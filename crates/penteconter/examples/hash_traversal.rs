//! Repeated traversal/cache experiment. Opt in with --features perft-experiment.
//! No documentation files are generated; stdout retains timings and counters.

use penteconter::{
    Position,
    perft::{
        LeafMode,
        experiments::hash_traversal::{self, Associativity, Cache, DepthStats, Legality},
    },
};
use std::{env, hint::black_box, process::ExitCode, time::Instant};

const USAGE: &str =
    "Usage: hash_traversal [--warmup N] [--samples N] [--cache-mib N] [--deeper] [--compare-ways]

Defaults: 3 warmup rounds, 11 measured rounds, 16 MiB cache.
Cache MiB must be a power of two, at most 1024. --deeper adds one ply.
Use --release --features perft-experiment. Single-threaded; no CPU pinning.
Both legality variants use the same traversal, full retained Position updates,
leaf compiler barriers, and move order. Default: direct-mapped always-replace.
--compare-ways compares uncached and 1/2/4-way caches with placement-only legality.
All ways reuse the same allocation and 32-byte entry size. Wider buckets replace
an exact key/depth match, otherwise the first empty entry, otherwise the shallowest
entry (ties: first slot). New entries are always admitted. Associativity dispatch
occurs once outside recursion. No extra bucket alignment beyond 32-byte entries.
Cache entries: 32 bytes, full 64-bit key plus exact remaining depth, no state
verification against true Zobrist collisions. No draws or search bounds.
Each sample starts with an empty table. Allocation is outside timing; clearing
and traversal are timed separately, and their sum is reported. Workload order
rotates and reverses between rounds. Parsing/checks/output are outside timing.
Hit/expansion counters come from a separate instrumented pass. Apply mode keeps
complete final positions observable; Bulk skips applying accepted final moves.";

const FIXTURES: [(&str, &str, u8, [u64; 2]); 4] = [
    (
        "initial",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        5,
        [4_865_609, 119_060_324],
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4,
        [4_085_603, 193_690_690],
    ),
    (
        "endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        5,
        [674_624, 11_030_083],
    ),
    (
        "promotions",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4,
        [2_103_487, 89_941_194],
    ),
];

struct Options {
    warmup: usize,
    samples: usize,
    cache_mib: usize,
    deeper: bool,
    compare_ways: bool,
}

fn parse() -> Result<Option<Options>, String> {
    let mut options = Options {
        warmup: 3,
        samples: 11,
        cache_mib: 16,
        deeper: false,
        compare_ways: false,
    };
    let mut seen = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            return Ok(None);
        }
        if seen.contains(&arg) {
            return Err(format!("repeated argument: {arg}"));
        }
        seen.push(arg.clone());
        if arg == "--compare-ways" {
            options.compare_ways = true;
            continue;
        }
        if arg == "--deeper" {
            options.deeper = true;
            continue;
        }
        let slot = match arg.as_str() {
            "--warmup" => &mut options.warmup,
            "--samples" => &mut options.samples,
            "--cache-mib" => &mut options.cache_mib,
            _ => return Err(format!("unknown argument: {arg}")),
        };
        *slot = args
            .next()
            .ok_or_else(|| format!("missing value: {arg}"))?
            .parse()
            .map_err(|_| format!("invalid unsigned integer: {arg}"))?;
    }
    if options.samples == 0 {
        return Err("--samples must be positive".into());
    }
    if !options.cache_mib.is_power_of_two() || options.cache_mib > 1024 {
        return Err("--cache-mib must be a power of two from 1 through 1024".into());
    }
    Ok(Some(options))
}

#[derive(Clone, Copy)]
struct Sample {
    traverse_ns: u128,
    clear_ns: u128,
}

struct Case {
    name: &'static str,
    position: Position,
    depth: u8,
    leaves: u64,
    mode: LeafMode,
    legality: Legality,
    ways: Option<Associativity>,
    samples: Vec<Sample>,
    stats: Vec<DepthStats>,
}

impl Case {
    fn label(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.name,
            self.depth,
            if self.mode == LeafMode::Apply {
                "apply"
            } else {
                "bulk"
            },
            if self.legality == Legality::Eager {
                "eager"
            } else {
                "placement"
            },
            if self.ways.is_some() {
                "cached"
            } else {
                "uncached"
            },
            self.ways.map_or(0, Associativity::ways)
        )
    }

    fn measure(&self, cache: &mut Cache) -> Sample {
        let clear_ns = if let Some(ways) = self.ways {
            let start = Instant::now();
            cache.reset(ways);
            start.elapsed().as_nanos()
        } else {
            0
        };
        let start = Instant::now();
        let result = hash_traversal::run(
            black_box(&self.position),
            black_box(self.depth),
            black_box(self.mode),
            self.legality,
            self.ways.map(|_| cache),
            false,
        );
        let nodes = black_box(result.nodes);
        let traverse_ns = start.elapsed().as_nanos();
        assert_eq!(nodes, self.leaves, "{}", self.label());
        Sample {
            traverse_ns,
            clear_ns,
        }
    }
}

fn percentile(sorted: &[u128], fraction: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * fraction;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] as f64 + (sorted[high] as f64 - sorted[low] as f64) * index.fract()
}

fn execute(options: Options) {
    let mut cache = Cache::new(options.cache_mib * 1024 * 1024 / 32);
    assert_eq!(cache.bytes(), options.cache_mib * 1024 * 1024);
    let cache_configs = if options.compare_ways {
        vec![
            None,
            Some(Associativity::One),
            Some(Associativity::Two),
            Some(Associativity::Four),
        ]
    } else {
        vec![None, Some(Associativity::One)]
    };
    let legalities = if options.compare_ways {
        &[Legality::PlacementOnly][..]
    } else {
        &[Legality::Eager, Legality::PlacementOnly][..]
    };
    let mut cases = Vec::new();
    for (name, fen, depth, counts) in FIXTURES {
        for mode in [LeafMode::Bulk, LeafMode::Apply] {
            for &ways in &cache_configs {
                for &legality in legalities {
                    cases.push(Case {
                        name,
                        position: fen.parse().unwrap(),
                        depth: depth + u8::from(options.deeper),
                        leaves: counts[usize::from(options.deeper)],
                        mode,
                        legality,
                        ways,
                        samples: Vec::new(),
                        stats: Vec::new(),
                    });
                }
            }
        }
    }
    for round in 0..options.warmup + options.samples {
        for offset in 0..cases.len() {
            let step = if round % 2 == 0 {
                offset
            } else {
                cases.len() - 1 - offset
            };
            let index = (round + step) % cases.len();
            let sample = cases[index].measure(&mut cache);
            if round >= options.warmup {
                cases[index].samples.push(sample);
            }
        }
    }
    // Separate diagnostics, including deterministic agreement between variants.
    for index in 0..cases.len() {
        let case = &mut cases[index];
        cache.reset(case.ways.unwrap_or(Associativity::One));
        let result = hash_traversal::run(
            &case.position,
            case.depth,
            case.mode,
            case.legality,
            case.ways.map(|_| &mut cache),
            true,
        );
        assert_eq!(result.nodes, case.leaves);
        case.stats = result.by_depth;
        if !options.compare_ways && index % 2 == 1 {
            assert_eq!(cases[index].stats, cases[index - 1].stats);
        }
    }
    println!("Penteconter hash traversal experiment v2\n{USAGE}");
    println!(
        "warmup={},samples={},cache_bytes={},position_bytes={},deeper={},compare_ways={}",
        options.warmup,
        options.samples,
        cache.bytes(),
        size_of::<Position>(),
        options.deeper,
        options.compare_ways
    );
    println!("\nSummary:");
    println!(
        "name,depth,mode,legality,cache,ways,leaves,min_ns,p25_ns,median_ns,p75_ns,max_ns,median_clear_ns,median_total_ns"
    );
    for case in &cases {
        let mut traverse: Vec<_> = case.samples.iter().map(|s| s.traverse_ns).collect();
        let mut clear: Vec<_> = case.samples.iter().map(|s| s.clear_ns).collect();
        let mut total: Vec<_> = case
            .samples
            .iter()
            .map(|s| s.traverse_ns + s.clear_ns)
            .collect();
        traverse.sort_unstable();
        clear.sort_unstable();
        total.sort_unstable();
        println!(
            "{},{},{},{:.0},{:.0},{:.0},{},{:.0},{:.0}",
            case.label(),
            case.leaves,
            traverse[0],
            percentile(&traverse, 0.25),
            percentile(&traverse, 0.5),
            percentile(&traverse, 0.75),
            traverse[traverse.len() - 1],
            percentile(&clear, 0.5),
            percentile(&total, 0.5)
        );
    }
    println!("\nRaw samples:");
    println!("name,depth,mode,legality,cache,ways,sample,traverse_ns,clear_ns,total_ns");
    for case in &cases {
        for (i, sample) in case.samples.iter().enumerate() {
            println!(
                "{},{},{},{},{}",
                case.label(),
                i + 1,
                sample.traverse_ns,
                sample.clear_ns,
                sample.traverse_ns + sample.clear_ns
            );
        }
    }
    println!("\nUntimed diagnostics:");
    println!(
        "name,depth,mode,legality,cache,ways,remaining_depth,visited,expanded,hits,child_positions"
    );
    for case in &cases {
        for (depth, stats) in case.stats.iter().enumerate().rev() {
            println!(
                "{},{},{},{},{},{}",
                case.label(),
                depth,
                stats.visited,
                stats.expanded,
                stats.hits,
                stats.child_positions
            );
        }
    }
    println!("\nFixtures:");
    println!("name,depth,leaves,fen");
    for (name, fen, depth, counts) in FIXTURES {
        println!(
            "{},{},{},{}",
            name,
            depth + u8::from(options.deeper),
            counts[usize::from(options.deeper)],
            fen
        );
    }
}

fn main() -> ExitCode {
    let options = match parse() {
        Ok(Some(options)) => options,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("{error}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    if cfg!(debug_assertions) {
        eprintln!("Use --release: debug consistency checks dominate this experiment.");
        return ExitCode::FAILURE;
    }
    execute(options);
    ExitCode::SUCCESS
}
