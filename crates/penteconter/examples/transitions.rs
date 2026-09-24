//! Paired, uncached copy/apply versus make/unmake traversal measurements.

use penteconter::{
    Position,
    perft::{
        LeafMode,
        experiments::transitions::{self, Strategy},
    },
};
use std::{env, hint::black_box, process::ExitCode, time::Instant};

const USAGE: &str = "Usage: transitions [--warmup N] [--samples N] [--deeper]

Defaults: 3 warmup rounds, 21 paired samples. --deeper adds one ply.
Use --release --features perft-experiment. No transposition cache.
Both strategies use board-only legality and the same forward hash updates.
Apply exposes a reference to each complete leaf through black_box in both paths.
Bulk skips full transitions for final legal moves. Make/unmake restores all
accepted children, including final leaves in Apply mode.
Move buffers are allocated inside timing. FEN parsing, initial root copy,
count/restoration checks, output, and diagnostic counters are outside timing.
Each pair measures both strategies consecutively; first strategy alternates
between rounds, and case order rotates. Code/data are warmed; scheduling,
frequency, and other system activity are uncontrolled. Single-threaded.";

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
    deeper: bool,
}

fn parse() -> Result<Option<Options>, String> {
    let mut options = Options {
        warmup: 3,
        samples: 21,
        deeper: false,
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
        if arg == "--deeper" {
            options.deeper = true;
            continue;
        }
        let slot = match arg.as_str() {
            "--warmup" => &mut options.warmup,
            "--samples" => &mut options.samples,
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
    Ok(Some(options))
}

#[derive(Clone, Copy)]
struct Sample {
    copy_ns: u128,
    unmake_ns: u128,
    copy_first: bool,
}

struct Case {
    name: &'static str,
    position: Position,
    depth: u8,
    leaves: u64,
    mode: LeafMode,
    samples: Vec<Sample>,
}

impl Case {
    fn label(&self) -> String {
        format!(
            "{},{},{}",
            self.name,
            self.depth,
            if self.mode == LeafMode::Apply {
                "apply"
            } else {
                "bulk"
            }
        )
    }

    fn measure(&self, strategy: Strategy) -> u128 {
        let mut working = self.position;
        let start = Instant::now();
        let result = transitions::run(
            black_box(&mut working),
            black_box(self.depth),
            black_box(self.mode),
            strategy,
            false,
        );
        let nodes = black_box(result.nodes);
        let ns = start.elapsed().as_nanos();
        assert_eq!(nodes, self.leaves, "{} {strategy:?}", self.label());
        assert_eq!(
            working,
            self.position,
            "root changed: {} {strategy:?}",
            self.label()
        );
        assert_eq!(working.zobrist_key(), working.recompute_zobrist_key());
        ns
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * fraction;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * index.fract()
}

fn execute(options: Options) {
    let mut cases = Vec::new();
    for (name, fen, depth, counts) in FIXTURES {
        for mode in [LeafMode::Bulk, LeafMode::Apply] {
            cases.push(Case {
                name,
                position: fen.parse().unwrap(),
                depth: depth + u8::from(options.deeper),
                leaves: counts[usize::from(options.deeper)],
                mode,
                samples: Vec::new(),
            });
        }
    }
    for round in 0..options.warmup + options.samples {
        for offset in 0..cases.len() {
            let index = (round + offset) % cases.len();
            let case = &mut cases[index];
            let copy_first = round % 2 == 0;
            let (copy_ns, unmake_ns) = if copy_first {
                (
                    case.measure(Strategy::CopyApply),
                    case.measure(Strategy::MakeUnmake),
                )
            } else {
                let unmake_ns = case.measure(Strategy::MakeUnmake);
                (case.measure(Strategy::CopyApply), unmake_ns)
            };
            if round >= options.warmup {
                case.samples.push(Sample {
                    copy_ns,
                    unmake_ns,
                    copy_first,
                });
            }
        }
    }
    println!("Penteconter transition experiment v1\n{USAGE}");
    println!(
        "warmup={},samples={},position_bytes={},undo_bytes={},deeper={}",
        options.warmup,
        options.samples,
        size_of::<Position>(),
        transitions::UNDO_BYTES,
        options.deeper
    );
    println!("\nSummary:");
    println!("name,depth,mode,strategy,leaves,min_ns,p25_ns,median_ns,p75_ns,max_ns,median_mnps");
    for case in &cases {
        for (strategy, copy) in [("copy_apply", true), ("make_unmake", false)] {
            let mut times: Vec<_> = case
                .samples
                .iter()
                .map(|s| {
                    if copy {
                        s.copy_ns as f64
                    } else {
                        s.unmake_ns as f64
                    }
                })
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "{},{},{},{:.0},{:.0},{:.0},{:.0},{:.0},{:.3}",
                case.label(),
                strategy,
                case.leaves,
                times[0],
                percentile(&times, 0.25),
                percentile(&times, 0.5),
                percentile(&times, 0.75),
                times[times.len() - 1],
                case.leaves as f64 * 1000.0 / percentile(&times, 0.5)
            );
        }
    }
    println!("\nPaired ratios (unmake time / copy time; below 1 favors unmake):");
    println!("name,depth,mode,p25_ratio,median_ratio,p75_ratio");
    for case in &cases {
        let mut ratios: Vec<_> = case
            .samples
            .iter()
            .map(|s| s.unmake_ns as f64 / s.copy_ns as f64)
            .collect();
        ratios.sort_by(f64::total_cmp);
        println!(
            "{},{:.4},{:.4},{:.4}",
            case.label(),
            percentile(&ratios, 0.25),
            percentile(&ratios, 0.5),
            percentile(&ratios, 0.75)
        );
    }
    println!("\nRaw pairs:");
    println!("name,depth,mode,sample,first,copy_ns,unmake_ns");
    for case in &cases {
        for (i, s) in case.samples.iter().enumerate() {
            println!(
                "{},{},{},{},{}",
                case.label(),
                i + 1,
                if s.copy_first {
                    "copy_apply"
                } else {
                    "make_unmake"
                },
                s.copy_ns,
                s.unmake_ns
            );
        }
    }
    println!("\nUntimed diagnostics:");
    println!("name,depth,mode,leaves,expanded,transitions");
    for case in &cases {
        let mut working = case.position;
        let copy = transitions::run(
            &mut working,
            case.depth,
            case.mode,
            Strategy::CopyApply,
            true,
        );
        assert_eq!(working, case.position);
        let unmake = transitions::run(
            &mut working,
            case.depth,
            case.mode,
            Strategy::MakeUnmake,
            true,
        );
        assert_eq!(working, case.position);
        assert_eq!(copy, unmake);
        assert_eq!(copy.nodes, case.leaves);
        println!(
            "{},{},{},{}",
            case.label(),
            copy.nodes,
            copy.expanded,
            copy.transitions
        );
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
