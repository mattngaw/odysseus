//! Repeated uncached perft and full Zobrist recomputation measurements.
//! Run with --release; stdout contains summaries, fixtures, and every raw sample.

use std::{
    env,
    hint::black_box,
    process::ExitCode,
    time::{Duration, Instant},
};

use penteconter::{
    Position,
    perft::{self, LeafMode},
};

const USAGE: &str = "Usage: baseline [--warmup N] [--samples N] [--hash-iterations N]

  --warmup N           Untimed rounds per workload (default 3; zero allowed).
  --samples N          Measured rounds per workload (default 11; at least 1).
  --hash-iterations N   Recomputations per hash sample (default 1000000; at least 1).
  --help               Show this help.

Use --release. Each perft sample is one traversal, including buffer setup.
Each hash sample repeats one fixed position with black_box on input and output;
the reported cost includes the loop and compiler barriers. No overhead subtraction.
FEN parsing, count assertions, statistics, and output are outside timing.
Workload order rotates each round. These are warm-cache, single-thread timings;
CPU placement, frequency, and other system activity are not controlled.
Perft uses board-only legality and hashes retained positions, without a cache. Apply mode
keeps final positions and their keys observable; bulk may optimize unused work.
Redirect stdout to retain the complete report, including raw elapsed nanoseconds.";

const INITIAL: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

// Fixed depths and counts from https://www.chessprogramming.org/Perft_Results.
const PERFT_CASES: [(&str, &str, u8, u64); 4] = [
    ("initial", INITIAL, 5, 4_865_609),
    ("kiwipete", KIWIPETE, 4, 4_085_603),
    (
        "endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        5,
        674_624,
    ),
    (
        "promotions",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4,
        2_103_487,
    ),
];

// Each EP pair differs only in the stored target. Compare within a pair to
// avoid confusing the EP cost with a different number or placement of pieces.
const HASH_CASES: [(&str, &str); 8] = [
    ("initial", INITIAL),
    ("kiwipete", KIWIPETE),
    ("uncapturable_absent", "4k3/8/8/3p4/8/8/8/4K3 w - - 0 1"),
    ("uncapturable_target", "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1"),
    ("legal_absent", "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1"),
    ("legal_target", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
    ("pinned_absent", "k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1"),
    ("pinned_target", "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
];

struct Options {
    warmup: usize,
    samples: usize,
    hash_iterations: u64,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, String> {
    let mut args = args.into_iter();
    let (mut warmup, mut samples, mut hash_iterations) = (None, None, None);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            return Ok(None);
        }
        let slot = match arg.as_str() {
            "--warmup" if warmup.is_none() => &mut warmup,
            "--samples" if samples.is_none() => &mut samples,
            "--hash-iterations" if hash_iterations.is_none() => &mut hash_iterations,
            _ => return Err(format!("unknown or repeated argument: {arg}")),
        };
        let value = args
            .next()
            .ok_or_else(|| format!("{arg} requires a value"))?;
        if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
            return Err(format!("{arg} requires an unsigned integer"));
        }
        *slot = Some(
            value
                .parse::<u64>()
                .map_err(|_| format!("{arg} is too large"))?,
        );
    }
    let warmup = usize::try_from(warmup.unwrap_or(3)).map_err(|_| "--warmup is too large")?;
    let samples = usize::try_from(samples.unwrap_or(11)).map_err(|_| "--samples is too large")?;
    let hash_iterations = hash_iterations.unwrap_or(1_000_000);
    if samples == 0 || hash_iterations == 0 {
        return Err("--samples and --hash-iterations must be at least 1".into());
    }
    Ok(Some(Options {
        warmup,
        samples,
        hash_iterations,
    }))
}

#[derive(Clone, Copy)]
enum Work {
    Perft {
        depth: u8,
        mode: LeafMode,
        leaves: u64,
    },
    Hash {
        iterations: u64,
    },
}

impl Work {
    fn labels(self) -> (&'static str, u8, &'static str) {
        match self {
            Self::Perft { depth, mode, .. } => (
                "perft",
                depth,
                match mode {
                    LeafMode::Bulk => "bulk",
                    LeafMode::Apply => "apply",
                },
            ),
            Self::Hash { .. } => ("hash", 0, "recompute"),
        }
    }

    fn units(self) -> u64 {
        match self {
            Self::Perft { leaves, .. } => leaves,
            Self::Hash { iterations } => iterations,
        }
    }
}

struct Case {
    name: &'static str,
    position: Position,
    work: Work,
    samples: Vec<Duration>,
}

impl Case {
    fn measure(&self) -> Duration {
        match self.work {
            Work::Perft {
                depth,
                mode,
                leaves,
            } => {
                let start = Instant::now();
                let nodes = black_box(
                    perft::run(
                        black_box(&self.position),
                        black_box(depth),
                        black_box(mode),
                        false,
                    )
                    .nodes,
                );
                let elapsed = start.elapsed();
                assert_eq!(nodes, leaves, "incorrect perft count: {}", self.name);
                elapsed
            }
            Work::Hash { iterations } => {
                let start = Instant::now();
                for _ in 0..iterations {
                    // Both barriers matter: prevent hoisting the pure function
                    // out of the loop and discarding an unused result.
                    // https://doc.rust-lang.org/std/hint/fn.black_box.html
                    black_box(black_box(&self.position).recompute_zobrist_key());
                }
                start.elapsed()
            }
        }
    }
}

fn cases(options: &Options) -> Vec<Case> {
    let mut cases = Vec::new();
    for (name, fen, depth, leaves) in PERFT_CASES {
        for mode in [LeafMode::Bulk, LeafMode::Apply] {
            cases.push(Case {
                name,
                position: fen.parse().expect("valid perft fixture"),
                work: Work::Perft {
                    depth,
                    mode,
                    leaves,
                },
                samples: Vec::with_capacity(options.samples),
            });
        }
    }
    for (name, fen) in HASH_CASES {
        cases.push(Case {
            name,
            position: fen.parse().expect("valid hash fixture"),
            work: Work::Hash {
                iterations: options.hash_iterations,
            },
            samples: Vec::with_capacity(options.samples),
        });
    }
    cases
}

// Linear interpolation at (n - 1) * fraction, including the even-sample median.
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * fraction;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * index.fract()
}

fn report(cases: &[Case], options: &Options) {
    println!("Penteconter performance baseline v3 (board-only legality)");
    println!(
        "Platform: {}-{}; optimized build with debug assertions disabled",
        env::consts::ARCH,
        env::consts::OS
    );
    println!("Position size: {} bytes", size_of::<Position>());
    println!(
        "Warmup rounds: {}; measured rounds: {}; hash calls/sample: {}",
        options.warmup, options.samples, options.hash_iterations
    );
    println!(
        "Order: rotate starting workload by one per round; raw sample numbers are per-workload chronological order"
    );
    println!(
        "Perft: uncached, board-only legality, incremental child hashing, no divide; includes move buffers and legal filtering; counts checked after timing"
    );
    println!(
        "Bulk skips final reapplication and may optimize unused hashes; apply includes it and black_box of each leaf position, including its key"
    );
    println!(
        "Hash: fixed position per sample; black_box input and output on every call; includes loop/barrier overhead"
    );
    println!(
        "FEN parsing, assertions, statistics, and output excluded; warm-cache single thread; scheduling/frequency uncontrolled"
    );
    println!(
        "Quartiles: linear interpolation at (n-1)*p; range and IQR describe samples, not confidence intervals"
    );

    println!("\nSummary: ns/leaf for perft, ns/key for hashing; units/s derived from median");
    println!(
        "kind,name,depth,mode,units_per_sample,min_ns,p25_ns,median_ns,p75_ns,max_ns,median_units_per_s"
    );
    for case in cases {
        let (kind, depth, mode) = case.work.labels();
        let units = case.work.units();
        let mut values: Vec<f64> = case
            .samples
            .iter()
            .map(|elapsed| elapsed.as_secs_f64() * 1e9 / units as f64)
            .collect();
        values.sort_unstable_by(f64::total_cmp);
        let median = quantile(&values, 0.5);
        println!(
            "{kind},{},{depth},{mode},{units},{:.3},{:.3},{median:.3},{:.3},{:.3},{:.0}",
            case.name,
            values[0],
            quantile(&values, 0.25),
            quantile(&values, 0.75),
            values[values.len() - 1],
            1e9 / median
        );
    }

    println!("\nRaw samples:");
    println!("kind,name,depth,mode,sample,units,elapsed_ns");
    for case in cases {
        let (kind, depth, mode) = case.work.labels();
        for (sample, elapsed) in case.samples.iter().enumerate() {
            println!(
                "{kind},{},{depth},{mode},{},{},{}",
                case.name,
                sample + 1,
                case.work.units(),
                elapsed.as_nanos()
            );
        }
    }
    println!("\nFixtures:");
    println!("kind,name,depth,mode,pieces,key,fen");
    for case in cases {
        let (kind, depth, mode) = case.work.labels();
        println!(
            "{kind},{},{depth},{mode},{},{:#018x},{}",
            case.name,
            case.position.board().occupied().count(),
            case.position.recompute_zobrist_key(),
            case.position.to_fen()
        );
    }
}

fn run() -> Result<(), String> {
    let Some(options) = parse(env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err("benchmark requires --release (debug assertions must be disabled)".into());
    }
    let mut cases = cases(&options);
    eprintln!(
        "Warming up {} workloads for {} rounds...",
        cases.len(),
        options.warmup
    );
    for round in 0..options.warmup {
        for offset in 0..cases.len() {
            black_box(cases[(round % cases.len() + offset) % cases.len()].measure());
        }
    }
    eprintln!("Measuring {} rounds...", options.samples);
    for round in 0..options.samples {
        for offset in 0..cases.len() {
            let index = (round % cases.len() + offset) % cases.len();
            let elapsed = cases[index].measure();
            if elapsed.is_zero() {
                return Err("sample below timer resolution; increase --hash-iterations".into());
            }
            cases[index].samples.push(elapsed);
        }
    }
    report(&cases, &options);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}\n\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use penteconter::MoveKind;

    #[test]
    fn quartiles_interpolate_even_odd_and_single_sample_sets() {
        for (values, expected) in [
            (vec![10., 20., 30., 40., 50.], [20., 30., 40.]),
            (vec![10., 20., 30., 40.], [17.5, 25., 32.5]),
            (vec![12.], [12., 12., 12.]),
        ] {
            for (fraction, expected) in [0.25, 0.5, 0.75].into_iter().zip(expected) {
                assert_eq!(quantile(&values, fraction), expected);
            }
        }
    }

    #[test]
    fn matched_ep_fixtures_have_the_intended_legality_and_hash_relationships() {
        for (pair, expected_captures) in HASH_CASES[2..].chunks_exact(2).zip([0, 1, 0]) {
            let absent: Position = pair[0].1.parse().unwrap();
            let target: Position = pair[1].1.parse().unwrap();
            assert_eq!(absent.board(), target.board());
            assert_eq!(absent.side_to_move(), target.side_to_move());
            assert_eq!(absent.castling_rights(), target.castling_rights());
            assert_eq!(absent.halfmove_clock(), target.halfmove_clock());
            assert_eq!(absent.fullmove_number(), target.fullmove_number());
            assert!(absent.en_passant_target().is_none());
            assert!(target.en_passant_target().is_some());
            let mut moves = Vec::new();
            target.generate_legal_moves(&mut moves);
            assert_eq!(
                moves
                    .iter()
                    .filter(|mv| mv.kind() == MoveKind::EnPassant)
                    .count(),
                expected_captures
            );
            assert_eq!(
                absent.recompute_zobrist_key() != target.recompute_zobrist_key(),
                expected_captures != 0
            );
        }
    }
}
