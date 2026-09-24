use std::{env, process::ExitCode, time::Instant};

use penteconter::{
    Position,
    perft::{self, LeafMode},
};

const INITIAL: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const USAGE: &str = "Usage: perft --depth N [--fen \"FEN\"] [--mode bulk|apply] [--divide]

  --depth N          Required; 0..255. Depth zero counts the root as one leaf.
  --fen \"FEN\"       Defaults to the initial position; quote all six FEN fields.
  --mode bulk        Count final legal moves without reapplying them (default).
  --mode apply       Apply final moves too; black_box preserves that work.
  --divide           Print counts per root move, after timing finishes.
  --help             Show this help.

Legal filtering applies candidates in both modes. Traversal reapplies accepted
moves, except at the final ply in bulk mode. Incremental hashing; no transposition cache.
Timing includes traversal, buffer setup, and optional divide collection;
it excludes FEN parsing and console output. Use a release build for timing.";

#[derive(Debug)]
struct Options {
    depth: u8,
    fen: String,
    mode: LeafMode,
    divide: bool,
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, String> {
    let mut args = args.into_iter();
    let mut depth = None;
    let mut fen = None;
    let mut mode = None;
    let mut divide = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--depth" if depth.is_none() => {
                let value = args.next().ok_or("--depth requires a value")?;
                if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
                    return Err("depth must be an integer from 0 to 255".into());
                }
                depth = Some(
                    value
                        .parse::<u8>()
                        .map_err(|_| "depth must be an integer from 0 to 255")?,
                );
            }
            "--fen" if fen.is_none() => fen = Some(args.next().ok_or("--fen requires a value")?),
            "--mode" if mode.is_none() => {
                mode = Some(match args.next().as_deref() {
                    Some("bulk") => LeafMode::Bulk,
                    Some("apply") => LeafMode::Apply,
                    _ => return Err("--mode requires bulk or apply".into()),
                });
            }
            "--divide" if !divide => divide = true,
            _ => return Err(format!("unknown or repeated argument: {arg}")),
        }
    }
    Ok(Some(Options {
        depth: depth.ok_or("--depth is required")?,
        fen: fen.unwrap_or_else(|| INITIAL.into()),
        mode: mode.unwrap_or(LeafMode::Bulk),
        divide,
    }))
}

fn run() -> Result<(), String> {
    let Some(options) = parse(env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    let position: Position = options
        .fen
        .parse()
        .map_err(|e| format!("invalid FEN: {e}"))?;
    let start = Instant::now();
    let result = perft::run(&position, options.depth, options.mode, options.divide);
    let elapsed = start.elapsed();

    println!("FEN: {}", position.to_fen());
    println!("Depth: {}", options.depth);
    println!(
        "Build: {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!(
        "Leaf mode: {}",
        match options.mode {
            LeafMode::Bulk => "bulk (count final legal moves without reapplying)",
            LeafMode::Apply => "apply (apply final moves; black_box each resulting position)",
        }
    );
    println!("Traversal: copy-and-apply; one reused move buffer per depth; no hash cache");
    println!("Hashing: maintained during apply; full recomputation assertions in debug builds");
    println!(
        "Work: legal filtering applies candidates; traversal reapplies accepted moves, except bulk leaves"
    );
    println!("Timing: traversal + buffer setup + divide collection; excludes parsing and output");
    if let Some(rows) = result.divide {
        if options.depth == 0 {
            println!("Divide: no root moves at depth zero; the root is one leaf");
        } else {
            println!("Divide:");
            for (mv, nodes) in rows {
                println!("  {mv}: {nodes}");
            }
        }
    }
    println!("Nodes: {}", result.nodes);
    println!("Elapsed: {:.6} s", elapsed.as_secs_f64());
    if elapsed.is_zero() {
        println!("Nodes/s: n/a (elapsed time below timer resolution)");
    } else {
        println!(
            "Nodes/s: {:.0}",
            result.nodes as f64 / elapsed.as_secs_f64()
        );
    }
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

    fn options(args: &[&str]) -> Result<Option<Options>, String> {
        parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn parses_defaults_and_explicit_workload_options() {
        let defaults = options(&["--depth", "0"]).unwrap().unwrap();
        assert_eq!(defaults.depth, 0);
        assert_eq!(defaults.fen, INITIAL);
        assert_eq!(defaults.mode, LeafMode::Bulk);
        assert!(!defaults.divide);
        let explicit = options(&[
            "--fen", INITIAL, "--divide", "--mode", "apply", "--depth", "5",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(explicit.depth, 5);
        assert_eq!(explicit.fen, INITIAL);
        assert_eq!(explicit.mode, LeafMode::Apply);
        assert!(explicit.divide);
        assert!(options(&["--help"]).unwrap().is_none());
    }

    #[test]
    fn rejects_missing_malformed_unknown_and_repeated_arguments() {
        for args in [
            vec![],
            vec!["--depth"],
            vec!["--depth", "-1"],
            vec!["--depth", "256"],
            vec!["--depth", "1.5"],
            vec!["--depth", "2", "--mode", "fast"],
            vec!["--depth", "2", "--fen"],
            vec!["--depth", "2", "--wat"],
            vec!["--depth", "2", "--depth", "3"],
            vec!["--depth", "2", "--divide", "--divide"],
        ] {
            assert!(options(&args).is_err(), "{args:?}");
        }
    }
}
