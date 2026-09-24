//! Play one checkpoint-backed game and export only completed-game training data.
//! Run with --help for settings. Inference is CPU FP32, one persistent worker.

use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use odysseus::{
    neural::{NeuralEvaluator, default_python},
    self_play::{SelfPlayConfig, SelfPlayOutcome, jsonl::write_completed_game, play_game},
};
use penteconter::{Game, Position};

const USAGE: &str = "Usage: self_play_neural --checkpoint PATH --output NEW.jsonl [options]
  --python PATH       Interpreter with neurodiktyon installed (default: workspace .venv)
  --simulations N     Simulations per move (default: 32, positive)
  --seed N            Move-sampling seed (default: 42)
  --temperature T     Fixed visit-sampling temperature (default: 1; 0 = greedy)
  --max-plies N       Maximum played plies (default: 256; 0 allowed)
  --fen FEN           Starting position (default: standard opening; no earlier history)
  --help              Show this help

Both sides use the checkpoint; PUCT exploration is fixed at 1.0.
Search uses a fresh tree per move. There is no root noise or temperature schedule.
The output must not exist. A cap or search failure leaves it empty, without labels.
An initially terminal position exports an empty completed game without loading Python.";

struct Options {
    python: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
    position: Position,
    config: SelfPlayConfig,
}

impl Options {
    fn parse() -> Result<Option<Self>, Box<dyn Error>> {
        let mut args = std::env::args().skip(1);
        let mut python = PathBuf::from(default_python());
        let mut checkpoint = None;
        let mut output = None;
        let mut position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?;
        let mut config = SelfPlayConfig {
            simulations_per_move: 32,
            exploration: 1.0,
            temperature: 1.0,
            seed: 42,
            max_plies: 256,
        };
        while let Some(flag) = args.next() {
            if flag == "--help" {
                println!("{USAGE}");
                return Ok(None);
            }
            if !matches!(
                flag.as_str(),
                "--python"
                    | "--checkpoint"
                    | "--output"
                    | "--simulations"
                    | "--seed"
                    | "--temperature"
                    | "--max-plies"
                    | "--fen"
            ) {
                return Err(format!("unknown option {flag}; use --help").into());
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--python" => python = value.into(),
                "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
                "--output" => output = Some(PathBuf::from(value)),
                "--simulations" => config.simulations_per_move = value.parse()?,
                "--seed" => config.seed = value.parse()?,
                "--temperature" => config.temperature = value.parse()?,
                "--max-plies" => config.max_plies = value.parse()?,
                "--fen" => position = value.parse()?,
                _ => unreachable!(),
            }
        }
        if config.simulations_per_move == 0 {
            return Err("--simulations must be positive".into());
        }
        if !config.temperature.is_finite() || config.temperature < 0.0 {
            return Err("--temperature must be finite and nonnegative".into());
        }
        Ok(Some(Self {
            python,
            checkpoint: checkpoint.ok_or("--checkpoint is required; use --help")?,
            output: output.ok_or("--output is required; use --help")?,
            position,
            config,
        }))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(options) = Options::parse()? else {
        return Ok(());
    };
    // Reserve a new file before doing any inference; never overwrite prior data.
    let mut output = BufWriter::new(File::create_new(&options.output)?);
    let mut evaluator = NeuralEvaluator::new(&options.python, Some(options.checkpoint.clone()));
    let mut game = Game::new(options.position);
    println!("Checkpoint: {}", options.checkpoint.display());
    println!("Python: {}; CPU FP32", options.python.display());
    println!("Settings: {:?}", options.config);
    println!("Starting FEN: {}", game.position().to_fen());
    let result = play_game(&mut game, &mut evaluator, options.config)?;
    println!("Played {} plies:", result.moves.len());
    println!(
        "{}",
        result
            .moves
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("Final FEN: {}", game.position().to_fen());
    match result.outcome {
        SelfPlayOutcome::Completed(completed) => {
            write_completed_game(&mut output, &completed)?;
            output.flush()?;
            println!(
                "Completed: {:?}; wrote {} labeled examples to {}",
                completed.adjudication,
                completed.examples.len(),
                options.output.display()
            );
        }
        SelfPlayOutcome::Truncated(recorder) => println!(
            "Truncated: excluded {} unlabeled roots; {} is empty",
            recorder.roots().len(),
            options.output.display()
        ),
    }
    Ok(())
}
