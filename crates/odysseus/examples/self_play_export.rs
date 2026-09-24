//! Export short, searched self-play games for the Python training_data example.
//! Usage: self_play_export <new-output.jsonl>

use std::{error::Error, fs::File, io::BufWriter, io::Write};

use odysseus::self_play::{
    SelfPlayConfig, SelfPlayOutcome, jsonl::write_completed_game, play_game,
};
use penteconter::Game;
use pyxis::UniformEvaluator;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: self_play_export <new-output.jsonl>")?;
    if args.next().is_some() {
        return Err("usage: self_play_export <new-output.jsonl>".into());
    }
    // Avoid silently replacing previous measurements or training records.
    let mut output = BufWriter::new(File::create_new(&path)?);
    let config = SelfPlayConfig {
        simulations_per_move: 32,
        exploration: 1.0,
        temperature: 0.0,
        seed: 42,
        max_plies: 2,
    };
    println!("Uniform evaluator; 32 simulations/root, greedy visits, seed=42, cap=2.");
    println!("Small hand-picked starting positions; moves chosen by search, not scripted.");
    for (name, fen) in [
        ("White mate", "k7/8/1QK5/8/8/8/8/8 w - - 0 1"),
        // Use the other side of the board: distinct encoded inputs make this
        // fixture suitable for a fixed-batch overfitting check.
        ("Black mate", "8/8/8/8/8/5kq1/8/7K b - - 0 1"),
        ("75-move draw", "4k3/8/8/8/8/8/8/R3K3 w - - 148 75"),
        ("Forced loss", "k7/8/2K5/1Q6/8/8/8/8 b - - 0 1"),
        (
            "Truncated opening",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
    ] {
        let result = play_game(&mut Game::new(fen.parse()?), &mut UniformEvaluator, config)?;
        match result.outcome {
            SelfPlayOutcome::Completed(completed) => {
                write_completed_game(&mut output, &completed)?;
                println!(
                    "{name}: {:?}; wrote {} examples",
                    completed.adjudication,
                    completed.examples.len()
                );
            }
            SelfPlayOutcome::Truncated(recorder) => println!(
                "{name}: excluded {} unlabeled roots",
                recorder.roots().len()
            ),
        }
    }
    output.flush()?;
    println!("JSONL written to {path}");
    Ok(())
}
