//! Run the complete self-play loop with a deterministic uniform evaluator.
//! Usage: self_play_game [seed] [max_plies]

use std::error::Error;

use odysseus::self_play::{SelfPlayConfig, SelfPlayOutcome, play_game};
use penteconter::Game;
use pyxis::UniformEvaluator;

fn demonstrate(label: &str, fen: &str, config: SelfPlayConfig) -> Result<(), Box<dyn Error>> {
    let mut game = Game::new(fen.parse()?);
    println!("\n{label}\n  {config:?}");
    let result = play_game(&mut game, &mut UniformEvaluator, config)?;
    println!("  Played {} plies:", result.moves.len());
    for chunk in result.moves.chunks(12) {
        println!(
            "    {}",
            chunk
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!("  Final FEN: {}", game.position().to_fen());
    match result.outcome {
        SelfPlayOutcome::Completed(completed) => {
            println!(
                "  Completed: {:?}; {} labeled examples",
                completed.adjudication,
                completed.examples.len()
            );
            for example in completed.examples.iter().take(4) {
                println!(
                    "    Ply {} {:?}: {} visits, W/D/L {:?}",
                    example.root.ply(),
                    example.root.side_to_move(),
                    example.root.total_visits(),
                    example.value_target
                );
            }
        }
        SelfPlayOutcome::Truncated(recorder) => {
            println!(
                "  Ply limit reached: {} unfinished roots; no W/D/L labels.",
                recorder.roots().len()
            );
            if let Some(root) = recorder.roots().first() {
                println!(
                    "    First input=[{},{}], legal moves={}, visits={}, policy sum={:.6}",
                    root.input().len(),
                    root.input()[0].len(),
                    root.policy().len(),
                    root.total_visits(),
                    root.policy_target().iter().sum::<f32>()
                );
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let seed = args
        .next()
        .map(|arg| arg.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    let max_plies = args
        .next()
        .map(|arg| arg.parse::<u32>())
        .transpose()?
        .unwrap_or(32);
    if args.next().is_some() {
        return Err("usage: self_play_game [seed] [max_plies]".into());
    }
    let config = SelfPlayConfig {
        simulations_per_move: 32,
        exploration: 1.0,
        temperature: 1.0,
        seed,
        max_plies,
    };
    println!("Uniform evaluator: zero values, equal legal priors; no neural model required.");
    println!("Every played move is selected from its search visits; no scripted game moves.");
    demonstrate(
        "Start position, sampled moves",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        config,
    )?;
    demonstrate(
        "Mate available, greedy selection, one-ply limit",
        "k7/8/1QK5/8/8/8/8/8 w - - 0 1",
        SelfPlayConfig {
            temperature: 0.0,
            max_plies: 1,
            ..config
        },
    )?;
    Ok(())
}
