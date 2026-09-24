//! Real search reports on scripted lines, demonstrating recording and final labels.

use std::error::Error;

use odysseus::self_play::{RecordError, Recorder};
use penteconter::Game;
use pyxis::{UniformEvaluator, search};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn demonstrate(label: &str, line: &[&str]) -> Result<(), Box<dyn Error>> {
    let mut game = Game::new(START.parse()?);
    let mut recorder = Recorder::new();
    println!("\n{label}");
    for &coordinate in line {
        let report = search(&mut game, &mut UniformEvaluator, 8, 1.0)?;
        recorder.record(&game, &report)?;
        let root = recorder.roots().last().unwrap();
        println!(
            "  Before {coordinate}: ply={}, {:?}, input=[{},{}], legal={}, visited={}, total visits={}, policy sum={:.6}",
            root.ply(),
            root.side_to_move(),
            root.input().len(),
            root.input()[0].len(),
            root.policy().len(),
            root.policy()
                .iter()
                .filter(|entry| entry.visits > 0)
                .count(),
            root.total_visits(),
            root.policy_target().iter().sum::<f32>(),
        );
        if root.ply() == 0 {
            println!("  Sample policy entries: relative move / slot / visits / target");
            let target = root.policy_target();
            for entry in root.policy().iter().take(10) {
                println!(
                    "    {} / {} / {} / {:.3}",
                    entry.index.entry(),
                    entry.index.index(),
                    entry.visits,
                    target[entry.index.index()]
                );
            }
        }
        let mut legal = Vec::new();
        game.position().generate_legal_moves(&mut legal);
        let mv = legal
            .into_iter()
            .find(|mv| mv.to_string() == coordinate)
            .ok_or_else(|| format!("invalid scripted move {coordinate}"))?;
        game.play(mv)?;
    }
    match recorder.finish(&game) {
        Ok(completed) => {
            println!("  Outcome: {:?}", completed.adjudication);
            for example in completed.examples {
                println!(
                    "  Ply {} ({:?}): W/D/L target {:?}",
                    example.root.ply(),
                    example.root.side_to_move(),
                    example.value_target
                );
            }
        }
        Err(RecordError::UnfinishedGame) => {
            println!(
                "  Unfinished: {} snapshots retained without outcome labels.",
                recorder.roots().len()
            );
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Uniform evaluator, 8 simulations per root, exploration=1.");
    println!("Moves are scripted to demonstrate outcomes; search supplies the visit targets.");
    demonstrate("Black wins by checkmate", &["f2f3", "e7e5", "g2g4", "d8h4"])?;
    demonstrate(
        "Third occurrence ends as a search/self-play draw",
        &[
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ],
    )?;
    demonstrate("Interrupted game", &["e2e4"])?;
    Ok(())
}
