use std::error::Error;

use penteconter::Game;
use pyxis::{SearchReport, UniformEvaluator, search};

fn main() -> Result<(), Box<dyn Error>> {
    let simulations = std::env::args()
        .nth(1)
        .map(|argument| argument.parse())
        .transpose()?
        .unwrap_or(32);
    for (label, fen) in [
        (
            "Starting position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        ("Mate available", "k7/8/1QK5/8/8/8/8/8 w - - 0 1"),
        ("Already checkmated", "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1"),
    ] {
        let mut game = Game::new(fen.parse()?);
        let original = *game.position();
        println!("\n{label} ({:?} to move)", game.position().side_to_move());
        match search(&mut game, &mut UniformEvaluator, simulations, 1.0)? {
            SearchReport::Terminal(value) => {
                println!("  Exact value: {}, no move, zero simulations", value.get());
            }
            SearchReport::Nonterminal {
                best_move,
                simulations,
                moves,
            } => {
                println!("  Suggested move: {best_move}; completed simulations: {simulations}");
                println!("  All root moves: move    N       Q      P     N/S");
                for entry in moves {
                    println!(
                        "                  {:5} {:3}  {:+.3}  {:.3}  {:.3}",
                        entry.mv.to_string(),
                        entry.stats.visits(),
                        entry.stats.mean_value(),
                        entry.stats.prior(),
                        entry
                            .visit_fraction
                            .expect("fixed-budget search completed at least one simulation"),
                    );
                }
            }
        }
        assert_eq!(*game.position(), original);
        assert_eq!(game.undo(), None);
    }
    println!("\nAll games are unchanged. The evaluator uses uniform weights and zero values.");
    Ok(())
}
