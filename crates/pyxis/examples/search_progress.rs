use std::error::Error;

use penteconter::Game;
use pyxis::{SearchReport, Tree, UniformEvaluator, resolve_node};

fn main() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new("k7/8/1QK5/8/8/8/8/8 w - - 0 1".parse()?);
    let original = *game.position();
    let mut evaluator = UniformEvaluator;
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator)?);
    let mut completed = 0;
    println!("One retained tree; pause, inspect, and resume at each checkpoint.");
    for checkpoint in [0, 1, 8, 32] {
        while completed < checkpoint {
            // A GUI worker can check for a stop request here. Each simulation
            // finishes and restores the root game before control returns.
            tree.simulate(&mut game, &mut evaluator, 1.0)?;
            completed += 1;
        }
        match tree.report() {
            SearchReport::Terminal(value) => {
                println!("Exact terminal value: {}", value.get());
                break;
            }
            SearchReport::Nonterminal {
                best_move,
                simulations,
                moves,
            } => {
                let best = moves.iter().find(|entry| entry.mv == best_move).unwrap();
                let fraction = best.visit_fraction.map_or_else(
                    || "undefined (no visits)".to_owned(),
                    |fraction| format!("{fraction:.3}"),
                );
                println!(
                    "S={simulations:2}: move {best_move}, N={}, P={:.3}, Q={:+.3}, N/S={fraction}",
                    best.stats.visits(),
                    best.stats.prior(),
                    best.stats.mean_value()
                );
            }
        }
        assert_eq!(*game.position(), original);
        assert_eq!(game.undo(), None);
    }
    println!("Before visits the move is a prior-based fallback; afterward it uses visits.");
    Ok(())
}
