use std::error::Error;

use penteconter::Game;
use pyxis::{Node, UniformEvaluator, resolve_node};

fn main() -> Result<(), Box<dyn Error>> {
    let mut evaluator = UniformEvaluator;
    for (label, fen) in [
        (
            "Starting position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ),
        ("Black checkmated", "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1"),
        ("Black stalemated", "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1"),
        ("Insufficient material", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ] {
        let game = Game::new(fen.parse()?);
        println!("{label} ({:?} to move)", game.position().side_to_move());
        match resolve_node(&game, &mut evaluator)? {
            Node::Terminal(value) => {
                println!("  Exact value: {}, evaluator skipped", value.get());
            }
            Node::Expanded(node) => {
                println!("  Evaluator value: {}", node.value().get());
                println!("  Legal edges: {}", node.edges().len());
                let first = node.edges()[0];
                println!(
                    "  First edge: {}, P={}, N={}, W={}",
                    first.mv(),
                    first.stats().prior(),
                    first.stats().visits(),
                    first.stats().value_sum(),
                );
            }
        }
    }
    Ok(())
}
