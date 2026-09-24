use std::{convert::Infallible, error::Error};

use penteconter::{Game, Move};
use pyxis::{Evaluation, Evaluator, Node, Tree, UniformEvaluator, resolve_node};

#[derive(Default)]
struct CountingEvaluator {
    calls: usize,
}

impl Evaluator for CountingEvaluator {
    type Error = Infallible;

    fn evaluate(&mut self, game: &Game, moves: &[Move]) -> Result<Evaluation, Self::Error> {
        self.calls += 1;
        UniformEvaluator.evaluate(game, moves)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    let root_position = *game.position();
    let mut evaluator = CountingEvaluator::default();
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator)?);
    println!(
        "Root setup: nodes={}, evaluator calls={}, root visits=0",
        tree.node_count(),
        evaluator.calls
    );
    // Exercise the one-simulation operation directly; see `search` for the driver.
    // With uniform priors and zero values, the first 20 visits explore root moves.
    for completed in 1..=24 {
        tree.simulate(&mut game, &mut evaluator, 1.0)?;
        let Node::Expanded(root) = tree.node(tree.root()).unwrap() else {
            unreachable!()
        };
        let visits: u32 = root.edges().iter().map(|edge| edge.stats().visits()).sum();
        assert_eq!(visits, completed);
        assert_eq!(*game.position(), root_position);
        assert_eq!(game.repetition_count(), 1);
        if [1, 20, 21, 24].contains(&completed) {
            println!(
                "After {completed:2} simulations: nodes={}, evaluator calls={}, root visits={visits}",
                tree.node_count(),
                evaluator.calls
            );
        }
    }
    let Node::Expanded(root) = tree.node(tree.root()).unwrap() else {
        unreachable!()
    };
    println!("\nRoot edges visited more than once (search continued below their children):");
    for edge in root.edges().iter().filter(|edge| edge.stats().visits() > 1) {
        println!(
            "  {}: N={}, Q={:+.2}",
            edge.mv(),
            edge.stats().visits(),
            edge.stats().mean_value()
        );
    }
    assert_eq!(game.undo(), None);
    println!("\nThe game is back at the root after every call; the tree retains its statistics.");
    Ok(())
}
