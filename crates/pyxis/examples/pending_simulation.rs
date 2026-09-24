use std::error::Error;

use penteconter::Game;
use pyxis::{Evaluator, SimulationStep, Tree, UniformEvaluator, resolve_node};

fn main() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    let original = *game.position();
    let mut evaluator = UniformEvaluator;
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator)?);
    let original_report = tree.report();
    println!("Root: {}", game.position().to_fen());

    let SimulationStep::NeedsEvaluation(pending) = tree.begin_simulation(&mut game, 1.0)? else {
        unreachable!("the starting position's first child is nonterminal")
    };
    let selected_leaf = *pending.game().position();
    println!("\nPaused at: {}", selected_leaf.to_fen());
    println!(
        "Evaluator receives {} ordered legal moves.",
        pending.legal_moves().len()
    );
    // While pending exists, Rust prevents using tree or game directly.
    drop(pending);
    assert_eq!(*game.position(), original);
    assert_eq!(tree.report(), original_report);
    assert_eq!(tree.node_count(), 1);
    println!("Canceled: game restored, still 1 node and 0 completed simulations.");

    let SimulationStep::NeedsEvaluation(pending) = tree.begin_simulation(&mut game, 1.0)? else {
        unreachable!()
    };
    assert_eq!(*pending.game().position(), selected_leaf);
    println!("\nRetry selected the same leaf.");
    // This is the explicit pause boundary. The caller decides when to evaluate.
    let evaluation = evaluator.evaluate(pending.game(), pending.legal_moves())?;
    println!(
        "Received value {:+.2} from the leaf player's perspective.",
        evaluation.value.get()
    );
    pending.complete(evaluation)?;
    assert_eq!(*game.position(), original);
    assert_eq!(game.undo(), None);
    assert_eq!(tree.node_count(), 2);
    println!("Completed: game restored, now 2 nodes and 1 completed simulation.");

    // Existing callers can still perform the whole operation in one call.
    tree.simulate(&mut game, &mut evaluator, 1.0)?;
    assert_eq!(*game.position(), original);
    println!("Synchronous wrapper: game restored, now 2 completed simulations.");
    Ok(())
}
