use std::error::Error;

use penteconter::Game;
use pyxis::{
    BatchEvaluator, EvaluationInput, MaterialEvaluator, SearchReport, SequentialBatchEvaluator,
    SimulationStep, Tree, resolve_node,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut games = Vec::new();
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
        "3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1",
    ] {
        games.push(Game::new(fen.parse()?));
    }
    let original: Vec<_> = games.iter().map(|g| *g.position()).collect();
    let mut evaluator = MaterialEvaluator;
    let mut trees: Vec<_> = games
        .iter()
        .map(|game| resolve_node(game, &mut evaluator).map(Tree::new))
        .collect::<Result<_, _>>()?;

    let mut pending = Vec::new();
    for (i, (tree, game)) in trees.iter_mut().zip(&mut games).enumerate() {
        match tree.begin_simulation(game, 1.0)? {
            SimulationStep::Completed => println!("Tree {i}: terminal path completed immediately."),
            SimulationStep::NeedsEvaluation(request) => {
                println!(
                    "Tree {i}: paused with {:?} to move and {} legal moves.",
                    request.game().position().side_to_move(),
                    request.legal_moves().len(),
                );
                pending.push(request);
            }
        }
    }
    let inputs: Vec<_> = pending
        .iter()
        .map(|p| EvaluationInput {
            game: p.game(),
            legal_moves: p.legal_moves(),
        })
        .collect();
    println!(
        "\nOne batch call with {} inputs; this adapter evaluates them sequentially.",
        inputs.len()
    );
    // If this call fails, `?` drops every pending request and restores its game.
    let results = SequentialBatchEvaluator::new(&mut evaluator).evaluate_batch(&inputs)?;
    // Check before zip, which would otherwise silently truncate unequal lengths.
    assert_eq!(results.len(), pending.len());
    for (i, (request, evaluation)) in pending.into_iter().zip(results).enumerate() {
        println!(
            "Result {i}: leaf value {:+.4}, {} policy weights.",
            evaluation.value.get(),
            evaluation.policy_weights.len()
        );
        request.complete(evaluation)?;
    }

    println!();
    for (i, (tree, game)) in trees.iter().zip(&mut games).enumerate() {
        assert_eq!(*game.position(), original[i]);
        assert_eq!(game.undo(), None);
        let SearchReport::Nonterminal {
            best_move,
            simulations,
            ..
        } = tree.report()
        else {
            unreachable!("all fixture roots are nonterminal")
        };
        assert_eq!(simulations, 1);
        println!(
            "Tree {i}: root restored, {simulations} completed simulation, current choice {best_move}."
        );
    }
    Ok(())
}
