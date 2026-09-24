use std::{cell::Cell, io};

use penteconter::{Game, Move};
use pyxis::{
    BatchEvaluator, Evaluation, EvaluationInput, Evaluator, MaterialEvaluator,
    SequentialBatchEvaluator, SimulationStep, Tree, UniformEvaluator, resolve_node,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct TestEvaluator<F>(F);

impl<F> Evaluator for TestEvaluator<F>
where
    F: FnMut(&Game, &[Move]) -> Result<Evaluation, io::Error>,
{
    type Error = io::Error;

    fn evaluate(&mut self, game: &Game, moves: &[Move]) -> Result<Evaluation, Self::Error> {
        (self.0)(game, moves)
    }
}

fn legal_moves(game: &Game) -> Vec<Move> {
    assert_eq!(game.outcome(), None);
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    moves
}

#[test]
fn forwards_borrowed_inputs_in_order_with_each_games_move_order_and_value_perspective() {
    let games = [
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/R3K3 b - - 0 1",
        // Black has one legal move, so the policy lengths differ across inputs.
        "k7/8/2K5/8/8/8/8/R7 b - - 0 1",
    ]
    .map(|fen| Game::new(fen.parse().unwrap()));
    let mut moves = games.each_ref().map(legal_moves);
    moves[0].reverse();
    // Include a repeated input: it still needs its own result in the right slot.
    let inputs = [2, 0, 1, 0].map(|i| EvaluationInput {
        game: &games[i],
        legal_moves: &moves[i],
    });
    let calls = Cell::new(0);
    let mut evaluator = TestEvaluator(|game: &Game, moves: &[Move]| {
        let i = calls.get();
        calls.set(i + 1);
        assert!(std::ptr::eq(game, inputs[i].game));
        assert_eq!(moves, inputs[i].legal_moves);
        let mut evaluation = MaterialEvaluator.evaluate(game, moves).unwrap();
        // Distinct, unnormalized weights reveal any reordering or normalization.
        evaluation.policy_weights = (1..=moves.len()).map(|n| (n * (i + 1)) as f32).collect();
        Ok(evaluation)
    });
    let results = SequentialBatchEvaluator::new(&mut evaluator)
        .evaluate_batch(&inputs)
        .unwrap();
    assert_eq!(calls.get(), 4);
    assert_eq!(results.len(), inputs.len());
    for (i, (result, expected_value)) in results.iter().zip([-0.5, 0.5, -0.5, 0.5]).enumerate() {
        assert_eq!(result.value.get(), expected_value);
        let expected: Vec<_> = (1..=inputs[i].legal_moves.len())
            .map(|n| (n * (i + 1)) as f32)
            .collect();
        assert_eq!(result.policy_weights, expected);
    }
}

#[test]
fn empty_batch_makes_no_calls_and_single_input_produces_one_result() {
    let game = Game::new(START.parse().unwrap());
    let moves = legal_moves(&game);
    let calls = Cell::new(0);
    let mut evaluator = TestEvaluator(|game: &Game, moves: &[Move]| {
        calls.set(calls.get() + 1);
        Ok(UniformEvaluator.evaluate(game, moves).unwrap())
    });
    let mut batch = SequentialBatchEvaluator::new(&mut evaluator);
    assert!(batch.evaluate_batch(&[]).unwrap().is_empty());
    assert_eq!(calls.get(), 0);
    let results = batch
        .evaluate_batch(&[EvaluationInput {
            game: &game,
            legal_moves: &moves,
        }])
        .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].value.get(), 0.0);
    assert_eq!(results[0].policy_weights, vec![1.0; 20]);
}

#[test]
fn first_error_is_preserved_without_partial_results_or_later_calls() {
    let game = Game::new(START.parse().unwrap());
    let moves = legal_moves(&game);
    let inputs = [EvaluationInput {
        game: &game,
        legal_moves: &moves,
    }; 3];
    for fail_at in 1..=3 {
        let calls = Cell::new(0);
        let mut evaluator = TestEvaluator(|game: &Game, moves: &[Move]| {
            calls.set(calls.get() + 1);
            if calls.get() == fail_at {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "fixture timeout"));
            }
            Ok(UniformEvaluator.evaluate(game, moves).unwrap())
        });
        let mut batch = SequentialBatchEvaluator::new(&mut evaluator);
        let error = batch.evaluate_batch(&inputs).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert_eq!(error.to_string(), "fixture timeout");
        assert_eq!(calls.get(), fail_at);
        // The backend's state survived the error and it can be called again.
        assert_eq!(batch.evaluate_batch(&inputs[..1]).unwrap().len(), 1);
        assert_eq!(calls.get(), fail_at + 1);
    }
}

#[test]
fn independent_trees_match_single_position_search_after_each_completed_batch() {
    let fens = [
        START,
        "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
        "3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1",
    ];
    let mut games = fens.map(|fen| Game::new(fen.parse().unwrap()));
    let mut reference_games = fens.map(|fen| Game::new(fen.parse().unwrap()));
    let mut evaluator = MaterialEvaluator;
    let mut trees = games
        .each_ref()
        .map(|game| Tree::new(resolve_node(game, &mut evaluator).unwrap()));
    let mut reference_trees = reference_games
        .each_ref()
        .map(|game| Tree::new(resolve_node(game, &mut MaterialEvaluator).unwrap()));
    let mut batch = SequentialBatchEvaluator::new(&mut evaluator);
    for _ in 0..32 {
        let mut pending = Vec::new();
        for (tree, game) in trees.iter_mut().zip(&mut games) {
            if let SimulationStep::NeedsEvaluation(request) =
                tree.begin_simulation(game, 1.0).unwrap()
            {
                pending.push(request);
            }
        }
        let inputs: Vec<_> = pending
            .iter()
            .map(|p| EvaluationInput {
                game: p.game(),
                legal_moves: p.legal_moves(),
            })
            .collect();
        let results = batch.evaluate_batch(&inputs).unwrap();
        assert_eq!(results.len(), pending.len());
        for (request, evaluation) in pending.into_iter().zip(results) {
            request.complete(evaluation).unwrap();
        }
        for (tree, game) in reference_trees.iter_mut().zip(&mut reference_games) {
            tree.simulate(game, &mut MaterialEvaluator, 1.0).unwrap();
        }
        for i in 0..3 {
            assert_eq!(
                format!("{:?}", trees[i]),
                format!("{:?}", reference_trees[i])
            );
            assert_eq!(trees[i].report(), reference_trees[i].report());
            assert_eq!(games[i].position(), reference_games[i].position());
            assert_eq!(games[i].repetition_count(), 1);
        }
    }
    for game in &mut games {
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn failed_batch_cancels_every_pending_request_on_early_return_and_restores_history() {
    let mut games: Vec<_> = (0..3).map(|_| Game::new(START.parse().unwrap())).collect();
    let mut history = Vec::new();
    for (i, game) in games.iter_mut().enumerate() {
        for coordinates in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            let mv = legal_moves(game)
                .into_iter()
                .find(|m| m.to_string() == coordinates)
                .unwrap();
            game.play(mv).unwrap();
            if i == 0 {
                history.push(mv);
            }
        }
        assert_eq!(game.repetition_count(), 2);
    }
    let original = *games[0].position();
    let calls = Cell::new(0);
    let mut evaluator = TestEvaluator(|game: &Game, moves: &[Move]| {
        calls.set(calls.get() + 1);
        // Three root evaluations, one successful leaf, then a failed leaf.
        if calls.get() == 5 {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "fixture timeout"));
        }
        assert_eq!(
            game.repetition_count(),
            if calls.get() <= 3 { 2 } else { 1 }
        );
        Ok(UniformEvaluator.evaluate(game, moves).unwrap())
    });
    let mut trees: Vec<_> = games
        .iter()
        .map(|game| Tree::new(resolve_node(game, &mut evaluator).unwrap()))
        .collect();
    let before: Vec<_> = trees.iter().map(|t| format!("{t:?}")).collect();
    let result = (|| -> Result<(), io::Error> {
        let pending: Vec<_> = trees
            .iter_mut()
            .zip(&mut games)
            .map(|(tree, game)| {
                let SimulationStep::NeedsEvaluation(pending) =
                    tree.begin_simulation(game, 1.0).unwrap()
                else {
                    panic!("expected a nonterminal leaf")
                };
                pending
            })
            .collect();
        let inputs: Vec<_> = pending
            .iter()
            .map(|p| EvaluationInput {
                game: p.game(),
                legal_moves: p.legal_moves(),
            })
            .collect();
        let results = SequentialBatchEvaluator::new(&mut evaluator).evaluate_batch(&inputs)?;
        assert_eq!(results.len(), pending.len());
        for (request, evaluation) in pending.into_iter().zip(results) {
            request.complete(evaluation).unwrap();
        }
        Ok(())
    })();
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
    assert_eq!(calls.get(), 5);
    assert_eq!(
        trees.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>(),
        before
    );
    for game in &mut games {
        assert_eq!(*game.position(), original);
        assert_eq!(game.repetition_count(), 2);
        for &mv in history.iter().rev() {
            assert_eq!(game.undo(), Some(mv));
        }
        assert_eq!(game.undo(), None);
        assert_eq!(*game.position(), START.parse().unwrap());
        assert_eq!(game.repetition_count(), 1);
    }
}
