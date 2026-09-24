use std::convert::Infallible;

use penteconter::Game;
use pyxis::{Evaluation, Evaluator, UniformEvaluator, normalize_policy};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[test]
fn neutral_value_and_unit_weights_for_either_side_to_move() {
    let mut evaluator = UniformEvaluator;
    for (fen, expected_moves) in [
        (START, 20),
        // Black is in check and has exactly one legal reply, Ka8-b8.
        ("k7/8/2K5/8/8/8/8/R7 b - - 0 1", 1),
    ] {
        let game = Game::new(fen.parse().unwrap());
        assert_eq!(game.outcome(), None);
        let mut legal_moves = Vec::new();
        game.position().generate_legal_moves(&mut legal_moves);
        assert_eq!(legal_moves.len(), expected_moves);

        let result: Result<Evaluation, Infallible> = evaluator.evaluate(&game, &legal_moves);
        let evaluation = result.unwrap();
        assert_eq!(evaluation.value.get(), 0.0);
        assert_eq!(evaluation.policy_weights.len(), expected_moves);
        assert!(
            evaluation
                .policy_weights
                .iter()
                .all(|&weight| weight == 1.0)
        );
    }
}

#[test]
fn normalizing_the_predictions_gives_uniform_legal_priors() {
    let game = Game::new(START.parse().unwrap());
    let mut legal_moves = Vec::new();
    game.position().generate_legal_moves(&mut legal_moves);
    let mut evaluation = UniformEvaluator.evaluate(&game, &legal_moves).unwrap();

    normalize_policy(&mut evaluation.policy_weights).unwrap();

    assert_eq!(evaluation.policy_weights.len(), legal_moves.len());
    assert_eq!(evaluation.policy_weights, vec![0.05; 20]);
    let sum: f64 = evaluation.policy_weights.into_iter().map(f64::from).sum();
    assert!((sum - 1.0).abs() <= f64::from(f32::EPSILON));
}
