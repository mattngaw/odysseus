use std::io;

use penteconter::{Color, Game, Move, MoveKind, Square};
use pyxis::{
    Evaluation, Evaluator, ExpansionError, InvalidPolicyWeight, Node, ResolveError, Value,
    resolve_node,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct TestEvaluator<F> {
    evaluate: F,
    calls: usize,
}

fn evaluator<F>(evaluate: F) -> TestEvaluator<F>
where
    F: FnMut(&Game, &[Move]) -> Result<Evaluation, io::Error>,
{
    TestEvaluator { evaluate, calls: 0 }
}

impl<F> Evaluator for TestEvaluator<F>
where
    F: FnMut(&Game, &[Move]) -> Result<Evaluation, io::Error>,
{
    type Error = io::Error;

    fn evaluate(&mut self, game: &Game, moves: &[Move]) -> Result<Evaluation, Self::Error> {
        self.calls += 1;
        (self.evaluate)(game, moves)
    }
}

fn evaluation(policy_weights: Vec<f32>) -> Evaluation {
    Evaluation {
        value: Value::new(-0.625).unwrap(),
        policy_weights,
    }
}

fn normal(coordinates: &str) -> Move {
    let b = coordinates.as_bytes();
    Move::new(
        Square::from_coords(b[0] - b'a', b[1] - b'1').unwrap(),
        Square::from_coords(b[2] - b'a', b[3] - b'1').unwrap(),
        MoveKind::Normal,
    )
    .unwrap()
}

#[test]
fn nonterminal_resolution_preserves_move_order_and_side_to_move_value() {
    for (fen, side) in [
        (START, Color::White),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            Color::Black,
        ),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let before = *game.position();
        let mut expected = Vec::new();
        game.position().generate_legal_moves(&mut expected);
        let mut evaluator = evaluator(|supplied_game, moves| {
            assert!(std::ptr::eq(supplied_game, &game));
            assert_eq!(supplied_game.position().side_to_move(), side);
            assert_eq!(moves, expected);
            let mut weights = vec![0.0; moves.len()];
            weights[0] = 1.0;
            weights[1] = 3.0;
            Ok(evaluation(weights))
        });

        let Node::Expanded(node) = resolve_node(&game, &mut evaluator).unwrap() else {
            panic!("expected expansion");
        };
        assert_eq!(evaluator.calls, 1);
        assert_eq!(node.value().get(), -0.625);
        assert_eq!(node.edges().len(), expected.len());
        for (index, edge) in node.edges().iter().enumerate() {
            assert_eq!(edge.mv(), expected[index]);
            let prior = match index {
                0 => 0.25,
                1 => 0.75,
                _ => 0.0,
            };
            assert_eq!(edge.stats().prior(), prior);
            assert_eq!(edge.stats().visits(), 0);
            assert_eq!(edge.stats().value_sum(), 0.0);
            assert_eq!(edge.child(), None);
        }
        assert_eq!(*game.position(), before);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn terminal_positions_bypass_evaluation_including_draws_with_legal_moves() {
    let mut evaluator = evaluator(|_, _| panic!("terminal games must bypass evaluation"));
    for (fen, value) in [
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", -1.0),
        ("8/8/8/8/8/2k5/1q6/K7 w - - 0 1", -1.0),
        // Checkmate still wins over the 150-halfmove draw rule.
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 150 1", -1.0),
        ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", 0.0),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0.0),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 150 76", 0.0),
    ] {
        let game = Game::new(fen.parse().unwrap());
        let Node::Terminal(actual) = resolve_node(&game, &mut evaluator).unwrap() else {
            panic!("expected terminal node: {fen}");
        };
        assert_eq!(actual.get(), value);
    }
    assert_eq!(evaluator.calls, 0);
}

#[test]
fn repetition_resolution_uses_recorded_history_and_tracks_undo() {
    let mut game = Game::new(START.parse().unwrap());
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal);
    let mut evaluator = evaluator(|_, moves| Ok(evaluation(vec![0.0; moves.len()])));
    assert!(matches!(
        resolve_node(&game, &mut evaluator).unwrap(),
        Node::Expanded(_)
    ));
    for occurrence in 2..=3 {
        for mv in cycle {
            game.play(mv).unwrap();
        }
        assert_eq!(game.repetition_count(), occurrence);
        let before = *game.position();
        let node = resolve_node(&game, &mut evaluator).unwrap();
        if occurrence < 3 {
            let Node::Expanded(node) = node else {
                panic!("second occurrences still reach the evaluator")
            };
            assert!(node.edges().iter().all(|edge| edge.stats().prior() == 0.05));
        } else {
            let Node::Terminal(value) = node else {
                panic!("expected threefold adjudication")
            };
            assert_eq!(value.get(), 0.0);
        }
        assert_eq!(*game.position(), before);
        assert_eq!(game.repetition_count(), occurrence);
        assert_eq!(game.outcome(), None);
    }
    assert_eq!(evaluator.calls, 2);
    assert_eq!(game.undo(), Some(cycle[3]));
    assert!(matches!(
        resolve_node(&game, &mut evaluator).unwrap(),
        Node::Expanded(_)
    ));
    assert_eq!(evaluator.calls, 3);
    for index in (0..7).rev() {
        assert_eq!(game.undo(), Some(cycle[index % 4]));
    }
    assert_eq!(game.undo(), None);
    assert_eq!(*game.position(), START.parse().unwrap());
    assert_eq!(game.repetition_count(), 1);
}

#[test]
fn evaluator_failure_preserves_the_original_error_and_game() {
    let mut game = Game::new(START.parse().unwrap());
    let mv = normal("e2e4");
    game.play(mv).unwrap();
    let before = *game.position();
    let mut evaluator =
        evaluator(|_, _| Err(io::Error::new(io::ErrorKind::TimedOut, "fixture timeout")));
    let error = resolve_node(&game, &mut evaluator).unwrap_err();
    assert_eq!(evaluator.calls, 1);
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "fixture timeout"
    );
    let ResolveError::Evaluator(error) = error else {
        panic!("expected evaluator error")
    };
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert_eq!(*game.position(), before);
    assert_eq!(game.repetition_count(), 1);
    assert_eq!(game.undo(), Some(mv));
    assert_eq!(*game.position(), START.parse().unwrap());
}

#[test]
fn malformed_policy_is_reported_as_an_expansion_error() {
    let game = Game::new(START.parse().unwrap());
    for actual in [19, 21] {
        let mut evaluator = evaluator(|_, _| Ok(evaluation(vec![1.0; actual])));
        let error = resolve_node(&game, &mut evaluator).unwrap_err();
        assert!(
            matches!(error, ResolveError::Expansion(ExpansionError::PolicyLengthMismatch { expected: 20, actual: length }) if length == actual)
        );
    }
    for invalid in [-1.0, f32::NAN, f32::INFINITY] {
        let mut evaluator = evaluator(|_, moves| {
            let mut weights = vec![1.0; moves.len()];
            weights[7] = invalid;
            Ok(evaluation(weights))
        });
        let error = resolve_node(&game, &mut evaluator).unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert!(matches!(
            error,
            ResolveError::Expansion(ExpansionError::InvalidPolicyWeight(InvalidPolicyWeight {
                index: 7
            }))
        ));
    }
}
