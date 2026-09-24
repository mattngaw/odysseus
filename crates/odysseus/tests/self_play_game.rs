use std::io;

use odysseus::self_play::{SelfPlayConfig, SelfPlayError, SelfPlayOutcome, play_game};
use penteconter::{Color, DrawReason, Game, GameOutcome, Move, Position};
use pyxis::{
    Adjudication, Evaluation, Evaluator, ResolveError, SearchError, SimulationError,
    UniformEvaluator, Value, adjudicate, encoding::encode, vocabulary::index_for_move,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const CYCLE: [&str; 8] = [
    "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
];

fn config(max_plies: u32) -> SelfPlayConfig {
    SelfPlayConfig {
        simulations_per_move: 1,
        exploration: 1.0,
        temperature: 1.0,
        seed: 42,
        max_plies,
    }
}

fn play(game: &mut Game, coordinate: &str) {
    let mut legal = Vec::new();
    game.position().generate_legal_moves(&mut legal);
    game.play(
        legal
            .into_iter()
            .find(|mv| mv.to_string() == coordinate)
            .unwrap(),
    )
    .unwrap();
}

fn history(game: &Game) -> Vec<(Position, usize)> {
    game.positions()
        .map(|frame| (*frame.position, frame.repetition_count))
        .collect()
}

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
        assert_eq!(
            adjudicate(game),
            None,
            "terminal games must bypass evaluation"
        );
        (self.evaluate)(game, moves)
    }
}

fn scripted<'a>(
    line: &'a [&'a str],
) -> TestEvaluator<impl FnMut(&Game, &[Move]) -> Result<Evaluation, io::Error> + 'a> {
    evaluator(move |game, moves| {
        let mut weights = vec![1.0; moves.len()];
        if let Some(coordinate) = line.get(game.positions().count() - 1) {
            weights.fill(0.0);
            weights[moves
                .iter()
                .position(|mv| mv.to_string() == *coordinate)
                .unwrap()] = 1.0;
        }
        Ok(Evaluation {
            value: Value::new(0.0).unwrap(),
            policy_weights: weights,
        })
    })
}

#[test]
fn seeded_run_replays_legal_moves_and_captures_inputs_before_play_with_existing_history() {
    let config = SelfPlayConfig {
        simulations_per_move: 32,
        ..config(6)
    };
    let mut game = Game::new(START.parse().unwrap());
    let mut replay = Game::new(START.parse().unwrap());
    for coordinate in ["e2e4", "e7e5"] {
        play(&mut game, coordinate);
        play(&mut replay, coordinate);
    }
    let result = play_game(&mut game, &mut UniformEvaluator, config).unwrap();
    let SelfPlayOutcome::Truncated(records) = result.outcome else {
        panic!("expected ply cap");
    };
    assert_eq!(result.moves.len(), 6);
    assert_eq!(records.roots().len(), 6);
    for (offset, (root, &mv)) in records.roots().iter().zip(&result.moves).enumerate() {
        assert_eq!(adjudicate(&replay), None);
        assert_eq!(root.ply(), offset + 2);
        assert_eq!(root.side_to_move(), replay.position().side_to_move());
        assert_eq!(root.input(), &encode(&replay));
        assert_eq!(root.total_visits(), 32);
        let slot = index_for_move(mv, root.side_to_move()).unwrap().index();
        assert!(root.legal_mask()[slot]);
        assert!(root.policy_target()[slot] > 0.0);
        assert!((root.policy_target().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        replay.play(mv).unwrap();
    }
    assert_eq!(history(&game), history(&replay));
    assert_eq!(adjudicate(&game), None);

    let mut again = Game::new(START.parse().unwrap());
    for coordinate in ["e2e4", "e7e5"] {
        play(&mut again, coordinate);
    }
    let repeated = play_game(&mut again, &mut UniformEvaluator, config).unwrap();
    assert_eq!(result.moves, repeated.moves);
    let SelfPlayOutcome::Truncated(repeated_records) = repeated.outcome else {
        panic!("expected ply cap");
    };
    assert_eq!(records.roots(), repeated_records.roots());
    assert_eq!(history(&game), history(&again));
}

#[test]
fn checkmate_on_the_last_allowed_ply_completes_and_labels_both_player_perspectives() {
    for (line, winner) in [
        (&["f2f3", "e7e5", "g2g4", "d8h4"][..], Color::Black),
        (&["e2e4", "f7f6", "d2d4", "g7g5", "d1h5"][..], Color::White),
    ] {
        let mut game = Game::new(START.parse().unwrap());
        let mut evaluator = scripted(line);
        let result = play_game(&mut game, &mut evaluator, config(line.len() as u32)).unwrap();
        assert_eq!(
            result
                .moves
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            line
        );
        assert_eq!(evaluator.calls, 2 * line.len() - 1);
        let SelfPlayOutcome::Completed(completed) = result.outcome else {
            panic!("mate must beat cap");
        };
        assert_eq!(
            completed.adjudication,
            Adjudication::Automatic(GameOutcome::Checkmate { winner })
        );
        assert_eq!(completed.examples.len(), result.moves.len());
        for example in completed.examples {
            assert_eq!(
                example.value_target,
                if example.root.side_to_move() == winner {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            );
        }
    }
}

#[test]
fn third_occurrence_stops_immediately_and_uses_supplied_history() {
    for prefix in [0, 7] {
        let mut game = Game::new(START.parse().unwrap());
        for coordinate in &CYCLE[..prefix] {
            play(&mut game, coordinate);
        }
        let initial_input = encode(&game);
        let mut evaluator = scripted(&CYCLE);
        let result = play_game(&mut game, &mut evaluator, config(20)).unwrap();
        assert_eq!(result.moves.len(), 8 - prefix);
        assert_eq!(game.repetition_count(), 3);
        assert_eq!(game.outcome(), None); // Threefold remains a search/self-play policy.
        assert_eq!(evaluator.calls, 2 * (8 - prefix) - 1);
        let SelfPlayOutcome::Completed(completed) = result.outcome else {
            panic!("threefold must complete");
        };
        assert_eq!(
            completed.adjudication,
            Adjudication::ThreefoldRepetitionDraw
        );
        assert_eq!(completed.examples.len(), 8 - prefix);
        assert_eq!(completed.examples[0].root.ply(), prefix);
        assert_eq!(completed.examples[0].root.input(), &initial_input);
        assert!(
            completed
                .examples
                .iter()
                .all(|example| example.value_target == [0.0, 1.0, 0.0])
        );
    }
}

#[test]
fn clock_threshold_and_mate_precedence_survive_the_ply_cap() {
    for clock in [99, 149] {
        let mut game = Game::new(
            format!("4k3/8/8/8/8/8/8/R3K3 w - - {clock} 75")
                .parse()
                .unwrap(),
        );
        let result = play_game(&mut game, &mut scripted(&["e1e2"]), config(1)).unwrap();
        assert_eq!(game.position().halfmove_clock(), clock + 1);
        if clock == 99 {
            let SelfPlayOutcome::Truncated(records) = result.outcome else {
                panic!("100 halfmoves is not terminal");
            };
            assert_eq!(records.roots().len(), 1);
        } else {
            let SelfPlayOutcome::Completed(completed) = result.outcome else {
                panic!("150 halfmoves must complete");
            };
            assert_eq!(
                completed.adjudication,
                Adjudication::Automatic(GameOutcome::Draw {
                    reason: DrawReason::SeventyFiveMoveRule
                })
            );
            assert_eq!(completed.examples[0].value_target, [0.0, 1.0, 0.0]);
        }
    }
    let mut game = Game::new("k7/8/1QK5/8/8/8/8/8 w - - 149 75".parse().unwrap());
    let result = play_game(&mut game, &mut scripted(&["b6b7"]), config(1)).unwrap();
    assert_eq!(game.position().halfmove_clock(), 150);
    let SelfPlayOutcome::Completed(completed) = result.outcome else {
        panic!("mate must complete");
    };
    assert_eq!(
        completed.adjudication,
        Adjudication::Automatic(GameOutcome::Checkmate {
            winner: Color::White
        })
    );
    assert_eq!(completed.examples[0].value_target, [1.0, 0.0, 0.0]);
}

#[test]
fn terminal_starts_and_zero_ply_caps_never_evaluate_or_fabricate_examples() {
    for fen in [
        START,
        "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let before = history(&game);
        let terminal = adjudicate(&game);
        let mut evaluator = evaluator(|_, _| panic!("must not evaluate"));
        let result = play_game(&mut game, &mut evaluator, config(0)).unwrap();
        assert!(result.moves.is_empty());
        assert_eq!(evaluator.calls, 0);
        assert_eq!(history(&game), before);
        match (result.outcome, terminal) {
            (SelfPlayOutcome::Completed(completed), Some(expected)) => {
                assert_eq!(completed.adjudication, expected);
                assert!(completed.examples.is_empty());
            }
            (SelfPlayOutcome::Truncated(records), None) => assert!(records.roots().is_empty()),
            _ => panic!("terminal status and cap disagree"),
        }
    }
}

#[test]
fn settings_are_validated_even_before_a_zero_cap_or_terminal_start() {
    let mut cases = vec![(
        SelfPlayConfig {
            simulations_per_move: 0,
            ..config(0)
        },
        0,
    )];
    for exploration in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        cases.push((
            SelfPlayConfig {
                exploration,
                ..config(0)
            },
            1,
        ));
    }
    for temperature in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.push((
            SelfPlayConfig {
                temperature,
                ..config(0)
            },
            2,
        ));
    }
    for fen in [START, "4k3/8/8/8/8/8/8/4K3 w - - 0 1"] {
        for &(config, kind) in &cases {
            let mut game = Game::new(fen.parse().unwrap());
            let before = history(&game);
            let mut evaluator = evaluator(|_, _| panic!("must not evaluate invalid settings"));
            let error = play_game(&mut game, &mut evaluator, config).unwrap_err();
            assert!(matches!(
                (kind, error),
                (0, SelfPlayError::ZeroSimulations)
                    | (1, SelfPlayError::InvalidExploration)
                    | (2, SelfPlayError::InvalidTemperature)
            ));
            assert_eq!(evaluator.calls, 0);
            assert_eq!(history(&game), before);
        }
    }
}

#[test]
fn evaluator_errors_propagate_with_only_already_played_moves_left_in_game() {
    for fail_at in [1, 2, 4] {
        let mut calls = 0;
        let mut evaluator = evaluator(|_, moves| {
            calls += 1;
            if calls == fail_at {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "fixture evaluation failure",
                ));
            }
            Ok(Evaluation {
                value: Value::new(0.0).unwrap(),
                policy_weights: vec![1.0; moves.len()],
            })
        });
        let mut game = Game::new(START.parse().unwrap());
        let error = play_game(&mut game, &mut evaluator, config(4)).unwrap_err();
        assert_eq!(evaluator.calls, fail_at);
        let cause = match error {
            SelfPlayError::Search(SearchError::RootResolution(ResolveError::Evaluator(cause))) => {
                cause
            }
            SelfPlayError::Search(SearchError::Simulation(SimulationError::Resolution(
                ResolveError::Evaluator(cause),
            ))) => cause,
            unexpected => panic!("unexpected failure: {unexpected:?}"),
        };
        assert_eq!(cause.kind(), io::ErrorKind::TimedOut);
        let mut expected = Game::new(START.parse().unwrap());
        if fail_at == 4 {
            play(&mut expected, "a2a3");
        }
        assert_eq!(history(&game), history(&expected));
        assert_eq!(adjudicate(&game), None);
    }
}
