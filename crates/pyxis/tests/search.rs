use std::io;

use penteconter::{Game, Move};
use pyxis::{
    Evaluation, Evaluator, ExpansionError, ResolveError, SearchError, SearchReport,
    SimulationError, Tree, Value, resolve_node, search,
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

fn evaluation(weights: Vec<f32>, value: f32) -> Evaluation {
    Evaluation {
        value: Value::new(value).unwrap(),
        policy_weights: weights,
    }
}

fn play(game: &mut Game, coordinates: &str) -> Move {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let mv = moves
        .into_iter()
        .find(|mv| mv.to_string() == coordinates)
        .unwrap();
    game.play(mv).unwrap();
    mv
}

#[test]
fn budget_counts_distributions_and_legal_move_order_are_reported_exactly() {
    for budget in [1, 20, 24] {
        let mut game = Game::new(START.parse().unwrap());
        let original = *game.position();
        let mut legal = Vec::new();
        game.position().generate_legal_moves(&mut legal);
        let mut evaluator = evaluator(|_, moves| Ok(evaluation(vec![1.0; moves.len()], 0.0)));
        let result = search(&mut game, &mut evaluator, budget, 1.0).unwrap();
        let SearchReport::Nonterminal {
            best_move,
            simulations,
            moves,
        } = result
        else {
            panic!("expected completed search");
        };
        assert_eq!(simulations, u64::from(budget));
        assert_eq!(evaluator.calls, budget as usize + 1); // Root setup is extra.
        assert_eq!(best_move, legal[0]);
        assert_eq!(moves.len(), legal.len());
        assert_eq!(
            moves.iter().map(|entry| entry.stats.visits()).sum::<u32>(),
            budget
        );
        assert!(
            (moves
                .iter()
                .map(|entry| f64::from(entry.visit_fraction.unwrap()))
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-6
        );
        for (index, (entry, mv)) in moves.iter().zip(&legal).enumerate() {
            // Equal priors and zero values balance visits, breaking ties by order.
            let expected = budget / 20 + u32::from((index as u32) < budget % 20);
            assert_eq!(entry.mv, *mv);
            assert_eq!(entry.stats.visits(), expected);
            assert_eq!(entry.stats.prior(), 0.05);
            assert_eq!(entry.stats.value_sum(), 0.0);
            assert_eq!(entry.stats.mean_value(), 0.0);
            assert!((entry.visit_fraction.unwrap() - expected as f32 / budget as f32).abs() < 1e-6);
            if expected == 0 {
                assert_eq!(entry.visit_fraction, Some(0.0));
            }
        }
        assert_eq!(*game.position(), original);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), None);
        // A second call is a fresh search, with no retained visits or cache.
        let again = search(&mut game, &mut evaluator, budget, 1.0).unwrap();
        assert_eq!(
            again,
            SearchReport::Nonterminal {
                best_move,
                simulations,
                moves
            }
        );
        assert_eq!(evaluator.calls, 2 * (budget as usize + 1));
    }
}

#[test]
fn most_visits_win_over_value_prior_and_puct_with_first_move_on_ties() {
    for (values, budget, visits_a, visits_b, expected_best) in [
        // A gets two visits but finishes with worse Q and PUCT than B.
        ([0.0, -0.25, -0.75, -0.5], 3, 2, 1, "a2a3"),
        // B wins despite having the lower initial prior.
        ([0.0, 0.5, -0.2, 0.0], 3, 1, 2, "b2b3"),
        // Equal visits: A wins by order even though B has the higher Q.
        ([0.0, 0.5, -0.2, 0.0], 2, 1, 1, "a2a3"),
    ] {
        let mut call = 0;
        let mut evaluator = evaluator(|_, moves| {
            let mut weights = vec![1.0; moves.len()];
            if call == 0 {
                weights.fill(0.0);
                for (mv, weight) in [("a2a3", 0.8), ("b2b3", 0.2)] {
                    let index = moves.iter().position(|m| m.to_string() == mv).unwrap();
                    weights[index] = weight;
                }
            }
            let value = values[call];
            call += 1;
            Ok(evaluation(weights, value))
        });
        let mut game = Game::new(START.parse().unwrap());
        let SearchReport::Nonterminal {
            best_move, moves, ..
        } = search(&mut game, &mut evaluator, budget, 1.0).unwrap()
        else {
            panic!("expected completed search");
        };
        let a = moves
            .iter()
            .find(|entry| entry.mv.to_string() == "a2a3")
            .unwrap();
        let b = moves
            .iter()
            .find(|entry| entry.mv.to_string() == "b2b3")
            .unwrap();
        assert_eq!(a.stats.visits(), visits_a);
        assert_eq!(b.stats.visits(), visits_b);
        assert!(b.stats.mean_value() > a.stats.mean_value());
        assert!(a.stats.prior() > b.stats.prior());
        assert_eq!(best_move.to_string(), expected_best);
        if visits_a > visits_b {
            assert!(
                pyxis::puct_score(b.stats, u64::from(budget), 1.0)
                    > pyxis::puct_score(a.stats, u64::from(budget), 1.0)
            );
        }
    }
}

#[test]
fn terminal_roots_return_exact_values_without_evaluation_or_simulations() {
    let mut evaluator = evaluator(|_, _| panic!("terminal root must bypass evaluation"));
    let mut games = Vec::new();
    for (fen, value) in [
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", -1.0),
        ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", 0.0),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0.0),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 150 76", 0.0),
    ] {
        games.push((Game::new(fen.parse().unwrap()), value));
    }
    let mut repeated = Game::new(START.parse().unwrap());
    let mut history = Vec::new();
    for _ in 0..2 {
        for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            history.push(play(&mut repeated, mv));
        }
    }
    assert_eq!(repeated.repetition_count(), 3);
    games.push((repeated, 0.0));
    for (mut game, value) in games {
        let original = *game.position();
        let repetitions = game.repetition_count();
        assert_eq!(
            search(&mut game, &mut evaluator, 32, 1.0).unwrap(),
            SearchReport::Terminal(Value::new(value).unwrap())
        );
        assert_eq!(*game.position(), original);
        assert_eq!(game.repetition_count(), repetitions);
        if repetitions == 3 {
            for &mv in history.iter().rev() {
                assert_eq!(game.undo(), Some(mv));
            }
            assert_eq!(*game.position(), START.parse().unwrap());
            assert_eq!(game.repetition_count(), 1);
        }
        assert_eq!(game.undo(), None);
    }
    assert_eq!(evaluator.calls, 0);
}

#[test]
fn invalid_settings_are_rejected_before_root_evaluation_even_for_terminal_games() {
    let mut evaluator = evaluator(|_, _| panic!("invalid settings must bypass evaluation"));
    for fen in [START, "4k3/8/8/8/8/8/8/4K3 w - - 0 1"] {
        let mut game = Game::new(fen.parse().unwrap());
        let original = *game.position();
        assert!(matches!(
            search(&mut game, &mut evaluator, 0, 1.0),
            Err(SearchError::ZeroSimulations)
        ));
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                search(&mut game, &mut evaluator, 1, invalid),
                Err(SearchError::InvalidExploration)
            ));
        }
        assert_eq!(*game.position(), original);
        assert_eq!(game.undo(), None);
    }
    assert_eq!(evaluator.calls, 0);
}

#[test]
fn root_and_later_failures_preserve_sources_and_restore_the_original_history() {
    for fail_at in [1, 4] {
        for malformed in [false, true] {
            let mut game = Game::new(START.parse().unwrap());
            let mut history = Vec::new();
            for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
                history.push(play(&mut game, mv));
            }
            let original = *game.position();
            assert_eq!(game.repetition_count(), 2);
            let mut call = 0;
            let mut evaluator = evaluator(|_, moves| {
                call += 1;
                if call == fail_at {
                    return if malformed {
                        Ok(evaluation(vec![], 0.0))
                    } else {
                        Err(io::Error::new(io::ErrorKind::TimedOut, "fixture timeout"))
                    };
                }
                let mut weights = vec![0.0; moves.len()];
                weights[0] = 1.0; // Force successive simulations down the same line.
                Ok(evaluation(weights, 0.0))
            });
            let error = search(&mut game, &mut evaluator, 8, 1.0).unwrap_err();
            assert!(std::error::Error::source(&error).is_some());
            let resolution = match (fail_at, error) {
                (1, SearchError::RootResolution(error)) => error,
                (4, SearchError::Simulation(SimulationError::Resolution(error))) => error,
                (_, other) => panic!("unexpected error: {other}"),
            };
            match (malformed, resolution) {
                (false, ResolveError::Evaluator(error)) => {
                    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
                    assert_eq!(error.to_string(), "fixture timeout");
                }
                (
                    true,
                    ResolveError::Expansion(ExpansionError::PolicyLengthMismatch {
                        actual: 0, ..
                    }),
                ) => {}
                (_, other) => panic!("unexpected resolution error: {other}"),
            }
            assert_eq!(evaluator.calls, fail_at);
            assert_eq!(*game.position(), original);
            assert_eq!(game.repetition_count(), 2);
            for mv in history.into_iter().rev() {
                assert_eq!(game.undo(), Some(mv));
            }
            assert_eq!(game.undo(), None);
            assert_eq!(*game.position(), START.parse().unwrap());
            assert_eq!(game.repetition_count(), 1);
        }
    }
}

#[test]
fn unvisited_reports_use_priors_without_inventing_a_visit_distribution() {
    for tied in [false, true] {
        let mut game = Game::new(START.parse().unwrap());
        let original = *game.position();
        let mut evaluator = evaluator(|_, moves| {
            let mut weights = vec![0.0; moves.len()];
            weights[0] = 1.0;
            weights[5] = 4.0;
            if tied {
                weights[7] = 4.0;
            }
            Ok(evaluation(weights, 0.9))
        });
        let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
        let initial = tree.report();
        let SearchReport::Nonterminal {
            best_move,
            simulations,
            moves,
        } = &initial
        else {
            panic!("expected nonterminal report");
        };
        assert_eq!(*simulations, 0);
        assert_eq!(*best_move, moves[5].mv);
        assert!(moves[5].stats.prior() > moves[0].stats.prior());
        if tied {
            assert_eq!(moves[5].stats.prior(), moves[7].stats.prior());
        }
        for entry in moves {
            assert_eq!(entry.stats.visits(), 0);
            assert_eq!(entry.stats.value_sum(), 0.0);
            assert_eq!(entry.stats.mean_value(), 0.0);
            assert_eq!(entry.visit_fraction, None);
        }
        assert_eq!(tree.report(), initial);
        assert_eq!(tree.node_count(), 1);
        assert_eq!(evaluator.calls, 1); // Only root resolution evaluated.

        if !tied {
            // The high-prior move is tried first. Its negative backed-up value
            // sends simulation two to the first move. Equal visits now choose
            // that first move, even though it still has the lower prior.
            for _ in 0..2 {
                tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
            }
            let SearchReport::Nonterminal {
                best_move,
                simulations,
                moves,
            } = tree.report()
            else {
                unreachable!();
            };
            assert_eq!(simulations, 2);
            assert_eq!(best_move, moves[0].mv);
            for index in [0, 5] {
                assert_eq!(moves[index].stats.visits(), 1);
                assert_eq!(moves[index].visit_fraction, Some(0.5));
            }
            assert_eq!(moves[7].visit_fraction, Some(0.0));
        }
        assert_eq!(*game.position(), original);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn reporting_and_resuming_match_an_uninterrupted_fixed_budget_search() {
    let mut game = Game::new(START.parse().unwrap());
    let mut comparison = Game::new(START.parse().unwrap());
    let mut history = Vec::new();
    for coordinates in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        history.push(play(&mut game, coordinates));
        play(&mut comparison, coordinates);
    }
    let original = *game.position();
    let mut evaluator = evaluator(|_, moves| Ok(evaluation(vec![1.0; moves.len()], 0.0)));
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    let mut completed = 0;
    let mut snapshots = Vec::new();
    for checkpoint in [0, 1, 8, 32] {
        while completed < checkpoint {
            tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
            completed += 1;
        }
        let report = tree.report();
        let SearchReport::Nonterminal { simulations, .. } = &report else {
            unreachable!();
        };
        assert_eq!(*simulations, checkpoint);
        assert_eq!(tree.report(), report);
        assert_eq!(evaluator.calls, checkpoint as usize + 1);
        assert_eq!(*game.position(), original);
        assert_eq!(game.repetition_count(), 2);
        snapshots.push(report);
    }
    // Reports own their statistics; later simulations do not rewrite them.
    for (snapshot, expected) in snapshots.iter().zip([0, 1, 8, 32]) {
        let SearchReport::Nonterminal {
            simulations, moves, ..
        } = snapshot
        else {
            unreachable!();
        };
        assert_eq!(*simulations, expected);
        assert_eq!(
            moves
                .iter()
                .map(|entry| u64::from(entry.stats.visits()))
                .sum::<u64>(),
            expected
        );
    }
    let uninterrupted = search(&mut comparison, &mut pyxis::UniformEvaluator, 32, 1.0).unwrap();
    assert_eq!(tree.report(), uninterrupted);
    for mv in history.into_iter().rev() {
        assert_eq!(game.undo(), Some(mv));
        assert_eq!(comparison.undo(), Some(mv));
    }
    assert_eq!(game.undo(), None);
    assert_eq!(comparison.undo(), None);
}

#[test]
fn failed_simulations_leave_the_last_report_available_and_can_be_retried() {
    for fail_at in [2usize, 4] {
        for malformed in [false, true] {
            let mut game = Game::new(START.parse().unwrap());
            let original = *game.position();
            let mut call = 0;
            let mut evaluator = evaluator(|_, moves| {
                call += 1;
                if call == fail_at {
                    return if malformed {
                        Ok(evaluation(vec![], 0.0))
                    } else {
                        Err(io::Error::new(io::ErrorKind::TimedOut, "fixture timeout"))
                    };
                }
                let mut weights = vec![0.0; moves.len()];
                weights[0] = 1.0;
                Ok(evaluation(weights, 0.0))
            });
            let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
            for _ in 0..fail_at - 2 {
                tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
            }
            let before = tree.report();
            assert!(matches!(
                tree.simulate(&mut game, &mut evaluator, 1.0),
                Err(SimulationError::Resolution(_))
            ));
            assert_eq!(tree.report(), before);
            assert_eq!(evaluator.calls, fail_at);
            assert_eq!(*game.position(), original);
            assert_eq!(game.repetition_count(), 1);
            assert_eq!(game.undo(), None);

            tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
            let SearchReport::Nonterminal { simulations, .. } = tree.report() else {
                unreachable!();
            };
            assert_eq!(simulations, (fail_at - 1) as u64);
            assert_eq!(*game.position(), original);
            assert_eq!(game.undo(), None);
        }
    }
}
