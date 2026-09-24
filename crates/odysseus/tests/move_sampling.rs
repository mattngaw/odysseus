use odysseus::self_play::{MoveSampler, Recorder, SamplingError, move_probabilities};
use penteconter::Game;
use pyxis::{SearchReport, Tree, UniformEvaluator, Value, resolve_node};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn report(counts: &[(&str, u32)]) -> SearchReport {
    let game = Game::new(START.parse().unwrap());
    let mut report = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap()).report();
    let SearchReport::Nonterminal {
        simulations, moves, ..
    } = &mut report
    else {
        unreachable!()
    };
    for &(coordinate, count) in counts {
        let entry = moves
            .iter_mut()
            .find(|entry| entry.mv.to_string() == coordinate)
            .unwrap();
        for _ in 0..count {
            entry.stats.record(Value::new(0.9).unwrap());
        }
        *simulations += u64::from(count);
    }
    // best_move and visit_fraction intentionally remain their pre-search values.
    report
}

fn index(report: &SearchReport, coordinate: &str) -> usize {
    let SearchReport::Nonterminal { moves, .. } = report else {
        unreachable!()
    };
    moves
        .iter()
        .position(|entry| entry.mv.to_string() == coordinate)
        .unwrap()
}

#[test]
fn known_temperature_distributions_preserve_zero_visit_exclusion() {
    let report = report(&[("e2e4", 60), ("d2d4", 30), ("g1f3", 10)]);
    let selected = [
        index(&report, "e2e4"),
        index(&report, "d2d4"),
        index(&report, "g1f3"),
    ];
    let sum_roots = 6f64.sqrt() + 3f64.sqrt() + 1.0;
    for (temperature, expected) in [
        (0.0, [1.0, 0.0, 0.0]),
        (0.5, [36.0 / 46.0, 9.0 / 46.0, 1.0 / 46.0]),
        (1.0, [0.6, 0.3, 0.1]),
        (
            2.0,
            [
                6f64.sqrt() / sum_roots,
                3f64.sqrt() / sum_roots,
                1.0 / sum_roots,
            ],
        ),
    ] {
        let probabilities = move_probabilities(&report, temperature).unwrap();
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for (index, expected) in selected.into_iter().zip(expected) {
            assert!((probabilities[index] - expected).abs() < 1e-12);
        }
        for (index, &probability) in probabilities.iter().enumerate() {
            if !selected.contains(&index) {
                assert_eq!(probability, 0.0);
            }
        }
    }
}

#[test]
fn extreme_temperatures_are_finite_and_zero_temperature_keeps_first_tie() {
    let mut report = report(&[("e2e4", 10), ("d2d4", 10), ("g1f3", 1)]);
    for reverse in [false, true] {
        let SearchReport::Nonterminal { moves, .. } = &mut report else {
            unreachable!()
        };
        if reverse {
            moves.reverse();
        }
        let first_max = moves
            .iter()
            .position(|entry| entry.stats.visits() == 10)
            .unwrap();
        let expected_move = moves[first_max].mv;
        let zero = move_probabilities(&report, 0.0).unwrap();
        assert_eq!(zero[first_max], 1.0);
        assert_eq!(
            MoveSampler::new(42).sample(&report, 0.0).unwrap(),
            expected_move
        );
        for temperature in [f64::from_bits(1), f64::MIN_POSITIVE, f64::MAX] {
            let probabilities = move_probabilities(&report, temperature).unwrap();
            assert!(
                probabilities
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
            );
            assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let expected = if temperature == f64::MAX {
                1.0 / 3.0
            } else {
                0.5
            };
            assert!((probabilities[first_max] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn seeded_replay_and_sampling_frequencies_match_visit_distribution() {
    let report = report(&[("e2e4", 60), ("d2d4", 30), ("g1f3", 10)]);
    let mut first = MoveSampler::new(1234);
    let mut replay = MoveSampler::new(1234);
    let mut different = MoveSampler::new(5678);
    let mut frequencies = [0; 3];
    let mut differs = false;
    for _ in 0..20_000 {
        let chosen = first.sample(&report, 1.0).unwrap();
        assert_eq!(chosen, replay.sample(&report, 1.0).unwrap());
        differs |= chosen != different.sample(&report, 1.0).unwrap();
        frequencies[match chosen.to_string().as_str() {
            "e2e4" => 0,
            "d2d4" => 1,
            "g1f3" => 2,
            unexpected => panic!("sampled unvisited move {unexpected}"),
        }] += 1;
    }
    assert!(differs);
    for (count, expected) in frequencies.into_iter().zip([0.6, 0.3, 0.1]) {
        assert!((f64::from(count) / 20_000.0 - expected).abs() < 0.02);
    }
}

#[test]
fn invalid_inputs_and_zero_temperature_do_not_advance_the_random_stream() {
    let valid = report(&[("e2e4", 6), ("d2d4", 3), ("g1f3", 1)]);
    let mut sampler = MoveSampler::new(42);
    for temperature in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            sampler.sample(&valid, temperature),
            Err(SamplingError::InvalidTemperature)
        );
    }
    assert_eq!(
        sampler.sample(&report(&[]), 1.0),
        Err(SamplingError::NoVisits)
    );
    assert_eq!(
        sampler.sample(&SearchReport::Terminal(Value::new(0.0).unwrap()), 1.0),
        Err(SamplingError::TerminalRoot)
    );
    for defect in 0..3 {
        let mut malformed = report(&[("e2e4", 1)]);
        let SearchReport::Nonterminal {
            simulations, moves, ..
        } = &mut malformed
        else {
            unreachable!()
        };
        match defect {
            0 => *simulations += 1,
            1 => moves[1] = moves[0],
            _ => moves.clear(),
        }
        assert_eq!(
            sampler.sample(&malformed, 1.0),
            Err(SamplingError::InvalidReport)
        );
    }
    sampler.sample(&valid, 0.0).unwrap();
    let mut fresh = MoveSampler::new(42);
    for _ in 0..50 {
        assert_eq!(
            sampler.sample(&valid, 1.0).unwrap(),
            fresh.sample(&valid, 1.0).unwrap()
        );
    }
}

#[test]
fn sampling_does_not_change_reports_or_recorded_training_targets() {
    let report = report(&[("e2e4", 60), ("d2d4", 30), ("g1f3", 10)]);
    let game = Game::new(START.parse().unwrap());
    let mut recorder = Recorder::new();
    recorder.record(&game, &report).unwrap();
    let before = recorder.roots()[0].policy_target();
    let mut sampler = MoveSampler::new(0);
    for temperature in [0.0, 0.5, 1.0, 2.0] {
        sampler.sample(&report, temperature).unwrap();
    }
    let mut after = Recorder::new();
    after.record(&game, &report).unwrap();
    assert_eq!(recorder.roots(), after.roots());
    assert_eq!(before, after.roots()[0].policy_target());
    assert_eq!(before[322], 0.6);
    assert_eq!(before[159], 0.1);
}

#[test]
fn single_visited_move_is_always_selected_at_every_temperature() {
    let report = report(&[("g1f3", 1)]);
    let mut sampler = MoveSampler::new(u64::MAX);
    for temperature in [0.0, f64::MIN_POSITIVE, 0.5, 1.0, 2.0, f64::MAX] {
        assert_eq!(
            sampler.sample(&report, temperature).unwrap().to_string(),
            "g1f3"
        );
    }
}
