use penteconter::{Color, Game, Move, MoveKind, Position, Square};
use pyxis::{
    Evaluation, Evaluator, Node, PolicyLogitsError, Value, policy_from_logits, resolve_node,
    vocabulary::{POLICY_SIZE, PolicyIndex, index_for_move},
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn moves(fen: &str) -> Vec<Move> {
    let mut legal = Vec::new();
    fen.parse::<Position>()
        .unwrap()
        .generate_legal_moves(&mut legal);
    legal
}

fn slot(mv: Move) -> usize {
    index_for_move(mv, Color::White).unwrap().index()
}

fn near(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
}

fn distribution(policy: &[f32]) {
    assert!(
        policy
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
    );
    let total: f64 = policy.iter().copied().map(f64::from).sum();
    assert!(
        (total - 1.0).abs() <= f64::from(f32::EPSILON),
        "sum: {total}"
    );
}

#[test]
fn known_probability_ratios_follow_the_callers_move_order() {
    let mut legal = moves(START);
    let mut logits = [0.0; POLICY_SIZE];
    logits[322] = 2.0_f32.ln(); // e2e4
    logits[159] = 4.0_f32.ln(); // g1f3
    // 18 moves with weight 1, plus weights 2 and 4: denominator 24.
    for _ in 0..2 {
        let policy = policy_from_logits(&logits, Color::White, &legal).unwrap();
        assert_eq!(policy.len(), legal.len());
        distribution(&policy);
        for (mv, p) in legal.iter().zip(policy) {
            let numerator = match mv.to_string().as_str() {
                "e2e4" => 2.0,
                "g1f3" => 4.0,
                _ => 1.0,
            };
            near(p, numerator / 24.0);
        }
        legal.reverse();
    }
}

#[test]
fn all_unused_logits_are_ignored_including_huge_values_nan_and_infinities() {
    let legal = moves(START);
    let mut baseline = [0.0; POLICY_SIZE];
    for (i, &mv) in legal.iter().enumerate() {
        baseline[slot(mv)] = i as f32 - 10.0;
    }
    let expected = policy_from_logits(&baseline, Color::White, &legal).unwrap();
    for unused in [
        f32::MAX,
        f32::MIN,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let mut logits = [unused; POLICY_SIZE];
        for &mv in &legal {
            logits[slot(mv)] = baseline[slot(mv)];
        }
        assert_eq!(
            policy_from_logits(&logits, Color::White, &legal).unwrap(),
            expected
        );
    }
}

#[test]
fn constant_offsets_leave_probabilities_unchanged() {
    let legal = moves(START);
    let mut baseline = [0.0; POLICY_SIZE];
    for (i, &mv) in legal.iter().enumerate() {
        baseline[slot(mv)] = (i % 5) as f32 - 2.0;
    }
    let expected = policy_from_logits(&baseline, Color::White, &legal).unwrap();
    for offset in [-10000.0, 10000.0] {
        let shifted = baseline.map(|x| x + offset);
        assert_eq!(
            policy_from_logits(&shifted, Color::White, &legal).unwrap(),
            expected
        );
    }
}

#[test]
fn equal_extreme_and_widely_separated_finite_logits_stay_valid() {
    let legal = moves(START);
    for logit in [f32::MIN, -1000.0, 0.0, 1000.0, f32::MAX] {
        let policy = policy_from_logits(&[logit; POLICY_SIZE], Color::White, &legal).unwrap();
        assert_eq!(policy, vec![0.05; 20]);
        distribution(&policy);
    }
    // A selected subset is normalized over just those moves.
    for (selected, expected) in [
        ([f32::MAX, f32::MIN, f32::MAX], [0.5, 0.0, 0.5]),
        ([-1000.0, -2000.0, 0.0], [0.0, 0.0, 1.0]),
    ] {
        let mut logits = [0.0; POLICY_SIZE];
        for (&mv, logit) in legal.iter().zip(selected) {
            logits[slot(mv)] = logit;
        }
        let policy = policy_from_logits(&logits, Color::White, &legal[..3]).unwrap();
        assert_eq!(policy, expected);
        distribution(&policy);
    }
}

#[test]
fn empty_and_single_move_lists_are_well_defined() {
    assert_eq!(
        policy_from_logits(&[f32::NAN; POLICY_SIZE], Color::White, &[]),
        Ok(vec![])
    );
    let legal = moves(START);
    for logit in [f32::MIN, -3.0, 0.0, f32::MAX] {
        let mut logits = [f32::NAN; POLICY_SIZE];
        logits[slot(legal[0])] = logit;
        assert_eq!(
            policy_from_logits(&logits, Color::White, &legal[..1]),
            Ok(vec![1.0])
        );
    }
}

#[test]
fn selected_nonfinite_logits_and_unencodable_moves_report_caller_indices() {
    let legal = moves(START);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for legal_index in [0, 7, 19] {
            let mut logits = [0.0; POLICY_SIZE];
            let policy_index = index_for_move(legal[legal_index], Color::White).unwrap();
            logits[policy_index.index()] = invalid;
            // Later failures do not hide the first one in caller order.
            logits[slot(legal[19])] = invalid;
            assert_eq!(
                policy_from_logits(&logits, Color::White, &legal),
                Err(PolicyLogitsError::NonFiniteLogit {
                    legal_index,
                    policy_index
                })
            );
        }
    }
    let invalid = Move::new(
        Square::new(0).unwrap(),
        Square::new(26).unwrap(),
        MoveKind::Normal,
    )
    .unwrap(); // a1c4
    assert_eq!(
        policy_from_logits(&[0.0; POLICY_SIZE], Color::White, &[legal[0], invalid]),
        Err(PolicyLogitsError::UnencodableMove { legal_index: 1 })
    );
}

#[test]
fn perspective_castling_ep_and_promotions_gather_the_correct_slots() {
    for (fen, side, coordinate, index) in [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            Color::Black,
            "e7e5",
            322,
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            Color::White,
            "e1g1",
            103,
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            Color::Black,
            "e8c8",
            97,
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            Color::White,
            "e5d6",
            1041,
        ),
        (
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
            Color::Black,
            "e4d3",
            1041,
        ),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            Color::White,
            "a7a8n",
            1401,
        ),
        (
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
            Color::Black,
            "a2a1q",
            1792,
        ),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            Color::White,
            "a7a8r",
            1793,
        ),
        (
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
            Color::Black,
            "a2a1b",
            1794,
        ),
    ] {
        let legal = moves(fen);
        let target = legal
            .iter()
            .position(|mv| mv.to_string() == coordinate)
            .unwrap();
        let mut logits = [0.0; POLICY_SIZE];
        logits[index] = 2.0_f32.ln();
        let policy = policy_from_logits(&logits, side, &legal).unwrap();
        distribution(&policy);
        for (i, &p) in policy.iter().enumerate() {
            near(
                p,
                if i == target { 2.0 } else { 1.0 } / (legal.len() + 1) as f32,
            );
        }
    }
}

#[test]
fn evaluator_output_becomes_the_expected_search_edge_priors() {
    struct FixedLogits([f32; POLICY_SIZE]);
    impl Evaluator for FixedLogits {
        type Error = PolicyLogitsError;

        fn evaluate(&mut self, game: &Game, legal: &[Move]) -> Result<Evaluation, Self::Error> {
            Ok(Evaluation {
                value: Value::new(0.25).unwrap(),
                policy_weights: policy_from_logits(&self.0, game.position().side_to_move(), legal)?,
            })
        }
    }
    let mut evaluator = FixedLogits([0.0; POLICY_SIZE]);
    evaluator.0[322] = 4.0_f32.ln();
    // a1b1 is a vocabulary entry but unavailable in the starting position.
    assert_eq!(PolicyIndex::new(0).unwrap().entry().to_string(), "a1b1");
    evaluator.0[0] = f32::MAX;
    let game = Game::new(START.parse().unwrap());
    let Node::Expanded(node) = resolve_node(&game, &mut evaluator).unwrap() else {
        panic!()
    };
    assert_eq!(node.value().get(), 0.25);
    let legal = moves(START);
    assert_eq!(node.edges().len(), legal.len());
    for (&mv, edge) in legal.iter().zip(node.edges()) {
        assert_eq!(edge.mv(), mv);
        near(
            edge.stats().prior(),
            if mv.to_string() == "e2e4" {
                4.0 / 23.0
            } else {
                1.0 / 23.0
            },
        );
    }
}
