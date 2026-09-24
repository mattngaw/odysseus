use penteconter::Position;

use super::*;

fn legal_moves() -> Vec<Move> {
    let position: Position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        .parse()
        .unwrap();
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    moves
}

fn evaluation(weights: Vec<f32>) -> Evaluation {
    Evaluation {
        value: Value::new(-0.625).unwrap(),
        policy_weights: weights,
    }
}

#[test]
fn expansion_preserves_order_and_value_without_recording_outgoing_visits() {
    let mut moves = legal_moves();
    moves.reverse(); // The caller's order, rather than the generator's, is binding.
    let mut weights = vec![0.0; moves.len()];
    weights[..4].copy_from_slice(&[2.0, 0.0, 1.0, 1.0]);
    let node = ExpandedNode::new(&moves, evaluation(weights)).unwrap();

    assert_eq!(node.value().get(), -0.625);
    assert_eq!(node.edges().len(), moves.len());
    for (index, edge) in node.edges().iter().enumerate() {
        assert_eq!(edge.mv(), moves[index]);
        assert_eq!(edge.child(), None);
        let expected_prior = match index {
            0 => 0.5,
            2 | 3 => 0.25,
            _ => 0.0,
        };
        assert_eq!(edge.stats().prior(), expected_prior);
        assert_eq!(edge.stats().visits(), 0);
        assert_eq!(edge.stats().value_sum(), 0.0);
        assert_eq!(edge.stats().mean_value(), 0.0);
    }
}

#[test]
fn all_zero_weights_expand_to_uniform_priors_with_a_stable_first_choice() {
    let moves = legal_moves();
    let node = ExpandedNode::new(&moves, evaluation(vec![0.0; moves.len()])).unwrap();
    assert_eq!(node.edges().len(), 20);
    assert!(node.edges().iter().all(|edge| edge.stats().prior() == 0.05));
    assert_eq!(node.select_edge(1.0), Some(0));
}

#[test]
fn too_few_or_too_many_weights_are_rejected_instead_of_truncated() {
    let moves = legal_moves();
    for actual in [0, moves.len() - 1, moves.len() + 1] {
        let error = ExpandedNode::new(&moves, evaluation(vec![1.0; actual])).unwrap_err();
        assert_eq!(
            error,
            ExpansionError::PolicyLengthMismatch {
                expected: moves.len(),
                actual,
            }
        );
    }
}

#[test]
fn empty_moves_do_not_create_an_expanded_nonterminal_node() {
    assert_eq!(
        ExpandedNode::new(&[], evaluation(vec![])).unwrap_err(),
        ExpansionError::NoLegalMoves
    );
}

#[test]
fn invalid_weights_report_the_original_move_index() {
    let moves = legal_moves();
    for invalid in [-1.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let mut weights = vec![1.0; moves.len()];
        weights[7] = invalid;
        assert_eq!(
            ExpandedNode::new(&moves, evaluation(weights)).unwrap_err(),
            ExpansionError::InvalidPolicyWeight(InvalidPolicyWeight { index: 7 })
        );
    }
}

#[test]
fn node_selection_reads_the_embedded_statistics() {
    let moves = legal_moves();
    let mut weights = vec![0.0; moves.len()];
    weights[..2].copy_from_slice(&[1.0, 3.0]);
    let mut node = ExpandedNode::new(&moves, evaluation(weights)).unwrap();
    assert_eq!(node.select_edge(1.0), Some(1));

    // Script samples directly inside this module; tree backup is a later step.
    node.edges[0].stats.record(Value::new(0.75).unwrap());
    node.edges[1].stats.record(Value::new(-0.75).unwrap());
    assert_eq!(node.select_edge(1.0), Some(0));
    assert_eq!(node.edges()[0].mv(), moves[0]);
    assert_eq!(node.select_edge(f32::NAN), None);
    assert_eq!(node.value().get(), -0.625);
}
