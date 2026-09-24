use penteconter::Game;

use super::*;
use crate::{Edge, ExpandedNode, UniformEvaluator, resolve_node};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn expanded(tree: &Tree, id: NodeId) -> &ExpandedNode {
    let Node::Expanded(node) = tree.node(id).unwrap() else {
        panic!("expected an expanded node")
    };
    node
}

fn snapshot(tree: &Tree) -> Vec<(Value, Vec<Edge>)> {
    tree.nodes
        .iter()
        .map(|node| match node {
            Node::Expanded(node) => (node.value(), node.edges().to_vec()),
            Node::Terminal(value) => (*value, vec![]),
        })
        .collect()
}

fn line(fen: &str, moves: &[&str]) -> (Tree, Vec<(NodeId, usize)>) {
    let mut game = Game::new(fen.parse().unwrap());
    let mut evaluator = UniformEvaluator;
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    let mut parent = tree.root();
    let mut path = Vec::new();
    for &coordinates in moves {
        let edges = expanded(&tree, parent).edges();
        let index = edges
            .iter()
            .position(|edge| edge.mv().to_string() == coordinates)
            .unwrap();
        game.play(edges[index].mv()).unwrap();
        let node = resolve_node(&game, &mut evaluator).unwrap();
        path.push((parent, index));
        parent = tree.add_child(parent, index, node).unwrap();
    }
    (tree, path)
}

#[test]
fn backup_alternates_perspective_at_each_depth() {
    let moves = ["e2e4", "e7e5", "g1f3"];
    for depth in 1..=3 {
        let (mut tree, path) = line(START, &moves[..depth]);
        tree.backup(&path, Value::new(0.75).unwrap()).unwrap();
        for (index, &(parent, edge_index)) in path.iter().enumerate() {
            let stats = expanded(&tree, parent).edges()[edge_index].stats();
            let expected = if (depth - index) % 2 == 0 {
                0.75
            } else {
                -0.75
            };
            assert_eq!(stats.visits(), 1);
            assert_eq!(stats.value_sum(), expected);
            assert_eq!(stats.mean_value(), expected);
        }
    }
}

#[test]
fn repeated_samples_accumulate_only_on_the_traversed_edges() {
    let (mut tree, path) = line(START, &["e2e4", "e7e5"]);
    let before = snapshot(&tree);
    for value in [0.75, -0.25, 0.0] {
        tree.backup(&path, Value::new(value).unwrap()).unwrap();
    }
    let after = snapshot(&tree);
    assert_eq!(after.len(), before.len());
    for (node_index, ((old_value, old_edges), (value, edges))) in
        before.iter().zip(&after).enumerate()
    {
        assert_eq!(value, old_value);
        assert_eq!(edges.len(), old_edges.len());
        for (edge_index, (old, edge)) in old_edges.iter().zip(edges).enumerate() {
            let step = path
                .iter()
                .position(|&(id, index)| id.index() == node_index && index == edge_index);
            if let Some(step) = step {
                assert_eq!(edge.mv(), old.mv());
                assert_eq!(edge.child(), old.child());
                assert_eq!(edge.stats().prior(), old.stats().prior());
                assert_eq!(edge.stats().visits(), 3);
                let sum = if step == 0 { 0.5 } else { -0.5 };
                assert_eq!(edge.stats().value_sum(), sum);
                assert_eq!(edge.stats().mean_value(), sum / 3.0);
            } else {
                assert_eq!(edge, old);
            }
        }
    }
}

#[test]
fn empty_paths_preserve_expanded_and_terminal_roots() {
    for fen in [START, "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1"] {
        let (mut tree, _) = line(fen, &[]);
        let before = snapshot(&tree);
        tree.backup(&[], Value::new(-1.0).unwrap()).unwrap();
        assert_eq!(snapshot(&tree), before);
    }
}

#[test]
fn terminal_leaf_values_back_up_without_changing_the_stored_result() {
    for (mv, leaf_value, parent_value) in [("b6b7", -1.0, 1.0), ("b6c7", 0.0, 0.0)] {
        let (mut tree, path) = line("k7/8/1QK5/8/8/8/8/8 w - - 0 1", &[mv]);
        let (parent, index) = path[0];
        let child = expanded(&tree, parent).edges()[index].child().unwrap();
        let Node::Terminal(value) = *tree.node(child).unwrap() else {
            panic!("expected terminal leaf")
        };
        assert_eq!(value.get(), leaf_value);
        for _ in 0..2 {
            tree.backup(&path, value).unwrap();
        }
        let stats = expanded(&tree, parent).edges()[index].stats();
        assert_eq!(stats.visits(), 2);
        assert_eq!(stats.mean_value(), parent_value);
        assert!(matches!(tree.node(child), Some(Node::Terminal(stored)) if *stored == value));
    }
}

#[test]
fn malformed_paths_leave_every_node_unchanged() {
    let (mut tree, path) = line(START, &["e2e4", "e7e5", "g1f3"]);
    tree.backup(&path, Value::new(0.5).unwrap()).unwrap();
    let before = snapshot(&tree);
    let unlinked = expanded(&tree, path[1].0)
        .edges()
        .iter()
        .position(|edge| edge.child().is_none())
        .unwrap();
    for (bad, error) in [
        (vec![(NodeId(999), 0)], BackupError::UnknownParent),
        (vec![path[1]], BackupError::DisconnectedPath),
        (vec![path[0], path[2]], BackupError::DisconnectedPath),
        (vec![path[0], path[0]], BackupError::DisconnectedPath),
        (vec![path[0], (NodeId(999), 0)], BackupError::UnknownParent),
        (
            vec![path[0], (path[1].0, usize::MAX)],
            BackupError::InvalidEdge,
        ),
        (
            vec![path[0], (path[1].0, unlinked)],
            BackupError::MissingChild,
        ),
    ] {
        assert_eq!(tree.backup(&bad, Value::new(-0.5).unwrap()), Err(error));
        assert_eq!(snapshot(&tree), before);
    }
}

#[test]
fn a_path_cannot_continue_through_a_terminal_node() {
    let (mut tree, mut path) = line("k7/8/1QK5/8/8/8/8/8 w - - 0 1", &["b6b7"]);
    let (parent, index) = path[0];
    let child = expanded(&tree, parent).edges()[index].child().unwrap();
    path.push((child, 0));
    let before = snapshot(&tree);
    assert_eq!(
        tree.backup(&path, Value::new(0.0).unwrap()),
        Err(BackupError::TerminalParent)
    );
    assert_eq!(snapshot(&tree), before);
}
