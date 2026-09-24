use penteconter::Game;

use super::*;
use crate::{Evaluator, UniformEvaluator};

fn game() -> Game {
    Game::new(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap(),
    )
}

fn expanded(game: &Game) -> Node {
    assert_eq!(game.outcome(), None);
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let evaluation = UniformEvaluator.evaluate(game, &moves).unwrap();
    Node::Expanded(ExpandedNode::new(&moves, evaluation).unwrap())
}

fn choices(tree: &Tree, id: NodeId) -> &ExpandedNode {
    let Node::Expanded(node) = tree.node(id).unwrap() else {
        panic!("expected an expanded node")
    };
    node
}

#[test]
fn only_the_root_exists_before_a_move_is_explored() {
    let tree = Tree::new(expanded(&game()));
    assert_eq!(tree.root().index(), 0);
    assert_eq!(tree.node_count(), 1);
    assert!(
        choices(&tree, tree.root())
            .edges()
            .iter()
            .all(|edge| edge.child().is_none())
    );
    assert!(tree.node(NodeId(1)).is_none());
}

#[test]
fn growing_storage_preserves_ids_links_and_unvisited_statistics() {
    let mut game = game();
    let mut tree = Tree::new(expanded(&game));
    let root = tree.root();
    let original_capacity = tree.nodes.capacity();
    let root_edges = choices(&tree, root).edges().to_vec();
    let mut links = Vec::new();

    for (edge_index, edge) in root_edges.iter().enumerate() {
        game.play_unchecked(edge.mv());
        let child = tree.add_child(root, edge_index, expanded(&game)).unwrap();
        assert_eq!(child.index(), edge_index + 1);
        links.push(child);
        assert_eq!(game.undo(), Some(edge.mv()));
    }
    assert!(tree.nodes.capacity() > original_capacity);
    assert_eq!(tree.node_count(), root_edges.len() + 1);
    for (index, edge) in choices(&tree, root).edges().iter().enumerate() {
        assert_eq!(edge.child(), Some(links[index]));
        assert_eq!(edge.mv(), root_edges[index].mv());
        assert_eq!(edge.stats(), root_edges[index].stats());
        assert!(
            choices(&tree, links[index])
                .edges()
                .iter()
                .all(|edge| edge.child().is_none())
        );
    }
}

#[test]
fn descendants_link_to_their_actual_parent() {
    let mut game = game();
    let mut tree = Tree::new(expanded(&game));
    let root = tree.root();
    let first = choices(&tree, root).edges()[0].mv();
    game.play_unchecked(first);
    let child = tree.add_child(root, 0, expanded(&game)).unwrap();
    let reply = choices(&tree, child).edges()[0].mv();
    game.play_unchecked(reply);
    let grandchild = tree.add_child(child, 0, expanded(&game)).unwrap();

    assert_eq!(tree.node_count(), 3);
    assert_eq!(choices(&tree, root).edges()[0].child(), Some(child));
    assert_eq!(choices(&tree, child).edges()[0].child(), Some(grandchild));
    assert!(
        choices(&tree, root).edges()[1..]
            .iter()
            .all(|edge| edge.child().is_none())
    );
}

#[test]
fn rejected_links_leave_the_tree_unchanged() {
    let game = game();
    let mut tree = Tree::new(expanded(&game));
    let root = tree.root();
    for (parent, edge_index, error) in [
        (NodeId(999), 0, AddChildError::UnknownParent),
        (root, usize::MAX, AddChildError::InvalidEdge),
    ] {
        assert_eq!(
            tree.add_child(parent, edge_index, expanded(&game)),
            Err(error)
        );
        assert_eq!(tree.node_count(), 1);
        assert!(
            choices(&tree, root)
                .edges()
                .iter()
                .all(|edge| edge.child().is_none())
        );
    }

    let mut child_game = game;
    child_game.play_unchecked(choices(&tree, root).edges()[0].mv());
    let child = tree.add_child(root, 0, expanded(&child_game)).unwrap();
    assert_eq!(
        tree.add_child(root, 0, expanded(&child_game)),
        Err(AddChildError::ChildAlreadyExists)
    );
    assert_eq!(tree.node_count(), 2);
    assert_eq!(choices(&tree, root).edges()[0].child(), Some(child));
}

#[test]
fn terminal_roots_preserve_their_value_and_cannot_have_children() {
    for raw in [-1.0, 0.0, 1.0] {
        let value = Value::new(raw).unwrap();
        let mut tree = Tree::new(Node::Terminal(value));
        let root = tree.root();
        assert_eq!(
            tree.add_child(root, 0, Node::Terminal(value)),
            Err(AddChildError::TerminalParent)
        );
        assert_eq!(tree.node_count(), 1);
        assert!(matches!(tree.node(root), Some(Node::Terminal(stored)) if *stored == value));
    }
}

#[test]
fn a_terminal_child_stores_its_own_perspective_without_backing_up_a_visit() {
    let mut game = Game::new("k7/8/1QK5/8/8/8/8/8 w - - 0 1".parse().unwrap());
    let mut tree = Tree::new(expanded(&game));
    let root = tree.root();
    let node = choices(&tree, root);
    let edge_index = node
        .edges()
        .iter()
        .position(|edge| edge.mv().to_string() == "b6b7")
        .unwrap();
    let edge = node.edges()[edge_index];
    game.play_unchecked(edge.mv());
    assert!(matches!(
        game.outcome(),
        Some(penteconter::GameOutcome::Checkmate {
            winner: penteconter::Color::White
        })
    ));
    let loss = Value::new(-1.0).unwrap(); // Black, the child's side to move, is mated.
    let child = tree
        .add_child(root, edge_index, Node::Terminal(loss))
        .unwrap();
    assert!(matches!(tree.node(child), Some(Node::Terminal(value)) if *value == loss));
    assert_eq!(
        choices(&tree, root).edges()[edge_index].stats(),
        edge.stats()
    );
}
