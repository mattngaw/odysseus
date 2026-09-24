use std::error::Error;

use penteconter::Game;
use pyxis::{ExpandedNode, Node, NodeId, Tree, UniformEvaluator, resolve_node};

fn choices(tree: &Tree, id: NodeId) -> &ExpandedNode {
    let Node::Expanded(node) = tree.node(id).expect("an ID from this tree") else {
        unreachable!("this example evaluates only nonterminal positions")
    };
    node
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    let starting_position = *game.position();
    let mut evaluator = UniformEvaluator;
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator)?);
    let root = tree.root();
    let root_node = choices(&tree, root);
    println!("Root ID: {}", root.index());
    println!(
        "Stored nodes: {}, root moves: {}",
        tree.node_count(),
        root_node.edges().len()
    );

    let edge_index = root_node.select_edge(1.0).unwrap();
    let edge = root_node.edges()[edge_index];
    println!(
        "Selected root edge {edge_index}: {}, child: {:?}",
        edge.mv(),
        edge.child()
    );

    // Perform one explicit play/evaluate/link operation; no simulation or backup.
    game.play_unchecked(edge.mv());
    let child = tree.add_child(root, edge_index, resolve_node(&game, &mut evaluator)?)?;
    println!(
        "\nLinked child ID: {}, side to move: {:?}",
        child.index(),
        game.position().side_to_move()
    );
    println!("Child moves: {}", choices(&tree, child).edges().len());
    let root_node = choices(&tree, root);
    println!(
        "Stored nodes: {}, linked root edges: {}",
        tree.node_count(),
        root_node
            .edges()
            .iter()
            .filter(|edge| edge.child().is_some())
            .count()
    );
    println!(
        "Parent edge visits: {} (linking does not count a simulation)",
        root_node.edges()[edge_index].stats().visits()
    );

    assert_eq!(game.undo(), Some(edge.mv()));
    assert_eq!(*game.position(), starting_position);
    println!(
        "\nGame restored to the root; the tree still contains {} nodes.",
        tree.node_count()
    );
    Ok(())
}
