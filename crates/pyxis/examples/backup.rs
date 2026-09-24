use std::error::Error;

use penteconter::Game;
use pyxis::{ExpandedNode, Node, NodeId, Tree, UniformEvaluator, Value, resolve_node};

fn expanded(tree: &Tree, id: NodeId) -> &ExpandedNode {
    let Node::Expanded(node) = tree.node(id).unwrap() else {
        unreachable!("the scripted line is nonterminal")
    };
    node
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    let start = *game.position();
    let mut evaluator = UniformEvaluator;
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator)?);
    let mut parent = tree.root();
    let mut path = Vec::new();
    // Build a fixed two-ply path for demonstration, rather than run selection.
    for coordinates in ["e2e4", "e7e5"] {
        let edges = expanded(&tree, parent).edges();
        let index = edges
            .iter()
            .position(|edge| edge.mv().to_string() == coordinates)
            .unwrap();
        let mv = edges[index].mv();
        game.play(mv)?;
        path.push((parent, index));
        parent = tree.add_child(parent, index, resolve_node(&game, &mut evaluator)?)?;
    }

    println!("Fixed path: White e2e4 -> Black e7e5 -> White leaf");
    println!("Scripted leaf samples below replace the dummy evaluator's zero.");
    for sample in [0.75, -0.25] {
        println!("\nBacking up {sample:+.2} from White's perspective:");
        tree.backup(&path, Value::new(sample).unwrap())?;
        for &(id, index) in &path {
            let edge = expanded(&tree, id).edges()[index];
            let stats = edge.stats();
            println!(
                "  {}: N={}, W={:+.2}, Q={:+.2}, P={:.2}",
                edge.mv(),
                stats.visits(),
                stats.value_sum(),
                stats.mean_value(),
                stats.prior(),
            );
        }
    }
    for &(id, index) in path.iter().rev() {
        assert_eq!(game.undo(), Some(expanded(&tree, id).edges()[index].mv()));
    }
    assert_eq!(*game.position(), start);
    println!("\nGame restored; tree retains both backed-up samples.");
    Ok(())
}
