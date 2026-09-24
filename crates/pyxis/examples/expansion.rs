use std::error::Error;

use penteconter::Game;
use pyxis::{Evaluator, ExpandedNode, UniformEvaluator};

fn main() -> Result<(), Box<dyn Error>> {
    let game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    assert_eq!(game.outcome(), None);
    let mut legal_moves = Vec::new();
    game.position().generate_legal_moves(&mut legal_moves);
    let evaluation = UniformEvaluator.evaluate(&game, &legal_moves)?;
    println!("Legal moves: {}", legal_moves.len());
    println!("Raw evaluator weights: {:?}", evaluation.policy_weights);

    let node = ExpandedNode::new(&legal_moves, evaluation)?;
    println!(
        "Node value: {:+.3} from {:?}'s perspective",
        node.value().get(),
        game.position().side_to_move()
    );
    println!(
        "\n{:>5} {:<8} {:>8} {:>4} {:>8} {:>8}",
        "index", "move", "prior P", "N", "W", "Q"
    );
    for (index, edge) in node.edges().iter().enumerate() {
        let stats = edge.stats();
        println!(
            "{index:>5} {:<8} {:>8.5} {:>4} {:>8.3} {:>8.3}",
            edge.mv().to_string(),
            stats.prior(),
            stats.visits(),
            stats.value_sum(),
            stats.mean_value()
        );
    }
    let selected = node
        .select_edge(1.0)
        .expect("valid coefficient and nonempty node");
    println!(
        "\nSelected: {} (index {selected})",
        node.edges()[selected].mv()
    );
    println!("Equal priors at zero visits select the first move in the supplied order.");
    println!("Expansion stores the evaluator value without visiting any outgoing edge.");
    Ok(())
}
