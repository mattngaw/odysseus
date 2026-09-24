use std::error::Error;

use penteconter::Game;
use pyxis::{Evaluation, Evaluator, UniformEvaluator, normalize_policy};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn main() -> Result<(), Box<dyn Error>> {
    let game = Game::new(START.parse()?);
    println!("{}\n", game.position());

    // The caller handles terminal games and supplies the complete legal list.
    assert_eq!(game.outcome(), None);
    let mut legal_moves = Vec::new();
    game.position().generate_legal_moves(&mut legal_moves);

    let Evaluation {
        value,
        policy_weights: raw_weights,
    } = UniformEvaluator.evaluate(&game, &legal_moves)?;
    assert_eq!(raw_weights.len(), legal_moves.len());
    let mut priors = raw_weights.clone();
    normalize_policy(&mut priors)?;

    println!("Evaluator: UniformEvaluator");
    println!(
        "Value: {:+.3} from {:?}'s perspective (the side to move)",
        value.get(),
        game.position().side_to_move()
    );
    println!("This placeholder's zero value does not establish a draw.");
    println!("Legal moves: {}", legal_moves.len());
    println!(
        "\n{:>5}  {:<8} {:>10} {:>10}",
        "index", "move", "weight", "prior P"
    );
    for (index, ((mv, weight), prior)) in legal_moves
        .iter()
        .zip(&raw_weights)
        .zip(&priors)
        .enumerate()
    {
        println!(
            "{index:>5}  {:<8} {weight:>10.3} {prior:>10.5}",
            mv.to_string()
        );
    }
    println!(
        "Prior sum: {:.6}",
        priors.iter().copied().map(f64::from).sum::<f64>()
    );
    println!("Rows preserve the exact move order supplied to the evaluator.");

    println!("\nNormalization alone: four synthetic legal-move weights, in fixed order.");
    for raw in [[2.0, 1.0, 0.0, 1.0], [0.0; 4]] {
        let mut normalized = raw;
        normalize_policy(&mut normalized)?;
        println!("  {raw:?} -> {normalized:?}");
    }
    println!("All-zero weights fall back to a uniform distribution.");
    Ok(())
}
