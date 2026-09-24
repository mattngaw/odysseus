use std::error::Error;

use penteconter::Game;
use pyxis::{Evaluator, MaterialEvaluator, SearchReport, UniformEvaluator, search};

fn main() -> Result<(), Box<dyn Error>> {
    let simulations = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(128);
    let fen = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1";
    let mut game = Game::new(fen.parse()?);
    let original = *game.position();
    println!("White can capture the undefended queen with Rd1xd5.\nFEN: {fen}\n");
    println!("Material values: P=1, N=B=3, R=5, Q=9; v = delta / (5 + |delta|).");
    let mut legal = Vec::new();
    game.position().generate_legal_moves(&mut legal);
    println!(
        "Before capture: White to move, delta=-4, v={:+.6}",
        MaterialEvaluator.evaluate(&game, &legal)?.value.get()
    );
    let capture = legal
        .iter()
        .find(|mv| mv.to_string() == "d1d5")
        .copied()
        .unwrap();
    game.play(capture)?;
    legal.clear();
    game.position().generate_legal_moves(&mut legal);
    let child_value = MaterialEvaluator.evaluate(&game, &legal)?.value;
    println!(
        "After capture: Black to move, delta=-5, v={:+.6}; White's backed-up sample={:+.6}",
        child_value.get(),
        (-child_value).get()
    );
    game.undo();
    for (label, report) in [
        (
            "UniformEvaluator",
            search(&mut game, &mut UniformEvaluator, simulations, 1.0)?,
        ),
        (
            "MaterialEvaluator",
            search(&mut game, &mut MaterialEvaluator, simulations, 1.0)?,
        ),
    ] {
        if let SearchReport::Nonterminal {
            best_move,
            moves,
            simulations,
        } = report
        {
            println!("\n{label}: {simulations} simulations, bestmove {best_move}");
            println!("move       N        Q        P      N/S");
            for entry in moves {
                println!(
                    "{:5}  {:5}  {:+.4}  {:.4}  {:.4}",
                    entry.mv.to_string(),
                    entry.stats.visits(),
                    entry.stats.mean_value(),
                    entry.stats.prior(),
                    entry.visit_fraction.unwrap()
                );
            }
        }
    }
    assert_eq!(*game.position(), original);
    assert_eq!(game.undo(), None);
    println!(
        "\nBoth searches used the same uniform policy. Material values are diagnostic heuristics."
    );
    Ok(())
}
