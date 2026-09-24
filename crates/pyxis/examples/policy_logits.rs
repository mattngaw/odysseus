use std::error::Error;

use penteconter::Game;
use pyxis::{
    Evaluation, ExpandedNode, Value, policy_from_logits,
    vocabulary::{POLICY_SIZE, index_for_move},
};

fn main() -> Result<(), Box<dyn Error>> {
    let game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    let side = game.position().side_to_move();
    let mut legal = Vec::new();
    game.position().generate_legal_moves(&mut legal);

    // Synthetic model outputs: these are illustrative preferences, not chess evaluation.
    let mut logits = [0.0; POLICY_SIZE];
    logits[322] = 4.0_f32.ln(); // e2e4: weight 4
    logits[159] = 2.0_f32.ln(); // g1f3: weight 2
    let baseline = policy_from_logits(&logits, side, &legal)?;
    logits[0] = 1e30; // a1b1 is in the vocabulary, but illegal here.
    let policy = policy_from_logits(&logits, side, &legal)?;
    assert_eq!(policy, baseline);

    let node = ExpandedNode::new(
        &legal,
        Evaluation {
            value: Value::new(0.0).unwrap(),
            policy_weights: policy.clone(),
        },
    )?;
    println!("1,858 synthetic logits -> 20 legal logits -> softmax -> search priors.");
    println!("e2e4 has weight 4, g1f3 has weight 2, and the other 18 moves have weight 1.");
    println!("The illegal a1b1 logit is 1e30; legal probabilities are exactly unchanged.");
    println!("\nMove   Vocabulary index    Logit   Probability   Edge prior");
    for ((&mv, &p), edge) in legal.iter().zip(&policy).zip(node.edges()) {
        let index = index_for_move(mv, side).unwrap().index();
        println!(
            "{mv}   {index:>16}  {:>7.3}   {p:>11.6}  {:>11.6}",
            logits[index],
            edge.stats().prior()
        );
    }
    println!(
        "Sum: {:.8}",
        policy.iter().copied().map(f64::from).sum::<f64>()
    );

    // Negative logits are valid, and a common offset cancels in softmax.
    for &mv in &legal {
        logits[index_for_move(mv, side).unwrap().index()] -= 10.0;
    }
    let shifted = policy_from_logits(&logits, side, &legal)?;
    let difference = policy
        .iter()
        .zip(shifted)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(difference < 1e-6);
    println!(
        "Subtracting 10 from every legal logit changes probabilities by at most {difference:.2e} (f32 rounding)."
    );
    Ok(())
}
