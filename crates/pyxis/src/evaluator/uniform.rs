use std::convert::Infallible;

use penteconter::{Game, Move};

use super::{Evaluation, Evaluator};
use crate::Value;

/// A placeholder evaluator with a neutral value and equal weights for all moves.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    type Error = Infallible;

    fn evaluate(&mut self, _game: &Game, legal_moves: &[Move]) -> Result<Evaluation, Self::Error> {
        Ok(Evaluation {
            value: Value::new(0.0).expect("zero is a valid evaluation value"),
            policy_weights: vec![1.0; legal_moves.len()],
        })
    }
}
