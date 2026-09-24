use penteconter::{Game, Move};

use crate::Value;

mod batch;
mod material;
mod uniform;

pub use batch::{BatchEvaluator, EvaluationInput, SequentialBatchEvaluator};
pub use material::MaterialEvaluator;
pub use uniform::UniformEvaluator;

/// The predictions returned by one evaluator call for one position.
#[derive(Debug)]
pub struct Evaluation {
    /// Expected game outcome from the evaluated position's side-to-move perspective.
    pub value: Value,
    /// Unnormalized weights, in the same order as the supplied legal moves.
    ///
    /// There must be one finite, nonnegative weight per move. All-zero weights
    /// are allowed. The consumer must check the length and use
    /// [`crate::normalize_policy`] before treating the weights as priors.
    pub policy_weights: Vec<f32>,
}

/// Supplies policy and value predictions to search, one position at a time.
///
/// Implementations may maintain buffers or other backend state. The game is
/// borrowed immutably; evaluation does not advance or undo the line of play.
pub trait Evaluator {
    /// An implementation-specific evaluation failure. Infallible evaluators can
    /// use [`std::convert::Infallible`].
    type Error: std::error::Error;

    /// Evaluates the current position using the supplied ordered legal moves.
    ///
    /// The caller must supply a game for which [`crate::adjudicate`] returns
    /// `None`, and its complete, nonempty legal-move list. Search handles terminal
    /// results directly, including third occurrences with legal moves remaining.
    /// Implementations must return weights matching this list's length and order,
    /// with the value expressed from the current side-to-move's perspective.
    ///
    /// # Errors
    ///
    /// Returns an implementation-specific error if evaluation cannot complete.
    fn evaluate(&mut self, game: &Game, legal_moves: &[Move]) -> Result<Evaluation, Self::Error>;
}
