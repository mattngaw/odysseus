use penteconter::{Game, Move};

use super::{Evaluation, Evaluator};

/// One borrowed game and the ordered legal moves its policy weights describe.
///
/// As with [`Evaluator::evaluate`], the caller supplies a game nonterminal under
/// [`crate::adjudicate`] and its complete, nonempty legal-move list. No validation
/// happens when constructing this input. The game includes recorded history.
#[derive(Clone, Copy, Debug)]
pub struct EvaluationInput<'a> {
    pub game: &'a Game,
    pub legal_moves: &'a [Move],
}

/// Supplies policy/value predictions for an ordered batch of independent inputs.
///
/// This interface does not select leaves, own pending simulations, or schedule
/// work. Inputs may borrow games and move lists from separate pending requests.
pub trait BatchEvaluator {
    type Error: std::error::Error;

    /// Returns exactly one evaluation per input, in the same order.
    ///
    /// Each value uses its own game's side-to-move perspective. Each policy
    /// contains unnormalized weights matching that input's legal-move order,
    /// with the same validity requirements as [`Evaluator::evaluate`]. Inputs
    /// may have different move counts. Repeated inputs still receive separate
    /// results; an empty input slice succeeds with an empty output vector.
    ///
    /// Consumers must check the returned batch length before pairing results
    /// with inputs, and validate each policy when expanding its node. The trait
    /// signature itself cannot enforce a backend's output length or ordering.
    ///
    /// # Errors
    ///
    /// Returns a backend error without partial results. Evaluator state is not
    /// rolled back. The caller still owns any pending requests and can drop
    /// them all to cancel those simulations; this method does not complete them.
    fn evaluate_batch(
        &mut self,
        inputs: &[EvaluationInput<'_>],
    ) -> Result<Vec<Evaluation>, Self::Error>;
}

/// Adapts an existing single-position evaluator by calling it sequentially.
///
/// Borrows the same evaluator instance, preserving its state across calls. This
/// provides the batch contract for existing evaluators, with no parallelism or
/// inference speedup. A future backend can implement [`BatchEvaluator`] directly.
#[derive(Debug)]
pub struct SequentialBatchEvaluator<'a, E: ?Sized> {
    evaluator: &'a mut E,
}

impl<'a, E: Evaluator + ?Sized> SequentialBatchEvaluator<'a, E> {
    pub fn new(evaluator: &'a mut E) -> Self {
        Self { evaluator }
    }
}

impl<E: Evaluator + ?Sized> BatchEvaluator for SequentialBatchEvaluator<'_, E> {
    type Error = E::Error;

    /// Visits inputs in order, stopping at the first error. Successful earlier
    /// results are discarded on error; later inputs are not evaluated. An empty
    /// batch makes no calls to the wrapped evaluator.
    fn evaluate_batch(
        &mut self,
        inputs: &[EvaluationInput<'_>],
    ) -> Result<Vec<Evaluation>, Self::Error> {
        inputs
            .iter()
            .map(|input| self.evaluator.evaluate(input.game, input.legal_moves))
            .collect()
    }
}
