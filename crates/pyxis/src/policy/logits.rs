use std::fmt;

use penteconter::{Color, Move};

use crate::vocabulary::{POLICY_SIZE, PolicyIndex, index_for_move};

use super::normalize_policy;

/// Gathers vocabulary logits and returns softmax probabilities in legal-move order.
///
/// The caller supplies distinct legal moves from one position with this side to
/// move. This function does not generate or validate legal moves. Normally the
/// list is complete; supplying a subset normalizes over that subset alone.
/// Empty lists return an empty vector. Unselected vocabulary slots are never
/// read, even if they contain NaN or infinity.
///
/// Softmax uses temperature 1 and subtracts the largest selected logit before
/// exponentiating. Nonempty results are finite, nonnegative, and sum to one
/// within f32 rounding. Equal logits produce a uniform distribution. Very small
/// probabilities may underflow to zero. The inputs remain unchanged.
///
/// The returned vector can be used as [`crate::Evaluation::policy_weights`].
/// Expansion still applies its usual weight validation and normalization.
///
/// # Errors
///
/// Returns the first move without a vocabulary slot or selected nonfinite logit,
/// in caller order. Negative finite logits are valid.
pub fn policy_from_logits(
    logits: &[f32; POLICY_SIZE],
    side_to_move: Color,
    legal_moves: &[Move],
) -> Result<Vec<f32>, PolicyLogitsError> {
    let mut policy = Vec::with_capacity(legal_moves.len());
    let mut max = f32::NEG_INFINITY;
    for (legal_index, &mv) in legal_moves.iter().enumerate() {
        let policy_index = index_for_move(mv, side_to_move)
            .ok_or(PolicyLogitsError::UnencodableMove { legal_index })?;
        let logit = logits[policy_index.index()];
        if !logit.is_finite() {
            return Err(PolicyLogitsError::NonFiniteLogit {
                legal_index,
                policy_index,
            });
        }
        max = max.max(logit);
        policy.push(logit);
    }
    for weight in &mut policy {
        // Widen before subtraction so opposite finite f32 extremes do not
        // overflow. Every exponent is <= 0, and the largest weight is 1.
        *weight = (f64::from(*weight) - f64::from(max)).exp() as f32;
    }
    normalize_policy(&mut policy).expect("shifted finite logits produce valid weights");
    Ok(policy)
}

/// A supplied move or its selected vocabulary logit cannot form a policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyLogitsError {
    /// Index into the caller's move list, not the vocabulary.
    UnencodableMove { legal_index: usize },
    /// Both indices identify the first selected NaN or infinity.
    NonFiniteLogit {
        legal_index: usize,
        policy_index: PolicyIndex,
    },
}

impl fmt::Display for PolicyLogitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnencodableMove { legal_index } => {
                write!(
                    f,
                    "move at legal-list index {legal_index} has no policy slot"
                )
            }
            Self::NonFiniteLogit {
                legal_index,
                policy_index,
            } => write!(
                f,
                "policy logit at vocabulary index {} for legal-list index {legal_index} must be finite",
                policy_index.index()
            ),
        }
    }
}

impl std::error::Error for PolicyLogitsError {}
