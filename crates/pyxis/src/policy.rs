use std::fmt;

mod logits;

pub use logits::{PolicyLogitsError, policy_from_logits};

/// A policy weight is negative or nonfinite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidPolicyWeight {
    /// Index of the first invalid weight in the supplied slice.
    pub index: usize,
}

/// Normalizes nonnegative weights for legal moves into priors, in place.
///
/// The caller supplies one weight per legal move, in matching order. This
/// function preserves that order and does not allocate. Nonempty outputs sum
/// to one within floating-point rounding. If every weight is zero, the result
/// is uniform. An empty input stays empty.
///
/// # Errors
///
/// Returns [`InvalidPolicyWeight`] for a negative or nonfinite weight, leaving
/// the entire input unchanged.
pub fn normalize_policy(weights: &mut [f32]) -> Result<(), InvalidPolicyWeight> {
    if weights.is_empty() {
        return Ok(());
    }

    // Accumulate in f64 so even a sum of large, finite f32 weights stays finite.
    let mut total = 0.0_f64;
    for (index, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(InvalidPolicyWeight { index });
        }
        total += f64::from(weight);
    }

    if total == 0.0 {
        weights.fill((1.0 / weights.len() as f64) as f32);
    } else {
        for weight in weights {
            *weight = (f64::from(*weight) / total) as f32;
        }
    }
    Ok(())
}

impl fmt::Display for InvalidPolicyWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "policy weight at index {} must be finite and nonnegative",
            self.index
        )
    }
}

impl std::error::Error for InvalidPolicyWeight {}

#[cfg(test)]
mod tests {
    use super::{InvalidPolicyWeight, normalize_policy};

    #[test]
    fn weights_become_proportional_priors_in_the_same_order() {
        let mut weights = [2.0, 1.0, 0.0, 1.0];
        normalize_policy(&mut weights).unwrap();
        assert_eq!(weights, [0.5, 0.25, 0.0, 0.25]);
    }

    #[test]
    fn all_zero_weights_fall_back_to_uniform_priors() {
        let mut weights = [0.0, -0.0, 0.0];
        normalize_policy(&mut weights).unwrap();
        assert_eq!(weights, [1.0 / 3.0; 3]);
    }

    #[test]
    fn empty_and_single_move_policies_are_valid() {
        assert_eq!(normalize_policy(&mut []), Ok(()));
        for weight in [0.0, 7.0] {
            let mut weights = [weight];
            normalize_policy(&mut weights).unwrap();
            assert_eq!(weights, [1.0]);
        }
    }

    #[test]
    fn invalid_weights_are_rejected_without_changing_the_input() {
        for invalid in [
            -1.0,
            -f32::from_bits(1),
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ] {
            for index in 0..3 {
                let mut weights = [2.0, 1.0, 1.0];
                weights[index] = invalid;
                let original_bits = weights.map(f32::to_bits);
                assert_eq!(
                    normalize_policy(&mut weights),
                    Err(InvalidPolicyWeight { index })
                );
                assert_eq!(weights.map(f32::to_bits), original_bits);
            }
        }
    }

    #[test]
    fn large_weights_do_not_overflow_the_total() {
        let mut weights = [f32::MAX, f32::MAX];
        normalize_policy(&mut weights).unwrap();
        assert_eq!(weights, [0.5, 0.5]);
    }

    #[test]
    fn tiny_positive_weights_keep_their_relative_proportions() {
        let mut weights = [f32::from_bits(1), f32::from_bits(3)];
        normalize_policy(&mut weights).unwrap();
        assert_eq!(weights, [0.25, 0.75]);
    }

    #[test]
    fn rounded_priors_form_a_distribution() {
        let mut weights = [0.0, 1.0, 2.0, 4.0];
        normalize_policy(&mut weights).unwrap();
        for prior in weights {
            assert!(prior.is_finite() && (0.0..=1.0).contains(&prior));
        }
        let sum: f64 = weights.into_iter().map(f64::from).sum();
        assert!((sum - 1.0).abs() <= f64::from(f32::EPSILON));
        assert!(weights[1] < weights[2] && weights[2] < weights[3]);
    }
}
