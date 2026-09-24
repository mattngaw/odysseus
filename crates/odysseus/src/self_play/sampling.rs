use std::fmt;

use penteconter::Move;
use pyxis::{RootMove, SearchReport};

/// Probabilities for playing each move, in the report's original order.
///
/// Temperature must be finite and nonnegative. At zero, the first most-visited
/// move has probability one, matching search's tie rule. At positive tau, weights
/// are N^(1/tau), normalized over visited moves; zero-visit moves stay excluded.
/// At tau=1 these are the raw visit fractions. Priors, Q, best_move and the
/// report's rounded visit_fraction fields are not used.
///
/// Requires a nonterminal report with distinct moves and a positive visit sum
/// equal to its reported simulation count. The caller supplies the legal report
/// from the current game; this function has no board on which to check legality.
/// Does not mutate the report or the recorder's untempered training targets.
pub fn move_probabilities(
    report: &SearchReport,
    temperature: f64,
) -> Result<Vec<f64>, SamplingError> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    let moves = validated_moves(report)?;
    let mut best = 0;
    for index in 1..moves.len() {
        if moves[index].stats.visits() > moves[best].stats.visits() {
            best = index;
        }
    }
    if temperature == 0.0 {
        let mut probabilities = vec![0.0; moves.len()];
        probabilities[best] = 1.0;
        return Ok(probabilities);
    }

    let maximum = f64::from(moves[best].stats.visits());
    let mut probabilities: Vec<_> = moves
        .iter()
        .map(|entry| {
            let visits = f64::from(entry.stats.visits());
            if visits == 0.0 {
                0.0
            } else if temperature == 1.0 {
                visits
            } else {
                // Subtract the maximum log-weight before exponentiating. A max
                // count always has weight 1, even for extremely small tau.
                ((visits / maximum).ln() / temperature).exp()
            }
        })
        .collect();
    let total: f64 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

/// A seeded stream for sampling played moves after search.
///
/// Replay requires the same reports, move order, temperatures, and floating-point
/// behavior. Each successful positive-temperature call consumes one random draw;
/// zero temperature and errors do not advance the stream.
#[derive(Debug)]
pub struct MoveSampler {
    state: u64,
}

impl MoveSampler {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn sample(
        &mut self,
        report: &SearchReport,
        temperature: f64,
    ) -> Result<Move, SamplingError> {
        let probabilities = move_probabilities(report, temperature)?;
        let SearchReport::Nonterminal { moves, .. } = report else {
            unreachable!("move_probabilities rejects terminal reports");
        };
        let unit = if temperature == 0.0 {
            0.0
        } else {
            // Exactly representable 53-bit fraction in [0, 1), never 1.
            (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        };
        Ok(moves[sample_index(&probabilities, unit)].mv)
    }

    // SplitMix64, public-domain reference by Sebastiano Vigna:
    // https://prng.di.unimi.it/splitmix64.c
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn validated_moves(report: &SearchReport) -> Result<&[RootMove], SamplingError> {
    let SearchReport::Nonterminal {
        simulations, moves, ..
    } = report
    else {
        return Err(SamplingError::TerminalRoot);
    };
    let mut total = 0u64;
    for (index, entry) in moves.iter().enumerate() {
        if moves[..index]
            .iter()
            .any(|previous| previous.mv == entry.mv)
        {
            return Err(SamplingError::InvalidReport);
        }
        total = total
            .checked_add(u64::from(entry.stats.visits()))
            .ok_or(SamplingError::InvalidReport)?;
    }
    if total != *simulations {
        return Err(SamplingError::InvalidReport);
    }
    if total == 0 {
        return Err(SamplingError::NoVisits);
    }
    Ok(moves)
}

fn sample_index(probabilities: &[f64], unit: f64) -> usize {
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > 0.0 {
            last_positive = Some(index);
            cumulative += probability;
            if unit < cumulative {
                return index;
            }
        }
    }
    // Rounding may leave the cumulative sum slightly below 1. Never fall back
    // to a trailing zero-visit move.
    last_positive.expect("a validated distribution has positive mass")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingError {
    InvalidTemperature,
    TerminalRoot,
    InvalidReport,
    NoVisits,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidTemperature => "temperature must be finite and nonnegative",
            Self::TerminalRoot => "cannot sample a terminal root",
            Self::InvalidReport => "report must have distinct moves and matching visit totals",
            Self::NoVisits => "move sampling requires at least one completed visit",
        })
    }
}

impl std::error::Error for SamplingError {}

#[cfg(test)]
mod tests {
    use super::{MoveSampler, sample_index};

    #[test]
    fn seeded_random_stream_matches_splitmix64_reference() {
        let mut sampler = MoveSampler::new(0);
        assert_eq!(sampler.next_u64(), 0xe220a8397b1dcdaf);
        assert_eq!(sampler.next_u64(), 0x6e789e6aa1b965f4);
        assert_eq!(sampler.next_u64(), 0x06c45d188009454f);
    }

    #[test]
    fn cumulative_boundaries_and_roundoff_never_choose_zero_mass() {
        let probabilities = [0.0, 0.25, 0.0, 0.75, 0.0];
        assert_eq!(sample_index(&probabilities, 0.0), 1);
        assert_eq!(sample_index(&probabilities, 0.25 - f64::EPSILON), 1);
        assert_eq!(sample_index(&probabilities, 0.25), 3);
        assert_eq!(sample_index(&probabilities, 1.0 - f64::EPSILON / 2.0), 3);
        assert_eq!(
            sample_index(&[0.5, 0.5 - f64::EPSILON, 0.0], 1.0 - f64::EPSILON / 2.0),
            1
        );
    }
}
