use std::fmt;

use penteconter::{Game, IllegalMove, Move};
use pyxis::{Evaluator, SearchError, adjudicate, search};

use super::{CompletedGame, MoveSampler, RecordError, Recorder, SamplingError};

/// Settings for one sequential run, with a fresh search tree on every ply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfPlayConfig {
    pub simulations_per_move: u32,
    pub exploration: f32,
    /// Fixed throughout this run; affects played moves, not training targets.
    pub temperature: f64,
    pub seed: u64,
    /// Maximum additional plies, excluding the supplied Game's existing history.
    /// Zero is allowed: adjudicate the current state, otherwise truncate immediately.
    pub max_plies: u32,
}

#[derive(Debug)]
pub enum SelfPlayOutcome {
    /// Terminal starting games produce an empty example list without evaluation.
    Completed(CompletedGame),
    /// Ply budget exhausted. Snapshots remain available and have no W/D/L labels.
    Truncated(Recorder),
}

#[derive(Debug)]
pub struct SelfPlayResult {
    /// Actual played moves, excluding both existing history and search variations.
    pub moves: Vec<Move>,
    pub outcome: SelfPlayOutcome,
}

/// Advances a game through search -> record -> sample -> play until it ends or caps.
///
/// Both players use the supplied evaluator. Search restores its temporary moves;
/// only the sampled moves remain in Game. Existing history supplies repetition
/// counts and input frames. The move list and examples cover only this run.
/// Each ply uses a fresh fixed-budget search and the same seeded sampling stream.
/// Replay also requires identical starting history and evaluator behavior/state.
///
/// Pyxis adjudication runs before the cap check, including after the final allowed
/// move. Threefold ends the run; the 150-halfmove rule is unchanged. The cap never
/// supplies an outcome label. No evaluator calls occur for an adjudicated root.
///
/// # Errors
///
/// Validates positive simulations, finite positive exploration, and finite
/// nonnegative temperature before any evaluation or game changes, even at a zero
/// cap or terminal root. Later errors propagate their source; Game retains moves
/// already played and local training records are discarded without finalization.
/// Evaluator state is not rolled back. Core move-counter overflow may still panic.
pub fn play_game<E: Evaluator>(
    game: &mut Game,
    evaluator: &mut E,
    config: SelfPlayConfig,
) -> Result<SelfPlayResult, SelfPlayError<E::Error>> {
    if config.simulations_per_move == 0 {
        return Err(SelfPlayError::ZeroSimulations);
    }
    if !config.exploration.is_finite() || config.exploration <= 0.0 {
        return Err(SelfPlayError::InvalidExploration);
    }
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(SelfPlayError::InvalidTemperature);
    }
    let mut recorder = Recorder::new();
    let mut sampler = MoveSampler::new(config.seed);
    let mut moves = Vec::new();
    loop {
        if let Some(adjudication) = adjudicate(game) {
            let completed = if moves.is_empty() {
                CompletedGame {
                    adjudication,
                    examples: Vec::new(),
                }
            } else {
                recorder.finish(game).map_err(SelfPlayError::Record)?
            };
            return Ok(SelfPlayResult {
                moves,
                outcome: SelfPlayOutcome::Completed(completed),
            });
        }
        if moves.len() >= config.max_plies as usize {
            return Ok(SelfPlayResult {
                moves,
                outcome: SelfPlayOutcome::Truncated(recorder),
            });
        }
        let report = search(
            game,
            evaluator,
            config.simulations_per_move,
            config.exploration,
        )
        .map_err(SelfPlayError::Search)?;
        recorder
            .record(game, &report)
            .map_err(SelfPlayError::Record)?;
        let mv = sampler
            .sample(&report, config.temperature)
            .map_err(SelfPlayError::Sampling)?;
        game.play(mv).map_err(SelfPlayError::IllegalMove)?;
        moves.push(mv);
    }
}

#[derive(Debug)]
pub enum SelfPlayError<E> {
    ZeroSimulations,
    InvalidExploration,
    InvalidTemperature,
    Search(SearchError<E>),
    Record(RecordError),
    Sampling(SamplingError),
    IllegalMove(IllegalMove),
}

impl<E: fmt::Display> fmt::Display for SelfPlayError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSimulations => {
                f.write_str("self-play requires at least one simulation per move")
            }
            Self::InvalidExploration => f.write_str("exploration must be finite and positive"),
            Self::InvalidTemperature => f.write_str("temperature must be finite and nonnegative"),
            Self::Search(error) => write!(f, "self-play search failed: {error}"),
            Self::Record(error) => write!(f, "self-play recording failed: {error}"),
            Self::Sampling(error) => write!(f, "self-play sampling failed: {error}"),
            Self::IllegalMove(error) => write!(f, "self-play move failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SelfPlayError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Search(error) => Some(error),
            Self::Record(error) => Some(error),
            Self::Sampling(error) => Some(error),
            Self::IllegalMove(error) => Some(error),
            _ => None,
        }
    }
}
