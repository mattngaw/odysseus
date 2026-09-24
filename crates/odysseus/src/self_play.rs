//! Sequential self-play, move sampling, and training records.

pub mod jsonl;
mod runner;
mod sampling;

pub use runner::{SelfPlayConfig, SelfPlayError, SelfPlayOutcome, SelfPlayResult, play_game};
pub use sampling::{MoveSampler, SamplingError, move_probabilities};

use std::fmt;

use penteconter::{Color, Game, GameOutcome, Position};
use pyxis::{
    Adjudication, SearchReport, adjudicate,
    encoding::{EncodedInput, encode},
    vocabulary::{POLICY_SIZE, PolicyIndex, index_for_move},
};

/// One legal action, including actions with no completed visits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyVisits {
    pub index: PolicyIndex,
    pub visits: u32,
}

/// An owned snapshot taken before playing a move. It has no outcome label yet.
#[derive(Debug, PartialEq)]
pub struct RecordedRoot {
    input: Box<EncodedInput>,
    side_to_move: Color,
    ply: usize,
    policy: Vec<PolicyVisits>,
    total_visits: u64,
}

impl RecordedRoot {
    pub fn input(&self) -> &EncodedInput {
        &self.input
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Ply relative to the Game's supplied starting position, not its FEN clock.
    pub fn ply(&self) -> usize {
        self.ply
    }

    /// All legal slots and raw counts, in the search report's order.
    pub fn policy(&self) -> &[PolicyVisits] {
        &self.policy
    }

    pub fn total_visits(&self) -> u64 {
        self.total_visits
    }

    /// N(a) / sum N, scattered into the fixed vocabulary. No temperature is applied.
    pub fn policy_target(&self) -> [f32; POLICY_SIZE] {
        let mut target = [0.0; POLICY_SIZE];
        for entry in &self.policy {
            target[entry.index.index()] =
                (f64::from(entry.visits) / self.total_visits as f64) as f32;
        }
        target
    }

    /// Distinguishes unvisited legal actions from illegal zero-target slots.
    pub fn legal_mask(&self) -> [bool; POLICY_SIZE] {
        let mut mask = [false; POLICY_SIZE];
        for entry in &self.policy {
            mask[entry.index.index()] = true;
        }
        mask
    }
}

#[derive(Debug, PartialEq)]
pub struct TrainingExample {
    pub root: RecordedRoot,
    /// One-hot [win, draw, loss] for this root's side to move.
    pub value_target: [f32; 3],
}

#[derive(Debug, PartialEq)]
pub struct CompletedGame {
    /// Retains the distinction between an automatic outcome and our draw policy.
    pub adjudication: Adjudication,
    pub examples: Vec<TrainingExample>,
}

/// Collects snapshots along one recorded game line; does not choose or play moves.
///
/// Call record after search and before play. Missing plies are allowed, but each
/// recorded root must advance the same line. Position snapshots of the known
/// history guard against undo, branching, duplicate roots, or unrelated games.
/// Search may temporarily play/undo provided it restores the original Game.
/// The caller must stop play at the first adjudicated outcome; this recorder
/// does not drive the game or retroactively adjudicate skipped plies.
#[derive(Debug, Default)]
pub struct Recorder {
    roots: Vec<RecordedRoot>,
    line: Vec<Position>,
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn roots(&self) -> &[RecordedRoot] {
        &self.roots
    }

    /// Records a nonterminal root with at least one completed simulation.
    ///
    /// The caller must supply the report searched at this exact Game/history:
    /// SearchReport carries no root identity. This validates the complete legal
    /// move set and sum of counts, but cannot establish a report's provenance.
    /// Priors, Q, best_move, and rounded visit_fraction are not training targets.
    /// All errors leave the recorder unchanged.
    pub fn record(&mut self, game: &Game, report: &SearchReport) -> Result<(), RecordError> {
        let line = self.continuation(game)?;
        if line.len() == self.line.len() {
            return Err(RecordError::DuplicateRoot);
        }
        if adjudicate(game).is_some() {
            return Err(RecordError::TerminalRoot);
        }
        let SearchReport::Nonterminal {
            simulations, moves, ..
        } = report
        else {
            return Err(RecordError::InvalidReport);
        };
        let mut legal_moves = Vec::new();
        game.position().generate_legal_moves(&mut legal_moves);
        if moves.len() != legal_moves.len() {
            return Err(RecordError::InvalidReport);
        }
        let side_to_move = game.position().side_to_move();
        let mut seen = [false; POLICY_SIZE];
        let mut policy = Vec::with_capacity(moves.len());
        let mut total_visits = 0u64;
        for entry in moves {
            if !legal_moves.contains(&entry.mv) {
                return Err(RecordError::InvalidReport);
            }
            let index = index_for_move(entry.mv, side_to_move).ok_or(RecordError::InvalidReport)?;
            if std::mem::replace(&mut seen[index.index()], true) {
                return Err(RecordError::InvalidReport);
            }
            let visits = entry.stats.visits();
            total_visits += u64::from(visits);
            policy.push(PolicyVisits { index, visits });
        }
        if total_visits != *simulations {
            return Err(RecordError::InvalidReport);
        }
        if total_visits == 0 {
            return Err(RecordError::NoVisits);
        }
        self.roots.push(RecordedRoot {
            input: Box::new(encode(game)),
            side_to_move,
            ply: line.len() - 1,
            policy,
            total_visits,
        });
        self.line = line;
        Ok(())
    }

    /// Labels and drains the records only when the continuing game is adjudicated.
    ///
    /// Uses Pyxis's shared threefold policy and unchanged 150-halfmove threshold.
    /// Interrupted/truncated games return UnfinishedGame, preserving all records
    /// without labels. All other errors also preserve records. On success the
    /// recorder is empty and can be reused for another game.
    pub fn finish(&mut self, game: &Game) -> Result<CompletedGame, RecordError> {
        if self.roots.is_empty() {
            return Err(RecordError::EmptyRecording);
        }
        self.continuation(game)?;
        let adjudication = adjudicate(game).ok_or(RecordError::UnfinishedGame)?;
        let examples = std::mem::take(&mut self.roots)
            .into_iter()
            .map(|root| {
                let value_target = match adjudication {
                    Adjudication::Automatic(GameOutcome::Checkmate { winner })
                        if winner == root.side_to_move =>
                    {
                        [1.0, 0.0, 0.0]
                    }
                    Adjudication::Automatic(GameOutcome::Checkmate { .. }) => [0.0, 0.0, 1.0],
                    Adjudication::Automatic(GameOutcome::Draw { .. })
                    | Adjudication::ThreefoldRepetitionDraw => [0.0, 1.0, 0.0],
                };
                TrainingExample { root, value_target }
            })
            .collect();
        self.line.clear();
        Ok(CompletedGame {
            adjudication,
            examples,
        })
    }

    fn continuation(&self, game: &Game) -> Result<Vec<Position>, RecordError> {
        let line: Vec<_> = game.positions().map(|frame| *frame.position).collect();
        if !line.starts_with(&self.line) {
            return Err(RecordError::HistoryMismatch);
        }
        Ok(line)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordError {
    TerminalRoot,
    InvalidReport,
    NoVisits,
    HistoryMismatch,
    DuplicateRoot,
    EmptyRecording,
    UnfinishedGame,
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TerminalRoot => "cannot record an adjudicated root",
            Self::InvalidReport => {
                "report must contain every legal move once and matching visit totals"
            }
            Self::NoVisits => "a policy target requires at least one completed visit",
            Self::HistoryMismatch => "game does not continue the recorded history",
            Self::DuplicateRoot => "this root has already been recorded",
            Self::EmptyRecording => "no searched roots have been recorded",
            Self::UnfinishedGame => "game has no adjudicated outcome; records remain unfinished",
        })
    }
}

impl std::error::Error for RecordError {}
