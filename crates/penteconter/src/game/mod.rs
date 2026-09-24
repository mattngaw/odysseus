//! A current position and the move history recorded from a supplied starting point.

use crate::position::RepetitionState;
use crate::{IllegalMove, Move, Position};
use std::collections::{HashMap, hash_map::Entry};

mod history;
mod outcome;

pub use history::RecordedPosition;
pub use outcome::{DrawReason, GameOutcome};

/// One line of play, with complete position snapshots for undo.
///
/// History begins at the supplied position, regardless of its move counters.
/// Earlier moves cannot be recovered from a FEN. Undo removes the last recorded
/// move; playing again continues from the restored position.
///
/// An occurrence index counts exact repetition states across the recorded line,
/// including the current position. Older entries are retained across pawn moves,
/// captures, and changes to castling rights, so undo can revisit them directly.
///
/// [`Self::outcome`] reports recognized outcomes on demand. Play validates moves
/// independently of outcomes; callers decide when to stop recording play.
#[derive(Debug)]
pub struct Game {
    position: Position,
    history: Vec<HistoryEntry>,
    repetitions: HashMap<RepetitionState, usize>,
}

#[derive(Debug)]
struct HistoryEntry {
    mv: Move,
    before: Position,
    before_repetition_count: usize,
}

impl Game {
    /// Starts a game with no recorded moves, using the supplied position as-is.
    pub fn new(position: Position) -> Self {
        Self {
            position,
            history: Vec::new(),
            repetitions: HashMap::from([(RepetitionState(position), 1)]),
        }
    }

    /// Borrows the current position. Changes go through play or undo so that
    /// the current position and recorded history stay together.
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Number of occurrences of the current repetition state in the recorded
    /// line, including the current position. The supplied starting state counts
    /// as one; earlier history cannot be recovered from its move counters.
    ///
    /// Uses a maintained hash-table index, without scanning history or allocating.
    /// Hash matches are verified using [`Position::same_repetition_state`].
    /// This is an occurrence count, not a draw claim or game-outcome decision.
    pub fn repetition_count(&self) -> usize {
        *self
            .repetitions
            .get(&RepetitionState(self.position))
            .expect("the current position has a recorded occurrence")
    }

    /// Validates a move, records the previous position, and advances the game.
    ///
    /// Uses [`Position::play`], including its exact move-kind matching.
    /// Does not check [`Self::outcome`]: a legal move may be played even after
    /// a draw is reported. The caller decides whether the game should continue.
    ///
    /// # Errors
    ///
    /// Returns [`IllegalMove`] without changing the position, history, or
    /// repetition counts if the move is not legal in the current position.
    ///
    /// # Panics
    ///
    /// Panics if a legal move would overflow a u32 move counter. The position,
    /// history, and repetition counts are unchanged in that case.
    pub fn play(&mut self, mv: Move) -> Result<(), IllegalMove> {
        let next = self.position.play(mv)?;
        self.record(mv, next);
        Ok(())
    }

    /// Records and plays a move already known to be legal in the current position.
    ///
    /// Uses the safe Rust method [`Position::play_unchecked`]: the caller must
    /// supply a legal move for this exact position, including its kind and
    /// promotion piece. Legality is checked only with debug assertions enabled.
    /// Otherwise, invalid input may panic or produce an invalid chess position.
    /// Recording history and indexing occurrences may allocate even when
    /// legality checks are disabled.
    /// Like [`Self::play`], does not check [`Self::outcome`]; the caller decides
    /// whether to continue after a reported draw.
    ///
    /// # Panics
    ///
    /// Panics on an illegal move with debug assertions enabled, or if a legal
    /// move would overflow a u32 move counter in any build. These failures leave
    /// the position, history, and repetition counts unchanged.
    pub fn play_unchecked(&mut self, mv: Move) {
        let next = self.position.play_unchecked(mv);
        self.record(mv, next);
    }

    /// Removes the last recorded move and restores its complete parent position,
    /// including the raw en passant target, move counters, and Zobrist key.
    /// Removes one occurrence of the state being left from the repetition index.
    /// Returns the undone move, or `None` without changes at the starting position.
    pub fn undo(&mut self) -> Option<Move> {
        let entry = self.history.pop()?;
        match self.repetitions.entry(RepetitionState(self.position)) {
            Entry::Occupied(mut occurrence) => {
                if *occurrence.get() == 1 {
                    occurrence.remove();
                } else {
                    *occurrence.get_mut() -= 1;
                }
            }
            Entry::Vacant(_) => unreachable!("the current position has a recorded occurrence"),
        }
        self.position = entry.before;
        Some(entry.mv)
    }

    // Both play paths finish constructing the child before touching history.
    // Store the parent before replacing the current position.
    fn record(&mut self, mv: Move, next: Position) {
        let before_repetition_count = self.repetition_count();
        // Reserve history before changing counts. The final push then cannot
        // allocate after the index has already recorded the new position.
        self.history.reserve(1);
        *self.repetitions.entry(RepetitionState(next)).or_insert(0) += 1;
        self.history.push(HistoryEntry {
            mv,
            before: self.position,
            before_repetition_count,
        });
        self.position = next;
    }
}

#[cfg(test)]
mod tests;
