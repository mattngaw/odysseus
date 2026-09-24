use super::Game;
use crate::Position;

/// A borrowed position and its occurrence count when that position was reached.
#[derive(Clone, Copy, Debug)]
pub struct RecordedPosition<'a> {
    pub position: &'a Position,
    /// Includes this occurrence, using only history up to this position.
    /// Later occurrences in the current game do not change this value.
    pub repetition_count: usize,
}

impl Game {
    /// Iterates the recorded line from its supplied starting position through
    /// the current position, with each position's historical occurrence count.
    /// Use `.rev()` for newest first, or `.rev().take(n)` for recent positions.
    /// The current position is always present, even before any moves are played.
    /// A FEN's move counters do not create missing history.
    ///
    /// Borrows existing snapshots without copying positions, allocating, or
    /// scanning for repetitions. One current-count lookup happens when creating
    /// the iterator. Stored counts use exact repetition identity, like
    /// [`Self::repetition_count`]. Undo and subsequent play update the viewed line.
    pub fn positions(&self) -> impl DoubleEndedIterator<Item = RecordedPosition<'_>> {
        self.history
            .iter()
            .map(|entry| RecordedPosition {
                position: &entry.before,
                repetition_count: entry.before_repetition_count,
            })
            .chain(std::iter::once(RecordedPosition {
                position: &self.position,
                repetition_count: self.repetition_count(),
            }))
    }
}
