use std::fmt;

use crate::{Move, Position};

/// The supplied move is not legal in this position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IllegalMove;

impl Position {
    /// Returns the position after a legal move, leaving this position unchanged.
    ///
    /// The move must match a generated legal move exactly, including its kind
    /// and any promotion piece. Castling and en passant kinds are not inferred
    /// from the source and destination squares.
    ///
    /// Allocates a temporary legal-move list before applying the move. This
    /// checks movement and king safety, not historical reachability or game
    /// outcomes such as draws. Successful transitions maintain the Zobrist key.
    /// Use [`Self::play_unchecked`] when the move is already known to be legal.
    ///
    /// # Errors
    ///
    /// Returns [`IllegalMove`] if the move is absent from the legal-move list.
    ///
    /// # Panics
    ///
    /// Panics if applying a legal move would overflow a u32 move counter, as
    /// with the internal transition routine. Legality is checked first.
    pub fn play(&self, mv: Move) -> Result<Self, IllegalMove> {
        if !self.is_legal_move(mv) {
            return Err(IllegalMove);
        }
        Ok(self.apply(mv))
    }

    /// Returns the position after a move already known to be legal here.
    ///
    /// The caller must supply a legal move for this exact position, including
    /// its kind and promotion piece. A move from this position's legal-move
    /// list satisfies that contract; a move generated for another position may
    /// not. Use [`Self::play`] to validate an arbitrary move description.
    ///
    /// With debug assertions enabled, regenerates the legal-move list and
    /// asserts membership. Otherwise, skips legality validation and its list
    /// allocation. Like `play`, preserves the parent and maintains the child's
    /// Zobrist key. Invalid input may panic or produce an invalid chess position;
    /// this is a chess-correctness contract, not a Rust memory-safety contract.
    ///
    /// # Panics
    ///
    /// Panics on an illegal move when debug assertions are enabled, or if a
    /// legal move would overflow a u32 move counter in any build.
    #[must_use = "returns a new position without changing the original"]
    #[inline]
    pub fn play_unchecked(&self, mv: Move) -> Self {
        debug_assert!(
            self.is_legal_move(mv),
            "move must be legal in this position: {mv:?}"
        );
        self.apply(mv)
    }

    fn is_legal_move(&self, mv: Move) -> bool {
        let mut legal_moves = Vec::new();
        self.generate_legal_moves(&mut legal_moves);
        legal_moves.contains(&mv)
    }
}

impl fmt::Display for IllegalMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("move is not legal in this position")
    }
}

impl std::error::Error for IllegalMove {}
