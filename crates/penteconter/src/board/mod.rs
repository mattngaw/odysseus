//! Piece placement and the board-only operations used for legality checks.

use std::fmt;

use crate::{Bitboard, Color, Piece, PieceKind, Square};

mod attack_queries;
mod placement;

/// Piece placement only; turn, move legality, and game history belong elsewhere.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Board {
    by_color: [Bitboard; 2],
    by_kind: [Bitboard; 6],
    mailbox: [Option<Piece>; 64],
}

impl Board {
    /// Creates an empty placement, including no kings.
    pub const fn empty() -> Self {
        Self {
            by_color: [Bitboard::EMPTY; 2],
            by_kind: [Bitboard::EMPTY; 6],
            mailbox: [None; 64],
        }
    }

    pub const fn piece_at(&self, square: Square) -> Option<Piece> {
        self.mailbox[square.index()]
    }

    pub const fn by_color(&self, color: Color) -> Bitboard {
        self.by_color[color.index()]
    }

    pub const fn by_kind(&self, kind: PieceKind) -> Bitboard {
        self.by_kind[kind.index()]
    }

    /// Returns the squares occupied by pieces of this color and kind.
    pub fn pieces(&self, color: Color, kind: PieceKind) -> Bitboard {
        self.by_color(color) & self.by_kind(kind)
    }

    pub fn occupied(&self) -> Bitboard {
        self.by_color(Color::White) | self.by_color(Color::Black)
    }

    /// Places or replaces a piece, returning the previous occupant.
    /// This changes placement without checking chess legality.
    pub fn set_piece(&mut self, square: Square, piece: Piece) -> Option<Piece> {
        self.set(square, Some(piece))
    }

    /// Removes a piece, returning the previous occupant, if any.
    pub fn remove_piece(&mut self, square: Square) -> Option<Piece> {
        self.set(square, None)
    }

    fn set(&mut self, square: Square, piece: Option<Piece>) -> Option<Piece> {
        let previous = self.mailbox[square.index()];

        // Clear the old membership first, including when the kind is unchanged.
        if let Some(previous) = previous {
            self.by_color[previous.color().index()].remove(square);
            self.by_kind[previous.kind().index()].remove(square);
        }
        if let Some(piece) = piece {
            self.by_color[piece.color().index()].insert(square);
            self.by_kind[piece.kind().index()].insert(square);
        }
        self.mailbox[square.index()] = piece;

        debug_assert!(self.is_consistent());
        previous
    }

    /// Checks agreement between the mailbox and bitboards, not chess legality.
    /// This scans all 64 squares and is also called after updates in debug builds.
    pub fn is_consistent(&self) -> bool {
        let mut by_color = [0u64; 2];
        let mut by_kind = [0u64; 6];
        for (index, piece) in self.mailbox.iter().enumerate() {
            if let Some(piece) = piece {
                by_color[piece.color().index()] |= 1u64 << index;
                by_kind[piece.kind().index()] |= 1u64 << index;
            }
        }
        self.by_color == by_color.map(Bitboard::from_bits)
            && self.by_kind == by_kind.map(Bitboard::from_bits)
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::formatting::grid(f, |square| self.piece_at(square).map_or('.', Piece::symbol))
    }
}

#[cfg(test)]
mod tests;
