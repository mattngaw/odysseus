//! Exact repetition identity, shared with Zobrist feature selection.

use super::Position;
use crate::{Bitboard, Move, MoveKind, PieceKind, Square, attacks};
use std::hash::{Hash, Hasher};

/// An immutable representative of a repetition state, for internal indexes.
/// Ordinary Position equality includes counters and the raw EP target; this
/// wrapper instead uses the repetition identity for both Hash and Eq.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RepetitionState(pub(crate) Position);

impl Hash for RepetitionState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.zobrist_key().hash(state);
    }
}

impl PartialEq for RepetitionState {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_repetition_state(&other.0)
    }
}

impl Eq for RepetitionState {}

impl Position {
    /// Compares the state that determines repetition identity.
    ///
    /// Includes colored piece placement, side to move, retained castling rights,
    /// and the en passant file only when at least one legal capture exists.
    /// Ignores both move counters and any uncapturable raw en passant target.
    /// `Position`'s ordinary field-by-field equality remains unchanged.
    ///
    /// Different cached Zobrist keys reject a match immediately. Equal keys are
    /// followed by exact state comparison, so a hash collision cannot establish
    /// equality. En passant normalization checks at most two candidates per
    /// position, without allocation or advancing move counters.
    ///
    /// This compares two snapshots; it does not count occurrences or adjudicate
    /// a draw, which require game history.
    pub fn same_repetition_state(&self, other: &Self) -> bool {
        self.zobrist_key == other.zobrist_key
            && self.side_to_move == other.side_to_move
            && self.castling_rights == other.castling_rights
            && self.board == other.board
            && self.legal_en_passant_file() == other.legal_en_passant_file()
    }

    // Shared by exact comparison and hashing so both normalize EP identically.
    pub(super) fn legal_en_passant_file(&self) -> Option<u8> {
        self.legal_en_passant_target().map(Square::file)
    }

    /// Returns the capture-target square only when at least one legal en passant
    /// capture exists. Unlike [`Self::en_passant_target`], an uncapturable or
    /// king-exposing FEN target is omitted. This is the same normalization used
    /// by repetition identity and Zobrist hashing.
    ///
    /// Checks at most two captures on scratch boards, without allocation,
    /// generating the full legal-move list, or advancing the position.
    pub fn legal_en_passant_target(&self) -> Option<Square> {
        let target = self.en_passant_target?;
        let mover = self.side_to_move;
        let mut sources = attacks::pawn_attacks(mover.opposite(), Bitboard::from_square(target))
            & self.board.pieces(mover, PieceKind::Pawn);

        while let Some(from) = sources.pop_first() {
            // Position's structural invariants guarantee the target is empty
            // and the opposing double-pushed pawn is present on this square.
            let mv = Move::new(from, target, MoveKind::EnPassant).unwrap();
            let scratch = self.board.after_move(mv);
            // Do not advance counters or recursively invoke hash maintenance.
            if !scratch.in_check(mover) {
                return Some(target);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests;
