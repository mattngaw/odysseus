use super::Position;
use crate::{Bitboard, Color, PieceKind};

/// A position with no legal moves: checkmate or stalemate.
///
/// This does not represent other game outcomes or draw claims.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalStatus {
    /// The side to move is in check and has no legal moves.
    Checkmate { winner: Color },
    /// The side to move is not in check and has no legal moves.
    Stalemate,
}

impl Position {
    /// Returns checkmate or stalemate when the side to move has no legal moves.
    ///
    /// `None` means at least one legal move exists, not that the game must
    /// continue: this query does not adjudicate other draws or consult history.
    /// Like move generation, it does not establish historical reachability.
    ///
    /// Generates a complete temporary legal-move list, which may allocate.
    /// Leaves the position unchanged without advancing counters or updating hashes.
    pub fn terminal_status(&self) -> Option<TerminalStatus> {
        let mut moves = Vec::new();
        self.generate_legal_moves(&mut moves);
        if !moves.is_empty() {
            return None;
        }

        Some(if self.in_check(self.side_to_move) {
            TerminalStatus::Checkmate {
                winner: self.side_to_move.opposite(),
            }
        } else {
            TerminalStatus::Stalemate
        })
    }

    /// Whether the remaining material proves that neither side can checkmate.
    ///
    /// Recognizes bare kings, a lone bishop or knight besides the kings, and
    /// kings with any number of bishops all on the same square color, regardless
    /// of ownership. Any pawn, rook, or queen excludes recognition.
    ///
    /// `true` establishes insufficient material. `false` only means these cases
    /// do not apply; this is not a complete dead-position or game-outcome query.
    /// In particular, two knights versus a bare king and opposite-colored
    /// bishops are not recognized: inability to force mate is not sufficient.
    ///
    /// Uses placement bitboards only, without move generation, allocation, or
    /// hashing. Does not depend on turn, counters, or game history.
    pub fn has_insufficient_material(&self) -> bool {
        let board = &self.board;
        if !(board.by_kind(PieceKind::Pawn)
            | board.by_kind(PieceKind::Rook)
            | board.by_kind(PieceKind::Queen))
        .is_empty()
        {
            return false;
        }

        let bishops = board.by_kind(PieceKind::Bishop);
        let knights = board.by_kind(PieceKind::Knight);
        if !knights.is_empty() {
            return bishops.is_empty() && knights.count() == 1;
        }

        // a1 is dark. With no knights, only kings and bishops remain.
        const DARK_SQUARES: Bitboard = Bitboard::from_bits(0xaa55_aa55_aa55_aa55);
        (bishops & DARK_SQUARES).is_empty() || (bishops & !DARK_SQUARES).is_empty()
    }
}
