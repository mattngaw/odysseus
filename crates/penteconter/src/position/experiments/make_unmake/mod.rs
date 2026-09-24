use crate::{CastlingRights, Move, MoveKind, Piece, PieceKind, Position, Square};

#[cfg(test)]
mod tests;

// Linear-use record: private fields, no Clone/Copy. The caller already has mv.
// Placement is reversed; irreversible metadata and the old key are restored.
#[must_use = "retain the undo record to restore the position"]
pub(crate) struct Undo {
    key: u64,
    halfmove_clock: u32,
    fullmove_number: u32,
    castling_rights: CastlingRights,
    en_passant_target: Option<Square>,
    moved: Piece,
    captured: Option<Piece>,
}

impl Position {
    /// Experimental in-place transition with the same preconditions as apply.
    /// Counter overflow leaves the position unchanged. Other contract violations
    /// are caller errors, with no guarantee of recovery after a debug panic.
    pub(crate) fn make(&mut self, mv: Move) -> Undo {
        let captured_square = if mv.kind() == MoveKind::EnPassant {
            Square::from_coords(mv.to().file(), mv.from().rank()).unwrap()
        } else {
            mv.to()
        };
        let undo = Undo {
            key: self.zobrist_key,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
            castling_rights: self.castling_rights,
            en_passant_target: self.en_passant_target,
            moved: self
                .board
                .piece_at(mv.from())
                .expect("source must be occupied"),
            captured: self.board.piece_at(captured_square),
        };
        self.apply_in_place(mv);
        undo
    }

    /// Restore the immediate child of the paired make. Deeper moves must already
    /// have been undone; mv and undo must come from that exact transition.
    pub(crate) fn unmake(&mut self, mv: Move, undo: Undo) {
        debug_assert_eq!(self.side_to_move, undo.moved.color().opposite());
        debug_assert_eq!(
            self.board.piece_at(mv.to()),
            Some(match mv.kind() {
                MoveKind::Promotion(kind) => Piece::new(undo.moved.color(), kind),
                _ => undo.moved,
            })
        );
        self.board.remove_piece(mv.to());
        self.board.set_piece(mv.from(), undo.moved);
        if mv.kind() == MoveKind::Castling {
            let (home_file, transit_file) = match mv.to().file() {
                6 => (7, 5),
                2 => (0, 3),
                _ => panic!("castling king must land on file c or g"),
            };
            self.board
                .remove_piece(Square::from_coords(transit_file, mv.from().rank()).unwrap());
            self.board.set_piece(
                Square::from_coords(home_file, mv.from().rank()).unwrap(),
                Piece::new(undo.moved.color(), PieceKind::Rook),
            );
        }
        if let Some(captured) = undo.captured {
            let square = if mv.kind() == MoveKind::EnPassant {
                Square::from_coords(mv.to().file(), mv.from().rank()).unwrap()
            } else {
                mv.to()
            };
            self.board.set_piece(square, captured);
        }
        self.side_to_move = undo.moved.color();
        self.castling_rights = undo.castling_rights;
        self.en_passant_target = undo.en_passant_target;
        self.halfmove_clock = undo.halfmove_clock;
        self.fullmove_number = undo.fullmove_number;
        self.zobrist_key = undo.key;
        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist_key());
    }
}
