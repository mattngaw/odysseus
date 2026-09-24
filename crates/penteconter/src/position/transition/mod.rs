use super::{Position, zobrist::piece_square_key};
use crate::{CastlingSide, Color, Move, MoveKind, Piece, PieceKind, Square};

impl Position {
    /// Copies this position and applies a move, including standard castling.
    ///
    /// The caller must supply correct ownership, movement geometry, clear paths,
    /// and move kind, with no friendly/king capture. Promotions must reach the
    /// back rank; en passant must use the current target; castling requires the
    /// retained right, home king/rook, and an empty path between them.
    ///
    /// King safety is not checked. The legality filter must check the
    /// resulting position AND, for castling, the king's starting and transit
    /// squares. Debug assertions catch some contract violations, not all.
    ///
    /// Counter overflow also panics, identically in debug and release, rather
    /// than wrapping metadata from a FEN containing maximal u32 counters.
    /// Maintains the Zobrist key and checks it against full recomputation in
    /// debug builds, including for candidates the legality filter will reject.
    pub(crate) fn apply(&self, mv: Move) -> Self {
        let mut next = *self;
        next.apply_in_place(mv);
        next
    }

    // Shared forward operation. Callers must uphold apply's contracts.
    // Check counter overflow before any mutation so that failure leaves self intact.
    #[inline]
    pub(crate) fn apply_in_place(&mut self, mv: Move) {
        let from = mv.from();
        let to = mv.to();
        let piece = self.board.piece_at(from).expect("source must be occupied");
        let captured = self.board.piece_at(to);
        debug_assert_eq!(piece.color(), self.side_to_move);
        debug_assert!(
            captured.is_none_or(|p| { p.color() != piece.color() && p.kind() != PieceKind::King })
        );

        // The original piece is still a pawn for promotion and en passant,
        // including en passant's capture on a square other than the destination.
        let halfmove_clock = if piece.kind() == PieceKind::Pawn || captured.is_some() {
            0
        } else {
            self.halfmove_clock
                .checked_add(1)
                .expect("halfmove clock overflow")
        };
        let fullmove_number = match self.side_to_move {
            Color::White => self.fullmove_number,
            Color::Black => self
                .fullmove_number
                .checked_add(1)
                .expect("fullmove number overflow"),
        };
        // Remove metadata while it still describes the original placement.
        // Until the final update below, only the placement contribution remains.
        self.zobrist_key ^= self.metadata_zobrist_key();
        self.remove_piece(from);
        match mv.kind() {
            MoveKind::Normal => {
                debug_assert!(piece.kind() != PieceKind::Pawn || (1..7).contains(&to.rank()));
                self.set_piece(to, piece);
            }
            MoveKind::Promotion(kind) => {
                debug_assert_eq!(piece.kind(), PieceKind::Pawn);
                debug_assert_eq!(to.rank(), if piece.color() == Color::White { 7 } else { 0 });
                self.set_piece(to, Piece::new(piece.color(), kind));
            }
            MoveKind::EnPassant => {
                debug_assert_eq!(piece.kind(), PieceKind::Pawn);
                debug_assert_eq!(Some(to), self.en_passant_target);
                debug_assert!(captured.is_none());
                let pawn_square = Square::from_coords(to.file(), from.rank()).unwrap();
                debug_assert_eq!(
                    self.board.piece_at(pawn_square),
                    Some(Piece::new(piece.color().opposite(), PieceKind::Pawn))
                );
                self.remove_piece(pawn_square);
                self.set_piece(to, piece);
            }
            MoveKind::Castling => {
                let rank = if piece.color() == Color::White { 0 } else { 7 };
                let (side, rook_file, rook_destination) = match to.file() {
                    6 => (CastlingSide::Kingside, 7, 5),
                    2 => (CastlingSide::Queenside, 0, 3),
                    _ => panic!("castling king must land on file c or g"),
                };
                debug_assert_eq!(piece.kind(), PieceKind::King);
                debug_assert_eq!(from, Square::from_coords(4, rank).unwrap());
                debug_assert_eq!(to.rank(), rank);
                debug_assert!(self.castling_rights.contains(piece.color(), side));
                let rook_from = Square::from_coords(rook_file, rank).unwrap();
                let rook_to = Square::from_coords(rook_destination, rank).unwrap();
                let rook = Piece::new(piece.color(), PieceKind::Rook);
                debug_assert_eq!(self.board.piece_at(rook_from), Some(rook));
                debug_assert!((rook_file.min(4) + 1..rook_file.max(4)).all(|file| {
                    self.board
                        .piece_at(Square::from_coords(file, rank).unwrap())
                        .is_none()
                }));
                self.remove_piece(rook_from);
                self.set_piece(rook_to, rook);
                self.set_piece(to, piece);
            }
        }

        if piece.kind() == PieceKind::King {
            self.castling_rights
                .remove(piece.color(), CastlingSide::Kingside);
            self.castling_rights
                .remove(piece.color(), CastlingSide::Queenside);
        }
        // In a structurally valid position, a retained right guarantees the
        // corresponding rook occupies its corner. Leaving or capturing on that
        // square therefore revokes the right, regardless of the capturing piece.
        self.revoke_corner_right(from);
        self.revoke_corner_right(to);

        self.en_passant_target = if mv.kind() == MoveKind::Normal
            && piece.kind() == PieceKind::Pawn
            && from.index().abs_diff(to.index()) == 16
        {
            // Preserve the FEN target even when no opponent pawn can capture.
            Square::new(((from.index() + to.index()) / 2) as u8)
        } else {
            None
        };
        self.halfmove_clock = halfmove_clock;
        self.fullmove_number = fullmove_number;
        self.side_to_move = self.side_to_move.opposite();
        // Evaluate new EP availability from the new side's perspective.
        self.zobrist_key ^= self.metadata_zobrist_key();
        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist_key());
    }

    // Placement and its hash contribution change together, including captures
    // performed by replacement. Board remains responsible for its own indices.
    fn remove_piece(&mut self, square: Square) {
        if let Some(piece) = self.board.remove_piece(square) {
            self.zobrist_key ^= piece_square_key(piece, square);
        }
    }

    fn set_piece(&mut self, square: Square, piece: Piece) {
        if let Some(captured) = self.board.set_piece(square, piece) {
            self.zobrist_key ^= piece_square_key(captured, square);
        }
        self.zobrist_key ^= piece_square_key(piece, square);
    }

    fn revoke_corner_right(&mut self, square: Square) {
        let (color, side) = match square.index() {
            0 => (Color::White, CastlingSide::Queenside),
            7 => (Color::White, CastlingSide::Kingside),
            56 => (Color::Black, CastlingSide::Queenside),
            63 => (Color::Black, CastlingSide::Kingside),
            _ => return,
        };
        self.castling_rights.remove(color, side);
    }
}

#[cfg(test)]
mod tests;
