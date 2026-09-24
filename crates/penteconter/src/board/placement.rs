use crate::{Board, Move, MoveKind, Piece, PieceKind, Square};

impl Board {
    // Placement-only scratch for generated candidates. The caller supplies
    // correct geometry/ownership and any required castling or EP metadata.
    // Board setters keep the mailbox and all bitboards synchronized.
    pub(crate) fn after_move(&self, mv: Move) -> Self {
        let mut next = *self;
        let piece = next
            .remove_piece(mv.from())
            .expect("source must be occupied");
        match mv.kind() {
            MoveKind::Normal => {
                next.set_piece(mv.to(), piece);
            }
            MoveKind::Promotion(kind) => {
                next.set_piece(mv.to(), Piece::new(piece.color(), kind));
            }
            MoveKind::EnPassant => {
                next.remove_piece(Square::from_coords(mv.to().file(), mv.from().rank()).unwrap());
                next.set_piece(mv.to(), piece);
            }
            MoveKind::Castling => {
                let (from_file, to_file) = match mv.to().file() {
                    6 => (7, 5),
                    2 => (0, 3),
                    _ => panic!("castling king must land on file c or g"),
                };
                next.remove_piece(Square::from_coords(from_file, mv.from().rank()).unwrap());
                next.set_piece(
                    Square::from_coords(to_file, mv.from().rank()).unwrap(),
                    Piece::new(piece.color(), PieceKind::Rook),
                );
                next.set_piece(mv.to(), piece);
            }
        }
        next
    }
}
