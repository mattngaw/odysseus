use crate::{Bitboard, Board, Color, PieceKind, Square, attacks};

// Placement is sufficient for attack queries. These stay crate-private so
// legality checks can inspect a Board without constructing a Position.
impl Board {
    pub(crate) fn attackers_to(&self, square: Square, by: Color) -> Bitboard {
        let board = self;
        let occupied = board.occupied();
        let queens = board.by_kind(PieceKind::Queen);

        // Work backward from the target. A white pawn source is one rank below
        // it, hence the opposite-color pawn attacks from the target square.
        let pawns = attacks::pawn_attacks(by.opposite(), Bitboard::from_square(square))
            & board.by_kind(PieceKind::Pawn);
        let knights = attacks::knight_attacks(square) & board.by_kind(PieceKind::Knight);
        let kings = attacks::king_attacks(square) & board.by_kind(PieceKind::King);
        let diagonals =
            attacks::bishop_attacks(square, occupied) & (board.by_kind(PieceKind::Bishop) | queens);
        let orthogonals =
            attacks::rook_attacks(square, occupied) & (board.by_kind(PieceKind::Rook) | queens);

        (pawns | knights | kings | diagonals | orthogonals) & board.by_color(by)
    }

    pub(crate) fn in_check(&self, color: Color) -> bool {
        let king = self
            .pieces(color, PieceKind::King)
            .pop_first()
            .expect("position must contain one king per color");
        !self.attackers_to(king, color.opposite()).is_empty()
    }
}
