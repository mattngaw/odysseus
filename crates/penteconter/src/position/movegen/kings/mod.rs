use crate::{Bitboard, CastlingSide, Color, Move, MoveKind, PieceKind, Position, Square, attacks};

impl Position {
    /// Appends ordinary king moves and standard castling candidates.
    /// Friendly destinations and captures of the opposing king are excluded.
    /// Castling requires its retained right and a clear path between king and
    /// rook; the position's structural invariants guarantee their home placement.
    ///
    /// Attacked squares are not filtered. Legality must check the resulting
    /// king square and, for castling, the starting and transit squares as well.
    /// Ordinary destinations are emitted in ascending index order, followed by
    /// kingside then queenside castling. Existing vector entries are preserved.
    pub(crate) fn generate_king_moves(&self, moves: &mut Vec<Move>) {
        let color = self.side_to_move;
        let from = self
            .board
            .pieces(color, PieceKind::King)
            .pop_first()
            .expect("position must contain one king per color");
        let mut targets = attacks::king_attacks(from)
            & !(self.ours() | self.board.pieces(color.opposite(), PieceKind::King));
        while let Some(to) = targets.pop_first() {
            moves.push(Move::new(from, to, MoveKind::Normal).unwrap());
        }

        let rank = if color == Color::White { 0 } else { 7 };
        let occupied = self.board.occupied();
        for (side, file, path) in [
            (CastlingSide::Kingside, 6, 0x60u64),  // f1, g1 (or f8, g8)
            (CastlingSide::Queenside, 2, 0x0eu64), // b1, c1, d1 (or b8, c8, d8)
        ] {
            if self.castling_rights.contains(color, side)
                && (occupied & Bitboard::from_bits(path << (8 * rank))).is_empty()
            {
                let to = Square::from_coords(file, rank).unwrap();
                moves.push(Move::new(from, to, MoveKind::Castling).unwrap());
            }
        }
    }
}

#[cfg(test)]
mod tests;
