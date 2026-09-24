use crate::{Bitboard, Color, Move, MoveKind, PieceKind, Position, Square, attacks};

impl Position {
    /// Appends pawn pushes, captures, promotions, and en passant candidates.
    /// Pins and king safety are not checked. En passant relies on the position's
    /// structurally validated target and double-pushed pawn.
    ///
    /// Order is singles, doubles, west captures, east captures; within each
    /// direction ordinary captures precede en passant. Targets are ascending
    /// square indices, and promotions expand in N/B/R/Q order.
    pub(crate) fn generate_pawn_moves(&self, moves: &mut Vec<Move>) {
        let color = self.side_to_move;
        let pawns = self.board.pieces(color, PieceKind::Pawn);
        let empty = !self.board.occupied().bits();
        let (singles, doubles, step) = match color {
            Color::White => {
                let singles = (pawns.bits() << 8) & empty;
                // Only pawns that just reached rank 3 can take a second step.
                let doubles = ((singles & (0xff << 16)) << 8) & empty;
                (singles, doubles, 8)
            }
            Color::Black => {
                let singles = (pawns.bits() >> 8) & empty;
                // Rank 6 is Black's intermediate rank.
                let doubles = ((singles & (0xff << 40)) >> 8) & empty;
                (singles, doubles, -8)
            }
        };
        append_targets(moves, Bitboard::from_bits(singles), step, MoveKind::Normal);
        append_targets(
            moves,
            Bitboard::from_bits(doubles),
            2 * step,
            MoveKind::Normal,
        );

        let capturable = self.theirs() & !self.board.pieces(color.opposite(), PieceKind::King);
        let en_passant = self
            .en_passant_target
            .map_or(Bitboard::EMPTY, Bitboard::from_square);
        // Separate directions preserve both sources when two pawns attack the
        // same destination. West/east are absolute file directions for both colors.
        for (targets, offset) in [
            (attacks::pawn_attacks_west(color, pawns), step - 1),
            (attacks::pawn_attacks_east(color, pawns), step + 1),
        ] {
            append_targets(moves, targets & capturable, offset, MoveKind::Normal);
            append_targets(moves, targets & en_passant, offset, MoveKind::EnPassant);
        }
    }
}

// Every target in a directional set has exactly one source, recovered by
// subtracting that direction's signed square-index offset.
fn append_targets(moves: &mut Vec<Move>, mut targets: Bitboard, offset: i8, kind: MoveKind) {
    while let Some(to) = targets.pop_first() {
        let from = Square::new((to.index() as i8 - offset) as u8)
            .expect("pawn geometry keeps the source on the board");
        if to.rank() == 0 || to.rank() == 7 {
            for promotion in [
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
            ] {
                moves.push(Move::new(from, to, MoveKind::Promotion(promotion)).unwrap());
            }
        } else {
            moves.push(Move::new(from, to, kind).unwrap());
        }
    }
}

#[cfg(test)]
mod tests;
