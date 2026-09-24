use super::Position;
use crate::{Move, MoveKind, PieceKind, attacks};

mod kings;
mod legal;
mod pawns;

impl Position {
    /// Appends all candidates in pawn, N/B/R/Q, then king/castling order.
    /// Castling candidates include those with attacked starting/transit squares.
    pub(crate) fn generate_pseudo_legal_moves(&self, moves: &mut Vec<Move>) {
        self.generate_pawn_moves(moves);
        self.generate_knight_and_slider_moves(moves);
        self.generate_king_moves(moves);
    }

    /// Appends pseudo-legal knight, bishop, rook, and queen moves for the side
    /// to move. Pawns and kings are left to subsequent generator components.
    ///
    /// Friendly destinations and captures of the opposing king are excluded.
    /// Pins and check evasions are not filtered. Every emitted move is `Normal`,
    /// whether it is quiet or a capture. Existing entries are preserved; callers
    /// can clear and reuse their vector's allocation between positions.
    ///
    /// Order is N/B/R/Q, then ascending source and destination square indices.
    pub(crate) fn generate_knight_and_slider_moves(&self, moves: &mut Vec<Move>) {
        let occupied = self.board.occupied();
        let destinations = !(self.ours()
            | self
                .board
                .pieces(self.side_to_move.opposite(), PieceKind::King));

        for kind in [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ] {
            let mut sources = self.board.pieces(self.side_to_move, kind);
            while let Some(from) = sources.pop_first() {
                let attacks = match kind {
                    PieceKind::Knight => attacks::knight_attacks(from),
                    PieceKind::Bishop => attacks::bishop_attacks(from, occupied),
                    PieceKind::Rook => attacks::rook_attacks(from, occupied),
                    PieceKind::Queen => attacks::queen_attacks(from, occupied),
                    PieceKind::Pawn | PieceKind::King => unreachable!("separate generators"),
                };
                let mut targets = attacks & destinations;
                while let Some(to) = targets.pop_first() {
                    moves.push(
                        Move::new(from, to, MoveKind::Normal)
                            .expect("attack geometry excludes the source square"),
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
