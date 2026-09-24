use crate::{Move, MoveKind, Position, Square};

impl Position {
    /// Appends legal moves for the side to move, preserving existing entries
    /// without inspecting them. Surviving candidates retain their generation
    /// order: pawns, N/B/R/Q, ordinary king moves, then castling.
    ///
    /// Filters king safety on copied board placements, including en passant's removal
    /// of the captured pawn. Castling also checks the starting and transit
    /// squares. This does not establish historical reachability or adjudicate
    /// draws; callers can clear and reuse the vector between positions.
    ///
    /// Does not advance counters or compute temporary position hashes.
    pub fn generate_legal_moves(&self, moves: &mut Vec<Move>) {
        let start = moves.len();
        self.generate_pseudo_legal_moves(moves);
        let mut write = start;
        for read in start..moves.len() {
            let mv = moves[read];
            if self.is_legal_candidate(mv) {
                moves[write] = mv;
                write += 1;
            }
        }
        moves.truncate(write);
    }

    // Only for candidates produced by our generators, not arbitrary Move values.
    fn is_legal_candidate(&self, mv: Move) -> bool {
        let mover = self.side_to_move;
        if mv.kind() == MoveKind::Castling {
            if self.in_check(mover) {
                return false;
            }
            let transit = Square::new(((mv.from().index() + mv.to().index()) / 2) as u8).unwrap();
            let step = Move::new(mv.from(), transit, MoveKind::Normal).unwrap();
            // Use the occupancy after the king leaves its source. The rook is
            // still at home during this transit check.
            if self.board.after_move(step).in_check(mover) {
                return false;
            }
        }
        !self.board.after_move(mv).in_check(mover)
    }
}

#[cfg(test)]
mod tests;
