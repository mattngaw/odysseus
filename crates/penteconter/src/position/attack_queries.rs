use super::Position;
use crate::{Bitboard, Color, Square};

impl Position {
    /// Returns the source squares of `by`'s pieces attacking `square`.
    ///
    /// Uses current occupancy, regardless of whose turn it is. The target may
    /// be empty or occupied by either color. Pinned pieces still attack; pawn
    /// pushes, en passant captures, and castling are not attack geometry.
    pub fn attackers_to(&self, square: Square, by: Color) -> Bitboard {
        self.board.attackers_to(square, by)
    }

    /// Whether `by` attacks `square` under the same rules as `attackers_to`.
    pub fn is_square_attacked(&self, square: Square, by: Color) -> bool {
        !self.attackers_to(square, by).is_empty()
    }

    /// Whether `color`'s king is attacked, independently of the side to move.
    /// After applying a candidate move, query the color that just moved.
    pub fn in_check(&self, color: Color) -> bool {
        self.board.in_check(color)
    }
}
