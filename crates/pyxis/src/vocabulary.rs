//! Fixed policy vocabulary, in Lc0's 1,858-entry order.
//!
//! Squares use the input encoder's current-player perspective: a1 = 0 through
//! h8 = 63, flipping ranks for Black. Entries 0..1792 are source/destination
//! pairs with queen or knight movement geometry, ordered by source then
//! destination index. Entries 1792..1858 add promotions from relative rank 7
//! to rank 8, ordered by source file, destination file, then queen/rook/bishop.
//! A knight promotion uses the ordinary source/destination entry.
//!
//! Standard castling uses king-to-rook entries: e1h1/e1a1. Only this boundary
//! translates Penteconter's e1g1/e1c1 moves. En passant uses its source and actual
//! capture-target square. There are no additional special-move or claim slots.
//!
//! An entry describes geometry, not a legal move or a unique move kind across
//! all positions. Decode by matching the caller's legal moves. This module does
//! not generate legal moves, score actions, or normalize policy logits.
//!
//! Ordering reference: <https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc>.
//! Tables are generated here from geometry, without importing Lc0 source/data.

use std::fmt;

use penteconter::{Color, Move, MoveKind, PieceKind, Square};

use crate::encoding::relative_square;

mod tables;

pub const POLICY_SIZE: usize = 1858;
pub const BASE_MOVE_COUNT: usize = 1792;

/// A checked index into the fixed policy output vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PolicyIndex(u16);

impl PolicyIndex {
    pub const fn new(index: usize) -> Option<Self> {
        if index < POLICY_SIZE {
            Some(Self(index as u16))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Describes this slot in relative coordinates, without position context.
    pub fn entry(self) -> PolicyEntry {
        tables::TABLES.entries[self.index()]
    }
}

/// The relative geometry named by one vocabulary slot.
///
/// `promotion == None` denotes a base slot. It can represent a knight promotion
/// when the legal mover is a promoting pawn, en passant, or king-to-rook castling.
/// The appended slots have an explicit queen, rook, or bishop promotion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyEntry {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
}

/// A vocabulary label, not necessarily a playable UCI move in a given position.
impl fmt::Display for PolicyEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from, self.to)?;
        if let Some(piece) = self.promotion {
            let suffix = match piece {
                PieceKind::Pawn => 'p',
                PieceKind::Knight => 'n',
                PieceKind::Bishop => 'b',
                PieceKind::Rook => 'r',
                PieceKind::Queen => 'q',
                PieceKind::King => 'k',
            };
            write!(f, "{suffix}")?;
        }
        Ok(())
    }
}

/// Maps an absolute move to its policy slot for the supplied side to move.
///
/// All legal standard-chess moves have a slot. `None` rejects unsupported
/// geometry, including malformed promotion, en passant, or castling geometry.
/// `Some` does NOT establish legality: this checks no board, occupancy, rights,
/// or king safety. Callers normally encode their existing legal-move list.
/// Lookup does not allocate or generate moves.
pub fn index_for_move(mv: Move, side_to_move: Color) -> Option<PolicyIndex> {
    let from = relative_square(mv.from(), side_to_move);
    let mut to = relative_square(mv.to(), side_to_move);
    match mv.kind() {
        MoveKind::Normal => {}
        MoveKind::Promotion(piece) => {
            if from.rank() != 6 || to.rank() != 7 || from.file().abs_diff(to.file()) > 1 {
                return None;
            }
            let promotion = match piece {
                PieceKind::Knight => None,
                PieceKind::Queen => Some(0),
                PieceKind::Rook => Some(1),
                PieceKind::Bishop => Some(2),
                PieceKind::Pawn | PieceKind::King => return None,
            };
            if let Some(promotion) = promotion {
                let index =
                    tables::TABLES.promotions[from.file() as usize][to.file() as usize][promotion];
                return PolicyIndex::new(index as usize);
            }
        }
        MoveKind::EnPassant => {
            if from.rank() != 4 || to.rank() != 5 || from.file().abs_diff(to.file()) != 1 {
                return None;
            }
        }
        MoveKind::Castling => {
            if from.index() != 4 || to.rank() != 0 {
                return None;
            }
            to = match to.file() {
                6 => Square::new(7).unwrap(),
                2 => Square::new(0).unwrap(),
                _ => return None,
            };
        }
    }
    PolicyIndex::new(tables::TABLES.base[from.index()][to.index()] as usize)
}

/// Finds this slot's move in a caller-supplied legal list, preserving its kind.
///
/// The list must belong to one position with this side to move. Legal moves in
/// that position have distinct policy indices. This scans only the supplied
/// list, allocating nothing and doing no move generation or legality checking.
/// Returns `None` for slots unavailable in this position, including empty lists.
pub fn move_for_index(
    index: PolicyIndex,
    side_to_move: Color,
    legal_moves: &[Move],
) -> Option<Move> {
    legal_moves
        .iter()
        .copied()
        .find(|&mv| index_for_move(mv, side_to_move) == Some(index))
}
