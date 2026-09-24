//! Position state, validation, FEN, move generation, transitions, and hashing.

use std::fmt;

use crate::{Bitboard, Board, CastlingRights, CastlingSide, Color, Piece, PieceKind, Square};

mod attack_queries;
mod fen;
mod movegen;
mod play;
mod repetition;
mod terminal;
mod transition;
mod zobrist;

pub use fen::FenError;
pub use play::IllegalMove;
pub(crate) use repetition::RepetitionState;
pub use terminal::TerminalStatus;

#[cfg(feature = "perft-experiment")]
mod experiments;
#[cfg(feature = "perft-experiment")]
pub(crate) use experiments::Undo;

/// Placement and metadata for one position, without game history.
///
/// Construction checks a limited set of structural properties, not legality or
/// reachability. Equality compares the full position, including the move
/// counters; it is not the equivalence relation used for repetition detection.
/// The cached Zobrist key is derived from that state and cannot be set separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Position {
    board: Board,
    side_to_move: Color,
    castling_rights: CastlingRights,
    en_passant_target: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
    zobrist_key: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionError {
    KingCount { color: Color, count: u32 },
    PawnOnBackRank,
    InvalidEnPassantTarget { square: Square },
    InconsistentEnPassant { square: Square },
    InconsistentCastling { color: Color, side: CastlingSide },
    ZeroFullmoveNumber,
}

impl Position {
    /// Checks exactly one king per color, no pawns on ranks 1 or 8, a nonzero
    /// fullmove number, and an empty en passant target on rank 6 (White to move)
    /// or rank 3 (Black to move). Retained castling rights require the matching
    /// king and rook on their standard starting squares. An en passant target
    /// requires the opposing pawn at its double-push destination, an empty
    /// starting square, and a zero halfmove clock.
    ///
    /// This does not check attacks, castling path clearance, piece counts beyond
    /// kings, or historical reachability. An en passant target does not promise
    /// that a legal en passant capture exists.
    pub fn new(
        board: Board,
        side_to_move: Color,
        castling_rights: CastlingRights,
        en_passant_target: Option<Square>,
        halfmove_clock: u32,
        fullmove_number: u32,
    ) -> Result<Self, PositionError> {
        for color in [Color::White, Color::Black] {
            let count = board.pieces(color, PieceKind::King).count();
            if count != 1 {
                return Err(PositionError::KingCount { color, count });
            }
        }
        let back_ranks = Bitboard::from_bits(0xff00_0000_0000_00ff);
        if !(board.by_kind(PieceKind::Pawn) & back_ranks).is_empty() {
            return Err(PositionError::PawnOnBackRank);
        }
        if fullmove_number == 0 {
            return Err(PositionError::ZeroFullmoveNumber);
        }
        for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
            for (side, rook_file) in [(CastlingSide::Kingside, 7), (CastlingSide::Queenside, 0)] {
                if castling_rights.contains(color, side) {
                    let king = Square::from_coords(4, rank).unwrap();
                    let rook = Square::from_coords(rook_file, rank).unwrap();
                    if board.piece_at(king) != Some(Piece::new(color, PieceKind::King))
                        || board.piece_at(rook) != Some(Piece::new(color, PieceKind::Rook))
                    {
                        return Err(PositionError::InconsistentCastling { color, side });
                    }
                }
            }
        }
        if let Some(square) = en_passant_target {
            let (rank, pawn_rank, origin_rank) = match side_to_move {
                Color::White => (5, 4, 6),
                Color::Black => (2, 3, 1),
            };
            if square.rank() != rank || board.piece_at(square).is_some() {
                return Err(PositionError::InvalidEnPassantTarget { square });
            }
            let pawn_square = Square::from_coords(square.file(), pawn_rank).unwrap();
            let origin = Square::from_coords(square.file(), origin_rank).unwrap();
            let pawn = Piece::new(side_to_move.opposite(), PieceKind::Pawn);
            if board.piece_at(pawn_square) != Some(pawn)
                || board.piece_at(origin).is_some()
                || halfmove_clock != 0
            {
                return Err(PositionError::InconsistentEnPassant { square });
            }
        }
        let mut position = Self {
            board,
            side_to_move,
            castling_rights,
            en_passant_target,
            halfmove_clock,
            fullmove_number,
            zobrist_key: 0,
        };
        position.zobrist_key = position.recompute_zobrist_key();
        Ok(position)
    }

    pub const fn board(&self) -> &Board {
        &self.board
    }

    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Occupancy of the side to move, in fixed board coordinates.
    pub const fn ours(&self) -> Bitboard {
        self.board.by_color(self.side_to_move)
    }

    /// Occupancy of the opponent, in fixed board coordinates.
    pub const fn theirs(&self) -> Bitboard {
        self.board.by_color(self.side_to_move.opposite())
    }

    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    pub const fn en_passant_target(&self) -> Option<Square> {
        self.en_passant_target
    }

    /// Halfmoves since the last pawn move or capture.
    pub const fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    /// Starts at one and advances after Black's move.
    pub const fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }
}

/// A board diagram followed by position metadata, for human inspection.
impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}\n", self.board)?;
        writeln!(f, "To move: {:?}", self.side_to_move)?;
        writeln!(f, "Castling: {}", self.castling_rights)?;
        match self.en_passant_target {
            Some(square) => writeln!(f, "En passant: {square}")?,
            None => writeln!(f, "En passant: -")?,
        }
        writeln!(f, "Halfmove clock: {}", self.halfmove_clock)?;
        write!(f, "Fullmove number: {}", self.fullmove_number)
    }
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KingCount { color, count } => {
                write!(f, "expected one {color:?} king, found {count}")
            }
            Self::PawnOnBackRank => f.write_str("a pawn occupies rank 1 or 8"),
            Self::InvalidEnPassantTarget { square } => {
                write!(
                    f,
                    "en passant target {square} is occupied or on the wrong rank"
                )
            }
            Self::ZeroFullmoveNumber => f.write_str("fullmove number must be at least one"),
            Self::InconsistentEnPassant { square } => {
                write!(
                    f,
                    "en passant target {square} contradicts the pawn placement or halfmove clock"
                )
            }
            Self::InconsistentCastling { color, side } => {
                write!(
                    f,
                    "{color:?} {side:?} castling right requires its king and rook on their starting squares"
                )
            }
        }
    }
}

impl std::error::Error for PositionError {}

#[cfg(test)]
mod tests;
