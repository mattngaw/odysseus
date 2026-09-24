pub mod attacks;
pub mod perft;

mod board;
mod formatting;
mod game;
mod position;
mod types;

pub use board::Board;
pub use game::{DrawReason, Game, GameOutcome, RecordedPosition};
pub use position::{FenError, IllegalMove, Position, PositionError, TerminalStatus};
pub use types::{
    Bitboard, CastlingRights, CastlingSide, Color, Move, MoveKind, Piece, PieceKind, Square,
};
