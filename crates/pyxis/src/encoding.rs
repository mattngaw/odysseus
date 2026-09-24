//! Baseline model input: 64 square tokens, each with 110 `f32` features.
//!
//! Tokens run a1, b1, ..., h8 in the current player's coordinates. For Black,
//! ranks are reversed but files are preserved. All eight frames use this same
//! perspective, including ownership; they do not alternate with historical turns.
//!
//! Frame `t` occupies channels `13*t..13*t+13`, newest first:
//! - offsets 0..6: our pawn, knight, bishop, rook, queen, king;
//! - offsets 6..12: their same six piece types;
//! - offset 12: all ones if this position had occurred before when reached.
//!
//! Missing frames are entirely zero. Current-only metadata occupies channels
//! 104..110: our kingside/queenside rights, their kingside/queenside rights,
//! legal en passant target, and the halfmove clock clamped to 150 and divided
//! by 150. Castling and clock values are repeated at every square. Castling
//! flags represent retained rights, even when castling is currently blocked.
//!
//! This pure representation can encode terminal games for inspection. Callers
//! must apply [`crate::adjudicate`] before requesting model evaluation. Neither
//! policy indexing nor learned positional embeddings are defined here.

use penteconter::{CastlingSide, Color, Game, PieceKind, Square};

pub const SQUARE_COUNT: usize = 64;
pub const HISTORY_FRAMES: usize = 8;
pub const FRAME_FEATURES: usize = 13;
pub const FEATURE_COUNT: usize = 110;

pub const OUR_KINGSIDE_CASTLING: usize = 104;
pub const OUR_QUEENSIDE_CASTLING: usize = 105;
pub const THEIR_KINGSIDE_CASTLING: usize = 106;
pub const THEIR_QUEENSIDE_CASTLING: usize = 107;
pub const EN_PASSANT: usize = 108;
pub const HALFMOVE_CLOCK: usize = 109;

/// Contiguous square-major input, indexed `[relative_square][feature]`.
/// A batch adds an outer dimension: `[batch, 64, 110]`.
pub type EncodedInput = [[f32; FEATURE_COUNT]; SQUARE_COUNT];

/// Converts between absolute and player-relative coordinates. Applying the
/// transform twice restores the original square; Black's a8 becomes a1.
pub const fn relative_square(square: Square, perspective: Color) -> Square {
    match perspective {
        Color::White => square,
        Color::Black => Square::from_coords(square.file(), 7 - square.rank()).unwrap(),
    }
}

/// Encodes the current position and up to seven previous recorded plies.
/// Does not allocate or modify the game. Values are binary except the clock.
pub fn encode(game: &Game) -> EncodedInput {
    let mut output = [[0.0; FEATURE_COUNT]; SQUARE_COUNT];
    encode_into(game, &mut output);
    output
}

/// Overwrites a reusable input buffer, including zeroing unavailable history.
pub fn encode_into(game: &Game, output: &mut EncodedInput) {
    output.fill([0.0; FEATURE_COUNT]);
    let current = game.position();
    let perspective = current.side_to_move();
    for (frame_index, frame) in game.positions().rev().take(HISTORY_FRAMES).enumerate() {
        let base = frame_index * FRAME_FEATURES;
        for (owner, color) in [perspective, perspective.opposite()]
            .into_iter()
            .enumerate()
        {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::King,
            ] {
                let channel = base + owner * 6 + kind.index();
                let mut squares = frame.position.board().pieces(color, kind);
                while let Some(square) = squares.pop_first() {
                    output[relative_square(square, perspective).index()][channel] = 1.0;
                }
            }
        }
        if frame.repetition_count >= 2 {
            for token in output.iter_mut() {
                token[base + 12] = 1.0;
            }
        }
    }

    let rights = current.castling_rights();
    let flags = [
        (OUR_KINGSIDE_CASTLING, perspective, CastlingSide::Kingside),
        (OUR_QUEENSIDE_CASTLING, perspective, CastlingSide::Queenside),
        (
            THEIR_KINGSIDE_CASTLING,
            perspective.opposite(),
            CastlingSide::Kingside,
        ),
        (
            THEIR_QUEENSIDE_CASTLING,
            perspective.opposite(),
            CastlingSide::Queenside,
        ),
    ];
    for token in output.iter_mut() {
        for (channel, color, side) in flags {
            token[channel] = if rights.contains(color, side) {
                1.0
            } else {
                0.0
            };
        }
        token[HALFMOVE_CLOCK] = current.halfmove_clock().min(150) as f32 / 150.0;
    }
    if let Some(target) = current.legal_en_passant_target() {
        output[relative_square(target, perspective).index()][EN_PASSANT] = 1.0;
    }
}
