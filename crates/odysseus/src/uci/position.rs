use std::{fmt, str::SplitWhitespace};

use penteconter::{Color, FenError, Game, PieceKind};

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Parses the words after `position` into a fresh game. Earlier history is
/// unavailable; all supplied moves are replayed, even after a recognized draw.
pub(super) fn parse(mut words: SplitWhitespace<'_>) -> Result<Game, PositionCommandError> {
    let position = match words.next() {
        Some("startpos") => START_FEN.parse().expect("valid starting FEN"),
        Some("fen") => {
            let mut fields = [""; 6];
            for field in &mut fields {
                *field = words
                    .next()
                    .ok_or(PositionCommandError::Fen(FenError::FieldCount))?;
            }
            fields
                .join(" ")
                .parse()
                .map_err(PositionCommandError::Fen)?
        }
        _ => return Err(PositionCommandError::ExpectedStart),
    };
    let mut game = Game::new(position);
    match words.next() {
        None => return Ok(game),
        Some("moves") => {}
        _ => return Err(PositionCommandError::ExpectedMoves),
    }

    let mut legal = Vec::new();
    for (index, text) in words.enumerate() {
        let ply = index + 1;
        let position = game.position();
        legal.clear();
        position.generate_legal_moves(&mut legal);
        // Matching the existing coordinate formatter recovers exact move kinds,
        // including castling and en passant, without a second move encoding.
        let mv = legal
            .iter()
            .copied()
            .find(|mv| mv.to_string() == text)
            .ok_or_else(|| PositionCommandError::IllegalMove {
                ply,
                text: text.to_owned(),
            })?;

        // Core transitions panic on counter overflow. Reject it at the input
        // boundary while still allowing pawn moves/captures to reset the clock.
        let board = position.board();
        let pawn = board.piece_at(mv.from()).unwrap().kind() == PieceKind::Pawn;
        let capture = board.piece_at(mv.to()).is_some();
        if (position.halfmove_clock() == u32::MAX && !pawn && !capture)
            || (position.fullmove_number() == u32::MAX && position.side_to_move() == Color::Black)
        {
            return Err(PositionCommandError::CounterOverflow { ply });
        }
        // The move came from this exact position's legal list.
        game.play_unchecked(mv);
    }
    Ok(game)
}

#[derive(Debug)]
pub(super) enum PositionCommandError {
    ExpectedStart,
    Fen(FenError),
    ExpectedMoves,
    IllegalMove { ply: usize, text: String },
    CounterOverflow { ply: usize },
}

impl fmt::Display for PositionCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpectedStart => f.write_str("expected startpos or fen"),
            Self::Fen(error) => error.fmt(f),
            Self::ExpectedMoves => f.write_str("expected moves before the move list"),
            Self::IllegalMove { ply, text } => write!(f, "illegal move {ply}: {text}"),
            Self::CounterOverflow { ply } => write!(f, "move {ply} would overflow a move counter"),
        }
    }
}

impl std::error::Error for PositionCommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fen(error) => Some(error),
            _ => None,
        }
    }
}
