use std::{fmt, fmt::Write, str::FromStr};

use crate::{
    Board, CastlingRights, CastlingSide, Color, Piece, PieceKind, Position, PositionError, Square,
};

/// A malformed FEN field or a position that fails structural validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FenError {
    FieldCount,
    PiecePlacement,
    SideToMove,
    CastlingRights,
    EnPassantTarget,
    HalfmoveClock,
    FullmoveNumber,
    InvalidPosition(PositionError),
}

impl FromStr for Position {
    type Err = FenError;

    /// Parses all six fields of standard-chess FEN, then calls `Position::new`.
    /// ASCII whitespace and any ordering of distinct KQkq rights are accepted.
    /// Counters must be unsigned decimal integers fitting in `u32`.
    fn from_str(fen: &str) -> Result<Self, Self::Err> {
        let mut fields = fen.split_ascii_whitespace();
        let mut parts = [""; 6];
        for part in &mut parts {
            *part = fields.next().ok_or(FenError::FieldCount)?;
        }
        if fields.next().is_some() {
            return Err(FenError::FieldCount);
        }
        let board = parse_board(parts[0])?;
        let side = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError::SideToMove),
        };
        let rights = parse_castling(parts[2])?;
        let target = match parts[3].as_bytes() {
            b"-" => None,
            [file @ b'a'..=b'h', rank @ b'1'..=b'8'] => {
                Square::from_coords(file - b'a', rank - b'1')
            }
            _ => return Err(FenError::EnPassantTarget),
        };
        let halfmove = parse_counter(parts[4]).ok_or(FenError::HalfmoveClock)?;
        let fullmove = parse_counter(parts[5]).ok_or(FenError::FullmoveNumber)?;
        Self::new(board, side, rights, target, halfmove, fullmove)
            .map_err(FenError::InvalidPosition)
    }
}

impl Position {
    /// Serializes all six FEN fields with canonical spacing and KQkq ordering.
    /// Preserves the en passant target even when no pawn can capture there.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            if rank != 7 {
                fen.push('/');
            }
            let mut empty = 0u8;
            for file in 0..8 {
                let square = Square::from_coords(file, rank).unwrap();
                if let Some(piece) = self.board().piece_at(square) {
                    if empty != 0 {
                        fen.push(char::from(b'0' + empty));
                        empty = 0;
                    }
                    fen.push(piece.symbol());
                } else {
                    empty += 1;
                }
            }
            if empty != 0 {
                fen.push(char::from(b'0' + empty));
            }
        }
        let side = match self.side_to_move() {
            Color::White => 'w',
            Color::Black => 'b',
        };
        write!(fen, " {side} {} ", self.castling_rights())
            .expect("writing to a String cannot fail");
        match self.en_passant_target() {
            Some(square) => write!(fen, "{square}").expect("writing to a String cannot fail"),
            None => fen.push('-'),
        }
        write!(fen, " {} {}", self.halfmove_clock(), self.fullmove_number())
            .expect("writing to a String cannot fail");
        fen
    }
}

fn parse_board(field: &str) -> Result<Board, FenError> {
    let mut board = Board::empty();
    let mut ranks = field.split('/');
    for rank in (0..8).rev() {
        let row = ranks.next().ok_or(FenError::PiecePlacement)?;
        let mut file = 0u8;
        for byte in row.bytes() {
            if (b'1'..=b'8').contains(&byte) {
                file += byte - b'0';
                if file > 8 {
                    return Err(FenError::PiecePlacement);
                }
            } else {
                let kind = match byte.to_ascii_lowercase() {
                    b'p' => PieceKind::Pawn,
                    b'n' => PieceKind::Knight,
                    b'b' => PieceKind::Bishop,
                    b'r' => PieceKind::Rook,
                    b'q' => PieceKind::Queen,
                    b'k' => PieceKind::King,
                    _ => return Err(FenError::PiecePlacement),
                };
                let color = if byte.is_ascii_uppercase() {
                    Color::White
                } else {
                    Color::Black
                };
                let square = Square::from_coords(file, rank).ok_or(FenError::PiecePlacement)?;
                board.set_piece(square, Piece::new(color, kind));
                file += 1;
            }
        }
        if file != 8 {
            return Err(FenError::PiecePlacement);
        }
    }
    if ranks.next().is_some() {
        return Err(FenError::PiecePlacement);
    }
    Ok(board)
}

fn parse_castling(field: &str) -> Result<CastlingRights, FenError> {
    let mut rights = CastlingRights::NONE;
    if field == "-" {
        return Ok(rights);
    }
    for byte in field.bytes() {
        let (color, side) = match byte {
            b'K' => (Color::White, CastlingSide::Kingside),
            b'Q' => (Color::White, CastlingSide::Queenside),
            b'k' => (Color::Black, CastlingSide::Kingside),
            b'q' => (Color::Black, CastlingSide::Queenside),
            _ => return Err(FenError::CastlingRights),
        };
        if rights.contains(color, side) {
            return Err(FenError::CastlingRights);
        }
        rights.insert(color, side);
    }
    Ok(rights)
}

fn parse_counter(field: &str) -> Option<u32> {
    if field.bytes().all(|byte| byte.is_ascii_digit()) {
        field.parse().ok()
    } else {
        None
    }
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid FEN: ")?;
        match self {
            Self::FieldCount => f.write_str("expected six fields"),
            Self::PiecePlacement => f.write_str(
                "expected eight ranks of eight squares using PNBRQKpnbrqk and digits 1 through 8",
            ),
            Self::SideToMove => f.write_str("side to move must be w or b"),
            Self::CastlingRights => {
                f.write_str("castling rights must be - or distinct KQkq letters")
            }
            Self::EnPassantTarget => {
                f.write_str("en passant target must be - or a lowercase square name")
            }
            Self::HalfmoveClock => f.write_str("halfmove clock must be a decimal u32"),
            Self::FullmoveNumber => f.write_str("fullmove number must be a decimal u32"),
            Self::InvalidPosition(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPosition(error) => Some(error),
            _ => None,
        }
    }
}
