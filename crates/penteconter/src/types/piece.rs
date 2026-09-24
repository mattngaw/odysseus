use std::{fmt, num::NonZeroU8};

/// The absolute color of a piece or side to move.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    /// Returns an index into a two-element array: white = 0, black = 1.
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// A piece's kind, independent of its color.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceKind {
    /// Returns an index into a six-element array, in declaration order.
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// A colored piece. Both `Piece` and `Option<Piece>` occupy one byte.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct Piece(NonZeroU8);

impl Piece {
    // Bits 0..=2 encode kind + 1; bit 3 encodes color. Valid encodings are
    // 1..=6 for white and 9..=14 for black. Zero is reserved for Option's None.
    pub const fn new(color: Color, kind: PieceKind) -> Self {
        let encoded = ((color as u8) << 3) | (kind as u8 + 1);
        Self(NonZeroU8::new(encoded).expect("piece encoding is nonzero"))
    }

    pub const fn color(self) -> Color {
        if self.0.get() & 8 == 0 {
            Color::White
        } else {
            Color::Black
        }
    }

    pub const fn kind(self) -> PieceKind {
        match self.0.get() & 7 {
            1 => PieceKind::Pawn,
            2 => PieceKind::Knight,
            3 => PieceKind::Bishop,
            4 => PieceKind::Rook,
            5 => PieceKind::Queen,
            6 => PieceKind::King,
            _ => panic!("invalid piece encoding"),
        }
    }

    pub(crate) fn symbol(self) -> char {
        let symbol = match self.kind() {
            PieceKind::Pawn => 'P',
            PieceKind::Knight => 'N',
            PieceKind::Bishop => 'B',
            PieceKind::Rook => 'R',
            PieceKind::Queen => 'Q',
            PieceKind::King => 'K',
        };
        match self.color() {
            Color::White => symbol,
            Color::Black => symbol.to_ascii_lowercase(),
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Piece")
            .field("color", &self.color())
            .field("kind", &self.kind())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Color, Piece, PieceKind};

    const COLORS: [Color; 2] = [Color::White, Color::Black];
    const KINDS: [PieceKind; 6] = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
        PieceKind::King,
    ];

    #[test]
    fn colors_have_distinct_indices_and_opposites() {
        assert_eq!(Color::White.opposite(), Color::Black);
        assert_eq!(Color::Black.opposite(), Color::White);
        for (index, color) in COLORS.into_iter().enumerate() {
            assert_eq!(color.index(), index);
            assert_eq!(color.opposite().opposite(), color);
        }
    }

    #[test]
    fn kinds_index_the_six_piece_bitboards() {
        for (index, kind) in KINDS.into_iter().enumerate() {
            assert_eq!(kind.index(), index);
        }
    }

    #[test]
    fn all_twelve_pieces_are_distinct_and_round_trip() {
        let mut pieces = Vec::new();
        for color in COLORS {
            for kind in KINDS {
                let piece = Piece::new(color, kind);
                assert_eq!(piece.color(), color);
                assert_eq!(piece.kind(), kind);
                assert!(!pieces.contains(&piece));
                pieces.push(piece);
            }
        }
        assert_eq!(pieces.len(), 12);
    }

    #[test]
    fn construction_and_access_work_in_constants() {
        const PIECE: Piece = Piece::new(Color::Black, PieceKind::Knight);
        const COLOR: Color = PIECE.color();
        const KIND: PieceKind = PIECE.kind();
        assert_eq!(COLOR, Color::Black);
        assert_eq!(KIND, PieceKind::Knight);
    }

    #[test]
    fn optional_piece_fits_in_one_mailbox_byte() {
        assert_eq!(size_of::<Piece>(), 1);
        assert_eq!(align_of::<Piece>(), 1);
        assert_eq!(size_of::<Option<Piece>>(), 1);
        assert_eq!(align_of::<Option<Piece>>(), 1);
        assert_eq!(size_of::<[Option<Piece>; 64]>(), 64);
    }
}
