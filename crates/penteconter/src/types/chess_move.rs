use std::{fmt, num::NonZeroU16};

use crate::{PieceKind, Square};

/// The information needed beyond a move's source and destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveKind {
    /// Includes ordinary captures and double pawn pushes.
    Normal,
    /// The promoted piece; construction rejects pawns and kings.
    Promotion(PieceKind),
    EnPassant,
    /// Source and destination are the king's squares, e.g. e1 to g1.
    Castling,
}

/// A move description, not a guarantee of legality in any position.
///
/// Both `Move` and `Option<Move>` occupy two bytes. Captured pieces and other
/// information needed to reverse a transition belong to the transition state.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct Move(NonZeroU16);

impl Move {
    // Bits 0..=5: source; 6..=11: destination; 12..=13: promotion N/B/R/Q;
    // 14..=15: normal/promotion/en passant/castling. Keep this encoding private.

    /// Returns `None` for identical squares or a pawn/king promotion.
    ///
    /// Does not check movement geometry, occupancy, or king safety. Castling
    /// uses the king's actual destination; same-square Chess960 castling is
    /// outside this representation's current scope. Null moves are not encoded.
    pub const fn new(from: Square, to: Square, kind: MoveKind) -> Option<Self> {
        if from.index() == to.index() {
            return None;
        }
        let flags = match kind {
            MoveKind::Normal => 0,
            MoveKind::Promotion(piece) => {
                let promotion = match piece {
                    PieceKind::Knight => 0,
                    PieceKind::Bishop => 1,
                    PieceKind::Rook => 2,
                    PieceKind::Queen => 3,
                    PieceKind::Pawn | PieceKind::King => return None,
                };
                (1 << 14) | (promotion << 12)
            }
            MoveKind::EnPassant => 2 << 14,
            MoveKind::Castling => 3 << 14,
        };
        let encoded = from.index() as u16 | ((to.index() as u16) << 6) | flags;
        // Distinct source/destination squares cannot both have index zero.
        Some(Self(
            NonZeroU16::new(encoded).expect("move encoding is nonzero"),
        ))
    }

    pub const fn from(self) -> Square {
        Square::new((self.0.get() & 63) as u8).expect("six-bit square index")
    }

    pub const fn to(self) -> Square {
        Square::new(((self.0.get() >> 6) & 63) as u8).expect("six-bit square index")
    }

    pub const fn kind(self) -> MoveKind {
        match self.0.get() >> 14 {
            0 => MoveKind::Normal,
            1 => MoveKind::Promotion(match (self.0.get() >> 12) & 3 {
                0 => PieceKind::Knight,
                1 => PieceKind::Bishop,
                2 => PieceKind::Rook,
                _ => PieceKind::Queen,
            }),
            2 => MoveKind::EnPassant,
            _ => MoveKind::Castling,
        }
    }
}

/// Coordinate notation, e.g. `e2e4` or `e7e8q`, rather than SAN.
///
/// Castling and en passant have no suffix; their kinds require position context
/// to recover from this text.
impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let MoveKind::Promotion(piece) = self.kind() {
            let suffix = match piece {
                PieceKind::Knight => 'n',
                PieceKind::Bishop => 'b',
                PieceKind::Rook => 'r',
                PieceKind::Queen => 'q',
                PieceKind::Pawn | PieceKind::King => unreachable!("invalid promotion"),
            };
            write!(f, "{suffix}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Move")
            .field("from", &self.from())
            .field("to", &self.to())
            .field("kind", &self.kind())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Move, MoveKind};
    use crate::{PieceKind, Square};

    const KINDS: [MoveKind; 7] = [
        MoveKind::Normal,
        MoveKind::Promotion(PieceKind::Knight),
        MoveKind::Promotion(PieceKind::Bishop),
        MoveKind::Promotion(PieceKind::Rook),
        MoveKind::Promotion(PieceKind::Queen),
        MoveKind::EnPassant,
        MoveKind::Castling,
    ];

    #[test]
    fn all_encodable_moves_are_distinct_and_round_trip() {
        let mut seen = [false; 1 << 16];
        for from in 0..64 {
            let from = Square::new(from).unwrap();
            for to in 0..64 {
                let to = Square::new(to).unwrap();
                for kind in KINDS {
                    let result = Move::new(from, to, kind);
                    if from == to {
                        assert_eq!(result, None);
                        continue;
                    }
                    let mv = result.unwrap();
                    assert_eq!((mv.from(), mv.to(), mv.kind()), (from, to, kind));
                    let slot = &mut seen[usize::from(mv.0.get())];
                    assert!(!*slot, "duplicate encoding: {mv:?}");
                    *slot = true;
                }
            }
        }
        assert_eq!(seen.into_iter().filter(|&used| used).count(), 64 * 63 * 7);
    }

    #[test]
    fn pawn_and_king_promotions_are_rejected() {
        let from = Square::from_coords(4, 6).unwrap();
        let to = Square::from_coords(4, 7).unwrap();
        for piece in [PieceKind::Pawn, PieceKind::King] {
            assert_eq!(Move::new(from, to, MoveKind::Promotion(piece)), None);
        }
    }

    #[test]
    fn compact_moves_support_constant_construction_and_access() {
        const FROM: Square = Square::new(12).unwrap();
        const TO: Square = Square::new(28).unwrap();
        const MOVE: Move = Move::new(FROM, TO, MoveKind::Normal).unwrap();
        const PARTS: (Square, Square, MoveKind) = (MOVE.from(), MOVE.to(), MOVE.kind());
        assert_eq!(PARTS, (FROM, TO, MoveKind::Normal));
        assert_eq!(size_of::<Move>(), 2);
        assert_eq!(size_of::<Option<Move>>(), 2);
    }

    #[test]
    fn formatting_shows_coordinates_and_typed_debug_fields() {
        let cases = [
            (12, 28, MoveKind::Normal, "e2e4"),
            (4, 6, MoveKind::Castling, "e1g1"),
            (60, 58, MoveKind::Castling, "e8c8"),
            (36, 43, MoveKind::EnPassant, "e5d6"),
            (52, 60, MoveKind::Promotion(PieceKind::Knight), "e7e8n"),
            (52, 60, MoveKind::Promotion(PieceKind::Bishop), "e7e8b"),
            (52, 60, MoveKind::Promotion(PieceKind::Rook), "e7e8r"),
            (52, 60, MoveKind::Promotion(PieceKind::Queen), "e7e8q"),
            (9, 0, MoveKind::Promotion(PieceKind::Queen), "b2a1q"),
        ];
        for (from, to, kind, expected) in cases {
            let mv = Move::new(Square::new(from).unwrap(), Square::new(to).unwrap(), kind).unwrap();
            assert_eq!(mv.to_string(), expected);
        }
        let mv = Move::new(
            Square::new(12).unwrap(),
            Square::new(28).unwrap(),
            MoveKind::Normal,
        )
        .unwrap();
        assert_eq!(
            format!("{mv:?}"),
            "Move { from: Square(e2), to: Square(e4), kind: Normal }"
        );
    }
}
