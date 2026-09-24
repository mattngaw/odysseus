use std::fmt;

use crate::Color;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CastlingSide {
    Kingside = 0,
    Queenside = 1,
}

/// Retained castling rights, not whether castling is currently a legal move.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(0b1111);

    pub const fn contains(self, color: Color, side: CastlingSide) -> bool {
        self.0 & Self::mask(color, side) != 0
    }

    pub fn insert(&mut self, color: Color, side: CastlingSide) {
        self.0 |= Self::mask(color, side);
    }

    pub fn remove(&mut self, color: Color, side: CastlingSide) {
        self.0 &= !Self::mask(color, side);
    }

    const fn mask(color: Color, side: CastlingSide) -> u8 {
        1 << (color.index() * 2 + side as usize)
    }
}

impl fmt::Display for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::NONE {
            return f.write_str("-");
        }
        for (color, side, symbol) in [
            (Color::White, CastlingSide::Kingside, "K"),
            (Color::White, CastlingSide::Queenside, "Q"),
            (Color::Black, CastlingSide::Kingside, "k"),
            (Color::Black, CastlingSide::Queenside, "q"),
        ] {
            if self.contains(color, side) {
                f.write_str(symbol)?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for CastlingRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CastlingRights")
            .field(
                "white_kingside",
                &self.contains(Color::White, CastlingSide::Kingside),
            )
            .field(
                "white_queenside",
                &self.contains(Color::White, CastlingSide::Queenside),
            )
            .field(
                "black_kingside",
                &self.contains(Color::Black, CastlingSide::Kingside),
            )
            .field(
                "black_queenside",
                &self.contains(Color::Black, CastlingSide::Queenside),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_subsets_support_independent_idempotent_updates() {
        let rights = [
            (Color::White, CastlingSide::Kingside),
            (Color::White, CastlingSide::Queenside),
            (Color::Black, CastlingSide::Kingside),
            (Color::Black, CastlingSide::Queenside),
        ];
        for subset in 0u8..16 {
            let mut original = CastlingRights::NONE;
            for (i, &(color, side)) in rights.iter().enumerate() {
                if subset & (1 << i) != 0 {
                    original.insert(color, side);
                }
            }
            for (i, &(color, side)) in rights.iter().enumerate() {
                assert_eq!(original.contains(color, side), subset & (1 << i) != 0);
                let mut added = original;
                let mut removed = original;
                for _ in 0..2 {
                    added.insert(color, side);
                    removed.remove(color, side);
                }
                for (j, &(other_color, other_side)) in rights.iter().enumerate() {
                    let previously_present = subset & (1 << j) != 0;
                    assert_eq!(
                        added.contains(other_color, other_side),
                        i == j || previously_present
                    );
                    assert_eq!(
                        removed.contains(other_color, other_side),
                        i != j && previously_present
                    );
                }
            }
            if subset == 15 {
                assert_eq!(original, CastlingRights::ALL);
            }
        }
        assert_eq!(CastlingRights::default(), CastlingRights::NONE);
        assert_eq!(size_of::<CastlingRights>(), 1);
    }
}
