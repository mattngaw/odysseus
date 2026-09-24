use std::{
    fmt,
    ops::{BitAnd, BitOr, BitXor, Not},
};

use crate::Square;

/// A set of squares, with bit 0 representing a1 and bit 63 representing h8.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u64::MAX);

    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn from_square(square: Square) -> Self {
        Self(1u64 << square.index())
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub const fn contains(self, square: Square) -> bool {
        self.0 & Self::from_square(square).0 != 0
    }

    pub fn insert(&mut self, square: Square) {
        self.0 |= Self::from_square(square).0;
    }

    pub fn remove(&mut self, square: Square) {
        self.0 &= !Self::from_square(square).0;
    }

    /// Removes and returns the lowest-indexed square, or `None` if empty.
    pub fn pop_first(&mut self) -> Option<Square> {
        if self.is_empty() {
            return None;
        }
        let square = Square::new(self.0.trailing_zeros() as u8);
        self.0 &= self.0 - 1;
        square
    }

    /// Visits every subset in increasing numeric order, including empty and self.
    /// The empty bitboard has one subset. This is lazy: there are 2^count()
    /// subsets, so exhausting the iterator is practical only for small masks.
    pub fn subsets(self) -> impl Iterator<Item = Self> {
        std::iter::successors(Some(Self::EMPTY), move |subset| {
            // Subtracting the mask carries through the gaps between its set
            // bits; masking again keeps only the next subset. After the full
            // mask, this wraps back to zero and enumeration stops.
            let next = subset.0.wrapping_sub(self.0) & self.0;
            (next != 0).then_some(Self(next))
        })
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard(0x{:016x})", self.0)
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::formatting::grid(f, |square| if self.contains(square) { 'x' } else { '.' })
    }
}

#[cfg(test)]
mod tests {
    use super::{Bitboard, Square};

    #[test]
    fn singleton_updates_cover_every_square() {
        for index in 0..64 {
            let square = Square::new(index).unwrap();
            let singleton = Bitboard::from_square(square);
            assert_eq!(singleton.bits(), 1u64 << index);
            assert_eq!(singleton.count(), 1);
            assert!(singleton.contains(square));

            let mut board = Bitboard::EMPTY;
            board.remove(square);
            assert!(board.is_empty());
            board.insert(square);
            board.insert(square);
            assert_eq!(board, singleton);
            board.remove(square);
            assert!(board.is_empty());

            let mut board = Bitboard::FULL;
            board.remove(square);
            assert!(!board.contains(square));
            assert_eq!(board.count(), 63);
            board.insert(square);
            assert_eq!(board, Bitboard::FULL);
        }
    }

    #[test]
    fn set_operations_match_membership() {
        let a = Bitboard::from_bits(0xAA55_AA55_AA55_AA55);
        let b = Bitboard::from_bits(0xFFFF_0000_FFFF_0000);
        for index in 0..64 {
            let s = Square::new(index).unwrap();
            assert_eq!((a & b).contains(s), a.contains(s) && b.contains(s));
            assert_eq!((a | b).contains(s), a.contains(s) || b.contains(s));
            assert_eq!((a ^ b).contains(s), a.contains(s) != b.contains(s));
            assert_eq!((!a).contains(s), !a.contains(s));
        }
    }

    #[test]
    fn subsets_match_explicit_bit_deposition_including_high_bits() {
        for bits in [
            0,
            1,
            1u64 << 63,
            0x8000_0000_0000_0081,
            0xff,
            0x8100_0000_0000_0081,
        ] {
            let mask = Bitboard::from_bits(bits);
            let positions: Vec<_> = (0..64).filter(|i| bits & (1u64 << i) != 0).collect();
            let mut subsets = mask.subsets();
            for packed in 0..(1u64 << positions.len()) {
                let mut expected = 0;
                for (j, position) in positions.iter().enumerate() {
                    if packed & (1 << j) != 0 {
                        expected |= 1u64 << position;
                    }
                }
                assert_eq!(subsets.next(), Some(Bitboard::from_bits(expected)));
            }
            assert_eq!(subsets.next(), None);
            assert_eq!(subsets.next(), None);
        }
        // A full mask is valid too; no allocation or 2^64 length calculation.
        assert_eq!(
            Bitboard::FULL
                .subsets()
                .take(4)
                .map(Bitboard::bits)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn popping_visits_each_member_once_in_ascending_order() {
        for bits in [0, 1, 1u64 << 63, 0x8000_0000_0000_0081, u64::MAX] {
            let mut board = Bitboard::from_bits(bits);
            for index in 0..64 {
                if bits & (1u64 << index) != 0 {
                    assert_eq!(board.pop_first(), Square::new(index));
                }
            }
            assert!(board.is_empty());
            assert_eq!(board.pop_first(), None);
            assert_eq!(board.pop_first(), None);
        }
    }
}
