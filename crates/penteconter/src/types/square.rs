use std::fmt;

/// A square numbered by rank, from a1 = 0 through h8 = 63.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    /// Returns `None` for indices outside the board.
    pub const fn new(index: u8) -> Option<Self> {
        if index < 64 { Some(Self(index)) } else { None }
    }

    /// Constructs a square from a zero-based file and rank, each in 0..8.
    pub const fn from_coords(file: u8, rank: u8) -> Option<Self> {
        if file < 8 && rank < 8 {
            Some(Self(rank * 8 + file))
        } else {
            None
        }
    }

    /// Returns the index for a bitboard bit or mailbox entry.
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the zero-based file: a = 0, h = 7.
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Returns the zero-based rank: first = 0, eighth = 7.
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", char::from(b'a' + self.file()), self.rank() + 1)
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Square({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::Square;

    #[test]
    fn every_square_round_trips_through_coordinates() {
        for index in 0..64 {
            let square = Square::new(index).unwrap();
            assert_eq!(square.index(), usize::from(index));
            assert_eq!(
                Square::from_coords(square.file(), square.rank()),
                Some(square)
            );
        }
        assert_eq!(Square::from_coords(0, 0).unwrap().index(), 0);
        assert_eq!(Square::from_coords(7, 0).unwrap().index(), 7);
        assert_eq!(Square::from_coords(0, 7).unwrap().index(), 56);
        assert_eq!(Square::from_coords(7, 7).unwrap().index(), 63);
    }

    #[test]
    fn invalid_indices_and_coordinates_are_rejected() {
        for index in 64..=u8::MAX {
            assert_eq!(Square::new(index), None);
        }
        for invalid in 8..=u8::MAX {
            assert_eq!(Square::from_coords(invalid, 0), None);
            assert_eq!(Square::from_coords(0, invalid), None);
        }
    }
}
