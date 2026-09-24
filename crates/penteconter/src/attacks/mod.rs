//! Attack geometry in fixed board coordinates.
//!
//! The non-sliding functions below do not inspect occupancy, pins, or check.
//! Their results include attacked squares occupied by friendly pieces. Pawn
//! attacks are diagonals only; pushes and en passant legality belong to move
//! generation. Sliding ray walkers live separately in [`reference`].

/// Explicit reference implementations for correctness checks and benchmarks.
pub mod reference;

/// Magic lookups and relevant occupancy masks.
pub mod magic;

pub use magic::{bishop_attacks, queen_attacks, rook_attacks};

use crate::{Bitboard, Color, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;

// Each table occupies 64 * 8 = 512 bytes and is built at compile time.
static KNIGHT_ATTACKS: [Bitboard; 64] = attack_table([
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]);
static KING_ATTACKS: [Bitboard; 64] = attack_table([
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]);

/// Returns all squares a knight on `square` attacks.
#[inline]
pub fn knight_attacks(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index()]
}

/// Returns all adjacent squares, without checking whether a king can move there.
#[inline]
pub fn king_attacks(square: Square) -> Bitboard {
    KING_ATTACKS[square.index()]
}

/// Attacks toward the a-file, for every pawn in `pawns`.
/// West is absolute: northwest for White, southwest for Black.
#[inline]
pub const fn pawn_attacks_west(color: Color, pawns: Bitboard) -> Bitboard {
    let pawns = pawns.bits() & !FILE_A;
    Bitboard::from_bits(match color {
        Color::White => pawns << 7,
        Color::Black => pawns >> 9,
    })
}

/// Attacks toward the h-file, for every pawn in `pawns`.
/// East is absolute: northeast for White, southeast for Black.
#[inline]
pub const fn pawn_attacks_east(color: Color, pawns: Bitboard) -> Bitboard {
    let pawns = pawns.bits() & !FILE_H;
    Bitboard::from_bits(match color {
        Color::White => pawns << 9,
        Color::Black => pawns >> 7,
    })
}

/// The union of both attack directions for every pawn in `pawns`.
/// Use the directional functions when source squares or double attacks matter.
#[inline]
pub const fn pawn_attacks(color: Color, pawns: Bitboard) -> Bitboard {
    Bitboard::from_bits(
        pawn_attacks_west(color, pawns).bits() | pawn_attacks_east(color, pawns).bits(),
    )
}

const fn attack_table(offsets: [(i8, i8); 8]) -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let mut index = 0;
    while index < 64 {
        let mut bits = 0;
        let mut offset = 0;
        while offset < offsets.len() {
            let file = (index % 8) as i8 + offsets[offset].0;
            let rank = (index / 8) as i8 + offsets[offset].1;
            if file >= 0 && file < 8 && rank >= 0 && rank < 8 {
                bits |= 1u64 << (rank * 8 + file);
            }
            offset += 1;
        }
        table[index] = Bitboard::from_bits(bits);
        index += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knight_and_king_tables_match_coordinate_distances_for_every_square_pair() {
        for from in 0..64 {
            let from = Square::new(from).unwrap();
            let knight = knight_attacks(from);
            let king = king_attacks(from);
            for to in 0..64 {
                let to = Square::new(to).unwrap();
                let df = from.file().abs_diff(to.file());
                let dr = from.rank().abs_diff(to.rank());
                assert_eq!(
                    knight.contains(to),
                    (df == 1 && dr == 2) || (df == 2 && dr == 1),
                    "knight {from} -> {to}"
                );
                assert_eq!(king.contains(to), df.max(dr) == 1, "king {from} -> {to}");
            }
        }
    }

    #[test]
    fn pawn_directions_match_coordinates_for_every_square_pair_and_color() {
        for color in [Color::White, Color::Black] {
            let forward = if color == Color::White { 1 } else { -1 };
            for from in 0..64 {
                let from = Square::new(from).unwrap();
                let pawns = Bitboard::from_square(from);
                let west = pawn_attacks_west(color, pawns);
                let east = pawn_attacks_east(color, pawns);
                let both = pawn_attacks(color, pawns);
                for to in 0..64 {
                    let to = Square::new(to).unwrap();
                    let df = i16::from(to.file()) - i16::from(from.file());
                    let dr = i16::from(to.rank()) - i16::from(from.rank());
                    assert_eq!(
                        west.contains(to),
                        df == -1 && dr == forward,
                        "{color:?} west {from} -> {to}"
                    );
                    assert_eq!(
                        east.contains(to),
                        df == 1 && dr == forward,
                        "{color:?} east {from} -> {to}"
                    );
                    assert_eq!(
                        both.contains(to),
                        df.abs() == 1 && dr == forward,
                        "{color:?} union {from} -> {to}"
                    );
                }
            }
        }
    }

    fn assert_pawn_set_matches_coordinates(color: Color, pawns: Bitboard) {
        let mut expected_west = Bitboard::EMPTY;
        let mut expected_east = Bitboard::EMPTY;
        for index in 0..64 {
            let from = Square::new(index).unwrap();
            if !pawns.contains(from) {
                continue;
            }
            let rank = i16::from(from.rank()) + if color == Color::White { 1 } else { -1 };
            if !(0..8).contains(&rank) {
                continue;
            }
            if from.file() > 0 {
                expected_west.insert(Square::from_coords(from.file() - 1, rank as u8).unwrap());
            }
            if from.file() < 7 {
                expected_east.insert(Square::from_coords(from.file() + 1, rank as u8).unwrap());
            }
        }
        assert_eq!(pawn_attacks_west(color, pawns), expected_west);
        assert_eq!(pawn_attacks_east(color, pawns), expected_east);
        assert_eq!(pawn_attacks(color, pawns), expected_west | expected_east);
    }

    #[test]
    fn whole_pawn_sets_match_coordinate_oracle() {
        for color in [Color::White, Color::Black] {
            // Every arrangement on each rank, including edges and back ranks.
            for rank in 0..8 {
                for arrangement in 0u64..256 {
                    assert_pawn_set_matches_coordinates(
                        color,
                        Bitboard::from_bits(arrangement << (8 * rank)),
                    );
                }
            }
            // Multiple ranks, both occupied edge files, dense and sparse sets.
            for bits in [
                u64::MAX,
                0x8181_8181_8181_8181,
                0xAA55_AA55_AA55_AA55,
                0x8040_2010_0804_0201,
            ] {
                assert_pawn_set_matches_coordinates(color, Bitboard::from_bits(bits));
            }
        }
    }

    #[test]
    fn separate_directions_preserve_two_sources_attacking_one_target() {
        for (color, rank, target_rank) in [(Color::White, 3, 4), (Color::Black, 4, 3)] {
            let left = Square::from_coords(2, rank).unwrap();
            let right = Square::from_coords(4, rank).unwrap();
            let target = Square::from_coords(3, target_rank).unwrap();
            let pawns = Bitboard::from_square(left) | Bitboard::from_square(right);
            let west = pawn_attacks_west(color, pawns);
            let east = pawn_attacks_east(color, pawns);
            assert_eq!(west & east, Bitboard::from_square(target));
            assert_eq!(west.count() + east.count(), 4);
            assert_eq!(pawn_attacks(color, pawns).count(), 3);
        }
    }
}
