//! Simple ray-walking implementations for verification and benchmarking.
//!
//! Keep these independent of future optimized sliding attacks. Callers must
//! explicitly select `attacks::reference`; these are not the main attack API.
//! Occupancy includes both colors. The first blocker is included, regardless
//! of color, and squares behind it are excluded. The origin is never attacked,
//! and whether it is present in occupancy does not affect the result.
//! The build script also uses these functions to construct lookup tables.

use crate::{Bitboard, Square};

/// Walks the four diagonal rays, stopping after the first blocker on each.
pub const fn bishop_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    walk_rays(square, occupied, &[(-1, -1), (-1, 1), (1, -1), (1, 1)])
}

/// Walks the four orthogonal rays, stopping after the first blocker on each.
pub const fn rook_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    walk_rays(square, occupied, &[(-1, 0), (1, 0), (0, -1), (0, 1)])
}

/// Combines the reference bishop and rook attacks.
pub const fn queen_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    Bitboard::from_bits(
        bishop_attacks(square, occupied).bits() | rook_attacks(square, occupied).bits(),
    )
}

const fn walk_rays(square: Square, occupied: Bitboard, directions: &[(i8, i8)]) -> Bitboard {
    let mut attacks = 0;
    let occupied = occupied.bits();
    let mut direction = 0;
    while direction < directions.len() {
        let (df, dr) = directions[direction];
        let mut file = square.file() as i8 + df;
        let mut rank = square.rank() as i8 + dr;
        while file >= 0 && file < 8 && rank >= 0 && rank < 8 {
            let target = 1u64 << (rank * 8 + file);
            attacks |= target;
            if occupied & target != 0 {
                break;
            }
            file += df;
            rank += dr;
        }
        direction += 1;
    }
    Bitboard::from_bits(attacks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(name: &str) -> Square {
        let bytes = name.as_bytes();
        Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap()
    }

    fn squares(names: &[&str]) -> Bitboard {
        let mut result = Bitboard::EMPTY;
        for name in names {
            result.insert(square(name));
        }
        result
    }

    #[test]
    fn first_blockers_are_included_and_hide_everything_beyond_them() {
        let occupied = squares(&["d4", "d6", "d7", "f4", "h4", "b2", "a1", "f6", "h8"]);
        let bishop = squares(&["e5", "f6", "c5", "b6", "a7", "e3", "f2", "g1", "c3", "b2"]);
        let rook = squares(&["d5", "d6", "d3", "d2", "d1", "e4", "f4", "c4", "b4", "a4"]);
        assert_eq!(bishop_attacks(square("d4"), occupied), bishop);
        assert_eq!(rook_attacks(square("d4"), occupied), rook);
        assert_eq!(queen_attacks(square("d4"), occupied), bishop | rook);
    }

    // Independent oracle: consider each destination, then check whether any
    // occupied square lies strictly inside its geometric line segment. It
    // shares neither direction lists nor ray stepping with the implementation.
    fn visible_targets(from: Square, occupied: Bitboard) -> (Bitboard, Bitboard) {
        let mut bishop = Bitboard::EMPTY;
        let mut rook = Bitboard::EMPTY;
        for index in 0..64 {
            let to = Square::new(index).unwrap();
            if to == from {
                continue;
            }
            let df = i16::from(to.file()) - i16::from(from.file());
            let dr = i16::from(to.rank()) - i16::from(from.rank());
            let diagonal = df.abs() == dr.abs();
            let orthogonal = df == 0 || dr == 0;
            if !diagonal && !orthogonal {
                continue;
            }
            let mut blocked = false;
            for blocker in 0..64 {
                let blocker = Square::new(blocker).unwrap();
                if !occupied.contains(blocker) {
                    continue;
                }
                let bf = i16::from(blocker.file()) - i16::from(from.file());
                let br = i16::from(blocker.rank()) - i16::from(from.rank());
                let dot = bf * df + br * dr;
                if bf * dr == br * df && dot > 0 && dot < df * df + dr * dr {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                if diagonal {
                    bishop.insert(to);
                }
                if orthogonal {
                    rook.insert(to);
                }
            }
        }
        (bishop, rook)
    }

    #[test]
    fn every_origin_matches_visibility_for_single_and_multiple_blockers() {
        let mut occupancies: Vec<_> = (0..64)
            .map(|index| Bitboard::from_bits(1u64 << index))
            .collect();
        occupancies.extend([
            Bitboard::EMPTY,
            Bitboard::FULL,
            Bitboard::from_bits(0xAA55_AA55_AA55_AA55),
            Bitboard::from_bits(0xFF81_8181_8181_81FF),
            Bitboard::from_bits(0x8040_2010_0804_0201),
            Bitboard::from_bits(0x0000_1818_0000_0000),
        ]);
        for index in 0..64 {
            let from = Square::new(index).unwrap();
            for &occupied in &occupancies {
                let (bishop, rook) = visible_targets(from, occupied);
                // Both representations of origin occupancy must give the same attacks.
                for occupied in [
                    occupied | Bitboard::from_square(from),
                    occupied & !Bitboard::from_square(from),
                ] {
                    assert_eq!(
                        bishop_attacks(from, occupied),
                        bishop,
                        "bishop {from}, {occupied:?}"
                    );
                    assert_eq!(
                        rook_attacks(from, occupied),
                        rook,
                        "rook {from}, {occupied:?}"
                    );
                    assert_eq!(
                        queen_attacks(from, occupied),
                        bishop | rook,
                        "queen {from}, {occupied:?}"
                    );
                }
            }
        }
    }
}
