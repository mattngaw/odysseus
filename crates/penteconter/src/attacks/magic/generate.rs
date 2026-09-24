//! Native table construction, compiled only for build.rs and library tests.

use super::layout::{Magic, TABLE_LEN};
use super::reference;
use crate::{Bitboard, Square};

pub(super) fn build_attacks(magics: &[Magic; 128]) -> Vec<Bitboard> {
    let mut table = vec![Bitboard::EMPTY; TABLE_LEN];
    for (entry, &magic) in magics.iter().enumerate() {
        let square = Square::new((entry % 64) as u8).unwrap();
        for occupied in magic.mask.subsets() {
            let attacks = reference_attacks(entry, square, occupied);
            let index = magic.index(occupied);
            // Sliding attacks are never empty, so zero marks an unused slot.
            assert!(
                table[index].is_empty() || table[index] == attacks,
                "destructive magic collision at entry {entry}, occupancy {occupied:?}"
            );
            table[index] = attacks;
        }
    }
    verify_attacks(magics, &table);
    table
}

fn reference_attacks(entry: usize, square: Square, occupied: Bitboard) -> Bitboard {
    if entry < 64 {
        reference::rook_attacks(square, occupied)
    } else {
        reference::bishop_attacks(square, occupied)
    }
}

fn verify_attacks(magics: &[Magic; 128], table: &[Bitboard]) {
    assert_eq!(table.len(), TABLE_LEN);
    // Revisit the completed table, also setting every excluded occupancy bit.
    for (entry, &magic) in magics.iter().enumerate() {
        let square = Square::new((entry % 64) as u8).unwrap();
        for subset in magic.mask.subsets() {
            for occupied in [subset, subset | !magic.mask] {
                assert_eq!(
                    table[magic.index(occupied)],
                    reference_attacks(entry, square, occupied),
                    "magic verification failed at entry {entry}, occupancy {occupied:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attacks::magic::{ATTACKS, layout::MAGICS};

    #[test]
    fn native_generation_matches_every_embedded_slot() {
        assert_eq!(build_attacks(&MAGICS).as_slice(), ATTACKS.as_slice());
    }

    #[test]
    #[should_panic(expected = "destructive magic collision")]
    fn destructive_collisions_abort_generation() {
        let mut magics = MAGICS;
        magics[0].multiplier = 0; // Every rook-a1 occupancy lands in one slot.
        build_attacks(&magics);
    }

    #[test]
    #[should_panic(expected = "magic verification failed")]
    fn verification_rejects_a_corrupted_table() {
        let mut table = ATTACKS.to_vec();
        table[MAGICS[0].index(Bitboard::EMPTY)] = Bitboard::EMPTY;
        verify_attacks(&MAGICS, &table);
    }
}
