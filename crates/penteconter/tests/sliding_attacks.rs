use penteconter::attacks::{bishop_attacks, magic, queen_attacks, reference, rook_attacks};
use penteconter::{Bitboard, Square};

#[test]
fn every_relevant_occupancy_matches_the_reference_through_the_main_api() {
    type AttackFn = fn(Square, Bitboard) -> Bitboard;
    let mut checked = 0;
    for index in 0..64 {
        let square = Square::new(index).unwrap();
        let cases: [(Bitboard, AttackFn, AttackFn); 2] = [
            (
                magic::rook_mask(square),
                rook_attacks,
                reference::rook_attacks,
            ),
            (
                magic::bishop_mask(square),
                bishop_attacks,
                reference::bishop_attacks,
            ),
        ];
        for (mask, lookup, reference) in cases {
            for subset in mask.subsets() {
                for occupied in [subset, subset | !mask] {
                    assert_eq!(
                        lookup(square, occupied),
                        reference(square, occupied),
                        "{square}, mask {mask:?}, occupancy {occupied:?}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 107_648);
}

#[test]
fn combined_rank_file_and_diagonal_blockers_match_for_all_three_sliders() {
    let mut patterns = vec![Bitboard::EMPTY, Bitboard::FULL];
    // Deterministic mixed occupancies exercise rook and bishop rays together.
    let mut bits = 0x0123_4567_89ab_cdefu64;
    for _ in 0..64 {
        bits = bits.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        patterns.push(Bitboard::from_bits(bits));
    }
    for index in 0..64 {
        let square = Square::new(index).unwrap();
        for &occupied in &patterns {
            assert_eq!(
                rook_attacks(square, occupied),
                reference::rook_attacks(square, occupied)
            );
            assert_eq!(
                bishop_attacks(square, occupied),
                reference::bishop_attacks(square, occupied)
            );
            assert_eq!(
                queen_attacks(square, occupied),
                reference::queen_attacks(square, occupied)
            );
        }
    }
}
