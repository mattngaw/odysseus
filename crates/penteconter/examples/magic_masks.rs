use std::collections::BTreeSet;

use penteconter::attacks::{magic, reference};
use penteconter::{Bitboard, Square};

fn main() {
    let a1 = Square::from_coords(0, 0).unwrap();
    let mask = magic::rook_mask(a1);
    println!("Rook on {a1}: relevant occupancy mask\n{mask}\n");
    let attacks: BTreeSet<_> = mask
        .subsets()
        .map(|occupied| reference::rook_attacks(a1, occupied).bits())
        .collect();
    println!("Relevant bits: {}", mask.count());
    println!("Occupancy subsets: {}", mask.subsets().count());
    println!("Distinct attack bitboards: {}", attacks.len());
    println!("First four subsets:");
    for subset in mask.subsets().take(4) {
        println!("  {subset:?}");
    }
    println!("Empty mask subsets: {}", Bitboard::EMPTY.subsets().count());

    let d4 = Square::from_coords(3, 3).unwrap();
    println!(
        "\nBishop on {d4}: relevant occupancy mask\n{}",
        magic::bishop_mask(d4)
    );
}
