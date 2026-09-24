use penteconter::attacks::reference;
use penteconter::{Bitboard, Square};

fn main() {
    let origin = Square::from_coords(3, 3).unwrap(); // d4
    let mut occupied = Bitboard::from_square(origin);
    for (file, rank) in [
        (3, 5),
        (3, 6),
        (5, 3),
        (7, 3),
        (1, 1),
        (0, 0),
        (5, 5),
        (7, 7),
    ] {
        occupied.insert(Square::from_coords(file, rank).unwrap());
    }
    println!("Occupancy (including origin {origin}):\n{occupied}\n");
    println!(
        "Reference bishop attacks from {origin}:\n{}\n",
        reference::bishop_attacks(origin, occupied)
    );
    println!(
        "Reference rook attacks from {origin}:\n{}\n",
        reference::rook_attacks(origin, occupied)
    );
    println!(
        "Reference queen attacks from {origin}:\n{}",
        reference::queen_attacks(origin, occupied)
    );
}
