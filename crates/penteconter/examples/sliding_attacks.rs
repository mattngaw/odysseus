use penteconter::attacks;
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
        "Magic bishop attacks from {origin}:\n{}\n",
        attacks::bishop_attacks(origin, occupied)
    );
    println!(
        "Magic rook attacks from {origin}:\n{}\n",
        attacks::rook_attacks(origin, occupied)
    );
    println!(
        "Magic queen attacks from {origin}:\n{}",
        attacks::queen_attacks(origin, occupied)
    );
}
