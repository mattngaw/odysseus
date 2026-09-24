use penteconter::attacks::{
    king_attacks, knight_attacks, pawn_attacks, pawn_attacks_east, pawn_attacks_west,
};
use penteconter::{Bitboard, Color, Square};

fn main() {
    let d4 = Square::from_coords(3, 3).unwrap();
    println!("Knight on {d4}:\n{}\n", knight_attacks(d4));
    println!("King on {d4}:\n{}\n", king_attacks(d4));

    let c4 = Square::from_coords(2, 3).unwrap();
    let e4 = Square::from_coords(4, 3).unwrap();
    let pawns = Bitboard::from_square(c4) | Bitboard::from_square(e4);
    println!("Pawns on {c4} and {e4}:\n{pawns}\n");
    for color in [Color::White, Color::Black] {
        println!(
            "{color:?} pawn attacks west:\n{}\n",
            pawn_attacks_west(color, pawns)
        );
        println!(
            "{color:?} pawn attacks east:\n{}\n",
            pawn_attacks_east(color, pawns)
        );
        println!(
            "{color:?} pawn attacks combined:\n{}\n",
            pawn_attacks(color, pawns)
        );
    }
}
