use penteconter::{Board, CastlingRights, Color, Piece, PieceKind, Position, Square};

fn main() {
    let mut board = Board::empty();
    for (file, rank, color, kind) in [
        (0, 0, Color::White, PieceKind::Knight),
        (4, 0, Color::White, PieceKind::King),
        (4, 3, Color::White, PieceKind::Pawn),
        (2, 5, Color::Black, PieceKind::Bishop),
        (7, 7, Color::Black, PieceKind::King),
    ] {
        board.set_piece(
            Square::from_coords(file, rank).unwrap(),
            Piece::new(color, kind),
        );
    }

    println!("Board:\n{board}\n");
    let occupied = board.occupied();
    println!("Occupied squares:\n{occupied}\n");
    println!("Occupied bits: {occupied:?}");

    let square = Square::from_coords(0, 0).unwrap();
    let piece = board.piece_at(square).unwrap();
    println!("Square: {square} / {square:?}");
    println!("Piece: {piece} / {piece:?}");
    let position = Position::new(board, Color::White, CastlingRights::NONE, None, 0, 12)
        .expect("example position meets the construction checks");
    println!("\nPosition (Display):\n{position}");
    println!("\nPosition (pretty Debug):\n{position:#?}");
}
