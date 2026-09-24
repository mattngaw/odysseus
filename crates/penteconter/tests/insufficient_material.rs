use penteconter::{
    Board, CastlingRights, Color, Game, Move, MoveKind, Piece, PieceKind, Position, Square,
};

const COLORS: [Color; 2] = [Color::White, Color::Black];

fn sq(coordinates: &str) -> Square {
    let bytes = coordinates.as_bytes();
    Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap()
}

fn position(pieces: &[(Square, Color, PieceKind)]) -> Position {
    let mut board = Board::empty();
    board.set_piece(sq("a1"), Piece::new(Color::White, PieceKind::King));
    board.set_piece(sq("h8"), Piece::new(Color::Black, PieceKind::King));
    for &(square, color, kind) in pieces {
        assert_eq!(board.set_piece(square, Piece::new(color, kind)), None);
    }
    Position::new(board, Color::White, CastlingRights::NONE, None, 0, 1).unwrap()
}

fn assert_insufficient(position: &Position, expected: bool) {
    let before = *position;
    assert_eq!(position.has_insufficient_material(), expected, "{position}");
    assert_eq!(*position, before);
}

#[test]
fn bare_kings_and_a_single_minor_piece_are_insufficient() {
    assert_insufficient(&position(&[]), true);
    for color in COLORS {
        for kind in [PieceKind::Knight, PieceKind::Bishop] {
            for square in [sq("c3"), sq("c4")] {
                assert_insufficient(&position(&[(square, color, kind)]), true);
            }
        }
    }
}

#[test]
fn bishop_pairs_depend_on_square_color_not_ownership() {
    // Compare coordinate parity against the implementation's bitboard masks.
    for first in [sq("c3"), sq("c4")] {
        for index in 0..64 {
            let second = Square::new(index).unwrap();
            if [sq("a1"), sq("h8"), first].contains(&second) {
                continue;
            }
            let same_color =
                (first.file() + first.rank()) % 2 == (second.file() + second.rank()) % 2;
            for first_owner in COLORS {
                for second_owner in COLORS {
                    let position = position(&[
                        (first, first_owner, PieceKind::Bishop),
                        (second, second_owner, PieceKind::Bishop),
                    ]);
                    assert_insufficient(&position, same_color);
                }
            }
        }
    }
}

#[test]
fn multiple_promoted_bishops_are_insufficient_only_on_one_square_color() {
    for (squares, opposite) in [
        (["b4", "d6", "f8", "h6"], "c4"), // Dark squares.
        (["b3", "d5", "f7", "h5"], "c3"), // Light squares.
    ] {
        // Include bishops all owned by either player and split between them.
        for ownership in 0..16 {
            let mut pieces = Vec::new();
            for (index, square) in squares.into_iter().enumerate() {
                let owner = COLORS[(ownership >> index) & 1];
                pieces.push((sq(square), owner, PieceKind::Bishop));
            }
            assert_insufficient(&position(&pieces), true);
            pieces.push((sq(opposite), Color::White, PieceKind::Bishop));
            assert_insufficient(&position(&pieces), false);
        }
    }
}

#[test]
fn two_knights_or_a_knight_and_bishop_are_not_insufficient() {
    for second_kind in [PieceKind::Knight, PieceKind::Bishop] {
        for first_owner in COLORS {
            for second_owner in COLORS {
                // Covers KNN-K, KN-KN, KBN-K, and KN-KB, with colors reversed.
                let position = position(&[
                    (sq("c3"), first_owner, PieceKind::Knight),
                    (sq("e4"), second_owner, second_kind),
                ]);
                assert_insufficient(&position, false);
            }
        }
    }
}

#[test]
fn any_pawn_rook_or_queen_prevents_material_recognition() {
    for color in COLORS {
        for kind in [PieceKind::Pawn, PieceKind::Rook, PieceKind::Queen] {
            let mut pieces = vec![(sq("c4"), color, kind)];
            assert_insufficient(&position(&pieces), false);
            pieces.extend([
                (sq("b4"), Color::White, PieceKind::Bishop),
                (sq("d6"), Color::Black, PieceKind::Bishop),
            ]);
            assert_insufficient(&position(&pieces), false);
        }
    }
    let start: Position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        .parse()
        .unwrap();
    assert_insufficient(&start, false);
}

#[test]
fn turn_and_move_counters_do_not_change_material_recognition() {
    for (fen, expected) in [
        ("7k/8/8/8/8/8/8/K7 w - - 0 1", true),
        ("7k/8/8/8/8/2N5/8/K7 w - - 0 1", true),
        ("7k/8/3b4/8/1B6/8/8/K7 w - - 0 1", true),
        ("7k/8/8/8/4n3/2N5/8/K7 w - - 0 1", false),
        ("7k/8/3b4/8/2B5/8/8/K7 w - - 0 1", false),
    ] {
        let base: Position = fen.parse().unwrap();
        for color in COLORS {
            for (halfmove, fullmove) in [(0, 1), (150, 100), (u32::MAX, u32::MAX)] {
                let changed = Position::new(
                    *base.board(),
                    color,
                    CastlingRights::NONE,
                    None,
                    halfmove,
                    fullmove,
                )
                .unwrap();
                assert_insufficient(&changed, expected);
            }
        }
    }
}

#[test]
fn a_capture_can_leave_insufficient_material_and_undo_restores_it() {
    let root: Position = "4k3/8/8/8/8/8/3r4/2B1K3 w - - 0 1".parse().unwrap();
    let mut game = Game::new(root);
    let capture = Move::new(sq("c1"), sq("d2"), MoveKind::Normal).unwrap();
    assert_insufficient(game.position(), false);
    game.play(capture).unwrap();
    assert_insufficient(game.position(), true);
    assert_eq!(game.undo(), Some(capture));
    assert_eq!(*game.position(), root);
    assert_insufficient(game.position(), false);
}

#[test]
fn promotion_kind_and_bishop_square_color_determine_the_new_material() {
    for (fen, bishop_draw, knight_draw) in [
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", true, true),
        // a8 and c2 are light; b2 is dark.
        ("4k3/P7/8/8/8/8/2B5/4K3 w - - 0 1", true, false),
        ("4k3/P7/8/8/8/8/1B6/4K3 w - - 0 1", false, false),
    ] {
        let root: Position = fen.parse().unwrap();
        for (kind, expected) in [
            (PieceKind::Knight, knight_draw),
            (PieceKind::Bishop, bishop_draw),
            (PieceKind::Rook, false),
            (PieceKind::Queen, false),
        ] {
            let mut game = Game::new(root);
            let promotion = Move::new(sq("a7"), sq("a8"), MoveKind::Promotion(kind)).unwrap();
            assert_insufficient(game.position(), false);
            game.play(promotion).unwrap();
            assert_insufficient(game.position(), expected);
            assert_eq!(game.undo(), Some(promotion));
            assert_eq!(*game.position(), root);
            assert_insufficient(game.position(), false);
        }
    }
}
