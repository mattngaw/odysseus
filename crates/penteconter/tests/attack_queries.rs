use penteconter::{
    Bitboard, Board, CastlingRights, Color, Piece, PieceKind, Position, Square, attacks,
};

fn square(name: &str) -> Square {
    let bytes = name.as_bytes();
    Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap()
}

fn sources(names: &[&str]) -> Bitboard {
    let mut result = Bitboard::EMPTY;
    for name in names {
        result.insert(square(name));
    }
    result
}

#[test]
fn pinned_pieces_still_attack_empty_and_friendly_occupied_squares() {
    for (fen, target, expected) in [
        ("4k3/4n3/8/8/8/8/8/K3R3 w - - 0 1", "f5", "e7"),
        ("4k3/4n3/8/5p2/8/8/8/K3R3 w - - 0 1", "f5", "e7"),
        ("4k3/4p3/8/8/8/8/8/K3R3 w - - 0 1", "f6", "e7"),
    ] {
        let position: Position = fen.parse().unwrap();
        assert_eq!(
            position.attackers_to(square(target), Color::Black),
            sources(&[expected])
        );
        assert!(position.is_square_attacked(square(target), Color::Black));
        assert!(!position.in_check(Color::Black));
    }
}

#[test]
fn pawns_attack_diagonals_not_pushes_or_the_en_passant_victim() {
    let position: Position = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 8".parse().unwrap();
    for (color, target, expected) in [
        (Color::White, "d6", sources(&["e5"])),
        (Color::White, "f6", sources(&["e5"])),
        (Color::White, "e6", Bitboard::EMPTY),
        (Color::White, "d5", Bitboard::EMPTY),
        (Color::Black, "c4", sources(&["d5"])),
        (Color::Black, "e4", sources(&["d5"])),
        (Color::Black, "d4", Bitboard::EMPTY),
    ] {
        assert_eq!(position.attackers_to(square(target), color), expected);
    }
}

#[test]
fn blockers_hide_sliders_of_either_color_and_targets_do_not_block_the_query() {
    for blocker in ['P', 'p'] {
        let fen = format!("k3r3/8/8/4{blocker}3/8/8/8/4K3 w - - 0 1");
        let position: Position = fen.parse().unwrap();
        assert_eq!(
            position.attackers_to(square("e5"), Color::Black),
            sources(&["e8"])
        );
        assert!(!position.is_square_attacked(square("e4"), Color::Black));
        assert!(!position.in_check(Color::White));
    }
    let position: Position = "k3r3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
    assert!(position.in_check(Color::White));
    assert!(!position.in_check(Color::Black));
}

#[test]
fn check_queries_support_double_check_and_either_color_regardless_of_turn() {
    for turn in ["w", "b"] {
        let position: Position = format!("k3r3/8/8/8/1b6/8/8/4K3 {turn} - - 0 1")
            .parse()
            .unwrap();
        assert_eq!(
            position.attackers_to(square("e1"), Color::Black),
            sources(&["e8", "b4"])
        );
        assert!(position.in_check(Color::White));
        assert!(!position.in_check(Color::Black));
        let position: Position = format!("4k3/8/8/1B6/8/8/8/K3R3 {turn} - - 0 1")
            .parse()
            .unwrap();
        assert_eq!(
            position.attackers_to(square("e8"), Color::White),
            sources(&["e1", "b5"])
        );
        assert!(position.in_check(Color::Black));
        assert!(!position.in_check(Color::White));
    }
    // Structurally accepted positions can have adjacent kings. Neither king's
    // attacks may recursively depend on whether its move would be legal.
    let position: Position = "8/8/8/8/8/8/4k3/4K3 w - - 0 1".parse().unwrap();
    assert!(position.in_check(Color::White));
    assert!(position.in_check(Color::Black));
}

// Independent query direction: enumerate pieces from the mailbox and generate
// forward attacks, using ray walking instead of the production magic lookups.
fn forward_attackers(position: &Position, target: Square, color: Color) -> Bitboard {
    let mut result = Bitboard::EMPTY;
    for index in 0..64 {
        let source = Square::new(index).unwrap();
        let Some(piece) = position.board().piece_at(source) else {
            continue;
        };
        if piece.color() != color {
            continue;
        }
        let occupied = position.board().occupied();
        let attacks = match piece.kind() {
            PieceKind::Pawn => attacks::pawn_attacks(color, Bitboard::from_square(source)),
            PieceKind::Knight => attacks::knight_attacks(source),
            PieceKind::King => attacks::king_attacks(source),
            PieceKind::Bishop => attacks::reference::bishop_attacks(source, occupied),
            PieceKind::Rook => attacks::reference::rook_attacks(source, occupied),
            PieceKind::Queen => attacks::reference::queen_attacks(source, occupied),
        };
        if attacks.contains(target) {
            result.insert(source);
        }
    }
    result
}

#[test]
fn all_targets_match_forward_attacks_on_sparse_and_dense_placements() {
    let kinds = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
    ];
    let mut state = 0x4f44_5953_5345_5553u64;
    for sample in 0..64 {
        let mut board = Board::empty();
        for index in 0..64 {
            let square = Square::new(index).unwrap();
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = state >> 32;
            if bits % 4 > sample % 4 {
                continue;
            }
            let kind = kinds[(bits as usize / 4) % kinds.len()];
            if kind == PieceKind::Pawn && (square.rank() == 0 || square.rank() == 7) {
                continue;
            }
            let color = if bits & 32 == 0 {
                Color::White
            } else {
                Color::Black
            };
            board.set_piece(square, Piece::new(color, kind));
        }
        let white_king = Square::new(sample as u8).unwrap();
        let black_king = Square::new(((sample + 31) % 64) as u8).unwrap();
        board.set_piece(white_king, Piece::new(Color::White, PieceKind::King));
        board.set_piece(black_king, Piece::new(Color::Black, PieceKind::King));
        let position =
            Position::new(board, Color::White, CastlingRights::NONE, None, 0, 1).unwrap();
        for index in 0..64 {
            let target = Square::new(index).unwrap();
            for color in [Color::White, Color::Black] {
                let expected = forward_attackers(&position, target, color);
                assert_eq!(
                    position.attackers_to(target, color),
                    expected,
                    "sample {sample}, {color:?} attacking {target}"
                );
                assert_eq!(
                    position.is_square_attacked(target, color),
                    !expected.is_empty()
                );
            }
        }
        for (color, king) in [(Color::White, white_king), (Color::Black, black_king)] {
            assert_eq!(
                position.in_check(color),
                !forward_attackers(&position, king, color.opposite()).is_empty()
            );
        }
    }
}
