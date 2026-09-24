use super::*;
use crate::{Board, CastlingRights, Piece};

fn square(name: &str) -> Square {
    let bytes = name.as_bytes();
    Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap()
}

fn play(position: &Position, coordinates: &str) -> Position {
    play_kind(position, coordinates, MoveKind::Normal)
}

fn play_kind(position: &Position, coordinates: &str, kind: MoveKind) -> Position {
    let mv = Move::new(square(&coordinates[..2]), square(&coordinates[2..]), kind).unwrap();
    let before = *position;
    let next = position.apply(mv);
    assert_eq!(position.board().after_move(mv), *next.board());
    #[cfg(feature = "perft-experiment")]
    {
        let mut working = *position;
        let undo = working.make(mv);
        assert_eq!(working, next);
        working.unmake(mv, undo);
        assert_eq!(working, *position, "unmake failed for {coordinates}");
        assert_eq!(working.zobrist_key(), working.recompute_zobrist_key());
    }
    assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
    assert_eq!(next.zobrist_key(), next.recompute_zobrist_key());
    assert_eq!(*position, before, "parent changed");
    assert!(next.board().is_consistent());
    // Reparse to check the constructor's structural invariants as well as
    // agreement between the serialized mailbox and the stored bitboards.
    assert_eq!(next.to_fen().parse::<Position>().unwrap(), next);
    next
}

#[test]
fn opening_sequence_updates_turns_clocks_and_en_passant() {
    let mut position: Position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        .parse()
        .unwrap();
    for (mv, expected) in [
        (
            "e2e4",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ),
        (
            "d7d5",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        ),
        (
            "e4d5",
            "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        ),
        (
            "d8d5",
            "rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
        ),
        (
            "b1c3",
            "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 3",
        ),
        (
            "d5d6",
            "rnb1kbnr/ppp1pppp/3q4/8/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4",
        ),
    ] {
        position = play(&position, mv);
        assert_eq!(position.to_fen(), expected);
    }
}

#[test]
fn ordinary_moves_clear_expired_targets_and_reset_clocks_only_when_required() {
    for (before, mv, expected) in [
        (
            "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 8",
            "e1d1",
            "4k3/8/8/3p4/8/8/8/3K4 b - - 1 8",
        ),
        (
            "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 8",
            "e8d8",
            "3k4/8/8/8/4P3/8/8/4K3 w - - 1 9",
        ),
        (
            "4k3/8/8/8/8/8/P7/4K3 w - - 17 8",
            "a2a3",
            "4k3/8/8/8/8/P7/8/4K3 b - - 0 8",
        ),
        (
            "4k3/p7/8/8/8/8/8/4K3 b - - 17 8",
            "a7a6",
            "4k3/8/p7/8/8/8/8/4K3 w - - 0 9",
        ),
        (
            "4k3/8/8/8/3r4/2B5/8/4K3 w - - 17 8",
            "c3d4",
            "4k3/8/8/8/3B4/8/8/4K3 b - - 0 8",
        ),
        (
            "4k3/8/2b5/3R4/8/8/8/4K3 b - - 17 8",
            "c6d5",
            "4k3/8/8/3b4/8/8/8/4K3 w - - 0 9",
        ),
    ] {
        assert_eq!(play(&before.parse().unwrap(), mv).to_fen(), expected);
    }
}

fn castle_position(side_to_move: Color, rights: CastlingRights) -> Position {
    let mut board = Board::empty();
    for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
        for (file, kind) in [
            (0, PieceKind::Rook),
            (4, PieceKind::King),
            (7, PieceKind::Rook),
        ] {
            board.set_piece(
                Square::from_coords(file, rank).unwrap(),
                Piece::new(color, kind),
            );
        }
    }
    Position::new(board, side_to_move, rights, None, 5, 12).unwrap()
}

#[test]
fn king_moves_rook_moves_and_corner_captures_only_remove_affected_rights() {
    let all = [
        (Color::White, CastlingSide::Kingside),
        (Color::White, CastlingSide::Queenside),
        (Color::Black, CastlingSide::Kingside),
        (Color::Black, CastlingSide::Queenside),
    ];
    for subset in 0..16 {
        let mut rights = CastlingRights::NONE;
        for (i, &(color, side)) in all.iter().enumerate() {
            if subset & (1 << i) != 0 {
                rights.insert(color, side);
            }
        }
        for (color, mv, affected) in [
            (Color::White, "e1d1", 0b0011),
            (Color::Black, "e8d8", 0b1100),
            (Color::White, "h1h2", 0b0001),
            (Color::White, "a1a2", 0b0010),
            (Color::Black, "h8h7", 0b0100),
            (Color::Black, "a8a7", 0b1000),
            (Color::White, "h1h8", 0b0101),
            (Color::Black, "h8h1", 0b0101),
            (Color::White, "a1a8", 0b1010),
            (Color::Black, "a8a1", 0b1010),
        ] {
            let next = play(&castle_position(color, rights), mv);
            for (i, &(color, side)) in all.iter().enumerate() {
                assert_eq!(
                    next.castling_rights().contains(color, side),
                    subset & !affected & (1 << i) != 0,
                    "{mv}, subset {subset}"
                );
            }
        }
    }
}

#[test]
fn returning_rooks_does_not_restore_castling_rights() {
    let mut position = castle_position(Color::White, CastlingRights::ALL);
    for mv in ["h1h2", "h8h7", "h2h1", "h7h8"] {
        position = play(&position, mv);
    }
    assert_eq!(position.to_fen(), "r3k2r/8/8/8/8/8/8/R3K2R w Qq - 9 14");
}

#[test]
fn applying_a_pinned_move_leaves_king_safety_to_the_caller() {
    let position: Position = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1".parse().unwrap();
    assert!(!position.in_check(Color::White));
    let next = play(&position, "e2d2");
    assert!(next.in_check(Color::White));
    assert!(!next.in_check(next.side_to_move()));
    assert_eq!(next.to_fen(), "k3r3/8/8/8/8/8/3R4/4K3 b - - 1 1");
}

#[test]
#[should_panic(expected = "halfmove clock overflow")]
fn halfmove_overflow_is_explicit() {
    let position: Position = "4k3/8/8/8/8/8/8/4K3 w - - 4294967295 1".parse().unwrap();
    play(&position, "e1d1");
}

#[test]
#[should_panic(expected = "fullmove number overflow")]
fn fullmove_overflow_is_explicit() {
    let position: Position = "4k3/8/8/8/8/8/8/4K3 b - - 0 4294967295".parse().unwrap();
    play(&position, "e8d8");
}

#[test]
fn promotions_replace_the_pawn_and_revoke_captured_rooks_rights() {
    for (kind, symbol) in [
        (PieceKind::Knight, 'N'),
        (PieceKind::Bishop, 'B'),
        (PieceKind::Rook, 'R'),
        (PieceKind::Queen, 'Q'),
    ] {
        for (before, mv, expected, color) in [
            (
                "k7/4P3/8/8/8/8/8/K7 w - - 17 8",
                "e7e8",
                "k3X3/8/8/8/8/8/8/K7 b - - 0 8",
                Color::White,
            ),
            (
                "k7/8/8/8/8/8/4p3/K7 b - - 17 8",
                "e2e1",
                "k7/8/8/8/8/8/8/K3X3 w - - 0 9",
                Color::Black,
            ),
            (
                "r3k2r/6P1/8/8/8/8/8/4K3 w kq - 17 8",
                "g7h8",
                "r3k2X/8/8/8/8/8/8/4K3 b q - 0 8",
                Color::White,
            ),
            (
                "r3k2r/1P6/8/8/8/8/8/4K3 w kq - 17 8",
                "b7a8",
                "X3k2r/8/8/8/8/8/8/4K3 b k - 0 8",
                Color::White,
            ),
            (
                "4k3/8/8/8/8/8/6p1/R3K2R b KQ - 17 8",
                "g2h1",
                "4k3/8/8/8/8/8/8/R3K2X w Q - 0 9",
                Color::Black,
            ),
            (
                "4k3/8/8/8/8/8/1p6/R3K2R b KQ - 17 8",
                "b2a1",
                "4k3/8/8/8/8/8/8/X3K2R w K - 0 9",
                Color::Black,
            ),
        ] {
            let symbol = if color == Color::White {
                symbol
            } else {
                symbol.to_ascii_lowercase()
            };
            let next = play_kind(&before.parse().unwrap(), mv, MoveKind::Promotion(kind));
            assert_eq!(next.to_fen(), expected.replace('X', &symbol.to_string()));
            assert!(next.board().by_kind(PieceKind::Pawn).is_empty());
        }
    }
}

#[test]
fn en_passant_removes_the_bypassed_pawn_in_both_directions_for_both_colors() {
    for (before, mv, expected) in [
        (
            "4k3/8/8/pP6/8/8/8/4K3 w - a6 0 8",
            "b5a6",
            "4k3/8/P7/8/8/8/8/4K3 b - - 0 8",
        ),
        (
            "4k3/8/8/6Pp/8/8/8/4K3 w - h6 0 8",
            "g5h6",
            "4k3/8/7P/8/8/8/8/4K3 b - - 0 8",
        ),
        (
            "4k3/8/8/8/Pp6/8/8/4K3 b - a3 0 8",
            "b4a3",
            "4k3/8/8/8/8/p7/8/4K3 w - - 0 9",
        ),
        (
            "4k3/8/8/8/6pP/8/8/4K3 b - h3 0 8",
            "g4h3",
            "4k3/8/8/8/8/7p/8/4K3 w - - 0 9",
        ),
    ] {
        let next = play_kind(&before.parse().unwrap(), mv, MoveKind::EnPassant);
        assert_eq!(next.to_fen(), expected);
        assert_eq!(next.board().by_kind(PieceKind::Pawn).count(), 1);
    }
}

#[test]
fn castling_moves_both_pieces_and_updates_shared_metadata() {
    for (color, mv, expected) in [
        (Color::White, "e1g1", "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 6 12"),
        (Color::White, "e1c1", "r3k2r/8/8/8/8/8/8/2KR3R b kq - 6 12"),
        (Color::Black, "e8g8", "r4rk1/8/8/8/8/8/8/R3K2R w KQ - 6 13"),
        (Color::Black, "e8c8", "2kr3r/8/8/8/8/8/8/R3K2R w KQ - 6 13"),
    ] {
        let position = castle_position(color, CastlingRights::ALL);
        assert_eq!(
            play_kind(&position, mv, MoveKind::Castling).to_fen(),
            expected
        );
    }
}

#[test]
fn special_move_king_safety_is_left_to_the_caller() {
    // Removing both rank-five pawns exposes the white king to the rook.
    let position: Position = "k7/8/8/K2pP2r/8/8/8/8 w - d6 0 8".parse().unwrap();
    assert!(!position.in_check(Color::White));
    let next = play_kind(&position, "e5d6", MoveKind::EnPassant);
    assert!(next.in_check(Color::White));
    assert_eq!(next.to_fen(), "k7/8/3P4/K6r/8/8/8/8 b - - 0 8");

    // The final king square is safe, but the transit square f1 is attacked.
    let position: Position = "k4r2/8/8/8/8/8/8/4K2R w K - 5 12".parse().unwrap();
    assert!(!position.in_check(Color::White));
    assert!(position.is_square_attacked(square("f1"), Color::Black));
    let next = play_kind(&position, "e1g1", MoveKind::Castling);
    assert!(!next.in_check(Color::White));
    assert_eq!(next.to_fen(), "k4r2/8/8/8/8/8/8/5RK1 b - - 6 12");
}
