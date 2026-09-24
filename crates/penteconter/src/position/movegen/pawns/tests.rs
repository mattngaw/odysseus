use super::*;
use crate::{Board, CastlingRights, Piece};

fn square(name: &str) -> Square {
    let bytes = name.as_bytes();
    Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap()
}

fn generated(position: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    position.generate_pawn_moves(&mut moves);
    moves
}

#[test]
fn opening_has_sixteen_pawn_moves_and_supports_appending_and_reuse() {
    for (turn, from_rank, one_rank, two_rank) in [("w", 2, 3, 4), ("b", 7, 6, 5)] {
        let position: Position =
            format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {turn} KQkq - 0 1")
                .parse()
                .unwrap();
        let expected: Vec<String> = [one_rank, two_rank]
            .into_iter()
            .flat_map(|to_rank| {
                ('a'..='h').map(move |file| format!("{file}{from_rank}{file}{to_rank}"))
            })
            .collect();
        let mut moves = Vec::with_capacity(17);
        let sentinel = Move::new(square("a1"), square("b1"), MoveKind::Normal).unwrap();
        moves.push(sentinel);
        let allocation = moves.as_ptr();
        position.generate_pawn_moves(&mut moves);
        assert_eq!(moves[0], sentinel);
        assert_eq!(
            moves[1..]
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(moves.iter().all(|mv| mv.kind() == MoveKind::Normal));
        moves.clear();
        position.generate_pawn_moves(&mut moves);
        assert_eq!(
            moves.iter().map(ToString::to_string).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(moves.as_ptr(), allocation);
    }
}

#[test]
fn double_pushes_require_both_squares_empty_and_the_starting_rank() {
    for (color, start, intermediate, destination, advanced) in
        [(Color::White, 1, 2, 3, 3), (Color::Black, 6, 5, 4, 4)]
    {
        for file in 0..8 {
            let from = Square::from_coords(file, start).unwrap();
            let to = Square::from_coords(file, destination).unwrap();
            for blocked_rank in [None, Some(intermediate), Some(destination)] {
                for blocker_color in [Color::White, Color::Black] {
                    let mut board = kings();
                    board.set_piece(from, Piece::new(color, PieceKind::Pawn));
                    if let Some(rank) = blocked_rank {
                        board.set_piece(
                            Square::from_coords(file, rank).unwrap(),
                            Piece::new(blocker_color, PieceKind::Knight),
                        );
                    }
                    let position =
                        Position::new(board, color, CastlingRights::NONE, None, 0, 1).unwrap();
                    let moves = generated(&position);
                    assert_eq!(
                        moves.iter().any(|mv| mv.from() == from && mv.to() == to),
                        blocked_rank.is_none()
                    );
                    assert_eq!(
                        moves.len(),
                        match blocked_rank {
                            None => 2,
                            Some(rank) if rank == intermediate => 0,
                            _ => 1,
                        }
                    );
                }
            }
            let mut board = kings();
            board.set_piece(
                Square::from_coords(file, advanced).unwrap(),
                Piece::new(color, PieceKind::Pawn),
            );
            let position = Position::new(board, color, CastlingRights::NONE, None, 0, 1).unwrap();
            assert_eq!(generated(&position).len(), 1);
        }
    }
}

#[test]
fn promotion_pushes_and_shared_capture_targets_keep_all_four_choices() {
    for (fen, paths) in [
        (
            "1r2k3/P1P5/8/8/8/8/8/4K3 w - - 0 1",
            ["a7a8", "c7c8", "c7b8", "a7b8"],
        ),
        (
            "4k3/8/8/8/8/8/p1p5/1R2K3 b - - 0 1",
            ["a2a1", "c2c1", "c2b1", "a2b1"],
        ),
    ] {
        let position: Position = fen.parse().unwrap();
        let moves = generated(&position);
        let expected: Vec<String> = paths
            .into_iter()
            .flat_map(|path| {
                ['n', 'b', 'r', 'q']
                    .into_iter()
                    .map(move |suffix| format!("{path}{suffix}"))
            })
            .collect();
        assert_eq!(
            moves.iter().map(ToString::to_string).collect::<Vec<_>>(),
            expected
        );
        assert!(
            moves
                .iter()
                .all(|mv| matches!(mv.kind(), MoveKind::Promotion(_)))
        );
        check_against_reference(&position);
    }
}

#[test]
fn en_passant_uses_current_target_preserves_both_sources_and_does_not_filter_check() {
    for (fen, expected) in [
        ("4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 8", vec!["e5d6", "c5d6"]),
        ("4k3/8/8/8/2pPp3/8/8/4K3 b - d3 0 8", vec!["e4d3", "c4d3"]),
        ("4k3/8/8/pP6/8/8/8/4K3 w - a6 0 8", vec!["b5a6"]),
        ("4k3/8/8/6Pp/8/8/8/4K3 w - h6 0 8", vec!["g5h6"]),
        ("4k3/8/8/8/Pp6/8/8/4K3 b - a3 0 8", vec!["b4a3"]),
        ("4k3/8/8/8/6pP/8/8/4K3 b - h3 0 8", vec!["g4h3"]),
        ("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 8", vec![]),
        ("4k3/8/8/2PpP3/8/8/8/4K3 w - - 0 8", vec![]),
    ] {
        let position: Position = fen.parse().unwrap();
        let moves = generated(&position);
        assert_eq!(
            moves
                .iter()
                .filter(|mv| mv.kind() == MoveKind::EnPassant)
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            expected
        );
        check_against_reference(&position);
    }
    let position: Position = "k7/8/8/K2pP2r/8/8/8/8 w - d6 0 8".parse().unwrap();
    let moves = generated(&position);
    let ep = moves
        .iter()
        .find(|mv| mv.kind() == MoveKind::EnPassant)
        .unwrap();
    assert!(!position.in_check(Color::White));
    assert!(position.apply(*ep).in_check(Color::White));
}

fn kings() -> Board {
    let mut board = Board::empty();
    board.set_piece(square("e1"), Piece::new(Color::White, PieceKind::King));
    board.set_piece(square("e8"), Piece::new(Color::Black, PieceKind::King));
    board
}

// Per-pawn coordinate oracle: no shifts, attack helpers, or production
// emission helper. Scan all destinations to check pushes and diagonals.
fn reference(position: &Position) -> Vec<Move> {
    let color = position.side_to_move();
    let forward = if color == Color::White { 1 } else { -1 };
    let start = if color == Color::White { 1 } else { 6 };
    let mut moves = Vec::new();
    for from_index in 0..64 {
        let from = Square::new(from_index).unwrap();
        if position.board().piece_at(from) != Some(Piece::new(color, PieceKind::Pawn)) {
            continue;
        }
        for to_index in 0..64 {
            let to = Square::new(to_index).unwrap();
            let df = to.file() as i8 - from.file() as i8;
            let dr = to.rank() as i8 - from.rank() as i8;
            let target = position.board().piece_at(to);
            let push = df == 0
                && target.is_none()
                && (dr == forward
                    || (from.rank() == start
                        && dr == 2 * forward
                        && position
                            .board()
                            .piece_at(
                                Square::from_coords(
                                    from.file(),
                                    (from.rank() as i8 + forward) as u8,
                                )
                                .unwrap(),
                            )
                            .is_none()));
            let capture = df.abs() == 1
                && dr == forward
                && target
                    .is_some_and(|piece| piece.color() != color && piece.kind() != PieceKind::King);
            let ep = df.abs() == 1 && dr == forward && Some(to) == position.en_passant_target();
            if !(push || capture || ep) {
                continue;
            }
            if to.rank() == 0 || to.rank() == 7 {
                for kind in [
                    PieceKind::Knight,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                    PieceKind::Queen,
                ] {
                    moves.push(Move::new(from, to, MoveKind::Promotion(kind)).unwrap());
                }
            } else {
                moves.push(
                    Move::new(
                        from,
                        to,
                        if ep {
                            MoveKind::EnPassant
                        } else {
                            MoveKind::Normal
                        },
                    )
                    .unwrap(),
                );
            }
        }
    }
    moves
}

fn check_against_reference(position: &Position) {
    let before = *position;
    let mut actual = generated(position);
    let mut expected = reference(position);
    let key = |mv: &Move| {
        (
            mv.from().index(),
            mv.to().index(),
            match mv.kind() {
                MoveKind::Normal => 0,
                MoveKind::Promotion(kind) => 1 + kind.index(),
                MoveKind::EnPassant => 7,
                MoveKind::Castling => 8,
            },
        )
    };
    actual.sort_by_key(key);
    expected.sort_by_key(key);
    assert!(actual.windows(2).all(|pair| pair[0] != pair[1]));
    assert_eq!(actual, expected, "{position}");
    for mv in actual {
        let next = position.apply(mv);
        assert!(next.board().is_consistent());
        assert_eq!(next.to_fen().parse::<Position>().unwrap(), next);
    }
    assert_eq!(*position, before);
}

#[test]
fn coordinate_oracle_covers_edges_blockers_king_exclusion_and_mixed_pawn_sets() {
    let mut seed = 0x4f44_5953_5345_5553u64;
    for sample in 0..64 {
        let mut board = Board::empty();
        for index in 0..64 {
            let sq = Square::new(index).unwrap();
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = seed >> 32;
            if bits % 4 > sample % 4 {
                continue;
            }
            let color = if bits & 16 == 0 {
                Color::White
            } else {
                Color::Black
            };
            let kind = if sq.rank() == 0 || sq.rank() == 7 || bits & 32 != 0 {
                PieceKind::Rook
            } else {
                PieceKind::Pawn
            };
            board.set_piece(sq, Piece::new(color, kind));
        }
        board.set_piece(
            Square::new(sample as u8).unwrap(),
            Piece::new(Color::White, PieceKind::King),
        );
        board.set_piece(
            Square::new(((sample + 31) % 64) as u8).unwrap(),
            Piece::new(Color::Black, PieceKind::King),
        );
        for color in [Color::White, Color::Black] {
            let position = Position::new(board, color, CastlingRights::NONE, None, 17, 8).unwrap();
            check_against_reference(&position);
        }
    }
}
