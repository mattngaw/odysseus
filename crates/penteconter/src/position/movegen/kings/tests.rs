use super::*;
use crate::{Board, CastlingRights, Piece};

fn sq(file: u8, rank: u8) -> Square {
    Square::from_coords(file, rank).unwrap()
}

fn generated(position: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    position.generate_king_moves(&mut moves);
    moves
}

fn home_board() -> Board {
    let mut board = Board::empty();
    for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
        for (file, kind) in [
            (0, PieceKind::Rook),
            (4, PieceKind::King),
            (7, PieceKind::Rook),
        ] {
            board.set_piece(sq(file, rank), Piece::new(color, kind));
        }
    }
    board
}

#[test]
fn ordinary_moves_match_coordinate_geometry_for_every_king_square_and_color() {
    for from_index in 0..64 {
        let from = Square::new(from_index).unwrap();
        for color in [Color::White, Color::Black] {
            for pattern in 0..4 {
                let mut board = Board::empty();
                for index in 0..64 {
                    if pattern == 0 || index % 4 != pattern - 1 {
                        continue;
                    }
                    let owner = if index % 3 == 0 {
                        color
                    } else {
                        color.opposite()
                    };
                    board.set_piece(
                        Square::new(index).unwrap(),
                        Piece::new(owner, PieceKind::Knight),
                    );
                }
                // Include neighboring enemy kings to distinguish excluding
                // their square from filtering squares they attack.
                let other = Square::new((from_index + 1) % 64).unwrap();
                board.set_piece(other, Piece::new(color.opposite(), PieceKind::King));
                board.set_piece(from, Piece::new(color, PieceKind::King));
                let position =
                    Position::new(board, color, CastlingRights::NONE, None, 7, 12).unwrap();
                let expected: Vec<Move> = (0..64)
                    .filter_map(|index| {
                        let to = Square::new(index).unwrap();
                        let adjacent = from
                            .file()
                            .abs_diff(to.file())
                            .max(from.rank().abs_diff(to.rank()))
                            == 1;
                        let excluded = board.piece_at(to).is_some_and(|piece| {
                            piece.color() == color || piece.kind() == PieceKind::King
                        });
                        (adjacent && !excluded)
                            .then(|| Move::new(from, to, MoveKind::Normal).unwrap())
                    })
                    .collect();
                let moves = generated(&position);
                assert_eq!(moves, expected, "{color:?} on {from}, pattern {pattern}");
                for mv in moves {
                    let next = position.apply(mv);
                    assert!(next.board().is_consistent());
                    assert_eq!(next.to_fen().parse::<Position>().unwrap(), next);
                }
                assert_eq!(*position.board(), board);
            }
        }
    }
}

#[test]
fn castling_candidates_require_the_right_and_every_path_square_empty() {
    let all_rights = [
        (Color::White, CastlingSide::Kingside),
        (Color::White, CastlingSide::Queenside),
        (Color::Black, CastlingSide::Kingside),
        (Color::Black, CastlingSide::Queenside),
    ];
    for subset in 0..16 {
        let mut rights = CastlingRights::NONE;
        for (i, &(color, side)) in all_rights.iter().enumerate() {
            if subset & (1 << i) != 0 {
                rights.insert(color, side);
            }
        }
        for (color, rank, right_shift) in [(Color::White, 0, 0), (Color::Black, 7, 2)] {
            for blockers in 0..32 {
                for blocker_color in [Color::White, Color::Black] {
                    let mut board = home_board();
                    for (i, file) in [1, 2, 3, 5, 6].into_iter().enumerate() {
                        if blockers & (1 << i) != 0 {
                            board.set_piece(
                                sq(file, rank),
                                Piece::new(blocker_color, PieceKind::Knight),
                            );
                        }
                    }
                    let position = Position::new(board, color, rights, None, 5, 12).unwrap();
                    let castles: Vec<Move> = generated(&position)
                        .into_iter()
                        .filter(|mv| mv.kind() == MoveKind::Castling)
                        .collect();
                    let mut expected = Vec::new();
                    if subset & (1 << right_shift) != 0 && blockers & 0b11000 == 0 {
                        expected
                            .push(Move::new(sq(4, rank), sq(6, rank), MoveKind::Castling).unwrap());
                    }
                    if subset & (2 << right_shift) != 0 && blockers & 0b00111 == 0 {
                        expected
                            .push(Move::new(sq(4, rank), sq(2, rank), MoveKind::Castling).unwrap());
                    }
                    assert_eq!(
                        castles, expected,
                        "rights {subset}, blockers {blockers}, {color:?}"
                    );
                    for mv in castles {
                        let next = position.apply(mv);
                        assert_eq!(next.to_fen().parse::<Position>().unwrap(), next);
                    }
                }
            }
        }
    }
}

#[test]
fn attacked_start_transit_and_destination_do_not_remove_castling_candidates() {
    for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
        for (side, transit, destination) in [
            (CastlingSide::Kingside, 5, 6),
            (CastlingSide::Queenside, 3, 2),
        ] {
            for attacked_file in [4, transit, destination] {
                let mut board = Board::empty();
                board.set_piece(sq(4, rank), Piece::new(color, PieceKind::King));
                let rook_file = if side == CastlingSide::Kingside { 7 } else { 0 };
                board.set_piece(sq(rook_file, rank), Piece::new(color, PieceKind::Rook));
                board.set_piece(
                    sq(1, 7 - rank),
                    Piece::new(color.opposite(), PieceKind::King),
                );
                board.set_piece(
                    sq(attacked_file, 7 - rank),
                    Piece::new(color.opposite(), PieceKind::Rook),
                );
                let mut rights = CastlingRights::NONE;
                rights.insert(color, side);
                let position = Position::new(board, color, rights, None, 5, 12).unwrap();
                let castle =
                    Move::new(sq(4, rank), sq(destination, rank), MoveKind::Castling).unwrap();
                assert!(position.is_square_attacked(sq(attacked_file, rank), color.opposite()));
                assert!(generated(&position).contains(&castle));
                assert_eq!(position.in_check(color), attacked_file == 4);
                assert_eq!(
                    position.apply(castle).in_check(color),
                    attacked_file == destination
                );
            }
        }
    }
}

#[test]
fn ordinary_king_moves_keep_attacked_destinations_and_defended_captures() {
    for fen in [
        "k4r2/8/8/8/8/8/8/4K3 w - - 0 1",
        "k4r2/8/8/8/8/8/8/4Kn2 w - - 0 1",
    ] {
        let position: Position = fen.parse().unwrap();
        let mv = Move::new(sq(4, 0), sq(5, 0), MoveKind::Normal).unwrap();
        assert!(generated(&position).contains(&mv));
        assert!(position.apply(mv).in_check(Color::White));
    }
}

#[test]
fn appending_preserves_entries_and_reuses_capacity_including_when_no_moves_exist() {
    let position =
        Position::new(home_board(), Color::White, CastlingRights::ALL, None, 0, 1).unwrap();
    let expected = generated(&position);
    assert_eq!(expected.len(), 7); // Five ordinary destinations, two castles.
    let mut moves = Vec::with_capacity(expected.len() + 1);
    let allocation = moves.as_ptr();
    let sentinel = Move::new(sq(0, 1), sq(0, 2), MoveKind::Normal).unwrap();
    moves.push(sentinel);
    position.generate_king_moves(&mut moves);
    assert_eq!(moves[0], sentinel);
    assert_eq!(&moves[1..], expected);
    moves.clear();
    position.generate_king_moves(&mut moves);
    assert_eq!(moves, expected);
    assert_eq!(moves.as_ptr(), allocation);
    for turn in ["w", "b"] {
        let blocked: Position =
            format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {turn} KQkq - 0 1")
                .parse()
                .unwrap();
        blocked.generate_king_moves(&mut moves);
        assert_eq!(moves, expected);
    }
}
