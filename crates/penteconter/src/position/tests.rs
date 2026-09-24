use super::*;
use crate::Piece;

fn square(file: u8, rank: u8) -> Square {
    Square::from_coords(file, rank).unwrap()
}

fn kings() -> Board {
    let mut board = Board::empty();
    board.set_piece(square(4, 0), Piece::new(Color::White, PieceKind::King));
    board.set_piece(square(4, 7), Piece::new(Color::Black, PieceKind::King));
    board
}

fn position(board: Board) -> Result<Position, PositionError> {
    Position::new(board, Color::White, CastlingRights::NONE, None, 0, 1)
}

#[test]
fn relative_occupancy_changes_without_changing_absolute_placement() {
    let mut board = kings();
    board.set_piece(square(0, 2), Piece::new(Color::White, PieceKind::Knight));
    board.set_piece(square(6, 4), Piece::new(Color::Black, PieceKind::Rook));
    for color in [Color::White, Color::Black] {
        let p = Position::new(board, color, CastlingRights::NONE, None, 17, 42).unwrap();
        assert_eq!(p.board(), &board);
        assert_eq!(p.side_to_move(), color);
        assert_eq!(p.ours(), board.by_color(color));
        assert_eq!(p.theirs(), board.by_color(color.opposite()));
        assert_eq!(p.castling_rights(), CastlingRights::NONE);
        assert_eq!(p.en_passant_target(), None);
        assert_eq!(p.halfmove_clock(), 17);
        assert_eq!(p.fullmove_number(), 42);
    }
}

#[test]
fn rejects_missing_and_duplicate_kings_of_either_color() {
    for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
        let mut missing = kings();
        missing.remove_piece(square(4, rank));
        assert_eq!(
            position(missing),
            Err(PositionError::KingCount { color, count: 0 })
        );
        let mut duplicate = kings();
        duplicate.set_piece(square(0, rank), Piece::new(color, PieceKind::King));
        assert_eq!(
            position(duplicate),
            Err(PositionError::KingCount { color, count: 2 })
        );
    }
}

#[test]
fn rejects_pawns_on_both_back_ranks_but_accepts_interior_ranks() {
    for color in [Color::White, Color::Black] {
        for rank in 0..8 {
            for file in 0..8 {
                let mut board = kings();
                // Move kings away from the back ranks so all files are tested.
                board.remove_piece(square(4, 0));
                board.remove_piece(square(4, 7));
                board.set_piece(
                    square((file + 1) % 8, 2),
                    Piece::new(Color::White, PieceKind::King),
                );
                board.set_piece(
                    square((file + 1) % 8, 5),
                    Piece::new(Color::Black, PieceKind::King),
                );
                board.set_piece(square(file, rank), Piece::new(color, PieceKind::Pawn));
                let result = position(board);
                if rank == 0 || rank == 7 {
                    assert_eq!(result, Err(PositionError::PawnOnBackRank));
                } else {
                    assert!(result.is_ok());
                }
            }
        }
    }
}

#[test]
fn en_passant_checks_target_rank_and_occupancy_for_both_turns() {
    for (color, target_rank, pawn_rank) in [(Color::White, 5, 4), (Color::Black, 2, 3)] {
        for file in 0..8 {
            let mut board = kings();
            // A double-pushed pawn with no opposing pawn available to capture.
            board.set_piece(
                square(file, pawn_rank),
                Piece::new(color.opposite(), PieceKind::Pawn),
            );
            for rank in 0..8 {
                let target = square(file, rank);
                let result = Position::new(board, color, CastlingRights::NONE, Some(target), 0, 2);
                if rank == target_rank {
                    assert_eq!(result.unwrap().en_passant_target(), Some(target));
                } else {
                    assert_eq!(
                        result,
                        Err(PositionError::InvalidEnPassantTarget { square: target })
                    );
                }
            }
            let target = square(file, target_rank);
            board.set_piece(target, Piece::new(color, PieceKind::Knight));
            assert_eq!(
                Position::new(board, color, CastlingRights::NONE, Some(target), 0, 2),
                Err(PositionError::InvalidEnPassantTarget { square: target })
            );
        }
    }
}

#[test]
fn retained_castling_rights_require_the_matching_king_and_rook() {
    for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
        for (side, file) in [(CastlingSide::Kingside, 7), (CastlingSide::Queenside, 0)] {
            let mut rights = CastlingRights::NONE;
            rights.insert(color, side);
            let mut board = kings();
            let rook = square(file, rank);
            board.set_piece(rook, Piece::new(color, PieceKind::Rook));
            // The path can be blocked: these are rights, not legal moves.
            let blocker_file = if side == CastlingSide::Kingside { 5 } else { 1 };
            board.set_piece(
                square(blocker_file, rank),
                Piece::new(color, PieceKind::Bishop),
            );
            let build = |board| Position::new(board, color, rights, None, 0, 1);
            assert_eq!(build(board).unwrap().castling_rights(), rights);
            let expected = Err(PositionError::InconsistentCastling { color, side });
            for replacement in [
                None,
                Some(Piece::new(color.opposite(), PieceKind::Rook)),
                Some(Piece::new(color, PieceKind::Knight)),
            ] {
                let mut changed = board;
                changed.remove_piece(rook);
                if let Some(piece) = replacement {
                    changed.set_piece(rook, piece);
                }
                assert_eq!(build(changed), expected);
                // Missing rooks are fine when no corresponding right is asserted.
                assert!(Position::new(changed, color, CastlingRights::NONE, None, 0, 1).is_ok());
            }
            board.remove_piece(square(4, rank));
            board.set_piece(square(3, rank), Piece::new(color, PieceKind::King));
            assert_eq!(build(board), expected);
        }
    }
}

#[test]
fn en_passant_requires_a_consistent_double_push() {
    for (color, target_rank, pawn_rank, origin_rank) in
        [(Color::White, 5, 4, 6), (Color::Black, 2, 3, 1)]
    {
        for file in 0..8 {
            let target = square(file, target_rank);
            let pawn_square = square(file, pawn_rank);
            let mut board = kings();
            let build = |board, clock| {
                Position::new(board, color, CastlingRights::NONE, Some(target), clock, 2)
            };
            let expected = Err(PositionError::InconsistentEnPassant { square: target });
            for replacement in [
                None,
                Some(Piece::new(color, PieceKind::Pawn)),
                Some(Piece::new(color.opposite(), PieceKind::Knight)),
            ] {
                board.remove_piece(pawn_square);
                if let Some(piece) = replacement {
                    board.set_piece(pawn_square, piece);
                }
                assert_eq!(build(board, 0), expected);
            }
            board.set_piece(pawn_square, Piece::new(color.opposite(), PieceKind::Pawn));
            assert!(build(board, 0).is_ok());
            assert_eq!(build(board, 1), expected);
            board.set_piece(
                square(file, origin_rank),
                Piece::new(color, PieceKind::Knight),
            );
            assert_eq!(build(board, 0), expected);
        }
    }
}

#[test]
fn counters_preserve_their_values_and_fullmove_starts_at_one() {
    assert_eq!(
        Position::new(kings(), Color::White, CastlingRights::NONE, None, 0, 0),
        Err(PositionError::ZeroFullmoveNumber)
    );
    let p = Position::new(
        kings(),
        Color::White,
        CastlingRights::NONE,
        None,
        u32::MAX,
        u32::MAX,
    )
    .unwrap();
    assert_eq!(p.halfmove_clock(), u32::MAX);
    assert_eq!(p.fullmove_number(), u32::MAX);
}
