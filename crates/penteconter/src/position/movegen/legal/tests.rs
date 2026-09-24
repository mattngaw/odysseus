use super::*;
use crate::{Board, CastlingRights, CastlingSide, Color, Piece, PieceKind};

fn sq(file: u8, rank: u8) -> Square {
    Square::from_coords(file, rank).unwrap()
}

fn normal(from: &str, to: &str) -> Move {
    let square = |s: &str| sq(s.as_bytes()[0] - b'a', s.as_bytes()[1] - b'1');
    Move::new(square(from), square(to), MoveKind::Normal).unwrap()
}

fn legal(position: &Position) -> Vec<Move> {
    let before = *position;
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    for (i, &mv) in moves.iter().enumerate() {
        assert!(!moves[..i].contains(&mv));
        let next = position.apply(mv);
        assert!(!next.in_check(position.side_to_move()));
        assert_eq!(next.to_fen().parse::<Position>().unwrap(), next);
    }
    assert_eq!(*position, before);
    moves
}

#[test]
fn combines_components_and_preserves_prefix_order_and_allocation() {
    let start: Position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        .parse()
        .unwrap();
    let mut candidates = Vec::new();
    start.generate_pseudo_legal_moves(&mut candidates);
    assert_eq!(candidates.len(), 20);
    assert_eq!(legal(&start), candidates);

    let position: Position = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1".parse().unwrap();
    candidates.clear();
    position.generate_pseudo_legal_moves(&mut candidates);
    let expected: Vec<Move> = candidates
        .iter()
        .copied()
        .filter(|&mv| !position.apply(mv).in_check(Color::White))
        .collect();
    assert!(expected.len() < candidates.len());
    let sentinel = normal("e2", "d2"); // Illegal here, but existing entries are opaque.
    let mut moves = Vec::with_capacity(candidates.len() + 1);
    let allocation = moves.as_ptr();
    moves.push(sentinel);
    position.generate_legal_moves(&mut moves);
    assert_eq!(moves[0], sentinel);
    assert_eq!(&moves[1..], expected);
    moves.clear();
    position.generate_legal_moves(&mut moves);
    assert_eq!(moves, expected);
    assert_eq!(moves.as_ptr(), allocation);
}

#[test]
fn pins_and_double_check_are_filtered_but_capturing_the_checker_is_allowed() {
    let position: Position = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1".parse().unwrap();
    let moves = legal(&position);
    assert!(!moves.contains(&normal("e2", "d2")));
    assert!(moves.contains(&normal("e2", "e8")));
    let position: Position = "k3r3/8/8/8/1b6/8/R7/4K3 w - - 0 1".parse().unwrap();
    let moves = legal(&position);
    assert!(!moves.is_empty());
    assert!(moves.iter().all(|mv| mv.from() == sq(4, 0)));
}

#[test]
fn en_passant_can_expose_a_rook_or_remove_a_checking_pawn() {
    for (fen, allowed) in [
        ("k7/8/8/K2pP2r/8/8/8/8 w - d6 0 8", false),
        ("k7/8/8/3pP3/4K3/8/8/8 w - d6 0 8", true),
    ] {
        let position: Position = fen.parse().unwrap();
        let ep = Move::new(sq(4, 4), sq(3, 5), MoveKind::EnPassant).unwrap();
        let mut candidates = Vec::new();
        position.generate_pseudo_legal_moves(&mut candidates);
        assert!(candidates.contains(&ep));
        assert_eq!(legal(&position).contains(&ep), allowed);
    }
}

#[test]
fn king_safety_uses_updated_occupancy_and_counts_pinned_attackers() {
    let position: Position = "k3r3/8/8/8/8/8/4K3/8 w - - 0 1".parse().unwrap();
    assert!(!position.is_square_attacked(sq(4, 0), Color::Black));
    assert!(!legal(&position).contains(&normal("e2", "e1")));
    let position: Position = "4k3/4n3/8/6K1/8/8/8/4R3 w - - 0 1".parse().unwrap();
    assert!(!legal(&position).contains(&normal("g5", "f5")));
    let position: Position = "k4r2/8/8/8/8/8/8/4Kn2 w - - 0 1".parse().unwrap();
    assert!(!legal(&position).contains(&normal("e1", "f1")));
}

#[test]
fn castling_checks_start_transit_and_destination_but_not_the_rook_only_path() {
    for (color, rank) in [(Color::White, 0), (Color::Black, 7)] {
        for (side, rook_file, transit, destination) in [
            (CastlingSide::Kingside, 7, 5, 6),
            (CastlingSide::Queenside, 0, 3, 2),
        ] {
            // Attacks on b1/b8 or on the rook's home square alone do not
            // forbid castling. Both colors and both sides are covered.
            for attacked in [
                None,
                Some(4),
                Some(transit),
                Some(destination),
                Some(rook_file),
                Some(1),
            ] {
                let mut board = Board::empty();
                board.set_piece(sq(4, rank), Piece::new(color, PieceKind::King));
                board.set_piece(sq(rook_file, rank), Piece::new(color, PieceKind::Rook));
                board.set_piece(
                    sq(4, 7 - rank),
                    Piece::new(color.opposite(), PieceKind::King),
                );
                if let Some(file) = attacked {
                    board.set_piece(
                        sq(file, if rank == 0 { 5 } else { 2 }),
                        Piece::new(color.opposite(), PieceKind::Rook),
                    );
                }
                let mut rights = CastlingRights::NONE;
                rights.insert(color, side);
                let position = Position::new(board, color, rights, None, 5, 12).unwrap();
                let castle =
                    Move::new(sq(4, rank), sq(destination, rank), MoveKind::Castling).unwrap();
                let allowed =
                    !attacked.is_some_and(|file| [4, transit, destination].contains(&file));
                assert_eq!(
                    legal(&position).contains(&castle),
                    allowed,
                    "{color:?}, {side:?}, attacked file {attacked:?}"
                );
            }
        }
    }
}

#[test]
fn mate_and_stalemate_append_nothing_but_draw_counters_do_not_stop_generation() {
    for (fen, check) in [
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", true),
        ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", false),
    ] {
        let position: Position = fen.parse().unwrap();
        assert_eq!(position.in_check(Color::Black), check);
        assert!(legal(&position).is_empty());
        let sentinel = normal("a1", "a2");
        let mut moves = vec![sentinel];
        position.generate_legal_moves(&mut moves);
        assert_eq!(moves, [sentinel]);
    }
    let position: Position = "4k3/8/8/8/8/8/8/4K3 w - - 150 100".parse().unwrap();
    assert_eq!(legal(&position).len(), 5);
}

#[test]
fn legality_does_not_advance_maximal_counters_or_change_position() {
    for side in ["w", "b"] {
        let position: Position =
            format!("4k3/8/8/8/8/8/8/4K3 {side} - - {} {}", u32::MAX, u32::MAX)
                .parse()
                .unwrap();
        let before = position;
        let mut moves = Vec::new();
        position.generate_legal_moves(&mut moves);
        assert_eq!(moves.len(), 5);
        assert_eq!(position, before);
        assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
    }
}
