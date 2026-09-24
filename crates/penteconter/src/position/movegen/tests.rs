use super::*;
use crate::{Board, CastlingRights, Color, Piece, Square};

fn square(name: &str) -> Square {
    let bytes = name.as_bytes();
    Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap()
}

#[test]
fn opening_generates_only_four_knight_moves_for_either_turn() {
    for (turn, expected) in [
        ("w", ["b1a3", "b1c3", "g1f3", "g1h3"]),
        ("b", ["b8a6", "b8c6", "g8f6", "g8h6"]),
    ] {
        let position: Position =
            format!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {turn} KQkq - 0 1")
                .parse()
                .unwrap();
        let mut moves = Vec::new();
        position.generate_knight_and_slider_moves(&mut moves);
        assert_eq!(
            moves.iter().map(ToString::to_string).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn append_preserves_existing_entries_and_clear_reuses_capacity() {
    let position: Position = "4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1".parse().unwrap();
    let sentinel = Move::new(square("e1"), square("f1"), MoveKind::Normal).unwrap();
    let mut expected = Vec::new();
    position.generate_knight_and_slider_moves(&mut expected);
    let mut moves = Vec::with_capacity(expected.len() + 1);
    let capacity = moves.capacity();
    let allocation = moves.as_ptr();
    moves.push(sentinel);
    position.generate_knight_and_slider_moves(&mut moves);
    assert_eq!(moves[0], sentinel);
    assert_eq!(&moves[1..], expected);
    moves.clear();
    position.generate_knight_and_slider_moves(&mut moves);
    assert_eq!(moves, expected);
    assert_eq!(moves.capacity(), capacity);
    assert_eq!(moves.as_ptr(), allocation);

    let empty: Position = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".parse().unwrap();
    empty.generate_knight_and_slider_moves(&mut moves);
    assert_eq!(moves, expected);
}

#[test]
fn pinned_moves_and_moves_that_do_not_evade_check_are_kept() {
    for fen in [
        "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "k3r3/8/8/8/8/8/R7/4K3 w - - 0 1",
    ] {
        let position: Position = fen.parse().unwrap();
        let mut moves = Vec::new();
        position.generate_knight_and_slider_moves(&mut moves);
        let mv = moves.iter().find(|mv| mv.to() == square("d2")).unwrap();
        assert!(position.apply(*mv).in_check(Color::White));
    }
}

#[test]
fn sliders_stop_at_friendly_pieces_captures_and_the_enemy_king() {
    for (occupant, includes_blocker) in [('P', false), ('p', true), ('k', false)] {
        let top = if occupant == 'k' { "8" } else { "k7" };
        let position: Position = format!("{top}/8/3{occupant}4/8/3R4/8/8/K7 w - - 0 1")
            .parse()
            .unwrap();
        let mut moves = Vec::new();
        position.generate_knight_and_slider_moves(&mut moves);
        assert!(moves.iter().any(|mv| mv.to() == square("d5")));
        assert_eq!(
            moves.iter().any(|mv| mv.to() == square("d6")),
            includes_blocker
        );
        assert!(
            !moves
                .iter()
                .any(|mv| [square("d7"), square("d8")].contains(&mv.to()))
        );
    }
}

// Deliberately uses mailbox geometry and stepwise path clearance rather
// than attack tables, bitboard enumeration, or the production target mask.
fn reference_moves(position: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    for from_index in 0..64 {
        let from = Square::new(from_index).unwrap();
        let Some(piece) = position.board().piece_at(from) else {
            continue;
        };
        if piece.color() != position.side_to_move() {
            continue;
        }
        for to_index in 0..64 {
            let to = Square::new(to_index).unwrap();
            if from == to
                || position.board().piece_at(to).is_some_and(|target| {
                    target.color() == piece.color() || target.kind() == PieceKind::King
                })
            {
                continue;
            }
            let df = to.file() as i8 - from.file() as i8;
            let dr = to.rank() as i8 - from.rank() as i8;
            let diagonal = df.abs() == dr.abs();
            let straight = df == 0 || dr == 0;
            let geometry = match piece.kind() {
                PieceKind::Knight => {
                    (df.abs() == 1 && dr.abs() == 2) || (df.abs() == 2 && dr.abs() == 1)
                }
                PieceKind::Bishop => diagonal,
                PieceKind::Rook => straight,
                PieceKind::Queen => diagonal || straight,
                _ => false,
            };
            if !geometry {
                continue;
            }
            if piece.kind() != PieceKind::Knight {
                let blocked = (1..df.abs().max(dr.abs())).any(|step| {
                    let square = Square::from_coords(
                        (from.file() as i8 + step * df.signum()) as u8,
                        (from.rank() as i8 + step * dr.signum()) as u8,
                    )
                    .unwrap();
                    position.board().piece_at(square).is_some()
                });
                if blocked {
                    continue;
                }
            }
            moves.push(Move::new(from, to, MoveKind::Normal).unwrap());
        }
    }
    moves
}

#[test]
fn generated_moves_match_mailbox_oracle_and_apply_consistently() {
    let kinds = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
    ];
    let mut seed = 0x4f44_5953_5345_5553u64;
    for sample in 0..32 {
        let mut board = Board::empty();
        for index in 0..64 {
            let square = Square::new(index).unwrap();
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = seed >> 32;
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
        board.set_piece(square("a1"), Piece::new(Color::White, PieceKind::King));
        board.set_piece(square("h8"), Piece::new(Color::Black, PieceKind::King));
        for color in [Color::White, Color::Black] {
            let position = Position::new(board, color, CastlingRights::NONE, None, 9, 20).unwrap();
            let mut moves = Vec::new();
            position.generate_knight_and_slider_moves(&mut moves);
            let key = |mv: &Move| {
                (
                    position.board().piece_at(mv.from()).unwrap().kind().index(),
                    mv.from().index(),
                    mv.to().index(),
                )
            };
            // Strict ordering also rejects duplicate emissions.
            assert!(moves.windows(2).all(|pair| key(&pair[0]) < key(&pair[1])));
            let mut expected = reference_moves(&position);
            expected.sort_by_key(key);
            assert_eq!(moves, expected, "sample {sample}, {color:?}");
            for mv in moves {
                let next = position.apply(mv);
                assert!(next.board().is_consistent());
                assert_eq!(next.to_fen().parse::<Position>().unwrap(), next);
            }
            assert_eq!(*position.board(), board);
        }
    }
}
