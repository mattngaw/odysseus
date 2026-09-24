use super::*;

const COLORS: [Color; 2] = [Color::White, Color::Black];
const KINDS: [PieceKind; 6] = [
    PieceKind::Pawn,
    PieceKind::Knight,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::Queen,
    PieceKind::King,
];

fn all_pieces() -> [Piece; 12] {
    std::array::from_fn(|i| Piece::new(COLORS[i / 6], KINDS[i % 6]))
}

fn assert_single_piece(board: &Board, square: Square, piece: Piece) {
    let singleton = Bitboard::from_square(square);
    assert_eq!(board.piece_at(square), Some(piece));
    assert_eq!(board.occupied(), singleton);
    for color in COLORS {
        let expected = if color == piece.color() {
            singleton
        } else {
            Bitboard::EMPTY
        };
        assert_eq!(board.by_color(color), expected);
    }
    for kind in KINDS {
        let expected = if kind == piece.kind() {
            singleton
        } else {
            Bitboard::EMPTY
        };
        assert_eq!(board.by_kind(kind), expected);
    }
    assert_eq!(board.pieces(piece.color(), piece.kind()), singleton);
    assert!(board.is_consistent());
}

fn assert_matches_placement(board: &Board, placement: &[Option<Piece>; 64]) {
    for (index, expected) in placement.iter().copied().enumerate() {
        let square = Square::new(index as u8).unwrap();
        assert_eq!(board.piece_at(square), expected);
        assert_eq!(board.occupied().contains(square), expected.is_some());
        for color in COLORS {
            assert_eq!(
                board.by_color(color).contains(square),
                expected.is_some_and(|piece| piece.color() == color)
            );
        }
        for kind in KINDS {
            assert_eq!(
                board.by_kind(kind).contains(square),
                expected.is_some_and(|piece| piece.kind() == kind)
            );
            for color in COLORS {
                assert_eq!(
                    board.pieces(color, kind).contains(square),
                    expected == Some(Piece::new(color, kind))
                );
            }
        }
    }
    assert!(board.is_consistent());
}

#[test]
fn empty_board_has_no_pieces() {
    let board = Board::empty();
    assert_eq!(board, Board::default());
    assert_matches_placement(&board, &[None; 64]);
    assert_eq!(size_of::<Board>(), 128);
}

#[test]
fn every_piece_can_be_placed_and_removed_on_every_square() {
    for index in 0..64 {
        let square = Square::new(index).unwrap();
        for piece in all_pieces() {
            let mut board = Board::empty();
            assert_eq!(board.remove_piece(square), None);
            assert_eq!(board.set_piece(square, piece), None);
            assert_single_piece(&board, square, piece);
            assert_eq!(board.remove_piece(square), Some(piece));
            assert_eq!(board.remove_piece(square), None);
            assert_eq!(board, Board::empty());
        }
    }
}

#[test]
fn every_replacement_preserves_only_the_new_membership() {
    for index in 0..64 {
        let square = Square::new(index).unwrap();
        for before in all_pieces() {
            for after in all_pieces() {
                let mut board = Board::empty();
                board.set_piece(square, before);
                assert_eq!(board.set_piece(square, after), Some(before));
                assert_single_piece(&board, square, after);
            }
        }
    }
}

#[test]
fn mixed_placements_preserve_untouched_squares_and_copies() {
    let mut board = Board::empty();
    let mut placement = [None; 64];
    let pieces = all_pieces();

    for step in 0..128 {
        // Visit every square once per pass, changing its piece on pass two.
        let index = (step * 37) % 64;
        let square = Square::new(index as u8).unwrap();
        let piece = pieces[step % pieces.len()];
        let snapshot = board;
        assert_eq!(board.set_piece(square, piece), placement[index]);
        assert_matches_placement(&snapshot, &placement);
        placement[index] = Some(piece);
        assert_matches_placement(&board, &placement);
    }
    assert_eq!(board.occupied(), Bitboard::FULL);

    for step in 0..64 {
        let index = (step * 19) % 64;
        let square = Square::new(index as u8).unwrap();
        assert_eq!(board.remove_piece(square), placement[index]);
        placement[index] = None;
        assert_matches_placement(&board, &placement);
    }
    assert_eq!(board, Board::empty());
}

#[test]
fn consistency_check_detects_disagreement_in_each_representation() {
    let square = Square::new(0).unwrap();
    let piece = Piece::new(Color::White, PieceKind::Rook);
    let mut board = Board::empty();
    board.set_piece(square, piece);

    let mut wrong_color = board;
    wrong_color.by_color[Color::Black.index()].insert(square);
    assert!(!wrong_color.is_consistent());

    let mut wrong_kind = board;
    wrong_kind.by_kind[PieceKind::Bishop.index()].insert(square);
    assert!(!wrong_kind.is_consistent());

    let mut missing_kind = board;
    missing_kind.by_kind[PieceKind::Rook.index()].remove(square);
    assert!(!missing_kind.is_consistent());

    let mut wrong_mailbox = board;
    wrong_mailbox.mailbox[square.index()] = None;
    assert!(!wrong_mailbox.is_consistent());
}
