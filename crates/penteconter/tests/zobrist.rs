use std::collections::HashSet;

use penteconter::{Board, CastlingRights, Color, MoveKind, Piece, PieceKind, Position, Square};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[test]
fn known_positions_have_reproducible_keys() {
    // Fixed vectors make accidental seed, generator, or feature-order changes visible.
    for (fen, expected) in [
        (START, 0xd396_10a2_aaa8_1afb),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            0xed1e_a8db_7895_b7cc,
        ),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0xd491_5cb8_6f00_81d0),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 0x1def_fb21_21be_e8a4),
    ] {
        let position: Position = fen.parse().unwrap();
        assert_eq!(position.zobrist_key(), expected, "{fen}");
        assert_eq!(position.recompute_zobrist_key(), expected, "{fen}");
        assert_eq!(
            position
                .to_fen()
                .parse::<Position>()
                .unwrap()
                .recompute_zobrist_key(),
            expected,
        );
    }
}

#[test]
fn placement_distinguishes_color_kind_and_square_including_kings() {
    let base: Position = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
    let mut keys = HashSet::from([base.recompute_zobrist_key()]);
    for color in [Color::White, Color::Black] {
        for kind in [
            PieceKind::Pawn,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ] {
            for file in [2, 3] {
                let mut board = *base.board();
                if kind == PieceKind::King {
                    let original = board.pieces(color, kind).pop_first().unwrap();
                    board.remove_piece(original);
                }
                board.set_piece(
                    Square::from_coords(file, 2).unwrap(),
                    Piece::new(color, kind),
                );
                let position =
                    Position::new(board, Color::White, CastlingRights::NONE, None, 0, 1).unwrap();
                // These particular fixtures must be distinct, not a proof that
                // arbitrary different positions cannot collide.
                assert!(keys.insert(position.recompute_zobrist_key()));
            }
        }
    }
}

#[test]
fn turn_is_included_and_both_counters_are_excluded_without_changing_equality() {
    let base: Position = START.parse().unwrap();
    let black: Position = START.replace(" w ", " b ").parse().unwrap();
    assert_ne!(base.recompute_zobrist_key(), black.recompute_zobrist_key());

    for (halfmove, fullmove) in [(1, 1), (0, 2), (u32::MAX, u32::MAX)] {
        let changed = Position::new(
            *base.board(),
            base.side_to_move(),
            base.castling_rights(),
            None,
            halfmove,
            fullmove,
        )
        .unwrap();
        assert_ne!(base, changed);
        assert_eq!(
            base.recompute_zobrist_key(),
            changed.recompute_zobrist_key()
        );
    }
}

#[test]
fn all_castling_rights_contribute_even_when_paths_are_blocked() {
    let mut keys = HashSet::new();
    for subset in 0..16 {
        let mut rights = String::new();
        for (index, symbol) in ['K', 'Q', 'k', 'q'].into_iter().enumerate() {
            if subset & (1 << index) != 0 {
                rights.push(symbol);
            }
        }
        if rights.is_empty() {
            rights.push('-');
        }
        let position: Position = START.replace("KQkq", &rights).parse().unwrap();
        assert!(keys.insert(position.recompute_zobrist_key()), "{rights}");
    }
}

#[test]
fn legal_en_passant_contributes_a_distinct_key_for_each_file_for_both_colors() {
    let mut deltas = [[0; 8]; 2];
    for (color, pawn_rank, target_rank) in [(Color::White, 4, 5), (Color::Black, 3, 2)] {
        for file in 0..8 {
            let mut board = Board::empty();
            board.set_piece(
                Square::new(0).unwrap(),
                Piece::new(Color::White, PieceKind::King),
            );
            board.set_piece(
                Square::new(63).unwrap(),
                Piece::new(Color::Black, PieceKind::King),
            );
            board.set_piece(
                Square::from_coords(file, pawn_rank).unwrap(),
                Piece::new(color.opposite(), PieceKind::Pawn),
            );
            let source_file = if file == 0 { 1 } else { file - 1 };
            board.set_piece(
                Square::from_coords(source_file, pawn_rank).unwrap(),
                Piece::new(color, PieceKind::Pawn),
            );
            let target = Square::from_coords(file, target_rank).unwrap();
            let with =
                Position::new(board, color, CastlingRights::NONE, Some(target), 0, 1).unwrap();
            let without = Position::new(board, color, CastlingRights::NONE, None, 0, 1).unwrap();
            assert_en_passant_identity(&with.to_fen(), 1);
            let delta = with.recompute_zobrist_key() ^ without.recompute_zobrist_key();
            assert_ne!(delta, 0);
            deltas[color.index()][file as usize] = delta;
        }
    }
    assert_eq!(deltas[0], deltas[1]); // The EP feature identifies a file, not a color.
    assert_eq!(deltas[0].into_iter().collect::<HashSet<_>>().len(), 8);
}

fn assert_en_passant_identity(fen: &str, legal_capture_count: usize) {
    let position: Position = fen.parse().unwrap();
    assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
    assert!(position.en_passant_target().is_some());
    let without = Position::new(
        *position.board(),
        position.side_to_move(),
        position.castling_rights(),
        None,
        position.halfmove_clock(),
        position.fullmove_number(),
    )
    .unwrap();
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    assert_eq!(
        moves
            .iter()
            .filter(|mv| mv.kind() == MoveKind::EnPassant)
            .count(),
        legal_capture_count,
        "{fen}",
    );
    assert_eq!(
        position.recompute_zobrist_key() != without.recompute_zobrist_key(),
        legal_capture_count != 0,
        "{fen}",
    );
    assert_ne!(position, without); // Raw target remains part of Position equality.
    assert_eq!(position.to_fen(), fen);
}

#[test]
fn uncapturable_en_passant_targets_are_ignored() {
    for fen in [
        "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3P4/8/8/4K3 b - d3 0 1",
    ] {
        assert_en_passant_identity(fen, 0);
    }
}

#[test]
fn en_passant_must_not_expose_or_leave_the_moving_king_in_check() {
    for fen in [
        // Capturing pawn is pinned along its file, in both colors.
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3Pp3/8/8/K3R3 b - d3 0 1",
        // Both pawns disappear from a rank, exposing a rook attack.
        "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
        "8/8/8/8/R4Ppk/8/8/4K3 b - f3 0 1",
        // Capture would leave an unrelated check unanswered.
        "k3r3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1",
    ] {
        assert_en_passant_identity(fen, 0);
    }
}

#[test]
fn legal_en_passant_can_capture_a_checker_or_block_a_check() {
    for fen in [
        "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1", // Captures the checking pawn.
        "kb6/8/8/2Pp4/8/8/7K/8 w - d6 0 1",  // Blocks b8-h2 bishop check.
    ] {
        let position: Position = fen.parse().unwrap();
        assert!(position.in_check(position.side_to_move()));
        assert_en_passant_identity(fen, 1);
    }
}

#[test]
fn one_legal_en_passant_candidate_is_sufficient_even_if_the_first_is_pinned() {
    assert_en_passant_identity("k1r5/8/8/2PpP3/8/8/8/2K5 w - d6 0 1", 1);
    assert_en_passant_identity("4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1", 2);
}

#[test]
fn en_passant_identity_does_not_advance_the_fullmove_counter() {
    // Applying a Black move here would overflow; recomputation is still defined.
    let maximal: Position = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 4294967295"
        .parse()
        .unwrap();
    let ordinary: Position = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1".parse().unwrap();
    let without: Position = "4k3/8/8/8/3Pp3/8/8/4K3 b - - 0 1".parse().unwrap();
    assert_eq!(maximal.zobrist_key(), maximal.recompute_zobrist_key());
    assert_eq!(
        maximal.recompute_zobrist_key(),
        ordinary.recompute_zobrist_key()
    );
    assert_ne!(
        maximal.recompute_zobrist_key(),
        without.recompute_zobrist_key()
    );
}
