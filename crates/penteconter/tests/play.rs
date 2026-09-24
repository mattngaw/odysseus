use penteconter::{IllegalMove, Move, MoveKind, PieceKind, Position, Square};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn mv(coordinates: &str, kind: MoveKind) -> Move {
    let bytes = coordinates.as_bytes();
    let from = Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap();
    let to = Square::from_coords(bytes[2] - b'a', bytes[3] - b'1').unwrap();
    Move::new(from, to, kind).unwrap()
}

fn expect_child(parent: &Position, mv: Move, fen: &str) -> Position {
    let before = *parent;
    let child = parent.play(mv).unwrap();
    assert_eq!(parent.play_unchecked(mv), child);
    assert_eq!(child, fen.parse::<Position>().unwrap());
    assert_eq!(child.zobrist_key(), child.recompute_zobrist_key());
    assert!(child.board().is_consistent());
    assert_eq!(*parent, before);
    child
}

#[test]
fn unchecked_play_accepts_moves_selected_from_the_current_legal_list() {
    for fen in [
        START,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ] {
        let position: Position = fen.parse().unwrap();
        let before = position;
        let mut moves = Vec::new();
        position.generate_legal_moves(&mut moves);
        for mv in moves {
            let child = position.play_unchecked(mv);
            assert_eq!(Ok(child), position.play(mv));
            assert_eq!(child.zobrist_key(), child.recompute_zobrist_key());
            assert!(child.board().is_consistent());
            assert_eq!(position, before);
        }
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "move must be legal in this position")]
fn unchecked_play_debug_check_rejects_exposing_own_king() {
    // The internal transition permits this pseudo-legal rook move. The public
    // entry point's debug check must reject it before applying the move.
    let position: Position = "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1".parse().unwrap();
    let _ = position.play_unchecked(mv("e2d2", MoveKind::Normal));
}

#[test]
fn ordinary_moves_follow_turns_and_update_placement_counters_and_hash() {
    let mut position: Position = START.parse().unwrap();
    for (coordinates, fen) in [
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
            "g8f6",
            "rnbqkb1r/ppp1pppp/5n2/3P4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
        ),
    ] {
        position = expect_child(&position, mv(coordinates, MoveKind::Normal), fen);
    }
}

#[test]
fn special_moves_require_the_correct_kind_and_apply_all_their_effects() {
    for (fen, coordinates, kind, expected) in [
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 8",
            "e5d6",
            MoveKind::EnPassant,
            "4k3/8/3P4/8/8/8/8/4K3 b - - 0 8",
        ),
        (
            "r3k3/8/8/8/8/8/8/4K3 b q - 7 8",
            "e8c8",
            MoveKind::Castling,
            "2kr4/8/8/8/8/8/8/4K3 w - - 8 9",
        ),
        (
            "1r5k/P7/8/8/8/8/8/4K3 w - - 4 8",
            "a7b8",
            MoveKind::Promotion(PieceKind::Knight),
            "1N5k/8/8/8/8/8/8/4K3 b - - 0 8",
        ),
    ] {
        let position: Position = fen.parse().unwrap();
        let before = position;
        assert_eq!(
            position.play(mv(coordinates, MoveKind::Normal)),
            Err(IllegalMove)
        );
        assert_eq!(position, before);
        expect_child(&position, mv(coordinates, kind), expected);
    }
}

#[test]
fn arbitrary_move_descriptions_are_rejected_without_entering_the_transition() {
    let position: Position = START.parse().unwrap();
    let before = position;
    for (coordinates, kind) in [
        ("e3e4", MoveKind::Normal), // Empty source.
        ("e7e5", MoveKind::Normal), // Opponent's piece.
        ("e2e5", MoveKind::Normal), // Invalid movement geometry.
        ("c1h6", MoveKind::Normal), // Blocked slider.
        ("e1e2", MoveKind::Normal), // Friendly capture.
        ("e2f3", MoveKind::Normal), // Pawn diagonal to an empty square.
        ("g1f3", MoveKind::Castling),
        ("e2e4", MoveKind::EnPassant),
        ("e2e4", MoveKind::Promotion(PieceKind::Queen)),
    ] {
        let mv = mv(coordinates, kind);
        assert_eq!(position.play(mv), Err(IllegalMove), "{mv:?}");
        assert_eq!(position, before);
    }
}

#[test]
fn king_safety_and_terminal_positions_are_checked() {
    for (fen, coordinates, kind) in [
        ("k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1", "e2d2", MoveKind::Normal),
        ("k3r3/8/8/8/8/8/4K3/8 w - - 0 1", "e2e1", MoveKind::Normal),
        (
            "k7/8/8/K2pP2r/8/8/8/8 w - d6 0 8",
            "e5d6",
            MoveKind::EnPassant,
        ),
        // Castling through an attacked transit square, then out of check.
        (
            "k4r2/8/8/8/8/8/8/4K2R w K - 0 1",
            "e1g1",
            MoveKind::Castling,
        ),
        (
            "k3r3/8/8/8/8/8/8/4K2R w K - 0 1",
            "e1g1",
            MoveKind::Castling,
        ),
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", "a8a7", MoveKind::Normal),
        ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", "a8a7", MoveKind::Normal),
        // Structurally accepted input still cannot permit capturing a king.
        ("4k3/4Q3/8/8/8/8/8/4K3 w - - 0 1", "e7e8", MoveKind::Normal),
    ] {
        let position: Position = fen.parse().unwrap();
        let before = position;
        assert_eq!(
            position.play(mv(coordinates, kind)),
            Err(IllegalMove),
            "{fen}"
        );
        assert_eq!(position, before);
    }
}

#[test]
fn counter_limits_preserve_legality_first_and_existing_overflow_behavior() {
    for (fen, legal) in [
        ("4k3/8/8/8/8/8/8/4K3 w - - 4294967295 1", "e1d1"),
        ("4k3/8/8/8/8/8/8/4K3 b - - 0 4294967295", "e8d8"),
    ] {
        let position: Position = fen.parse().unwrap();
        let before = position;
        assert_eq!(
            position.play(mv("a2a3", MoveKind::Normal)),
            Err(IllegalMove)
        );
        let result = std::panic::catch_unwind(|| position.play(mv(legal, MoveKind::Normal)));
        assert!(result.is_err(), "legal move must report counter overflow");
        let result =
            std::panic::catch_unwind(|| position.play_unchecked(mv(legal, MoveKind::Normal)));
        assert!(
            result.is_err(),
            "unchecked play must also report counter overflow"
        );
        assert_eq!(position, before);
    }
    // A pawn move resets the halfmove clock, and White does not advance the
    // fullmove number. Maximal counters alone must not reject a legal move.
    let position: Position = "4k3/8/8/8/8/8/4P3/4K3 w - - 4294967295 4294967295"
        .parse()
        .unwrap();
    expect_child(
        &position,
        mv("e2e4", MoveKind::Normal),
        "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 4294967295",
    );
}
