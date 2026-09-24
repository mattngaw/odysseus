use penteconter::{Move, MoveKind, Position, Square};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn assert_identity(left: &Position, right: &Position, expected: bool) {
    let snapshots = (*left, *right);
    assert!(left.same_repetition_state(left));
    assert!(right.same_repetition_state(right));
    assert_eq!(left.same_repetition_state(right), expected);
    assert_eq!(right.same_repetition_state(left), expected);
    if expected {
        assert_eq!(left.zobrist_key(), right.zobrist_key());
    }
    assert_eq!((*left, *right), snapshots);
}

#[test]
fn repetition_identity_ignores_counters_without_changing_position_equality() {
    let base: Position = START.parse().unwrap();
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
        assert_identity(&base, &changed, true);
    }
}

#[test]
fn repetition_identity_distinguishes_placement_turn_and_retained_castling_rights() {
    let base: Position = START.parse().unwrap();
    for fen in [
        START.replace(" w ", " b "),
        START.replace("KQkq", "KQk"), // The blocked path does not erase the right.
        START.replace("PPPPPPPP/RNBQKBNR", "PPPPPPPP/RBBQKBNR"),
        START.replace("PPPPPPPP/RNBQKBNR", "PPPPPPPP/RnBQKBNR"),
        START.replace("8/PPPPPPPP", "P7/1PPPPPPP"),
    ] {
        assert_identity(&base, &fen.parse().unwrap(), false);
    }
}

#[test]
fn repetition_identity_normalizes_en_passant_by_legal_capturability() {
    for (fen, capturable) in [
        ("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1", false),
        ("4k3/8/8/8/3P4/8/8/4K3 b - d3 0 1", false),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", true),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", true),
        // File pin, removal of both rank blockers, and an unanswered check.
        ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false),
        ("4k3/8/8/8/3Pp3/8/8/K3R3 b - d3 0 1", false),
        ("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1", false),
        ("k3r3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1", false),
        // A legal second candidate is sufficient even when the first is pinned.
        ("k1r5/8/8/2PpP3/8/8/8/2K5 w - d6 0 1", true),
        ("4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1", true),
    ] {
        let with: Position = fen.parse().unwrap();
        let without = Position::new(
            *with.board(),
            with.side_to_move(),
            with.castling_rights(),
            None,
            with.halfmove_clock(),
            with.fullmove_number(),
        )
        .unwrap();
        assert_ne!(with, without);
        assert_identity(&with, &without, !capturable);
        assert_eq!(with.to_fen(), fen);
    }
}

#[test]
fn different_raw_targets_can_normalize_to_the_same_repetition_state() {
    let d6: Position = "4k3/8/8/3p1p2/8/8/8/4K3 w - d6 0 1".parse().unwrap();
    let f6: Position = "4k3/8/8/3p1p2/8/8/8/4K3 w - f6 0 1".parse().unwrap();
    let none: Position = "4k3/8/8/3p1p2/8/8/8/4K3 w - - 0 1".parse().unwrap();
    assert_identity(&d6, &f6, true);
    assert_identity(&d6, &none, true);
    assert_identity(&f6, &none, true);
}

#[test]
fn repetition_identity_with_legal_ep_does_not_advance_counters() {
    let maximal: Position = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 4294967295"
        .parse()
        .unwrap();
    let ordinary: Position = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1".parse().unwrap();
    assert_identity(&maximal, &ordinary, true);
}

#[test]
fn a_played_cycle_returns_to_the_same_repetition_state() {
    let original: Position = START.parse().unwrap();
    let mut current = original;
    for coordinates in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        let bytes = coordinates.as_bytes();
        let from = Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap();
        let to = Square::from_coords(bytes[2] - b'a', bytes[3] - b'1').unwrap();
        current = current
            .play(Move::new(from, to, MoveKind::Normal).unwrap())
            .unwrap();
    }
    assert_ne!(current, original);
    assert_identity(&current, &original, true);
}
