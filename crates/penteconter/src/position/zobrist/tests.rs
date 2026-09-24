use super::*;
use crate::{Move, MoveKind};

const INITIAL: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn play(position: &Position, coordinates: &str) -> Position {
    let square = |name: &str| {
        Square::from_coords(name.as_bytes()[0] - b'a', name.as_bytes()[1] - b'1').unwrap()
    };
    let mv = Move::new(
        square(&coordinates[..2]),
        square(&coordinates[2..]),
        MoveKind::Normal,
    )
    .unwrap();
    let mut legal = Vec::new();
    position.generate_legal_moves(&mut legal);
    assert!(
        legal.contains(&mv),
        "{coordinates} in {}",
        position.to_fen()
    );
    let snapshot = *position;
    let next = position.apply(mv);
    assert_eq!(*position, snapshot);
    assert_eq!(next.zobrist_key(), next.recompute_zobrist_key());
    assert_eq!(next, next.to_fen().parse().unwrap());
    next
}

#[test]
fn incremental_keys_match_recomputation_through_legal_trees() {
    fn visit(position: &Position, depth: u8) -> u64 {
        // An ordinary assertion deliberately also checks optimized test builds.
        assert_eq!(
            position.zobrist_key(),
            position.recompute_zobrist_key(),
            "{}",
            position.to_fen()
        );
        if depth == 0 {
            return 1;
        }
        let snapshot = *position;
        let mut moves = Vec::new();
        position.generate_legal_moves(&mut moves);
        let mut nodes = 0;
        for mv in moves {
            nodes += visit(&position.apply(mv), depth - 1);
        }
        assert_eq!(*position, snapshot);
        nodes
    }

    for (fen, depth, leaves) in [
        (INITIAL, 3, 8_902),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            3,
            97_862,
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43_238),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            3,
            62_379,
        ),
    ] {
        assert_eq!(visit(&fen.parse().unwrap(), depth), leaves, "{fen}");
    }
}

#[test]
fn rejected_candidates_also_have_consistent_keys() {
    for fen in [
        "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1",
        "k4r2/8/8/8/8/8/8/4K2R w K - 0 1",
    ] {
        let position: Position = fen.parse().unwrap();
        let mut candidates = Vec::new();
        let mut legal = Vec::new();
        position.generate_pseudo_legal_moves(&mut candidates);
        position.generate_legal_moves(&mut legal);
        assert!(candidates.len() > legal.len());
        for mv in candidates {
            let child = position.apply(mv);
            assert_eq!(
                child.zobrist_key(),
                child.recompute_zobrist_key(),
                "{fen}: {mv}"
            );
        }
    }
}

#[test]
fn double_pushes_hash_new_en_passant_only_when_legal_for_the_opponent() {
    for (fen, mv, capturable) in [
        // White double-pushes: no candidate, legal candidate, pinned candidate,
        // and two candidates where the first is pinned and the second is legal.
        ("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1", "d2d4", false),
        ("4k3/8/8/8/4p3/8/3P4/4K3 w - - 0 1", "d2d4", true),
        ("4k3/8/8/8/4p3/8/3P4/K3R3 w - - 0 1", "d2d4", false),
        ("2k5/8/8/8/2p1p3/8/3P4/K1R5 w - - 0 1", "d2d4", true),
        // The corresponding Black double-push cases.
        ("4k3/3p4/8/8/8/8/8/4K3 b - - 0 1", "d7d5", false),
        ("4k3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", "d7d5", true),
        ("k3r3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", "d7d5", false),
        ("k1r5/3p4/8/2P1P3/8/8/8/2K5 b - - 0 1", "d7d5", true),
    ] {
        let next = play(&fen.parse().unwrap(), mv);
        assert!(next.en_passant_target().is_some());
        let without = Position::new(
            *next.board(),
            next.side_to_move(),
            next.castling_rights(),
            None,
            next.halfmove_clock(),
            next.fullmove_number(),
        )
        .unwrap();
        assert_eq!(
            next.zobrist_key() != without.zobrist_key(),
            capturable,
            "{fen}"
        );
    }
}

#[test]
fn old_en_passant_expires_using_the_original_placement() {
    for (fen, mv) in [
        ("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1", "e1d1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e1d1"),
        // Moving the king breaks the pin. Testing EP after this placement
        // change would incorrectly remove a feature the old key never had.
        ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e1d1"),
        ("4k3/8/8/8/3P4/8/8/4K3 b - d3 0 1", "e8d8"),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e8d8"),
        ("4k3/8/8/8/3Pp3/8/8/K3R3 b - d3 0 1", "e8d8"),
    ] {
        let next = play(&fen.parse().unwrap(), mv);
        assert!(next.en_passant_target().is_none());
    }
}

#[test]
fn a_reversible_cycle_restores_the_key_but_not_the_move_counters() {
    let original: Position = INITIAL.parse().unwrap();
    let mut position = original;
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        position = play(&position, mv);
    }
    assert_eq!(position.board(), original.board());
    assert_eq!(position.zobrist_key(), original.zobrist_key());
    assert_ne!(position, original);
    assert_eq!(position.halfmove_clock(), 4);
    assert_eq!(position.fullmove_number(), 3);
}

#[test]
fn full_recomputation_ignores_the_cached_key() {
    let mut position: Position = INITIAL.parse().unwrap();
    let expected = position.zobrist_key();
    // Deliberately corrupt the private cache to test oracle independence.
    position.zobrist_key ^= u64::MAX;
    assert_ne!(position.zobrist_key(), expected);
    assert_eq!(position.recompute_zobrist_key(), expected);
}
