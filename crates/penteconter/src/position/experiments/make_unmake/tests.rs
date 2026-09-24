use super::*;

fn check_tree(position: &mut Position, depth: u8) -> u64 {
    assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
    if depth == 0 {
        return 1;
    }
    let before = *position;
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    let mut count = 0;
    for mv in moves {
        let expected = before.apply(mv);
        let undo = position.make(mv);
        assert_eq!(*position, expected);
        count += check_tree(position, depth - 1);
        position.unmake(mv, undo);
        assert_eq!(*position, before, "parent not restored after {mv}");
        assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
    }
    count
}

#[test]
fn every_parent_and_hash_is_restored_through_standard_perft_trees() {
    for (fen, depth, count) in [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            3,
            8902,
        ),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            3,
            97862,
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            3,
            62379,
        ),
    ] {
        let mut position: Position = fen.parse().unwrap();
        assert_eq!(check_tree(&mut position, depth), count);
        assert_eq!(position.to_fen(), fen);
    }
}

#[test]
fn rejected_candidates_and_raw_ep_metadata_round_trip() {
    for fen in [
        "k3r3/8/8/8/8/8/4R3/4K3 w - - 7 8",
        "k4r2/8/8/8/8/8/8/4K2R w K - 0 1",
        "k7/8/8/K2pP2r/8/8/8/8 w - d6 0 8",
        "k7/8/8/3pP3/4K3/8/8/8 w - d6 0 8",
        "8/8/8/4k3/3Pp3/8/8/K7 b - d3 0 8",
        "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1",
    ] {
        let before: Position = fen.parse().unwrap();
        let mut working = before;
        let mut candidates = Vec::new();
        before.generate_pseudo_legal_moves(&mut candidates);
        for mv in candidates {
            let undo = working.make(mv);
            assert_eq!(working, before.apply(mv));
            assert_eq!(working.zobrist_key(), working.recompute_zobrist_key());
            working.unmake(mv, undo);
            assert_eq!(working, before);
            assert_eq!(working.to_fen(), fen);
            assert_eq!(working.zobrist_key(), working.recompute_zobrist_key());
        }
    }
}

#[test]
fn long_lines_unwind_in_reverse_order_to_each_exact_parent() {
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ] {
        let root: Position = fen.parse().unwrap();
        let mut position = root;
        let mut stack = Vec::new();
        let mut seed = 0x4f44595353455553u64;
        for _ in 0..256 {
            let mut moves = Vec::new();
            position.generate_legal_moves(&mut moves);
            if moves.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mv = moves[(seed >> 32) as usize % moves.len()];
            let parent = position;
            let undo = position.make(mv);
            assert_eq!(position, parent.apply(mv));
            assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
            stack.push((mv, undo, parent));
        }
        assert!(stack.len() > 10);
        while let Some((mv, undo, parent)) = stack.pop() {
            position.unmake(mv, undo);
            assert_eq!(position, parent);
            assert_eq!(position.zobrist_key(), position.recompute_zobrist_key());
        }
        assert_eq!(position, root);
    }
}

#[test]
fn make_counter_overflow_leaves_the_entire_position_unchanged() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    for (fen, from, to) in [
        (format!("4k3/8/8/8/8/8/8/4K3 w - - {} 1", u32::MAX), 4, 3),
        (format!("4k3/8/8/8/8/8/8/4K3 b - - 0 {}", u32::MAX), 60, 59),
    ] {
        let before: Position = fen.parse().unwrap();
        let mut working = before;
        let mv = Move::new(
            Square::new(from).unwrap(),
            Square::new(to).unwrap(),
            MoveKind::Normal,
        )
        .unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _undo = working.make(mv);
        }));
        assert!(result.is_err());
        assert_eq!(working, before);
        assert_eq!(working.zobrist_key(), working.recompute_zobrist_key());
    }
}
