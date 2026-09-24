use super::*;

const CASES: [(&str, u8, u64); 4] = [
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
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        3,
        62379,
    ),
];

fn compare_candidates(position: &Position, depth: u8) {
    if depth == 0 {
        return;
    }
    let mut candidates = Vec::new();
    position.generate_pseudo_legal_moves(&mut candidates);
    for mv in candidates {
        let child = position.apply(mv);
        assert_eq!(
            position.board().after_move(mv),
            *child.board(),
            "{mv}, {}",
            position.to_fen()
        );
        assert_eq!(child.zobrist_key(), child.recompute_zobrist_key());
    }
    let mut eager = Vec::new();
    let mut placement = Vec::new();
    generate_eager_legal(position, &mut eager);
    position.generate_legal_moves(&mut placement);
    assert_eq!(eager, placement, "{}", position.to_fen());
    for mv in eager {
        compare_candidates(&position.apply(mv), depth - 1);
    }
}

#[test]
fn scratch_placement_and_order_match_full_apply_including_rejected_candidates() {
    for (fen, _, _) in CASES {
        compare_candidates(&fen.parse().unwrap(), 3);
    }
    for fen in [
        "k7/8/8/K2pP2r/8/8/8/8 w - d6 0 8",
        "k7/8/8/3pP3/4K3/8/8/8 w - d6 0 8",
        "8/8/8/4k3/3Pp3/8/8/K7 b - d3 0 8",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/8/8/8/7p/R3K3 b Q - 0 1",
        "k4r2/8/8/8/8/8/8/4K2R w K - 0 1",
    ] {
        compare_candidates(&fen.parse().unwrap(), 2);
    }
}

#[test]
fn cache_requires_full_key_and_exact_depth_and_replaces_conflicts() {
    let mut cache = Cache::new(1);
    assert_eq!(cache.bytes(), 32);
    assert_eq!(cache.probe::<1>(0, 0), None);
    cache.store::<1>(0, 1, 0); // Zero key and terminal zero count are both valid.
    assert_eq!(cache.probe::<1>(0, 1), Some(0));
    assert_eq!(cache.probe::<1>(0, 2), None);
    assert_eq!(cache.probe::<1>(1 << 40, 1), None);
    cache.store::<1>(1 << 40, 1, 42);
    assert_eq!(cache.probe::<1>(0, 1), None);
    assert_eq!(cache.probe::<1>(1 << 40, 1), Some(42));
    cache.clear();
    assert_eq!(cache.probe::<1>(1 << 40, 1), None);
}

#[test]
fn both_legality_and_leaf_modes_match_known_counts_with_eviction() {
    for (fen, depth, expected) in CASES {
        let position: Position = fen.parse().unwrap();
        for legality in [Legality::Eager, Legality::PlacementOnly] {
            for mode in [LeafMode::Bulk, LeafMode::Apply] {
                let plain = run(&position, depth, mode, legality, None, true);
                assert_eq!(plain.nodes, expected);
                check_stats(&plain, depth, mode, false);
                for ways in [Associativity::One, Associativity::Two, Associativity::Four] {
                    for capacity in [ways.ways(), 1024] {
                        let mut cache = Cache::new(capacity);
                        cache.reset(ways);
                        let result = run(&position, depth, mode, legality, Some(&mut cache), true);
                        assert_eq!(result.nodes, expected);
                        check_stats(&result, depth, mode, true);
                        // Repeat without clearing: the stored root must be a hit.
                        let warm = run(&position, depth, mode, legality, Some(&mut cache), true);
                        assert_eq!(warm.nodes, expected);
                        assert_eq!(warm.by_depth[depth as usize].hits, 1);
                        assert_eq!(warm.by_depth.iter().map(|s| s.expanded).sum::<u64>(), 0);
                        cache.clear();
                        let untracked =
                            run(&position, depth, mode, legality, Some(&mut cache), false);
                        assert_eq!(untracked.nodes, expected);
                        assert!(untracked.by_depth.is_empty());
                    }
                }
            }
        }
    }
}

fn check_stats(result: &Result, depth: u8, mode: LeafMode, cached: bool) {
    let stats = &result.by_depth;
    assert_eq!(stats[depth as usize].visited, 1);
    for d in 1..=depth as usize {
        assert_eq!(stats[d].visited, stats[d].expanded + stats[d].hits);
        assert_eq!(stats[d - 1].visited, stats[d].child_positions);
        if !cached {
            assert_eq!(stats[d].hits, 0);
        }
    }
    if mode == LeafMode::Bulk {
        assert_eq!(stats[0].visited, 0);
    } else if !cached {
        assert_eq!(stats[0].visited, result.nodes);
    }
}

#[test]
fn equivalent_counters_and_uncapturable_ep_can_reuse_cache_but_legal_ep_cannot() {
    for (a, b, same) in [
        (
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 18 42",
            true,
        ),
        (
            "4k3/8/8/3p4/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1",
            true,
        ),
        (
            "k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1",
            "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            true,
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            false,
        ),
    ] {
        let a: Position = a.parse().unwrap();
        let b: Position = b.parse().unwrap();
        let mut cache = Cache::new(1024);
        run(
            &a,
            2,
            LeafMode::Bulk,
            Legality::Eager,
            Some(&mut cache),
            false,
        );
        let result = run(
            &b,
            2,
            LeafMode::Apply,
            Legality::PlacementOnly,
            Some(&mut cache),
            true,
        );
        assert_eq!(result.by_depth[2].hits == 1, same);
        assert_eq!(
            result.nodes,
            crate::perft::run(&b, 2, LeafMode::Apply, false).nodes
        );
    }
}

#[test]
fn terminal_positions_and_depth_zero_are_not_confused_with_empty_entries() {
    for fen in [
        "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
        "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
    ] {
        let position = fen.parse().unwrap();
        let mut cache = Cache::new(1);
        for depth in [0, 1, 2] {
            for legality in [Legality::Eager, Legality::PlacementOnly] {
                for mode in [LeafMode::Bulk, LeafMode::Apply] {
                    assert_eq!(
                        run(&position, depth, mode, legality, Some(&mut cache), true).nodes,
                        u64::from(depth == 0)
                    );
                }
            }
        }
    }
}
