//! Uncached legal-move traversal for correctness checks and baseline timing.

use crate::{Move, Position};

#[cfg(feature = "perft-experiment")]
pub mod experiments;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeafMode {
    /// Count legal moves at depth one without applying them again.
    Bulk,
    /// Apply every accepted final move and pass its position to `black_box`.
    Apply,
}

#[derive(Debug)]
pub struct PerftResult {
    /// Leaf positions at the requested depth, not all visited positions.
    pub nodes: u64,
    /// Counts per root move when requested. At depth zero this is an empty list,
    /// although `nodes` is one because the root itself is the leaf.
    pub divide: Option<Vec<(Move, u64)>>,
}

/// Counts legal move paths with one reused vector per depth and no hash cache.
/// Depth zero counts one leaf; terminal positions at positive depth count zero.
/// Draw adjudication is not part of perft.
///
/// Legal filtering checks scratch board placements. Recursion constructs full
/// accepted child positions; `Bulk` skips this at the final ply. `Apply`
/// uses a compiler barrier for final positions so those updates remain observable.
/// Buffer setup and optional divide collection are part of this call; parsing,
/// timing, and output are left to the caller.
///
/// Panics on u64 node-count overflow or a position move-counter overflow.
pub fn run(position: &Position, depth: u8, mode: LeafMode, divide: bool) -> PerftResult {
    let mut buffers: Vec<Vec<Move>> = (0..depth).map(|_| Vec::new()).collect();
    let mut rows = divide.then(Vec::new);
    let nodes = visit(position, &mut buffers, mode, rows.as_mut());
    PerftResult {
        nodes,
        divide: rows,
    }
}

fn visit(
    position: &Position,
    buffers: &mut [Vec<Move>],
    mode: LeafMode,
    mut divide: Option<&mut Vec<(Move, u64)>>,
) -> u64 {
    let Some((moves, remaining)) = buffers.split_first_mut() else {
        return 1;
    };
    moves.clear();
    position.generate_legal_moves(moves);
    if remaining.is_empty() && mode == LeafMode::Bulk {
        if let Some(rows) = divide {
            rows.extend(moves.iter().map(|&mv| (mv, 1)));
        }
        return moves.len() as u64;
    }
    let mut nodes = 0u64;
    for &mv in moves.iter() {
        let child = position.apply(mv);
        let count = if remaining.is_empty() {
            // This branch is Apply mode: keep even otherwise-unused leaf
            // placement and metadata updates observable in optimized builds.
            std::hint::black_box(child);
            1
        } else {
            visit(&child, remaining, mode, None)
        };
        nodes = nodes.checked_add(count).expect("perft node count overflow");
        if let Some(rows) = divide.as_mut() {
            rows.push((mv, count));
        }
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_modes_and_divide_agree_on_counts_and_root_moves() {
        for (fen, depth, expected) in [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                3,
                8902,
            ),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                2,
                2039,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812),
            (
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                2,
                1486,
            ),
        ] {
            let position: Position = fen.parse().unwrap();
            let mut root_moves = Vec::new();
            position.generate_legal_moves(&mut root_moves);
            let bulk = run(&position, depth, LeafMode::Bulk, true);
            let apply = run(&position, depth, LeafMode::Apply, true);
            assert_eq!(bulk.nodes, expected);
            assert_eq!(apply.nodes, expected);
            assert_eq!(bulk.divide, apply.divide);
            let rows = bulk.divide.unwrap();
            assert_eq!(
                rows.iter().map(|&(mv, _)| mv).collect::<Vec<_>>(),
                root_moves
            );
            assert_eq!(rows.iter().map(|&(_, count)| count).sum::<u64>(), expected);
            for mode in [LeafMode::Bulk, LeafMode::Apply] {
                let result = run(&position, depth, mode, false);
                assert_eq!(result.nodes, expected);
                assert!(result.divide.is_none());
            }
        }
    }

    #[test]
    fn divide_handles_depth_zero_depth_one_and_terminal_positions() {
        for fen in [
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
            "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
        ] {
            let position: Position = fen.parse().unwrap();
            let mut moves = Vec::new();
            position.generate_legal_moves(&mut moves);
            for mode in [LeafMode::Bulk, LeafMode::Apply] {
                let zero = run(&position, 0, mode, true);
                assert_eq!(zero.nodes, 1);
                assert!(zero.divide.unwrap().is_empty());
                let one = run(&position, 1, mode, true);
                assert_eq!(one.nodes, moves.len() as u64);
                assert_eq!(
                    one.divide.unwrap(),
                    moves.iter().map(|&mv| (mv, 1)).collect::<Vec<_>>()
                );
            }
        }
    }
}
