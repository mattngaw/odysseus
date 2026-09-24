//! Uncached comparison of copy/apply and make/unmake, using default legality.

use crate::perft::LeafMode;
use crate::{Move, Position};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Strategy {
    /// Copy into a local child, then update it without returning another Position.
    CopyApply,
    MakeUnmake,
}

/// Includes alignment/padding, not just the sum of saved field sizes.
pub const UNDO_BYTES: usize = size_of::<crate::position::Undo>();

#[derive(Debug, Default, Eq, PartialEq)]
pub struct Result {
    pub nodes: u64,
    /// Legal-generator calls; zero when instrumentation is disabled.
    pub expanded: u64,
    /// Accepted child positions constructed; zero without instrumentation.
    pub transitions: u64,
}

/// Both strategies leave the root unchanged on successful return. The caller
/// checks it outside timing. There is no unwind guard if a deeper move panics.
///
/// Allocate one move vector per depth; use identical legal generation, move
/// order, hash updates, and leaf observation. Apply mode passes a REFERENCE to
/// each complete leaf through black_box: both placements and keys are accessible,
/// without requiring a by-value Position copy just to observe make/unmake's leaf.
/// Bulk counts final legal moves without performing their full transitions.
/// Copy/apply constructs its child at the call site; the default perft traversal
/// retains the returning `Position::apply` wrapper for comparison.
///
/// Statistics and strategy selection specialize outside the recursive loop.
pub fn run(
    position: &mut Position,
    depth: u8,
    mode: LeafMode,
    strategy: Strategy,
    instrument: bool,
) -> Result {
    match (strategy, instrument) {
        (Strategy::CopyApply, false) => traverse::<false, false>(position, depth, mode),
        (Strategy::CopyApply, true) => traverse::<false, true>(position, depth, mode),
        (Strategy::MakeUnmake, false) => traverse::<true, false>(position, depth, mode),
        (Strategy::MakeUnmake, true) => traverse::<true, true>(position, depth, mode),
    }
}

fn traverse<const UNMAKE: bool, const TRACK: bool>(
    position: &mut Position,
    depth: u8,
    mode: LeafMode,
) -> Result {
    let mut buffers: Vec<Vec<Move>> = (0..depth).map(|_| Vec::new()).collect();
    let mut result = Result::default();
    result.nodes = visit::<UNMAKE, TRACK>(position, &mut buffers, mode, &mut result);
    result
}

fn visit<const UNMAKE: bool, const TRACK: bool>(
    position: &mut Position,
    buffers: &mut [Vec<Move>],
    mode: LeafMode,
    result: &mut Result,
) -> u64 {
    let Some((moves, remaining)) = buffers.split_first_mut() else {
        return 1;
    };
    if TRACK {
        result.expanded += 1;
    }
    moves.clear();
    position.generate_legal_moves(moves);
    if remaining.is_empty() && mode == LeafMode::Bulk {
        return moves.len() as u64;
    }
    if TRACK {
        result.transitions += moves.len() as u64;
    }
    let mut nodes = 0u64;
    for &mv in moves.iter() {
        let count = if UNMAKE {
            let undo = position.make(mv);
            let count = descend::<UNMAKE, TRACK>(position, remaining, mode, result);
            position.unmake(mv, undo);
            count
        } else {
            let mut child = *position;
            child.apply_in_place(mv);
            descend::<UNMAKE, TRACK>(&mut child, remaining, mode, result)
        };
        nodes = nodes.checked_add(count).expect("perft node count overflow");
    }
    nodes
}

#[inline]
fn descend<const UNMAKE: bool, const TRACK: bool>(
    position: &mut Position,
    remaining: &mut [Vec<Move>],
    mode: LeafMode,
    result: &mut Result,
) -> u64 {
    if remaining.is_empty() {
        std::hint::black_box(&*position);
        1
    } else {
        visit::<UNMAKE, TRACK>(position, remaining, mode, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategies_match_counts_work_and_original_positions() {
        for (fen, depth, expected) in [
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
            ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", 2, 0),
            ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", 2, 0),
        ] {
            let root: Position = fen.parse().unwrap();
            for mode in [LeafMode::Bulk, LeafMode::Apply] {
                let mut working = root;
                let copy = run(&mut working, depth, mode, Strategy::CopyApply, true);
                assert_eq!(working, root);
                let unmake = run(&mut working, depth, mode, Strategy::MakeUnmake, true);
                assert_eq!(copy, unmake);
                assert_eq!(unmake.nodes, expected);
                assert_eq!(working, root);
                for strategy in [Strategy::CopyApply, Strategy::MakeUnmake] {
                    let plain = run(&mut working, depth, mode, strategy, false);
                    assert_eq!(plain.nodes, expected);
                    assert_eq!((plain.expanded, plain.transitions), (0, 0));
                    assert_eq!(working, root);
                    assert_eq!(run(&mut working, 0, mode, strategy, true).nodes, 1);
                    assert_eq!(working, root);
                }
            }
        }
    }
}
