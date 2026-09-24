//! Opt-in comparison of eager and placement-only legality, with optional caching.
//! Board-only legality is the default; eager legality is retained as a reference.

use crate::perft::LeafMode;
use crate::{Move, MoveKind, Position, Square};

mod cache;
pub use cache::{Associativity, Cache};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Legality {
    /// The previous generator: full copy-and-apply, including hashing.
    Eager,
    /// Check candidate placements only; hash accepted traversal children normally.
    PlacementOnly,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DepthStats {
    /// Entered positions, including cache hits. At zero: actually applied leaves.
    pub visited: u64,
    /// Positions for which legal moves were generated (cache misses if enabled).
    pub expanded: u64,
    pub hits: u64,
    /// Accepted child positions actually constructed at this remaining depth.
    pub child_positions: u64,
}

#[derive(Debug)]
pub struct Result {
    /// Logical leaf count, including subtrees skipped by cache hits.
    pub nodes: u64,
    /// Indexed by remaining depth; empty when instrumentation is disabled.
    pub by_depth: Vec<DepthStats>,
}

/// Cache ownership/allocation/clearing is the caller's responsibility. The same
/// cache may be reused across modes: both count the same leaves. Move buffers
/// are allocated here, matching the uncached baseline's lifetime.
///
/// Instrumentation is compiled out of the timed specialization. Cache,
/// legality, and associativity selection dispatch outside recursive traversal.
/// Complete final positions pass through `black_box` in Apply mode.
pub fn run(
    position: &Position,
    depth: u8,
    mode: LeafMode,
    legality: Legality,
    cache: Option<&mut Cache>,
    instrument: bool,
) -> Result {
    match cache
        .as_ref()
        .map(|cache| cache.associativity())
        .unwrap_or(Associativity::One)
    {
        Associativity::One => {
            run_with_ways::<1>(position, depth, mode, legality, cache, instrument)
        }
        Associativity::Two => {
            run_with_ways::<2>(position, depth, mode, legality, cache, instrument)
        }
        Associativity::Four => {
            run_with_ways::<4>(position, depth, mode, legality, cache, instrument)
        }
    }
}

fn run_with_ways<const WAYS: usize>(
    position: &Position,
    depth: u8,
    mode: LeafMode,
    legality: Legality,
    cache: Option<&mut Cache>,
    instrument: bool,
) -> Result {
    match (legality, cache.is_some(), instrument) {
        (Legality::Eager, false, false) => {
            traverse::<false, false, false, WAYS>(position, depth, mode, cache)
        }
        (Legality::Eager, false, true) => {
            traverse::<false, false, true, WAYS>(position, depth, mode, cache)
        }
        (Legality::Eager, true, false) => {
            traverse::<false, true, false, WAYS>(position, depth, mode, cache)
        }
        (Legality::Eager, true, true) => {
            traverse::<false, true, true, WAYS>(position, depth, mode, cache)
        }
        (Legality::PlacementOnly, false, false) => {
            traverse::<true, false, false, WAYS>(position, depth, mode, cache)
        }
        (Legality::PlacementOnly, false, true) => {
            traverse::<true, false, true, WAYS>(position, depth, mode, cache)
        }
        (Legality::PlacementOnly, true, false) => {
            traverse::<true, true, false, WAYS>(position, depth, mode, cache)
        }
        (Legality::PlacementOnly, true, true) => {
            traverse::<true, true, true, WAYS>(position, depth, mode, cache)
        }
    }
}

fn traverse<const PLACEMENT: bool, const CACHED: bool, const TRACK: bool, const WAYS: usize>(
    position: &Position,
    depth: u8,
    mode: LeafMode,
    mut cache: Option<&mut Cache>,
) -> Result {
    let mut buffers: Vec<Vec<Move>> = (0..depth).map(|_| Vec::new()).collect();
    let mut by_depth = if TRACK {
        vec![DepthStats::default(); usize::from(depth) + 1]
    } else {
        Vec::new()
    };
    let nodes = visit::<PLACEMENT, CACHED, TRACK, WAYS>(
        position,
        &mut buffers,
        mode,
        &mut cache,
        &mut by_depth,
    );
    Result { nodes, by_depth }
}

fn visit<const PLACEMENT: bool, const CACHED: bool, const TRACK: bool, const WAYS: usize>(
    position: &Position,
    buffers: &mut [Vec<Move>],
    mode: LeafMode,
    cache: &mut Option<&mut Cache>,
    stats: &mut [DepthStats],
) -> u64 {
    let depth = buffers.len();
    if TRACK {
        stats[depth].visited += 1;
    }
    let Some((moves, remaining)) = buffers.split_first_mut() else {
        return 1;
    };
    if CACHED
        && let Some(count) = cache
            .as_ref()
            .unwrap()
            .probe::<WAYS>(position.zobrist_key(), depth as u8)
    {
        if TRACK {
            stats[depth].hits += 1;
        }
        return count;
    }
    if TRACK {
        stats[depth].expanded += 1;
    }
    moves.clear();
    if PLACEMENT {
        position.generate_legal_moves(moves);
    } else {
        generate_eager_legal(position, moves);
    }
    let mut nodes = 0u64;
    if remaining.is_empty() && mode == LeafMode::Bulk {
        nodes = moves.len() as u64;
    } else {
        if TRACK {
            stats[depth].child_positions += moves.len() as u64;
        }
        for &mv in moves.iter() {
            let child = position.apply(mv);
            let count = if remaining.is_empty() {
                std::hint::black_box(child);
                if TRACK {
                    stats[0].visited += 1;
                }
                1
            } else {
                visit::<PLACEMENT, CACHED, TRACK, WAYS>(&child, remaining, mode, cache, stats)
            };
            nodes = nodes.checked_add(count).expect("perft node count overflow");
        }
    }
    if CACHED {
        cache
            .as_mut()
            .unwrap()
            .store::<WAYS>(position.zobrist_key(), depth as u8, nodes);
    }
    nodes
}

fn generate_eager_legal(position: &Position, moves: &mut Vec<Move>) {
    let start = moves.len();
    position.generate_pseudo_legal_moves(moves);
    let mut write = start;
    for read in start..moves.len() {
        let mv = moves[read];
        if eager_is_legal(position, mv) {
            moves[write] = mv;
            write += 1;
        }
    }
    moves.truncate(write);
}

// Only for candidates produced by our generators, not arbitrary Move values.
fn eager_is_legal(position: &Position, mv: Move) -> bool {
    let mover = position.side_to_move();
    if mv.kind() == MoveKind::Castling {
        if position.in_check(mover) {
            return false;
        }
        let transit = Square::new(((mv.from().index() + mv.to().index()) / 2) as u8).unwrap();
        let step = Move::new(mv.from(), transit, MoveKind::Normal).unwrap();
        // Use the occupancy after the king leaves its source. The rook is
        // still at home during this transit check.
        if position.apply(step).in_check(mover) {
            return false;
        }
    }
    !position.apply(mv).in_check(mover)
}
