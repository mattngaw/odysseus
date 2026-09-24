use std::fmt;

use penteconter::Game;

use crate::{Evaluator, Node, ResolveError, SearchReport, SimulationError, Tree, resolve_node};

/// Runs a fresh sequential search with a fixed evaluator and simulation budget.
///
/// Validates `simulations >= 1` and finite `exploration > 0` before inspecting
/// the root or calling the evaluator. Resolves the root outside the budget;
/// a terminal root returns immediately with no simulations. Otherwise, runs
/// exactly the requested number and reports raw root statistics and N/S.
/// Their visit counts sum to the budget; rounded visit fractions sum to
/// approximately one. Move selection uses raw counts, not Q or a PUCT score.
/// Reporting is shared with [`Tree::report`]. The internal tree is discarded
/// afterward. For a search controlled by external stop conditions, retain a
/// [`Tree`] and call [`Tree::simulate`] and [`Tree::report`] between those checks.
///
/// The supplied game's position and recorded history are restored on `Ok` and
/// `Err`. Evaluator state is not rolled back. As with [`Tree::simulate`], these
/// guarantees cover returned results, not unwinding panics.
///
/// # Errors
///
/// Rejects invalid settings and propagates root-resolution or simulation
/// failures with their original error sources. A failed search returns no
/// partial report, even if earlier simulations completed.
pub fn search<E: Evaluator>(
    game: &mut Game,
    evaluator: &mut E,
    simulations: u32,
    exploration: f32,
) -> Result<SearchReport, SearchError<E::Error>> {
    if simulations == 0 {
        return Err(SearchError::ZeroSimulations);
    }
    if !exploration.is_finite() || exploration <= 0.0 {
        return Err(SearchError::InvalidExploration);
    }
    let root = resolve_node(game, evaluator).map_err(SearchError::RootResolution)?;
    let mut tree = Tree::new(root);
    if matches!(tree.node(tree.root()), Some(Node::Terminal(_))) {
        return Ok(tree.report());
    }
    for _ in 0..simulations {
        tree.simulate(game, evaluator, exploration)
            .map_err(SearchError::Simulation)?;
    }

    Ok(tree.report())
}

#[derive(Debug)]
pub enum SearchError<E> {
    ZeroSimulations,
    InvalidExploration,
    RootResolution(ResolveError<E>),
    Simulation(SimulationError<E>),
}

impl<E: fmt::Display> fmt::Display for SearchError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSimulations => f.write_str("search requires at least one simulation"),
            Self::InvalidExploration => f.write_str("exploration must be finite and positive"),
            Self::RootResolution(error) => write!(f, "root resolution failed: {error}"),
            Self::Simulation(error) => write!(f, "simulation failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SearchError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RootResolution(error) => Some(error),
            Self::Simulation(error) => Some(error),
            _ => None,
        }
    }
}
