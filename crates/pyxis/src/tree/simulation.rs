use std::fmt;

use penteconter::Game;

use super::{Node, PendingSimulation, Tree, pending::Traversal};
use crate::resolution::{PreparedNode, prepare_node};
use crate::{Evaluator, ResolveError};

/// Starting a simulation either completes a terminal path or requests evaluation.
#[derive(Debug)]
#[must_use = "handle completion, or complete/drop the pending evaluation"]
pub enum SimulationStep<'a> {
    /// One simulation was backed up and the root game has been restored.
    Completed,
    /// No simulation has been counted yet; the game remains at the selected leaf.
    NeedsEvaluation(PendingSimulation<'a>),
}

impl Tree {
    /// Descends from an already resolved root, without calling an evaluator.
    ///
    /// A terminal leaf is linked if necessary and its exact value backed up;
    /// returns [`SimulationStep::Completed`] with the root game restored.
    /// Otherwise returns a [`PendingSimulation`] borrowing this tree and game
    /// exclusively, with the game at the leaf. The tree remains unchanged until
    /// completion. Dropping the request restores the game without recording a
    /// visit. Only one simulation can be pending on a tree at a time.
    ///
    /// The caller supplies the root's exact position AND recorded history.
    /// Stored nodes and moves must correspond to their legal game states;
    /// descent plays these moves with [`Game::play_unchecked`]. Use the same
    /// evaluator for root setup and all pending evaluations on this tree.
    ///
    /// # Errors
    ///
    /// Invalid exploration, a terminal root, failed selection, or an exhausted
    /// visit counter leave both tree and root game/history unchanged.
    pub fn begin_simulation<'a>(
        &'a mut self,
        game: &'a mut Game,
        exploration: f32,
    ) -> Result<SimulationStep<'a>, BeginSimulationError> {
        if !exploration.is_finite() || exploration <= 0.0 {
            return Err(BeginSimulationError::InvalidExploration);
        }
        if matches!(self.nodes[0], Node::Terminal(_)) {
            return Err(BeginSimulationError::TerminalRoot);
        }

        let mut current = self.root();
        let mut traversal = Traversal {
            tree: self,
            game,
            path: Vec::new(),
        };
        loop {
            let node = match &traversal.tree.nodes[current.0] {
                Node::Terminal(value) => {
                    traversal.finish(*value, None);
                    return Ok(SimulationStep::Completed);
                }
                Node::Expanded(node) => node,
            };
            let index = node
                .select_edge(exploration)
                .ok_or(BeginSimulationError::SelectionFailed)?;
            let edge = node.edges()[index];
            // Preflight before playing or requesting evaluation. Once the child
            // is linked, backup cannot fail: the connected path has room in N.
            if edge.stats().visits() == u32::MAX {
                return Err(BeginSimulationError::VisitOverflow);
            }
            traversal.path.reserve(1);
            traversal.game.play_unchecked(edge.mv());
            traversal.path.push((current, index));
            if let Some(child) = edge.child() {
                current = child;
                continue;
            }
            match prepare_node(traversal.game) {
                PreparedNode::Terminal(value) => {
                    traversal.finish(value, Some(Node::Terminal(value)));
                    return Ok(SimulationStep::Completed);
                }
                PreparedNode::NeedsEvaluation(moves) => {
                    return Ok(SimulationStep::NeedsEvaluation(PendingSimulation::new(
                        traversal, moves,
                    )));
                }
            }
        }
    }

    /// Completes one sequential simulation from an already resolved root.
    ///
    /// Selects and plays moves until reaching a terminal node or an unlinked
    /// edge. Resolves and links at most one new child, then backs up its value.
    /// An existing terminal child reuses its exact value without evaluation.
    /// Newly expanded outgoing edges remain unvisited in this simulation.
    /// Success records exactly one additional visit among the root's edges.
    ///
    /// The caller must supply the same position AND recorded history that the
    /// root represents, and keep the evaluator fixed across setup and simulation.
    /// Nodes and child links must represent their corresponding legal game
    /// states. The tree does not store enough state to check these contracts;
    /// traversal plays its stored legal moves with `Game::play_unchecked`.
    ///
    /// On either `Ok` or `Err`, restores the supplied game's position and
    /// recorded history. Evaluator state (such as a call counter) is not rolled
    /// back. Root setup, a simulation budget, and final move choice are separate.
    /// This wraps [`Self::begin_simulation`], immediately evaluating any pending leaf.
    ///
    /// # Errors
    ///
    /// Rejects nonfinite/nonpositive exploration, terminal roots, failed edge
    /// selection, exhausted visit counters, or failed node resolution. Returned
    /// errors leave the tree unchanged; no partial simulation is counted.
    ///
    /// # Panics
    ///
    /// An evaluator panic that unwinds drops the pending request and restores
    /// the game. There is no general recovery guarantee for other
    /// panics, such as a violated tree/game contract, or for process aborts.
    pub fn simulate<E: Evaluator>(
        &mut self,
        game: &mut Game,
        evaluator: &mut E,
        exploration: f32,
    ) -> Result<(), SimulationError<E::Error>> {
        match self.begin_simulation(game, exploration)? {
            SimulationStep::Completed => Ok(()),
            SimulationStep::NeedsEvaluation(pending) => {
                let evaluation = evaluator
                    .evaluate(pending.game(), pending.legal_moves())
                    .map_err(|error| SimulationError::Resolution(ResolveError::Evaluator(error)))?;
                pending
                    .complete(evaluation)
                    .map_err(|error| SimulationError::Resolution(ResolveError::Expansion(error)))
            }
        }
    }
}

/// Selection failed before an evaluation was requested or a visit recorded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BeginSimulationError {
    InvalidExploration,
    TerminalRoot,
    SelectionFailed,
    VisitOverflow,
}

impl fmt::Display for BeginSimulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidExploration => "exploration must be finite and positive",
            Self::TerminalRoot => "a terminal root has no simulation to run",
            Self::SelectionFailed => "could not select an outgoing edge",
            Self::VisitOverflow => "selected edge visit count would overflow",
        })
    }
}

impl std::error::Error for BeginSimulationError {}

#[derive(Debug)]
pub enum SimulationError<E> {
    InvalidExploration,
    TerminalRoot,
    SelectionFailed,
    VisitOverflow,
    Resolution(ResolveError<E>),
}

impl<E> From<BeginSimulationError> for SimulationError<E> {
    fn from(error: BeginSimulationError) -> Self {
        match error {
            BeginSimulationError::InvalidExploration => Self::InvalidExploration,
            BeginSimulationError::TerminalRoot => Self::TerminalRoot,
            BeginSimulationError::SelectionFailed => Self::SelectionFailed,
            BeginSimulationError::VisitOverflow => Self::VisitOverflow,
        }
    }
}

impl<E: fmt::Display> fmt::Display for SimulationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExploration => f.write_str("exploration must be finite and positive"),
            Self::TerminalRoot => f.write_str("a terminal root has no simulation to run"),
            Self::SelectionFailed => f.write_str("could not select an outgoing edge"),
            Self::VisitOverflow => f.write_str("selected edge visit count would overflow"),
            Self::Resolution(error) => write!(f, "leaf resolution failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SimulationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolution(error) => Some(error),
            _ => None,
        }
    }
}
