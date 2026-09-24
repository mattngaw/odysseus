use penteconter::Move;

use super::{Node, Tree};
use crate::{EdgeStats, Value};

/// A snapshot of one legal root move's statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootMove {
    pub mv: Move,
    /// Prior P, visits N, sum W, and derived mean Q in the root player's perspective.
    pub stats: EdgeStats,
    /// Visit fraction N/S, rounded to f32; `None` when the root has no visits.
    /// Once S > 0, unvisited moves have `Some(0.0)`. This is distinct from
    /// the evaluator prior and is not a win probability.
    pub visit_fraction: Option<f32>,
}

/// An owned snapshot, independent of whether the caller will continue searching.
#[derive(Debug, PartialEq)]
pub enum SearchReport {
    /// Exact outcome from the root's side-to-move perspective; no move or visits.
    Terminal(Value),
    Nonterminal {
        /// Most-visited move, or the highest-prior move when S = 0.
        /// Exact ties keep the first move in the root's legal-move order.
        best_move: Move,
        /// Sum of root-edge visits: completed simulations, excluding root setup.
        /// Wider than an individual edge count so summing visits cannot truncate it.
        simulations: u64,
        /// Every legal move, including unvisited moves, in the original order.
        moves: Vec<RootMove>,
    },
}

impl Tree {
    /// Reports the current root without changing the tree or evaluating a position.
    ///
    /// May be called immediately after root resolution, between simulations, or
    /// after a returned simulation error. Before any visits, the fallback uses
    /// the highest prior and no visit distribution exists. Otherwise, choice
    /// uses visits alone, with first-in-order ties; fractions sum to approximately
    /// one. Neither Q nor PUCT breaks a visit-count tie.
    ///
    /// The returned statistics are copies; reporting does not consume the tree.
    /// A caller can check its stop condition between [`Self::simulate`] calls,
    /// report, and later resume with the same root game/history and evaluator.
    /// Root resolution and an in-flight simulation are not interrupted by this API.
    pub fn report(&self) -> SearchReport {
        let root = match &self.nodes[0] {
            Node::Terminal(value) => return SearchReport::Terminal(*value),
            Node::Expanded(root) => root,
        };
        let simulations = root
            .edges()
            .iter()
            .map(|edge| u64::from(edge.stats().visits()))
            .sum::<u64>();
        let mut best = root.edges()[0];
        let mut moves = Vec::with_capacity(root.edges().len());
        for &edge in root.edges() {
            let better = if simulations == 0 {
                edge.stats().prior() > best.stats().prior()
            } else {
                edge.stats().visits() > best.stats().visits()
            };
            if better {
                best = edge;
            }
            moves.push(RootMove {
                mv: edge.mv(),
                stats: edge.stats(),
                visit_fraction: (simulations != 0)
                    .then(|| (f64::from(edge.stats().visits()) / simulations as f64) as f32),
            });
        }
        SearchReport::Nonterminal {
            best_move: best.mv(),
            simulations,
            moves,
        }
    }
}
