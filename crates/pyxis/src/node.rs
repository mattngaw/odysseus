use std::fmt;

use penteconter::Move;

use crate::{EdgeStats, Evaluation, InvalidPolicyWeight, NodeId, Value, normalize_policy};

/// One outgoing move and its statistics, in the parent node player's perspective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    mv: Move,
    pub(crate) stats: EdgeStats,
    pub(crate) child: Option<NodeId>,
}

impl Edge {
    pub const fn mv(&self) -> Move {
        self.mv
    }

    pub const fn stats(&self) -> EdgeStats {
        self.stats
    }

    /// The linked child in this node's owning tree, or `None` if none is stored yet.
    pub const fn child(&self) -> Option<NodeId> {
        self.child
    }
}

/// An evaluated nonterminal node's original value and ordered outgoing edges.
///
/// This contains no game position. The evaluator value uses
/// this node's side-to-move perspective; expansion does not record it as a
/// sample on any outgoing edge.
#[derive(Debug)]
pub struct ExpandedNode {
    value: Value,
    pub(crate) edges: Vec<Edge>,
}

impl ExpandedNode {
    /// Checks and normalizes an evaluation, pairing priors with moves in order.
    ///
    /// The caller supplies a nonempty list of available legal moves from a
    /// nonterminal game, and the evaluation for that game and list. Normally
    /// this is the complete legal list; a root may intentionally restrict it.
    /// Evaluators still receive the complete list; a caller restricting the root
    /// must gather the corresponding weights before construction. This checks
    /// the shape and weights, not chess legality or game outcomes. Moves are
    /// copied into the node; the caller may reuse its move buffer afterward.
    ///
    /// # Errors
    ///
    /// Rejects an empty move list, a different number of weights, or negative
    /// or nonfinite weights. All-zero weights produce uniform priors.
    pub fn new(legal_moves: &[Move], mut evaluation: Evaluation) -> Result<Self, ExpansionError> {
        if legal_moves.is_empty() {
            return Err(ExpansionError::NoLegalMoves);
        }
        if evaluation.policy_weights.len() != legal_moves.len() {
            return Err(ExpansionError::PolicyLengthMismatch {
                expected: legal_moves.len(),
                actual: evaluation.policy_weights.len(),
            });
        }
        normalize_policy(&mut evaluation.policy_weights)
            .map_err(ExpansionError::InvalidPolicyWeight)?;

        let edges = legal_moves
            .iter()
            .copied()
            .zip(evaluation.policy_weights)
            .map(|(mv, prior)| Edge {
                mv,
                stats: EdgeStats::new(prior).expect("normalization produces a valid prior"),
                child: None,
            })
            .collect();
        Ok(Self {
            value: evaluation.value,
            edges,
        })
    }

    /// The original evaluator value, not an average of later search samples.
    pub const fn value(&self) -> Value {
        self.value
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Returns an index into `edges()`, using the rules of [`crate::select_edge`].
    pub fn select_edge(&self, exploration: f32) -> Option<usize> {
        crate::selection::select_edge_by(&self.edges, exploration, Edge::stats)
    }
}

/// Evaluator output could not be turned into a nonterminal node's edges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionError {
    NoLegalMoves,
    PolicyLengthMismatch { expected: usize, actual: usize },
    InvalidPolicyWeight(InvalidPolicyWeight),
}

impl fmt::Display for ExpansionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLegalMoves => f.write_str("expansion requires at least one legal move"),
            Self::PolicyLengthMismatch { expected, actual } => {
                write!(f, "expected {expected} policy weights, received {actual}")
            }
            Self::InvalidPolicyWeight(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ExpansionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPolicyWeight(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
