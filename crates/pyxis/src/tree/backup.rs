use std::fmt;

use super::{Node, NodeId, Tree};
use crate::Value;

impl Tree {
    /// Backs up one leaf value along a root-to-leaf path of `(parent, edge_index)`.
    ///
    /// Each edge must have a linked child, which must be the next entry's parent
    /// when another entry follows. IDs must belong to this tree. The caller
    /// supplies the reached leaf's value from that final child's side-to-move
    /// perspective; the tree does not contain the game needed to verify it.
    ///
    /// Walks backward, negating the value before recording one sample per edge.
    /// Priors, child links, and stored node values stay unchanged. An empty path
    /// is a no-op, including for a terminal root. No allocation is required.
    ///
    /// # Errors
    ///
    /// Rejects unknown parents, paths that do not follow child links from the
    /// root, terminal parents, invalid edge indices, unlinked edges, or exhausted
    /// visit counters. Validates the entire path before updating any statistics;
    /// errors leave the tree unchanged.
    pub fn backup(
        &mut self,
        path: &[(NodeId, usize)],
        mut value: Value,
    ) -> Result<(), BackupError> {
        let mut expected_parent = self.root();
        for &(parent, edge_index) in path {
            let node = self.node(parent).ok_or(BackupError::UnknownParent)?;
            if parent != expected_parent {
                return Err(BackupError::DisconnectedPath);
            }
            let Node::Expanded(node) = node else {
                return Err(BackupError::TerminalParent);
            };
            let edge = node
                .edges()
                .get(edge_index)
                .ok_or(BackupError::InvalidEdge)?;
            expected_parent = edge.child().ok_or(BackupError::MissingChild)?;
            if edge.stats().visits() == u32::MAX {
                return Err(BackupError::VisitOverflow);
            }
        }

        // A connected path in this tree cannot repeat an edge: add_child only
        // links fresh descendants. Each checked counter will be incremented once.
        for &(parent, edge_index) in path.iter().rev() {
            let Node::Expanded(node) = &mut self.nodes[parent.0] else {
                unreachable!("every path parent was validated before backup")
            };
            value = -value;
            node.edges[edge_index].stats.record(value);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackupError {
    UnknownParent,
    DisconnectedPath,
    TerminalParent,
    InvalidEdge,
    MissingChild,
    VisitOverflow,
}

impl fmt::Display for BackupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownParent => "path parent is not in this tree",
            Self::DisconnectedPath => "path must follow child links from the root",
            Self::TerminalParent => "terminal nodes have no outgoing edges",
            Self::InvalidEdge => "path edge index is outside the parent's move list",
            Self::MissingChild => "path edge has no linked child",
            Self::VisitOverflow => "path edge visit count would overflow",
        })
    }
}

impl std::error::Error for BackupError {}

#[cfg(test)]
mod tests;
