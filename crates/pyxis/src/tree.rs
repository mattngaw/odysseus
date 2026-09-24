use std::fmt;

use crate::{ExpandedNode, Value};

mod backup;
mod pending;
mod report;
mod simulation;

pub use backup::BackupError;
pub use pending::PendingSimulation;
pub use report::{RootMove, SearchReport};
pub use simulation::{BeginSimulationError, SimulationError, SimulationStep};

/// An index into the tree that created it, stable as that tree grows.
///
/// IDs are local to one tree, not position identities. An ID from a different
/// tree may have the same index; callers must keep each ID with its owning tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeId(usize);

impl NodeId {
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A resolved search node. An edge without a child ID has no stored node yet.
#[derive(Debug)]
pub enum Node {
    Expanded(ExpandedNode),
    /// An exact game-outcome value from this node's side-to-move perspective.
    /// The caller establishes the outcome and supplies -1, 0, or +1.
    Terminal(Value),
}

/// An append-only tree whose edges each own a distinct child node.
///
/// Nodes contain search data; the caller maintains the corresponding game and
/// history. No nodes are merged by position identity or shared between edges.
#[derive(Debug)]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    /// Starts with one evaluated or terminal root and no linked children.
    pub fn new(root: Node) -> Self {
        Self { nodes: vec![root] }
    }

    pub const fn root(&self) -> NodeId {
        NodeId(0)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Borrows a node by its tree-local ID; out-of-range IDs return `None`.
    /// Bounds checking does not establish that an ID came from this tree.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0)
    }

    /// Appends a fresh child and links it to one edge of an expanded parent.
    ///
    /// The caller resolves the child for the game reached by that edge's move.
    /// This only changes the tree structure; it records no visits or values.
    ///
    /// # Errors
    ///
    /// Rejects an unknown parent, a terminal parent, an invalid edge index, or
    /// an edge that already has a child. Errors leave the tree unchanged.
    pub fn add_child(
        &mut self,
        parent: NodeId,
        edge_index: usize,
        child: Node,
    ) -> Result<NodeId, AddChildError> {
        let parent_node = self
            .nodes
            .get(parent.0)
            .ok_or(AddChildError::UnknownParent)?;
        let Node::Expanded(parent_node) = parent_node else {
            return Err(AddChildError::TerminalParent);
        };
        let edge = parent_node
            .edges()
            .get(edge_index)
            .ok_or(AddChildError::InvalidEdge)?;
        if edge.child().is_some() {
            return Err(AddChildError::ChildAlreadyExists);
        }

        let child_id = NodeId(self.nodes.len());
        self.nodes.push(child);
        // Appending may relocate the Vec. Reacquire the parent by its stable ID.
        let Node::Expanded(parent_node) = &mut self.nodes[parent.0] else {
            unreachable!("the parent was checked before appending")
        };
        parent_node.edges[edge_index].child = Some(child_id);
        Ok(child_id)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddChildError {
    UnknownParent,
    TerminalParent,
    InvalidEdge,
    ChildAlreadyExists,
}

impl fmt::Display for AddChildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnknownParent => "parent node is not in this tree",
            Self::TerminalParent => "terminal nodes have no outgoing edges",
            Self::InvalidEdge => "edge index is outside the parent's move list",
            Self::ChildAlreadyExists => "the edge already has a child node",
        })
    }
}

impl std::error::Error for AddChildError {}

#[cfg(test)]
mod tests;
