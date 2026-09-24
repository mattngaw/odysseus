use penteconter::{Game, Move};

use super::{Node, NodeId, Tree};
use crate::{Evaluation, ExpandedNode, ExpansionError, Value};

/// A simulation paused at a nonterminal leaf, awaiting its policy and value.
///
/// Exclusively borrows the tree and game until consumed or dropped. The game
/// stays at the leaf; no child or statistics have been added to the tree yet.
/// Only immutable evaluator inputs are exposed. Complete with an evaluation
/// for this game and ordered move list, using the same evaluator as the tree.
///
/// Dropping an unfinished request cancels it: the original game and history
/// are restored and the tree is unchanged. This also applies when an evaluator
/// returns an error with `?` or unwinds through the request's owning scope.
///
/// A pending request prevents another simulation on the same tree:
/// ```compile_fail,E0499
/// # use penteconter::Game;
/// # use pyxis::Tree;
/// # fn overlapping(tree: &mut Tree, game: &mut Game, other_game: &mut Game) {
/// let pending = tree.begin_simulation(game, 1.0).unwrap();
/// let second = tree.begin_simulation(other_game, 1.0);
/// drop(pending);
/// # }
/// ```
/// The game cannot be changed while its request is alive:
/// ```compile_fail,E0499
/// # use penteconter::Game;
/// # use pyxis::Tree;
/// # fn mutate_game(tree: &mut Tree, game: &mut Game) {
/// let pending = tree.begin_simulation(game, 1.0).unwrap();
/// game.undo();
/// drop(pending);
/// # }
/// ```
#[derive(Debug)]
#[must_use = "complete the pending simulation, or drop it to cancel"]
pub struct PendingSimulation<'a> {
    traversal: Traversal<'a>,
    legal_moves: Vec<Move>,
}

impl<'a> PendingSimulation<'a> {
    pub(super) fn new(traversal: Traversal<'a>, legal_moves: Vec<Move>) -> Self {
        Self {
            traversal,
            legal_moves,
        }
    }

    /// The selected leaf's game, including its recorded history.
    pub fn game(&self) -> &Game {
        self.traversal.game
    }

    /// Complete legal-move list in the order required by the returned weights.
    pub fn legal_moves(&self) -> &[Move] {
        &self.legal_moves
    }

    /// Validates the evaluation, links the leaf, and backs up exactly one sample.
    /// The value must use the leaf's side-to-move perspective.
    ///
    /// Consumes the request and restores the root game and history on both
    /// success and error. Malformed policy output leaves the tree unchanged.
    /// A failed request is canceled; start another simulation to retry.
    ///
    /// # Errors
    ///
    /// Returns [`ExpansionError`] for invalid policy length or weights.
    ///
    /// A request cannot be completed twice:
    /// ```compile_fail,E0382
    /// # use pyxis::{Evaluation, PendingSimulation};
    /// # fn twice(pending: PendingSimulation<'_>, first: Evaluation, second: Evaluation) {
    /// pending.complete(first).unwrap();
    /// pending.complete(second).unwrap();
    /// # }
    /// ```
    pub fn complete(mut self, evaluation: Evaluation) -> Result<(), ExpansionError> {
        let node = ExpandedNode::new(&self.legal_moves, evaluation)?;
        self.traversal
            .finish(node.value(), Some(Node::Expanded(node)));
        Ok(())
    }
}

/// Owns restoration throughout descent, including before a request is returned.
#[derive(Debug)]
pub(super) struct Traversal<'a> {
    pub tree: &'a mut Tree,
    pub game: &'a mut Game,
    pub path: Vec<(NodeId, usize)>,
}

impl Traversal<'_> {
    pub fn finish(&mut self, value: Value, child: Option<Node>) {
        if let Some(child) = child {
            let &(parent, index) = self.path.last().expect("a simulation traverses an edge");
            self.tree
                .add_child(parent, index, child)
                .expect("the selected edge is unlinked and belongs to this tree");
        }
        // Descent preflights every counter. Exclusive borrowing prevents changes
        // to the path, links, or counters while waiting for an evaluation.
        self.tree
            .backup(&self.path, value)
            .expect("the traversed path is linked and every visit counter has room");
    }
}

impl Drop for Traversal<'_> {
    fn drop(&mut self) {
        for _ in self.path.iter().rev() {
            self.game
                .undo()
                .expect("each path entry has one recorded move");
        }
    }
}
