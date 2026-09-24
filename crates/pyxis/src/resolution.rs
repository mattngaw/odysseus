use std::fmt;

use penteconter::{Game, GameOutcome, Move};

use crate::{Adjudication, Evaluator, ExpandedNode, ExpansionError, Node, Value, adjudicate};

/// Resolves the current game into a terminal or evaluated search node.
///
/// Outcomes under [`adjudicate`] bypass the evaluator, including our threefold
/// repetition convention. Otherwise, the evaluator receives
/// the complete ordered legal-move list, and expansion validates and normalizes
/// its weights. Both terminal and evaluated values use the side-to-move's
/// perspective. Resolving a node neither links it into a tree nor records visits.
/// The game and its history are only borrowed, including on errors.
///
/// This baseline generates moves again after `Game::outcome()` has internally
/// generated them. Sharing that work is a separate optimization.
///
/// # Errors
///
/// Preserves evaluator failures and rejects malformed policy output through
/// [`ExpandedNode::new`]. No node is returned on failure.
pub fn resolve_node<E: Evaluator>(
    game: &Game,
    evaluator: &mut E,
) -> Result<Node, ResolveError<E::Error>> {
    let moves = match prepare_node(game) {
        PreparedNode::Terminal(value) => return Ok(Node::Terminal(value)),
        PreparedNode::NeedsEvaluation(moves) => moves,
    };
    let evaluation = evaluator
        .evaluate(game, &moves)
        .map_err(ResolveError::Evaluator)?;
    let node = ExpandedNode::new(&moves, evaluation).map_err(ResolveError::Expansion)?;
    Ok(Node::Expanded(node))
}

/// Shared preparation for immediate root resolution and paused leaf evaluation.
pub(crate) enum PreparedNode {
    Terminal(Value),
    NeedsEvaluation(Vec<Move>),
}

pub(crate) fn prepare_node(game: &Game) -> PreparedNode {
    if let Some(outcome) = adjudicate(game) {
        let value = match outcome {
            // The checkmated player is the player to move.
            Adjudication::Automatic(GameOutcome::Checkmate { .. }) => -1.0,
            Adjudication::Automatic(GameOutcome::Draw { .. })
            | Adjudication::ThreefoldRepetitionDraw => 0.0,
        };
        return PreparedNode::Terminal(Value::new(value).expect("exact outcomes are valid values"));
    }

    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    PreparedNode::NeedsEvaluation(moves)
}

#[derive(Debug)]
pub enum ResolveError<E> {
    Evaluator(E),
    Expansion(ExpansionError),
}

impl<E: fmt::Display> fmt::Display for ResolveError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluator(error) => write!(f, "evaluation failed: {error}"),
            Self::Expansion(error) => write!(f, "expansion failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ResolveError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluator(error) => Some(error),
            Self::Expansion(error) => Some(error),
        }
    }
}
