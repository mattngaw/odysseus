use std::convert::Infallible;

use penteconter::{Game, Move, PieceKind};

use super::{Evaluation, Evaluator};
use crate::Value;

/// A diagnostic material heuristic with uniform policy weights.
///
/// Piece values in pawn units are P=1, N=B=3, R=5, Q=9; kings are excluded.
/// For side-to-move material advantage `delta`, the value is
/// `delta / (5 + abs(delta))`. Five extra pawns map to +0.5. This is a bounded
/// heuristic, not a calibrated expected outcome. It ignores positional factors.
/// Search resolves terminal outcomes before calling the evaluator.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    type Error = Infallible;

    fn evaluate(&mut self, game: &Game, legal_moves: &[Move]) -> Result<Evaluation, Self::Error> {
        const SCALE: f32 = 5.0;
        const PIECE_VALUES: [(PieceKind, i32); 5] = [
            (PieceKind::Pawn, 1),
            (PieceKind::Knight, 3),
            (PieceKind::Bishop, 3),
            (PieceKind::Rook, 5),
            (PieceKind::Queen, 9),
        ];
        let position = game.position();
        let board = position.board();
        let us = position.side_to_move();
        let mut delta = 0;
        for (kind, worth) in PIECE_VALUES {
            let ours = board.pieces(us, kind).count() as i32;
            let theirs = board.pieces(us.opposite(), kind).count() as i32;
            delta += worth * (ours - theirs);
        }
        let delta = delta as f32;
        Ok(Evaluation {
            value: Value::new(delta / (SCALE + delta.abs()))
                .expect("bounded material and positive scale produce a valid value"),
            policy_weights: vec![1.0; legal_moves.len()],
        })
    }
}
