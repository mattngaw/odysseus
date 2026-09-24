use super::*;
use crate::{MoveKind, Square};

#[test]
fn index_retains_earlier_states_and_removes_abandoned_states_at_zero() {
    let root: Position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 42"
        .parse()
        .unwrap();
    let mut game = Game::new(root);
    let double_push = Move::new(
        Square::from_coords(4, 1).unwrap(),
        Square::from_coords(4, 3).unwrap(),
        MoveKind::Normal,
    )
    .unwrap();

    assert_eq!(game.repetitions.len(), 1);
    game.play(double_push).unwrap();
    assert_eq!(game.repetitions.len(), 2);
    assert_eq!(game.repetitions.get(&RepetitionState(root)), Some(&1));
    let abandoned = RepetitionState(*game.position());
    assert_eq!(game.repetitions.get(&abandoned), Some(&1));

    assert_eq!(game.undo(), Some(double_push));
    assert_eq!(game.repetitions.len(), 1);
    assert!(!game.repetitions.contains_key(&abandoned));
    assert_eq!(game.repetition_count(), 1);
    assert_eq!(game.undo(), None);
    assert_eq!(game.repetitions.len(), 1);

    game.play(double_push).unwrap();
    assert_eq!(game.repetitions.len(), 2);
    assert_eq!(game.repetition_count(), 1);
}
