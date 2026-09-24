use penteconter::{Game, Move, Position};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn play(game: &mut Game, coordinate: &str, checked: bool) -> Move {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let mv = moves
        .into_iter()
        .find(|mv| mv.to_string() == coordinate)
        .unwrap();
    if checked {
        game.play(mv).unwrap();
    } else {
        game.play_unchecked(mv);
    }
    mv
}

fn verify(game: &Game, expected: &[Position]) {
    let frames: Vec<_> = game.positions().collect();
    assert_eq!(frames.len(), expected.len());
    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(*frame.position, expected[i]);
        let count = expected[..=i]
            .iter()
            .filter(|p| p.same_repetition_state(&expected[i]))
            .count();
        assert_eq!(frame.repetition_count, count, "frame {i}");
    }
    assert!(std::ptr::eq(
        frames.last().unwrap().position,
        game.position()
    ));
    let recent: Vec<_> = game
        .positions()
        .rev()
        .take(8)
        .map(|p| *p.position)
        .collect();
    assert_eq!(
        recent,
        expected.iter().rev().take(8).copied().collect::<Vec<_>>()
    );
}

#[test]
fn history_preserves_prefix_counts_through_play_undo_and_branching() {
    for checked in [false, true] {
        let mut game = Game::new(START.parse().unwrap());
        let mut expected = vec![*game.position()];
        let mut moves = Vec::new();
        verify(&game, &expected);
        for coordinate in [
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "e7e5", "g1f3",
            "b8c6", "f1c4", "g8f6", "e1g1", "f8c5", "c4f7",
        ] {
            moves.push(play(&mut game, coordinate, checked));
            expected.push(*game.position());
            verify(&game, &expected);
        }
        for _ in 0..13 {
            assert_eq!(game.undo(), moves.pop());
            expected.pop();
            verify(&game, &expected);
        }
        assert_eq!(game.repetition_count(), 2);
        assert_eq!(game.positions().next().unwrap().repetition_count, 1);
        // Replace the abandoned branch; its positions and counts must vanish.
        moves.push(play(&mut game, "d2d4", checked));
        expected.push(*game.position());
        verify(&game, &expected);
        while let Some(mv) = moves.pop() {
            assert_eq!(game.undo(), Some(mv));
            expected.pop();
            verify(&game, &expected);
        }
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn fen_only_history_contains_exactly_one_position_even_with_large_counters() {
    let position = "4k3/8/8/8/8/8/8/R3K3 b - - 149 1000".parse().unwrap();
    let game = Game::new(position);
    verify(&game, &[position]);
    assert_eq!(game.positions().next_back().unwrap().repetition_count, 1);
}
