use penteconter::{Game, IllegalMove, Move, MoveKind, PieceKind, Position, Square};
use std::panic::{AssertUnwindSafe, catch_unwind};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn mv(coordinates: &str, kind: MoveKind) -> Move {
    let bytes = coordinates.as_bytes();
    let from = Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap();
    let to = Square::from_coords(bytes[2] - b'a', bytes[3] - b'1').unwrap();
    Move::new(from, to, kind).unwrap()
}

fn normal(coordinates: &str) -> Move {
    mv(coordinates, MoveKind::Normal)
}

fn play(game: &mut Game, mv: Move, checked: bool) {
    if checked {
        assert_eq!(game.play(mv), Ok(()));
    } else {
        game.play_unchecked(mv);
    }
}

fn assert_position(game: &Game, expected: &Position) {
    assert_eq!(game.position(), expected);
    assert_eq!(
        game.position().zobrist_key(),
        game.position().recompute_zobrist_key()
    );
    assert!(game.position().board().is_consistent());
}

fn assert_repetition_count(game: &Game, recorded: &[Position]) {
    // Test oracle: count snapshots independently of the maintained index.
    let expected = recorded
        .iter()
        .filter(|position| position.same_repetition_state(game.position()))
        .count();
    assert_eq!(game.repetition_count(), expected);
}

fn check_line(root: Position, moves: &[Move]) {
    // Compute expected positions independently of Game's history machinery.
    let mut positions = vec![root];
    for &mv in moves {
        positions.push(positions.last().unwrap().play(mv).unwrap());
    }
    for checked in [true, false] {
        let mut game = Game::new(root);
        assert_repetition_count(&game, &positions[..1]);
        for (index, &mv) in moves.iter().enumerate() {
            play(&mut game, mv, checked);
            assert_position(&game, &positions[index + 1]);
            assert_repetition_count(&game, &positions[..index + 2]);
        }
        for (index, &mv) in moves.iter().enumerate().rev() {
            assert_eq!(game.undo(), Some(mv));
            assert_position(&game, &positions[index]);
            assert_repetition_count(&game, &positions[..index + 1]);
        }
        assert_eq!(game.undo(), None);
        assert_position(&game, &root);
        assert_eq!(game.repetition_count(), 1);
    }
}

#[test]
fn supplied_position_is_the_beginning_of_recorded_history() {
    let root: Position = "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 42".parse().unwrap();
    let mut game = Game::new(root);
    assert_position(&game, &root);
    assert_eq!(game.repetition_count(), 1);
    for _ in 0..2 {
        assert_eq!(game.undo(), None);
        assert_position(&game, &root);
        assert_eq!(game.repetition_count(), 1);
    }
}

#[test]
fn ordinary_play_and_undo_restore_every_position_in_order() {
    let moves = ["e2e4", "d7d5", "e4d5", "g8f6", "g1f3", "f6d5"].map(normal);
    check_line(START.parse().unwrap(), &moves);
}

#[test]
fn special_moves_and_expiring_ep_restore_complete_snapshots() {
    for (fen, coordinates, kind) in [
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 42",
            "e1g1",
            MoveKind::Castling,
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 7 42",
            "e8c8",
            MoveKind::Castling,
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 42",
            "e5d6",
            MoveKind::EnPassant,
        ),
        (
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 42",
            "e4d3",
            MoveKind::EnPassant,
        ),
        // The uncapturable raw target is still part of the snapshot.
        (
            "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 42",
            "e1d1",
            MoveKind::Normal,
        ),
    ] {
        check_line(fen.parse().unwrap(), &[mv(coordinates, kind)]);
    }
    // Restore the original pawn, captured rook, and its retained castling right.
    for kind in [
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Rook,
        PieceKind::Queen,
    ] {
        check_line(
            "r3k3/1P6/8/8/8/8/8/4K3 w q - 4 42".parse().unwrap(),
            &[mv("b7a8", MoveKind::Promotion(kind))],
        );
    }
}

#[test]
fn rejected_moves_preserve_the_current_position_and_existing_history() {
    let root: Position = START.parse().unwrap();
    let mut game = Game::new(root);
    let first = normal("e2e4");
    let second = normal("d7d5");
    game.play(first).unwrap();
    let after_first = *game.position();
    game.play(second).unwrap();
    let before_rejection = *game.position();
    for mv in [
        normal("e2e3"),
        normal("d5d4"),
        mv("e1g1", MoveKind::Castling),
    ] {
        assert_eq!(game.play(mv), Err(IllegalMove));
        assert_position(&game, &before_rejection);
        assert_eq!(game.repetition_count(), 1);
    }
    assert_eq!(game.undo(), Some(second));
    assert_position(&game, &after_first);
    assert_eq!(game.undo(), Some(first));
    assert_position(&game, &root);
    assert_eq!(game.undo(), None);
}

#[test]
fn playing_after_undo_continues_from_the_restored_position() {
    let root: Position = START.parse().unwrap();
    for checked in [true, false] {
        let mut game = Game::new(root);
        play(&mut game, normal("e2e4"), checked);
        let after_first = *game.position();
        play(&mut game, normal("d7d5"), checked);
        assert_eq!(game.undo(), Some(normal("d7d5")));
        play(&mut game, normal("c7c5"), checked);
        assert_position(&game, &after_first.play(normal("c7c5")).unwrap());
        assert_eq!(game.undo(), Some(normal("c7c5")));
        assert_position(&game, &after_first);
        assert_eq!(game.undo(), Some(normal("e2e4")));
        assert_position(&game, &root);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn a_generated_line_survives_history_growth_and_full_unwinding() {
    let root: Position = START.parse().unwrap();
    let mut position = root;
    let mut line = Vec::new();
    let mut seed = 0x4f44_5953_5345_5553u64;
    for _ in 0..128 {
        let mut moves = Vec::new();
        position.generate_legal_moves(&mut moves);
        if moves.is_empty() {
            break;
        }
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mv = moves[(seed >> 32) as usize % moves.len()];
        position = position.play_unchecked(mv);
        line.push(mv);
    }
    assert!(line.len() > 16);
    check_line(root, &line);
}

#[test]
fn counter_overflow_preserves_the_position_and_existing_history() {
    for fen in [
        "4k3/8/8/8/8/8/8/4K3 w - - 4294967294 42",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 4294967295",
    ] {
        let root: Position = fen.parse().unwrap();
        for checked in [true, false] {
            let mut game = Game::new(root);
            play(&mut game, normal("e1d1"), checked);
            let before_overflow = *game.position();
            let mut legal_moves = Vec::new();
            game.position().generate_legal_moves(&mut legal_moves);
            assert!(legal_moves.contains(&normal("e8d8")));
            let result = catch_unwind(AssertUnwindSafe(|| {
                play(&mut game, normal("e8d8"), checked)
            }));
            assert!(result.is_err(), "move counter must overflow");
            assert_position(&game, &before_overflow);
            assert_eq!(game.repetition_count(), 1);
            assert_eq!(game.undo(), Some(normal("e1d1")));
            assert_position(&game, &root);
            assert_eq!(game.undo(), None);
        }
    }
}

#[cfg(debug_assertions)]
#[test]
fn unchecked_debug_rejection_preserves_the_position_and_existing_history() {
    let root: Position = "k3r3/8/8/8/8/8/4R3/4K3 b - - 7 42".parse().unwrap();
    let mut game = Game::new(root);
    let first = normal("a8b8");
    game.play_unchecked(first);
    let before_rejection = *game.position();
    let result = catch_unwind(AssertUnwindSafe(|| game.play_unchecked(normal("e2d2"))));
    assert!(result.is_err(), "the pinned rook cannot expose its king");
    assert_position(&game, &before_rejection);
    assert_eq!(game.repetition_count(), 1);
    assert_eq!(game.undo(), Some(first));
    assert_position(&game, &root);
    assert_eq!(game.undo(), None);
}

#[test]
fn repetitions_accumulate_across_different_cycles_and_follow_the_current_state() {
    let root: Position = START.parse().unwrap();
    let moves = [
        "g1f3", "g8f6", "f3g1", "f6g8", // Initial state appears a second time.
        "b1c3", "b8c6", "c3b1", "c6b8", // A different cycle returns to it again.
        "g1f3", // This state has appeared twice, although the parent appeared three times.
    ]
    .map(normal);
    let counts = [1, 1, 1, 1, 2, 1, 1, 1, 3, 2];
    for checked in [true, false] {
        let mut game = Game::new(root);
        for (i, &mv) in moves.iter().enumerate() {
            play(&mut game, mv, checked);
            assert_eq!(game.repetition_count(), counts[i + 1]);
        }
        for (i, &mv) in moves.iter().enumerate().rev() {
            assert_eq!(game.undo(), Some(mv));
            assert_eq!(game.repetition_count(), counts[i]);
        }
        assert_eq!(game.undo(), None);
        assert_eq!(game.repetition_count(), 1);
    }
    check_line(root, &moves);
}

#[test]
fn abandoning_a_repeated_continuation_removes_only_its_occurrences() {
    let root: Position = START.parse().unwrap();
    let kingside_cycle = ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal);
    let queenside_cycle = ["b1c3", "b8c6", "c3b1", "c6b8"].map(normal);
    for checked in [true, false] {
        let mut game = Game::new(root);
        for count in [2, 3] {
            for &mv in &kingside_cycle {
                play(&mut game, mv, checked);
            }
            assert_eq!(game.repetition_count(), count);
        }
        for &mv in kingside_cycle.iter().rev() {
            assert_eq!(game.undo(), Some(mv));
        }
        assert_eq!(game.repetition_count(), 2);
        for &mv in &queenside_cycle {
            play(&mut game, mv, checked);
        }
        assert_eq!(game.repetition_count(), 3);
        for cycle in [&queenside_cycle, &kingside_cycle] {
            for &mv in cycle.iter().rev() {
                assert_eq!(game.undo(), Some(mv));
            }
        }
        assert_position(&game, &root);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn expiring_ep_merges_only_when_the_original_capture_was_not_legal() {
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal);
    for (fen, first_return_count) in [
        ("4k1n1/8/8/3p4/8/8/8/4K1N1 w - d6 0 42", 2),
        ("k3r1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 42", 2),
        ("4k1n1/8/8/3pP3/8/8/8/4K1N1 w - d6 0 42", 1),
    ] {
        let root: Position = fen.parse().unwrap();
        for checked in [true, false] {
            let mut game = Game::new(root);
            for count in [first_return_count, first_return_count + 1] {
                for &mv in &cycle {
                    play(&mut game, mv, checked);
                }
                assert_eq!(game.position().board(), root.board());
                assert_eq!(game.position().en_passant_target(), None);
                assert_eq!(game.repetition_count(), count, "{fen}");
            }
            for _ in 0..2 {
                for &mv in cycle.iter().rev() {
                    assert_eq!(game.undo(), Some(mv));
                }
            }
            assert_position(&game, &root);
            assert_eq!(game.repetition_count(), 1);
        }
    }
}

#[test]
fn losing_castling_rights_starts_a_distinct_state_and_undo_restores_old_counts() {
    let root: Position = "4k1n1/8/8/8/8/8/8/4K2R w K - 0 1".parse().unwrap();
    let cycle = ["h1h2", "g8f6", "h2h1", "f6g8"].map(normal);
    for checked in [true, false] {
        let mut game = Game::new(root);
        for count in [1, 2] {
            for &mv in &cycle {
                play(&mut game, mv, checked);
            }
            assert_eq!(game.position().board(), root.board());
            assert_ne!(game.position().castling_rights(), root.castling_rights());
            assert_eq!(game.repetition_count(), count);
        }
        for _ in 0..2 {
            for &mv in cycle.iter().rev() {
                assert_eq!(game.undo(), Some(mv));
            }
        }
        assert_position(&game, &root);
        assert_eq!(game.repetition_count(), 1);
    }
    check_line(root, &[cycle, cycle].concat());
}

#[test]
fn rejection_and_overflow_preserve_counts_of_a_repeated_state() {
    let root: Position = START.replace("0 1", "4294967291 42").parse().unwrap();
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal);
    for checked in [true, false] {
        let mut game = Game::new(root);
        for &mv in &cycle {
            play(&mut game, mv, checked);
        }
        assert_eq!(game.position().halfmove_clock(), u32::MAX);
        assert_eq!(game.repetition_count(), 2);
        let snapshot = *game.position();

        assert_eq!(game.play(normal("e2e5")), Err(IllegalMove));
        assert_position(&game, &snapshot);
        assert_eq!(game.repetition_count(), 2);
        let overflow = catch_unwind(AssertUnwindSafe(|| {
            play(&mut game, normal("g1f3"), checked)
        }));
        assert!(overflow.is_err());
        assert_position(&game, &snapshot);
        assert_eq!(game.repetition_count(), 2);

        // A pawn move resets the clock; undo must recover the retained count.
        play(&mut game, normal("e2e4"), checked);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), Some(normal("e2e4")));
        assert_position(&game, &snapshot);
        assert_eq!(game.repetition_count(), 2);
        for &mv in cycle.iter().rev() {
            assert_eq!(game.undo(), Some(mv));
        }
        assert_position(&game, &root);
        assert_eq!(game.repetition_count(), 1);
    }
}
