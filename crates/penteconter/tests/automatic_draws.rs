use penteconter::{
    Color, DrawReason, Game, GameOutcome, Move, MoveKind, PieceKind, Position, Square,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const FIVEFOLD: Option<GameOutcome> = Some(GameOutcome::Draw {
    reason: DrawReason::FivefoldRepetition,
});
const SEVENTY_FIVE: Option<GameOutcome> = Some(GameOutcome::Draw {
    reason: DrawReason::SeventyFiveMoveRule,
});

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
        game.play(mv).unwrap();
    } else {
        game.play_unchecked(mv);
    }
}

fn assert_outcome(game: &Game, expected: Option<GameOutcome>) {
    let before = *game.position();
    let count = game.repetition_count();
    assert_eq!(game.outcome(), expected, "{before}; occurrences: {count}");
    assert_eq!(*game.position(), before);
    assert_eq!(game.repetition_count(), count);
}

#[test]
fn fivefold_counts_occurrences_across_different_cycles_and_undo() {
    let cycles = [
        ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal),
        ["b1c3", "b8c6", "c3b1", "c6b8"].map(normal),
    ];
    let root: Position = START.parse().unwrap();
    for checked in [true, false] {
        let mut game = Game::new(root);
        assert_eq!(game.repetition_count(), 1);
        assert_outcome(&game, None);
        // Reach occurrences 2 through 6. Three and four are not automatic draws;
        // six must still be recognized, since callers may continue after five.
        for cycle_index in 0..5 {
            for (ply, mv) in cycles[cycle_index % 2].into_iter().enumerate() {
                play(&mut game, mv, checked);
                if ply < 3 {
                    // These intermediate states each occur at most three times,
                    // even when a previous root occurrence already reported a draw.
                    assert!(game.repetition_count() < 5);
                    assert_outcome(&game, None);
                }
            }
            let count = cycle_index + 2;
            assert_eq!(game.repetition_count(), count);
            assert_outcome(&game, if count >= 5 { FIVEFOLD } else { None });
        }

        let before_pawn = *game.position();
        play(&mut game, normal("e2e4"), checked);
        assert_eq!(game.repetition_count(), 1);
        assert_outcome(&game, None);
        assert_eq!(game.undo(), Some(normal("e2e4")));
        assert_eq!(*game.position(), before_pawn);
        assert_outcome(&game, FIVEFOLD);

        for cycle_index in (0..5).rev() {
            for mv in cycles[cycle_index % 2].into_iter().rev() {
                assert_eq!(game.undo(), Some(mv));
            }
            let count = cycle_index + 1;
            assert_eq!(game.repetition_count(), count);
            assert_outcome(&game, if count >= 5 { FIVEFOLD } else { None });
        }
        assert_eq!(*game.position(), root);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn fen_clock_is_used_but_earlier_repetitions_are_not_inferred() {
    for side in ["w", "b"] {
        for halfmove in [0, 99, 100, 149, 150, 151, u32::MAX] {
            let root: Position = format!("4k3/8/8/8/8/8/8/R3K3 {side} - - {halfmove} {}", u32::MAX)
                .parse()
                .unwrap();
            let game = Game::new(root);
            assert_eq!(game.repetition_count(), 1);
            assert_outcome(&game, if halfmove >= 150 { SEVENTY_FIVE } else { None });
        }
    }
}

#[test]
fn quiet_moves_cross_the_clock_threshold_and_undo_restores_it() {
    for (side, coordinates) in [("w", ["e1e2", "e8e7"]), ("b", ["e8e7", "e1e2"])] {
        let root: Position = format!("4k3/8/8/8/8/8/8/R3K3 {side} - - 149 75")
            .parse()
            .unwrap();
        let moves = coordinates.map(normal);
        for checked in [true, false] {
            let mut game = Game::new(root);
            assert_outcome(&game, None);
            for (index, mv) in moves.into_iter().enumerate() {
                play(&mut game, mv, checked);
                assert_eq!(game.position().halfmove_clock(), 150 + index as u32);
                assert_outcome(&game, SEVENTY_FIVE);
            }
            assert_eq!(game.undo(), Some(moves[1]));
            assert_eq!(game.position().halfmove_clock(), 150);
            assert_outcome(&game, SEVENTY_FIVE);
            assert_eq!(game.undo(), Some(moves[0]));
            assert_eq!(*game.position(), root);
            assert_outcome(&game, None);
        }
    }
}

#[test]
fn pawn_moves_captures_and_promotions_reset_the_clock_draw() {
    for (placement, coordinates, kind) in [
        ("4k3/8/8/8/8/8/P7/R3K3", "a2a3", MoveKind::Normal),
        ("4k3/8/8/8/8/8/r7/R3K3", "a1a2", MoveKind::Normal),
        (
            "4k3/P7/8/8/8/8/8/4K3",
            "a7a8",
            MoveKind::Promotion(PieceKind::Queen),
        ),
    ] {
        for halfmove in [149, 150] {
            let root: Position = format!("{placement} w - - {halfmove} 75").parse().unwrap();
            let expected = if halfmove == 150 { SEVENTY_FIVE } else { None };
            let mv = mv(coordinates, kind);
            for checked in [true, false] {
                let mut game = Game::new(root);
                assert_outcome(&game, expected);
                play(&mut game, mv, checked);
                assert_eq!(game.position().halfmove_clock(), 0);
                assert_outcome(&game, None);
                assert_eq!(game.undo(), Some(mv));
                assert_eq!(*game.position(), root);
                assert_outcome(&game, expected);
            }
        }
    }
}

#[test]
fn checkmate_on_the_150th_halfmove_takes_precedence_for_both_colors() {
    for (fen, coordinates, winner) in [
        ("k7/8/1QK5/8/8/8/8/8 w - - 149 75", "b6b7", Color::White),
        ("8/8/8/8/8/1qk5/8/K7 b - - 149 75", "b3b2", Color::Black),
    ] {
        let root: Position = fen.parse().unwrap();
        let mating_move = normal(coordinates);
        for checked in [true, false] {
            let mut game = Game::new(root);
            assert_outcome(&game, None);
            play(&mut game, mating_move, checked);
            assert_eq!(game.position().halfmove_clock(), 150);
            assert_outcome(&game, Some(GameOutcome::Checkmate { winner }));
            assert_eq!(game.undo(), Some(mating_move));
            assert_eq!(*game.position(), root);
            assert_outcome(&game, None);
        }
    }
}

#[test]
fn existing_draw_reasons_keep_priority_but_check_alone_does_not() {
    for (fen, expected) in [
        (
            "k7/2Q5/2K5/8/8/8/8/8 b - - 150 75",
            Some(GameOutcome::Draw {
                reason: DrawReason::Stalemate,
            }),
        ),
        (
            "4k3/8/8/8/8/8/8/4K3 w - - 150 75",
            Some(GameOutcome::Draw {
                reason: DrawReason::InsufficientMaterial,
            }),
        ),
    ] {
        let game = Game::new(fen.parse().unwrap());
        assert_outcome(&game, expected);
    }
    // Being in check does not defer the move-counter draw when an escape exists.
    let game = Game::new("4k3/8/8/8/8/8/4r3/4K3 w - - 150 75".parse().unwrap());
    assert!(game.position().in_check(Color::White));
    assert_eq!(game.position().terminal_status(), None);
    assert_outcome(&game, SEVENTY_FIVE);
}

#[test]
fn fivefold_reason_precedes_the_clock_when_both_thresholds_are_reached() {
    let root: Position = START.replace("0 1", "134 75").parse().unwrap();
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal);
    for checked in [true, false] {
        let mut game = Game::new(root);
        for cycle_index in 0..4 {
            for mv in cycle {
                assert_outcome(&game, None);
                play(&mut game, mv, checked);
            }
            assert_eq!(game.repetition_count(), cycle_index + 2);
        }
        assert_eq!(game.position().halfmove_clock(), 150);
        assert_outcome(&game, FIVEFOLD);

        // A new position loses the repetition reason but retains the clock draw.
        play(&mut game, normal("b1c3"), checked);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.position().halfmove_clock(), 151);
        assert_outcome(&game, SEVENTY_FIVE);
        assert_eq!(game.undo(), Some(normal("b1c3")));
        assert_outcome(&game, FIVEFOLD);
        assert_eq!(game.undo(), Some(cycle[3]));
        assert_eq!(game.position().halfmove_clock(), 149);
        assert_eq!(game.repetition_count(), 4);
        assert_outcome(&game, None);
    }
}
