use penteconter::{
    Color, DrawReason, Game, GameOutcome, IllegalMove, Move, MoveKind, Position, Square,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MATERIAL_DRAW: Option<GameOutcome> = Some(GameOutcome::Draw {
    reason: DrawReason::InsufficientMaterial,
});
const STALEMATE: Option<GameOutcome> = Some(GameOutcome::Draw {
    reason: DrawReason::Stalemate,
});

fn normal(coordinates: &str) -> Move {
    let bytes = coordinates.as_bytes();
    let from = Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap();
    let to = Square::from_coords(bytes[2] - b'a', bytes[3] - b'1').unwrap();
    Move::new(from, to, MoveKind::Normal).unwrap()
}

fn assert_outcome(game: &Game, expected: Option<GameOutcome>) {
    let before = *game.position();
    let repetitions = game.repetition_count();
    assert_eq!(game.outcome(), expected, "{before}");
    assert_eq!(game.outcome(), expected, "repeating the query: {before}");
    assert_eq!(*game.position(), before);
    assert_eq!(game.repetition_count(), repetitions);
}

fn play(game: &mut Game, mv: Move, checked: bool) {
    if checked {
        game.play(mv).unwrap();
    } else {
        game.play_unchecked(mv);
    }
}

#[test]
fn supplied_positions_report_winner_or_draw_reason_without_recorded_moves() {
    for (fen, expected) in [
        (START, None),
        // In check, but with legal escapes.
        ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", None),
        (
            "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
            Some(GameOutcome::Checkmate {
                winner: Color::White,
            }),
        ),
        (
            "8/8/8/8/8/2k5/1q6/K7 w - - 0 1",
            Some(GameOutcome::Checkmate {
                winner: Color::Black,
            }),
        ),
        ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", STALEMATE),
        ("8/8/8/8/8/2k5/2q5/K7 w - - 0 1", STALEMATE),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", MATERIAL_DRAW),
        ("4k3/8/8/8/8/8/8/2B1K3 b - - 0 1", MATERIAL_DRAW),
        ("4k3/8/8/8/8/8/8/2N1K3 w - - 0 1", MATERIAL_DRAW),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        assert_outcome(&game, expected);
        assert_eq!(game.undo(), None);
        assert_outcome(&game, expected);
    }
}

#[test]
fn stalemate_takes_priority_when_material_is_also_insufficient() {
    // Black has no legal move, despite this also being KB-K.
    let game = Game::new("8/8/8/8/8/1B6/2K5/k7 b - - 0 1".parse().unwrap());
    assert!(game.position().has_insufficient_material());
    assert_outcome(&game, STALEMATE);
}

#[test]
fn play_and_undo_update_all_recognized_outcomes() {
    for (fen, coordinates, expected) in [
        (
            "k7/8/1QK5/8/8/8/8/8 w - - 0 1",
            "b6b7",
            Some(GameOutcome::Checkmate {
                winner: Color::White,
            }),
        ),
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6c7", STALEMATE),
        ("4k3/8/8/8/8/8/3r4/2B1K3 w - - 0 1", "c1d2", MATERIAL_DRAW),
    ] {
        let root: Position = fen.parse().unwrap();
        let mv = normal(coordinates);
        for checked in [true, false] {
            let mut game = Game::new(root);
            assert_outcome(&game, None);
            play(&mut game, mv, checked);
            assert_outcome(&game, expected);
            assert_eq!(game.undo(), Some(mv));
            assert_eq!(*game.position(), root);
            assert_outcome(&game, None);
            // A previous query/result does not remain latched after undo.
            play(&mut game, mv, checked);
            assert_outcome(&game, expected);
        }
    }
}

#[test]
fn both_play_apis_allow_legal_moves_after_a_reported_draw() {
    let root: Position = "4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap();
    let moves = ["e1e2", "e8e7", "e2e1", "e7e8"].map(normal);
    for checked in [true, false] {
        let mut game = Game::new(root);
        assert_outcome(&game, MATERIAL_DRAW);
        // A reported draw does not relax ordinary move legality.
        assert_eq!(game.play(normal("e1e3")), Err(IllegalMove));
        assert_outcome(&game, MATERIAL_DRAW);

        let mut expected = root;
        for mv in moves {
            expected = expected.play(mv).unwrap();
            play(&mut game, mv, checked);
            assert_eq!(*game.position(), expected);
            assert_outcome(&game, MATERIAL_DRAW);
        }
        assert_eq!(game.repetition_count(), 2);
        for mv in moves.into_iter().rev() {
            assert_eq!(game.undo(), Some(mv));
            assert_outcome(&game, MATERIAL_DRAW);
        }
        assert_eq!(*game.position(), root);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn checkmate_and_stalemate_still_have_no_legal_moves_to_play() {
    for (fen, expected) in [
        (
            "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
            Some(GameOutcome::Checkmate {
                winner: Color::White,
            }),
        ),
        ("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1", STALEMATE),
    ] {
        let root: Position = fen.parse().unwrap();
        let mut game = Game::new(root);
        assert_outcome(&game, expected);
        assert_eq!(game.play(normal("a8a7")), Err(IllegalMove));
        assert_eq!(*game.position(), root);
        assert_eq!(game.undo(), None);
        assert_outcome(&game, expected);
    }
}
