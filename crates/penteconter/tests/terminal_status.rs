use penteconter::{Color, Game, Move, MoveKind, PieceKind, Position, Square, TerminalStatus};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn mv(coordinates: &str, kind: MoveKind) -> Move {
    let bytes = coordinates.as_bytes();
    let from = Square::from_coords(bytes[0] - b'a', bytes[1] - b'1').unwrap();
    let to = Square::from_coords(bytes[2] - b'a', bytes[3] - b'1').unwrap();
    Move::new(from, to, kind).unwrap()
}

fn assert_status(position: &Position, expected: Option<TerminalStatus>) {
    let before = *position;
    assert_eq!(position.terminal_status(), expected, "{position}");
    assert_eq!(*position, before);
}

fn assert_only_moves(position: &Position, expected: &[Move]) {
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    assert_eq!(moves.len(), expected.len(), "{position}: {moves:?}");
    for &mv in expected {
        assert!(moves.contains(&mv), "{position}: missing {mv:?}");
        let next = position.play(mv).unwrap();
        assert!(!next.in_check(position.side_to_move()));
    }
}

#[test]
fn checkmate_reports_the_opponents_color_as_winner() {
    for (fen, winner) in [
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", Color::White),
        ("8/8/8/8/8/2k5/1q6/K7 w - - 0 1", Color::Black),
    ] {
        let position: Position = fen.parse().unwrap();
        assert_status(&position, Some(TerminalStatus::Checkmate { winner }));
    }
}

#[test]
fn stalemate_is_distinct_from_checkmate_for_both_colors() {
    for fen in [
        "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/2k5/2q5/K7 w - - 0 1",
    ] {
        let position: Position = fen.parse().unwrap();
        assert_status(&position, Some(TerminalStatus::Stalemate));
    }
}

#[test]
fn a_legal_move_prevents_mate_or_stalemate_including_nonking_evasions() {
    for fen in [
        START,
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", // The king can capture its checker.
        "r7/8/8/8/8/8/2k5/K1B5 w - - 0 1", // Only Ba3 blocks the check.
    ] {
        let position: Position = fen.parse().unwrap();
        assert_status(&position, None);
    }
    let position: Position = "r7/8/8/8/8/8/2k5/K1B5 w - - 0 1".parse().unwrap();
    assert_only_moves(&position, &[mv("c1a3", MoveKind::Normal)]);
}

#[test]
fn en_passant_can_be_the_only_escape_from_checkmate() {
    for (placement, side, target, coordinates, winner) in [
        ("1r6/8/8/Pp6/K1k5/8/8/2b5", "w", "b6", "a5b6", Color::Black),
        ("2B5/8/8/k1K5/pP6/8/8/1R6", "b", "b3", "a4b3", Color::White),
    ] {
        let position: Position = format!("{placement} {side} - {target} 0 1")
            .parse()
            .unwrap();
        assert!(position.in_check(position.side_to_move()));
        assert_only_moves(&position, &[mv(coordinates, MoveKind::EnPassant)]);
        assert_status(&position, None);

        // With the same placement but no EP opportunity, the check is mate.
        let without_ep: Position = format!("{placement} {side} - - 0 1").parse().unwrap();
        assert_status(&without_ep, Some(TerminalStatus::Checkmate { winner }));
    }
}

#[test]
fn en_passant_can_be_the_only_move_preventing_stalemate() {
    let position: Position = "7k/8/4p3/3pP3/8/1q6/8/K7 w - d6 0 1".parse().unwrap();
    assert!(!position.in_check(Color::White));
    assert_only_moves(&position, &[mv("e5d6", MoveKind::EnPassant)]);
    assert_status(&position, None);

    let without_ep: Position = "7k/8/4p3/3pP3/8/1q6/8/K7 w - - 0 1".parse().unwrap();
    assert_status(&without_ep, Some(TerminalStatus::Stalemate));
}

#[test]
fn an_illegal_en_passant_candidate_does_not_prevent_stalemate() {
    // EP would expose the king on a5 to the rook on h5; the pawn's push is blocked.
    let position: Position = "k7/1q6/4p3/K2pP2r/8/8/2b5/8 w - d6 0 1".parse().unwrap();
    assert!(position.play(mv("e5d6", MoveKind::EnPassant)).is_err());
    assert_status(&position, Some(TerminalStatus::Stalemate));
}

#[test]
fn promotions_can_be_the_only_legal_moves_including_capturing_a_checker() {
    for (fen, coordinates, in_check) in [
        ("7k/4P3/8/8/8/1q6/8/K7 w - - 0 1", "e7e8", false),
        ("r7/1P6/8/8/8/8/2k5/K7 w - - 0 1", "b7a8", true),
    ] {
        let position: Position = fen.parse().unwrap();
        let promotions = [
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
        ]
        .map(|kind| mv(coordinates, MoveKind::Promotion(kind)));
        assert_eq!(position.in_check(Color::White), in_check);
        assert_only_moves(&position, &promotions);
        assert_status(&position, None);
    }
}

#[test]
fn counters_and_other_draw_conditions_do_not_affect_the_query() {
    for side in ["w", "b"] {
        for (halfmove, fullmove) in [(0, 1), (150, 100), (u32::MAX, u32::MAX)] {
            // Bare kings have legal moves even though no checkmate is possible.
            let position: Position =
                format!("4k3/8/8/8/8/8/8/4K3 {side} - - {halfmove} {fullmove}")
                    .parse()
                    .unwrap();
            assert_status(&position, None);
        }
    }
    for (placement, expected) in [
        (
            "k7/1Q6/2K5/8/8/8/8/8",
            TerminalStatus::Checkmate {
                winner: Color::White,
            },
        ),
        ("k7/2Q5/2K5/8/8/8/8/8", TerminalStatus::Stalemate),
    ] {
        let position: Position = format!("{placement} b - - {} {}", u32::MAX, u32::MAX)
            .parse()
            .unwrap();
        assert_status(&position, Some(expected));
    }
}

#[test]
fn game_play_and_undo_expose_the_current_positions_status() {
    let mut game = Game::new(START.parse().unwrap());
    for coordinates in ["f2f3", "e7e5", "g2g4"] {
        assert_status(game.position(), None);
        game.play(mv(coordinates, MoveKind::Normal)).unwrap();
    }
    let mating_move = mv("d8h4", MoveKind::Normal);
    assert_status(game.position(), None);
    game.play(mating_move).unwrap();
    assert_status(
        game.position(),
        Some(TerminalStatus::Checkmate {
            winner: Color::Black,
        }),
    );
    assert_eq!(game.undo(), Some(mating_move));
    assert_status(game.position(), None);

    let root: Position = "k7/8/1QK5/8/8/8/8/8 w - - 0 1".parse().unwrap();
    let mut game = Game::new(root);
    let stalemating_move = mv("b6c7", MoveKind::Normal);
    assert_status(game.position(), None);
    game.play(stalemating_move).unwrap();
    assert_status(game.position(), Some(TerminalStatus::Stalemate));
    assert_eq!(game.undo(), Some(stalemating_move));
    assert_eq!(*game.position(), root);
    assert_status(game.position(), None);
}
