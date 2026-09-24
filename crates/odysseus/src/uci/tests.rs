use penteconter::{Game, Move, MoveKind, PieceKind, Position, Square};

use super::Session;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const CYCLES: &str = "startpos moves g1f3 g8f6 f3g1 f6g8 b1c3 b8c6 c3b1 c6b8";

fn send(session: &mut Session, input: &str) -> String {
    let mut output = Vec::new();
    session
        .run(std::io::Cursor::new(input.as_bytes().to_vec()), &mut output)
        .unwrap();
    String::from_utf8(output).unwrap()
}

fn load(arguments: &str) -> Game {
    let mut session = Session::default();
    assert_eq!(send(&mut session, &format!("position {arguments}")), "");
    session.game.unwrap()
}

fn history(mut game: Game) -> Vec<(Position, usize, Option<Move>)> {
    let mut states = Vec::new();
    loop {
        let position = *game.position();
        let repetitions = game.repetition_count();
        let mv = game.undo();
        states.push((position, repetitions, mv));
        if mv.is_none() {
            return states;
        }
    }
}

#[test]
fn whitespace_unknown_commands_and_eof_are_handled() {
    let mut session = Session::default();
    let input = "\n\t \r\nunknown command\nucinewgame\nstop\n  isready\t\r\nisready";
    assert_eq!(send(&mut session, input), "readyok\nreadyok\n");
    assert!(session.game.is_none());
}

#[test]
fn checkpoint_options_preserve_path_spelling_and_support_explicit_clearing() {
    let mut session = Session::default();
    assert!(session.search_options.neural_checkpoint.is_none());
    assert_eq!(
        send(
            &mut session,
            "setoption\tname\tNeuralCheckpoint value /tmp/Small  Net.PT\nisready"
        ),
        "readyok\n"
    );
    assert_eq!(
        session.search_options.neural_checkpoint.as_deref(),
        Some("/tmp/Small  Net.PT")
    );
    assert_eq!(
        send(
            &mut session,
            "setoption name NeuralCheckpointExtra value wrong\nsetoption name NeuralCheckpoint"
        ),
        ""
    );
    assert_eq!(
        session.search_options.neural_checkpoint.as_deref(),
        Some("/tmp/Small  Net.PT")
    );
    for value in ["<empty>", ""] {
        send(
            &mut session,
            "setoption name NeuralCheckpoint value Previous.pt",
        );
        send(
            &mut session,
            &format!("setoption name neuralcheckpoint value {value}"),
        );
        assert!(session.search_options.neural_checkpoint.is_none());
    }
    send(
        &mut session,
        "setoption name NeuralPython value /tmp/My  Python",
    );
    assert_eq!(session.search_options.neural_python, "/tmp/My  Python");
    assert!(
        send(&mut session, "setoption name NeuralPython value")
            .contains("must name a Python interpreter")
    );
    assert_eq!(session.search_options.neural_python, "/tmp/My  Python");
}

#[test]
fn quit_stops_before_subsequent_commands() {
    let mut session = Session::default();
    assert_eq!(
        send(
            &mut session,
            "isready\nquit\nposition startpos\nuci\nisready\n"
        ),
        "readyok\n"
    );
    assert!(session.game.is_none());
}

#[test]
fn startpos_replays_moves_and_replacement_starts_fresh() {
    let mut session = Session::default();
    assert_eq!(
        send(
            &mut session,
            " \tposition\tstartpos moves e2e4\te7e5 g1f3\r\nisready\n"
        ),
        "readyok\n"
    );
    let game = session.game.as_mut().unwrap();
    assert_eq!(
        game.position().to_fen(),
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    );
    for expected in ["g1f3", "e7e5", "e2e4"] {
        assert_eq!(game.undo().unwrap().to_string(), expected);
    }
    assert_eq!(game.position().to_fen(), START);
    assert_eq!(game.undo(), None);

    // A repeated position command replaces the line rather than appending it.
    for _ in 0..2 {
        assert_eq!(send(&mut session, &format!("position {CYCLES}")), "");
        assert_eq!(session.game.as_ref().unwrap().repetition_count(), 3);
    }
    for arguments in ["startpos", "startpos moves"] {
        assert_eq!(send(&mut session, &format!("position {arguments}")), "");
        let game = session.game.as_mut().unwrap();
        assert_eq!(game.position().to_fen(), START);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn fen_preserves_all_fields_but_cannot_supply_prior_history() {
    let fen = "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 42";
    for suffix in ["", " moves"] {
        let mut game = load(&format!("fen {fen}{suffix}"));
        assert_eq!(game.position().to_fen(), fen);
        assert_eq!(game.undo(), None);
    }
    let mut game = load(CYCLES);
    assert_eq!(game.repetition_count(), 3);
    assert_eq!(game.position().halfmove_clock(), 8);
    assert_eq!(game.position().fullmove_number(), 5);
    let mut from_fen = load(&format!("fen {}", game.position().to_fen()));
    assert_eq!(from_fen.position(), game.position());
    assert_eq!(from_fen.repetition_count(), 1);
    assert_eq!(from_fen.undo(), None);
    for expected in ["c6b8", "c3b1", "b8c6", "b1c3"] {
        assert_eq!(game.undo().unwrap().to_string(), expected);
    }
    assert_eq!(game.repetition_count(), 2);
}

#[test]
fn castling_and_en_passant_recover_their_move_kinds() {
    for (fen, coordinate, kind, expected) in [
        (
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 42",
            "e1g1",
            MoveKind::Castling,
            "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 8 42",
        ),
        (
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 7 42",
            "e8c8",
            MoveKind::Castling,
            "2kr3r/8/8/8/8/8/8/R3K2R w KQ - 8 43",
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 42",
            "e5d6",
            MoveKind::EnPassant,
            "4k3/8/3P4/8/8/8/8/4K3 b - - 0 42",
        ),
        (
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 42",
            "e4d3",
            MoveKind::EnPassant,
            "4k3/8/8/8/8/3p4/8/4K3 w - - 0 43",
        ),
    ] {
        let mut game = load(&format!("fen {fen} moves {coordinate}"));
        assert_eq!(game.position().to_fen(), expected);
        let mv = game.undo().unwrap();
        assert_eq!(mv.to_string(), coordinate);
        assert_eq!(mv.kind(), kind);
        assert_eq!(game.position().to_fen(), fen);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn promotion_suffix_selects_the_requested_piece_for_both_colors() {
    for (fen, coordinates, square) in [
        ("r3k3/1P6/8/8/8/8/8/4K3 w q - 4 42", "b7a8", 56),
        ("4k3/8/8/8/8/8/1p6/R3K3 b Q - 4 42", "b2a1", 0),
    ] {
        for (suffix, kind) in [
            ('n', PieceKind::Knight),
            ('b', PieceKind::Bishop),
            ('r', PieceKind::Rook),
            ('q', PieceKind::Queen),
        ] {
            let mut game = load(&format!("fen {fen} moves {coordinates}{suffix}"));
            assert_eq!(
                game.position()
                    .board()
                    .piece_at(Square::new(square).unwrap())
                    .unwrap()
                    .kind(),
                kind
            );
            assert_eq!(game.position().halfmove_clock(), 0);
            assert_eq!(game.undo().unwrap().kind(), MoveKind::Promotion(kind));
            assert_eq!(game.position().to_fen(), fen);
            assert_eq!(game.undo(), None);
        }
    }
}

#[test]
fn failed_commands_preserve_the_entire_previous_game_and_allow_more_commands() {
    for arguments in [
        "",
        "unknown",
        "fen",
        "fen 4k3/8/8/8/8/8/8/4K3 w - - 0",
        "fen 8/8/8/8/8/8/8/8 w - - 0 1",
        "fen 4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "fen 4k3/8/8/8/8/8/8/4K3 w - d6 0 1",
        "startpos e2e4",
        "fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
        "startpos moves e2e5",
        "startpos moves e2e4 e2e4",
        "startpos moves e2e4 e7e5 Nf3",
        "startpos moves E2E4",
        "startpos moves e2e4q",
        "startpos moves 0000",
        "startpos moves ♞",
        "fen k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1 moves e2d2",
        "fen k4r2/8/8/8/8/8/8/4K2R w K - 0 1 moves e1g1",
        "fen k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1 moves e5d6",
        "fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1 moves a7a8",
        "fen 4k3/P7/8/8/8/8/8/4K3 w - - 0 1 moves a7a8k",
        "fen 4k3/8/8/8/8/8/8/4K3 w - - 4294967294 42 moves e1d1 e8d8",
        "fen 4k3/8/8/8/8/8/8/4K3 w - - 0 4294967295 moves e1d1 e8d8",
    ] {
        let mut session = Session {
            game: Some(load(CYCLES)),
            ..Session::default()
        };
        let output = send(&mut session, &format!("position {arguments}\nisready\n"));
        assert!(
            output.starts_with("info string position rejected: "),
            "{arguments}: {output}"
        );
        assert!(output.ends_with("\nreadyok\n"), "{arguments}: {output}");
        assert_eq!(
            history(session.game.take().unwrap()),
            history(load(CYCLES)),
            "{arguments}"
        );
        // The session can recover with a later valid command.
        assert_eq!(send(&mut session, "position startpos"), "");
        assert_eq!(session.game.unwrap().position().to_fen(), START);
    }
}

#[test]
fn initial_failure_reports_the_offending_ply_without_installing_a_game() {
    let mut session = Session::default();
    assert_eq!(
        send(
            &mut session,
            "position startpos moves e2e4 e7e5 e2e3\nisready"
        ),
        "info string position rejected: illegal move 3: e2e3\nreadyok\n"
    );
    assert!(session.game.is_none());
}

#[test]
fn maximal_counters_are_allowed_when_the_move_does_not_overflow() {
    for (fen, coordinates) in [
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 4294967295 42", "e2e4"),
        ("4k3/8/8/8/8/8/4r3/4K3 w - - 4294967295 42", "e1e2"),
    ] {
        let game = load(&format!("fen {fen} moves {coordinates}"));
        assert_eq!(game.position().halfmove_clock(), 0);
    }
    let game = load("fen 4k3/8/8/8/8/8/8/4K3 w - - 0 4294967295 moves e1d1");
    assert_eq!(game.position().fullmove_number(), u32::MAX);
}

#[test]
fn replay_is_not_stopped_by_a_recognized_draw() {
    let mut game = load("fen 4k3/8/8/8/8/8/8/4K3 w - - 150 42 moves e1e2 e8e7");
    assert!(game.outcome().is_some());
    assert_eq!(game.position().to_fen(), "8/4k3/8/8/8/8/4K3/8 w - - 152 43");
    assert_eq!(game.undo().unwrap().to_string(), "e8e7");
    assert_eq!(game.undo().unwrap().to_string(), "e1e2");
    assert_eq!(game.undo(), None);
}
