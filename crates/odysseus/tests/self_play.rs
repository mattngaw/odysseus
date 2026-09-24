use odysseus::self_play::{RecordError, Recorder};
use penteconter::{Color, DrawReason, Game, GameOutcome, Move};
use pyxis::{
    Adjudication, SearchReport, Tree, UniformEvaluator, Value, encoding::encode, resolve_node,
    search, vocabulary::index_for_move,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn legal(game: &Game) -> Vec<Move> {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    moves
}

fn play(game: &mut Game, coordinate: &str) {
    let mv = legal(game)
        .into_iter()
        .find(|mv| mv.to_string() == coordinate)
        .unwrap();
    game.play(mv).unwrap();
}

fn record(recorder: &mut Recorder, game: &mut Game) {
    let report = search(game, &mut UniformEvaluator, 8, 1.0).unwrap();
    recorder.record(game, &report).unwrap();
}

#[test]
fn snapshot_retains_all_legal_slots_counts_and_pre_move_input() {
    let mut game = Game::new(START.parse().unwrap());
    let input = encode(&game);
    let mut report = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap()).report();
    let SearchReport::Nonterminal {
        simulations, moves, ..
    } = &mut report
    else {
        unreachable!();
    };
    *simulations = 100;
    for entry in moves {
        let count = match entry.mv.to_string().as_str() {
            "e2e4" => 60,
            "d2d4" => 30,
            "g1f3" => 10,
            _ => 0,
        };
        for _ in 0..count {
            entry.stats.record(Value::new(0.9).unwrap());
        }
        // Fractions deliberately remain None: targets must come from raw counts.
    }
    let mut recorder = Recorder::new();
    recorder.record(&game, &report).unwrap();
    play(&mut game, "e2e4");
    record(&mut recorder, &mut game);
    let white = &recorder.roots()[0];
    assert_eq!(white.input(), &input);
    assert_eq!(white.side_to_move(), Color::White);
    assert_eq!(white.ply(), 0);
    assert_eq!(white.policy().len(), 20);
    assert_eq!(white.total_visits(), 100);
    let mask = white.legal_mask();
    let target = white.policy_target();
    assert_eq!(mask.iter().filter(|&&x| x).count(), 20);
    assert_eq!(
        white
            .policy()
            .iter()
            .filter(|entry| entry.visits == 0)
            .count(),
        17
    );
    assert_eq!(target[322], 0.6); // e2e4
    assert_eq!(target[159], 0.1); // g1f3
    assert!((target.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(target.iter().zip(mask).all(|(&p, legal)| legal || p == 0.0));
    for entry in white.policy() {
        assert!(mask[entry.index.index()]);
        assert_eq!(target[entry.index.index()], entry.visits as f32 / 100.0);
    }

    let black = &recorder.roots()[1];
    assert_eq!(black.input(), &encode(&game));
    assert_eq!(black.side_to_move(), Color::Black);
    assert_eq!(black.ply(), 1);
    // Black's e7 pawn is on relative e2; White's e4 pawn is on relative e5.
    assert_eq!(black.input()[12][0], 1.0);
    assert_eq!(black.input()[36][6], 1.0);
    assert_eq!(black.input()[52][13 + 6], 1.0); // Previous-frame white e2 pawn.
    let e5 = legal(&game)
        .into_iter()
        .find(|mv| mv.to_string() == "e7e5")
        .unwrap();
    assert_eq!(index_for_move(e5, Color::Black).unwrap().index(), 322);
    assert!(black.legal_mask()[322]);
}

#[test]
fn special_moves_keep_distinct_relative_policy_slots() {
    for (fen, expected) in [
        (
            "r3k2r/8/8/8/3Pp3/8/8/R3K2R b Kq d3 0 1",
            vec![("e8c8", "e1a1"), ("e4d3", "e5d6")],
        ),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            vec![
                ("a7a8q", "a7a8q"),
                ("a7a8r", "a7a8r"),
                ("a7a8b", "a7a8b"),
                ("a7a8n", "a7a8"),
            ],
        ),
        (
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
            vec![
                ("a2a1q", "a7a8q"),
                ("a2a1r", "a7a8r"),
                ("a2a1b", "a7a8b"),
                ("a2a1n", "a7a8"),
            ],
        ),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let mut recorder = Recorder::new();
        record(&mut recorder, &mut game);
        let root = &recorder.roots()[0];
        for (absolute, relative) in expected {
            let mv = legal(&game)
                .into_iter()
                .find(|mv| mv.to_string() == absolute)
                .unwrap();
            let index = index_for_move(mv, root.side_to_move()).unwrap();
            assert_eq!(index.entry().to_string(), relative);
            assert_eq!(
                root.policy()
                    .iter()
                    .filter(|entry| entry.index == index)
                    .count(),
                1
            );
            assert!(root.legal_mask()[index.index()]);
        }
    }
}

#[test]
fn both_winners_label_each_roots_perspective_and_finish_drains_records() {
    for (line, winner) in [
        (vec!["f2f3", "e7e5", "g2g4", "d8h4"], Color::Black),
        (vec!["e2e4", "f7f6", "d2d4", "g7g5", "d1h5"], Color::White),
    ] {
        let mut game = Game::new(START.parse().unwrap());
        let mut recorder = Recorder::new();
        for coordinate in &line {
            record(&mut recorder, &mut game);
            let saved = recorder.roots().len();
            assert_eq!(recorder.finish(&game), Err(RecordError::UnfinishedGame));
            assert_eq!(recorder.roots().len(), saved);
            play(&mut game, coordinate);
        }
        let completed = recorder.finish(&game).unwrap();
        assert_eq!(
            completed.adjudication,
            Adjudication::Automatic(GameOutcome::Checkmate { winner })
        );
        assert_eq!(completed.examples.len(), line.len());
        for (ply, example) in completed.examples.iter().enumerate() {
            assert_eq!(example.root.ply(), ply);
            assert_eq!(
                example.value_target,
                if example.root.side_to_move() == winner {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                }
            );
        }
        assert!(recorder.roots().is_empty());
        assert_eq!(recorder.finish(&game), Err(RecordError::EmptyRecording));
        // A successful drain also resets the history guard for the next game.
        record(&mut recorder, &mut Game::new(START.parse().unwrap()));
    }
}

#[test]
fn threefold_labels_draws_without_changing_automatic_outcomes_or_old_features() {
    let mut game = Game::new(START.parse().unwrap());
    let mut recorder = Recorder::new();
    for coordinate in ["g1f3", "g8f6", "f3g1", "f6g8"].into_iter().cycle().take(8) {
        record(&mut recorder, &mut game);
        play(&mut game, coordinate);
    }
    assert_eq!(game.repetition_count(), 3);
    assert_eq!(game.outcome(), None);
    assert!(
        recorder.roots()[0]
            .input()
            .iter()
            .all(|token| token[12] == 0.0)
    );
    assert!(
        recorder.roots()[4]
            .input()
            .iter()
            .all(|token| token[12] == 1.0)
    );
    let terminal = search(&mut game, &mut UniformEvaluator, 8, 1.0).unwrap();
    assert_eq!(
        recorder.record(&game, &terminal),
        Err(RecordError::TerminalRoot)
    );
    let completed = recorder.finish(&game).unwrap();
    assert_eq!(
        completed.adjudication,
        Adjudication::ThreefoldRepetitionDraw
    );
    assert_eq!(completed.examples.len(), 8);
    assert!(
        completed
            .examples
            .iter()
            .all(|example| example.value_target == [0.0, 1.0, 0.0])
    );
}

#[test]
fn clock_draw_still_requires_150_halfmoves_and_mate_takes_precedence() {
    for clock in [99, 149] {
        let mut game = Game::new(
            format!("4k3/8/8/8/8/8/8/R3K3 w - - {clock} 75")
                .parse()
                .unwrap(),
        );
        let mut recorder = Recorder::new();
        record(&mut recorder, &mut game);
        play(&mut game, "e1e2");
        if clock == 99 {
            assert_eq!(recorder.finish(&game), Err(RecordError::UnfinishedGame));
            assert_eq!(recorder.roots().len(), 1);
        } else {
            let completed = recorder.finish(&game).unwrap();
            assert_eq!(
                completed.adjudication,
                Adjudication::Automatic(GameOutcome::Draw {
                    reason: DrawReason::SeventyFiveMoveRule
                })
            );
            assert_eq!(completed.examples[0].value_target, [0.0, 1.0, 0.0]);
        }
    }
    let mut game = Game::new("k7/8/1QK5/8/8/8/8/8 w - - 149 75".parse().unwrap());
    let mut recorder = Recorder::new();
    record(&mut recorder, &mut game);
    play(&mut game, "b6b7");
    assert_eq!(game.position().halfmove_clock(), 150);
    assert_eq!(
        recorder.finish(&game).unwrap().examples[0].value_target,
        [1.0, 0.0, 0.0]
    );
}

#[test]
fn malformed_reports_and_zero_visits_leave_recording_empty() {
    let mut game = Game::new(START.parse().unwrap());
    let mut recorder = Recorder::new();
    let unvisited = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap()).report();
    assert_eq!(
        recorder.record(&game, &unvisited),
        Err(RecordError::NoVisits)
    );
    assert_eq!(
        recorder.record(&game, &SearchReport::Terminal(Value::new(0.0).unwrap())),
        Err(RecordError::InvalidReport)
    );
    for defect in 0..4 {
        let mut report = search(&mut game, &mut UniformEvaluator, 8, 1.0).unwrap();
        let SearchReport::Nonterminal {
            simulations, moves, ..
        } = &mut report
        else {
            unreachable!()
        };
        match defect {
            0 => {
                moves.pop();
            }
            1 => moves[1] = moves[0],
            2 => *simulations += 1,
            _ => {
                let mut other = Game::new(START.parse().unwrap());
                play(&mut other, "e2e4");
                moves[0].mv = legal(&other)[0];
            }
        }
        assert_eq!(
            recorder.record(&game, &report),
            Err(RecordError::InvalidReport)
        );
        assert!(recorder.roots().is_empty());
    }
    record(&mut recorder, &mut game);
}

#[test]
fn history_guard_rejects_duplicate_branch_truncation_and_unrelated_final_game() {
    let mut game = Game::new(START.parse().unwrap());
    let mut recorder = Recorder::new();
    play(&mut game, "e2e4");
    record(&mut recorder, &mut game); // Recording may begin after earlier plies.
    let report = search(&mut game, &mut UniformEvaluator, 8, 1.0).unwrap();
    assert_eq!(
        recorder.record(&game, &report),
        Err(RecordError::DuplicateRoot)
    );
    let fen_only = Game::new(*game.position());
    assert_eq!(
        recorder.record(&fen_only, &report),
        Err(RecordError::HistoryMismatch)
    );
    game.undo().unwrap();
    assert_eq!(recorder.finish(&game), Err(RecordError::HistoryMismatch));
    play(&mut game, "d2d4");
    assert_eq!(
        recorder.record(&game, &report),
        Err(RecordError::HistoryMismatch)
    );
    let unrelated_mate = Game::new("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1".parse().unwrap());
    assert_eq!(
        recorder.finish(&unrelated_mate),
        Err(RecordError::HistoryMismatch)
    );
    assert_eq!(recorder.roots().len(), 1);
    game.undo().unwrap();
    play(&mut game, "e2e4");
    play(&mut game, "e7e5");
    record(&mut recorder, &mut game);
    assert_eq!(recorder.roots()[0].ply(), 1);
    assert_eq!(recorder.roots()[1].ply(), 2);
}
