use std::io::{self, Write};

use odysseus::self_play::{CompletedGame, Recorder, jsonl::write_completed_game};
use penteconter::{Color, DrawReason, Game, GameOutcome};
use pyxis::{Adjudication, UniformEvaluator, search};
use serde_json::Value;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn completed(fen: &str, line: &[&str]) -> CompletedGame {
    let mut game = Game::new(fen.parse().unwrap());
    let mut recorder = Recorder::new();
    for &coordinate in line {
        let report = search(&mut game, &mut UniformEvaluator, 8, 1.0).unwrap();
        recorder.record(&game, &report).unwrap();
        let mut moves = Vec::new();
        game.position().generate_legal_moves(&mut moves);
        game.play(
            moves
                .into_iter()
                .find(|mv| mv.to_string() == coordinate)
                .unwrap(),
        )
        .unwrap();
    }
    recorder.finish(&game).unwrap()
}

#[test]
fn jsonl_preserves_inputs_raw_counts_labels_and_complete_game_boundaries() {
    let games = [
        completed(START, &["f2f3", "e7e5", "g2g4", "d8h4"]),
        completed(START, &["e2e4", "f7f6", "d2d4", "g7g5", "d1h5"]),
        completed(
            START,
            &[
                "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
            ],
        ),
        completed("4k3/8/8/8/8/8/8/R3K3 w - - 149 75", &["e1e2"]),
    ];
    let mut bytes = Vec::new();
    for game in &games {
        write_completed_game(&mut bytes, game).unwrap();
    }
    assert_eq!(bytes.last(), Some(&b'\n'));
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(text.lines().count(), games.len());
    for ((game, line), kind) in games.iter().zip(text.lines()).zip([
        "automatic_checkmate",
        "automatic_checkmate",
        "self_play_threefold_repetition",
        "automatic_seventy_five_move_rule",
    ]) {
        let parsed: Value = serde_json::from_str(line).unwrap();
        assert_eq!(parsed["format"], "odysseus.self_play");
        assert_eq!(parsed["version"], 1);
        assert_eq!(parsed["input_encoding"], "pyxis-110-v1");
        assert_eq!(parsed["policy_vocabulary"], "lc0-1858-v1");
        assert_eq!(parsed["adjudication"]["kind"], kind);
        let entries = parsed["examples"].as_array().unwrap();
        assert_eq!(entries.len(), game.examples.len());
        for (example, raw) in game.examples.iter().zip(entries) {
            let root = &example.root;
            assert_eq!(raw["ply"].as_u64().unwrap(), root.ply() as u64);
            let side = match root.side_to_move() {
                Color::White => "white",
                Color::Black => "black",
            };
            assert_eq!(raw["side_to_move"], side);
            assert_eq!(raw["total_visits"].as_u64().unwrap(), root.total_visits());
            assert_eq!(raw["value_target"], serde_json::json!(example.value_target));
            let input: Vec<Vec<f32>> = serde_json::from_value(raw["input"].clone()).unwrap();
            assert_eq!(input.len(), 64);
            for (original, restored) in root.input().iter().zip(input) {
                assert_eq!(restored.len(), 110);
                for (a, b) in original.iter().zip(restored) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            let policy = raw["policy"].as_array().unwrap();
            assert_eq!(policy.len(), root.policy().len());
            for (original, restored) in root.policy().iter().zip(policy) {
                assert_eq!(
                    restored["index"].as_u64().unwrap(),
                    original.index.index() as u64
                );
                assert_eq!(
                    restored["visits"].as_u64().unwrap(),
                    u64::from(original.visits)
                );
            }
        }
        let mut again = Vec::new();
        write_completed_game(&mut again, game).unwrap();
        assert_eq!(again, format!("{line}\n").as_bytes());
    }
}

#[test]
fn automatic_and_policy_draw_reasons_stay_distinct_even_without_examples() {
    for (adjudication, kind) in [
        (
            Adjudication::ThreefoldRepetitionDraw,
            "self_play_threefold_repetition",
        ),
        (
            Adjudication::Automatic(GameOutcome::Draw {
                reason: DrawReason::Stalemate,
            }),
            "automatic_stalemate",
        ),
        (
            Adjudication::Automatic(GameOutcome::Draw {
                reason: DrawReason::InsufficientMaterial,
            }),
            "automatic_insufficient_material",
        ),
        (
            Adjudication::Automatic(GameOutcome::Draw {
                reason: DrawReason::FivefoldRepetition,
            }),
            "automatic_fivefold_repetition",
        ),
        (
            Adjudication::Automatic(GameOutcome::Draw {
                reason: DrawReason::SeventyFiveMoveRule,
            }),
            "automatic_seventy_five_move_rule",
        ),
    ] {
        let game = CompletedGame {
            adjudication,
            examples: vec![],
        };
        let mut bytes = Vec::new();
        write_completed_game(&mut bytes, &game).unwrap();
        let raw: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(raw["adjudication"], serde_json::json!({"kind": kind}));
        assert_eq!(raw["examples"], serde_json::json!([]));
    }
}

#[test]
fn corrupted_public_labels_or_order_fail_before_writing() {
    let mut game = completed(START, &["f2f3", "e7e5", "g2g4", "d8h4"]);
    let mut bytes = Vec::new();
    for bad in [[1.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [0.5, 0.0, 0.5]] {
        game.examples[0].value_target = bad;
        assert_eq!(
            write_completed_game(&mut bytes, &game).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(bytes.is_empty());
    }
    game.examples[0].value_target = [0.0, 0.0, 1.0];
    game.examples.swap(0, 1);
    assert!(write_completed_game(&mut bytes, &game).is_err());
    assert!(bytes.is_empty());
}

#[test]
fn writer_failures_are_returned() {
    struct Failing;
    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "closed test writer",
            ))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let game = completed(START, &["f2f3", "e7e5", "g2g4", "d8h4"]);
    let error = write_completed_game(Failing, &game).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    assert!(error.to_string().contains("closed test writer"));
}
