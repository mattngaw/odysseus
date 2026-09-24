use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};

use penteconter::{Game, Position};

const TIMEOUT: Duration = Duration::from_secs(10);
const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MATE_IN_ONE: &str = "k7/8/1QK5/8/8/8/8/8 w - - 0 1";

struct Engine {
    child: Child,
    lines: Receiver<String>,
}

impl Engine {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_odysseus"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if sender.send(line.unwrap()).is_err() {
                    break;
                }
            }
        });
        Self { child, lines }
    }

    fn send(&mut self, command: &str) {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{command}").unwrap();
        stdin.flush().unwrap();
    }

    fn until(&self, prefix: &str) -> Vec<String> {
        let deadline = Instant::now() + TIMEOUT;
        let mut lines = Vec::new();
        loop {
            let line = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|error| {
                    panic!("waiting for {prefix}: {error}; received {lines:?}")
                });
            let done = line.starts_with(prefix);
            lines.push(line);
            if done {
                return lines;
            }
        }
    }

    fn search(&mut self, command: &str) -> Vec<String> {
        self.send(command);
        self.until("bestmove ")
    }

    fn ready_without_extra_bestmove(&mut self) {
        self.send("isready");
        let lines = self.until("readyok");
        assert!(
            !lines.iter().any(|s| s.starts_with("bestmove ")),
            "{lines:?}"
        );
    }

    fn wait_exit(&mut self) {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "{status}");
                return;
            }
            assert!(Instant::now() < deadline, "engine did not exit");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn bestmove(lines: &[String]) -> &str {
    let moves: Vec<_> = lines
        .iter()
        .filter_map(|s| s.strip_prefix("bestmove "))
        .collect();
    assert_eq!(moves.len(), 1, "{lines:?}");
    moves[0]
}

fn nodes(lines: &[String]) -> u64 {
    lines
        .iter()
        .rev()
        .find_map(|line| {
            let words: Vec<_> = line.split_whitespace().collect();
            (words.first() == Some(&"info") && words.get(1) != Some(&"string"))
                .then(|| {
                    words
                        .windows(2)
                        .find(|pair| pair[0] == "nodes")
                        .map(|pair| pair[1].parse().unwrap())
                })
                .flatten()
        })
        .expect("missing node count")
}

fn assert_legal(fen: &str, coordinate: &str) {
    let position: Position = fen.parse().unwrap();
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    assert!(
        moves.iter().any(|mv| mv.to_string() == coordinate),
        "{coordinate} in {fen}"
    );
}

#[test]
fn fixed_budget_is_exact_and_each_search_starts_from_the_unchanged_game() {
    let mut engine = Engine::new();
    engine.send("position startpos");
    let first = engine.search("go nodes 32");
    assert_eq!(nodes(&first), 32);
    assert_legal(START, bestmove(&first));
    let second = engine.search("go nodes 32");
    assert_eq!(nodes(&second), 32);
    assert_eq!(bestmove(&first), bestmove(&second));
    engine.ready_without_extra_bestmove();

    engine.send(&format!("position fen {MATE_IN_ONE}"));
    let mate = engine.search("go nodes 32");
    assert_eq!(nodes(&mate), 32);
    assert_eq!(bestmove(&mate), "b6b7");
}

#[test]
fn time_zero_budget_and_clock_searches_return_legal_moves() {
    let mut engine = Engine::new();
    engine.send("position startpos");
    for command in ["go nodes 0", "go movetime 0", "go wtime 0 btime 10000"] {
        let result = engine.search(command);
        assert_eq!(nodes(&result), 0);
        assert_legal(START, bestmove(&result));
    }
    let start = Instant::now();
    let result = engine.search("go movetime 100");
    assert!(start.elapsed() >= Duration::from_millis(100));
    assert!(nodes(&result) > 0);
    assert_legal(START, bestmove(&result));
    engine.ready_without_extra_bestmove();
}

#[test]
fn infinite_search_is_ready_responsive_stoppable_and_resource_bounded() {
    let mut engine = Engine::new();
    engine.send("position startpos\ngo infinite\nisready");
    let ready = engine.until("readyok");
    assert!(!ready.iter().any(|s| s.starts_with("bestmove ")));
    engine.send("stop\nstop");
    let result = engine.until("bestmove ");
    assert_legal(START, bestmove(&result));
    engine.ready_without_extra_bestmove();

    engine.send("setoption name MaxTreeNodes value 1");
    let result = engine.search("go nodes 100");
    assert_eq!(nodes(&result), 0);
    assert!(result.iter().any(|s| s.contains("tree node limit reached")));
    engine.send("go infinite");
    engine.until("info string tree node limit reached");
    engine.ready_without_extra_bestmove();
    let result = engine.search("stop");
    assert_eq!(nodes(&result), 0);
    assert_legal(START, bestmove(&result));
}

#[test]
fn queued_position_and_search_finish_the_previous_job_before_reporting_the_next() {
    let mut engine = Engine::new();
    engine.send(&format!(
        "position startpos\ngo infinite\nposition fen {MATE_IN_ONE}\ngo nodes 32"
    ));
    let old = engine.until("bestmove ");
    assert_legal(START, bestmove(&old));
    let new = engine.until("bestmove ");
    assert_eq!(bestmove(&new), "b6b7");
    assert_eq!(nodes(&new), 32);
    engine.ready_without_extra_bestmove();

    engine.send("go infinite\nucinewgame\nisready");
    let old = engine.until("bestmove ");
    assert_legal(MATE_IN_ONE, bestmove(&old));
    assert_eq!(engine.until("readyok"), ["readyok"]);
    assert_eq!(bestmove(&engine.search("go nodes 1")), "0000");
    engine.send("position startpos");
    assert_legal(START, bestmove(&engine.search("go nodes 1")));
}

#[test]
fn invalid_commands_leave_the_session_usable_and_preserve_the_previous_position() {
    let mut engine = Engine::new();
    engine.send("position startpos\nposition startpos moves e2e5");
    let rejected = engine.until("info string position rejected:");
    assert_eq!(rejected.len(), 1);
    for command in ["go nodes -1", "go depth 4", "go nodes 10 searchmoves e2e5"] {
        let result = engine.search(command);
        assert!(result[0].starts_with("info string "), "{result:?}");
        assert_eq!(bestmove(&result), "0000");
        engine.ready_without_extra_bestmove();
    }
    assert_legal(START, bestmove(&engine.search("go nodes 16")));
}

#[test]
fn root_move_restriction_handles_standard_castling_en_passant_and_underpromotion() {
    let mut engine = Engine::new();
    for (fen, mv) in [
        (START, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
    ] {
        engine.send(&format!("position fen {fen}"));
        let result = engine.search(&format!("go searchmoves {mv} {mv} nodes 8"));
        assert_eq!(bestmove(&result), mv);
        assert_eq!(nodes(&result), 8);
        assert_legal(fen, mv);
    }
    engine.send("position startpos");
    let result = engine.search("go nodes 32 searchmoves e2e4 d2d4");
    assert!(["e2e4", "d2d4"].contains(&bestmove(&result)));
}

#[test]
fn terminal_roots_return_null_moves_and_infinite_terminal_search_waits_for_stop() {
    let mut engine = Engine::new();
    for fen in [
        "k7/1Q6/2K5/8/8/8/8/8 b - - 1 1",    // Checkmate.
        "k7/8/1QK5/8/8/8/8/8 b - - 1 1",     // Stalemate.
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",     // Insufficient material.
        "4k3/8/8/8/8/8/8/R3K3 w - - 150 76", // 75-move rule.
    ] {
        engine.send(&format!("position fen {fen}"));
        let result = engine.search("go nodes 32");
        assert_eq!(bestmove(&result), "0000");
        assert_eq!(nodes(&result), 0);
    }
    engine.send("go infinite\nisready");
    assert!(
        !engine
            .until("readyok")
            .iter()
            .any(|s| s.starts_with("bestmove "))
    );
    assert_eq!(bestmove(&engine.search("stop")), "0000");
}

#[test]
fn playing_a_complete_repetition_game_preserves_history_across_searches() {
    let mut engine = Engine::new();
    let mut game = Game::new(START.parse().unwrap());
    let mut history = String::new();
    for coordinate in ["g1f3", "g8f6", "f3g1", "f6g8"].into_iter().cycle().take(8) {
        engine.send(&format!("position startpos moves {history}"));
        let result = engine.search(&format!("go nodes 8 searchmoves {coordinate}"));
        assert_eq!(bestmove(&result), coordinate);
        let mut moves = Vec::new();
        game.position().generate_legal_moves(&mut moves);
        let mv = moves
            .into_iter()
            .find(|mv| mv.to_string() == coordinate)
            .unwrap();
        game.play(mv).unwrap();
        history.push_str(coordinate);
        history.push(' ');
    }
    assert_eq!(game.repetition_count(), 3);
    assert_eq!(game.outcome(), None);
    engine.send(&format!("position startpos moves {history}"));
    let result = engine.search("go nodes 8");
    assert_eq!(bestmove(&result), "0000");
    assert!(
        result
            .iter()
            .any(|line| line
                == "info string draw adjudicated by search policy: threefold repetition")
    );
    // FEN alone has the same board but cannot carry the earlier repetitions.
    engine.send(&format!("position fen {}", game.position().to_fen()));
    assert_legal(START, bestmove(&engine.search("go nodes 8")));
}

#[test]
fn quit_and_eof_cancel_active_workers_and_exit_cleanly() {
    for quit in [true, false] {
        let mut engine = Engine::new();
        engine.send("position startpos\ngo infinite\nisready");
        engine.until("readyok");
        if quit {
            engine.send("quit");
        } else {
            drop(engine.child.stdin.take());
        }
        engine.wait_exit();
    }
}

#[derive(Debug)]
struct MoveStatistics {
    mv: String,
    visits: u64,
    prior: f64,
    q: f64,
}

fn statistics_snapshots(lines: &[String]) -> Vec<(u64, Vec<MoveStatistics>)> {
    let mut snapshots = Vec::new();
    let mut moves = Vec::new();
    for line in lines {
        if let Some(total) = line.strip_prefix("info string node N: ") {
            let total: u64 = total.parse().unwrap();
            assert_eq!(
                moves.iter().map(|m: &MoveStatistics| m.visits).sum::<u64>(),
                total
            );
            snapshots.push((total, std::mem::take(&mut moves)));
        } else if line.starts_with("info string ") && line.contains(" (P: ") {
            let words: Vec<_> = line.split_whitespace().collect();
            assert_eq!(words.len(), 9, "{line}");
            moves.push(MoveStatistics {
                mv: words[2].to_owned(),
                visits: words[4].parse().unwrap(),
                prior: words[6].strip_suffix("%)").unwrap().parse::<f64>().unwrap() / 100.0,
                q: words[8].strip_suffix(')').unwrap().parse().unwrap(),
            });
        }
    }
    assert!(
        moves.is_empty(),
        "incomplete statistics snapshot: {moves:?}"
    );
    snapshots
}

#[test]
fn reported_root_statistics_match_pyxis_including_zero_visits_and_both_perspectives() {
    use pyxis::{SearchReport, Tree, UniformEvaluator, resolve_node};
    let mut engine = Engine::new();
    let mut nonzero_q_for_both_colors = 0;
    for (fen, budget) in [
        (START, 0),
        (START, 1),
        (MATE_IN_ONE, 32),
        ("8/8/8/8/8/1qk5/8/K7 b - - 0 1", 32),
    ] {
        engine.send(&format!("position fen {fen}"));
        let lines = engine.search(&format!("go nodes {budget}"));
        let snapshots = statistics_snapshots(&lines);
        assert_eq!(snapshots.first().unwrap().0, 0);
        assert_eq!(snapshots.last().unwrap().0, budget);
        let mut game = Game::new(fen.parse().unwrap());
        let mut tree = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap());
        let mut completed = 0;
        let mut saw_nonzero_q = false;
        for (total, actual) in snapshots {
            while completed < total {
                tree.simulate(&mut game, &mut UniformEvaluator, 1.0)
                    .unwrap();
                completed += 1;
            }
            let SearchReport::Nonterminal {
                moves,
                best_move,
                simulations,
            } = tree.report()
            else {
                panic!()
            };
            assert_eq!(total, simulations);
            assert_eq!(actual.len(), moves.len());
            let unique: std::collections::HashSet<_> = actual.iter().map(|m| &m.mv).collect();
            assert_eq!(unique.len(), moves.len());
            // Nibbler takes the last verbose move as its preferred move.
            assert_eq!(actual.last().unwrap().mv, best_move.to_string());
            for expected in moves {
                let found = actual
                    .iter()
                    .find(|m| m.mv == expected.mv.to_string())
                    .unwrap();
                assert_eq!(found.visits, u64::from(expected.stats.visits()));
                assert!((found.prior - f64::from(expected.stats.prior())).abs() <= 1e-8);
                assert!((found.q - f64::from(expected.stats.mean_value())).abs() <= 1e-8);
            }
            if total > 0 && actual.iter().any(|m| m.q > 0.0) {
                saw_nonzero_q = true;
            }
        }
        nonzero_q_for_both_colors += usize::from(saw_nonzero_q);
    }
    assert_eq!(nonzero_q_for_both_colors, 2);
}

#[test]
fn verbose_option_toggles_only_reporting_and_focused_roots_report_normalized_priors() {
    let mut engine = Engine::new();
    engine.send("position startpos");
    let enabled = engine.search("go nodes 32");
    assert!(!statistics_snapshots(&enabled).is_empty());
    engine.send("setoption name VerboseMoveStats value false");
    let disabled = engine.search("go nodes 32");
    assert!(statistics_snapshots(&disabled).is_empty());
    assert_eq!(nodes(&enabled), nodes(&disabled));
    assert_eq!(bestmove(&enabled), bestmove(&disabled));
    engine.send("setoption name VerboseMoveStats value nonsense");
    assert_eq!(
        engine.until("info string "),
        ["info string VerboseMoveStats must be true or false"]
    );
    assert!(statistics_snapshots(&engine.search("go nodes 1")).is_empty());
    engine.send("setoption name VerboseMoveStats value true");
    let focused = engine.search("go nodes 7 searchmoves e2e4 d2d4");
    for (_, moves) in statistics_snapshots(&focused) {
        assert_eq!(moves.len(), 2);
        for mv in moves {
            assert!(["e2e4", "d2d4"].contains(&mv.mv.as_str()));
            assert!((mv.prior - 0.5).abs() < 1e-8);
        }
    }
    engine.send("setoption name MaxTreeNodes value 2\ngo infinite");
    let capped = engine.until("info string tree node limit reached");
    assert_eq!(statistics_snapshots(&capped).last().unwrap().0, 1);
    let stopped = engine.search("stop");
    assert_eq!(statistics_snapshots(&stopped).last().unwrap().0, 1);
    engine.ready_without_extra_bestmove();
}

#[test]
fn evaluator_selection_changes_values_and_visits_but_preserves_uniform_policy() {
    const HANGING_QUEEN: &str = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1";
    let mut engine = Engine::new();
    engine.send(&format!("position fen {HANGING_QUEEN}"));
    let uniform = engine.search("go nodes 128");
    assert_eq!(bestmove(&uniform), "d1a1");
    let uniform_stats = statistics_snapshots(&uniform).pop().unwrap().1;
    assert!(uniform_stats.iter().all(|m| m.q == 0.0));
    engine.send("setoption name Evaluator value material");
    let material = engine.search("go nodes 128");
    assert_eq!(nodes(&material), 128);
    assert_eq!(bestmove(&material), "d1d5");
    let material_stats = statistics_snapshots(&material).pop().unwrap().1;
    let capture = material_stats.iter().find(|m| m.mv == "d1d5").unwrap();
    assert_eq!(capture.q, 0.5);
    assert!(capture.visits > 64);
    for entry in &material_stats {
        assert_eq!(
            entry.prior,
            uniform_stats
                .iter()
                .find(|m| m.mv == entry.mv)
                .unwrap()
                .prior
        );
    }
    engine.send("setoption name Evaluator value unknown");
    assert_eq!(
        engine.until("info string "),
        ["info string Evaluator must be uniform, material, or neural"]
    );
    let still_material = engine.search("go nodes 1 searchmoves d1d5");
    assert_eq!(
        statistics_snapshots(&still_material).last().unwrap().1[0].q,
        0.5
    );
    engine.send("setoption name Evaluator value uniform");
    let back_to_uniform = engine.search("go nodes 1 searchmoves d1d5");
    assert_eq!(
        statistics_snapshots(&back_to_uniform).last().unwrap().1[0].q,
        0.0
    );
    engine.send(
        "setoption name Evaluator value material\nposition fen k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
    );
    assert_eq!(bestmove(&engine.search("go nodes 128")), "0000");
}

#[test]
fn changing_evaluator_finishes_the_old_search_before_starting_the_new_one() {
    let mut engine = Engine::new();
    engine.send("position fen 4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1\nsetoption name MaxTreeNodes value 2\ngo infinite searchmoves d1d5");
    engine.until("info string tree node limit reached");
    engine.send("setoption name Evaluator value material\nsetoption name MaxTreeNodes value 1000\ngo nodes 128 searchmoves d1d5");
    let old = engine.until("bestmove ");
    assert_eq!(nodes(&old), 1);
    assert_eq!(statistics_snapshots(&old).last().unwrap().1[0].q, 0.0);
    let new = engine.until("bestmove ");
    assert_eq!(nodes(&new), 128);
    assert_eq!(statistics_snapshots(&new).last().unwrap().1[0].q, 0.5);
    engine.ready_without_extra_bestmove();
}
