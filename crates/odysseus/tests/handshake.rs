use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::mpsc::{self, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

// Ensure a failed assertion or timeout cannot leave the engine running.
struct Engine(Child);

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn executable_replies_and_quits_while_stdin_remains_open() {
    let mut engine = Engine(
        Command::new(env!("CARGO_BIN_EXE_odysseus"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let stdout = engine.0.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let timeout = Duration::from_secs(5);

    // Send each command only after receiving the preceding response. A
    // transcript sent all at once would miss replies buffered until exit.
    for _ in 0..2 {
        let input = engine.0.stdin.as_mut().unwrap();
        writeln!(input, "uci").unwrap();
        input.flush().unwrap();
        for expected in [
            format!("id name Odysseus {}", env!("CARGO_PKG_VERSION")),
            "id author Matt".to_owned(),
            "option name MaxTreeNodes type spin default 100000 min 1 max 10000000".to_owned(),
            "option name MultiPV type spin default 1 min 1 max 1".to_owned(),
            "option name VerboseMoveStats type check default true".to_owned(),
            "option name Evaluator type combo default uniform var uniform var material var neural"
                .to_owned(),
            format!(
                "option name NeuralPython type string default {}",
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join(".venv/bin/python")
                    .display()
            ),
            "option name NeuralCheckpoint type string default <empty>".to_owned(),
            "uciok".to_owned(),
        ] {
            assert_eq!(receiver.recv_timeout(timeout).unwrap().unwrap(), expected);
        }
        let input = engine.0.stdin.as_mut().unwrap();
        writeln!(input, "isready").unwrap();
        input.flush().unwrap();
        assert_eq!(receiver.recv_timeout(timeout).unwrap().unwrap(), "readyok");
    }

    let input = engine.0.stdin.as_mut().unwrap();
    writeln!(input, "position startpos moves e2e4 e7e5\nisready").unwrap();
    input.flush().unwrap();
    assert_eq!(receiver.recv_timeout(timeout).unwrap().unwrap(), "readyok");

    let input = engine.0.stdin.as_mut().unwrap();
    writeln!(input, "position startpos moves e2e5\nisready").unwrap();
    input.flush().unwrap();
    assert_eq!(
        receiver.recv_timeout(timeout).unwrap().unwrap(),
        "info string position rejected: illegal move 1: e2e5"
    );
    assert_eq!(receiver.recv_timeout(timeout).unwrap().unwrap(), "readyok");

    let input = engine.0.stdin.as_mut().unwrap();
    writeln!(input, "quit").unwrap();
    input.flush().unwrap();
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = engine.0.try_wait().unwrap() {
            assert!(status.success(), "engine exited with {status}");
            break;
        }
        assert!(Instant::now() < deadline, "engine did not honor quit");
        thread::sleep(Duration::from_millis(10));
    }
    assert!(matches!(
        receiver.recv_timeout(timeout),
        Err(RecvTimeoutError::Disconnected)
    ));
    reader.join().unwrap();
}
