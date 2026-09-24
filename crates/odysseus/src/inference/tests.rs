use super::*;

fn header() -> Vec<u8> {
    let mut bytes = b"ODNN".to_vec();
    for value in [1u32, 7040, 1861] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[test]
fn handshake_checks_magic_version_and_both_tensor_sizes() {
    read_handshake(&mut header().as_slice()).unwrap();
    for index in [0, 4, 8, 12] {
        let mut bytes = header();
        bytes[index] ^= 1;
        assert_eq!(
            read_handshake(&mut bytes.as_slice()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
    assert_eq!(
        read_handshake(&mut &header()[..15]).unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
}

#[test]
fn startup_error_frames_propagate_one_line_and_reject_invalid_lengths_or_utf8() {
    let frame = |version: u32, count: u32, reserved: u32, text: &[u8]| {
        let mut bytes = b"ODNE".to_vec();
        for number in [version, count, reserved] {
            bytes.extend_from_slice(&number.to_le_bytes());
        }
        bytes.extend_from_slice(text);
        bytes
    };
    let text = "checkpoint modèle.pt:\nunsupported encoding".as_bytes();
    let error = read_handshake(&mut frame(1, text.len() as u32, 0, text).as_slice()).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(
        error.to_string(),
        "worker startup failed: checkpoint modèle.pt: unsupported encoding"
    );
    for bytes in [
        frame(2, 1, 0, b"x"),
        frame(1, 0, 0, b""),
        frame(1, 4097, 0, b""),
        frame(1, u32::MAX, 0, b""),
        frame(1, 1, 1, b"x"),
        frame(1, 1, 0, &[255]),
    ] {
        assert_eq!(
            read_handshake(&mut bytes.as_slice()).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
    assert_eq!(
        read_handshake(&mut frame(1, 2, 0, b"x").as_slice())
            .unwrap_err()
            .kind(),
        io::ErrorKind::UnexpectedEof
    );
}

#[test]
fn reply_decodes_little_endian_policy_then_wdl_without_normalizing() {
    let values: Vec<_> = (0..OUTPUT_FLOATS).map(|i| i as f32 / 16.0 - 10.0).collect();
    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let prediction = read_prediction(&mut bytes.as_slice()).unwrap();
    assert_eq!(prediction.policy_logits, values[..POLICY_SIZE]);
    assert_eq!(prediction.value_logits, values[POLICY_SIZE..]);
}

#[test]
fn truncated_or_nonfinite_reply_is_rejected() {
    let mut bytes = vec![0; OUTPUT_FLOATS * 4];
    assert_eq!(
        read_prediction(&mut &bytes[..bytes.len() - 1])
            .unwrap_err()
            .kind(),
        io::ErrorKind::UnexpectedEof
    );
    for index in [0, POLICY_SIZE, OUTPUT_FLOATS - 1] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            assert_eq!(
                read_prediction(&mut bytes.as_slice()).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
        bytes[index * 4..index * 4 + 4].fill(0);
    }
}

// These process tests need a Python interpreter but never import PyTorch.
// Set ODYSSEUS_TEST_PYTHON to override the workspace's .venv interpreter.
fn python(script: &str) -> Command {
    let default = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    let path = std::env::var_os("ODYSSEUS_TEST_PYTHON").unwrap_or_else(|| default.into_os_string());
    let mut command = Command::new(path);
    command.args(["-u", "-c", script]);
    command
}

const READY: &str = "import sys, struct, time\nsys.stdout.buffer.write(struct.pack('<4sIII', b'ODNN', 1, 7040, 1861))\nsys.stdout.buffer.flush()\n";

#[test]
#[ignore = "requires Python; run cargo test -p odysseus --lib -- --ignored"]
fn process_startup_rejects_wrong_header_or_exit() {
    for script in [
        "pass",
        "import sys; sys.stdout.buffer.write(b'wrong protocol!!')",
    ] {
        let result = InferenceWorker::spawn(&mut python(script), Duration::from_secs(5));
        assert!(result.is_err());
    }
}

#[test]
#[ignore = "requires Python; run cargo test -p odysseus --lib -- --ignored"]
fn process_startup_and_exchange_timeouts_terminate_and_reap_the_worker() {
    let result = InferenceWorker::spawn(
        &mut python("import time; time.sleep(60)"),
        Duration::from_millis(500),
    );
    let Err(error) = result else {
        panic!("stalled startup succeeded")
    };
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);

    let mut worker = InferenceWorker::spawn(
        &mut python(&format!("{READY}time.sleep(60)")),
        Duration::from_secs(5),
    )
    .unwrap();
    worker.timeout = Duration::from_millis(50);
    let input = [[0.0; FEATURE_COUNT]; SQUARE_COUNT];
    assert_eq!(
        worker.infer(&input).unwrap_err().kind(),
        io::ErrorKind::TimedOut
    );
    assert!(worker.child.try_wait().unwrap().is_some());
    assert!(worker.io_thread.is_none());
    assert_eq!(
        worker.infer(&input).unwrap_err().kind(),
        io::ErrorKind::BrokenPipe
    );
}

#[test]
#[ignore = "requires Python; run cargo test -p odysseus --lib -- --ignored"]
fn process_bad_replies_close_the_connection_and_prevent_reuse() {
    for (response, kind) in [
        ("b''", io::ErrorKind::UnexpectedEof),
        ("bytes(4)", io::ErrorKind::UnexpectedEof),
        (
            "struct.pack('<1861f', *([float('nan')] * 1861))",
            io::ErrorKind::InvalidData,
        ),
    ] {
        let script = format!(
            "{READY}sys.stdin.buffer.read(7040*4)\nsys.stdout.buffer.write({response})\nsys.stdout.buffer.flush()\n"
        );
        let mut worker =
            InferenceWorker::spawn(&mut python(&script), Duration::from_secs(5)).unwrap();
        let input = [[0.0; FEATURE_COUNT]; SQUARE_COUNT];
        assert_eq!(worker.infer(&input).unwrap_err().kind(), kind);
        assert!(worker.child.try_wait().unwrap().is_some());
        assert_eq!(
            worker.infer(&input).unwrap_err().kind(),
            io::ErrorKind::BrokenPipe
        );
    }
}

#[test]
#[ignore = "requires Python; run cargo test -p odysseus --lib -- --ignored"]
fn process_invalid_input_sends_nothing_and_a_valid_request_still_works() {
    let script = format!(
        "{READY}sys.stdin.buffer.read(7040*4)\nsys.stdout.buffer.write(bytes(1861*4))\nsys.stdout.buffer.flush()\ntime.sleep(60)\n"
    );
    let mut worker = InferenceWorker::spawn(&mut python(&script), Duration::from_secs(5)).unwrap();
    let mut input = [[0.0; FEATURE_COUNT]; SQUARE_COUNT];
    input[17][33] = f32::NAN;
    assert_eq!(
        worker.infer(&input).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    input[17][33] = 0.0;
    let prediction = worker.infer(&input).unwrap();
    assert_eq!(prediction.value_logits, [0.0; 3]);
    worker.terminate();
    assert!(worker.child.try_wait().unwrap().is_some());
}
