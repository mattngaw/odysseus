//! Persistent Python inference, independent of the UCI search lifecycle.
//!
//! Protocol v1 uses a 16-byte startup header: `ODNN`, then little-endian u32
//! version, input float count, output float count. Each request is square-major
//! 64*110 little-endian f32 features. Each reply is 1858 policy logits then 3
//! side-to-move W/D/L logits. Encoding/vocabulary changes require a version bump.
//! Only one request is in flight. Model loading and each exchange have a timeout.
//! A startup failure may instead return `ODNE`, u32 version=1, u32 UTF-8 byte
//! count (1..=4096), u32 reserved=0, then that message. No requests follow it.

use std::{
    io::{self, Read, Write},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use pyxis::{
    encoding::{EncodedInput, FEATURE_COUNT, SQUARE_COUNT},
    vocabulary::POLICY_SIZE,
};

const INPUT_FLOATS: usize = SQUARE_COUNT * FEATURE_COUNT;
const OUTPUT_FLOATS: usize = POLICY_SIZE + 3;

/// Raw network output. Legal policy normalization and value conversion are external.
#[derive(Debug, PartialEq)]
pub struct Prediction {
    pub policy_logits: [f32; POLICY_SIZE],
    pub value_logits: [f32; 3],
}

enum Reply {
    Ready,
    Prediction(Box<Prediction>),
}

/// Owns one worker process, kept alive across calls. Drop kills and reaps it.
/// A failed exchange terminates the worker; callers must start a new one.
pub struct InferenceWorker {
    child: Child,
    requests: Option<Sender<Vec<u8>>>,
    responses: Receiver<io::Result<Reply>>,
    io_thread: Option<JoinHandle<()>>,
    timeout: Duration,
}

impl InferenceWorker {
    /// Launch a command such as `python -m neurodiktyon.inference_worker`.
    /// The caller selects the interpreter, module arguments, and working directory.
    /// Stdout/stdin are reserved for the protocol; stderr is inherited.
    pub fn spawn(command: &mut Command, timeout: Duration) -> io::Result<Self> {
        Self::spawn_cancellable(command, timeout, &AtomicBool::new(false))
    }

    /// Like spawn, but cancellation interrupts startup and reaps the child.
    pub fn spawn_cancellable(
        command: &mut Command,
        timeout: Duration,
        cancel: &AtomicBool,
    ) -> io::Result<Self> {
        if cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "inference cancelled",
            ));
        }
        if timeout.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "timeout must be positive",
            ));
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (requests, incoming) = mpsc::channel();
        let (outgoing, responses) = mpsc::channel();
        // Both writing and reading happen here: a worker that stops reading must
        // not block the caller inside write_all before it can enforce its timeout.
        let io_thread = match thread::Builder::new()
            .name("neural-inference-io".into())
            .spawn(move || {
                communicate(stdin, stdout, incoming, outgoing);
            }) {
            Ok(handle) => handle,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let worker = Self {
            child,
            requests: Some(requests),
            responses,
            io_thread: Some(io_thread),
            timeout,
        };
        match worker.receive(cancel)? {
            Reply::Ready => Ok(worker),
            Reply::Prediction(_) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected worker handshake",
            )),
        }
    }

    pub fn infer(&mut self, input: &EncodedInput) -> io::Result<Prediction> {
        self.infer_cancellable(input, &AtomicBool::new(false))
    }

    /// Cancelling an in-flight request terminates the child to avoid stale replies.
    pub fn infer_cancellable(
        &mut self,
        input: &EncodedInput,
        cancel: &AtomicBool,
    ) -> io::Result<Prediction> {
        if cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "inference cancelled",
            ));
        }
        let Some(requests) = &self.requests else {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "inference worker is closed",
            ));
        };
        if input.iter().flatten().any(|value| !value.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "inference features must be finite",
            ));
        }
        let mut bytes = Vec::with_capacity(INPUT_FLOATS * 4);
        for value in input.iter().flatten() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let result = requests
            .send(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "inference I/O thread exited"))
            .and_then(|()| self.receive(cancel))
            .and_then(|reply| match reply {
                Reply::Prediction(prediction) => Ok(*prediction),
                Reply::Ready => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unexpected worker handshake",
                )),
            });
        if result.is_err() {
            self.terminate();
        }
        result
    }

    fn receive(&self, cancel: &AtomicBool) -> io::Result<Reply> {
        let start = Instant::now();
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "inference cancelled",
                ));
            }
            let remaining = self.timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "waiting for inference worker",
                ));
            }
            match self
                .responses
                .recv_timeout(remaining.min(Duration::from_millis(10)))
            {
                Ok(reply) => return reply,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "inference I/O thread exited",
                    ));
                }
            }
        }
    }

    fn terminate(&mut self) {
        self.requests.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.io_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn communicate(
    mut stdin: impl Write,
    mut stdout: impl Read,
    requests: Receiver<Vec<u8>>,
    responses: Sender<io::Result<Reply>>,
) {
    let ready = read_handshake(&mut stdout).map(|()| Reply::Ready);
    let failed = ready.is_err();
    if responses.send(ready).is_err() || failed {
        return;
    }
    for bytes in requests {
        let result = stdin
            .write_all(&bytes)
            .and_then(|()| stdin.flush())
            .and_then(|()| read_prediction(&mut stdout))
            .map(|prediction| Reply::Prediction(Box::new(prediction)));
        let failed = result.is_err();
        if responses.send(result).is_err() || failed {
            return;
        }
    }
}

fn read_handshake(source: &mut impl Read) -> io::Result<()> {
    let mut header = [0u8; 16];
    source.read_exact(&mut header)?;
    let field = |i| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    if &header[..4] == b"ODNE" {
        if field(4) != 1 || !(1..=4096).contains(&field(8)) || field(12) != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid startup error frame",
            ));
        }
        let mut message = vec![0; field(8) as usize];
        source.read_exact(&mut message)?;
        let message = String::from_utf8(message)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        // UCI errors occupy one line, even if a library emitted a multiline error.
        let message = message.split_whitespace().collect::<Vec<_>>().join(" ");
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("worker startup failed: {message}"),
        ));
    }
    if &header[..4] != b"ODNN"
        || field(4) != 1
        || field(8) != INPUT_FLOATS as u32
        || field(12) != OUTPUT_FLOATS as u32
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incompatible inference protocol or tensor sizes",
        ));
    }
    Ok(())
}

fn read_prediction(source: &mut impl Read) -> io::Result<Prediction> {
    let mut bytes = [0u8; OUTPUT_FLOATS * 4];
    source.read_exact(&mut bytes)?;
    let values: [f32; OUTPUT_FLOATS] =
        std::array::from_fn(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    if values.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "worker returned nonfinite logits",
        ));
    }
    Ok(Prediction {
        policy_logits: values[..POLICY_SIZE].try_into().unwrap(),
        value_logits: values[POLICY_SIZE..].try_into().unwrap(),
    })
}

#[cfg(test)]
mod tests;
