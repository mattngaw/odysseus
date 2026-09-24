//! Neural evaluation shared by UCI and recorded self-play.

use std::{
    io,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use penteconter::{Game, Move};
use pyxis::{Evaluation, Evaluator, Value, encoding, policy_from_logits};

use crate::inference::InferenceWorker;

/// Development interpreter in this workspace's editable Python environment.
pub fn default_python() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate is under workspace/crates")
        .join(".venv/bin/python")
        .to_string_lossy()
        .into_owned()
}

/// Owns a lazy, persistent CPU FP32 worker; dropping the evaluator reaps it.
/// Settings are fixed; loaded weights are retained until the worker is discarded.
pub struct NeuralEvaluator {
    python: PathBuf,
    checkpoint: Option<PathBuf>,
    worker: Option<InferenceWorker>,
}

impl NeuralEvaluator {
    /// `None` selects the untrained seed-0 model. A supplied checkpoint must load
    /// successfully: errors never fall back to random weights. Loading is deferred
    /// until evaluation, so terminal games need no Python process.
    pub fn new(python: impl Into<PathBuf>, checkpoint: Option<PathBuf>) -> Self {
        Self {
            python: python.into(),
            checkpoint,
            worker: None,
        }
    }

    /// Cancellation interrupts worker startup or inference. A failed exchange
    /// discards the worker; a later call can start a fresh process. Startup and
    /// each exchange have a 30-second timeout. UCI owns the cancellation flag;
    /// sequential self-play can use the ordinary `Evaluator` implementation.
    pub fn evaluate_cancellable(
        &mut self,
        game: &Game,
        legal_moves: &[Move],
        stop: &AtomicBool,
    ) -> io::Result<Evaluation> {
        if stop.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "search stopped"));
        }
        if self.worker.is_none() {
            let mut command = Command::new(&self.python);
            command.args([
                "-m",
                "neurodiktyon.inference_worker",
                "--seed",
                "0",
                "--threads",
                "1",
            ]);
            if let Some(checkpoint) = &self.checkpoint {
                command.arg("--checkpoint").arg(checkpoint);
            }
            self.worker = Some(InferenceWorker::spawn_cancellable(
                &mut command,
                Duration::from_secs(30),
                stop,
            )?);
        }
        let prediction = match self
            .worker
            .as_mut()
            .unwrap()
            .infer_cancellable(&encoding::encode(game), stop)
        {
            Ok(prediction) => prediction,
            Err(error) => {
                self.worker.take();
                return Err(error);
            }
        };
        Ok(Evaluation {
            value: value_from_logits(prediction.value_logits)?,
            policy_weights: policy_from_logits(
                &prediction.policy_logits,
                game.position().side_to_move(),
                legal_moves,
            )
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        })
    }
}

impl Evaluator for NeuralEvaluator {
    type Error = io::Error;

    fn evaluate(&mut self, game: &Game, legal_moves: &[Move]) -> io::Result<Evaluation> {
        self.evaluate_cancellable(game, legal_moves, &AtomicBool::new(false))
    }
}

fn value_from_logits(logits: [f32; 3]) -> io::Result<Value> {
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "nonfinite W/D/L logits",
        ));
    }
    let max = logits.into_iter().fold(f32::NEG_INFINITY, f32::max) as f64;
    let weights = logits.map(|x| (f64::from(x) - max).exp());
    let value = ((weights[0] - weights[2]) / weights.iter().sum::<f64>()) as f32;
    Value::new(value)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid neural value"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wdl_value_uses_side_to_move_order_and_stable_softmax() {
        assert_eq!(value_from_logits([0.0; 3]).unwrap().get(), 0.0);
        let result = value_from_logits([2.0f32.ln(), 3.0f32.ln(), 5.0f32.ln()]).unwrap();
        assert!((result.get() + 0.3).abs() < 1e-6);
        assert_eq!(
            value_from_logits([f32::MAX, -f32::MAX, 0.0]).unwrap().get(),
            1.0
        );
        assert_eq!(
            value_from_logits([-f32::MAX, 0.0, f32::MAX]).unwrap().get(),
            -1.0
        );
        assert_eq!(value_from_logits([0.0, f32::MAX, 0.0]).unwrap().get(), 0.0);
        assert!(value_from_logits([f32::NAN, 0.0, 0.0]).is_err());
    }
}
