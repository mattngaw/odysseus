use std::{
    io,
    sync::{Mutex, atomic::AtomicBool},
};

use odysseus::neural::NeuralEvaluator;
pub(super) use odysseus::neural::default_python;
use penteconter::{Game, Move};
use pyxis::{Evaluation, Evaluator, MaterialEvaluator, UniformEvaluator};

/// UCI's evaluator selection; Pyxis remains independent of Python.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Choice {
    #[default]
    Uniform,
    Material,
    Neural,
}

pub(super) type Cache = Mutex<Option<NeuralEvaluator>>;

pub(super) fn clear_cache(cache: &Cache) {
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    cache.clear_poison();
}

/// The session retains the evaluator across searches; one search uses it at a time.
/// Settings changes clear the cache before another adapter uses it.
pub(super) struct Adapter<'a> {
    pub choice: Choice,
    pub python: &'a str,
    pub checkpoint: Option<&'a str>,
    pub cache: &'a Cache,
    pub stop: &'a AtomicBool,
}

impl Evaluator for Adapter<'_> {
    type Error = io::Error;

    fn evaluate(&mut self, game: &Game, legal_moves: &[Move]) -> Result<Evaluation, Self::Error> {
        match self.choice {
            Choice::Uniform => UniformEvaluator
                .evaluate(game, legal_moves)
                .map_err(|e| match e {}),
            Choice::Material => MaterialEvaluator
                .evaluate(game, legal_moves)
                .map_err(|e| match e {}),
            Choice::Neural => self
                .cache
                .lock()
                .map_err(|_| io::Error::other("inference cache poisoned"))?
                .get_or_insert_with(|| {
                    NeuralEvaluator::new(self.python, self.checkpoint.map(Into::into))
                })
                .evaluate_cancellable(game, legal_moves, self.stop),
        }
    }
}
