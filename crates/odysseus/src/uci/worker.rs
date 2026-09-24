use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use super::{Event, evaluator, limits::Limits, position, report::report_lines};
use pyxis::{
    Adjudication, Evaluation, ExpandedNode, Node, ResolveError, Tree, adjudicate, resolve_node,
};

/// Cloned into each worker so options stay fixed throughout a search.
#[derive(Clone)]
pub(super) struct SearchOptions {
    pub max_tree_nodes: usize,
    pub verbose_move_stats: bool,
    pub evaluator: evaluator::Choice,
    pub neural_python: String,
    pub neural_checkpoint: Option<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_tree_nodes: 100_000,
            verbose_move_stats: true,
            evaluator: evaluator::Choice::Uniform,
            neural_python: evaluator::default_python(),
            neural_checkpoint: None,
        }
    }
}

pub(super) struct Active {
    pub id: u64,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Active {
    pub fn start(
        id: u64,
        position: String,
        limits: Limits,
        options: SearchOptions,
        cache: Arc<evaluator::Cache>,
        events: Sender<Event>,
    ) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("odysseus-search".into())
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    search(id, &position, limits, options, &signal, &events, &cache)
                }));
                let lines = match result {
                    Ok(Ok(lines)) => lines,
                    Ok(Err(error)) => vec![
                        format!("info string search failed: {error}"),
                        "bestmove 0000".into(),
                    ],
                    Err(_) => vec![
                        "info string search failed: worker panicked".into(),
                        "bestmove 0000".into(),
                    ],
                };
                let _ = events.send(Event::Finished { id, lines });
            })?;
        Ok(Self {
            id,
            stop,
            handle: Some(handle),
        })
    }

    pub fn stop(&self) {
        // Only a flag is communicated; no other data needs synchronization.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = &self.handle {
            handle.thread().unpark();
        }
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn search(
    id: u64,
    source: &str,
    limits: Limits,
    options: SearchOptions,
    stop: &AtomicBool,
    events: &Sender<Event>,
    cache: &evaluator::Cache,
) -> Result<Vec<String>, String> {
    let start = Instant::now();
    // Reconstruct the supplied history for this worker. The coordinator keeps
    // its original Game intact, even if a worker fails or panics during traversal.
    let mut game = position::parse(source.split_whitespace()).map_err(|e| e.to_string())?;
    let mut evaluator = evaluator::Adapter {
        choice: options.evaluator,
        python: &options.neural_python,
        checkpoint: options.neural_checkpoint.as_deref(),
        cache,
        stop,
    };
    if options.evaluator == evaluator::Choice::Neural {
        let description = options.neural_checkpoint.as_ref().map_or_else(
            || "untrained seed-0 CPU FP32 model".to_owned(),
            |path| format!("CPU FP32; checkpoint={path}"),
        );
        let _ = events.send(Event::Progress {
            id,
            lines: vec![format!("info string neural evaluator: {description}")],
        });
    }
    let verbose_move_stats = options.verbose_move_stats;
    let mut root = match resolve_node(&game, &mut evaluator) {
        Ok(root) => root,
        Err(ResolveError::Evaluator(error))
            if error.kind() == std::io::ErrorKind::Interrupted && stop.load(Ordering::Relaxed) =>
        {
            // No evaluated root exists yet. Honour stop with a legal allowed move,
            // explicitly identifying the unevaluated fallback instead of resigning.
            let mut moves = Vec::new();
            game.position().generate_legal_moves(&mut moves);
            for requested in &limits.searchmoves {
                if !moves.iter().any(|mv| mv.to_string() == *requested) {
                    return Err(format!("illegal searchmoves entry: {requested}"));
                }
            }
            let fallback = moves.iter().find(|mv| {
                limits.searchmoves.is_empty() || limits.searchmoves.contains(&mv.to_string())
            });
            return Ok(vec![
                "info string stopped before root evaluation; returning an unevaluated legal move"
                    .into(),
                format!(
                    "bestmove {}",
                    fallback.map_or_else(|| "0000".into(), ToString::to_string)
                ),
            ]);
        }
        Err(error) => return Err(error.to_string()),
    };
    if matches!(root, Node::Terminal(_))
        && adjudicate(&game) == Some(Adjudication::ThreefoldRepetitionDraw)
    {
        let _ = events.send(Event::Progress {
            id,
            lines: vec![
                "info string draw adjudicated by search policy: threefold repetition".into(),
            ],
        });
    }
    if let Node::Expanded(node) = &root
        && !limits.searchmoves.is_empty()
    {
        for requested in &limits.searchmoves {
            if !node
                .edges()
                .iter()
                .any(|edge| edge.mv().to_string() == *requested)
            {
                return Err(format!("illegal searchmoves entry: {requested}"));
            }
        }
        // Evaluate the complete legal list first, then gather and renormalize
        // allowed root moves. Descendants retain all legal moves.
        let allowed: Vec<_> = node
            .edges()
            .iter()
            .filter(|edge| limits.searchmoves.contains(&edge.mv().to_string()))
            .collect();
        let moves: Vec<_> = allowed.iter().map(|edge| edge.mv()).collect();
        let evaluation = Evaluation {
            value: node.value(),
            policy_weights: allowed.iter().map(|edge| edge.stats().prior()).collect(),
        };
        root = Node::Expanded(ExpandedNode::new(&moves, evaluation).map_err(|e| e.to_string())?);
    }
    let mut tree = Tree::new(root);
    let terminal = matches!(tree.node(tree.root()), Some(Node::Terminal(_)));
    let mut completed = 0u64;
    let mut last_info = Instant::now();
    let _ = events.send(Event::Progress {
        id,
        lines: report_lines(&tree.report(), start.elapsed(), false, verbose_move_stats),
    });
    let mut reason = None;
    while !terminal && !stop.load(Ordering::Relaxed) {
        if limits.nodes.is_some_and(|limit| completed >= limit)
            || limits.time.is_some_and(|limit| start.elapsed() >= limit)
        {
            break;
        }
        if tree.node_count() >= options.max_tree_nodes {
            reason = Some("tree node limit reached");
            break;
        }
        if let Err(error) = tree.simulate(&mut game, &mut evaluator, 1.0) {
            // A returned error preserves earlier completed simulations.
            let mut lines = report_lines(&tree.report(), start.elapsed(), true, verbose_move_stats);
            if !stop.load(Ordering::Relaxed) {
                lines.insert(0, format!("info string simulation failed: {error}"));
            }
            return Ok(lines);
        }
        completed += 1;
        if last_info.elapsed() >= Duration::from_millis(250) {
            if events
                .send(Event::Progress {
                    id,
                    lines: report_lines(&tree.report(), start.elapsed(), false, verbose_move_stats),
                })
                .is_err()
            {
                break;
            }
            last_info = Instant::now();
        }
    }
    if let Some(reason) = reason {
        let mut lines = report_lines(&tree.report(), start.elapsed(), false, verbose_move_stats);
        lines.push(format!("info string {reason}"));
        let _ = events.send(Event::Progress { id, lines });
    }
    // Infinite analysis waits for stop even after a terminal root or resource
    // limit. Park rather than spinning or allocating more tree storage.
    while limits.wait_for_stop && !stop.load(Ordering::Relaxed) {
        thread::park_timeout(Duration::from_millis(100));
    }
    Ok(report_lines(
        &tree.report(),
        start.elapsed(),
        true,
        verbose_move_stats,
    ))
}
