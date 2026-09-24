use penteconter::Game;
use std::{
    collections::VecDeque,
    io::{self, BufRead, Write},
    sync::{Arc, mpsc},
    thread,
};

mod evaluator;
mod limits;
mod position;
mod report;
mod worker;

enum Event {
    Input(String),
    End(io::Result<()>),
    Progress { id: u64, lines: Vec<String> },
    Finished { id: u64, lines: Vec<String> },
}

/// The coordinator owns the current game and all protocol output.
#[derive(Default)]
pub(crate) struct Session {
    game: Option<Game>,
    position_source: Option<String>,
    search_options: worker::SearchOptions,
    neural_cache: Arc<evaluator::Cache>,
}

impl Session {
    pub(crate) fn run(
        &mut self,
        mut input: impl BufRead + Send + 'static,
        mut output: impl Write,
    ) -> io::Result<()> {
        let (sender, receiver) = mpsc::channel();
        let inputs = sender.clone();
        // Stdin may block until process exit after quit. This reader owns its
        // input and is deliberately not joined; search workers are always joined.
        thread::Builder::new()
            .name("odysseus-input".into())
            .spawn(move || {
                loop {
                    let mut line = String::new();
                    match input.read_line(&mut line) {
                        Ok(0) => {
                            let _ = inputs.send(Event::End(Ok(())));
                            break;
                        }
                        Ok(_) => {
                            if inputs.send(Event::Input(line)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = inputs.send(Event::End(Err(error)));
                            break;
                        }
                    }
                }
            })?;
        let mut pending = VecDeque::new();
        let mut active: Option<worker::Active> = None;
        let mut next_id = 0u64;
        loop {
            match receiver.recv().expect("the coordinator retains a sender") {
                Event::End(result) => return result, // Active's Drop cancels and joins.
                Event::Input(line) => {
                    if line.split_whitespace().next() == Some("quit") {
                        return Ok(());
                    }
                    pending.push_back(line);
                }
                Event::Progress { id, lines } => {
                    if active.as_ref().is_some_and(|job| job.id == id) {
                        write_lines(&mut output, lines)?;
                    }
                }
                Event::Finished { id, lines } => {
                    if active.as_ref().is_some_and(|job| job.id == id) {
                        // Finish the old search before changing its position or
                        // starting another. Never interleave reports from jobs.
                        drop(active.take());
                        write_lines(&mut output, lines)?;
                    }
                }
            }
            while let Some(line) = pending.front() {
                let command = line.split_whitespace().next();
                if matches!(
                    command,
                    Some("position" | "go" | "ucinewgame" | "setoption")
                ) && let Some(job) = &active
                {
                    job.stop();
                    break;
                }
                let line = pending.pop_front().unwrap();
                let mut words = line.split_whitespace();
                match words.next() {
                    Some("uci") => {
                        writeln!(output, "id name Odysseus {}", env!("CARGO_PKG_VERSION"))?;
                        writeln!(output, "id author Matt")?;
                        writeln!(
                            output,
                            "option name MaxTreeNodes type spin default 100000 min 1 max 10000000"
                        )?;
                        writeln!(
                            output,
                            "option name MultiPV type spin default 1 min 1 max 1"
                        )?;
                        writeln!(
                            output,
                            "option name VerboseMoveStats type check default true"
                        )?;
                        writeln!(
                            output,
                            "option name Evaluator type combo default uniform var uniform var material var neural"
                        )?;
                        writeln!(
                            output,
                            "option name NeuralPython type string default {}",
                            evaluator::default_python()
                        )?;
                        writeln!(
                            output,
                            "option name NeuralCheckpoint type string default <empty>"
                        )?;
                        writeln!(output, "uciok")?;
                    }
                    Some("isready") => writeln!(output, "readyok")?,
                    Some("stop") => {
                        if let Some(job) = &active {
                            job.stop();
                        }
                    }
                    Some("ucinewgame") => {
                        self.game = None;
                        self.position_source = None;
                    }
                    Some("position") => {
                        let source = words.collect::<Vec<_>>().join(" ");
                        match position::parse(source.split_whitespace()) {
                            Ok(game) => {
                                self.game = Some(game);
                                self.position_source = Some(source);
                            }
                            Err(error) => {
                                writeln!(output, "info string position rejected: {error}")?
                            }
                        }
                    }
                    Some("setoption") => {
                        let original_text = words.collect::<Vec<_>>().join(" ");
                        let text = original_text.to_ascii_lowercase();
                        if let Some(value) = text.strip_prefix("name maxtreenodes value ") {
                            match value.parse::<usize>() {
                                Ok(value @ 1..=10_000_000) => {
                                    self.search_options.max_tree_nodes = value
                                }
                                _ => writeln!(
                                    output,
                                    "info string MaxTreeNodes must be between 1 and 10000000"
                                )?,
                            }
                        }
                        if let Some(value) = text.strip_prefix("name verbosemovestats value ") {
                            match value {
                                "true" => self.search_options.verbose_move_stats = true,
                                "false" => self.search_options.verbose_move_stats = false,
                                _ => writeln!(
                                    output,
                                    "info string VerboseMoveStats must be true or false"
                                )?,
                            }
                        }
                        if let Some(value) = text.strip_prefix("name evaluator value ") {
                            match value {
                                "uniform" => {
                                    self.search_options.evaluator = evaluator::Choice::Uniform
                                }
                                "material" => {
                                    self.search_options.evaluator = evaluator::Choice::Material
                                }
                                "neural" => {
                                    self.search_options.evaluator = evaluator::Choice::Neural
                                }
                                _ => writeln!(
                                    output,
                                    "info string Evaluator must be uniform, material, or neural"
                                )?,
                            }
                            if self.search_options.evaluator != evaluator::Choice::Neural {
                                evaluator::clear_cache(&self.neural_cache);
                            }
                        }
                        if let Some(value) = string_option(&line, "NeuralPython") {
                            if value.is_empty() {
                                writeln!(
                                    output,
                                    "info string NeuralPython must name a Python interpreter"
                                )?;
                            } else if self.search_options.neural_python != value {
                                evaluator::clear_cache(&self.neural_cache);
                                self.search_options.neural_python = value.to_owned();
                            }
                        }
                        if let Some(value) = string_option(&line, "NeuralCheckpoint") {
                            // Explicitly setting the same path also reloads the file.
                            // The coordinator has already stopped/joined active search.
                            evaluator::clear_cache(&self.neural_cache);
                            self.search_options.neural_checkpoint =
                                if value.is_empty() || value == "<empty>" {
                                    None
                                } else {
                                    Some(value.to_owned())
                                };
                        }
                        // MultiPV is fixed at one. Unknown options are ignored.
                    }
                    Some("go") => {
                        let limits = self
                            .game
                            .as_ref()
                            .ok_or_else(|| "no position loaded".to_owned())
                            .and_then(|game| {
                                limits::Limits::parse(words, game.position().side_to_move())
                            });
                        match limits {
                            Ok(limits) => {
                                next_id = next_id.checked_add(1).expect("search ID exhausted");
                                active = Some(worker::Active::start(
                                    next_id,
                                    self.position_source
                                        .clone()
                                        .expect("a loaded game has its source"),
                                    limits,
                                    self.search_options.clone(),
                                    Arc::clone(&self.neural_cache),
                                    sender.clone(),
                                )?);
                            }
                            Err(error) => {
                                writeln!(output, "info string go rejected: {error}")?;
                                writeln!(output, "bestmove 0000")?;
                            }
                        }
                    }
                    _ => {}
                }
                output.flush()?;
            }
        }
    }
}

/// Parse a single-word string option without changing case or internal path spaces.
fn string_option<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let mut remaining = line.trim_start();
    for expected in ["setoption", "name", name, "value"] {
        let token = remaining.split_whitespace().next()?;
        if !token.eq_ignore_ascii_case(expected) {
            return None;
        }
        remaining = remaining[token.len()..].trim_start();
    }
    Some(remaining.trim_end())
}

fn write_lines(output: &mut impl Write, lines: Vec<String>) -> io::Result<()> {
    for line in lines {
        writeln!(output, "{line}")?;
    }
    output.flush()
}

#[cfg(test)]
mod tests;
