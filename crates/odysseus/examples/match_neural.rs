//! One checkpoint-versus-checkpoint game, with streamed JSONL move diagnostics.
//! The Python match_checkpoints example schedules paired openings and aggregates.

use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use odysseus::neural::{NeuralEvaluator, default_python};
use penteconter::{Color, DrawReason, Game, GameOutcome};
use pyxis::{Adjudication, Evaluator, SearchReport, adjudicate, search};
use serde_json::{Value, json};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const USAGE: &str = "Usage: match_neural --white PATH --black PATH --output NEW.jsonl [options]
  --python PATH       Interpreter (default: workspace .venv)
  --moves 'e2e4 ...'   Replay opening UCI moves, preserving history
  --fen FEN           Initial position before opening moves (default: standard start)
  --simulations N     Fixed simulations per move (default: 32, positive)
  --max-plies N       Maximum additional plies after the opening (default: 1024)
  --help              Show this help

CPU FP32, one thread per checkpoint worker. Fresh search tree per turn; PUCT=1.
Greedy visits, stable first-in-order ties, no root noise, no resignation.
Pyxis adjudication (including threefold and 150 halfmoves) precedes the ply cap.
A capped game is unfinished. On errors, partial logs have no result event.";

struct Options {
    python: PathBuf,
    white: PathBuf,
    black: PathBuf,
    output: PathBuf,
    fen: String,
    opening: Vec<String>,
    simulations: u32,
    max_plies: u32,
}

impl Options {
    fn parse() -> Result<Option<Self>, Box<dyn Error>> {
        let mut args = std::env::args().skip(1);
        let mut python = default_python().into();
        let (mut white, mut black, mut output) = (None, None, None);
        let mut fen = START.to_owned();
        let mut opening = Vec::new();
        let (mut simulations, mut max_plies) = (32, 1024);
        while let Some(flag) = args.next() {
            if flag == "--help" {
                println!("{USAGE}");
                return Ok(None);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--python" => python = value.into(),
                "--white" => white = Some(PathBuf::from(value)),
                "--black" => black = Some(PathBuf::from(value)),
                "--output" => output = Some(PathBuf::from(value)),
                "--fen" => fen = value,
                "--moves" => opening = value.split_whitespace().map(str::to_owned).collect(),
                "--simulations" => simulations = value.parse()?,
                "--max-plies" => max_plies = value.parse()?,
                _ => return Err(format!("unknown option {flag}; use --help").into()),
            }
        }
        if simulations == 0 {
            return Err("--simulations must be positive".into());
        }
        Ok(Some(Self {
            python,
            white: white.ok_or("--white required")?,
            black: black.ok_or("--black required")?,
            output: output.ok_or("--output required")?,
            fen,
            opening,
            simulations,
            max_plies,
        }))
    }
}

fn replay(fen: &str, opening: &[String]) -> Result<Game, Box<dyn Error>> {
    let mut game = Game::new(fen.parse()?);
    for text in opening {
        if adjudicate(&game).is_some() {
            return Err("opening continues beyond an adjudicated outcome".into());
        }
        let mut legal = Vec::new();
        game.position().generate_legal_moves(&mut legal);
        let mv = legal
            .into_iter()
            .find(|mv| mv.to_string() == *text)
            .ok_or_else(|| format!("illegal opening move {text}"))?;
        game.play(mv)?;
    }
    Ok(game)
}

fn color(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn outcome(result: Option<Adjudication>) -> (&'static str, Option<&'static str>) {
    match result {
        None => ("unfinished_ply_cap", None),
        Some(Adjudication::ThreefoldRepetitionDraw) => ("search_policy_threefold", None),
        Some(Adjudication::Automatic(GameOutcome::Checkmate { winner })) => {
            ("automatic_checkmate", Some(color(winner)))
        }
        Some(Adjudication::Automatic(GameOutcome::Draw { reason })) => (
            match reason {
                DrawReason::Stalemate => "automatic_stalemate",
                DrawReason::InsufficientMaterial => "automatic_insufficient_material",
                DrawReason::FivefoldRepetition => "automatic_fivefold_repetition",
                DrawReason::SeventyFiveMoveRule => "automatic_seventy_five_move_rule",
            },
            None,
        ),
    }
}

fn emit(output: &mut impl Write, event: Value) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(&mut *output, &event)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn play<E: Evaluator>(
    game: &mut Game,
    white: &mut E,
    black: &mut E,
    simulations: u32,
    max_plies: u32,
    output: &mut impl Write,
) -> Result<(), Box<dyn Error>>
where
    E::Error: 'static,
{
    let started = Instant::now();
    for ply in 0..=max_plies {
        let adjudication = adjudicate(game);
        if adjudication.is_some() || ply == max_plies {
            let (reason, winner) = outcome(adjudication);
            emit(
                output,
                json!({
                    "event": "result", "completed": adjudication.is_some(),
                    "reason": reason, "winner": winner, "played_plies": ply,
                    "final_fen": game.position().to_fen(), "repetition_count": game.repetition_count(),
                    "elapsed_seconds": started.elapsed().as_secs_f64(),
                }),
            )?;
            return Ok(());
        }
        let side = game.position().side_to_move();
        // Select once at the real root. Every descendant uses this same model,
        // even when search temporarily changes the side to move.
        let evaluator = match side {
            Color::White => &mut *white,
            Color::Black => &mut *black,
        };
        let search_started = Instant::now();
        let SearchReport::Nonterminal {
            best_move,
            simulations: count,
            moves,
        } = search(game, evaluator, simulations, 1.0)?
        else {
            return Err("nonterminal root produced a terminal search report".into());
        };
        let search_seconds = search_started.elapsed().as_secs_f64();
        let root: Vec<_> = moves
            .iter()
            .map(|entry| {
                json!({
                    "move": entry.mv.to_string(), "visits": entry.stats.visits(),
                    "prior": entry.stats.prior(), "q": entry.stats.mean_value(),
                })
            })
            .collect();
        game.play(best_move)?;
        emit(
            output,
            json!({
                "event": "move", "ply": ply, "side": color(side),
                "move": best_move.to_string(), "simulations": count,
                "search_seconds": search_seconds, "root": root,
                "fen_after": game.position().to_fen(), "repetition_count": game.repetition_count(),
            }),
        )?;
    }
    unreachable!()
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(options) = Options::parse()? else {
        return Ok(());
    };
    let mut game = replay(&options.fen, &options.opening)?;
    let mut output = BufWriter::new(File::create_new(&options.output)?);
    emit(
        &mut output,
        json!({
            "event": "start", "format": "odysseus.match", "version": 1,
            "white_checkpoint": options.white, "black_checkpoint": options.black,
            "python": options.python, "device": "cpu", "dtype": "float32", "threads_per_worker": 1,
            "initial_fen": options.fen, "opening_moves": options.opening,
            "starting_fen": game.position().to_fen(), "repetition_count": game.repetition_count(),
            "simulations_per_move": options.simulations, "max_additional_plies": options.max_plies,
            "exploration": 1.0, "temperature": 0, "root_noise": false, "tree_reuse": false,
            "tie_break": "first_in_legal_move_order", "resignations": false,
        }),
    )?;
    let mut white = NeuralEvaluator::new(&options.python, Some(options.white));
    let mut black = NeuralEvaluator::new(&options.python, Some(options.black));
    play(
        &mut game,
        &mut white,
        &mut black,
        options.simulations,
        options.max_plies,
        &mut output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use penteconter::Move;
    use pyxis::{Evaluation, UniformEvaluator};
    use std::convert::Infallible;

    #[derive(Default)]
    struct Spy(Vec<Color>);
    impl Evaluator for Spy {
        type Error = Infallible;
        fn evaluate(&mut self, game: &Game, legal: &[Move]) -> Result<Evaluation, Infallible> {
            self.0.push(game.position().side_to_move());
            UniformEvaluator.evaluate(game, legal)
        }
    }

    #[test]
    fn each_player_owns_the_entire_search_and_only_real_moves_remain() {
        let mut game = replay(START, &[]).unwrap();
        let (mut white, mut black) = (Spy::default(), Spy::default());
        let mut bytes = Vec::new();
        play(&mut game, &mut white, &mut black, 1, 2, &mut bytes).unwrap();
        assert_eq!(white.0, [Color::White, Color::Black]);
        assert_eq!(black.0, [Color::Black, Color::White]);
        assert!(game.undo().is_some());
        assert!(game.undo().is_some());
        assert!(game.undo().is_none());
        let events: Vec<Value> = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events[2]["completed"], false);
        assert_eq!(events[2]["winner"], Value::Null);
    }

    #[test]
    fn repetition_history_is_retained_and_terminal_precedes_zero_cap() {
        let opening = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8";
        let moves: Vec<_> = opening.split_whitespace().map(str::to_owned).collect();
        let mut game = replay(START, &moves).unwrap();
        assert_eq!(game.repetition_count(), 3);
        let (mut white, mut black) = (Spy::default(), Spy::default());
        let mut output = Vec::new();
        play(&mut game, &mut white, &mut black, 1, 0, &mut output).unwrap();
        assert!(white.0.is_empty() && black.0.is_empty());
        let result: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(result["completed"], true);
        assert_eq!(result["reason"], "search_policy_threefold");
    }
}
