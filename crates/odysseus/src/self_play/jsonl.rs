//! Version 1 interchange: one completed game per JSON line.
//!
//! Inputs are square-major [64,110] FP32 values; sparse policy entries contain
//! every legal slot, including zero visits. W/D/L labels use each root's player.
//! Encoding/vocabulary identifiers describe semantics, not just tensor shapes;
//! changing channel meanings or slot ordering requires a new identifier.
//! This inspectable baseline is not a compact replay buffer format.

use std::io::{self, Write};

use penteconter::{Color, DrawReason, GameOutcome};
use pyxis::Adjudication;
use serde::Serialize;

use super::CompletedGame;

pub const FORMAT: &str = "odysseus.self_play";
pub const VERSION: u32 = 1;
pub const INPUT_ENCODING: &str = "pyxis-110-v1";
pub const POLICY_VOCABULARY: &str = "lc0-1858-v1";

#[derive(Serialize)]
struct GameRecord<'a> {
    format: &'static str,
    version: u32,
    input_encoding: &'static str,
    policy_vocabulary: &'static str,
    adjudication: Outcome,
    examples: Vec<Example<'a>>,
}

#[derive(Serialize)]
struct Example<'a> {
    ply: usize,
    side_to_move: &'static str,
    input: Vec<&'a [f32]>,
    policy: Vec<Policy>,
    total_visits: u64,
    value_target: [f32; 3],
}

#[derive(Serialize)]
struct Policy {
    index: usize,
    visits: u32,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Outcome {
    AutomaticCheckmate { winner: &'static str },
    AutomaticStalemate,
    AutomaticInsufficientMaterial,
    AutomaticFivefoldRepetition,
    AutomaticSeventyFiveMoveRule,
    SelfPlayThreefoldRepetition,
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

/// Appends one completed game and a newline to the supplied writer.
///
/// Accepts no unfinished Recorder/SelfPlayOutcome. The caller supplies a trusted
/// CompletedGame; this does not replay moves to prove its provenance. Public
/// outcome labels are checked for consistency before writing any bytes. Empty
/// completed games (terminal starting positions) are valid and yield no examples.
/// On I/O failure the writer may contain a partial line; discard/repair that line
/// before appending more games. Flush/durability and file ownership are caller work.
pub fn write_completed_game(writer: impl Write, game: &CompletedGame) -> io::Result<()> {
    let (adjudication, winner) = match game.adjudication {
        Adjudication::Automatic(GameOutcome::Checkmate { winner }) => (
            Outcome::AutomaticCheckmate {
                winner: color_name(winner),
            },
            Some(winner),
        ),
        Adjudication::Automatic(GameOutcome::Draw { reason }) => (
            match reason {
                DrawReason::Stalemate => Outcome::AutomaticStalemate,
                DrawReason::InsufficientMaterial => Outcome::AutomaticInsufficientMaterial,
                DrawReason::FivefoldRepetition => Outcome::AutomaticFivefoldRepetition,
                DrawReason::SeventyFiveMoveRule => Outcome::AutomaticSeventyFiveMoveRule,
            },
            None,
        ),
        Adjudication::ThreefoldRepetitionDraw => (Outcome::SelfPlayThreefoldRepetition, None),
    };
    let mut examples = Vec::with_capacity(game.examples.len());
    let mut previous = None;
    for example in &game.examples {
        let root = &example.root;
        let expected = match winner {
            Some(color) if color == root.side_to_move() => [1.0, 0.0, 0.0],
            Some(_) => [0.0, 0.0, 1.0],
            None => [0.0, 1.0, 0.0],
        };
        if example.value_target != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "W/D/L label disagrees with the completed game's adjudication",
            ));
        }
        if let Some((ply, side)) = previous
            && (root.ply() <= ply || (root.ply() % 2 == ply % 2) != (root.side_to_move() == side))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "recorded plies must increase with consistent side-to-move parity",
            ));
        }
        previous = Some((root.ply(), root.side_to_move()));
        examples.push(Example {
            ply: root.ply(),
            side_to_move: color_name(root.side_to_move()),
            input: root.input().iter().map(|row| row.as_slice()).collect(),
            policy: root
                .policy()
                .iter()
                .map(|entry| Policy {
                    index: entry.index.index(),
                    visits: entry.visits,
                })
                .collect(),
            total_visits: root.total_visits(),
            value_target: example.value_target,
        });
    }
    let record = GameRecord {
        format: FORMAT,
        version: VERSION,
        input_encoding: INPUT_ENCODING,
        policy_vocabulary: POLICY_VOCABULARY,
        adjudication,
        examples,
    };
    let mut writer = writer;
    serde_json::to_writer(&mut writer, &record).map_err(|error| {
        io::Error::new(
            error.io_error_kind().unwrap_or(io::ErrorKind::InvalidData),
            error,
        )
    })?;
    writer.write_all(b"\n")
}
