//! Round-trip real Pyxis encodings through one persistent Python model.
//! Usage: inference_bridge PYTHON [DUMP_FILE [CHECKPOINT]]
//! Optional dump: consecutive little-endian f32 records, each input then P/WDL.

use std::{error::Error, fs::File, io::Write, process::Command, time::Duration};

use odysseus::inference::InferenceWorker;
use penteconter::Game;
use pyxis::encoding::encode;

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn play(game: &mut Game, coordinate: &str) -> Result<(), Box<dyn Error>> {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let mv = moves
        .into_iter()
        .find(|mv| mv.to_string() == coordinate)
        .ok_or_else(|| format!("illegal example move {coordinate}"))?;
    game.play(mv)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let python = args
        .next()
        .ok_or("usage: inference_bridge PYTHON [DUMP_FILE [CHECKPOINT]]")?;
    let dump_path = args.next();
    let checkpoint = args.next();
    if args.next().is_some() {
        return Err("too many arguments".into());
    }

    let mut game = Game::new(START.parse()?);
    let mut fixtures = vec![("Start position, White", encode(&game))];
    play(&mut game, "g1f3")?;
    fixtures.push(("After Nf3, Black with history", encode(&game)));
    for mv in ["g8f6", "f3g1", "f6g8"] {
        play(&mut game, mv)?;
    }
    fixtures.push(("Second occurrence of start position", encode(&game)));
    fixtures.push((
        "Black: castling rights and legal EP",
        encode(&Game::new(
            "r3k2r/8/8/8/3Pp3/8/8/R3K2R b Kq d3 0 1".parse()?,
        )),
    ));
    fixtures.push((
        "White: promotion available",
        encode(&Game::new("4k3/P7/8/8/8/8/8/4K3 w - - 0 1".parse()?)),
    ));
    fixtures.push(("Repeat first request", fixtures[0].1));

    let mut command = Command::new(python);
    command.args([
        "-m",
        "neurodiktyon.inference_worker",
        "--seed",
        "0",
        "--threads",
        "1",
    ]);
    if let Some(path) = &checkpoint {
        command.arg("--checkpoint").arg(path);
    }
    let mut worker = InferenceWorker::spawn(&mut command, Duration::from_secs(30))?;
    let mut dump = dump_path.map(File::create).transpose()?;
    let mut first = None;
    let count = fixtures.len();
    if let Some(path) = &checkpoint {
        println!(
            "One persistent CPU FP32 worker; checkpoint {}.",
            path.to_string_lossy()
        );
    } else {
        println!("One persistent CPU FP32 worker; seed 0; untrained default ChessModel.");
    }
    println!("Each request: [1,64,110]; response: [1,1858] policy + [1,3] W/D/L logits.");
    for (index, (label, input)) in fixtures.into_iter().enumerate() {
        let prediction = worker.infer(&input)?;
        println!(
            "{index}: {label}; raw W/D/L = {:?}",
            prediction.value_logits
        );
        if let Some(file) = &mut dump {
            for value in input
                .iter()
                .flatten()
                .chain(&prediction.policy_logits)
                .chain(&prediction.value_logits)
            {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        if index == count - 1 {
            if first.as_ref() != Some(&prediction) {
                return Err("repeated input changed predictions".into());
            }
            println!("Repeated start-position request exactly matches its first response.");
        }
        if index == 0 {
            first = Some(prediction);
        }
    }
    Ok(())
}
