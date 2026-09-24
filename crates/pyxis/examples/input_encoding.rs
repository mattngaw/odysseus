use std::error::Error;

use penteconter::Game;
use pyxis::{
    adjudicate,
    encoding::{EncodedInput, FEATURE_COUNT, encode},
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const LEGAL_EP: &str = "r3k2r/8/8/8/3Pp3/8/8/R3K2R b Kq d3 0 1";

fn play(game: &mut Game, coordinate: &str) -> Result<(), Box<dyn Error>> {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let mv = moves
        .into_iter()
        .find(|mv| mv.to_string() == coordinate)
        .ok_or_else(|| format!("illegal move {coordinate} in {}", game.position().to_fen()))?;
    game.play(mv)?;
    Ok(())
}

fn channel_name(channel: usize) -> String {
    if channel < 104 {
        let frame = channel / 13;
        let feature = channel % 13;
        if feature == 12 {
            return format!("frame {frame}: previously seen (broadcast)");
        }
        let owner = if feature < 6 { "ours" } else { "theirs" };
        let kind = ["pawn", "knight", "bishop", "rook", "queen", "king"][feature % 6];
        format!("frame {frame}: {owner} {kind}")
    } else {
        [
            "our kingside right",
            "our queenside right",
            "their kingside right",
            "their queenside right",
            "legal en passant target",
            "halfmove clock / 150",
        ][channel - 104]
            .into()
    }
}

fn plane(input: &EncodedInput, channel: usize) {
    println!("\nChannel {channel}: {}", channel_name(channel));
    for rank in (0..8).rev() {
        print!("{} ", rank + 1);
        for file in 0..8 {
            print!(" {:5.3}", input[rank * 8 + file][channel]);
        }
        println!();
    }
    println!("      a     b     c     d     e     f     g     h");
}

fn inspect(title: &str, game: &Game, channel: usize) {
    let input = encode(game);
    println!("\n{title}\nFEN: {}", game.position().to_fen());
    println!(
        "Input [64, 110]; all frames oriented for {:?}.",
        game.position().side_to_move()
    );
    println!("Adjudication: {:?}", adjudicate(game));
    println!(
        "Raw EP: {:?}; legal EP (absolute): {:?}",
        game.position().en_passant_target(),
        game.position().legal_en_passant_target()
    );
    println!(
        "Castling [ours K,Q; theirs K,Q]: {:?}; clock: {:.4}",
        &input[0][104..108],
        input[0][109]
    );
    println!("Frame  Channels   Historical occurrences  Repetition feature");
    let mut frames = game.positions().rev();
    for t in 0..8 {
        if let Some(frame) = frames.next() {
            println!(
                "{t:>5}  {:>3}..{:<3}   {:>22}  {:>18.0}",
                t * 13,
                t * 13 + 12,
                frame.repetition_count,
                input[0][t * 13 + 12]
            );
        } else {
            println!("{t:>5}  {:>3}..{:<3}   zero padding", t * 13, t * 13 + 12);
        }
    }
    plane(&input, channel);
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    if let Some(first) = args.next() {
        if first == "--help" {
            println!("Usage: input_encoding [CHANNEL [\"FEN\" [UCI_MOVE ...]]]");
            println!("No arguments: built-in history, legal/pinned EP, and clock examples.");
            println!("With a channel: inspect that feature; omitted FEN uses startpos.");
            println!("Channels 13*t+[0..5 ours PNBRQK, 6..11 theirs, 12 repetition].");
            println!("104..107 castling rights, 108 legal EP, 109 normalized clock.");
            return Ok(());
        }
        let channel: usize = first.parse()?;
        if channel >= FEATURE_COUNT {
            return Err("channel must be in 0..110".into());
        }
        let fen = args.next().unwrap_or_else(|| START.into());
        let mut game = Game::new(fen.parse()?);
        for mv in args {
            play(&mut game, &mv)?;
        }
        inspect("Custom input", &game, channel);
        return Ok(());
    }

    let mut game = Game::new(START.parse()?);
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3"] {
        play(&mut game, mv)?;
    }
    inspect("Black perspective with repeated positions", &game, 1);
    plane(&encode(&game), 27); // Our knights two plies ago, in the same coordinates.
    inspect(
        "Legal EP: absolute d3 becomes relative d6",
        &Game::new(LEGAL_EP.parse()?),
        108,
    );
    inspect(
        "Pinned pawn: raw d6 exists, but EP feature is zero",
        &Game::new("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1".parse()?),
        108,
    );
    inspect(
        "FEN-only history; 100 halfmoves is still nonterminal",
        &Game::new("4k3/8/8/8/8/8/8/R3K3 w - - 100 76".parse()?),
        109,
    );
    Ok(())
}
