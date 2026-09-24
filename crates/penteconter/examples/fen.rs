use penteconter::{FenError, Position};

fn main() -> Result<(), FenError> {
    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let position: Position = fen.parse()?;
    println!("{position}\n\nFEN: {}", position.to_fen());
    Ok(())
}
