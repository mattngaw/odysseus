use std::error::Error;

use penteconter::Position;
use pyxis::vocabulary::{POLICY_SIZE, PolicyIndex, index_for_move, move_for_index};

fn inspect(fen: &str, selected: &[&str]) -> Result<(), Box<dyn Error>> {
    let position: Position = fen.parse()?;
    let side = position.side_to_move();
    let mut legal = Vec::new();
    position.generate_legal_moves(&mut legal);
    println!("\nFEN: {fen}");
    println!("Perspective: {side:?}; {} legal moves", legal.len());
    println!("Absolute move  Kind                 Index  Relative entry  Decoded move");
    let mut printed = 0;
    for &mv in &legal {
        let coordinate = mv.to_string();
        if !selected.is_empty() && !selected.contains(&coordinate.as_str()) {
            continue;
        }
        let index = index_for_move(mv, side).expect("legal move has a policy slot");
        let decoded = move_for_index(index, side, &legal).expect("slot is in legal list");
        assert_eq!(decoded, mv);
        println!(
            "{coordinate:<13}  {:<19}  {:>5}  {:<14}  {decoded}",
            format!("{:?}", mv.kind()),
            index.index(),
            index.entry().to_string()
        );
        printed += 1;
    }
    if !selected.is_empty() {
        assert_eq!(
            printed,
            selected.len(),
            "all requested demo moves are legal"
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("expected no arguments, --dump, --help, or one quoted FEN".into());
    }
    if let Some(arg) = args.first() {
        match arg.as_str() {
            "--help" => {
                println!("Usage: policy_vocabulary [--dump | \"FEN\"]");
                println!("No arguments: selected legal moves illustrating each convention.");
                println!("FEN: every legal move and its index, entry, and decoded move.");
                println!("--dump: all 1,858 relative entry labels, one per line, in index order.");
            }
            "--dump" => {
                for index in 0..POLICY_SIZE {
                    println!("{}", PolicyIndex::new(index).unwrap().entry());
                }
            }
            fen => inspect(fen, &[])?,
        }
        return Ok(());
    }
    println!("Fixed policy vocabulary: 1,792 base slots + 66 promotion slots.");
    println!("Black flips ranks, preserving files. Castling entries point to the rook.");
    println!("Knight promotions use base slots; queen/rook/bishop use appended slots.");
    println!("Decoding matches the supplied legal moves, preserving the move kind.");
    let cases: &[(&str, &[&str])] = &[
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["e2e4", "g1f3"],
        ),
        (
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            &["e7e5", "g8f6"],
        ),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", &["e1g1", "e1c1"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", &["e8g8", "e8c8"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", &["e5d6"]),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", &["e4d3"]),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            &["a7a8n", "a7a8q", "a7a8r", "a7a8b"],
        ),
        (
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
            &["a2a1n", "a2a1q", "a2a1r", "a2a1b"],
        ),
    ];
    for &(fen, selected) in cases {
        inspect(fen, selected)?;
    }
    Ok(())
}
