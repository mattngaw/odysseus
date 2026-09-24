use penteconter::{
    Position,
    perft::{self, LeafMode},
};

#[test]
fn standard_perft_positions_match_published_counts() {
    // Position data and expected leaf counts:
    // https://www.chessprogramming.org/Perft_Results
    // Position 4's mirrored counterpart checks color symmetry as well.
    let cases: [(&str, &str, &[u64]); 7] = [
        (
            "initial",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[1, 20, 400, 8902, 197281],
        ),
        (
            "kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[1, 48, 2039, 97862],
        ),
        (
            "position 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[1, 14, 191, 2812, 43238],
        ),
        (
            "position 4",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[1, 6, 264, 9467],
        ),
        (
            "position 4 mirrored",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            &[1, 6, 264, 9467],
        ),
        (
            "position 5",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[1, 44, 1486, 62379],
        ),
        (
            "position 6",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[1, 46, 2079, 89890],
        ),
    ];
    for (name, fen, counts) in cases {
        let position: Position = fen.parse().unwrap();
        let before = position;
        for (depth, &expected) in counts.iter().enumerate() {
            let actual = perft::run(&position, depth as u8, LeafMode::Bulk, false).nodes;
            assert_eq!(actual, expected, "{name}, depth {depth}");
            println!("{name}: depth {depth} = {actual}");
        }
        assert_eq!(position, before);
    }
}

#[test]
fn perft_counts_depth_zero_as_one_even_for_terminal_positions() {
    for fen in [
        "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
        "k7/2Q5/2K5/8/8/8/8/8 b - - 0 1",
    ] {
        let position: Position = fen.parse().unwrap();
        assert_eq!(perft::run(&position, 0, LeafMode::Bulk, false).nodes, 1);
        assert_eq!(perft::run(&position, 1, LeafMode::Bulk, false).nodes, 0);
        assert_eq!(perft::run(&position, 2, LeafMode::Bulk, false).nodes, 0);
    }
}
