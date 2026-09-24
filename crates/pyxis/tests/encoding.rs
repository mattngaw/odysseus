use penteconter::{Color, Game, MoveKind, Position, Square};
use pyxis::encoding::{encode, encode_into, relative_square};
use pyxis::{SimulationStep, Tree, UniformEvaluator, resolve_node};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn sq(s: &str) -> Square {
    let b = s.as_bytes();
    Square::from_coords(b[0] - b'a', b[1] - b'1').unwrap()
}

fn play(game: &mut Game, s: &str) {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let mv = moves.into_iter().find(|mv| mv.to_string() == s).unwrap();
    game.play(mv).unwrap();
}

#[test]
fn coordinates_preserve_files_and_flip_only_black_ranks() {
    for i in 0..64 {
        let square = Square::new(i).unwrap();
        assert_eq!(relative_square(square, Color::White), square);
        let flipped = relative_square(square, Color::Black);
        assert_eq!(flipped.file(), square.file());
        assert_eq!(flipped.rank(), 7 - square.rank());
        assert_eq!(relative_square(flipped, Color::Black), square);
    }
    assert_eq!(relative_square(sq("a8"), Color::Black), sq("a1"));
    assert_eq!(relative_square(sq("e7"), Color::Black), sq("e2"));
}

#[test]
fn all_history_pieces_use_the_current_players_coordinates_and_ownership() {
    let mut game = Game::new(START.parse().unwrap());
    let mut snapshots = vec![*game.position()];
    for mv in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        play(&mut game, mv);
        snapshots.push(*game.position());
        let output = encode(&game);
        // Explicit shape and numeric channel assertions lock the wire format.
        assert_eq!(output.len(), 64);
        assert_eq!(output[0].len(), 110);
        let black = game.position().side_to_move() == Color::Black;
        for (t, position) in snapshots.iter().rev().enumerate() {
            for index in 0..64u8 {
                let square = Square::new(index).unwrap();
                let absolute = Square::from_coords(
                    square.file(),
                    if black {
                        7 - square.rank()
                    } else {
                        square.rank()
                    },
                )
                .unwrap();
                // Mailbox oracle checks all 12 channels, including empty squares.
                let piece = position.board().piece_at(absolute);
                for channel in 0..12 {
                    let expected = piece.is_some_and(|p| {
                        let theirs = p.color() != game.position().side_to_move();
                        channel == usize::from(theirs) * 6 + p.kind().index()
                    });
                    assert_eq!(
                        output[index as usize][13 * t + channel],
                        if expected { 1.0 } else { 0.0 }
                    );
                }
            }
        }
        assert!(
            output
                .iter()
                .all(|token| token[snapshots.len() * 13..104].iter().all(|x| *x == 0.0))
        );
        if snapshots.len() == 4 {
            // Black to move after Nf3.
            assert_eq!(output[sq("f6").index()][7], 1.0); // Their current knight.
            assert_eq!(output[sq("g8").index()][13 + 7], 1.0); // Same knight one ply ago.
            assert_eq!(output[sq("e2").index()][26], 1.0); // Our pawn before ...e5.
            assert_eq!(output[sq("e1").index()][39 + 5], 1.0); // Our king in initial frame.
        }
    }
}

#[test]
fn repeated_frames_use_prefix_counts_and_only_the_latest_eight_positions() {
    let mut game = Game::new(START.parse().unwrap());
    for s in ["g1f3", "g8f6", "f3g1", "f6g8"].into_iter().cycle().take(8) {
        play(&mut game, s);
    }
    // Terminal games remain encodable for inspection, though search bypasses NN.
    assert_eq!(game.repetition_count(), 3);
    let output = encode(&game);
    for token in &output {
        let flags: Vec<_> = (0..8).map(|t| token[13 * t + 12]).collect();
        assert_eq!(flags, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }
    // Oldest retained frame is after Nf3, not the supplied starting position.
    assert_eq!(output[sq("f3").index()][7 * 13 + 1], 1.0);
    assert_eq!(output[sq("g1").index()][7 * 13 + 1], 0.0);
    let before = encode(&game);
    game.undo().unwrap();
    play(&mut game, "f6g8");
    assert_eq!(encode(&game), before);
}

#[test]
fn reuse_clears_old_history_and_fen_counters_do_not_fabricate_frames() {
    let mut game = Game::new(START.parse().unwrap());
    for s in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        play(&mut game, s);
    }
    let mut output = encode(&game);
    let fen_only = Game::new("4k3/8/8/8/8/8/8/R3K3 b - - 100 1000".parse().unwrap());
    encode_into(&fen_only, &mut output);
    assert_eq!(output, encode(&fen_only));
    assert!(
        output
            .iter()
            .all(|token| token[12..109].iter().all(|x| *x == 0.0))
    );
    assert!(output.iter().all(|token| token[109] == 100.0 / 150.0));
}

#[test]
fn castling_flags_are_current_relative_rights_even_with_blocked_paths() {
    for side in ["w", "b"] {
        for mask in 0..16 {
            let mut rights = String::new();
            for (i, letter) in ['K', 'Q', 'k', 'q'].into_iter().enumerate() {
                if mask & (1 << i) != 0 {
                    rights.push(letter);
                }
            }
            if rights.is_empty() {
                rights.push('-');
            }
            let fen = START.replace("w KQkq", &format!("{side} {rights}"));
            let game = Game::new(fen.parse().unwrap());
            let output = encode(&game);
            let order = if side == "w" {
                [0, 1, 2, 3]
            } else {
                [2, 3, 0, 1]
            };
            for token in &output {
                for (channel, bit) in order.into_iter().enumerate() {
                    assert_eq!(
                        token[104 + channel],
                        if mask & (1 << bit) != 0 { 1.0 } else { 0.0 }
                    );
                }
            }
        }
    }
}

#[test]
fn clock_is_broadcast_scaled_by_150_and_clamped_without_affecting_other_features() {
    for clock in [0, 1, 75, 100, 149, 150, 151, u32::MAX] {
        let game = Game::new(
            format!("4k3/8/8/8/8/8/8/R3K3 w - - {clock} 1000")
                .parse()
                .unwrap(),
        );
        let output = encode(&game);
        for token in &output {
            assert_eq!(token[109], clock.min(150) as f32 / 150.0);
            assert!(
                token
                    .iter()
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
            );
        }
    }
}

#[test]
fn en_passant_marks_the_legal_target_with_pins_and_check_evasions_handled() {
    for (fen, target) in [
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", Some("d6")),
        ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", Some("d6")),
        ("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1", None),
        ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", None),
        ("4k3/8/8/8/3Pp3/8/8/K3R3 b - d3 0 1", None),
        ("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1", None),
        ("k3r3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1", None),
        ("4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1", Some("d6")),
        ("kb6/8/8/2Pp4/8/8/7K/8 w - d6 0 1", Some("d6")),
        ("k1r5/8/8/2PpP3/8/8/8/2K5 w - d6 0 1", Some("d6")),
        ("4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1", Some("d6")),
    ] {
        let original: Position = fen.parse().unwrap();
        let mut game = Game::new(original);
        let output = encode(&game);
        let targets: Vec<_> = (0..64).filter(|i| output[*i][108] == 1.0).collect();
        assert_eq!(
            targets,
            target.map(|s| vec![sq(s).index()]).unwrap_or_default(),
            "{fen}"
        );
        let mut moves = Vec::new();
        original.generate_legal_moves(&mut moves);
        assert_eq!(
            original.legal_en_passant_target(),
            moves
                .iter()
                .find(|mv| mv.kind() == MoveKind::EnPassant)
                .map(|mv| mv.to())
        );
        assert_eq!(*game.position(), original);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn pending_leaf_encoding_includes_search_history_and_cancellation_restores_input() {
    let mut game = Game::new(START.parse().unwrap());
    for s in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        play(&mut game, s);
    }
    let before = encode(&game);
    let root = *game.position();
    let mut tree = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap());
    let SimulationStep::NeedsEvaluation(pending) = tree.begin_simulation(&mut game, 1.0).unwrap()
    else {
        panic!("first simulation should request a nonterminal leaf");
    };
    let leaf = pending.game();
    let frames: Vec<_> = leaf.positions().rev().collect();
    assert_eq!(frames.len(), 6);
    assert_eq!(*frames[1].position, root);
    assert_eq!(frames[1].repetition_count, 2);
    let input = encode(leaf);
    assert_eq!(leaf.position().side_to_move(), Color::Black);
    for token in &input {
        assert_eq!(token[13 + 12], 1.0);
        assert_eq!(token[5 * 13 + 12], 0.0);
    }
    drop(pending);
    assert_eq!(encode(&game), before);
    assert_eq!(game.repetition_count(), 2);
}
