use penteconter::{Color, DrawReason, Game, GameOutcome, Move, MoveKind, Square};
use pyxis::{Adjudication, Node, UniformEvaluator, adjudicate, resolve_node};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn normal(s: &str) -> Move {
    let b = s.as_bytes();
    Move::new(
        Square::from_coords(b[0] - b'a', b[1] - b'1').unwrap(),
        Square::from_coords(b[2] - b'a', b[3] - b'1').unwrap(),
        MoveKind::Normal,
    )
    .unwrap()
}

#[test]
fn adjudication_preserves_automatic_outcomes_and_exact_occurrence_counts() {
    let mut game = Game::new(START.parse().unwrap());
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"].map(normal);
    for n in 1..=5 {
        assert_eq!(game.repetition_count(), n);
        let expected = match n {
            1 | 2 => None,
            3 | 4 => Some(Adjudication::ThreefoldRepetitionDraw),
            _ => Some(Adjudication::Automatic(GameOutcome::Draw {
                reason: DrawReason::FivefoldRepetition,
            })),
        };
        let position = *game.position();
        assert_eq!(adjudicate(&game), expected);
        assert_eq!(*game.position(), position);
        assert_eq!(game.repetition_count(), n);
        assert_eq!(game.outcome().is_some(), n == 5);
        if n < 5 {
            // Core play intentionally remains allowed after adjudication.
            for mv in cycle {
                game.play(mv).unwrap();
            }
        }
    }
    for _ in 0..12 {
        game.undo().unwrap();
    }
    assert_eq!(game.repetition_count(), 2);
    assert_eq!(adjudicate(&game), None);
    // A FEN supplies no record of earlier occurrences.
    assert_eq!(adjudicate(&Game::new(*game.position())), None);
}

#[test]
fn move_clock_policy_still_uses_150_halfmoves_and_preserves_mate_precedence() {
    for clock in [99, 100, 149, 150, 151] {
        let fen = format!("4k3/8/8/8/8/8/8/R3K3 w - - {clock} 76");
        let game = Game::new(fen.parse().unwrap());
        let expected = (clock >= 150).then_some(Adjudication::Automatic(GameOutcome::Draw {
            reason: DrawReason::SeventyFiveMoveRule,
        }));
        assert_eq!(adjudicate(&game), expected);
        assert_eq!(
            matches!(
                resolve_node(&game, &mut UniformEvaluator).unwrap(),
                Node::Terminal(_)
            ),
            clock >= 150,
        );
    }
    let game = Game::new("k7/1Q6/2K5/8/8/8/8/8 b - - 150 76".parse().unwrap());
    assert_eq!(
        adjudicate(&game),
        Some(Adjudication::Automatic(GameOutcome::Checkmate {
            winner: Color::White,
        }))
    );
}

#[test]
fn automatic_draw_reasons_take_precedence_over_threefold_adjudication() {
    for (fen, cycle, reason) in [
        (
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            ["e1e2", "e8e7", "e2e1", "e7e8"],
            DrawReason::InsufficientMaterial,
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 142 72",
            ["g1f3", "g8f6", "f3g1", "f6g8"],
            DrawReason::SeventyFiveMoveRule,
        ),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        for s in cycle.into_iter().cycle().take(8) {
            game.play(normal(s)).unwrap();
        }
        assert_eq!(game.repetition_count(), 3);
        assert_eq!(
            adjudicate(&game),
            Some(Adjudication::Automatic(GameOutcome::Draw { reason }))
        );
    }
}
