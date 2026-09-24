use penteconter::Game;
use pyxis::{
    Evaluation, Evaluator, MaterialEvaluator, Node, SearchReport, UniformEvaluator, resolve_node,
    search,
};

fn evaluate(game: &Game) -> Evaluation {
    assert_eq!(game.outcome(), None);
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let original = *game.position();
    let repetitions = game.repetition_count();
    let result = MaterialEvaluator.evaluate(game, &moves).unwrap();
    assert_eq!(result.policy_weights, vec![1.0; moves.len()]);
    assert_eq!(*game.position(), original);
    assert_eq!(game.repetition_count(), repetitions);
    result
}

#[test]
fn piece_values_and_turn_perspective_match_the_agreed_mapping() {
    for (piece, expected) in [
        ('P', 1.0 / 6.0),
        ('N', 3.0 / 8.0),
        ('B', 3.0 / 8.0),
        ('R', 0.5),
        ('Q', 9.0 / 14.0),
    ] {
        // Balanced a-pawns keep the minor-piece cases nonterminal.
        for (side, sign) in [("w", 1.0), ("b", -1.0)] {
            let game = Game::new(
                format!("4k3/p7/8/8/8/8/P6{piece}/4K3 {side} - - 0 1")
                    .parse()
                    .unwrap(),
            );
            assert!((evaluate(&game).value.get() - sign * expected).abs() < 1e-7);
        }
    }
    let black_advantage = Game::new("4k3/p6q/8/8/8/8/P7/4K3 w - - 0 1".parse().unwrap());
    assert!((evaluate(&black_advantage).value.get() + 9.0 / 14.0).abs() < 1e-7);
}

#[test]
fn equal_material_is_neutral_and_multiple_queens_all_count() {
    let equal = Game::new(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            .parse()
            .unwrap(),
    );
    assert_eq!(evaluate(&equal).value.get(), 0.0);
    let promoted = Game::new("4k3/p7/8/8/8/7Q/P6Q/4K3 w - - 0 1".parse().unwrap());
    let value = evaluate(&promoted).value.get();
    assert!((value - 18.0 / 23.0).abs() < 1e-7);
    assert!(value > 0.0 && value < 1.0);
}

#[test]
fn queen_capture_changes_material_and_backup_restores_the_parent_perspective() {
    use pyxis::Tree;
    for (fen, capture) in [
        ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5"),
        ("3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1", "d8d4"),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let original = *game.position();
        assert!((evaluate(&game).value.get() + 4.0 / 9.0).abs() < 1e-7);
        let root = resolve_node(&game, &mut MaterialEvaluator).unwrap();
        let Node::Expanded(node) = &root else {
            panic!()
        };
        let (index, edge) = node
            .edges()
            .iter()
            .enumerate()
            .find(|(_, edge)| edge.mv().to_string() == capture)
            .unwrap();
        let mv = edge.mv();
        let mut tree = Tree::new(root);
        // An explicit one-edge sample isolates the backup sign from selection.
        game.play(mv).unwrap();
        let Node::Expanded(child) = resolve_node(&game, &mut MaterialEvaluator).unwrap() else {
            panic!()
        };
        assert_eq!(child.value().get(), -0.5);
        let sample = child.value();
        tree.add_child(tree.root(), index, Node::Expanded(child))
            .unwrap();
        tree.backup(&[(tree.root(), index)], sample).unwrap();
        game.undo().unwrap();
        let SearchReport::Nonterminal { moves, .. } = tree.report() else {
            panic!()
        };
        assert_eq!(moves[index].stats.mean_value(), 0.5);
        assert_eq!(*game.position(), original);
    }
}

#[test]
fn exact_outcomes_override_material_advantage_or_deficit() {
    for (fen, expected) in [
        ("k7/1Q6/2K5/8/8/8/8/8 b - - 0 1", -1.0),
        ("k7/8/1QK5/8/8/8/8/8 b - - 0 1", 0.0),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 150 76", 0.0),
    ] {
        let game = Game::new(fen.parse().unwrap());
        let Node::Terminal(value) = resolve_node(&game, &mut MaterialEvaluator).unwrap() else {
            panic!()
        };
        assert_eq!(value.get(), expected);
    }
}

#[test]
fn material_search_prefers_the_hanging_queen_with_uniform_priors() {
    for (fen, capture) in [
        ("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5"),
        ("3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1", "d8d4"),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let original = *game.position();
        let SearchReport::Nonterminal {
            best_move, moves, ..
        } = search(&mut game, &mut MaterialEvaluator, 128, 1.0).unwrap()
        else {
            panic!()
        };
        assert_eq!(best_move.to_string(), capture);
        let best = moves.iter().find(|entry| entry.mv == best_move).unwrap();
        assert!(best.stats.mean_value() > 0.0);
        assert!(best.stats.visits() > 64);
        let SearchReport::Nonterminal { moves: uniform, .. } =
            search(&mut game, &mut UniformEvaluator, 128, 1.0).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            moves
                .iter()
                .map(|m| (m.mv, m.stats.prior()))
                .collect::<Vec<_>>(),
            uniform
                .iter()
                .map(|m| (m.mv, m.stats.prior()))
                .collect::<Vec<_>>()
        );
        assert!(
            best.stats.visits()
                > uniform
                    .iter()
                    .find(|entry| entry.mv == best_move)
                    .unwrap()
                    .stats
                    .visits()
        );
        assert_eq!(*game.position(), original);
        assert_eq!(game.undo(), None);
    }
}
