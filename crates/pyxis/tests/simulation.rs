use std::{collections::VecDeque, io};

use penteconter::{Game, Move, Position};
use pyxis::{
    BeginSimulationError, Edge, Evaluation, Evaluator, ExpandedNode, ExpansionError,
    InvalidPolicyWeight, MaterialEvaluator, Node, NodeId, ResolveError, SimulationError,
    SimulationStep, Tree, UniformEvaluator, Value, resolve_node,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

struct TestEvaluator<F> {
    evaluate: F,
    calls: usize,
}

fn evaluator<F>(evaluate: F) -> TestEvaluator<F>
where
    F: FnMut(&Game, &[Move]) -> Result<Evaluation, io::Error>,
{
    TestEvaluator { evaluate, calls: 0 }
}

impl<F> Evaluator for TestEvaluator<F>
where
    F: FnMut(&Game, &[Move]) -> Result<Evaluation, io::Error>,
{
    type Error = io::Error;

    fn evaluate(&mut self, game: &Game, moves: &[Move]) -> Result<Evaluation, Self::Error> {
        self.calls += 1;
        (self.evaluate)(game, moves)
    }
}

fn choices(tree: &Tree, id: NodeId) -> &ExpandedNode {
    let Node::Expanded(node) = tree.node(id).unwrap() else {
        panic!("expected expansion")
    };
    node
}

fn snapshot(tree: &Tree) -> Vec<(NodeId, Value, Vec<Edge>)> {
    let mut pending = vec![tree.root()];
    let mut result = Vec::new();
    while let Some(id) = pending.pop() {
        match tree.node(id).unwrap() {
            Node::Expanded(node) => {
                pending.extend(node.edges().iter().filter_map(Edge::child));
                result.push((id, node.value(), node.edges().to_vec()));
            }
            Node::Terminal(value) => result.push((id, *value, vec![])),
        }
    }
    assert_eq!(result.len(), tree.node_count());
    result
}

fn play(game: &mut Game, coordinates: &str) -> Move {
    let mut moves = Vec::new();
    game.position().generate_legal_moves(&mut moves);
    let mv = moves
        .into_iter()
        .find(|mv| mv.to_string() == coordinates)
        .unwrap();
    game.play(mv).unwrap();
    mv
}

fn position_after(moves: &[&str]) -> Position {
    let mut game = Game::new(START.parse().unwrap());
    for mv in moves {
        play(&mut game, mv);
    }
    *game.position()
}

fn evaluation(moves: &[Move], priors: &[(&str, f32)], value: f32) -> Evaluation {
    let mut weights = vec![0.0; moves.len()];
    for &(coordinates, weight) in priors {
        let index = moves
            .iter()
            .position(|mv| mv.to_string() == coordinates)
            .unwrap();
        weights[index] = weight;
    }
    Evaluation {
        value: Value::new(value).unwrap(),
        policy_weights: weights,
    }
}

#[test]
fn five_complete_simulations_reproduce_the_worked_paths_and_values() {
    // Embed the paper's synthetic branches into legal chess move sequences.
    // A=a2a3, B=b2b3; Black's reply on either branch is a7a6, then c2c3.
    // Unlike that toy tree, chess has additional legal replies even when P=0.
    // c=2 keeps the illustrated replies preferred for these five simulations;
    // at c=1, Black would switch replies on simulation 5 (tested separately).
    let mut script = VecDeque::from([
        (position_after(&[]), vec![("a2a3", 0.5), ("b2b3", 0.5)], 0.0),
        (position_after(&["a2a3"]), vec![("a7a6", 1.0)], -0.2),
        (position_after(&["b2b3"]), vec![("a7a6", 1.0)], 0.4),
        (position_after(&["a2a3", "a7a6"]), vec![("c2c3", 1.0)], -0.8),
        (position_after(&["b2b3", "a7a6"]), vec![("c2c3", 1.0)], 0.6),
        (
            position_after(&["b2b3", "a7a6", "c2c3"]),
            vec![("b7b6", 1.0)],
            -0.6,
        ),
    ]);
    let mut evaluator = evaluator(|game, moves| {
        let (position, weights, value) = script.pop_front().expect("unexpected evaluation");
        assert_eq!(*game.position(), position);
        Ok(evaluation(moves, &weights, value))
    });
    let mut game = Game::new(START.parse().unwrap());
    let root_position = *game.position();
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    let root = tree.root();
    let a = choices(&tree, root)
        .edges()
        .iter()
        .position(|edge| edge.mv().to_string() == "a2a3")
        .unwrap();
    let b = choices(&tree, root)
        .edges()
        .iter()
        .position(|edge| edge.mv().to_string() == "b2b3")
        .unwrap();
    assert_eq!(evaluator.calls, 1);
    assert!(
        choices(&tree, root)
            .edges()
            .iter()
            .all(|edge| edge.stats().visits() == 0)
    );

    for (simulation, (na, wa, nb, wb)) in [
        (1, 0.2, 0, 0.0),
        (1, 0.2, 1, -0.4),
        (2, -0.6, 1, -0.4),
        (2, -0.6, 2, 0.2),
        (2, -0.6, 3, 0.8),
    ]
    .into_iter()
    .enumerate()
    {
        tree.simulate(&mut game, &mut evaluator, 2.0).unwrap();
        let edges = choices(&tree, root).edges();
        for (index, visits, sum) in [(a, na, wa), (b, nb, wb)] {
            assert_eq!(edges[index].stats().visits(), visits);
            assert!((edges[index].stats().value_sum() - sum).abs() < 1e-6);
            assert_eq!(edges[index].stats().prior(), 0.5);
        }
        assert!(
            edges
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != a && *index != b)
                .all(|(_, edge)| edge.stats().visits() == 0 && edge.child().is_none())
        );
        assert_eq!(
            edges.iter().map(|edge| edge.stats().visits()).sum::<u32>(),
            simulation as u32 + 1
        );
        assert_eq!(tree.node_count(), simulation + 2);
        assert_eq!(evaluator.calls, simulation + 2);
        // The newly created leaf is the greatest ID in an append-only tree.
        let nodes = snapshot(&tree);
        let (_, _, leaf_edges) = nodes.iter().max_by_key(|(id, _, _)| id.index()).unwrap();
        assert!(
            leaf_edges
                .iter()
                .all(|edge| edge.stats().visits() == 0 && edge.child().is_none())
        );
        assert_eq!(*game.position(), root_position);
        assert_eq!(game.repetition_count(), 1);
        assert_eq!(game.undo(), None);
    }
    assert!(script.is_empty());
    let b_child = choices(&tree, root).edges()[b].child().unwrap();
    let reply = choices(&tree, b_child)
        .edges()
        .iter()
        .find(|edge| edge.mv().to_string() == "a7a6")
        .unwrap();
    assert_eq!(reply.stats().visits(), 2);
    assert!((reply.stats().value_sum() + 1.2).abs() < 1e-6);
}

#[test]
fn zero_prior_legal_moves_remain_selectable_when_the_explored_reply_scores_worse() {
    let mut game = Game::new(position_after(&["b2b3"]));
    let root_position = *game.position();
    let mut evaluator = evaluator(|game, moves| {
        if *game.position() == root_position {
            Ok(evaluation(moves, &[("a7a6", 1.0)], 0.4))
        } else {
            Ok(Evaluation {
                value: Value::new(0.6).unwrap(),
                policy_weights: vec![1.0; moves.len()],
            })
        }
    });
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    for _ in 0..2 {
        tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
    }
    // After the first simulation, a7a6 has Q=-0.6 and U=0.5, so a
    // zero-prior unvisited alternative (Q=U=0) wins the second selection.
    let edges = choices(&tree, tree.root()).edges();
    let preferred = edges
        .iter()
        .find(|edge| edge.mv().to_string() == "a7a6")
        .unwrap();
    assert_eq!(preferred.stats().visits(), 1);
    assert_eq!(preferred.stats().value_sum(), -0.6);
    assert_eq!(
        edges
            .iter()
            .filter(|edge| edge.stats().prior() == 0.0 && edge.stats().visits() == 1)
            .count(),
        1
    );
    assert_eq!(tree.node_count(), 3);
    assert_eq!(evaluator.calls, 3);
    assert_eq!(*game.position(), root_position);
    assert_eq!(game.undo(), None);
}

#[test]
fn terminal_children_count_visits_and_are_reused_without_evaluation() {
    for (fen, mv, leaf_value) in [
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6b7", -1.0),
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6c7", 0.0),
        ("4k3/8/8/8/8/8/3r4/2B1K3 w - - 0 1", "c1d2", 0.0),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 149 76", "a1a2", 0.0),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let root_position = *game.position();
        let mut evaluator = evaluator(|evaluated_game, moves| {
            assert_eq!(
                *evaluated_game.position(),
                root_position,
                "terminal leaf evaluated"
            );
            Ok(evaluation(moves, &[(mv, 1.0)], -0.75))
        });
        let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
        let root = tree.root();
        for count in 1..=2 {
            tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
            assert_eq!(evaluator.calls, 1);
            assert_eq!(tree.node_count(), 2);
            let edge = choices(&tree, root)
                .edges()
                .iter()
                .find(|edge| edge.mv().to_string() == mv)
                .unwrap();
            assert_eq!(edge.stats().visits(), count);
            assert_eq!(edge.stats().mean_value(), -leaf_value);
            assert!(
                matches!(tree.node(edge.child().unwrap()), Some(Node::Terminal(value)) if value.get() == leaf_value)
            );
            assert_eq!(*game.position(), root_position);
            assert_eq!(game.undo(), None);
        }
    }
}

#[test]
fn threefold_draw_uses_and_restores_the_existing_game_history() {
    let mut game = Game::new(START.parse().unwrap());
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"];
    let mut played = Vec::new();
    for index in 0..7 {
        played.push(play(&mut game, cycle[index % 4]));
    }
    let root_position = *game.position();
    assert_eq!(game.repetition_count(), 2);
    let mut evaluator = evaluator(|game, moves| {
        assert_eq!(*game.position(), root_position, "threefold leaf evaluated");
        Ok(evaluation(moves, &[("f6g8", 1.0)], 0.0))
    });
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    for iteration in 0..2 {
        if iteration == 0 {
            assert!(matches!(
                tree.begin_simulation(&mut game, 1.0).unwrap(),
                SimulationStep::Completed
            ));
        } else {
            tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
        }
        assert_eq!(*game.position(), root_position);
        assert_eq!(game.repetition_count(), 2);
        assert_eq!(game.outcome(), None);
        assert_eq!(evaluator.calls, 1);
        assert_eq!(tree.node_count(), 2);
        let child = choices(&tree, tree.root())
            .edges()
            .iter()
            .find_map(Edge::child)
            .unwrap();
        assert!(matches!(tree.node(child), Some(Node::Terminal(value)) if value.get() == 0.0));
    }
    for mv in played.into_iter().rev() {
        assert_eq!(game.undo(), Some(mv));
    }
    assert_eq!(game.undo(), None);
    assert_eq!(*game.position(), START.parse().unwrap());
    assert_eq!(game.repetition_count(), 1);
}

#[test]
fn failed_deep_resolution_restores_game_and_tree_and_can_be_retried() {
    for failure in 0..3 {
        let mut game = Game::new(START.parse().unwrap());
        let history = [play(&mut game, "g1f3"), play(&mut game, "g8f6")];
        let root_position = *game.position();
        let mut calls = 0;
        let mut evaluator = evaluator(|_, moves| {
            calls += 1;
            if calls == 4 && failure == 0 {
                return Err(io::Error::new(io::ErrorKind::TimedOut, "fixture timeout"));
            }
            let mut weights = vec![0.0; moves.len()];
            weights[0] = 1.0;
            if calls == 4 {
                if failure == 1 {
                    weights.pop();
                } else {
                    weights[0] = f32::NAN;
                }
            }
            Ok(Evaluation {
                value: Value::new(0.0).unwrap(),
                policy_weights: weights,
            })
        });
        let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
        for _ in 0..2 {
            tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
        }
        let before = snapshot(&tree);
        let error = tree.simulate(&mut game, &mut evaluator, 1.0).unwrap_err();
        match (failure, error) {
            (0, SimulationError::Resolution(ResolveError::Evaluator(error))) => {
                assert_eq!(error.kind(), io::ErrorKind::TimedOut)
            }
            (
                1,
                SimulationError::Resolution(ResolveError::Expansion(
                    ExpansionError::PolicyLengthMismatch { .. },
                )),
            ) => {}
            (
                2,
                SimulationError::Resolution(ResolveError::Expansion(
                    ExpansionError::InvalidPolicyWeight(InvalidPolicyWeight { index: 0 }),
                )),
            ) => {}
            (_, other) => panic!("unexpected error: {other}"),
        }
        assert_eq!(snapshot(&tree), before);
        assert_eq!(evaluator.calls, 4);
        assert_eq!(*game.position(), root_position);
        assert_eq!(game.repetition_count(), 1);
        tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
        assert_eq!(evaluator.calls, 5);
        assert_eq!(tree.node_count(), 4);
        assert_eq!(choices(&tree, tree.root()).edges()[0].stats().visits(), 3);
        assert_eq!(*game.position(), root_position);
        for mv in history.into_iter().rev() {
            assert_eq!(game.undo(), Some(mv));
        }
        assert_eq!(game.undo(), None);
        assert_eq!(*game.position(), START.parse().unwrap());
    }
}

#[test]
fn invalid_exploration_and_terminal_roots_do_not_run_a_simulation() {
    let mut game = Game::new(START.parse().unwrap());
    let root_position = *game.position();
    let mut evaluator = evaluator(|_, moves| {
        Ok(Evaluation {
            value: Value::new(0.0).unwrap(),
            policy_weights: vec![1.0; moves.len()],
        })
    });
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    let before = snapshot(&tree);
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            tree.simulate(&mut game, &mut evaluator, invalid),
            Err(SimulationError::InvalidExploration)
        ));
        assert_eq!(snapshot(&tree), before);
        assert_eq!(*game.position(), root_position);
        assert_eq!(evaluator.calls, 1);
        assert_eq!(game.undo(), None);
    }
    for fen in [
        "k7/1Q6/2K5/8/8/8/8/8 b - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let root_position = *game.position();
        let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
        let before = snapshot(&tree);
        assert!(matches!(
            tree.simulate(&mut game, &mut evaluator, 1.0),
            Err(SimulationError::TerminalRoot)
        ));
        assert_eq!(snapshot(&tree), before);
        assert_eq!(*game.position(), root_position);
        assert_eq!(evaluator.calls, 1);
        assert_eq!(game.undo(), None);
    }
}

#[test]
fn pending_completion_preserves_leaf_inputs_and_backs_up_one_sample() {
    let mut game = Game::new(START.parse().unwrap());
    let history = [play(&mut game, "e2e4"), play(&mut game, "e7e5")];
    let original = *game.position();
    let mut tree = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap());
    let selected = choices(&tree, tree.root()).edges()[0].mv();
    let expected = original.play(selected).unwrap();
    let mut moves = Vec::new();
    expected.generate_legal_moves(&mut moves);

    let SimulationStep::NeedsEvaluation(pending) = tree.begin_simulation(&mut game, 1.0).unwrap()
    else {
        panic!("expected a request")
    };
    assert_eq!(*pending.game().position(), expected);
    assert_eq!(pending.game().outcome(), None);
    assert_eq!(pending.legal_moves(), moves);
    // Different weights make any move/weight reordering observable.
    let weights: Vec<_> = (1..=moves.len()).map(|n| n as f32).collect();
    let total: f32 = weights.iter().sum();
    pending
        .complete(Evaluation {
            value: Value::new(0.625).unwrap(),
            policy_weights: weights.clone(),
        })
        .unwrap();

    assert_eq!(tree.node_count(), 2);
    let root = choices(&tree, tree.root());
    let edge = root.edges()[0];
    assert_eq!(edge.stats().visits(), 1);
    assert_eq!(edge.stats().mean_value(), -0.625);
    assert!(root.edges()[1..].iter().all(|e| e.stats().visits() == 0));
    let child = choices(&tree, edge.child().unwrap());
    assert_eq!(child.value().get(), 0.625);
    for (index, edge) in child.edges().iter().enumerate() {
        assert_eq!(edge.mv(), moves[index]);
        assert!((edge.stats().prior() - weights[index] / total).abs() < 1e-7);
        assert_eq!(edge.stats().visits(), 0);
        assert_eq!(edge.child(), None);
    }
    assert_eq!(*game.position(), original);
    for mv in history.into_iter().rev() {
        assert_eq!(game.undo(), Some(mv));
    }
    assert_eq!(game.undo(), None);
    assert_eq!(*game.position(), START.parse().unwrap());
}

#[test]
fn dropped_and_invalid_deep_requests_restore_all_recorded_history_and_allow_retry() {
    let mut game = Game::new(START.parse().unwrap());
    let mut history = Vec::new();
    for mv in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        let position = *game.position();
        let repetitions = game.repetition_count();
        history.push((play(&mut game, mv), position, repetitions));
    }
    let original = *game.position();
    assert_eq!(game.repetition_count(), 2);
    let mut evaluator = evaluator(|_, moves| {
        let mut weights = vec![0.0; moves.len()];
        weights[0] = 1.0;
        Ok(Evaluation {
            value: Value::new(0.0).unwrap(),
            policy_weights: weights,
        })
    });
    let mut tree = Tree::new(resolve_node(&game, &mut evaluator).unwrap());
    for _ in 0..2 {
        tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
    }
    let before = snapshot(&tree);
    let mut expected = original;
    for _ in 0..3 {
        let mut moves = Vec::new();
        expected.generate_legal_moves(&mut moves);
        expected = expected.play(moves[0]).unwrap();
    }
    for failure in 0..3 {
        let SimulationStep::NeedsEvaluation(pending) =
            tree.begin_simulation(&mut game, 1.0).unwrap()
        else {
            panic!("expected a request three plies below the root")
        };
        assert_eq!(*pending.game().position(), expected);
        if failure == 0 {
            drop(pending);
        } else {
            let mut weights = vec![1.0; pending.legal_moves().len()];
            if failure == 1 {
                weights.pop();
            } else {
                weights[0] = f32::NAN;
            }
            let error = pending.complete(Evaluation {
                value: Value::new(0.0).unwrap(),
                policy_weights: weights,
            });
            match (failure, error) {
                (1, Err(ExpansionError::PolicyLengthMismatch { .. })) => {}
                (2, Err(ExpansionError::InvalidPolicyWeight(_))) => {}
                other => panic!("unexpected error: {other:?}"),
            }
        }
        assert_eq!(snapshot(&tree), before);
        assert_eq!(*game.position(), original);
        assert_eq!(game.repetition_count(), 2);
        assert_eq!(game.outcome(), None);
    }
    // Cancellation and invalid output did not allocate nodes or spend visits.
    assert_eq!(evaluator.calls, 3);
    tree.simulate(&mut game, &mut evaluator, 1.0).unwrap();
    assert_eq!(tree.node_count(), 4);
    assert_eq!(choices(&tree, tree.root()).edges()[0].stats().visits(), 3);
    for (mv, position, repetitions) in history.into_iter().rev() {
        assert_eq!(game.undo(), Some(mv));
        assert_eq!(*game.position(), position);
        assert_eq!(game.repetition_count(), repetitions);
    }
    assert_eq!(game.undo(), None);
}

#[test]
fn evaluator_unwinding_drops_the_request_and_restores_game_and_tree() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut game = Game::new(START.parse().unwrap());
    let history = [play(&mut game, "e2e4"), play(&mut game, "e7e5")];
    let original = *game.position();
    let mut tree = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap());
    let before = snapshot(&tree);
    let mut panics = evaluator(|_, _| panic!("fixture evaluator panic"));
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            tree.simulate(&mut game, &mut panics, 1.0).unwrap();
        }))
        .is_err()
    );
    assert_eq!(snapshot(&tree), before);
    assert_eq!(*game.position(), original);
    assert_eq!(game.repetition_count(), 1);
    for mv in history.into_iter().rev() {
        assert_eq!(game.undo(), Some(mv));
    }
    assert_eq!(game.undo(), None);
}

#[test]
fn terminal_paths_complete_immediately_without_a_pending_request() {
    for (fen, mv, parent_value) in [
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6b7", 1.0),
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", "b6c7", 0.0),
        ("4k3/8/8/8/8/8/3r4/2B1K3 w - - 0 1", "c1d2", 0.0),
        ("4k3/8/8/8/8/8/8/R3K3 w - - 149 76", "a1a2", 0.0),
    ] {
        let mut game = Game::new(fen.parse().unwrap());
        let original = *game.position();
        let mut moves = Vec::new();
        game.position().generate_legal_moves(&mut moves);
        let node = ExpandedNode::new(&moves, evaluation(&moves, &[(mv, 1.0)], 0.0)).unwrap();
        let mut tree = Tree::new(Node::Expanded(node));
        // Both a newly discovered terminal and a stored terminal complete here.
        for n in 1..=2 {
            assert!(matches!(
                tree.begin_simulation(&mut game, 1.0).unwrap(),
                SimulationStep::Completed
            ));
            assert_eq!(tree.node_count(), 2);
            let edge = choices(&tree, tree.root())
                .edges()
                .iter()
                .find(|e| e.mv().to_string() == mv)
                .unwrap();
            assert_eq!(edge.stats().visits(), n);
            assert_eq!(edge.stats().mean_value(), parent_value);
            assert_eq!(*game.position(), original);
            assert_eq!(game.undo(), None);
        }
    }
}

#[test]
fn beginning_rejects_invalid_inputs_without_changing_game_or_tree() {
    let mut game = Game::new(START.parse().unwrap());
    let original = *game.position();
    let mut tree = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap());
    let before = snapshot(&tree);
    for exploration in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            tree.begin_simulation(&mut game, exploration).unwrap_err(),
            BeginSimulationError::InvalidExploration
        );
        assert_eq!(snapshot(&tree), before);
        assert_eq!(*game.position(), original);
        assert_eq!(game.undo(), None);
    }
    let mut game = Game::new("4k3/8/8/8/8/8/8/4K3 w - - 0 1".parse().unwrap());
    let original = *game.position();
    let mut tree = Tree::new(resolve_node(&game, &mut UniformEvaluator).unwrap());
    let before = snapshot(&tree);
    assert_eq!(
        tree.begin_simulation(&mut game, 1.0).unwrap_err(),
        BeginSimulationError::TerminalRoot
    );
    assert_eq!(snapshot(&tree), before);
    assert_eq!(*game.position(), original);
    assert_eq!(game.undo(), None);
}

#[test]
fn explicit_completion_matches_synchronous_search_after_every_simulation() {
    fn check<E: Evaluator + Default>(fen: &str) {
        let mut sync_game = Game::new(fen.parse().unwrap());
        let mut split_game = Game::new(fen.parse().unwrap());
        let mut sync_evaluator = E::default();
        let mut split_evaluator = E::default();
        let mut sync = Tree::new(resolve_node(&sync_game, &mut sync_evaluator).unwrap());
        let mut split = Tree::new(resolve_node(&split_game, &mut split_evaluator).unwrap());
        for _ in 0..128 {
            sync.simulate(&mut sync_game, &mut sync_evaluator, 1.0)
                .unwrap();
            if let SimulationStep::NeedsEvaluation(pending) =
                split.begin_simulation(&mut split_game, 1.0).unwrap()
            {
                let result = split_evaluator
                    .evaluate(pending.game(), pending.legal_moves())
                    .unwrap();
                pending.complete(result).unwrap();
            }
            assert_eq!(snapshot(&sync), snapshot(&split));
            assert_eq!(sync.report(), split.report());
            assert_eq!(sync_game.position(), split_game.position());
            assert_eq!(sync_game.repetition_count(), split_game.repetition_count());
        }
        assert_eq!(sync_game.undo(), None);
        assert_eq!(split_game.undo(), None);
    }
    for fen in [
        START,
        "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
        "3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1",
    ] {
        check::<UniformEvaluator>(fen);
        check::<MaterialEvaluator>(fen);
    }
}
