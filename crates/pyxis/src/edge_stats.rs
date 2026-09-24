use crate::Value;

/// Statistics for one outgoing move, from the parent node's player perspective.
///
/// The prior stays fixed as samples accumulate. Values are stored and averaged
/// with f32 arithmetic; the caller handles perspective changes during backup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeStats {
    prior: f32,
    visits: u32,
    value_sum: f32,
}

impl EdgeStats {
    /// Creates an unvisited edge with the supplied normalized prior P.
    /// Returns `None` if the prior is nonfinite or outside [0, 1].
    /// Normalizing the priors across sibling edges is the caller's responsibility.
    pub const fn new(prior: f32) -> Option<Self> {
        if prior >= 0.0 && prior <= 1.0 {
            Some(Self {
                prior,
                visits: 0,
                value_sum: 0.0,
            })
        } else {
            None
        }
    }

    /// The policy prior P assigned when this edge was created.
    pub const fn prior(self) -> f32 {
        self.prior
    }

    /// Number of samples N recorded through this edge.
    pub const fn visits(self) -> u32 {
        self.visits
    }

    /// Sum W of recorded values, in the parent player's perspective.
    pub const fn value_sum(self) -> f32 {
        self.value_sum
    }

    /// Mean value Q = W / N, or the neutral placeholder zero when unvisited.
    /// The placeholder does not contribute a sample or visit.
    pub fn mean_value(self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    /// Records one sample already expressed from the parent player's perspective.
    /// The prior is unchanged; this method does not negate the supplied value.
    ///
    /// # Panics
    ///
    /// Panics without changing the statistics if the visit count would overflow.
    pub fn record(&mut self, value: Value) {
        let visits = self
            .visits
            .checked_add(1)
            .expect("edge visit count overflow");
        self.visits = visits;
        self.value_sum += value.get();
    }
}

#[cfg(test)]
mod tests {
    use super::EdgeStats;
    use crate::Value;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn new_edges_have_a_prior_and_no_samples() {
        for prior in [0.0, f32::from_bits(1), 0.25, 1.0] {
            let stats = EdgeStats::new(prior).unwrap();
            assert_eq!(stats.prior(), prior);
            assert_eq!(stats.visits(), 0);
            assert_eq!(stats.value_sum(), 0.0);
            assert_eq!(stats.mean_value(), 0.0);
        }
    }

    #[test]
    fn invalid_priors_are_rejected() {
        for prior in [
            -f32::from_bits(1),
            1.0_f32.next_up(),
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
        ] {
            assert!(EdgeStats::new(prior).is_none(), "accepted {prior}");
        }
    }

    #[test]
    fn samples_update_the_mean_without_changing_perspective_or_prior() {
        let mut stats = EdgeStats::new(0.25).unwrap();
        for (sample, visits, sum, mean) in [
            (0.75, 1, 0.75, 0.75),
            (-0.25, 2, 0.5, 0.25),
            (0.0, 3, 0.5, 1.0 / 6.0),
        ] {
            stats.record(Value::new(sample).unwrap());
            assert_eq!(stats.prior(), 0.25);
            assert_eq!(stats.visits(), visits);
            assert_eq!(stats.value_sum(), sum);
            assert!((stats.mean_value() - mean).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn value_sums_can_exceed_the_range_of_an_individual_sample() {
        for sample in [-1.0, 1.0] {
            let mut stats = EdgeStats::new(1.0).unwrap();
            for _ in 0..3 {
                stats.record(Value::new(sample).unwrap());
            }
            assert_eq!(stats.visits(), 3);
            assert_eq!(stats.value_sum(), 3.0 * sample);
            assert_eq!(stats.mean_value(), sample);
        }
    }

    #[test]
    fn visit_overflow_preserves_the_statistics() {
        let mut stats = EdgeStats::new(0.5).unwrap();
        stats.visits = u32::MAX;
        let before = stats;

        let result = catch_unwind(AssertUnwindSafe(|| {
            stats.record(Value::new(0.5).unwrap());
        }));

        assert!(result.is_err());
        assert_eq!(stats, before);
    }

    #[test]
    fn tree_backup_checks_all_counters_before_updating_any_edge() {
        use crate::{BackupError, Node, Tree, UniformEvaluator, resolve_node};
        use penteconter::Game;

        // This test lives here so it can prepare a near-overflow counter without
        // exposing a counter setter or performing billions of backup calls.
        for exhausted_step in 0..2 {
            let mut game = Game::new(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    .parse()
                    .unwrap(),
            );
            let mut nodes = Vec::new();
            for step in 0..3 {
                let Node::Expanded(mut node) = resolve_node(&game, &mut UniformEvaluator).unwrap()
                else {
                    panic!("expected expansion");
                };
                if step == exhausted_step {
                    node.edges[0].stats.visits = u32::MAX - 1;
                }
                game.play(node.edges()[0].mv()).unwrap();
                nodes.push(Node::Expanded(node));
            }
            let mut nodes = nodes.into_iter();
            let mut tree = Tree::new(nodes.next().unwrap());
            let root = tree.root();
            let child = tree.add_child(root, 0, nodes.next().unwrap()).unwrap();
            tree.add_child(child, 0, nodes.next().unwrap()).unwrap();
            let path = [(root, 0), (child, 0)];
            let sample = Value::new(0.5).unwrap();
            // Reaching the maximum is valid; exceeding it must change nothing.
            tree.backup(&path, sample).unwrap();
            let snapshot = |tree: &Tree| {
                path.map(|(id, index)| {
                    let Node::Expanded(node) = tree.node(id).unwrap() else {
                        unreachable!()
                    };
                    node.edges()[index].stats()
                })
            };
            let before = snapshot(&tree);
            assert_eq!(before[exhausted_step].visits(), u32::MAX);
            assert_eq!(tree.backup(&path, sample), Err(BackupError::VisitOverflow));
            assert_eq!(snapshot(&tree), before);
        }
    }

    #[test]
    fn simulation_overflow_restores_the_game_without_evaluation_or_tree_changes() {
        use crate::{
            Evaluation, Evaluator, Node, SimulationError, Tree, UniformEvaluator, resolve_node,
        };
        use penteconter::{Game, Move};
        use std::convert::Infallible;

        struct NeverEvaluate;
        impl Evaluator for NeverEvaluate {
            type Error = Infallible;

            fn evaluate(&mut self, _: &Game, _: &[Move]) -> Result<Evaluation, Self::Error> {
                panic!("overflow must be rejected before evaluating a new leaf")
            }
        }

        // Prepare private counters here, without adding public test setters.
        for exhausted_step in 0..2 {
            let mut game = Game::new(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                    .parse()
                    .unwrap(),
            );
            let original = *game.position();
            let mut nodes = Vec::new();
            for step in 0..2 {
                let Node::Expanded(mut node) = resolve_node(&game, &mut UniformEvaluator).unwrap()
                else {
                    panic!("expected expansion");
                };
                for (index, edge) in node.edges.iter_mut().enumerate() {
                    edge.stats = EdgeStats::new(if index == 0 { 1.0 } else { 0.0 }).unwrap();
                }
                if step == exhausted_step {
                    node.edges[0].stats.visits = u32::MAX;
                }
                if step == 0 {
                    game.play(node.edges()[0].mv()).unwrap();
                }
                nodes.push(Node::Expanded(node));
            }
            game.undo().unwrap();
            let mut nodes = nodes.into_iter();
            let mut tree = Tree::new(nodes.next().unwrap());
            tree.add_child(tree.root(), 0, nodes.next().unwrap())
                .unwrap();
            let before = format!("{tree:?}");
            assert!(matches!(
                tree.simulate(&mut game, &mut NeverEvaluate, 1.0),
                Err(SimulationError::VisitOverflow)
            ));
            assert_eq!(format!("{tree:?}"), before);
            assert_eq!(*game.position(), original);
            assert_eq!(game.repetition_count(), 1);
            assert_eq!(game.undo(), None);
        }
    }

    #[test]
    fn reports_sum_root_visits_without_truncating_to_an_edge_counter() {
        use crate::{Node, SearchReport, Tree, UniformEvaluator, resolve_node};
        use penteconter::Game;

        // Prepare private counters here instead of performing billions of visits.
        let game = Game::new(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                .parse()
                .unwrap(),
        );
        let Node::Expanded(mut root) = resolve_node(&game, &mut UniformEvaluator).unwrap() else {
            unreachable!();
        };
        root.edges[0].stats.visits = u32::MAX;
        root.edges[1].stats.visits = 10;
        let tree = Tree::new(Node::Expanded(root));
        let SearchReport::Nonterminal {
            best_move,
            simulations,
            moves,
        } = tree.report()
        else {
            unreachable!();
        };
        assert_eq!(simulations, u64::from(u32::MAX) + 10);
        assert_eq!(best_move, moves[0].mv);
        assert_eq!(
            moves[1].visit_fraction,
            Some((10.0 / simulations as f64) as f32)
        );
        assert_eq!(moves[2].visit_fraction, Some(0.0));
    }
}
