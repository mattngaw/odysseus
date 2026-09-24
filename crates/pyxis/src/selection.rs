use crate::{EdgeStats, puct_score};

/// Selects one outgoing edge, returning its index in the supplied slice.
///
/// The parent visit total is the sum of all sibling visits. At zero visits,
/// select the highest prior; otherwise select the highest PUCT score. Exact
/// ties keep the first edge in input order. Selection neither allocates nor
/// changes statistics. Supplying normalized sibling priors is the caller's job.
///
/// Returns `None` for an empty slice, a nonfinite or nonpositive exploration
/// coefficient, or a visit total that would overflow u64.
pub fn select_edge(edges: &[EdgeStats], exploration: f32) -> Option<usize> {
    select_edge_by(edges, exploration, |stats| *stats)
}

// Share the two-pass selection loop with nodes without collecting their stats.
pub(crate) fn select_edge_by<T>(
    edges: &[T],
    exploration: f32,
    stats: impl Fn(&T) -> EdgeStats,
) -> Option<usize> {
    if !exploration.is_finite() || exploration <= 0.0 {
        return None;
    }
    let parent_visits = edges.iter().try_fold(0u64, |total, edge| {
        total.checked_add(u64::from(stats(edge).visits()))
    })?;

    let mut best_index = None;
    let mut best_score = f64::NEG_INFINITY;
    for (index, edge) in edges.iter().enumerate() {
        let edge = stats(edge);
        let score = if parent_visits == 0 {
            f64::from(edge.prior())
        } else {
            puct_score(edge, parent_visits, exploration)?
        };
        // A strict comparison preserves the earlier edge on an exact tie.
        if score > best_score {
            best_index = Some(index);
            best_score = score;
        }
    }
    best_index
}

#[cfg(test)]
mod tests {
    use super::select_edge;
    use crate::{EdgeStats, Value};

    fn edge(prior: f32, samples: &[f32]) -> EdgeStats {
        let mut edge = EdgeStats::new(prior).unwrap();
        for &sample in samples {
            edge.record(Value::new(sample).unwrap());
        }
        edge
    }

    #[test]
    fn empty_and_single_edge_lists() {
        assert_eq!(select_edge(&[], 1.0), None);
        assert_eq!(select_edge(&[edge(1.0, &[])], 1.0), Some(0));
        assert_eq!(select_edge(&[edge(1.0, &[-1.0])], 1.0), Some(0));
    }

    #[test]
    fn startup_uses_the_highest_prior_and_keeps_the_first_tie() {
        let edges = [edge(0.1, &[]), edge(0.2, &[]), edge(0.7, &[])];
        assert_eq!(select_edge(&edges, 1.0), Some(2));
        let tied = [edge(0.2, &[]), edge(0.4, &[]), edge(0.4, &[])];
        assert_eq!(select_edge(&tied, 1.0), Some(1));
    }

    #[test]
    fn exploration_can_favor_an_unvisited_edge_over_the_highest_mean() {
        let edges = [
            edge(0.5, &[0.25, 0.5, 0.75, 0.5]),
            edge(0.3, &[0.25]),
            edge(0.2, &[]),
        ];
        // T=5. At c=1 the scores are about [0.724, 0.585, 0.447].
        // At c=3 they are about [1.171, 1.256, 1.342].
        assert_eq!(select_edge(&edges, 1.0), Some(0));
        assert_eq!(select_edge(&edges, 3.0), Some(2));
    }

    #[test]
    fn negative_scores_still_select_the_highest() {
        let edges = [edge(0.75, &[-1.0]), edge(0.25, &[-0.25])];
        // Both scores are negative; the lower-prior edge has the better score.
        assert_eq!(select_edge(&edges, 0.1), Some(1));
    }

    #[test]
    fn exact_score_ties_keep_input_order_even_with_different_statistics() {
        let unvisited = edge(0.25, &[]);
        let visited = edge(0.75, &[0.0, 0.0]);
        let other = edge(0.0, &[-0.5, -0.5]);
        // T=4: both tied edges score exactly 0.5; the other scores -0.5.
        assert_eq!(select_edge(&[unvisited, visited, other], 1.0), Some(0));
        assert_eq!(select_edge(&[visited, unvisited, other], 1.0), Some(0));
    }

    #[test]
    fn scripted_backups_reproduce_the_five_simulation_root_choices() {
        let mut edges = [edge(0.5, &[]), edge(0.5, &[])];
        // Root-only replay of the worked example: A, B, A, B, B. Values have
        // already been converted to the root player's perspective.
        for (expected, value) in [(0, 0.2), (1, -0.4), (0, -0.8), (1, 0.6), (1, 0.6)] {
            let selected = select_edge(&edges, 1.0).unwrap();
            assert_eq!(selected, expected);
            edges[selected].record(Value::new(value).unwrap());
        }
        assert_eq!(edges.map(EdgeStats::visits), [2, 3]);
    }

    #[test]
    fn invalid_exploration_is_rejected_at_startup_and_after_visits() {
        for exploration in [0.0, -0.0, -1.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert_eq!(select_edge(&[], exploration), None);
            assert_eq!(select_edge(&[edge(1.0, &[])], exploration), None);
            assert_eq!(select_edge(&[edge(1.0, &[0.5])], exploration), None);
        }
    }
}
