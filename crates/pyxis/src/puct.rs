use crate::EdgeStats;

/// Computes Q + c * P * sqrt(T) / (1 + N) for one outgoing edge.
///
/// `parent_visits` is T: the sum of visits across all of the parent's outgoing
/// edges, including this edge. `exploration` is the positive coefficient c.
/// Scores use the parent player's perspective and can exceed one.
///
/// The stored f32 mean and prior are widened for scoring. Temporary arithmetic
/// uses f64 so finite f32 coefficients cannot overflow the exploration bonus.
///
/// When T is zero, unvisited edges all score zero. The caller handles the
/// highest-prior startup rule and subsequent selection ties separately.
///
/// Returns `None` if `exploration` is nonfinite or nonpositive, or if the parent
/// total is smaller than this edge's visit count. The caller must ensure that
/// the total actually matches the sibling edges; this function cannot verify it.
pub fn puct_score(stats: EdgeStats, parent_visits: u64, exploration: f32) -> Option<f64> {
    if !exploration.is_finite() || exploration <= 0.0 || parent_visits < u64::from(stats.visits()) {
        return None;
    }

    let bonus = f64::from(exploration) * f64::from(stats.prior()) * (parent_visits as f64).sqrt()
        / (1.0 + f64::from(stats.visits()));
    Some(f64::from(stats.mean_value()) + bonus)
}

#[cfg(test)]
mod tests {
    use super::puct_score;
    use crate::{EdgeStats, Value};

    #[test]
    fn matches_the_worked_example() {
        let mut stats = EdgeStats::new(0.5).unwrap();
        stats.record(Value::new(0.25).unwrap());

        assert_eq!(puct_score(stats, 4, 1.0), Some(0.75));
    }

    #[test]
    fn higher_value_or_prior_increases_the_score() {
        let mut baseline = EdgeStats::new(0.25).unwrap();
        baseline.record(Value::new(-0.25).unwrap());
        let mut higher_value = EdgeStats::new(0.25).unwrap();
        higher_value.record(Value::new(0.25).unwrap());
        let mut higher_prior = EdgeStats::new(0.75).unwrap();
        higher_prior.record(Value::new(-0.25).unwrap());

        assert_eq!(puct_score(baseline, 4, 1.0), Some(0.0));
        assert_eq!(puct_score(higher_value, 4, 1.0), Some(0.5));
        assert_eq!(puct_score(higher_prior, 4, 1.0), Some(0.5));
    }

    #[test]
    fn exploration_strength_and_parent_visits_increase_the_bonus() {
        let mut stats = EdgeStats::new(0.5).unwrap();
        stats.record(Value::new(0.25).unwrap());

        assert_eq!(puct_score(stats, 4, 1.0), Some(0.75));
        assert_eq!(puct_score(stats, 4, 2.0), Some(1.25));
        assert_eq!(puct_score(stats, 16, 1.0), Some(1.25));
    }

    #[test]
    fn more_edge_visits_reduce_the_bonus_at_the_same_mean() {
        let mut stats = EdgeStats::new(0.5).unwrap();
        stats.record(Value::new(0.25).unwrap());
        assert_eq!(puct_score(stats, 4, 1.0), Some(0.75));

        stats.record(Value::new(0.25).unwrap());
        stats.record(Value::new(0.25).unwrap());
        assert_eq!(stats.mean_value(), 0.25);
        assert_eq!(puct_score(stats, 4, 1.0), Some(0.5));
    }

    #[test]
    fn unvisited_edges_tie_at_startup_and_use_their_prior_afterward() {
        let lower = EdgeStats::new(0.25).unwrap();
        let higher = EdgeStats::new(0.75).unwrap();

        assert_eq!(puct_score(lower, 0, 1.0), Some(0.0));
        assert_eq!(puct_score(higher, 0, 1.0), Some(0.0));
        assert_eq!(puct_score(lower, 4, 1.0), Some(0.5));
        assert_eq!(puct_score(higher, 4, 1.0), Some(1.5));
    }

    #[test]
    fn zero_prior_contributes_no_exploration_bonus() {
        let mut stats = EdgeStats::new(0.0).unwrap();
        stats.record(Value::new(-0.25).unwrap());

        assert_eq!(puct_score(stats, u64::MAX, f32::MAX), Some(-0.25));
    }

    #[test]
    fn invalid_coefficients_and_inconsistent_totals_are_rejected() {
        let mut stats = EdgeStats::new(0.5).unwrap();
        for exploration in [0.0, -0.0, -1.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            assert_eq!(puct_score(stats, 0, exploration), None);
        }
        stats.record(Value::new(0.25).unwrap());
        assert_eq!(puct_score(stats, 0, 1.0), None);
    }

    #[test]
    fn extreme_finite_coefficients_produce_finite_scores() {
        let stats = EdgeStats::new(1.0).unwrap();

        assert_eq!(
            puct_score(stats, 4, f32::MAX),
            Some(2.0 * f64::from(f32::MAX))
        );
        assert!(puct_score(stats, u64::MAX, f32::MAX).unwrap().is_finite());
        assert_eq!(
            puct_score(stats, 1, f32::from_bits(1)),
            Some(f64::from(f32::from_bits(1)))
        );
    }
}
