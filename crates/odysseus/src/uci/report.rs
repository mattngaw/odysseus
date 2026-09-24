use std::time::Duration;

use pyxis::SearchReport;

/// Formats one complete snapshot. The coordinator writes it without mixing jobs.
pub(super) fn report_lines(
    report: &SearchReport,
    elapsed: Duration,
    finished: bool,
    verbose_move_stats: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    let milliseconds = elapsed.as_millis();
    match report {
        SearchReport::Terminal(_) => {
            lines.push(format!("info time {milliseconds} nodes 0 nps 0"));
            if finished {
                lines.push("bestmove 0000".into());
            }
        }
        SearchReport::Nonterminal {
            best_move,
            simulations,
            moves,
        } => {
            let nps = u128::from(*simulations) * 1000 / milliseconds.max(1);
            // Nodes count completed simulations. The ordinary UCI line remains
            // useful to other GUIs; scalar Q does not imply centipawns or WDL.
            lines.push(format!(
                "info time {milliseconds} nodes {simulations} nps {nps} multipv 1 pv {best_move}"
            ));
            if verbose_move_stats {
                // Nibbler interprets Lc0's verbose stats order as worst-to-best.
                // Stable sorting from reversed input order preserves Pyxis's
                // first-in-order tie winner as the last (best) emitted move.
                let mut ranked: Vec<_> = moves.iter().rev().collect();
                ranked.sort_by(|a, b| {
                    if *simulations == 0 {
                        a.stats.prior().total_cmp(&b.stats.prior())
                    } else {
                        a.stats.visits().cmp(&b.stats.visits())
                    }
                });
                for entry in ranked {
                    // P is a percentage; Q is signed, in the root player's
                    // perspective. Unvisited Q retains Pyxis's zero placeholder.
                    lines.push(format!(
                        "info string {} N: {} (P: {:.6}%) (Q: {:+.8})",
                        entry.mv,
                        entry.stats.visits(),
                        f64::from(entry.stats.prior()) * 100.0,
                        entry.stats.mean_value(),
                    ));
                }
                // Supplies the denominator for N/S and resets Nibbler's ordering
                // counter for the next snapshot. Root setup contributes no visit.
                lines.push(format!("info string node N: {simulations}"));
            }
            if finished {
                lines.push(format!("bestmove {best_move}"));
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use penteconter::Position;
    use pyxis::{EdgeStats, RootMove, Value};

    #[test]
    fn values_keep_their_sign_and_priors_keep_their_units() {
        let position: Position = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1".parse().unwrap();
        let mut legal = Vec::new();
        position.generate_legal_moves(&mut legal);
        let mut low = EdgeStats::new(0.25).unwrap();
        low.record(Value::new(-0.5).unwrap());
        let high = EdgeStats::new(0.75).unwrap();
        let report = SearchReport::Nonterminal {
            best_move: legal[0],
            simulations: 1,
            moves: vec![
                RootMove {
                    mv: legal[0],
                    stats: low,
                    visit_fraction: Some(1.0),
                },
                RootMove {
                    mv: legal[1],
                    stats: high,
                    visit_fraction: Some(0.0),
                },
            ],
        };
        let lines = report_lines(&report, Duration::from_millis(10), true, true);
        assert_eq!(
            lines[1],
            format!(
                "info string {} N: 0 (P: 75.000000%) (Q: +0.00000000)",
                legal[1]
            )
        );
        assert_eq!(
            lines[2],
            format!(
                "info string {} N: 1 (P: 25.000000%) (Q: -0.50000000)",
                legal[0]
            )
        );
        assert_eq!(lines[3], "info string node N: 1");
        assert_eq!(lines[4], format!("bestmove {}", legal[0]));
    }

    #[test]
    fn terminal_root_has_no_move_statistics() {
        assert_eq!(
            report_lines(
                &SearchReport::Terminal(Value::new(-1.0).unwrap()),
                Duration::ZERO,
                true,
                true
            ),
            ["info time 0 nodes 0 nps 0", "bestmove 0000"]
        );
    }
}
