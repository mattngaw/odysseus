use pyxis::{EdgeStats, Value, puct_score, select_edge};

fn print_scores(label: &str, edges: &[(&str, EdgeStats)], exploration: f32) {
    let parent_visits: u64 = edges.iter().map(|(_, edge)| u64::from(edge.visits())).sum();
    println!("\n{label}: c={exploration:.1}, T={parent_visits}");
    println!(
        "{:>4} {:>7} {:>4} {:>8} {:>8} {:>10} {:>10}",
        "edge", "P", "N", "W", "Q", "bonus U", "score Q+U"
    );
    for &(name, edge) in edges {
        let score = puct_score(edge, parent_visits, exploration).expect("valid scoring inputs");
        let mean = f64::from(edge.mean_value());
        // Recover the bonus from the library's score instead of duplicating PUCT.
        let bonus = score - mean;
        println!(
            "{name:>4} {:>7.3} {:>4} {:>+8.3} {mean:>+8.3} {bonus:>10.6} {score:>+10.6}",
            edge.prior(),
            edge.visits(),
            edge.value_sum()
        );
    }
    let stats: Vec<_> = edges.iter().map(|&(_, edge)| edge).collect();
    let selected = select_edge(&stats, exploration).expect("valid inputs and a nonempty edge list");
    println!("Selected: {} (index {selected})", edges[selected].0);
}

fn main() {
    let mut edges = [("A", 0.5), ("B", 0.3), ("C", 0.2)]
        .map(|(name, prior)| (name, EdgeStats::new(prior).expect("a valid prior")));
    println!("PUCT score = Q + U, where U = c * P * sqrt(T) / (1 + N).");
    println!("T is the sum of sibling visits. Values use the parent player's perspective.");
    print_scores("Before any visits", &edges, 1.0);
    println!("All scores are zero at T=0, even with unequal priors.");
    println!("The selector uses highest prior at startup and keeps the first edge on exact ties.");

    // These are scripted samples, not the result of a search simulation.
    for raw in [0.25, 0.5, 0.75, 0.5] {
        edges[0].1.record(Value::new(raw).unwrap());
    }
    edges[1].1.record(Value::new(0.25).unwrap());
    println!("\nScripted parent-perspective samples: A=[0.25, 0.5, 0.75, 0.5], B=[0.25], C=[].");
    print_scores("After recording samples", &edges, 1.0);
    print_scores("Same statistics, stronger exploration", &edges, 3.0);
    println!("\nChanging c changes only the exploration bonus; P, N, W, and Q stay fixed.");
    println!("An unvisited edge can gain a nonzero bonus once its siblings have visits.");
    println!("A PUCT score is a selection score, not a bounded game-outcome value.");
}
