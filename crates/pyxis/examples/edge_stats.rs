use pyxis::{EdgeStats, Value};

fn main() {
    let mut edge = EdgeStats::new(0.25).expect("a valid prior");
    println!("Synthetic samples for a White-parent -> Black-child edge.");
    println!("Returned samples use Black's perspective; stored statistics use White's.");
    println!("We negate explicitly before record(); record() itself does not flip values.");
    println!("P = fixed prior, N = visits, W = value sum, Q = W/N.");
    println!(
        "\n{:>6} {:>10} {:>10} {:>7} {:>4} {:>8} {:>8}",
        "sample", "Black v", "White v", "P", "N", "W", "Q"
    );
    println!(
        "{:>6} {:>10} {:>10} {:>7.3} {:>4} {:>8.3} {:>8.3}",
        "start",
        "--",
        "--",
        edge.prior(),
        edge.visits(),
        edge.value_sum(),
        edge.mean_value()
    );

    for (index, raw) in [-0.75, 0.25, -1.0].into_iter().enumerate() {
        let child_value = Value::new(raw).expect("a finite value in [-1, 1]");
        let parent_value = -child_value;
        edge.record(parent_value);
        println!(
            "{:>6} {:>+10.3} {:>+10.3} {:>7.3} {:>4} {:>+8.3} {:>+8.3}",
            index + 1,
            child_value.get(),
            parent_value.get(),
            edge.prior(),
            edge.visits(),
            edge.value_sum(),
            edge.mean_value()
        );
    }
    println!("\nAt N=0, Q=0 is only a placeholder; no sample has been recorded.");
    println!("Individual values stay in [-1, 1], but their sum W can exceed that range.");
    println!("For a deeper path, backup must flip perspective once per edge.");
}
