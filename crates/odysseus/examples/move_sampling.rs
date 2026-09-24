//! Inspect played-move temperatures with a fixed seed and known visit counts.

use std::error::Error;

use odysseus::self_play::{MoveSampler, Recorder, move_probabilities};
use penteconter::Game;
use pyxis::{SearchReport, Tree, UniformEvaluator, Value, resolve_node};

fn main() -> Result<(), Box<dyn Error>> {
    let seed = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<u64>())
        .transpose()?
        .unwrap_or(42);
    let game = Game::new("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".parse()?);
    let mut report = Tree::new(resolve_node(&game, &mut UniformEvaluator)?).report();
    let SearchReport::Nonterminal {
        simulations, moves, ..
    } = &mut report
    else {
        unreachable!()
    };
    for entry in moves {
        let count: u32 = match entry.mv.to_string().as_str() {
            "e2e4" => 60,
            "d2d4" => 30,
            "g1f3" => 10,
            _ => 0,
        };
        for _ in 0..count {
            entry.stats.record(Value::new(0.0).unwrap());
        }
        *simulations += u64::from(count);
    }
    let mut recorder = Recorder::new();
    recorder.record(&game, &report)?;
    let SearchReport::Nonterminal { moves, .. } = &report else {
        unreachable!()
    };
    let indices: Vec<_> = ["e2e4", "d2d4", "g1f3"]
        .into_iter()
        .map(|s| {
            moves
                .iter()
                .position(|entry| entry.mv.to_string() == s)
                .unwrap()
        })
        .collect();
    println!("Start-position legal moves, synthetic visits: e4=60, d4=30, Nf3=10; 17 unvisited.");
    println!("Seed={seed}; resetting the stream for each temperature; 10,000 draws each.");
    println!("Temperature  P(e4)    P(d4)    P(Nf3)   Samples: e4 / d4 / Nf3");
    for temperature in [0.0, 0.5, 1.0, 2.0] {
        let probabilities = move_probabilities(&report, temperature)?;
        let mut sampler = MoveSampler::new(seed);
        let mut counts = [0u32; 3];
        for _ in 0..10_000 {
            let chosen = sampler.sample(&report, temperature)?;
            let index = indices
                .iter()
                .position(|&index| moves[index].mv == chosen)
                .unwrap();
            counts[index] += 1;
        }
        println!(
            "{temperature:11.1}  {:.5}  {:.5}  {:.5}  {:5} / {:5} / {:5}",
            probabilities[indices[0]],
            probabilities[indices[1]],
            probabilities[indices[2]],
            counts[0],
            counts[1],
            counts[2]
        );
    }
    let target = recorder.roots()[0].policy_target();
    println!("Training targets remain N / sum N at every play temperature:");
    for &index in &indices {
        let slot = recorder.roots()[0].policy()[index].index;
        println!("  {}: {:.3}", moves[index].mv, target[slot.index()]);
    }
    Ok(())
}
