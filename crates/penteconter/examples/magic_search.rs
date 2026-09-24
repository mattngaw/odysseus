//! Explicit development tool; it does not run during engine initialization.
//! Usage: cargo run -p penteconter --release --example magic_search -- [rook|bishop [square]]
//! With no arguments, searches all 128 bishop/rook squares. CSV goes to stdout;
//! verification summaries and failures go to stderr. Constants are not installed.
//! `--rust` emits the complete Rust multiplier file after all searches verify.

use std::{env, process::ExitCode};

use penteconter::attacks::{magic, reference};
use penteconter::{Bitboard, Square};

const BASE_SEED: u64 = 0x4f44_5953_5345_5553;
const MAX_CANDIDATES: u64 = 10_000_000;

#[derive(Clone, Copy, Debug)]
enum Slider {
    Rook,
    Bishop,
}

impl Slider {
    fn name(self) -> &'static str {
        match self {
            Self::Rook => "rook",
            Self::Bishop => "bishop",
        }
    }

    fn mask(self, square: Square) -> Bitboard {
        match self {
            Self::Rook => magic::rook_mask(square),
            Self::Bishop => magic::bishop_mask(square),
        }
    }

    fn attacks(self, square: Square, occupied: Bitboard) -> Bitboard {
        match self {
            Self::Rook => reference::rook_attacks(square, occupied),
            Self::Bishop => reference::bishop_attacks(square, occupied),
        }
    }

    fn seed(self, square: Square) -> u64 {
        let kind = match self {
            Self::Rook => 0,
            Self::Bishop => 1,
        };
        // Independent deterministic streams: a single-square run matches an all-square run.
        BASE_SEED
            .wrapping_add((kind * 64 + square.index() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }
}

/// SplitMix64, used only to generate reproducible search candidates.
struct Candidates(u64);

impl Candidates {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn sparse(&mut self) -> u64 {
        // A search heuristic: approximately one eighth of the bits remain set.
        // Every accepted candidate still has to pass all occupancy cases.
        self.next_u64() & self.next_u64() & self.next_u64()
    }
}

#[derive(Clone, Copy, Default)]
struct Slot {
    generation: u64,
    attacks: Bitboard,
}

#[derive(Debug, Eq, PartialEq)]
struct Found {
    multiplier: u64,
    shift: u32,
    candidates: u64,
    table: Vec<Option<Bitboard>>,
}

fn index(occupied: Bitboard, mask: Bitboard, multiplier: u64, shift: u32) -> usize {
    ((occupied.bits() & mask.bits()).wrapping_mul(multiplier) >> shift) as usize
}

fn try_candidate(
    cases: &[(Bitboard, Bitboard)],
    mask: Bitboard,
    multiplier: u64,
    shift: u32,
    generation: u64,
    slots: &mut [Slot],
) -> bool {
    for &(occupied, attacks) in cases {
        let slot = &mut slots[index(occupied, mask, multiplier, shift)];
        if slot.generation != generation {
            // A new generation logically clears the table without touching
            // every slot on each attempt. Rejected candidates leave stale slots.
            *slot = Slot {
                generation,
                attacks,
            };
        } else if slot.attacks != attacks {
            return false; // Destructive collision: two answers need the same slot.
        }
        // A collision with the same attack bitboard is valid.
    }
    true
}

fn find_magic(slider: Slider, square: Square, seed: u64, limit: u64) -> Option<Found> {
    let mask = slider.mask(square);
    let shift = 64 - mask.count();
    let cases: Vec<_> = mask
        .subsets()
        .map(|occupied| (occupied, slider.attacks(square, occupied)))
        .collect();
    let mut slots = vec![Slot::default(); 1 << mask.count()];
    let mut candidates = Candidates(seed);
    for generation in 1..=limit {
        let multiplier = candidates.sparse();
        if try_candidate(&cases, mask, multiplier, shift, generation, &mut slots) {
            let table = slots
                .into_iter()
                .map(|slot| (slot.generation == generation).then_some(slot.attacks))
                .collect();
            return Some(Found {
                multiplier,
                shift,
                candidates: generation,
                table,
            });
        }
    }
    None
}

fn verify(slider: Slider, square: Square, found: &Found) -> Result<usize, String> {
    let mask = slider.mask(square);
    let mut count = 0;
    // Re-enumerate and recompute from the ray walker after search, checking the
    // materialized table rather than the generation-stamped search scratch space.
    for occupied in mask.subsets() {
        for occupied in [occupied, occupied | !mask] {
            let entry = found
                .table
                .get(index(occupied, mask, found.multiplier, found.shift));
            if entry != Some(&Some(slider.attacks(square, occupied))) {
                return Err(format!(
                    "verification failed for {} {square}, {occupied:?}",
                    slider.name()
                ));
            }
        }
        count += 1;
    }
    Ok(count)
}

fn run(args: &[String]) -> Result<(), String> {
    let usage = "usage: magic_search [rook|bishop [a1..h8]] | --rust";
    if args == ["--help"] {
        println!(
            "{usage}\nNo arguments searches all 128 squares. Fixed base seed: {BASE_SEED:#018x}; limit: {MAX_CANDIDATES} candidates per square."
        );
        return Ok(());
    }
    let rust_output = args == ["--rust"];
    let args = if rust_output { &[] } else { args };
    if args.len() > 2 {
        return Err(usage.into());
    }
    let sliders = match args.first().map(String::as_str) {
        None => vec![Slider::Rook, Slider::Bishop],
        Some("rook") => vec![Slider::Rook],
        Some("bishop") => vec![Slider::Bishop],
        _ => return Err(usage.into()),
    };
    let squares: Vec<_> = match args.get(1) {
        None => (0..64).map(|i| Square::new(i).unwrap()).collect(),
        Some(name) => match name.as_bytes() {
            [file @ b'a'..=b'h', rank @ b'1'..=b'8'] => {
                vec![Square::from_coords(file - b'a', rank - b'1').unwrap()]
            }
            _ => return Err(usage.into()),
        },
    };
    if !rust_output {
        println!(
            "piece,square,mask,multiplier,shift,seed,candidates,table_slots,used_slots,constructive_collisions"
        );
    }
    let mut multipliers = Vec::new();
    let mut verified_squares = 0;
    let mut verified_subsets = 0;
    for slider in sliders {
        for &square in &squares {
            let seed = slider.seed(square);
            let found = find_magic(slider, square, seed, MAX_CANDIDATES).ok_or_else(|| {
                format!("search exhausted {MAX_CANDIDATES} candidates for {} {square}, seed {seed:#018x}; any preceding CSV rows are partial results", slider.name())
            })?;
            let count = verify(slider, square, &found)?;
            let used = found.table.iter().filter(|slot| slot.is_some()).count();
            if rust_output {
                multipliers.push(found.multiplier);
            } else {
                println!(
                    "{},{square},{:#018x},{:#018x},{},{seed:#018x},{},{},{used},{}",
                    slider.name(),
                    slider.mask(square).bits(),
                    found.multiplier,
                    found.shift,
                    found.candidates,
                    found.table.len(),
                    count - used
                );
            }
            verified_squares += 1;
            verified_subsets += count;
        }
    }
    if rust_output {
        println!(
            "// Generated by: cargo run -p penteconter --release --example magic_search --locked --offline -- --rust"
        );
        println!("// Base seed: {BASE_SEED:#018x}. Each array is ordered a1 through h8.");
        println!("// Regenerate with the development tool; do not hand-edit multipliers.");
        for (kind, name) in ["ROOK", "BISHOP"].into_iter().enumerate() {
            println!("\npub(super) const {name}_MULTIPLIERS: [u64; 64] = [");
            for index in 0..64 {
                println!(
                    "    {:#018x}, // {}",
                    multipliers[kind * 64 + index],
                    Square::new(index as u8).unwrap()
                );
            }
            println!("];");
        }
    }
    eprintln!(
        "Verified {verified_squares} squares and {verified_subsets} relevant occupancy subsets, each with excluded bits both clear and set."
    );
    Ok(())
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinguishes_constructive_and_destructive_collisions() {
        let mask = Bitboard::from_bits(3);
        let a = Bitboard::from_bits(0x10);
        let b = Bitboard::from_bits(0x20);
        let mut slots = vec![Slot::default(); 4];
        // A zero multiplier deliberately sends every occupancy to slot zero.
        let constructive: Vec<_> = mask.subsets().map(|occupied| (occupied, a)).collect();
        assert!(try_candidate(&constructive, mask, 0, 62, 1, &mut slots));
        let destructive = [(Bitboard::EMPTY, a), (Bitboard::from_bits(1), b)];
        assert!(!try_candidate(&destructive, mask, 0, 62, 2, &mut slots));
        let different_answer = [(Bitboard::EMPTY, b)];
        assert!(try_candidate(&different_answer, mask, 0, 62, 3, &mut slots));
        assert_eq!(slots[0].attacks, b); // Earlier generations cannot interfere.
    }

    #[test]
    fn search_is_repeatable_and_the_result_verifies() {
        let square = Square::new(0).unwrap();
        let seed = Slider::Bishop.seed(square);
        let first = find_magic(Slider::Bishop, square, seed, MAX_CANDIDATES).unwrap();
        let second = find_magic(Slider::Bishop, square, seed, MAX_CANDIDATES).unwrap();
        assert_eq!(first, second);
        assert_eq!(verify(Slider::Bishop, square, &first), Ok(64));
    }

    #[test]
    fn budget_exhaustion_and_corrupted_tables_are_detected() {
        let square = Square::new(0).unwrap();
        let seed = Slider::Bishop.seed(square);
        assert!(find_magic(Slider::Bishop, square, seed, 0).is_none());
        let mut found = find_magic(Slider::Bishop, square, seed, MAX_CANDIDATES).unwrap();
        let slot = index(
            Bitboard::EMPTY,
            Slider::Bishop.mask(square),
            found.multiplier,
            found.shift,
        );
        found.table[slot] = None;
        assert!(verify(Slider::Bishop, square, &found).is_err());
    }
}
