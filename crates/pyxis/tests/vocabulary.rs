use std::collections::HashSet;

use penteconter::{Bitboard, Color, Move, MoveKind, PieceKind, Position, Square, attacks};
use pyxis::vocabulary::{
    BASE_MOVE_COUNT, POLICY_SIZE, PolicyIndex, index_for_move, move_for_index,
};

const START: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

fn sq(s: &str) -> Square {
    let b = s.as_bytes();
    Square::from_coords(b[0] - b'a', b[1] - b'1').unwrap()
}

fn mv(s: &str, kind: MoveKind) -> Move {
    Move::new(sq(&s[..2]), sq(&s[2..4]), kind).unwrap()
}

#[test]
fn index_bounds_and_every_vocabulary_entry_round_trip_in_both_perspectives() {
    assert_eq!(POLICY_SIZE, 1858);
    assert_eq!(BASE_MOVE_COUNT, 1792);
    for invalid in [1858, 65535, usize::MAX] {
        assert_eq!(PolicyIndex::new(invalid), None);
    }
    let mut labels = HashSet::new();
    let mut checksum = 0xcbf29ce484222325u64;
    for raw in 0..1858 {
        let index = PolicyIndex::new(raw).unwrap();
        assert_eq!(index.index(), raw);
        let entry = index.entry();
        assert_eq!(entry.promotion.is_some(), raw >= 1792);
        let label = entry.to_string();
        assert!(labels.insert(label.clone()));
        for b in label.bytes().chain([b'\n']) {
            checksum = (checksum ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
        for side in [Color::White, Color::Black] {
            let absolute = |s: Square| {
                Square::new(s.index() as u8 ^ if side == Color::Black { 56 } else { 0 }).unwrap()
            };
            let kind = entry
                .promotion
                .map_or(MoveKind::Normal, MoveKind::Promotion);
            let descriptor = Move::new(absolute(entry.from), absolute(entry.to), kind).unwrap();
            assert_eq!(index_for_move(descriptor, side), Some(index));
        }
    }
    // Frozen from all kMoveStrs labels, newline-delimited, in Lc0 encoder.cc.
    // Source SHA256 f8b1fc47e5f7ab85c9d38a21d1b03f09dba98a5c6760f61f8ebacacf5afbdfae.
    assert_eq!(checksum, 0x32dcaf215253e599);
}

#[test]
fn base_slots_match_the_independent_attack_geometry_oracle() {
    let mut count = 0;
    for from in 0..64 {
        let from = Square::new(from).unwrap();
        let allowed = attacks::reference::rook_attacks(from, Bitboard::EMPTY)
            | attacks::reference::bishop_attacks(from, Bitboard::EMPTY)
            | attacks::knight_attacks(from);
        for to in 0..64 {
            let to = Square::new(to).unwrap();
            if let Some(mv) = Move::new(from, to, MoveKind::Normal) {
                let index = index_for_move(mv, Color::White);
                assert_eq!(index.is_some(), allowed.contains(to), "{mv}");
                if let Some(index) = index {
                    assert!(index.index() < BASE_MOVE_COUNT);
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 1792);
}

#[test]
fn known_indices_lock_ordering_special_moves_and_black_orientation() {
    for (text, kind, side, raw, label) in [
        ("a1b1", MoveKind::Normal, Color::White, 0, "a1b1"),
        ("e2e4", MoveKind::Normal, Color::White, 322, "e2e4"),
        ("e7e5", MoveKind::Normal, Color::Black, 322, "e2e4"),
        ("g1f3", MoveKind::Normal, Color::White, 159, "g1f3"),
        ("g8f6", MoveKind::Normal, Color::Black, 159, "g1f3"),
        ("e1g1", MoveKind::Castling, Color::White, 103, "e1h1"),
        ("e8g8", MoveKind::Castling, Color::Black, 103, "e1h1"),
        ("e1c1", MoveKind::Castling, Color::White, 97, "e1a1"),
        ("e8c8", MoveKind::Castling, Color::Black, 97, "e1a1"),
        ("e1g1", MoveKind::Normal, Color::White, 102, "e1g1"),
        (
            "a7a8",
            MoveKind::Promotion(PieceKind::Knight),
            Color::White,
            1401,
            "a7a8",
        ),
        (
            "a2a1",
            MoveKind::Promotion(PieceKind::Knight),
            Color::Black,
            1401,
            "a7a8",
        ),
        (
            "a7a8",
            MoveKind::Promotion(PieceKind::Queen),
            Color::White,
            1792,
            "a7a8q",
        ),
        (
            "a7a8",
            MoveKind::Promotion(PieceKind::Rook),
            Color::White,
            1793,
            "a7a8r",
        ),
        (
            "a7a8",
            MoveKind::Promotion(PieceKind::Bishop),
            Color::White,
            1794,
            "a7a8b",
        ),
        ("h8g8", MoveKind::Normal, Color::White, 1791, "h8g8"),
        (
            "h7h8",
            MoveKind::Promotion(PieceKind::Bishop),
            Color::White,
            1857,
            "h7h8b",
        ),
    ] {
        let index = index_for_move(mv(text, kind), side).unwrap();
        assert_eq!(index.index(), raw, "{text}");
        assert_eq!(index.entry().to_string(), label);
    }
    for (text, side, label) in [
        ("e5d6", Color::White, "e5d6"),
        ("e4d3", Color::Black, "e5d6"),
    ] {
        let index = index_for_move(mv(text, MoveKind::EnPassant), side).unwrap();
        assert_eq!(index.entry().to_string(), label);
    }
}

#[test]
fn all_promotion_routes_have_four_distinct_slots_in_both_colors() {
    for side in [Color::White, Color::Black] {
        let (from_rank, to_rank) = if side == Color::White { (6, 7) } else { (1, 0) };
        let mut seen = HashSet::new();
        for from_file in 0u8..8 {
            for to_file in 0u8..8 {
                for kind in [
                    PieceKind::Knight,
                    PieceKind::Queen,
                    PieceKind::Rook,
                    PieceKind::Bishop,
                ] {
                    let mv = Move::new(
                        Square::from_coords(from_file, from_rank).unwrap(),
                        Square::from_coords(to_file, to_rank).unwrap(),
                        MoveKind::Promotion(kind),
                    )
                    .unwrap();
                    let index = index_for_move(mv, side);
                    assert_eq!(index.is_some(), from_file.abs_diff(to_file) <= 1);
                    if let Some(index) = index {
                        assert!(seen.insert(index));
                        assert_eq!(index.index() < BASE_MOVE_COUNT, kind == PieceKind::Knight);
                    }
                }
            }
        }
        assert_eq!(seen.len(), 22 * 4);
    }
}

fn check_legal(position: Position, depth: u8) {
    let mut moves = Vec::new();
    position.generate_legal_moves(&mut moves);
    let mut seen = HashSet::new();
    for &mv in &moves {
        let index =
            index_for_move(mv, position.side_to_move()).expect("legal moves must have slots");
        assert!(
            seen.insert(index),
            "duplicate legal index in {}: {mv}",
            position.to_fen()
        );
        assert_eq!(
            move_for_index(index, position.side_to_move(), &moves),
            Some(mv)
        );
        if depth != 0 {
            check_legal(position.play_unchecked(mv), depth - 1);
        }
    }
}

#[test]
fn legal_moves_map_injectively_and_decode_with_exact_kinds_across_game_trees() {
    for fen in [
        START,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/2pPp3/8/8/4K3 b - d3 0 1",
        "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1",
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ] {
        check_legal(fen.parse().unwrap(), 2);
    }
}

#[test]
fn decoding_requires_a_matching_legal_move_and_preserves_ambiguous_base_slot_kinds() {
    for (fen, coordinate, kind) in [
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "e5d6",
            MoveKind::EnPassant,
        ),
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", MoveKind::Castling),
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "a7a8",
            MoveKind::Promotion(PieceKind::Knight),
        ),
        ("4k3/R7/8/8/8/8/8/4K3 w - - 0 1", "a7a8", MoveKind::Normal),
    ] {
        let position: Position = fen.parse().unwrap();
        let mut moves = Vec::new();
        position.generate_legal_moves(&mut moves);
        let expected = mv(coordinate, kind);
        let index = index_for_move(expected, Color::White).unwrap();
        assert_eq!(move_for_index(index, Color::White, &moves), Some(expected));
        moves.retain(|&mv| mv != expected);
        assert_eq!(move_for_index(index, Color::White, &moves), None);
        assert_eq!(move_for_index(index, Color::White, &[]), None);
    }
}

#[test]
fn unsupported_geometry_is_rejected_but_encoding_is_not_a_legality_check() {
    for (s, kind) in [
        ("a1c4", MoveKind::Normal),
        ("a6a7", MoveKind::Promotion(PieceKind::Knight)),
        ("a7c8", MoveKind::Promotion(PieceKind::Queen)),
        ("e2d3", MoveKind::EnPassant),
        ("e5e6", MoveKind::EnPassant),
        ("e1h1", MoveKind::Castling),
        ("d1f1", MoveKind::Castling),
        ("e8g8", MoveKind::Castling),
    ] {
        assert_eq!(index_for_move(mv(s, kind), Color::White), None, "{s}");
    }
    // Geometrically encodable does not mean this pawn can move three squares.
    let impossible = mv("e2e5", MoveKind::Normal);
    let index = index_for_move(impossible, Color::White).unwrap();
    let mut legal = Vec::new();
    START
        .parse::<Position>()
        .unwrap()
        .generate_legal_moves(&mut legal);
    assert_eq!(move_for_index(index, Color::White, &legal), None);
}
