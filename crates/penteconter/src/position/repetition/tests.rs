use super::{Position, RepetitionState};
use std::collections::HashMap;

#[test]
fn equal_keys_do_not_hide_differences_in_repetition_state() {
    let base: Position = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".parse().unwrap();
    let legal_ep: Position = "4k3/8/8/2PpPpP1/8/8/8/4K3 w - d6 0 1".parse().unwrap();
    for (left, right_fen) in [
        (base, "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        (base, "r3k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1"),
        (base, "r3k2r/8/8/8/8/8/N7/R3K2R w KQkq - 0 1"),
        (legal_ep, "4k3/8/8/2PpPpP1/8/8/8/4K3 w - - 0 1"),
        (legal_ep, "4k3/8/8/2PpPpP1/8/8/8/4K3 w - f6 0 1"),
    ] {
        let mut right: Position = right_fen.parse().unwrap();
        assert_ne!(left.zobrist_key, right.zobrist_key);
        // Deliberately simulate a full-key collision. Only this private test can
        // overwrite the derived key; production callers cannot inject one.
        right.zobrist_key = left.zobrist_key;
        assert!(!left.same_repetition_state(&right), "{right_fen}");
        assert!(!right.same_repetition_state(&left), "{right_fen}");

        // Equal Zobrist keys must also remain separate keys in the index.
        let mut counts = HashMap::new();
        counts.insert(RepetitionState(left), 2);
        counts.insert(RepetitionState(right), 7);
        assert_eq!(counts.len(), 2, "{right_fen}");
        assert_eq!(counts.get(&RepetitionState(left)), Some(&2));
        assert_eq!(counts.get(&RepetitionState(right)), Some(&7));
        counts.remove(&RepetitionState(right));
        assert_eq!(counts.get(&RepetitionState(left)), Some(&2));
    }
}
