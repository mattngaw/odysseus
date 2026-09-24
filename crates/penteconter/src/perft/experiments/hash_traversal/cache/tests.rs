use super::*;

#[test]
fn every_layout_uses_the_same_storage_and_reset_preserves_the_allocation() {
    let mut cache = Cache::new(1024);
    let allocation = cache.entries.as_ptr();
    assert_eq!(size_of::<Entry>(), 32);
    for ways in [Associativity::One, Associativity::Two, Associativity::Four] {
        cache.reset(ways);
        assert_eq!(cache.associativity(), ways);
        assert_eq!(cache.bytes(), 32 * 1024);
        assert_eq!(cache.entries.as_ptr(), allocation);
        assert!(cache.entries.iter().all(|entry| entry.depth == 0));
        cache.entries[0] = Entry {
            key: 7,
            depth: 1,
            nodes: 42,
        };
    }
}

fn check_keys_and_depth<const WAYS: usize>(ways: Associativity) {
    let mut cache = Cache::new(WAYS);
    cache.reset(ways);
    assert_eq!(cache.probe::<WAYS>(0, 0), None);
    cache.store::<WAYS>(0, 1, 0);
    assert_eq!(cache.probe::<WAYS>(0, 1), Some(0));
    assert_eq!(cache.probe::<WAYS>(0, 2), None);
    assert_eq!(cache.probe::<WAYS>(1 << 40, 1), None);
    cache.store::<WAYS>(0, 1, 42);
    assert_eq!(cache.probe::<WAYS>(0, 1), Some(42));
    assert_eq!(
        cache
            .entries
            .iter()
            .filter(|entry| entry.depth != 0)
            .count(),
        1
    );
    cache.store::<WAYS>(0, 2, 100);
    assert_eq!(cache.probe::<WAYS>(0, 2), Some(100));
    assert_eq!(
        cache.probe::<WAYS>(0, 1),
        if WAYS == 1 { None } else { Some(42) }
    );
    cache.clear();
    assert_eq!(cache.probe::<WAYS>(0, 1), None);
    assert_eq!(cache.probe::<WAYS>(0, 2), None);
}

#[test]
fn all_ways_require_full_key_and_exact_depth_and_accept_zero_counts() {
    check_keys_and_depth::<1>(Associativity::One);
    check_keys_and_depth::<2>(Associativity::Two);
    check_keys_and_depth::<4>(Associativity::Four);
}

fn check_replacement<const WAYS: usize>(ways: Associativity) {
    let mut cache = Cache::new(WAYS * 2);
    cache.reset(ways);
    // Even keys contend for bucket zero; odd keys occupy a separate bucket.
    cache.store::<WAYS>(1, 1, 99);
    for i in 0..WAYS {
        cache.store::<WAYS>((i * 2) as u64, 2, i as u64);
    }
    for i in 0..WAYS {
        assert_eq!(cache.probe::<WAYS>((i * 2) as u64, 2), Some(i as u64));
    }
    // All depths tie, so a new key replaces the first entry only.
    cache.store::<WAYS>(100, 1, 1000);
    assert_eq!(cache.probe::<WAYS>(0, 2), None);
    assert_eq!(cache.probe::<WAYS>(100, 1), Some(1000));
    // The shallower entry is next to go, for another equally shallow result.
    cache.store::<WAYS>(102, 1, 2000);
    assert_eq!(cache.probe::<WAYS>(100, 1), None);
    for i in 1..WAYS {
        assert_eq!(cache.probe::<WAYS>((i * 2) as u64, 2), Some(i as u64));
    }
    assert_eq!(cache.probe::<WAYS>(1, 1), Some(99));

    // Also place the unique shallowest result at the end of a full bucket.
    cache.clear();
    for i in 0..WAYS {
        cache.store::<WAYS>((i * 2) as u64, (WAYS - i) as u8, i as u64);
    }
    cache.store::<WAYS>(100, 1, 42);
    assert_eq!(cache.probe::<WAYS>(((WAYS - 1) * 2) as u64, 1), None);
    for i in 0..WAYS - 1 {
        assert_eq!(
            cache.probe::<WAYS>((i * 2) as u64, (WAYS - i) as u8),
            Some(i as u64)
        );
    }
}

#[test]
fn contested_buckets_fill_then_replace_shallowest_with_deterministic_ties() {
    check_replacement::<1>(Associativity::One);
    check_replacement::<2>(Associativity::Two);
    check_replacement::<4>(Associativity::Four);
}

#[test]
fn updating_an_existing_deeper_entry_does_not_evict_a_shallower_neighbor() {
    let mut cache = Cache::new(4);
    cache.reset(Associativity::Four);
    cache.store::<4>(1, 1, 10);
    cache.store::<4>(2, 4, 20);
    cache.store::<4>(2, 4, 30);
    assert_eq!(cache.probe::<4>(1, 1), Some(10));
    assert_eq!(cache.probe::<4>(2, 4), Some(30));
    assert_eq!(
        cache
            .entries
            .iter()
            .filter(|entry| entry.depth != 0)
            .count(),
        2
    );
}

#[test]
#[should_panic]
fn reset_rejects_more_ways_than_entries() {
    Cache::new(2).reset(Associativity::Four);
}
