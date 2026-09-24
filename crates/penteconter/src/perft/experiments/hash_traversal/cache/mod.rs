//! Equal-capacity perft caches with one, two, or four entries per bucket.

/// Number of candidate entries examined for a key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Associativity {
    One,
    Two,
    Four,
}

impl Associativity {
    pub const fn ways(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
        }
    }
}

// Preserve the baseline's size and alignment for every configuration. Buckets
// are contiguous but have no additional cache-line alignment guarantee.
// Depth zero marks an empty entry and is never cached.
#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    nodes: u64,
    depth: u8,
}

/// Bounded perft cache, with a full 64-bit key and exact remaining depth.
///
/// Storage replaces an existing key/depth match, otherwise the first empty
/// entry, otherwise the shallowest entry (ties choose the first slot). A new
/// entry is always admitted. One way reduces to direct-mapped always-replace.
///
/// This is a probabilistic experiment: full-key Zobrist collisions are not
/// resolved by state comparison. It ignores draws and search bounds.
pub struct Cache {
    entries: Vec<Entry>,
    associativity: Associativity,
}

impl Cache {
    /// Creates a direct-mapped cache. `entries` must be a nonzero power of two;
    /// each entry occupies 32 bytes. Allocation occurs only at construction.
    pub fn new(entries: usize) -> Self {
        assert!(entries.is_power_of_two());
        Self {
            entries: vec![Entry::default(); entries],
            associativity: Associativity::One,
        }
    }

    /// Allocated entry storage, excluding this small owning handle.
    pub fn bytes(&self) -> usize {
        self.entries.len() * size_of::<Entry>()
    }

    pub const fn associativity(&self) -> Associativity {
        self.associativity
    }

    pub fn clear(&mut self) {
        self.entries.fill(Entry::default());
    }

    /// Clears all entries and selects a bucket width, reusing the allocation.
    /// Panics before changing the cache if it has fewer entries than ways.
    pub fn reset(&mut self, associativity: Associativity) {
        assert!(self.entries.len() >= associativity.ways());
        self.clear();
        self.associativity = associativity;
    }

    // WAYS is selected once at traversal entry, allowing specialized probes.
    fn bucket<const WAYS: usize>(&self, key: u64) -> usize {
        debug_assert_eq!(WAYS, self.associativity.ways());
        (key as usize & (self.entries.len() / WAYS - 1)) * WAYS
    }

    pub(super) fn probe<const WAYS: usize>(&self, key: u64, depth: u8) -> Option<u64> {
        let start = self.bucket::<WAYS>(key);
        for entry in &self.entries[start..start + WAYS] {
            if entry.depth != 0 && entry.key == key && entry.depth == depth {
                return Some(entry.nodes);
            }
        }
        None
    }

    pub(super) fn store<const WAYS: usize>(&mut self, key: u64, depth: u8, nodes: u64) {
        debug_assert_ne!(depth, 0);
        let start = self.bucket::<WAYS>(key);
        let bucket = &mut self.entries[start..start + WAYS];
        let mut victim = 0;
        for index in 0..WAYS {
            let entry = &bucket[index];
            if entry.depth == depth && entry.key == key {
                victim = index;
                break;
            }
            // Empty entries have depth zero, so the same minimum also selects
            // the first empty entry. Still inspect later slots for a match.
            if entry.depth < bucket[victim].depth {
                victim = index;
            }
        }
        bucket[victim] = Entry { key, nodes, depth };
    }
}

#[cfg(test)]
mod tests;
