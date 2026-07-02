//! Minimal deterministic RNG (SplitMix64). No external dependency so solver
//! results are reproducible across platforms for a given seed.

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Derive an independent stream from a seed and arbitrary stream tags.
    /// Used to give each (round, segment) pair its own deterministic RNG.
    pub fn derive(seed: u64, tag_a: u64, tag_b: u64) -> Self {
        let mut rng = Self::new(seed ^ tag_a.rotate_left(17) ^ tag_b.rotate_left(41));
        // Burn a few outputs so nearby tags decorrelate.
        rng.next_u64();
        rng.next_u64();
        rng
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `0..bound` (bound must be > 0).
    pub fn next_below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        // 128-bit multiply avoids modulo bias well enough for heuristics.
        let wide = u128::from(self.next_u64()) * bound as u128;
        (wide >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn deterministic_for_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn next_below_stays_in_range() {
        let mut rng = SplitMix64::new(7);
        for bound in [1usize, 2, 3, 10, 1000] {
            for _ in 0..100 {
                assert!(rng.next_below(bound) < bound);
            }
        }
    }

    #[test]
    fn derived_streams_differ() {
        let mut a = SplitMix64::derive(1, 0, 0);
        let mut b = SplitMix64::derive(1, 0, 1);
        let same = (0..8).filter(|_| a.next_u64() == b.next_u64()).count();
        assert!(same < 8);
    }
}
