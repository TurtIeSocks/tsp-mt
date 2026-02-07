use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Clone, Debug)]
pub(crate) struct LkhConfig {
    /// Trials per process.
    max_trials: usize,
    /// LKH RUNS per process.
    runs: usize,
    /// Base seed used to generate per-run seeds.
    base_seed: u64,
    /// LKH TRACE_LEVEL.
    trace_level: usize,
    /// Seconds to run LKH.
    time_limit: usize,
    max_candidates: usize,
}

impl LkhConfig {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            max_trials: (n * 3).max(1_000).min(100_000),
            time_limit: (n / 512).max(2),
            ..Default::default()
        }
    }

    pub(crate) fn preprocessing(n: usize) -> Self {
        Self {
            max_trials: n,
            runs: 1,
            time_limit: 1,
            ..Default::default()
        }
    }

    pub(crate) fn base_seed(&self) -> u64 {
        self.base_seed
    }

    pub(crate) fn time_limit(&self) -> usize {
        self.time_limit
    }

    /// Deterministic seed generation; replace with your favorite scheme.
    /// Produces `count` distinct u64 seeds from `base_seed`.
    pub(crate) fn generate_seeds(&self, count: usize) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(self.base_seed);
        (0..count).map(|_| rng.random::<u64>()).collect()
    }

    pub(crate) fn param_file(&self) -> String {
        format!(
            "\
RUNS = {}
MAX_TRIALS = {}
TRACE_LEVEL = {}
TIME_LIMIT = {}
MAX_CANDIDATES = {} SYMMETRIC
",
            self.runs, self.max_trials, self.trace_level, self.time_limit, self.max_candidates
        )
    }
}

impl Default for LkhConfig {
    fn default() -> Self {
        Self {
            max_trials: 10000,
            runs: 1,
            base_seed: 12345,
            trace_level: 1,
            time_limit: 60,
            max_candidates: 32,
        }
    }
}
