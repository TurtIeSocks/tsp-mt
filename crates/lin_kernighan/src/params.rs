use std::time::Duration;

/// Solver knobs.
///
/// This is the narrow surface tsp_mt_core actually uses, deliberately
/// avoiding LKH's 87-field parameter struct. New knobs are only added when
/// a real call site needs them.
#[derive(Clone, Debug)]
pub struct Params {
    /// Maximum candidate edges per node (LKH `MAX_CANDIDATES SYMMETRIC`).
    pub max_candidates: usize,
    /// Maximum LK trials per run (LKH `MAX_TRIALS`).
    pub max_trials: usize,
    /// Wall-clock budget per run (LKH `TIME_LIMIT`).
    pub time_limit: Duration,
    /// Random seed for reproducible runs.
    pub seed: u64,
    /// Verbosity: 0=quiet, 1=normal progress logs.
    pub trace_level: usize,
    /// Maximum k for sequential k-opt moves (LKH `MOVE_TYPE`).
    /// LKH default is 5; we ship k≤3 for now.
    pub move_type: usize,
    /// Initial tour to seed the search with (LKH `INITIAL_TOUR_FILE`).
    /// `None` ⇒ build a greedy initial tour from the candidate set.
    pub initial_tour: Option<Vec<usize>>,
    /// Stop the trial loop early after this many consecutive double-bridge
    /// kicks fail to improve the best tour. Roughly approximates LKH's
    /// "no improvement seen in N trials" termination — keeps lin_kernighan
    /// from burning its full `time_limit` once the tour has plateaued.
    pub max_no_improvement: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_candidates: 5,
            max_trials: 1_000,
            time_limit: Duration::from_secs(10),
            seed: 1,
            trace_level: 0,
            move_type: 3,
            initial_tour: None,
            max_no_improvement: 500,
        }
    }
}

impl Params {
    pub fn with_max_candidates(mut self, v: usize) -> Self {
        self.max_candidates = v;
        self
    }
    pub fn with_max_trials(mut self, v: usize) -> Self {
        self.max_trials = v;
        self
    }
    pub fn with_time_limit(mut self, v: Duration) -> Self {
        self.time_limit = v;
        self
    }
    pub fn with_seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }
    pub fn with_trace_level(mut self, v: usize) -> Self {
        self.trace_level = v;
        self
    }
    pub fn with_move_type(mut self, v: usize) -> Self {
        self.move_type = v;
        self
    }
    pub fn with_initial_tour(mut self, v: Vec<usize>) -> Self {
        self.initial_tour = Some(v);
        self
    }
    pub fn with_max_no_improvement(mut self, v: usize) -> Self {
        self.max_no_improvement = v;
        self
    }
}
