/// Runtime options for LKH solving behavior.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    /// Radius used by local tangent-plane projection (in meters).
    pub projection_radius: f64,
    /// Maximum number of points per H3 chunk before hierarchical chunking is applied.
    pub max_chunk_size: usize,
    /// Random seed used when ordering chunk centroids with LKH.
    pub centroid_order_seed: u64,
    /// `MAX_TRIALS` for centroid-ordering LKH run.
    pub centroid_order_max_trials: usize,
    /// `TIME_LIMIT` (seconds) for centroid-ordering LKH run.
    pub centroid_order_time_limit: usize,
    /// Window size for boundary-local 2-opt refinement after chunk stitching.
    pub boundary_2opt_window: usize,
    /// Number of passes for boundary-local 2-opt refinement.
    pub boundary_2opt_passes: usize,
    /// Emit progress logs to stderr when true.
    pub verbose: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            projection_radius: 70.0,
            max_chunk_size: 5_000,
            centroid_order_seed: 999,
            centroid_order_max_trials: 20_000,
            centroid_order_time_limit: 10,
            boundary_2opt_window: 500,
            boundary_2opt_passes: 50,
            verbose: true,
        }
    }
}
