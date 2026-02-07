pub(crate) const PROBLEM_FILE_TSP: &str = "problem.tsp";
pub(crate) const PROBLEM_FILE_CANDIDATE: &str = "problem.cand";
pub(crate) const PROBLEM_FILE_PI: &str = "problem.pi";
pub(crate) const PROBLEM_NAME: &str = "problem";

pub(crate) const MIN_CYCLE_POINTS: usize = 3;
pub(crate) const PLANE_PROJECTION_RADIUS: f64 = 70.0;

pub(crate) const THREAD_FALLBACK_PARALLELISM: usize = 2;
pub(crate) const THREAD_MIN_PARALLELISM: usize = 2;
pub(crate) const THREAD_RESERVED_CORES: usize = 1;

pub(crate) const PREP_CANDIDATES_PAR_FILE: &str = "prep_candidates.par";
pub(crate) const PREP_CANDIDATES_TOUR_FILE: &str = "prep_candidates.tour";

pub(crate) const RUN_FILE_PREFIX: &str = "run_";
pub(crate) const RUN_PAR_EXTENSION: &str = ".par";
pub(crate) const RUN_TOUR_EXTENSION: &str = ".tour";

pub(crate) const CENTROIDS_TSP_FILE: &str = "centroids.tsp";
pub(crate) const CENTROIDS_PAR_FILE: &str = "centroids.par";
pub(crate) const CENTROIDS_TOUR_FILE: &str = "centroids.tour";
pub(crate) const CENTROIDS_NAME: &str = "centroids";

pub(crate) const CHUNK_DIR_PREFIX: &str = "chunk_";
pub(crate) const CHUNK_ORDER_DIR: &str = "chunk_order";

pub(crate) fn run_par_file(idx: usize) -> String {
    format!("{RUN_FILE_PREFIX}{idx}{RUN_PAR_EXTENSION}")
}

pub(crate) fn run_tour_file(idx: usize) -> String {
    format!("{RUN_FILE_PREFIX}{idx}{RUN_TOUR_EXTENSION}")
}
