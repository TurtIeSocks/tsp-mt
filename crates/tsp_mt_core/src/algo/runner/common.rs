use std::{fs, path::Path, thread, time::Instant};

use lkh::{
    LkhError, LkhResult,
    parameters::{CandidateLimit, LkhParameters},
    problem::{EdgeWeightType, NodeCoord, TsplibProblem, TsplibProblemType},
    solver::LkhSolver,
    tour::{TsplibTour, TsplibTourType},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{LKHNode, SolverInput, SolverOptions, h3_chunking, stitching};

pub(super) const MIN_CYCLE_POINTS: usize = 3;
pub(super) const DEFAULT_BASE_SEED: u64 = 12_345;
pub(super) const MULTI_SEED_TRACE_LEVEL: usize = 1;
pub(super) const CENTROID_TRACE_LEVEL: usize = 0;
pub(super) const MAX_CENTROIDS_WITH_TRIVIAL_ORDER: usize = 2;
pub(super) const ERR_NO_RESULTS: &str = "No results";

const DEFAULT_MAX_CANDIDATES: usize = 32;
const SINGLE_RUNS: usize = 1;
const MAX_TRIALS_MULTIPLIER: usize = 3;
const MIN_MAX_TRIALS: usize = 1_000;
const MAX_MAX_TRIALS: usize = 100_000;
const TIME_LIMIT_DIVISOR: usize = 1024;
const MIN_TIME_LIMIT_SECONDS: usize = 2;
const THREAD_FALLBACK_PARALLELISM: usize = 2;
const THREAD_RESERVED_CORES: usize = 1;
const ROUNDER_FACTOR: f64 = 1000.0;

const STITCH_CANDIDATE_CHECK_LIMIT: usize = 200;
const STITCH_PORTAL_NODE_FLOOR: usize = 12;
const STITCH_PORTAL_NODE_CAP: usize = 48;
const STITCH_PORTAL_PAIR_CHECK_LIMIT: usize = 180;
const STITCH_LARGE_JUMP_DISTANCE_THRESHOLD: f64 = 900.0;
const STITCH_LARGE_JUMP_PENALTY: f64 = 700.0;
const STITCH_VERY_LARGE_JUMP_DISTANCE_THRESHOLD: f64 = 2_200.0;
const STITCH_VERY_LARGE_JUMP_PENALTY: f64 = 1_600.0;
const STITCH_NON_NEIGHBOR_BRIDGE_PENALTY: f64 = 350.0;
const STITCH_NON_NEIGHBOR_SELECTION_PENALTY_MULTIPLIER: f64 = 2.5;
const STITCH_LONG_EDGE_BOUNDARY_MULTIPLIER: f64 = 2.0;
const STITCH_LONG_EDGE_BOUNDARY_LIMIT: usize = 16;
const INITIAL_TOUR_FILE_NAME: &str = "initial.tour";

const ERR_INVALID_POINT: &str = "Input contains invalid lat/lng values";
const ERR_INVALID_PROJECTION_RADIUS: &str = "projection_radius must be > 0";
const ERR_INVALID_MAX_CHUNK_SIZE: &str = "max_chunk_size must be > 0";
const ERR_INVALID_SPIKE_REPAIR_WINDOW: &str =
    "spike_repair_window must be > 0 when spike_repair_top_n > 0";
const ERR_INVALID_SPIKE_REPAIR_PASSES: &str =
    "spike_repair_passes must be > 0 when spike_repair_top_n > 0";
const ERR_INVALID_OUTLIER_THRESHOLD: &str = "outlier_threshold must be > 0";

pub(super) fn build_problem(points: &[LKHNode]) -> TsplibProblem {
    TsplibProblem::new(TsplibProblemType::Tsp)
        .with_node_coord_section(
            points
                .iter()
                .enumerate()
                .map(|(idx, n)| {
                    NodeCoord::twod(
                        idx + 1,
                        (n.y * ROUNDER_FACTOR).round(),
                        (n.x * ROUNDER_FACTOR).round(),
                    )
                })
                .collect::<Vec<_>>(),
        )
        .with_dimension(points.len())
        .with_edge_weight_type(EdgeWeightType::Euc2d)
}

pub(super) fn seeded_params(
    problem_file: &Path,
    seed: u64,
    max_trials: usize,
    time_limit: f64,
    trace_level: usize,
) -> LkhParameters {
    LkhParameters::new(problem_file)
        .with_max_candidates(CandidateLimit::new(DEFAULT_MAX_CANDIDATES, true))
        .with_max_trials(max_trials)
        .with_runs(SINGLE_RUNS)
        .with_seed(seed)
        .with_time_limit(time_limit)
        .with_trace_level(trace_level)
}

pub(super) fn maybe_attach_initial_tour_file(
    params: &mut LkhParameters,
    problem_file: &Path,
    node_count: usize,
    enable: bool,
) -> LkhResult<()> {
    if !enable {
        return Ok(());
    }
    let parent = problem_file.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let path = parent.join(INITIAL_TOUR_FILE_NAME);
    write_initial_tour_file(&path, node_count)?;
    params.initial_tour_file = Some(path);
    Ok(())
}

fn write_initial_tour_file(path: &Path, node_count: usize) -> LkhResult<()> {
    let mut tour = TsplibTour::new();
    tour.tour_type = Some(TsplibTourType::Tour);
    tour.dimension = Some(node_count);
    tour.tour_section = (1..=node_count).collect();
    tour.emit_eof = true;
    tour.write_to_file(path)
}

pub(super) fn available_seed_runs() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(THREAD_FALLBACK_PARALLELISM)
        .saturating_sub(THREAD_RESERVED_CORES)
        .max(1)
}

pub(super) fn scaled_max_trials(n: usize) -> usize {
    (n * MAX_TRIALS_MULTIPLIER).clamp(MIN_MAX_TRIALS, MAX_MAX_TRIALS)
}

pub(super) fn scaled_time_limit_seconds(n: usize) -> f64 {
    ((n / TIME_LIMIT_DIVISOR).max(MIN_TIME_LIMIT_SECONDS)) as f64
}

pub(super) fn generate_seeds(base_seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(base_seed);
    (0..count).map(|_| rng.random::<u64>()).collect()
}

pub(super) fn canonicalize_cycle_order(order: Vec<usize>) -> Vec<usize> {
    if order.len() <= 1 {
        return order;
    }

    let rotate_to_min = |src: &[usize]| -> Vec<usize> {
        let Some((min_pos, _)) = src.iter().enumerate().min_by_key(|(_, value)| **value) else {
            return Vec::new();
        };
        let mut rotated = Vec::with_capacity(src.len());
        rotated.extend_from_slice(&src[min_pos..]);
        rotated.extend_from_slice(&src[..min_pos]);
        rotated
    };

    let forward = rotate_to_min(&order);
    let mut reversed = order;
    reversed.reverse();
    let reversed = rotate_to_min(&reversed);

    if reversed < forward {
        reversed
    } else {
        forward
    }
}

pub(super) fn cycle_length(points: &[LKHNode], tour: &[usize]) -> f64 {
    (0..tour.len())
        .map(|idx| {
            let a = points[tour[idx]];
            let b = points[tour[(idx + 1) % tour.len()]];
            a.dist(&b)
        })
        .sum()
}

pub(super) fn validate_min_cycle_points(input: &SolverInput) -> LkhResult<()> {
    if input.points_len() < MIN_CYCLE_POINTS {
        return Err(LkhError::invalid_input(format!(
            "Need at least {MIN_CYCLE_POINTS} points for a cycle"
        )));
    }
    Ok(())
}

pub(super) fn validate_points(input: &SolverInput) -> LkhResult<()> {
    if input.nodes.iter().any(|point| !point.is_valid()) {
        return Err(LkhError::invalid_input(ERR_INVALID_POINT));
    }
    Ok(())
}

pub(super) fn validate_projection_radius(options: &SolverOptions) -> LkhResult<()> {
    if options.projection_radius <= 0.0 {
        return Err(LkhError::invalid_input(ERR_INVALID_PROJECTION_RADIUS));
    }
    Ok(())
}

pub(super) fn validate_chunk_size(options: &SolverOptions) -> LkhResult<()> {
    if options.max_chunk_size == 0 {
        return Err(LkhError::invalid_input(ERR_INVALID_MAX_CHUNK_SIZE));
    }
    Ok(())
}

pub(super) fn validate_outlier_threshold(options: &SolverOptions) -> LkhResult<()> {
    if options.outlier_threshold <= 0.0 {
        return Err(LkhError::invalid_input(ERR_INVALID_OUTLIER_THRESHOLD));
    }
    Ok(())
}

pub(super) fn solve_chunk_indices(
    chunk_id: usize,
    idxs: &[usize],
    projected_points: &[LKHNode],
    work_dir: &Path,
    lkh_exe: &Path,
    use_initial_tour: bool,
) -> LkhResult<Vec<usize>> {
    if idxs.len() < MIN_CYCLE_POINTS {
        return Ok(idxs.to_vec());
    }

    let chunk_points: Vec<LKHNode> = idxs.iter().map(|&idx| projected_points[idx]).collect();
    let problem_file = work_dir
        .join(format!("chunk_{chunk_id}"))
        .join("problem.tsp");
    let mut params = seeded_params(
        &problem_file,
        DEFAULT_BASE_SEED,
        scaled_max_trials(chunk_points.len()),
        scaled_time_limit_seconds(chunk_points.len()),
        MULTI_SEED_TRACE_LEVEL,
    );
    maybe_attach_initial_tour_file(
        &mut params,
        &problem_file,
        chunk_points.len(),
        use_initial_tour,
    )?;
    let solver = LkhSolver::new(build_problem(&chunk_points), params)?;

    let now = Instant::now();
    let tour = solver.run_with_exe(lkh_exe)?;
    log::info!(
        "chunk: done id={chunk_id} n={} secs={:.2}",
        idxs.len(),
        now.elapsed().as_secs_f32()
    );

    let local_order = tour.zero_based_tour()?;
    Ok(local_order
        .into_iter()
        .map(|local_idx| idxs[local_idx])
        .collect())
}

pub(super) fn build_node_to_chunk_map(
    node_count: usize,
    chunk_partitions: &[h3_chunking::ChunkPartition],
) -> Vec<usize> {
    let mut node_to_chunk: Vec<usize> = vec![usize::MAX; node_count];
    for (chunk_idx, chunk) in chunk_partitions.iter().enumerate() {
        for &node_idx in &chunk.indices {
            if node_idx < node_to_chunk.len() {
                node_to_chunk[node_idx] = chunk_idx;
            }
        }
    }
    node_to_chunk
}

pub(super) fn build_stitching_tuning(
    options: &SolverOptions,
) -> LkhResult<stitching::StitchingTuning> {
    if options.spike_repair_top_n > 0 && options.spike_repair_window == 0 {
        return Err(LkhError::invalid_input(ERR_INVALID_SPIKE_REPAIR_WINDOW));
    }
    if options.spike_repair_top_n > 0 && options.spike_repair_passes == 0 {
        return Err(LkhError::invalid_input(ERR_INVALID_SPIKE_REPAIR_PASSES));
    }

    Ok(stitching::StitchingTuning {
        candidate_pair_check_limit: STITCH_CANDIDATE_CHECK_LIMIT,
        portal_node_floor: STITCH_PORTAL_NODE_FLOOR,
        portal_node_cap: STITCH_PORTAL_NODE_CAP,
        portal_pair_check_limit: STITCH_PORTAL_PAIR_CHECK_LIMIT,
        large_jump_distance_threshold: STITCH_LARGE_JUMP_DISTANCE_THRESHOLD,
        large_jump_penalty: STITCH_LARGE_JUMP_PENALTY,
        very_large_jump_distance_threshold: STITCH_VERY_LARGE_JUMP_DISTANCE_THRESHOLD,
        very_large_jump_penalty: STITCH_VERY_LARGE_JUMP_PENALTY,
        non_neighbor_bridge_penalty: STITCH_NON_NEIGHBOR_BRIDGE_PENALTY,
        non_neighbor_selection_penalty_multiplier: STITCH_NON_NEIGHBOR_SELECTION_PENALTY_MULTIPLIER,
        long_edge_boundary_multiplier: STITCH_LONG_EDGE_BOUNDARY_MULTIPLIER,
        long_edge_boundary_limit: STITCH_LONG_EDGE_BOUNDARY_LIMIT,
        spike_repair_top_n: options.spike_repair_top_n,
        spike_repair_window: options.spike_repair_window,
        spike_repair_passes: options.spike_repair_passes,
    })
}

#[cfg(test)]
mod tests {
    use super::canonicalize_cycle_order;

    #[test]
    fn canonicalize_cycle_order_normalizes_rotation_and_direction() {
        let a = canonicalize_cycle_order(vec![2, 3, 4, 1, 0]);
        let b = canonicalize_cycle_order(vec![0, 2, 3, 4, 1]);
        let c = canonicalize_cycle_order(vec![0, 1, 4, 3, 2]);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a[0], 0);
    }
}
