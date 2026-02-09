use std::{path::Path, thread, time::Instant};

use lkh::{
    LkhError, LkhResult,
    parameters::{CandidateLimit, LkhParameters},
    problem::{EdgeWeightType, NodeCoord, TsplibProblem, TsplibProblemType},
    solver::LkhSolver,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::{
    LKHNode, SolverInput, SolverOptions, constants::MIN_CYCLE_POINTS, geometry, h3_chunking,
    projection::PlaneProjection, stitching,
};

const DEFAULT_MAX_CANDIDATES: usize = 32;
const DEFAULT_BASE_SEED: u64 = 12_345;
const SINGLE_RUNS: usize = 1;
const MULTI_SEED_TRACE_LEVEL: usize = 1;
const CENTROID_TRACE_LEVEL: usize = 0;
const MAX_TRIALS_MULTIPLIER: usize = 3;
const MIN_MAX_TRIALS: usize = 1_000;
const MAX_MAX_TRIALS: usize = 100_000;
const TIME_LIMIT_DIVISOR: usize = 128;
const MIN_TIME_LIMIT_SECONDS: usize = 2;
const THREAD_FALLBACK_PARALLELISM: usize = 2;
const THREAD_RESERVED_CORES: usize = 1;
const MAX_CENTROIDS_WITH_TRIVIAL_ORDER: usize = 2;

const ERR_NO_RESULTS: &str = "No results";
const ERR_INVALID_POINT: &str = "Input contains invalid lat/lng values";
const ERR_INVALID_PROJECTION_RADIUS: &str = "projection_radius must be > 0";
const ERR_INVALID_MAX_CHUNK_SIZE: &str = "max_chunk_size must be > 0";

fn build_problem(points: &[LKHNode]) -> TsplibProblem {
    TsplibProblem::new(TsplibProblemType::Tsp)
        .with_node_coord_section(
            points
                .iter()
                .enumerate()
                .map(|(idx, n)| {
                    NodeCoord::twod(idx + 1, (n.y * 1000.0).round(), (n.x * 1000.0).round())
                })
                .collect::<Vec<_>>(),
        )
        .with_dimension(points.len())
        .with_edge_weight_type(EdgeWeightType::Euc2d)
}

fn seeded_params(
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

fn available_seed_runs() -> usize {
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(THREAD_FALLBACK_PARALLELISM)
        .saturating_sub(THREAD_RESERVED_CORES)
        .max(1)
}

fn scaled_max_trials(n: usize) -> usize {
    (n * MAX_TRIALS_MULTIPLIER).clamp(MIN_MAX_TRIALS, MAX_MAX_TRIALS)
}

fn scaled_time_limit_seconds(n: usize) -> f64 {
    ((n / TIME_LIMIT_DIVISOR).max(MIN_TIME_LIMIT_SECONDS)) as f64
}

fn generate_seeds(base_seed: u64, count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(base_seed);
    (0..count).map(|_| rng.random::<u64>()).collect()
}

fn cycle_length(points: &[LKHNode], tour: &[usize]) -> f64 {
    (0..tour.len())
        .map(|idx| {
            let a = points[tour[idx]];
            let b = points[tour[(idx + 1) % tour.len()]];
            a.dist(&b)
        })
        .sum()
}

fn validate_min_cycle_points(input: &SolverInput) -> LkhResult<()> {
    if input.points_len() < MIN_CYCLE_POINTS {
        return Err(LkhError::invalid_input(format!(
            "Need at least {MIN_CYCLE_POINTS} points for a cycle"
        )));
    }
    Ok(())
}

fn validate_points(input: &SolverInput) -> LkhResult<()> {
    if input.nodes.iter().any(|point| !point.is_valid()) {
        return Err(LkhError::invalid_input(ERR_INVALID_POINT));
    }
    Ok(())
}

fn validate_projection_radius(options: &SolverOptions) -> LkhResult<()> {
    if options.projection_radius <= 0.0 {
        return Err(LkhError::invalid_input(ERR_INVALID_PROJECTION_RADIUS));
    }
    Ok(())
}

fn validate_chunk_size(options: &SolverOptions) -> LkhResult<()> {
    if options.max_chunk_size == 0 {
        return Err(LkhError::invalid_input(ERR_INVALID_MAX_CHUNK_SIZE));
    }
    Ok(())
}

#[tsp_mt_derive::timer()]
pub fn lkh_single(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    validate_min_cycle_points(&input)?;
    validate_points(&input)?;

    let problem_file_path = options.work_dir.join("problem.tsp");

    let tour = LkhSolver::new(
        build_problem(&input.nodes),
        LkhParameters::new(&problem_file_path)
            .with_max_candidates(CandidateLimit::new(DEFAULT_MAX_CANDIDATES, true))
            .with_max_trials(scaled_max_trials(input.n()))
            .with_runs(SINGLE_RUNS)
            .with_seed(DEFAULT_BASE_SEED)
            .with_time_limit(scaled_time_limit_seconds(input.n()))
            .with_trace_level(MULTI_SEED_TRACE_LEVEL),
    )?
    .run()?;

    Ok(tour
        .zero_based_tour()?
        .into_iter()
        .map(|idx| input.get_point(idx))
        .collect())
}

#[tsp_mt_derive::timer()]
pub fn lkh_multi_seed(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    validate_min_cycle_points(&input)?;
    validate_points(&input)?;

    let points = input.nodes;
    let projected_points = PlaneProjection::new(&points)
        .radius(options.projection_radius)
        .project();
    let run_count = available_seed_runs();
    let max_trials = scaled_max_trials(points.len());
    let time_limit = scaled_time_limit_seconds(points.len());
    let seeds = generate_seeds(DEFAULT_BASE_SEED, run_count);
    let work_dir = options.work_dir.clone();

    let run_results: LkhResult<Vec<(Vec<usize>, f64)>> = seeds
        .into_par_iter()
        .enumerate()
        .map(|(run_idx, seed)| -> LkhResult<(Vec<usize>, f64)> {
            let run_dir = work_dir.join(format!("seed_{run_idx}"));
            let problem_file = run_dir.join("problem.tsp");
            let tour = LkhSolver::new(
                build_problem(&projected_points),
                seeded_params(
                    &problem_file,
                    seed,
                    max_trials,
                    time_limit,
                    MULTI_SEED_TRACE_LEVEL,
                ),
            )?
            .run()?;

            let order = tour.zero_based_tour()?;
            let length = cycle_length(&points, &order);
            Ok((order, length))
        })
        .collect();

    let best_tour = run_results?
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| LkhError::other(ERR_NO_RESULTS))?
        .0;

    Ok(best_tour.into_iter().map(|idx| points[idx]).collect())
}

#[tsp_mt_derive::timer()]
pub fn lkh_multi_parallel(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    validate_chunk_size(&options)?;
    validate_projection_radius(&options)?;
    validate_min_cycle_points(&input)?;
    validate_points(&input)?;

    if input.points_len() <= options.max_chunk_size {
        return lkh_multi_seed(input, options);
    }

    let points = input.nodes;
    let projected_points = PlaneProjection::new(&points)
        .radius(options.projection_radius)
        .project();
    let chunks = h3_chunking::partition_indices(&points, options.max_chunk_size)
        .map_err(|err| LkhError::other(err.to_string()))?;

    log::info!(
        "chunker: partitioned n={} chunks={} max_chunk_size={}",
        points.len(),
        chunks.len(),
        options.max_chunk_size
    );

    let work_dir = options.work_dir.clone();
    let solved_chunk_tours: LkhResult<Vec<Vec<usize>>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, idxs)| -> LkhResult<Vec<usize>> {
            if idxs.len() < MIN_CYCLE_POINTS {
                return Ok(idxs.to_vec());
            }

            let chunk_points: Vec<LKHNode> =
                idxs.iter().map(|&idx| projected_points[idx]).collect();
            let solver = LkhSolver::new(
                build_problem(&chunk_points),
                seeded_params(
                    &work_dir
                        .join(format!("chunk_{chunk_id}"))
                        .join("problem.tsp"),
                    DEFAULT_BASE_SEED,
                    scaled_max_trials(chunk_points.len()),
                    scaled_time_limit_seconds(chunk_points.len()),
                    MULTI_SEED_TRACE_LEVEL,
                ),
            )?;

            let now = Instant::now();
            let tour = solver.run_with_exe(&options.lkh_exe)?;
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
        })
        .collect();

    let solved_chunk_tours = solved_chunk_tours?;
    let centroids: Vec<LKHNode> = chunks
        .iter()
        .map(|idxs| geometry::centroid_of_indices(&points, idxs))
        .collect();
    let projected_centroids = PlaneProjection::new(&centroids)
        .radius(options.projection_radius)
        .project();

    let chunk_order = if centroids.len() <= MAX_CENTROIDS_WITH_TRIVIAL_ORDER {
        (0..centroids.len()).collect::<Vec<_>>()
    } else {
        let order_problem_file = work_dir.join("chunk_order").join("problem.tsp");
        let order_tour = LkhSolver::new(
            build_problem(&projected_centroids),
            seeded_params(
                &order_problem_file,
                options.centroid_order_seed,
                options.centroid_order_max_trials,
                options.centroid_order_time_limit as f64,
                CENTROID_TRACE_LEVEL,
            ),
        )?
        .run()?;

        order_tour.zero_based_tour()?
    };

    let ordered_chunk_tours: LkhResult<Vec<Vec<usize>>> = chunk_order
        .into_iter()
        .map(|chunk_idx| {
            solved_chunk_tours.get(chunk_idx).cloned().ok_or_else(|| {
                LkhError::invalid_data(format!(
                    "Chunk order index {chunk_idx} out of bounds for {} chunks",
                    solved_chunk_tours.len()
                ))
            })
        })
        .collect();

    let (mut merged_tour, boundaries) =
        stitching::stitch_chunk_tours_dense(&projected_points, ordered_chunk_tours?);
    stitching::boundary_two_opt(
        &projected_points,
        &mut merged_tour,
        &boundaries,
        options.boundary_2opt_window,
        options.boundary_2opt_passes,
    );

    Ok(merged_tour.into_iter().map(|idx| points[idx]).collect())
}
