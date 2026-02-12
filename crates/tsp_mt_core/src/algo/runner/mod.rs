use lkh::{LkhError, LkhResult, solver::LkhSolver};
use rayon::prelude::*;

use crate::{
    LKHNode, SolverInput, SolverOptions, geometry, h3_chunking, projection::PlaneProjection,
    stitching,
};

mod common;
mod metric_spike;

use common::{
    CENTROID_TRACE_LEVEL, DEFAULT_BASE_SEED, ERR_NO_RESULTS, MAX_CENTROIDS_WITH_TRIVIAL_ORDER,
    MULTI_SEED_TRACE_LEVEL, available_seed_runs, build_problem, build_stitching_tuning,
    canonicalize_cycle_order, cycle_length, generate_seeds, maybe_attach_initial_tour_file,
    scaled_max_trials, scaled_time_limit_seconds, seeded_params, solve_chunk_indices,
    validate_chunk_size, validate_min_cycle_points, validate_outlier_threshold, validate_points,
    validate_projection_radius,
};
use metric_spike::{log_metric_spike_breakdown, repair_metric_spikes_with_global_two_opt};

#[tsp_mt_derive::timer()]
pub fn lkh_single(input: SolverInput, options: SolverOptions) -> LkhResult<Vec<LKHNode>> {
    validate_min_cycle_points(&input)?;
    validate_points(&input)?;

    let problem_file_path = options.work_dir.join("problem.tsp");
    let mut params = seeded_params(
        &problem_file_path,
        DEFAULT_BASE_SEED,
        scaled_max_trials(input.n()),
        scaled_time_limit_seconds(input.n()),
        MULTI_SEED_TRACE_LEVEL,
    );
    maybe_attach_initial_tour_file(
        &mut params,
        &problem_file_path,
        input.n(),
        options.use_initial_tour,
    )?;

    let tour =
        LkhSolver::new(build_problem(&input.nodes), params)?.run_with_exe(&options.lkh_exe)?;

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
            let mut params = seeded_params(
                &problem_file,
                seed,
                max_trials,
                time_limit,
                MULTI_SEED_TRACE_LEVEL,
            );
            maybe_attach_initial_tour_file(
                &mut params,
                &problem_file,
                projected_points.len(),
                options.use_initial_tour,
            )?;
            let tour = LkhSolver::new(build_problem(&projected_points), params)?
                .run_with_exe(&options.lkh_exe)?;

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
    validate_outlier_threshold(&options)?;
    validate_min_cycle_points(&input)?;
    validate_points(&input)?;
    let stitching_tuning = build_stitching_tuning(&options)?;

    if input.points_len() <= options.max_chunk_size {
        return lkh_multi_seed(input, options);
    }

    let points = input.nodes;
    let projected_points = PlaneProjection::new(&points)
        .radius(options.projection_radius)
        .project();
    let chunk_partitions = h3_chunking::partition_with_metadata(&points, options.max_chunk_size)
        .map_err(|err| LkhError::other(err.to_string()))?;

    log::info!(
        "chunker: partitioned n={} chunks={} max_chunk_size={}",
        points.len(),
        chunk_partitions.len(),
        options.max_chunk_size
    );

    let work_dir = options.work_dir.clone();
    let solved_chunk_tours: LkhResult<Vec<Vec<usize>>> = chunk_partitions
        .par_iter()
        .enumerate()
        .map(|(chunk_id, chunk)| {
            solve_chunk_indices(
                chunk_id,
                &chunk.indices,
                &projected_points,
                &work_dir,
                &options.lkh_exe,
                options.use_initial_tour,
            )
        })
        .collect();

    let solved_chunk_tours = solved_chunk_tours?;

    let centroids: Vec<LKHNode> = chunk_partitions
        .iter()
        .map(|chunk| geometry::centroid_of_indices(&points, &chunk.indices))
        .collect();
    let projected_centroids = PlaneProjection::new(&centroids)
        .radius(options.projection_radius)
        .project();

    let chunk_order = if centroids.len() <= MAX_CENTROIDS_WITH_TRIVIAL_ORDER {
        (0..centroids.len()).collect::<Vec<_>>()
    } else {
        let order_problem_file = work_dir.join("chunk_order").join("problem.tsp");
        let mut params = seeded_params(
            &order_problem_file,
            options.centroid_order_seed,
            options.centroid_order_max_trials,
            options.centroid_order_time_limit as f64,
            CENTROID_TRACE_LEVEL,
        );
        maybe_attach_initial_tour_file(
            &mut params,
            &order_problem_file,
            projected_centroids.len(),
            options.use_initial_tour,
        )?;
        let order_tour = LkhSolver::new(build_problem(&projected_centroids), params)?
            .run_with_exe(&options.lkh_exe)?;

        order_tour.zero_based_tour()?
    };
    let chunk_order = canonicalize_cycle_order(chunk_order);

    let ordered_chunks: LkhResult<Vec<(Vec<usize>, stitching::ChunkStitchState)>> = chunk_order
        .into_iter()
        .map(|chunk_idx| {
            let tour = solved_chunk_tours.get(chunk_idx).cloned().ok_or_else(|| {
                LkhError::invalid_data(format!(
                    "Chunk order index {chunk_idx} out of bounds for {} chunks",
                    solved_chunk_tours.len()
                ))
            })?;
            let chunk = chunk_partitions.get(chunk_idx).ok_or_else(|| {
                LkhError::invalid_data(format!(
                    "Chunk partition index {chunk_idx} out of bounds for {} chunks",
                    chunk_partitions.len()
                ))
            })?;
            let centroid = projected_centroids.get(chunk_idx).copied().ok_or_else(|| {
                LkhError::invalid_data(format!(
                    "Chunk centroid index {chunk_idx} out of bounds for {} centroids",
                    projected_centroids.len()
                ))
            })?;

            Ok((
                tour,
                stitching::ChunkStitchState {
                    cell: chunk.cell,
                    resolution: chunk.resolution,
                    centroid,
                    size: chunk.indices.len(),
                },
            ))
        })
        .collect();

    let ordered_chunks = ordered_chunks?;
    let (ordered_chunk_tours, ordered_chunk_states): (
        Vec<Vec<usize>>,
        Vec<stitching::ChunkStitchState>,
    ) = ordered_chunks.into_iter().unzip();

    let (mut merged_tour, boundaries) = stitching::stitch_chunk_tours_dense_with_state(
        &projected_points,
        ordered_chunk_tours,
        ordered_chunk_states,
        stitching_tuning,
    );
    let focused_boundaries = stitching::augment_boundaries_with_long_edges(
        &projected_points,
        &merged_tour,
        &boundaries,
        stitching_tuning,
    );
    stitching::boundary_two_opt(
        &projected_points,
        &mut merged_tour,
        &focused_boundaries,
        options.boundary_2opt_window,
        options.boundary_2opt_passes,
    );
    stitching::spike_repair_two_opt(&projected_points, &mut merged_tour, stitching_tuning);
    repair_metric_spikes_with_global_two_opt(&points, &mut merged_tour, options.outlier_threshold);

    log_metric_spike_breakdown(
        &points,
        &merged_tour,
        options.outlier_threshold,
        &chunk_partitions,
        "final",
    );

    Ok(merged_tour.into_iter().map(|idx| points[idx]).collect())
}
