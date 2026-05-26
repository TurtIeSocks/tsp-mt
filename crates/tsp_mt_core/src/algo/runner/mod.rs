use lin_kernighan::Solver as LkSolver;
use rayon::prelude::*;

use crate::{
    Error, LKHNode, Result, SolverInput, SolverOptions, geometry, h3_chunking,
    projection::PlaneProjection, stitching,
};

mod common;
mod metric_spike;

use common::{
    CENTROID_TRACE_LEVEL, DEFAULT_BASE_SEED, ERR_NO_RESULTS, MAX_CENTROIDS_WITH_TRIVIAL_ORDER,
    MIN_REFINE_TIME_LIMIT_SECONDS, MULTI_SEED_TRACE_LEVEL, SHARED_MAX_CANDIDATES,
    available_seed_runs, build_global_candidates, build_problem, build_stitching_tuning,
    canonicalize_cycle_order, cycle_length, generate_seeds, maybe_attach_initial_tour,
    scaled_max_trials, scaled_time_limit_seconds, seeded_params, solve_chunk_indices,
    validate_chunk_size, validate_min_cycle_points, validate_outlier_threshold, validate_points,
    validate_projection_radius,
};
use metric_spike::{log_metric_spike_breakdown, repair_metric_spikes_with_global_two_opt};

#[tsp_mt_derive::timer()]
pub fn lkh_single(input: SolverInput, options: SolverOptions) -> Result<Vec<LKHNode>> {
    validate_min_cycle_points(&input)?;
    validate_points(&input)?;

    let params = seeded_params(
        DEFAULT_BASE_SEED,
        scaled_max_trials(input.n()),
        scaled_time_limit_seconds(input.n()) / 2.0,
        MULTI_SEED_TRACE_LEVEL,
    );
    let params = maybe_attach_initial_tour(params, input.n(), options.use_initial_tour);
    let problem = build_problem(&input.nodes)?;
    let outcome = LkSolver::new(problem, params).solve()?;

    Ok(outcome.tour.into_iter().map(|idx| input.get_point(idx)).collect())
}

#[tsp_mt_derive::timer()]
pub fn lkh_multi_seed(input: SolverInput, options: SolverOptions) -> Result<Vec<LKHNode>> {
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

    let run_results: Result<Vec<(Vec<usize>, f64)>> = seeds
        .into_par_iter()
        .map(|seed| -> Result<(Vec<usize>, f64)> {
            let params = seeded_params(seed, max_trials, time_limit, MULTI_SEED_TRACE_LEVEL);
            let params = maybe_attach_initial_tour(params, projected_points.len(), options.use_initial_tour);
            let problem = build_problem(&projected_points)?;
            let outcome = LkSolver::new(problem, params).solve()?;

            let length = cycle_length(&points, &outcome.tour);
            Ok((outcome.tour, length))
        })
        .collect();

    let best_tour = run_results?
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?
        .0;

    Ok(best_tour.into_iter().map(|idx| points[idx]).collect())
}

#[tsp_mt_derive::timer()]
pub fn lkh_multi_parallel(input: SolverInput, options: SolverOptions) -> Result<Vec<LKHNode>> {
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
        .map_err(|err| Error::other(err.to_string()))?;

    log::info!(
        "chunker: partitioned n={} chunks={} max_chunk_size={}",
        points.len(),
        chunk_partitions.len(),
        options.max_chunk_size
    );

    // Build a global NN candidate set once on the full projected
    // point set. Each chunk slices a local subset via
    // `CandidateSet::subset`, avoiding an O(n log n) k-d tree build
    // per chunk. For input-10k+ this turns dozens of redundant
    // candidate builds into one.
    let global_cands_start = std::time::Instant::now();
    let global_candidates = build_global_candidates(&projected_points, SHARED_MAX_CANDIDATES)?;
    log::info!(
        "shared candidate set built in {:.3}s (n={})",
        global_cands_start.elapsed().as_secs_f64(),
        projected_points.len()
    );

    let solved_chunk_tours: Result<Vec<Vec<usize>>> = chunk_partitions
        .par_iter()
        .enumerate()
        .map(|(chunk_id, chunk)| {
            solve_chunk_indices(
                chunk_id,
                &chunk.indices,
                &projected_points,
                options.use_initial_tour,
                Some(&global_candidates),
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
        let params = seeded_params(
            options.centroid_order_seed,
            options.centroid_order_max_trials,
            options.centroid_order_time_limit as f64,
            CENTROID_TRACE_LEVEL,
        );
        let params = maybe_attach_initial_tour(
            params,
            projected_centroids.len(),
            options.use_initial_tour,
        );
        let problem = build_problem(&projected_centroids)?;
        let outcome = LkSolver::new(problem, params).solve()?;
        outcome.tour
    };
    let chunk_order = canonicalize_cycle_order(chunk_order);

    let ordered_chunks: Result<Vec<(Vec<usize>, stitching::ChunkStitchState)>> = chunk_order
        .into_iter()
        .map(|chunk_idx| {
            let tour = solved_chunk_tours.get(chunk_idx).cloned().ok_or_else(|| {
                Error::invalid_data(format!(
                    "Chunk order index {chunk_idx} out of bounds for {} chunks",
                    solved_chunk_tours.len()
                ))
            })?;
            let chunk = chunk_partitions.get(chunk_idx).ok_or_else(|| {
                Error::invalid_data(format!(
                    "Chunk partition index {chunk_idx} out of bounds for {} chunks",
                    chunk_partitions.len()
                ))
            })?;
            let centroid = projected_centroids.get(chunk_idx).copied().ok_or_else(|| {
                Error::invalid_data(format!(
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

    // First-pass cleanup: the boundary_2opt and spike_repair passes
    // target specific edges (seam crossings and metric outliers) that
    // the NN-candidate-restricted LK refinement below cannot see —
    // long seam edges connect spatially-distant nodes that aren't in
    // each other's nearest-neighbour lists, so LK never gets a
    // chance to break them. Running these targeted passes first
    // produces a much better initial tour for the LK refinement to
    // polish.
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

    // Second-pass refinement: with seams pre-broken by the targeted
    // passes, lin_kernighan now has a near-locally-optimal starting
    // tour. A single LK pass over the whole tour with the shared
    // global candidates surfaces cross-chunk 2-opt/3-opt moves the
    // per-chunk solver couldn't reach by construction.
    let refine_start = std::time::Instant::now();
    let refine_problem = build_problem(&projected_points)?;
    // Time-bound refinement to roughly the per-chunk budget. The
    // stitched tour is already near a local optimum, so kicks have
    // limited room; but the search loop still wants the
    // double-bridge → re-improve cycle to escape seam-induced local
    // minima.
    // Refinement deadline scales sub-linearly: most improvement happens
    // in the first second or two as cross-seam moves surface. Anything
    // beyond that is diminishing returns.
    let refine_time_limit = scaled_time_limit_seconds(projected_points.len())
        .max(MIN_REFINE_TIME_LIMIT_SECONDS as f64)
        .min(8.0);
    // Refinement starts from a near-locally-optimal stitched tour.
    // Fewer kicks needed than a from-scratch chunk solve, so cap
    // `max_no_improvement` lower to exit faster once seam moves are
    // exhausted. ~200 kicks of no improvement is plenty.
    const REFINE_MAX_NO_IMPROVEMENT: usize = 200;
    let refine_params = seeded_params(
        DEFAULT_BASE_SEED,
        scaled_max_trials(projected_points.len()),
        refine_time_limit,
        MULTI_SEED_TRACE_LEVEL,
    )
    .with_initial_tour(merged_tour.clone())
    .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT);
    let refine_outcome = LkSolver::new_with_candidates(
        refine_problem,
        refine_params,
        global_candidates.clone(),
    )
    .solve()?;
    merged_tour = refine_outcome.tour;
    log::info!(
        "post-stitch LK refinement: {:.2}s length={}",
        refine_start.elapsed().as_secs_f64(),
        refine_outcome.length
    );

    // Final safety net: keep the metric-spike repair pass. The above
    // LK refinement minimises tour length but doesn't have explicit
    // outlier-edge awareness, so the spike pass occasionally still
    // helps on pathological topologies where one or two cross-region
    // edges dominate. Cheap relative to the LK refinement.
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
