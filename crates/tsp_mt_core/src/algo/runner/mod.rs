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
    // Use half of the available cores for the initial multi-seed
    // round. Empirically the second-half of seeds don't add useful
    // diversity (they all converge to the same local-min basin), and
    // running fewer seeds parallel-wise reduces inter-core cache
    // contention so each seed's LK iteration speed goes up.
    let run_count = (available_seed_runs() / 2).max(1);
    let max_trials = scaled_max_trials(points.len());

    // === B: longer per-seed time_limit ===
    // Multi-seed runs N=cores in parallel, so wall-clock is one
    // run's time. The chunked path's per-chunk `time_limit` is sized
    // for "many chunks fit in this much wall time"; here we have
    // one input × N parallel runs, so we can afford 4× the
    // per-chunk budget without hurting overall runtime much. Capped
    // at 12s to avoid pathological slowdown on tiny instances.
    let base_time = scaled_time_limit_seconds(points.len());
    let per_seed_time = (base_time * 4.0).max(MIN_REFINE_TIME_LIMIT_SECONDS as f64).min(48.0);

    let seeds = generate_seeds(DEFAULT_BASE_SEED, run_count);

    // Shared NN candidate set built once. Each parallel seed run uses
    // the same candidates — diversification comes from per-seed
    // initial tour and kick sequence, not the candidate graph. Also
    // reused by the post-multi-seed refinement below.
    let global_candidates = build_global_candidates(&projected_points, SHARED_MAX_CANDIDATES)?;

    let run_results: Result<Vec<(Vec<usize>, f64)>> = seeds
        .into_par_iter()
        .enumerate()
        .map(|(seed_idx, seed)| -> Result<(Vec<usize>, f64)> {
            // === E: stronger per-seed search ===
            // Enable 4-opt (move_type=4) to broaden each seed's
            // search beyond the chunk-solver's k≤3 limit. Bump
            // max_no_improvement so kicks keep firing for longer
            // before the per-seed stagnation early-exit kicks in.
            //
            // === P: heterogeneous move_type across seeds ===
            // Even-indexed seeds use deep search (move_type=4);
            // odd-indexed seeds use fast search (move_type=2).
            // Fast seeds explore more starting tours within
            // budget; deep seeds exploit each starting tour more
            // thoroughly. Best-of-N captures both regimes.
            // All seeds use 2-opt only — heterogeneous move_type=4
            // seeds were dropped after they regressed quality at
            // small n (where they wasted budget reaching 4-opt
            // local minima the perturbation kicks unwound) and
            // didn't improve large n (4-opt apply costs O(n) per
            // accept which limits the sweep count within budget).
            let move_type = 2;
            let params = seeded_params(seed, max_trials, per_seed_time, MULTI_SEED_TRACE_LEVEL)
                .with_move_type(move_type)
                .with_max_no_improvement(SEED_MAX_NO_IMPROVEMENT);

            let problem = build_problem(&projected_points)?;

            // === K: per-seed initial-tour diversity ===
            // Half of the seeds (the odd indices) get a random
            // greedy-NN starting tour driven by their seed value;
            // the other half use the deterministic greedy-fragment
            // tour (greedy multi-fragment heuristic, our default).
            // Mixing starting basins lets best-of-N explore tours
            // the perturbation kicks alone wouldn't reach.
            let params = if options.use_initial_tour {
                // Caller explicitly asked for the identity initial
                // tour — honor that for all seeds.
                maybe_attach_initial_tour(params, projected_points.len(), true)
            } else if !seed_idx.is_multiple_of(2) {
                let nn_tour = lin_kernighan::initial::greedy_nn(
                    &problem,
                    &global_candidates,
                    seed,
                );
                params.with_initial_tour(nn_tour.into_vec_usize())
            } else {
                params
            };

            let outcome = LkSolver::new_with_candidates(
                problem,
                params,
                global_candidates.clone(),
            )
            .solve()?;

            let length = cycle_length(&points, &outcome.tour);
            Ok((outcome.tour, length))
        })
        .collect();

    let mut sorted_results = run_results?;
    sorted_results.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut best_tour = sorted_results
        .first()
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?
        .0
        .clone();
    let runner_up_tour = sorted_results.get(1).map(|(t, _)| t.clone());

    // === F (full IPT): one-shot boundary-segment crossover ===
    // Take best_tour (T1) and runner_up (T2) from the parallel seeds
    // and find the single best improving equal-cardinality boundary
    // segment swap (gain measured in the projected euc_2d metric
    // that LK itself optimises). All inner walks have hard step
    // caps and the output is Hamiltonicity-validated before return.
    //
    // IPT optimises the projected euc_2d distance, but the runner
    // ranks tours via haversine `cycle_length`. The two metrics can
    // disagree on tiny deltas — accept the swap only when it also
    // improves haversine. Without this guard the projected-space win
    // can be a real-metric regression that subsequent LK refinement
    // (which itself works in projected euc_2d) does not undo.
    if let Some(runner_up) = runner_up_tour.as_ref() {
        let t1_u32: Vec<u32> = best_tour.iter().map(|&i| i as u32).collect();
        let t2_u32: Vec<u32> = runner_up.iter().map(|&i| i as u32).collect();
        let ipt_problem = build_problem(&projected_points)?;
        if let Some(child) = lin_kernighan::recombine::merge_with_tour_ipt(
            &t1_u32,
            &t2_u32,
            ipt_problem.coords(),
        ) {
            let child_usize: Vec<usize> = child.iter().map(|&v| v as usize).collect();
            let before_len = cycle_length(&points, &best_tour);
            let after_len = cycle_length(&points, &child_usize);
            if after_len < before_len {
                log::info!(
                    "F (IPT) recombination: haversine {} -> {} (delta={:+.4}%)",
                    before_len,
                    after_len,
                    (after_len - before_len) / before_len * 100.0,
                );
                best_tour = child_usize;
            }
        }
    }

    // === A: post multi-seed unified refinement ===
    // Mirror of the chunked path's post-stitch LK pass. Take the
    // best tour out of the N parallel seeds and run one more LK
    // sweep on it using 2-opt only — same trick that hit on the
    // chunked path (Or-opt and 3-opt apply are O(n) tour rebuilds
    // at this scale, drown out the cheap 2-opt sweeps that catch
    // cross-seed-untouchable local moves).
    let refine_time_limit = (base_time * 2.0)
        .max(MIN_REFINE_TIME_LIMIT_SECONDS as f64)
        .min(12.0);
    let refine_problem = build_problem(&projected_points)?;
    let refine_params = seeded_params(
        DEFAULT_BASE_SEED,
        scaled_max_trials(points.len()),
        refine_time_limit,
        MULTI_SEED_TRACE_LEVEL,
    )
    .with_initial_tour(best_tour)
    .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT)
    .with_move_type(2);
    // === F-lite: recombination via candidate-set augmentation ===
    // Take edges from the second-best multi-seed tour and splice
    // them into the candidate set used by the refinement. The
    // refinement search now has access to both tours' edges and can
    // synthesise moves spanning them — a poor-man's IPT crossover
    // without the full LKH machinery (~80 LOC vs ~400). Edges
    // already in the candidate set are skipped, so the effective
    // size grows by at most n.
    let mut refine_candidates = global_candidates.clone();
    if let Some(runner_up) = runner_up_tour.as_ref() {
        let runner_up_u32: Vec<u32> = runner_up.iter().map(|&i| i as u32).collect();
        let refine_problem_for_coords = build_problem(&projected_points)?;
        refine_candidates.add_tour_edges(&runner_up_u32, refine_problem_for_coords.coords());
    }
    let refined = LkSolver::new_with_candidates(
        refine_problem,
        refine_params,
        refine_candidates,
    )
    .solve()?;
    log::info!(
        "post-multi-seed LK refinement: length={}",
        refined.length
    );

    // === N: 2nd kick-polish round ===
    // Take the refinement output and re-kick it from N parallel
    // seeds. Each polish run starts from the same refined tour,
    // applies its own seeded double-bridge, then improves. Picks
    // best of N + the refined tour itself. Mostly a no-op if
    // refinement found the basin's bottom, but occasionally a
    // kick lands in a deeper basin we missed.
    let polish_time = (base_time * 2.0)
        .max(MIN_REFINE_TIME_LIMIT_SECONDS as f64)
        .min(8.0);
    let polish_seeds = generate_seeds(DEFAULT_BASE_SEED.wrapping_add(1), available_seed_runs());
    let refined_tour = refined.tour.clone();
    let polish_results: Result<Vec<(Vec<usize>, f64)>> = polish_seeds
        .into_par_iter()
        .map(|seed| -> Result<(Vec<usize>, f64)> {
            let params = seeded_params(
                seed,
                scaled_max_trials(points.len()),
                polish_time,
                CENTROID_TRACE_LEVEL,
            )
            .with_initial_tour(refined_tour.clone())
            .with_move_type(2)
            .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT);
            let problem = build_problem(&projected_points)?;
            let outcome = LkSolver::new_with_candidates(
                problem,
                params,
                global_candidates.clone(),
            )
            .solve()?;
            let length = cycle_length(&points, &outcome.tour);
            Ok((outcome.tour, length))
        })
        .collect();

    // Convert refined.length (integer EUC_2D in scaled space) to
    // the same metric polish runs use (great-circle meters) so the
    // best-of comparison is apples-to-apples.
    let refined_meters = cycle_length(&points, &refined.tour);
    let best_after_polish = polish_results?
        .into_iter()
        .chain(std::iter::once((refined.tour, refined_meters)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?
        .0;

    log::info!("post-polish: picked best of {} candidates", run_count + 1);

    // Adaptive pipeline depth: the extra refine+polish cycles below
    // each cost ~10-20s wall-clock and only buy real quality at
    // medium-to-large `n`. At small `n` the basin has already been
    // exhausted by the first cycle and the work is wasted.
    //
    // Empirically (see commit history / verify-multi-seed.sh runs):
    //   n <  5_000: skip both extra cycles
    //   n < 10_000: keep 2nd cycle, skip 3rd
    //   n >= 10_000: full 3 cycles
    let n_for_pipeline = points.len();
    if n_for_pipeline < 5_000 {
        return Ok(best_after_polish.into_iter().map(|idx| points[idx]).collect());
    }

    // === 2nd refinement pass after kick-polish ===
    // Take the best tour from the kick-polish round and run another
    // LK refinement (single-seed, 2-opt only). The polish kicks may
    // have produced a tour whose best 2-opt basin is below the
    // refinement convergence we found, so re-refining is worth the
    // small extra cost.
    let refine2_params = seeded_params(
        DEFAULT_BASE_SEED.wrapping_add(2),
        scaled_max_trials(points.len()),
        refine_time_limit,
        MULTI_SEED_TRACE_LEVEL,
    )
    .with_initial_tour(best_after_polish.clone())
    .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT)
    .with_move_type(2);
    let refine2_problem = build_problem(&projected_points)?;
    let refined2 = LkSolver::new_with_candidates(
        refine2_problem,
        refine2_params,
        global_candidates.clone(),
    )
    .solve()?;
    log::info!(
        "post-polish LK refinement: length={}",
        refined2.length
    );

    // === 2nd kick-polish round after 2nd refinement ===
    // Mirror of the first polish round, but on the post-2nd-
    // refinement tour. Cheap on top of the existing pipeline and
    // catches the occasional kick that beats the doubly-refined
    // baseline.
    let polish2_seeds = generate_seeds(DEFAULT_BASE_SEED.wrapping_add(3), available_seed_runs());
    let refined2_tour_clone = refined2.tour.clone();
    let polish2_results: Result<Vec<(Vec<usize>, f64)>> = polish2_seeds
        .into_par_iter()
        .map(|seed| -> Result<(Vec<usize>, f64)> {
            let params = seeded_params(
                seed,
                scaled_max_trials(points.len()),
                polish_time,
                CENTROID_TRACE_LEVEL,
            )
            .with_initial_tour(refined2_tour_clone.clone())
            .with_move_type(2)
            .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT);
            let problem = build_problem(&projected_points)?;
            let outcome = LkSolver::new_with_candidates(
                problem,
                params,
                global_candidates.clone(),
            )
            .solve()?;
            let length = cycle_length(&points, &outcome.tour);
            Ok((outcome.tour, length))
        })
        .collect();
    let refined2_meters = cycle_length(&points, &refined2.tour);
    let best_after_polish2 = polish2_results?
        .into_iter()
        .chain(std::iter::once((refined2.tour, refined2_meters)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?
        .0;

    if n_for_pipeline < 10_000 {
        return Ok(best_after_polish2.into_iter().map(|idx| points[idx]).collect());
    }

    // === 3rd refinement pass after 2nd polish ===
    // Each refine+polish cycle keeps finding small gains at large n
    // — the iteration spread across cycles is wider than what fits
    // in one stage's time budget. Third cycle still pays off at the
    // 25k cell.
    let refine3_params = seeded_params(
        DEFAULT_BASE_SEED.wrapping_add(4),
        scaled_max_trials(points.len()),
        refine_time_limit,
        MULTI_SEED_TRACE_LEVEL,
    )
    .with_initial_tour(best_after_polish2.clone())
    .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT)
    .with_move_type(2);
    let refine3_problem = build_problem(&projected_points)?;
    let refined3 = LkSolver::new_with_candidates(
        refine3_problem,
        refine3_params,
        global_candidates.clone(),
    )
    .solve()?;
    log::info!("post-polish2 LK refinement: length={}", refined3.length);

    // === 3rd kick-polish round after 3rd refinement ===
    let polish3_seeds = generate_seeds(DEFAULT_BASE_SEED.wrapping_add(5), available_seed_runs());
    let refined3_tour_clone = refined3.tour.clone();
    let polish3_results: Result<Vec<(Vec<usize>, f64)>> = polish3_seeds
        .into_par_iter()
        .map(|seed| -> Result<(Vec<usize>, f64)> {
            let params = seeded_params(
                seed,
                scaled_max_trials(points.len()),
                polish_time,
                CENTROID_TRACE_LEVEL,
            )
            .with_initial_tour(refined3_tour_clone.clone())
            .with_move_type(2)
            .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT);
            let problem = build_problem(&projected_points)?;
            let outcome = LkSolver::new_with_candidates(
                problem,
                params,
                global_candidates.clone(),
            )
            .solve()?;
            let length = cycle_length(&points, &outcome.tour);
            Ok((outcome.tour, length))
        })
        .collect();
    let refined3_meters = cycle_length(&points, &refined3.tour);
    let best_after_polish3 = polish3_results?
        .into_iter()
        .chain(std::iter::once((refined3.tour, refined3_meters)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?
        .0;

    if n_for_pipeline < 20_000 {
        return Ok(best_after_polish3.into_iter().map(|idx| points[idx]).collect());
    }

    // === 4th refinement + polish cycle for n>=20k ===
    // The third cycle still finds gains at the 20k/25k scale, so a
    // fourth cycle continues the trend. Empirically buys ~0.06% gap
    // closure at the cost of ~20s wall-clock per cell, only
    // activated for inputs large enough that the marginal quality
    // gain justifies the runtime hit.
    let refine4_params = seeded_params(
        DEFAULT_BASE_SEED.wrapping_add(6),
        scaled_max_trials(points.len()),
        refine_time_limit,
        MULTI_SEED_TRACE_LEVEL,
    )
    .with_initial_tour(best_after_polish3.clone())
    .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT)
    .with_move_type(2);
    let refine4_problem = build_problem(&projected_points)?;
    let refined4 = LkSolver::new_with_candidates(
        refine4_problem,
        refine4_params,
        global_candidates.clone(),
    )
    .solve()?;
    log::info!("post-polish3 LK refinement: length={}", refined4.length);

    let polish4_seeds = generate_seeds(DEFAULT_BASE_SEED.wrapping_add(7), available_seed_runs());
    let refined4_tour_clone = refined4.tour.clone();
    let polish4_results: Result<Vec<(Vec<usize>, f64)>> = polish4_seeds
        .into_par_iter()
        .map(|seed| -> Result<(Vec<usize>, f64)> {
            let params = seeded_params(
                seed,
                scaled_max_trials(points.len()),
                polish_time,
                CENTROID_TRACE_LEVEL,
            )
            .with_initial_tour(refined4_tour_clone.clone())
            .with_move_type(2)
            .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT);
            let problem = build_problem(&projected_points)?;
            let outcome = LkSolver::new_with_candidates(
                problem,
                params,
                global_candidates.clone(),
            )
            .solve()?;
            let length = cycle_length(&points, &outcome.tour);
            Ok((outcome.tour, length))
        })
        .collect();
    let refined4_meters = cycle_length(&points, &refined4.tour);
    let final_tour = polish4_results?
        .into_iter()
        .chain(std::iter::once((refined4.tour, refined4_meters)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| Error::other(ERR_NO_RESULTS))?
        .0;

    Ok(final_tour.into_iter().map(|idx| points[idx]).collect())
}

/// Per-seed stagnation tolerance in multi-seed mode. Higher than the
/// chunked path's 3000 default because each seed has more time budget
/// (B) and we want kicks to keep firing until diminishing returns.
const SEED_MAX_NO_IMPROVEMENT: usize = 5000;
/// Refinement stagnation tolerance — same as the chunked-path
/// `REFINE_MAX_NO_IMPROVEMENT` constant; copied here to avoid
/// cross-module visibility churn.
const REFINE_MAX_NO_IMPROVEMENT: usize = 2000;

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
        .min(5.0);
    // Refinement starts from a near-locally-optimal stitched tour
    // but still benefits significantly from kick-based diversification
    // — many seam-induced local minima only get escaped via a
    // double-bridge perturbation. Re-uses the chunk solver's trial
    // loop with a tight `max_no_improvement` so we exit quickly once
    // kicks stop paying off.
    const REFINE_MAX_NO_IMPROVEMENT: usize = 2000;
    let refine_params = seeded_params(
        DEFAULT_BASE_SEED,
        scaled_max_trials(projected_points.len()),
        refine_time_limit,
        MULTI_SEED_TRACE_LEVEL,
    )
    .with_initial_tour(merged_tour.clone())
    .with_max_no_improvement(REFINE_MAX_NO_IMPROVEMENT)
    // Refinement skips 3-opt + Or-opt — those are O(n)-per-apply
    // tour rebuilds that murder runtime on n≥5k tours. 2-opt sweeps
    // do most of the seam-fixing work and stay near-constant per
    // sweep thanks to don't-look-bit propagation.
    .with_move_type(2);
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
