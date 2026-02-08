use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use rayon::prelude::*;

use crate::{
    Error, Result, SolverInput, Tour, config::LkhConfig, constants::MIN_CYCLE_POINTS, file_cleanup,
    geometry, h3_chunking, node::LKHNode, options::SolverOptions, problem::TsplibProblemWriter,
    process::LkhProcess, projection::PlaneProjection, solver::LkhSolver, stitching,
};

const RUN_INDEX_SINGLE: usize = 0;
const MAX_CENTROIDS_WITH_TRIVIAL_ORDER: usize = 2;
const CHUNK_DIR_PREFIX: &str = "chunk_";
const CHUNK_ORDER_DIR: &str = "chunk_order";

const CENTROIDS_BASENAME: &str = "centroids";
const PAR_EXTENSION: &str = ".par";
const TOUR_EXTENSION: &str = ".tour";
const TSP_EXTENSION: &str = ".tsp";

const ERR_LKH_CHUNK_FAILED: &str = "LKH chunk failed";
const ERR_CENTROID_ORDERING_FAILED: &str = "Centroid ordering LKH failed";
const ERR_INVALID_POINT: &str = "Input contains invalid lat/lng values";
const ERR_INVALID_PROJECTION_RADIUS: &str = "projection_radius must be > 0";
const ERR_INVALID_MAX_CHUNK_SIZE: &str = "max_chunk_size must be > 0";

fn file_name(stem: &str, extension: &str) -> String {
    format!("{stem}{extension}")
}

struct ChunkSolver(PathBuf);

impl Drop for ChunkSolver {
    fn drop(&mut self) {
        // log::debug!("Dropping chunksolver");
        file_cleanup::cleanup_workdir(&self.0)
    }
}

impl ChunkSolver {
    fn new(work_dir: &Path) -> Self {
        file_cleanup::register_workdir_for_shutdown_cleanup(work_dir);
        Self(work_dir.to_path_buf())
    }

    fn solve_single(
        &self,
        lkh_exe: &Path,
        work_dir: &Path,
        chunk_points: &[LKHNode],
        options: &SolverOptions,
    ) -> Result<Vec<usize>> {
        let n = chunk_points.len();
        if n < MIN_CYCLE_POINTS {
            return Ok(chunk_points
                .iter()
                .enumerate()
                .map(|(idx, _)| idx)
                .collect());
        }

        let solver = LkhSolver::new(lkh_exe, work_dir);
        solver.create_work_dir()?;

        let pts = PlaneProjection::new(chunk_points)
            .radius(options.projection_radius)
            .project();

        solver.create_problem_file(&pts)?;

        let cfg = solver.parallel_run_config(n);
        let run_par = solver.run_par_path(RUN_INDEX_SINGLE);
        let run_tour = solver.run_tour_path(RUN_INDEX_SINGLE);
        let seed = cfg.base_seed();
        let run_cfg = cfg.with_seed(seed).with_output_tour_file(&run_tour);

        run_cfg.write_to_file(&run_par)?;

        let out = solver.run(&run_par)?;
        LkhProcess::ensure_success(ERR_LKH_CHUNK_FAILED, &out)?;

        LkhProcess::parse_tsplib_tour(&run_tour, n)
    }

    fn order_by_centroid_tsp(
        &self,
        lkh_exe: &Path,
        work_dir: &Path,
        centroids: &[LKHNode],
        options: &SolverOptions,
    ) -> Result<Vec<usize>> {
        if centroids.len() <= MAX_CENTROIDS_WITH_TRIVIAL_ORDER {
            log::debug!(
                "chunked.order: skip_lkh centroids={} reason=trivial",
                centroids.len()
            );
            return Ok((0..centroids.len()).collect());
        }

        log::debug!("chunked.order: start centroids={}", centroids.len());
        fs::create_dir_all(work_dir)?;

        let problem = work_dir.join(file_name(CENTROIDS_BASENAME, TSP_EXTENSION));
        TsplibProblemWriter::write_euc2d(&problem, CENTROIDS_BASENAME, centroids)?;

        let par_path = work_dir.join(file_name(CENTROIDS_BASENAME, PAR_EXTENSION));
        let tour_path = work_dir.join(file_name(CENTROIDS_BASENAME, TOUR_EXTENSION));

        let cfg = LkhConfig::for_small_problem(
            &problem,
            options.centroid_order_max_trials,
            options.centroid_order_time_limit,
        )
        .with_seed(options.centroid_order_seed)
        .with_output_tour_file(&tour_path);

        cfg.write_to_file(&par_path)?;

        let out = Command::new(lkh_exe)
            .arg(&par_path)
            .current_dir(work_dir)
            .output()
            .map_err(Error::from)?;
        LkhProcess::ensure_success(ERR_CENTROID_ORDERING_FAILED, &out)?;

        log::debug!("chunked.order: done centroids={}", centroids.len());
        LkhProcess::parse_tsplib_tour(&tour_path, centroids.len())
    }
}

#[tsp_mt_derive::timer("chunked")]
pub fn solve_tsp_with_lkh_h3_chunked(input: SolverInput, options: SolverOptions) -> Result<Tour> {
    // cleans up workdir on drop
    let chunk_solver = ChunkSolver::new(&options.work_dir);

    if options.max_chunk_size == 0 {
        return Err(Error::invalid_input(ERR_INVALID_MAX_CHUNK_SIZE));
    }
    if options.projection_radius <= 0.0 {
        return Err(Error::invalid_input(ERR_INVALID_PROJECTION_RADIUS));
    }
    if input.points.iter().any(|p| !p.is_valid()) {
        return Err(Error::invalid_input(ERR_INVALID_POINT));
    }

    if input.n() <= options.max_chunk_size {
        log::info!(
            "chunker: bypass n={} max_chunk_size={} mode=parallel",
            input.n(),
            options.max_chunk_size
        );
        return crate::solve_tsp_with_lkh_parallel(input, options);
    }

    let global_coords = PlaneProjection::new(&input.points)
        .radius(options.projection_radius)
        .project();

    let chunks = h3_chunking::partition_indices(&input.points, options.max_chunk_size)?;
    log::info!(
        "chunker: partitioned n={} chunks={} max_chunk_size={}",
        input.n(),
        chunks.len(),
        options.max_chunk_size
    );

    let solved_chunk_tours: Vec<Vec<usize>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, idxs)| -> Result<Vec<usize>> {
            let chunk_points: Vec<LKHNode> = idxs.iter().map(|&i| input.get_point(i)).collect();
            let chunk_dir = options
                .work_dir
                .join(format!("{CHUNK_DIR_PREFIX}{chunk_id}"));

            let now = Instant::now();
            let tour_local =
                chunk_solver.solve_single(&options.lkh_exe, &chunk_dir, &chunk_points, &options)?;
            let tour_global: Vec<usize> = tour_local.into_iter().map(|li| idxs[li]).collect();

            log::info!(
                "chunk: done id={chunk_id} n={} secs={:.2}",
                idxs.len(),
                now.elapsed().as_secs_f32()
            );

            Ok(tour_global)
        })
        .collect::<Result<Vec<_>>>()?;

    let centroids: Vec<LKHNode> = chunks
        .iter()
        .map(|idxs| geometry::centroid_of_indices(&global_coords, idxs))
        .collect();

    log::info!("chunker: ordering centroids count={}", centroids.len());
    let order_dir = options.work_dir.join(CHUNK_ORDER_DIR);
    let order =
        chunk_solver.order_by_centroid_tsp(&options.lkh_exe, &order_dir, &centroids, &options)?;

    let mut ordered_tours: Vec<Vec<usize>> = Vec::with_capacity(solved_chunk_tours.len());
    for ci in order {
        ordered_tours.push(solved_chunk_tours[ci].clone());
    }

    let (mut merged, boundaries) =
        stitching::stitch_chunk_tours_dense(&global_coords, ordered_tours);
    log::info!(
        "chunker: stitched n={} boundaries={}",
        merged.len(),
        boundaries.len()
    );

    stitching::boundary_two_opt(
        &global_coords,
        &mut merged,
        &boundaries,
        options.boundary_2opt_window,
        options.boundary_2opt_passes,
    );
    log::info!(
        "chunker: complete n={} chunks={}",
        merged.len(),
        chunks.len()
    );

    Ok(Tour::new(
        merged.into_iter().map(|i| input.get_point(i)).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::solve_tsp_with_lkh_h3_chunked;
    use crate::{LKHNode, SolverInput, SolverOptions};

    fn sample_input(points: Vec<LKHNode>) -> SolverInput {
        SolverInput::new(&points)
    }

    #[test]
    fn solve_chunked_rejects_zero_max_chunk_size() {
        let input = sample_input(vec![LKHNode::from_lat_lng(10.0, 20.0)]);
        let options = SolverOptions {
            max_chunk_size: 0,
            ..SolverOptions::default()
        };

        let err = solve_tsp_with_lkh_h3_chunked(input, options).expect_err("must fail");
        assert!(err.to_string().contains("max_chunk_size must be > 0"));
    }

    #[test]
    fn solve_chunked_rejects_non_positive_projection_radius() {
        let input = sample_input(vec![LKHNode::from_lat_lng(10.0, 20.0)]);
        let options = SolverOptions {
            projection_radius: 0.0,
            ..SolverOptions::default()
        };

        let err = solve_tsp_with_lkh_h3_chunked(input, options).expect_err("must fail");
        assert!(err.to_string().contains("projection_radius must be > 0"));
    }

    #[test]
    fn solve_chunked_rejects_invalid_lat_lng_points() {
        let input = sample_input(vec![LKHNode::from_lat_lng(95.0, 20.0)]);
        let options = SolverOptions::default();

        let err = solve_tsp_with_lkh_h3_chunked(input, options).expect_err("must fail");
        assert!(err.to_string().contains("invalid lat/lng values"));
    }
}
