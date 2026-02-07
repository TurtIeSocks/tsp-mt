use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use geo::Coord;
use rayon::prelude::*;

use crate::lkh::{
    config::LkhConfig,
    constants::{CENTROIDS_FILE, MIN_CYCLE_POINTS, RUN_FILE},
    geometry::TourGeometry,
    h3_chunking::H3Chunker,
    options::SolverOptions,
    point::Point,
    problem::TsplibProblemWriter,
    process::LkhProcess,
    projection::PlaneProjection,
    run_spec::RunSpec,
    solver::LkhSolver,
    stitching::TourStitcher,
};

const RUN_INDEX_SINGLE: usize = 0;
const MAX_CENTROIDS_WITH_TRIVIAL_ORDER: usize = 2;
const CHUNK_DIR_PREFIX: &str = "chunk_";
const CHUNK_ORDER_DIR: &str = "chunk_order";

const ERR_LKH_CHUNK_FAILED: &str = "LKH chunk failed";
const ERR_CENTROID_ORDERING_FAILED: &str = "Centroid ordering LKH failed";
const ERR_INVALID_POINT: &str = "Input contains invalid lat/lng values";
const ERR_INVALID_PROJECTION_RADIUS: &str = "projection_radius must be > 0";
const ERR_INVALID_MAX_CHUNK_SIZE: &str = "max_chunk_size must be > 0";

struct ChunkSolver;

impl ChunkSolver {
    fn solve_single(
        lkh_exe: PathBuf,
        work_dir: PathBuf,
        chunk_points: &[Point],
        options: &SolverOptions,
    ) -> io::Result<Vec<usize>> {
        let n = chunk_points.len();
        if n < MIN_CYCLE_POINTS {
            return Ok(chunk_points
                .iter()
                .enumerate()
                .map(|(idx, _)| idx)
                .collect());
        }

        let cfg = LkhConfig::new(n);
        let solver = LkhSolver::new(lkh_exe, work_dir);
        solver.create_work_dir()?;

        let pts = PlaneProjection::new(chunk_points)
            .radius(options.projection_radius)
            .project();

        solver.create_problem_file(&pts)?;
        solver.ensure_candidate_file(pts.len())?;

        let rs = RunSpec::new(
            RUN_INDEX_SINGLE,
            cfg.base_seed(),
            solver
                .work_dir()
                .join(RUN_FILE.par_idx(RUN_INDEX_SINGLE)),
            solver
                .work_dir()
                .join(RUN_FILE.tour_idx(RUN_INDEX_SINGLE)),
        );
        rs.write_lkh_par(&cfg, &solver)?;

        let out = solver.run(rs.par_path())?;
        LkhProcess::ensure_success(ERR_LKH_CHUNK_FAILED, &out)?;

        rs.parse_tsplib_tour(n)
    }

    fn order_by_centroid_tsp(
        lkh_exe: &Path,
        work_dir: &Path,
        centroids: &[Coord],
        options: &SolverOptions,
    ) -> io::Result<Vec<usize>> {
        if centroids.len() <= MAX_CENTROIDS_WITH_TRIVIAL_ORDER {
            return Ok((0..centroids.len()).collect());
        }

        fs::create_dir_all(work_dir)?;

        let problem = work_dir.join(CENTROIDS_FILE.tsp());
        TsplibProblemWriter::write_euc2d(&problem, CENTROIDS_FILE.name(), centroids)?;

        let rs = RunSpec::new(
            RUN_INDEX_SINGLE,
            options.centroid_order_seed,
            work_dir.join(CENTROIDS_FILE.par()),
            work_dir.join(CENTROIDS_FILE.tour()),
        );
        rs.write_lkh_par_small(
            &problem,
            options.centroid_order_max_trials,
            options.centroid_order_time_limit,
        )?;

        let out = Command::new(lkh_exe)
            .arg(rs.par_path())
            .current_dir(work_dir)
            .output()?;
        LkhProcess::ensure_success(ERR_CENTROID_ORDERING_FAILED, &out)?;

        rs.parse_tsplib_tour(centroids.len())
    }
}

pub(crate) fn solve_tsp_with_lkh_h3_chunked_with_options(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
    options: &SolverOptions,
) -> io::Result<Vec<Point>> {
    if options.max_chunk_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            ERR_INVALID_MAX_CHUNK_SIZE,
        ));
    }
    if options.projection_radius <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            ERR_INVALID_PROJECTION_RADIUS,
        ));
    }
    if input.iter().any(|p| !p.is_valid()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            ERR_INVALID_POINT,
        ));
    }

    if input.len() <= options.max_chunk_size {
        return super::solve_tsp_with_lkh_parallel_with_options(lkh_exe, work_dir, input, options);
    }

    let global_coords = PlaneProjection::new(input)
        .radius(options.projection_radius)
        .project();

    let chunks = H3Chunker::partition_indices(input, options.max_chunk_size)?;
    if options.verbose {
        eprintln!(
            "Chunked {} points into {} chunks (max {})",
            input.len(),
            chunks.len(),
            options.max_chunk_size
        );
    }

    let solved_chunk_tours: Vec<Vec<usize>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, idxs)| -> io::Result<Vec<usize>> {
            let chunk_points: Vec<Point> = idxs.iter().map(|&i| input[i]).collect();
            let chunk_dir = work_dir.join(format!("{CHUNK_DIR_PREFIX}{chunk_id}"));

            let now = Instant::now();
            let tour_local =
                ChunkSolver::solve_single(lkh_exe.clone(), chunk_dir, &chunk_points, options)?;
            let tour_global: Vec<usize> = tour_local.into_iter().map(|li| idxs[li]).collect();

            if options.verbose {
                eprintln!(
                    "chunk {chunk_id}: n={} solved in {:.2}s",
                    idxs.len(),
                    now.elapsed().as_secs_f32()
                );
            }

            Ok(tour_global)
        })
        .collect::<io::Result<Vec<_>>>()?;

    let centroids: Vec<Coord> = chunks
        .iter()
        .map(|idxs| TourGeometry::centroid_of_indices(&global_coords, idxs))
        .collect();

    let order_dir = work_dir.join(CHUNK_ORDER_DIR);
    let order = ChunkSolver::order_by_centroid_tsp(&lkh_exe, &order_dir, &centroids, options)?;

    let mut ordered_tours: Vec<Vec<usize>> = Vec::with_capacity(solved_chunk_tours.len());
    for ci in order {
        ordered_tours.push(solved_chunk_tours[ci].clone());
    }

    let (mut merged, boundaries) =
        TourStitcher::stitch_chunk_tours_dense(&global_coords, ordered_tours);

    TourStitcher::boundary_two_opt(
        &global_coords,
        &mut merged,
        &boundaries,
        options.boundary_2opt_window,
        options.boundary_2opt_passes,
        options.verbose,
    );

    Ok(merged.into_iter().map(|i| input[i]).collect())
}
