use std::{
    fs, io,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use geo::Coord;
use rayon::prelude::*;

use crate::{
    lkh::{
        config::LkhConfig,
        constants::{
            CENTROIDS_NAME, CENTROIDS_PAR_FILE, CENTROIDS_TOUR_FILE, CENTROIDS_TSP_FILE,
            CHUNK_DIR_PREFIX, CHUNK_ORDER_DIR, MIN_CYCLE_POINTS, PLANE_PROJECTION_RADIUS,
            run_par_file, run_tour_file,
        },
        geometry::TourGeometry,
        h3_chunking::H3Chunker,
        problem::TsplibProblemWriter,
        process::LkhProcess,
        run_spec::RunSpec,
        solver::LkhSolver,
        stitching::TourStitcher,
    },
    project::Plane,
    utils::Point,
};

const MAX_CHUNK: usize = 5_000;
const CENTROID_ORDER_SEED: u64 = 999;
const CENTROID_ORDER_MAX_TRIALS: usize = 20_000;
const CENTROID_ORDER_TIME_LIMIT: usize = 10;
const BOUNDARY_2OPT_WINDOW: usize = 500;
const BOUNDARY_2OPT_PASSES: usize = 50;
const RUN_INDEX_SINGLE: usize = 0;
const MAX_CENTROIDS_WITH_TRIVIAL_ORDER: usize = 2;
const ERR_LKH_CHUNK_FAILED: &str = "LKH chunk failed";
const ERR_CENTROID_ORDERING_FAILED: &str = "Centroid ordering LKH failed";

struct ChunkSolver;

impl ChunkSolver {
    fn solve_single(
        lkh_exe: PathBuf,
        work_dir: PathBuf,
        chunk_points: &[Point],
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

        let pts = Plane::new(chunk_points)
            .radius(PLANE_PROJECTION_RADIUS)
            .project();

        solver.create_problem_file(&pts)?;
        solver.ensure_candidate_file(pts.len())?;

        let rs = RunSpec::new(
            RUN_INDEX_SINGLE,
            cfg.base_seed(),
            solver.work_dir().join(run_par_file(RUN_INDEX_SINGLE)),
            solver.work_dir().join(run_tour_file(RUN_INDEX_SINGLE)),
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
    ) -> io::Result<Vec<usize>> {
        if centroids.len() <= MAX_CENTROIDS_WITH_TRIVIAL_ORDER {
            return Ok((0..centroids.len()).collect());
        }

        fs::create_dir_all(work_dir)?;

        let problem = work_dir.join(CENTROIDS_TSP_FILE);
        TsplibProblemWriter::write_euc2d(&problem, CENTROIDS_NAME, centroids)?;

        let rs = RunSpec::new(
            RUN_INDEX_SINGLE,
            CENTROID_ORDER_SEED,
            work_dir.join(CENTROIDS_PAR_FILE),
            work_dir.join(CENTROIDS_TOUR_FILE),
        );
        rs.write_lkh_par_small(
            &problem,
            CENTROID_ORDER_MAX_TRIALS,
            CENTROID_ORDER_TIME_LIMIT,
        )?;

        let out = Command::new(lkh_exe)
            .arg(rs.par_path())
            .current_dir(work_dir)
            .output()?;
        LkhProcess::ensure_success(ERR_CENTROID_ORDERING_FAILED, &out)?;

        rs.parse_tsplib_tour(centroids.len())
    }
}

pub fn solve_tsp_with_lkh_h3_chunked(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    if input.len() <= MAX_CHUNK {
        return super::solve_tsp_with_lkh_parallel(lkh_exe, work_dir, input);
    }

    let global_coords = Plane::new(input).radius(PLANE_PROJECTION_RADIUS).project();

    let chunks = H3Chunker::partition_indices(input, MAX_CHUNK);
    eprintln!(
        "Chunked {} points into {} chunks (max {})",
        input.len(),
        chunks.len(),
        MAX_CHUNK
    );

    let solved_chunk_tours: Vec<Vec<usize>> = chunks
        .par_iter()
        .enumerate()
        .map(|(chunk_id, idxs)| -> io::Result<Vec<usize>> {
            let chunk_points: Vec<Point> = idxs.iter().map(|&i| input[i]).collect();
            let chunk_dir = work_dir.join(format!("{CHUNK_DIR_PREFIX}{chunk_id}"));

            let now = Instant::now();
            let tour_local = ChunkSolver::solve_single(lkh_exe.clone(), chunk_dir, &chunk_points)?;
            let tour_global: Vec<usize> = tour_local.into_iter().map(|li| idxs[li]).collect();

            eprintln!(
                "chunk {chunk_id}: n={} solved in {:.2}s",
                idxs.len(),
                now.elapsed().as_secs_f32()
            );

            Ok(tour_global)
        })
        .collect::<io::Result<Vec<_>>>()?;

    let centroids: Vec<Coord> = chunks
        .iter()
        .map(|idxs| TourGeometry::centroid_of_indices(&global_coords, idxs))
        .collect();

    let order_dir = work_dir.join(CHUNK_ORDER_DIR);
    let order = ChunkSolver::order_by_centroid_tsp(&lkh_exe, &order_dir, &centroids)?;

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
        BOUNDARY_2OPT_WINDOW,
        BOUNDARY_2OPT_PASSES,
    );

    Ok(merged.into_iter().map(|i| input[i]).collect())
}
