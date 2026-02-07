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
        config::LkhConfig, geometry::TourGeometry, h3_chunking::H3Chunker, run_spec::RunSpec,
        solver::LkhSolver, stitching::TourStitcher,
    },
    project::Plane,
    utils::Point,
};

struct ChunkSolver;

impl ChunkSolver {
    fn solve_single(
        lkh_exe: PathBuf,
        work_dir: PathBuf,
        chunk_points: &[Point],
    ) -> io::Result<Vec<usize>> {
        let n = chunk_points.len();
        if n < 3 {
            return Ok(chunk_points
                .iter()
                .enumerate()
                .map(|(idx, _)| idx)
                .collect());
        }

        let cfg = LkhConfig::new(n);
        let solver = LkhSolver::new(lkh_exe, work_dir);
        solver.create_work_dir()?;

        let pts = Plane::new(&chunk_points.to_vec()).radius(70.0).project();

        solver.create_problem_file(&pts)?;
        solver.ensure_candidate_file(pts.len())?;

        let rs = RunSpec {
            idx: 0,
            seed: cfg.base_seed(),
            par_path: solver.work_dir().join("run_0.par"),
            tour_path: solver.work_dir().join("run_0.tour"),
        };
        rs.write_lkh_par(&cfg, &solver)?;

        let out = solver.run(&rs.par_path)?;

        if !out.status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "LKH chunk failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr),
                ),
            ));
        }

        rs.parse_tsplib_tour(n)
    }

    fn order_by_centroid_tsp(
        lkh_exe: &Path,
        work_dir: &Path,
        centroids: &[Coord],
    ) -> io::Result<Vec<usize>> {
        if centroids.len() <= 2 {
            return Ok((0..centroids.len()).collect());
        }

        fs::create_dir_all(work_dir)?;

        let problem = work_dir.join("centroids.tsp");
        Self::write_centroid_problem(&problem, centroids)?;

        let rs = RunSpec {
            idx: 0,
            seed: 999,
            par_path: work_dir.join("centroids.par"),
            tour_path: work_dir.join("centroids.tour"),
        };
        rs.write_lkh_par_small(&problem, 20_000, 10)?;

        let out = Command::new(lkh_exe)
            .arg(&rs.par_path)
            .current_dir(work_dir)
            .output()?;

        if !out.status.success() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Centroid ordering LKH failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr),
                ),
            ));
        }

        rs.parse_tsplib_tour(centroids.len())
    }

    fn write_centroid_problem(problem: &Path, centroids: &[Coord]) -> io::Result<()> {
        let mut s = String::new();
        s.push_str("NAME: centroids\nTYPE: TSP\n");
        s.push_str(&format!("DIMENSION: {}\n", centroids.len()));
        s.push_str("EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n");
        for (i, p) in centroids.iter().enumerate() {
            s.push_str(&format!(
                "{} {:.0} {:.0}\n",
                i + 1,
                p.x * 1000.0,
                p.y * 1000.0
            ));
        }
        s.push_str("EOF\n");
        fs::write(problem, s)
    }
}

pub fn solve_tsp_with_lkh_h3_chunked(
    lkh_exe: PathBuf,
    work_dir: PathBuf,
    input: &[Point],
) -> io::Result<Vec<Point>> {
    const MAX_CHUNK: usize = 5_000;

    if input.len() <= MAX_CHUNK {
        return super::solve_tsp_with_lkh_parallel(lkh_exe, work_dir, input);
    }

    let global_coords = Plane::new(&input.to_vec()).radius(70.0).project();

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
            let chunk_dir = work_dir.join(format!("chunk_{chunk_id}"));

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

    let order_dir = work_dir.join("chunk_order");
    let order = ChunkSolver::order_by_centroid_tsp(&lkh_exe, &order_dir, &centroids)?;

    let mut ordered_tours: Vec<Vec<usize>> = Vec::with_capacity(solved_chunk_tours.len());
    for ci in order {
        ordered_tours.push(solved_chunk_tours[ci].clone());
    }

    let (mut merged, boundaries) =
        TourStitcher::stitch_chunk_tours_dense(&global_coords, ordered_tours);

    TourStitcher::boundary_two_opt(&global_coords, &mut merged, &boundaries, 500, 50);

    Ok(merged.into_iter().map(|i| input[i]).collect())
}
