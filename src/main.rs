use std::time::Instant;

use tsp_mt::{SolverInput, SolverOptions, solve_tsp_with_lkh_h3_chunked, utils};

fn main() -> std::io::Result<()> {
    let now = Instant::now();
    let input = SolverInput::from_args().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("input parsing error:\n{e}"),
        )
    })?;
    let options = SolverOptions::from_args().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("argument parsing error:\n{e}"),
        )
    })?;

    eprintln!("Input length: {}", input.points_len());
    eprintln!();

    eprintln!(
        "Workdir: {:?} | LKH: {:?}",
        input.work_dir_path(),
        input.lkh_path()
    );

    let route = solve_tsp_with_lkh_h3_chunked(input, options)?;

    for point in route.iter() {
        println!("{point}");
    }

    eprintln!("Output length: {}", route.len());
    eprintln!("Time: {:.2}s", now.elapsed().as_secs_f32());

    utils::measure_distance_open(&route);

    Ok(())
}
