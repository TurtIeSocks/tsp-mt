use std::time::Instant;

use log::info;

use tsp_mt::{Result, SolverInput, SolverOptions, logging, solve_tsp_with_lkh_h3_chunked, utils};

fn main() -> Result<()> {
    let now = Instant::now();
    let options = SolverOptions::from_args()?;
    logging::init_logger(&options)?;
    let input = SolverInput::from_args()?;

    info!("Input length: {}", input.points_len());

    info!(
        "Workdir: {:?} | LKH: {:?}",
        input.work_dir_path(),
        input.lkh_path()
    );

    let route = solve_tsp_with_lkh_h3_chunked(input, options)?;

    for point in route.iter() {
        println!("{point}");
    }

    info!("Output length: {}", route.len());
    info!("Time: {:.2}s", now.elapsed().as_secs_f32());

    utils::measure_distance_open(&route);

    Ok(())
}
