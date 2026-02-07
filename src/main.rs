use std::time::Instant;

use log::info;

use tsp_mt_core::{
    Result, SolverInput, SolverOptions, logging, solve_tsp_with_lkh_h3_chunked, utils,
};

fn main() -> Result<()> {
    let now = Instant::now();
    let options = SolverOptions::from_args()?;
    logging::init_logger(&options)?;
    let input = SolverInput::from_args()?;

    info!("input: {input}");
    info!("options: {options}");

    let route = solve_tsp_with_lkh_h3_chunked(input, options)?;

    for point in route.iter() {
        println!("{point}");
    }

    info!(
        "output: n={} time={:.2}s",
        route.len(),
        now.elapsed().as_secs_f32()
    );

    utils::tour_distance(&route);

    Ok(())
}
