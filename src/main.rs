use log::info;

use tsp_mt_core::{
    Result, SolverInput, SolverOptions, logging, solve_tsp_with_lkh_h3_chunked, utils,
};

fn main() -> Result<()> {
    let options = SolverOptions::from_args()?;
    logging::init_logger(&options)?;

    // We needed to init the logger before the timer macro
    main_inner(options)
}

#[tsp_mt_derive::timer("main")]
fn main_inner(options: SolverOptions) -> Result<()> {
    let input = SolverInput::from_args()?;

    info!("input: {input}");
    info!("options: {options}");

    let route = solve_tsp_with_lkh_h3_chunked(input, options)?;

    for point in route.iter() {
        println!("{point}");
    }

    utils::tour_distance(&route);

    Ok(())
}
