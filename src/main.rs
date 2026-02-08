use std::{
    env,
    fs::File,
    io::{Write, stdout},
    path::PathBuf,
};

use log::info;

use tsp_mt_core::{Result, SolverInput, SolverOptions, logging, solve_tsp_with_lkh_h3_chunked};

fn main() -> Result<()> {
    if env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        println!("{}", SolverOptions::usage());
        return Ok(());
    }

    let options = SolverOptions::from_args()?;
    logging::init_logger(&options)?;

    // We needed to init the logger before the timer macro
    main_inner(options)
}

#[tsp_mt_derive::timer("main")]
fn main_inner(options: SolverOptions) -> Result<()> {
    let input = SolverInput::from_args()?;
    let output_path = options.output_path().map(PathBuf::from);
    let outlier_threshold = options.outlier_threshold;

    info!("input: {input}");
    info!("options: {options}");

    let tour = solve_tsp_with_lkh_h3_chunked(input, options)?;
    tour.tour_metrics(outlier_threshold);
    write_route_output(&tour.nodes, output_path.as_deref())?;

    Ok(())
}

fn write_route_output(
    route: &[tsp_mt_core::LKHNode],
    output_path: Option<&std::path::Path>,
) -> Result<()> {
    let mut writer: Box<dyn Write> = match output_path {
        Some(path) => Box::new(File::create(path)?),
        None => Box::new(stdout()),
    };

    for point in route {
        writeln!(writer, "{point}")?;
    }
    writer.flush()?;
    Ok(())
}
