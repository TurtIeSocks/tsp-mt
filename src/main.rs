mod input;
mod logging;
mod options;

use std::{
    fs::{self, File},
    io::{Write, stdout},
};

use log::info;

pub(crate) use tsp_geo::{Error, Result};
use tsp_geo::{GeoPoint, Tour};

use input::SolverInput;
use options::SolverOptions;

fn main() -> Result<()> {
    if std::env::args()
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
    let input = SolverInput::from_args(&options)?;
    let config = options.to_config()?;
    let output_path = options.output_path();
    let outlier_threshold = options.outlier_threshold;

    info!("input: {input}");
    info!("options: {options}");
    let tour = Tour::new(tsp_geo::solve(&input.nodes, &config)?);
    tour.tour_metrics(outlier_threshold);
    write_route_output(&tour.nodes, output_path.as_deref())?;

    Ok(())
}

fn write_route_output(route: &[GeoPoint], output_path: Option<&std::path::Path>) -> Result<()> {
    let mut writer: Box<dyn Write> = match output_path {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            Box::new(File::create(path)?)
        }
        None => Box::new(stdout()),
    };

    for point in route {
        writeln!(writer, "{point}")?;
    }
    writer.flush()?;
    Ok(())
}
