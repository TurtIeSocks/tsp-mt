use std::{env, time::Instant};

// use crate::{first::WrapperOptions};

// mod first;
// mod outlier_repair;
// mod outlier_v2;
mod old;
mod outliers;
mod virus;
// mod recursive;
mod processing;
mod utils;

fn main() -> std::io::Result<()> {
    let now = Instant::now();

    // let points = vec![
    //     Point { lat: 40.0, lng: -73.0 },
    //     Point { lat: 40.1, lng: -73.1 },
    //     // ...
    // ];

    let Ok(points) = utils::read_points_from_stdin() else {
        panic!("can't create points");
    };

    eprintln!("Input length: {}", points.len());

    let mut path = env::current_dir()?;
    path.push("tsp");
    let path = format!("{}", path.to_str().unwrap_or_default());

    let args = parse_args();
    let opts = old::Options {
        tsp_path: path.into(),
        leaf_size: args.leaf_size,
        max_leaf_size: args.max_leaf_size,
        // portals: args.portals,
        // seam_refine: args.seam_refine,
        ..Default::default()
    };

    let route = old::solve(&points, &opts)?;

    // let route = match recursive::solve_tsp_parallel(
    //     &points,
    //     &Options {
    //         tsp_path: path.clone(),
    //         ..Default::default()
    //     },
    // ) {
    //     Ok(points) => points,
    //     Err(err) => panic!("TSP run error: {:?}", err),
    // };

    // let route = match first::solve_parallel_tsp(
    //     &points,
    //     &WrapperOptions {
    //         tsp_path: path.clone(),
    //         ..Default::default()
    //     },
    // ) {
    //     Ok(points) => points,
    //     Err(err) => panic!("TSP run error: {:?}", err),
    // };

    for point in route.iter() {
        println!("{}", point.to_string());
    }

    eprintln!("Output length: {}", route.len());
    // eprintln!("Unique length: {}", seen.len());
    eprintln!("Time: {:.2}", now.elapsed().as_secs_f32());
    eprintln!();

    utils::measure_distance_open(&route);
    // run_single(&path, &points);

    Ok(())
}

struct Args {
    leaf_size: usize,
    max_leaf_size: usize,
    // portals: usize,
    // seam_refine: bool,
}

fn parse_args() -> Args {
    let mut args = env::args().skip(1);

    let mut out = Args {
        leaf_size: virus::Options::default().leaf_size,
        max_leaf_size: virus::Options::default().max_leaf_size,
        // portals: virus::Options::default().portals,
        // seam_refine: virus::Options::default().seam_refine,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--leaf_size" => {
                let v = args.next().expect("missing value for --leaf_size");
                out.leaf_size = v.parse().expect("invalid --leaf_size");
            }
            "--max_leaf_size" => {
                let v = args.next().expect("missing value for --max_leaf_size");
                out.max_leaf_size = v.parse().expect("invalid --max_leaf_size");
            }
            // "--portals" => {
            //     let v = args.next().expect("missing value for --portals");
            //     out.portals = v.parse().expect("invalid --portals");
            // }
            // "--seam_refine" => {
            //     let v = args.next().expect("missing value for --seam_refine");
            //     out.seam_refine = parse_bool(&v).expect("invalid --seam_refine");
            // }
            _ => {
                panic!("unknown arg: {arg}");
            }
        }
    }

    out
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" | "on" => Some(true),
        "false" | "0" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

// fn run_single(path: &str, points: &[read_points::Point]) {
//     eprintln!("Running single TSP now");

//     let now = Instant::now();

//     let route = match read_points::run_external_tsp(path, points) {
//         Ok(points) => points,
//         Err(err) => panic!("TSP run error: {:?}", err),
//     };

//     eprintln!("Route length: {}", route.len());
//     eprintln!("Time: {:.2}", now.elapsed().as_secs_f32());
// }
