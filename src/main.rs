use std::{env, time::Instant};

use tsp_mt::{LKHNode, SolverInput, SolverOptions, solve_tsp_with_lkh_h3_chunked, utils};

// mod outliers;
// mod processing;
// mod stats;
// mod tsp;

fn main() -> std::io::Result<()> {
    let now = Instant::now();

    let Ok(points) = utils::read_points_from_stdin() else {
        panic!("can't create points");
    };

    eprintln!("Input length: {}", points.len());
    eprintln!();

    let current = env::current_dir()?;
    let lkh = current.join("lkh/LKH-3.0.13/LKH");
    let work_dir = current.join("temp");

    eprintln!("Workdir: {:?} | LKH: {lkh:?}", &work_dir);

    let input: Vec<LKHNode> = points
        .iter()
        .map(|p| LKHNode::new(p.lat, p.lng))
        .collect();

    let route = solve_tsp_with_lkh_h3_chunked(
        SolverInput::new(&lkh, &work_dir, &input),
        SolverOptions::default(),
    )?;
    // let path = format!("{}", path.to_str().unwrap_or_default());

    // let args = parse_args();
    // let opts = tsp::Options {
    //     tsp_path: path.into(),
    //     leaf_size: args.leaf_size,
    //     max_leaf_size: args.max_leaf_size,
    //     // portals: args.portals,
    //     // seam_refine: args.seam_refine,
    //     ..Default::default()
    // };

    // let route = tsp::solve(&points, &opts)?;
    // let route = route.into_iter().map(|idx| points[idx]).collect::<Vec<_>>();

    for point in route.iter() {
        println!("{point}");
    }

    eprintln!("Output length: {}", route.len());
    eprintln!("Time: {:.2}s", now.elapsed().as_secs_f32());

    let route_for_metrics: Vec<utils::Point> = route
        .iter()
        .map(|p| utils::Point { lat: p.y, lng: p.x })
        .collect();
    utils::measure_distance_open(&route_for_metrics);
    // run_single(&path, &points);

    Ok(())
}

// struct Args {
//     leaf_size: usize,
//     max_leaf_size: usize,
//     // portals: usize,
//     // seam_refine: bool,
// }

// fn parse_args() -> Args {
//     let mut args = env::args().skip(1);

//     let mut out = Args {
//         leaf_size: tsp::Options::default().leaf_size,
//         max_leaf_size: tsp::Options::default().max_leaf_size,
//         // portals: old::Options::default().portals,
//         // seam_refine: old::Options::default().seam_refine,
//     };

//     while let Some(arg) = args.next() {
//         match arg.as_str() {
//             "--leaf_size" => {
//                 let v = args.next().expect("missing value for --leaf_size");
//                 out.leaf_size = v.parse().expect("invalid --leaf_size");
//             }
//             "--max_leaf_size" => {
//                 let v = args.next().expect("missing value for --max_leaf_size");
//                 out.max_leaf_size = v.parse().expect("invalid --max_leaf_size");
//             }
//             // "--portals" => {
//             //     let v = args.next().expect("missing value for --portals");
//             //     out.portals = v.parse().expect("invalid --portals");
//             // }
//             // "--seam_refine" => {
//             //     let v = args.next().expect("missing value for --seam_refine");
//             //     out.seam_refine = parse_bool(&v).expect("invalid --seam_refine");
//             // }
//             _ => {
//                 panic!("unknown arg: {arg}");
//             }
//         }
//     }

//     out
// }

// fn parse_bool(s: &str) -> Option<bool> {
//     match s.to_ascii_lowercase().as_str() {
//         "true" | "1" | "yes" | "y" | "on" => Some(true),
//         "false" | "0" | "no" | "n" | "off" => Some(false),
//         _ => None,
//     }
// }

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
