use std::env;

/// Runtime options for LKH solving behavior.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    /// Radius used by local tangent-plane projection (in meters).
    pub projection_radius: f64,
    /// Maximum number of points per H3 chunk before hierarchical chunking is applied.
    pub max_chunk_size: usize,
    /// Random seed used when ordering chunk centroids with LKH.
    pub centroid_order_seed: u64,
    /// `MAX_TRIALS` for centroid-ordering LKH run.
    pub centroid_order_max_trials: usize,
    /// `TIME_LIMIT` (seconds) for centroid-ordering LKH run.
    pub centroid_order_time_limit: usize,
    /// Window size for boundary-local 2-opt refinement after chunk stitching.
    pub boundary_2opt_window: usize,
    /// Number of passes for boundary-local 2-opt refinement.
    pub boundary_2opt_passes: usize,
    /// Emit progress logs to stderr when true.
    pub verbose: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            projection_radius: 70.0,
            max_chunk_size: 5_000,
            centroid_order_seed: 999,
            centroid_order_max_trials: 20_000,
            centroid_order_time_limit: 10,
            boundary_2opt_window: 500,
            boundary_2opt_passes: 50,
            verbose: true,
        }
    }
}

impl SolverOptions {
    pub fn from_args() -> Result<Self, String> {
        let mut options = Self::default();
        let mut args = env::args().skip(1).peekable();

        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                return Err(Self::usage().to_string());
            }

            let Some(raw_name) = arg.strip_prefix("--") else {
                return Err(format!("Unexpected argument: {arg}\n\n{}", Self::usage()));
            };

            if raw_name.is_empty() {
                return Err(format!("Invalid option name: {arg}\n\n{}", Self::usage()));
            }

            let (name, value) = split_arg(raw_name, &mut args);

            match name.as_str() {
                "projection-radius" => {
                    options.projection_radius = parse_value::<f64>(&name, value)?;
                }
                "max-chunk-size" => {
                    options.max_chunk_size = parse_value::<usize>(&name, value)?;
                }
                "centroid-order-seed" => {
                    options.centroid_order_seed = parse_value::<u64>(&name, value)?;
                }
                "centroid-order-max-trials" => {
                    options.centroid_order_max_trials = parse_value::<usize>(&name, value)?;
                }
                "centroid-order-time-limit" => {
                    options.centroid_order_time_limit = parse_value::<usize>(&name, value)?;
                }
                "boundary-2opt-window" => {
                    options.boundary_2opt_window = parse_value::<usize>(&name, value)?;
                }
                "boundary-2opt-passes" => {
                    options.boundary_2opt_passes = parse_value::<usize>(&name, value)?;
                }
                "verbose" => {
                    options.verbose = match value {
                        Some(v) => parse_bool(&name, &v)?,
                        None => true,
                    };
                }
                "no-verbose" => {
                    if value.is_some() {
                        return Err(format!("Flag --{name} does not take a value"));
                    }
                    options.verbose = false;
                }
                // Handled by SolverInput::from_args; accepted here to allow dual parsing.
                "lkh-exe" | "work-dir" => {
                    if value.is_none() {
                        return Err(format!("Missing value for --{name}"));
                    }
                }
                _ => {
                    return Err(format!("Unknown option: --{name}\n\n{}", Self::usage()));
                }
            }
        }

        Ok(options)
    }

    pub fn usage() -> &'static str {
        concat!(
            "Usage:\n",
            "  tsp-mt [options] < points.txt\n\n",
            "Options:\n",
            "  --lkh-exe <path>\n",
            "  --work-dir <path>\n",
            "  --projection-radius <f64>\n",
            "  --max-chunk-size <usize>\n",
            "  --centroid-order-seed <u64>\n",
            "  --centroid-order-max-trials <usize>\n",
            "  --centroid-order-time-limit <usize>\n",
            "  --boundary-2opt-window <usize>\n",
            "  --boundary-2opt-passes <usize>\n",
            "  --verbose[=<bool>]\n",
            "  --no-verbose\n",
            "  --help\n",
            "\n",
            "Examples:\n",
            "  tsp-mt --max-chunk-size 2000 --no-verbose < points.txt\n",
            "  tsp-mt --projection-radius=100 --verbose=false < points.txt\n",
        )
    }
}

fn parse_value<T>(name: &str, value: Option<String>) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = value.ok_or_else(|| format!("Missing value for --{name}"))?;
    raw.parse::<T>()
        .map_err(|e| format!("Invalid value for --{name}: {raw} ({e})"))
}

fn parse_bool(name: &str, value: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "False" | "no" | "NO" | "off" | "OFF" => Ok(false),
        _ => Err(format!(
            "Invalid boolean for --{name}: {value} (expected true/false)"
        )),
    }
}

fn split_arg(
    raw_name: &str,
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
) -> (String, Option<String>) {
    if let Some((k, v)) = raw_name.split_once('=') {
        return (k.to_string(), Some(v.to_string()));
    }

    let value = match args.peek() {
        Some(next) if !next.starts_with("--") => args.next(),
        _ => None,
    };

    (raw_name.to_string(), value)
}
