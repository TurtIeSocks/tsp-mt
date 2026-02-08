use std::{env, path::Path};

use log::LevelFilter;
use tsp_mt_derive::{CliOptions, CliValue, KvDisplay};

use crate::{Error, Result};

/// Runtime options for LKH solving behavior.
#[derive(Clone, Debug, CliOptions, KvDisplay)]
pub struct SolverOptions {
    /// Radius used by local tangent-plane projection (in meters).
    #[cli(long = "projection-radius")]
    pub projection_radius: f64,
    /// Maximum number of points per H3 chunk before hierarchical chunking is applied.
    #[cli(long = "max-chunk-size")]
    pub max_chunk_size: usize,
    /// Random seed used when ordering chunk centroids with LKH.
    #[cli(long = "centroid-order-seed")]
    pub centroid_order_seed: u64,
    /// `MAX_TRIALS` for centroid-ordering LKH run.
    #[cli(long = "centroid-order-max-trials")]
    pub centroid_order_max_trials: usize,
    /// `TIME_LIMIT` (seconds) for centroid-ordering LKH run.
    #[cli(long = "centroid-order-time-limit")]
    pub centroid_order_time_limit: usize,
    /// Window size for boundary-local 2-opt refinement after chunk stitching.
    #[cli(long = "boundary-2opt-window")]
    pub boundary_2opt_window: usize,
    /// Number of passes for boundary-local 2-opt refinement.
    #[cli(long = "boundary-2opt-passes")]
    pub boundary_2opt_passes: usize,
    /// Structured logging level.
    #[cli(long = "log-level", parse_with = "LogLevel::parse")]
    pub log_level: LogLevel,
    /// Logging output format.
    #[cli(long = "log-format", parse_with = "LogFormat::parse")]
    pub log_format: LogFormat,
    /// Include timestamps in log lines.
    pub log_timestamp: bool,
    /// Optional output file path for logs and metrics. Empty means stderr.
    #[cli(long = "log-output")]
    pub log_output: String,
    /// Optional output file path for ordered route points. Empty means stdout.
    #[cli(long = "output")]
    pub output: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, CliValue)]
#[cli_value(option = "log-level")]
pub enum LogLevel {
    Error,
    #[cli(alias = "warning")]
    Warn,
    Info,
    Debug,
    Trace,
    Off,
}

impl LogLevel {
    pub fn to_filter(self) -> LevelFilter {
        match self {
            Self::Error => LevelFilter::Error,
            Self::Warn => LevelFilter::Warn,
            Self::Info => LevelFilter::Info,
            Self::Debug => LevelFilter::Debug,
            Self::Trace => LevelFilter::Trace,
            Self::Off => LevelFilter::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, CliValue)]
#[cli_value(option = "log-format")]
pub enum LogFormat {
    Compact,
    Pretty,
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
            log_level: LogLevel::Warn,
            log_format: LogFormat::Compact,
            log_timestamp: true,
            log_output: String::new(),
            output: String::new(),
        }
    }
}

impl SolverOptions {
    pub fn from_args() -> Result<Self> {
        Self::parse_from_iter(env::args().skip(1))
    }

    fn parse_from_iter<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut options = Self::default();
        let mut args = args
            .into_iter()
            .map(|arg| arg.as_ref().to_owned())
            .peekable();

        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                return Err(Error::invalid_input(Self::usage()));
            }

            let Some(raw_name) = arg.strip_prefix("--") else {
                return Err(Error::invalid_input(format!(
                    "Unexpected argument: {arg}\n\n{}",
                    Self::usage()
                )));
            };

            if raw_name.is_empty() {
                return Err(Error::invalid_input(format!(
                    "Invalid option name: {arg}\n\n{}",
                    Self::usage()
                )));
            }

            let (name, value) = Self::split_arg(raw_name, &mut args);

            if options.apply_cli_option(&name, value.clone())? {
                continue;
            }

            match name.as_str() {
                "log-timestamp" => {
                    options.log_timestamp = match value {
                        Some(v) => parse_bool(&name, &v)?,
                        None => true,
                    };
                }
                "no-log-timestamp" => {
                    if value.is_some() {
                        return Err(Error::invalid_input(format!(
                            "Flag --{name} does not take a value"
                        )));
                    }
                    options.log_timestamp = false;
                }
                // Handled by SolverInput::from_args; accepted here to allow dual parsing.
                "lkh-exe" | "work-dir" => {
                    if value.is_none() {
                        return Err(Error::invalid_input(format!("Missing value for --{name}")));
                    }
                }
                _ => {
                    return Err(Error::invalid_input(format!(
                        "Unknown option: --{name}\n\n{}",
                        Self::usage()
                    )));
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
            "  --log-level <error|warn|info|debug|trace|off>\n",
            "  --log-format <compact|pretty>\n",
            "  --log-timestamp[=<bool>]\n",
            "  --no-log-timestamp\n",
            "  --log-output <path>\n",
            "  --output <path>\n",
            "  --help\n",
            "\n",
            "Examples:\n",
            "  tsp-mt --max-chunk-size 2000 --log-level warn --output output.txt < points.txt\n",
            "  tsp-mt --log-level=info --log-output run.log < points.txt\n",
            "  tsp-mt --log-level=debug --log-format=pretty --log-timestamp < points.txt\n",
            "  tsp-mt --projection-radius=100 --log-level=info < points.txt\n",
        )
    }

    pub fn log_output_path(&self) -> Option<&Path> {
        let log_output = self.log_output.trim();
        if log_output.is_empty() || log_output == "-" {
            None
        } else {
            Some(Path::new(log_output))
        }
    }

    pub fn output_path(&self) -> Option<&Path> {
        let output = self.output.trim();
        if output.is_empty() || output == "-" {
            None
        } else {
            Some(Path::new(output))
        }
    }
}

fn parse_bool(name: &str, value: &str) -> Result<bool> {
    match value {
        "1" | "true" | "TRUE" | "True" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "False" | "no" | "NO" | "off" | "OFF" => Ok(false),
        _ => Err(Error::invalid_input(format!(
            "Invalid boolean for --{name}: {value} (expected true/false)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use log::LevelFilter;

    use super::{LogFormat, LogLevel, SolverOptions, parse_bool};

    #[test]
    fn parse_bool_accepts_common_true_values() {
        assert!(parse_bool("x", "true").expect("parse"));
        assert!(parse_bool("x", "1").expect("parse"));
        assert!(parse_bool("x", "YES").expect("parse"));
        assert!(parse_bool("x", "ON").expect("parse"));
    }

    #[test]
    fn parse_bool_accepts_common_false_values() {
        assert!(!parse_bool("x", "false").expect("parse"));
        assert!(!parse_bool("x", "0").expect("parse"));
        assert!(!parse_bool("x", "NO").expect("parse"));
        assert!(!parse_bool("x", "off").expect("parse"));
    }

    #[test]
    fn parse_bool_rejects_unknown_values() {
        let err = parse_bool("log-timestamp", "maybe").expect_err("invalid bool should fail");
        assert!(
            err.to_string()
                .contains("Invalid boolean for --log-timestamp: maybe")
        );
    }

    #[test]
    fn log_level_maps_to_expected_filter() {
        assert_eq!(LogLevel::Error.to_filter(), LevelFilter::Error);
        assert_eq!(LogLevel::Warn.to_filter(), LevelFilter::Warn);
        assert_eq!(LogLevel::Info.to_filter(), LevelFilter::Info);
        assert_eq!(LogLevel::Debug.to_filter(), LevelFilter::Debug);
        assert_eq!(LogLevel::Trace.to_filter(), LevelFilter::Trace);
        assert_eq!(LogLevel::Off.to_filter(), LevelFilter::Off);
    }

    #[test]
    fn parse_from_iter_applies_known_cli_options() {
        let options = SolverOptions::parse_from_iter([
            "--projection-radius=120.5",
            "--max-chunk-size=42",
            "--centroid-order-seed=77",
            "--centroid-order-max-trials=200",
            "--centroid-order-time-limit=9",
            "--boundary-2opt-window=8",
            "--boundary-2opt-passes=7",
            "--log-level=debug",
            "--log-format=pretty",
            "--log-timestamp=false",
            "--log-output=run.log",
            "--output=route.txt",
        ])
        .expect("parse options");

        assert_eq!(options.projection_radius, 120.5);
        assert_eq!(options.max_chunk_size, 42);
        assert_eq!(options.centroid_order_seed, 77);
        assert_eq!(options.centroid_order_max_trials, 200);
        assert_eq!(options.centroid_order_time_limit, 9);
        assert_eq!(options.boundary_2opt_window, 8);
        assert_eq!(options.boundary_2opt_passes, 7);
        assert_eq!(options.log_level, LogLevel::Debug);
        assert_eq!(options.log_format, LogFormat::Pretty);
        assert!(!options.log_timestamp);
        assert_eq!(options.log_output, "run.log");
        assert_eq!(options.output, "route.txt");
    }

    #[test]
    fn parse_from_iter_accepts_no_log_timestamp_flag() {
        let options =
            SolverOptions::parse_from_iter(["--no-log-timestamp"]).expect("parse options");
        assert!(!options.log_timestamp);
    }

    #[test]
    fn parse_from_iter_rejects_no_log_timestamp_with_value() {
        let err = SolverOptions::parse_from_iter(["--no-log-timestamp=true"])
            .expect_err("expected flag value rejection");
        assert!(err.to_string().contains("does not take a value"));
    }

    #[test]
    fn parse_from_iter_rejects_unknown_option() {
        let err = SolverOptions::parse_from_iter(["--unknown-opt=1"])
            .expect_err("expected unknown option error");
        assert!(err.to_string().contains("Unknown option: --unknown-opt"));
    }

    #[test]
    fn parse_from_iter_rejects_unexpected_positional_argument() {
        let err =
            SolverOptions::parse_from_iter(["points.txt"]).expect_err("expected positional error");
        assert!(err.to_string().contains("Unexpected argument: points.txt"));
    }

    #[test]
    fn parse_from_iter_allows_input_passthrough_options_with_values() {
        SolverOptions::parse_from_iter(["--lkh-exe=/bin/lkh", "--work-dir=/tmp/work"])
            .expect("passthrough options should be accepted");
    }

    #[test]
    fn parse_from_iter_requires_value_for_passthrough_options() {
        let err =
            SolverOptions::parse_from_iter(["--lkh-exe"]).expect_err("missing value should fail");
        assert!(err.to_string().contains("Missing value for --lkh-exe"));
    }

    #[test]
    fn parse_from_iter_help_returns_usage_error() {
        let err =
            SolverOptions::parse_from_iter(["--help"]).expect_err("help should short-circuit");
        assert!(err.to_string().contains("Usage:"));
    }

    #[test]
    fn output_path_treats_empty_and_dash_as_stdout() {
        let options = SolverOptions::default();
        assert!(options.output_path().is_none());

        let options = SolverOptions {
            output: "-".to_string(),
            ..SolverOptions::default()
        };
        assert!(options.output_path().is_none());
    }

    #[test]
    fn output_path_returns_path_for_non_empty_value() {
        let options = SolverOptions {
            output: "out/route.txt".to_string(),
            ..SolverOptions::default()
        };
        assert_eq!(
            options.output_path().expect("path should exist"),
            std::path::Path::new("out/route.txt")
        );
    }

    #[test]
    fn log_output_path_treats_empty_and_dash_as_stderr() {
        let options = SolverOptions::default();
        assert!(options.log_output_path().is_none());

        let options = SolverOptions {
            log_output: "-".to_string(),
            ..SolverOptions::default()
        };
        assert!(options.log_output_path().is_none());
    }

    #[test]
    fn log_output_path_returns_path_for_non_empty_value() {
        let options = SolverOptions {
            log_output: "out/run.log".to_string(),
            ..SolverOptions::default()
        };
        assert_eq!(
            options.log_output_path().expect("path should exist"),
            std::path::Path::new("out/run.log")
        );
    }
}
