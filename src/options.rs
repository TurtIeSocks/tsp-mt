use std::{env, path::PathBuf};

use log::LevelFilter;
use tsp_mt_derive::{CliOptions, CliValue, KvDisplay};

use crate::{Error, Result};

/// Anything above this is treated as invalid rather than fed to
/// `Duration::from_secs_f64`, which panics on huge or non-finite values.
const MAX_TIME_LIMIT_SECS: f64 = 1e9;

/// Runtime options for the built-in solver.
#[derive(Clone, Debug, CliOptions, KvDisplay)]
pub struct SolverOptions {
    /// Wall-clock time budget in seconds. `0` picks one from the input size.
    #[cli(long = "time-limit")]
    pub time_limit: f64,
    /// Random seed for the solver's perturbation phase.
    #[cli(long = "seed")]
    pub seed: u64,
    /// Worker threads. `0` uses all available cores.
    #[cli(long = "threads")]
    pub threads: usize,
    /// Nearest-neighbor count for candidate-edge generation.
    #[cli(long = "max-neighbors")]
    pub max_neighbors: usize,
    /// Distance threshold factor (multiples of the average edge) used when
    /// counting route outlier spikes in metrics logs.
    #[cli(long = "outlier-threshold")]
    pub outlier_threshold: f64,
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
    /// Optional input file path for points. Empty means stdin.
    #[cli(long = "input")]
    pub input: String,
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
            time_limit: 0.0,
            seed: 12_345,
            threads: 0,
            max_neighbors: 10,
            outlier_threshold: 10.0,
            log_level: LogLevel::Warn,
            log_format: LogFormat::Compact,
            log_timestamp: true,
            log_output: String::new(),
            input: String::new(),
            output: String::new(),
        }
    }
}

impl SolverOptions {
    pub fn from_args() -> Result<Self> {
        let cli_args: Vec<String> = env::args().skip(1).collect();
        Self::parse_from_iter(cli_args)
    }

    fn parse_from_iter<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut options = Self::default();
        Self::apply_args(&mut options, args)?;
        Ok(options)
    }

    fn apply_args<I, S>(options: &mut Self, args: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
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
                _ => {
                    return Err(Error::invalid_input(format!(
                        "Unknown option: --{name}\n\n{}",
                        Self::usage()
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn usage() -> &'static str {
        concat!(
            "Usage:\n",
            "  tsp-mt [options] [--input points.txt]\n",
            "  tsp-mt [options] < points.txt\n\n",
            "Options:\n",
            "  --time-limit <seconds>      wall-clock budget (0 = auto from input size)\n",
            "  --seed <u64>                RNG seed for the perturbation phase\n",
            "  --threads <usize>           worker threads (0 = all cores)\n",
            "  --max-neighbors <usize>     candidate edges per point (default 10)\n",
            "  --outlier-threshold <f64>   spike threshold as multiple of avg edge\n",
            "  --log-level <error|warn|info|debug|trace|off>\n",
            "  --log-format <compact|pretty>\n",
            "  --log-timestamp[=<bool>]\n",
            "  --no-log-timestamp\n",
            "  --log-output <path>\n",
            "  --input <path>\n",
            "  --output <path>\n",
            "  --help\n",
            "\n",
            "Examples:\n",
            "  tsp-mt --output route.txt < points.txt\n",
            "  tsp-mt --input points.txt --output route.txt --time-limit 30\n",
            "  tsp-mt --threads 8 --seed 42 --log-level=info < points.txt\n",
        )
    }

    /// Translate CLI options into a solver configuration, validating ranges
    /// that would otherwise panic downstream (`Duration::from_secs_f64`).
    pub fn to_config(&self) -> Result<tsp_geo::SolverConfig> {
        if !self.time_limit.is_finite()
            || self.time_limit < 0.0
            || self.time_limit > MAX_TIME_LIMIT_SECS
        {
            return Err(Error::invalid_input(format!(
                "--time-limit must be a finite number of seconds in [0, {MAX_TIME_LIMIT_SECS:.0}], got {}",
                self.time_limit
            )));
        }
        let mut cfg = tsp_geo::SolverConfig::default();
        if self.time_limit > 0.0 {
            cfg.time_limit = Some(std::time::Duration::from_secs_f64(self.time_limit));
        }
        cfg.seed = self.seed;
        cfg.threads = self.threads;
        cfg.max_neighbors = self.max_neighbors.max(4);
        cfg.max_candidates = cfg.max_candidates.max(cfg.max_neighbors + 6);
        Ok(cfg)
    }

    pub fn log_output_path(&self) -> Option<PathBuf> {
        check_path(&self.log_output)
    }

    pub fn output_path(&self) -> Option<PathBuf> {
        check_path(&self.output)
    }

    pub fn input_path(&self) -> Option<PathBuf> {
        check_path(&self.input)
    }
}

fn check_path(path_str: &str) -> Option<PathBuf> {
    let path_str = path_str.trim();
    if path_str.is_empty() || path_str == "-" {
        None
    } else {
        normalize_path(path_str).ok()
    }
}

fn normalize_path(path: impl Into<PathBuf>) -> Result<PathBuf> {
    let path = path.into();
    if path == PathBuf::new() {
        return Ok(path);
    }
    std::path::absolute(path).map_err(Error::Io)
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
            "--time-limit=30.5",
            "--seed=77",
            "--threads=8",
            "--max-neighbors=12",
            "--outlier-threshold=12.5",
            "--log-level=debug",
            "--log-format=pretty",
            "--log-timestamp=false",
            "--log-output=run.log",
            "--input=points.txt",
            "--output=route.txt",
        ])
        .expect("parse options");

        assert_eq!(options.time_limit, 30.5);
        assert_eq!(options.seed, 77);
        assert_eq!(options.threads, 8);
        assert_eq!(options.max_neighbors, 12);
        assert_eq!(options.outlier_threshold, 12.5);
        assert_eq!(options.log_level, LogLevel::Debug);
        assert_eq!(options.log_format, LogFormat::Pretty);
        assert!(!options.log_timestamp);
        assert_eq!(options.log_output, "run.log");
        assert_eq!(options.input, "points.txt");
        assert_eq!(options.output, "route.txt");
    }

    #[test]
    fn to_config_maps_options_and_scales_candidate_cap() {
        let options = SolverOptions {
            time_limit: 30.0,
            seed: 7,
            threads: 4,
            max_neighbors: 24,
            ..SolverOptions::default()
        };
        let cfg = options.to_config().expect("valid config");
        assert_eq!(cfg.time_limit, Some(std::time::Duration::from_secs(30)));
        assert_eq!(cfg.seed, 7);
        assert_eq!(cfg.threads, 4);
        assert_eq!(cfg.max_neighbors, 24);
        assert!(cfg.max_candidates >= 30, "cap must scale with neighbors");
    }

    #[test]
    fn to_config_rejects_non_finite_or_absurd_time_limits() {
        for bad in [f64::INFINITY, f64::NAN, -1.0, 1e300] {
            let options = SolverOptions {
                time_limit: bad,
                ..SolverOptions::default()
            };
            let err = options
                .to_config()
                .expect_err("bad time limit should be rejected");
            assert!(err.to_string().contains("--time-limit"), "{err}");
        }
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
    fn parse_from_iter_rejects_removed_lkh_options() {
        for stale in [
            "--lkh-exe=/bin/lkh",
            "--max-chunk-size=100",
            "--work-dir=/tmp",
        ] {
            let err = SolverOptions::parse_from_iter([stale])
                .expect_err("removed options should be rejected");
            assert!(err.to_string().contains("Unknown option"));
        }
    }

    #[test]
    fn parse_from_iter_rejects_unexpected_positional_argument() {
        let err =
            SolverOptions::parse_from_iter(["points.txt"]).expect_err("expected positional error");
        assert!(err.to_string().contains("Unexpected argument: points.txt"));
    }

    #[test]
    fn parse_from_iter_requires_value_for_options() {
        let err =
            SolverOptions::parse_from_iter(["--seed"]).expect_err("missing value should fail");
        assert!(err.to_string().contains("Missing value for --seed"));
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
    fn input_path_treats_empty_and_dash_as_stdin() {
        let options = SolverOptions::default();
        assert!(options.input_path().is_none());

        let options = SolverOptions {
            input: "-".to_string(),
            ..SolverOptions::default()
        };
        assert!(options.input_path().is_none());
    }

    #[test]
    fn input_path_returns_path_for_non_empty_value() {
        let options = SolverOptions {
            input: "in/points.txt".to_string(),
            ..SolverOptions::default()
        };
        let expected = std::path::absolute("in/points.txt").expect("absolute path");
        assert_eq!(options.input_path().expect("path should exist"), expected);
    }

    #[test]
    fn output_path_returns_path_for_non_empty_value() {
        let options = SolverOptions {
            output: "out/route.txt".to_string(),
            ..SolverOptions::default()
        };
        let expected = std::path::absolute("out/route.txt").expect("absolute path");
        assert_eq!(options.output_path().expect("path should exist"), expected);
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
        let expected = std::path::absolute("out/run.log").expect("absolute path");
        assert_eq!(
            options.log_output_path().expect("path should exist"),
            expected
        );
    }
}
