use std::{
    env,
    path::{Path, PathBuf},
    process,
};

#[cfg(feature = "fetch-lkh")]
use lkh::embedded_lkh;
use log::LevelFilter;
use tsp_mt_derive::{CliOptions, CliValue, KvDisplay};
// use walkdir::WalkDir;

use crate::{Error, Result};

// const CONFIG_FILE_NAME: &str = ".tsp-mt.config.toml";
// const CONFIG_PATH_ENV: &str = "TSP_MT_CONFIG";

/// Runtime options for LKH solving behavior.
#[derive(Clone, Debug, CliOptions, KvDisplay)]
pub struct SolverOptions {
    /// Path to the LKH executable.
    #[cli(long = "lkh-exe")]
    #[kv(fmt = "path")]
    pub lkh_exe: PathBuf,
    /// Working directory for temporary files and run artifacts.
    #[cli(long = "work-dir")]
    #[kv(fmt = "path")]
    pub work_dir: PathBuf,
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
    /// Solver strategy to run: `single`, `multi-seed`, or `multi-parallel`.
    #[cli(long = "solver-mode", parse_with = "SolverMode::parse")]
    pub solver_mode: SolverMode,
    /// Whether to remove the solver work directory after the run.
    pub cleanup: bool,
    /// Window size for boundary-local 2-opt refinement after chunk stitching.
    #[cli(long = "boundary-2opt-window")]
    pub boundary_2opt_window: usize,
    /// Number of passes for boundary-local 2-opt refinement.
    #[cli(long = "boundary-2opt-passes")]
    pub boundary_2opt_passes: usize,
    /// Number of longest edges targeted by the post-stitch spike-repair pass.
    #[cli(long = "spike-repair-top-n")]
    pub spike_repair_top_n: usize,
    /// 2-opt window used by the post-stitch spike-repair pass.
    #[cli(long = "spike-repair-window")]
    pub spike_repair_window: usize,
    /// 2-opt passes used by the post-stitch spike-repair pass.
    #[cli(long = "spike-repair-passes")]
    pub spike_repair_passes: usize,
    /// Distance threshold (meters) used when counting route outlier spikes in metrics logs.
    #[cli(long = "outlier-threshold")]
    pub outlier_threshold: f64,
    /// Build and pass an `INITIAL_TOUR_FILE` to each LKH run using the input point order.
    pub use_initial_tour: bool,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, CliValue)]
#[cli_value(option = "solver-mode")]
pub enum SolverMode {
    Single,
    MultiSeed,
    MultiParallel,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            lkh_exe: PathBuf::new(),
            work_dir: default_work_dir(),
            projection_radius: 70.0,
            max_chunk_size: 5_000,
            centroid_order_seed: 999,
            centroid_order_max_trials: 20_000,
            centroid_order_time_limit: 10,
            solver_mode: SolverMode::MultiParallel,
            cleanup: true,
            boundary_2opt_window: 500,
            boundary_2opt_passes: 50,
            spike_repair_top_n: 48,
            spike_repair_window: 700,
            spike_repair_passes: 5,
            outlier_threshold: 10.0,
            use_initial_tour: false,
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

        #[cfg(not(feature = "fetch-lkh"))]
        {
            let (options, saw_lkh_exe) = Self::parse_with_config(cli_args)?;
            if !saw_lkh_exe {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "`fetch-lkh feature is not enabled and the `--lkh-exe` arg was not provided",
                )));
            }
            Ok(options)
        }

        #[cfg(feature = "fetch-lkh")]
        {
            let (mut options, saw_lkh_exe) = Self::parse_with_config(cli_args)?;
            if !saw_lkh_exe {
                options.lkh_exe = embedded_lkh::embedded_path()?;
            }
            Ok(options)
        }
    }

    fn parse_with_config<I, S>(cli_args: I) -> Result<(Self, bool)>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut options = Self::default();
        let mut saw_lkh_exe = false;

        // if let Some(config_path) = find_config_file()? {
        //     let config_args = Self::config_args_from_path(&config_path)?;
        //     saw_lkh_exe |= Self::apply_args(&mut options, config_args)?;
        // }

        saw_lkh_exe |= Self::apply_args(&mut options, cli_args)?;

        options.work_dir = normalize_path(options.work_dir)?;
        options.lkh_exe = normalize_path(options.lkh_exe)?;
        Ok((options, saw_lkh_exe))
    }

    #[cfg(test)]
    fn parse_from_iter<I, S>(args: I) -> Result<(Self, bool)>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut options = Self::default();
        let saw_lkh_exe = Self::apply_args(&mut options, args)?;
        options.work_dir = normalize_path(options.work_dir)?;
        options.lkh_exe = normalize_path(options.lkh_exe)?;
        Ok((options, saw_lkh_exe))
    }

    fn apply_args<I, S>(options: &mut Self, args: I) -> Result<bool>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut saw_lkh_exe = false;
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
                if name == "lkh-exe" {
                    saw_lkh_exe = true;
                }
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
                "cleanup" => {
                    options.cleanup = match value {
                        Some(v) => parse_bool(&name, &v)?,
                        None => true,
                    };
                }
                "no-cleanup" => {
                    if value.is_some() {
                        return Err(Error::invalid_input(format!(
                            "Flag --{name} does not take a value"
                        )));
                    }
                    options.cleanup = false;
                }
                "use-initial-tour" => {
                    options.use_initial_tour = match value {
                        Some(v) => parse_bool(&name, &v)?,
                        None => true,
                    };
                }
                "no-use-initial-tour" => {
                    if value.is_some() {
                        return Err(Error::invalid_input(format!(
                            "Flag --{name} does not take a value"
                        )));
                    }
                    options.use_initial_tour = false;
                }
                _ => {
                    return Err(Error::invalid_input(format!(
                        "Unknown option: --{name}\n\n{}",
                        Self::usage()
                    )));
                }
            }
        }
        Ok(saw_lkh_exe)
    }

    // fn config_args_from_path(path: &Path) -> Result<Vec<String>> {
    //     let contents = std::fs::read_to_string(path).map_err(Error::Io)?;
    //     Self::config_args_from_str(&contents)
    //         .map_err(|e| Error::invalid_input(format!("{e} (config: {})", path.display())))
    // }

    // fn config_args_from_str(contents: &str) -> std::result::Result<Vec<String>, String> {
    //     let value: toml::Value = toml::from_str(contents)
    //         .map_err(|e| format!("Invalid TOML in {CONFIG_FILE_NAME}: {e}"))?;
    //     let Some(table) = value.as_table() else {
    //         return Err(format!("Config file must be a TOML table at top-level"));
    //     };

    //     let mut args = Vec::with_capacity(table.len());
    //     for (raw_key, value) in table {
    //         let key = raw_key.replace('_', "-");
    //         let arg = match value {
    //             toml::Value::String(v) => format!("--{key}={v}"),
    //             toml::Value::Integer(v) => format!("--{key}={v}"),
    //             toml::Value::Float(v) => format!("--{key}={v}"),
    //             toml::Value::Boolean(v) => format!("--{key}={v}"),
    //             _ => {
    //                 return Err(format!(
    //                     "Unsupported value type for key `{raw_key}` in config file"
    //                 ));
    //             }
    //         };
    //         args.push(arg);
    //     }

    //     Ok(args)
    // }

    pub fn usage() -> &'static str {
        concat!(
            "Usage:\n",
            "  tsp-mt [options] [--input points.txt]\n",
            "  tsp-mt [options] < points.txt\n\n",
            "Options:\n",
            "  --lkh-exe <path>\n",
            "  --work-dir <path>\n",
            "  --projection-radius <f64>\n",
            "  --max-chunk-size <usize>\n",
            "  --centroid-order-seed <u64>\n",
            "  --centroid-order-max-trials <usize>\n",
            "  --centroid-order-time-limit <usize>\n",
            "  --solver-mode <single|multi-seed|multi-parallel>\n",
            "  --boundary-2opt-window <usize>\n",
            "  --boundary-2opt-passes <usize>\n",
            "  --spike-repair-top-n <usize>\n",
            "  --spike-repair-window <usize>\n",
            "  --spike-repair-passes <usize>\n",
            "  --outlier-threshold <f64>\n",
            "  --log-level <error|warn|info|debug|trace|off>\n",
            "  --log-format <compact|pretty>\n",
            "  --log-timestamp[=<bool>]\n",
            "  --no-log-timestamp\n",
            "  --cleanup[=<bool>]\n",
            "  --no-cleanup\n",
            "  --use-initial-tour[=<bool>]\n",
            "  --no-use-initial-tour\n",
            "  --log-output <path>\n",
            "  --input <path>\n",
            "  --output <path>\n",
            "  --help\n",
            "\n",
            "Examples:\n",
            "  tsp-mt --max-chunk-size 2000 --log-level warn --output output.txt < points.txt\n",
            "  tsp-mt --input points.txt --output output.txt\n",
            "  tsp-mt --log-level=info --log-output run.log < points.txt\n",
            "  tsp-mt --log-level=debug --log-format=pretty --log-timestamp < points.txt\n",
            "  tsp-mt --solver-mode=multi-seed --log-level=info < points.txt\n",
            "  tsp-mt --projection-radius=100 --log-level=info < points.txt\n",
        )
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

    pub fn lkh_path(&self) -> &Path {
        &self.lkh_exe
    }

    pub fn work_dir_path(&self) -> &Path {
        &self.work_dir
    }
}

fn default_work_dir() -> PathBuf {
    env::temp_dir().join(format!("tsp-mt-{}", process::id()))
}

// fn find_config_file() -> Result<Option<PathBuf>> {
//     if let Some(path) = config_from_env()? {
//         return Ok(Some(path));
//     }
//     if let Some(path) = config_from_current_dir_tree()? {
//         return Ok(Some(path));
//     }

//     #[cfg(unix)]
//     {
//         Ok(find_config_system_wide_unix())
//     }
//     #[cfg(not(unix))]
//     {
//         Ok(None)
//     }
// }

// fn config_from_env() -> Result<Option<PathBuf>> {
//     let Ok(raw) = env::var(CONFIG_PATH_ENV) else {
//         return Ok(None);
//     };
//     let trimmed = raw.trim();
//     if trimmed.is_empty() {
//         return Ok(None);
//     }
//     Ok(Some(normalize_path(PathBuf::from(trimmed))?))
// }

// fn config_from_current_dir_tree() -> Result<Option<PathBuf>> {
//     let mut dir = env::current_dir().map_err(Error::Io)?;
//     loop {
//         let candidate = dir.join(CONFIG_FILE_NAME);
//         if candidate.is_file() {
//             return Ok(Some(normalize_path(candidate)?));
//         }
//         if !dir.pop() {
//             break;
//         }
//     }
//     Ok(None)
// }

// #[cfg(unix)]
// fn find_config_system_wide_unix() -> Option<PathBuf> {
//     for entry in WalkDir::new("/")
//         .follow_links(false)
//         .into_iter()
//         .filter_map(|entry| entry.ok())
//     {
//         if !entry.file_type().is_file() {
//             continue;
//         }
//         if entry.file_name() == CONFIG_FILE_NAME {
//             if let Ok(path) = normalize_path(entry.path().to_path_buf()) {
//                 return Some(path);
//             }
//         }
//     }
//     None
// }

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
    use std::{
        path::PathBuf,
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use log::LevelFilter;

    use super::{LogFormat, LogLevel, SolverMode, SolverOptions, parse_bool};

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
        let (options, _) = SolverOptions::parse_from_iter([
            "--projection-radius=120.5",
            "--max-chunk-size=42",
            "--centroid-order-seed=77",
            "--centroid-order-max-trials=200",
            "--centroid-order-time-limit=9",
            "--solver-mode=single",
            "--boundary-2opt-window=8",
            "--boundary-2opt-passes=7",
            "--spike-repair-top-n=20",
            "--spike-repair-window=240",
            "--spike-repair-passes=2",
            "--outlier-threshold=12.5",
            "--log-level=debug",
            "--log-format=pretty",
            "--log-timestamp=false",
            "--cleanup=false",
            "--use-initial-tour=true",
            "--log-output=run.log",
            "--input=points.txt",
            "--output=route.txt",
        ])
        .expect("parse options");

        assert_eq!(options.projection_radius, 120.5);
        assert_eq!(options.max_chunk_size, 42);
        assert_eq!(options.centroid_order_seed, 77);
        assert_eq!(options.centroid_order_max_trials, 200);
        assert_eq!(options.centroid_order_time_limit, 9);
        assert_eq!(options.solver_mode, SolverMode::Single);
        assert_eq!(options.boundary_2opt_window, 8);
        assert_eq!(options.boundary_2opt_passes, 7);
        assert_eq!(options.spike_repair_top_n, 20);
        assert_eq!(options.spike_repair_window, 240);
        assert_eq!(options.spike_repair_passes, 2);
        assert_eq!(options.outlier_threshold, 12.5);
        assert_eq!(options.log_level, LogLevel::Debug);
        assert_eq!(options.log_format, LogFormat::Pretty);
        assert!(!options.log_timestamp);
        assert!(!options.cleanup);
        assert!(options.use_initial_tour);
        assert_eq!(options.log_output, "run.log");
        assert_eq!(options.input, "points.txt");
        assert_eq!(options.output, "route.txt");
    }

    // #[test]
    // fn config_args_from_str_supports_basic_types_and_snake_case_keys() {
    //     let args = SolverOptions::config_args_from_str(
    //         r#"
    //         max_chunk_size = 42
    //         projection_radius = 123.5
    //         cleanup = false
    //         log_level = "info"
    //         "#,
    //     )
    //     .expect("config parse");

    //     assert!(args.contains(&"--max-chunk-size=42".to_string()));
    //     assert!(args.contains(&"--projection-radius=123.5".to_string()));
    //     assert!(args.contains(&"--cleanup=false".to_string()));
    //     assert!(args.contains(&"--log-level=info".to_string()));
    // }

    // #[test]
    // fn config_args_from_str_rejects_non_scalar_values() {
    //     let err = SolverOptions::config_args_from_str(
    //         r#"
    //         input = ["a", "b"]
    //         "#,
    //     )
    //     .expect_err("array values should be rejected");
    //     assert!(err.contains("Unsupported value type"));
    // }

    // #[test]
    // fn cli_values_override_config_values() {
    //     let config_args = SolverOptions::config_args_from_str(
    //         r#"
    //         max_chunk_size = 12
    //         log_level = "error"
    //         "#,
    //     )
    //     .expect("config parse");

    //     let mut options = SolverOptions::default();
    //     let _ = SolverOptions::apply_args(&mut options, config_args).expect("config apply");
    //     let _ =
    //         SolverOptions::apply_args(&mut options, ["--max-chunk-size=99", "--log-level=debug"])
    //             .expect("cli apply");

    //     assert_eq!(options.max_chunk_size, 99);
    //     assert_eq!(options.log_level, LogLevel::Debug);
    // }
    #[test]
    fn parse_from_iter_accepts_no_log_timestamp_flag() {
        let (options, _) =
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
    fn parse_from_iter_accepts_no_cleanup_flag() {
        let (options, _) = SolverOptions::parse_from_iter(["--no-cleanup"]).expect("parse options");
        assert!(!options.cleanup);
    }

    #[test]
    fn parse_from_iter_rejects_no_cleanup_with_value() {
        let err = SolverOptions::parse_from_iter(["--no-cleanup=true"])
            .expect_err("expected flag value rejection");
        assert!(err.to_string().contains("does not take a value"));
    }

    #[test]
    fn parse_from_iter_accepts_use_initial_tour_flag() {
        let (options, _) =
            SolverOptions::parse_from_iter(["--use-initial-tour"]).expect("parse options");
        assert!(options.use_initial_tour);
    }

    #[test]
    fn parse_from_iter_accepts_no_use_initial_tour_flag() {
        let (options, _) =
            SolverOptions::parse_from_iter(["--use-initial-tour", "--no-use-initial-tour"])
                .expect("parse options");
        assert!(!options.use_initial_tour);
    }

    #[test]
    fn parse_from_iter_rejects_no_use_initial_tour_with_value() {
        let err = SolverOptions::parse_from_iter(["--no-use-initial-tour=true"])
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
    fn parse_from_iter_reads_lkh_exe_and_work_dir() {
        let (options, saw_lkh) =
            SolverOptions::parse_from_iter(["--lkh-exe=/bin/lkh", "--work-dir=/tmp/work"])
                .expect("options should be parsed");

        assert!(saw_lkh);
        assert_eq!(options.lkh_path(), std::path::Path::new("/bin/lkh"));
        assert_eq!(options.work_dir_path(), std::path::Path::new("/tmp/work"));
    }

    #[test]
    fn parse_from_iter_normalizes_relative_work_dir_to_absolute_path() {
        let relative = std::path::PathBuf::from("tmp").join("..").join("work");
        let expected = std::path::absolute(&relative).expect("absolute path");

        let (options, _) =
            SolverOptions::parse_from_iter([format!("--work-dir={}", relative.display())])
                .expect("options should be parsed");

        assert_eq!(options.work_dir_path(), expected);
    }

    #[test]
    fn parse_from_iter_requires_value_for_lkh_exe() {
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
    fn solver_mode_defaults_to_multi_parallel() {
        let options = SolverOptions::default();
        assert_eq!(options.solver_mode, SolverMode::MultiParallel);
    }

    #[test]
    fn cleanup_defaults_to_true() {
        let options = SolverOptions::default();
        assert!(options.cleanup);
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

    #[test]
    fn output_path_without_extension_has_no_side_effects() {
        let unique = format!(
            "tsp-mt-options-{}-{}",
            process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        );
        let nested = std::env::temp_dir().join(unique).join("logs").join("run");
        let output = nested.to_string_lossy().into_owned();
        let options = SolverOptions {
            output,
            ..SolverOptions::default()
        };

        let normalized = options.output_path().expect("path should exist");
        let expected = std::path::absolute(PathBuf::from(&nested)).expect("absolute path");
        assert_eq!(normalized, expected);
        assert!(
            !normalized.exists(),
            "normalization should not create a file"
        );
        let parent = normalized.parent().expect("parent path");
        assert!(
            !parent.exists(),
            "normalization should not create parent directories"
        );
    }
}
