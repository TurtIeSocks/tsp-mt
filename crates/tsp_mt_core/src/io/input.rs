use std::{
    env,
    io::Read,
    path::{Path, PathBuf},
    process,
};
use tsp_mt_derive::{CliOptions, KvDisplay};

use crate::{Error, LKHNode, Result, embedded_lkh};

/// Runtime input for LKH solver.
#[derive(Clone, Debug, CliOptions, KvDisplay)]
pub struct SolverInput {
    #[cli(long = "lkh-exe")]
    #[kv(fmt = "path")]
    pub(crate) lkh_exe: PathBuf,
    #[cli(long = "work-dir")]
    #[kv(fmt = "path")]
    pub(crate) work_dir: PathBuf,
    #[kv(fmt = "len")]
    pub(crate) points: Vec<LKHNode>,
}

impl SolverInput {
    pub fn new(lkh_exe: &Path, work_dir: &Path, points: &[LKHNode]) -> Self {
        Self {
            lkh_exe: lkh_exe.to_path_buf(),
            work_dir: work_dir.to_path_buf(),
            points: points.to_vec(),
        }
    }

    pub fn from_args() -> Result<Self> {
        let (mut input, saw_lkh_exe) = Self::parse_cli_args(env::args().skip(1))?;

        if !saw_lkh_exe {
            input.lkh_exe = embedded_lkh::ensure_lkh_executable()?;
        }
        input.points = read_points_from_stdin()?;
        Ok(input)
    }

    fn parse_cli_args<I, S>(args: I) -> Result<(Self, bool)>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut input = Self {
            lkh_exe: PathBuf::new(),
            work_dir: default_work_dir(),
            points: Vec::new(),
        };
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
                continue;
            };

            let (name, value) = Self::split_arg(raw_name, &mut args);
            if input.apply_cli_option(&name, value)? && name == "lkh-exe" {
                saw_lkh_exe = true;
            }
        }

        Ok((input, saw_lkh_exe))
    }

    pub fn usage() -> &'static str {
        concat!(
            "Input options:\n",
            "  --lkh-exe <path>   Path to the LKH executable\n",
            "  --work-dir <path>  Working directory for temp files\n",
        )
    }

    pub fn points_len(&self) -> usize {
        self.points.len()
    }

    pub fn lkh_path(&self) -> &Path {
        &self.lkh_exe
    }

    pub fn work_dir_path(&self) -> &Path {
        &self.work_dir
    }

    pub(crate) fn n(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn get_point(&self, idx: usize) -> LKHNode {
        self.points[idx]
    }
}

fn default_work_dir() -> PathBuf {
    env::temp_dir().join(format!("tsp-mt-{}", process::id()))
}

fn read_points_from_stdin() -> Result<Vec<LKHNode>> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    parse_points(&input)
}

fn parse_points(input: &str) -> Result<Vec<LKHNode>> {
    let mut points = Vec::new();
    for (idx, tok) in input.split_whitespace().enumerate() {
        let mut it = tok.split(',');
        let lat_s = it
            .next()
            .ok_or_else(|| Error::invalid_input(format!("Token {}: missing latitude", idx + 1)))?;
        let lon_s = it
            .next()
            .ok_or_else(|| Error::invalid_input(format!("Token {}: missing longitude", idx + 1)))?;

        if it.next().is_some() {
            return Err(Error::invalid_input(format!(
                "Token {}: expected 'lat,lng' but got extra comma fields: {tok}",
                idx + 1
            )));
        }

        let lat: f64 = lat_s.parse().map_err(|_| {
            Error::invalid_input(format!("Token {}: invalid latitude: {}", idx + 1, lat_s))
        })?;
        let lon: f64 = lon_s.parse().map_err(|_| {
            Error::invalid_input(format!("Token {}: invalid longitude: {}", idx + 1, lon_s))
        })?;

        points.push(LKHNode::new(lat, lon));
    }

    if points.is_empty() {
        return Err(Error::invalid_input("No points provided on stdin."));
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::{SolverInput, parse_points};

    #[test]
    fn parse_cli_args_reads_lkh_exe_and_work_dir() {
        let (input, saw_lkh) = SolverInput::parse_cli_args([
            "--lkh-exe",
            "/bin/lkh",
            "--work-dir",
            "/tmp/work",
        ])
        .expect("parse args");

        assert!(saw_lkh);
        assert_eq!(
            input.lkh_path().to_str().expect("utf8 path"),
            "/bin/lkh"
        );
        assert_eq!(
            input.work_dir_path().to_str().expect("utf8 path"),
            "/tmp/work"
        );
    }

    #[test]
    fn parse_cli_args_help_returns_usage_error() {
        let err = SolverInput::parse_cli_args(["--help"]).expect_err("help should short-circuit");
        assert!(err.to_string().contains("Input options:"));
    }

    #[test]
    fn parse_cli_args_without_lkh_tracks_missing_lkh() {
        let (_input, saw_lkh) =
            SolverInput::parse_cli_args(["--work-dir", "/tmp/work"]).expect("parse args");
        assert!(!saw_lkh);
    }

    #[test]
    fn parse_points_parses_whitespace_separated_lat_lng_tokens() {
        let points = parse_points("1.0,2.0\n3.0,4.0 5.0,6.0").expect("parse points");
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].to_string(), "1.0,2.0");
        assert_eq!(points[2].to_string(), "5.0,6.0");
    }

    #[test]
    fn parse_points_rejects_empty_input() {
        let err = parse_points(" \n\t ").expect_err("empty input should fail");
        assert!(err.to_string().contains("No points provided on stdin."));
    }

    #[test]
    fn parse_points_rejects_extra_comma_fields() {
        let err = parse_points("1,2,3").expect_err("extra fields should fail");
        assert!(err.to_string().contains("expected 'lat,lng'"));
    }

    #[test]
    fn parse_points_rejects_non_numeric_coordinates() {
        let err = parse_points("a,2").expect_err("invalid latitude should fail");
        assert!(err.to_string().contains("invalid latitude"));
    }
}
