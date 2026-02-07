use std::{
    env,
    io::Read,
    path::{Path, PathBuf},
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
    pub fn new(lkh_exe: &PathBuf, work_dir: &PathBuf, points: &[LKHNode]) -> Self {
        Self {
            lkh_exe: lkh_exe.clone(),
            work_dir: work_dir.clone(),
            points: points.to_vec(),
        }
    }

    pub fn from_args() -> Result<Self> {
        let current = env::current_dir()?;
        let mut input = Self {
            lkh_exe: PathBuf::new(),
            work_dir: current.join(".temp"),
            points: Vec::new(),
        };
        let mut saw_lkh_exe = false;

        let mut args = env::args().skip(1).peekable();
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

        if !saw_lkh_exe {
            input.lkh_exe = embedded_lkh::ensure_lkh_executable()?;
        }
        input.points = read_points_from_stdin()?;
        Ok(input)
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

fn read_points_from_stdin() -> Result<Vec<LKHNode>> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

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
