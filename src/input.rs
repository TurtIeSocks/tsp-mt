use std::{
    env, io::{self, Read},
    path::{Path, PathBuf},
};

use crate::LKHNode;

/// Runtime input for LKH solver.
#[derive(Clone, Debug)]
pub struct SolverInput {
    pub(crate) lkh_exe: PathBuf,
    pub(crate) work_dir: PathBuf,
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

    pub fn from_args() -> io::Result<Self> {
        let current = env::current_dir()?;
        let mut lkh_exe = current.join("lkh/LKH-3.0.13/LKH");
        let mut work_dir = current.join("temp");

        let mut args = env::args().skip(1).peekable();
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    Self::usage().to_string(),
                ));
            }
            let Some(raw_name) = arg.strip_prefix("--") else {
                continue;
            };

            let (name, value) = split_arg(raw_name, &mut args);

            match name.as_str() {
                "lkh-exe" => {
                    let raw = value.ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("Missing value for --{}", name),
                        )
                    })?;
                    lkh_exe = PathBuf::from(raw);
                }
                "work-dir" => {
                    let raw = value.ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("Missing value for --{}", name),
                        )
                    })?;
                    work_dir = PathBuf::from(raw);
                }
                _ => {}
            }
        }

        let points = read_points_from_stdin()?;
        Ok(Self {
            lkh_exe,
            work_dir,
            points,
        })
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

fn read_points_from_stdin() -> std::io::Result<Vec<LKHNode>> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let mut points = Vec::new();

    for (idx, tok) in input.split_whitespace().enumerate() {
        let mut it = tok.split(',');
        let lat_s = it.next().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Token {}: missing latitude", idx + 1),
            )
        })?;
        let lon_s = it.next().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Token {}: missing longitude", idx + 1),
            )
        })?;

        if it.next().is_some() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Token {}: expected 'lat,lng' but got extra comma fields: {tok}",
                    idx + 1
                ),
            ));
        }

        let lat: f64 = lat_s.parse().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Token {}: invalid latitude: {}", idx + 1, lat_s),
            )
        })?;
        let lon: f64 = lon_s.parse().map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Token {}: invalid longitude: {}", idx + 1, lon_s),
            )
        })?;

        points.push(LKHNode::new(lat, lon));
    }

    if points.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No points provided on stdin.",
        ));
    }

    Ok(points)
}
