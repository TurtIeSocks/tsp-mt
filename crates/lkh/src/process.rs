//! Low-level process runner for invoking an LKH executable.
//!
//! This module is intentionally minimal: it only spawns a process with a
//! parameter file argument and surfaces stdout/stderr on failure.
//!
//! # Example
//!
//! ```no_run
//! use lkh::process::LkhProcess;
//!
//! fn main() -> lkh::LkhResult<()> {
//!     let process = LkhProcess::new("/usr/local/bin/LKH");
//!     let _output = process.run("work/problem.par")?;
//!     Ok(())
//! }
//! ```
//!
use std::{
    path::PathBuf,
    process::{Command, Output},
};

use lkh_derive::WithMethods;

#[cfg(feature = "embedded-lkh")]
use crate::embedded_lkh;
use crate::{LkhError, LkhResult, with_methods_error};

/// Configurable LKH process invocation.
#[derive(WithMethods)]
#[with(error = LkhError)]
pub struct LkhProcess {
    exe_path: PathBuf,
    context: Option<String>,
    current_dir: Option<PathBuf>,
}

with_methods_error!(LkhProcessWithMethodsError);

impl LkhProcess {
    /// Creates a process wrapper for a specific LKH executable path.
    pub fn new(exe_path: impl Into<PathBuf>) -> Self {
        Self {
            exe_path: exe_path.into(),
            context: None,
            current_dir: None,
        }
    }

    /// Executes `LKH <par_file_path>`.
    ///
    /// Returns raw process output on success. On non-zero exit status, returns
    /// `LkhError::ProcessFailed` with captured stdout/stderr.
    pub fn run(&self, par_file_path: impl Into<PathBuf>) -> LkhResult<Output> {
        let mut command = Command::new(&self.exe_path);
        if let Some(current_dir) = &self.current_dir {
            command.current_dir(current_dir);
        }

        let output = command
            .arg(par_file_path.into())
            .output()
            .map_err(LkhError::from)?;

        if output.status.success() {
            Ok(output)
        } else {
            Err(LkhError::ProcessFailed {
                context: self.context.clone().unwrap_or_default(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            })
        }
    }

    #[cfg(feature = "embedded-lkh")]
    /// Creates a process wrapper backed by the embedded LKH executable.
    pub fn try_default() -> LkhResult<Self> {
        Ok(Self {
            exe_path: embedded_lkh::embedded_path()?,
            context: None,
            current_dir: None,
        })
    }
}

#[cfg(feature = "embedded-lkh")]
impl Default for LkhProcess {
    fn default() -> Self {
        Self::try_default().expect("failed to initialize default embedded LKH process")
    }
}
