use std::{
    path::PathBuf,
    process::{Command, Output},
};

use lkh_derive::WithMethods;

#[cfg(feature = "embedded-lkh")]
use crate::embedded_lkh;
use crate::{LkhError, LkhResult, with_methods_error};

#[derive(WithMethods)]
#[with(error = LkhError)]
pub struct LkhProcess {
    exe_path: PathBuf,
    context: Option<String>,
}

with_methods_error!(LkhProcessWithMethodsError);

impl LkhProcess {
    pub fn new(exe_path: impl Into<PathBuf>) -> Self {
        Self {
            exe_path: exe_path.into(),
            context: None,
        }
    }

    pub fn run(&self, par_file_path: impl Into<PathBuf>) -> LkhResult<Output> {
        let output = Command::new(&self.exe_path)
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
}

#[cfg(feature = "embedded-lkh")]
impl Default for LkhProcess {
    fn default() -> Self {
        let exe_path = match embedded_lkh::embedded_path() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("Error getting the embedded path: {err}");
                panic!("Error getting the default embedded path, check stderr")
            }
        };

        Self {
            exe_path,
            context: None,
        }
    }
}
