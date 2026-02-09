use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[cfg(feature = "embedded-lkh")]
use crate::embedded_lkh;
use crate::{LkhError, LkhResult};

pub struct LkhProcess {
    exe_path: PathBuf,
}

impl Default for LkhProcess {
    fn default() -> Self {
        #[cfg(not(feature = "embedded-lkh"))]
        {
            panic!("LkhProcess::default requires the 'embedded-lkh' feature")
        }
        let exe_path = match embedded_lkh::embedded_path() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("Error getting the embedded path: {err}");
                panic!("Error getting the default embedded path, check stderr")
            }
        };

        Self { exe_path }
    }
}

impl LkhProcess {
    pub fn run(&self, par_path: &Path, context: impl ToString) -> LkhResult<Output> {
        let output = Command::new(&self.exe_path)
            .arg(par_path)
            // .current_dir(self.work_dir())
            .output()
            .map_err(LkhError::from)?;

        if output.status.success() {
            Ok(output)
        } else {
            Err(LkhError::ProcessFailed {
                context: context.to_string(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            })
        }
    }
}
