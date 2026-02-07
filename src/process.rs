use std::process::Output;

use crate::{Error, Result};

pub(crate) struct LkhProcess;

impl LkhProcess {
    pub(crate) fn ensure_success(context: &str, out: &Output) -> Result<()> {
        if out.status.success() {
            return Ok(());
        }

        Err(Error::ProcessFailed {
            context: context.to_string(),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        })
    }
}
