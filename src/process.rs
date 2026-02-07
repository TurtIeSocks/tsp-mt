use std::{io, process::Output};

pub(crate) struct LkhProcess;

impl LkhProcess {
    pub(crate) fn ensure_success(context: &str, out: &Output) -> io::Result<()> {
        if out.status.success() {
            return Ok(());
        }

        Err(io::Error::other(format!(
            "{context}.\nSTDOUT:\n{}\nSTDERR:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        )))
    }
}
