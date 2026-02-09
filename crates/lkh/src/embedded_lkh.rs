use std::{env, fs, path::PathBuf, sync::OnceLock};

include!(concat!(env!("OUT_DIR"), "/embedded_lkh.rs"));

use crate::LkhResult;

#[cfg(target_os = "windows")]
const EXECUTABLE_SUFFIX: &str = ".exe";
#[cfg(not(target_os = "windows"))]
const EXECUTABLE_SUFFIX: &str = "";

static LKH_EXECUTABLE_PATH: OnceLock<PathBuf> = OnceLock::new();

pub fn embedded_path() -> LkhResult<PathBuf> {
    if let Some(path) = LKH_EXECUTABLE_PATH.get() {
        return Ok(path.clone());
    }

    let mut path = env::temp_dir();
    path.push(format!("lkh-embedded-{}{}", LKH_VERSION, EXECUTABLE_SUFFIX));

    let needs_write = match fs::metadata(&path) {
        Ok(meta) => meta.len() != LKH_EXECUTABLE_BYTES.len() as u64,
        Err(_) => true,
    };

    if needs_write {
        fs::write(&path, LKH_EXECUTABLE_BYTES)?;
        set_executable_permissions(&path)?;
    }

    let _ = LKH_EXECUTABLE_PATH.set(path.clone());
    Ok(path)
}

#[cfg(unix)]
fn set_executable_permissions(path: &std::path::Path) -> LkhResult<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable_permissions(_path: &std::path::Path) -> LkhResult<()> {
    Ok(())
}
