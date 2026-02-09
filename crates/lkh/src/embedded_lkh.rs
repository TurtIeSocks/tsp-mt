use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

include!(concat!(env!("OUT_DIR"), "/embedded_lkh.rs"));

use crate::{LkhError, LkhResult};

#[cfg(target_os = "windows")]
const EXECUTABLE_SUFFIX: &str = ".exe";
#[cfg(not(target_os = "windows"))]
const EXECUTABLE_SUFFIX: &str = "";

static LKH_EXECUTABLE_PATH: OnceLock<PathBuf> = OnceLock::new();

pub fn embedded_path() -> LkhResult<PathBuf> {
    if let Some(path) = LKH_EXECUTABLE_PATH.get() {
        return Ok(path.clone());
    }

    let path = embedded_executable_path();
    ensure_embedded_executable(&path)?;

    let _ = LKH_EXECUTABLE_PATH.set(path.clone());
    Ok(path)
}

fn embedded_executable_path() -> PathBuf {
    let mut path = env::temp_dir();
    path.push(format!("lkh-embedded-{}{}", LKH_VERSION, EXECUTABLE_SUFFIX));
    path
}

fn ensure_embedded_executable(path: &Path) -> LkhResult<()> {
    if file_matches_embedded(path)? {
        return Ok(());
    }

    let temp_path = write_embedded_temp_file(path)?;
    finalize_embedded_temp_file(path, &temp_path)?;

    if !file_matches_embedded(path)? {
        return Err(LkhError::other(format!(
            "embedded executable write verification failed at {}",
            path.display()
        )));
    }

    Ok(())
}

fn file_matches_embedded(path: &Path) -> LkhResult<bool> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err.into()),
    };
    Ok(bytes == LKH_EXECUTABLE_BYTES)
}

fn write_embedded_temp_file(target_path: &Path) -> LkhResult<PathBuf> {
    let parent = target_path.parent().ok_or_else(|| {
        LkhError::other(format!(
            "unable to resolve parent directory for {}",
            target_path.display()
        ))
    })?;
    fs::create_dir_all(parent)?;

    let file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("lkh-embedded");

    for attempt in 0..16 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let candidate = parent.join(format!(
            "{file_name}.{}.{}.{}.tmp",
            std::process::id(),
            nanos,
            attempt
        ));

        let mut file = match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err.into()),
        };

        file.write_all(LKH_EXECUTABLE_BYTES)?;
        file.sync_all()?;
        set_executable_permissions(&candidate)?;
        return Ok(candidate);
    }

    Err(LkhError::other(
        "failed to allocate temporary file for embedded LKH",
    ))
}

fn finalize_embedded_temp_file(target_path: &Path, temp_path: &Path) -> LkhResult<()> {
    match fs::rename(temp_path, target_path) {
        Ok(()) => Ok(()),
        Err(_) => {
            if file_matches_embedded(target_path)? {
                let _ = fs::remove_file(temp_path);
                return Ok(());
            }

            let _ = fs::remove_file(target_path);
            match fs::rename(temp_path, target_path) {
                Ok(()) => Ok(()),
                Err(err) => {
                    let _ = fs::remove_file(temp_path);
                    Err(err.into())
                }
            }
        }
    }
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
