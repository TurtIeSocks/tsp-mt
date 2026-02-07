use std::{env, fs, io, path::PathBuf};

include!(concat!(env!("OUT_DIR"), "/embedded_lkh.rs"));

pub(crate) fn ensure_lkh_executable() -> io::Result<PathBuf> {
    let mut path = env::temp_dir();
    path.push(format!("tsp-mt-lkh-{}", LKH_VERSION));

    let needs_write = match fs::metadata(&path) {
        Ok(meta) => meta.len() != LKH_EXECUTABLE_BYTES.len() as u64,
        Err(_) => true,
    };

    if needs_write {
        fs::write(&path, LKH_EXECUTABLE_BYTES)?;
        set_executable_permissions(&path)?;
    }

    Ok(path)
}

#[cfg(unix)]
fn set_executable_permissions(path: &std::path::Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
fn set_executable_permissions(_path: &std::path::Path) -> io::Result<()> {
    Ok(())
}
