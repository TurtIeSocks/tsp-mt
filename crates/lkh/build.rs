use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

const LKH_VERSION: &str = "3.0.13";
const LKH_URL: &str =
    "https://codeload.github.com/blaulan/LKH-3/tar.gz/3da421394c922921b841c0be8e2d176b60fdcfe6";
const LKH_ARCHIVE_SHA256: &str = "7f696cf6a38cf1bfdc2bb4b00dc386dcd3959a34b1afc3da8707030234c9560f";
const LKH_ARCHIVE_SHA256_ENV: &str = "TSP_MT_LKH_SHA256";
const LKH_ALLOW_INSECURE_HTTP_ENV: &str = "TSP_MT_ALLOW_INSECURE_HTTP_LKH";
const LKH_WINDOWS_EXE: &str = "LKH.exe";
const LKH_WINDOWS_URL: &str = "https://github.com/blaulan/LKH-3";
const LKH_WINDOWS_SHA256_ENV: &str = "TSP_MT_LKH_WINDOWS_EXE_SHA256";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_EMBEDDED_LKH");
    if env::var_os("CARGO_FEATURE_EMBEDDED_LKH").is_none() {
        return Ok(());
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("failed to resolve workspace root")?;
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={LKH_ARCHIVE_SHA256_ENV}");
    println!("cargo:rerun-if-env-changed={LKH_ALLOW_INSECURE_HTTP_ENV}");
    println!("cargo:rerun-if-env-changed={LKH_WINDOWS_SHA256_ENV}");

    if cfg!(target_os = "windows") {
        let root_exe = workspace_root.join(LKH_WINDOWS_EXE);
        if root_exe.exists() {
            println!("cargo:rerun-if-changed={}", root_exe.display());
        }
        return build_windows(workspace_root, &out_dir);
    }

    if !cfg!(target_family = "unix") {
        return Err("LKH build is only supported on unix-like targets and Windows".into());
    }

    println!("cargo:rerun-if-env-changed=TSP_MT_LKH_URL");
    let vendored_archive = workspace_root.join("lkh").join("lkh.tgz");
    if vendored_archive.exists() {
        println!("cargo:rerun-if-changed={}", vendored_archive.display());
    }

    build_unix(workspace_root, &out_dir)
}

fn build_windows(workspace_root: &Path, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let root_exe = workspace_root.join(LKH_WINDOWS_EXE);
    if !root_exe.exists() {
        return Err(format!(
            "windows build requires {LKH_WINDOWS_EXE} in repository root ({}). \
             download it from {LKH_WINDOWS_URL}",
            root_exe.display(),
        )
        .into());
    }

    if let Some(expected) = env_non_empty(LKH_WINDOWS_SHA256_ENV) {
        verify_sha256(&root_exe, &expected)?;
    }

    let bundled_bin = out_dir.join("lkh.bin");
    copy_if_different(&root_exe, &bundled_bin)?;
    write_embedded_source(out_dir, &bundled_bin)?;
    Ok(())
}

fn build_unix(workspace_root: &Path, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let build_root = out_dir.join("lkh-build");
    fs::create_dir_all(&build_root)?;

    let archive_path = build_root.join(format!("LKH-{LKH_VERSION}.tgz"));
    let src_dir = build_root.join(format!("LKH-{LKH_VERSION}"));
    let built_exe = src_dir.join("LKH");

    if !built_exe.exists() {
        ensure_archive(workspace_root, &archive_path)?;
        if !src_dir.exists() {
            fs::create_dir_all(&src_dir)?;
            run_cmd(
                Command::new("tar")
                    .arg("-xzf")
                    .arg(&archive_path)
                    .arg("--strip-components=1")
                    .arg("-C")
                    .arg(&src_dir),
                "extracting LKH archive",
            )?;
        }
        fs::create_dir_all(src_dir.join("SRC").join("OBJ"))?;
        run_cmd(
            Command::new("make").current_dir(&src_dir),
            "building LKH via make",
        )?;
    }

    if !built_exe.exists() {
        return Err(format!("LKH binary was not produced at {}", built_exe.display()).into());
    }

    let bundled_bin = out_dir.join("lkh.bin");
    copy_if_different(&built_exe, &bundled_bin)?;
    write_embedded_source(out_dir, &bundled_bin)?;

    Ok(())
}

fn write_embedded_source(out_dir: &Path, bundled_bin: &Path) -> Result<(), Box<dyn Error>> {
    let generated = out_dir.join("embedded_lkh.rs");
    let source = format!(
        "pub const LKH_VERSION: &str = \"{LKH_VERSION}\";\n\
         pub static LKH_EXECUTABLE_BYTES: &[u8] = include_bytes!(r#\"{}\"#);\n",
        bundled_bin.display()
    );
    write_if_different(&generated, source.as_bytes())?;
    Ok(())
}

fn copy_if_different(src: &Path, dst: &Path) -> io::Result<()> {
    let src_bytes = fs::read(src)?;
    write_if_different(dst, &src_bytes)
}

fn write_if_different(path: &Path, data: &[u8]) -> io::Result<()> {
    if let Ok(existing) = fs::read(path)
        && existing == data
    {
        return Ok(());
    }
    fs::write(path, data)
}

fn ensure_archive(workspace_root: &Path, archive_path: &Path) -> Result<(), Box<dyn Error>> {
    if archive_path.exists() {
        verify_archive(archive_path)?;
        return Ok(());
    }

    if let Some(cached) = find_cached_archive(workspace_root)? {
        verify_archive(&cached)?;
        fs::copy(cached, archive_path)?;
        verify_archive(archive_path)?;
        return Ok(());
    }

    let vendored = workspace_root.join("lkh").join("lkh.tgz");
    if vendored.exists() {
        verify_archive(&vendored)?;
        fs::copy(&vendored, archive_path)?;
        verify_archive(archive_path)?;
        return Ok(());
    }

    let mut download_errors = Vec::new();
    if let Some(url) = env_non_empty("TSP_MT_LKH_URL") {
        if let Err(err) = try_download_and_verify_archive(&url, archive_path) {
            download_errors.push(format!("{url}: {err}"));
        } else {
            return Ok(());
        }
    } else {
        for url in [LKH_URL] {
            if let Err(err) = try_download_and_verify_archive(url, archive_path) {
                download_errors.push(format!("{url}: {err}"));
                continue;
            }
            return Ok(());
        }
    }

    Err(format!(
        "failed to download and verify LKH archive. \
         Tried: {}. place a verified fallback at {} or set TSP_MT_LKH_URL/TSP_MT_LKH_SHA256",
        download_errors.join(" | "),
        vendored.display(),
    )
    .into())
}

fn find_cached_archive(workspace_root: &Path) -> io::Result<Option<PathBuf>> {
    let target_dir = workspace_root.join("target");
    if !target_dir.is_dir() {
        return Ok(None);
    }

    let profiles = ["debug", "release"];
    for profile in profiles {
        let build_dir = target_dir.join(profile).join("build");
        if !build_dir.is_dir() {
            continue;
        }
        for entry in fs::read_dir(build_dir)? {
            let candidate = entry?
                .path()
                .join("out")
                .join("lkh-build")
                .join(format!("LKH-{LKH_VERSION}.tgz"));
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
    }

    Ok(None)
}

fn download_archive(url: &str, archive_path: &Path) -> io::Result<()> {
    if run_cmd(
        Command::new("curl")
            .arg("-fsSL")
            .arg(url)
            .arg("-o")
            .arg(archive_path),
        "downloading LKH archive with curl",
    )
    .is_ok()
    {
        return Ok(());
    }

    run_cmd(
        Command::new("wget")
            .arg("-q")
            .arg("-O")
            .arg(archive_path)
            .arg(url),
        "downloading LKH archive with wget",
    )
}

fn try_download_and_verify_archive(url: &str, archive_path: &Path) -> Result<(), Box<dyn Error>> {
    validate_download_url(url)?;
    download_archive(url, archive_path)?;
    if let Err(err) = verify_archive(archive_path) {
        let _ = fs::remove_file(archive_path);
        return Err(err);
    }
    Ok(())
}

fn run_cmd(cmd: &mut Command, context: &str) -> io::Result<()> {
    let rendered = render_cmd(cmd);
    let out = cmd.output()?;
    if out.status.success() {
        return Ok(());
    }

    Err(io::Error::other(format!(
        "{context} failed ({rendered})\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )))
}

fn render_cmd(cmd: &Command) -> String {
    let program = cmd.get_program().to_string_lossy();
    let args = cmd
        .get_args()
        .map(OsStr::to_string_lossy)
        .collect::<Vec<_>>()
        .join(" ");
    if args.is_empty() {
        program.into_owned()
    } else {
        format!("{program} {args}")
    }
}

fn verify_archive(path: &Path) -> Result<(), Box<dyn Error>> {
    let expected = env::var(LKH_ARCHIVE_SHA256_ENV).unwrap_or_else(|_| LKH_ARCHIVE_SHA256.into());
    verify_sha256(path, &expected)
}

fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), Box<dyn Error>> {
    let expected = normalize_sha256_hex(expected_hex)?;
    let bytes = fs::read(path)?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!(
            "sha256 mismatch for {}: expected {}, got {}",
            path.display(),
            expected,
            actual
        )
        .into());
    }
    Ok(())
}

fn normalize_sha256_hex(raw: &str) -> Result<String, Box<dyn Error>> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.len() != 64 || !normalized.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("invalid sha256 value: {raw}").into());
    }
    Ok(normalized)
}

fn validate_download_url(url: &str) -> Result<(), Box<dyn Error>> {
    if url.starts_with("https://") {
        return Ok(());
    }

    if url.starts_with("http://") {
        if allow_insecure_http() {
            return Ok(());
        }
        return Err(format!(
            "refusing insecure URL '{url}'. \
             Use HTTPS or set {LKH_ALLOW_INSECURE_HTTP_ENV}=1 to allow HTTP."
        )
        .into());
    }

    Err(format!("unsupported URL scheme for TSP_MT_LKH_URL: {url}").into())
}

fn allow_insecure_http() -> bool {
    env_non_empty(LKH_ALLOW_INSECURE_HTTP_ENV)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn env_non_empty(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
