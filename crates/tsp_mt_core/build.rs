use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const LKH_VERSION: &str = "3.0.13";
const LKH_URL: &str = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz";
const LKH_WINDOWS_EXE: &str = "LKH.exe";
const LKH_WINDOWS_URL: &str = "http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.exe";

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("failed to resolve workspace root")?;
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    println!("cargo:rerun-if-changed=build.rs");

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
            run_cmd(
                Command::new("tar")
                    .arg("-xzf")
                    .arg(&archive_path)
                    .arg("-C")
                    .arg(&build_root),
                "extracting LKH archive",
            )?;
        }
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
    if let Ok(existing) = fs::read(path) {
        if existing == data {
            return Ok(());
        }
    }
    fs::write(path, data)
}

fn ensure_archive(workspace_root: &Path, archive_path: &Path) -> Result<(), Box<dyn Error>> {
    if archive_path.exists() {
        return Ok(());
    }

    let url = env::var("TSP_MT_LKH_URL").unwrap_or_else(|_| LKH_URL.to_string());
    if download_archive(&url, archive_path).is_ok() {
        return Ok(());
    }

    let vendored = workspace_root.join("lkh").join("lkh.tgz");
    if vendored.exists() {
        fs::copy(&vendored, archive_path)?;
        return Ok(());
    }

    Err(format!(
        "failed to download LKH archive from {url} and no fallback at {}",
        vendored.display()
    )
    .into())
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
