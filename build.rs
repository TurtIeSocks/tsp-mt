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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=lkh/lkh.tgz");
    println!("cargo:rerun-if-changed={LKH_WINDOWS_EXE}");
    println!("cargo:rerun-if-env-changed=TSP_MT_LKH_URL");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    if cfg!(target_os = "windows") {
        return build_windows(&manifest_dir, &out_dir);
    }
    if !cfg!(target_family = "unix") {
        return Err("LKH build is only supported on unix-like targets and Windows".into());
    }

    build_unix(&manifest_dir, &out_dir)
}

fn build_windows(manifest_dir: &Path, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let root_exe = manifest_dir.join(LKH_WINDOWS_EXE);
    if !root_exe.exists() {
        return Err(format!(
            "windows build requires {LKH_WINDOWS_EXE} in repository root ({}). \
             download it from {LKH_WINDOWS_URL}",
            root_exe.display(),
        )
        .into());
    }

    let bundled_bin = out_dir.join("lkh.bin");
    fs::copy(&root_exe, &bundled_bin)?;
    write_embedded_source(out_dir, &bundled_bin)?;
    Ok(())
}

fn build_unix(manifest_dir: &Path, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let build_root = out_dir.join("lkh-build");
    fs::create_dir_all(&build_root)?;

    let archive_path = build_root.join(format!("LKH-{LKH_VERSION}.tgz"));
    let src_dir = build_root.join(format!("LKH-{LKH_VERSION}"));
    let built_exe = src_dir.join("LKH");

    if !built_exe.exists() {
        ensure_archive(&manifest_dir, &archive_path)?;
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
    fs::copy(&built_exe, &bundled_bin)?;
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
    fs::write(generated, source)?;
    Ok(())
}

fn ensure_archive(manifest_dir: &Path, archive_path: &Path) -> Result<(), Box<dyn Error>> {
    if archive_path.exists() {
        return Ok(());
    }

    let url = env::var("TSP_MT_LKH_URL").unwrap_or_else(|_| LKH_URL.to_string());
    if download_archive(&url, archive_path).is_ok() {
        return Ok(());
    }

    let vendored = manifest_dir.join("lkh").join("lkh.tgz");
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
