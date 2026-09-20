//! Run the denoising through the **local mbirjax Python library** instead of
//! the native Rust port: the matrix goes out as a `.npy` file, a small bridge
//! script (`python/run_hyper_denoise.py`) calls `mbirjax.hsnt` from this
//! repository, and the result comes back as a `.npy` file.  Progress is
//! streamed over stdout lines (`##PROGRESS <fraction> <stage>`); setting
//! `cancel` kills the interpreter and the call returns an error containing
//! "cancelled", like the native solver.
//!
//! The interpreter is `$MBIRJAX_PYTHON` when set, else `python3`, else
//! `python`; it must be able to import mbirjax's dependencies (numpy, scipy,
//! scikit-learn, h5py, matplotlib) — e.g. the repository's pixi environment.

use crate::hsnt::{DatasetType, HsntParams};
use anyhow::{anyhow, bail, Context, Result};
use ndarray::Array2;
use ndarray_npy::{read_npy, write_npy};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;

pub struct PyDenoise {
    /// Denoised matrix, points × bands, same layout as the input.
    pub matrix: Array2<f32>,
    pub subspace_dimension: usize,
    /// Version reported by the imported mbirjax package.
    pub mbirjax_version: String,
}

pub struct PyEstimate {
    pub num_materials: usize,
    pub subspace_dimension: usize,
    pub mbirjax_version: String,
}

/// Denoise via `mbirjax.hsnt.hyper_denoise` on the local library.
pub fn hyper_denoise_py(
    x: &Array2<f64>,
    params: &HsntParams,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<PyDenoise> {
    let work = WorkDir::new()?;
    let input = work.path.join("input.npy");
    let output = work.path.join("output.npy");
    progress("Writing data for mbirjax", 0.0);
    write_npy(&input, x).context("writing the input .npy for mbirjax")?;

    let args = vec![
        "--mode".into(),
        "denoise".into(),
        "--input".into(),
        input.display().to_string(),
        "--output".into(),
        output.display().to_string(),
        "--dataset-type".into(),
        params.dataset_type.label().to_owned(),
        "--num-materials".into(),
        params.num_materials.to_string(),
        "--safety-factor".into(),
        params.safety_factor.to_string(),
        "--beta-loss".into(),
        params.beta_loss.label().to_owned(),
        "--max-iter".into(),
        params.max_iter.to_string(),
        "--tolerance".into(),
        params.tolerance.to_string(),
    ];
    let report = run_bridge(&args, cancel, progress)?;

    let matrix: Array2<f32> = read_npy(&output).context("reading the mbirjax result .npy")?;
    if matrix.dim() != x.dim() {
        bail!(
            "mbirjax returned a {:?} matrix for a {:?} input",
            matrix.dim(),
            x.dim()
        );
    }
    Ok(PyDenoise {
        matrix,
        subspace_dimension: report
            .subspace_dimension
            .unwrap_or_else(|| params.subspace_dimension()),
        mbirjax_version: report.version,
    })
}

/// Estimate the number of materials from sampled pixel spectra (raw values —
/// the script applies the same preprocessing as `mbirjax.hsnt.dehydrate`)
/// via `mbirjax.hsnt._estimate_subspace_dimension` on the local library.
pub fn estimate_num_materials_py(
    sample: &Array2<f64>,
    dataset_type: DatasetType,
    safety_factor: f64,
) -> Result<PyEstimate> {
    let work = WorkDir::new()?;
    let input = work.path.join("sample.npy");
    write_npy(&input, sample).context("writing the sample .npy for mbirjax")?;

    let args = vec![
        "--mode".into(),
        "estimate".into(),
        "--input".into(),
        input.display().to_string(),
        "--dataset-type".into(),
        dataset_type.label().to_owned(),
        "--safety-factor".into(),
        safety_factor.to_string(),
    ];
    let none = AtomicBool::new(false);
    let report = run_bridge(&args, &none, &mut |_, _| {})?;
    let (num_materials, subspace_dimension) = report
        .estimate
        .ok_or_else(|| anyhow!("the bridge script reported no estimate"))?;
    Ok(PyEstimate {
        num_materials,
        subspace_dimension,
        mbirjax_version: report.version,
    })
}

struct BridgeReport {
    version: String,
    subspace_dimension: Option<usize>,
    estimate: Option<(usize, usize)>,
}

/// Spawn the bridge script and pump its stdout protocol until it exits.
fn run_bridge(
    args: &[String],
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<BridgeReport> {
    let script = bridge_script()?;
    let python = python_interpreter();
    let mut child = Command::new(&python)
        .arg(&script)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "launching '{python}' (set MBIRJAX_PYTHON to a Python that can import mbirjax)"
            )
        })?;

    // Stdout lines stream over a channel so this thread can watch `cancel`
    // while the interpreter works.
    let stdout = child.stdout.take().expect("stdout was piped");
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let mut stderr = child.stderr.take().expect("stderr was piped");
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });

    let mut report = BridgeReport {
        version: "unknown".to_owned(),
        subspace_dimension: None,
        estimate: None,
    };
    let mut cancelled = false;
    loop {
        if cancel.load(Ordering::Relaxed) && !cancelled {
            let _ = child.kill();
            cancelled = true;
        }
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(line) => parse_line(&line, &mut report, progress),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = reader.join();
    let stderr_text = stderr_reader.join().unwrap_or_default();
    let status = child.wait().context("waiting for the mbirjax bridge")?;

    if cancelled || cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    if !status.success() {
        let tail: Vec<&str> = stderr_text.lines().rev().take(12).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        bail!(
            "the mbirjax bridge failed ({status}).\n{}",
            tail.join("\n")
        );
    }
    Ok(report)
}

fn parse_line(line: &str, report: &mut BridgeReport, progress: &mut dyn FnMut(&str, f32)) {
    let Some(rest) = line.strip_prefix("##") else {
        return; // library chatter — ignored
    };
    let mut parts = rest.splitn(2, ' ');
    let tag = parts.next().unwrap_or_default();
    let payload = parts.next().unwrap_or_default();
    match tag {
        "PROGRESS" => {
            let mut p = payload.splitn(2, ' ');
            let fraction: f32 = p.next().and_then(|f| f.parse().ok()).unwrap_or(0.0);
            let stage = p.next().unwrap_or("mbirjax");
            progress(stage, fraction.clamp(0.0, 1.0));
        }
        "VERSION" => {
            if let Some(v) = payload.split_whitespace().next() {
                report.version = v.to_owned();
            }
        }
        "SUBSPACE_DIM" => report.subspace_dimension = payload.trim().parse().ok(),
        "ESTIMATE" => {
            let mut p = payload.split_whitespace();
            if let (Some(m), Some(d)) = (
                p.next().and_then(|v| v.parse().ok()),
                p.next().and_then(|v| v.parse().ok()),
            ) {
                report.estimate = Some((m, d));
            }
        }
        _ => {}
    }
}

fn python_interpreter() -> String {
    if let Ok(p) = std::env::var("MBIRJAX_PYTHON") {
        if !p.trim().is_empty() {
            return p;
        }
    }
    for candidate in ["python3", "python"] {
        if Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return candidate.to_owned();
        }
    }
    "python3".to_owned()
}

/// The bridge script: `$MBIRJAX_HSNT_BRIDGE`, else `python/run_hyper_denoise.py`
/// next to this crate (compile-time path — the app lives inside the mbirjax
/// repository), else next to the executable.
fn bridge_script() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("MBIRJAX_HSNT_BRIDGE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        bail!("MBIRJAX_HSNT_BRIDGE points to '{}', which does not exist", p.display());
    }
    let in_repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("run_hyper_denoise.py");
    if in_repo.is_file() {
        return Ok(in_repo);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("run_hyper_denoise.py");
            if beside.is_file() {
                return Ok(beside);
            }
        }
    }
    bail!(
        "cannot find run_hyper_denoise.py (looked at {}; set MBIRJAX_HSNT_BRIDGE)",
        in_repo.display()
    )
}

/// A per-call scratch folder under the system temp dir, removed on drop.
struct WorkDir {
    path: PathBuf,
}

impl WorkDir {
    fn new() -> Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dehydration_hydration_{}_{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating scratch folder {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
