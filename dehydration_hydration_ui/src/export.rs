//! Exporting the corrected stack as 32-bit float TIFF files (one per input
//! image, keeping the input file names) together with a provenance file
//! (`correction_config.json`) recording exactly how they were produced.
//! Also: CSV export of the profile plots.
//!
//! [`export_corrected`] is the synchronous, GUI-free core (also used by the
//! headless `--run` mode); [`start_export`] wraps it on a background thread
//! for the egui app.

use crate::correction::CorrectionParams;
use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

/// Everything `correction_config.json` records about a corrected stack.
pub struct Provenance {
    pub input_folder: PathBuf,
    pub num_images: usize,
    pub image_width: usize,
    pub image_height: usize,
    pub params: CorrectionParams,
    pub subspace_dimension: usize,
    /// Spatial binning factor of the run (1 = full resolution).
    pub bin: usize,
    pub elapsed_seconds: f64,
    /// Version of the mbirjax library that ran the correction.
    pub mbirjax_version: String,
}

/// `<output>/<input-folder-name>_dehydration_hydration_corrected`, suffixed
/// `_1`, `_2`, … when it already exists (the notebook's
/// `make_or_increment_folder_name`). The folder is created.
pub fn make_export_folder(output_dir: &Path, input_dir_name: &str) -> Result<PathBuf> {
    let base = output_dir.join(format!("{input_dir_name}_dehydration_hydration_corrected"));
    let mut candidate = base.clone();
    let mut i = 0;
    while candidate.exists() {
        i += 1;
        candidate = PathBuf::from(format!("{}_{i}", base.display()));
    }
    std::fs::create_dir_all(&candidate)
        .with_context(|| format!("create {}", candidate.display()))?;
    Ok(candidate)
}

/// Write one frame as a grayscale 32-bit float TIFF. `undo_display_transpose`
/// restores the on-disk orientation for frames the loader transposed (TIFF
/// input — same convention as rust_roi_selector).
pub fn write_f32_tiff(path: &Path, frame: &Array2<f32>, undo_display_transpose: bool) -> Result<()> {
    use tiff::encoder::{colortype::Gray32Float, TiffEncoder};

    let data = if undo_display_transpose {
        frame.t().as_standard_layout().into_owned()
    } else {
        frame.as_standard_layout().into_owned()
    };
    let (h, w) = (data.nrows(), data.ncols());
    let file =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = TiffEncoder::new(std::io::BufWriter::new(file))
        .with_context(|| format!("init TIFF encoder for {}", path.display()))?;
    enc.write_image::<Gray32Float>(w as u32, h as u32, data.as_slice().unwrap())
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Output file name: the input file's stem with a `.tif` extension.
pub fn output_name(input: &Path) -> String {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_owned());
    format!("{stem}.tif")
}

/// Write the corrected stack + provenance file into a fresh export folder
/// under `output_dir`, reporting `(files_done, files_total)`. Returns the
/// created folder.
pub fn export_corrected(
    output_dir: &Path,
    input_dir_name: &str,
    frames: &[Array2<f32>],
    sources: &[PathBuf],
    undo_display_transpose: bool,
    provenance: &Provenance,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<PathBuf> {
    let folder = make_export_folder(output_dir, input_dir_name)?;
    let total = frames.len();
    for (i, frame) in frames.iter().enumerate() {
        let name = sources
            .get(i)
            .map(|p| output_name(p))
            .unwrap_or_else(|| format!("image_{i:05}.tif"));
        write_f32_tiff(&folder.join(name), frame, undo_display_transpose)?;
        progress(i + 1, total);
    }
    let json = provenance_json(provenance);
    std::fs::write(folder.join("correction_config.json"), json)
        .with_context(|| format!("write provenance in {}", folder.display()))?;
    Ok(folder)
}

/// The provenance file contents. Hand-written JSON: the structure is small
/// and flat, not worth a serde dependency.
pub fn provenance_json(p: &Provenance) -> String {
    format!(
        r#"{{
  "tool": "dehydration_hydration_ui",
  "tool_version": "{version}",
  "algorithm": "mbirjax.hsnt.hyper_denoise (local mbirjax library)",
  "mbirjax_version": "{mbirjax}",
  "created_utc": "{time}",
  "input_folder": "{input}",
  "num_images": {n},
  "image_width": {w},
  "image_height": {h},
  "parameters": {{
    "dataset_type": "{dtype}",
    "num_materials": {materials},
    "beta_loss": "{beta}",
    "max_iter": {max_iter},
    "safety_factor": 2,
    "subspace_dimension": {subdim}
  }},
  "spatial_binning": {bin},
  "elapsed_seconds": {elapsed:.1}
}}
"#,
        version = env!("CARGO_PKG_VERSION"),
        mbirjax = json_escape(&p.mbirjax_version),
        time = iso8601_utc_now(),
        input = json_escape(&p.input_folder.display().to_string()),
        n = p.num_images,
        w = p.image_width,
        h = p.image_height,
        dtype = p.params.dataset_type.label(),
        materials = p.params.num_materials,
        beta = p.params.beta_loss.label(),
        max_iter = p.params.max_iter,
        subdim = p.subspace_dimension,
        bin = p.bin,
        elapsed = p.elapsed_seconds,
    )
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ`, from the system clock only
/// (no chrono dependency; civil-from-days per Howard Hinnant).
pub fn iso8601_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

/// Gregorian calendar date from days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Write the profile plot's data as CSV. `tof_us`/`lambda_a` are optional
/// extra axis columns (must match the profile length when present).
pub fn write_profiles_csv(
    path: &Path,
    uncorrected: &[f64],
    corrected: &[f64],
    tof_us: Option<&[f64]>,
    lambda_a: Option<&[f64]>,
) -> Result<()> {
    use std::io::Write;

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    write!(out, "image_index")?;
    if tof_us.is_some() {
        write!(out, ",tof_us")?;
    }
    if lambda_a.is_some() {
        write!(out, ",lambda_angstroms")?;
    }
    writeln!(out, ",uncorrected,corrected")?;
    for i in 0..uncorrected.len().min(corrected.len()) {
        write!(out, "{i}")?;
        if let Some(t) = tof_us {
            write!(out, ",{}", t.get(i).copied().unwrap_or(f64::NAN))?;
        }
        if let Some(l) = lambda_a {
            write!(out, ",{}", l.get(i).copied().unwrap_or(f64::NAN))?;
        }
        writeln!(out, ",{},{}", uncorrected[i], corrected[i])?;
    }
    Ok(())
}

pub enum ExportMsg {
    Progress { done: usize, total: usize },
    Done(Result<PathBuf, String>),
}

/// Run [`export_corrected`] on a background thread for the GUI.
#[allow(clippy::too_many_arguments)]
pub fn start_export(
    output_dir: PathBuf,
    input_dir_name: String,
    frames: std::sync::Arc<Vec<Array2<f32>>>,
    sources: Vec<PathBuf>,
    undo_display_transpose: bool,
    provenance: Provenance,
    ctx: egui::Context,
) -> Receiver<ExportMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let progress_tx = tx.clone();
        let progress_ctx = ctx.clone();
        let mut progress = move |done: usize, total: usize| {
            let _ = progress_tx.send(ExportMsg::Progress { done, total });
            progress_ctx.request_repaint();
        };
        let result = export_corrected(
            &output_dir,
            &input_dir_name,
            &frames,
            &sources,
            undo_display_transpose,
            &provenance,
            &mut progress,
        );
        let _ = tx.send(ExportMsg::Done(result.map_err(|e| format!("{e:#}"))));
        ctx.request_repaint();
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dehydration_export_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn provenance() -> Provenance {
        Provenance {
            input_folder: PathBuf::from("/data/Run_1"),
            num_images: 2,
            image_width: 5,
            image_height: 3,
            params: CorrectionParams::default(),
            subspace_dimension: 4,
            bin: 1,
            elapsed_seconds: 1.5,
            mbirjax_version: "0.7.2".to_owned(),
        }
    }

    #[test]
    fn export_folder_increments_on_collision() {
        let dir = tmp_dir("incr");
        let a = make_export_folder(&dir, "Run_1234").unwrap();
        let b = make_export_folder(&dir, "Run_1234").unwrap();
        let c = make_export_folder(&dir, "Run_1234").unwrap();
        assert!(a.ends_with("Run_1234_dehydration_hydration_corrected"));
        assert!(b.ends_with("Run_1234_dehydration_hydration_corrected_1"));
        assert!(c.ends_with("Run_1234_dehydration_hydration_corrected_2"));
    }

    #[test]
    fn f32_tiff_roundtrips_through_the_loader() {
        let dir = tmp_dir("tiff");
        let path = dir.join("img.tif");
        let frame = Array2::from_shape_fn((3, 5), |(y, x)| (y * 5 + x) as f32 * 0.5);
        // Written with the transpose undone, the loader (which transposes
        // TIFFs on read) must round-trip to the in-memory orientation.
        write_f32_tiff(&path, &frame, true).unwrap();
        let stack = crate::loader::load_paths(&[path]).unwrap();
        assert_eq!((stack.height, stack.width), (3, 5));
        assert_eq!(stack.frames[0], frame);
    }

    #[test]
    fn output_name_forces_tif_extension() {
        assert_eq!(output_name(Path::new("/a/b/img_0001.tiff")), "img_0001.tif");
        assert_eq!(output_name(Path::new("/a/b/img_0001.tif")), "img_0001.tif");
    }

    #[test]
    fn export_writes_images_and_provenance() {
        let dir = tmp_dir("full");
        let frames = vec![
            Array2::from_elem((3, 5), 1.0f32),
            Array2::from_elem((3, 5), 2.0f32),
        ];
        let sources = vec![PathBuf::from("a_0000.tiff"), PathBuf::from("a_0001.tiff")];
        let mut ticks = Vec::new();
        let folder = export_corrected(
            &dir,
            "Run_1",
            &frames,
            &sources,
            true,
            &provenance(),
            &mut |d, t| ticks.push((d, t)),
        )
        .unwrap();
        assert!(folder.join("a_0000.tif").is_file());
        assert!(folder.join("a_0001.tif").is_file());
        assert_eq!(ticks, vec![(1, 2), (2, 2)]);
        let json = std::fs::read_to_string(folder.join("correction_config.json")).unwrap();
        assert!(json.contains("\"num_materials\": 2"));
        assert!(json.contains("\"mbirjax_version\": \"0.7.2\""));
        assert!(json.contains("/data/Run_1"));
    }

    #[test]
    fn timestamp_looks_like_iso8601() {
        let t = iso8601_utc_now();
        assert_eq!(t.len(), 20, "{t}");
        assert!(t.ends_with('Z'));
        assert_eq!(&t[4..5], "-");
        assert_eq!(&t[10..11], "T");
        // Sanity: the year is in a plausible range.
        let year: i32 = t[0..4].parse().unwrap();
        assert!((2026..2100).contains(&year), "{t}");
    }

    #[test]
    fn civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // leap-year checks
        assert_eq!(civil_from_days(19_723 + 59), (2024, 2, 29));
    }

    #[test]
    fn profiles_csv_includes_optional_axes() {
        let dir = tmp_dir("csv");
        let path = dir.join("profiles.csv");
        write_profiles_csv(
            &path,
            &[1.0, 2.0],
            &[0.9, 1.9],
            Some(&[6.08, 11.2]),
            Some(&[0.001, 0.002]),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("image_index,tof_us,lambda_angstroms,uncorrected,corrected\n"));
        assert!(text.contains("0,6.08,0.001,1,0.9"));

        // Without axes: only three columns.
        let path2 = dir.join("plain.csv");
        write_profiles_csv(&path2, &[1.0], &[2.0], None, None).unwrap();
        assert!(std::fs::read_to_string(&path2)
            .unwrap()
            .starts_with("image_index,uncorrected,corrected\n"));
    }
}
