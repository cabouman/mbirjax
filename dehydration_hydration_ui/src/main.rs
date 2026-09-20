//! Dehydration/Hydration Correction — GUI for the VENUS
//! dehydration_hydration workflow: load a stack of TIFF images, denoise it
//! with the NMF dehydrate/rehydrate algorithm of the local mbirjax library
//! (mbirjax.hsnt), inspect the result, and export the corrected stack.
//! `--run` does the same without a GUI, for scripting and pipelines.

use dehydration_hydration::app::DehydrationApp;
use dehydration_hydration::correction::{run_correction, CorrectionParams};
use dehydration_hydration::export::{export_corrected, Provenance};
use dehydration_hydration::hsnt::DatasetType;
use dehydration_hydration::loader;
use dehydration_hydration::nmf::BetaLoss;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

const USAGE: &str = "\
dehydration_hydration — NMF dehydration/hydration denoising of a TIFF stack

USAGE:
  dehydration_hydration [OPTIONS] [INPUT ...]

ARGS:
  INPUT   TIFF file(s) or a folder of TIFF images (subfolders are searched
          when the folder itself has none). When omitted, the data can be
          opened from within the application.

OPTIONS:
  --run                    Headless mode: run the correction and export the
                           corrected stack without opening a window
                           (requires INPUT and --output)
  -o, --output <DIR>       Folder receiving the corrected subfolder
                           '<input>_dehydration_hydration_corrected'
  --materials <N>          Number of materials (default 2)
  --dataset-type <TYPE>    attenuation | transmission (default attenuation)
  --beta-loss <LOSS>       frobenius | kullback-leibler (default frobenius)
  --max-iter <N>           NMF iteration cap (default 300)
  --bin <B>                Spatial binning factor (default 1 = full
                           resolution; >1 exports a binned preview)
  -h, --help               Show this help

The correction reproduces the dehydration_hydration notebook:
mbirjax.hsnt.hyper_denoise — M. S. N. Chowdhury et al., \"Fast Hyperspectral
Neutron Tomography\", IEEE Trans. Comput. Imaging 11, 663-677 (2025).
";

struct Cli {
    inputs: Vec<PathBuf>,
    run: bool,
    output: Option<PathBuf>,
    params: CorrectionParams,
    bin: usize,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli {
        inputs: Vec::new(),
        run: false,
        output: None,
        params: CorrectionParams::default(),
        bin: 1,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--run" => cli.run = true,
            "-o" | "--output" => cli.output = Some(PathBuf::from(value("--output")?)),
            "--materials" => {
                cli.params.num_materials = value("--materials")?
                    .parse()
                    .map_err(|_| "--materials must be a positive integer".to_owned())?;
            }
            "--dataset-type" | "--dataset_type" => {
                cli.params.dataset_type = match value("--dataset-type")?.as_str() {
                    "attenuation" => DatasetType::Attenuation,
                    "transmission" => DatasetType::Transmission,
                    other => return Err(format!("unknown dataset type '{other}'")),
                };
            }
            "--beta-loss" | "--beta_loss" => {
                cli.params.beta_loss = match value("--beta-loss")?.as_str() {
                    "frobenius" => BetaLoss::Frobenius,
                    "kullback-leibler" => BetaLoss::KullbackLeibler,
                    other => return Err(format!("unknown beta loss '{other}'")),
                };
            }
            "--max-iter" | "--max_iter" => {
                cli.params.max_iter = value("--max-iter")?
                    .parse()
                    .map_err(|_| "--max-iter must be a positive integer".to_owned())?;
            }
            "--bin" => {
                cli.bin = value("--bin")?
                    .parse()
                    .map_err(|_| "--bin must be a positive integer".to_owned())?;
                if cli.bin == 0 {
                    return Err("--bin must be at least 1".to_owned());
                }
            }
            s if s.starts_with('-') => return Err(format!("unknown option: {s}")),
            _ => cli.inputs.push(PathBuf::from(a)),
        }
    }
    if cli.params.num_materials == 0 {
        return Err("--materials must be at least 1".to_owned());
    }
    if cli.run {
        if cli.inputs.is_empty() {
            return Err("--run requires an INPUT folder or files".to_owned());
        }
        if cli.output.is_none() {
            return Err("--run requires --output <DIR>".to_owned());
        }
    }
    Ok(cli)
}

/// Expand folders to the image files they contain, so errors (missing
/// folder, no images) surface on stderr early.
fn expand_inputs(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for input in inputs {
        if input.is_dir() {
            match loader::list_supported_in_dir(input) {
                Ok(found) => files.extend(found),
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    std::process::exit(1);
                }
            }
        } else {
            files.push(input.clone());
        }
    }
    files
}

/// Headless: load, correct, export, print the export folder on stdout.
fn run_headless(cli: &Cli, files: Vec<PathBuf>) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    eprintln!("Loading {} file(s)…", files.len());
    let last_decile = AtomicUsize::new(0);
    let stack = loader::load_paths_with_progress(&files, |done, total| {
        let decile = done * 10 / total.max(1);
        if decile > last_decile.swap(decile, Ordering::Relaxed) {
            eprintln!("  loaded {done}/{total}");
        }
    })?;
    eprintln!(
        "Loaded {} image(s), {}×{} px ({} NaN/Inf pixel(s) zeroed).",
        stack.n_frames(),
        stack.width,
        stack.height,
        stack.nonfinite_fixed
    );

    let cancel = AtomicBool::new(false);
    let mut last_stage = String::new();
    let mut progress = |stage: &str, fraction: f32| {
        if stage != last_stage {
            eprintln!("[{:>3.0}%] {stage}", fraction * 100.0);
            last_stage = stage.to_owned();
        }
    };
    let out = run_correction(&stack, cli.params, cli.bin, &cancel, &mut progress)?;
    eprintln!(
        "Correction done in {:.1} s (subspace dimension {}).",
        out.elapsed_seconds, out.subspace_dimension
    );

    let input_dir = files
        .first()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let input_dir_name = input_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "images".to_owned());
    let (h, w) = out.integrated_mean.dim();
    let provenance = Provenance {
        input_folder: input_dir,
        num_images: out.frames.len(),
        image_width: w,
        image_height: h,
        params: cli.params,
        subspace_dimension: out.subspace_dimension,
        bin: out.bin,
        elapsed_seconds: out.elapsed_seconds,
        mbirjax_version: out.mbirjax_version.clone(),
    };
    let folder = export_corrected(
        cli.output.as_deref().expect("checked in parse_args"),
        &input_dir_name,
        &out.frames,
        &stack.sources,
        stack.transposed_on_load,
        &provenance,
        &mut |_, _| {},
    )?;
    eprintln!("Exported {} corrected image(s).", out.frames.len());
    println!("{}", folder.display());
    Ok(())
}

fn main() -> eframe::Result<()> {
    let cli = match parse_args() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let files = expand_inputs(&cli.inputs);

    if cli.run {
        if let Err(e) = run_headless(&cli, files) {
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("VENUS Dehydration / Hydration Correction"),
        ..Default::default()
    };

    eframe::run_native(
        "VENUS Dehydration / Hydration Correction",
        native_options,
        Box::new(move |cc| {
            // Saved light/dark preference, shared by all the VENUS rust
            // tools (dark when none is saved); the toolbar has a toggle.
            cc.egui_ctx.set_theme(dehydration_hydration::theme::load());
            let mut app = DehydrationApp::new();
            if !files.is_empty() {
                app.start_load(files, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}
