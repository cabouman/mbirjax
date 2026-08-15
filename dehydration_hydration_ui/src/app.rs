//! The egui/eframe application: load a TIFF stack, tune the correction
//! parameters, run the NMF dehydration/hydration denoising on a background
//! thread, compare corrected vs raw images, inspect region profiles, and
//! export the corrected stack — the same workflow as the
//! dehydration_hydration notebook, in one native window.

use crate::colormap::Colormap;
use crate::correction::{start_correction, CorrectionMsg, CorrectionParams};
use crate::export::{start_export, ExportMsg, Provenance};
use crate::hsnt::DatasetType;
use crate::py_bridge::{self, PyEstimate};
use crate::loader::{self, ImageStack};
use crate::nmf::BetaLoss;
use crate::spectra;

use egui::{Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use egui_plot::{Corner, CoordinatesFormatter, Legend, MarkerShape, Plot, PlotPoints, Points};
use ndarray::{s, Array2};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

/// Width of the colorbar column: gradient strip + ticks + value labels.
const COLORBAR_WIDTH: f32 = 78.0;

/// Spatial binning factor of the fast preview run.
const PREVIEW_BIN: usize = 2;

/// Number of randomly sampled pixel spectra used by the material estimation.
const ESTIMATE_SAMPLE: usize = 384;

/// ORNL Neutron Imaging team logo (same asset as the other rust
/// applications) and the MBIRJAX logo (the Purdue library the correction is
/// a port of), embedded in the binary and shown at the bottom-left of the
/// window. The MBIRJAX logo has a light- and a dark-background variant;
/// the one matching the active theme is displayed.
const IMAGING_LOGO_BYTES: &[u8] = include_bytes!("../logos/ImagingLogo.png");
const MBIRJAX_LOGO_LIGHT_BYTES: &[u8] = include_bytes!("../logos/mbirjax_logo.png");
const MBIRJAX_LOGO_DARK_BYTES: &[u8] = include_bytes!("../logos/mbirjax_logo_dark_background.png");
const LOGO_HEIGHT: f32 = 44.0;

fn load_logo(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(name, color_image, TextureOptions::LINEAR))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Raw,
    Result,
    Profiles,
}

/// Messages sent from the background loading thread to the UI.
enum LoadMsg {
    Progress { done: usize, total: usize },
    Done(anyhow::Result<ImageStack>),
}

struct LoadJob {
    rx: Receiver<LoadMsg>,
    done: usize,
    total: usize,
}

struct CorrJob {
    rx: Receiver<CorrectionMsg>,
    stage: String,
    fraction: f32,
    cancel: Arc<AtomicBool>,
}

struct ExportJob {
    rx: Receiver<ExportMsg>,
    done: usize,
    total: usize,
}

/// A finished correction, kept alongside the parameters that produced it.
struct ResultState {
    frames: Arc<Vec<Array2<f32>>>,
    integrated_mean: Array2<f32>,
    /// The (binned) raw frames the run compared against — present only for
    /// preview runs (bin > 1), where the input stack is not like-for-like.
    binned_raw: Option<Vec<Array2<f32>>>,
    /// Spatial binning factor (1 = full resolution, >1 = preview).
    bin: usize,
    subspace_dimension: usize,
    elapsed_seconds: f64,
    params: CorrectionParams,
    /// Version of the mbirjax library that ran the correction.
    mbirjax_version: String,
}

impl ResultState {
    fn is_preview(&self) -> bool {
        self.bin > 1
    }

    /// Corrected-frame dimensions (h, w) — the stack's when bin = 1.
    fn dims(&self) -> (usize, usize) {
        self.integrated_mean.dim()
    }

    /// The raw frame comparable to corrected frame `i` (binned for previews).
    fn raw_frame<'a>(&'a self, stack: &'a ImageStack, i: usize) -> Option<&'a Array2<f32>> {
        match &self.binned_raw {
            Some(binned) => binned.get(i),
            None => stack.frames.get(i),
        }
    }
}

/// What the right pane of the "Corrected vs raw" view shows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RightPane {
    Raw,
    /// corrected − raw, on a symmetric color range around 0.
    Difference,
}

/// X-axis of the profile plots.
#[derive(Clone, Copy, PartialEq, Eq)]
enum XAxis {
    Index,
    TofUs,
    LambdaAngstrom,
}

impl XAxis {
    fn label(self) -> &'static str {
        match self {
            XAxis::Index => "Image index",
            XAxis::TofUs => "TOF (µs)",
            XAxis::LambdaAngstrom => "Wavelength (Å)",
        }
    }
}

/// Pixel region for the profile plots (half-open: `left..right`, `top..bottom`).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Region {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

impl Region {
    fn clamped(mut self, w: usize, h: usize) -> Self {
        self.left = self.left.min(w.saturating_sub(1));
        self.right = self.right.clamp(self.left + 1, w);
        self.top = self.top.min(h.saturating_sub(1));
        self.bottom = self.bottom.clamp(self.top + 1, h);
        self
    }

    fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.left as f32 && x <= self.right as f32 && y >= self.top as f32 && y <= self.bottom as f32
    }

    /// The 8 resize handles as `((hx, vy), (image x, image y))`:
    /// hx/vy ∈ {-1, 0, 1} select the edge(s) the handle drags.
    fn handles(&self) -> [((i8, i8), (f32, f32)); 8] {
        let (x0, y0) = (self.left as f32, self.top as f32);
        let (x1, y1) = (self.right as f32, self.bottom as f32);
        let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        [
            ((-1, -1), (x0, y0)),
            ((0, -1), (mx, y0)),
            ((1, -1), (x1, y0)),
            ((-1, 0), (x0, my)),
            ((1, 0), (x1, my)),
            ((-1, 1), (x0, y1)),
            ((0, 1), (mx, y1)),
            ((1, 1), (x1, y1)),
        ]
    }
}

/// A drag in progress on the profile region. `rect` is the working rectangle
/// in image coordinates (`[x0, y0, x1, y1]`, unordered while a resize crosses
/// itself); it is committed to the integer [`Region`] on every drag frame.
struct RegionDrag {
    mode: DragMode,
    rect: [f32; 4],
}

enum DragMode {
    /// Drawing a new region; `rect[0..2]` holds the anchor corner.
    Draw,
    /// Moving the whole region; `last` is the previous pointer position.
    Move { last: (f32, f32) },
    /// Dragging the handle that controls these edges.
    Resize { hx: i8, vy: i8 },
}

fn cursor_for_handle(hx: i8, vy: i8) -> egui::CursorIcon {
    match (hx, vy) {
        (0, _) => egui::CursorIcon::ResizeVertical,
        (_, 0) => egui::CursorIcon::ResizeHorizontal,
        _ if hx == vy => egui::CursorIcon::ResizeNwSe,
        _ => egui::CursorIcon::ResizeNeSw,
    }
}

pub struct DehydrationApp {
    stack: Option<Arc<ImageStack>>,
    loading: Option<LoadJob>,
    /// Folder the images came from: names the export folder and seeds the
    /// export dialog location.
    input_dir: Option<PathBuf>,
    /// Recently loaded dataset folders, most recent first (persisted).
    recent: Vec<PathBuf>,

    view: View,
    frame_idx: usize,
    colormap: Colormap,
    /// Track the displayed image's full range automatically until the user
    /// touches the contrast values; "Auto" re-engages it.
    contrast_auto: bool,
    vmin: f32,
    vmax: f32,
    data_min: f32,
    data_max: f32,

    /// Per-pixel sum over the raw frames (the notebook's integrated image).
    integrated_raw: Option<Array2<f32>>,
    result: Option<ResultState>,

    params: CorrectionParams,
    corr_job: Option<CorrJob>,
    export_job: Option<ExportJob>,
    last_export: Option<PathBuf>,
    /// Background material-count estimation ("Auto" button).
    estimate_job: Option<Receiver<Result<PyEstimate, String>>>,
    /// mbirjax version reported by the last bridge run, for the About dialog.
    mbirjax_version: Option<String>,

    /// Right pane of the "Corrected vs raw" view.
    right_pane: RightPane,

    tex_left: Option<TextureHandle>,
    tex_right: Option<TextureHandle>,
    cbar_tex: Option<TextureHandle>,
    tex_dirty: bool,
    /// The images currently on screen (owned copies, parallel to the
    /// textures), for the cursor value read-out.
    pane_cache: (Option<Array2<f32>>, Option<Array2<f32>>),

    region: Option<Region>,
    /// Live drag on the profile region: drawing a new one, moving it, or
    /// resizing it by one of its handles.
    region_drag: Option<RegionDrag>,
    /// (uncorrected, corrected) mean intensity per frame over the region.
    profiles: Option<(Vec<f64>, Vec<f64>)>,
    /// (uncorrected, corrected) intensity per frame of the marked pixel.
    pixel_profiles: Option<(Vec<f64>, Vec<f64>)>,
    /// Pixel marked by clicking the profiles image (result coordinates).
    pixel_marker: Option<(usize, usize)>,
    profiles_dirty: bool,
    /// Plot the profiles on a log10 y-axis (non-positive values are hidden).
    log_y: bool,
    /// TOF axis (µs, one value per image) from the folder's *_Spectra.txt.
    spectra_tof_us: Option<Vec<f64>>,
    x_axis: XAxis,
    /// Source–detector distance for the wavelength conversion (m).
    distance_m: f64,

    scale: f32,
    fit_requested: bool,
    cursor: Option<(usize, usize, f32)>,
    status: String,
    /// The "ℹ mbirjax" About dialog (algorithm provenance and versions).
    show_about: bool,
    /// (imaging, mbirjax-light-bg, mbirjax-dark-bg) logo textures, loaded on
    /// the first frame.
    logo_tex: Option<[Option<TextureHandle>; 3]>,
}

impl Default for DehydrationApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DehydrationApp {
    pub fn new() -> Self {
        Self {
            stack: None,
            loading: None,
            input_dir: None,
            recent: crate::recent::load(),
            view: View::Raw,
            frame_idx: 0,
            colormap: Colormap::Viridis,
            contrast_auto: true,
            vmin: 0.0,
            vmax: 1.0,
            data_min: 0.0,
            data_max: 1.0,
            integrated_raw: None,
            result: None,
            params: CorrectionParams::default(),
            corr_job: None,
            export_job: None,
            last_export: None,
            estimate_job: None,
            mbirjax_version: None,
            right_pane: RightPane::Raw,
            tex_left: None,
            tex_right: None,
            cbar_tex: None,
            tex_dirty: false,
            pane_cache: (None, None),
            region: None,
            region_drag: None,
            profiles: None,
            pixel_profiles: None,
            pixel_marker: None,
            profiles_dirty: false,
            log_y: false,
            spectra_tof_us: None,
            x_axis: XAxis::Index,
            distance_m: spectra::DEFAULT_DISTANCE_M,
            scale: 1.0,
            fit_requested: false,
            cursor: None,
            status: "Open a folder of TIFF images to begin.".to_owned(),
            show_about: false,
            logo_tex: None,
        }
    }

    /// The two logos, side by side. The MBIRJAX variant matching the active
    /// theme is shown: the official transparent PNG (dark text) on light,
    /// the white-on-black variant on dark.
    fn logos_row(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let [imaging, mbirjax_light, mbirjax_dark] = self.logo_tex.get_or_insert_with(|| {
            [
                load_logo(&ctx, "imaging_logo", IMAGING_LOGO_BYTES),
                load_logo(&ctx, "mbirjax_logo_light", MBIRJAX_LOGO_LIGHT_BYTES),
                load_logo(&ctx, "mbirjax_logo_dark", MBIRJAX_LOGO_DARK_BYTES),
            ]
        });
        let mbirjax = match ctx.theme() {
            egui::Theme::Dark => mbirjax_dark,
            egui::Theme::Light => mbirjax_light,
        };
        ui.horizontal(|ui| {
            if let Some(tex) = imaging {
                ui.add(egui::Image::from_texture(&*tex).max_height(LOGO_HEIGHT))
                    .on_hover_text("Neutron Imaging — Oak Ridge National Laboratory");
            }
            if let Some(tex) = mbirjax {
                ui.add(egui::Image::from_texture(&*tex).max_height(LOGO_HEIGHT))
                    .on_hover_text("MBIRJAX — Purdue University");
            }
        });
    }

    // ----- about dialog ------------------------------------------------------

    /// Modal showing what algorithm this tool runs and which mbirjax version
    /// the local library reported.
    fn about_modal(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        let modal = egui::Modal::new(egui::Id::new("about_modal")).show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.heading("Dehydration / Hydration Correction");
            ui.label(format!("Application version {}", env!("CARGO_PKG_VERSION")));
            ui.separator();
            ui.add_space(4.0);
            ui.label("The correction runs the dehydrate/rehydrate denoising of the \
                      mbirjax library checked out in this repository (module \
                      mbirjax.hsnt, function hyper_denoise), through a Python bridge.");
            ui.add_space(6.0);
            let version_line = match &self.mbirjax_version {
                Some(v) => format!("mbirjax version: {v}"),
                None => "mbirjax version: run a correction or estimation to query it".to_owned(),
            };
            ui.label(egui::RichText::new(version_line).strong());
            ui.add_space(6.0);
            ui.label("Algorithm reference:");
            ui.label(
                egui::RichText::new(
                    "M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, \
                     H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, \"Fast Hyperspectral \
                     Neutron Tomography\", IEEE Transactions on Computational Imaging, \
                     vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854",
                )
                .small(),
            );
            ui.hyperlink_to(
                "mbirjax hsnt documentation",
                "https://mbirjax.readthedocs.io/en/latest/usr_hsnt.html",
            );
            ui.add_space(10.0);
            ui.vertical_centered(|ui| {
                if ui.button("  Close  ").clicked() {
                    self.show_about = false;
                }
            });
        });
        if modal.should_close() {
            self.show_about = false;
        }
    }

    // ----- loading -----------------------------------------------------------

    pub fn start_load(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        if paths.is_empty() {
            return;
        }
        if self.corr_job.is_some() || self.export_job.is_some() {
            self.status = "Wait for the running job to finish (or cancel it) first.".to_owned();
            return;
        }
        let total = paths.len();
        let (tx, rx) = std::sync::mpsc::channel();
        // Loading is parallel, so the progress callback fires from worker
        // threads; the channel sender goes behind a mutex (Sender is !Sync).
        let progress_tx = std::sync::Mutex::new(tx.clone());
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = loader::load_paths_with_progress(&paths, |done, total| {
                if let Ok(sender) = progress_tx.lock() {
                    let _ = sender.send(LoadMsg::Progress { done, total });
                }
                ctx.request_repaint();
            });
            let _ = tx.send(LoadMsg::Done(result));
            ctx.request_repaint();
        });
        self.loading = Some(LoadJob { rx, done: 0, total });
        self.status = format!("Loading {total} file(s)…");
    }

    fn poll_load(&mut self) {
        let mut result = None;
        if let Some(job) = &mut self.loading {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    LoadMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                    }
                    LoadMsg::Done(res) => result = Some(res),
                }
            }
        }
        if let Some(res) = result {
            self.loading = None;
            match res {
                Ok(stack) => self.apply_stack(stack),
                Err(e) => self.status = format!("Load failed: {e:#}"),
            }
        }
    }

    fn apply_stack(&mut self, stack: ImageStack) {
        let n = stack.n_frames();
        let (w, h) = (stack.width, stack.height);
        self.input_dir = stack
            .sources
            .first()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        if let Some(dir) = &self.input_dir {
            self.recent = crate::recent::add(dir);
        }

        let mut acc = Array2::<f32>::zeros((h, w));
        for f in &stack.frames {
            acc += f;
        }
        self.integrated_raw = Some(acc);

        // TOF axis from the folder's *_Spectra.txt, when it matches the
        // stack (one TOF value per image).
        self.spectra_tof_us = self
            .input_dir
            .as_ref()
            .and_then(|dir| spectra::find_spectra_file(dir))
            .and_then(|path| spectra::load_tof_us(&path).ok())
            .filter(|tof| tof.len() == n);
        if self.spectra_tof_us.is_none() {
            self.x_axis = XAxis::Index;
        }

        self.stack = Some(Arc::new(stack));
        self.result = None;
        self.profiles = None;
        self.pixel_profiles = None;
        self.pixel_marker = None;
        self.region = None;
        self.last_export = None;
        self.view = View::Raw;
        self.frame_idx = 0;
        self.contrast_auto = true;
        self.fit_requested = true;
        self.tex_dirty = true;
        let spectra_note = if self.spectra_tof_us.is_some() {
            " TOF axis found (Spectra.txt)."
        } else {
            ""
        };
        self.status = format!("Loaded {n} image(s), {w}×{h} px.{spectra_note}");
    }

    // ----- material estimation ("Auto") --------------------------------------

    /// Estimate the number of materials from a random sample of pixel
    /// spectra, on a background thread — the estimation itself runs in the
    /// local mbirjax library (`_estimate_subspace_dimension`).
    fn start_estimation(&mut self, ctx: &egui::Context) {
        let Some(stack) = self.stack.clone() else {
            return;
        };
        let dataset_type = self.params.dataset_type;
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let (h, w, n) = (stack.height, stack.width, stack.frames.len());
            let points = h * w;
            let sample_size = ESTIMATE_SAMPLE.min(points);
            let mut rng = crate::linalg::Rng::new(0xE571_AA7E);
            // Raw values — the bridge applies dehydrate()'s preprocessing so
            // the estimation sees the data exactly as the denoising would.
            let mut sample = Array2::<f64>::zeros((sample_size, n));
            for row in 0..sample_size {
                let p = (rng.next_u64() % points as u64) as usize;
                let (y, x) = (p / w, p % w);
                for (i, frame) in stack.frames.iter().enumerate() {
                    sample[(row, i)] = f64::from(frame[(y, x)]);
                }
            }
            let estimate = py_bridge::estimate_num_materials_py(&sample, dataset_type, 2.0)
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(estimate);
            ctx.request_repaint();
        });
        self.estimate_job = Some(rx);
        self.status = "Estimating the number of materials (local mbirjax)…".to_owned();
    }

    fn poll_estimation(&mut self) {
        let Some(rx) = &self.estimate_job else { return };
        if let Ok(result) = rx.try_recv() {
            self.estimate_job = None;
            match result {
                Ok(est) => {
                    self.mbirjax_version = Some(est.mbirjax_version.clone());
                    self.params.num_materials = est.num_materials.clamp(1, 10);
                    self.status = format!(
                        "Estimated {} material(s) from {ESTIMATE_SAMPLE} sampled pixel spectra \
                         (subspace dimension {}, mbirjax {}).",
                        est.num_materials, est.subspace_dimension, est.mbirjax_version
                    );
                }
                Err(e) => self.status = format!("Estimation failed: {e}"),
            }
        }
    }

    // ----- correction job ----------------------------------------------------

    fn start_correction_job(&mut self, ctx: &egui::Context, bin: usize) {
        let Some(stack) = &self.stack else { return };
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = start_correction(stack.clone(), self.params, bin, cancel.clone(), ctx.clone());
        self.corr_job = Some(CorrJob {
            rx,
            stage: "Starting…".to_owned(),
            fraction: 0.0,
            cancel,
        });
        let mode = if bin > 1 {
            format!("preview, {bin}×{bin} binned, ")
        } else {
            String::new()
        };
        self.status = format!(
            "Correction running ({mode}{}, {} materials, {}, {} iterations max)…",
            self.params.dataset_type.label(),
            self.params.num_materials,
            self.params.beta_loss.label(),
            self.params.max_iter,
        );
    }

    fn poll_correction(&mut self) {
        let mut done = None;
        if let Some(job) = &mut self.corr_job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    CorrectionMsg::Progress { stage, fraction } => {
                        job.stage = stage;
                        job.fraction = fraction;
                    }
                    CorrectionMsg::Done(res) => done = Some(res),
                }
            }
        }
        let Some(res) = done else { return };
        self.corr_job = None;
        match res {
            Ok(out) => {
                let (h, w) = out.integrated_mean.dim();
                let preview_note = if out.is_preview() {
                    format!(" — PREVIEW at {0}×{0} binning", out.bin)
                } else {
                    String::new()
                };
                self.status = format!(
                    "Correction done in {:.1} s (subspace dimension {}, mbirjax {}){preview_note}.",
                    out.elapsed_seconds, out.subspace_dimension, out.mbirjax_version
                );
                self.mbirjax_version = Some(out.mbirjax_version.clone());
                self.result = Some(ResultState {
                    frames: Arc::new(out.frames),
                    integrated_mean: out.integrated_mean,
                    binned_raw: out.binned_raw,
                    bin: out.bin,
                    subspace_dimension: out.subspace_dimension,
                    elapsed_seconds: out.elapsed_seconds,
                    params: self.params,
                    mbirjax_version: out.mbirjax_version,
                });
                self.pixel_marker = None;
                self.pixel_profiles = None;
                // The notebook's default profile region: the second quarter
                // of the image in both directions.
                self.region = Some(
                    Region {
                        left: w / 4,
                        right: 2 * w / 4,
                        top: h / 4,
                        bottom: 2 * h / 4,
                    }
                    .clamped(w, h),
                );
                self.profiles_dirty = true;
                self.view = View::Result;
                self.contrast_auto = true;
                self.tex_dirty = true;
            }
            Err(e) if e.contains("cancelled") => {
                self.status = "Correction cancelled.".to_owned();
            }
            Err(e) => self.status = format!("Correction failed: {e}"),
        }
    }

    // ----- export job --------------------------------------------------------

    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(result) = &self.result else { return };
        let Some(stack) = &self.stack else { return };
        if result.is_preview() {
            self.status =
                "Preview results are binned — run the full correction before exporting.".to_owned();
            return;
        }
        let input_dir_name = self
            .input_dir
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "images".to_owned());
        let mut dialog = rfd::FileDialog::new()
            .set_title("Choose the folder that will receive the corrected images");
        if let Some(parent) = self.input_dir.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(parent);
        }
        let Some(output_dir) = dialog.pick_folder() else {
            return;
        };
        let (h, w) = result.dims();
        let provenance = Provenance {
            input_folder: self.input_dir.clone().unwrap_or_default(),
            num_images: result.frames.len(),
            image_width: w,
            image_height: h,
            params: result.params,
            subspace_dimension: result.subspace_dimension,
            bin: result.bin,
            elapsed_seconds: result.elapsed_seconds,
            mbirjax_version: result.mbirjax_version.clone(),
        };
        let rx = start_export(
            output_dir,
            input_dir_name,
            result.frames.clone(),
            stack.sources.clone(),
            stack.transposed_on_load,
            provenance,
            ctx.clone(),
        );
        self.export_job = Some(ExportJob {
            rx,
            done: 0,
            total: result.frames.len(),
        });
        self.status = "Exporting corrected images…".to_owned();
    }

    /// Save the region profiles (plus TOF/λ columns when available) as CSV.
    fn save_profiles_csv(&mut self) {
        let Some((uncorrected, corrected)) = self.profiles.clone() else {
            return;
        };
        let mut dialog = rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .set_file_name("profiles.csv")
            .set_title("Save the region profiles as CSV");
        if let Some(dir) = &self.input_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else {
            return;
        };
        let tof = self.spectra_tof_us.clone();
        let lambda: Option<Vec<f64>> = tof.as_ref().map(|tof| {
            tof.iter()
                .map(|&t| spectra::tof_us_to_lambda_angstroms(t, self.distance_m))
                .collect()
        });
        match crate::export::write_profiles_csv(
            &path,
            &uncorrected,
            &corrected,
            tof.as_deref(),
            lambda.as_deref(),
        ) {
            Ok(()) => self.status = format!("Profiles saved to {}", path.display()),
            Err(e) => self.status = format!("CSV save failed: {e:#}"),
        }
    }

    fn poll_export(&mut self) {
        let mut done = None;
        if let Some(job) = &mut self.export_job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    ExportMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                    }
                    ExportMsg::Done(res) => done = Some(res),
                }
            }
        }
        let Some(res) = done else { return };
        self.export_job = None;
        match res {
            Ok(folder) => {
                self.status = format!("Corrected images exported to {}", folder.display());
                self.last_export = Some(folder);
            }
            Err(e) => self.status = format!("Export failed: {e}"),
        }
    }

    // ----- profiles ----------------------------------------------------------

    fn recompute_profiles(&mut self) {
        if !self.profiles_dirty {
            return;
        }
        self.profiles_dirty = false;
        self.profiles = None;
        self.pixel_profiles = None;
        let (Some(stack), Some(result)) = (&self.stack, &self.result) else {
            return;
        };
        let (h, w) = result.dims();
        let n = result.frames.len();

        if let Some(region) = self.region {
            let r = region.clamped(w, h);
            let mean_of = |frame: &Array2<f32>| -> f64 {
                frame
                    .slice(s![r.top..r.bottom, r.left..r.right])
                    .mapv(f64::from)
                    .mean()
                    .unwrap_or(0.0)
            };
            let uncorrected: Vec<f64> = (0..n)
                .filter_map(|i| result.raw_frame(stack, i))
                .map(mean_of)
                .collect();
            let corrected: Vec<f64> = result.frames.iter().map(mean_of).collect();
            self.profiles = Some((uncorrected, corrected));
        }

        if let Some((py, px)) = self.pixel_marker {
            if py < h && px < w {
                let uncorrected: Vec<f64> = (0..n)
                    .filter_map(|i| result.raw_frame(stack, i))
                    .map(|f| f64::from(f[(py, px)]))
                    .collect();
                let corrected: Vec<f64> = result
                    .frames
                    .iter()
                    .map(|f| f64::from(f[(py, px)]))
                    .collect();
                self.pixel_profiles = Some((uncorrected, corrected));
            }
        }
    }

    /// X-axis values for the profile plots, per the selected axis.
    fn x_values(&self, n: usize) -> Vec<f64> {
        match (self.x_axis, &self.spectra_tof_us) {
            (XAxis::TofUs, Some(tof)) => tof.iter().take(n).copied().collect(),
            (XAxis::LambdaAngstrom, Some(tof)) => tof
                .iter()
                .take(n)
                .map(|&t| spectra::tof_us_to_lambda_angstroms(t, self.distance_m))
                .collect(),
            _ => (0..n).map(|i| i as f64).collect(),
        }
    }

    // ----- textures ----------------------------------------------------------

    /// The images of the current view as owned copies: `(left, right,
    /// right_is_difference)` — right is `None` in the Profiles view (the
    /// plot takes its place).
    fn display_images(&self) -> (Option<Array2<f32>>, Option<Array2<f32>>, bool) {
        match self.view {
            View::Raw => (
                self.stack
                    .as_ref()
                    .and_then(|s| s.frames.get(self.frame_idx))
                    .cloned(),
                self.integrated_raw.clone(),
                false,
            ),
            View::Result => {
                let (Some(stack), Some(result)) = (&self.stack, &self.result) else {
                    return (None, None, false);
                };
                let corrected = result.frames.get(self.frame_idx).cloned();
                let raw = result.raw_frame(stack, self.frame_idx).cloned();
                match self.right_pane {
                    RightPane::Raw => (corrected, raw, false),
                    RightPane::Difference => {
                        let diff = match (&corrected, &raw) {
                            (Some(c), Some(r)) if c.dim() == r.dim() => Some(c - r),
                            _ => None,
                        };
                        (corrected, diff, true)
                    }
                }
            }
            View::Profiles => (
                self.result.as_ref().map(|r| r.integrated_mean.clone()),
                None,
                false,
            ),
        }
    }

    fn pane_titles(&self) -> (String, String) {
        let preview = self
            .result
            .as_ref()
            .filter(|r| r.is_preview())
            .map(|r| format!("  [PREVIEW {0}×{0} binned]", r.bin))
            .unwrap_or_default();
        match self.view {
            View::Raw => (
                format!("Raw image #{}", self.frame_idx),
                "Integrated image (sum)".to_owned(),
            ),
            View::Result => (
                format!("Corrected image #{}{preview}", self.frame_idx),
                match self.right_pane {
                    RightPane::Raw => format!("Uncorrected image #{}{preview}", self.frame_idx),
                    RightPane::Difference => {
                        format!("Corrected − raw #{}{preview}", self.frame_idx)
                    }
                },
            ),
            View::Profiles => (
                format!("Integrated corrected image (mean){preview}"),
                String::new(),
            ),
        }
    }

    fn ensure_textures(&mut self, ctx: &egui::Context) {
        if !self.tex_dirty {
            return;
        }
        let lut = self.colormap.lut();

        let (left, right, right_is_diff) = self.display_images();

        // Contrast range follows the left image while on auto.
        let range = left.as_ref().map(finite_range);
        if let Some((lo, hi)) = range {
            self.data_min = lo;
            self.data_max = hi;
            if self.contrast_auto {
                self.vmin = lo;
                self.vmax = hi;
            }
        }
        let (vmin, vmax) = (self.vmin, self.vmax);
        let left_color = left.as_ref().map(|img| colorize(img, vmin, vmax, &lut));
        let right_color = right.as_ref().map(|img| {
            let (lo, hi) = if right_is_diff {
                // Symmetric range around 0: structure in the difference
                // stands out regardless of sign.
                let (lo, hi) = finite_range(img);
                let a = lo.abs().max(hi.abs()).max(1e-12);
                (-a, a)
            } else {
                match self.view {
                    // Raw view: the integrated image has its own scale;
                    // Result view: corrected and raw share the contrast.
                    View::Raw => finite_range(img),
                    _ => (vmin, vmax),
                }
            };
            colorize(img, lo, hi, &lut)
        });
        self.pane_cache = (left, right);

        self.tex_left =
            left_color.map(|c| ctx.load_texture("pane_left", c, TextureOptions::NEAREST));
        self.tex_right =
            right_color.map(|c| ctx.load_texture("pane_right", c, TextureOptions::NEAREST));

        // Colorbar strip: 1×256, high values at the top.
        let mut cbuf = vec![0u8; 256 * 4];
        for j in 0..256 {
            let [r, g, b] = lut[255 - j];
            cbuf[j * 4] = r;
            cbuf[j * 4 + 1] = g;
            cbuf[j * 4 + 2] = b;
            cbuf[j * 4 + 3] = 255;
        }
        let cbar = egui::ColorImage::from_rgba_unmultiplied([1, 256], &cbuf);
        self.cbar_tex = Some(ctx.load_texture("colorbar", cbar, TextureOptions::LINEAR));

        self.tex_dirty = false;
    }

    // ----- dialogs -----------------------------------------------------------

    fn open_files_dialog(&mut self, ctx: &egui::Context) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Images", loader::SUPPORTED_EXTENSIONS)
            .set_title("Open TIFF image(s)")
            .pick_files()
        {
            self.start_load(files, ctx);
        }
    }

    /// The "🕒 Recent" drop-down: the last [`crate::recent::MAX_RECENT`]
    /// dataset folders, most recent first. Folders that no longer exist are
    /// shown disabled.
    fn recent_menu(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.set_min_width(280.0);
        for dir in self.recent.clone() {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| dir.display().to_string());
            let exists = dir.is_dir();
            let label = if exists {
                name
            } else {
                format!("{name}  (missing)")
            };
            if ui
                .add_enabled(exists, egui::Button::new(label).wrap_mode(egui::TextWrapMode::Extend))
                .on_hover_text(dir.display().to_string())
                .on_disabled_hover_text(format!("{} no longer exists", dir.display()))
                .clicked()
            {
                match loader::list_supported_in_dir(&dir) {
                    Ok(files) => self.start_load(files, ctx),
                    Err(e) => self.status = format!("{e:#}"),
                }
                ui.close();
            }
        }
    }

    fn open_folder_dialog(&mut self, ctx: &egui::Context) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Open the folder containing the images to correct")
            .pick_folder()
        {
            match loader::list_supported_in_dir(&dir) {
                Ok(files) => self.start_load(files, ctx),
                Err(e) => self.status = format!("{e:#}"),
            }
        }
    }

    // ----- UI: toolbar, parameters, status -----------------------------------

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Far right: the mbirjax/About button; everything else flows in
            // from the left inside the nested layout.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("ℹ mbirjax")
                    .on_hover_text("Algorithm provenance — runs the local mbirjax library")
                    .clicked()
                {
                    self.show_about = true;
                }
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    self.toolbar_left(ui);
                });
            });
        });
    }

    /// The left-flowing part of the toolbar (everything except the
    /// right-aligned mbirjax button).
    fn toolbar_left(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let busy = self.loading.is_some();
            let ctx = ui.ctx().clone();
            if ui
                .add_enabled(!busy, egui::Button::new("📁 Open Folder…"))
                .clicked()
            {
                self.open_folder_dialog(&ctx);
            }
            if ui
                .add_enabled(!busy, egui::Button::new("📂 Open Files…"))
                .clicked()
            {
                self.open_files_dialog(&ctx);
            }
            ui.add_enabled_ui(!busy && !self.recent.is_empty(), |ui| {
                ui.menu_button("🕒 Recent", |ui| {
                    self.recent_menu(ui, &ctx);
                });
            });

            ui.separator();

            let has_result = self.result.is_some();
            let mut changed = false;
            changed |= ui
                .selectable_value(&mut self.view, View::Raw, "Raw data")
                .changed();
            ui.add_enabled_ui(has_result, |ui| {
                changed |= ui
                    .selectable_value(&mut self.view, View::Result, "Corrected vs raw")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.view, View::Profiles, "Profiles")
                    .changed();
            });

            if self.view == View::Result {
                ui.separator();
                ui.label("vs:");
                changed |= ui
                    .selectable_value(&mut self.right_pane, RightPane::Raw, "Raw")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.right_pane, RightPane::Difference, "Difference")
                    .on_hover_text(
                        "corrected − raw on a symmetric color range: structure here is \
                         what the correction removed (or invented)",
                    )
                    .changed();
            }

            let n_frames = self.stack.as_ref().map(|s| s.n_frames()).unwrap_or(0);
            if matches!(self.view, View::Raw | View::Result) && n_frames > 1 {
                ui.separator();
                ui.label("Image:");
                let max = n_frames - 1;
                let source = self
                    .stack
                    .as_ref()
                    .and_then(|s| s.sources.get(self.frame_idx.min(max)))
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                changed |= ui
                    .add(egui::Slider::new(&mut self.frame_idx, 0..=max))
                    .on_hover_text(source)
                    .changed();
            }
            if changed {
                self.frame_idx = self.frame_idx.min(n_frames.saturating_sub(1));
                self.tex_dirty = true;
            }

            ui.separator();

            ui.label("Contrast:");
            let range = self.data_min..=self.data_max;
            let speed = (self.data_max - self.data_min).max(1.0) / 200.0;
            let r1 = ui.add(
                egui::DragValue::new(&mut self.vmin)
                    .speed(speed)
                    .range(range.clone()),
            );
            let r2 = ui.add(egui::DragValue::new(&mut self.vmax).speed(speed).range(range));
            if r1.changed() || r2.changed() {
                self.contrast_auto = false;
                self.tex_dirty = true;
            }
            if ui
                .button("Auto")
                .on_hover_text("Track the displayed image's full range")
                .clicked()
            {
                self.contrast_auto = true;
                self.tex_dirty = true;
            }

            ui.separator();

            let mut cmap_changed = false;
            egui::ComboBox::from_id_salt("colormap")
                .selected_text(format!("Colormap: {}", self.colormap.label()))
                .show_ui(ui, |ui| {
                    for c in Colormap::ALL {
                        cmap_changed |= ui
                            .selectable_value(&mut self.colormap, c, c.label())
                            .changed();
                    }
                });
            if cmap_changed {
                self.tex_dirty = true;
            }

            ui.separator();
            ui.label("Zoom:");
            if ui.button("−").clicked() {
                self.scale = (self.scale / 1.25).max(0.02);
            }
            if ui.button("+").clicked() {
                self.scale = (self.scale * 1.25).min(64.0);
            }
            if ui.button("Fit").clicked() {
                self.fit_requested = true;
            }
            ui.label(format!("{:.0}%", self.scale * 100.0));

            ui.separator();
            crate::theme::toggle_button(ui);
        });
    }

    fn params_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(stack) = &self.stack {
            ui.heading("Data set");
            ui.add_space(2.0);
            if let Some(dir) = &self.input_dir {
                ui.label(
                    egui::RichText::new(dir.display().to_string())
                        .small()
                        .weak(),
                )
                .on_hover_text("Folder the images were loaded from");
            }
            let (n, h, w) = (stack.n_frames(), stack.height, stack.width);
            ui.label(format!("{n} images of {w}×{h} px"));
            ui.label(format!(
                "In memory: {} (f32), correction needs ≈{} more",
                fmt_bytes(n as u64 * (h * w) as u64 * 4),
                // (points × bands) f64 working matrix + f32 corrected stack
                fmt_bytes(n as u64 * (h * w) as u64 * 12),
            ));
            if stack.nonfinite_fixed > 0 {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!(
                        "{} NaN/Inf pixel(s) replaced by 0 on load",
                        stack.nonfinite_fixed
                    ),
                );
            }
            if let (Some(first), Some(last)) = (stack.sources.first(), stack.sources.last()) {
                let name = |p: &std::path::Path| {
                    p.file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default()
                };
                ui.label(
                    egui::RichText::new(format!("{} … {}", name(first), name(last)))
                        .small()
                        .weak(),
                )
                .on_hover_text("First and last image file of the stack");
            }
            ui.add_space(6.0);
            ui.separator();
        }

        ui.heading("Correction parameters");
        ui.add_space(4.0);

        let running = self.corr_job.is_some();
        ui.add_enabled_ui(!running, |ui| {
            egui::ComboBox::from_label("Dataset type")
                .selected_text(self.params.dataset_type.label())
                .show_ui(ui, |ui| {
                    for t in [DatasetType::Attenuation, DatasetType::Transmission] {
                        ui.selectable_value(&mut self.params.dataset_type, t, t.label());
                    }
                });
            ui.label("attenuation = −log(transmission)")
                .on_hover_text("Pick 'transmission' when the images are normalized transmission data");
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.params.num_materials, 1..=10)
                        .text("Number of materials"),
                )
                .on_hover_text("How many different materials the data set contains");
                let estimating = self.estimate_job.is_some();
                let ctx = ui.ctx().clone();
                if estimating {
                    ui.spinner();
                } else if ui
                    .add_enabled(self.stack.is_some(), egui::Button::new("Auto"))
                    .on_hover_text(
                        "Estimate from the data: fits a noise model to the singular values \
                         of sampled pixel spectra and counts the ones above it",
                    )
                    .clicked()
                {
                    self.start_estimation(&ctx);
                }
            });

            egui::ComboBox::from_label("Beta loss")
                .selected_text(self.params.beta_loss.label())
                .show_ui(ui, |ui| {
                    for b in [BetaLoss::Frobenius, BetaLoss::KullbackLeibler] {
                        ui.selectable_value(&mut self.params.beta_loss, b, b.label());
                    }
                });

            ui.add(egui::Slider::new(&mut self.params.max_iter, 50..=1000).text("Max iterations"))
                .on_hover_text("Maximum iterations for the NMF solver");
        });

        ui.add_space(8.0);
        let ctx = ui.ctx().clone();
        if let Some(job) = &self.corr_job {
            ui.add(
                egui::ProgressBar::new(job.fraction)
                    .show_percentage()
                    .text(job.stage.clone()),
            );
            if ui.button("✖ Cancel").clicked() {
                job.cancel.store(true, Ordering::Relaxed);
            }
        } else {
            let can_run = self.stack.is_some() && self.loading.is_none();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_run, egui::Button::new("▶ Perform correction"))
                    .on_hover_text(
                        "Runs the NMF dehydration/hydration denoising on the whole stack",
                    )
                    .clicked()
                {
                    self.start_correction_job(&ctx, 1);
                }
                if ui
                    .add_enabled(can_run, egui::Button::new("⚡ Preview"))
                    .on_hover_text(format!(
                        "Fast preview on {PREVIEW_BIN}×{PREVIEW_BIN}-binned pixels (~{}× \
                         faster) for parameter tuning; previews cannot be exported",
                        PREVIEW_BIN * PREVIEW_BIN
                    ))
                    .clicked()
                {
                    self.start_correction_job(&ctx, PREVIEW_BIN);
                }
            });
        }

        if let Some(result) = &self.result {
            ui.add_space(10.0);
            ui.separator();
            if result.is_preview() {
                ui.heading(format!("Result — PREVIEW ({0}×{0} binned)", result.bin));
            } else {
                ui.heading("Result");
            }
            ui.label(format!(
                "{} materials → subspace dimension {}",
                result.params.num_materials, result.subspace_dimension
            ));
            ui.label(format!(
                "{} · {} · {} iterations max",
                result.params.dataset_type.label(),
                result.params.beta_loss.label(),
                result.params.max_iter
            ));
            ui.label(format!("Computed in {:.1} s", result.elapsed_seconds));
            if self.params != result.params {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "Parameters changed since this result — run again to apply.",
                );
            }

            ui.add_space(8.0);
            if let Some(job) = &self.export_job {
                let frac = if job.total > 0 {
                    job.done as f32 / job.total as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("Exporting {} / {}", job.done, job.total)),
                );
            } else {
                let exportable = !result.is_preview();
                if ui
                    .add_enabled(exportable, egui::Button::new("💾 Export corrected images…"))
                    .on_hover_text(
                        "Choose an output folder; the corrected stack is written as \
                         32-bit float TIFFs (plus correction_config.json recording the \
                         parameters) into a new subfolder named after the input folder",
                    )
                    .on_disabled_hover_text(
                        "Preview results are binned — run the full correction to export",
                    )
                    .clicked()
                {
                    self.export_dialog(&ctx);
                }
            }
            if let Some(folder) = &self.last_export {
                ui.add_space(4.0);
                ui.label("Exported to:");
                ui.label(
                    egui::RichText::new(folder.display().to_string())
                        .small()
                        .weak(),
                );
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(&self.status);
            if let Some(stack) = &self.stack {
                ui.separator();
                ui.label(format!(
                    "{} images, {}×{} px",
                    stack.n_frames(),
                    stack.width,
                    stack.height
                ));
            }
            if let Some((x, y, v)) = self.cursor {
                ui.separator();
                ui.label(format!("({x}, {y}) = {v:.4}"));
            }
        });
    }

    // ----- UI: image viewers -------------------------------------------------

    fn viewer(&mut self, ui: &mut egui::Ui) {
        if let Some(job) = &self.loading {
            let frac = if job.total > 0 {
                job.done as f32 / job.total as f32
            } else {
                0.0
            };
            ui.centered_and_justified(|ui| {
                ui.add_sized(
                    [320.0, 24.0],
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("⏳ Loading {} / {} files", job.done, job.total)),
                );
            });
            return;
        }
        if self.stack.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("No images loaded — use 'Open Folder…' to select the data to correct.");
            });
            return;
        }
        match self.view {
            View::Raw | View::Result => self.dual_viewer(ui),
            View::Profiles => self.profiles_view(ui),
        }
    }

    /// Two images side by side (raw + integrated, or corrected + raw /
    /// difference) with a shared zoom and the colorbar of the
    /// contrast-controlled left pane.
    fn dual_viewer(&mut self, ui: &mut egui::Ui) {
        // Dimensions come from the displayed image, not the stack: preview
        // results are spatially binned.
        let Some((h, w)) = self.pane_cache.0.as_ref().map(|img| img.dim()) else {
            return;
        };
        let (title_left, title_right) = self.pane_titles();

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(title_left).strong());
            ui.label(" | ");
            ui.label(egui::RichText::new(title_right).strong());
        });

        ui.horizontal_top(|ui| {
            let avail = ui.available_size();
            let view_w = (avail.x - COLORBAR_WIDTH - ui.spacing().item_spacing.x).max(50.0);
            if self.fit_requested && w > 0 && h > 0 {
                let gap = ui.spacing().item_spacing.x;
                let s = ((view_w - gap) / (2.0 * w as f32)).min(avail.y / h as f32);
                self.scale = s.clamp(0.02, 64.0);
                self.fit_requested = false;
            }

            let panes = [
                (self.tex_left.clone(), self.pane_cache.0.clone()),
                (self.tex_right.clone(), self.pane_cache.1.clone()),
            ];
            ui.allocate_ui(egui::vec2(view_w, avail.y), |ui| {
                ui.set_min_size(egui::vec2(view_w, avail.y));
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.horizontal_top(|ui| {
                            let size = egui::vec2(w as f32 * self.scale, h as f32 * self.scale);
                            let mut cursor = None;
                            for (tex, img) in &panes {
                                let Some(tex) = tex else { continue };
                                let (rect, response) =
                                    ui.allocate_exact_size(size, Sense::hover());
                                let full_uv =
                                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                                ui.painter_at(rect).image(
                                    tex.id(),
                                    rect,
                                    full_uv,
                                    Color32::WHITE,
                                );
                                if let (Some(p), Some(img)) = (response.hover_pos(), img) {
                                    let ix = ((p.x - rect.left()) / self.scale).floor() as i64;
                                    let iy = ((p.y - rect.top()) / self.scale).floor() as i64;
                                    if ix >= 0 && iy >= 0 && (ix as usize) < w && (iy as usize) < h
                                    {
                                        cursor = Some((
                                            ix as usize,
                                            iy as usize,
                                            img[(iy as usize, ix as usize)],
                                        ));
                                    }
                                }
                            }
                            self.cursor = cursor;
                        });
                    });
            });

            self.colorbar(ui, avail.y);
        });
    }

    /// Profiles view: the integrated corrected image with a draggable region
    /// on the left, the region's mean-intensity profiles on the right.
    fn profiles_view(&mut self, ui: &mut egui::Ui) {
        let Some((h, w)) = self.result.as_ref().map(|r| r.dims()) else {
            return;
        };
        let (title_left, _) = self.pane_titles();

        ui.horizontal_top(|ui| {
            let avail = ui.available_size();
            let img_w = (avail.x * 0.42).max(120.0);

            // ---- left: image + region ----
            ui.vertical(|ui| {
                ui.set_max_width(img_w);
                ui.label(egui::RichText::new(title_left).strong());
                ui.label(
                    "Drag to draw the region, drag inside it to move it, drag a handle to \
                     resize it. Click a pixel to plot its spectrum.",
                );
                if self.fit_requested && w > 0 && h > 0 {
                    let s = (img_w / w as f32).min((avail.y - 60.0).max(50.0) / h as f32);
                    self.scale = s.clamp(0.02, 64.0);
                    self.fit_requested = false;
                }
                egui::ScrollArea::both()
                    .id_salt("profile_img")
                    .auto_shrink([false, false])
                    .max_height(avail.y - 40.0)
                    .show(ui, |ui| {
                        self.profile_image(ui, w, h);
                    });
            });

            ui.separator();

            // ---- right: region fields + plot ----
            ui.vertical(|ui| {
                if let Some(region) = &mut self.region {
                    let mut r = *region;
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Left:");
                        changed |= drag_usize(ui, &mut r.left, w);
                        ui.label("Right:");
                        changed |= drag_usize(ui, &mut r.right, w);
                        ui.label("Top:");
                        changed |= drag_usize(ui, &mut r.top, h);
                        ui.label("Bottom:");
                        changed |= drag_usize(ui, &mut r.bottom, h);
                    });
                    if changed {
                        *region = r.clamped(w, h);
                        self.profiles_dirty = true;
                    }
                }
                ui.horizontal(|ui| {
                    if let Some(region) = self.region {
                        ui.label(format!(
                            "Region (rows {}:{}, columns {}:{})",
                            region.top, region.bottom, region.left, region.right
                        ));
                        ui.separator();
                    }
                    ui.label("Y-axis:");
                    ui.selectable_value(&mut self.log_y, false, "Linear");
                    ui.selectable_value(&mut self.log_y, true, "Log")
                        .on_hover_text("log₁₀ scale; non-positive values are hidden");
                    ui.separator();
                    if ui
                        .add_enabled(self.profiles.is_some(), egui::Button::new("📄 Save CSV…"))
                        .on_hover_text(
                            "Write the plotted profiles (with TOF/wavelength columns when \
                             a Spectra.txt was found) to a CSV file",
                        )
                        .clicked()
                    {
                        self.save_profiles_csv();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("X-axis:");
                    ui.selectable_value(&mut self.x_axis, XAxis::Index, "Image index");
                    ui.add_enabled_ui(self.spectra_tof_us.is_some(), |ui| {
                        ui.selectable_value(&mut self.x_axis, XAxis::TofUs, "TOF (µs)")
                            .on_disabled_hover_text("No *_Spectra.txt found next to the images");
                        ui.selectable_value(
                            &mut self.x_axis,
                            XAxis::LambdaAngstrom,
                            "Wavelength (Å)",
                        )
                        .on_disabled_hover_text("No *_Spectra.txt found next to the images");
                    });
                    if self.x_axis == XAxis::LambdaAngstrom {
                        ui.label("L (m):");
                        ui.add(
                            egui::DragValue::new(&mut self.distance_m)
                                .speed(0.01)
                                .range(0.1..=100.0),
                        )
                        .on_hover_text("Source–detector distance for λ = h·t/(mₙ·L)");
                    }
                    if let Some((py, px)) = self.pixel_marker {
                        ui.separator();
                        ui.label(format!("Pixel ({px}, {py})"));
                        if ui.small_button("✖").on_hover_text("Remove the pixel marker").clicked()
                        {
                            self.pixel_marker = None;
                            self.pixel_profiles = None;
                        }
                    }
                });
                if let Some((uncorrected, corrected)) = &self.profiles {
                    // In log mode the plotted values are log10(y); the axis
                    // ticks and the cursor read-out convert back.
                    let log_y = self.log_y;
                    let xs = self.x_values(uncorrected.len().max(corrected.len()));
                    let series = |vals: &[f64]| -> PlotPoints {
                        vals.iter()
                            .enumerate()
                            .filter(|&(_, &v)| !log_y || v > 0.0)
                            .map(|(i, &v)| {
                                [
                                    xs.get(i).copied().unwrap_or(i as f64),
                                    if log_y { v.log10() } else { v },
                                ]
                            })
                            .collect()
                    };
                    let uncorr = series(uncorrected);
                    let corr = series(corrected);
                    let pixel_series = self
                        .pixel_profiles
                        .as_ref()
                        .map(|(u, c)| (series(u), series(c)));
                    let x_axis = self.x_axis;
                    let mut plot = Plot::new(("profiles_plot", log_y, x_axis.label()))
                        .legend(Legend::default())
                        .x_axis_label(x_axis.label())
                        .y_axis_label(if log_y {
                            "Average intensity (log)"
                        } else {
                            "Average intensity"
                        })
                        .coordinates_formatter(
                            Corner::LeftBottom,
                            CoordinatesFormatter::new(move |p, _| {
                                let y = if log_y { 10f64.powf(p.y) } else { p.y };
                                let x = match x_axis {
                                    XAxis::Index => format!("image {:.0}", p.x),
                                    XAxis::TofUs => format!("{:.2} µs", p.x),
                                    XAxis::LambdaAngstrom => format!("{:.4} Å", p.x),
                                };
                                format!("{x}  —  intensity {y:.5}")
                            }),
                        );
                    if log_y {
                        plot = plot
                            .y_axis_formatter(|mark, _| fmt_axis(10f64.powf(mark.value)));
                    }
                    plot.show(ui, |plot_ui| {
                        plot_ui.points(
                            Points::new("Uncorrected profile", uncorr)
                                .shape(MarkerShape::Cross)
                                .radius(3.0),
                        );
                        plot_ui.points(
                            Points::new("Corrected profile", corr)
                                .shape(MarkerShape::Circle)
                                .radius(2.5),
                        );
                        if let Some((pu, pc)) = pixel_series {
                            plot_ui.points(
                                Points::new("Pixel uncorrected", pu)
                                    .shape(MarkerShape::Asterisk)
                                    .radius(2.0),
                            );
                            plot_ui.points(
                                Points::new("Pixel corrected", pc)
                                    .shape(MarkerShape::Diamond)
                                    .radius(2.0),
                            );
                        }
                    });
                }
            });
        });
    }

    /// The integrated corrected image with the profile region rectangle;
    /// dragging draws a new region.
    fn profile_image(&mut self, ui: &mut egui::Ui, w: usize, h: usize) {
        let Some(tex) = self.tex_left.clone() else {
            return;
        };
        let size = egui::vec2(w as f32 * self.scale, h as f32 * self.scale);
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(tex.id(), rect, full_uv, Color32::WHITE);

        let scale = self.scale;
        let to_img =
            |p: Pos2| -> (f32, f32) { ((p.x - rect.left()) / scale, (p.y - rect.top()) / scale) };

        // Cursor read-out on the integrated corrected image.
        self.cursor = None;
        if let Some(p) = response.hover_pos() {
            let (ix, iy) = to_img(p);
            let (xi, yi) = (ix.floor() as i64, iy.floor() as i64);
            if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                if let Some(result) = &self.result {
                    self.cursor =
                        Some((xi as usize, yi as usize, result.integrated_mean[(yi as usize, xi as usize)]));
                }
            }
        }

        const HANDLE_HIT: f32 = 10.0;
        const HANDLE_SIZE: f32 = 9.0;
        let to_screen = |ix: f32, iy: f32| -> Pos2 {
            Pos2::new(rect.left() + ix * scale, rect.top() + iy * scale)
        };

        // Plain click (no drag): mark the pixel whose spectrum to plot.
        if response.clicked() {
            if let Some(p) = response.interact_pointer_pos() {
                let (ix, iy) = to_img(p);
                let (xi, yi) = (ix.floor() as i64, iy.floor() as i64);
                if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                    let new = (yi as usize, xi as usize);
                    self.pixel_marker = if self.pixel_marker == Some(new) {
                        None // clicking the marked pixel clears it
                    } else {
                        Some(new)
                    };
                    self.profiles_dirty = true;
                }
            }
        }

        // A drag starts on a handle (resize), inside the region (move), or
        // anywhere else (draw a new region).
        if response.drag_started() {
            if let Some(sp) = response.interact_pointer_pos() {
                let p = to_img(sp);
                let started = self.region.and_then(|r| {
                    let rectf = [
                        r.left as f32,
                        r.top as f32,
                        r.right as f32,
                        r.bottom as f32,
                    ];
                    if let Some(((hx, vy), _)) = r
                        .handles()
                        .into_iter()
                        .find(|(_, (ix, iy))| to_screen(*ix, *iy).distance(sp) <= HANDLE_HIT)
                    {
                        Some(RegionDrag {
                            mode: DragMode::Resize { hx, vy },
                            rect: rectf,
                        })
                    } else if r.contains(p.0, p.1) {
                        Some(RegionDrag {
                            mode: DragMode::Move { last: p },
                            rect: rectf,
                        })
                    } else {
                        None
                    }
                });
                self.region_drag = Some(started.unwrap_or(RegionDrag {
                    mode: DragMode::Draw,
                    rect: [p.0, p.1, p.0, p.1],
                }));
            }
        }

        if response.dragged() || response.drag_stopped() {
            if let (Some(drag), Some(sp)) =
                (self.region_drag.as_mut(), response.interact_pointer_pos())
            {
                let p = to_img(sp);
                match &mut drag.mode {
                    DragMode::Draw => {
                        drag.rect[2] = p.0;
                        drag.rect[3] = p.1;
                    }
                    DragMode::Move { last } => {
                        // Translate, keeping the region fully inside the image.
                        let (dx, dy) = (p.0 - last.0, p.1 - last.1);
                        let rw = drag.rect[2] - drag.rect[0];
                        let rh = drag.rect[3] - drag.rect[1];
                        let nx0 = (drag.rect[0] + dx).clamp(0.0, (w as f32 - rw).max(0.0));
                        let ny0 = (drag.rect[1] + dy).clamp(0.0, (h as f32 - rh).max(0.0));
                        drag.rect = [nx0, ny0, nx0 + rw, ny0 + rh];
                        *last = p;
                    }
                    DragMode::Resize { hx, vy } => {
                        if *hx == -1 {
                            drag.rect[0] = p.0;
                        } else if *hx == 1 {
                            drag.rect[2] = p.0;
                        }
                        if *vy == -1 {
                            drag.rect[1] = p.1;
                        } else if *vy == 1 {
                            drag.rect[3] = p.1;
                        }
                    }
                }

                // Commit the working rectangle to the integer region.
                let (x0, x1) = (drag.rect[0].min(drag.rect[2]), drag.rect[0].max(drag.rect[2]));
                let (y0, y1) = (drag.rect[1].min(drag.rect[3]), drag.rect[1].max(drag.rect[3]));
                let new = Region {
                    left: x0.round().max(0.0) as usize,
                    right: (x1.round().max(0.0) as usize).min(w),
                    top: y0.round().max(0.0) as usize,
                    bottom: (y1.round().max(0.0) as usize).min(h),
                }
                .clamped(w, h);
                // A fresh draw only takes effect once it grows beyond a click.
                let keep = match drag.mode {
                    DragMode::Draw => new.right > new.left + 1 || new.bottom > new.top + 1,
                    _ => true,
                };
                if keep && self.region != Some(new) {
                    self.region = Some(new);
                    self.profiles_dirty = true;
                }
            }
            if response.drag_stopped() {
                self.region_drag = None;
            }
        }

        // Hover cursor: resize arrows over a handle, grab inside the region.
        if !response.dragged() {
            if let (Some(hp), Some(r)) = (response.hover_pos(), self.region) {
                if let Some(((hx, vy), _)) = r
                    .handles()
                    .into_iter()
                    .find(|(_, (ix, iy))| to_screen(*ix, *iy).distance(hp) <= HANDLE_HIT)
                {
                    ui.ctx().set_cursor_icon(cursor_for_handle(hx, vy));
                } else {
                    let (ix, iy) = to_img(hp);
                    if r.contains(ix, iy) {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                }
            }
        }

        // Draw the region and its handles.
        if let Some(r) = self.region {
            let rr = Rect::from_min_max(
                to_screen(r.left as f32, r.top as f32),
                to_screen(r.right as f32, r.bottom as f32),
            );
            painter.rect_stroke(
                rr,
                egui::CornerRadius::ZERO,
                Stroke::new(2.0, Color32::RED),
                egui::StrokeKind::Middle,
            );
            for (_, (ix, iy)) in r.handles() {
                let hr = Rect::from_center_size(
                    to_screen(ix, iy),
                    egui::vec2(HANDLE_SIZE, HANDLE_SIZE),
                );
                painter.rect_filled(hr, egui::CornerRadius::ZERO, Color32::WHITE);
                painter.rect_stroke(
                    hr,
                    egui::CornerRadius::ZERO,
                    Stroke::new(1.0, Color32::BLACK),
                    egui::StrokeKind::Middle,
                );
            }
        }

        // Crosshair on the marked pixel.
        if let Some((py, px)) = self.pixel_marker {
            let c = to_screen(px as f32 + 0.5, py as f32 + 0.5);
            let arm = 7.0;
            for (a, b) in [
                (Pos2::new(c.x - arm, c.y), Pos2::new(c.x + arm, c.y)),
                (Pos2::new(c.x, c.y - arm), Pos2::new(c.x, c.y + arm)),
            ] {
                painter.line_segment([a, b], Stroke::new(3.0, Color32::BLACK));
                painter.line_segment([a, b], Stroke::new(1.5, Color32::YELLOW));
            }
        }
    }

    /// Vertical colorbar for the current contrast range (left pane).
    fn colorbar(&self, ui: &mut egui::Ui, height: f32) {
        let Some(tex) = &self.cbar_tex else { return };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(COLORBAR_WIDTH, height), Sense::hover());
        let painter = ui.painter_at(rect);
        let font = egui::TextStyle::Small.resolve(ui.style());
        let text_color = ui.visuals().text_color();

        let pad = font.size * 0.6;
        let bar = Rect::from_min_max(
            Pos2::new(rect.left() + 2.0, rect.top() + pad),
            Pos2::new(rect.left() + 2.0 + 16.0, rect.bottom() - pad),
        );
        if bar.height() < 20.0 {
            return;
        }
        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(tex.id(), bar, full_uv, Color32::WHITE);
        painter.rect_stroke(
            bar,
            egui::CornerRadius::ZERO,
            Stroke::new(1.0, ui.visuals().widgets.noninteractive.fg_stroke.color),
            egui::StrokeKind::Outside,
        );

        let n_ticks = ((bar.height() / 60.0).floor() as usize + 2).clamp(2, 7);
        for i in 0..n_ticks {
            let t = i as f32 / (n_ticks - 1) as f32;
            let y = bar.bottom() - t * bar.height();
            let v = self.vmin + t * (self.vmax - self.vmin);
            painter.line_segment(
                [Pos2::new(bar.right(), y), Pos2::new(bar.right() + 4.0, y)],
                Stroke::new(1.0, text_color),
            );
            painter.text(
                Pos2::new(bar.right() + 6.0, y),
                egui::Align2::LEFT_CENTER,
                fmt_tick(v),
                font.clone(),
                text_color,
            );
        }
    }
}

fn drag_usize(ui: &mut egui::Ui, v: &mut usize, max: usize) -> bool {
    ui.add(egui::DragValue::new(v).speed(1).range(0..=max)).changed()
}

/// Finite min/max of an image (0..1 fallback for all-NaN data).
fn finite_range(img: &Array2<f32>) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in img.iter() {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

/// Map an image through the colormap LUT into an RGBA texture image.
fn colorize(img: &Array2<f32>, vmin: f32, vmax: f32, lut: &[[u8; 3]; 256]) -> egui::ColorImage {
    let (h, w) = (img.nrows(), img.ncols());
    let span = (vmax - vmin).max(1e-12);
    let mut buf = vec![0u8; w * h * 4];
    for (i, &v) in img.iter().enumerate() {
        let t = ((v - vmin) / span).clamp(0.0, 1.0);
        let idx = ((t * 255.0).round() as usize).min(255);
        let [r, g, b] = lut[idx];
        buf[i * 4] = r;
        buf[i * 4 + 1] = g;
        buf[i * 4 + 2] = b;
        buf[i * 4 + 3] = 255;
    }
    egui::ColorImage::from_rgba_unmultiplied([w, h], &buf)
}

/// Human-readable byte count (KB/MB/GB, decimal).
fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= 1000.0 && unit < UNITS.len() - 1 {
        v /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

/// Compact tick label for the log y-axis: linear-space value of a tick.
fn fmt_axis(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".to_owned()
    } else if a >= 100_000.0 || a < 0.001 {
        format!("{v:.1e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

/// Compact tick label: plain decimals in a comfortable range, scientific
/// notation for very large/small magnitudes.
fn fmt_tick(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".to_owned()
    } else if a >= 100_000.0 || a < 0.001 {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

impl eframe::App for DehydrationApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_load();
        self.poll_correction();
        self.poll_export();
        self.poll_estimation();
        if self.loading.is_some()
            || self.corr_job.is_some()
            || self.export_job.is_some()
            || self.estimate_job.is_some()
        {
            ctx.request_repaint();
        }
        self.recompute_profiles();
        self.ensure_textures(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.status_bar(ui);
        });
        egui::Panel::left("params")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                // Logos pinned to the bottom-left corner of the window: the
                // parameters scroll in the space above an explicitly reserved
                // logo row, and a spacer pushes the row to the very bottom
                // when the parameters are short.
                let logo_row_height = LOGO_HEIGHT + 24.0;
                let scroll_height = (ui.available_height() - logo_row_height).max(0.0);
                egui::ScrollArea::vertical()
                    .max_height(scroll_height)
                    .show(ui, |ui| {
                        self.params_panel(ui);
                    });
                ui.add_space((ui.available_height() - logo_row_height).max(0.0));
                ui.separator();
                self.logos_row(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| {
            self.viewer(ui);
        });

        self.about_modal(&ctx);
    }
}
