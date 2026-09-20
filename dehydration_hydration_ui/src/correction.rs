//! Running the correction: build the (points × bands) hyperspectral matrix
//! from the image stack, denoise it, and reshape the result back into a
//! stack of frames.
//!
//! [`run_correction`] is the synchronous, GUI-free core (also used by the
//! headless `--run` mode); it calls the **local mbirjax Python library**
//! through [`crate::py_bridge`]. [`run_correction_native`] runs the bundled
//! Rust port of the algorithm instead (kept for tests and cross-checking).
//! [`start_correction`] wraps the mbirjax path on a background thread
//! streaming progress to the egui app over a channel.

use crate::hsnt::{hyper_denoise, DatasetType, HsntParams};
use crate::loader::ImageStack;
use crate::nmf::BetaLoss;
use anyhow::Result;
use ndarray::Array2;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

/// The user-facing parameters — the ones the notebook exposes. The rest of
/// [`HsntParams`] keeps the notebook's defaults.
#[derive(Clone, Copy, PartialEq)]
pub struct CorrectionParams {
    pub dataset_type: DatasetType,
    pub num_materials: usize,
    pub beta_loss: BetaLoss,
    pub max_iter: usize,
}

impl Default for CorrectionParams {
    fn default() -> Self {
        Self {
            dataset_type: DatasetType::Attenuation,
            num_materials: 2,
            beta_loss: BetaLoss::Frobenius,
            max_iter: 300,
        }
    }
}

impl CorrectionParams {
    pub fn to_hsnt(self) -> HsntParams {
        HsntParams {
            dataset_type: self.dataset_type,
            num_materials: self.num_materials,
            beta_loss: self.beta_loss,
            max_iter: self.max_iter,
            ..HsntParams::default()
        }
    }
}

pub struct CorrectionOutput {
    /// Corrected frames, same order as the input stack (spatially binned by
    /// `bin` when this is a preview run).
    pub frames: Vec<Array2<f32>>,
    /// Per-pixel mean over the corrected frames (the notebook's "integrated
    /// corrected image").
    pub integrated_mean: Array2<f32>,
    /// The raw frames the correction actually ran on, kept only when they
    /// differ from the input stack (bin > 1) so the comparison views and
    /// profiles have a like-for-like reference.
    pub binned_raw: Option<Vec<Array2<f32>>>,
    /// Spatial binning factor (1 = full resolution, >1 = preview).
    pub bin: usize,
    pub subspace_dimension: usize,
    pub elapsed_seconds: f64,
    /// Version of the mbirjax library that ran the correction (or a note
    /// that the native port did).
    pub mbirjax_version: String,
}

impl CorrectionOutput {
    pub fn is_preview(&self) -> bool {
        self.bin > 1
    }
}

/// Mean-bin every frame over `bin`×`bin` blocks (edges that do not fill a
/// block are dropped). `bin = 1` returns a plain copy.
pub fn bin_frames(frames: &[Array2<f32>], bin: usize) -> Vec<Array2<f32>> {
    if bin <= 1 {
        return frames.to_vec();
    }
    frames
        .iter()
        .map(|f| {
            let (h, w) = (f.nrows() / bin, f.ncols() / bin);
            Array2::from_shape_fn((h, w), |(y, x)| {
                let mut acc = 0.0f32;
                for dy in 0..bin {
                    for dx in 0..bin {
                        acc += f[(y * bin + dy, x * bin + dx)];
                    }
                }
                acc / (bin * bin) as f32
            })
        })
        .collect()
}

enum Backend {
    /// The local mbirjax Python library, through [`crate::py_bridge`].
    Mbirjax,
    /// The bundled Rust port ([`crate::hsnt`]).
    NativePort,
}

/// Synchronous correction of a whole stack through the local mbirjax
/// library. `bin > 1` runs a spatially binned preview. Progress arrives as
/// `(stage, fraction)`; setting `cancel` aborts the run (the error message
/// contains "cancelled").
pub fn run_correction(
    stack: &ImageStack,
    params: CorrectionParams,
    bin: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<CorrectionOutput> {
    run_correction_impl(stack, params, bin, cancel, progress, Backend::Mbirjax)
}

/// Same correction on the bundled Rust port of the algorithm — no Python
/// needed. Used by the unit tests and the `cross_check` example.
pub fn run_correction_native(
    stack: &ImageStack,
    params: CorrectionParams,
    bin: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<CorrectionOutput> {
    run_correction_impl(stack, params, bin, cancel, progress, Backend::NativePort)
}

fn run_correction_impl(
    stack: &ImageStack,
    params: CorrectionParams,
    bin: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
    backend: Backend,
) -> Result<CorrectionOutput> {
    let started = std::time::Instant::now();

    progress("Preparing data", 0.0);
    let bin = bin.max(1);
    let (work_frames, binned_raw) = if bin > 1 {
        let binned = bin_frames(&stack.frames, bin);
        (binned.clone(), Some(binned))
    } else {
        (stack.frames.clone(), None)
    };
    let (h, w) = match work_frames.first() {
        Some(f) => (f.nrows(), f.ncols()),
        None => anyhow::bail!("empty stack"),
    };
    let x = frames_to_matrix(&work_frames, h, w);
    progress("Preparing data", 1.0);

    let (denoised, subspace_dimension, mbirjax_version) = match backend {
        Backend::Mbirjax => {
            let out = crate::py_bridge::hyper_denoise_py(&x, &params.to_hsnt(), cancel, progress)?;
            (out.matrix, out.subspace_dimension, out.mbirjax_version)
        }
        Backend::NativePort => {
            let denoised = hyper_denoise(x, &params.to_hsnt(), cancel, progress)?;
            (
                denoised,
                params.to_hsnt().subspace_dimension(),
                format!("native port of {}", crate::hsnt::MBIRJAX_VERSION),
            )
        }
    };

    progress("Assembling corrected stack", 0.0);
    let frames = matrix_to_frames(&denoised, h, w);
    let integrated_mean = integrated_mean(&frames, h, w);
    progress("Assembling corrected stack", 1.0);

    Ok(CorrectionOutput {
        frames,
        integrated_mean,
        binned_raw,
        bin,
        subspace_dimension,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        mbirjax_version,
    })
}

pub enum CorrectionMsg {
    Progress { stage: String, fraction: f32 },
    Done(Result<CorrectionOutput, String>),
}

/// Spawn the correction thread for the GUI. Poll the returned receiver each
/// frame; store `cancel` and set it to stop the solver.
pub fn start_correction(
    stack: Arc<ImageStack>,
    params: CorrectionParams,
    bin: usize,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
) -> Receiver<CorrectionMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut progress = |stage: &str, fraction: f32| {
            let _ = tx.send(CorrectionMsg::Progress {
                stage: stage.to_owned(),
                fraction,
            });
            ctx.request_repaint();
        };
        let result = run_correction(&stack, params, bin, &cancel, &mut progress);
        let _ = tx.send(CorrectionMsg::Done(result.map_err(|e| format!("{e:#}"))));
        ctx.request_repaint();
    });
    rx
}

/// (points × bands) matrix: row = pixel (row-major over the frame), column =
/// image index. The image index is the spectral axis — the same layout as
/// the notebook's `swapaxes(raw, 0, 2)` + reshape, up to a pixel ordering
/// that the round-trip undoes.
pub fn frames_to_matrix(frames: &[Array2<f32>], h: usize, w: usize) -> Array2<f64> {
    let n = frames.len();
    let mut x = Array2::<f64>::zeros((h * w, n));
    for (i, frame) in frames.iter().enumerate() {
        let mut col = x.column_mut(i);
        for (dst, &v) in col.iter_mut().zip(frame.iter()) {
            *dst = f64::from(v);
        }
    }
    x
}

fn matrix_to_frames(x: &Array2<f32>, h: usize, w: usize) -> Vec<Array2<f32>> {
    let n = x.ncols();
    (0..n)
        .map(|i| {
            let col = x.column(i);
            Array2::from_shape_fn((h, w), |(y, xx)| col[y * w + xx])
        })
        .collect()
}

fn integrated_mean(frames: &[Array2<f32>], h: usize, w: usize) -> Array2<f32> {
    let mut acc = Array2::<f32>::zeros((h, w));
    for f in frames {
        acc += f;
    }
    if !frames.is_empty() {
        acc /= frames.len() as f32;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_stack(frames: Vec<Array2<f32>>) -> ImageStack {
        let (h, w) = (frames[0].nrows(), frames[0].ncols());
        let sources = frames.iter().map(|_| PathBuf::from("x.tif")).collect();
        ImageStack {
            frames,
            width: w,
            height: h,
            sources,
            transposed_on_load: true,
            nonfinite_fixed: 0,
        }
    }

    #[test]
    fn matrix_roundtrip_preserves_pixels() {
        let f0 = Array2::from_shape_fn((3, 4), |(y, x)| (y * 4 + x) as f32);
        let f1 = f0.mapv(|v| v * 10.0);
        let stack = make_stack(vec![f0.clone(), f1.clone()]);
        let x = frames_to_matrix(&stack.frames, 3, 4);
        assert_eq!(x.dim(), (12, 2));
        assert_eq!(x[[5, 0]], 5.0);
        assert_eq!(x[[5, 1]], 50.0);
        let back = matrix_to_frames(&x.mapv(|v| v as f32), 3, 4);
        assert_eq!(back[0], f0);
        assert_eq!(back[1], f1);
    }

    #[test]
    fn binning_averages_blocks_and_drops_ragged_edges() {
        let f = Array2::from_shape_fn((5, 6), |(y, x)| (y * 6 + x) as f32);
        let binned = bin_frames(&[f], 2);
        assert_eq!(binned[0].dim(), (2, 3)); // 5→2 rows (last dropped), 6→3 cols
        // Block (0,0): mean of values at (0,0),(0,1),(1,0),(1,1) = (0+1+6+7)/4
        assert_eq!(binned[0][(0, 0)], 3.5);
    }

    #[test]
    fn preview_run_returns_binned_frames_and_reference() {
        let frames: Vec<Array2<f32>> = (0..6)
            .map(|i| {
                Array2::from_shape_fn((8, 8), |(y, x)| {
                    1.0 + (i as f32) * 0.1 + ((y + x) as f32) * 0.01
                })
            })
            .collect();
        let stack = make_stack(frames);
        let cancel = AtomicBool::new(false);
        let out = run_correction_native(
            &stack,
            CorrectionParams {
                max_iter: 60,
                ..Default::default()
            },
            2,
            &cancel,
            &mut |_, _| {},
        )
        .unwrap();
        assert!(out.is_preview());
        assert_eq!(out.frames[0].dim(), (4, 4));
        assert_eq!(out.binned_raw.as_ref().unwrap()[0].dim(), (4, 4));
        assert_eq!(out.frames.len(), 6);
    }
}
