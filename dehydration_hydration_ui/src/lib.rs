//! Dehydration/Hydration correction library: stack loading, the NMF-based
//! hyperspectral denoising (run in the local mbirjax library through
//! `py_bridge`, with a native Rust port kept for tests), and TIFF export.
//!
//! The GUI binary (`main.rs`) is a thin shell around these modules; they are
//! exposed here so they can be unit/integration tested without a display.

pub mod app;
pub mod colormap;
pub mod correction;
pub mod export;
pub mod hsnt;
pub mod linalg;
pub mod loader;
pub mod nmf;
pub mod py_bridge;
pub mod recent;
pub mod spectra;
pub mod theme;
