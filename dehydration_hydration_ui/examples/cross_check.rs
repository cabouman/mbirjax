//! Cross-validation helper: run the Rust hyper_denoise on a (points × bands)
//! .npy matrix and write the result, for comparison against the Python
//! `mbirjax.hsnt.hyper_denoise` output.
//!
//! Usage: cross_check <input.npy> <output.npy> [frobenius|kullback-leibler] [attenuation|transmission]

use dehydration_hydration::hsnt::{hyper_denoise, DatasetType, HsntParams};
use dehydration_hydration::nmf::BetaLoss;
use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::sync::atomic::AtomicBool;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = &args[1];
    let output = &args[2];
    let beta = match args.get(3).map(String::as_str) {
        Some("kullback-leibler") => BetaLoss::KullbackLeibler,
        _ => BetaLoss::Frobenius,
    };
    let dtype = match args.get(4).map(String::as_str) {
        Some("transmission") => DatasetType::Transmission,
        _ => DatasetType::Attenuation,
    };

    let file = std::fs::File::open(input).expect("open input");
    let x = Array2::<f64>::read_npy(file).expect("read npy");

    let params = HsntParams {
        dataset_type: dtype,
        num_materials: 2,
        beta_loss: beta,
        max_iter: 300,
        ..HsntParams::default()
    };
    let cancel = AtomicBool::new(false);
    let start = std::time::Instant::now();
    let y = hyper_denoise(x, &params, &cancel, &mut |_, _| {}).expect("denoise");
    eprintln!("rust hyper_denoise: {:.2} s", start.elapsed().as_secs_f64());

    let out = std::fs::File::create(output).expect("create output");
    y.write_npy(out).expect("write npy");
}
