//! Colormaps for displaying the integrated image.
//!
//! Each colormap is compiled to a 256-entry RGB lookup table so the per-pixel
//! display path is just an array index. Scientific maps come from `colorgrad`;
//! classic MATLAB-style "jet" is built from its control colors.

use colorgrad::Gradient;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Gray,
    Viridis,
    Inferno,
    Magma,
    Plasma,
    Cividis,
    Turbo,
    Jet,
}

impl Colormap {
    pub const ALL: [Colormap; 8] = [
        Colormap::Gray,
        Colormap::Viridis,
        Colormap::Inferno,
        Colormap::Magma,
        Colormap::Plasma,
        Colormap::Cividis,
        Colormap::Turbo,
        Colormap::Jet,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Colormap::Gray => "Gray",
            Colormap::Viridis => "Viridis",
            Colormap::Inferno => "Inferno",
            Colormap::Magma => "Magma",
            Colormap::Plasma => "Plasma",
            Colormap::Cividis => "Cividis",
            Colormap::Turbo => "Turbo",
            Colormap::Jet => "Jet",
        }
    }

    /// 256-entry RGB lookup table for this colormap.
    pub fn lut(self) -> [[u8; 3]; 256] {
        let mut lut = [[0u8; 3]; 256];
        match self {
            Colormap::Gray => {
                for (i, slot) in lut.iter_mut().enumerate() {
                    *slot = [i as u8; 3];
                }
            }
            Colormap::Viridis => fill(&mut lut, &colorgrad::preset::viridis()),
            Colormap::Inferno => fill(&mut lut, &colorgrad::preset::inferno()),
            Colormap::Magma => fill(&mut lut, &colorgrad::preset::magma()),
            Colormap::Plasma => fill(&mut lut, &colorgrad::preset::plasma()),
            Colormap::Cividis => fill(&mut lut, &colorgrad::preset::cividis()),
            Colormap::Turbo => fill(&mut lut, &colorgrad::preset::turbo()),
            Colormap::Jet => fill(&mut lut, &jet()),
        }
        lut
    }
}

fn fill<G: Gradient>(lut: &mut [[u8; 3]; 256], g: &G) {
    for (i, slot) in lut.iter_mut().enumerate() {
        let c = g.at(i as f32 / 255.0);
        *slot = [to_u8(c.r), to_u8(c.g), to_u8(c.b)];
    }
}

fn to_u8(v: f32) -> u8 {
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Classic MATLAB "jet" colormap from its control colors.
fn jet() -> colorgrad::LinearGradient {
    colorgrad::GradientBuilder::new()
        .html_colors(&[
            "#000080", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#800000",
        ])
        .domain(&[0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
        .build::<colorgrad::LinearGradient>()
        .expect("valid jet gradient")
}
