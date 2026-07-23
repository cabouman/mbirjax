"""Registry of IQ evaluation test cases.

Each case documents the baseline settings for one dataset so that comparisons
across mbirjax versions and settings are stable. Run cases with run_recon.py.
All dataset paths assume /depot is mounted.

snr_db and sharpness are set explicitly in every case (they are deliberately NOT
in DEFAULTS, so adding a case without them fails loudly). Settings not listed in
a case fall back to DEFAULTS. Any loader kwarg for the case's type (see
LOADER_KEYS in run_recon.py) may be added per case.
"""

# Shared recon defaults; individual cases may override.
DEFAULTS = dict(
    weight_type='transmission_root',
    max_iterations=15,
)

TEST_CASES = {
    # Purdue BGA (solder drops), Zeiss equiangle scan.
    'bga_no_hart': dict(
        type='zeiss',
        path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm',
        downsample_factor=3,
        subsample_view_factor=5,
        snr_db=35.0,
        sharpness=1.5,
    ),

    # ORNL HFN scan, pymbir hdf5 file (same data as hfn_scan.tgz used in mbirjax_applications/vcls).
    'hfn': dict(
        type='pymbir',
        path='/depot/bouman/data/ORNL/pymbir/hfn_scan/TCR_Single_Channeled_SRC_M_2019-03-18_13-08-09.hdf5',
        bh_correction=True,
        snr_db=35.0,
        sharpness=1.0,
    ),

    # AFRL Nano CT Sample C, Zeiss scan. Settings from mbirjax_applications demo_zeiss.py.
    'nano_ct_c': dict(
        type='zeiss',
        path='/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-C_CS0.txrm',
        downsample_factor=2,
        subsample_view_factor=2,
        snr_db=35.0,
        sharpness=2.0,
    ),

    # Lilly autoinjector, NSI scan.
    'lilly_autoinjector': dict(
        type='nsi',
        path='/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal',
        downsample_factor=2,
        subsample_view_factor=2,
        snr_db=35.0,
        sharpness=1.0,
    ),
}
