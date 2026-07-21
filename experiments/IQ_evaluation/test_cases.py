"""Registry of IQ evaluation test cases.

Each case documents the baseline settings for one dataset so that comparisons
across mbirjax versions and settings are stable. Run cases with run_recon.py.
All dataset paths assume /depot is mounted.

Settings not listed in a case fall back to DEFAULTS. Any loader kwarg for the
case's type (see LOADER_KEYS in run_recon.py) may be added per case.
"""

# Shared recon defaults; individual cases may override.
DEFAULTS = dict(
    sharpness=1.0,
    snr_db=35.0,
    weight_type='transmission_root',
    max_iterations=15,
)

TEST_CASES = {
    # Purdue BGA (solder drops), equiangle scan. Settings carried over from center_slice_zeiss.py.
    'bga_no_hart': dict(
        type='zeiss',
        path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm',
        downsample_factor=3,
        subsample_view_factor=5,
        sharpness=1.5,
    ),

    # ORNL HFN scan, pymbir hdf5 file (same data as hfn_scan.tgz used in mbirjax_applications/vcls).
    'hfn': dict(
        type='pymbir',
        path='/depot/bouman/data/ORNL/pymbir/hfn_scan/TCR_Single_Channeled_SRC_M_2019-03-18_13-08-09.hdf5',
        bh_correction=True,
    ),

    # Lilly autoinjector, NSI scan. PROVISIONAL baseline - adjust downsampling after first run.
    'lilly_autoinjector': dict(
        type='nsi',
        path='/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal',
        downsample_factor=4,
        subsample_view_factor=4,
    ),

    # Candidates to add later (see center_slice_zeiss.py for the full Zeiss list):
    # 'bga_hart': /depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_HART_360_HART.txrm
}
