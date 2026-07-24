"""Registry of IQ evaluation datasets.

Each case documents the baseline settings for one dataset so that comparisons
across mbirjax versions and settings are stable. Run cases with run_recons.py.
All dataset paths assume /depot is mounted.

snr_db and sharpness are set explicitly in every case (they are deliberately NOT
in DEFAULTS, so adding a case without them fails loudly). Settings not listed in
a case fall back to DEFAULTS. Any loader kwarg for the case's type (see
LOADER_KEYS in run_recons.py) may be added per case.

full_res_subsample_view_factor is the view subsampling used by --full-res runs,
chosen per case so num_views is roughly 1/4 to 1/2 of num_det_channels;
--full-views overrides it to 1.
"""

# Shared recon defaults; individual cases may override.
DEFAULTS = dict(
    weight_type='transmission_root',
    max_iterations=15,
)

DATASETS = {
    # Purdue BGA (solder drops), Zeiss equiangle scan.
    'bga_no_hart': dict(
        type='zeiss',
        path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm',
        downsample_factor=3,
        subsample_view_factor=5,
        full_res_subsample_view_factor=4,   # 2401 views -> 601, 1532 channels
        snr_db=35.0,
        sharpness=1.5,
    ),

    # Purdue BGA (solder drops), Zeiss 360-degree HART scan of the same object.
    'bga_hart': dict(
        type='zeiss',
        path='/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_HART_360_HART.txrm',
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
        full_res_subsample_view_factor=2,   # 1050 views -> 525, 1024 channels
        snr_db=35.0,
        sharpness=1.0,
    ),

    # AFRL Nano CT Sample C, Zeiss scan. Settings from mbirjax_applications demo_zeiss.py.
    'nano_ct_c': dict(
        type='zeiss',
        path='/depot/bouman/data/AFRL/lipp/Black_Sheep_tomo-C_CS0.txrm',
        downsample_factor=2,
        subsample_view_factor=2,
        full_res_subsample_view_factor=2,   # 901 views -> 451, 1024 channels
        snr_db=35.0,
        sharpness=2.0,
    ),

    # Lilly autoinjector, NSI scan.
    'lilly_autoinjector': dict(
        type='nsi',
        path='/depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal',
        downsample_factor=2,
        subsample_view_factor=2,
        full_res_subsample_view_factor=2,   # 1800 views -> 900, 1880 channels
        snr_db=35.0,
        sharpness=1.0,
    ),
}
