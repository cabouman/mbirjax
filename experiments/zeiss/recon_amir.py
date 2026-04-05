# ── pyMBIR availability check and auto-install ────────────────────────────────
import importlib, subprocess, sys
from pathlib import Path

def _conda_install(pkg, channel=None):
    """Install a package into the current conda environment."""
    # Locate the conda executable relative to the running Python
    conda = Path(sys.executable).parent.parent / "bin" / "conda"
    if not conda.exists():
        conda = "conda"   # fall back to PATH
    cmd = [str(conda), "install", "-y", "--prefix", str(Path(sys.executable).parent.parent)]
    if channel:
        cmd += ["-c", channel]
    cmd.append(pkg)
    subprocess.check_call(cmd)


def _ensure_pymbir():
    """
    Check whether pyMBIR is importable.  If not:
      1. conda-install astra-toolbox (conda-only, not on PyPI)
      2. pip-install the remaining dependencies from environment.yml
      3. pip-install pyMBIR from GitHub
    """
    if importlib.util.find_spec("pyMBIR") is not None:
        return   # already installed

    print("pyMBIR not found — installing dependencies and pyMBIR from GitHub...")

    # astra-toolbox is only distributed via conda
    if importlib.util.find_spec("astra") is None:
        print("  conda install astra-toolbox (channel: astra-toolbox/label/dev)")
        try:
            _conda_install("astra-toolbox", channel="astra-toolbox/label/dev")
        except Exception as exc:
            raise RuntimeError(
                "Could not conda-install astra-toolbox.\n"
                "Install it manually with:\n"
                "    conda install -c astra-toolbox/label/dev astra-toolbox"
            ) from exc

    # pip-installable packages from environment.yml
    for pkg in ["numpy", "scipy", "matplotlib", "psutil"]:
        print(f"  pip install {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    # Install pyMBIR from GitHub (uses setup.py, so needs --no-build-isolation)
    pymbir_url = "https://github.com/svvenkatakrishnan/pyMBIR"
    print(f"  pip install git+{pymbir_url}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-build-isolation",
        f"git+{pymbir_url}",
    ])

    if importlib.util.find_spec("pyMBIR") is None:
        raise ImportError("pyMBIR installation appeared to succeed but is still not importable.")
    print("pyMBIR installed successfully.")

_ensure_pymbir()
# ─────────────────────────────────────────────────────────────────────────────

from scipy.ndimage.interpolation import shift
import numpy as np
import concurrent.futures as cf
# import dxchange
import psutil
# import pyqtgraph as pg
from pyMBIR.reconEngine import *
import os
from readOle_xtrm_distances import *
import tifffile
import mbirjax as mj
# from skimage.filters import threshold_otsu
from tqdm import tqdm


def read_tifffolder(dir_tiff):
    # reads the tiff sequance from dir_tiff folder
    tif_files = [f for f in sorted(os.listdir(dir_tiff)) if f.endswith(('tif', 'tiff'))]
    tif_files = [os.path.join(dir_tiff, s) for s in tif_files]
    shape_inp = tifffile.imread(tif_files[0])
    read_FDK = np.zeros((len(tif_files), shape_inp.shape[0], shape_inp.shape[1]))
    for i in range(len(tif_files)):
        read_FDK[i, :, :] = tifffile.imread(tif_files[i])

    return read_FDK


def apply_proj_offsets(proj, proj_offsets, ncore=None, out=None):
    if not ncore:
        ncore = psutil.cpu_count(True)
    if out is None:
        out = np.empty(proj.shape, dtype=proj.dtype)
    with cf.ThreadPoolExecutor(ncore) as e:
        futures = [e.submit(shift, proj[i], proj_offsets[i], out[i], order=1, mode='nearest') for i in
                   range(proj.shape[0])]  # nearest
        cf.wait(futures)
    return out


def create_circle_mask(y, x, center, rad):
    mask = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1]) <= rad * rad
    return mask


def apply_cylind_mask(im_size_o, num_slice_o, disc_height, center1, disc_rad1, rec_fdk):
    obj_mask = np.zeros((im_size_o, im_size_o, num_slice_o)).astype(np.float32)
    y, x = np.ogrid[-im_size_o / 2:im_size_o / 2, -im_size_o / 2:im_size_o / 2]
    height_idx = slice(0, disc_height)
    mask = create_circle_mask(y, x, center1, disc_rad1)
    obj_mask[mask, height_idx] = 1

    obj_maskT = np.transpose(obj_mask, [2, 0, 1])
    rec_fdk_f = np.multiply(rec_fdk, obj_maskT)

    return rec_fdk_f


data_folder = '/depot/bouman/data/ORNL/versa'
filenames = ['ParAM-Round-1_Z62.txrm']

sigma = 1  # 4#100#2#1
SHORTSCAN = False
view_subsamp = 2  # 5#
filtsize = 5
for file_name in filenames:
    data_path = os.path.join(data_folder, file_name)
    write_op = False
    gpu_index = [0]
    NUM_ITER = 120

    orig_det, src_orig = calc_XrayDet_distances(data_path)  # 32.24  # 97.5594 #mm
    # orig_det = 13.083  # mm
    center_offset_col = 0
    center_offset_row = 0
    n_vox_x = 1024  # 982#
    n_vox_y = 1024  # 982#
    n_vox_z = 1024  # 962#
    recon_voxel_fact = src_orig / (
                src_orig + orig_det)  # 1  # 1.0045*2#20/96 #0.5 #Size of recon voxel relavive to det pixel size

    ######End of inputs#######
    weight_data, metadata = read_txrm(data_path)
    center_shift = metadata['center_shift']
    opt_mag = np.round(metadata['opt_mag'], 2)
    # pull proj_offsets from metadata
    proj_offsets = np.array((metadata['y-shifts'], metadata['x-shifts'])).T
    angles_1 = metadata['thetas'].astype(np.float32)

    if np.abs(angles_1[0] - angles_1[-1]) < 2 * np.pi - 1:
        SHORTSCAN = True

    dang = np.diff(angles_1)
    angles = np.zeros(len(angles_1))
    for i in range(len(dang)):
        angles[i + 1] = - dang[i] + angles[i]
    I0 = metadata['reference']

    ##View sub-sampling####
    if view_subsamp > 1:
        weight_data = weight_data[::view_subsamp, :, :]
        angles = angles[::view_subsamp]
    num_angles = len(angles)
    proj_offsets = proj_offsets[::view_subsamp]

    det_row = metadata['image_height']
    det_col = metadata['image_width']

    det_pix_x = opt_mag * metadata['pixel_size'] / recon_voxel_fact / 1e3
    det_pix_y = opt_mag * metadata['pixel_size'] / recon_voxel_fact / 1e3
    print('Pixel size %f micron' % (metadata['pixel_size']))
    det_size = np.array([det_row, det_col])
    vol_xy = metadata['pixel_size'] / 1e3
    vol_z = vol_xy  # det_pix_y*recon_voxel_fact
    # rat = np.max(weight_data)/np.max(I0)
    proj_data = weight_data / I0  # /rat
    # temp = np.copy(proj_data[0])
    # temp = scipy.ndimage.median_filter(temp,footprint=np.ones((filtsize,filtsize)))
    #
    # proj_data /=(np.max(temp)+1)
    proj_data = -np.log(proj_data)
    proj_data[np.isnan(proj_data)] = 0
    proj_data[np.isinf(proj_data)] = 0

    # weight_data = apply_proj_offsets(weight_orig, proj_offsets)
    proj_data = apply_proj_offsets(proj_data, proj_offsets)

    weight_data = weight_data.swapaxes(0, 1).astype(np.float32)
    proj_data = proj_data.swapaxes(0, 1).astype(np.float32)

    # Display object
    print('Actual projection shape (%d,%d,%d)' % proj_data.shape)
    '''
    temp_proj_data=np.swapaxes(proj_data,0,1)
    temp_proj_data=np.swapaxes(temp_proj_data,1,2)
    pg.image(temp_proj_data);pg.QtGui.QApplication.exec_()
    '''

    proj_dims = np.array([det_row, num_angles, det_col])  # .astype(np.uint16)
    proj_params = {}
    proj_params['type'] = 'cone'
    proj_params['dims'] = proj_dims
    proj_params['angles'] = angles
    cone_params = {}
    cone_params['pix_x'] = det_pix_x
    cone_params['pix_y'] = det_pix_y
    cone_params['src_orig'] = src_orig
    cone_params['orig_det'] = opt_mag * src_orig / recon_voxel_fact - src_orig  # orig_det
    # cone_params['src_orig'] = src_orig
    # cone_params['orig_det'] = orig_det

    proj_params['cone_params'] = cone_params
    proj_params['forward_model_idx'] = 2

    miscalib = {}
    miscalib['delta_u'] = center_offset_col
    miscalib['delta_v'] = center_offset_row
    # miscalib['phi'] = 0

    rec_params = {}
    rec_params['num_iter'] = NUM_ITER
    rec_params['gpu_index'] = gpu_index
    # rec_params['MRF_P'] = MRF_P
    # rec_params['MRF_SIGMA'] = MRF_SIGMA
    rec_params['MRF_P'] = 1.2
    rec_params['MRF_SIGMA'] = sigma  # * (1.0 / vol_xy ** 2)  # 1.5
    print('Regularization paramter = %f' % rec_params['MRF_SIGMA'])
    rec_params['debug'] = False
    # rec_params['sigma'] = 1
    # rec_params['reject_frac'] = 0.1
    rec_params['verbose'] = True
    rec_params['stop_thresh'] = 0.001  # percent
    # rec_params['vol_row'] = n_vox_z
    # rec_params['vol_col'] = n_vox_x

    vol_params = {}
    vol_params['vox_xy'] = vol_xy
    vol_params['vox_z'] = vol_z
    vol_params['n_vox_x'] = n_vox_x
    vol_params['n_vox_y'] = n_vox_y
    vol_params['n_vox_z'] = n_vox_z

    rec_params['shep_logan_filt'] = False
    rec_params['short_scan'] = SHORTSCAN  # True#False
    miscalib['delta_u'] = center_shift * det_pix_x  #
    # miscalib['delta_v'] = -5 * det_pix_y
    A = generateAmatrix(proj_params, miscalib, vol_params, gpu_index)

    rec_fdk_2_s = analytic(proj_data, proj_params, miscalib, vol_params, rec_params)
    mj.slice_viewer(rec_fdk_2_s, slice_axis=0)
