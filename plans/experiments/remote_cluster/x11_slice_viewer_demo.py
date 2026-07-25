"""X11 smoke test: run a small mbirjax recon ON A GPU NODE and display it with
slice_viewer on the user's Mac, over ssh X11 forwarding + slurm's X11 prolog.

Prints a diagnostic block FIRST so that if no window appears we can tell which link
failed (DISPLAY not forwarded / backend fell back to Agg / no GPU) instead of guessing.

Small by design (64 views, 96x90 detector): every slice_viewer redraw ships pixels over
the network, so this is about proving the path works, not about a realistic workload.
"""
import os
import sys

print("=" * 68, flush=True)
print("NODE       :", os.uname().nodename, flush=True)
print("DISPLAY    :", os.environ.get("DISPLAY", "<UNSET -- X11 not forwarded>"), flush=True)
print("XAUTHORITY :", os.environ.get("XAUTHORITY", "<unset>"), flush=True)

import matplotlib
print("mpl backend BEFORE mbirjax :", matplotlib.get_backend(), flush=True)

import mbirjax as mj
# mbirjax/viewer.py does matplotlib.use('TkAgg') at import; if DISPLAY is missing or
# tkinter is broken it warns and leaves the backend at Agg, which draws nothing.
print("mpl backend AFTER  mbirjax :", matplotlib.get_backend(), flush=True)

import jax
print("jax devices:", jax.devices(), flush=True)

backend = matplotlib.get_backend().lower()
if backend == "agg":
    print("\nFATAL: backend is Agg (non-interactive) -- no window can appear.", flush=True)
    print("       Either DISPLAY is not set on this node, or TkAgg failed to load.", flush=True)
    sys.exit(3)

# Prove the GUI toolkit can actually open a connection before spending time on a recon.
try:
    import tkinter
    _root = tkinter.Tk()
    _root.withdraw()
    print("tkinter opened a display connection OK", flush=True)
    _root.destroy()
except Exception as e:
    print("\nFATAL: tkinter could not open the display:", type(e).__name__, e, flush=True)
    sys.exit(4)
print("=" * 68, flush=True)

# ── the actual work: small cone-beam demo data + a short recon, on the GPU ──────
print("\ngenerating demo data (small) ...", flush=True)
phantom, sinogram, params = mj.generate_demo_data(
    object_type='shepp-logan', model_type='cone',
    num_views=64, num_det_rows=96, num_det_channels=90)
angles = params['angles']
print("  sinogram", sinogram.shape, " phantom", phantom.shape, flush=True)

ct_model = mj.ConeBeamModel(sinogram.shape, angles,
                            source_detector_dist=params['source_detector_dist'],
                            source_iso_dist=params['source_iso_dist'])
ct_model.set_params(sharpness=1.0)
weights = mj.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

print("reconstructing (max_iterations=10) ...", flush=True)
recon, recon_dict = ct_model.recon(sinogram, weights=weights, max_iterations=10)
print("  recon", recon.shape, flush=True)
mj.get_memory_stats()

print("\nopening slice_viewer -- the window should appear on the Mac.", flush=True)
print("(this blocks until you close the window)", flush=True)
mj.slice_viewer(
    phantom, recon, slice_label=['Phantom', 'MBIR recon'],
    title='mbirjax on {} -- displayed on your Mac via X11'.format(os.uname().nodename))
print("slice_viewer closed cleanly.", flush=True)
