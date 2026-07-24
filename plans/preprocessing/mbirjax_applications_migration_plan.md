# mbirjax_applications migration plan (match the preprocessing refactor)

**STATUS: EXECUTED (clean break, Greg-approved).** All 9 files migrated via fan-out (1 agent/file) +
py_compile-clean; net -60 lines (manual construct+set_params+auto boilerplate collapsed). The 3
highest-judgment diffs reviewed by hand: Lilly_recon/Lilly_recon_ps auto_crop_sino_conebeam -> auto_crop=
flag; demo_zeiss scanner_type branch -> reader auto-select; demo_vcls_ornl_part raw-param recovery via
get_all_params (angles wrapped np.asarray for python-list indexing; build_reference_object needed NO
recovery). Reviewer notes: demo_fdk_mar_compare offset_correction default (True) matches the old implicit
default; scripts that already called auto_set_recon_geometry are behavior-identical, any that did not now
get a correctly-sized recon grid. STAGED in the mbirjax_applications working tree (NOT committed) for
Greg's review + data validation in PyCharm.


Target repo: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_applications` (separate git repo).
Goal: update every caller of the now-REMOVED public preprocessing API to the new `get_sino_and_model`.
Cannot run the scripts (no scanner data) -> `py_compile` + pattern review; Greg validates on data.

## What changed in the library (the delta to migrate against)

- `<reader>.compute_sino_and_params(...)` -> RENAMED to private `_compute_sino_and_params`. Public
  replacement: `<reader>.get_sino_and_model(...)` returning a READY model (it runs
  construct -> set_params -> auto_set_recon_geometry internally):
  - nsi:       `sino, model = mjp.nsi.get_sino_and_model(dataset_dir, *, downsample_factor, subsample_view_factor, crop_pixels_*, auto_crop=False, verbose, offset_correction)`
  - pymbir:    `sino, model = mjp.pymbir.get_sino_and_model(filename, *, bh_correction=True, auto_crop=False)`
  - zeiss:     `sino, model = mjp.zeiss.get_sino_and_model(dataset_url, *, downsample_factor, subsample_view_factor, crop_pixels_*, alu_unit, bg_option, zinger_correction, auto_crop=False, verbose)`  (class AUTO-SELECTED: ultra->Parallel, else->Cone)
  - zeiss_tct: `sino, model, weights = mjp.zeiss_tct.get_sino_and_model(dataset_dir, *, crop_pixels_*, alu_unit, verbose)`  (RETURNS weights)
  All args after the first are KEYWORD-ONLY.
- `auto_crop_sino_conebeam(...)` -> REMOVED. Its job is now the `auto_crop=True` flag on the cone-beam
  readers (nsi/pymbir/zeiss); the detector-plane machinery is `detect_blank_margins` + `apply_detector_crop`.
- `est_crop_width` -> renamed `detect_blank_margins` (no app callers).
- The param dicts are no longer returned. If a script needs them, recover via
  `required, optional, regularization = model.get_all_params()`.

## Caller inventory (9 files)

NSI (6):
- `nsi/Lilly_recon.py:51` compute + `:57` **auto_crop_sino_conebeam** (conditional on `args.sino_cropping`)
- `nsi/Lilly_recon_ps.py:70` compute + `:76` **auto_crop_sino_conebeam** (same pattern)
- `nsi/demo_fdk_mar_compare.py:47` compute (uses `mj.preprocess.nsi.` full path)
- `nsi/demo_fdk_mbir_compare.py:46` compute
- `nsi/demo_plastic_metal.py:91` compute
- `nsi/demo_split_sino_recon.py:40` compute  (same script as the mbirjax experiments/ one migrated)
pymbir/vcls (2):
- `vcls/build_reference_object.py:46` pymbir.compute
- `vcls/demo_vcls_ornl_part.py:58` pymbir.compute
zeiss (1):
- `zeiss/demo_zeiss.py:190` zeiss.compute (4-tuple + `zeiss_metadata['scanner_type']` branch)

Plus stale docstring/comment references ("see mbirjax.preprocess.nsi.compute_sino_and_params") in
Lilly_recon, demo_fdk_*, demo_plastic_metal, demo_split_sino_recon -> update the name to get_sino_and_model.

## Migration patterns

1. **Standard** (most files): replace `sino, cbp, opt = <reader>.compute_sino_and_params(...)` +
   `ct_model = mj.<Model>(**cbp); ct_model.set_params(**opt); ct_model.auto_set_recon_geometry()` with
   `sino, ct_model = mjp.<reader>.get_sino_and_model(...)` (same kwargs, keyword form). KEEP the later
   tweaks (`set_params(sharpness=..., positivity_flag=..., partition_sequence=..., snr_db=...)`).
2. **auto_crop** (Lilly_recon, Lilly_recon_ps): the `if cropping: sino,... = auto_crop_sino_conebeam(...)`
   block folds into the reader call: `get_sino_and_model(..., auto_crop=bool(args.sino_cropping))`. Note:
   `sino = np.maximum(sino, 0.0)` then moves AFTER the call (harmless -- auto-crop detection runs on the
   pre-clip sino inside the reader, same as before; the model uses sino SHAPE, not values).
3. **zeiss scanner-type branch** (demo_zeiss): delete the `if zeiss_metadata['scanner_type']=='ultra':
   ParallelBeamModel else ConeBeamModel` branch AND `zeiss_metadata` -- get_sino_and_model auto-selects
   the class. Keep the post-tweaks.
4. **raw-param recovery** (per-file check, like the mbirjax experiments/ migration): if a script reads
   `cbp['angles']` / `cbp['sinogram_shape']` / `opt[...]` for anything other than building the model
   (e.g. vcls build_reference_object / demo_vcls short-scan subsets, by analogy to the nozzle bh script),
   recover with `required, optional, _ = ct_model.get_all_params()`.

## Behavior notes to flag per file

- **auto_set_recon_geometry fix**: scripts that built the model WITHOUT calling auto (some demos) now get a
  correctly-sized recon grid via get_sino_and_model -> recon output changes (intended fix). Lilly_recon /
  demo_zeiss already call auto, so those are behavior-identical.
- **weights**: only zeiss_tct returns weights; no app file uses that reader (no tct/ caller found), so no
  weights handling needed here.
- **np.maximum ordering** (Lilly_recon*): moves after the reader call; behavior-neutral (see pattern 2).

## Execution + verification

- Same fan-out approach used for the mbirjax experiments/ migration (1 agent per file, edit + `py_compile`,
  recover raw-params via get_all_params, flag behavior changes), then review the aggregate diff.
- Verify: `python -m py_compile <file>` for each (repo has no automated test suite for these demos).
- Greg reviews the diff and validates on real data before committing in the mbirjax_applications repo.

## Open decisions for Greg

- Keep the clean break (no deprecated alias) consistent with the library, OR temporarily add a
  `compute_sino_and_params` shim in mbirjax to ease this migration (NOT recommended -- library already
  hard-broke intentionally).
- Whether to migrate all 9 now or stage by directory (nsi first, then vcls, then zeiss).
