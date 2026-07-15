# Preprocessing pipeline refactor — plan & design record

Branch: `refactor_preprocessing` (from `prerelease`). Local working doc (plans/ is untracked scratch).

## Goal

Replace the four-step footgun that every scanner reader documents today —

```python
sino, cone_beam_params, optional_params = compute_sino_and_params(dataset_dir)
model = mbirjax.ConeBeamModel(**cone_beam_params)
model.set_params(**optional_params)
model.auto_set_recon_geometry()          # forgetting this sizes the grid with DEFAULT pitches -> OOM
```

— with a single call that returns a ready-to-reconstruct model:

```python
sino, model = mbirjax.preprocess.nsi.get_sino_and_model(dataset_dir)
recon, recon_dict = model.recon(sino, weights=mbirjax.gen_weights(sino))
```

Design of record: `docs/source/dev_preprocessing_api.rst` (+ stub `docs/source/_proposals/preprocessing_api.py`),
published to `/depot/bouman/www/mbirjax/preprocessing/`.

## Phase 1 — model-param + crop foundation  (DONE, reviewed, full CPU suite green: 242 passed)

- `TomographyModel.get_all_params()` — the single source of truth. Returns `(required, optional,
  regularization)`: constructor args (cone view components unpacked from `view_params_array`) with
  `geometry_type=str(type(self))` in `required`; regularization = the six strength/meta knobs
  (`sigma_y/sigma_x/sigma_prox/snr_db/sharpness/auto_regularize_flag`); prior SHAPE params
  (`p/q/T/qggmrf_nbr_wts`) stay structural in `optional`.
- `build_model(required, optional=None, regularization=None)` — registry-resolved class from
  `required['geometry_type']`; construct -> set_params(no_warning) -> auto_set_recon_geometry; a pinned
  `recon_shape` is re-applied AFTER auto so it survives.
- `copy_ct_model` rewritten as a thin wrapper on the two above.
- `detect_blank_margins` (renamed from `est_crop_width`) — pure sinogram-intensity margin detection.
- `apply_detector_crop(required, optional, crop_top, crop_bottom, crop_left, crop_right)` — the single,
  **detector-plane-only, geometry-general** crop->geometry primitive (see rationale below).
- `_auto_crop_sino` (private) replaces the removed cone-only `auto_crop_sino_conebeam`: detect + slice +
  apply_detector_crop. Now geometry-general.

### KEY RATIONALE — why `apply_detector_crop` is detector-plane-only (do not "restore" recon_slice_offset)

The removed `auto_crop_sino_conebeam` did three things after cropping: update `sinogram_shape`, shift
`det_row_offset`/`det_channel_offset`, AND correct
`recon_slice_offset -= (crop_bottom-crop_top)/2 * delta_det_row / magnification`.

`apply_detector_crop` keeps the first two (universal, detector-plane) and **deliberately drops the
recon_slice_offset correction.** This is behavior-preserving, not a regression:

- Cone `auto_set_recon_geometry` (`cone_beam.py:268`) sets `recon_slice_offset = 0.5*(z_min+z_max)` from
  the helical z-shifts ALONE, ignoring `det_row_offset`, and overwrites it **unconditionally**
  (`cone_beam.py:270`, via set_params).
- It runs after the crop in every correct path: in the constructor, and in `build_model` unconditionally.
- Therefore any `recon_slice_offset` the crop wrote is thrown away before recon. The correction only ever
  survived in the no-auto path — which is exactly the stale-geometry OOM footgun the refactor removes.

**Git archaeology (load-bearing):** the correction has been *dead code since commit `0fbd350`
(v0.6.15)*. It was written at `021fef8` (v0.6.14), when `auto_set_recon_geometry` did NOT set
recon_slice_offset; `0fbd350` changed auto to overwrite it. This refactor does not touch `cone_beam.py`.

Confirmed by an 18-agent adversarial review (find -> independently verify): the strongest refutation
attempt ("asymmetric row crops make the dropped correction matter") was itself refuted on exactly the
grounds above. No implementation-logic defect was found; the five confirmed findings were all
test-discrimination / doc-nitpick and have been fixed (sentinel test that pins auto's overwrite, exact
deterministic detect_blank_margins values, object-position slice-direction check, real ParallelBeamModel
round-trip, and a `:func:`build_model`` nitpick-xref fix).

Geometry coverage of the offset formula: cone/translation/multiaxis carry `det_row_offset` +
`det_channel_offset`; parallel has `det_channel_offset` only (rows map straight to slices; its projector
ignores det_row_offset even though the param exists at 0.0). Guarded per-offset by a membership test.

## Phase 2 — unify the CONFIG crop through apply_detector_crop  (DONE for NSI + Zeiss; TCT deferred to Phase 3)

Scope decision (Greg): NSI + Zeiss now; Zeiss-TCT folded into its Phase-3 `get_sino_and_model` rewrite
(it crops before converting offsets to ALU and pins recon_shape from the cropped shape → needs an
internal reorder better done when restructuring that reader wholesale).

`convert_nsi_to_mbirjax_params` (nsi.py) and `convert_zeiss_to_mbirjax_params` (zeiss.py, both the
`ultra`/parallel and `versa`/cone branches) now route the config crop through `apply_detector_crop` via
a small temp-dict at RAW detector resolution, BEFORE the downsample rescale. Result:

- **Symmetric crops are byte-identical** to the old behavior (offset shift = 0; shape math identical).
  NSI forces top==bottom (nsi.py:78-81) + symmetric sides, so NSI is byte-identical on every real path.
- **Asymmetric top/bottom crops now shift `det_row_offset`** by `(crop_bottom−crop_top)/2 · raw δ_row`
  — the previously-uncompensated bug (Zeiss allows asymmetric; probe: NSI 0.1→−0.9, Zeiss 0.15→−0.1).
- Units: the shift uses the RAW pitch (crop is raw pixels applied before downsample), so it is
  independent of `downsample_factor`. Verified by probe + `TestConfigCropUnification` (6 tests).
- Blast radius clean: `convert_zeiss` is called only by the zeiss reader; `zeiss_tct` has its OWN
  `convert_zeiss_to_mbirjax_params` (zeiss_tct.py:274) and is untouched.

Adversarial review (9 agents): core claims (byte-identical, raw-pitch units, blast radius, offset sign)
all verified clean; two byte-identical nitpicks (numpy-vs-Python-int container, `−0.0→+0.0` sign bit)
refuted as value-identical + neutralized by `normalize_scalar`. One low-severity finding fixed: added a
guard assert at the top of `apply_detector_crop` (mirrors `crop_view_data`'s) so a crop ≥ dimension
raises on the geometry path instead of silently yielding a negative `sinogram_shape` (matters for direct
`convert_*` callers like `experiments/sharding/time_scan_to_sino.py`). Full CPU suite green (248 passed).

Known cosmetic wrinkle (moot): NSI computes `recon_slice_offset = −det_row_offset/mag` before the crop,
so it would be stale for a hypothetical asymmetric NSI crop — but NSI blocks asymmetric upstream and auto
overwrites recon_slice_offset, so it is unreachable/harmless.

## Phase 3 — per-reader `get_sino_and_model`

Model-class selection inside each reader (nsi/zeiss/zeiss_tct/pymbir); rename `compute_sino_and_params`
-> `_compute_sino_and_params` (private helper) with normalized return; `compute_weight` stays public for
zeiss_tct. auto_crop wired ON for cone (and parallel is a clean candidate), OFF by default for translation
(zeiss_tct) pending real-TCT validation.

## Phase 4 — docs + review

Fold the proposal into `usr_preprocess.rst`; full CPU suite; adversarial review on the diff.

## Follow-up (separate repo)

Migrate 3 mbirjax_applications scripts (nsi/Lilly_recon_ps.py, nsi/demo_split_sino_recon.py,
vcls/build_reference_object.py) once the library API lands.
