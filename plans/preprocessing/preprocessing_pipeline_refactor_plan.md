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

## Phase 3 — per-reader `get_sino_and_model`  (NSI DONE = the template; zeiss/zeiss_tct/pymbir next)

Template established on NSI (full suite green 251; adversarial review clean — 0 library/test/doc defects):

- `get_sino_and_model(dataset_dir, *, ..., auto_crop=False, ...)` -> `(sino, model)`: a thin wrapper =
  `_compute_sino_and_params` -> optional `mjp.utilities._auto_crop_sino` -> `mbirjax.build_model`.
- `compute_sino_and_params` -> `_compute_sino_and_params` (private). NORMALIZED return: it appends
  `required_params['geometry_type'] = str(mbirjax.ConeBeamModel)` so `build_model` resolves the class.
  Putting the class identity in `_compute` (not the wrapper) is what generalizes to Zeiss's
  parallel-vs-cone choice (each reader's `_compute` knows its geometry).
- `import mbirjax` added to nsi.py; `_auto_crop_sino` reached via `mjp.utilities._auto_crop_sino`
  (private, not star-exported); `build_model` via `mbirjax.build_model`.
- Behavior-equivalence verified: `get_sino_and_model(auto_crop=False)` == old
  `ConeBeamModel(**cbp)+set_params(**opt)+auto` (NSI optional has no recon_shape, so build_model's auto
  sizes the grid; no recon_shape pin).
- Tests: `TestGetSinoAndModel` mocks `_compute_sino_and_params` (no real dataset needed) -> builds a
  ready ConeBeamModel; auto_crop shrinks + stays consistent. Docs: NSI autofunction -> get_sino_and_model,
  stale save_preprocessing docstring ref -> get_all_params; nitpicky build clean.

Readers: NSI DONE, **pymbir DONE** (cone-only, no crop/downsample; `filename` arg; deferred imports).
**zeiss DONE** (parallel/cone; `_compute` resolves geometry_type from scanner_type mirroring
convert_zeiss: `'ultra'` -> parallel, else -> cone; return went 4-tuple -> normalized TRIPLE, dropping
`zeiss_metadata` which only held `scanner_type`, now encoded as the model class; Greg-decided
`'unknown'` -> CONE (behavior-preserving vs the old convert else-branch + the classify_zeiss_system
"assuming cone" warning; the earlier end-of-function `raise ValueError` was removed). Adversarial review
of zeiss+pymbir: behavior-equivalence (both branches), geometry_type registry match, zeiss_metadata drop,
pymbir deferred imports, blast radius all CLEAN; only real finding was the `'unknown'` behavior, now
resolved.  Remaining: **zeiss_tct** (translation; folds in the deferred Phase-2 config-crop + its
offset-reorder; auto_crop OFF by default pending real-TCT validation). `compute_weight` stays public for
zeiss_tct.

## Migration list — callers of the REMOVED public `compute_sino_and_params` (clean break, no alias)

Greg chose a hard rename (no deprecated alias), so every caller of a renamed reader's public
`compute_sino_and_params` must move to `get_sino_and_model(...) -> (sino, model)` (or the private
`_compute_sino_and_params` if the raw `(sino, required, optional)` triple is needed). No library/test
code is affected; the breaks are research scripts + the external mbirjax_applications repo.

- NSI (renamed): `experiments/sharding/collect_nsi_golden.py:82`,
  `experiments/split_sino_recon/demo_split_sino_recon.py:44`,
  `experiments/preprocessing/offset_correction.py:31,45`.
- pymbir (renamed): `experiments/bh_curve_fit/bh_curve_fit_experiment_nozzle_data.py:123,128`.
- zeiss / zeiss_tct: TBD when renamed.
- External (separate repo, per the earlier plan): mbirjax_applications `nsi/Lilly_recon_ps.py`,
  `nsi/demo_split_sino_recon.py`, `vcls/build_reference_object.py`.

### zeiss_tct DONE (last reader; translation)

- `get_sino_and_model(dataset_dir, *, ...) -> (sino, model, weights)` -- Greg-decided 3-tuple return, since
  zeiss_tct produces a data-specific `compute_weight` dark-boundary mask (kept public). No `auto_crop`
  param (translation auto-crop deferred pending real-TCT validation).
- Deferred Phase-2 config-crop folded in: `convert` now converts offsets to ALU BEFORE the crop and
  routes it through `apply_detector_crop`; recon_shape from the cropped shape. Probe: symmetric/default
  byte-identical (det_row_offset 0.15 unchanged), asymmetric shifts (-> -0.1). No downsample, so raw
  pitch = final pitch.
- recon_shape equivalence VERIFIED empirically: build_model PINS recon_shape (optional carries it from
  convert) and pinned == auto-sized (3,43,41) -- calc_tct_recon_params is aspect-idempotent -- so the pin
  matches the old auto flow.
- Adversarial review: 3 findings, ALL out-of-scope experiments/ callers; behavior-equivalence + crop
  reorder + recon_shape + weights + geometry_type + blast radius all CLEAN.

**PHASE 3 COMPLETE.** All four readers (nsi, pymbir, zeiss, zeiss_tct) converted to get_sino_and_model +
private _compute_sino_and_params, each adversarially reviewed clean. Full CPU suite green (257 passed).

## DRY pass (Greg-requested) -- A/B/D DONE; C (OLE) deferred

Fan-out scan (23 candidates) + adversarial judge rejected the over-abstraction traps (geometry_type
one-liners, downsample blocks with guard divergence, look-alikes with real behavioral differences).
Four distinct wins surfaced; Greg picked A + B + D now, C as a follow-up. All three new helpers live in
`preprocess/utilities.py` (exported), probe-confirmed byte-identical convert output, tests green:

- **A `apply_config_crop(...)`** -- scalar-in/out adapter around apply_detector_crop; collapses the 3x
  copy-pasted temp-dict marshaling in convert_nsi/convert_zeiss/convert_zeiss_tct. Keyword-only crop args
  (per the judge, to avoid a wide-positional footgun).
- **B `finalize_model(sino, required, optional, *, auto_crop=False)`** -- the shared get_sino_and_model
  tail (_auto_crop_sino + build_model); all four readers call it, which also RETIRES the awkward
  `mjp.utilities._auto_crop_sino` private reach-in (now contained inside utilities.py). Dropped the
  now-unused `import mbirjax` from nsi/pymbir/zeiss get_sino_and_model (kept in _compute for
  str(GeometryModel)).
- **D `to_alu(value, from_unit, alu_unit)` + `_ALU_UNIT_CONVERSION`** -- one shared unit table; replaced
  the duplicated dict + the `*= conv[u]/conv[alu]` idiom in convert_zeiss (5x) and convert_zeiss_tct (8x).
- New tests: `test_apply_config_crop_matches_formula`, `test_to_alu` (helpers covered indirectly by the
  reader tests too).

### C -- FOLLOW-UP (do NOT lose): shared Xradia/OLE reader module

Biggest single duplication and NOT yet done (Greg deferred it): **9 byte-identical OLE-reader helpers**
duplicated between `zeiss.py` and `zeiss_tct.py` -- `_check_read`, `_get_ole_data_type`, `_log_imported_data`,
`_read_ole_struct`, `_read_ole_value`, `_read_ole_arr`, `_read_ole_image`, `_read_ole_str`,
`get_index_in_list` -- plus near-identical `read_xrm` / `read_xrm_dir`. Extract to a shared private module
(e.g. `mbirjax/preprocess/_xradia_ole.py`) and import into both. KEEP `read_metadata` per-reader (zeiss
handles ReferenceData/MultiReferenceData, tct is scalar) -- real behavioral divergence, do not merge.
Medium effort, low behavior risk, BUT this layer has NO test coverage -- do it as its own focused change
with careful verification, not bundled into the API refactor.

## Phase 4 — docs + review  (IN PROGRESS)

- DONE: folded a concise one-call-API narrative into `usr_preprocess.rst` (get_sino_and_model pattern,
  class auto-selection, weights note, auto_crop); retired the proposal scaffolding -- reverted the
  `conf.py` `_proposals` sys.path and the `index.rst` `dev_preprocessing_api` toctree entry (both were
  committed referencing the UNcommitted proposal page, so a fresh checkout would have broken; now clean),
  and moved the untracked `_proposals/preprocessing_api.py` + `dev_preprocessing_api.rst` to scratchpad
  (Greg to permanently delete). Nitpicky docs build CLEAN. Full suite still 259 (docs-only changes).
- DONE: final holistic adversarial review of the whole refactor vs prerelease (~1700-line diff). It
  earned its keep -- found 3 REAL CROSS-PHASE findings no per-phase review could see (all fixed):
  1. [medium] save_preprocessing/load_preprocessing docstrings still showed the old reload
     `ConeBeamModel(**cone_beam_params)` + set_params -- which now TypeErrors because get_all_params
     injects geometry_type (not a constructor arg). Fixed -> `mbirjax.build_model(cone_beam_params,
     optional_params)` in both docstrings.
  2. [low] build_model applied ALL params with `no_warning=True`, which also disabled the param-NAME
     validation guard (a typo'd reader/round-trip key was silently accepted). Fixed: validate `optional`
     (no_warning=False -> typo raises ValueError, verified) and apply `regularization` separately with
     no_warning=True to suppress only the "directly setting regularization" advisory; copy_ct_model now
     passes reg as the separate arg (was merging into optional). 43 utilities+preprocessing tests green.
  3. [low] usr_preprocess.rst "auto_crop to a cone-beam reader" was imprecise (it is on nsi/pymbir/zeiss;
     zeiss can be parallel). Reworded to name the readers. (Refuted: a public/private-split "inconsistency"
     -- the split is intentional and consistent.)
  Docs build clean; full suite re-run after fixes.
- DONE: migrated the 9 LIVE experiments/ scripts (Greg's "live ones only" -- skipped the 2 ephemeral
  capture scripts collect_sibling_baseline.py + collect_nsi_golden.py). Fan-out (1 agent/script) ->
  get_sino_and_model; all 9 py_compile-clean; no compute_sino_and_params CALLS remain (only .md plan-doc
  text). CANNOT run them (no scanner data) -- py_compile + pattern-reviewed only; Greg validates on data.
  Notes: (a) 7 clean; the "old code never called auto_set_recon_geometry -> now correctly sized" recon-grid
  fix applies to demo_split_sino_recon, offset_correction, alignment_exp_ORNL (flagged, intended). (b) raw
  param-dict reads recovered via model.get_all_params() in bh_curve_fit, center_slice_zeiss, TCT_BGA. (c)
  the two TCT scripts kept `weights = None` (deliberate authorial choice preserved; the new data-specific
  mask is now available if wanted -- delete the `weights = None` line to opt in). TWO JUDGMENT CALLS for
  Greg: (1) exp_view_offset_gradient.py READER changed zeiss -> zeiss_tct (a real bug fix: it built a
  TranslationModel on TCT data but called the cone/parallel zeiss reader whose params lack
  translation_vectors -> was hard-broken); (2) center_slice_zeiss.py load_file/else branch restructured +
  get_all_params recovery for its pickle.dump (more involved -- review).

**PREPROCESSING REFACTOR COMPLETE** (Phases 1-4 + DRY). Follow-ups in flight: OLE dedup chip (C, separate
session) and the mbirjax_applications migration (separate repo).

## Follow-up (separate repo)

Migrate 3 mbirjax_applications scripts (nsi/Lilly_recon_ps.py, nsi/demo_split_sino_recon.py,
vcls/build_reference_object.py) once the library API lands.
