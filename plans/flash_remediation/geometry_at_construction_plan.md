# Implementation plan — geometry-at-construction unit fix (Option A) + axial cap + OOM memory-stats

**Status: PLAN ONLY — no code written yet.** Awaiting go-ahead. Discussion-first; stage only.

Origin: `plans/current_plans.md` §1 (flash remediation tail). Triggered by an OOM on the Lilly
`Connected_Autoinjector_Horizontal` scan where the cone-beam automatic `recon_shape` came out
`(1880, 1880, 4007)` instead of `(1880, 1880, ~1751)`.

## 1. Root cause (confirmed)

`source_detector_dist` / `source_iso_dist` are physical distances that are meaningless without a
length unit; the **detector pitch is that unit**. Both `ConeBeamModel` and `TranslationModel` take
SID/SDD at construction but NOT the pitches — pitches arrive later via `set_params` (the
preprocessors' `optional_params`). `TomographyModel.__init__` runs `auto_set_recon_geometry` at
construction ([tomography_model.py:122](../../mbirjax/tomography_model.py)) with **default unit
pitches (1.0)**. For cone, the per-end axial extension `excess ≈ v_top·(1/mag + R/SDD) − H_iso/2`
mixes a default-pixel `R` (= ½ the in-plane FoV width, ≈ 1/pitch too large) with real-mm SID/SDD →
the slab inflates ≈ 2.8×. `set_params(**optional_params)` applies the real pitches but does **not**
re-run the geometry, so `recon()` uses the stale inflated shape → OOM. Verified by staging the NSI
flow: after `ConeBeamModel(**cbp)` `recon_shape=(…,4212)`; after `set_params(**optional_params)`
still 4212; only an explicit `auto_set_recon_geometry()` gives 1842. This scan's real fan is a
normal 26° (R/SID = 0.23); the correct extension is a modest ×1.23.

Chosen fix = **Option A**: provide the detector geometry at *construction* so the construction-time
auto is correct, introducing **no** implicit `set_params` side-effects. (The rejected alternative —
eager auto-refresh inside `set_params` — was killed by an adversarial panel that found four
confirmed correctness regressions, all from that implicit refresh: `set_view_parameters` desync,
translation `voxel_row_aspect` non-idempotency, vcls sibling axial divergence, and a `delta_voxel`
contract regression.) Two adversarial design panels reviewed A; this plan folds in their deltas.

Scope: **cone and translation** (the two geometries whose auto reads pitches × SID/SDD). Parallel
/ multiaxis recon shapes are pitch-ratio-invariant → unaffected, and the read-only freeze is
geometry-aware so it does NOT touch them. Only cone has an axial extension (hence the cap). Cone
preprocessors in scope: **nsi, pymbir, and `zeiss.py`'s "versa" (cone) branch**; translation:
`zeiss_tct`. `zeiss.py`'s "ultra" branch is parallel and out of scope. Empirically the *OOM* bites
the real-mm-pitch cone flows that do not pin `recon_shape` (nsi; zeiss versa without the explicit
auto call) — pymbir pitches are hardcoded 1.0 (never inflates) and zeiss_tct already pins its
`recon_shape` — but the structural fix and the read-only contract apply to all the cone/translation
flows.

## 2. Design summary (five pieces)

1. **Geometry at construction** — add the detector-pitch/offset params as *keyword-only* constructor
   kwargs on cone (4) and translation (pitches only, 2), forwarded into `super().__init__` before
   the line-122 auto. Route the preprocessors to supply them in the constructor dict.
2. **Read-only, deprecation-warn** — after construction the four detector-geometry keys are frozen;
   setting them via `set_params` emits a `DeprecationWarning` ("set detector geometry at
   construction") and still applies the value (soft deprecation now, hard error next release). The
   one internal post-construction setter (`split_sino_recon`'s per-half `det_row_offset`) is routed
   through construction via a new `copy_ct_model` override, so nothing internal trips the warning.
3. **Fan-reach axial cap (cone only)** — `R_eff = min(support_radius, MAX_FAN_REACH·SID)`,
   `MAX_FAN_REACH = 1.0`, used in the extension; warn **unconditionally** (not gated on
   `no_warning`) when it binds. Backstops the residual "hand-built model with a forgotten pitch"
   case (bounds the slab to ≈2× base instead of OOM). Inert on real scans (binds only R/SID > 1).
4. **`get_memory_stats` on OOM** — dump per-device memory in `_handle_jax_error` before the existing
   guidance (broad try/except so diagnostics never mask the error).
5. **Docs + tests** — update preprocessor/`set_params` docstrings; add the cone+translation
   regression tests below.

## 3. Files to change (all in `mbirjax/` unless noted)

### Constructors / core

**`mbirjax/cone_beam.py`**
- `ConeBeamModel.__init__` (72–96): add keyword-only kwargs
  `*, delta_det_channel=1.0, delta_det_row=1.0, det_channel_offset=0.0, det_row_offset=0.0` and
  forward **all four** explicitly into the `super().__init__(...)` call (93–96). *(Blocker: the
  constructor forwards named args, not `**kwargs` — a missed forward makes the fix a silent no-op.)*
- Module constant `MAX_FAN_REACH = 1.0` (near the existing cone kernel-guard constants).
- `auto_set_recon_geometry` extension block (277–313): also read `source_iso_dist`; compute
  `R_eff = min(support_radius, MAX_FAN_REACH * source_iso_dist)`; use `R_eff` (not `support_radius`)
  in `z_per_v_far_side`; when `R_eff < support_radius`, emit a warning **regardless of
  `no_warning`** (gated only on `verbose>0`) reporting pre/post `num_slices_top`/`bot`,
  `det_row_offset`, and the `recon_shape` override path. Add a one-line comment that
  `split_sino_recon` is intentionally NOT capped (its overlap is bounded by the small
  `half_overlap_sino`).
- `split_sino_recon._recon_one_half` (~1487–1499): pass `det_row_offset=det_row_offset` into the
  `copy_ct_model(...)` call (construction) and **remove** the post-construction
  `model.set_params(det_row_offset=det_row_offset)` at line 1496.

**`mbirjax/translation_model.py`**
- `TranslationModel.__init__` (72–80): add keyword-only `*, delta_det_channel=1.0,
  delta_det_row=1.0` (pitches only — `calc_tct_recon_params` ignores offsets, so offset kwargs
  would be inert/misleading); forward both into `super().__init__(...)`.

**`mbirjax/tomography_model.py`**
- `__init__`: set `self._geometry_frozen = False` immediately after `super().__init__()` (119) and
  before the first `set_params` (120), with a comment that it must precede the first `set_params`;
  set `self._geometry_frozen = True` at the **end** of `__init__` (after `set_devices()` etc.).
- Module/class constant `_CONSTRUCTION_ONLY_GEOMETRY_PARAMS = frozenset({'delta_det_channel',
  'delta_det_row', 'det_channel_offset', 'det_row_offset'})`.
- **Geometry-aware freeze:** class attribute `_DETECTOR_GEOMETRY_CONSTRUCTION_ONLY = False` on
  `TomographyModel`, overridden `= True` on `ConeBeamModel` and `TranslationModel` only. The freeze
  applies ONLY to the SID/SDD geometries — parallel / multiaxis `recon_shape` is pitch-invariant
  (no stale-shape footgun) and setting a parallel pitch is a legitimate voxel-size change, so they
  stay settable. This also keeps the zeiss "ultra" (parallel) flow's post-construction
  `set_params(delta_det_*)` warning-free without touching `ParallelBeamModel`.
- `set_params` (2383): before/around the existing body, if `self._DETECTOR_GEOMETRY_CONSTRUCTION_ONLY`
  and any key in `_CONSTRUCTION_ONLY_GEOMETRY_PARAMS` is in `kwargs` and
  `getattr(self, '_geometry_frozen', False)` and not `no_warning`, emit a `DeprecationWarning`
  naming the construction-time path. Still apply the value (soft deprecation). `getattr(..., False)`
  guards the pre-init window.
- `_handle_jax_error` (~2961–2982): inside `if is_oom(traceback.format_exc()):`, before
  `log_oom_guidance`, add:
  `try: buf=io.StringIO(); mj.get_memory_stats(print_results=True, file=buf);
  self.logger.error("Device memory at OOM:\n"+buf.getvalue())` / `except Exception: pass`.
  (`io` and `import mbirjax as mj` already imported.)
- Docstrings: rewrite the `set_params` example at ~2625 (`set_params(delta_det_channel=100.0)`) to a
  construction-time example; note the four detector-geometry params are construction-only
  (deprecation window). Base `auto_set_recon_geometry` docstring (~2608): note geometry is fixed at
  construction.

**`mbirjax/utilities.py`**
- `copy_ct_model`: add an optional `det_row_offset=None` (and, for symmetry/future,
  `det_channel_offset=None`) override applied to `required_params` before
  `type(ct_model)(**required_params)` (1745), so a per-half detector offset is supplied at
  *construction*. This is what lets `split_sino_recon` stop setting it post-construction. Confirmed
  by panel: `get_required_param_names` enumerates KEYWORD_ONLY params, so the new pitch/offset
  constructor kwargs already flow through the copy/vcls reflection paths at construction (this
  actually *fixes* a latent `copy_ct_model` stale-shape bug).

### Preprocessors

**`mbirjax/preprocess/nsi.py`** — `convert_nsi_to_mbirjax_params` (423–427): move
`delta_det_channel`, `delta_det_row`, `det_channel_offset`, `det_row_offset` from `optional_params`
into `cone_beam_params`. **Keep** `delta_voxel`, `alu_*` in `optional_params`. Rewrite the
`compute_sino_and_params` Note (47–53): detector geometry is now a constructor parameter; the
explicit `auto_set_recon_geometry()` in the documented recipe is retained as an idempotent no-op for
back-compat with pre-change sidecars.

**`mbirjax/preprocess/pymbir.py`** — same move of the four keys (125–129) into the cone constructor
dict. **Keep `delta_voxel` in `optional_params`** — pymbir sets pitches to 1.0 and `delta_voxel` is a
deliberate independent voxel-size override; dropping it would mis-size the pymbir recon. Update
docstring.

**`mbirjax/preprocess/zeiss_tct.py`** — move `delta_det_channel`, `delta_det_row` (pitches only,
357–358) from `optional_params` into `translation_params`. **Keep** `recon_shape`, `delta_voxel`,
`voxel_row_aspect`, `det_row_offset`, `det_channel_offset`, `alu_*` in `optional_params` (the
documented flow has no explicit auto call, so the pinned `recon_shape` is what sizes the recon —
keeping it is a zero-output-change belt-and-suspenders). Update docstring.

**`mbirjax/preprocess/zeiss.py`** — two branches (`scanner_type`): **`else`/"versa" is CONE**
(`geometry_params` carries `source_detector_dist`/`source_iso_dist`, 409–410; no `recon_shape`
pinned) and has the identical NSI inflation bug. Move `delta_det_channel`, `delta_det_row`,
`det_channel_offset`, `det_row_offset` from `optional_params` (413–417) into `geometry_params` for
that branch. **Keep** `delta_voxel` (= `iso_pixel_pitch` = `delta_det_channel/mag`, redundant) and
`alu_*` in `optional_params`. The **`"ultra"` branch is PARALLEL** (no SID/SDD, `delta_voxel =
delta_det_channel`, pitch-invariant `recon_shape`) → NOT in scope; leave it unchanged.
`mbirjax_applications/zeiss/demo_zeiss.py` already calls `auto_set_recon_geometry()` after
`set_params` (line 203) so it is correct today; routing makes it robust and that explicit call
becomes an idempotent no-op. Update docstring.

**`mbirjax/preprocess/utilities.py`** — `auto_crop_sino_conebeam` (~892–912): **atomic with the
above** — change the reads of `optional_params['delta_det_row'/'delta_det_channel']` (909) to
`cone_beam_params[...]`, and the `+=` writes of `det_row_offset`/`det_channel_offset` (910–911) to
`cone_beam_params[...]`; assert those keys pre-exist in `cone_beam_params`; update the Returns
docstring (currently claims it updates `optional_params` offsets). *(Blocker: if offsets move but
the `+=` still targets `optional_params`, the crop offset correction is silently dropped and the
volume is mis-registered by ≈crop/2 voxels.)*

### Tests

**`tests/geometries/test_auto_geometry.py`**
- Construction-time shape (the fix): build a cone model with real mm pitches via the new kwargs;
  assert construction-time `recon_shape` == the value after an explicit `auto_set_recon_geometry()`
  (the 4007→~1751 case); assert a following `set_params(**optional_params)` leaves it unchanged.
- Fan-reach cap **binding**: forgotten-pitch cone (default 1.0 + mm SID/SDD); assert slices ≈ 2× base
  (not ~4007), no OOM, and the cap warning fires **at construction**.
- Fan-reach cap **non-binding**: R/SID < 1 with a large `det_row_offset` (Lilly-like); assert the
  full offset-driven extension is preserved and the cap does not bind (pins the prior fixed-cap
  defect fix).
- Read-only: `set_params(delta_det_channel=…)` post-construction raises `DeprecationWarning`; the
  construction-time kwarg does not.

**`tests/geometries/test_split_overlap.py`** — assert `split_sino_recon` still produces the correct
per-half `det_row_offset` via the `copy_ct_model` override (no post-construction geometry set); seam
metric unchanged vs the pre-change value.

**`tests/test_preprocessing.py`** — nsi routing + crop regression (**highest priority; currently no
coverage**): run `convert_nsi_to_mbirjax_params` → `auto_crop_sino_conebeam` →
`ConeBeamModel(**cone_beam_params)` and assert the constructed model's `det_channel_offset`/
`det_row_offset` equal the **crop-adjusted** values, that pitches are present in `cone_beam_params`,
and that `recon_shape` is the expected cropped size. This is the one seam whose failure is silent.

**`tests/geometries/test_translation_banded.py`** (or a dedicated translation geometry test) —
construct a `TranslationModel` with the new pitch kwargs; assert construction-time
`recon_shape`/`delta_voxel`/`voxel_row_aspect` match the pinned zeiss_tct values byte-identically;
assert the KEEP path (set_params of the pinned optional_params) is identical. Plus a forgotten-pitch
translation case: document/pin its behavior (bounded vs OOM-with-diagnostic — see Open items).

**`tests/test_utilities.py`** (or `test_recon_restart.py`) —
- OOM memory-stats safety: monkeypatch `get_memory_stats` to raise `KeyError`/`psutil` error;
  assert the try/except swallows it, `log_oom_guidance` still runs, and the original error re-raises.
- Reflection paths: `copy_ct_model` and `vcls._make_sibling` on a real-pitch cone model; assert the
  keyword-only pitch params are enumerated by `get_required_param_names` and forwarded (no KeyError),
  and `recon_shape` matches the parent (confirm `split_sino_recon` still overrides).
- Positional back-compat: `ConeBeamModel(shape, angles, sdd, sid, z_shifts, True)` binds
  `helical_z_shifts`/`use_curved_detector` positionally; a 7th positional arg raises (keyword-only
  enforced).

**`tests/test_pallas_kernels.py:593`** — the one existing test that sets `delta_det_row=8.0,
delta_det_channel=8.0` **post-construction**: change it to pass those pitches at *construction* (else
it now emits the deprecation warning). Test-hygiene fix required by the read-only path.

### Docs / tracking

- Inline docstrings above (nsi, pymbir, zeiss_tct, tomography_model).
- `docs/source/` parameter docs / `_static/new_model_template.py`: check whether any doc describes
  `delta_det_*` as `set_params`-able and add a construction-time note (low priority; verify during
  implementation).
- `plans/current_plans.md` §1: append a line noting the geometry-at-construction fix, the fan-reach
  cap, the read-only deprecation, and the OOM memory-stats (propose separately).
- `plans/flash_remediation/README.md` / `plans/README.md`: index this plan doc.
- **mbirjax_metrics `recon_shape` pin** — SEPARATE coordinated change (spawned as its own task/chip):
  pin the cone/translation cell shapes so the padding-policy change does not move the time/memory
  baselines.

## 4. Read-only freeze — mechanism detail

- Freeze flag lives only as an instance attribute; init `False` between
  `super().__init__()` and the first `set_params` in `TomographyModel.__init__`; set `True` at the
  very end of `__init__`. The guard also checks the class attribute
  `_DETECTOR_GEOMETRY_CONSTRUCTION_ONLY` (True only on cone/translation), so parallel/multiaxis are
  never frozen — the zeiss "ultra" (parallel) `set_params(delta_det_*)` stays warning-free and
  `ParallelBeamModel` needs no change.
- The construction path sets pitches at line 120 (before freeze); the construction-time
  `auto_set_recon_geometry` (122) writes only `recon_shape`/`delta_voxel`/`recon_slice_offset` — none
  of the four frozen keys — so it never trips the guard.
- After A's routing, the **only** in-tree post-construction writer of a frozen key is
  `split_sino_recon` (per-half `det_row_offset`), which moves to the `copy_ct_model` construction
  override → no internal deprecation warnings. `copy_ct_model` reconstructs via the constructor
  (`type(ct_model)(**required_params)`), and the four keys are now KEYWORD_ONLY constructor params
  → enumerated by `get_required_param_names` → routed to `required_params` (construction), NOT
  `other_params`, so `copy_ct_model`'s `set_params(**other_params)` never carries them.
- External callers doing `set_params(delta_det_…=…)` get a `DeprecationWarning`; behavior otherwise
  unchanged (soft deprecation). Harden to a hard error in a later release (follow-up item).

## 5. Validation

- Full CPU suite green: `python -m pytest -n auto tests/` (+ the new tests).
- Cluster repro (no script change to `mbirjax_applications/nsi/Lilly_recon.py`): confirm
  construction-time `recon_shape` ≈ `(1880,1880,1751)` not `(…,4007)`, and a feasible-downsample
  recon no longer OOMs; re-run the geometry probe.
- Metrics re-baseline check (panel watch item): confirm no seeded cone/translation baseline's
  construction-time shape or per-device peak shifts unexpectedly (synthetic cells use pitch 1.0 → no
  inflation; real-pitch flows go through the pinned-shape or explicit-auto path). Run the harness A/B
  before shipping; coordinate with the metrics `recon_shape`-pin change.

## 6. Open items / decisions

- **Forgotten-pitch translation has no cap** (no extension). A hand-built `TranslationModel` with mm
  SID/SDD + default pitch could size a large `recon_shape` and OOM at construction. Recommendation:
  rely on #3 (memory-stats-on-OOM diagnoses it) + a docstring note, rather than invent a
  translation-side threshold. Confirm the sizing behavior in the forgotten-pitch translation test.
- **Hard-error migration** for the deprecation window (next release).
- **`det_channel_offset` override in `copy_ct_model`**: add now for symmetry or only `det_row_offset`
  (the sole current need)? Lean: add both, use `det_row_offset`.
