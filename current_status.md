Current Status

Forward projection v3
- Added forward_project_pixel_batch_to_one_view_v3 in mbirjax/parallel_beam.py.
- It keeps the original projection math but uses a GEMV + safe scatter approach:
  - Computes A_chan_n weights per offset.
  - Applies a validity mask and clips indices to avoid out-of-bounds scatter.
  - Uses voxel_values.T @ A_chan_n to form a per-channel vector and scatters into sinogram_view.
  - Uses jax.lax.fori_loop to avoid Python-level unrolling.

Comparison plan (v1 vs v2 vs v3)
- Create a micro-benchmark that times each variant on representative sizes:
  - num_pixels ~ 2K, num_det_rows ~ 1-2K, num_det_channels ~ 1-2K, psf_radius ~ 1.
- Run with a fixed set of pixel_indices and angles to make results comparable.
- Record compile time and steady-state runtime for each variant.
- Verify numerical equivalence (max/mean absolute error) against v1 for one or two random seeds.


Notes for conversion from conda to uv/pixi while maintaining centralization of envs:

Conda: your current model (centralized prefix)

What you’re doing (symlinking ~/.conda to scratch) is basically: “all envs and caches live in one big writable place.” Conda supports that model directly via config (envs_dirs, pkgs_dirs) too.  ￼

Pros (for clusters):
	•	One place to monitor/clean.
	•	Easy to “nuke and rebuild” without touching home quota.
	•	Works well with shared scratch policies.

Cons:
	•	Still paying solver + conda activation overhead even if you’re mostly pip.

uv: you can centralize both cache and the venv location

uv defaults to a per-project .venv, which is the opposite of conda’s centralization. But uv supports two knobs that make it cluster-friendly:
	•	Centralize the download/build cache: UV_CACHE_DIR=/scratch/...  ￼
	•	Centralize (or redirect) the project environment directory: UV_PROJECT_ENVIRONMENT=/scratch/.../envs/<name>  ￼

So you can keep the “envs live on scratch” behavior while still using a modern, fast pip/venv tool.

Also: uv’s project workflow installs your repo editable by default (so it maps nicely to your current pip install -e . habit).  ￼

Practical shape (conceptually):
	•	UV_PROJECT_ENVIRONMENT=/scratch/$USER/uv-envs/mbirjax
	•	UV_CACHE_DIR=/scratch/$USER/uv-cache

Then uv sync / uv run uses that scratch-based environment for the repo.

Pros for you:
	•	Keeps the “centralized env on scratch” advantage.
	•	Much faster installs/resolution than classic conda workflows for Python-only stacks.
	•	Editable install behavior matches your dev flow.  ￼

Con: you’ll likely want a small wrapper (modulefile / shell function) so everyone gets the same env vars on the cluster.

Pixi: cache is easy to move; envs are usually per-project

Pixi is explicitly more “project-local” than conda (often creating an env alongside the project), which some HPC docs call out as the main conceptual difference.  ￼

But pixi does let you move/cache-on-scratch:
	•	PIXI_CACHE_DIR (or RATTLER_CACHE_DIR) to relocate the cache  ￼
	•	PIXI_HOME to relocate pixi’s global data  ￼

So pixi can help with the disk pressure problem, but its default “env near the project” model is less like your conda centralization unless you standardize where repos live (e.g., all repos on scratch) or you build conventions around it.

⸻

My recommendation given your constraints
	•	If you truly are Python + JAX + wheels, and conda is mostly “a convenient bucket on scratch”: uv is the cleanest replacement, because you can keep the bucket-on-scratch model via UV_PROJECT_ENVIRONMENT + UV_CACHE_DIR.  ￼
	•	If you want to stay in the conda ecosystem but modernize: pixi is great, but expect more “project-local envs” unless you adopt repo-location conventions; you can still move caches off home.  ￼

If you tell me how you currently pin JAX (CPU vs CUDA wheels, and whether CUDA comes from modules), I can suggest a concrete “team-friendly” layout for scratch paths and a one-command bootstrap that mirrors your current conda create && pip install -e . experience.