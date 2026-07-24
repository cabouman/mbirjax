"""
device_put_check.py  —  standalone (no mbirjax) probe for the multi-GPU
device_put corruption bug (jax issue #36308 / PR #36524) and for the
specific cross-device operations the planned threading-with-manual-collectives
sharding scheme depends on.

Background
----------
Reported bug: `jax.device_put(<device-resident jax array>, NamedSharding)`
silently zeros / corrupts the shard on non-default GPUs.  Documented
workaround: materialize to numpy first — `jax.device_put(np.asarray(x), sh)`.
The bug does NOT reproduce on CPU / simulated devices, so this must run on a
node with >= 2 real GPUs.

Verified results (2026-05-29, JAX 0.10.1):
  - 2x H100 80GB HBM3 : A,B,C,D all PASS  (direct d2d safe; no host bounce)
  - 2x L40S           : A,C FAIL (non-default shard silently zeroed),
                        B,D PASS (host-sourced / pre-placed assembly OK)
=> the transfer helper must pick d2d vs host-bounce from an empirical probe.

What this tests (all need >= 2 GPUs):
  A. TRIGGER   : device_put(jax_array, NamedSharding)        — the suspect path
  B. WORKAROUND: device_put(np.asarray(jax_array), NamedSharding)
  C. D2D COPY  : device_put(array_on_dev0, dev1)             — hot-loop all-gather
  D. ASSEMBLE  : make_array_from_single_device_arrays(...)   — Path-G output build

Each array carries known nonzero values so corruption-to-zero (or garbage) is
detectable.  Run:
    python device_put_check.py > device_put_check_out.txt 2>&1
in a JAX-with-GPU environment, then inspect the output file.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    import jaxlib
    jaxlib_ver = jaxlib.__version__
except Exception:
    jaxlib_ver = "?"

print("=" * 70)
print("device_put_check.py")
print("=" * 70)
print("jax     :", jax.__version__)
print("jaxlib  :", jaxlib_ver)
print("backend :", jax.default_backend())

all_devices = jax.devices()
print("devices :", all_devices)
try:
    gpus = jax.devices("gpu")
except RuntimeError:
    gpus = []
print("n gpus  :", len(gpus))
print()

if len(gpus) < 2:
    print("!! FEWER THAN 2 GPUs VISIBLE.")
    print("   The corruption bug does NOT reproduce on CPU / simulated devices,")
    print("   so this run cannot validate it.  Re-run on a node with >= 2 GPUs")
    print("   (and check CUDA_VISIBLE_DEVICES / the job's GPU allocation).")
    print("   Proceeding anyway on the first <=2 available devices as a smoke test.")
    devs = all_devices[:2]
else:
    devs = gpus[:2]

if len(devs) < 2:
    print("Only one device total; cannot run multi-device tests. Aborting.")
    raise SystemExit(0)

print("Using devices:", devs)
print()

results = {}

def verdict(name, ok, detail=""):
    results[name] = ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


mesh = Mesh(np.array(devs).reshape(2), ("d",))
sh = NamedSharding(mesh, P("d", None))

# Known, nonzero, per-row-distinct host data: row0=[1,2,3,4], row1=[11,12,13,14]
host = (np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
        + 10 * np.arange(2)[:, None] + 1.0)
expected = host.copy()


def shard_report(arr):
    """Return dict device->values for each addressable shard."""
    out = {}
    for s in arr.addressable_shards:
        out[str(s.device)] = np.asarray(s.data).ravel().tolist()
    return out


# ── Test A: TRIGGER — device_put of a device-resident jax array w/ NamedSharding
print("Test A — device_put(jax_array, NamedSharding)  [suspect path]")
try:
    jax_arr = jnp.asarray(host)                 # lives on default device
    arr = jax.device_put(jax_arr, sh)           # the reported-buggy call
    back = np.asarray(arr)
    ok = np.array_equal(back, expected)
    print("    per-shard:", shard_report(arr))
    if not ok:
        print("    readback :", back.tolist(), " expected:", expected.tolist())
    verdict("A_device_put_jaxarray_NamedSharding", ok,
            "" if ok else "corruption — non-default shard wrong")
except Exception as e:
    verdict("A_device_put_jaxarray_NamedSharding", False, f"exception: {e!r}")
print()

# ── Test B: WORKAROUND — materialize to numpy first
print("Test B — device_put(np.asarray(jax_array), NamedSharding)  [workaround]")
try:
    jax_arr = jnp.asarray(host)
    arr = jax.device_put(np.asarray(jax_arr), sh)
    back = np.asarray(arr)
    ok = np.array_equal(back, expected)
    print("    per-shard:", shard_report(arr))
    if not ok:
        print("    readback :", back.tolist(), " expected:", expected.tolist())
    verdict("B_device_put_numpy_NamedSharding", ok)
except Exception as e:
    verdict("B_device_put_numpy_NamedSharding", False, f"exception: {e!r}")
print()

# ── Test C: D2D — direct device-to-device single-device copy (hot-loop gather)
print("Test C — device_put(array_on_dev0, dev1)  [direct d2d, no host bounce]")
try:
    row = np.array([1., 2., 3., 4.], dtype=np.float32)
    x0 = jax.device_put(jnp.asarray(row), devs[0])
    x1 = jax.device_put(x0, devs[1])            # move dev0 -> dev1 directly
    back = np.asarray(x1)
    ok = np.array_equal(back, row)
    print("    on dev1  :", back.tolist(), " expected:", row.tolist())
    verdict("C_d2d_single_device_device_put", ok)
except Exception as e:
    verdict("C_d2d_single_device_device_put", False, f"exception: {e!r}")
print()

# ── Test D: ASSEMBLE — make_array_from_single_device_arrays (Path-G output)
print("Test D — make_array_from_single_device_arrays  [Path-G output assembly]")
try:
    parts = [jax.device_put(host[i:i + 1], devs[i]) for i in range(2)]
    assembled = jax.make_array_from_single_device_arrays(host.shape, sh, parts)
    back = np.asarray(assembled)
    ok = np.array_equal(back, expected)
    print("    per-shard:", shard_report(assembled))
    if not ok:
        print("    readback :", back.tolist(), " expected:", expected.tolist())
    verdict("D_make_array_from_single_device_arrays", ok)
except Exception as e:
    verdict("D_make_array_from_single_device_arrays", False, f"exception: {e!r}")
print()

# ── Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
n_fail = sum(1 for v in results.values() if not v)
print()
if n_fail == 0 and len(gpus) >= 2:
    print("All multi-GPU operations correct on this hardware/JAX version.")
    print("=> Direct device-to-device copies are safe; the threading scheme can")
    print("   use them without a host (numpy) bounce in the hot loop.")
elif n_fail and len(gpus) >= 2:
    print(f"{n_fail} test(s) FAILED on real GPUs — corruption present.")
    print("=> If Test A fails but B passes: device_put-of-jax-array is buggy;")
    print("   the numpy-materialize workaround is still required.")
    print("=> If Test C fails: direct d2d copies corrupt; the threading hot loop")
    print("   would need a host bounce, which undermines compute/transfer overlap.")
else:
    print("Inconclusive: ran without >= 2 real GPUs. Re-run on a GPU node.")
