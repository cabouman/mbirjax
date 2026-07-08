"""
Analytic memory envelope: 'full' vs 'hybrid' vs 'sharded' (NO measurement).

Purpose
-------
Back-of-the-envelope for the hybrid keep/drop decision (v2 plan note 3 / 2-Drop).
Question (Greg): given a GPU budget of G GB and sino == recon == N^3 floats, what N
OOMs in 'full' (everything on one GPU) vs 'hybrid' (recon on CPU, sino+projections on
GPU) vs 'sharded' over n GPUs?  How much bigger a recon does hybrid buy on a single-GPU
box, and how does that compare to just adding GPUs?

Model
-----
One N^3 float32 volume is V(N) = BYTES_PER_ELEM * N^3 bytes (same for sino or recon,
since both are taken as N^3).  Per-device peak memory is modeled as a small integer
number of such volumes:

    full   peak = (A_RECON + A_SINO) * V(N)        # 6 recon copies + ~15 sino-side at peak
    hybrid peak = (A_RECON_RESIDENT + A_SINO) * V(N)  # recon offloaded to CPU; only a
                                                       # working batch resident on GPU
    sharded(n) peak/device = (A_RECON + A_SINO) * V(N) / n   # both axes shard 1/n

These are the load-bearing, UNCERTAIN assumptions -- edit them and re-run:
  * A_RECON   : recon-side multiplier in 'full' mode (Greg's "6R", an estimate).
  * A_SINO    : sino-side peak multiplier (error sino, weights, weighted-error, delta,
                + projection transients).  Greg: "about 15 at best."
  * A_RECON_RESIDENT : GPU-resident recon working set in hybrid (a voxel/cylinder batch),
                in volume units.  Small; ~1 by default.

The hybrid gain depends ENTIRELY on the recon fraction A_RECON / (A_RECON + A_SINO):
hybrid only removes the recon side, so if sino dominates, hybrid buys little.

Sanity cross-check
------------------
Prints the effective total multiplier implied by a measured anchor (default: leak-fixed
1-device-mesh 504^3 -> 6.9 GB).  If that disagrees with A_RECON + A_SINO, the coefficients
(or which code path they describe) need revisiting -- suspect the ruler.
"""

# ------------------------- CONFIG (edit and re-run) -------------------------
BYTES_PER_ELEM = 4          # float32

A_RECON = 7.0               # 'full'-mode recon-side multiplier (volumes)
A_SINO = 8.0               # sino-side peak multiplier (volumes)
A_RECON_RESIDENT = 1.0      # hybrid: GPU-resident recon working set (volumes)

GPU_BUDGETS_GB = [16, 24, 40, 48, 80, 141]   # cards to consider
SHARD_DEVICE_COUNTS = [2, 4, 8]              # 'sharded' comparison points

# Measured anchor for the cross-check (mode, N, GB).  Default = current 1-device path.
ANCHOR_N = 504
ANCHOR_GB = 6.9
ANCHOR_LABEL = "leak-fixed 1-device-mesh 504^3 const-weights"
# ---------------------------------------------------------------------------

GB = 1024 ** 3


def volume_bytes(n):
    return BYTES_PER_ELEM * n ** 3


def n_max(budget_gb, coeff):
    """Largest N with coeff * V(N) <= budget."""
    return (budget_gb * GB / (coeff * BYTES_PER_ELEM)) ** (1.0 / 3.0)


def main():
    full_coeff = A_RECON + A_SINO
    hybrid_coeff = A_RECON_RESIDENT + A_SINO

    print("=" * 78)
    print("Analytic memory envelope: full vs hybrid vs sharded")
    print("=" * 78)
    print(f"Assumptions: sino == recon == N^3 float{8*BYTES_PER_ELEM}")
    print(f"  full   peak coeff = A_RECON + A_SINO          = {full_coeff:.1f} volumes")
    print(f"  hybrid peak coeff = A_RECON_RESIDENT + A_SINO = {hybrid_coeff:.1f} volumes")
    print(f"  recon fraction (what hybrid removes) = {A_RECON/full_coeff:.1%}")
    print()

    # Cross-check the total coefficient against a measured anchor.
    c_eff = ANCHOR_GB * GB / (BYTES_PER_ELEM * ANCHOR_N ** 3)
    print(f"Cross-check vs measured anchor ({ANCHOR_LABEL}):")
    print(f"  {ANCHOR_GB} GB at N={ANCHOR_N}  ->  effective total = {c_eff:.1f} volumes")
    print(f"  (model 'full' total = {full_coeff:.1f}; if these disagree, the coeffs"
          f" describe a different code path -- revisit.)")
    print()

    hybrid_N_gain = (full_coeff / hybrid_coeff) ** (1.0 / 3.0)
    print(f"Hybrid vs full, single GPU:  N x{hybrid_N_gain:.3f}  "
          f"(volume x{hybrid_N_gain**3:.2f})  -- independent of G.")
    print()

    header = f"{'G (GB)':>8} | {'full N':>8} | {'hybrid N':>9} | " + \
             " | ".join(f"shard{n} N" for n in SHARD_DEVICE_COUNTS)
    print(header)
    print("-" * len(header))
    for g in GPU_BUDGETS_GB:
        nf = n_max(g, full_coeff)
        nh = n_max(g, hybrid_coeff)
        shard_cols = " | ".join(f"{nf * (n ** (1.0/3.0)):8.0f}" for n in SHARD_DEVICE_COUNTS)
        print(f"{g:>8} | {nf:8.0f} | {nh:9.0f} | {shard_cols}")
    print()
    print("Reading: hybrid is a one-time N x{:.2f} on a SINGLE GPU; each shard column is".format(hybrid_N_gain))
    print("the same 'full' working set spread over n GPUs (N grows as n^(1/3); volume as n).")
    print("Compare hybrid N against shard2 N to see whether 2 GPUs already beat hybrid.")


if __name__ == "__main__":
    main()
