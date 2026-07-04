"""Host-arithmetic census of ragged-tail cost in the sparse-projector batching.

Pure numpy/python -- no jax, runs in seconds.  For realistic (recon size, view count,
device count, granularity) configurations, this computes the batch counts and ragged
remainders the projector drivers actually see on each batched axis, and prices four
tail-handling strategies:

  ragged   -- status quo: the remainder runs as a separate odd-shaped initial batch
              (extra inlined kernel in the driver HLO; zero wasted FLOPs).
  pad      -- pad the axis up to a multiple of the batch size (single kernel shape;
              wasted FLOPs on the pad; input pad copy).
  overlap  -- fixed-size windows with a clamped last start (single kernel shape;
              same wasted FLOPs as pad, as recompute; needs a mask on sum axes).
  balanced -- choose B* = ceil(n / ceil(n / B_max)) <= B_max so batches are near-equal;
              the residual (< num_batches items) is handled by a tiny overlap.
              Wasted FLOPs = residual/n; single kernel shape.

The driver call shapes mirror the current call sites (verified against the code
2026-07-03; see projector_batching_characterization.md in this directory):

  * driver VIEW axis: n = views_per_device = ceil(num_views / n_dev)
      - forward driver: views are vmapped and CONCATENATED (tail strategy cannot
        change values); back driver: views are SUMMED (overlap needs a mask).
  * driver PIXEL axis: n depends on the path:
      - VCD subset calls: n = subset = ceil(ror_pixels / num_subsets)
      - cone sharded gather-forward (n_dev > 1): the host pre-batches pixels at
        PIXEL_BATCH, so the driver sees exactly PIXEL_BATCH (plus one tail-size
        executable -- a full extra trace+compile, not just an inlined odd batch).

Run:  python batching_census.py   (no arguments; edit the config block below)
"""

import math

# ----------------------------------------------------------------------------------
# Config -- edit here
# ----------------------------------------------------------------------------------
# (N, num_views) pairs: cubic recons N^3 with a realistic span of view counts.
# Power-of-two N with views = N and 1.5N, plus common experimental (odd) view counts.
CONFIGS = [
    (256, 256), (256, 360), (256, 384),
    (512, 512), (512, 720), (512, 768),
    (1024, 1024), (1024, 1200), (1024, 1536),
    (2048, 2048), (2048, 3142),
]
DEVICE_COUNTS = [1, 2, 3, 4, 6, 8]

# Current hardwired knobs (tomography_model.py)
VIEW_BATCH = 128
PIXEL_BATCH = 2048

# Default granularity list and partition_sequence (_utils.py): subsets actually used
GRANULARITY = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PARTITION_SEQUENCE = [0, 2, 4, 6, 7]
SUBSETS_USED = sorted({GRANULARITY[i] for i in PARTITION_SEQUENCE})

# Report only axes whose worst strategy wastes at least this fraction (keeps the
# table focused on configs where strategies actually diverge).
WASTE_REPORT_THRESHOLD = 0.005


# ----------------------------------------------------------------------------------
# Strategy pricing for one batched axis of length n with target batch size B
# ----------------------------------------------------------------------------------
def price_axis(n, B):
    """Return dict of per-strategy (waste_fraction, num_kernel_shapes) for one axis.

    waste_fraction: extra FLOPs as a fraction of the n items' real work.
    num_kernel_shapes: distinct batch-body shapes the driver HLO contains for this
    axis (1 = uniform; 2 = uniform body + odd tail).
    """
    if n <= 0:
        raise ValueError('n must be positive')
    B = min(B, n)
    r = n % B                       # ragged remainder under fixed B
    pad = (B - r) % B               # items of pad/recompute under fixed B

    num_b = math.ceil(n / B)        # balanced: fewest batches with size <= B
    b_star = math.ceil(n / num_b)   # near-equal batch size
    residual = num_b * b_star - n   # < num_b items of overlap/recompute

    return {
        'n': n, 'B': B, 'remainder': r, 'b_star': b_star, 'num_batches': num_b,
        'ragged':   {'waste': 0.0,          'shapes': 1 if r == 0 else 2},
        'pad':      {'waste': pad / n,      'shapes': 1},
        'overlap':  {'waste': pad / n,      'shapes': 1},
        'balanced': {'waste': residual / n, 'shapes': 1},
    }


def views_per_device(num_views, n_dev):
    """Device-form per-shard view count (views padded up to a multiple of n_dev)."""
    return math.ceil(num_views / n_dev)


def ror_pixel_count(N):
    """Pixels inside the inscribed-ellipse RoR mask of an N x N recon plane.

    Exact count of the discrete mask in vcd_utils.get_2d_ror_mask (not pi/4 * N^2):
    the census must see the same subset sizes the code sees.
    """
    center = (N - 1) / 2.0
    radius = center
    count = 0
    for i in range(N):
        yy = ((i - center) / radius) ** 2
        # count j with ((j-center)/radius)^2 <= 1 - yy
        rem = 1.0 - yy
        if rem < 0:
            continue
        half = radius * math.sqrt(rem)
        j0 = math.ceil(center - half)
        j1 = math.floor(center + half)
        count += max(0, j1 - j0 + 1)
    return count


def subset_size(N, num_subsets):
    """VCD subset size: gen_pixel_partition pads so all subsets are exactly equal."""
    return math.ceil(ror_pixel_count(N) / num_subsets)


# ----------------------------------------------------------------------------------
# Census
# ----------------------------------------------------------------------------------
def fmt_pct(x):
    return f'{100 * x:6.2f}%'


def report_axis(label, n, B, rows):
    p = price_axis(n, B)
    worst = max(p['pad']['waste'], p['balanced']['waste'])
    rows.append((worst, label, p))


def main():
    rows = []

    # --- driver VIEW axis: one entry per (views, n_dev) ---
    for (_, num_views) in sorted({(0, v) for (_, v) in CONFIGS}):
        for n_dev in DEVICE_COUNTS:
            n = views_per_device(num_views, n_dev)
            report_axis(f'views  v={num_views:5d} ndev={n_dev}', n, VIEW_BATCH, rows)

    # --- driver PIXEL axis: VCD subsets per (N, num_subsets) ---
    for N in sorted({N for (N, _) in CONFIGS}):
        for s in SUBSETS_USED:
            n = subset_size(N, s)
            report_axis(f'pixels N={N:5d} subsets={s:3d}', n, PIXEL_BATCH, rows)

    # Print, worst waste first, filtered
    print(f'{"axis / config":34s} {"n":>8s} {"B":>5s} {"rem":>5s} '
          f'{"B*":>5s} {"pad/ovl":>8s} {"balanced":>9s} {"shapes r/p/o/b"}')
    shown = 0
    for worst, label, p in sorted(rows, reverse=True, key=lambda t: t[0]):
        if worst < WASTE_REPORT_THRESHOLD:
            continue
        shown += 1
        print(f'{label:34s} {p["n"]:8d} {p["B"]:5d} {p["remainder"]:5d} '
              f'{p["b_star"]:5d} {fmt_pct(p["pad"]["waste"]):>8s} '
              f'{fmt_pct(p["balanced"]["waste"]):>9s}   '
              f'{p["ragged"]["shapes"]}/{p["pad"]["shapes"]}'
              f'/{p["overlap"]["shapes"]}/{p["balanced"]["shapes"]}')
    print(f'\n[{shown} of {len(rows)} axes above the {100 * WASTE_REPORT_THRESHOLD:.1f}% '
          f'waste threshold; all others are ~free under every strategy]')

    # --- aggregate: distribution of worst-case waste ---
    for strat in ('pad', 'balanced'):
        waste = [p[strat]['waste'] for _, _, p in rows]
        waste.sort(reverse=True)
        k = len(waste)
        print(f'{strat:9s}: max {fmt_pct(waste[0])}   '
              f'p90 {fmt_pct(waste[k // 10])}   median {fmt_pct(waste[k // 2])}')

    ragged_axes = sum(1 for _, _, p in rows if p['ragged']['shapes'] == 2)
    print(f'ragged   : {ragged_axes}/{len(rows)} axes carry an odd tail today '
          f'(an extra inlined kernel each; zero wasted FLOPs)')

    # --- balanced-B* sanity: how far below B_max does balancing push the batch? ---
    shrink = [(p['B'] - p['b_star']) / p['B'] for _, _, p in rows if p['n'] > p['B']]
    if shrink:
        shrink.sort(reverse=True)
        print(f'balanced B* shrink vs B_max (batched axes only): '
              f'max {fmt_pct(shrink[0])}   median {fmt_pct(shrink[len(shrink) // 2])} '
              f'(memory bound is an upper bound, so shrink is always safe)')


if __name__ == '__main__':
    main()
