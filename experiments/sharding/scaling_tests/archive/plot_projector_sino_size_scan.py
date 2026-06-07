"""
experiments/sharding/scaling_tests/plot_projector_sino_size_scan.py
──────────────────────────────────────────────────────────────────
Plot the single-device projector timing scans produced by
``projector_sino_size_scan.py`` (``results/projector_sino_size_scan_{cpu,gpu}.yaml``).

For each platform present, draws a 2x2 figure — one panel per sweep
(views / rows / channels / uniform) — showing forward and back time vs the swept
sinogram axis on log-log axes.  **Power-of-2 sizes are marked with a star (★)** so
you can see at a glance that they sit on the same smooth curve as the
non-powers-of-two: the channel-major layout fix removed the power-of-2
``num_det_channels`` cache-aliasing spike, so the channels panel (where the spike
lived) should now be smooth.

Writes ``results/projector_sino_size_scan_{plat}.png`` next to each YAML.

Run:  python experiments/sharding/scaling_tests/plot_projector_sino_size_scan.py
"""
import os

import yaml
import matplotlib
matplotlib.use("Agg")           # headless: save PNGs, no display
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SWEEPS = ["views", "rows", "channels", "uniform"]
COLORS = {"forward": "tab:blue", "back": "tab:orange"}


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _plot_series(ax, xs, ys, color, label):
    """Line through all points, with non-pow2 as circles and pow2 as stars."""
    pts = sorted(zip(xs, ys))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-", color=color, alpha=0.5, linewidth=1.4, label=label, zorder=1)
    npx = [x for x in xs if not _is_pow2(x)]
    npy = [y for x, y in zip(xs, ys) if not _is_pow2(x)]
    p2x = [x for x in xs if _is_pow2(x)]
    p2y = [y for x, y in zip(xs, ys) if _is_pow2(x)]
    ax.scatter(npx, npy, marker="o", s=26, color=color, zorder=2)
    ax.scatter(p2x, p2y, marker="*", s=170, color=color,
               edgecolor="black", linewidth=0.5, zorder=3)


def plot_platform(plat, data, out_png):
    base = data["base"]
    rows = [r for r in data["rows"] if "forward_ms" in r]
    by_sweep = {s: [r for r in rows if r["sweep"] == s] for s in SWEEPS}
    held = {
        "views":    f"rows={base[1]}, channels={base[2]}",
        "rows":     f"views={base[0]}, channels={base[2]}",
        "channels": f"views={base[0]}, rows={base[1]}",
        "uniform":  "n in each axis",
    }
    xlabel = {"views": "num_views", "rows": "num_det_rows",
              "channels": "num_det_channels", "uniform": "n  (n,n,n)"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, s in zip(axes.flat, SWEEPS):
        rs = by_sweep.get(s) or []
        if not rs:
            ax.set_visible(False)
            continue
        xs = [r["value"] for r in rs]
        _plot_series(ax, xs, [r["forward_ms"] for r in rs], COLORS["forward"], "forward")
        _plot_series(ax, xs, [r["back_ms"] for r in rs], COLORS["back"], "back")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel[s])
        ax.set_ylabel("time (ms)")
        ax.set_title(f"{s} sweep   ({held[s]} held)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"Single-device projector timing — {plat.upper()}  "
        f"({data.get('device', '')})\n"
        "★ = power-of-2 size — should lie on the curve (no power-of-2 cache aliasing)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    found = False
    for plat in ("cpu", "gpu"):
        path = os.path.join(RESULTS_DIR, f"projector_sino_size_scan_{plat}.yaml")
        if not os.path.exists(path):
            print(f"(skip {plat}: {path} not found)")
            continue
        found = True
        with open(path) as f:
            data = yaml.safe_load(f)
        plot_platform(plat, data,
                      os.path.join(RESULTS_DIR, f"projector_sino_size_scan_{plat}.png"))
    if not found:
        print("No projector_sino_size_scan_*.yaml found in results/; run the scan first.")
    print("Done.")


if __name__ == "__main__":
    main()
