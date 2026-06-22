Performance Dashboard
=====================

MBIRJAX's reconstruction performance — run time, peak memory, and **correctness** — is tracked
automatically over time, on both CPU and GPU, by a companion project,
`mbirjax_metrics <https://github.com/gbuzzard/mbirjax_metrics>`__.  When a tracked branch changes, a
scheduled job re-measures it and publishes an interactive dashboard:

**Live dashboard:** https://gbuzzard.github.io/mbirjax_metrics/

The dashboard rebuilds and republishes automatically whenever new measurements are pushed, so that
link is always current.  This page is a guide to reading it; you do not need to run anything.

What it measures
----------------

For each tracked branch, the job runs MBIRJAX's reconstruction operators — the FBP filter, forward
projection, back projection, and the iterative VCD reconstruction — across a range of problem sizes
and device counts, on both CPU and GPU.  For every configuration it records:

- **run time** (the minimum over repeated trials),
- **peak memory**, and
- a numeric **fingerprint** of the output, used to detect correctness changes.

The dashboard is the view onto that growing time series.  Read it top to bottom.

Tiles
-----

A row of cards summarizing the currently-selected run, each split **CPU | GPU**:

- **configs measured** — how many (geometry × op × size × device-count) cells ran, and how many failed.
- **correctness** — how many configurations diverge from a trusted reference (see `Correctness`_ below).
- **performance regressions** — configurations whose time or memory regressed versus the reference.
- **tests failed** — unit-test failures from that commit.
- **run shown** — which commit you are viewing (branch · platform · date).

Click any card to drill into the specifics.

Correctness banner
------------------

If any branch's latest run produces a **different reconstruction** than its reference, a red banner at
the top of the page lists the offending configurations, and the browser tab gets a warning badge.
This is the loudest signal on the page: correctness is treated as more important than speed.  The
banner clears when the divergence goes away or is acknowledged as reviewed.

History
-------

Time-series panels (commit time on the horizontal axis) spanning both platforms and all branches:
**time** and **peak memory** at the largest size, plus a **performance-regressions** count.  Controls
select a **branch**, a **geometry group** (cone + parallel, or translation + multiaxis), and a
**device count**.  Click any point to load that run into the tiles and scaling views.

Scaling
-------

For the selected run and operation:

- **time vs size** and **memory vs size** (log–log), each with an "ideal" slope for reference.
- **speedup vs devices** and **per-device memory** — i.e. whether the work actually shards across GPUs.
- **compare against** overlays the same curves from ``main``, ``prerelease``, the prior run, or the best-ever.

Colors and marks
----------------

- **Color = platform:** GPU is blue, CPU is amber.
- **Line style = geometry** within each group (one solid, one dashed).
- **Red ✕** = a correctness fail (output mismatch versus the reference) · **red △** = failing tests ·
  **amber ring** = a GPU that ran hot · **amber disc** = a GPU that throttled (its timing is unreliable).

Correctness
-----------

Each output's fingerprint is checked against up to four references:

- the **prior run** on the same branch — did this commit change the result?
- the latest **main** — does this branch still match the canonical answer?
- **single- vs multi-device** within the same run — does sharding change the result?
- the **other platform** — do CPU and GPU agree?

A change beyond a small tolerance is flagged.  Reviewed or expected changes can be **acknowledged** so
they stop alerting, without erasing the record.

Running it yourself
-------------------

The dashboard is a single self-contained page generated from a YAML time series; no server is needed.
The measurement engine, the nightly harness, and the build script all live in the
`mbirjax_metrics <https://github.com/gbuzzard/mbirjax_metrics>`__ repository — see its ``README`` and
the ``action_scripts/`` and ``tooling/`` guides there for how runs are measured, gated, and scheduled,
and how to build the dashboard locally.
