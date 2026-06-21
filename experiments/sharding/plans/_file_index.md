# experiments/sharding/plans — file index

- `sharding_implementation_plan_v2.md` — **the current forward plan** (2026-06-02 on).  Placement
  architecture (`recon_placement`/`sino_placement`, the
  `move_cylinders_to_sino`/`sum_cylinders_to_recon` movement interface, uniform
  pixel-batched streaming) + the re-sequenced phases P1–P6 (Phase D re-opened on
  the new interface) + the device-config UX (`configure_devices`).  Read this for
  what comes next.
- `sharding_implementation_plan.md` — **completed-work record** (Phases
  0/A/B/F1/D/F2 case studies) + still-valid cross-cutting principles, verified
  hardware facts, and resolved open questions O1–O4.  Superseded for forward
  planning by `sharding_implementation_plan_v2.md`; read it for history and principles.
- `sharding_status.md` — short living status: current phase, phase tracker,
  verified hardware facts, open items.
- `correctness_gating_redesign.md` — **design note (2026-06-21, PLAN/for-review)** for the metrics
  harness (`mbirjax_metrics`): make correctness its own severity tier, add layered references (vs-main +
  cross-device + CPU↔GPU) to stop the prior-run ratchet, surface it as a dashboard banner + tab/favicon
  badge (no email), and clear reviewed divergences via an acknowledge-through-date file.  Phased plan §5.
