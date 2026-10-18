# Campaign evidence — IQ3_S decode on gfx1100

Snapshot of the rollout workspace's evidence tree (`~/projects/strata-amd`), pushed here so the
references in `GFX1100.md` resolve in the repository. The workspace itself is not a git repository;
this is a curated copy, not the live one.

- `iq3s-decode-campaign-20261018/` — pack fingerprint, upstream recheck, campaign log, and the
  execution backlog (`tasks-v2.md` current; `tasks-v1-superseded.md` kept for history).
- `results/` — the per-window records (`record-iq3s-*.json`): numbers, gates, verdicts.
- `iq3s-decode-timeline-20261018.md` — V01 boundary table, review outcome, calibration, revised
  recommendation. `iq3s-u01-audit-20261018.md` — component/premises audit and the nightly decision.
- `logs/` — the summary logs of the r31* windows only (per-cell logs stay local: 22 MB).
- `scripts/rocm/` — the r30*/r31* scripts that produced the windows.
- `TASKS-snapshot.md`, `STATUS-snapshot.md` — dated snapshots of the workspace's work queue and
  status log.

Not included: the 298 MB log tree, model weights, packs, or the pristine `Strata/` checkout.
