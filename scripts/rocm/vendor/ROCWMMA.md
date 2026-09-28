Two rocWMMA checkouts are vendored here (MIT, pinned; excluded from this release tree):

- rocwmma-711/  rocm-7.1.1 (commit 1ab208f) - HAS native gfx1100 support (wave32).
  Used by rollout patch 0007 (the opt-in WMMA QSA arm). Fetch:
  git clone --depth 1 --branch rocm-7.1.1 https://github.com/ROCm/rocWMMA rocwmma-711

- rocwmma-src/  v0.8 (commit 1b59aaa) - CDNA-only; mma_sync does NOT compile for
  gfx1100 (measured; kept only as the recorded negative result). Do not use.

See PORTING.md 12g-12h for the probe verdicts and the precision contract.
