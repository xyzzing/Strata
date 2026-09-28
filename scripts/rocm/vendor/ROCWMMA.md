rocWMMA checkouts vendored here (MIT, pinned; excluded from this release tree):
- rocwmma-711/  rocm-7.1.1 (commit 1ab208f) - native gfx1100 (wave32). Used by patch 0007.
  git clone --depth 1 --branch rocm-7.1.1 https://github.com/ROCm/rocWMMA rocwmma-711
- rocwmma-src/  v0.8 (1b59aaa) - CDNA-only; mma_sync does NOT compile for gfx1100 (recorded negative).
See PORTING.md 12g-12h.
