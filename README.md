# Strata HIP — AMD ROCm (gfx1100) port, tests under way

An AMD HIP backend for [Strata](https://github.com/Niko1221/Strata), built by
porting the pinned upstream CUDA sources mechanically (HIPIFY) and validating
every kernel against an independent reference on real hardware.

**Status: kernel-layer tests are under way and passing. Full-model inference is
not tested yet** — no weights have been downloaded, and no performance claims
are made. Everything below states exactly what is and is not proven.

Applied to upstream **v0.1.15** (`d551edf4…`); v0.1.16 exists upstream and is
**not** ported here. On v0.1.15 no `src/kernels/cuda/*.cu` file changed, so the
recorded kernel evidence applies to the tag as it stands.

## What is verified (on an AMD Radeon RX 7900 XTX, gfx1100)

- **Build**: the whole HIP tree compiles from the pinned upstream revision with
  zero compiler errors and zero hand-edited generated files.
- **Kernel parity suite**: each ported device kernel is compared on the card
  against an independent reference — a float64 host implementation, an exact
  bit-level model, or ggml's own CPU decoder cross-checked against gguf-py.
  Tolerances are derived per kernel from the reduction length (K·2⁻²⁴·Σ|terms|),
  never tuned to make a test pass. Quantized formats are compared bit-exactly.
- **Coverage**: every kernel on the default, non-speculative forward path has a
  passing test. Known gaps are listed as skips with declared output signatures —
  a skipped test exits nonzero and is never reported as a pass.

## What is NOT verified

- Full-model inference (weights not downloaded; the Stage-4 feasibility gate
  covers kernel-layer only, and says nothing about the shape of a forward pass).
- The QSA scorer's NVIDIA-tensor-core path (replaced by a correct portable
  reference path; a fast path is future work).
- Performance — nothing has been timed against anything.
- Speculative decoding and vision — out of scope for a first release.

## Reproducing

On Fedora with ROCm (tested: ROCm 7.x, clang from `rocml`, gfx1100):

```sh
./strata-rocm doctor          # read-only diagnostics
./strata-rocm prepare         # vendor the hash-pinned translator, create a venv
./strata-rocm hipify          # regenerate the HIP tree from the pinned revision
./strata-rocm build           # cmake + ninja, gfx1100 only, keep-going
./strata-rocm test --stage kernels   # the parity suite (needs the GPU)
./strata-rocm report          # redacted REPORT.md from recorded state
```

`./strata-rocm selftest` runs the tooling's own test suite (`tests/harness/`);
it injects a defect and proves the classifier calls it failed.

Design notes, the Fedora-ROCm traps, and every tolerance decision are recorded
in `PORTING.md`. Machine-readable state: `artifacts/rocm/state.json` and
`artifacts/rocm/results/`.

## How the port stays honest

- `Strata/` is a pristine checkout at a pinned tag; nothing inside it is edited.
  The HIP tree is *generated* (`build-hip/`, by a hash-pinned hipify-perl plus a
  declared, hash-recorded patch table) and never hand-edited.
- Exit codes distinguish pass (0), failure (1/4), blocked (3), skipped (5) and
  timeout (124). Skipped is not passed.
- Unsupported NVIDIA constructs (inline PTX, tensor-core MMA, cooperative
  groups, CUDA graphs…) are inventoried in `HIPIFY_REPORT.md` with the stage
  that must replace them, not silently compiled away.

## Licensing

The upstream revision this port is pinned to (v0.1.15) publishes no LICENSE
file, so that revision is local use only; upstream **v0.1.16 added an MIT
LICENSE** (with ggml, its font and the projection vector keeping their own).
This repository publishes the porting tooling, patches and tests; the generated
HIP tree is a derivative of upstream and is regenerable rather than distributed.
Model weights are not included and are not downloaded by any script here without
an explicit, recorded approval.
