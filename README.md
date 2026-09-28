# Strata HIP — AMD ROCm (gfx1100) port, tests under way

An AMD HIP backend for [Strata](https://github.com/Niko1221/Strata), built by
porting the pinned upstream CUDA sources mechanically (HIPIFY) and validating
every kernel against an independent reference on real hardware.

**Status: the kernel suite passes (40 passed / 0 failed / 2 declared skips), a
full model generates end to end, and the server has been tested at the model's
full 98,304-token context** — one-shot and through upstream's web UI + OpenAI
API on loopback 8081. The model-level correctness claim is deliberately
qualified; see `STATUS.md`. No performance claims are made.

Pinned to upstream tag **v0.1.16** (`c0107c3…`, MIT). The v0.1.15 → v0.1.16
range changed no `src/kernels/cuda/*.cu` file and no kernel header, so the
recorded per-kernel evidence applies to this tag; the full suite was re-run on
it (40/0/2).

## Fixes beyond the port itself (rollout patches)

- **Patch 0005 — gfx1100 shared-memory limit**: `fused_gr_read_multi` staged 80 KB of dynamic LDS (RTX allows it; RDNA3 caps a workgroup at 64 KB). Fixed by shrinking the tile constant; the tile loop is exact for any divisor, and the per-row accumulation is bitwise identical (reviewer-verified).
- **Patch 0006 — file-backed experts for IQ packs (the OOM fix)**: upstream's `--mmap-experts` only understood Q2_0-uniform `experts.bin`, so IQ models pinned a 47.9 GiB registered arena in RAM — which OOM-killed the machine repeatedly under a 96 GB desktop load. The patch maps the variable-layout IQ `experts.bin` directly: same bytes (A/B-validated: 16/16 identical greedy tokens vs the arena path), zero pinned RAM, experts live in the evictable page cache. Cost: the documented mmap trade (~1.9× cold slower decode); correctness unaffected.

## 98K context — tested

Serving with `--kv int8 --kv-resident 32768` (upstream's big-context profile) at the model's native **98,304-token window** was verified end to end on the card: server up, chat completion correct ("Name the capital of France in one word." → "Paris"), ~59 GiB of RAM still available with file-backed experts. The pre-fix configuration (pinned arena) OOM-crashed at this point — see `PORTING.md` §12d–12e for the full post-mortem.

## First inference (2026-09-28)

The full IQ3_S model loads — 46.84 GiB of experts mmapped from the sha256-verified
shards, PLE table from shard 2, MTP draft resident — and generates: *"The capital
of France is"* → *" Paris. The capital of Germany is Berlin"* (one-shot, greedy,
32.35 tok/s beside the user's own 8080 server), and via the OpenAI-compatible web
API: *"Name the capital of France in one word."* → **"Paris"**. The claim is
qualified by design: no same-weights logit reference has run yet — the per-kernel
parity suite below is the correctness evidence.

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

## Installing and running

### 0. Prerequisites

- Fedora (tested on 44) with AMD ROCm userspace installed (`hipcc`, `cmake`,
  `ninja`, ROCm clang). No sudo is needed anywhere below: everything lands in
  the project directory. **Do not install or replace your system ROCm.**
- An AMD RDNA3 GPU — the build is validated for **gfx1100** (RX 7900 XT/XTX/GRE)
  only and refuses other `CMAKE_HIP_ARCHITECTURES` by design.
- ~3 GB free for the build, plus room for model weights when you get there
  (the IQ3_S variant used here: ~84 GB of shards; experts are mmapped from the
  GGUF, not copied).

### 1. Build the port (no weights needed)

```sh
git clone --branch v0.1.16 https://github.com/Niko1221/Strata Strata   # pristine upstream checkout
./strata-rocm doctor          # read-only diagnostics; everything should be 'ready'
./strata-rocm prepare         # vendors the hash-pinned HIPIFY translator, creates .venv-rocm
./strata-rocm hipify          # generates build-hip/ from the pinned upstream revision
./strata-rocm build           # cmake + ninja, gfx1100 only, keep-going, complete logs
./strata-rocm test --stage kernels   # the parity suite on the GPU (40 pass / 2 declared skips)
./strata-rocm report          # redacted REPORT.md generated from recorded state
```

Every step is idempotent and recorded under `artifacts/rocm/`. Exit codes mean
what they say: 0 ok, 1/4 failure, 3 blocked, **5 skipped — never a pass**, 124
timeout.

### 2. Get the model

The engine reads the upstream project's own GGUF release
(`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, apache-2.0). Pick one variant's
two shards (IQ3_S is the one validated here) plus the mmproj file:

```sh
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  "mmproj-Qwen3.8-Flash-Next-BF16.gguf" --local-dir ~/models/qwen38-flash-next
```

Then verify the sha256s against `MODEL-DOWNLOAD-PROPOSAL.md` (it is the
checksum manifest), build the small pack, and prepare the MTP draft layer:

```sh
/usr/bin/python3 build-hip/hipify/tools/iq_pack.py \
  --gguf ~/models/qwen38-flash-next/IQ3_S/*-00001-of-00002.gguf --out packs/iq3_s
/usr/bin/python3 build-hip/hipify/tools/strata_tokenizer.py \
  --gguf ~/models/qwen38-flash-next/IQ3_S/*-00001-of-00002.gguf --out packs/iq3_s
python3 build-hip/hipify/tools/mtp_fetch.py fetch --out ~/models/mtp
STRATA_GGUF_PY=build-hip/llama.cpp/gguf-py python3 build-hip/hipify/tools/mtp_pack.py \
  --src ~/models/mtp --experts q2_0 --out ~/models/mtp/mtp-q2_0.gguf
STRATA_GGUF_PY=build-hip/llama.cpp/gguf-py python3 build-hip/hipify/tools/mtp_rt.py \
  --gguf ~/models/mtp/mtp-q2_0.gguf --out ~/models/mtp/rt
```

### 3. Run inference (one-shot)

```sh
./build-hip/cmake/strata --pack packs/iq3_s --tokens "760,6511,314,9338,369" \
  --native <shard1.gguf> --ple-gguf <shard2.gguf> \
  --spec 2 --mtp ~/models/mtp/rt --prefill 1024 \
  --expert-profile Strata/data/expert-profile.bin --expert-cache auto \
  --max-new 16 --greedy --max-context 4096
```

(`--tokens` are comma-separated token ids; with the shipped pack,
`760,6511,314,9338,369` is *"The capital of France is"*, which answers
*" Paris. The capital of Germany is Berlin"*.) `--max-context` caps VRAM use;
on a 24 GB card shared with another model server, expect modest rates.

### 4. Run the server (web UI + OpenAI-compatible API)

The server binds **loopback only**. Port **8081** is this port's target; never
point it at a port another service already uses. Add `--mmap-experts` when
something else holds RAM (the default expert arena pins ~47 GB; see
`PORTING.md` §12b):

```sh
# artifacts/rocm/serve/strata-hip.json is a ready-made config: edit the shard
# paths in it, then:
cd build-hip/hipify && ../../.venv-rocm/bin/python serve/server.py \
  --engine strata --config ../../artifacts/rocm/serve/strata-hip.json \
  --port 8081 --host 127.0.0.1
```

Then open <http://127.0.0.1:8081/> — a chat UI plus a live Monitor — or use the
OpenAI-compatible API:

```sh
curl http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Name the capital of France in one word."}],"max_tokens":128,"temperature":0}'
# -> "Paris"
```

`/health`, `/v1/models` and the Anthropic-style `/v1/messages` are also served.
Stop with Ctrl-C (the engine child dies with it). The first start loads ~47 GB
of weights and takes a minute or two.

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

## AMD GPUs on Windows

Upstream's Windows build is NVIDIA-only. AMD's HIP SDK for Windows does
support the RX 7900 XTX, but this port is **Linux-only for now**: its build
plumbing targets the Linux ROCm toolchain, and a Windows port needs on-hardware
iteration that has not happened (see `PORTING.md` §13 for the recorded
feasibility analysis). WSL2 for RDNA3 is not officially supported by AMD.

## Licensing

Upstream is **MIT licensed as of v0.1.16** (the ggml code, its font and the
projection vector keep their own licences). This repository publishes the
porting tooling, declared patches and tests under the same MIT terms; the
generated HIP tree is a derivative of the pristine upstream checkout and is
regenerable rather than distributed. Model weights are not included and are not
downloaded by any script here without an explicit, recorded approval.
