# STATUS — Strata HIP/gfx1100 rollout

**working (kernel layer: fully tested; full model: first inference done, qualified claim)**

Last updated: 2026-09-30 (session 9) · Stage reached: **4 of 7**

> **The port has produced its first tokens.** Upstream **v0.1.16** (MIT) merged —
> zero device-kernel changes, all five declared patches apply. The kernel suite
> is **40 passed, 0 failed, 2 skipped** (both skips weight-gated; the MMQ IQ2
> gap S3.2b' closed this session via uniform-imatrix fixtures, validated on the
> card at 0.001 of bound). The full model (IQ3_S, sha256-verified shards) loads
> — 46.84 GiB of experts mapped, PLE table attached, MTP draft resident — and
> generates: *"The capital of France is" → " Paris. The capital of Germany is
> Berlin"*, and through upstream's web UI + OpenAI API on loopback **8081**:
> *"Name the capital of France in one word." → "Paris"*. Decode 32.35 tok/s
> beside the user's own 8080 server.
> **The claim is qualified, deliberately**: no same-weights logit reference
> exists (the 8080 server runs the 27B variant; a second ~84 GB-resident
> reference cannot coexist in 96 GB RAM). One real forward-path HIP defect was
> found and fixed on the way: `fused_gr_read_multi` staged 80 KB of dynamic
> LDS, past gfx1100's 64 KB limit — declared patch 0005, no math change. The
> serve engine's pinned 47 GB expert arena caused an OOM under coexistence; the
> kernel killed OUR process (the user's server was not kernel-killed but is
> currently down — restarting it is the user's action), and the serve config now
> defaults to `--mmap-experts`.

---

## Environment (measured, not assumed)

| | |
| --- | --- |
| OS / kernel | Fedora Linux 44 (KDE), `7.2.7-200.fc44.x86_64` |
| CPU / RAM | Ryzen 9 7900X, 24 threads, 96118 MiB |
| GPU | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M], `[1002:744c]`, **gfx1100**, 24 GiB VRAM |
| Driver | `amdgpu` loaded, `amdgpu.runpm=0` on the kernel command line (left alone) |
| ROCm / HIP | Fedora RPMs, HIP `7.1.52802-9999`, ROCm `7.1.1`, device libs for `gfx1100` present |
| Toolchain | cmake 4.3.0, ninja, ROCm clang 20 (device), GCC 16.2.1 (host), python 3.14.7, perl |
| Disk | free space on `/home` fluctuated 55–83 GiB during session 2 (measured thrice); the model download targets `/mnt/LINUXWINSHARE` instead (ext4, remounted **rw** by the user) |
| Existing service | llama.cpp on **:8080**, model `qwen27b-ista-iq3s` (IQ3_S, 27B), holding ~19.6 GiB VRAM — **preserved, never touched** |

## Revision history

- Session 1: upstream `main` at v0.1.13 (`b89c989a…`), chosen over the plan's
  inspected `6da1f667…` deliberately — the plan asks for newer upstream work to
  be inspected first, and there is **no HIP implementation upstream** (open PRs
  are all CUDA-side).
- Session 2: bumped to v0.1.15 (`d551edf…`) at the user's request; verified no
  device kernel changed; `iq_avx2.cpp` joined the CPU-expert trio.
- Session 3: v0.1.16 (above).

Every result file records the revision it was produced from. Six untracked
files in the checkout (`AGENTS.md`, `CLAUDE.md`,
`.agentic/checks.json`, `.agentic/install-state.json`, `.agentic/verify.py`,
`.agentic/workflow.md`) came from an external agent-onboarding hook. They are
not upstream source and do not affect the revision; the rollout's AGENTS.md
exception list now names them. `.agentic/checks.json` was configured to run
the rollout's real checks; it currently exits **5** (skipped coverage), which
is the honest result while two references are missing.

## Gates

| gate | status | evidence |
| --- | --- | --- |
| S0-hardware-hip-smoke | **passed** | `results/smoke-run.json`, `artifacts/rocm/logs/smoke-*.log` |
| S0-source-audit | **passed** | `HIPIFY_REPORT.md`, `results/hipify.json` (re-run on v0.1.15: 124 rewritten, 0 unsupported) |
| S1-harness-selfcheck | **passed** | `TASKS.md` § "Harness self-check"; now in standing, re-runnable form as `./strata-rocm selftest` (35 tests, includes an injected-defect case) |
| S1-cpu-reference | **passed with declared gaps** | IQ fixtures verified against gguf-py (9/10 types; Q2_0's reference is ggml's C decoder — gguf-py cannot decode it; recorded in `results/fixtures-reference.json`) |
| S2-kernel-suite-gfx1100 | **passed with declared gaps** | `results/test-kernels.json`: 40 passed, 0 failed, 2 skipped (v0.1.16) |
| S3-attention-prefill | **passed with declared gaps** | S3.1 (PTX replacement) tested; S3.2 GEMM tested; **all 8 MMQ types verified on the card** (S3.2b' closed session 3: uniform-imatrix fixtures, IQ2 worst at 0.001 of bound) |
| S3.4-s2-expert-grouped | **passed** | `testlogs/s2_expert_grouped_parity.log` |
| S3-feasibility | **passed** | every kernel on the default forward path tested; the `native_gdn` condition is closed (all three fused entry points tested); see below |
| S5-concurrency | partial | graph semantics **measured on HIP**, now including the per-step-pointer hazard (arguments frozen at capture — measured real) and concurrent replay (safe): `testlogs/core_graph_parity.log`. The repeated-request/stress suite itself is S5.3 work |
| S4-first-inference | **passed (qualified)** | `logs/first-inference-20260928.log` + `record-web-ui-serve`: first tokens + web UI; no same-weights logit reference yet |
| S5.1/S5.3 … S7-release | pending | next: rocWMMA fast path, engine benchmarks (S6.1), launch script (S7.1) |

## Revision

`Strata/` is at `c1e9033…` — upstream tag **v0.1.20** (tracked tree clean; upstream is now **MIT licensed**, which also
unblocks S0.6 for the tooling). Session 2 built against v0.1.15; session 3
verified the v0.1.15→v0.1.16 range touches **no `src/kernels/cuda/*.cu` and no
kernel headers** (only `src/kernels/cpu/expert_layout.cpp`), re-applied all
patches (a fifth, 0005, is new — see below), and re-ran the full suite.

## Session 6 — instrument audit, the Monitor on ROCm, and the first upstream PRs (2026-09-29)

The rollout's *measuring devices* were the work this session; the engine itself
was not touched. Everything below is recorded in `results/` and reproducible
with `./strata-rocm speedaudit`.

**The web UI's live tok/s was not a rate.** `GET /metrics -> live.tok_s` was
`generated / (now - first_token)`, a cumulative mean whose first sample is
`1/elapsed`. Replaying a known 25.0 tok/s pace through the real
`serve/server.py` (`speedaudit --probe`) made the number the Monitor displays
read **27,235.7 tok/s — 1,092x the engine's own rate** — and then undershoot for
the first second of every request. Declared patch **0009** makes it the rate
over the last 2 s, keeps the mean as `live.tok_s_mean`, and stops a request that
dies without a `DONE` line from recording the previous request's counters as its
`decode_tok_s`. Same probe after the patch: **max 28.0 tok/s against the 25.0
pace (1.12x)**. The engine's per-request figure — the "last request" card — was
never wrong, and the 10+ tok/s readings are real engine state, not a UI fault
(see the attribution below and `PORTING.md` §14d).

**The Monitor's hardware half spoke only NVML**, so on this ROCm box GPU load,
VRAM, temperature, power and PCIe were all empty and About said *"not readable
(NVML)"*; with no `psutil` in the venv the disk rate and core count were missing
too. Declared patch **0010** adds an amdgpu sysfs backend (edge-sensor
temperature, power, link, VRAM, utilisation) plus `/proc/diskstats` and
`/proc/cpuinfo` fallbacks. After the restart the live server reports
`gpu_util`, VRAM `23.0/24.0 GB`, `gpu_temp 49–59 °C`, `gpu_power 35–79 W` of a
339 W limit, `Gen4 x16`, a live disk rate, `cores 12` and
`gpu_name "Navi 31 [Radeon RX 7900 XT/XTX/GRE] (gfx1100)"`.

**Attribution of the low decode figures** (`speedaudit --ab`, three arms with
warmups): the earliest-peak one-shot measured **1.52 tok/s cold** (page cache
evicted; the 32.35 tok/s record was a warm-cache figure), the live config
measured **30.6 tok/s streamed and 35.5 tok/s non-streamed** on a 61/65-token
prompt, and 11,824- / 47,134-token prompts read at 115.6 / 98.0 tok/s with
decode 16.8 / 15.2 — the 47k arm crosses the `--kv-resident 32768` cliff, where
the prompt read collapses from ~480 to ~100 tok/s. The resident server, minutes
later, measured 10.7–11.4 tok/s for a 65-token prompt (85.6% / 76.3% hit rates
against the fresh engine's 96%) — same counters, different state. That
fresh-vs-resident gap is the open question this session leaves behind; it is
recorded, not explained.

**Upstream.** `Strata/` stays pristine; the two platform-independent patches
were proposed to `Niko1221/Strata` from `xyzzing` and apply to upstream `main`
(`f1b1d96`, Engine 0.1.21) unchanged:

| what | where | state |
| --- | --- | --- |
| the live tok/s rate + stale-counter fix (patch 0009) | PR **#104** | open; `serve.test_server` 28 tests OK, both new cases fail on unpatched code |
| the AMD telemetry backend (patch 0010) | PR **#105** | open; `serve.test_telemetry` 5 tests OK |
| the working port, its measured numbers and the defect list | issue **#106** | open |

The declared patches stay in `scripts/rocm/patches/` until a PR merges — if one
does, that patch is removed and the tree regenerated, so it can never
double-apply on the next upstream bump.

**README** — the measured-rates row now reads `36.7–38.1 tok/s (50 warm);
--spec-split live` with the two decode figures explained in the paragraph below
it, per the user's request.

## Session 7 — the stall-tail diagnosis and Task 1's stall-distribution pass (2026-09-29)

**The "10+ tok/s after all the patches" regression is identified: it is a stall
tail, not a level change.** The expert-panel verdict (llama.cpp-serving +
GPU-kernel), reconciled against the recorded corpus (98 pooled tune samples, one
binary): healthy decode is **41–53 tok/s — at reference speed**; the defect is
per-request blocking stalls injecting non-device time. The corpus spans
1.4–52.8 tok/s with `51.5, 2.4, 52.3` sandwiched inside one run
(`tune-20260929T015309Z`); per-step cost flips 53–56 ↔ 119–196 ms while
tok/step stays 1.6–2.2 in every regime — the stalls buy no drafting, they are
pure waiting. Kernel patches cannot move it (patch 0008's decode A/B was
measured **equal**), and the tuning verdicts (`--spec-split` +5.7 %,
`--prefill 8192` +28 %) cannot be resolved at 37× corpus variance — re-derive
or retract in Task 6. Panel claims corrected against measurements:
`/mnt/LINUXWINSHARE` is local **ext4 on `nvme0n1p5` (93 % full)**, not CIFS;
the "MTP/drafting collapse" branch is falsified by the tok/step derivation; NUMA
is falsified (single node, `numa_balancing=0`); the decisive pair's identical
KV stats (~90.4 % both sides) retire the kv-cliff as its cause. Plan and task
contracts: `tasks/plan.md`, `tasks/todo.md` (DDD ubiquitous language included).

**Task 1 ran — `speedaudit --regimes`, a new repeatable harness**
(`scripts/rocm/lib/speedaudit.py`). 2 warmup + 20 measured probes in one engine
session, predeclared rule `decode_tok_s < 25` = stall-contaminated:
**stall rate 0.0 — a clean baseline; the 10+ state did not reproduce today.**
Distribution 30.6–45.8 (median 43.0), healthy `ms/step` 45.25. The two mild
dips (probes 13/18 at 30.6/35.4 tok/s) moved `ms/step` 45→64/51 with **flat
counters** (majflt 0, read_bytes at the session median) — host-side shape, not
storage. Warmup 0 carried the real I/O signature: an 11.5 MB read burst + 22
major faults at t≈6.6 s (borrow/refill + page-cache warm). Per-probe
read_bytes fell 274 KB → 131 KB as the PLE row cache self-warmed the identical
probe — recorded as a probe-design artifact, not speed evidence. The pure-GPU
floor attempt failed cleanly (`session_replay_full: not captured with post
graphs`: native packs skip graph capture); a bounded no-pack retry is inside
the running Task 2/3 schedule. Evidence:
`results/speedaudit-regimes-20260929T071923Z.json`; selftest 57 green.

**Tasks 2+3 launched — attribution schedule**
(`scripts/rocm/lib/attribution.py`, interleaved round-robin per the tune
controller's founding lesson). Arms on the card: PLE row-cache 0/1M/4M, injected
`--ple-delay-us` 200/2000 (the slow-signature control), `STRATA_TG_FLUSH_US`
0/5/20/100 vs the default 2000 (the CUDA/WDDM-era claim "0 flushes run as fast
as 5 us ones" has never been re-measured on ROCm), `--pool-workers` 6/8 vs the
default 11 (11 workers + host loop on 12 cores), and `--no-prefill-borrow` —
each pass with per-request `/proc` majflt/read_bytes deltas, fincore residency
of the PLE GGUF + MTP sidecar, and one-shot `--stats` decomposition runs (the
serve path never prints `--stats`). Checkpointed after every pass under
`results/attribution-*.json`.

## Session 7b — speed analysis: decomposition, opportunity map, MMQ build (2026-09-29)

The user's objective: decode **and** prefill/TTFT; the acceptance metric is the
highest steady-state generation rate. Everything below is measured on the
DEPLOYED engine (fork `xyzzing/Strata` main `2ee6419`, engine 0.1.24, config
`artifacts/rocm/serve/strata-hip-0124.json`), not on the v0.1.20 pipeline binary.

**Decisive measurement** (new repeatable script `scripts/rocm/lib/gpu_decomp.py`,
`STRATA_DECODE_TIMING` + `STRATA_VERIFY_PROFILE`, record
`results/gpu-decomp-20260929T171204Z.json`): steady-state decode is **25.5
ms/token = 22.2 ms GPU-reach wait (87%)** + 0.75 CPU pool + 0.85 drafting + 0.47
commit/emit. The host-stall theory is retired for steady state: the headroom is
the per-layer GPU/hand-off chain and tok/window yield (1.75–1.8). The stage
report's `(gap)`/`hc0` columns are buggy (~1e14), so the residual is not yet
split between GPU-idle-on-flag and unnamed kernel work — the declared falsifier
is a no-doorbell window capture (speed-map D8).

**Two audits of the fork** (subagent, evidence-tagged, `tasks/speed-map.md`):
95 commits c1e9033..2ee6419 inventoried. Three findings changed the plan: the
deployed `build-hip` was configured with **`STRATA_PREFILL_MMQ=OFF`** (the
documented biggest prefill knob was compiled out); `STRATA_PF_IDX_BATCH` is a
doc-only no-op; the upstream "+14.6% prompt" tensor-core QSA work is CUDA-only
(dead on gfx1100; rocWMMA port remains backlog).

**Build + correctness gate**: `build-hip-mmq/` (MMQ ON, tests ON; deployed
`build-hip/` untouched for rollback) built after two declared CMake fixes —
`native_mmvq_multi` guarded (`bench/micro/native_mmvq_multi.cpp` exists in no
commit of this repo or upstream) and `qsa_prompt_attn_parity` scoped to CUDA
(it linked `CUDA::cudart` unconditionally). CTest: **36 passed**, including
**`hip_prefill_mmq_parity`** (the MMQ gate), 1 documented SKIP
(`hip_prefill_hipblaslt_gemm`), 2 fixture-blocked failures (`expert_parity`,
`pool_test` need a Q2_0 `pack/full/experts.bin`; only IQ3_S packs exist here —
reported as failures, not passes). The shipped hipBLASLt table is accepted at
runtime: `prefill gemm: hipBLASLt tuning enabled (30 rows, gfx1100, version
100100)` — open question resolved.

**A/B verdicts** (frozen contract: 5 interleaved pairs × 2 prompt slots,
min-gain 5%, stall-tag, 3% protected cap, holdout — `results/record-speed-ab-batch-20260929.json`):

- **Promoted — decode +38.0%:** dropping `--spec-split` (43.0→59.4 tok/s,
  holdout 59.5). The old "+5.7% with --spec-split" claim is **retracted and
  inverted** (it cost ~27% decode on 0.1.24).
- **Promoted — prompt +112.1% (long-slot fresh prefill 416→971 tok/s):** the
  MMQ build (`build-hip-mmq`, `STRATA_PREFILL_MMQ=ON`) + `STRATA_HIPBLASLT_TUNING`
  + `STRATA_PF_EMBED_BATCH=1` + `STRATA_PREFILL_RING=96`; decode −0.8% within
  the protected cap. Promotion = config `exe` + a new config `env` block
  (`serve/server.py` `child_env` merges `cfg["env"]` — survives restarts).
- **Retracted — `--prefill 8192` +28%:** exactly 0.0% at 1.6k AND at 10,477
  fresh tokens (slot-1 619 vs 619 tok/s).
- **Rejected:** `STRATA_VERIFY_DEVICE_PLAN=1` (−4.1% decode); D1c native-subset
  flags (+4.1%, below the frozen 5% bar — recorded, the rule was not bent).
- **Unavailable:** the fork's `--mmap-experts`/`--resident-cpu-experts` pair is
  refused for native (IQ) packs (generate.cpp:1758-1760) — structural.
- **Headline confirmation** (one paired run vs the original configuration):
  **decode +38.3% (43.0→59.5 tok/s, holdout 59.5), prompt +120.2%**.

The live config carries the promotion (`strata-hip-0124.json`; rollback:
`.bak` + the untouched `build-hip/strata`). The resident `strata-8081` runs the
winner state, smoke-verified. Fork verification gate (`python3
.agentic/verify.py`, configured this session): **PASS** — CTest 36 green incl.
`hip_prefill_mmq_parity`, 1 documented SKIP, 2 declared fixture-blocked failures
(`expert_parity`, `pool_test` need a Q2_0 `pack/full` fixture that does not exist
on this host; weights immutable, downloads forbidden).

## What was run, and what came out (session 3, on v0.1.16)

```
./strata-rocm doctor        ready 14  blocked 0  unverified 0   (outside the sandbox)
./strata-rocm hipify        266 tracked files, 125 rewritten, 0 unsupported construct classes
./strata-rocm build         ok, 44 executables, 0 compiler errors, 5 declared patches applied
./strata-rocm test --stage kernels
                            40 passed, 0 failed, 2 skipped  -> exit 5 (skipped is not a pass)
./strata-rocm report        regenerated from the recorded state
FIRST INFERENCE             "The capital of France is" -> " Paris. The capital of Germany is Berlin"
                            (one-shot, greedy, spec 2; 32.35 tok/s decode, TTFT 524 ms; log:
                             artifacts/rocm/logs/first-inference-20260928.log)
WEB UI                      upstream serve/server.py + our HIP engine on loopback 8081:
                            /health ok, /v1/models, chat completion "Name the capital of
                            France in one word." -> "Paris" (finish=stop)
```

Session-3 events recorded in `results/`: `record-s4-artifacts-verified.json`
(all model files sha256-verified against the manifest), `record-first-inference`,
`record-web-ui-serve`, `record-oom-incident` (serve engine's pinned expert arena
triggered global memory pressure; the kernel killed OUR strata process; the
user's 8080 server was not kernel-killed but is currently down — restarting it
is the user's action; the serve config now defaults to `--mmap-experts`),
`record-revision-bump` (v0.1.15→v0.1.16), and the S3.2b' + S5.2 additions below.

The smoke test that established the hardware gate:

```
device_name=AMD Radeon RX 7900 XTX
gcn_arch=gfx1100
warp_size=32                 (RDNA3 wave32 default, not 64)
int_exact_mismatches=0
float_worst_rel_error=0.000e+00
SMOKE_RESULT=pass
```

## What is verified

- HIP can allocate, launch and compute on this card, with a bit-exact integer
  result and an exact float result.
- The CUDA→HIP translation is reproducible from the pinned revision: one
  hash-pinned translator, a declared patch table, before-images kept, no
  hand-edited generated file.
- 39 suites agree with their independent references on `gfx1100` (3 more are
  declared skips), covering dequantization, elementwise, normalization, RoPE,
  routing, GEMV variants, KV quantized streaming, quantized activation,
  sampling, and the QSA, GDN (including all three fused entry points), GR and
  shared-expert paths.
- **The QSA scorer's NVIDIA tensor-core PTX has been replaced by a portable
  reference path and independently tested** (`PORTING.md` §6b): 7 cases, 0
  failures, bit-exact against a host model of the same arithmetic, deviation
  1.1e-4 of the row magnitude against a per-row derived bound, no writes outside
  `[0, n_kv)`, invalid-count guard verified. With that, the tree has **no
  unsupported translation units left**.
- **The prefill GEMM runs on hipBLAS and is independently verified**
  (`PORTING.md` §6c, `tests/rocm/prefill_gemm_parity.cpp`): 7 cases, 0 failures,
  covering the transpose trap (T, N, K all distinct), the `ldy` row stride,
  `beta=1` accumulation, both BF16 and FP16 paths, a K=512 reduction and
  single-token decode -- all inside a derived summation bound, worst case 0.107
  of it. The rest of `src/prefill` (ggml's MMQ kernels, and the 1421-line host
  `prefill.cpp`) **compiles for gfx1100 but is untested**.
- **They pass while the user's model server holds 18.7 GiB of the card's 24 GiB**
  — that is the real coexistence condition, not an idle GPU.

## Session 8 — upstream 0.1.27 merge, the Monitor's AMD readings, local rollout (2026-09-30)

**What works now (all measured or gate-green this session):**

- **Engine 0.1.27 is live on 8081.** Upstream merged (`9e67295`…`4da9c59`); both
  builds green; CTest **38/38** with the declared exclusions unchanged and one
  documented SKIP; `.agentic/verify.py` PASS. The version A/B against the 0.1.24
  winner snapshot (`tune-20260930T015729Z.json`, frozen contract) reads as
  **equivalence: decode 59.9 vs 59.95 tok/s (±0.1%), prompt 531.6 vs 526.9
  (+0.9% for 0.1.27)** — the merge costs nothing. Rollback snapshots:
  `artifacts/rocm/rollback/strata-0.1.24-{mmq,hip}`.
- **The Monitor shows the AMD card's readings** on the live service: name
  `Navi 31 [Radeon RX 7900 XT/XTX/GRE] (gfx1100)`, util/VRAM/temp/power/PCIe
  live on `/metrics`, on the multi-card Monitor (`f7262d0`) with the `25e518d`
  sysfs backend rebased. The card is the one at the model's HIP device (PCI-slot
  match, not "the index-th amdgpu" — an iGPU would take that), and AMD readings
  win over NVML whenever the model runs on AMD (`backend` key or the engine's
  own `libamdhip64` linkage). `serve/test_telemetry.py` 8/8.
- **The rollout config is coherent:** `strata-hip-0124.json` on 0.1.27 with
  `backend: hip`, `STRATA_PF_EMBED_BATCH` retired (unconditional upstream),
  hipBLASLt 100100 table + `STRATA_PREFILL_RING=96` kept.

**What is unverified:** the 0.1.27 features not exercised here — the CJK draft
subset (`data/draft_vocab.bin` not installed: model sidecars are immutable,
that install is your call), `--kv k8v4`, `STRATA_MTP_BATCH`, `--mmap-experts` on
native IQ packs (de159b5 unblocks the D1 candidate; not yet measured).

**Harness note (PORTING §17):** `restore_resident` was keeping a stale engine
alive through two false "restarted" claims (wrong unit stopped; transient-unit
"already loaded"; a self-matching `pgrep | grep` check). Fixed and the serving
state re-verified against the process command line. No measurement was affected
(the A/B drives its own engines).

**Records:** `artifacts/rocm/results/record-upstream-0127-20260930.json`,
`tune-20260930T015729Z.json`, `speedaudit-regimes-*.json`.

## A note on the existing model server

During this session the unit `llama-model@qwen27b.service` (a *user* unit,
`Restart=on-failure`, `NRestarts=0`) was stopped at 15:31 local and started
again at 15:41 local. The journal records a clean `Stopped …` line, not a
failure or a signal, and the unit's restart counter is zero, so this was a
requested stop-and-start by something outside this session — not a crash, and
not the rollout: no command run here stops, starts or reconfigures that service.
It is back up, healthy, and serving `qwen27b-ista-iq3s` on :8080.
The one measurement this distorted: a `strata-rocm run` check taken inside that
ten-minute gap read 23.7 GiB free VRAM; the 18.7 GiB reading above is the real
one.


- **The runtime core runs on the card.** `strata-device --selftest` reports
  `AMD Radeon RX 7900 XTX / architecture gfx1100 (HIP capability 11.0)`, prints
  the planner's memory plan against the card's real free VRAM, allocates a
  64 MiB poisoned arena, honours 256- and 4096-byte alignment, and refuses an
  over-allocation. The architecture gate is a validated-target list
  (`PORTING.md` §6d), not a widened constant.
- **HIP graph semantics were measured, not assumed** — the plan's Stage 5
  warning, answered with numbers: capture/instantiate/launch works, and a replay
  **re-reads its input buffer** rather than replaying a snapshot (3 replays, 0
  stale values). That is the assumption whose failure would silently corrupt an
  engine, and it holds on gfx1100.


- **The `strata` engine binary builds, links and runs on this machine.** All eight
  `strata_engine` sources, `strata_spec`, the CPU expert kernels and the whole
  prefill stack are in the link (26 executables). On the card it parses its
  arguments and refuses cleanly at the first missing artifact
  (`cannot open <pack>/index.txt`, exit 1). **That is not a result** — no token was
  produced — it is evidence that the startup path is intact.


- **ggml's CPU backend builds and links** (S2.7c), through upstream's own
  `add_subdirectory` on the pinned llama.cpp. `native_experts_available()` and 26
  native-expert symbols are in the `strata` binary, so an IQ expert weight means
  what it means in llama.cpp rather than what a reimplementation assumes.
- **A suspected correctness hole turned out to be a compile-time split.** The
  missing `mmq-instance-*.cu` files do not break linkage: `moe_mmq.cu` instantiates
  all eight `mul_mat_q_case<T>` templates itself. `PORTING.md` §6f records both the
  finding and the lesson — *"we skipped eight files" was the right thing to check
  and the wrong thing to assume.*
- **`native_expert_parity` builds for the first time**, and is **skipped for a
  missing input, not a missing reference**. The runner gained per-test argv
  (`tests/rocm/testargs.json`) because its blanket `--selftest` was being read as
  a GGUF filename and aborting the binary.


- **The MMQ prompt path runs on gfx1100, bit-exact for Q2_0** — after this session
  found and fixed a real defect in Strata's own HIP shim: it computed the device
  arch id with CUDA's `100*major + 10*minor` encoding (1110 for gfx1100), while
  ggml keys MMQ instance selection on an id parsed from the GCN string. Every
  launch aborted with *"no device code compatible with HIP arch 1300"*. The path
  **links and dispatches either way** — only executing it could show this.
- **A previous checkpoint's conclusion was wrong and is corrected.** The missing
  `mmq-instance-*.cu` files were recorded as "resolved by inspection". They are
  genuinely required; the new test failed to link and said so.


- **The MMQ prompt path is verified for six of the eight types it claims**,
  including **IQ3_S, which is what the model's weights are**: Q2_0 bit-exact, the
  five buildable i-quants ≤2.1e-05 worst error at ~0.001 of a derived bound. The
  oracle is ggml's own reference quantizer/dequantizer — the scalar definition of
  each format — and the activations are chosen so they quantize exactly, so the
  reference needs no decode of ggml's transposed q8_1 layout.


- **All ten IQ/QK quantization formats are verified** — previously the largest
  single gap. `./strata-rocm fixtures` builds the fixtures upstream never
  published: blocks from ggml's encoders (the only i-quant encoders that exist),
  reference values from **gguf-py** (a separate implementation in another
  language). Result: **`dequant rel 0.00e+00`, bit-exact for all ten**, and the
  quantized mat-vec at 4.3e-3–6.9e-3 against a declared 2e-2 bound. ggml's C
  decoder and gguf-py agree exactly on all nine types gguf-py implements.


- **Routing is verified, tie-breaking included** (`native_router_parity`): six cases
  including a tie exactly at the 10th/11th boundary, where the documented
  lower-index rule decides which expert runs. All ids match a float64 reference.
- **The q8_1 activation quantizer is verified** (`native_mmvq_quant_parity`),
  including the clause that the stored sum is of the *original* floats and **not**
  reconstructed from the quantized integers — a block is constructed where the two
  differ, so a naive implementation fails that case and nothing else.


- **Attention is verified** (`native_flash_attn_parity`) — seven cases including
  **NaN-poisoned padding**, which confirms the kernel synthesizes unused k/v rows
  and mask entries instead of reading them, and the invalid-step guard (status set,
  all 6144 outputs NaN). The heads are made distinguishable so the documented
  `head / 12` map cannot coincide with a wrong `head % 2`.
- **A contract gap is recorded**: the attention header documents the geometry but
  the implementation also requires `shapes.idx_block == 4` and
  `shapes.idx_top_k >= 256`, and its error message names neither. Our test hit it.


- **The GDN recurrence is verified** (`native_gdn_parity`) — seven cases including
  **eight consecutive steps over one state buffer** with the reference iterating the
  same eight, which is the only way a mishandled feedback path shows up. Worst
  relative error 9.4e-07 at 0.003 of a derived bound. Two details the header omits
  (decay-versus-delta order, and that the readout uses the *updated* state) are
  asserted against the algebra they imply and recorded as resolved-from-implementation.


- **All five GDN preprocessing helpers are verified** (`native_gdn_preprocess_parity`),
  and an epsilon asymmetry the header only implies — `l2_norm` divides epsilon by
  128, `out_norm` does not — is now tested in the regime where it bites (a
  small-magnitude row, where mixing them up would be 90% off) instead of asserted
  where it is indistinguishable.


- **A kernel that contradicted its own contract was found and fixed.**
  `native_moe_combine` promises FMA accumulation for experts 1..k-1; the
  project-wide `-ffp-contract=off` made it round separately, and 35–280 of 256–512
  elements differed from the contracted reference. Patch `0004` makes the fusion
  explicit. **A float64 comparison could never have caught this** — both orders are
  within 2.7e-07 of the ideal — so the test computes two exact fp32 simulations and
  reports which one the device matches.
- **`native_gr_postops` is a recorded suspect** for the same reason: its header
  contracts ordered FMA accumulation and nothing tests it yet.


- **`native_gr_postops` is verified and the FMA suspect is cleared.** All three
  functions pass, including the exact `output == residual` aliasing. The fusion was
  confirmed tolerance-free: both documented paths run on the same inputs **differ
  on 125 of 256 elements**, which could not happen if `-ffp-contract=off` had
  broken this kernel the way it broke `native_moe`. It did not, because this file
  names `__fmaf_rn` explicitly while `native_moe` trusted the compiler.


- **Two QSA sub-ops verified** (`native_qsa_ops_parity`):
  `native_qsa_rms_norm_weighted` on **both** its reduction branches (256-thread
  below `n_cols = 1024`, 1024-thread above — a bug in one would never show in the
  other) at ~7.1e-08, and `native_qsa_gate_apply` with a fixture that poisons the
  first half of each row, so a wrong-half read fails by ~1 instead of by a rounding.


- **RoPE is verified** (`native_rope_parity`): both head widths, two frequency
  bases, in-place and out-of-place, with and without the IMRoPE table. Three checks
  are not tolerances — the copy region is **byte-exact**, the rotation is asserted
  **as an isometry**, and pair 0 pins the angle itself. The flat tolerance in the
  first revision was wrong (the fp32 *angle* is thousands of radians for small
  pairs, so its own ulp dominates); the bound is now derived per element.


- **The Q5_K decode-step GEMV is verified** (`native_mmvq_q5k_parity`) — five
  fixtures across ncols 1/2/4 and a 2560-wide projection, `bound excess = 0`
  everywhere. Q5_K is one of the pack's formats and is absent from `iq_parity`'s
  type list, so this path had no test at all.
- **An audit error was found and corrected.** `native_mmvq_*` had been listed as
  untested on the strength of a name-based search; the dispatcher *is* exercised by
  `iq_parity` for all ten of its types. Re-deriving by symbol shrank the remaining
  set from four groups to **two**: `native_embed` and `native_qsa_indexer_append`.


- **The QSA key indexer is verified** (`native_qsa_indexer_parity`) — the most
  stateful kernel in the tree, driven as a *sequence* with a float64 state machine:
  `tail` byte-exact, `pooled` at 6e-08–1.6e-07, `block_pos` exact, and the spare row
  asserted against the stored `dead` key. The documented first-cell-vs-last-cell
  rotation trap is shown to be **detectable** (gap 1.07–1.82) rather than assumed.
- **Coverage is now measured per kernel header: 39 of 42 have a tested entry
  point.** The three that do not are dispositioned in `TASKS.md` (not called,
  weight-gated, and out of scope respectively); none is on the default path.

## S3-feasibility: PASSED for the default configuration

**39 of 42 kernel headers have a tested entry point.** Every kernel on the default,
non-speculative forward path passes an independent test on the real card (suite
re-run on upstream v0.1.15). The three exceptions were measured, not argued:

| header | measured status |
| --- | --- |
| `native_gr_norm` | **not called by the engine at all** |
| `native_ple_postops` | PLE path — **weight-gated** |
| `verify_kernels` | speculative decoding — **out of scope for release 1** |

Session 1 attached three conditions. Session 2 closes the first and leaves the
other two, which are weight-gated and scoping decisions respectively:

1. ~~**Enabling the `native_gdn` experiment puts untested code on the path.**~~
   **Closed**: all three `fused_gdn` entry points now have passing tests
   (`fused_gdn_parity`, `fused_gdn_conv_l2_parity`, `fused_gdn_ab_parity`), on
   top of the already-tested `native_gdn_step` and preprocessing. The flag
   still defaults to false; enabling it at Stage 4 is a configuration decision,
   no longer a correctness risk.
2. **The PLE path is unverified** pending weights.
3. **Speculative decoding is untested** and out of scope.

Passing this gate authorises *proposing* a download. It does not authorise one,
and it does not claim the port works — no token has been produced.

## What is NOT verified

- **PLE table path** (`ple_parity`) — needs the model's PLE table; no weights.
- **The QSA scorer's speed** — the portable path is a reference implementation,
  not an optimisation: one sequential F32 dot product per (row, head) instead of
  tensor cores. It is correct and slow, and no performance claim is made for it.
- **`native_expert_parity`** — its reference needs `ggml-cpu.h`/`ggml.h`. Those
  headers are now present (the pinned llama.cpp is fetched by `prepare`); the test
  still needs ggml's CPU backend compiled and linked.
- **MMQ type coverage** — **8 of 8 verified** (session 3: S3.2b' closed with
  uniform-imatrix fixtures; the old "needs an importance matrix" blocker applied
  only to calibration-driven quantization, not to fixtures). Note the recorded
  divergence: ggml's HIP `__vsub4` saturates where CUDA's wraps (`PORTING.md`
  §6c), a live suspect for any IQ-format mismatch.
- **Native (IQ) CPU experts** — built and linked (S2.7c), but **not one has ever
  been computed**: the reference compares CPU and GPU experts on real GGUF rows,
  so it needs a shard.
- **Upstream's eight explicit MMQ template instances** are not compiled (S2.7d):
  the wrapper links, but whether every quantization type dispatches to a real
  kernel is untested.
- **Full model** — not downloaded, not run, no logits compared.
- **Performance** — no AMD measurement exists. No NVIDIA throughput claim is
  transferable and none is made.
- **Distribution/licence** — no top-level LICENSE; local use only.

## Blockers

| blocker | kind | what unblocks it |
| --- | --- | --- |
| Model weights not on disk yet | **in progress (user-driven)** | the user is downloading to `/mnt/LINUXWINSHARE` (remounted rw by the user); when files land: sha256-verify against `MODEL-DOWNLOAD-PROPOSAL.md`, then S4.1 |
| QSA scorer fast path: **measured, not promoted** (session 5) | resolved by measurement | the WMMA arm (patch 0007) is parity-clean but ~2x slower than portable, and the whole scorer is ~0.1% of decode time - the portable path stays, wiring declined with numbers (`results/record-qsa-wmma-verdict.json`, PORTING §12h) |
| IQ2_XXS / IQ2_XS MMQ reference | missing upstream capability | needs quantization with an importance matrix (S3.2b'); `prefill_mmq_parity` reports it as a declared skip |
| GPU device nodes hidden from the default shell | environment | run GPU commands outside the sandbox; `doctor` names this explicitly so it is not mistaken for a driver fault |

No repair limit was reached (0 of 3 per signature). No tolerance was changed.
No retry loop was run.

## Budgets

Wall clock recorded: not instrumented per slice (see the note below) · downloads
**40551103 bytes** of 2 GiB (vendored hipify-perl 986704 B + the pinned
llama.cpp archive 39564399 B, per `state.json` and `download-proposal.json`;
the venv is counted separately as `venv_bytes`) · model weights **0 bytes
downloaded by the rollout** (the user is downloading weights themselves, which
is outside this budget by construction) · repair attempts: **0 of 3** used per
failure signature (`state.json` `budgets.repairs` is empty).

> Honest gap: `state.json`'s `wall_clock_s` is not yet accumulated by the
> wrapper, so the 60-minute execution slice from the plan is not enforced
> automatically. It is tracked by the operator. Closing this is a small task.

## Exact next action

**One action is yours: restart your 8080 llama-server.** It is not running —
not kernel-killed (the kernel log names only our own strata process as the OOM
victim), and never touched by the rollout; it likely exited during the memory
pressure window that the serve engine's pinned 47 GB arena created. My fault in
cause, not in action. Your previous invocation is preserved in the session
record: `llama-server --model ~/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/.../Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --alias qwen27b-ista-iq3s --device ROCm0 --n-gpu-layers 99 --ctx-size 98304 --cache-type-k q4_0 --cache-type-v q4_0 … --spec-type draft-mtp`.

Mine, in order (no weights needed beyond what is on disk):
1. D1 candidate now unblocked by upstream `de159b5`: `--mmap-experts` on the
   native IQ pack — measured expert-arena DRAM relief, paired A/B as usual.
2. The untried 0.1.27 knobs worth one paired run each: `STRATA_MTP_BATCH`,
   `--kv k8v4`.
3. rocWMMA QSA fast path — feasibility verified (v0.8 vendored, compiles for
   gfx1100); implement opt-in behind `qsa_score_parity` with the fp16-bound
   derived in PORTING first, then `./strata-rocm bench` vs the portable path.
4. Decide the two Monitor loose ends: `origin/fix/live-token-rate` (`b6e1176`,
   unmerged) and whether to install `data/draft_vocab.bin` (your call).
5. S7.1/S7.2: launch script + frozen known-good revision + fork release.


---

## Session 9 - the PR campaign (rocWMMA scorer, AMD rates row, D8) - IN FLIGHT

Goal: one PR to Niko1221/Strata carrying (1) the rocWMMA QSA-select TC block-scores arm,
(2) an AMD RX 7900 XTX IQ3_S row in the rates tables, (3) D8 GPU-floor attribution.

**Done and recorded:**
- The arm (fork `feature/rocwmma-qsa-select`, commits e4724ff/94acae8/b6ae967): parity 5/5
  cases including one-hot/coordinate-coded adversarial fixtures; the emitted code object
  runs 8 `v_wmma_f32_16x16x16_f16` (verified by objdump - no fallback can masquerade);
  bench 10.5x at 128K context (nq=256).  Records: `record-qsa-select-wmma-parity-*.json`.
- Rollout twin: patch `0011-qsa-select-wmma.patch` (4 file hunks) applies clean on a fresh
  tree; the generated build is 47 executables with 0 errors.  Parity suite registered
  (`qsa_select_wmma` in STRATA_HIP_PARITY); its GPU run batches with the campaign.
- D8 code fix (7a6c1ed): `kProfPer` 32->33 (the hc-read second half's up-stamp overwrote the
  NEXT layer's stage-0 base) + the unsigned-delta wraparound guard that printed ~1e14
  ms/window.  Reason first: PORTING.md 19.  GPU validation + the no-doorbell floor
  attribution are campaign stages (`d8_capture.py`).
- Pre-PR merge audit (`record-prepr-merge-audit-*.json`): upstream is 8 commits ahead
  (engine 0.1.28); exactly one conflict (verify.cpp, a pre-existing guard divergence),
  resolved on the integration branch `feature/rocwmma-qsa-select-upstream` (0165063),
  which builds 234/234 targets.  Overlapping open PRs: #216/#223/#225 (code), #208
  (docs/DETAILS.md).

**In flight (the `ab-driver` user service, memory-gated at 70 GB MemAvailable):**
1. the paired A/B (frozen contract: 5 pairs x 2 slots per length, 1K-128K + decode) - 20/120
   cells recorded; at 128K the effect is already visible (warp 93,378 ms vs wmma 85,285 ms
   GPU timeline, qsa phase 12,858 -> 5,600 ms);
2. the rates sweep (1K-128K, methodology pinned in `rates_amd.py`, matrix.json-compatible);
3. the D8 profile capture.

**What blocked progress:** the engine peaks ~63 GB anonymous at startup (expert-profile
dequantization) and the kernel OOM killer took several runs; the driver now waits for real
headroom and runs one cell at a time.  User stopped the 8080 llama-server (recorded), which
removed the GPU-noise source behind the early 50x outliers.

**Next:** let the campaign finish (it self-reports in `artifacts/rocm/logs/`), then gate
the promotion (>=5% at 128K; decode cap 3%), write the tables + PR body, `./strata-rocm
report`, push both branches.

**Campaign + PR state (2026-09-30, later):**
- **PR #228 to Niko1221/Strata is OPEN and READY FOR REVIEW** (was draft; cross-fork head
  `xyzzing:feature/rocwmma-qsa-select-upstream`). The body carries the kernel-vs-phase-vs-e2e
  honesty ladder (10.5x kernel / ~2.3x phase / +8.7% e2e at 128K), the full 1K-128K paired
  table, and the three gates.
- **A/B contract complete** (123 valid cells): 10/10 pairs at every length 1K-128K, zero
  exclusions (failed attempts rescued by retry-until-valid: 1 at 1K, 26 at 4K, 1 at 32K -
  declared in the record).  Paired gains: 1K +0.2%, 4K +0.9%, 32K -4.5% (interference-noise
  at ~22 s cells; 4 of 10 pairs flagged, never dropped), 64K +3.8%, 128K +8.7%.
- **Frozen gates all PASS -> verdict PROMOTE** (default flip earns its gates): gate 1 +8.7%,
  gate 2 +2.2% (decode, paired estimator - see PORTING 20 for why the estimator and not the
  threshold), gate 3 +3.6%.  Both gate-2 statistics recorded (`ratio` 1.022, `ratio_of_medians`
  0.937).
- **Upstream v0.1.29 merged** into the integration branch (merge `0975f72`; upstream merged
  PR #187's multi-query scorer, superseding our cherry-pick). Two conflicts, both comment-only
  in `qsa_select.cu` + `CMakeLists.txt`; merged tree builds 236/236. Parity suite's
  adversarial-fixture upgrade committed (`9e56773`).  Record:
  `record-upstream-0129-merge-20260930.json`.
- **Anchor move** (`record-strata-anchor-move-20260930.json`): `Strata/` c1e9033 -> 9e56773
  (user-requested); hook files verified byte-identical before clearing; `.agentic/checks.json`
  (configured exception) preserved.  Existing evidence keeps its named revisions.
- **Rates harness fixed twice** before collecting: `--native` passed bare swallowed `--ple-gguf`
  (399 exit=2 cells purged, declared in the record notes) and `STRATA_PREFILL_TIMING=1` was
  missing (no prefill line -> no prompt rate).  First clean cells landed (32K: prefill 840
  tok/s) before the GPU was yielded.
- **Resident `strata-8081` restored and verified** (HTTP 200 on /v1/models; unit active;
  config persisted at `artifacts/rocm/serve-8081.json`).  It had stopped at 15:03 (transient
  unit, collected on stop) - before the anchor move.  The GPU campaign is PAUSED while the
  user's service holds the card; resume = `systemctl --user start ab-driver` (rates sweep
  continues, then d8_capture/d8_attribute).

**Campaign resume (2026-09-30 late, post-reboot):** Pre-flight caught a real harness bug before
the GPU window: the rates done-check required only `decode_tok_s`, so a tier could "complete"
without `prefill_tok_s` (the rates row needs prompt tok/s). Fixed (now requires both); raw
engine logs moved from `/tmp` (wiped on reboot) to `artifacts/rocm/logs/rates-raw-*.log`; two
stale `prefill=None` cells (collected before the `STRATA_PREFILL_TIMING=1` env fix landed) purged
and re-collected. Verified: 1k now parses both rates (prefill 878.87 / decode 28.17) - short
prompts parse fine, matching the A/B's identical `PRE_RE`. Campaign restarted as stages-2-4-only
driver (protects the frozen A/B from re-touch); 2/5 tiers clean (1k, 32k), grinding 4k/64k/128k
then D8 capture + attribution.

**Rates methodology hardened (2026-09-30 late):** The published row must be defensible, so the
pipeline was fixed to match the declared 2026-09-29 methodology (median-of-3 per tier) instead of
single-cell: `rates_amd.py` fills 3 clean cells/tier, the driver done-check requires >=3/tier with
BOTH rates, `apply_rates_row.py` takes the median and refuses otherwise. Two real bugs found:
`apply_rates_row.py` crashed on `prefill_tok_s=None` (it guarded decode but not prefill), and a
mixed-state batch (32k pre-reboot, 1k/4k post) was physically inconsistent - 32k decoded FASTER
(44.28) than 1k (28.17) under HIGHER contention. Purged and re-collected as one fresh batch; the
clean post-reboot state confirms the old cells were contaminated: 1k decode 28.17 -> 52.9/51.3
(prefill stable ~858-859). Prior values preserved in the record's notes.

**D8 pipeline pre-flighted (2026-09-30 late):** Caught the SAME `--native`-bare bug that broke the
rates sweep, plus four dead paths (`/mnt/LINUXWINSHARE/strata/*` no longer exist) - d8_capture.py
would have failed twice over and wasted the whole D8 window. Fixed to the proven A/B config paths,
durable log path, and the correct workload token file. Improved d8_attribute.py to separate the
profiler's "(gap)" column from real GPU stages: the stages sum to the window total by construction,
so the old `total - sum` "unnamed_residual" was ~0 and hid the residual inside a "named" sum. Now the
residual is explicitly named (profiler gap + doorbell/pool round-trip from the no-doorbell floor),
with a crosscheck (~0) proving parse completeness. Both verified end-to-end on the known
engine-tune.log profile line: named_gpu + gap = total, crosscheck -0.38.

**Rates stall-handling (2026-09-30 late):** The sweep hit intermittent background-GPU stalls (one 4k
cell decoded at 1.29 tok/s vs its sibling's 48.17 under identical load/chrome; one 64k at 12.92 vs
~44) - a stall is not the engine's decode rate and must not become a published README/DETAILS number.
Applied the A/B's `retry_until_3_clean` pattern: each cell gets `stall_flag` (decode < 40% of the
tier's best-observed decode), stalls stay in the record (never dropped) but are excluded from the
median, and each tier keeps collecting until it has 3 NON-stalled cells. `apply_rates_row.py` publishes
the median of the non-stalled cells and reports `n_stalled_excluded`. Verified end-to-end: 4k 1.29 and
64k 12.92 flagged STALL; 1k all clean (50.5/51.3/52.9). Non-stalled: 1k=3, 32k=3, 4k=2, 64k=2, 128k=0.

---

## Session 10 (2026-09-30/10-01): deliverables closed out; PR re-scoped by the maintainer

**D8 GPU-floor attribution COMPLETE (deliverable 3).** The STRATA_VERIFY_PROFILE stage report is a
"ms/round" block (`verify window ... wait for rings X pool Y host Z commit W ms/round`), not the
"GDN layers ... over N windows" format the old capture regex expected (it matched nothing and left
`profile_report` None). Fixed `d8_capture.py`'s `PROFILE_RE` to capture the real block; rewrote
`d8_attribute.py` to parse it. Captured the baseline + the no-doorbell floor
(`STRATA_VERIFY_DEVICE_PLAN=1`). **Finding:** the ~26 ms/window residual is the `wait for rings`
column (28.005 ms/round) and the no-doorbell floor leaves it unchanged (28.167, delta -0.162) -
so it is **ring-wait** (GPU idle waiting for the host command rings), NOT the expert-pool doorbell
round-trip (that hypothesis is measured and rejected). Full stage decomposition named (pool 6.385,
dispatch run 5.844, pool multi gate/up 3.829, ...). Records: `record-d8-profile-capture.json`
(both runs) + `record-d8-gpu-floor-20260930.json` (attribution + verdict).

**Rates sweep COMPLETE (deliverable 2).** 5/5 tiers at 3 non-stalled cells each (median-of-clean):
prefill 1k 860 / 4k 1350 / 32k 602 / 64k 1449 / 128k 1403; decode 1k 51.3 / 4k 52.2 / 32k 50.1 /
64k 44.9 / 128k 52.1. The flat decode (44-52 across 1K-128K) is architectural (GDN linear attention
makes decode O(1) in context). The 32k prompt dip (602, vs 1350/1449 neighbours) is real - its
GPU-timeline (~46 s) matches 64k's for half the tokens - likely the `--kv-resident 32768` boundary;
all cells recorded in `amd-rates-20260930.json`. Row applied to README:33 + DETAILS:29,44 (262K "-").

**Maintainer re-scoped the PR (the pivot).** Niko1221 CLOSED PR #228 (the bundled 3-deliverable PR)
on 2026-09-30, NOT merged: *"We'd take the rocWMMA scorer as an opt-in (STRATA_SELECT_WMMA=1) but
not as the default ... open a new PR with only the scorer, its parity test, the bench and the CMake
detection - without .agentic/, AGENTS.md, CLAUDE.md, CI workflows, serve/telemetry.py, direct_file.cpp
and math_constants.h. We have no gfx1100 card, so your numbers carry it."* User decision: follow the
maintainer - scorer-only PR; D2/D3 held.

**PR #312 OPEN (deliverable 1, scorer-only).** https://github.com/Niko1221/Strata/pull/312 -
branch `feature/qsa-select-rocwmma-optin` (base: upstream main 0.1.30). Exactly 6 files (scorer,
parity, bench, CMake detection + the prefill call-site hook); ZERO excluded files. Opt-in tightened
to `STRATA_SELECT_WMMA=1` (default off; the arm that runs is printed so a fallback can't masquerade).
Cherry-picked the 4 scorer commits onto 0.1.30 (clean auto-merge); verified the merged prefill/CMake/
decl regions and that the scorer calls only functions present in 0.1.30 (`qsa_block_scores_tc/_topk`).
Numbers carried in the PR (parity 5/5, kernel 10.5x @128K, paired e2e 1K +0.18% ... 128K +8.7%).

**D2/D3 held on the fork (not in the PR, per the maintainer).** D2 rates row -> fork
`feature/amd-rates-row-20260930`; D3 D8 fix -> fork `feature/rocwmma-qsa-select` (+ local records).
All three deliverables are landed on the fork; only the scorer is proposed upstream.

**Resident 8081 restored** (campaign done): serving, context 65536, model qwen3.8-flash-next,
health ok. Rollback unchanged (`artifacts/rocm/rollback/strata-0.1.28-mmq`).

**Next / follow-ups:** (1) compile-verify the scorer on 0.1.30 + re-run parity against anchor
9e56773 (GPU, with resident stopped) - the code is verified well-formed but not yet rebuilt on 0.1.30;
(2) if the maintainer wants D2/D3, offer the held branches as separate PRs; (3) the resident runs
0.1.29 (`build-hip-0128`); upstream 0.1.30 has faster prompt kernels - a later upgrade is optional.

## Session 11 (2026-10-01): scorer validated; all three deliverables proposed upstream as separate PRs

**Scorer validated (personally reproduced on the GPU).** Re-ran the parity + kernel bench on
`build-hip-0128` (0.1.29 campaign build; the scorer comes in via patch `0011-qsa-select-wmma`).
Parity (full mode) **5/5 cases PASS**: warp arm bitwise == its host model; tail block bitwise == the
warp arm (shared tail kernel); every score inside the pre-derived bound `B = 2^-9.5*S + 8*2^-24*|score|`
vs the FP64 oracle (worst utilisation 0-11%); selection ids agree wherever gap > 2B (differing queries
0); the +2.5B negative control correctly FAILS. Kernel bench @128K (ctx=131072, nq=256, median-of-9):
warp 2.635 ms -> wmma 0.238 ms = **11.05x** (PR #312 cites 10.5x - consistent). Logs:
`validate-parity-0128-20260930.txt`, `validate-bench-0128-20260930.txt`, `record-validate-scorer-20260930.json`.

*Caveat (honest):* this is the 0.1.29 campaign build, not a fresh 0.1.30 rebuild. A source rebuild
through the ROCm patch pipeline double-applies patch 0011 (the scorer lives in BOTH the Strata/ source
and the patch), so it rejects "already applied". The scorer kernel is byte-identical across the
0.1.28/0.1.29/0.1.30 rebase (only the base differs), so this parity/bench validates the exact kernel the
PRs contribute; the 0.1.30 integration is separately verified well-formed (additive, matching call
signatures). PR #312 is the SOURCE form of this same scorer.

**All three deliverables now proposed upstream as separate PRs** (per the maintainer's "one PR per
concern" re-scoping of #228):
- **#312** scorer (opt-in `STRATA_SELECT_WMMA=1`) - `feature/qsa-select-rocwmma-optin`
- **#319** AMD RX 7900 XTX IQ3_S rates row (docs only, 3 lines) - `pr/amd-rates-row-20260930`
- **#320** verify: fix STRATA_VERIFY_PROFILE stage columns (metrics only, 2 files) - `pr/d8-verify-columns-20260930`

Each is a clean minimal diff against `origin/main` (0.1.30): #319 = README +1 / DETAILS +2;
#320 = verify.hpp +3 / verify.cpp +24. Both cherry-picked onto 0.1.30 cleanly (auto-merge), zero
excluded files (no `.agentic/`, `AGENTS.md`, `CLAUDE.md`, `direct_file.cpp`, `math_constants.h`, CI,
telemetry). #319's every cell is traced to `amd-rates-20260930.json`; #320 is the D8 column fix whose
attribution (ring-wait residual) is in `record-d8-gpu-floor-20260930.json`.

**Resident 8081 restored** after the GPU validation (health ok, context 65536, `qwen3.8-flash-next`).
`build-hip/hipify` preserved/restore from `artifacts/rocm/rollback/hipify-0129-runtime.tar.gz` so the
resident keeps its 0.1.29 runtime.

**Next:** await the maintainer on #312/#319/#320. If any is merged, the resident can be rebuilt on the
merged 0.1.30; if #320 lands, the D8 verify columns ship upstream and the ring-wait attribution stands.

## Session 12 (2026-10-01): the 0.1.30 caveat closed - the PR source scorer passes parity on a true 0.1.30 rebuild

Built PR #312's OWN source (branch `pr-qsa-select-rocwmma` = origin/main 0.1.30 + scorer) through the
ROCm pipeline with the scorer patch `0011-qsa-select-wmma` DISABLED (so the source scorer is built,
not double-applied), and re-ran parity. **Result: 5/5 cases PASS on the 0.1.30 source build** - warp arm
bitwise == host model; tail bitwise == warp arm; every score inside the pre-derived bound
`B = 2^-9.5*S + 8*2^-24*|score|` vs the FP64 oracle (worst utilisation 0-11%); selection ids agree where
gap > 2B (differing 0); the +2.5B negative control correctly FAILS. This SUPERSEDES the earlier
0.1.29-campaign parity as the caveat's answer: the exact scorer the PRs ship is now GPU-validated on the
0.1.30 base it targets. Log `validate-parity-0130-20261001.txt`, record
`record-validate-scorer-0130-20261001.json`.

**Two honest notes:**
1. The bench (`qsa_select_bench`) did NOT rebuild on 0.1.30 - it fails to link only because
   `qsa_prompt_attn.cu` uses `__trap` (a hipify translation gap in an UNRELATED kernel, nothing to do with
   the scorer). The scorer kernel is byte-identical to the 0.1.29 build that measured 11.05x and parity
   confirms correctness, so the kernel speed carries; the 0.1.29 bench number stands.
2. **Port finding:** the ROCm patch pipeline (0001-0011) is **pinned to the 0.1.29 base**. Applied to
   0.1.30, patches 0001-0010 reject (their context shifted upstream) and `__trap` goes unfixed. So a clean
   FULL 0.1.30 engine rebuild through the pipeline needs the patches re-based to 0.1.30 - a port task, not
   a scorer issue. The scorer kernel + parity + bench sources are clean and build independently of those
   patches (which is exactly why the parity could be validated on 0.1.30).

Resident 8081 restored (0.1.29 runtime preserved via `hipify-0129-runtime.tar.gz`). Scorer patch 0011
re-enabled. All three PRs (#312 scorer, #319 rates row, #320 D8 fix) stand open upstream.

## Session 13 (2026-10-01): the 0.1.31 upstream refresh - fork merged, rates re-measured (+13-27% decode), PR threads resolved, pipeline re-pinned

**Upstream state read (records: `record-upstream-0131-audit-20261001.json`).** main moved 0.1.30 -> **0.1.31**
(`9259cad`): the #262 AMD packed-byte intrinsics (`__byte_perm`->`v_perm_b32`, SWAR `__vsub4/__vsubss4/__vcmpne4`
in upstream's own hip_compat shim), hyper-connection read in 2 kernels + `STRATA_GR_V3` opt-in, CS-T low-RAM
(GGUF-in-place experts, partial pin), the 8 GiB arena pin cap now WDDM-only (Linux default unchanged), verify
window #267 rework, gfx1101/gfx1200 cards. PR threads: **#312 CLOSED** by the maintainer with a rebase request
("rebase onto #337's arch-dispatching entry point, keep STRATA_SELECT_WMMA=1 opt-in; a new PR is welcome");
**#319 asked to re-measure on 0.1.31** ("about +20% decode on the 7900 XTX") or move to COMMUNITY_BENCHMARKS.md;
**#320 accepted** into the 0.1.32 batch as-is.

**Fork merge (fork `feature/rocwmma-qsa-select-upstream` merge `c304952`).** 0.1.29 -> 0.1.31; 2 conflicts
(CMakeLists qsa_prompt_attn guard wording; AMD_HIP.md setup text - kept our amdgpu-sysfs Monitor line, upstream's
"Monitor shows no GPU statistics" is false on this fork). The D8 verify-columns fix survives the verify.cpp
auto-merge. Build `build-hip-0131`: configure needs `env -u LD_LIBRARY_PATH` (the zcode AppImage mount poisons
cmake's CMAKE_ROOT - found this session) and `CMAKE_HIP_COMPILER_ROCM_ROOT=/usr`. CTest: **50 passed**, 1 skip,
4 declared environment-gated failures (`expert_parity`/`pool_test` fixture-gated as before; `ple_parity` now
HARD-FAILS without the model's PLE table - new 0.1.31 semantics; `platform_memory_test` needs mlock beyond the
user manager's 8 MiB hard MEMLOCK cap - sudo forbidden). `qsa_select_wmma_parity` passes on 0.1.31.

**0.1.31 engine startup contract change (rates pre-flight caught it):** split-GGUF validation now REQUIRES the
full shard set by llama.cpp's `-NNNNN-of-MMMMM` names. Fixed with symlinks only (`/home/zacch/models/flash-next-iq3s/`
naming both existing shards; no file moved, no download), and `--ple-gguf` must name the SAME symlinked path as
the shard set or the loader appends the PLE shard twice ("duplicate split shard number").

**Rates re-measured on 0.1.31 (deliverable: PR #319 update, user-approved).** Frozen methodology (one-shot,
greedy, 256 tokens, median of 3 non-stalled cells/tier; stalls recorded, excluded; new record
`amd-rates-0131-20261001.json`): decode 1k **64.2** (+25.1%), 4k 59.1 (+13.2%), 32k 60.5 (+20.8%), 64k 56.8
(+26.5%), 128k 59.0 (+13.2%) - **the maintainer's "+20% from #262" reads confirmed**; prompt rates unchanged
outside 32K. **The 0.1.30-era 32k prefill dip is GONE** (602 -> 1,565 tok/s, in line with neighbours; the
`--kv-resident 32768` suspicion is moot on 0.1.31). Rows + engine-version footnotes pushed to
`xyzzing:pr/amd-rates-row-20260930`; measured table posted as a PR comment.

**Scorer PR rebuilt for the #337 rebase (branch `xyzzing:pr-qsa-select-wmma-rdna3`).** Merged #337 (gfx12 bf16
WMMA scorer + prompt-aware top-k) onto 0.1.31 main, then our three scorer commits cherry-picked with the design
flipped to what the maintainer asked: the arch dispatch is INSIDE `qsa_block_scores_tc` (gfx12 first; gfx1100
runs our FP16 WMMA arm under `STRATA_SELECT_WMMA=1`; else false -> warp kernel; `STRATA_SELECT_OLD=1` still
wins; the arm that ran is printed once). Our kernel renamed `block_scores_wmma_gfx11_kernel` (name collision
with #337's). The prefill call site is upstream's unchanged. Bench = union of both: #337's capacity arg + FP64
gate (single reference shared by all arms, 4x-warp bound), our median-of-9 timing + adversarial comparisons.
rocWMMA note: Fedora ships `/usr/include/rocwmma/rocwmma.hpp`, so `__has_include` finds it with no
STRATA_ROCWMMA_INCLUDE_DIR - the arm compiles in on every build here (verified in the 0128/0131 binaries).

**Pipeline re-pinned to 0.1.31 (S0 green again).** Anchor `Strata/` moved d02abdec -> **9259cad** (recorded,
hook files preserved; `data/expert-profile.bin` byte-identical - the running sweep was untouched). Patch stack
rebuilt against the new base: **0001, 0002, 0003, 0005, 0006, 0009, 0010 DROPPED - superseded upstream** (their
mma region is now `#if !defined(__HIPCC__)`-guarded with upstream's own scalar AMD fallback; the arch gate is
upstream's compiled_for/STRATA_HIP_ARCHS; MMQ device-cc is upstream's `ggml_cuda_parse_amd_id`; fused_gr TILEV
is capability-based; --mmap-experts is CS-T native; PR #104 and #105 MERGED upstream). Kept+rebased: 0004
(clean), 0007 (launch hunk adapted to the scalar/CUDA split), 0008 (fmaf sites updated for the SWAR intrinsic
rewrite), 0011 (arm-only insert, wired into qsa_block_scores_tc's AMD branch; prefill call-site hunk dropped -
upstream's call site already routes through _tc). NEW hipify rule `device-trap-builtin` (`__trap()` ->
`__builtin_trap()`, the session-12 bench-link blocker). Pipeline CMake for 0.1.31: + conversation_{snapshot,
state,memory}.cpp and remote_experts.cpp (session.cpp references it unconditionally), + kq_avx2.cpp (0.1.31
Q4_K/Q5_1/Q8_0 CPU kernels) with -mavx2, + mmq-instance-q8_0, + `STRATA_USE_HIP=1`/`STRATA_HIP_ARCHS="gfx1100"`
global defines (upstream's hip_backend sets them; their absence was invisible until 0.1.31's guards). Rollout
tests updated to 0.1.31 signatures: `native_rope_apply`/`native_qsa_indexer_append` now take a `RopeScaling`
struct ({None, freq_base} is the old contract); `qsa_score_parity`'s host model re-anchored to the scalar
kernel's sequential single-rounded FMA (old model was the tf32 path - bit-identity correctly FAILED until the
model matched); `iq_parity` now takes the fixture directory (registration special-cased).
**Gates: hipify clean (0 rejects, 5 patches), build 47 executables 0 errors, `test --stage kernels` 42 passed /
0 failed / 2 declared weight-gated skips.**

**Resident 8081 upgraded to 0.1.31 (user-approved before the session).** `serve-8081.json` repointed to
`build-hip-0131/strata` with the split-name symlinked shards (`.bak-0129` + untouched `build-hip-0128` are the
rollback); `restore_resident` restart verified against /proc; health ok (loaded, ctx 65536) and the smoke
completion *"Name the capital of France in one word."* → `"Paris"` (finish=stop), **decode 61.3 tok/s measured
through the serve path** (0.1.29 measured ~43 through the same path). The new scorer PR is **#368** (open),
the #319 update is pushed and commented. End state: every gate green on 0.1.31; nothing in flight.

**Opt-in pilot (2026-10-01, user asked "can we complete in 15 minutes?"):** no for the frozen contract, so a
predeclared pilot ran instead — `record-pilot-0131-optins-20261001.json`. Outcomes: the WMMA arm ENGAGES on
0.1.31 (arm print, no silent fallback) and its one clean cell matches the sweep rates (no regression), but the
paired e2e is INCONCLUSIVE — my pilot script had no VRAM-drain gate after stopping the resident: pair 1 ran
contaminated (189 s prefill vs the clean ~92 s) and pair 2 plus the script itself were OOM-killed (faults owned
in the record). STRATA_GR_V3: unmeasured, needs the frozen harness. STRATA_PARTIAL_PIN: **not applicable** on
this config (FileExpertSource's RAM-tier path never engages; upstream's own A/B measured a wash). Resident
restored and verified.

**Maintainer updates posted (2026-10-01, on top of the #319 table comment):** a refresh on **issue #106**
(our port thread still advertised the 0.1.21-era state and open #104/#105): both patches merged upstream,
port on 0.1.31 with the measured gfx1100 numbers, scorer rebased as #368, plus the two 0.1.31 environment
notes (split-GGUF shard-name requirement incl. the same-shard-set rule for --native/--ple-gguf; ple_parity's
new hard-fail-without-weights semantics). And a pointer comment on **PR #337** (the maintainer is actively
benchmarking it on an R9700): the gfx1100 companion arm is #368 under the same opt-in, with our measured
parity/top-k numbers and the merge-order note (#368 contains #337 wholesale — merge #337 first, or #368
directly). Nothing else upstream is ours to update: #320 is accepted as-is, and the #262 gfx1100 datapoint
already lives in the #319 thread where they asked for it.

## Session 15 (2026-10-02): the decode-and-exact-CLM campaign (Phase A on 0.1.33, Phase B reframed to validation)

**Skills + contract.** Updated `port-strata-hip` (restructured) and installed the NEW `enable-strata-clm`
bundle (zip, backups `.bak-20261002T063353Z`); the user's decode-and-exact-CLM agent prompt is the campaign
contract. Base decision resolved by the prompt's own upstream-relevance rule: Phase A runs on **0.1.33**.

**Fork refresh to 0.1.33** (merge `a1df415`): 0.1.32's RDNA router rewrite (+12% decode upstream), #337 merged
with maintainer follow-ups (WMMA scorer opt-in, top-k dispatch HIP-only), #329's STRATA_HIP_WMMA, GR_V3 Turing
guards. Conflicts resolved: upstream's #301 AMD Monitor backend supersedes the fork's telemetry (take upstream);
our gfx11 arm **re-plugged into the maintainer's opt-in dispatch** as the gfx1100 branch after gfx12 (the same
adaptation #368 needs — its rebase is now a cherry-pick); prefill's old selector reverted to upstream's form;
bench = the #368 per-arm-gate form. `build-hip-0133`: CTest **53 passed / 2 skipped / 4 declared env-gated /
0 regressions** (59 tests; `hip_prompt_attn_wmma` correctly skips on gfx1100).

**Desktop protection (user-reported crashes under campaign pressure).** Mechanism identified: cells launched at
~60 GB MemAvailable + engine's 52 GB burst = near-zero headroom, desktop squeezed (load avg hit 26). Structural
fixes, all recorded as contract amendments: every chunk under a user scope with **MemoryMax=58G** (a burst kills
OUR cell, never the desktop — note: this user manager rejects OOMScoreAdjust), launch gates raised to **66 GB
(light) / 70 GB (deep tiers)**, and `--vram-reserve-mib 2500` on every cell (desktop VRAM never evicted; both
arms identically). The measured admission gate (peak 51.9 GB, floor 10.0 GB) replaced the 70 GB constant.

**Phase A contract frozen** (`contract-phase-a-20261002.json`): accepted decode tok/s at fixed everything;
pinned 1K/4K/32K ladder + 128K deep (mapping declared; prose-pairing declared unavailable); 5 pairs × 2 slots;
weighted log-ratio aggregate + paired bootstrap; protections decode/TTFT/latency ≤3%, prefill ≤3%, memory caps;
predeclared candidates: WMMA (decision cells) + GR_V3; ≤6 candidates or 90 benchmark minutes.

**Campaign state (hardened driver `ab_driver_0133.sh`)**: stage A (WMMA decode protection) DONE — 10 valid
pairs, early pairing warp ~61-62.5 vs wmma ~59.5-59.7 tok/s (decode regression risk — gate 2 decides); stage B
(GR_V3) DONE — 5 pairs, **+2.7% aggregate decode, median +2.5% — below the 3% bar: ambiguous → extension to 10
pairs predeclared, queued after stage C**; stage C (128K/64K e2e, gate 1+3) running — first 128K cell 90.8 s GPU.

**Phase B reframed by the audit** (records: `record-session15-handover-20261002.json` + the audit report):
upstream 0.1.32/0.1.33's conversation cache (#342) **already implements the exact-reuse machinery** — full
state snapshots (GDN/PLE/QSA tails/KV/draft), turn-boundary checkpoint chains + root pinning + byte budgets,
exact token-prefix matching, deep-copy parking, prevalidation, byte-exact A/B/A ctests, and model-level parity
harnesses (`tools/conversation_cache_parity.py`, dry-run by default). Serve-layer revised messages already flow
through engine prefix-matching (`cached_tokens` surfaces reuse). Phase B = **validate on our stack, not rebuild**:
run the parity harness's `reuse`/`exchange` scenarios on build-hip-0133 (GPU, memory-gated), close the one [U]
gap (mid-page mid-block edit restore), report diagnostics. Spec-decode rollback confirmed positional-overwrite —
correctly NOT reusable as an edit checkpoint.

**Fork line → 0.1.35 (2026-10-02, merge `44663ae`; comparison in `record-fork-0135-merge-20261002.json`).**
The 0.1.34/0.1.35 delta is Windows AMD + the MCP setup server + serve fixes (#430 cancel-within-1s, #457 draft
metrics, #460, #447) + the #420 prefill-MMQ fallback; ONE conflict (AGENTS.md — upstream now ships repo
instructions, theirs taken). Our arm files are **byte-identical to PR #368's branch** after the merge, so the
PR and the fork line agree exactly. Gates: build-hip-0135 clean, CTest **55 passed / 0 regressions** (60 tests,
same 2 skips + 4 env-gated), `qsa_select_wmma_parity` + bench pass. Engine smoke on 0.1.35: **decode 67.45
tok/s, prefill ~810-830 tok/s @1K** (sweep medians 64.2/~850 — no regression). **Resident 8081 upgraded to the
0.1.35 engine** (rollback `serve-8081.json.bak-0131`); health ok, "Paris", serve-path decode 56.5/53.2 tok/s on
two requests (0.1.31's fresh-resident smoke was 61.3 on n=1 — within serve-path variance; the one-shot engine
comparison shows none). A leftover from the 0.1.33 merge (the duplicate-bench CMake cleanup lived only in the
working tree) was committed as a fixup before this merge.

**Decode knob pair (2026-10-02, Qwen3.8-Flash-Next on the 0.1.35 engine).** GR_V3 final at the contract's 15-pair
limit: **+2.79% aggregate decode, 12/15 pair wins, 95% CI [−1.8%, +7.5%]** — a stable, probable small win that
misses the 3% promotion bar; as a config-only opt-in it is adoptable for the user's serving (their call).
STRATA_MTP_BATCH_F16: **rejected** — the engagement fingerprint shows the draft pass unchanged (1.432 vs 1.432
ms/round) and the paired median is negative. FarSkip-Collective/Instella-MoE: **evaluated, not adopted** —
multi-GPU pre-training collectives, different architecture and stage; recorded in the donor log with the mapping.
The remaining recorded decode lever is the **ring-wait/submission campaign** (D8: ~26–28 ms/window of GPU idle
waiting for host command rings) — engine code in the verify window's submission path.

**pcie-frac sweep + the pool diagnostic (2026-10-03).** The STRATA_VERIFY_PROFILE report decomposed the
decode round precisely (spec 3, 0.1.35): wait-for-rings 26.1 ms/round with a 9.6 ms host pool call inside it
(gate/up 5.4 + down 3.1 ms of CPU rows at 41.3 GB/s aggregate; quantize 0.057 — which is why #500's quant
parallelization was a wash; dispatch ~1 ms; PCIe DMA in-graph for 55% of the misses; mtp drafting 3.4 ms/round).
The miss-split knob `--pcie-frac` was swept (0.75/0.9/1.0 vs the calibrated 0.55): no reliable gain on medians
(−0.95/+4.13/−23.2%), and the f100 collapse proves the DMA is in-graph-serialized, not overlapped free DMA —
the "convert CPU misses to idle-GPU PCIe work" premise was wrong. **Every knob on the decode critical path is
now measured; the remaining levers are engineering campaigns**: (1) row-kernel bandwidth 41.3→55+ GB/s
(+4-6% decode; AVX-512 prefetch/blocking at decode shapes), (2) conversation-adaptive expert placement
(fewer misses than the static frequency-ranked profile), (3) next-window submission pipelining (~1-3 ms/round).
The KWin VRAM incident was fixed the same day: serve config now carries `--vram-reserve-mib 4000`.

**Campaign 1 concluded (2026-10-03): the CPU-miss path is at the memory wall.** The STRATA_POOL_MT_MULT
sweep (1/6/12 vs the default 3 tasks/thread) was flat on medians; the kernel dispatch was already optimal
(ggml vec_dot at 1 token beats our AVX-512 by 23% — #152 routes it there); and the box's measured streaming
ceiling (28.6 GB/s 12-thread copy on AM5 dual-channel DDR5) sits BELOW the pool's 41.3 GB/s read-dominated
row bandwidth — the misses are DRAM-floored, not kernel-limited. **The remaining decode levers are
architectural**: fewer misses (placement/prediction, or the #505 IQ2_XS anomaly fix enabling a smaller
full-resident quant on 24 GB), and ring-wait pipelining to hide the floored pool time — the user-directed
next campaign.

**Ring-wait pipelining campaign concluded (2026-10-03, commit `ed43021`).** The wseq-gate implementation
followed the checkpointed design exactly (leading replay-safe gate kernel, arm/disarm semantics, fallback
rounds unchanged, abandoned-graph leak bounded and counted) and survived a genuinely useful smoke ladder
(device-alias bug, missing fallback disarm, missing fallback resets — each caught before exactness passed:
**greedy output bit-identical** with the gate on). The paired verdict: **median −0.94%, REJECTED** — the
host-side glue this design targeted is already amortized on this non-WDDM stack (the launch-submission cost
the CUDA-era notes described doesn't exist here). Kept merged as a documented exact opt-in. **The decode
board is now fully measured**: config wins adopted (spec 3 +9.5%, GR_V3 +2.8%), every other candidate
rejected with records. The remaining architectural levers: conversation-adaptive expert placement (fewer
misses) and the #505 IQ2_XS anomaly (24 GB full-residency path).

**Placement/prediction campaign concluded (2026-10-04): neutral on-distribution; conditional lever identified.**
Upstream's #477 (`--expert-profile-save`) already implements conversation-adaptive placement persistence — the
campaign validated the loop end-to-end (learning serve run → periodic save → reload) and measured learned-vs-static
on decode cells: **−1.53% median, REJECTED on an in-distribution workload** (the static profile was built from
code-agent routing; our learning workload was code-agent prompts). The lever is conditional on
out-of-distribution serving. With this, the decode board is fully measured at every level: config (spec 3 +
GR_V3 adopted, ~+12% combined), knobs (all rejected), engineering (pipelining, #500, DRAM floor — rejected/floored),
placement (neutral on-distribution).

**The 100500 table gap closed (2026-10-04 afternoon).** The T4/PR-#745 confound was real and is now
resolved by measurement: this host's ROCm stacks split by hipBLASLt version (system 7.1.1 → 100100,
nightly 10.2 SDK → 100500), and only the nightly engine was tableless. `tune_hipblaslt` (prebuilt
0.1.38-lineage binary) run against the nightly SDK's library produced `gfx1100-hipblaslt-100500.txt`
(32 rows = 16 shapes × {4096,8192} buckets, union of the shipped gfx1100 set and upstream's
gfx1201-100500 set; every row correctness-gated and its case's best). The serve-path A/B on the SAME
nightly binary (`scripts/rocm/hipblaslt1005_driver.py`, one-shot rates methodology, median of 3 clean
cells/arm, `--vram-reserve-mib 3072` — its absence is why the first cell batch refused at cublasCreate):
**128K prefill 1687.0 vs 926.1 tok/s median = +82.2% from the table alone**; no-table arm reproduces
T4's r10 cell (921); decode flat (62.3 vs 60.7, overlapping ranges); 1k sanity 69.4 tok/s. Deployed:
table committed on fork main (`c28ee6d`, local) + `serve-nightly-8082.json` env; the resident 8081
needs NO change (its 100100 table matches its stack). Submitted to PR #745 as a second results-only
commit (`f3d6827`, pushed; PR body updated) per COMMUNITY_BENCHMARKS.md. Upstream `tools/hip/` pickup
is a separate engine PR, deliberately not folded in. `record-hipblaslt-100500-20261004T103111Z.json`.
The resident 8081 was found STOPPED at session start and left as found — restart with
`./strata-rocm run` when serving is wanted.

**Production moved to the nightly stack (2026-10-04 evening).** serve-8081.json now runs the ROCm 10.2
nightly SDK engine (`build-nightly/strata`) with the gfx1100-hipblaslt-100500 table via `lib_dirs` +
env (args unchanged: the finalized knob lineage). Verified live: `hipBLASLt tuning enabled (32 rows,
gfx1100, version 100500)`, smoke decode 62.6 tok/s, draft 10/11, cache 82.1%. Start script:
`artifacts/rocm/serve-8081-start.sh` (MemoryMax 66G/MemoryHigh 61G desktop protection). Rollback kept:
`serve-8081.json.bak-0138-711` + the 7.1.1-stack build. The 8082 candidate is retired (records kept).
`record-production-move-20261004.json`. Caveat recorded: an SDK refresh invalidates the 100500 table
by design — retune is one tuner command (PR #755's tune-command.txt).

**7.1.1-stack engine builds removed (2026-10-04 evening, user-directed).** All eight fork-tree build
dirs (~3.8 GB) are gone; the tuner binary was preserved next to the SDK it retunes against
(`$EXP/tune_hipblaslt`), PR #745's tune-command.txt was repointed (commit `be18e4c`), the rollback
story is now rebuild-from-source (recipe in the .bak-0138-711 note), and a notice covers the
historical campaign scripts whose EXE paths dangle. Production 8081 verified unaffected.
`record-remove-711-builds-20261004.json`.
Post-check: 8081 was found cleanly stopped at 19:30Z (no systemd stop request — manual signal or
server.py self-exit); the production unit now runs `Restart=always RestartSec=10` and was relaunched
and reverified (table engaged, health ok). `build-cpu/` in the fork tree is a CPU build and was
deliberately not touched by the cleanup.

**Consolidation: one production home (2026-10-04 night).** Production = `$EXP`
(20261003-231142) alone: engine, SDK, 100500 table, pack, tokenizer, expert profile, venv,
hipify/serve, config, start script, logs. Verified with the full ladder (health, engine 0.1.38,
table engaged, smoke 'Paris' finish=stop 58.1 tok/s). First swap attempt OOMed on the dense upload
— async VRAM drain; the start script now gates on <2600 MiB VRAM before launch. strata-amd is
reduced to a records archive (3.2 → 1.1 GB: Strata anchor + artifacts + scripts + harness build);
its serve config/start script are pointer stubs. Share tidy: mtp/tensors intermediates (4.9G)
removed (manifests preserve the HF reproduction record); packs/iq3_s-full (49G, derived pack-form
backup of the live 52G local gguf) flagged for the user's decision. `record-consolidation-20261004.json`.

---

## Session 18 (2026-10-04 late): the RDNA3 WMMA prompt-attention arm + the #505 IQ2_XS decomposition

Campaign per the approved plan: Track A = port the gfx12-only WMMA prompt attention to gfx1100;
Track B = decompose the #505 IQ2_XS anomaly. Work tree: fork `/home/zacch/projects/strata`, branch
`feature/prompt-attn-wmma-rdna3` off **upstream main 99f3dbd** (0.1.38), build `build-pa` (nightly
SDK flags, production-mirroring).

**Track A — the arm is implemented, parity-clean, and fast.** Upstream's S6 kernel
(`qsa_prompt_attn.cu`) compiled its WMMA path only under `__gfx1200__/__gfx1201__` and gated the
env at runtime on a `gfx12` prefix; gfx1100 kept the ordered FP32 fallback, which A0 measured as
**the single largest prefill phase: 26.9% at 32K, 25.6% at 128K** (tonight's own production-stack
phase logs, `rates-raw-{32k,128k}.log` — zero new cells needed for the sizing). The gfx11 port
uses the same `v_wmma_f32_16x16x16_f16` ISA instruction through the **unsuffixed** builtin, whose
fragment layout (AMD Matrix Instruction Calculator, rdna3 tables, hardware-confirmed by a
standalone probe) differs from the gfx12 builtin's on all three operands: A[l&15][i], B[j][l&15],
D[(l>>4)+2j][l&15]. First attempt with gfx12-style packing failed parity at ~30× output scale;
the MIC tables + probe fixed it. Changes: arch guard + two `wmma_f16` overloads (wh8/wh16) +
`i8x16_to_h16` + re-indexed q/K/p.v packing and D-slot stores (softmax and staging untouched);
runtime prefix check accepts gfx11|gfx12; parity skip branch accepts gfx11. Objdump gate: 32 ×
`v_wmma_f32_16x16x16_f16` in the kernel (exactly the loop body). Parity 3/3 PASS (`hip_prompt_attn_wmma`,
upstream's own test, first time it runs on gfx1100): vs FP64 3.3–4.5e-06 (old kernel 1.8–2.1e-06,
same class), kernel **4.17×/4.91×/4.27×** at 32K×2048q / 1500 / 2100×256. Amdahl at the measured
share: **~+19–21% e2e prefill projected at 32K/128K** vs the ≥5% gate. Paired e2e A/B running
(`scripts/rocm/ab_pa_wmma.py`: 5 pairs × 1k/32k/128k, arms alternate, frozen gates ≥5% prefill
@128K, decode ≤3% cap) → `ab-pa-wmma-20261004.json`. Records:
`record-pa-wmma-rdna3-20261004.json`. Production 8081 was stopped for the measurement window per
the established campaign pattern (start script verified present for restore).

**Track B — #505 IQ2_XS anomaly: expert-row kernels EXONERATED; the packs differ by a per-tensor
format lottery; 24 GB full-residency is arithmetically impossible at any quant.** (1) Kernel
decomposition on the card (`iq_multi_parity --bench`, `native_grouped_parity --bench`, synthetic
blocks): IQ2_XS is **16% faster** than IQ3_XXS at the single-token mmvq shape (21.8 vs 26.0 µs)
and within +1.6% at every grouped-expert shape — the gfx1100 row kernels match the official
ordering, so the ~30% end-to-end decode inversion lawsir reports cannot live in them. (2) The GSQ
packs' HF `rco-allocation.txt` manifests show the RCO allocator assigns per-tensor formats within
a budget, so per-token-touched tensors (attn projections, the shared expert run every token) carry
DIFFERENT format mixes per pack (e.g. attn+proj: IQ2_XS 93×IQ4_XS+15×IQ3_S+12×Q6_K vs IQ3_XXS
30×IQ4_XS+22×Q4_K+8×Q5_K+48×Q6_K) — any remaining differential must come from this lottery or a
per-format decode kernel cost not yet priced. (3) Whole-pack sizes: IQ2_XS 63.35 GiB, IQ3_XXS
70.63, IQ3_S 77.88, Q2_0 61.86 — **the "IQ2_XS fix enables 24 GB full-residency" lever is
arithmetically dead** (correction owed to TASKS C2/STATUS lever lists); the anomaly stays relevant
to 48 GB boards where IQ2_XS is the natural resident quant. `record-505-iq2xs-decomposition-20261004.json`.
Upstream posts (both tracks) are drafted and user-gated.

User-supplied references evaluated this session (donors/notes, none adopted): AMD Matrix
Instruction Calculator (USED — the authoritative layout source), legacy-rocm-build#6025 (the
lane=row WMMA transposition trap, gfx12), Repeerc/flash-attention-v2-RDNA3-minimal (rocWMMA FA2 on
7900 XTX, +10–40% vs ZLUDA SDPA — external corroboration that RDNA3 WMMA attention pays),
adelj88/rocm_wmma_gemm + rocm_sgemm (raw-intrinsic GEMM, rocBLAS-competitive, tuning
methodology), seb-v + siboehm matmul write-ups (optimization-pass reading), ROCm
MoE/grouped-GEMM set (hipBLASLt ext, CK grouped GEMM, torch grouped_mm, RDNA4 matrix cores,
rocBLAS-Examples, hipfort).

**e2e verdict (same session, 33.4-min campaign): both gates PASS — the arm is strictly better at
this geometry.** 128K prefill **+22.8% median (5/5 pairs, per-pair 1.212–1.253)**, 1513.8 →
1863.2 tok/s; 32K +17.9% (5/5); decode flat (1k +2.3% noise, 32k −0.03%, 128k medians equal —
well inside the 3% protection cap). Attribution confirms the mechanism: the b arm's qsa-attn share
drops from 26–27% to 8.4–10.6% of prefill GPU time. Kept **opt-in** (`STRATA_HIP_WMMA=1`,
upstream's contract); default path byte-identical upstream behavior. Committed locally as
`831812b` on `feature/prompt-attn-wmma-rdna3` (2 files, +135/−12, off upstream main 99f3dbd —
the #368 PR shape; NOT pushed). Production 8081 restored and health-ok after the window.
`record-pa-wmma-e2e-verdict-20261004.json`. Remaining prefill terms at 128K (now 1863 vs RTX
5090's 5779): the MMQ expert path and hc/gdn phases. Upstream posts (PR #313 data + PR offer) are
user-gated.

## Session 18 continuation (2026-10-04 night): 0.1.39 refresh + production opt-in + PR #786 + fused-native verdict

**Upstream 0.1.39 refresh, arm rebased, production OPTED IN (user-directed).** Branch rebased onto
upstream main `6f32ec0` (one trivial conflict; rerere disabled per the standing rule) → commit
`49969d1`; rebuild green, objdump gate 32× `v_wmma_f32_16x16x16_f16`, parity 3/3 PASS (4.05–5.12×).
serve-8081.json now points at `build-pa/strata` (0.1.39 + arm) with `STRATA_HIP_WMMA=1`;
verified live (arm print, table 100500, health ok, 0.1.39 session line). Rollback:
`serve-8081.json.bak-nightly-0139-noarm`. **PR #786 opened** (current main, measured tables,
addresses the maintainer's two #313 review points; opt-in contract) + a measured-data comment
posted on #313 (comment 5981366917).

**Fused-native expert path (STRATA_PF_FUSED=1): REJECTED with a safety finding.** The local-model
recon surfaced the env-only lever for the ~31% expert bucket; the gate chain verified
(moe_fused.cu:411 + moe_fused_iq.cu:562-599; IQ3_S/IQ4_NL covered). Paired A/B (30 cells,
production-mirrored, resident-stopped window): 128K prefill **+2.5% median (3/4)** — far below the
5% gate; 32K +1.5%; 1k −5.4%; decode noisy-flat. **One GPU HANG in the fused arm at 128K**
(HW Exception / GPU Hang, `pf-raw-b-128k-p3.log`) — rare (4/5 cells clean) but disqualifying.
Knob stays off; the MMQ expert path is near the wall on this card/quant (consistent with B0).
`record-pf-fused-verdict-20261004.json`.

**Local-model-first recon workflow (user-directed, working)**: `scripts/rocm/llm_recon.py` drives
the resident 8081 through scoped excerpt questions → `artifacts/rocm/llm-recon/`. Strong at
single-excerpt dispatch questions (exact citations; its "NOT IN EXCERPT" hedge led to the
STRATA_PF_FUSED discovery); NOT viable for multi-file kernel inventories (thinking budget exhausts;
content NULL ×3 — those were finished by hand). Every claim verified against source before use.

**hc/gdn recon completed (hand-verified)**: hc read = norm + 3 hipBLASLt GEMMs + silu + mix per
half (`prefill.cpp:1976-1994`, `gr.cu`); GDN prefill = projections (hipBLASLt) + tiled conv +
**pipelined-columns recurrence** (`kernels.cu:908-927`) whose key-head variant is **CUDA-only**
(`:915-917` — another HIP gap) + out-norm. In-tree unmeasured knobs found:
`STRATA_GDN_REC_HEADS`/`STRATA_GDN_PIPELINE` (A/B arms), `STRATA_KV_HOST_DMA=1` (#579/#613,
targets the 6.5% wait-copy).

**Next campaign (user-directed, queued)**: hc-read kernel design (154→26 µs/site evidence from
exllamav3-rocm-moe, MIT, same card+model family — donor, credited; our hc = 6.7%), epoch-barrier
GDN chunk recurrence (their gdn_core_mk; ours = 7.4%; fallback = the CUDA-only kh variant port),
device-built dispatch tables. Local resident carries the bulk reading; GLM verifies/implements.
References evaluated-not-adopted this session: beellama.cpp (KV-cache quant fork — KVarN +
precision-tail ideas noted for a future KV campaign), RaBitQ-style WHT scaling (rotation already
in-tree `fwht256_inplace_cuda`; the scaling half would be a new Strata KV format — residency
play, not speed), AITER/vLLM-ROCm (CDNA/PyTorch — rejected, same shape as FarSkip).

## Session 19 (2026-10-05): KV_HOST_DMA verdict + hc deprioritized + GDN design checkpoint queued

**STRATA_KV_HOST_DMA=1 (#579/#613): REJECTED with the mechanism confirmed.** Paired A/B (30 cells,
production-mirrored, resident-stopped 50.8-min window, restored after): 128K prefill +0.35% (4/5) —
wash at the target tier; 1K −12.5% (0/5); decode flat. The knob ENGAGED (staged mode confirmed) but
the 'wait copy' phase is unchanged between arms (5.0% vs 5.0%) — the single in-graph DMA serializes
with the window, the same mechanism the pcie-frac sweep measured. The wait-copy lever is CLOSED for
in-graph copies; only out-of-graph double-buffered submission-path engineering could move it.
`record-kvhostdma-verdict-20261005.json`.

**hc-read DEPRIORITIZED (hand-verified correction to the queued plan).** The prefill hc phase
(6.7% at 128K) is GEMM-dominated: kPfHc brackets norm + 3 hipBLASLt GEMMs + silu + mix
(prefill.cpp:1984-1994); the elementwise kernels (prefill/kernels.cu:66-206) already carry
upstream's F-1/F-2 fusions (no FP32 normalized copy; write+norm fused). Remaining elementwise
share ≤ ~2% of prefill — the donor's 154→26 µs/site claim is decode-shaped, not transferable to
T=2048 chunks. (Also recorded: src/kernels/cuda/gr.cu is a DIFFERENT gr — the prefill hc kernels
live in src/prefill/kernels.cu.)

**Donor study (exllamav3-rocm-moe gdn.cu, fetched and read)**: epoch-based grid barrier
(single-use per launch, last block resets + publishes host-advanced epoch, co-resident ≤1 wave) +
a GDN megakernel fusing b/a GEMV + conv + gates + recurrence + gated-RMSNorm epilogue.
Transferable: launch fusion (small); the REAL lever their design implies for us is the
**chunk-parallel (WY-form) delta rule** to break gdn_rec_cols_pipe's serial T-walk (7.4% of 128K
prefill) — exact modulo summation order, needs its own derived parity bound. Queued as a design
campaign (the ringwait-pipelining-design pattern: checkpoint first, implement after).

**Local-model recon, final characterization**: scripts/rocm/llm_recon.py + 7 targets total.
Works: single-excerpt dispatch questions (moe-dispatch verified exact; led to the STRATA_PF_FUSED
discovery). Fails: multi-file inventories AND ~430-line single files (thinking budget exhausts,
content NULL ×4). ~215-line excerpts are the reliable envelope. hc-kernels-a delivered (honest
NOT-IN-EXCERPT behavior; its line citations for the WRONG gr.cu verified as a different file).

**Re-ranked prefill board (128K, arm-on, 70.3 s timeline)**: qsa select 14.8% (FIRST: verify which
select arm 0.1.39 runs on gfx1100 — its own WMMA select is gfx12-only per qsa_select.cu:421, our
#368 kernel 11.6× may be available) → GDN chunk-parallel recurrence 7.4% (WY design campaign) →
expert bucket ~31% (measured near-wall, closed) → hc 6.7% (GEMM-dominated, ≤2% elementwise,
deprioritized) → wait copy 6.5% (CLOSED, in-graph DMA serializes).

**Select-arm verification AUTOMATED and ANSWERED (2026-10-05).** `scripts/rocm/verify_select_arm.py`
(three layers: source dispatch greps → binary symbol + v_wmma count in qsa_select.cu.o → runtime log
scan with a stale-generation guard) — `verify-select-arm.json`. Verdict: gfx1100 select runs the
**warp kernel**; `STRATA_SELECT_WMMA=1` is a no-op on this card in 0.1.39 (the only WMMA select
kernel is gfx12-gated, `qsa_select.cu:1031`); the "prompt scorer arm = wmma" log line is a STALE
print from the pre-0.1.36 fork builds (append-mode log; 0.1.39 has no select print at all). The
#368 gfx11 kernel (11.6× scorer, +9.8% e2e @128K measured on 0.1.33) is NOT in the production
binary — it lives on the `pr-qsa-select-wmma-rdna3` branch (0.1.38 rebase, mergeable-clean).
**Next lever confirmed: bring #368 onto the current line** (rebase onto 0.1.39+arm or merge the
branch), verify parity + the select phase drop, paired A/B.

**Select campaign close-out (this line, 2026-10-05 morning).** Three deltas beyond the verdict
already recorded above: (1) **#368 refreshed upstream** — branch `pr-qsa-select-wmma-rdna3`
force-pushed to `a3cf018` = upstream/main 6f32ec0 + the two select commits cherry-picked (the PR
stays disjoint from #786), compile+parity-gated in a scratch build (`build-sel`, parity all-PASS)
before the push; measured-numbers comment posted (5985569138). (2) **GR_V3 is INERT on 0.1.39** —
the engine prints "STRATA_GR_V3=1 needs the two-half split on this card, which fails its checks
(#375): the default read is used"; the +2.79% decode attribution from 0.1.35 does not hold for the
env key on this line (harmless but dead; re-measure or remove for hygiene — small follow-up).
(3) **Production verified on BOTH WMMA arms** — current-generation prints for the select arm
("prompt scorer arm = wmma (gfx1100, opt-in)", fires once n_kv clears the top-k width bound; a
61-token smoke short-circuits select to the identity and never prints) and the prompt-attention
arm. `record-select-wmma-verdict-20261005.json` + `ab-select-wmma-20261005.json`.

**Session 22 addendum — agent session (2026-10-05 morning), the #368 land + records.** Complementing the
close-out above, five deltas: (1) **Parity on the nightly toolchain, demotion recorded BEFORE landing**
(PORTING.md §18.2, per the §7 rule): the warp-arm host-model contraction pin is demoted to a diagnostic
(six shapes enumerated; on 10.2-clang no host-expressible shape fits — best fma-left 14412/15512) because
the pin is a reference-isolation device on a compiler scheduling artifact, and the only flag that would
restore it (`-ffp-contract=off` on the kernel TU) changes the production arithmetic the A/B binds to. The
arm's hard gates are untouched: fresh run `logs/parity-select-wmma-20261005.log` all-PASS — worst oracle
utilisation 11.0%, 0 differing selection IDs, bitwise tail, canaries, negative control fails as required;
the one-hot case pins ALL six patterns bitwise (machinery intact). (2) **Fork main landed**: branch
`feature/select-wmma-rdna3-0139` (9687100 port + 972a637 parity harness) merged → `0731865`, origin docs
merged → **`fc19bf7`**, pushed; kernel sources verified bit-identical to the verified branch (empty diff).
`build-pa` rebuilt from merged main (exit 0, `logs/build-pa-merged-main-20261005.log`) — the next restart
picks up autotune machinery (session 21: defaults already fastest — behaviorally neutral) + the select arm.
(3) **Upstream**: PR #856 opened from the fork line, then CLOSED in favour of the refreshed
`pr-qsa-select-wmma-rdna3` @ `a3cf018` (disjoint-from-#786 path, per the close-out above) with a pointer
comment; fork main + branch remain pushed. (4) **GR_V3 inert — seconded, and the A/B is unaffected**: both
A/B arms carried the dead env equally; hygiene follow-up = drop `STRATA_GR_V3` from `serve-8081.json` at
the next restart window (production restart = user's call). (5) **WY/GDN design campaign opened** — donor
re-sourced and pinned (phoenixhaxor/exllamav3-rocm-moe@main: `gdn.cu` 2792 ln + vendor/fla
`gdn_chunk_fwd.py`/`gdn_wy_fast.py` MIT copies, sha256 in `artifacts/rocm/llm-recon/donor/`); resident
recon batch running (`scripts/rocm/llm_recon_wy.py`, 14 single-excerpt targets within the ~215-line
envelope, all first-pass-only); design checkpoint to land as `artifacts/rocm/wy-gdn-design-20261005.md`.

## Session 19 continuation (2026-10-05 morning): GR_V3 hygiene + the WY-GDN campaign opened (C2 done)

**GR_V3 hygiene done**: the inert `STRATA_GR_V3` key removed from serve-8081.json
(`.bak-grv3-inert` rollback; no runtime change — the engine already uses the default read; takes
effect at next restart). `#375` is the upstream gate that fails on this card.

**WY-form chunk-parallel GDN: design checkpoint written and C2 (the algebra proof) is GREEN.**
`tasks/gdn-wy-design-20261005.md` carries the full derivation from the kernel's actual update
(`S_t = g_t S_{t-1} + k_t β_t(v_t − g_t k_tᵀS_{t-1})`, scalar per-head decay): prefix decays, the
C×C S₀-INDEPENDENT triangular system for u (= δ), the S₀ correction factored as
`M = A⁻¹(diag(βg·dec)·K)` so every per-column step is a GEMV (the device kernel column-splits
exactly this), outputs as rank-C sums. `gdn_wy_check.cu` (new parity target, HIP-portable) proves
the chunked algebra against the engine's serial kernel at double precision: **8/8 fixtures
≤ 3.1e-13** (T around the C=64 boundary, no-decay, fast-decay, β=0, T=4099). The mid-derivation
erratum (u depends on the column through S₀) is handled by the standard split form — recorded in
the doc. Committed `403313f` on `feature/select-wmma-rdna3-0139`. **Next (C3)**: the device WY
kernel behind `STRATA_GDN_WY=1` (columns exactly as the host reference), then bench (the harness
has T=8192/32768) and the frozen A/B — expected ≈ +5% at 128K prefill if the 7.4% recurrence goes
4×+.

**WY/GDN checkpoint LANDED (2026-10-05, same session).** `artifacts/rocm/wy-gdn-design-20261005.md`:
the recurrence contract verified line-by-line (S_t = g_t(I − β_t k_t k_tᵀ)S_{t−1} + β_t k_t v_tᵀ,
S=128 square, HK=16/HV=48, T=8192 prefill chunks, shared out-norm outside the arm's diff surface);
WY-form decomposition + in-block kernel plan (BT=32 first, no global staging — fla's fused-KKT
corroborates); reassociation-ONLY parity bound (no dtype change — 2⁻¹⁶-class, no FP16 rounding),
fp64 oracle + negative control + canaries, serial arms keep bitwise; gates PRE-DECLARED: adopt
opt-in iff 128K prefill ≥ +2% median AND decode within the frozen 3% cap. Donor re-sourced and
pinned (phoenixhaxor/exllamav3-rocm-moe@main, MIT, sha256 recorded). Resident recon 10/14 with
content (4 envelope NULLs, all non-load-bearing). Implementation rows WY1-WY5 QUEUED behind this
checkpoint + a user go. **Session 22 next action: user go/no-go on WY implementation.**

**Reconciliation (2026-10-05 08:3x, same session).** The WY design checkpoint above landed as a
consolidated document AFTER a concurrent session had already begun the implementation on
`feature/select-wmma-rdna3-0139` (unpushed WIP): C2 `403313f` (host fp64 algebra proof — its
first green was vacuous, fixtures wiped k/v; real-v 8/8 ≤ 3.1e-13) and C3 `144d1a8` (device
kernel WIP; open: token-1 ~1e-4 fp64 mismatch, T=1 device NaN). The doc's §8 now maps the
C-states to the WY rows; §4 carries the CMake HIP-scope lesson; §5 keeps the C3 session's
pre-declared gates (≥+5% @128K prefill median, decode ≤3% cap). Branch untouched by this
session beyond 972a637 (origin still at 972a637; WIP stays local by design).

**C3 update (2026-10-05, parallel session): the device kernel is GREEN and integrated in-flight.**
A second session picked up the WY debugging where my checkpoint left off and found the actual
T=1 NaN root cause: my strided staging loops ran on `threadIdx.x` (c ∈ [0,32) under the
dim3(CB, RG) launch) instead of the flat thread id — **kt and A were 3/4 unwritten**
(uninitialized LDS fed redk rows 1..3; the redq path read global and stayed clean, which
localized the symptom). Commit `9f767ef` fixes the stride to `tid = rg*CB + c`, makes the
error metrics NaN-aware (std::max over a NaN diff silently reported 0.0), and keeps the
first-non-finite locator. **Green 3/3: device worst oc err 6.7e-08 vs the serial kernel; host
fp64 algebra 9/9 fixtures (worst 7.6e-11 at T=4099).** The engine integration is in flight
uncommitted: `gdn_rec_wy_kernel` in src/prefill/kernels.cu behind `STRATA_GDN_WY` (opt-in,
exact-modulo-summation-order contract, out-norm unchanged). My earlier token-1 host mismatch
resolved as the same A-line decay factor (the WYC→WY rename had silently no-op'd my first fix —
both files' fix verified in the green state). Remaining: bench (T=8192/32768), frozen A/B,
commit + upstream offer. This session STOOD DOWN from the fork tree files at this point
(two-writer race); the parallel session owns the tree, this session owns the board records.

**Session 22 continuation (2026-10-05, late morning): WY/GDN C3 closed — parity green, engine verdict REJECT-with-records.** After the concurrent session's WIP went quiet, this session took over the two open C3 bugs and closed the campaign honestly: (1) **Host token-1 mismatch: already resolved** in the committed tree (all 9 host fp64 fixtures pass; worst 7.6e-11 at T=4099/scale 29353 — the memory note was stale). (2) **Device NaN root-caused**: the WY kernel's two strided loops ran on `threadIdx.x` (= c ∈ [0,32) under dim3(CB,RG)) instead of the flat `tid` — only the first 32 columns of `kt` and a quarter of `A` were ever written; uninitialized LDS fed redk rows 1-3 (the redq path reads global and stayed clean — that asymmetry localized it). Fixed (`9f767ef`), green 3/3. Also fixed a harness metric bug: `std::max` silently DROPS NaN diffs (a NaN state read as err 0.0 for a whole run). (3) **Second device bug via extended fixtures**: fast decay underflows fp32 cumulative products to 0 → the `dec[t]/dec[s+1]` ratio becomes 0/0 NaN (serial multiplies per-step and lands at 0; fp64 host never sees it). Fix: the decay factor as the backward gate product (`g·dec[t]/dec[s+1]` = Π gates s+1..t exactly — the standalone g is consumed). (4) **WY3 gate landed**: all-heads fp64 oracle + derived reassociation bound (T/WY+1)·2⁻¹⁴·(1+scale); worst utilisation **1.0%** across all device fixtures — the bound has select-arm-class headroom. (5) **Engine wiring (WY2)**: `gdn_rec_wy_kernel` + `STRATA_GDN_WY` dispatch in kernels.cu (`0ee20fb`), opt-in default-off. (6) **Engine verdict: REJECT** (`record-gdn-wy-verdict-20261005.json`): 32k spot pair, WY arm **144× slower** (gdn recurrence 1318→189,878 ms; prefill 19.1→207.8 s) — the scalar-FMA port does O(w³) redundant per-thread folds with ~49 KB LDS occupancy pressure; the WY form needs tensor-core inner GEMMs (fla's tl.dot class) to pay. Full A/B moot at that magnitude; kernel stays in-tree dormant; escalation lever (wmma-ified folds) queued behind a revised checkpoint. Production 8081 stopped/restored for the spot cells per the standing pattern; healthy after, now on the branch build (WY dormant, select arm engaged).

**Session 23 start (2026-10-05, afternoon): WY WMMA-folds escalation — go recorded, revised checkpoint landed first.** The S12 escalation lever activated under the user's explicit go. Delegation mode for this campaign (user directive): the resident Qwen3.8-Flash-Next server (8081, `llm_recon_wy.py` mechanics, temp 0, `/no_think`) authors the bulk kernel/harness text as FIRST-PASS drafts; the session agent (GLM) authors the checkpoint/bound, diff-reviews every landed line against the contract, runs the gates, and owns the records. Revised checkpoint = design doc §9 + PORTING §22, landed BEFORE any kernel text: fp16-class bound `B = ⌈T/BT⌉·(2⁻⁷ + 2⁻¹⁴)·(1 + scale)` (≤8 fp16-dots per element path at 2u₁₆=2⁻¹⁰ each, fp16×fp16 products exact in fp32; fp32 2¹⁰-term class unchanged; chunk-compounded; (1+scale)-normalized as S12), explicit negative control (+2.5·B fails), fp32 arm held inside the OLD bound as regression canary. Gates re-affirmed unchanged (design §5 authority): adopt opt-in iff 128K prefill ≥ +5% median AND decode within the frozen 3% cap. Pre-flight: fork tree clean at `0ee20fb`, 8081 healthy (qwen3.8-flash-next loaded).

**Session 23 addendum (2026-10-05, ~10:45): bound N amended 8→16 BEFORE any kernel text landed.** The r1 Qwen draft review (findings: 4 transposed matrix_b operands on the Sin16/U16 GEMMs — caught at review, not at gate time; an LDS overrun from standalone pk/pq tiles; a Sin16 build race; a dead duplicate GEMM block) also showed the flat fp16-rounding path is 10 deep once the solve's lineage propagates through the output GEMM — undershooting §9's originally recorded N=8. §9/PORTING §22/TASKS S13 amended to N=16 (per-chunk 2⁻⁶) with the recount recorded; the amendment happened at design review, pre-implementation, which is the only place a bound may move. Guard-dance prerequisite verified: `kernels.cu`'s `<cuda_runtime.h>` resolves to `include/strata/hip_compat/cuda_runtime.h` → `intrinsics.hpp`, so the rocWMMA macro lift/restore dance works in the prefill TU. Round 2 fix prompt issued to Qwen.

**Session 23 close (2026-10-05, midday): WY WMMA-folds verdict REJECT — parity GREEN, performance negative, lever honestly closed.** The S12 escalation lever was implemented and gated end-to-end: revised checkpoint §9 + pre-derived fp16-class bound (amended 8→16 at design review, pre-kernel) → resident-Qwen3.8 bulk codegen (3 rounds, `enable_thinking=false` honored; every landed line diff-reviewed — the review caught 4 transposed `matrix_b` operands, an LDS overrun, and a Sin16 build race before they ever ran) → `gdn_wy_wmma_check` parity **GREEN: worst bound utilisation 3.5%**, negative control trips, fp32-arm regression green. The gate caught three real kernel defects (P-GEMM self-clobber of its own a_frag; state GEMM computing one 16-row tile per 32-row warp — half the state; fp16-domain overflow on random-sign long cells). Kernel bench: 80.1 ms per 8192-chunk. **Engine spot pair (32k, ×2 reproducible, arm print verified): prefill 1699→1003-1035 tok/s (−39..−40%), gdn recurrence phase 6.8%→38.7%** — the profiler phase is the all-layers total, so per layer-chunk the wmma folds are ~6× SLOWER than cols_pipe (an intermediate "4.1× faster" bench read wrongly compared one-layer to all-layers numbers; the engine data corrects it). **Verdict: REJECT per the pre-declared gates; A/B ladder moot** (`record-gdn-wy-wmma-verdict-20261005.json`). Kernel stays in-tree opt-in default-off (`STRATA_GDN_WY=1` → wmma arm, arch-guarded fma fallback; env-unset dispatch byte-identical; `gdn_wy_wmma_check` protects it). Production 8081 stopped/restored per the standing pattern; healthy after on the rebuilt build-pa (arm dormant). Remaining levers recorded in the verdict json (C=32 LDS-side restructure; GEMM fusion; or closing the WY lever for this geometry — cols_pipe at 6.8% caps the prize at ~5% even at infinite recurrence speed).

**Session 23 decision (2026-10-05, afternoon): WY lever CLOSED.** Per the user's call, the chunk-parallel WY lever is closed for this geometry — both implementations are measured with records (FMA folds 144× slower, WMMA folds ~6× slower per layer-chunk than cols_pipe) and the theoretical ceiling (~5% of 128K prefill, at infinite recurrence speed) does not justify further kernel work. The two kernels stay in-tree opt-in default-off and parity-protected as standing negative evidence. Branch `feature/select-wmma-rdna3-0139` (4 commits: fp32 arm, wmma arm, parity fixes, bench) remains local-only pending the user's merge/push go.

**Session 23 continuation (2026-10-05, afternoon): WY branch merged to fork main (pushed), production swapped to the merged tree, YaRN gate opened.** Merge fc19bf7→c81936d (7 WY commits: fp32 arm, wmma arm, parity fixes, bench; the two parity harnesses now on main). Found + fixed en route: this morning's build-pa rebuild had been made from the branch tree and LACKED the autotune line (decode_tuning.cpp et al — behaviorally identical since autotune keeps defaults, but provenance-wrong); build-pa rebuilt from merged main (11:30) and 8081 swapped per the standing pattern, healthy. Memory correction: the "gfx1100-100500 table not on fork main" note was stale — the table landed on main in f3d6827; what remains open on it are the upstream PRs. **YaRN gate (Y0) done — FEASIBLE via KV streaming:** the model is natively trained at 262,144 and yarn was needle-validated to 512K on the pre-GDN 0.1.17-era NV stack; the current stack has never served >131,072. VRAM is already 23.3/24 GiB at 64K, so 262K/512K all-resident cannot fit — but `--kv-resident` streaming keeps VRAM flat (whole K/V in pinned RAM: 3.1/6.2 GiB int8) at today's 97.6-99.7% block-hit rates. Checkpoint + pre-declared gates (needle recall at depth 50%, decode ≥60% of the 128K cell, prefill ≥400 tok/s at 512K) + manifest in `artifacts/rocm/yarn-512k-feasibility-20261005.md`; **ladder Y1 awaits the user's approval of the manifest** (~2-3 h, two stop/restore windows, no downloads).

**Session 23 final (2026-10-05, evening): YaRN ladder run — REJECT on the decode gate, with the hard part proven.** Y1 executed per the approved manifest: G0 green (rope_parity on merged main; drafter verified at mtp.cpp:515 to inherit the scaled config), 256K/512K workload files built (2×/4× canonical stream), then the ladder on the production pack with 8081 stopped/restored. **Needle recall is CORRECT at both tiers — 256,529 and 523,776 prompt tokens at depth 50% both FOUND** ("meadow-copper-504"), prefill 1901 tok/s @262K and 1536 tok/s @512K (streaming costs just 23% at 4× context, VRAM flat at 23.1 GiB via --kv-resident). **The decode gate fails: 28.0 tok/s at 512K vs the pre-declared 37.1 floor** — mechanism measured, not guessed: MTP draft acceptance collapses 0.803 → 0.056 at depth (224/234 rounds accept nothing; q4_0's unchanged decode rules out streaming bandwidth; a spec-off cell is impossible — native IQ packs require --spec ≥2). **Verdict: REJECT as pre-declared, no gates moved** (`record-yarn-ladder-20261005.json`). 8081 restored healthy. The 512K capability is real on recall+prefill; the missing piece is a depth-capable drafter — that is a model/training lever, not an engine knob.

**Session 23 closing item (2026-10-05, night): QSA-attention recon — premise corrected, ranking displaced; TheRock 7.10 dropped.** The Rock 7.10 refresh was DROPPED by user decision (recon only; the current stack is itself TheRock-built ROCm 10.2.0a nightly). The QSA-attention recon (checkpoint-first, per the rocwmma-qsa profile-first rule) corrected its own premise: production qsa-attn share at 128K is **9.9%**, not 18.7% (that number was a non-production arm's 32k profile). Same-window A/B vs `STRATA_PROMPT_ATTN_OLD=1`: **23,459 ms (27.5%) → 6,553 ms (9.9%)** — #786's WMMA arm already banked 3.6× on the term and 22.7% end-to-end prefill. Selection dump (STRATA_QSA_DUMP): ~68× cross-query locality, achieved logical-read rate ~131 GB/s vs ~960 peak — the kernel is issue-bound, not memory-bound; the remaining headroom is INT8-MMA precision-contract work, ranked below the now-dominant **MoE GEMM cluster (~36.5%: gate/up 17.4 + down 9.1 + dequant 5.8 + combine 4.2)**. No candidate opened; the checkpoint displaces the campaign it opened (`artifacts/rocm/qsa-attn-recon-20261005.md`). Board queue after this session: MoE-GEMM-cluster recon (new top target), upstream PR #888 watch, depth-drafter (model lever, parked).

**Session 24 (2026-10-06): INT8-Q feasibility fixture run to verdict — REJECT on the pre-declared precision gates; the QSA-attention INT8 lever is closed.** The checkpoint §7 fixture (tests/hip/qsa_int8_q_fixture.cu, committed to the fork at **65a3c8f**, log `artifacts/rocm/qsa-int8q-fixture-run.log`) compares three score paths on one 64×16×256 tile: device rocWMMA int8→i32 (the candidate), the fp16 hi/lo incumbent emulated in double, and fp64 truth. Four fixture bugs were found and fixed en route (warp-gating barrier divergence; B-operand read past the staged LDS tile; a single k=16 step covering only ¼ of each 64-group; a host k_deq aliasing that first produced a fraudulent bit-zero "PASS" — a pass with exactly-zero deviation against an independent error class is a broken reference, now a banked trap). Final state: double-launch probe 0/1024 diffs (deterministic); arithmetic isolation rel RMS **4.943e-08** — the int8 MMA + fp32 fold path is integer-exact; the gates fail on the QUANTIZATION class, not the arithmetic: softmax max weight Δ **9.545e-2** (bound 2e-2), output max |Δ| **1.551e-1** (bound 2e-2), score-domain max |s8−hilo| 4.07 with hilo ≡ fp64 truth to 4 digits. Independent numpy replication of the same distributions puts the class at max ~3.2 — the measurement is honest. Rejection is structural: int8 per-64 max-abs gives bulk elements a step of (group max)/254, i.e. D·2⁻⁸ relative error under the 12×-outlier class (~27% of groups affected), while the hi/lo incumbent is ~2⁻²². No kernel integration, no engagement cell (verdict §7.1 of `qsa-attn-recon-20261005.md`; handoff addendum 3). Banked: the green int8 rocWMMA fixture pattern (per-warp m-tiles, 4-k-step i32 accumulation, per-group scale fold, double-launch probe). Unverified: nothing material — the device path is proven exact and the deviation cross-checked. Board after S24: MoE GEMM cluster remains top target with its fused-WMMA candidate already REJECTed at parity (PR #999 open upstream) — the cluster's next candidate needs a new recon angle; upstream PR #888 watch; depth-drafter parked (model lever).

---

## Session 25 (2026-10-06): upstream 0.1.40 — history rewritten, ALL our PRs closed, fork content replayed as `refresh-0140` (build green)

**Upstream state.** Upstream released **0.1.40 / v0.1.40.1** (`82f46a8`) and **rewrote main's history** (fetch = forced-update; `6f32ec0` is no longer an ancestor; no merge base with fork main — a plain merge is impossible). Content: the **#783 multi-token decode batch (PR-a..k)**, the Aurora gfx1151 merge (incl. upstream's OWN gfx11 fused-native matrix-core arm `STRATA_NAT_W11` and the `gdn_recurrence_variant` dispatcher), **upstream's own gfx11 WMMA arms under the same opt-in keys we proposed** (prompt attention `601b776`, select scorer `92dde2f`), the **gfx1100 hipBLASLt-100500 table landed as `d74e110` — authored "Zac Chen", byte-identical to our #755**, the serve env-block already upstream, serve fixes (#1012/#1058/#997), a restructured README, and the dma_batch nvcc fix.

**Every one of our PRs changed state today (gh-verified).** #755 **merged for 0.1.40**; #786 **superseded with credit** ("your kernel and layout work was the starting point"); #744/#319/#745/#888/**#999 auto-closed by the maintainer's history cleanup** — explicitly "**not a rejection**", with re-file instructions (fresh branch from new main, cherry-pick own commits, new PR). None of the five auto-closed contents reached 0.1.40 (no autotune tool, no rates row, no bench folder, no WY, no rocWMMA fused arm upstream).

**The refresh (branch `refresh-0140` = v0.1.40.1 + 21 curated commits; main untouched, nothing pushed).** Carried: both bench folders, `qsa_locality.py`, the **full WY/GDN chain** (the one real conflict — 0.1.40's Aurora block vs our WY insertion in `kernels.cu` — resolved by keeping BOTH; no symbol collision; `STRATA_GDN_WY` dispatch auto-merged into the variant-0 path), the version-describe, the fused-WMMA + INT8-Q fixtures (self-contained), the CI gate fix, and docs (banner/port-status/onboarding re-applied with the 0.1.39-line numbers explicitly scoped and stale claims corrected). Dropped as **superseded upstream**: the prompt-attn arm, the select arm (the session-25 "re-file" queue item is dead — upstream ships the arm), the rocWMMA fused kernel (upstream `STRATA_NAT_W11` fills the slot; ours was engagement-REJECT dormant; fixtures kept), and the **autotune port** (#783 rebuilt the exact kernel machinery its ROWS templates wrap — a 668-line structural conflict = a re-port campaign, deferred with a record). Also dropped: old-era survivors with no callers or already-upstream (`append_chunk`, `direct_file.*`, telemetry delta, server.py env-block).

**Build gate GREEN, host tests 21/22.** Three build traps found and fixed en route (all recorded): (1) configure auto-picked the **system Fedora clang** as the HIP compiler — its builtins lack `__builtin_elementwise_exp10` against the 10.2a SDK headers; must pin `CMAKE_HIP_COMPILER` to the SDK's `amdclang++`; (2) `STRATA_PREFILL_MMQ=ON` is load-bearing on 0.1.40 — the HIP `strata_mmq` library and the HIP branch's `_strata_gg` set exist only under it; (3) fork CMake fix `d8d3529`: the fused-WMMA fixture target sits at top level where 0.1.40's `_strata_gg` is not visible — it now resolves the ggml path itself. Host CTest subset 21/22; the one failure is upstream's own declared gate ("--vram-elastic ... CUDA-only for now").

**Unverified (honest):** every GPU claim on the 0.1.40 line. The card window (stop 8081 per the standing pattern) owes: the full GPU CTest/parity set, upstream's own gfx11 arms' parity on **gfx1100** (upstream validated them on gfx1151 only), the paired A/B upstream-arms vs production's 0.1.39 arms, and the #1008 decode-regression check (likely answered by #783). Production ($EXP, 8081) still runs the 0.1.39-line engine with both WMMA arms — untouched, as is the `Strata/` anchor. Record: `artifacts/rocm/results/record-refresh-0140-20261006.json`.

**Session 25 continuation (2026-10-06 evening): the GPU window — parity green, upstream's arms verified on gfx1100, #1008 answered, 128k blocked by desktop memory.**
Window 19:04–19:31 (8081 stopped/drained in ~10 s/restored at the end, health 200 + smoke "Paris" finish=stop). **Parity/CTest on build-0140: 86/87 PASSED + 2 declared skips + 1 declared CUDA-only-by-design failure** — including upstream's OWN `hip_prompt_attn_wmma` parity passing on gfx1100 (3.66 s, first time it runs on this card), the carried WY checks, the fused fixture, and upstream's Aurora gfx11 fused tests (`hip_prefill_fused_iq` 54 s). **Upstream's arms statically verified in the gfx1100 code objects**: prompt-attn 48× `v_wmma_f32_16x16x16_f16`; select 48× `v_wmma_f32_16x16x16_bf16` — a genuine arithmetic-class difference from our fp16 select arm. **Paired A/B (production 0.1.39+our arms vs 0.1.40+upstream arms, engine-vs-engine, same args/env/table, versions stamped per cell): #1008 answered — NO decode regression on gfx1100 in 0.1.40** (decode medians 67.1 → 69.5, paired +2.5% median, 3/5 pairs; n=5, one noisy pair). The b-arm prompt-attn engagement prints at runtime. **Blocked**: the 32k/128k tiers — the first 32k cell was kernel-OOM-killed at a 48.5 GB peak against 49 GB MemAvailable (browsers hold ~19 GB; journal `run-p2319820 oom-kill`), so the window re-scoped pre-declared to 1k-only and the arms-vs-arms 128K comparison (where select was 11.8% of prefill) waits for a ≥66 GB window. Launch-gate amendment recorded in the driver (45 GB, cell-replaces-resident 1:1, MemoryMax-capped). Qwen reviewed the driver pre-window (findings adjudicated non-corrupting). Record: `record-refresh-0140-20261006.json` §gpu_window_20261006.

**Session 25 final (2026-10-06 night): the deep window — the select phase IS the whole 0.1.40 delta, and it favors our arm.**
Window 19:58–20:22 with browsers closed (88 GB available; the frozen 66 GB gate restored per its own terms; 20/20 cells rc=0 in 20.7 min; production restored + "Paris"). **Paired engine-vs-engine at the deep tiers: 0.1.40+upstream-arms is −2.27% prefill @128K (4/5 pairs in a tight 0.977–0.981 band) and −1.26% @32K, decode mixed/small.** The 128K phase table gives the mechanism exactly: **`qsa select` 1934 ms (3.2%) on production vs 3176 ms (5.1%) on 0.1.40 — our fp16 select arm is 1.64× faster than upstream's bf16 arm on gfx1100; every other phase is within 0.4%.** Production's select phase reproduces its recorded 1952 ms (session-19 provenance). Also recorded: quiet-desktop absolutes run +15% over the loaded-desktop historicals (2148–2153 vs 1863 @128K) — desktop memory pressure costs double-digit prefill on this DRAM-bound engine; paired verdicts unaffected. **Swap-decision data: as-is, 0.1.40 buys nothing and costs the select phase; re-landing our fp16 select arm on the 0.1.40 line would make it ≥ production everywhere measured.** We now hold a measured fp16-vs-bf16 select comparison upstream may want — user-gated post. Records: `ab-refresh0140-deep-20261006.json` + the record JSON §gpu_window2_deep_20261006.

**Session 26 (2026-10-07): autotune-UI question decided — NO UI, recorded.** Asked whether a UI should be layered on the 8081 webui for our autotune. Decided NO (user-approved): the workflow is batch (tune → byte-exact verify → e2e confirm → table → config env → restart), not interactive — the `./strata-rocm` CLI + records interface is the right shape; the webui is upstream's `serve/server.py`, and any patch on it is a permanent fork surface paying the replay tax on every refresh (the same reason the autotune port was dropped from the 0.1.40 refresh); the re-port itself is DEFERRED (TASKS R4) so there is no workflow to front; and rates must be record-sourced (the webui's live tok/s was already caught being a non-rate). Revisit trigger: when the re-port lands AND tuning becomes recurring (SDK-refresh ⇒ retune gate), extend `./strata-rocm status`/`report` with tune state; the maximum concession is a static read-only page over `artifacts/rocm/results/` JSON — never a webui patch. Record: `artifacts/rocm/results/record-autotune-ui-decision-20261007.json`; TASKS R13.

**Session 26 continuation (2026-10-07): the PR re-file round — 5 new upstream PRs carrying our authored commits.** Upstream's 2026-10-06 history rewrite (main force-pushed to `82f46a8c`, all commits re-landed with fresh dates) auto-closed our open PRs #319/#744/#745/#888/#999 with the maintainer's "not a rejection" template offering a re-file recipe — and #755 was merged for 0.1.40 (authorship kept) while #786 was superseded with credit. Executed the re-file onto the new main: rates row **#1292** (the old branch's first commit carried a stale `DETAILS.md` overwrite deleting ~193 lines of upstream text — reconstructed payload-only, refile diff +12/−2), community bench **#1293**, GDN WY **#1296** (the WY arm composed with main's new Aurora GDN kernels; hook files dropped; `gdn_wy_check` all-PASS + `gdn_wy_wmma_check` PASS 3.5% worst utilisation on the new base), autotune **#1297** (ships the ported+measured version, not the stale 0.1.27-era patch; `calibrate.py` reconciled with main's `host_worker_extras` machinery; decode_tuning_test + 34 python tests green). Plus the voluntary **#1298** — the gfx110x fp16 (rocWMMA) select arm promised on #944 — composing with 0.1.40's shipped bf16 arm (parity all-PASS on the new base, worst oracle utilisation 11.0%). **#999 NOT re-filed: superseded** — main's own Aurora fused-expert line went further than our PR's kernels; disposition commented. Cross-link comments on every old PR; GFX1100.md PR block rewritten on refresh-0140 (`09c85ff`). Every code gate run fresh on the new base (three full HIP builds via the nightly amdclang++ recipe — new trap recorded: pinning `CMAKE_HIP_COMPILER` requires a FRESH build dir, the cache-wipe silently dropped the `-D` flags). Objective served: ~12 authored commits (Zac Chen) now visible across 5 open upstream PRs. Record: `artifacts/rocm/results/record-pr-refile-20261007.json`; TASKS R14.

**Session 26 continuation (2026-10-07): the autotune seam rework, GPU verification, and the R13 tune-state surface.**
Following the re-file round: audited the staged re-port (`8d39e6a`, then rewritten losslessly as `66092b4` re-port + `23b447b` rework on `refile-744-autotune` after a concurrent-amend collision with the parallel session — both sessions' work preserved). **Audit (9 findings, `record-autotune-refile-audit-20261007.json`)**: the minder second opinion (local Strata qwen via :8390) REFUTED my baseline-bias direction and found the design-changing gap — dispatch-rerouting env knobs (`STRATA_NO_SUB16_GU` etc.) were not part of the table identity. **Rework (`23b447b`)**: kernel-selection-space token in the table header (a same-compiler kernel rework now refuses old tables as stale — the exact #783 lesson), dispatch-knob refusal, one X-macro rows list for parser/launchers/tuner, true-default tuner baseline (the engine's own entry, not the explicit-rows 8/8 pair) + the chooser-crash fix minder's diff review caught. CPU gates green (decode_tuning_test incl. 2 new refusal cases, 22+12 python tests). **GPU window (take 2; take 1 VOID — the parallel session pruned the /tmp worktrees mid-run and ctest "passed" zero matched tests from a fallback cwd; bogus lines in the take-1 log, hardened driver banked as `scripts/rocm/stage1-refile744.sh`)**: ctest strict ×4 PASS (decode_tuning_test, iq_parity, native_grouped_parity, iq_multi_parity); parity through an ENGAGED table PASS with the engagement print; refuse paths verified on card with the designed messages (wrong space 99, stale pre-space header, knob set); **tune campaign = honest null ×7** (18:20…23:20, default chain 67.9–73.1 µs, best 8/8 kept, 0 shapes changed — #783's hardcoded conclusions win on this card too); e2e correctly not run (nothing to install on a null). **Identity finding**: the same binary reports HIP runtime 70152802 (system libamdhip64) vs 71726392 (SDK lib via LD_LIBRARY_PATH) — the runtime field tracks the LOADED library, exactly what a measured table should key on; tune/run with the same library. Record: `record-autotune-refile-gpu-20261007.json` + `autotune-table-20261007.txt` + `logs/stage1-refile744-20261007T030705Z.log`. **Stage 2**: PR #1297's body appended (seam-rework section + measurements, minder-drafted, numbers verified against the record); #566 comment posted (the calibrate.py HIP surface rides with #1297, split offered). **Harness (R13's then_do, landed early)**: `scripts/rocm/lib/tune_state.py` + a Tuning artifacts section in `./strata-rocm status`/`report` (records-sourced, redaction-checked, decode shape table vs GEMM solution table named apart), PORTING.md §23 fixes the four-term language, state.json gained the tune block, the `build-pa` rollback hazard is live-flagged (a 0.1.39-era binary silently resurrects the old autotune line). **Next: await the maintainer on #1297**; the production-swap decision (and R5's A/B follow-ups) come after it moves.

**Session 26 continuation 3 (2026-10-07): the autotune next-optimization round — decode shape space CLOSED with evidence; DeepGEMM dispositioned Separate Ways.**
Through the port-strata-hip + analyze-strata-moe discipline (≤3 ranked candidates, cheapest discriminating experiment first). **Candidate 1 (staging axis — the last unmeasured decode knob, default from upstream's RTX 5070)**: paired alternating tune_decode arms, 7 expert pairs ×2 rounds — **wash for every production format** (≤0.2% on 18:x/21:x/23:x) but a real staging-OFF win for **IQ2_S gate/up (22:20 −4.59%, 22:42 −2.28%, tight repeats)** — the codebook lives in the 96 MB Infinity Cache here, so LDS staging is pure overhead for that format; mmvq staging an exact wash (13.8 µs all arms). Rows tuning stayed null in every arm. Disposition: docs + records, no new machinery (production pairs show no gain; a V2 staging axis only if an IQ2_S-class model lands). **Candidate 2 (workload re-derivation)**: `tune_decode` gained `--windows`/`--weights` (commit `33a73c5`, pushed — joins PR #1297); the real window distribution **did not exist in any record**, so it was captured: branch engine + production args + STRATA_DECODE_TIMING=1, 8 requests → **avg window width ≈2.4 (1.91–3.13), not the hardcoded proxy's 3.6**; the rows null re-confirmed under the measured mix. **DeepGEMM (user-requested donor review): NO port — Separate Ways** (ISA: RDNA3 has no fp8/fp4 matrix ops nor TMA — DeepGEMM-Ascend went to an ISA with fp8 cubes; format contract: fp8-block-scales vs fixed GGUF IQ = re-quantize; bottleneck: compute-bound GEMM — the class that already lost twice here). Reopen triggers recorded (RDNA4/CDNA, credible HIP fork, fp8 GGUF models). **Quirk nailed**: serve/server.py:589 — without a config `log` key the engine's stderr is DEVNULL (the capture lost its first two attempts to this + two script bugs, all recorded). Records: `record-autotune-nextopt-20261007.json` + five window logs + the capture. 8081 restored + healthy after every window; production config untouched. **Next: the counter-split (kernel-vs-PCIe-floor) — still the queued prerequisite for prefill tile-J work.**

**Session 26 continuation 4 (2026-10-07 evening): the counter-split ran — the tile-J candidate is REJECTED by the pre-declared gate.**
The moe-gemm-recon §3 counter-split (kernel-vs-PCIe-floor), queued since session 23 as the required-before-tiles measurement, ran on the CURRENT production stack: 3 one-shot 128K cells (build-0140, production args, `qsa-profile-131072.tokens`), parsed by `scripts/rocm/counter_split.py` (minder-drafted per the use-local-strata instruction; I caught its finditer-unpack bug and my own blob-constant arithmetic — the parser now derives 1.9517 MiB/pair from each log's own loaded line). **Verdict: the MoE GEMM cluster is 30.9% of the 128K GPU timeline (26.8 s of 86.6 s), and the routed-tail transfer floor is 21.2 s — 79% of the cluster and 24.5% of the entire prefill.** Exposed wait is only 4.3 s: 17.0 s of the transfer already hides behind compute, leaving **~5.5 s of absolute kernel slack — a 6.3% prefill ceiling even with zero-cost kernels**; a realistic tile-J win nets ≤2–3% e2e, below the §5 ≥3% adoption gate before any parity risk. REJECT-with-records, exactly the "tiles tuned against a PCIe-bound term" outcome the ordering was written to prevent. Residency today is better than the session-23 cell (14.9% vs 10.9% of routed instances) and the floor still dominates; streaming is 100% DMA with 1.3 s host staging. Re-run the split (one cheap window) only if residency ever rises materially. Records: `record-counter-split-20261007.json` + `countersplit-cell1..3-*.log`; TASKS R15 extended; moe-gemm-recon memory closed on the tile-J question. 8081 restored + healthy; production untouched.

**Session 26 continuation 5 (2026-10-07): the 512K decode refresh — the adopted config inverts past acceptance collapse; competitor comparison recorded.**
Correction first: the "#1 arms-vs-arms deep-tier" item I offered earlier today was **already done** the evening of 2026-10-06 (`ab-refresh0140-deep-20261006.json`: 20/20 cells, −2.27% prefill @128K for 0.1.40+upstream arms, mechanism = the select phase; the swap decision it fed was executed — production runs build-0140 with the re-landed fp16 select). The runnable item was #2: the Y1 ladder's 512K decode cell re-run as a declared engine+config-only comparison. **Result: 23.35 tok/s median (23.29/23.40) vs the 28.01 anchor = −16.7%** — the adopted decode config (spec 5 + SH_STREAM), depth-validated flat to 128K, **inverts past the MTP acceptance collapse** (~1.16 tokens/round): wider windows buy more verification per accepted token. Prefill improved +3.0% (1583). The pre-declared decode gate now fails harder (37.8% of the 128K rate vs the ≥60% floor) — **the 512K REJECT strengthens**; a spec-2 probe cannot reach the floor. Competitor side recorded with its regime caveat: exllamav3-rocm-moe's "up to 65 @512K" is content-driven n-gram drafting (does not collapse at depth), not a like-for-like prose number; our short-context prose decode remains faster than their 45–50 claim. Records: `record-yarn512-refresh-20261007.json` + two cell logs; minder drafted the comparison paragraph (numbers verified). 8081 healthy after the window; production untouched.

**Session 26 continuation 6 (2026-10-07 evening): production updated to upstream v0.1.40.2 — and the prefill re-baseline is a −29% surprise.**
User-directed refresh: upstream tagged v0.1.40.2 (= current main tip, 242 commits past our v0.1.40.1 base). Replay-never-merge executed in a fresh worktree: `refresh-01402` = 24 fork commits replayed (2 SKIPPED-absorbed — upstream took both bench/result folders into the release) + 1 critical fixup: **the cherry-pick auto-resolution silently dropped the fp16 select arm's entire `qsa_select.cu` content** (kept only CMake/hpp/parity — caught by post-replay inspection: zero f16 WMMA instructions in the tree); re-applied three-way, keep-both with upstream's new SIMT kernel, composition verified (our gfx1100-arch-gated arm tried first, falls through to gfx12 bf16). Build hit the known `STRATA_PREFILL_MMQ=ON` trap (recurring lesson). **GPU gates ×4 PASS** incl. the select arm's parity. **The profile cell is the headline: 128K prefill GPU timeline 61.2 s vs 86.6 s = −29% (~2143 tok/s)** — 0.1.40.2's prompt attention collapsed the qsa-attn phase 26.0% → 9.8%; gemm gate/up is now the largest prefill phase (16.9%); and **wait-for-rings persists at 21.8 ms/round — the graph-capture target is real on the new kernels (GO)**. The swap itself took four windows: three false failures from a broken smoke gate — **8 max_tokens against a thinking model starves content (reasoning_content fills, content null, finish length)**; the raw body finally showed it, `chat_template_kwargs enable_thinking:false` is the fix; plus one ~3-min production-down stretch from a stray diag pair holding VRAM (kill the engine child, not the wrapper pid). **Production now: 0.1.40.2-3ddb3f8 via the existing mirror, config env unchanged, rollback kept (`.bak-01402-pre` → build-0140).** Survey (the user's second ask): upstream has NOT done partial-residency window capture — their zero-doorbell graphs need 100% residency (48 GB-class), batch graphs have a live bug (#1201), DEVICE_PLAN measured −4.1% here, PDL is sm_90 — **the spike is planned and GO** (record has the two designs). New upstream levers queued for cells: `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_SPEC_GUMBEL=1`. Records: `record-swap-01402-20261007.json` + swap/profile logs.

**Correction (same evening): the "−29% prefill" in continuation 6 was a confound, now corrected everywhere.** The morning counter-split cells ran arms-OFF (minimal env, 1,513 tok/s @128K); the swap profile ran the production env arms-ON (2,143). The clean comparator — the deep A/B's arms-on band, 2,098–2,153 across 0.1.39 and 0.1.40 — brackets 0.1.40.2's 2,143–2,157: **flat, no regression, no new win**; the swap's real effect is carrying the re-landed fp16 select arm (the −2.3% bf16 deficit is gone in 0.1.40.2-production). GFX1100.md updated (`2fd3f27`): production identity 0.1.40.2-3ddb3f8, first-sweep cells (128K prefill 2,157 / decode 71.7–75.7, n=1/tier, desktop-loaded), the confession inline, median-of-3 sweep continuing merge-safe (blocked on the 68 GB desktop gate; one rerun of `scripts/rocm/rates_amd_01402.py` completes it). En-route finds: the sweep launcher's `env -u` argv stripped the SDK lib (cells ran the system HIP runtime, 2× drag — purged and re-run); production recovered from a stray-diag VRAM hold; the smoke-gate thinking-starve rule.

**Session 27 (2026-10-10): upstream 0.1.41 — history LINEAR this time; fork content replayed as `refresh-0141` (build green, host CTest run).**
User ask: update local strata to the latest upstream. Latest = tag **v0.1.41** (`fb58e0db`, 2026-10-08, == upstream/main; no newer tag). Unlike the 0.1.40 jump, `v0.1.40.2` IS an ancestor of `v0.1.41` (merge-base `e8ca9af`) — 151 commits / 258 files / +31080−470 — so the replay is ordered cherry-picks onto the tag, no forced-update reconstruction. Executed in a fresh worktree `/home/zacch/projects/strata-r141`, branch `refresh-0141` = **v0.1.41 + 24 replayed commits** (`-x` provenance; 2 of the 26 refresh-01402 commits were SKIPPED-absorbed upstream); `refresh-01402`, fork main, origin and production 8081 all untouched. Conflicts exactly as scoped: `kernels.cu` (upstream's new **STRATA_GDN_CHUNKED** WY-form chunked recurrence vs our dormant `STRATA_GDN_WY` — kept BOTH, chunked first per upstream precedence, our `!wy` kh-guard re-applied, fork note in the comment), `CMakeLists.txt` ×2 (the fused-WMMA ggml-path block and the `qsa_select_wmma_parity` target), `README.md` (fork banner/sections kept), `GFX1100.md` (delete/modify — upstream deleted the file; the fork file kept). **Gate P (post-replay source inspection — the mandatory check after the 0.1.40.2 silent-drop trap): byte-identical to refresh-01402 for every fork-touched file except the two expected conflict files; the fp16 select arm (`qsa_select.cu`/hpp/parity) replayed intact** (9 rocWMMA/gfx1100 markers, zero drop). Build **GREEN**: fresh `build-0141` dir, nightly SDK, `CMAKE_HIP_COMPILER` pin verified in cache, `STRATA_PREFILL_MMQ=ON`, ninja 349/349, `strata --version` = **0.1.41-60184598** (the git-describe stamp works). Host CTest (−`expert_cache_segmented_test`, −`isa_guard_check` declared): **93 PASS / 5 FAIL / 2 declared skips**; all four parity suites + `gdn_wy_check`/`gdn_wy_wmma_check` + `prefill_fused_iq_wmma_test` + `qsa_select_wmma_parity` PASS without the card. The 5 failures declared: `expert_parity`/`pool_test`/`ple_parity`/`platform_memory_test` = the known fixture/ulimit env-gated class; **`hip_prefill_fused_iq` = cudaMalloc OOM while 8081 held the card (confirmed on an immediate retry, 2/2 same signature) — NOT pre-declared, rerun in the GPU window decides it (not closed)**. Upstream's new `isa_guard_check` (x86 AVX-512 disassembly guard) excluded from the host pass with the reason recorded. Not touched: `strata-amd/Strata/` anchor (944b7ce, deliberately held). **Next: the user's GPU window** — strict 4-suite parity + the fused_iq rerun + static f16-WMMA check on `build-0141/strata` + the 128K verify cell in the production env (band 2098–2157 tok/s, select phase ~1.9–2.0 s); then the separate swap decision (`swap-0141.sh` pattern, `.bak-0141-pre`). Queued: `STRATA_GDN_CHUNKED` vs `STRATA_GDN_WY` A/B (upstream's arm may supersede #1296 — measure, don't assume); autotune re-port onto 0.1.41 (#1297). Record: `artifacts/rocm/results/record-refresh-0141-20261010.json`.

**Session 27 cont. (same day, independent second pass): the replay re-verified from scratch, the two skips PROVEN content-safe, GFX1100.md staged (`ac7c1fc2`).**
A re-dispatch of the same plan found the work already executed (the worktree/branch pre-existed — this pass verified instead of redoing). Results, all independently reproduced: (1) **whole-tree byte-identity** — every fork-touched file on `refresh-0141` is byte-identical to `refresh-01402` except the two expected conflict files; the fp16 select arm is intact; (2) **the 24-vs-26 deviation is now proven, not just claimed** — `7167453a`'s only real delta (fused-WMMA fixture HIP gate) exists structurally upstream-side (`if(STRATA_ENABLE_HIP)` at CMakeLists:1568; upstream touched no `.github` file in the window, so the carried workflows are complete), and `4be3d18a`'s README/GFX1100 content is present byte-identical; (3) **both GDN arms** coexist in the dispatch (~2096 `wy` / ~2102 `chunked`, fork note in the comment); (4) **ctest evidence reconciled**: the `LastTestsFailed.log` single entry is the OOM retry overwriting it — the full run in `/tmp/ctest-host-0141.log` shows 95% passed, 5 failed of 98 (4 pre-declared env-gated + `hip_prefill_fused_iq` OOM) + 2 declared skips; note `qsa_select_wmma_parity`'s host PASS (1.06 s) is the argless host-oracle path, **not device evidence** — R19 owns on-card parity; (5) premises re-verified (v0.1.41 = fb58e0db = upstream/main, merge-base `e8ca9af`, delta 258 files). Version-stamp note: the binary stamps `0.1.41-60184598` (configure-time); branch tip is `ac7c1fc2` after the GFX1100.md docs commit. Addendum: `reverification_20261010_second_pass` in the record. Nothing pushed; 8081 untouched; next unchanged: the user's GPU window (R19), then the swap decision (R20).

**Session 27 cont. 2 (same day): R19 GPU window executed — GREEN end to end; R20 (the swap) now awaiting the user's decision.**
Window ~12:20–12:50: 8081 had already been cleanly stopped at 11:26 (staged), MemAvailable 69–71 GB cleared the frozen gate, VRAM drained to 2.5 GiB. (1) **Static f16 check PASS** — the recorded llvm-objdump path sees nothing (device code lives in `.hip_fatbin`), so all 57 embedded gfx1100 code objects were carved and disassembled under `env -i`: our fp16 arm `block_scores_wmma_gfx11_kernel` carries the recorded 8× `v_wmma_f32_16x16x16_f16` signature, upstream's select arm 48× bf16 (both compiled, gfx1100 dispatches fp16). (2) **Strict parity 7/7 PASS on-card**, including `qsa_select_wmma_parity` at 50 s (the real device run — the earlier 1.06 s host "pass" was the argless host-oracle path) and **`hip_prefill_fused_iq` PASS 56 s — the session-27 OOM is closed as environmental** (8081 held the VRAM). (3) **128K verify cell exit 0**: prefill 131,071 tokens in 61,116 ms = **2145 tok/s** (band 2098–2157), qsa-select phase **1974 ms** (the 1.9–2.0 s band; 3.1 s would mean the bf16 arm), engagement print live, engine stamp 0.1.41-60184598; decode 53.33 tok/s is parity with the 01402 swap cell's identical cold 64-token shape (52.05) — not a serve-path figure. (4) 8081 restored in 28 s, health ok, smoke 'Paris' finish=stop (first attempt content=null = the recorded thinking-starve signature; `chat_template_kwargs enable_thinking:false` again required). Numbers land in the record (`gpu_window_20261010_R19`), TASKS R19, GFX1100.md banner. **Next: user decision R20** — swap 8081 to `build-0141/strata` per `swap-0141.sh` (`.bak-0141-pre`, rollback stays `build-0140`), or stay on 0.1.40.2.

**Session 27 cont. 3 (same day ~13:44): R20 EXECUTED (user-approved) — production 8081 now runs the 0.1.41 line.**
`scripts/rocm/swap-0141.sh` (template clone, three deltas recorded in its header): stop 8081 → strict 4-suite parity on `build-0141` **4/4 PASS** → config exe `r1402/build-hip/strata` → **`r141/build-0141/strata`** with `.bak-0141-pre` kept (rollback = that file → the `0.1.40.2-3ddb3f8` binary) → 8081 up in ~4 s, health ok, **engine log: `session is up (engine 0.1.41-60184598)`**, VRAM 23.3 GiB loaded, smoke 'Paris' finish=stop. The script bakes in the two lessons: smoke disables thinking via `chat_template_kwargs` (content=null+finish=length = starved = gate fail), and the piggyback 128K cell is intentionally not re-run (R19's identical cell 30 min prior on the identical binary; noted in-script, not silent). GFX1100.md banner rewritten for the production identity (rollback chain + gate trail + STAGE_PIN-unset note). Queued unchanged: R21 GDN A/B, R22 autotune re-port; nothing pushed; `refresh-01402` + builds kept for rollback.

## Session 27 (2026-10-10): the drafter-yield family measured — all five arms REJECT; production unchanged; greedy cross-restart drift found

The Tier-1 unswept flags (from the missed-flag audit earlier this session) ran as a paired
campaign on the PRODUCTION binary (`ab_draft_yield.py`, 0.1.40.2-3ddb3f8, production args/env,
5 pairs per tier-arm, arms alternate, `MemoryMax=66G` scope; resident stopped 13:55–15:05,
restored + smoke-verified). Verdicts (`results/ab-draft-yield-20261010-verdict.json`):

| arm | tier | median paired decode | wins | verdict |
| --- | --- | --- | --- | --- |
| `--lookup-chain 2` | 1k | −5.6% | 1/5 | REJECT |
| `STRATA_MTP_TOP2=1` | 1k | −1.1% | 1/5 | REJECT |
| `--mtp-hnorm stream` | 1k | −2.1% | 1/5 | REJECT |
| `--mtp-window 65536` | 128k | −4.6% | 1/5 | REJECT |
| `--lookup-chain 2` | 128k | −6.9% | 1/5 | REJECT |

Mechanisms (per-cell telemetry, raw logs kept): `mtp-window 65536` LOWERED MTP acceptance
0.717→0.677 and tok/round 2.46→2.31 while costing 4.51→5.38 ms/round — the "drafter can't see
deep context" hypothesis is falsified at 128K (the window knob is not the depth lever).
`lookup-chain` lifted suffix acceptance 0.78→0.89 but tok/round stayed flat — wider verify
windows cost more than chain proposals yield. `top2` lowered acceptance outright.

**New finding: greedy output is not cross-restart stable.** Same prompt, same binary, greedy:
base cells land in ≥4 discrete output variants across engine restarts (identical within a
block; one variant shared across two different arms). The engine's expert residency and
adaptive-tier composition differ per start (token-graph resident 8241/8245/8254 observed), and
GPU-vs-CPU expert-path rounding flips occasional argmaxes. Consequence: cross-restart
output-exactness gates are unsound on this engine; within-process exactness claims (the
windowpipe C4 verdict) are unaffected, and the exactness column of this campaign is recorded
as UNDECIDABLE rather than arm-caused.

Production config UNCHANGED (nothing cleared the +1.5%/4-of-5 gate). The `--prefill 8192`
dead-arg hygiene note from the audit remains open for the next config touch.

**Correction (2026-10-11):** the session-27 drafter-yield campaign ran the 0.1.40.2 rollback
binary (`strata-r1402/build-hip`), not the production binary — the 0.1.41 swap (R20) had already
landed at 13:44, before the 13:55–15:05 campaign window. The flags under test are upstream
engine code and the verdicts stand as measured on 0.1.40.2; they have not been re-measured on
the 0.1.41 production line. Production itself is verified on 0.1.41-60184598 (health ok, Paris
smoke, adopted config intact).

## Session 27 cont. (2026-10-11): R23 — production upgraded to the 0.1.42 line

`refresh-0142` = upstream **v0.1.42** (`61b3fb5d`, 235 upstream commits since v0.1.41) + the 27
fork commits replayed (3 CMakeLists conflicts resolved: the gfx906 swizzle probe kept alongside
our fused-int8 fixture; our fixture's HIP-guard fixup taken wholesale; the fp16-select parity
registration re-inserted). One build trap: my configure replication missed build-0141's explicit
`CMAKE_HIP_FLAGS` (`-isystem $SDK/include --hip-path=$SDK --rocm-path=$SDK`) — without it the
nightly clang pulls SYSTEM 7.1.1 HIP headers via /usr/include and `qsa_select.cu` dies on
undeclared `__ocml_*` intrinsics (the two-stack version split, again). With the flags: build
329/329, 0 errors.

Gates: ctest parity suites (iq_multi, iq, native_grouped, qsa_select_wmma) ×2 strict — 5/5 both
runs; band cells on the production config: 1K decode 76.25 tok/s, 128K prefill **2168 tok/s**
(0.1.41 band ~2145–2153), select phase 1931 ms (band ~1974), decode @128K 68.8 — all in band.
Swap: `swap-0142.sh` (`.bak-0142-pre` rollback → build-0141/strata 0.1.41-60184598), health ok,
Paris smoke finish=stop, engine stamp `0.1.42-0ebe4053`.

Two of the missed-flag audit's knobs became official HIP opt-ins in this delta (`#1479`:
`STRATA_HEAD_MIX_MULTI`, `STRATA_ONE_TOKEN_COMMIT`); single probe cells on 0.1.42 measured both
FLAT at 1K (76.5 / 76.2 vs 76.3 baseline, n=1) — they go to the probe queue, not adoption.
0.1.42 items queued for the test plan: `STRATA_ROUTE_TAIL_SKIP` (default-on CUDA, explicitly
not HIP), `STRATA_ROUTE_PRIOR` (output-changing, parity-gated), `STRATA_MMQ_RESIDENT_SORT_NE`
now open to native packs, the offline `replay_cache_policy.py` tool, spec chain re-probe
(`71eaee17`), and the #1655 community 4× RX 7900 XT 262K opt-in table.

## Session 27 cont. 2 (2026-10-12): XeStrata xe0.1.42 installed as the alternative engine (:8082)

Local, unprivileged (no dnf): the Fedora RPM sha256-verified, then extracted via
`rpm2archive` (rpm2cpio/cpio emit a truncated gzip-magic stream on this package — use
rpm2archive) into `xestrata-0142/rootfs/`; `libblosc2.so.6` fetched from Fedora 44's blosc2
2.22.0 rpm the same way (only missing dep). `start-8082.sh` / `stop-8082.sh` drive a transient
unit through the production home's serve/server.py with `serve-8082.json` (same shards, pack,
tokenizer, expert profile — read-only; `fit_max_tokens` on). Smoke window: 8081 paused ~2 min;
health ok, "Paris" finish=stop, engine stamp `strata xe0.1.42`; serve-path decode probes 76.9 /
85.7 tok/s incl TTFT (warm). Findings: (1) **`ROUTE_TAIL_SKIP` is ON by default in their AMD
build** (log: "1450 missed experts skipped since the start") — outputs differ slightly from
production by design (`=0` restores exactness); this is also our test-plan item 6 running for
free. (2) Their gfx1100 path is `oneMath (rocBLAS)` prompt GEMMs + their own DP4a kernels —
the hipBLASLt 100500 table env is inert there; it is a genuinely different kernel stack.
(3) The engine prints "48 compute units" for the 96-CU XTX — a query oddity to keep in mind
reading their numbers. 8082 is left STOPPED (single card, one engine at a time); production
restored and verified.

**XeStrata prefill gap quantified (2026-10-12, paired one-shot cells, same args, alternating
reps):** 32K prompt: ours 2076/2099 tok/s vs XeStrata 928/959 (TTFT 16.0 s vs 37.8-39.0);
128K: ours 2144/2157 vs 959/959 (TTFT 61.4 s vs 140.3) — **×2.2-2.25 slower prefill**, matching
the user's report. Binary-verified cause: their gfx1100 build has NO hipBLASLt, NO WMMA arms
(select/prompt-attention), NO PF_GEMM/PROMPT_F16/SCORES_THREAD (184 env knobs vs our 438);
prompt GEMMs ride oneMath (rocBLAS heuristic) + their own DP4a expert kernels. Their decode
side carries the compensating feature (ROUTE_TAIL_SKIP default-on). Conclusion for the
alternative's role: use 8082 for decode-side/long-residence work; 8081 remains the prefill
engine. Numbers go to the xestrata memory + config note.

## Session 27 cont. 3 (2026-10-12): PcieAuto ported to HIP, gated, LIVE in production

**Port** (`acb3deb0` on refresh-0142): upstream's in-window PCIe-share refinement (PcieAuto,
aa7b0cb) was CUDA-only by an unmeasured-only guard — the machinery is engine-generic (pool ms
per missed expert + miss counts vs the probed link rate). Guard removed with provenance;
opt-outs unchanged (layer splits off, `STRATA_PCIE_FRAC_DEFAULT=old` restores the fixed rule).
Scope honesty: XeStrata's actual stabilization commits target THEIR OWN subsystem
(`memory_read_rate`/`measure_pcie_split`/preload-thread — none exists in upstream v0.1.42),
so this is a re-implementation of their *finding* (the share should be measured, not assumed)
on our tree's mechanism. Our start-up probe was already stable (best-of-4, ±0.7%), so their
probe-retry ideas had nothing to fix here; the value is the in-run refinement HIP never had.

**Gate** (`gate_pcieauto_hip.sh`, 4 alternating starts per arm at 64K, resident paused):
no-regression PASS — prefill medians 2189 (base) vs 2180 (port), decode medians 65.4 vs 65.4,
one 59.69 noise outlier on the port side. Engagement: 0 refinements in one-shot cells (256
tokens cannot accumulate the 48 windows + 300 misses PcieAuto needs — design limitation of
the cell, not the port), so engagement was validated under production serve: a ~95K-token
request fired **update 1: pool 45 µs/missed expert (45.1 GB/s), link 28.4 GB/s → pcie_frac
0.55 → 0.25** — sane math (f = L/(L+1.85·r)), the pool reading matches the recorded 41-45 GB/s
DRAM wall, and the mechanism self-limits at 3 updates. Production 8081 now runs the ported
binary (Paris smoke green). Rollback: the env line or `git revert acb3deb0`.

**dsh-shape port DEFERRED, honestly:** XeStrata's position-aware short-read chooser
(`e76a2d7d`) sits on their own `ShortReadCost` subsystem which upstream does not have (our
v0.1.42 chooser is a fixed `--short-read` limit) — porting it is feature development
(~300+ lines + serve-loop measurement call-sites), not a cherry-pick. Queued behind the
test plan's higher-value cells; their measured 505→304 ms follow-up win is the target.

## Session 27 cont. 4 (2026-10-12): the dsh-shape chooser ported, gated — works, but our card's fixed 64 was already near-optimal

**Port** (`5690ed88`): XeStrata's measured short-read chooser (their b8a83451 + e76a2d7d, final
state) ported onto our serve loop — the structures mapped 1:1 (windows_ok/read_windows/
read_part/cuts). With `--short-read` unset, the serve loop measures both paths per part
(windows: median ms/token of the last 32 reads near the position; batched: least-squares
fixed+per-token through the last 32 short reads there; both position-aware within a factor of
2) and routes each part the cheaper way, exploring batched for 100+ ms parts read from
position ≤16384. `--short-read N` keeps the fixed behavior; the learned limit prints to
stderr. ~+150 lines in generate.cpp; server.py's PP comment unchanged in code.

**Gate v1 lesson:** total-wall follow-up timing is decode-dominated — the read-path effect
must be measured as streaming TTFT. **Gate v2** (paired serve cycles, ~25K-token conversation,
alternating short/long-tail follow-ups, TTFT via SSE): short-tail TTFT port 703-713/326-328/
348-836 vs base 821/341/432 — equal within noise, possible small win on the first post-list
follow-up (703-713 vs 821). Engagement verified by STRATA_TRACE: parts route by measured cost
(17-token answer 154.8 ms windows; 6-token line 40 ms; 72-token list answer 471.7 ms batched);
the learned-limit line stays silent until 3 batched samples exist near the position (had 2).

**The honest headline: the win does not reproduce on gfx1100 — because our windows path is
~7-9 ms/token where the 4070/9060 XT pay 12.5-16.** Their crossover at 18-22 tokens came from
expensive window tokens; ours lands at ~60-100 tokens, so upstream's fixed 64 was already
near-optimal on this card and base ≈ port at the dsh shape. The port stays: it is
self-measuring (protects against position/workload shifts), exact-by-construction (routing
only; answers unchanged per the within-process checks), no-regression in the gate, and it
makes future card/driver changes self-correcting. Production 8081 runs it (Paris green).
Rollback: `--short-read 64` in the serve config pins the old behavior, or `git revert`.

## Session 28 (2026-10-13): R24 — production upgraded to the 0.1.43.1 line

`refresh-0143` = upstream **v0.1.43.1** (`a0116b2`/`47bc6e06`, 105 commits since v0.1.42, linear —
v0.1.42 IS an ancestor, no forced-update reconstruction) + all 30 `refresh-0142` fork commits
replayed `-x` in a fresh worktree `strata-r143`. Conflict surface exactly 3 files (CMakeLists,
README, generate.cpp).

**The replay caught a real silent-revert.** Upstream 0.1.43 introduced `decode_defaults_arch_ok()`
(gfx1200/1201-only on HIP) gating their new PcieAuto default; the conflict resolution taking HEAD
verbatim would have switched our LIVE PcieAuto-on-gfx1100 port off without a word. Composed:
`(layer_split.empty() || pcie_auto_split_ok()) && !pf_old && (decode_defaults_arch_ok() ||
pcie_auto_fork_arch_ok())` — the new helper pins gfx1100 exactly (the only arch our no-regression
gate measured); every other AMD part keeps upstream's "earlier behaviour". Gate P: 16/19 fork
files byte-identical, select arm markers intact, dsh chooser verbatim (+123/−3 all present).
Upstream's PcieAuto machinery had converged with ours — `acb3deb0` itself was only +6/−3 (the
guard removal), and the machinery auto-deduplicated.

**Build + host gates**: build-0143 fresh dir with build-0142's exact cache flags (the R23
`CMAKE_HIP_FLAGS` two-stack trap avoided) — ninja 381/381, 0 errors, stamp **0.1.43.1-8a15c99c**.
Host ctest 105/113 (5 failures = exactly the declared env-gated class; 3 declared skips).

**R24 card window** (one batched stop/restore; MemAvailable 78.8 GB after stop — the desktop is
light now, the 49.7 GB consumer was the serve engine itself): strict parity 4-suite ×2 **8/8 PASS**,
`hip_prefill_fused_iq` **PASS** (host-OOM closed as environmental), static f16-WMMA fatbin check
**PASS** (our 8× f16 select arm beside upstream's 48× bf16). Band cells MISSED the pre-declared
absolute bands (128K prefill 2136.5 vs 2143–2189; 1K decode 70.9 vs ≥74.0) — per the standing rule
that means investigate, no tolerance moved. **Paired engine-vs-engine investigation**
(0142 vs 0143, interleaved, same shape): **+0.22% prefill @128K (2158.1 vs 2153.3, select 1932 vs
1934 ms) / +1.26% decode @1K (71.4 vs 70.5 median) = NO-REGRESSION**. The misses explained by
measurement: the same 0143 binary reads 2136.5–2158.1 across windows (~1% cross-session spread,
wider than my band edge), and R23's 76.3 1K anchor came from the session whose record JSON was
never landed — both binaries read ~70–71.5 on today's shape. Swap decision = GO on the paired
contract (recorded as the judgment it is).

**Flag survey (user ask) + probes (n=1 @1K)**: `--remote-expert-opt` N/A (multi-GPU remote-expert
path; single-card host); `--adapt-async` N/A (resident-RAM mode); `STRATA_KV_GROW` does not engage
on this config (engine says why); ADAPT_OVERLAP +0.6%, AUX_CPUS flat — queue. **`STRATA_ROUTE_
TAIL_SKIP=7` measured +8.6% decode (77.0 vs the ~71 cluster)** — the top test-plan cell's first
gfx1100 signal; output-changing, NOT adopted, gated campaign (KL/task gate) is next. Upstream
0.1.43 decode-defaults push (adaptive overlap) stays CUDA-default/HIP-opt-in — we keep it off.

**Incident (owned)**: the static-check dry-run corrupted the rollback binary — llvm-objcopy
without `-o` writes to the INPUT path in place; build-0142/strata became a fatbin stub. Running
engine unaffected (image in RAM; zero restarts, journal-checked); fixed by rm + ninja relink
(content = 5690ed88, cached stamp). The window script now objcopies a /tmp copy only; lesson in
env-quirks memory. Swap: `swap-0143.sh`, 4/4 parity, 8081 up in ~4 s, Paris smoke finish=stop,
engine log `session is up (engine 0.1.43.1-8a15c99c)`. Rollback `.bak-0143-pre` → build-0142.
Records: `record-refresh-0143-20261013.json`; GFX1100.md banner `5a68370f`. Nothing pushed.
Post-swap watch: PcieAuto engagement on the next natural long request (48 windows + 300 misses).

### Session 28 cont. (2026-10-13, evening): PR #2224 filed + R25 TAIL_SKIP campaign — ADOPTED

**PR #2224** (user-ordered): PcieAuto-on-HIP for gfx1100 filed upstream — `xyzzing:pcie-auto-hip-gfx1100`
based on v0.1.43.1 == main, one commit (+24/−2, the composed `pcie_auto_fork_arch_ok()` resolution from
the R24 replay), fresh-worktree build gate 373/373, dup-checked (no prior pcie-default PR). Evidence in
the body: the no-regression gate, the serve engagement traces, production service since 10-12. Record:
`record-pr-pcieauto-20261013.json`.

**R25 — `STRATA_ROUTE_TAIL_SKIP=7` ADOPTED on production 8081** (the test plan's top cell, run on the
R24 probe's +8.6% signal). Pre-declared gates (script header, before any rate): A = task battery +
drift-vs-restart-baseline, B = paired one-shot decode. **The literal KL was declared infeasible before
the campaign**: `--dump-logits` writes 0 bytes on native packs (measured; the source's own "a native
pack never reaches the per-token dump site") — the gate ran as task battery + drift + mechanism
identity, with upstream's KL 0.003–0.006 (same machinery, CUDA) and XeStrata's 0.003–0.006 (AMD) cited
as mechanism-class evidence only.

Results: **Gate A PASS** — battery 7/8 = 7/8 = 7/8 across A-B-A (the one failure is battery-constant;
0/8 answers differ arm-vs-base), prose drift 0.756 vs restart baseline 0.481 (real, "answers differ
slightly" class, characterized). **Gate B PASS, decisively** — 1K: −0.74/+13.72/+13.97/+16.60/+21.07%
(median **+13.97%**, 4/5); 128K: +18.56/+18.30/+19.17% (median **+18.56%**, 3/3). **Serve confirmation:
+9.3%** on the live API path (132.63 → 144.99 tok/s median, cached-resident shape). Adopted: env line in
serve-8081.json with the output-change note (`=0` restores base outputs; rollback = delete the line);
engagement print verified; production healthy on it now.

Records: `record-r25-tailskip-20261013.json`; script `scripts/rocm/r25-tailskip-campaign.sh` (gates in
the header). Not filed upstream per the 10-12 decision (maintainer mid-flight on the HIP tail-skip
gates) — our numbers are comment material if the question opens.

**#1292 refresh (2026-10-13, night):** the rates sweep re-ran on the shipped config (0.1.43.1 +
TAIL_SKIP=7): prefill 722 / 1,366 / 2,117 / 2,149 / 2,136 tok/s and decode 80.7 / 80.1 / 74.5 /
71.4 / 81.3 tok/s across 1K-128K (median of 3 clean cells per tier, no stalls). The PR branch is
rebased onto v0.1.43.1 — the rebase showed upstream had already absorbed the old 0.1.38 row into
main, so the PR now carries the numbers refresh (decode +23-37% per tier, prefill 32K-128K +29-42%)
with the TAIL_SKIP disclosure in the footnote; force-pushed + delta comment. Record:
`record-r1292-resweep-20261013.json`.

### Session 28 cont. 2 (2026-10-13, night): R26 — the decode-flag batch; --lookup-chain 2 ADOPTED

The Tier-1 re-probes + untried knobs ran as one batched window (38 one-shot cells; gates pre-declared
in the script headers) plus a serve tier. **The window-economics-shift hypothesis is CONFIRMED — for
lookup-chain**: `--lookup-chain 2`, rejected −5.6/−6.9% at full window cost on 0.1.40.2, now measures
**+6.30% @1K (3/3) and +19.92% @128K** one-shot with ROUTE_TAIL_SKIP=7 making deep windows cheap. The
arm's output ids are byte-identical across all three starts (within the recorded greedy-variant class;
base itself splits 3 ways) — exactness-preserving by construction, no battery owed. Serve confirmation:
**+3.54%** (141.01 → 146.00 tok/s, 6+6 probes). **ADOPTED** into serve-8081.json args.

The rest, by the gates: **spec 6 REJECT** (+1.03% @1K, −6.03% @128K — the raw-depth optimum didn't
move; the shift pays via lookup drafts, not window depth). **k8v4 REJECT** (+4.42% @1K but −4.93%
@128K fails the ≥0 gate — the slot gain doesn't cover q4_0-V precision at depth). **mtp-q4 proj
REJECT** (median −0.49%; pair 1's base cell 63.49 recorded as a stall-contaminated outlier). **pool
workers: default 11 re-confirmed** (8/10/13 all −0.2…−0.4%). **ADAPT_OVERLAP under serve: −3.36%,
auto-reverted** (the gate removed the env line; one-shot flat, serve negative — closed on this config).
`--calibrate` printed help instead of running (invocation needs its documented workload) — queued.
k8v4/mtp-q4 task batteries not owed (rejected at the rates gate before the battery stage).

Production 8081 now: 0.1.43.1-8a15c99c + TAIL_SKIP=7 + `--lookup-chain 2` (both 13-Oct adoptions,
each with a recorded rollback), healthy. Records: `record-r26-decodeflags-20261013.json`; scripts
`r26-decode-window.sh` + the serve-tier scripts. Logs `r26-*.log`.

**Deep tiers (2026-10-13, night): the first 262K row measured** — on the shipped config (0.1.43.1 +
TAIL_SKIP=7 + lookup-chain 2): 128K re-measured (prefill 2,157 [2157–2158]; decode 75.6 [75.5–80.4])
and **262K: prefill 2,043 [2038–2051], decode 69.3 [67.9–71.4]** (3/3 clean each, one-shot, median).
The 262K cell shape: the 262,144-token workload trimmed by 256 to fit the native cap with
`--max-new 256` (a full prompt + generation exceeds every valid `--max-context`; 278,528 headroom
parses but allocates past-native and hit a dense-upload VRAM OOM beside the resident). Decode decays
only 75.6 → 69.3 from 128K → 262K — the O(1) GDN decode story holds at the native max; prefill gives
back just 5%. #1292's empty 262K column is filled (branch `055fe9b1`, comment posted) and the x.com
card's comparison panel now shows the 128K/262K clusters (no community PR has a 262K row — ours is
the only bar there). Record: `record-deeptiers-20261013.json`.

## Session 29 (2026-10-14): R27 — production upgraded to the upstream v0.1.44.1 line

**The ask**: "update production to latest upstream version 0.1.44". Upstream's latest release is
**v0.1.44.1** (`296b59d2`, tagged 2026-10-14 01:51 +0200 — the same commit as upstream main HEAD),
an 11-commit hotfix over the `v0.1.44` tag (`ed4773eb`, ~5 h older) the request named. I asked and
the user confirmed 0.1.44.1.

**The refresh.** `refresh-0144` = v0.1.44.1 + the 32 fork commits replayed `-x` in a new worktree
`strata-r144`. The upstream delta is **282 commits / 358 files / +47208−1243** — the largest refresh
so far. Two commits conflicted, both in `src/program/generate.cpp`: upstream's new
`mtp_first_default_on()` landed exactly on the fork's `pcie_auto_fork_arch_ok()` helper (composed:
both kept), and upstream's new `--prefill-head-alias` landed on the dsh-chooser/ShortReadCost hunks
(4 adjacent-addition conflicts, all keep-both; upstream's head-restore block needed its own closing
brace because the shared `}` closes the fork's `if (!o.short_read_given)`).

**Gate P — the silent-revert trap did not recur.** 14/19 fork-touched files byte-identical to
refresh-0143; the 5 that differ differ only by upstream additions (verified line by line: version
bump + `STRATA_MMQ_KQUANTS` relocation + targets; the README Unsloth row removed upstream; a
`kCompiledGeometry` static_assert and gfx1103 in `qsa_select.cu`; NVFP4 `row_sd` scaling and gfx1103
in `prefill/kernels.cu`). Stronger than the ratchet: all **2986** distinct lines the 32 fork commits
added were checked against the composed tree — **0 present in refresh-0143 but lost in refresh-0144**
(149 are absent from both, i.e. lines a later fork commit rewrote). The PcieAuto engagement line
auto-merged intact this time; the 0.1.43 trap needed a manual compose, so the gate is still earning
its keep.

**Build + host gates.** `build-0144` from build-0143's exact cache recipe (the first attempt
correctly stopped at `STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON`; the full recipe was
recovered from record-refresh-0141), one recorded deviation — the pinned ggml checkout is reused
from build-0143's `_deps` (pin unchanged) instead of re-cloned. **ninja 403/403, 0 errors**, stamp
**`strata 0.1.44.1-7d4356d7`**. Host ctest **111/116**, with exactly the R24 declared env-gated
failure class and no new failure.

**The card window.** Parity 4-suite ×2 **8/8 PASS on-card**, `hip_prefill_fused_iq` **PASS**,
static f16 select-arm check PASS. The first cell attempt **skipped** at 64,495 MiB < the pre-declared
68,000 MiB gate — recorded as skipped, not waived; after the desktop released its browser memory the
re-run had 82,722 MiB.

**The paired verdict, and a disclosed bug in the verdict script.** Interleaved 0143-vs-0144 cells on
the production args (`--lookup-chain 2`, `STRATA_ROUTE_TAIL_SKIP=7`). The R24-era verdict carried two
problems. First, it is **two-sided** (`abs(delta) < 3`) under the name "no-regression", so it reported
0144's **+7.16%** 1K decode improvement as "REGRESSION SUSPECTED"; fixed in `r27c-verdict-0144.py`
(**direction** fixed, tolerance unchanged at 3%, measurements untouched, R24's +0.22/+1.26 never
surfaced it). Second — and this is the substantive point — **decode tok/s is not a cross-engine
metric**: it is `tokens-per-round / ms-per-round`, and tokens-per-round is set by speculative
acceptance, which follows the trajectory (greedy output is a recorded cross-restart variable). Proof
inside this one window, same binary 0143: 2.53 vs 2.87 tok/round → **65.53 vs 90.79 tok/s (38.5%)**.
The verdict therefore gates on `ms/round`, the cost the engine actually controls:

| axis (trajectory-independent) | 0144 | 0143 | paired |
| --- | --- | --- | --- |
| prefill @128K | 2151.4 tok/s | 2148.3 tok/s | **+0.14%** |
| 1K ms/round (median of 2) | 30.74 | 35.01 | **−12.19%** |
| 128K ms/round | 33.61 | 36.25 | **−7.29%** |
| select phase | 1941 ms | 1940 ms | fp16 arm engaged |

**NO-REGRESSION** on every axis 0144 does not get to choose. (Reported but not gated: 1K decode
tok/s median 83.75 vs 78.16 = +7.16%; 128K decode 59.50 vs 63.05 = −5.63% at 2.00 vs 2.29 tok/round —
both acceptance-driven.)

**What upstream changed under us (read-only 0.1.43.1→0.1.44.1 audit).** Two new **default-on**
gfx1100 paths with no production env line: `STRATA_COMMIT_BATCH` (batches each GDN layer's
conv+state commit and each QSA layer's tail restore + indexer append into one launch; upstream
measured it on gfx12/Windows) and the expert-arena huge-page gate (`pinned.cu` — `MADV_HUGEPAGE` is
skipped when `/proc/buddyinfo` shows < arena/2 free in ≥2 MiB blocks). Both are exercised by the
paired cells above. Also: upstream now **hard-checks the `--native` GGUF geometry** at startup
against `kCompiledGeometry` — the production shard's 19 pinned keys + `rms_eps` were read and match,
so the gate passes, but a differently-pruned file would now be refused rather than mis-run (a real
behaviour change worth knowing). gfx1103 was added to every WMMA whitelist (additive, gfx1100
retained); `STRATA_MMQ_KQUANTS`/`STRATA_Q6K_EXPERTS` went default-ON upstream and are runtime-inert
for this pack. Residual unmeasured risk recorded: the new batched commit kernel
`gdn_step_commit_layers_kernel` has no `__launch_bounds__` where the per-layer original had one.

**The frozen server.py stays (audited, not assumed).** Production runs a frozen
`$EXP/hipify/serve/server.py`; it is **byte-identical to upstream v0.1.40.1** (unmodified, four
releases behind). An audit of the argv set, stdin command set, stdout prefix set, protocol format
strings and the `FATAL_PREFIXES` greps found them unchanged between the refresh-0143 and refresh-0144
trees → **COMPATIBLE**; upstream's 0.1.44 server-side features (image generation, elastic vision,
start retry, `img_keys`) are simply **not adopted** — lost functionality, not breakage, the same
policy as R24.

**Production now**: `strata-8081.service` runs **`0.1.44.1-7d4356d7`**
(`/home/zacch/projects/strata-r144/build-0144/strata`), args and env unchanged, up in ~4 s, health
ok, ctx 131072, smoke *"Name the capital of France in one word." → "Paris"* finish=stop, engine log
`session is up (engine 0.1.44.1-7d4356d7)`. Swap gates: strict 4-suite parity 4/4 on the exact
binary. **Rollback**: `serve-8081.json.bak-0144-pre` → `build-0143/strata` `0.1.43.1-8a15c99c` (kept
in place; build-0143 was never rebuilt, moved or deleted). The 8080 server was never touched (it was
not listening before or after). Record: `record-refresh-0144-20261014.json`; TASKS R27; nothing
pushed.

### Session 29 cont. (2026-10-14): R28 — the two new default-ON paths measured; both safe, neither measurable

Upstream's 0.1.44 turned two gfx1100-relevant behaviours on by default with no env line and no
gfx1100 measurement of their own: **`STRATA_COMMIT_BATCH`** (one batched commit launch for every GDN
layer and every QSA layer, `=0` keeps the per-layer loop; their commit measured it on gfx1201) and
the **expert-arena huge-page gate** (skip `MADV_HUGEPAGE` when free 2 MiB blocks < arena/2 — the fix
for #2126). I A/B'd both on `build-0144`, the live production binary, 1K ×4 and 128K ×2 interleaved
cells per arm.

**I was wrong about the gate, and the engine's log corrected me.** My first reading said the gate
"fires in production" because buddyinfo taken with the engine *running* showed 82–126 MiB of
order≥9 blocks against a ~47 GiB arena. But the gate reads `free_huge_bytes()` at **arena setup** —
before the engine allocates, when the previous instance's memory is already free. The authority is
the production log: **15/15 recent sessions requested `MADV_HUGEPAGE`, and the gate-fired string
occurs 0 times in the whole 75,438-line log** (the 3 "using 4 KB pages" lines are at log lines
215/263/308 from long-superseded versions). So on a 96 GB box with a 47 GiB arena the gate is
essentially **never taken** — it is insurance for the overlapping-restart case #2126 describes.

**Consequence for the measurement: the `STRATA_ARENA_THP=1` arm is a no-op here** (base already gets
THP — all of base/A/B/C logged `MADV_HUGEPAGE`). I added arm **D (`STRATA_NO_ARENA_THP=1`)** to force
the 4 KB branch the gate *would* take. Result: steady state identical at matched mix (D 31.22/31.30
vs B 31.17/31.26 ms/round), but **startup wall +8 s @1K (29/31 s vs 21 s) and +5 s @128K (85 vs
80 s)**. The gate's normal branch is the fast one; its fallback costs a few seconds and buys nothing
in steady state.

**The `STRATA_COMMIT_BATCH` reading was a mix confound.** The 1K cells fall into exactly **two
trajectory states**, and the window-size mix alone swings ms/round ~10%: T3-heavy (rounds 94,
tokens/round 2.72, T3:69, acceptance 0.976) ≈ **28.2 ms/round**, T5-heavy (rounds ~90, 2.87,
T5:35, 0.792) ≈ **31.2 ms/round**. The naive 4-rep median read +5.11% for `=0` — but that compares
different mixes. **At matched mix the arms are indistinguishable**: 1K T3-heavy base 28.21/28.25/
28.25 vs `=0` 28.14/28.27; 128K base 35.97/36.16 vs `=0` 35.83/35.90. Prefill @128K flat (2152.4 vs
2151.1). So: **no regression, and upstream's 1.7 ms commit-graph saving does not surface as a
measurable share of a ~28–36 ms verify round on gfx1100.** The control arms set the noise floor at
±1.3% (B, a genuine no-op, read +1.13%; D read +1.27%).

This also sharpens R27's own method: `ms/round` is *less* trajectory-dependent than tok/s but **not
mix-independent** — R27's four 1K cells all happened to be T5-heavy with T5 counts 35–45, so that
comparison was mix-matched and stands; the general rule is now recorded.

Two cosmetic script bugs were found *after* the run and fixed (the preflight buddyinfo awk summed
orders 4–10 instead of 9–10; the page-strategy detector did not recognise the "skipped" wording).
Neither affected any number: the engine's own per-cell arena note was the engagement proof
throughout. Production stayed on `0.1.44.1-7d4356d7` with its config unchanged, verified healthy
after each run, no stray engines. Record: `record-r28-0144-defaults-ab-20261014.json`; TASKS R28.

**Filed upstream** (posting rule satisfied first: records green, every figure checked against the
record): issue [#2361](https://github.com/Niko1221/Strata/issues/2361) carries both measurements as
one report, and a comment on [#2126](https://github.com/Niko1221/Strata/issues/2126#issuecomment-6160027716)
carries the gate-specific data on the issue that motivated the fix — including the honest caveat
that the #2126 stall itself was not reproduced here.

## 2026-10-18: GFX1100.md optimization-ledger export (docs only, no measurements)

The user ordered a consolidated export of every optimization done or explored, why each closed line
stopped, and a ranking of the pending queue, written into GFX1100.md. Executed as a new section
"**The optimization ledger — adopted, closed, pending (2026-10-18 export)**" in
`~/projects/strata-r145/GFX1100.md` (the live refresh-0145 line), placed between "Measured and
rejected" and "Credits". Contents: (1) the adopted set — 11 rows, each with arm/tier + record name
(100500 table, both WMMA arms, spec 5 + min-p 0.70, SH_STREAM, TAIL_SKIP=7, lookup-chain 2,
PcieAuto-default-on, dsh chooser, autotune, inherited MMQ tiles + 0.1.45 wide top-k); (2) the closed
set grouped by closure class — headroom-did-not-exist (fused-WMMA −0.8%, tile-J transfer-floor 79%,
BLAS 4.1×, placement −1.53%, 24 GB residency impossible), precision-contract failures (INT8-Q,
NORM_ROPE, k8v4), measured-slower/inert (WY 144×/parity, TSUM, spec-split, ring-wait,
ADAPT_OVERLAP, the R28 defaults pair, the draft-yield family, N/A-refusing knobs), wrong-ISA/bottleneck
donor assessments (DeepGEMM, DS4, cafe-llama.cpp, StairCut, #846, MonoMoE-class, XeStrata,
HC/PLE), and capability limits (YaRN 512K acceptance collapse, issue-bound attention, no-logit-reference
+ infeasible-KL, greedy cross-restart instability); (3) **pending ranked 1-16** under the rule
"measured signal > unmeasured mechanism; one-window cells > campaigns": deep-tier 0.1.45 re-measure,
R21 GDN A/B, PF_GEMM isolate, SCORES_THREAD @262K, ROUTE_PRIOR, HC_UPMIX free A/B, the 0.1.45
dormant opt-ins, ADAPT_MIN_GAIN, routed-gather (user-gated), #1367-class INT8-MMA port, calibrate,
prefill-auto dead-arg, partial-residency graph capture (with a re-measure-first caveat),
Controller/depth policy, mtp-draft-vocab, ISA profile — plus the maintenance line (WY CUDA
dynamic-smem port, XeStrata remnants, strata-pack v2, x.com card refresh). Every figure traces to
an existing record; upstream-PR states deliberately out of scope per the user's instruction. The
edit is **uncommitted** in the strata-r145 worktree (docs commit on user ask). Note: STATUS has no
R29 (0.1.45 swap) section yet — the swap record and GFX1100.md banner exist
(record-refresh-0145-20261018.json); this section does not renumber or backfill it.

## 2026-10-18 (later): upstream PR sweep — 849 open PRs pulled, dispositions reconciled, R30 queued

User ask: pull all of upstream's PRs, read them against GFX1100.md's ledger and TASKS.md, and say
what is next. Executed read-only: every **open** PR head of `Niko1221/Strata` (**849**) was fetched
into a throwaway bare mirror inside the rollout workspace (`.scratch/pr-audit/strata-prs.git`,
106 MB, 0 failures, `--reference` to the local checkout), with metadata and a per-PR delta
(`open-prs.json`, `pr-diffs.json`). Upstream main is **exactly `v0.1.45` (`477ac13c`)** with
**0 commits after the tag**, so there is no next-refresh content to pull — the open PRs are the
only upstream work not in the production binary (`0.1.45-8eb3e85d`). Queue shape: 167 bench
reports, 22 docs, 33 setup, 97 serve, 18 web, 9 tests, 503 engine/other; **125** touch a
`.cu/.cuh`, 75 of them created on/after 2026-10-10.

The sweep's most consequential output is not a new PR: **the maintainer replied on 2026-10-17 to
four of our six open PRs**. #2224 (PcieAuto) was ported as opt-in with the default unchanged — so
the fork's gfx1100 measured-default-on is a disclosed divergence, and the PR itself is **open, not
merged**; #1296 (GDN WY) is **rejected** (WY loses −39…−40 % prefill on gfx1100 by our own numbers)
because #2299's chunked WMMA gfx11 recurrence is **already in main** — confirmed in our tree
(`gdn_rec_wmma` / `STRATA_GDN_CHUNKED`, `src/prefill/kernels.cu`); #1297 (autotune) is **not
carried** (the +10 % was `--spec-min-p 0.70`, already adopted); #1298 (fp16 select) is **not added
next to the exact thread scorer #1943** (bitwise-equal scores, 3.6–4× on the phase) unless we show
an end-to-end win. #1292/#1293 have no reply. Ledger corrections recorded: **#846 closed
unmerged** (no merged form to cherry-pick), **#1655 closed**, **#2299 closed with content in main**,
and **#1367 + #1943 are already in the 0.1.45 binary** — so ledger pending items 4 and 10 are
turn-on cells, not ports. One genuinely new gfx1100 candidate: **#2514** (9-line HIP decode QSA
top-k at 262K; 1.7–1.8× kernel at 98K–262K on the 7900 XTX, same ids, `STRATA_TOPK_DECODE_REG=0`
off switch).

What that means for the board: the production binary already contains **seven gfx11 knobs that are
dormant and unmeasured here (SCORES_THREAD only partially: ≤3 % @128K post-WMMA, never 262K)** —
`STRATA_SCORES_THREAD` (exact scorer; gfx1100 default OFF; parity target
`qsa_scores_gfx1100_parity` is built in build-0145), `STRATA_SELECT_SIMT`,
`STRATA_QSA_ROWS`, `STRATA_QSA_SPLIT_ATTN`, `STRATA_EXPDEC`, `STRATA_HC_UPMIX`, `STRATA_PF_F16` —
plus upstream's chunked GDN. **Correction (2026-10-18, same day):** `STRATA_PROMPT_ATTN_IMMA`
(#1367) is in main's source but **not compiled** into build-0145 (0 strings in the gfx1100 binary),
so the earlier "eight" was wrong and item 10 is not a turn-on cell here. The next step is therefore **not a port or a refresh**: it is one
batched window (R31) whose top cell is the **scorer three-way at 128K and 262K**
(`SELECT_WMMA=1` vs `SELECT_WMMA=0 + SCORES_THREAD=1` vs both off), which both refreshes the deep
tiers (ledger #1) and decides #1298. The same window carries the GDN chunked A/B (resolves R21 and
retires WY) and #2514 (one cherry-pick + rebuild; kernels: `qsa_topk_parity` then a 262K decode
cell). Documentation debt found: R29 had **no TASKS row** (backfilled) and has no STATUS section
(noted), `GFX1100.md`'s DOC1 export is still uncommitted, `refresh-0145` is unpushed, and the live
`serve-8081.json` note's "#2224 as merged" is wrong wording.

Evidence: `artifacts/rocm/upstream-pr-sweep-20261018.md` (full tables and the ranked queue),
`artifacts/rocm/results/record-upstream-pr-sweep-20261018.json`; TASKS R29/R30/R31/R32. No GPU
cell was run (needs the 8081 window and a widened sandbox), no PR was commented on or closed, and
nothing outside this workspace was written.

**Next command/decision:** approve the R31 window (`scripts/rocm/r29-gpu-window-0145.sh` shape:
stop 8081 → 68 000 MiB gate → strict parity → cells → restore), or run cell 2 alone first.

## 2026-10-18 (session 31): IQ3_S decode kernel campaign — windows A/B, two kernel candidates measured and rejected

User order: auto-start the IQ3_S decode optimization backlog (`tasks.md`, attached; copied verbatim to
`artifacts/rocm/iq3s-decode-campaign-20261018/tasks.md`). The pack's own `native_experts.txt` **confirms the
backlog's format table exactly** (gate/up IQ2_S 20, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1; down IQ4_NL 39, Q2_0 9;
512 experts; 7 format pairs; 98,227,200 B per expert = 50,292,326,400 B total) — the campaign was pinned on that,
not on the backlog's assumption. Upstream recheck: **#2514** is the only applicable external change (deep-tier
decode top-k; #3024/#2997 are gfx12, #646/#783 closed, llama.cpp #26301 not in the pin and K-quant only).

**Window A** (stop 8081 → gates 68,587 MiB / 2,235 MiB → parity → bench → restore): strict parity **6/6 PASS
including under `STRATA_EXPDEC=1`** — EXPDEC is bitwise-correct on gfx1100. The gate/up EXPDEC kernel is
**slower** at the grouped shape (922.2 → 991.1 µs at G20 T4; tie at G8 T1) → **K1 rejected at the kernel
level**. The bench did find a down win: `STRATA_EXPDEC_DOWN=1 _D20=1` is **−21…−34 % at the real G34 T1
envelope on every down format**, so it was carried to e2e. Attribution from the one-shot summary: wait-for-rings
**23.41 of 35.4 ms/round**; CPU experts 1.55 distinct / 2.15 routed per layer; **PCIe 55 % of misses**; routing
**k = 10** with 3.36 rows/round (the real verify-window envelope is G≈34 T≈1, not the denser bench shape).

**Window B** (baseline + down e2e + serve-path stage profile): baseline **1K 96.09 tok/s** (5 cells, 0.46 %
spread), **32K 66.32**, **128K 61.57** (one slow cell; paired arms are the rule), acceptance deterministic
across every arm. The down arm **regressed end to end: −3.34 % @1K (5/5 negative), −2.85 % @32K**, +3.01 % @128K
(noise) → **K2 rejected**; this reproduces `docs/EXPERTDEC.md`'s CUDA observation that the down bench win does
not survive the engine. The stage profile (serve path with `STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`,
read from the config's engine log — server.py does not forward engine stderr to its stdout) gives the decode
map: 35.07 ms/window staged = **waits 26.9 %**, projections+attention 25.3 %, **HC norm/down/up 15.6 %**,
expert VRAM hits 11.2 %, shared+quant 6.1 %, copy+combine+PCIe 5.3 %, router 4.4 %, head 3.3 %, plus MTP draft
4.77 ms/round (11 %) outside the stage total. Caveat recorded: the profiler disables sh_fork/branch/fuse_ab/
side_mixer, so the profiled round is 42.1 ms vs ~27.2 ms uninstrumented — shares indicative, never an adoption
number.

**Decision:** the in-binary decode kernel space is now exhausted with evidence — gate/up rejected, down rejected,
`QSA_ROWS`/`QSA_SPLIT_ATTN` ≤1.4 % each by the profile, `HC_UPMIX` is prefill-only, `HC_PERSIST` refuses gfx11.
The 27 % wait tracks the per-layer host CPU-expert serialization (5.40 ms/window), so the next choices are T08
overlap (port `STRATA_VERIFY_DEFER_CPU` #2690, output-changing, battery-gated) or a new gfx11 HC decode fusion
(15.6 % cluster). Nothing was adopted; production stayed on `0.1.45-8eb3e85d` and was restored healthy after
both windows (~24 s each), 8080 untouched.

**Window C — the #2514 port at depth (objective item):** ported as `ed3606a6` (opt-in
`STRATA_TOPK_DECODE_REG=1`, default OFF). The T02 check found the PR's premise does **not** hold on 0.1.45:
262K decode already takes the **wide** kernel (`campaign3`/`82922f55`), and 131K decode takes the register
kernel either way — so the port is a wide-vs-register choice at 262K only, not a ref-kernel fix. Parity
**4/4 PASS** (`qsa_topk_parity` and its reg-always variant, both defaults; same ids). Three paired 262K
cells: +4.45 % (the first base cell ran cold at 54.11), −0.16 %, −0.37 %; **median ms/round +0.17 % with a
4.44 spread → gate FAIL, NOT ADOPTED.** The port stays dormant; the production config is untouched (at
131 072 the dispatch is identical). Provenance recorded: `build-0145/strata` was rebuilt in place at
`ed3606a6` (sha256 `b0d443f9…`), its `STRATA_VERSION` string still `0.1.45-8eb3e85d` because the stamp is
baked at configure time, and the window trap restarted 8081 onto it (health ok). Side data point, not
like-for-like: 262K decode here is **56.5 tok/s** vs the 69.3 recorded on the 0.1.43.1 sweep — flagged for
the R31 deep-tier re-measure, not claimed as a regression.

**Next command/decision:** the objective's items are done (pin, attribution, EXPDEC/grouped/down gates, #2514
port at depth, records, production restored after every window). The campaign continues on IQ4: (a) the #2690
overlap port (output-changing, quality class of ROUTE_TAIL_SKIP) or (b) a gfx11 HC decode fusion (15.6 %
cluster). Campaign artifacts: `artifacts/rocm/iq3s-decode-campaign-20261018/` (`fingerprint.json`,
`upstream.md`, `campaign.md`), records `record-iq3s-window-{a,b,c}-20261018.json`.

## 2026-10-18 (session 31b): IQ4(a) — STRATA_VERIFY_DEFER_CPU (#2690) ported and rejected on measurement

User choice: do (a), the overlap port. Ported by merging the PR branch (based on v0.1.44.1) into
`refresh-0145` in a separate worktree `strata-r146` (branch `port-2690`, merge `3f04756b`); three
conflicts resolved compositely — the fork's `PcieModel` plus the PR's plan-sink fields, the fork's
`hflag_in_h_`/`herr_h_`/`hdrop_h_`/`h_bal_` handles plus `h_pflag_`, and upstream's `lb_+k` in
`batch_poll` with the PR's sink nulls; both docs sections kept. Full build green (309 targets,
binary sha256 `d2080793…`, stamp `0.1.45-ed3606a6-dirty`). Default path untouched; the feature is
opt-in and output-changing.

**Gate A (quality): passes in substance.** Parity 4/4 on the ported binary (default path); the
battery A-B-A is **7/8 = 7/8 = 7/8** with no regression and byte-identical answers (the single miss,
h2o, is the battery's ascii regex against the subscript answer, in every arm); the deferral
demonstrably engaged (every captured window printed ", deferred", 1/2/3/5/6/8-token windows). The
prose-drift check is noise-dominated: across two runs the restart floor was 0.748 and then 0.000,
with arm divergence 0.000 and 0.734 respectively — the cross-restart greedy instability makes it
unusable here, so the battery and the paired cells carry the gate.

**Gate B (performance): fails.** Paired decode on build-0146: **1K −11.41 % (1/5 wins), 32K
−3.57 %, 128K −3.61 %**. The mechanism is visible and is the point: the deferral makes the 32K
round faster (−15.5 % ms/round) but the changed output costs drafter acceptance (1K 162/166 →
146/179; 32K 70/103 → 57/77), so committed tok/s falls at every tier.

**Verdict: NOT ADOPTED.** Production was never switched — it stayed on `build-0145`, config
unchanged (no `STRATA_VERIFY_DEFER_CPU`), verified healthy after the window; 8080 untouched. The
port stays dormant on `port-2690`/`strata-r146`. The only declared follow-up is a bounded
`STRATA_VERIFY_DEFER_CPU_BELOW` sweep (0.05/0.02) trading divergence for acceptance; it was not run
because the sign is negative at two tiers and 0.1 is upstream's own default.

**Next:** IQ4(b), the gfx11 HC decode fusion (15.6 % stage cluster, no in-binary decode arm), or the
ledger's ROUTE_PRIOR campaign (raises the hit rate that drives the 27 % wait). Record:
`record-iq3s-window-d-defer-cpu-20261018.json`; logs `r30d-*`; script `r30d-defer-cpu-window.sh`.

## 2026-10-18 (session 31c): the six follow-ups — three adoptions, two rejections, one no-go

User order: do the six suggested follow-ups. All six ran, each with a pre-declared gate; production
was restored healthy after every window and is now on three new env-only arms.

1. **262K deep-tier A/B.** 0.1.43.1 vs 0.1.45, same 261,888-token workload/config: **0.1.45 is
   +3.96 % faster in decode** (pairs +2.70/+3.96/+39.11; the third 0.1.43.1 cell a stall) and +3 %
   in prefill. The 69.3-vs-56.5 gap is **config/methodology, not a line regression** — the suspected
   deep-tier regression is closed (`record-iq3s-262k-line-ab-20261018.json`).
2. **`STRATA_ROUTE_PRIOR=0.5` ADOPTED.** Screen λ 0.5/1.0/2.0 → +29.9/−3.8/+0.7 % @1K; battery
   A-B-A **7/8 = 7/8 = 7/8**, no regressions. Clean interleaved confirmation: **1K +22.20 % (5/5),
   32K +25.56 % (2/2), 128K +0.24 % tok/s with −15.4 % ms/round**. Mechanism measured: CPU
   experts/layer 1.61→0.44-0.73, PCIe 1.19→0.20-0.89 — the decode wall the campaign had priced.
   Live in `serve-8081.json` (`.bak-route-prior-pre`, health + Paris smoke). Honest split: at 1K the
   gain is mostly the drafter accepting the new trajectory, at 32K both round and acceptance
   improve, at 128K only the round (`record-iq3s-window-route-prior-20261018.json`).
3. **BELOW sweep rejected.** 0.05/0.02 on top of the live config: 1K −2.48 % (2/5), 32K −3.96 %,
   rounds slower — the #2690 deferral stays dormant (`record-iq3s-window-below-sweep-20261018.json`).
4. **gfx11 HC fusion no-go.** The only fused form refuses gfx11 (no `s_memrealtime`), its closest
   RDNA prior (R9700, #2226) saw no gain, and the 15.6 % target share is instrumented-only. Reopen
   with an uninstrumented HC share ≥10 % or a barrier-free shape
   (`record-iq3s-hc-fusion-feasibility-20261018.json`).
5. **Prefill cells.** Scorer three-way @128K: fp16 arm select **1,714 ms** vs exact thread
   **4,320 ms** vs radix **9,859-10,440 ms**; total prefill ties (2,066 vs 2,055 tok/s), so the exact
   scorer is not a gfx1100 win and #1298 was closed with this data. **`STRATA_GDN_CHUNKED=1` ADOPTED
   (+5.85 % prefill @32K)** and **`STRATA_PF_GEMM=1 STRATA_PF_PAD=1` ADOPTED (+4.41 % @128K;
   GEMM alone +2.56 %)**, gated by `hip_gdn_chunked` parity + an 8K needle answered identically in
   base/GDN/PF/base (the short battery cannot reach these prefill paths; declared).
   `.bak-prefill-pre` holds the rollback (`record-iq3s-window-prefill-20261018.json`).
6. **Hygiene.** `refresh-0145` pushed (`512e96b3` then `dbb0a561`); **#2224/#1296/#1297/#1298 all
   closed** with their measured dispositions; GFX1100.md carries the follow-up section and the
   three new adopted rows.

**Live env now:** prior adopted set + `STRATA_ROUTE_PRIOR=0.5`, `STRATA_GDN_CHUNKED=1`,
`STRATA_PF_GEMM=1`, `STRATA_PF_PAD=1` (all output-changing; each has a config backup). 8080
untouched throughout.

## 2026-10-18 (session 31d): second queue — re-baseline, tuning, T08, hygiene, branch decision

User order: the second six-item queue, plus a view on the dormant branches. All six ran; nothing new
was adopted (the live config already carries the three arms), and the headline is now measured on it.

1. **Re-baseline + re-profile.** Live env: 1K **101.8-142.0** across the drafter's two trajectories
   (100 % acceptance → ~140 tok/s; 200/200 → ~102), 32K **73.5**, 128K 52-60 (noisy; the paired
   prior check there is neutral), **262K 44-45 with the prior +27.8 % (3/3) vs off** (33-34 → 42-45;
   CPU 2.75→0.78, PCIe 2.33→0.46). Prefill **2,337-2,395 @32K / 2,387-2,421 @128K / 2,315 @262K**
   (the pre-adoption 128K figure was ~2,150). The profile shows the window moved exactly as the prior
   predicted: **waitB 9.45 → 3.02 ms**, per-layer CPU 0.93 (from 1.55-2.63), GPU-reach wait 24.78,
   and the prior's extra router kernel costs 0.77 ms for a 6.4 ms wait saving.
2. **Prior tuning + quality.** λ 0.5 is a sharp optimum (0.25 **−47.15 %**, 0.75 **−38.67 %** vs it);
   with `STRATA_ROUTE_TAIL_SKIP=0` the prior still wins **+38-43 %**; a **20-prompt battery is
   20/20 for both off and live**, no regressions — a better quality bound than the 8-prompt gate.
3. **T08 sweeps.** Live `--vram-reserve-mib 3072` is best (2048 −40/−48 %, 4096 −11/−11 %);
   `--pool-workers` auto stands (9/13 mixed); the n-gram lookahead stats need ≥100 windows and were
   not captured (declared, not claimed).
4. **Hygiene.** shard2's hash is recorded (`316b46f3…`); **spec-off is impossible on a native pack by
   design** — the engine refuses with "needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK"
   (rc 2), closing T03's open cell as "cannot exist"; the >LLC bench (G64/141 MB) shows 3.4 µs/expert
   vs 4.0 at G8 (no cliff); the graph-replay item is covered by the engine's captured windows.
5. **Upstream:** main is still `v0.1.45` (0 commits after the tag); #3024 is open and was updated
   today, gfx12-only so far.
6. **Dormant branches.** Keep **#2514** on the refresh line (`ed3606a6`: 9 lines, default-OFF,
   dispatch-neutral at 131K; re-test if upstream merges its own 262K top-k). Keep **#2690** as a
   standalone branch (`3f04756b`), not merged: its 226-line `verify.cpp` diff would cost a replay
   tax on every refresh, and the prior has just cut the CPU rows 60-75 % that were its premise.
   Re-test triggers are in GFX1100.md.

GFX1100.md now carries the new headline section and the branch decision (`50de2d59`, pushed).

## 2026-10-18 (session 31e): tasks-v2 adopted — no local-model executor; V01/U01 delivered

User brought `tasks-v2` (local implementation + frontier orchestration) and asked whether to skip the
local model. Decision: **yes, skip it as an executor** — there is no model-to-model relay in this
setup (v2 says so itself), the only local models are the specimen on 8081 (the object under test,
stopped for every window) and the off-limits 8080 llama.cpp, and the DSH agent already has the
filesystem/build/runner access a local LLM would be relayed through. What survives from v2 is the
independent review, now a fresh-context reviewer subagent at v2's mandatory checkpoints (bottleneck,
oracle, sync/graph/repack design, final acceptance). v2 is installed as the live backlog
(`tasks-v2.md`), v1 archived as `tasks-v1-superseded.md`, with a reconciliation table and the role
map.

**V01 (decode timeline)** is delivered: per window (T=3.13, round 35.08 ms, stage 28.89 ms) the HC
read is 5.20 ms at **758 GB/s**, the experts 4.07 at **730**, the head 1.06 at **734** — all at HBM
bandwidth, so V02/HC fusion has no headroom. The **dense projections are the only material gap**:
8.60 ms at **558 GB/s**, ~26 % under the neighbouring kernels, worth ~2.0 ms/window (~5.8 %) if
closed; router+plan (2.32) and shared+quant (2.06) are overhead-bound, not byte-bound. Hardware
counters are blocked (`perf_event_paranoid=2`, no sudo), so the table is derived from constants and
the measured profile (marked as such).

**U01 (components/primitives)** is delivered: TheRock 10.2.0, HIP 7.17.26392, amdclang 24.0.0git,
hipBLASLt 1.5.0, rocBLAS 5, comgr 3.5.0, rocWMMA in use, CK present but unused. CK Tile and rocSHMEM
are not applicable to the executed paths; host-NUMA VMM has no measured problem; the compiler lead
has no known gap. **No second nightly is installed** — the current stack is already a 10.2.0 nightly,
counters cannot judge a toolchain A/B, and there is no compiler-shaped symptom; reopen triggers are
recorded.

**Recommendation (V01):** V05 scoped to the dense-projection layout, preceded by one discriminating
micro-benchmark (current vs 1024-B row-padded vs column-interleaved, T=1 and T=4, achieved GB/s;
falsifier: <10 % gain means occupancy, drop V05 and take the router/plan fusion). A fresh-context
review is running at the checkpoint before any Phase 2 code.

## 2026-10-18 (session 31f): V01 review and calibration — V05 dropped, router fusion selected

The bounded fresh-context reviewer returned: it **confirmed** the HC (3.94 GB / 758 GB/s) and experts
(2.97 GB / 729 GB/s) arithmetic, and **challenged** four things, all accepted — the projection byte
basis' unlicensed ×T factor (defensible range 179-560 GB/s), the assumed 750 GB/s wall, the
"column-interleaved copy" arm (a Q8_0 re-encode, not a copy) and the head row (660 GB/s, not 734);
it also caught that the **8.60 ms projection row was wrong — the named GEMV stages sum to 6.25 ms**.

Calibration measurements: `q8_gemv_bench` on the real `native_mmvq` path shows the large GEMVs at the
wall (qkv 695, QSA q 769, head 912 GB/s) and the small/medium ones at 25-60% (gate 426, HC up 289,
shared 308-375, k/v 340). Two serve profiles (spec 2 vs spec 8) show the stage total going
**24.04 → 29.14 ms for +58% rows**, so weight bytes are largely amortized per window, and
**router+plan+bell is 2.25 → 2.37 ms — effectively flat, overhead-bound**.

**Decision: V05 is dropped by its own falsifier** (a storage permutation cannot recover a per-call
occupancy/launch gap; the interleave arm would be a re-encode). **First experiment selected: fuse
the router path** (router+prior+topk) behind an opt-in — 2.37 ms/window, T-flat, expected 0.7-1.0 ms
(2.0-2.9%), gated by the existing router parity; runner-up is batching the small GEMVs. Disclosed
incident: `q8_gemv_bench --help` once ran the bench against the live card; production stayed healthy
and the bench was then re-run properly in a stop/restore window.

## 2026-10-18 (session 31g): first v2 experiment — router-select skip, rejected

The selected first experiment (fuse/skip the redundant router path) was implemented and gated. With
`STRATA_ROUTE_PRIOR=0.5` the prior re-picks the top-10 from the same logits and overwrites
ids/weights, so `native_router_top10_multi`'s selection is dead work; `STRATA_ROUTER_SKIP_TOPK=1`
(commit 662d4736) dropped that one launch per layer. It is **exact by inspection** — `route_prior_k`
reads only logits and `res`; ids/weights are write-only. The cross-process routing-trace check is
invalid for this change (greedy restart drift: 99 vs 93 drafts in the two runs).

Paired cells: a first window showed 1K **+2.91% / 4-of-5**, 32K +2.45% / 2-of-2; a clean confirmation
did not reproduce it (1K **+1.31% / 3-of-5**, with accepted counts swinging 197↔207, 32K/128K inside
the band). It misses the pre-declared ≥+2% / 4-of-5 gate, so it was **not adopted** and the commit was
reverted (74ce6009); the built binary is back to the adopted set. The router path was already batched
(one projection + one top-10 per window); the only redundancy was one cheap launch per layer.

**Where this leaves the campaign:** the remaining bounded candidates are the small-GEMV batch
(z/rec/ab + shared, bound ~1.7-2.9%) and per-shape kernel work on the 25-60%-of-wall shapes (gate
426, HC up 289, shared 308-375, k/v 340 GB/s); no other single change clears the gate on the current
profile. GFX1100.md carries the full v2-phase record (push `fd3a9f7e`).

## 2026-10-18 (session 31h): one window of small GEMV — batch closed below the gate

The user asked for one window on the small-GEMV runner-up. The premise was corrected first: `z/rec/ab`
are stamps **inside** the fused GDN kernel, not separate launches, so the only separate small GEMVs
are the shared expert's gate/up/down and the k/v projections. `q8_gemv_bench` at the live token
counts (1/3/4 cols): shared gate/up **6.0/8.2/8.3 µs** (39→28 % of the 750 GB/s wall), shared down
7.4/9.5/10.2 (32→23 %), k/v 4.5/5.8/5.9, HC up 14.4/21.1/22.4 (32→21 %); in-situ (r31f) the
shared+quant stage is 1.57 ms and HC up 1.50 ms.

A single 3-job launch for the shared block would recover the launch/teardown and some occupancy at
most **~0.4–0.5 ms (1.2–1.5 % of the 35 ms round)** — the weights (1.74 MB each) must be read
regardless and the shapes are latency-bound. That is below the pre-declared ≥2 % gate, so no kernel
was written; the runner-up is closed. The one-shot run (114.04 tok/s, trajectory band) does not print
the serve-path stage lines, so the in-situ figures are from the spec-8 profile, as recorded.

## 2026-10-18 (session 31i): HC up — byte-calibrated, tested, stopped below the gate

The user asked to pursue HC up with a stop rule. The first finding is a correction: `w_up` is **bf16
[10240][320] = 6.55 MB** per (layer, half), so the engine's in-situ 15.6 µs/call is **~415 GB/s =
56 % of the 750 GB/s wall**, not the bench's synthetic-Q8_0 21–32 %. A new diagnostic
(`src/kernels/hc_up_bench.cpp`) replicated the row-dot and tested grid/ILP shapes in one window:
baseline 80×128 **182 GB/s**, rows/2 448, **rows/4 (320 blocks) 566 GB/s (75 %)**, 8 warps 461,
4-row ILP 282.

Bound: even at the wall, the 96 calls would save **0.67 ms/window = 1.9 %** — below the pre-declared
≥2 % gate; the best practical shape gives 0.4–0.5 ms (1.2–1.4 %). The bench's baseline replica is also
2.3× slower than the engine's actual multi-token kernel, so the bench-to-engine transfer is uncertain.
**Stopped by the stop rule; no engine change**; the bench is kept as a diagnostic and the record written.
