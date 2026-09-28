# STATUS — Strata HIP/gfx1100 rollout

**partially working**

Last updated: 2026-09-28 (session 2) · Stage reached: **3 of 7**

> The HIP backend builds reproducibly for `gfx1100` against upstream **v0.1.15**
> (bumped from v0.1.13 this session; no device kernel changed upstream, all four
> declared patches re-applied cleanly), **every** translation unit compiles (no
> unsupported exclusions), and **39 of 42 ported suites agree with their
> independent references on the actual RX 7900 XTX** (3 skipped, all declared in
> `tests/rocm/unavailable.json`). All three `fused_gdn` entry points are now
> tested (two new tests this session), so the S3-feasibility gate's condition 1
> is closed. **Full-model correctness is unverified**: three skips need model
> weights or calibration data, no native expert has ever been computed, and the
> engine has never produced a token.
> **The user is downloading model weights to `/mnt/LINUXWINSHARE`** (remounted
> read-write by the user). Stage 4 begins when the shards land and verify.

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

## Revision

`Strata/` is at `d551edf42c157f1c8c20f07f68c5dfe506b1edda` — upstream tag
**v0.1.15** (tracked tree clean). Session 1 built against v0.1.13
(`b89c989a…`); session 2 checked upstream releases at the user's request, found
v0.1.14/v0.1.15 (IQ-model stall fixes, Linux compile fixes, kernel preload),
verified **no `src/kernels/cuda/*.cu` file changed** in the range — so recorded
kernel parity evidence carries over semantically — and re-ran the full suite on
the new revision anyway. All four declared patches re-applied cleanly; the one
composition change is `iq_avx2.cpp` joining the CPU-expert trio with plain
`-mavx2`, mirroring upstream's CMake (recorded in
`results/record-revision-bump.json`).

The plan's inspected revision `6da1f667…` also exists locally. Upstream `main`
was chosen over it deliberately at session 1: the plan asks for newer upstream
work to be inspected first, and there is **no HIP implementation upstream** —
the open PRs are all CUDA-side. Every result file records the revision it was
produced from.

Six untracked files in the checkout (`AGENTS.md`, `CLAUDE.md`,
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
| S2-kernel-suite-gfx1100 | **passed with declared gaps** | `results/test-kernels.json`: 39 passed, 0 failed, 3 skipped (v0.1.15) |
| S3-attention-prefill | partial | S3.1 (PTX replacement) done and tested; S3.2 GEMM ported and independently tested; MMQ verified for 6 of 8 types (IQ2_XXS/IQ2_XS need an importance matrix, S3.2b'); `prefill_mmq_parity` now reports its two reference-less cases as a declared skip instead of an invisible pass |
| S3.4-s2-expert-grouped | **passed** | `testlogs/s2_expert_grouped_parity.log` |
| S3-feasibility | **passed** | every kernel on the default forward path tested; the `native_gdn` condition is closed (all three fused entry points tested); see below |
| S5-concurrency | partial | graph semantics **measured on HIP** (`testlogs/core_graph_parity.log`): replays re-read their buffers, the registry captures once, empty captures are errors. The repeated-request suite itself is Stage 5 work |
| S4-first-inference … S7-release | pending | blocked on the user's model download (`/mnt/LINUXWINSHARE`) |

## What was run, and what came out (session 2, on v0.1.15)

```
./strata-rocm doctor        ready 14  blocked 0  unverified 0   (outside the sandbox; server on 8080 preserved)
./strata-rocm hipify        260 tracked files, 124 rewritten, 0 unsupported construct classes
./strata-rocm build         ok, 44 executables, 0 compiler errors, 4 jobs, no unsupported exclusions
./strata-rocm test --stage kernels
                            39 passed, 0 failed, 3 skipped  -> exit 5 (skipped is not a pass)
./strata-rocm report        regenerated from the recorded state
./strata-rocm selftest      35 harness tests passed (tooling's own suite, incl. injected-defect case)
```

Session 1's numbers are preserved in `artifacts/rocm/results/history.jsonl`
(including its two failed builds and the v0.1.13 suite's 37/0/2).

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
- **MMQ type coverage** — 6 of 8 verified. `IQ2_XXS` and `IQ2_XS` cannot be built
  through ggml's public reference API: their quantizers need an importance matrix
  (S3.2b'). Note the recorded divergence: ggml's HIP `__vsub4` saturates where
  CUDA's wraps (`PORTING.md` §6c), a live suspect for any IQ-format mismatch.
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
| QSA scorer is correct but unoptimised | known limitation | optional fast path, only after the parity test exists (it does) |
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

**Waiting on your download, not a command from me.** You are downloading the
model weights to `/mnt/LINUXWINSHARE` (you also remounted it read-write — noted
in the record). When the files land, the next commands are mine to run:

1. identify the variant and **sha256-verify every shard** against the manifest in
   `MODEL-DOWNLOAD-PROPOSAL.md` (which also gets its stale space arithmetic
   re-derived at that point),
2. `./strata-rocm build` (already green on v0.1.15) and TASKS **S4.1**: first
   inference on loopback **8081**, short context, single request, speculative
   decoding and vision off, with the server on **8080 untouched**,

and the first correctness claim at Stage 4 is *qualified* unless a logit-level
llama.cpp reference is arranged — the proposal says why.
