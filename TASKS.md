# TASKS — Strata HIP/gfx1100 rollout

Ordered work with acceptance gates. A stage is complete only when its gate
passes; "compiles" is never the criterion. Status values: **done**, **partial**,
**todo**, **blocked**.

Baseline: `Strata/` at `b89c989a7155e984e544ddd90d1038dda7da9e3d` (see STATUS.md
for why this is newer than the plan's inspected revision, and how to go back).

---

## S0 — environment and source audit

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S0.1 | Detect OS/kernel/CPU/RAM/disk/GPU/ROCm/toolchain, read-only | **done** | `strata-rocm doctor` reports ready/blocked/unverified per prerequisite; manifest in `artifacts/rocm/env/manifest.json` |
| S0.2 | Compile and run a HIP allocation+arithmetic test on the card | **done** | bit-exact integer result and finite float agreement on `gfx1100`; `results/smoke-run.json` |
| S0.3 | Resolve and record the exact upstream revision and tree state | **done** | revision + tracked/untracked summary in `state.json` and every result record |
| S0.4 | Inventory CUDA runtime / BLAS / intrinsic / PTX / graph dependencies across the whole tree | **done** | `HIPIFY_REPORT.md` + `results/hipify.json`: 260 tracked files, 123 rewritten, every untranslatable construct listed with file and stage. The scanner ignores its own explanation comments, so a zero count means zero, not "zero except the comment" |
| S0.5 | Check for newer upstream HIP work before duplicating it | **done** | no HIP implementation upstream; 5 open PRs, all CUDA-side (VMM, multi-GPU). Recorded in STATUS.md |
| S0.6 | Verify licence terms before any distribution | **partial** | no top-level LICENSE file; README credits llama.cpp/ggml (MIT) and vendored `third_party/ggml/LICENSE`. **Distribution is not authorised** until the project's own terms are clarified with the user. Local use only |

## S1 — automation and reference foundation

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S1.1 | `strata-rocm` wrapper with the full command contract | **done** | doctor/prepare/build/test/bench/status/report/run/hipify all implemented; timeouts, meaningful nonzero exits, no interactive prompts |
| S1.2 | Resumable state and machine-readable results | **done** | `state.json` with stage, gates, limits, budgets, repair counters; one JSON result per operation plus `history.jsonl` |
| S1.3 | Bounded execution with timeouts distinguished from failure | **done** | `EX_TIMEOUT=124` separate from `EX_FAIL=1`; `EX_SKIP=5` separate from both |
| S1.4 | Show the harness catches a defect it is supposed to catch | **done** | see "Harness self-check" below — a real defect (fp contraction) was caught, diagnosed and fixed by the suite, with before/after evidence |
| S1.5 | Independent CPU/float64 references for the ported kernels | **partial** | the in-tree `src/kernels/*_parity.cpp` references are used and are independent of the kernels. Missing: IQ/QK format fixtures (S2.6) |
| S1.6 | Validate the harness against a *deliberately injected* defect and remove it | **done (session 2, standing form)** | `./strata-rocm selftest`: 35 harness tests; `test_testrun.py::test_injected_defect_classifies_as_failed_with_rc4` injects a failing binary and requires the classifier to report failed (rc 4), plus a declared-but-different-failure case that must NOT earn its skip. Session 1's real fp-contraction catch remains recorded below |

## S2 — basic HIP backend and kernels

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S2.1 | HIPIFY the tree reproducibly, headers and host files included | **done** | `./strata-rocm hipify` regenerates `build-hip/hipify/` from `git archive`; hash-pinned translator; `.prehip` before-images kept |
| S2.2 | Declared patches after HIPIFY, reviewable as data | **done** | `patches/hipify_patches.py` rules + hit counts in `HIPIFY_REPORT.md`; forced-include shim for NVIDIA intrinsics |
| S2.3 | Explicit HIP build, CUDA build untouched, NVIDIA flags separated | **done** | `scripts/rocm/hip/CMakeLists.txt`; refuses any arch but `gfx1100`; `Strata/CMakeLists.txt` unmodified |
| S2.4 | Port device discovery, memory, streams, events | **partial** | kernels ported and tested; `src/core/device.cu` still hard-requires `cc_major == 12` and `src/core` is not built yet (S2.7) |
| S2.5 | Port dequantization, elementwise, norm, RoPE, routing, GEMV kernels; run on gfx1100 against references | **done (20/22)** | 20 pass, 0 fail, 2 skipped for missing references — `results/test-kernels.json` |
| S2.6 | Independent references for the IQ/QK quantization formats | **done** | `./strata-rocm fixtures` builds all ten (upstream's generator lived in the omitted `tools/` tree). Blocks come from ggml's own encoders; the reference VALUES come from **gguf-py**, a separate implementation in a different language. `iq_parity` now runs and passes for all ten: **`dequant rel 0.00e+00`** — the ported dequantizers are bit-exact against gguf-py — and `mmvq rel` 4.3e-3 to 6.9e-3 against a declared 2e-2 bound. ggml's C decoder and gguf-py agree exactly on all nine types gguf-py implements; `Q2_0` (gguf-py raises `NotImplementedError`) keeps ggml's values and is independently cross-checked in `tests/rocm/prefill_mmq_parity.cpp` |
| S2.7 | Port `src/core` host runtime (device.cu arch gate, graphs, session) | **done** | `strata_hip_core` builds from the same six sources as upstream's `strata_core`, and `strata-device --selftest` **runs on the card**: reports `gfx1100 (HIP capability 11.0)`, allocates a 64 MiB poisoned arena, honours alignment and refuses over-allocation. Patch 0002 replaces the `cc_major == 12` gate with a validated-target list (`gfx1100`), not a widened constant. `strata_engine` (all eight sources), `strata_spec` and the **`strata` binary itself now build and link** — 26 executables. On the card the binary parses its arguments and refuses cleanly at the first missing artifact (`cannot open <pack>/index.txt`, exit 1); that is not a correctness test, it is evidence that the startup path is intact |
| S2.7c | Native (IQ) experts through ggml's CPU backend | **done (build) / untested (behaviour)** | `STRATA_NATIVE_EXPERTS` on, via upstream's own `add_subdirectory` on the pinned llama.cpp: `ggml-base` and `ggml-cpu` build and link (27 executables), and `strata::kernels::cpu::native_experts_available()` plus 26 native-expert symbols are present in the engine binary. **Untested**: no native expert has been computed — that needs a GGUF shard |
| S2.7d | Upstream's explicit MMQ template instances (`ggml-cuda/template-instances/mmq-instance-*.cu`) | **done — and the previous conclusion was WRONG** | The checkpoint before this one recorded this as *"resolved by inspection"*, claiming `moe_mmq.cu` instantiates the templates itself. It does not: `mul_mat_q_case<T>` is only *declared* in `mmq.cuh`, and `quantize_mmq_q8_1_cuda` lives in `quantize.cu`. Neither was in the build. The new MMQ test failed to link and said so. Upstream's nine translation units are now compiled (`strata_hip_mmq`), exactly as upstream composes them. Lesson recorded: an inspection that concludes "not needed" needs a test that would have failed without it |
| S2.8 | Check subgroup widths, overflow, boundary sizes per kernel | **partial** | wave32 established as the gfx1100 default; masks widened and verified only where a test exists. Needs an explicit per-kernel boundary/integer-extreme review |

## S3 — difficult attention and prefill paths (the feasibility gate)

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S3.1 | Replace the NVIDIA PTX in `native_qsa_score.cu` with a separately testable HIP reference path | **done** | `patches/0001-native-qsa-score-portable.patch` (sha256 `e372bc6b9a08…`) replaces the `ldmatrix`/`mma.tf32` fragments with a portable path preserving the documented contract. `tests/rocm/qsa_score_parity.cpp`: 7 cases, 0 failures, bit-exact against a host model of the same arithmetic, deviation 1.1e-4 of row magnitude inside a per-row derived bound, no writes past `n_kv`, invalid-step guard suppresses all writes. **No fast path yet, and none is allowed before this test exists** |
| S3.2 | Port `src/prefill/gemm.cu` to supported hipBLAS ops | **done (GEMM) / partial (MMQ)** | `tests/rocm/prefill_gemm_parity.cpp`: 7 cases, 0 failures — transpose trap (T,N,K all distinct), padded row stride (`ldy`), `beta=1` accumulation, BF16 and FP16 paths, K=512 reductions, single-token decode. Deviation inside a derived bound (`K*2^-24*sum|terms|`), worst case 0.107 of it. **Not verified:** `kernels.cu`/`moe_mmq.cu` (ggml MMQ) now *compile* for gfx1100 but have no reference test |
| S3.2b | Independent test for the ggml MMQ path (`src/prefill/{kernels,moe_mmq}.cu`) | **done for 6 of 8 types** | `tests/rocm/prefill_mmq_parity.cpp`: weights built and decoded by ggml's own reference quantizer/dequantizer, activations chosen to quantize exactly, compared against a float64 matmul.<br>**Q2_0: bit-exact** (and a hand-written Q2_0 decoder agrees with ggml's on 256/256 values). **IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS: 0 failures**, worst error 1.6e-05 at ~0.001 of a derived bound — the model's own **IQ3_S** is in this group.<br>**IQ2_XXS and IQ2_XS cannot be built through ggml's public reference API**: `from_float_ref` is null for both, because their quantizers need an importance matrix. Unverified, and blocked on calibration data rather than on effort |
| S3.2b' | Give `IQ2_XXS`/`IQ2_XS` a fixture | **todo** | needs an importance matrix, or a hand-built block fixture plus a hand-written decoder for two codebook formats. Not attempted; recorded rather than implied |
| S3.2c | Re-enable `native_expert_parity` | **done (builds) / skipped (needs weights)** | the test builds and is maintained in the suite. It takes a GGUF shard positionally, so it declares its own argv in `tests/rocm/testargs.json`, and `unavailable.json` classifies its `usage:` output as a missing input rather than a pass. It runs the moment weights exist |
| S3.3 | Audit and port custom attention, recurrent/stateful ops and all remaining forward-pass kernels | **partial** | The audit is done: every kernel entry point the layer/session code calls was cross-referenced against every existing test, giving a concrete list rather than a guess. Eleven of the highest-value gaps are now closed — **the QSA key indexer** (`native_qsa_indexer_parity`, the last forward-path function), **the Q5_K decode-step GEMV** (`native_mmvq_q5k_parity`; Q5_K is absent from `iq_parity`'s type list), **RoPE** (`native_rope_parity`, including the IMRoPE table path and a byte-exact copy region), **the QSA sub-ops** (`native_qsa_rms_norm_weighted` on both reduction branches, `native_qsa_gate_apply`), **`native_gr_postops`** (all three functions; the FMA suspect is **cleared**), **the MoE combine** (`native_moe_combine_parity`, which found and fixed a violated accumulation contract — see below), **the GDN preprocessing helpers** (conv_silu, l2_norm, out_norm, beta_gate, gate), **the GDN recurrence** (including an eight-step stateful case), **attention** (`native_flash_attn_parity`, including NaN-poisoned padding and the invalid-step guard), **routing** (`native_router_parity`, including the documented lower-index tie-break at the 10th/11th boundary) and the **q8_1 activation quantizer** (`native_mmvq_quant_parity`, which also pins the "sum of the ORIGINAL floats, not reconstructed from the integers" rule). **Coverage, measured per kernel header rather than by name:** 37 of 42 kernel headers have at least one tested entry point; **5 have none**, each with a disposition below. **`native_embed` needed no test at all** — it is a host accessor (`src/core/native_head.cpp`), not a kernel; its work is `iq_dequant_f32`, which `iq_parity` verifies to `rel 0.00e+00` for all ten types. **`native_mmvq_*` likewise** — exercised by `iq_parity`. Both were false positives of name-based audits, `native_qsa_gate_apply`, `native_qsa_indexer_append`, `native_qsa_rms_norm_weighted`, `native_rope_apply`, `native_q5_k_f32`, `native_embed`, `native_mmvq_*` |
| S3.4 | Feasibility gate for proposing the full-model download | **PASSED for the default configuration** | 38 of 42 kernel headers covered; the four exceptions measured, not argued — see "The gate, decided" below. Three conditions attached |

## S4 — first full-model inference

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S4.1 | Present the exact model manifest and get download approval | **proposal delivered, awaiting your decision** | `MODEL-DOWNLOAD-PROPOSAL.md` + `results/download-proposal.json`: exact files, sizes and SHA256 from the HuggingFace API (metadata only), licence **apache-2.0**. **The measured blocker: `/home` has 63.6 GB free and the smallest variant needs a 109.0 GB peak** (two shards + the pack output + the MTP layer) — 45.5 GB short. The other volume, `/mnt/LINUXWINSHARE`, has 77.5 GB free but is mounted **read-only**. Options and their consequences are in the proposal. **Nothing downloaded.** |
| S4.2 | Short-context single-request run, speculation and vision off | **todo** | clean completion + short generation |
| S4.3 | Layer/logit comparison against an independent reference with identical weights and tokenization | **todo** | logits and top-token margins recorded; fluent text is not evidence |

## S5 — concurrency and repeated use

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S5.1 | Restore overlap, pinned memory, streams/events and callbacks incrementally | **todo** | each step compared against the validated debug path |
| S5.2 | Graph capture only after ordinary execution is correct | **partial** | HIP graph semantics **measured, not assumed**: `tests/rocm/core_graph_parity.cpp` — capture/instantiate/launch works; **replays re-read their input buffers** (3 replays, 0 stale values), which is the assumption that would silently corrupt an engine if it were false; the registry captures a key exactly once; an empty capture is reported as an error. Still outstanding: the per-step-pointer hazard (a graph re-reads buffers but not kernel arguments) and replay under concurrency |
| S5.3 | Repeated requests, cancellation, state reset, memory cleanup | **todo** | no hangs, no stale state, no growing allocations; failures reproducible from logged seeds |

## S6 — performance and optional speculation

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S6.1 | Benchmark the correct baseline before tuning | **partial** | `strata-rocm bench` measures kernel-level wall time with settings and VRAM recorded. No TTFT/token-rate is reported: the engine links but cannot load a model (no weights, CPU expert path off), so there is nothing to time |
| S6.2 | Tune one demonstrated bottleneck at a time | **todo** | measured improvement without a correctness regression, or a documented decision to keep the simpler path |
| S6.3 | Speculative decoding (optional) | **todo** | acceptance/rejection, KV rollback and equivalence tests first |

## S7 — local release and handoff

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S7.1 | Reproducible build/launch commands, pinned dependencies, limitations, rollback | **partial** | PORTING.md + STATUS.md exist and are accurate; a launch command cannot exist before S4 |
| S7.2 | Preserve a working local revision before experimental optimisation | **todo** | |

---

## Harness self-check (S1.4) — what it actually caught

Not an injected defect: a real one, caught by the suite and diagnosed to a root
cause rather than papered over.

| build | `elementwise_parity` embedding gather | `quantize_act_parity` Q8_K small-magnitude |
| --- | --- | --- |
| default (`-ffp-contract=on`) | 10793 bit mismatches / 108 row cases | 2 wrong bytes of 131072 |
| `-ffp-contract=fast` | 10793 bit mismatches | 2 wrong bytes |
| `-ffp-contract=off` | **0 mismatches**, and the test's required `fma_diff > 0` still holds (8141) | **0 wrong bytes** |

Root cause: clang fuses `a*b+c` inside a statement; the in-tree reference
deliberately blocks fusion with a `volatile` temporary and asserts bit-equality
with the unfused result. No tolerance was changed — the arithmetic contract was
matched. Recorded in PORTING.md §7 with its performance cost.

## Immediate next tasks (renumbered session 2; the old list had accreted duplicates)

1. **S4.1 — first inference**, gated on the user's download completing in
   `/mnt/LINUXWINSHARE` (remounted rw by the user): sha256-verify the shards
   against `MODEL-DOWNLOAD-PROPOSAL.md`, re-derive that file's stale space
   arithmetic, then launch on loopback **8081** (short context, single request,
   speculation/vision off, **8080 untouched**) and compare against whatever
   reference the shards' provenance supports.
2. **S3.2b'** — `IQ2_XXS`/`IQ2_XS` in the *MMQ* path need an importance matrix
   (`iq_parity` covers both types' dequantization and MMVQ, but not their MMQ
   prompt path; `prefill_mmq_parity` now reports the two cases as a declared
   skip instead of an invisible pass).
3. **S4 config decision (no weights needed)** — whether `native_gdn_enabled()`
   is switched on for first inference: all three `fused_gdn` entry points are
   now tested, so this is a configuration decision, not a correctness risk.
   Optional alongside: the QSA fast path (its parity test exists and passes).

## Harness bug found and fixed this round

The HIPIFY patch step was not idempotent while the translate step was: a second
`strata-rocm hipify` reused the cached tree and re-applied the file patch, leaving
a `.rej` behind. It still compiled, which is the dangerous part — a half-patched
tree that builds and reports success. Fixed three ways, because one was not
enough: patches now run only against a tree exported in the same run (and the
stamp covers the patch set, so editing a patch still regenerates); **any `.rej`
after patching fails the run loudly** and marks the source audit failed; and the
`.hipify-revision` marker is no longer counted as a source file, so
`files_total` is the real 260.

## Done since the last checkpoint

**`s2_expert_grouped` is covered, and the gate is decided.**

- **The last untested kernel file has a test** (`s2_expert_grouped_parity`). It
  tests the part unique to that file — grouping, `dst_index` routing, slot indexing
  and scratch sizing — and leans explicitly on `dequant_s2_parity` and
  `s2_gemv_parity` for the expert arithmetic, which the header itself describes as
  composed from already-verified pieces. Results: a hit inside a group is
  **bit-identical** to the same hit evaluated alone; rows no hit was routed to stay
  untouched; the permuted routing is **demonstrably distinguishable** from
  sequential writes (so the failure the header warns about would be caught);
  scratch canaries survive at exactly `moe_hit_grouped_scratch_bytes`; and
  `x_scales = null` is reproducible while supplying scales changes 7680 values.

## The gate, decided

Coverage: **39 of 42 kernel headers have a tested entry point** (session 2; the
suite re-run on upstream v0.1.15). The three that do not, each measured rather
than argued:

| header | measured status |
| --- | --- |
| `native_gr_norm` | **not called by the engine at all** — `native_gr_rms_norm_weighted` appears nowhere in `src/core` or `src/program` |
| `native_ple_postops` | on the PLE path, which needs the model's PLE table — **weight-gated** |
| `verify_kernels` | the **speculative decoding** path, explicitly out of scope for the first release (plan §9) |

**S3-feasibility passes for the default configuration**, on this evidence: every
kernel on the default, non-speculative forward path has an independent test that
passes on the real card. Session 2 additionally tested all three `fused_gdn`
entry points (`fused_gdn_parity`, `fused_gdn_conv_l2_parity`,
`fused_gdn_ab_parity`), closing the gate's first condition.

Two conditions remain, and neither is a formality:

1. **The PLE path is unverified** pending weights. `ple_parity` and
   `native_ple_postops` both need the table.
2. **Speculative decoding is out of scope** for release 1 and untested.

`native_gdn_enabled()` still defaults to false, but switching it on is no longer
a correctness risk: the flag's entire code path is now tested. Passing this
gate authorises *proposing* a download, not downloading. The proposal's disk
arithmetic is refreshed when the user's download lands (S4.1).


**The last forward-path function is verified, and the audit instrument was fixed.**

- **`native_qsa_indexer_append` verified** (`native_qsa_indexer_parity`) — the most
  stateful kernel in the tree, so the test drives a **sequence** of cells and models
  the same state machine in float64. `tail` is asserted **byte-exact**, `pooled` by
  tolerance (6e-08 to 1.6e-07), `block_pos` exactly, and the spare row is asserted
  to equal the stored `dead` key rather than a fresh computation.
- **The documented trap is demonstrably detectable.** The header warns that
  "rotating at the block's LAST cell keeps every shape and every magnitude", so a
  wrong choice survives any shape or magnitude check. The test computes the
  last-cell rotation as well and shows a gap of **1.07 to 1.82** — the two are
  distinguishable here, which is what makes the assertion mean anything.
- **A latent bug in my own test helper, found by a byte-exact comparison.** Three
  of 384 tail entries differed by exactly one ulp. The cause was not the kernel: the
  hand-rolled `f32_to_f16` copied into four test files rounded half-**up** where
  `__float2half_rn` rounds half-to-**even**. It had been latent through several
  passing byte-exact checks that happened to use powers of two. Fixed **once**, in
  `tests/rocm/f16.h`, with the tie rule written out, and the four copies removed.

## The gate, measured

Coverage is now stated per kernel header rather than by grepping names — an
instrument that has now failed three times (a false gap on `native_mmvq`, a false
positive on `native_embed`, and a list that was mostly structs and sizing helpers).

**37 of 42 kernel headers have at least one tested entry point.** The five with none:

| header | disposition |
| --- | --- |
| `verify_kernels` (18 entry points) | **speculative decoding** — explicitly out of scope for the first release (plan §9) |
| `fused_gdn` (3) | *fused* variants of arithmetic already verified by `native_gdn_parity`, `native_gdn_preprocess_parity` and `gdn_parity` |
| `native_gr_norm` (1) | the GR analogue of `native_qsa_rms_norm_weighted`, which is verified |
| `native_ple_postops` (2) | the PLE path; `ple_parity` is skipped for want of the model's PLE table |
| **`s2_expert_grouped` (10)** | **a real gap** — the S2 expert path, which a forward pass uses |

**S3-feasibility was therefore not declared passed at that point.** (It is now — see "The gate, decided" below, which supersedes this section.) The reasoning below is kept because it is why `s2_expert_grouped` got tested: Four of the five are scoped and
arguable; `s2_expert_grouped` is not — it is real code on the expert path, and
calling the gate passed with it untested would be exactly the "compilation is not a
port" failure the plan warns against. It is **one test away**.


**The audit was re-derived by symbol, and it changed the feasibility picture.**

- **A name-based audit over-reported the gap.** The previous checkpoint listed
  `native_mmvq_*` as untested because no test file is called
  `native_mmvq_parity`. The dispatcher **is** exercised — `iq_parity` calls
  `native_mmvq(type, ...)` and compares against a float64 matvec for every type it
  covers, and that test passes. Re-deriving the list by grepping for *symbols*
  shrank the remaining set from four groups to two.
- **The real Q5_K gap is closed** (`native_mmvq_q5k_parity`). Q5_K is one of the
  pack's formats and it is **absent from `iq_parity`'s type list**, so the
  decode-step GEMV for it had no test at all. Five fixtures: exact-quantizing
  activations (isolating the GEMV from the quantizer) and ordinary activations
  (where the Q8_1 rounding dominates and the bound is derived from it), across
  ncols 1/2/4 and up to a 2560-wide projection. `bound excess = 0` everywhere.
- **A test bug the kernel's own validation caught**: `native_mmvq_weight_bytes`
  takes `(type, n_in, n_out)` and I passed `(type, n_out, n_in)`, which made it
  divide by the wrong block count and throw. Noted in the test so the next caller
  does not repeat it.


- **`native_rope_apply` verified** (`native_rope_parity`): six fixtures across both
  head widths (128/256), two frequency bases, in-place and out-of-place, with and
  without the IMRoPE table, plus a pair-0 anchor. Three checks are not tolerances:
  the **copy region is byte-exact** (channels from `n_rot` up are copied, not
  recomputed — a kernel that rotated the whole row would give plausible garbage
  there), the rotation is checked **as an isometry** per pair, and pair 0 pins the
  angle itself (`powf(theta_scale, 0) == 1` exactly, so `theta == position`).
- **A tolerance that was justified wrongly, corrected rather than widened.** The
  first revision used a flat relative tolerance and failed by 10-25x. The kernel
  was fine — the isometry held to 1.6e-07 throughout — but the *bound* was wrong:
  the angle is computed in fp32 as `pos * powf(theta_scale, pair)`, and for small
  `pair` that angle is thousands of radians, so one ulp of it is ~1.6e-04 and
  `cos`/`sin` carry that straight into the output. The observable error scales with
  the **angle**, not the output. The bound is now per element:
  `(|a|+|b|) * (a few ulps of theta) + a few ulps of the output`.


- **Two QSA sub-ops verified** (`native_qsa_ops_parity`):
  `native_qsa_rms_norm_weighted` on **both** its reduction branches — it selects a
  256-thread kernel below `n_cols = 1024` and a 1024-thread one above, so a bug in
  one branch would never appear in the other — plus the exact in-place alias the
  contract allows, at ~7.1e-08 relative. And `native_qsa_gate_apply`, including a
  fixture that **poisons the first half of each `q_full` row**: the kernel must read
  the *second* half, and a first-half read would produce a perfectly well-formed
  output that is wrong everywhere.
- A layout note recorded rather than silently worked around: the header writes the
  shape as `[n_cols,n_rows]`, which reads as column-major, but the kernel indexes
  `blockIdx.x * n_cols + col` — row-major with `n_cols` as the row width. The test
  follows the kernel and says so.


- **`native_gr_postops` verified, and the FMA suspect CLEARED.** All three
  functions (`down_silu`, `pre_gated` on both documented paths, `post`, including
  the exact `output == residual` aliasing the contract allows). The evidence that
  the fused path really contracts is tolerance-free: run both paths on the same
  random inputs and they **differ on 125 of 256 elements**; if `-ffp-contract=off`
  had broken this kernel as it broke `native_moe`, they would be *identical*.
  With `gate = 0` — making every operation exact — each path then matches its own
  simulated order **bit for bit**.
- **Why this file was innocent and its sibling was not**, which is the useful
  part: `native_gr_postops.cu` names `__fmaf_rn` explicitly, so no compiler flag
  can touch it. `native_moe.cu` wrote `sum += a*b` and depended on the compiler.
  Same contract, two different amounts of care — and only the second one lost.
- **A construction that defeated itself, recorded because it is a trap.** The
  first attempt made `gate = 0` to get `sigmoid(0) = 1/2` exactly, hoping for a
  bit-exact comparison of the accumulation *order*. It gives bit-exactness — and
  destroys the discrimination, because multiplying by 1/2 is exact, so
  `fma(x,w,sum)` and `sum + x*w` become the *same operation*. The test now uses
  two constructions, each asserted only for what it can actually establish.


**A real defect, found by a test that pinned rounding rather than value.**

- **`native_moe_combine` verified** (`native_moe_combine_parity`) — and the test
  found that the kernel did **not** do what its own header contracts. The header
  says experts 1..k-1 "accumulate with FMA in expert order"; this rollout builds
  every HIP translation unit with `-ffp-contract=off` (a deliberate decision made
  in session 1 so the in-tree parity references agree — `PORTING.md` §7), and that
  project-wide flag silently made the same expression round the product
  separately. Measured: **35 to 280 of 256 to 512 elements** differed from the
  contracted reference, and the output matched the separately-rounded order bit
  for bit.
- **Fixed by patch `0004-moe-combine-fma.patch`**, which names `__fmaf_rn`
  explicitly so the documented accumulation holds whatever the contraction setting
  is — strictly more robust than depending on a compiler default that differs
  between nvcc and clang. After the fix every case matches the FMA order bit for
  bit (`vs fused=0`, `vs rounded=35..280` — the numbers flip exactly).
- **A float64 comparison could never have found this.** Both accumulation orders
  are within 2.7e-07 of the ideal; they differ from each *other* only in the last
  bit. The test therefore computes two exact fp32 simulations and reports which one
  the device matches, which is the only construction that can see it.
- **`native_gr_postops` is now a known suspect**, not an unknown one: its header
  contracts "ordered FMA accumulation" too, and no test covers it. Recorded in the
  S3.3 row so the next session starts there rather than rediscovering it.


- **All five GDN preprocessing helpers are verified** (`native_gdn_preprocess_parity`)
  — `conv_silu` (four-tap convolution, history shift, SiLU), `l2_norm`, `out_norm`,
  `beta_gate` and `gate`. Worst relative error 1.7e-07; the history shift is exact.
- **An asymmetry the header does not spell out is now tested, not argued about:**
  `l2_norm` uses `epsilon/128` while `out_norm` uses `epsilon` as given. On ordinary
  data the two are 4e-07 apart and indistinguishable, so an earlier revision's
  "they must differ" assertion was vacuous. The test now runs a **small-magnitude
  row**, where the epsilon term dominates the mean square and mixing the two up
  would be **90% off**. The softplus threshold of 20 is exercised the same way: the
  fixture checks that 12 of 96 heads actually cross it.


- **The GDN recurrence is verified** (`native_gdn_parity`) — the stateful core of
  the recurrent layers, and the place where a porting error does not show up once
  but compounds. Seven cases: identity and grouped head maps, **eight steps over
  one state buffer** with the reference iterating the same eight, no decay,
  `beta=0`, zero initial state, and twelve steps under a decaying gate. Worst
  relative error 9.4e-07 at 0.003 of a derived bound.

  The multi-step case is the point: a recurrence that failed to feed its own
  output back would pass every single-step check and fail that one.

  Two details the header does not state were resolved from the implementation and
  are flagged as such in the test rather than presented as documented: the decay
  is applied as `state_new = g*state_old + k*((v - g*kv)*beta)` with `kv` from the
  *un-decayed* state, and the readout uses the *updated* state. Both are asserted
  against the algebra they imply, so an upstream change fails the test instead of
  silently agreeing with something else.


- **Attention is verified** (`native_flash_attn_parity`) — the critical path for
  every token, and previously untested. Seven cases: ordinary, additive mask,
  single key (bit-exact), full 256 context, **padding poisoned with NaN** with and
  without a mask, and the invalid-step guard. Worst error 0.001–0.061 of a bound
  derived from the reduction length.
  The NaN-poisoned cases matter most: the contract says unused k/v rows and mask
  entries "may contain arbitrary bytes" and are synthesized rather than read, and
  the kernel passes with NaNs sitting in exactly those bytes. The heads are also
  made distinguishable so the documented `head / 12` map cannot coincide with a
  wrong `head % 2`.
- **A contract gap found.** The header documents Q24x256/KV2x256 but the
  implementation *also* requires `shapes.idx_block == 4` and
  `shapes.idx_top_k >= 256`; a caller working from the header alone gets an
  `invalid_argument`. Our test hit exactly that. Nothing is patched upstream for
  it — the header is upstream's to fix — but it is recorded, and it is the kind of
  thing that costs a future caller an hour.


**S3.3's audit is done, and the two highest-value gaps in it are closed.**

The audit was mechanical rather than guessed: every `native_*`/`qsa_*`/`gdn_*`/... entry
point the layer and session code calls was cross-referenced against every test in
`tests/rocm/` and `src/kernels/*_parity.cpp`. That produced a concrete list of
unverified forward-pass operations, recorded in the S3.3 row rather than left as
"the remaining kernels".

- **Routing** (`native_router_parity`) - the plan asks for this by name: *"Expert
  IDs and scores; explicitly handle near-ties and documented tie-breaking."* Six
  cases: ordinary logits, all-equal logits, a tie inside the top ten, **a tie
  exactly at the 10th/11th boundary** (the one that changes which expert runs), a
  one-ulp near tie under a -900 offset, and +/-88 extremes. All ids match a float64
  top-10 with the documented lower-index rule; weights sum to 1 within 1e-4 and sit
  at 0.001 of a bound derived from the 512-expert reduction.
- **The q8_1 activation quantizer** (`native_mmvq_quant_parity`) - blocks decode to
  `d = fp16(amax/127)`, `q = round(x/d)` and the **sum of the ORIGINAL floats**. The
  test includes a block constructed so that "sum of the original" and "sum
  reconstructed from the quantized integers" differ, and asserts the former: a naive
  implementation would pass every other check and fail that one.


**The IQ/QK verification gap is closed.** Ten quantization formats that were
"skipped for want of upstream fixtures" now run and pass:

| | |
| --- | --- |
| types | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_M, IQ4_NL, IQ4_XS, Q2_0, Q3_K |
| dequantization accuracy | **`rel 0.00e+00`** — bit-exact against gguf-py for all ten |
| quantized mat-vec | `rel` 4.3e-3 to 6.9e-3 against the test's declared 2e-2 bound |
| oracle independence | values from **gguf-py** (Python, separate implementation); blocks from ggml's encoders (only they can encode i-quants) |
| extra cross-check | ggml's C decoder and gguf-py **agree exactly** on all nine types gguf-py implements |

`./strata-rocm fixtures` builds them, idempotently, and the test stage calls it —
so the suite is reproducible from a bare checkout rather than depending on a
directory someone once created. `Q2_0` is the one type gguf-py cannot decode
(`NotImplementedError`), so it keeps ggml's C values; it is separately
cross-checked against a hand-written decoder in the MMQ test.

Suite: **25 passed, 0 failed, 2 skipped**. Both remaining skips need model
weights.


**The MMQ prompt path is now verified for six of the eight types it claims,
including the one the model actually uses.**

| type | result |
| --- | --- |
| `Q2_0` | **bit-exact** (`worst|err| = 0.000e+00`); a hand-written Q2_0 decoder agrees with ggml's on 256/256 values |
| `IQ3_S` (the model's own) | 0 failures, worst error 1.6e-05 at **0.001 of the derived bound** |
| `IQ2_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS` | 0 failures, same order (≤2.1e-05, ≤0.001 of bound) |
| `IQ2_XXS`, `IQ2_XS` | **cannot be built**: ggml's public `from_float_ref` is null for both — their quantizers need an importance matrix |

The oracle is ggml's own `ggml_type_traits::from_float_ref` / `to_float`, which is the
scalar definition of each format against which the GPU kernel is an independent
implementation. The activations are chosen so they quantize to q8_1 *exactly*,
which is what lets the reference use the original floats without decoding ggml's
transposed, padded q8_1 layout — a test that fails because the test is wrong would
have been worse than no test.

Also found on the way: ggml's i-quant quantizers abort with *"forgot to call
ggml_quantize_init()?"* unless that is called first; and the i-quants are not
bit-exact against the scalar decoder, only ~1e-5 off, which is fp32 rounding of
the per-block scale application and is 1000x inside the derived bound.


**The prompt MMQ path works on gfx1100, and it is bit-exact** — but only after a
real defect was found and fixed.

- **A defect in Strata's own HIP shim made the entire MMQ path unusable on AMD.**
  `src/prefill/ggml_cuda_host.cu` computed the device arch id as
  `100*major + 10*minor` — CUDA's encoding, giving **1110** for gfx1100. ggml keys
  its MMQ instance selection on that number, so every launch aborted with
  *"HIP kernel mul_mat_q has no device code compatible with HIP arch 1300"*. ggml's
  own HIP path parses the GCN string instead. Fixed in patch `0003-mmq-device-cc.patch`
  by using ggml's parser verbatim. The `1300` in the message was a red herring.
- **S3.2b done for Q2_0**: `tests/rocm/prefill_mmq_parity.cpp` — 3 model-shaped
  cases, 0 failures, **bit-exact** (`worst|err| = 0.000e+00`) against a float64
  matmul of blocks decoded in the test, with activations that quantize exactly.
  The seven i-quant types remain uncovered.
- **S2.7d corrected.** The previous checkpoint closed it as *"resolved by
  inspection"*. That was wrong: the templates are only declared in `mmq.cuh` and
  `quantize_mmq_q8_1_cuda` lives in `quantize.cu`; neither was in the build. The
  new test failed to **link**, which is what a missing symbol looks like. All nine
  upstream translation units are now compiled.
- Two bugs in the new test itself, both fixed and both instructive: `quantize()`
  takes a **device** pointer for the activations (passing host memory faulted the
  GPU on a host address), and the ggml type ids for the i-quants were guessed
  rather than read (16,17,18,20,21,22,23,42 — not a neat run), which made the test
  print a wrong "supported" count.


- **Native (IQ) CPU experts are enabled** (S2.7c). Upstream's own mechanism —
  `add_subdirectory` on the pinned llama.cpp's ggml — builds and links
  `ggml-base` and `ggml-cpu`; 27 executables. `native_experts_available()` and 26
  native-expert symbols are in the engine binary. Untested at run time: no native
  expert has been computed, because that needs a GGUF shard.
- **S2.7d is not a hole, and saying so is the finding.** `moe_mmq.cu` calls
  `mul_mat_q_case<GGML_TYPE_X>` with explicit template arguments for all eight
  types, so the templates are implicitly instantiated in that one TU. Upstream's
  eight `mmq-instance-*.cu` files are a compile-time split, not a linkage
  requirement. "We skipped eight files" was the right thing to check and the
  wrong thing to assume.
- **`native_expert_parity` builds again**, for the first time. It is skipped, not
  failed: it takes a GGUF shard positionally, so it declares its own argv in
  `tests/rocm/testargs.json` (the runner's blanket `--selftest` had been read as a
  filename, aborting it), and its `usage:` output is the declared missing-input
  signature. Skipped for a missing input, not a missing reference.


- **The `strata` engine binary builds, links and runs.** All eight `strata_engine`
  sources, `strata_spec`, the CPU expert kernels and the whole prefill set are in
  the link — 26 executables. On the card it parses its arguments and fails
  cleanly at the first missing artifact (`cannot open <pack>/index.txt`, exit 1).
  A clean refusal is not a result, and it is not reported as one: it is evidence
  that the startup path is intact, and nothing more.


- **S2.7 core half complete, and the first Strata binary runs on the GPU.**
  `strata-device --selftest` executes on the RX 7900 XTX: it reports
  `gfx1100 (HIP capability 11.0)`, prints the planner's memory plan against the
  card's actual free VRAM, allocates a poisoned 64 MiB arena, honours both
  alignments, and refuses an over-allocation. Nothing about that is a compile.
- **The architecture gate is now honest.** Patch `0002-core-amd-arch.patch`
  (`e315fc9b4a04…`) replaces `if (cc_major != 12) throw` with a list of targets
  this port has actually been tested on — currently `gfx1100` — and the refusal
  message names the architecture. Widening it is a documented decision with a
  suite behind it, not a relaxed comparison.
- **HIP graph semantics measured rather than assumed** (the plan's Stage 5
  warning). `core_graph_parity`: capture/instantiate/launch works; a replay
  re-reads its input buffer rather than replaying a snapshot (3 trials, 0 stale
  values); the registry captures a key exactly once across four calls; an empty
  capture is an error.


- **S3.2 GEMM half complete.** `src/prefill/gemm.cu` goes through hipBLAS via the
  HIPIFY translation, and the independent test covers exactly the five things the
  plan named for this file. The whole prefill device set (`gemm.cu`,
  `kernels.cu`, `moe_mmq.cu`, `ggml_cuda_host.cu`) plus the host `prefill.cpp` now
  **build for gfx1100** — the first time any ggml MMQ code has compiled for this
  target here.
- The pinned llama.cpp sources are fetched by `./strata-rocm prepare` at the exact
  commit `setup.py` pins (`3cf03257…`, MIT, 39564399 bytes, sha256 `cbe23c59…`),
  into a regenerable directory, with the manifest in
  `artifacts/rocm/env/llama-cpp.json`. Cumulative downloads: 40.5 MB of the 2 GB
  budget. No model weights.


- **S3.1 complete.** `native_qsa_score.cu` is no longer excluded: the tree now
  has **no unsupported translation units**, and the scanner reports zero because
  it no longer counts its own explanatory comments.
- Full suite: **20 passed, 0 failed, 2 skipped** (was 19/21), on the card, with
  the user's model server holding its VRAM.
