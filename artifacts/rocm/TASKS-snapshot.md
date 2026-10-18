
| IQ6 | **IQ4(a): `STRATA_VERIFY_DEFER_CPU` (#2690) ported and gated** (user-chosen option a) | **done 2026-10-18, NOT ADOPTED** — merge `3f04756b` in worktree `strata-r146`/branch `port-2690`; build-0146 green (sha256 `d2080793…`); conflicts composed with the fork's PcieModel, `hflag_in_h_`/`herr`/`hdrop`/`h_bal` handles, `lb_+k` and both docs sections; default path untouched. Parity 4/4 on the ported binary; engagement verified (", deferred" per captured window). **Quality holds** (battery 7/8 = 7/8 = 7/8, no regressions, identical answers; prose drift noise-dominated across restarts). **Performance fails**: 1K **−11.41 %** (1/5), 32K −3.57 %, 128K −3.61 % — the output change costs drafter acceptance (1K 162/166 → 146/179) even though the 32K round is −15.5 % ms/round. Production unchanged (build-0145, no env), healthy. Port dormant on `port-2690`; only a `BELOW` sweep (0.05/0.02) declared as follow-up | `artifacts/rocm/results/record-iq3s-window-d-defer-cpu-20261018.json`; logs `r30d-*`; scripts/rocm/r30d-defer-cpu-window.sh |# TASKS — Strata HIP/gfx1100 rollout

| IQ7 | **The six-item follow-up queue** (user: "do 1 - 6") | **done 2026-10-18 — 3 adoptions, 2 rejections, 1 no-go** — (1) 262K A/B: **0.1.45 +3.96% decode / +3% prefill vs 0.1.43.1** on the same workload; the 69.3-vs-56.5 gap is config, not a line regression. (2) **`STRATA_ROUTE_PRIOR=0.5` ADOPTED**: screen λ0.5/1.0/2.0 = +29.9/−3.8/+0.7% @1K; battery 7/8=7/8; clean confirmation **1K +22.20% (5/5), 32K +25.56% (2/2), 128K +0.24% tok/s with −15.4% ms/round**; CPU experts/layer 1.61→0.44-0.73, PCIe 1.19→0.20-0.89; live in serve-8081.json (`.bak-route-prior-pre`, health+Paris smoke). (3) BELOW sweep **rejected** (−2.48% @1K, −3.96% @32K). (4) gfx11 HC fusion **no-go** (persist refuses gfx11; R9700 prior no gain; instrumented-only share). (5) prefill: scorer three-way fp16 select **1,714 ms** vs exact **4,320 ms** vs radix **~10,000 ms** (total ties; closes #1298's question); **GDN chunked +5.85% @32K ADOPTED** and **PF_GEMM+PAD +4.41% @128K ADOPTED** (parity + 8K needle; `.bak-prefill-pre`). (6) hygiene: `refresh-0145` pushed `dbb0a561`; **#2224/#1296/#1297/#1298 all closed** with their numbers | records `record-iq3s-262k-line-ab`, `-window-route-prior`, `-window-below-sweep`, `-hc-fusion-feasibility`, `-window-prefill` (all 20261018); logs `r30e-*`..`r30j-*`; scripts `r30e-262k-line-ab.sh` .. `r30j-prefill-gate.sh` |

| IQ8 | **Second follow-up queue** (user: "do 1 -6"; dormant-branch question) | **done 2026-10-18 — re-baselined, tuned, sweeps closed, branches decided** — (1) live-env re-baseline + re-profile: 1K 101.8-142.0 across trajectories, 32K 73.5, 128K 52-60 (noisy), **262K 44-45 with the prior +27.8% (3/3) vs off**; prefill 2,337-2,421 @32K/128K, 2,315 @262K; **waitB 9.45 -> 3.02 ms**, CPU 2.47, the prior's 0.77 ms router kernel buys 6.4 ms. (2) λ 0.5 is a sharp optimum (0.25 −47.15%, 0.75 −38.67%); TAIL_SKIP=0 still +38-43%; **20-prompt battery 20/20 = 20/20**. (3) T08: reserve 3072 and workers auto stand (2048 −40%, 4096 −11%). (4) hygiene: shard2 hash recorded, **spec-off impossible on native packs by engine refusal**, >LLC bench no cliff, replay covered by the engine windows. (5) upstream still v0.1.45, #3024 open (gfx12). (6) **branches: keep #2514 carried on refresh-0145; keep #2690 standalone with re-test triggers** (both documented in GFX1100.md) | `record-iq3s-second-queue-20261018.json`; logs `r31a-*`..`r31d-*`; scripts `r31a-rebaseline.sh`, `r31a2-262k-prior-ab.sh`, `r31b-route-prior-tune.sh`, `r31b2-quality-battery.sh`, `r31c-t08-sweeps.sh`, `r31d-hygiene.sh` |Ordered work with acceptance gates. A stage is complete only when its gate

| IQ9 | **tasks-v2 adopted; V01 + U01 Phase 1** (user: "should we skip using the local model?") | **Phase 0/1 done 2026-10-18** — v2 installed as the live backlog (`tasks-v2.md`, deployment note), v1 archived; **local-LLM executor removed**: no relay mechanism exists, the only local models are the specimen (8081) and the off-limits 8080, and the DSH agent implements directly with a fresh-context reviewer at v2's checkpoints. **V01** boundary table: HC 5.20 ms @758 GB/s, experts 4.07 @730, head 1.06 @734 (all at HBM bandwidth → V02/HC fusion has no headroom), **dense projections 8.60 ms @558 GB/s, the only material gap (~26% under, ~2.0 ms/window ≈ 5.8%)**; router/plan 2.32 and shared+quant 2.06 are overhead-bound. **U01**: TheRock 10.2.0 / HIP 7.17.26392 / amdclang 24.0.0git / hipBLASLt 1.5.0; CK and rocSHMEM unused and not applicable; **no second nightly installed** (counters blocked by `perf_event_paranoid=2`, no compiler-shaped symptom; reopen triggers stated). Recommendation: **V05 scoped to the dense-projection layout, preceded by one discriminating micro-benchmark** (current vs 1024-B row-padded vs interleaved, T=1/T=4; falsifier: <10% → occupancy, drop V05, take router/plan fusion) | `artifacts/rocm/iq3s-decode-timeline-20261018.md`, `iq3s-u01-audit-20261018.md`, `record-iq3s-v01-u01-20261018.json` |passes; "compiles" is never the criterion. Status values: **done**, **partial**,

| IQ10 | **V01 review + calibration; first experiment selected** | **done 2026-10-18 (round 2)** — the bounded fresh-context reviewer CONFIRMED the HC/experts arithmetic and CHALLENGED the projection byte basis (×T), the assumed wall, the interleave arm and the 8.60 ms row (named GEMV stages sum to 6.25). Calibration: `q8_gemv_bench` per-shape bandwidth (qkv 695, q 769, head 912; gate 426, HC up 289, shared 308-375, k/v 340) and spec-2 vs spec-8 profiles (stage 24.04→29.14 ms for +58% rows = weights amortized; **router+plan+bell 2.25→2.37 ms, T-flat**). **V05 dropped by its falsifier**; **first experiment = fuse the router path** (2.37 ms/window, expected 0.7-1.0 ms ≈ 2.0-2.9%, opt-in, router parity gate); runner-up = batch the small GEMVs. Incident disclosed: the bench was once run against the live card via `--help`; production stayed healthy | `record-iq3s-v01-review-calibration-20261018.json`; logs `r31e-*`, `r31f-*`; V01 addendum |**todo**, **blocked**.

| IQ11 | **First v2 experiment implemented + gated (router-select skip)** | **rejected 2026-10-18** — with `STRATA_ROUTE_PRIOR=0.5` the router's own `native_router_top10_multi` selection is dead work; `STRATA_ROUTER_SKIP_TOPK=1` (commit 662d4736, **reverted**) skipped it, **exact by inspection** (`route_prior_k` reads only logits+res; ids/weights are write-only). Paired cells: 1K **+1.31% median, 3/5** (a first window's +2.91%/4/5 did not reproduce), 32K/128K inside the trajectory band (accepted 197↔207); misses the pre-declared ≥+2%/4-of-5 gate → not adopted. The router path is already one projection + one top-10 per window; the remaining redundancy is one cheap launch per layer. Runner-up open: batch the small GEMVs (z/rec/ab + shared, bound ~1.7-2.9%) | `record-iq3s-router-skip-gate-20261018.json`; logs `r31g-*`, `r31h-*` |

| IQ12 | **One-window small-GEMV test (runner-up)** | **done 2026-10-18 — not worth building** — premise corrected: z/rec/ab are stamps inside the fused GDN kernel, so the only separate small GEMVs are the shared expert's gate/up/down (+k/v). One window: `q8_gemv_bench` at 1/3/4 cols — shared gate/up 6.0/8.2/8.3 µs (39→28% of wall), shared down 7.4/9.5/10.2 (32→23%), k/v 4.5/5.8/5.9; in-situ (r31f) shared+quant 1.57 ms, HC up 1.50, k/v 0.29. A 3-job launch could save ~0.4-0.5 ms (1.2-1.5%) — below the ≥2% gate; the weights must be read and the shapes are latency-bound. No code change; runner-up closed | `record-iq3s-small-gemv-window-20261018.json`; log `r31i-*` |Baseline: `Strata/` at `b89c989a7155e984e544ddd90d1038dda7da9e3d` (see STATUS.md

| IQ13 | **HC up investigation with stop rule** | **stopped 2026-10-18 — below the gate even at the wall** — real bytes: w_up is bf16 10240×320 = **6.55 MB** per call (the 3.48 MB Q8_0 bench figure was the artifact); in situ **15.6 µs/call = ~415 GB/s = 56%** of the 750 GB/s wall, not 21-32%. New `hc_up_bench` variants: baseline 182, rows/2 448, **rows/4 (320 blk) 566 GB/s (75%)**, 8 warps 461, 4-row ILP 282. Bound: at the wall the 96 calls save **0.67 ms (1.9%)** — below the ≥2% gate; the practical best shape gives 0.4-0.5 ms (1.2-1.4%); and the bench's baseline replica is 2.3× slower than the engine's actual multi-token kernel, so the transfer is uncertain. **No engine change**; bench kept as a diagnostic | `record-iq3s-hc-up-window-20261018.json`; `src/kernels/hc_up_bench.cpp`; log `r31j-*` |for why this is newer than the plan's inspected revision, and how to go back).

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
| S2.8 | Check subgroup widths, overflow, boundary sizes per kernel | **partial (disposition recorded session 3)** | wave32 established; masks widened by declared patch rules and verified wherever a test drives them. The boundary/integer-extreme space is covered where the suite already drives it: quantization extremes (`native_mmvq_quant_parity`: zeros, amax=±127, fp16-overflow guard), NaN-poisoned padding (`native_flash_attn_parity`), shape guards (`prefill_mmq_parity`), sentinel write-range checks in every byte-exact test, first-cell/last-cell rotation traps (`native_qsa_indexer_parity`). Residual: no *systematic* per-kernel sweep of untested boundary sizes; the suite's coverage instrument (39 of 42 headers) is the guard — new kernels cannot land without a test, and tests carry their own boundary cases. Not scheduled as separate work |

## S3 — difficult attention and prefill paths (the feasibility gate)

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S3.1 | Replace the NVIDIA PTX in `native_qsa_score.cu` with a separately testable HIP reference path | **done** | `patches/0001-native-qsa-score-portable.patch` (sha256 `e372bc6b9a08…`) replaces the `ldmatrix`/`mma.tf32` fragments with a portable path preserving the documented contract. `tests/rocm/qsa_score_parity.cpp`: 7 cases, 0 failures, bit-exact against a host model of the same arithmetic, deviation 1.1e-4 of row magnitude inside a per-row derived bound, no writes past `n_kv`, invalid-step guard suppresses all writes. **No fast path yet, and none is allowed before this test exists** |
| S3.2 | Port `src/prefill/gemm.cu` to supported hipBLAS ops | **done (GEMM) / partial (MMQ)** | `tests/rocm/prefill_gemm_parity.cpp`: 7 cases, 0 failures — transpose trap (T,N,K all distinct), padded row stride (`ldy`), `beta=1` accumulation, BF16 and FP16 paths, K=512 reductions, single-token decode. Deviation inside a derived bound (`K*2^-24*sum|terms|`), worst case 0.107 of it. **Not verified:** `kernels.cu`/`moe_mmq.cu` (ggml MMQ) now *compile* for gfx1100 but have no reference test |
| S3.2b | Independent test for the ggml MMQ path (`src/prefill/{kernels,moe_mmq}.cu`) | **done for 6 of 8 types** | `tests/rocm/prefill_mmq_parity.cpp`: weights built and decoded by ggml's own reference quantizer/dequantizer, activations chosen to quantize exactly, compared against a float64 matmul.<br>**Q2_0: bit-exact** (and a hand-written Q2_0 decoder agrees with ggml's on 256/256 values). **IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS: 0 failures**, worst error 1.6e-05 at ~0.001 of a derived bound — the model's own **IQ3_S** is in this group.<br>**IQ2_XXS and IQ2_XS cannot be built through ggml's public reference API**: `from_float_ref` is null for both, because their quantizers need an importance matrix. Unverified, and blocked on calibration data rather than on effort |
| S3.2b' | Give `IQ2_XXS`/`IQ2_XS` a fixture | **done (session 3)** | The blocker dissolved for a *fixture*: ggml's low-level `quantize_iq2_xxs/xs` accept any non-null imatrix (`ggml-quants.c` asserts only non-null), and `make_iq_fixtures.cpp` already used a uniform imatrix for the dequant fixtures. `prefill_mmq_parity build_weights()` now takes the same path; all 8 MMQ types tested on the card, IQ2_XXS worst \|err\| 2.1e-05 at **0.001 of the derived bound**, IQ2_XS 2.6e-05 at 0.001. The declared skip entry stays in `unavailable.json` as a regression guard |
| S3.2c | Re-enable `native_expert_parity` | **done (builds) / skipped (needs weights)** | the test builds and is maintained in the suite. It takes a GGUF shard positionally, so it declares its own argv in `tests/rocm/testargs.json`, and `unavailable.json` classifies its `usage:` output as a missing input rather than a pass. It runs the moment weights exist |
| S3.3 | Audit and port custom attention, recurrent/stateful ops and all remaining forward-pass kernels | **partial** | The audit is done: every kernel entry point the layer/session code calls was cross-referenced against every existing test, giving a concrete list rather than a guess. Eleven of the highest-value gaps are now closed — **the QSA key indexer** (`native_qsa_indexer_parity`, the last forward-path function), **the Q5_K decode-step GEMV** (`native_mmvq_q5k_parity`; Q5_K is absent from `iq_parity`'s type list), **RoPE** (`native_rope_parity`, including the IMRoPE table path and a byte-exact copy region), **the QSA sub-ops** (`native_qsa_rms_norm_weighted` on both reduction branches, `native_qsa_gate_apply`), **`native_gr_postops`** (all three functions; the FMA suspect is **cleared**), **the MoE combine** (`native_moe_combine_parity`, which found and fixed a violated accumulation contract — see below), **the GDN preprocessing helpers** (conv_silu, l2_norm, out_norm, beta_gate, gate), **the GDN recurrence** (including an eight-step stateful case), **attention** (`native_flash_attn_parity`, including NaN-poisoned padding and the invalid-step guard), **routing** (`native_router_parity`, including the documented lower-index tie-break at the 10th/11th boundary) and the **q8_1 activation quantizer** (`native_mmvq_quant_parity`, which also pins the "sum of the ORIGINAL floats, not reconstructed from the integers" rule). **Coverage, measured per kernel header rather than by name:** 37 of 42 kernel headers have at least one tested entry point; **5 have none**, each with a disposition below. **`native_embed` needed no test at all** — it is a host accessor (`src/core/native_head.cpp`), not a kernel; its work is `iq_dequant_f32`, which `iq_parity` verifies to `rel 0.00e+00` for all ten types. **`native_mmvq_*` likewise** — exercised by `iq_parity`. Both were false positives of name-based audits, `native_qsa_gate_apply`, `native_qsa_indexer_append`, `native_qsa_rms_norm_weighted`, `native_rope_apply`, `native_q5_k_f32`, `native_embed`, `native_mmvq_*` |
| S3.4 | Feasibility gate for proposing the full-model download | **PASSED for the default configuration** | 38 of 42 kernel headers covered; the four exceptions measured, not argued — see "The gate, decided" below. Three conditions attached |

## S4 — first full-model inference

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S4.1 | Present the exact model manifest and get download approval | **done (session 3)** | You chose IQ3_S and downloaded the shards (aria2, then moved to `~/models`); the rollout fetched mmproj + MTP. **Every artifact sha256-verified** against `results/download-proposal.json` (`record-s4-artifacts-verified.json`): shard 1 `4c1eb2ce…` (verified twice — the re-download was byte-identical), shard 2 `316b46f3…`, mmproj size-exact, MTP tensors manifest-hashed. The proposal's peak-space arithmetic turned out moot: the IQ pack is 1.5 GB (`--experts-bin` is opt-in; the engine mmaps experts from the GGUF), not ≈shard-1-sized |
| S4.2 | Short-context single-request run, speculation and vision off | **done (session 3, one deviation recorded)** | One-shot greedy run: *"The capital of France is"* → *" Paris. The capital of Germany is Berlin"*; 8 tokens in 247.3 ms (32.35 tok/s), TTFT 524 ms, beside the user's 8080 server. Deviation: the engine's IQ-native path requires `--spec T (T>=2) --mtp DIR` even one-shot, so the MTP draft layer is LOADED — speculation was used by the engine's window machinery; a spec-free run needs an engine-side change (recorded, not made). Web UI: upstream `serve/server.py` + our engine on loopback **8081** — `/health` ok, OpenAI chat completion *"Name the capital of France in one word."* → **"Paris"** (finish=stop). The serve engine's pinned 47 GB expert arena caused an OOM under coexistence (kernel killed OUR process; `record-oom-incident`); the config now defaults to `--mmap-experts` |
| S4.3 | Layer/logit comparison against an independent reference with identical weights and tokenization | **qualified / deferred, reasoning recorded** | No same-weights reference exists: the 8080 server serves the **27B** GSQ-RCO (different weights), and a second ~84 GB-resident llama.cpp reference cannot coexist in 96 GB RAM beside it. Recorded as the qualified claim: clean completion + correct short generation + tokenizer round-trip + the kernel suite's independent per-kernel evidence. A full logit comparison needs a RAM window with the 8080 server stopped (user's call) |

## S5 — concurrency and repeated use

| id | task | status | acceptance |
| --- | --- | --- | --- |
| S5.1 | Restore overlap, pinned memory, streams/events and callbacks incrementally | **todo** | each step compared against the validated debug path |
| S5.2 | Graph capture only after ordinary execution is correct | **partial → semantics complete (session 3)** | HIP graph semantics **measured, not assumed**: capture/instantiate/launch works; **replays re-read their input buffers** (3 replays, 0 stale); the registry captures a key exactly once; an empty capture is an error. Session 3 closed the two residuals: the **per-step-pointer hazard is REAL on gfx1100** (kernel arguments — pointer values — are frozen at capture; a replay after buffer reallocation acts on the OLD addresses, measured 256/256 updated at old, 0 touched at new), and **concurrent replay is safe** (same graphExec on two streams: both accepted, exact; two graphs on two streams with disjoint buffers: both exact). Consequence for the engine: re-capture whenever addresses change — now a measured requirement, not an assumption. Repeated-request/stress semantics remain S5.3 |
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

## Session 6 additions — the instruments, and the first upstream PRs (2026-09-29)

| id | task | status | acceptance |
| --- | --- | --- | --- |
| I1 | What the Monitor's tok/s numbers actually measure | **done** | `scripts/rocm/lib/speedaudit.py` (`--probe` no GPU/service, `--live` against a running server, `--ab` on the card); records `results/speedaudit-{probe,live,ab}-*.json`. The live feed read 27,235.7 tok/s against a known 25.0 tok/s pace; `PORTING.md` §14d has the four readouts |
| I2 | The live feed is a rate, and a failed request keeps no counters | **done** | declared patch `0009-serve-live-rate`; probe after: max 28.0 vs a 25.0 pace (1.12x); `tests/harness/test_serve_rate.py` (3 tests) green |
| I3 | The Monitor's GPU/disk readings on ROCm | **done** | declared patch `0010-telemetry-amd` (amdgpu sysfs, `/proc/diskstats`, `/proc/cpuinfo`); `tests/harness/test_telemetry_amd.py` (5 tests) green; live `/metrics` carries the GPU series |
| I4 | Earliest-peak vs live config attribution | **done, state-qualified** | `results/speedaudit-ab-20260929T050743Z.json`: one-shot 1.52 tok/s cold vs 32.35 warm-cache; live config 30.6/35.5 tok/s short prompt; 47k prompt reads at 98 tok/s (KV-residency cliff) |
| I5 | Propose the platform-independent patches upstream | **done** | PR #104 (live rate), PR #105 (AMD telemetry), issue #106 (the port); both patches apply to upstream `main` `f1b1d96` unchanged; upstream suites 28/5/55 tests OK |
| I6 | Why a fresh engine measures ~3x the resident one on the same config | **todo** | recorded observation only: 30.6–35.5 tok/s (fresh, 96% hit) vs 10.7–11.4 tok/s (resident, 85.6%/76.3%), same DONE counters. Needs its own interleaved measurement with tier/hit-rate/RAM captured per arm |
| I7 | Publish the port if the maintainer wants a HIP series | **pending their reply** | issue #106 asks; the patch set is the offer |

## Session 7 additions — the stall-tail diagnosis and the attribution schedules (2026-09-29)

| id | task | status | acceptance |
| --- | --- | --- | --- |
| D1 | Identify the "10+ tok/s after all the patches" regression (expert panel + corpus) | **done** | `tasks/plan.md` fact base: healthy decode is **41–53 tok/s at reference speed**; the defect is a **per-request stall tail** (corpus 1.4–52.8 tok/s, in-run `51.5, 2.4, 52.3` sandwich, `tune-20260929T015309Z`); `ms/step` flips 53–56↔119–196 while `tok/step` stays 1.6–2.2 (stalls are pure waiting, drafting exonerated); patches measured equal where it matters (0008 A/B); CIFS, NUMA, MTP-collapse and kv-cliff-as-pair-cause falsified against measurements |
| Task 1 | Stall-distribution pass (`speedaudit --regimes`) | **done — clean baseline** | `results/speedaudit-regimes-20260929T071923Z.json`: 20 measured probes in one session, predeclared `decode_tok_s < 25`; **stall rate 0.0** (the 10+ state did not reproduce), 30.6–45.8 (median 43.0), healthy `ms/step` 45.25; dips 13/18 flat-countered (majflt 0, read median) ⇒ host-side shape; warmup 0 I/O signature 11.5 MB + 22 majflt at t≈6.6 s (borrow/refill); identical-probe row-cache self-warming measured and recorded as artifact. Pure-GPU floor: native packs skip graph capture (`session_replay_full: not captured with post graphs`); bounded no-pack retry inside the Task 2/3 run |
| Task 2+3 | Attribution schedules (PLE knobs vs host-loop knobs) | **in flight** | `scripts/rocm/lib/attribution.py` + `tune.unit_start(extra_env=)`: interleaved round-robin arms (row-cache 0/1M/4M, `--ple-delay-us` 200/2000 controls, `STRATA_TG_FLUSH_US` 0/5/20/100 vs 2000 default, `--pool-workers` 6/8 vs 11, `--no-prefill-borrow`), per-request `/proc` deltas + fincore of PLE GGUF/MTP sidecar + one-shot `--stats` host-phase decomposition; `results/attribution-*.json` checkpointed per pass |

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

## Session 7b additions — speed analysis on the deployed fork (2026-09-29)

Scope: the user's objective is decode **and** prefill/TTFT, acceptance metric =
highest steady-state generation rate. Source of truth is the DEPLOYED fork
(`xyzzing/Strata` @ `2ee6419`, engine 0.1.24, in-tree HIP) — `Strata/` and the
hipify pipeline remain the v0.1.20 rollback reference.

- **Decomposition instrument** (replaces the impossible native-pack
  `--gpu-only-full`): `scripts/rocm/lib/gpu_decomp.py` drives the engine's own
  `STRATA_DECODE_TIMING` + `STRATA_VERIFY_PROFILE`. Measured steady state:
  25.5 ms/token = **GPU-reach wait 22.2 (87%)** + CPU pool 0.75 + draft 0.85 +
  commit 0.47 (`results/gpu-decomp-20260929T171204Z.json`). Host-stall theory
  retired; map in `tasks/speed-map.md`, decisions in `PORTING.md` §16.
- **Opportunity map**: evidence-tagged decode (D1–D8) and prefill (P1–P5)
  candidates with Amdahl estimates; ruled-out list includes the doc-only
  `STRATA_PF_IDX_BATCH` no-op.
- **MMQ build + gate**: `build-hip-mmq/` (MMQ ON, tests ON) after two declared
  CMake fixes (`native_mmvq_multi` EXISTS guard; `qsa_prompt_attn_parity`
  CUDA-only). CTest 36 passed incl. `hip_prefill_mmq_parity`, 1 documented SKIP,
  2 fixture-blocked (`expert_parity`, `pool_test`: no Q2_0 `pack/full` fixture on
  this host — declared failures, not passes). hipBLASLt table confirmed active
  at runtime (`tuning enabled (30 rows, gfx1100, version 100100)`).
- **A/B verdicts** (frozen contract; `results/record-speed-ab-batch-20260929.json`):
  promoted `--spec-split` removal (decode +38.0%; the legacy +5.7% claim
  RETRACTED AND INVERTED) and the MMQ build + prefill env (long-prompt prefill
  416→971 tok/s, prompt +112.1%, decode −0.8% within cap; promotion via config
  `exe` + `env`, merged by `serve/server.py` `child_env`). Retracted `--prefill
  8192` +28% (exactly 0.0% at 1.6k and 10.5k tokens). Rejected
  `STRATA_VERIFY_DEVICE_PLAN=1` (−4.1%) and D1c flags (+4.1%, below the frozen
  5% bar). Unavailable: `--mmap-experts`/`--resident-cpu-experts` for native
  packs (generate.cpp:1758-1760). Headline confirmation vs the original
  config: **decode +38.3%, prompt +120.2%** (holdout confirmed). Fork gate
  `python3 .agentic/verify.py` PASS (checks.json configured this session).

## Session 9 — the upstream PR campaign (open)

- **Deliverable 1, rocWMMA QSA-select arm**: code + parity (5/5 incl. adversarial
  fixtures) + bench (10.5× at 128K) DONE on `feature/rocwmma-qsa-select`
  (`e4724ff`/`94acae8`/`b6ae967`); rollout twin = patch `0011-qsa-select-wmma.patch`
  (applies clean, 47 executables, `qsa_select_wmma_parity` passes on the generated
  tree). OPEN: the paired e2e cells (campaign service `ab-driver`, 31/120 mid-run; gate 1 PASS at +7.3% on n=7
  paired pairs at 128K; decode + 64K-holdout gates pending) → gate verdict → promotion decision.
  Close-out tooling all written and dry-run-verified: `record_e2e.py` (gates + verdict),
  `apply_rates_row.py` (README 3-col row + both DETAILS 6-col rows, idempotent, evidence-gated),
  `d8_attribute.py` (baseline vs no-doorbell floor, stage 4 of the driver).  PR #228 (draft) is open.
- **Deliverable 2, AMD rates row**: `scripts/rocm/rates_amd.py` written
  (methodology pinned to `bench/results/2026-09-29-speed-0126`; definitions
  recorded since upstream's harness is private). OPEN: the 5 sweep cells (campaign
  stage 2) → README/DETAILS rows + `record-amd-rates-*.json`.
- **Deliverable 3, D8**: the two arithmetic defects FIXED (`7a6c1ed`; reason first,
  PORTING §19) — `kProfPer` 32→33 and the wraparound guard. OPEN: `d8_capture.py`
  (campaign stage 3) → attribute the residual / no-doorbell floor →
  `record-d8-*.json` completion.
- **Pre-PR merge audit** (`record-prepr-merge-audit-*.json`): upstream +8 (engine
  0.1.28), one trivial verify.cpp conflict resolved on
  `feature/rocwmma-qsa-select-upstream` (`0165063`, builds 234/234). Both branches
  PUSHED to origin. OPEN: fill `artifacts/rocm/pr-body-draft.md`'s slots, open the
  PR, `./strata-rocm report`, STATUS close-out.
- **Environment**: engine startup peaks ~63 GB anon (expert-profile dequant) —
  kernel OOM killed several runs; the campaign gates on 70 GB MemAvailable and runs
  one cell at a time. User stopped the 8080 llama-server (recorded); that removed
  the 50× GPU-timeline outlier source.

## Session 13 additions — the 0.1.31 upstream refresh (2026-10-01)

| id | task | status | evidence |
| --- | --- | --- | --- |
| U1 | Upstream 0.1.31 delta audit and capability map | **done** | `record-upstream-0131-audit-20261001.json`; 6 of 11 pipeline patches found superseded upstream |
| U2 | Fork merge 0.1.29→0.1.31 + build + gates | **done** | merge `c304952`; `build-hip-0131`; CTest 50 pass / 1 skip / 4 declared env-gated; `record-upstream-0131-merge-20261001.json` |
| U3 | Rates re-measure on 0.1.31 → PR #319 | **done** | `amd-rates-0131-20261001.json`; decode +13–27%; 32k prefill dip gone; branch pushed + PR comment |
| U4 | Scorer rebase onto #337 → new PR | **done** | `xyzzing:pr-qsa-select-wmma-rdna3`; parity 5/5 + bench 11.64x on 0.1.31; `record-validate-scorer-0131-pr337-20261001.json`; PR #368 open |
| U5 | Pipeline re-pin to 0.1.31 (patches, CMake, tests) | **done** | patches 0004/0007/0008/0011 (0 rejects) + `device-trap-builtin` rule; CMake conversation/remote_experts/kq_avx2/q8_0/STRATA_USE_HIP; rope/RopeScaling + fma host-model + iq fixture-dir test fixes; build 47 exe, `test --stage kernels` 42/0/2; S0-source-audit green |
| U6 | Resident 8081 → 0.1.31 | **done** | serve-8081.json repointed (`.bak-0129` rollback); health ok; "Paris" finish=stop; 61.3 tok/s serve-path decode |
| U7 | Split-GGUF shard-name contract (0.1.31 engine) | **done** | symlink dir `/home/zacch/models/flash-next-iq3s/` (no file moved); `--native`/`--ple-gguf` must name the same shard set |

| U8 | 15-min opt-in pilot: WMMA e2e @128K, STRATA_GR_V3, STRATA_PARTIAL_PIN | **partial (pilot, by design)** | `record-pilot-0131-optins-20261001.json`: WMMA arm ENGAGES on 0.1.31 and its one clean cell matches the sweep (no regression) — paired gain INCONCLUSIVE (my pilot script's missing VRAM-drain gate contaminated pair 1; OOM killed pair 2 + the script); GR_V3 unmeasured (needs the frozen harness); PARTIAL_PIN **not applicable** on this config (FileExpertSource RAM-tier path never engages; upstream's own A/B measured it a wash). Resident restored. Frozen harness is the follow-up if a 0.1.31 e2e number is wanted for #368. |

## Session 15 additions — decode-and-exact-CLM campaign (2026-10-02)

| id | task | status | evidence |
| --- | --- | --- | --- |
| A1 | Phase A setup: skills, 0.1.33 fork refresh, build, gates, measured memory gate, desktop protections | **done** | merge a1df415; CTest 53/0/2+4 env-gated; peak 51.9 GB measured; MemoryMax scope + 66/70 GB gates + VRAM reserve 2500 (user-reported desktop crashes drove the redesign); contract frozen + amended |
| A2 | Phase A candidate 1: WMMA arm (decision-scoped frozen gates) | **done: KEEP OPT-IN** | gate1 PASS +9.82% @128K (n=10), gate3 PASS +4.24% @64K, gate2 FAIL −3.77% decode (cap 3%); `record-qsa-select-e2e-0133-verdict.json`; maintainer's opt-in default validated with gfx1100 data |
| A3 | Phase A candidate 2: STRATA_GR_V3 (fingerprinted pairs) | **done: RECORDED-REJECTED** | +2.7% @5 pairs (ambiguous) → extension to 10 pairs per frozen rule → **+1.34% aggregate, below the 3% bar**; `record-grv3-0133-verdict.json` |
| B1 | Phase B: audit existing reuse machinery | **done** | #342 already implements exact reuse (audit in handover record); Phase B reframed to validation |
| B2 | Phase B: model-validate exact reuse on gfx1100 | **done: PASS** | reuse scenario byte-exact A/B/A, 99% token reuse on append, restore 30-36 ms; `record-phase-b-clm-20261002.json` |
| B3 | Phase B: exchange scenario | **declared gap** | stale harness assertion vs 0.1.33's incremental KV capture (parks/restores/KV reuse visible in the log); upstream harness patch is the fix |
| A5 | Fork line updated to 0.1.35 (merge 44663ae) + comparison vs our 0.1.33 line | **done** | `record-fork-0135-merge-20261002.json`: arm files byte-identical to PR #368 post-merge; CTest 55/0 regressions; engine smoke decode 67.45 vs sweep 64.2; resident upgraded (rollback .bak-0131); 1 conflict (AGENTS.md, upstream's taken); leftover CMake cleanup committed as fixup |
| A6 | Decode knob pair on Qwen3.8-Flash-Next (0.1.35): GR_V3 final + MTP_F16 A/B | **done** | GR_V3 final (15 pairs): **+2.79% aggregate, 12/15 wins, CI [−1.8,+7.5]** — below the 3% bar, positive point estimate; `record-grv3-final-20261002.json`. MTP_F16: **rejected** — drafting fingerprint identical (1.432 vs 1.432 ms/round, no engagement), median −1.58%; `record-mtpf16-verdict-20261002.json`. FarSkip/Instella-MoE donor: **evaluated-not-adopted** (multi-GPU pre-training collectives; domain+model+stage mismatch); `record-donor-farskip-20261002.json` |
| A7 | #500 trial: merge → build → paired A/B on both paths → revert on measured evidence; gfx1100 data posted upstream | **done** | decode (1K) −2.3% median (extra phase protocol > quant work at tiny n); 32K prefill −0.6% (CI spans zero); revert verified byte-identical to pre-p500; [comment on #500](https://github.com/Niko1221/Strata/pull/500#issuecomment-5956785686); `record-p500-verdict-20261003.json` + `record-p500-prefill-verdict-20261003.json` |
| A8 | Serving config finalized from the knob campaigns: `--spec 3` adopted (+9.5% median decode, 5/5, output-exact for greedy), `STRATA_GR_V3=1` adopted (+2.79% point estimate, CI spans zero); k8v4 structurally incompatible with `--kv-resident` (recorded); devplan/pw6/pw16/pciedma rejected | **done** | `record-knob-campaign-final-20261003.json`; resident restarted on the final config |
| A10 | pcie-frac sweep (0.75/0.9/1.0): the miss-split knob vs the 0.55 calibrated default | **done: REJECTED** | medians −0.95%/+4.13%/−23.2% — no reliable gain, f100 clearly bad (in-graph DMA serialization confirmed); `record-pcie-sweep-20261003.json`. Ring-wait knob layer now EXHAUSTED: every env/arg knob on the decode critical path is measured. Remaining levers are engineering: row-kernel bandwidth (41.3→55+ GB/s, +4-6% decode), conversation-adaptive placement (fewer misses), next-window pipelining (~1-3 ms/round) |
| C1 | Campaign 1 (row-kernel bandwidth) opened: baseline measured via native_expert_parity on the real shard | **in progress** | ggml vec_dot BEATS our avx512 at 1-token shape (175 vs 216 µs, +23%); crossover at ~2 tokens already handled by #152's dispatch; the loss is AGGREGATE efficiency (41.3 GB/s over 12 workers = 3.44/worker vs the 5-6 single ceiling) → target: task→worker blob locality + dispatch overhead (~1 ms/round); `record-campaign1-baseline-20261003.json` |
| C2 | Campaign 1 conclusion: bandwidth ceiling measured (28.6 GB/s 12-thread copy; pool rows at 41.3 GB/s read-dominated = AT the wall); MT_MULT sweep flat; dispatch optimal | **done — verdict** | `record-mtmult-sweep-20261003.json`. On AM5 the CPU-miss path is DRAM-floored; remaining decode levers = placement (fewer misses) + ring-wait pipelining (hide the floor). 24 GB full-residency impossible for 47 GiB unless the #505 IQ2_XS anomaly is fixed |
| C3 | Ring-wait pipelining: implementation design checkpointed (doorbell-protocol window gate, exact-by-design, fallback STRATA_WINDOW_PIPELINE=1) | **designed — implementation next session** | `ringwait-pipelining-design-20261003.md`: wseq gate replay-safe via device-side increment (the doorbell pattern), flag resets moved post-tail-sync (stream-order safe), apply_pending + serve penalty copy + row-0 staging moved pre-draft; validation ladder: replay ctest → paired --exe-b decode cells → gate ≥+1.5% median; stale verify.hpp comment fixed on the fork line |
| C4 | Ring-wait pipelining implemented per the checkpointed design (wseq gate, arm/disarm, fallback safety, round-loop wiring behind STRATA_WINDOW_PIPELINE=1) and A/B measured | **done: REJECTED (kept as opt-in)** | Three real bugs caught by the smoke ladder (device-alias, missing disarm on fallback, missing resets on fallback) before the exactness gate passed: **greedy output bit-identical**, suites green. Paired A/B (5 pairs): median **−0.94%**, 2/5 wins → the glue was already amortized (non-WDDM launch ~free); commit `ed43021` kept as a documented exact opt-in, default off. `record-windowpipe-verdict-20261003.json` |
| C6 | Fork refresh to 0.1.38 (merge 8d766ec) + gates + #516 fix validation on our hardware | **done** | CTest 58 pass/62; resident on build-hip-0138 with the shipped 3072 MiB desktop reserve — compositor clean, serve-path 56.0 tok/s; validation posted to #516; `record-fork-0138-refresh-20261003.json` |
| P1 | Placement/prediction campaign: #477 learn→save→reload loop exercised end-to-end; learned-vs-static profile A/B | **done: REJECTED on-distribution** | learned profile (code-agent workload) vs static global profile: −1.53% median decode, CI [−2.0,+0.7] — the static profile already covers the domain (in-distribution test); the lever is CONDITIONAL on out-of-distribution workloads. #477's mechanics verified: periodic save works, SIGTERM does not save (save-every N or stdin QUIT), reload via --expert-profile. `record-placement-verdict-20261004.json`. Harness: stability-gate VRAM quantum replaced with a KFD-pid check (desktop idle VRAM legitimately varies) |
| T1 | autotune patch applied to the rocm10 tree (0.1.38): union-resolved, CMake/setup/calibrate/hpp fixed; iq_kernels.cu BLOCKED (union mangled the patch's native_down_multi_kernel) | **blocked on patch re-base** | WIP committed; `record-autotune-patch-status-20261004.json`; config strata-iq3_s.json + my-agent-prompts.json ready; the two autotune commands pending the iq_kernels.cu re-derivation |
| T4 | Autotune PR pushed upstream (#744: fork main 4da9c59 + patch, clean); 128K comparison r7-vs-r10 | **done** | 128K decode 62.0 vs 61.0 (flat, O(1)); 128K prefill CONFOUNDED - the nightly lacks a gfx1100 hipBLASLt-100500 table (engine: version mismatch 100100/100500 -> hipBLASEx fallback, the #505 mechanism); record-autotune-128k-pr-20261004.json |
| T5 | Community-benchmarks submission: results-only PR #745 (ROCm 7.1.1 vs 10.2 nightly serve-path comparison on RX 7900 XTX, per docs/COMMUNITY_BENCHMARKS.md template) | **done: PR #745 OPEN** | branch bench-rx7900xtx-rocm102 off upstream main (results only, no engine changes); README per the template + runs.json (10 rocm10 / 9 rocm7 requests, stall kept+marked) + startup/repeat scripts; raw-log truncation declared; #744 (autotune) is the separate engine-changes PR |
| T6 | Build `gfx1100-hipblaslt-100500.txt` (the T4/#745 confound): tune against the nightly SDK's hipBLASLt, validate, deploy, unconfound 128K prefill, submit to PR #745 | **done** | tuner (build-hip-0138 binary + LD_LIBRARY_PATH=$SDK/lib) → meta `hipblaslt_version=100500`, 32/32 cases, every row its correctness-gated best; A/B on the same nightly binary (3 clean cells/arm, no stalls): prefill **1687.0 vs 926.1 tok/s median (+82.2%)**, decode flat 62.3 vs 60.7, 1k sanity 69.4; deployed: fork main `c28ee6d` (local) + serve-nightly-8082.json env; PR #745 second commit `f3d6827` pushed + body updated; serve-8081.json intentionally UNCHANGED (system stack = 100100, its table matches); `record-hipblaslt-100500-20261004T103111Z.json`; harness lessons: systemd-run needs `--wait --pipe`, every cell needs `--vram-reserve-mib 3072` + a VRAM-drain gate, consecutive-failure guard. Engine PR upstream: #755 (branch table-gfx1100-hipblaslt-100500 off upstream main 99f3dbd, commit 169cfc1: table + AMD_HIP.md shipped-tables entry, measurements tabled in-body) |
| T7 | Production move per user direction: 8081 → nightly stack, 8082 candidate retired | **done, verified live** | serve-8081.json exe=build-nightly/strata + lib_dirs=sdk/lib + 100500 table env (args = finalized knobs); start script artifacts/rocm/serve-8081-start.sh (MemoryMax 66G/MemoryHigh 61G; OOMScoreAdjust omitted - silently rejected by the user manager); verified: health ok, engine 0.1.38, `tuning enabled (32 rows, gfx1100, version 100500)`, smoke 62.6 tok/s decode, draft 10/11; rollback serve-8081.json.bak-0138-711 + old build KEPT; caveat: SDK refresh ⇒ retune the table (one command). `record-production-move-20261004.json` |
| T8 | Remove the 7.1.1-stack engine builds entirely (user-directed, post-migration) | **done** | 8 fork-tree build dirs (~3.8 GB) deleted after preserving the tuner binary to `$EXP/tune_hipblaslt`; PR #745 tune-command repointed (`be18e4c`); rollback = rebuild-from-source (recipe in serve-8081.json.bak-0138-711 + live config note); scripts/rocm/README-711-builds-removed.txt covers dangling campaign-driver EXEs; strata-amd/build-hip (hipify = production PYTHONPATH) and the fork source deliberately kept; system ROCm 7.1.1 untouched (constraint). `record-remove-711-builds-20261004.json` |
| T6 | gfx1100-hipblaslt-100500 table: tuned on the nightly SDK (26 shapes × {4096,8192} tokens), validated in-engine | **done** | 128K prefill 921 (untabled) → **1615 tok/s tuned** (beats rocm7's 1535); 128K decode 65.5 vs 62.0 — ROCm 10.2 now strictly faster un-confounded; `record-autotune-128k-pr-20261004.json`; table + #505 follow-up next |
| T7 | gfx1100-100500 table installed in the rocm10 tree's tools/hip + offered to #505 (table-only PR on request) | **done** | [comment](https://github.com/Niko1221/Strata/issues/505#issuecomment-5979575779); 8082 candidate restarted with the tuned table in its config env |
| T9 | Consolidate to one production home; prune strata-amd; tidy the models share | **done, verified** | single home `$EXP` (deps moved in, config repointed, start script with VRAM-drain gate + Restart=always); swap verified (smoke Paris/58.1 tok/s, table 100500); strata-amd 3.2→1.1 GB records archive with pointer stubs; share: mtp/tensors 4.9G removed, iq3_s-full 49G flagged user-decision; `record-consolidation-20261004.json` |
| T9 | #319 refreshed with the ROCm 10.2 + 0.1.38 ladder (760/1,275/1,641/1,594/1,494 prefill, 65.1/60.4/64.4/61.8/59.2 decode, footnotes name the nightly + tuned 100500 table), rebased onto newest main, **mergeable clean** | **done** | branch pr/amd-rates-row-20260930 @ 944b7ce; rebase conflicts resolved taking our incoming rows (the rebase side-flip gotcha); 1K/4K prefill declared slightly under the 0.1.31 row |

## Session 18 additions — RDNA3 WMMA prompt attention + #505 IQ2_XS decomposition (2026-10-04 late)

| id | task | status | evidence |
| --- | --- | --- | --- |
| W1 | A0 sizing: prompt-attention share of prefill on the production stack | **done (existing logs)** | qsa attn = 26.9% @32K, 25.6% @128K of prefill GPU time — the largest phase; `rates-raw-{32k,128k}.log` phase blocks; Amdahl at 4.2x ≈ +19–21% e2e |
| W2 | A1: gfx11 WMMA arm in qsa_prompt_attn.cu (+parity gfx11 branch) | **done — parity 3/3 PASS** | MIC rdna3 tables (A[l&15][i], B[j][l&15], D[(l>>4)+2j][l&15]), hw-confirmed by probe; objdump 32× v_wmma; kernel 4.17–4.91× vs FP32 fallback; branch `feature/prompt-attn-wmma-rdna3` off upstream 99f3dbd, build-pa; `record-pa-wmma-rdna3-20261004.json` |
| W3 | A2: paired e2e A/B (frozen: ≥5% prefill @128K, decode ≤3% cap) | **done — both gates PASS** | 128K prefill **+22.8% median, 5/5** (1513.8→1863.2 tok/s); 32K +17.9% (5/5); decode flat; attn share 26-27%→8.4-10.6%; `ab-pa-wmma-20261004.json`, `record-pa-wmma-e2e-verdict-20261004.json`; committed 831812b (not pushed); 8081 restored |
| B0 | #505 IQ2_XS anomaly: kernel-level decomposition | **done — row kernels EXONERATED** | IQ2_XS 16% FASTER than IQ3_XXS at mmvq ncols=1; grouped +1.6% worst — official ordering holds on gfx1100; `record-505-iq2xs-decomposition-20261004.json` |
| B1 | #505: mechanism candidates | **done — allocation-lottery finding** | GSQ rco-allocation manifests: per-token tensors (attn proj, shared expert) carry different format mixes per pack; remaining differential must be priced per format (K-quant ncols=1 bench) or measured on the packs (weights = user decision) |
| B2 | #505: full-residency math | **done — lever CORRECTED** | IQ2_XS pack 63.35 GiB (Q2_0 61.9, IQ3_XXS 70.6, IQ3_S 77.9) — 24 GB full-residency impossible at ANY quant for this model; supersedes the C2 lever line |

## Session 18 additions 2 — 0.1.39 refresh, production opt-in, fused-native verdict (2026-10-04 night)

| id | task | status | evidence |
| --- | --- | --- | --- |
| W4 | Rebase arm onto upstream 0.1.39 (user-directed refresh-first) + rebuild + gates | **done** | 49969d1 off 6f32ec0 (1 trivial conflict); parity 3/3 (4.05-5.12x); objdump 32×; `build-pa` |
| W5 | Production opt-in: 8081 → 0.1.39 + WMMA arm (STRATA_HIP_WMMA=1) | **done, verified live** | serve-8081.json + .bak-nightly-0139-noarm; arm print, table 100500, health ok |
| W6 | Upstream: PR #786 opened + #313 measured-data comment | **done** | PR #786 (2 files off 6f32ec0); comment 5981366917 |
| F1 | Fused-native expert path STRATA_PF_FUSED=1 (env-only, local-model-discovered) | **done: REJECTED** | 128K +2.5% (3/4) < 5% gate; 1 GPU HANG at 128K (b-arm, pf-raw-b-128k-p3.log); `record-pf-fused-verdict-20261004.json`; resident restored |
| L1 | Local-model-first recon workflow | **done, characterized** | llm_recon.py + 4 targets: single-excerpt dispatch = exact (moe-dispatch verified); multi-file inventories = budget-exhausted (hand-finished); all claims verified vs source |
| L2 | hc/gdn hand recon (verified) | **done** | hc = norm+3 GEMMs+silu+mix (prefill.cpp:1976-94); gdn recurrence pipelined-cols, kh variant CUDA-only (kernels.cu:915-17); unmeasured knobs STRATA_KV_HOST_DMA/GDN_PIPELINE/GDN_REC_HEADS |

## Session 19 additions — KV_HOST_DMA + hc re-rank + donor study (2026-10-05)

| id | task | status | evidence |
| --- | --- | --- | --- |
| K1 | STRATA_KV_HOST_DMA=1 paired A/B (wait-copy lever) | **done: REJECTED** | 128K +0.35% (4/5) wash; 1K −12.5% (0/5); wait-copy phase unchanged (5.0% both arms); in-graph DMA serializes (pciedma mechanism); `record-kvhostdma-verdict-20261005.json`; 8081 restored |
| K2 | hc-read fusion feasibility (donor-informed) | **done: DEPRIORITIZED** | hc phase GEMM-dominated (prefill.cpp:1984-94); elementwise ≤2% of prefill; upstream F-1/F-2 fusions already in; donor claim decode-shaped |
| K3 | Donor GDN study (exllamav3-rocm-moe gdn.cu) | **done: design checkpoint queued** | epoch barrier + megakernel read; WY-form chunk-parallel delta rule = the 7.4% recurrence lever, own parity bound required |
| K4 | Local-model recon envelope | **done: characterized** | ~215-line single-excerpt = reliable; larger/multi-file = thinking-budget NULL |

## Session 22 additions — #368 select arm landed + WY/GDN campaign opened (2026-10-05 morning)

| id | task | status | evidence |
| --- | --- | --- | --- |
| S1 | Parity-harness nightly-toolchain demotion (6 contraction shapes, pin → diagnostic) | **done, reason recorded pre-landing** | PORTING.md §18.2; `logs/parity-select-wmma-20261005.log` all-PASS (oracle util 11.0%, 0 differing IDs, bitwise tail, neg-control fails) |
| S2 | A/B verdict: #368 select arm on 0.1.39+arm | **done: PASS, ADOPT OPT-IN** | `record-select-wmma-verdict-20261005.json` (30/30 cells): 128K prefill **+11.2%** median (gate ≥+5%), 32K +5.9%, 1K +3.5%; decode -0.0/+1.1/+4.9% (cap 3% PASS); select phase 10249→1952 ms (5.25×) |
| S3 | Land on fork main + rebuild | **done** | merge `0731865`, origin-docs merge `fc19bf7` pushed; kernel-identity diff empty vs verified branch; `build-pa` rebuilt exit 0 |
| S4 | Upstream PR | **done: #856 opened then closed in favour of refreshed `pr-qsa-select-wmma-rdna3` @ a3cf018** | pointer comment on #856; disjoint-from-#786 path per close-out |
| S5 | Production adoption | **done (user-flipped 07:21), verified live** | `serve-8081.json` STRATA_SELECT_WMMA=1; arm print + health ok; rollback = remove env line |
| S6 | GR_V3 hygiene follow-up (dead env on 0.1.39) | **done by concurrent session** | STRATA_GR_V3 REMOVED from serve-8081.json (.bak-grv3-inert) before this session's S-rows were written |
| S7 | WY/GDN recon batch (resident, 14 targets) | **done: 10/14 content=YES, 4 NULL** (wy-gdn-caller answered from source: T=8192 prefill chunks; 3 donor targets = corroboration-only, envelope NULLs) | `scripts/rocm/llm_recon_wy.py` → `artifacts/rocm/llm-recon/wy-*.md`; donor pinned: `artifacts/rocm/llm-recon/donor/` (sha256 recorded) |
| S8 | WY design checkpoint doc (consolidated with the concurrent session's in-flight C2/C3 implementation) | **done** | `artifacts/rocm/wy-gdn-design-20261005.md` — contract verified line-by-line; in-block plan (C=64, LDS); reassociation-only bound; gates PRE-DELETED→pre-declared by the C3 session (≥+5% @128K, ≤3% decode); C2 403313f done (vacuous-green lesson), C3 144d1a8 WIP (token-1 fp64 mismatch + T=1 NaN open) |

## Session 19 additions 2 — GR_V3 hygiene + WY-GDN C2 (2026-10-05 morning)

| id | task | status | evidence |
| --- | --- | --- | --- |
| H1 | GR_V3 hygiene (inert env key on 0.1.39) | **done** | key removed from serve-8081.json (.bak-grv3-inert); #375 gate recorded |
| W-1 | WY-GDN design checkpoint (full algebra from the kernel's update) | **done** | tasks/gdn-wy-design-20261005.md; split form (M = A⁻¹(diag(βg·dec)·K)) keeps columns parallel |
| W-2 | C2: algebra proof | **done: 8/8 fixtures ≤ 3.1e-13** | gdn_wy_check.cu (new HIP-portable parity target, commit 403313f); boundary/no-decay/fast-decay/β=0/T=4099 |
| W-3 | C3: device WY kernel behind STRATA_GDN_WY=1 | **next session** | column structure = the host reference's per-column GEMVs; C=64, LDS 16 KiB matrices + staged k/q |
| W-4 | C4: bench (harness T=8192/32768) + frozen A/B | **queued after C3** | gates: prefill ≥5%, decode ≤3%; expected ≈ +5% @128K at 4× recurrence |

## Session 19 additions 3 — WY-GDN C3 in progress, honest state (2026-10-05 morning)

| id | task | status | evidence |
| --- | --- | --- | --- |
| W-3a | C2 SUPERSEDED: the 8/8 algebra "proof" was v-BLIND (fixture padding wipe zeroed k AND v — decay-only dynamics) | **found + fixed** | fixture padding loop removed; real-v fixtures expose the coupling algebra |
| W-3b | Host wy64: decay-indexing fixes (rhs S0 = beta*g*dec[t]*p^K — NOT dec[t-1]; A[t][s] = beta*g*(dec[t]/dec[s+1])*kk — no extra dec[t]) | **fixed** | token-0 exact (1e-17), token-1 now differs only ~1e-4-scale (down from 1e-5 pre-A-fix... still a real discrepancy under instrumentation) |
| W-3c | Device WY kernel written (column-parallel, WY=64 sub-chunks, A in LDS) + reduction barrier race FIXED (redk/redq split) | **written; NaN in pk at T=1 remains** (sane at T=64; runs vary = race or uninit class) | commit 144d1a8; WY_DBG=1 dumps both sides |
| W-3d | C3 completion: fix the T=1 NaN + the token-1 host discrepancy, then bench + A/B | **next** | instrumentation in place (WY_DBG dumps serial64/wy64/device intermediates); the 1e-4 token-1 host mismatch is fp64-real (algebra or indexing), NOT rounding |

| W-3e (parallel session 20) | T=1 NaN root cause + fix + green | **done (their commit 9f767ef)** | tid-stride bug (kt/A 3/4 unwritten); NaN-aware metrics; device 6.7e-08, host 9/9 (worst 7.6e-11 @T=4099) |
| W-3f | Engine integration behind STRATA_GDN_WY=1 | **in flight (session 20, uncommitted)** | gdn_rec_wy_kernel in src/prefill/kernels.cu +112 lines; this session stood down from fork-tree files (two-writer race) |

| S9 | WY C3 bugs closed (device NaN root cause + fix; host mismatch already resolved) | **done** | `9f767ef`; redk-vs-redq probe localized it; tid = rg*CB+c in both strided loops; NaN-aware metrics |
| S10 | WY3 fp64-oracle bound gate | **done: worst utilisation 1.0%** | all-heads oracle64; bound (T/WY+1)·2⁻¹⁴·(1+scale); fast-decay 0/0 NaN found+fixed (product-form decay) — `0ee20fb` |
| S11 | WY2 engine wiring (STRATA_GDN_WY dispatch) | **done** | kernels.cu gdn_rec_wy_kernel + dispatch; build-pa rebuilt |
| S12 | WY4/WY5 bench + A/B | **done: REJECT (144× slower at 32k spot; A/B moot)** | `record-gdn-wy-verdict-20261005.json`; wy-spot-{a,b}-32k.log; kernel opt-in default-off; escalation = wmma folds, needs revised checkpoint + go |
| S13 | WY6 revised checkpoint (wmma folds, user go recorded) + implementation | **in flight (session 23)** | design doc §9 — fp16-class bound `⌈T/BT⌉·(2⁻⁶+2⁻¹⁴)·(1+scale)` recorded BEFORE kernel text (N amended 8→16 at design review, pre-kernel, recount recorded); PORTING §22; bulk codegen delegated to resident Qwen3.8 (8081), first-pass only, session agent diffs+gates everything |

| S13 (update) | WY6 wmma-folds campaign complete | **done: REJECT (parity GREEN 3.5% utilisation; engine 32k spot −39..−40% prefill, gdn 6.8→38.7%, ×2 reproducible)** | `record-gdn-wy-wmma-verdict-20261005.json`; gdnwywmma-raw-*.log; wy-wmma-parity-20261005.log; branch commits 5fb90ce/62852a9+bench; kernel opt-in default-off (wmma arm, fma fallback); `gdn_wy_wmma_check` CTest protects it; remaining levers in the verdict json — next decision: another revised checkpoint or close the lever |
| S14 | WY lever disposition | **done: CLOSED by decision (2026-10-05)** — both variants measured (FMA 144×, wmma ~6× per layer-chunk); ceiling ~5% doesn't justify further work | `record-gdn-wy-wmma-verdict-20261005.json` next_action; design doc §10 decision; kernels in-tree opt-in dormant, parity-protected; branch unpushed pending user merge/push go |

## Session 23 additions 2 — merge + YaRN gate (2026-10-05, afternoon)

| id | task | status | evidence |
| --- | --- | --- | --- |
| S15 | Merge WY branch → fork main + push; rebuild build-pa from merged tree (restores the autotune line the morning rebuild lacked) | **done** | main fc19bf7→c81936d pushed; build-pa 11:30 binary; 8081 swapped + healthy; NOTE: the "gfx1100 table not on fork main" memory note was STALE — the table landed on main in f3d6827; the #745/#755 items are upstream-PR matters only |
| Y0 | YaRN 256K/512K feasibility gate | **done: FEASIBLE via kv-resident streaming; ladder manifest awaits approval** | `artifacts/rocm/yarn-512k-feasibility-20261005.md` — 12 QSA layers × 1056 B/cell (0.77 GiB @64K measured); VRAM 23.3/24 GiB today → all-resident impossible, streaming keeps VRAM flat (pinned 3.1/6.2 GiB); 2026-09-28 needle campaign already validated yarn past 262K on the pre-GDN NV stack; gates G0/G-adopt/G-reject pre-declared; P0-P4 manifest, ~2-3 h, no downloads |
| Y1 | 262144/524288 ladder (P0-P3, +P4 optional) | **queued behind user approval of the §5 manifest** | standing stop/restore windows; verdict record on completion |
| Y1 | 262144/524288 YaRN ladder | **done: REJECT (decode gate)** — recall ✓✓ (needle FOUND at 512K depth 50%), prefill ✓ (1536 tok/s @512K), decode ✗ (28.0 vs floor 37.1: MTP acceptance 0.803→0.056 at depth; q4 unchanged ⇒ not bandwidth) | `record-yarn-ladder-20261005.json`; yarn-ladder/needle jsons; gates pre-declared, none moved; 8081 restored healthy |
| S16 | Upstream PR for the WY campaign | **done: [Niko1221/Strata#888](https://github.com/Niko1221/Strata/pull/888)** — branch feature/gdn-wy-chunked-experimental off upstream/main 6f32ec0, campaign squashed to one self-contained commit (b24816d), archive refs scrubbed, build+parity green on the PR branch; concurrency check done (no GDN-WY PR from anyone upstream) |
| S17 | TheRock 7.10 refresh | **DROPPED by user decision (2026-10-05) — no comparison wanted** | recon-only (setup.sh/AMD_HIP.md install mechanics + disk check); nothing downloaded or built; production untouched. Context kept: current stack is already TheRock-built ROCm 10.2.0a20260930 nightly; community 7.10 reports ranged +1-4% to +29-34% (#432) on other hosts |
| S18 | QSA attention recon (checkpoint-first) | **done: PREMISE CORRECTED + DISPLACED** — production share at 128K is 9.9% (the 18.7% was a non-production arm's 32k profile); #786's WMMA arm already banked 3.6× on the term (old arm A/B: 23.5s→6.6s, prefill 85.4→66.0s); kernel not memory-bound (~131 GB/s logical, 68× cross-query locality); top block is now the MoE GEMM cluster (~36.5%) | `artifacts/rocm/qsa-attn-recon-20261005.md`; qsa-attn-recon-{old,dump}.log; 8081 restored healthy |
| S19 | Session-23 retro: 8 items applied | **done** — (1) run-build.sh/run-cell.sh wrappers, (2) parity-gate.sh one-command gate, (3) llm_codegen complete-files rule, (4) AGENTS.md numbers-provenance rule, (5) fork STRATA_VERSION git-describe (f570bab; gotcha: add_compile_definitions mangles parenthesized values), (6) tools/qsa_locality.py landed + pushed (e9f4603), (7) exploration-budget rule in memory, (8) rocwmma-qsa.md traps addendum | fork main c81936d..e9f4603 pushed; build-sel binary verified `0.1.39-759144f-dirty`; production build-pa picks the version string up at its next rebuild |
| S20 | MoE GEMM cluster recon (checkpoint-first, Qwen-driven) | **done: bracket measured, fused arm dead on gfx1100 as-shipped, candidate scoped** — MMQ vs BLAS 4.1× at gate/up (11.1s vs 45.3s); fused arm does not engage (PTX mma.sync class, gather phase persists); upstream lane clear (#820 dense-side corroborates); 7,970/24,576 experts resident ⇒ split kernel-vs-PCIe-floor BEFORE tile work; RDNA-aware dispatch-layer tiles = the scoped candidate (fork-owned call site, zero vendored edits) | `artifacts/rocm/moe-gemm-recon-20261005.md`; moe-recon-{base,fused,blas}.log; llm-recon/wy-wmma-moe-gemm-r{1,2}; gates pre-declared (+3%/3% cap/mmq parity); INCIDENTS: 8080 down (owner's), 8081 served by 0.1.24 unit all day (identity = owner's call) |
| S20 addendum | dispatch-layer RDNA-awareness candidate | **DISPLACED** — vendored mmq.cuh already ships mmq-config-rdna3.cuh, HIP device path compiles it deterministically, host cc-mapping verified by MMQ running (wrong mapping ⇒ #420 fallback to BLAS, which did not happen) | moe-gemm-recon checkpoint §7; remaining lever = fused-arm INT8-WMMA translation (own checkpoint + go) |
| S21 | fused-WMMA campaign checkpoint | **GREEN-LIT (user go) — M1 spike PASS** (rocWMMA int8→i32 bit-exact on gfx1100, adversarial reference); scope = 3 asm sites, 8 formats (format-generic loop, portable decoders), integer-exact contract + fp32 reassociation rider, gates pre-declared (+3%/3% cap/fixtures); M2 = IQ3_S gate/up translation first | `artifacts/rocm/moe-fused-wmma-checkpoint-20261005.md`; spike artifacts/rocm/spikes/rocwmma_i8_spike.hip.cu; inventory llm-recon/wy-wmma-moe-wmma-c1 |
| S22 | M2 translation — first pass | **in progress: checkpoint+spike+primitive done; kernel composition next** — Qwen draft v1 REJECTED (stubbed dot loops + a smuggled `(sc>>4)&1` semantic change, reverted), v2 degenerated (repetition loop), v3 single-mma primitive GOOD; handoff with composition plan: `artifacts/rocm/moe-wmma-m2/handoff.md` | session 24 opens with the kernel composition + fixtures |
| S22 addendum | M2 fixture round: m2f scaffold good / decode+scale-order GLM-owned; m2d truncated. Completion plan recorded in the handoff | **next session** |
| S23 | M2 COMPLETE: fused-WMMA IQ3_S gate/up fixture GREEN | **done: rel RMS 0.000e+00 (integer-exact dots), worst row 1.27e-05 (fp32 order only; bounds 1e-4/1e-3)** — decode+staging+mma+scale chain proven on gfx1100; CTest `prefill_fused_iq_wmma_test` registered (mmq branch, GGML_USE_HIP, shim macros undef'd); 5 fixture-caught bugs fixed (sign-spread port, missing decode pass, ws layout, all-threads fold, half_at integer conversion); pushed 66bbcc5 | next: M3 (all formats + down path + requant epilogue + in-tree native_kernel_wmma integration) → engagement cell + gates |
| S23 addendum 2 | M3 down path GREEN | **done: Q2_0 [2560x640] x requantized H — rel RMS 1.2e-07, decode integer-exact (REFQ2_0==DEVQ2_0); Qwen's Q2_0/IQ4_NL decode units landed (byte_perm pool → direct table, fixture-judged); 2 fixture-side test bugs fixed (blob stride 36→18, H-scale slot placement)** | pushed dd07d2a; remaining M3: in-tree native_kernel_wmma integration (routing tables + batch machinery) → M4 engagement cell → gates |
| S24 | M3-completion: in-tree native_kernel_wmma written + wired; fixture regression OPEN (dd07d2a rebuild fails worst-row-1.0 deterministically; earlier PASS suspect-stale) | **in progress — WIP stashed** (`git stash`, "M3-completion WIP") | handoff addendum: moe-wmma-m2/handoff.md; next session: fresh configure/build at dd07d2a, stage-trace debug, then M4 engagement cell |
| S25 | M3 down path: decode exact, aggregate open | **in progress: Q2_0 decode bit-exact vs to_float; aggregate NON-DETERMINISTIC (1.48-1.53 one build class, 2.6e5 another) — recorded skip-with-reason; race or uninitialized LDS read in the down kernel** | committed efc423c pushed; next: compute-sanitizer racecheck/initcheck on the down kernel, then the M4 engagement cell (gu path alone may already clear +3% — the gu term is 17.2+5.9 = 23.1% of the cluster) |
| S26 | M4/M5: engagement cell + verdict + upstream PR | **done: REJECT at parity (-0.8% end-to-end median, gate/up +1.7% noise; gate +3% FAIL) — arm opt-in dormant, parity-protected; upstream PR [Niko1221/Strata#999](https://github.com/Niko1221/Strata/pull/999)** (branch moe-fused-wmma-gfx1100 off upstream/main 6f32ec0, single commit bc5a005: kernel + fixture + CTest; build+fixture verified on the branch); 8081 restored healthy | `record-moe-fused-wmma-verdict-20261006.json`; moe-eng-{base,fused}-p{1,2,3}.log |
| S27 | QSA-attn recon continuation: INT8-Q candidate scoped | **done: checkpoint addendum §7** — INT8-Q path (per-64 max-abs int8 Q quant + rocWMMA int8->i32 + fp32 scale fold) halves the score-product MMA count (issue-bound kernel); precision class = the engine's accepted int8-codes-x-group-scales (K already int8 in KV); prototype scaffolded, acceptance pre-declared (5e-2/2e-2/2e-2) | qsa-attn-recon-20261005.md §7; next session: fixture implementation → go/no-go → kernel integration |
| S28 | INT8-Q fixture: per-group staging fixed; ONE constant-garbage element class remains (rel RMS 3.17e4, deterministic) | **in progress — WIP uncommitted** | tests/hip/qsa_int8_q_fixture.cu (untracked WIP); handoff moe-wmma-m2/handoff.md INT8-Q addendum; next: double-launch diff probe + the second-m-tile check (the MoE gu bug signature) |
| S29 | M2/M3 down-path fixture GREEN in-tree; gdn_wy_check fault = VRAM window (8081 loaded), not a regression — rerun in a stop window | **verified** | prefill_fused_iq_wmma_test: FIXTURE PASS + DOWN PASS; gdn green earlier today in the stop window |
| S30 | INT8-Q fixture GREEN after 4 fixes (barrier, B-ptr, 4-k-step mma, k_deq aliasing) → gates: isolation 4.9e-8 PASS, softmax Δ 9.5e-2 / output Δ 1.55e-1 FAIL → **VERDICT REJECT (structural: int8 max-abs pays D·2⁻⁸ under the 12× outlier class; hi/lo incumbent ≈ 2⁻²² ≈ fp64)** | **CLOSED — no kernel integration** | fork 65a3c8f (fixture committed), log artifacts/rocm/qsa-int8q-fixture-run.log, verdict §7.1 of qsa-attn-recon-20261005.md, handoff addendum 3; numpy cross-check max ~3.2 confirms honest quant class |
| S31 | CI fixed (b03e899: MoE fixture target gated under STRATA_ENABLE_HIP — CPU job green; HIP image → rocm/dev-ubuntu-24.04:10.1.0-full for the upstream `__shared__ alignas` error; CUDA job remains red: our WY kernel 50,436 B static LDS > nvcc's 48 KB static limit — dynamic-smem port queued as parity-gated follow-up) + production rebuilt/restarted on fork main 65a3c8f (engine 0.1.39-65a3c8f, hipBLASLt table loaded, smoke OK) + MMQ tile survey (RDNA3 table verified: IQ3_S I=64/J∈{16,32,48,64,96,128}; launch J selected in VENDORED switch_J — M≤96 breaks at J=96, M≈160 ties at J=96/I=128/256t; fork lever = re-dispatch via header templates or upstream STRATA_MMQ_TILE_J) + MonoMoE assessed (recon lens: decode-phase/FP8-wgmma — prices the decode-share measurement, no port) + upstream PR #1007 assessed (xjc10: QSA block scores → rocBLAS SGEMM + on-start solution tuner, gfx1030, +2.8% e2e 128K — COMPLEMENTARY to our select-WMMA arm; same phase, different hardware class; conflict surface qsa_select.cu + dispatch chain; possible gfx1100 A/B: WMMA scorer vs tuned-SGEMM scorer) | **RECORDED** | moe-gemm-recon §6 addendum; llm-recon/wy-wmma-moe-gemm-mmq-tile-r1.*; upstream: #999 #888 #786 #755 #744 open; un-PR'd fork content: select-WMMA arm (file with #1007 reference), STRATA_VERSION, qsa_locality.py, INT8-Q fixture |
| S31b | Releases brought current: tag **v0.1.39-rocm.1** cut at 116a8e1 — the release workflow ran for the FIRST time ever (prior releases were manual, no assets): release.yml HIP image 7.1→10.1.0-full + publish made resilient to a failed matrix arm; **hip-gfx1100 built GREEN, `strata-hip-gfx1100` + sha256 attached**, cuda-sm_89 failed as expected (WY LDS 48KB gap), curated title/notes applied per the fork convention | **DONE — release is Latest** | github.com/xyzzing/Strata/releases/tag/v0.1.39-rocm.1; workflow-fix commit on main; CUDA binary intentionally absent (gap recorded in the release notes) |

## Session 25 additions — the 0.1.40 refresh (2026-10-06)

| id | task | status | evidence |
| --- | --- | --- | --- |
| R1 | Update local strata to latest upstream | **done: branch `refresh-0140` = v0.1.40.1 + 21 curated commits, build GREEN, host CTest 21/22** — upstream rewrote history (forced-update; no merge base) so the update is a replay, not a merge; fork main c21dc05 untouched; nothing pushed | `artifacts/rocm/results/record-refresh-0140-20261006.json`; STATUS session 25; /tmp/build-0140d.log |
| R2 | "Patch with my submitted-but-unaccepted PRs?" — answered by evidence | **moot as a separate step: upstream 0.1.40 absorbed/superseded the class** — #755 MERGED (table byte-identical, authorship kept); #786 superseded WITH CREDIT (upstream's own gfx11 prompt-attn arm, same env key); select arm upstream-side (92dde2f; the re-file queue item is DEAD); #744/#319/#745/#888/#999 auto-closed by the history cleanup ("not a rejection") with re-file instructions; #745's bench folder still absent upstream | gh PR states 2026-10-06 06:40–10:12Z; close comments quoted in the record |
| R3 | WY/GDN chain onto 0.1.40 | **done: carried dormant** — kernels.cu conflict resolved keeping BOTH 0.1.40's Aurora block and our WY kernels (upstream `WY` is a scoped template param, no collision); STRATA_GDN_WY dispatch auto-merged into `gdn_recurrence_variant` variant-0 | branch commits 403313f..da2087a replayed; build-0140 gdn_wy targets present |
| R4 | Autotune port onto 0.1.40 (#744 re-file prerequisite) | **DONE — re-filed as upstream PR #1297 (open), GPU-verified 2026-10-07** — branch `refile-744-autotune` = re-port `66092b4` + seam rework `23b447b` (selection-space identity token, dispatch-knob refusal, one X-macro rows list, true-default tuner baseline); on-card: ctest strict ×4 PASS, parity through engaged table, refuse paths verified, tune campaign honest null ×7 (0 shapes changed, 67.9–73.1 µs); PR body carries the measurements; production swap waits for the maintainer | `record-autotune-refile-audit-20261007.json`, `record-autotune-refile-gpu-20261007.json`, `logs/stage1-refile744-20261007T030705Z.log`; https://github.com/Niko1221/Strata/pull/1297 |
| R5 | 0.1.40 re-baseline measurement (the real gate) | **PENDING the user's card window** — GPU CTest/parity full set; upstream's own gfx11 arms' parity on gfx1100 (validated on gfx1151 only per their close comments); paired A/B upstream-arms vs production 0.1.39 arms; #1008 decode-regression check (likely answered by #783); then main move + push + production swap decisions | record R1 `gpu_question_for_the_window` |
| R5 | 0.1.40 re-baseline measurement | **PARTIAL: parity DONE (86/87 + 2 declared skips), #1008 DONE (no decode regression on gfx1100, paired +2.5%), arms runtime/static engagement DONE — 32k/128k arms comparison BLOCKED by desktop memory** (kernel OOM at 48.5G peak vs 49 GB available; frozen ≥66 GB gate) | `record-refresh-0140-20261006.json` §gpu_window_20261006; /tmp/ctest-gpu-0140.log; ab-refresh0140-20261006.json |
| R6 | Production restore after the window | **done** — health 200, "Paris" finish=stop, engine 0.1.39-65a3c8f, 2958 MiB VRAM free | engine-8081.log current generation |
| R5b | Deep-tier window (32k/128k arms comparison) | **done: 20/20 cells; 0.1.40 −2.27% prefill @128K vs production, the ENTIRE delta = the qsa select phase (1934→3176 ms; our fp16 arm 1.64× faster than upstream's bf16), all else phase-flat; decode mixed ±2-5%** — swap options recorded (stay / as-is / re-land fp16 select then swap); quiet-desktop +15% context note recorded | `ab-refresh0140-deep-20261006.json`; record §gpu_window2_deep_20261006; STATUS session 25 final |
| R7 | Upstream post: the fp16-vs-bf16 select comparison | **done: posted on PR #944** (the shipped arm's thread) — [comment 6016189613](https://github.com/Niko1221/Strata/pull/944#issuecomment-6016189613): select phase 1934 ms (fp16) vs 3176 ms (bf16) = 1.64× on gfx1100, e2e −2.27% @128K, prompt-attn arms equivalent, re-file offer per the maintainer's own recipe, AI-disclosure line | link + record-refresh-0140-20261006.json |
| R8 | Re-land the fp16 select arm on the 0.1.40 line | **done: commit `700c026` on refresh-0140 — parity all-PASS (oracle util 0.0%), paired A/B vs production ×1.0009 (+0.09%, 4 clean pairs; p5-b flagged: the 8081 restore raced it) = PARITY; the bf16 arm's −2.27% fully recovered; gfx115x/gfx12 keep upstream's bf16 kernel** | `ab-reland-20261006.json`; record §fp16_select_reland_20261006; GFX1100.md 7f09401 |
| R9 | X.com graphic | **done: `artifacts/rocm/xpost/gfx1100-ladder-20261006.png`** — 2×2: prefill ladder 1,403→2,150; output tok/s vs RTX 5090 (165, IQ2_XS, byte-normalized near-parity) and RTX 5070 (53); prefill vs 5090 (5,779, gap 3.7×→2.7×); the fp16-vs-bf16 select phase (1.64×) | sources footnoted in the image; upstream bench/results/2026-09-30-community-rtx-5090 |
| R10 | Production swap to refresh-0140 + branch push | **done: 8081 runs engine `0.1.40-700c026`** (fp16 select + upstream prompt-attn prints current-generation, 100500 table engaged, smoke 'Paris'); server.py refreshed to the 0.1.40.1 hotfix line; **branch pushed** to xyzzing/Strata @ 700c026; #944 follow-up posted ([comment 6016997902](https://github.com/Niko1221/Strata/pull/944#issuecomment-6016997902)) with the PR offer | record §production_swap_20261006; rollback: exe→build-pa/strata + serve-backup tgz |
| R11 | Tonight's decode-ROI sweep (skills-contracted, <=3 candidates) | **done: +11.3% decode ADOPTED — spec 5 (+7.3%) and STRATA_SH_STREAM=1 (+4.0%) compose to 80.6 vs 72.3 baseline (one-shot @1K, 4 pairs each, all cells rc=0)**; max4tsum +0.9% below gate; **NORM_ROPE parity-REJECTED on gfx1100** (rope_parity WRONG 3644/49152 — upstream's HIP-off default vindicated, knob stays off); floor curve 0.70/0.60/0.50 → 77.4/66.0/72.4 | `record-ab-night-20261006.json`; ab-night-20261006.json (run-2 cells) + driver stdout for run-1; production restarted on the adopted config |
| R12 | Deep-tier validation of the adopted config | **done: HOLDS — spec5+SH_STREAM +7.3% @32K and +8.6% @128K decode (16/16 cells); the old spec-3 config sags with context (72.3→65.9) while the adopted config stays flat** — adoption fully validated at every tier; vs 4090's 102 @128K they keep the long-context decode lead (noted, not actionable tonight) | `ab-night-deep-20261006.json`; record §deep_tier_validation |
| R13 | Autotune UI on top of the 8081 web UI — "should there be a UI for our autotune?" | **decided: NO — not built (user-approved 2026-10-07)** — the workflow is batch (tune → byte-exact verify → e2e confirm → table → config env → restart), not interactive: the `./strata-rocm` CLI + records interface is the right shape; the webui is upstream's `serve/server.py` (a patch there is a permanent fork surface paying the replay tax on every refresh — the same reason the autotune port itself was dropped from the 0.1.40 refresh); nothing to front while the re-port is DEFERRED (R4); rates must be record-sourced (the webui live tok/s was already caught being a non-rate — STATUS "The web UI's live tok/s was not a rate") | `artifacts/rocm/results/record-autotune-ui-decision-20261007.json`; revisit trigger: the re-port lands + tuning becomes recurring (SDK-refresh ⇒ retune gate) → extend `./strata-rocm status`/`report` with tune state (**landed early 2026-10-07**: `scripts/rocm/lib/tune_state.py`, status/report "Tuning artifacts" section, records-sourced — see the session-26 STATUS entry); max concession = a static read-only page over `artifacts/rocm/results/` JSON — never a webui patch |
| R14 | Re-file the history-rewrite-closed PRs onto upstream's new main | **done 2026-10-07 — 5 PRs open: #1292 (rates row, payload-only reconstruction), #1293 (bench), #1296 (GDN WY, composed with main's Aurora kernels, parity re-run green), #1297 (autotune port + reconciled calibrate.py, 34 tests green), #1298 (select fp16 per the #944 offer, parity green); #999 superseded by main's own fused line (no re-file); #755 merged / #786 superseded upstream already. Cross-links on all old PRs; GFX1100.md updated (`09c85ff`)** | `artifacts/rocm/results/record-pr-refile-20261007.json` |
| R15 | Autotune next-optimization round (post-#1297): staging axis, workload flags, DeepGEMM | **DONE 2026-10-07** — staging: wash for production formats, IQ2_S gate/up −4.6/−2.3% OFF (docs+records, no code); rows null robust in every arm; `--windows/--weights` flags landed (33a73c5, in PR #1297); measured window mix ≈2.4 avg (capture on branch engine, production args); rows null re-confirmed on the measured mix; DeepGEMM = Separate Ways (no RDNA3 fp8/TMA, wrong format contract, wrong bottleneck); counter-split RAN 2026-10-07: **tile-J REJECTED** (transfer floor 79% of the cluster, kernel slack 6.3% prefill ceiling; `record-counter-split-20261007.json`, `scripts/rocm/counter_split.py`) | `record-autotune-nextopt-20261007.json`; logs nextopt-window1..5 + window-capture-*; memory moe-gemm-recon addendum |
| R16 | 512K decode refresh (Y1 cell on current production engine + adopted config) + competitor comparison | **DONE 2026-10-07** — 23.35 tok/s median vs the 28.01 anchor = **−16.7%: the adopted config inverts past MTP acceptance collapse** (gate II now 37.8% vs the 60% floor; 512K REJECT strengthens); prefill +3.0%; competitor's "up to 65 @512K" = content-driven n-gram regime, no like-for-like prose number on their side; levers unchanged: depth-aware draft policy or content-driven drafter. NOTE: the "arms-vs-arms deep tier" item was already done 2026-10-06 evening (ab-refresh0140-deep) — do not re-run | `record-yarn512-refresh-20261007.json`; yarn512-cell1..2-20261007T115451Z.log |
| R17 | Production → upstream v0.1.40.2 (user-directed) + graph-capture survey | **DONE 2026-10-07** — refresh-01402 (24 replayed + fixup: the replay auto-resolution dropped the fp16 arm's .cu content, re-landed keep-both with upstream's new SIMT kernel); parity ×4 PASS; **128K prefill −29% (61.2 s, attn 26→9.8%)**; rings-wait 21.8 ms/round persists → spike GO; **production = 0.1.40.2-3ddb3f8**, rollback .bak-01402-pre → build-0140; smoke-gate lesson: thinking-starved content=null (raw body or bust); survey: partial-residency capture still open upstream; queued cells: CPU_SHARE=auto, SPEC_GUMBEL=1 | `record-swap-01402-20261007.json` |

## Session 27 additions — the 0.1.41 refresh (2026-10-10)

| id | task | status | evidence |
| --- | --- | --- | --- |
| R18 | Update local strata to latest upstream (v0.1.41) | **done (branch + gates): branch `refresh-0141` = v0.1.41 (fb58e0db) + 24 replayed commits in fresh worktree `strata-r141`; history LINEAR (v0.1.40.2 is an ancestor — merge-base e8ca9af), ordered cherry-picks, no forced-update reconstruction; conflicts kept-both: kernels.cu (upstream's new STRATA_GDN_CHUNKED vs our STRATA_GDN_WY — both compiled, chunked first), CMakeLists ×2, README sections, GFX1100.md (delete/modify, kept); Gate P: every fork-touched file byte-identical to refresh-01402 except the two expected conflict files — the 0.1.40.2 silent-drop trap did NOT recur; build GREEN (fresh build-0141, SDK compiler pin verified, MMQ ON, 349/349), version 0.1.41-60184598; host CTest 93 PASS / 5 declared / 2 skips — all parity suites + fork gates green without the card; `hip_prefill_fused_iq` OOM (card held by 8081) NOT pre-declared — GPU rerun decides; nothing pushed; refresh-01402 + production 8081 untouched; Strata/ anchor held at 944b7ce. **Second pass (same day): independently re-verified — the 2 skips PROVEN content-safe (fixture gate = upstream's own `if(STRATA_ENABLE_HIP)` CMakeLists:1568; README/GFX1100 content byte-identical), full-run ctest evidence reconciled (`/tmp/ctest-host-0141.log`), GFX1100.md staged-banners + STRATA_STAGE_PIN-unset note committed as `ac7c1fc2`; see the record's `reverification_20261010_second_pass`** | `artifacts/rocm/results/record-refresh-0141-20261010.json`; /tmp/{cmake,build,ctest-host}-0141.log; STATUS session 27 (+cont.) |
| R19 | 0.1.41 GPU window (the real gate) | **done 2026-10-10, GREEN**: window ~12:20–12:50 (8081 was already cleanly stopped 11:26, MemAvailable 69–71 GB, VRAM drained); static f16 check PASS (carved all 57 fatbin code objects: our `block_scores_wmma_gfx11_kernel` = 8× f16 wmma, upstream arm = 48× bf16 — the drop trap did not fire); strict parity **7/7 PASS on-card** (iq 19 s / native_grouped 25 s / iq_multi 34 s / qsa_select_wmma 50 s device run / gdn_wy 21 s / gdn_wy_wmma 93 s / **`hip_prefill_fused_iq` 56 s — OOM question CLOSED, environmental**); 128K cell exit 0: prefill **2145 tok/s** (band 2098–2157), select **1974 ms** (1.9–2.0 s band, fp16 engaged print live), decode 53.33 tok/s = parity with the 01402 cell (52.05, same cold 64-token shape); 8081 restored 28 s, health ok, Paris smoke PASS (thinking disabled); engine stamp 0.1.41-60184598 | record-refresh-0141 §gpu_window_20261010_R19; logs/r141-profile-*.log; /tmp/0141-window/ |
| R20 | Production swap decision to 0.1.41 | **done 2026-10-10 ~13:44 (user-approved)** — `scripts/rocm/swap-0141.sh` (clone of the 01402 template): stop 8081 → 4-suite strict parity on build-0141 4/4 PASS → config exe swap `r1402/build-hip/strata` → `r141/build-0141/strata` (`.bak-0141-pre` kept) → 8081 up ~4 s, health ok, engine log `session is up (engine 0.1.41-60184598)`, smoke 'Paris' finish=stop with thinking disabled (the starve rule is now IN the script); rollback target = `.bak-0141-pre` → the 0.1.40.2 binary; the piggyback 128K cell was not re-run (R19's identical cell 30 min prior — noted in-script); VRAM 23.3 GiB loaded | logs/swap-0141-*.log; record-refresh-0141 §production_swap_20261010 |
| R21 | STRATA_GDN_CHUNKED (upstream, new) vs STRATA_GDN_WY (fork #1296, dormant) on gfx1100 | **queued** — upstream's chunked WY-form recurrence lands in the same dispatch; measure the A/B (both opt-in, both compiled) before any claim that one supersedes the other; the #1296 arm stays dormant-and-protected until then | refresh-0141 kernels.cu fork note |
| R22 | Autotune re-port onto 0.1.41 | **queued (R4 pattern)** — PR #1297 branch (`refile-744-autotune`) based on 0.1.40-era main; rebase/re-port campaign, not part of this refresh | TASKS R4; PR #1297 |
| R23 | Production swap decision to 0.1.42 | **done 2026-10-11 (backfilled 2026-10-13)** — `refresh-0142` = v0.1.42 (`61b3fb5d`, 235 commits) + 27 fork commits; `swap-0142.sh`, gates green (parity ×2 strict 5/5; bands 128K 2168 / select 1931 ms / 1K decode 76.3), engine `0.1.42-0ebe4053`, rollback `.bak-0142-pre` → build-0141. **Backfill note: this row's machine-readable record was never landed** — numbers sourced from STATUS.md session-27-cont (provenance named); the R24 band-investigation is partially calibrated against its unreconstructable 1K cell shape, which is why R24 gated the swap on a paired engine-vs-engine contract instead | STATUS.md session 27 cont.; record-refresh-0143-20261013 §provenance_gaps |
| R24 | Refresh production to upstream 0.1.43.1 | **done 2026-10-13** — `refresh-0143` = v0.1.43.1 (`a0116b2`, 105 commits, linear) + 30 fork commits replayed in `strata-r143`; Gate P 16/19 byte-identical and **caught upstream's `decode_defaults_arch_ok()` silently reverting the PcieAuto-on-gfx1100 port** (composed `pcie_auto_fork_arch_ok()`); dsh chooser verbatim. Build 381/381 stamp `0.1.43.1-8a15c99c`; host ctest 105/113 (5 = declared env-gated). GPU window: parity 4-suite ×2 8/8, fused_iq PASS, static f16-WMMA PASS. Absolute band cells missed (2136.5 vs 2143-2189; 70.9 vs ≥74) → paired engine-vs-engine investigation **NO-REGRESSION (+0.22% @128K / +1.26% @1K)**; swap 4/4 parity, Paris smoke, engine log `0.1.43.1-8a15c99c`; rollback `.bak-0143-pre` → build-0142 (relinked after the objcopy in-place incident — owned in the record). Flag probes: TAIL_SKIP **+8.6% decode n=1** (queued, output-changing), ADAPT_OVERLAP/AUX_CPUS flat, KV_GROW not eligible, remote-expert-opt N/A | record-refresh-0143-20261013.json; logs/r24-window-*, r24-inv-*, swap-0143-*; GFX1100.md `5a68370f` |
| R25 | STRATA_ROUTE_TAIL_SKIP gated campaign on HIP | **done 2026-10-13, ADOPTED** — campaign per testplan item 6 with gates pre-declared in the script header: KL declared infeasible on native packs before rates (dump-logits 0-byte measured); task battery A-B-A 7/8 = 7/8 = 7/8 (0/8 answers differ), prose drift 0.756 vs restart-baseline 0.481; paired decode **+13.97% @1K (4/5) / +18.56% @128K (3/3)**; serve-confirmed **+9.3%** (132.63→144.99 tok/s). Env line live in serve-8081.json (`=0` restores base outputs). Not filed upstream (maintainer mid-flight on the HIP tail-skip gates) | record-r25-tailskip-20261013.json; scripts/rocm/r25-tailskip-campaign.sh; logs/r25-campaign-*, r25-cell-* |
| R26 | Decode-flag batch: spec 6 / lookup-chain 2 / pool-workers / k8v4 / mtp-q4 / ADAPT_OVERLAP-serve | **done 2026-10-13, ADOPTED --lookup-chain 2** — the window-economics re-probe: lc2 (rejected -5.6/-6.9% pre-TAIL_SKIP) now +6.30% @1K (3/3) / +19.92% @128K, output ids byte-identical across arm starts, serve +3.54%; spec 6 REJECT (+1.03/-6.03 — depth optimum didn't move), k8v4 REJECT (+4.4 @1K / -4.9 @128K), mtp-q4 REJECT (-0.49), pool-workers 11 re-confirmed, ADAPT_OVERLAP serve -3.36% auto-reverted; calibrate queued (wrong invocation, printed help) | record-r26-decodeflags-20261013.json; scripts r26-decode-window.sh; logs r26-* |
| R28 | Measure the two new upstream 0.1.44 default-ON gfx1100 paths (`STRATA_COMMIT_BATCH`, arena huge-page gate) + file upstream | **done 2026-10-14** — A/B on build-0144 (the live production binary), 5 arms × 1K×4 / 128K×2 interleaved cells on production-shaped args/env. **Headline: both new defaults are safe on gfx1100, and neither shows a measurable steady-state effect here.** (1) `STRATA_COMMIT_BATCH`: the 1K cells fall into two trajectory states whose mix alone swings ms/round ~10% (T3-heavy ≈28.2, T5-heavy ≈31.2), so the naive 4-rep median reading of +5.11% was mix confounding — at **matched mix** base vs `=0` is 28.25 vs 28.20 (1K) and 36.06 vs 35.90 (128K) = sub-noise. (2) Arena gate: **never fires on this box** — 15/15 recent production sessions and 0 gate-fired lines in the whole 75,438-line log request `MADV_HUGEPAGE`, because the gate reads buddyinfo at arena SETUP (previous instance already freed), not while an engine runs; forcing the 4 KB branch costs **+8 s @1K / +5 s @128K of startup** with no steady-state change. The user-requested `STRATA_ARENA_THP=1` arm is a **no-op** in this window (base already gets THP); arm D (`NO_ARENA_THP=1`) was added to make the page comparison possible. Noise floor from the control arms ≈±1.3%. **Corrected**: my first reading claimed the gate fires in production (it was measured with the engine running — not the state the gate sees); the engine's own log is the authority. Scripts had two cosmetic bugs after the run (buddyinfo awk, page-strategy detector) — fixed. Production untouched and verified healthy after each run. **Filed upstream**: issue [#2361](https://github.com/Niko1221/Strata/issues/2361) (both paths) + comment on [#2126](https://github.com/Niko1221/Strata/issues/2126#issuecomment-6160027716) (the gate data, on the issue that motivated the fix) | record-r28-0144-defaults-ab-20261014.json; logs/r28-ab-*, r28b-reps-*, r28-cell-r[1-4]-*-*; scripts/rocm/r28-defaults-ab-0144.sh, r28b-commitbatch-reps.sh |
| R27 | Refresh production to upstream v0.1.44.1 (user-directed; "update production to latest upstream version 0.1.44") | **done 2026-10-14** — target confirmed with the user as **v0.1.44.1** (`296b59d2`, upstream main HEAD, an 11-commit hotfix over the `v0.1.44` tag they named). `refresh-0144` = v0.1.44.1 + 32 fork commits replayed `-x` in `strata-r144` (282 upstream commits / 358 files); 2 conflict commits, both in `generate.cpp` (upstream's new `mtp_first_default_on()` landing on the PcieAuto helper; upstream's new `--prefill-head-alias` landing on the ShortReadCost/dsh-chooser hunks) — all composed keep-both. **Gate P: 14/19 byte-identical, 2986 fork-added lines checked, 0 lost**; the 0.1.43 silent-revert trap did NOT recur (engagement line auto-merged intact). Build 403/403, stamp `0.1.44.1-7d4356d7`. Host ctest 111/116 (same 5 declared env-gated class). Card window: parity 4-suite ×2 **8/8**, `hip_prefill_fused_iq` PASS, static f16 arm PASS; first cell attempt SKIPPED at 64495 MiB < the 68000 gate, re-run at 82722 MiB (not waived). Paired cells, verdict on **trajectory-independent ms/round** (decode tok/s is acceptance-driven — same-binary 0143 spread 38.5%): prefill @128K **+0.14%**, 1K ms/round **−12.19%**, 128K ms/round **−7.29%**, select 1941 vs 1940 ms, fp16 engaged = **NO-REGRESSION**. Upstream audit found two new default-on gfx1100 paths (`STRATA_COMMIT_BATCH`, arena huge-page gate) — both exercised by the cells; frozen `server.py` (byte-identical to upstream v0.1.40.1) audited **COMPATIBLE**. Swap 4/4 parity, engine log `0.1.44.1-7d4356d7`, Paris smoke finish=stop; rollback `.bak-0144-pre` → build-0143 `0.1.43.1-8a15c99c`. **Disclosed**: the inherited R24-era paired verdict is two-sided and reported a +7.16% improvement as "REGRESSION SUSPECTED"; corrected in `r27c-verdict-0144.py` (direction fixed, tolerance unchanged at 3%) | record-refresh-0144-20261014.json; logs/r27-*, r27b-cells-*, r27c-verdict-*, swap-0144-*; scripts/rocm/build-0144.sh, r27-gpu-window-0144.sh, r27b-paired-cells-0144.sh, r27c-verdict-0144.py, swap-0144.sh |

| DOC1 | Export the optimization ledger (adopted / closed-with-reasons / pending ranked) into GFX1100.md (user-ordered) | **done 2026-10-18** — new section "The optimization ledger — adopted, closed, pending (2026-10-18 export)" in `~/projects/strata-r145/GFX1100.md` between "Measured and rejected" and "Credits"; 11 adopted rows, the closed set grouped by closure class with the stop reason per line, pending ranked 1-16 (top: deep-tier 0.1.45 re-measure, R21 GDN A/B, PF_GEMM isolate, SCORES_THREAD @262K, ROUTE_PRIOR) + a maintenance line; every figure names its existing record; no new measurements; upstream-PR states excluded per instruction; **uncommitted** in strata-r145 (docs commit on ask) | the ledger section itself; STATUS.md "2026-10-18" note |

## Session 30 additions — the upstream PR sweep and the queued R30 window (2026-10-18)

| id | task | status | evidence |
| --- | --- | --- | --- |
| R29 (backfill) | Refresh production to upstream v0.1.45 | **done 2026-10-18 (row backfilled 2026-10-18; its record landed but TASKS was never updated)** — `refresh-0145` = v0.1.45 (`477ac13c`, 264 commits / 267 files vs v0.1.44.1) + 35 fork commits replayed `-x` in `strata-r145`; Gate P 14/19 byte-identical, 3 851 fork lines checked / 8 rewritten in the 3 documented compositions / 0 lost; build 423/423 stamp `0.1.45-8eb3e85d`; host ctest 124/130; card window parity ×2 8/8, fused_iq + new f16 sibling PASS, static f16 PASS; paired one-sided contract prefill +0.45 % @128K, ms/round −0.83 % @128K / +0.83 % @1K = NO-REGRESSION; select **1 699 ms** vs the band's 1 850–2 050 = OUT OF BAND LOW, investigated to 0.1.45's `82922f55` wide-top-k-on-HIP (band amended with reason, ≥3 100 FAIL line untouched); swap 4/4 parity, engine `0.1.45-8eb3e85d`, Paris smoke; rollback `.bak-0145-pre` → build-0144. **Numbers sourced from the record; this row re-measures nothing** | `artifacts/rocm/results/record-refresh-0145-20261018.json`; logs/r29-window-*, r29-cell-*, swap-0145-*; scripts/rocm/r29-gpu-window-0145.sh, swap-0145.sh |
| R30 | Pull all open upstream PRs + reconcile the board (user ask) | **done 2026-10-18 (audit only; no measurement)** — **849 open PR heads fetched** into `.scratch/pr-audit/strata-prs.git` (bare mirror, 106 MB, 0 failures) + metadata and per-PR deltas; upstream main is exactly `v0.1.45` with **0 commits after the tag**, so no refresh content exists today. **Maintainer dispositions landed 2026-10-17 on four of our six PRs**: #2224 ported as opt-in (default unchanged), #1296 WY rejected (loses −39…−40 % prefill here; #2299 chunked gfx11 WMMA is in main), #1297 tuner not carried, #1298 fp16 select not added next to the exact scorer (#1943). Ledger corrections: #846 closed unmerged, #1655 closed, #2299 closed-with-content-in-main, #2224 **open** not merged, **#1367 and #1943 are already in the 0.1.45 binary** (dormant cells, not ports). New gfx1100 queue ranked: **#2514** (9-line HIP decode top-k @262K, 1.7–1.8× kernel on this card, same ids), #2588, #2964, #2630, #1525. Seven in-tree dormant knobs identified (SCORES_THREAD exact scorer, SELECT_SIMT, QSA_ROWS, QSA_SPLIT_ATTN, EXPDEC, HC_UPMIX, PF_F16); PROMPT_ATTN_IMMA is in main's source but not compiled into build-0145 (0 strings, corrected 2026-10-18). **Doc debt found**: no R29 row (backfilled above) and no R29 STATUS section; GFX1100.md DOC1 uncommitted; `refresh-0145` unpushed; live config note says "#2224 as merged" | `artifacts/rocm/upstream-pr-sweep-20261018.md`; `artifacts/rocm/results/record-upstream-pr-sweep-20261018.json`; `.scratch/pr-audit/{open-prs,pr-diffs}.json` |
| R31 | **R30 window (queued)** — deep tiers + scorer three-way + #2514 + GDN chunked | **queued; needs the user's 8081 window** — (1) 128K/262K deep-tier re-measure on the `0.1.45` binary (ledger pending #1); (2) **scorer three-way @128K and @262K**: `SELECT_WMMA=1` (current) vs `SELECT_WMMA=0 + SCORES_THREAD=1` (exact, bitwise) vs both off — if the exact scorer ≥ fp16 + 3 %, retire the fp16 arm and close #1298 with the data (this replaces ledger #4's framing); (3) cherry-pick **#2514**, rebuild, `qsa_topk_parity` (same ids) + 262K decode cell (gate ≥3 % ms/round); (4) **`STRATA_GDN_CHUNKED=1` vs default** at 8K/32K prompt (output-changing → battery + drift template), then close #1296; (5) if time, IMMA vs `HIP_WMMA` @128K and the free `SELECT_SIMT` / `HC_UPMIX` A/B. Mechanics: the `r29-gpu-window-0145.sh` shape (stop 8081, 68 000 MiB gate, strict parity first, restore trap); no rebuild except #2514 | this row; `upstream-pr-sweep-20261018.md` §6; `scripts/rocm/r29-gpu-window-0145.sh` |
| R32 | **PR hygiene (queued, after the cells)** — close the four disposed PRs with their own numbers, ping #1292/#1293 | **queued** — #2224 (reply with the gfx1100 default-on divergence), #1296 (WY numbers + the chunked decision), #1297 (accepted rejection), #1298 (the scorer A/B result); one ping each on #1292/#1293. No force-pushes, no rewrite | R30 row; maintainer comments 2026-10-17 |

## Session 31 additions — IQ3_S decode kernel campaign, windows A/B (2026-10-18)

| id | task | status | evidence |
| --- | --- | --- | --- |
| IQ1 | T01/T02 pin + harness readiness for the IQ3_S decode campaign (user-ordered, backlog `tasks.md`) | **done** — `fingerprint.json` pins code/build/model/pack/config/hardware; the pack's `native_experts.txt` **confirms the backlog's format table exactly** (gate/up IQ2_S 20, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1; down IQ4_NL 39, Q2_0 9; 512 experts; 98,227,200 B/expert; 7 pairs); `upstream.md` maps #2514 (only applicable external change), #3024/#2997 (gfx12), #646/#783 (closed), llama.cpp #26301 (not in pin, K-quant). Harness: `native_expert_bench` works on the IQ3_S shard; `STRATA_EXPDEC` **engages on gfx1100**; `expdec_bench` is CUDA-only in CMake (deviation recorded) | `artifacts/rocm/iq3s-decode-campaign-20261018/{fingerprint.json,upstream.md,campaign.md}` |
| IQ2 | **Window A**: parity + EXPDEC kernel matrix + attribution | **done 2026-10-18** — gates met (MemAvailable 68,587 MiB, VRAM 2,235 MiB); strict parity **6/6** incl. under `STRATA_EXPDEC=1` (EXPDEC bitwise-correct on gfx1100). **K1 REJECTED**: gate/up EXPDEC 922.2 → 991.1 µs (G20 T4), tie at G8 T1. **Down finding**: `_DOWN=1 _D20=1` is **−21…−34 % at the real G34 T1 envelope** (every format). Attribution: wait-for-rings 23.41 of 35.4 ms/round; CPU experts 1.55/2.15 per layer; **PCIe 55 % of misses**; routing **k=10**, 3.36 rows/round | `record-iq3s-window-a-20261018.json`; logs `r30-iq3s-*`; script `r30-iq3s-decode-window.sh` |
| IQ3 | **Window B**: campaign baseline + down e2e + GPU stage profile | **done 2026-10-18** — baseline **1K 96.09 tok/s** (5 cells, 0.46 % spread), **32K 66.32**, **128K 61.57** (noisy); acceptance deterministic (162/166, 70/103, 36/56). **K2 REJECTED e2e**: down arm **−3.34 % @1K (5/5), −2.85 % @32K** despite the kernel win (the CUDA EXPERTDEC lesson reproduces). Stage profile (serve path, fusions off; indicative): 35.07 ms/window = **waits 26.9 %**, projections+attention 25.3 %, **HC 15.6 %**, expert VRAM hits 11.2 %, shared+quant 6.1 %, copy+combine+PCIe 5.3 %, router 4.4 %, head 3.3 %; MTP draft 4.77 ms/round (11 %). **Conclusion: no in-binary decode kernel lever clears +2 %**; HC has no decode arm (`HC_UPMIX` prefill-only, `HC_PERSIST` refuses gfx11) | `record-iq3s-window-b-20261018.json`; logs `r30b-iq3s-*`; script `r30b-iq3s-window.sh` |
| IQ4 | **Next (queued)**: T08 overlap or a gfx11 HC decode fusion | **queued, decision recorded** — (a) port `STRATA_VERIFY_DEFER_CPU` (#2690, output-changing: a layer's light CPU rows join at the next layer's FFN half; the 27 % wait tracks the 5.40 ms/window host serialization) behind the battery/drift gate, or (b) write a gfx11 HC decode fusion kernel (15.6 % cluster; the persistent barrier is the gfx11 blocker). Choose by mechanism strength vs effort; (a) has the measured mechanism, (b) is pure kernel work | campaign.md D5–D7; stage profile in `record-iq3s-window-b` |
| IQ5 | **#2514 port at 262K** (objective item) | **done 2026-10-18, NOT ADOPTED** — port commit `ed3606a6` (opt-in `STRATA_TOPK_DECODE_REG=1`, default OFF); **the PR premise does not hold on 0.1.45**: 262K decode already takes the wide kernel (`campaign3`/`82922f55`), and 131K takes the register kernel either way, so the port is a wide-vs-register choice at 262K only. Parity **4/4** (`qsa_topk_parity` + reg-always, both defaults; same ids). 3 paired 262K cells: +4.45 % (cold first base), −0.16 %, −0.37 %; **median ms/round +0.17 % (spread 4.44) → gate FAIL, not adopted**. Dormant opt-in kept; production config untouched (131 072 context is dispatch-identical). **Binary provenance**: `build-0145/strata` rebuilt in place at `ed3606a6`, sha256 `b0d443f9…`, version string still `0.1.45-8eb3e85d` (baked at configure time); 8081 was restarted onto it and is healthy. Side data point (not like-for-like): 262K decode **56.5 tok/s** here vs 69.3 on the 0.1.43.1 sweep → R31 deep-tier item | `artifacts/rocm/results/record-iq3s-window-c-20261018.json`; logs `r30c-topk-*`; script `r30c-iq3s-262k-topk-window.sh` |