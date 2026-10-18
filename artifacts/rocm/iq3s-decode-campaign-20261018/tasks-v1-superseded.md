> **SUPERSEDED 2026-10-18** by `tasks-v2.md` (deployment note: no local implementation model;
> DSH agent implements, fresh-context subagent reviews). Kept for its execution history: the per-task
> status table, the two follow-up queues, and the evidence pointers. Do not execute from this file.

# Tasks: ISTA IQ3_S decode optimization for Strata on gfx1100

Updated: 2026-10-18

## Objective and scope

Improve committed decode tokens/second for ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, **IQ3_S**, on RX 7900 XTX using Strata's HIP backend under ROCm. Preserve the model's represented weights and validated inference behavior. Target the user's Ryzen 9 7900X / 96 GB RAM / Fedora system; confirm its current configuration before measuring.

## Execution status (2026-10-18) — this is the live tracking copy

The backlog said "no gfx1100 runtime measurements or HIP compilation were completed in this
workspace" when it was written. That is no longer true: four production windows ran, the model
was pinned from the pack, and every task below has a status and an evidence pointer. Original
task text is kept verbatim below; this table is the status, and `campaign.md` is the decision log.

| task | status | evidence |
| --- | --- | --- |
| T01 pin | **done** | `fingerprint.json` (code/build/model/pack/config/hardware); the format table below was confirmed from the pack's own `native_experts.txt` (identical counts) |
| T02 upstream recheck | **done** | `upstream.md`: #2514 applicable (and later measured), #3024/#2997 gfx12, #646/#783 closed unmerged, llama.cpp #26301 not in the pin |
| T03 decode/residency attribution | **done** | window A: wait-for-rings 23.41 of 35.4 ms/round; CPU 1.55/2.15 per layer; PCIe 55% of misses; k=10, 3.36 rows/round. Window B: baseline 96.09/66.32/61.57 tok/s; stage profile (waits 26.9%, projections 25.3%, HC 15.6%, expert hits 11.2%) — caveat: profiler disables fusions |
| T04 expert-kernel harness | **done** | `native_expert_bench` on real rows (7 pairs, bitwise `0 differing`); `iq_parity`/`iq_multi_parity`/`s2_expert_grouped_parity` incl. under `STRATA_EXPDEC=1`; `expdec_bench` is CUDA-only in CMake (declared deviation; matrix run through the parity suites instead) |
| T05 gate/up tuning (IQ2_S/IQ3_XXS/IQ3_S) | **tested, rejected so far** | EXPDEC gate/up slower at the grouped shape (922→991 µs G20 T4) → rejected. No other gate/up candidate has passed T09 yet |
| T06 IQ4_NL down / Q2_0 | **tested, rejected** | EXPDEC down −21…−34% at the kernel level (G34 T1) but −3.34% @1K / −2.85% @32K e2e → rejected (CUDA EXPERTDEC lesson reproduces) |
| T07 expert reuse during verification | **measured; no win** | k=10, 3.36 rows/round, entries/expert p50 6 (routing trace); the reuse mechanism tried was EXPDEC, rejected above |
| T08 placement / CPU / transfer overlap | **one adoption, one rejection; the task's own sweeps not run** | measured 27% wait + 5.40 ms host serialization; #2690 CPU-row deferral ported and rejected (−11.41% @1K) and its BELOW sweep rejected; **`STRATA_ROUTE_PRIOR=0.5` ADOPTED** (+22.20% @1K 5/5, +25.56% @32K, +0.24% @128K with −15.4% ms/round; CPU experts/layer 1.61→0.44-0.73, PCIe 1.19→0.20-0.89). Still open: the residency-budget sweep, the CPU-thread re-tune and the n-gram locality measurement |
| T09 acceptance gates | **applied to every candidate** | battery + paired 1K/32K/128K + parity per candidate; no tolerance moved; production restored after each window |
| T10 secondary GPU work | **top-k done; prefill separate and partly done** | #2514 (deep-tier top-k) ported and measured **neutral** → dormant; BF16 re-assessment N/A (no BF16 patch in this tree); the prefill cells ran separately and adopted GDN chunked (+5.85% @32K) and PF_GEMM+PAD (+4.41% @128K); the exact scorer measured 2.5x slower at the phase than the fp16 arm |
| P3 packaging | **not started, and not needed yet** | the three adopted items are env-only (no code lands upstream), so there is nothing to package; the two code ports stay dormant on their branches (`ed3606a6` on refresh-0145, `3f04756b` on port-2690) |

Follow-ups after the first four windows (the six-item queue the user approved):

1. **done 2026-10-18** — 262K deep-tier A/B, 0.1.43.1 vs 0.1.45 on the same workload/config:
   **0.1.45 is +3.96% faster** (median; pairs +2.70/+3.96/+39.11, the third 0143 cell a stall);
   prefill also +3%. The 69.3-vs-56.5 gap was **config/methodology, not a line regression**.
2. **done 2026-10-18 — ADOPTED λ=0.5.** Screen: λ 0.5/1.0/2.0 → +29.9%/−3.8%/+0.7% at 1K; battery
   A-B-A 7/8=7/8=7/8, no regressions. Clean confirmation: **1K +22.20% (5/5), 32K +25.56% (2/2),
   128K +0.24% tok/s with −15.4% ms/round**; CPU experts/layer 1.61→0.44-0.73 and PCIe 1.19→0.20-0.89.
   Live in serve-8081.json (`STRATA_ROUTE_PRIOR=0.5`, backup `.bak-route-prior-pre`, health+Paris smoke).
3. **done 2026-10-18 — no rescue.** BELOW 0.05/0.02 on top of the live ROUTE_PRIOR=0.5: 1K median
   **−2.48%** (2/5), 32K **−3.96%** (0/2), rounds slower — NOT ADOPTED; the deferral stays dormant
   (`record-iq3s-window-below-sweep-20261018.json`).
4. **NO-GO 2026-10-18** — gfx11 HC fusion: the only fused form refuses gfx11 (no `s_memrealtime`) and its
   closest RDNA prior (R9700, #2226) saw no gain; the 15.6% share is instrumented-only. Reopen with an
   uninstrumented HC share ≥10% or a barrier-free shape (`record-iq3s-hc-fusion-feasibility-20261018.json`).
5. **done 2026-10-18 — two adoptions and one answer.** Scorer three-way @128K: fp16 arm select
   1,714 ms vs exact 4,320 ms vs radix ~10,000 ms; total prefill ties — the exact scorer is not a
   gfx1100 win and #1298 closed with this. **GDN chunked +5.85% @32K ADOPTED** and **PF_GEMM+PAD
   +4.41% @128K ADOPTED** (prefill-only, output-changing; gate = `hip_gdn_chunked` parity + an 8K
   needle answered identically in every arm).
6. **done 2026-10-18** — refresh-0145 pushed (`dbb0a561`); **all four disposed PRs closed with
   their numbers**: #2224 (gfx1100 default-on divergence), #1296 (WY rejected), #1297 (tuner not
   carried), #1298 (fp16 arm 2.5x faster than the exact scorer at the phase, total ties).

Live production env after the queue (2026-10-18): the previously adopted set plus
`STRATA_ROUTE_PRIOR=0.5`, `STRATA_GDN_CHUNKED=1`, `STRATA_PF_GEMM=1`, `STRATA_PF_PAD=1` (all
output-changing; backups `.bak-route-prior-pre` and `.bak-prefill-pre`).

## Second queue (2026-10-18, late) — executed

1. **Live-env re-baseline + re-profile.** 1K **101.8-142.0** across the drafter's two trajectories
   (100 % acceptance → ~140, 200/200 → ~102), 32K **73.5**, 128K 52-60 (noisy, paired prior check
   neutral), 262K **44-45** with the prior **+27.8 % (3/3)** vs off; prefill **2,337-2,421 @32K/128K**,
   **2,315 @262K**. The window moved: **waitB 9.45 → 3.02 ms**, CPU 2.47, GPU-reach wait 24.78; the
   prior's extra router kernel (+0.77 ms) buys 6.4 ms.
2. **Prior tuning + quality.** λ 0.5 is a sharp optimum (0.25 **−47.15 %**, 0.75 **−38.67 %**);
   with `TAIL_SKIP=0` the prior still wins **+38-43 %**; **20-prompt battery 20/20 = 20/20**, no
   regressions.
3. **T08 sweeps.** Live `--vram-reserve-mib 3072` best (2048 −40/−48 %, 4096 −11/−11 %);
   `--pool-workers` auto best (9/13 mixed); lookahead stats need ≥100 windows (not captured).
4. **Hygiene.** shard2 hash recorded; **spec-off impossible on native packs by engine refusal**;
   >LLC bench shows no cliff; graph replay covered by the engine windows.
5. **Upstream:** main still `v0.1.45`; #3024 open (gfx12).
6. **Dormant branches:** keep #2514 carried on the refresh line; keep #2690 standalone, not merged,
   with re-test triggers (documented in GFX1100.md and the record).

## Remaining — what is NOT done, and why

The top-level path ran end to end and produced decisions, but "tasks.md is finished" would be
false. These items are open; each carries the reason it is open rather than a tick:

| task | open item | status |
| --- | --- | --- |
| T01 | shard2 content sha256 | **closed** — `316b46f3…` recorded in `fingerprint.json` |
| T03 | spec-off cell; live-env baselines | **closed** — spec-off is impossible on a native pack (engine refusal quoted); the live-env baselines are in the section above |
| T04 | >LLC working set; graph replay | **closed** — G64/141 MB 3.4 µs/expert vs G8 4.0 (no cliff); the verify windows are captured graphs and EXPDEC ran inside them |
| T05/T06/T07 | the task's bounded kernel sweeps | **still open, now lower priority** — the route prior cut CPU/PCIe 60-75 % and the live profile leaves no expert-kernel headroom in the measured wall (HC 5.2 ms, projections ~8.6 ms are the remaining shares); a re-opened kernel campaign needs a new profile first |
| T08 | residency budget, CPU threads, n-gram locality | **closed on the live config** — reserve 3072 and workers auto are optimal; lookahead stats need ≥100 windows (not captured) |
| T10 | BF16 re-assessment | **closed** — N/A (no patch in tree); HC fusion no-go; top-k neutral; prefill adopted |
| P3 | upstream packaging | **decided** — nothing code-level adopted; branches kept per the decision (workspace log) |
| — | longer-prompt quality of the adopted prior | **closed** — 20-prompt battery 20/20 = 20/20 |

This is an execution backlog, not a performance claim. The earlier BF16 experiment is unvalidated
and stays disabled in the baseline (no BF16 patch is present in this tree).

## Verified starting facts

The IQ3_S filename describes a mixed-quantization release. Counts below were computed from its published tensor allocation; they are layer coverage, not runtime percentages.

| Operation | Format | Layers |
| --- | --- | ---: |
| Routed gate/up | IQ2_S | 20 |
| Routed gate/up | IQ3_XXS | 17 |
| Routed gate/up | IQ3_S | 10 |
| Routed gate/up | IQ4_XS | 1 |
| Routed down | IQ4_NL | 39 |
| Routed down | Q2_0 | 9 |

Gate and up have matching formats within each layer. The expected expert dimensions are input 2560 and intermediate 640; verify against the loaded artifact. The model card lists 54.8 GB transformer weights and a separate 28.8 GB n-gram shard. Reserve VRAM for context state, scratch, desktop use and speculation before assigning experts.

- [x] Identify the requested variant as IQ3_S, replacing the earlier IQ3_XXS assumption.
- [x] Count routed-expert formats from the published IQ3_S allocation.
- [x] Identify mixed-format expert kernels as the primary investigation area.
- [ ] Verify all of these facts against the user's exact GGUF revision and native pack.

## Execution rules

- Follow the checkout's AGENTS.md and existing code conventions; inspect local modifications before editing.
- Extend existing kernels and dispatch. Avoid a parallel framework, broad refactor or model-specific hardcoding without shape/format checks.
- Keep experiments opt-in until correctness and end-to-end gates pass. Preserve fallback paths for other devices, formats and shapes.
- Work on one hypothesis at a time. A lower-priority task may move up only when profiling supports it; record why.
- Do not equate layer counts, summed overlapping kernel time or microbenchmark speedups with application speedup.
- Keep original GGUF quantization and activation arithmetic for initial experiments. Precision changes require a separate quality study.
- Record blocked hardware-dependent tasks explicitly; do not mark them complete after source review or compilation alone.

## P0 — Establish an actionable baseline

### T01 — Pin code, model and toolchain

- [ ] Record the user's fork, branch, commit, dirty diff, upstream base, ggml/llama.cpp dependency pin and local patches.
- [ ] Record ROCm/compiler versions, gfx1100 device identification, RAM configuration, PCIe link and relevant runtime flags.
- [ ] Pin Hugging Face revision, both GGUF shard hashes, tensor allocation, and native-pack identity/configuration.
- [ ] Confirm expert dimensions, format pairs, active-expert count, layer types and loaded backend dispatch from artifact/runtime evidence.
- [ ] Build the unchanged baseline using the checkout's documented commands; record the exact command and build log.

Done when: another run can reproduce the same binary, model and configuration. Baseline excludes the experimental BF16 patch.

### T02 — Recheck upstream work before implementing

Dependencies: T01.

- [ ] Read current versions and discussions of Strata PRs #3024, #2997, #2514, #646 and #783; identify what is already in the chosen base.
- [ ] Compare Strata's pinned llama.cpp/ggml kernels with current relevant matvec changes; inspect #26301 as a K-quant mechanism, not an IQ-format drop-in.
- [ ] Trace native expert packing, single-token dispatch, grouped verification dispatch, and CPU/GPU combination.
- [ ] Record each candidate's source, supported formats, measured hardware, correctness caveats, licensing and integration status.

Done when: proposed changes are demonstrably missing and applicable. Prior review found gfx12 runtime evidence for #3024, not proven gfx1100 gains; partial-residency correctness issues in earlier work require particular attention. Recheck these observations against current revisions.

### T03 — Measure decode and residency

Dependencies: T01. T02 can proceed independently.

- [ ] Capture resident experts per layer, CPU/GPU route counts, actual transferred bytes, peak VRAM/RAM and page faults.
- [ ] Attribute exposed wall time to expert gate/up, down, CPU work, copies, synchronization, attention/index selection and remaining projections.
- [ ] Determine whether host experts execute on CPU, transfer to GPU, or follow another path; do not assume all expert weights cross PCIe each token.
- [ ] Use fixed coding and general prompts with short context and a representative long context that fits. Record token IDs and exact context occupancy.
- [ ] Measure speculation off and the user's normal speculation configuration separately. Record verified positions, accepted tokens, rejection and rollback cost.
- [ ] Warm up and run at least five interleaved baseline/candidate pairs for later comparisons; use 256 committed output tokens where fixtures permit, recording early EOS.
- [ ] Record median, spread, TTFT, committed decode tokens/second, and cold versus warm behavior. Use profiler traces for attribution and uninstrumented runs for throughput.

Done when: there is a ranked cost table and a reproducible baseline command. If CPU or transfer cost dominates, prioritize T08 before GPU kernel tuning.

## P1 — Correctness harness and highest-value expert kernels

### T04 — Add a focused expert-kernel validation/benchmark harness

Dependencies: T01–T03.

- [ ] Cover every actual gate/up-down format pair found in IQ3_S, including the single IQ4_XS gate/up layer.
- [ ] Compare current and candidate kernels with an independent decoded-weight reference on representative expert tensors and adversarial inputs.
- [ ] Test single-token execution and actual supported verification widths, initially 2–4 where supported; include repeated and distinct expert groups, tails and fallback shapes.
- [ ] Check bounds/canaries, finite outputs, stream correctness and graph replay if used by production dispatch.
- [ ] Establish absolute/relative error tolerances before tuning from baseline behavior; require bitwise parity where arithmetic order is deliberately preserved.
- [ ] Benchmark both hot-cache reuse and a rotating working set larger than the GPU's last-level cache; record working-set size and allocations.

Done when: the harness detects wrong unpacking, scales, indexing and reductions without imposing artificial cache residency.

### T05 — Tune gate/up: IQ2_S, IQ3_XXS, then IQ3_S

Dependencies: T02–T04. Reorder formats if measured exposed time contradicts coverage.

- [ ] Start from existing grouped/native expert kernels in src/kernels/cuda/iq_kernels.cu and docs/EXPERTDEC.md.
- [ ] Test wave32-compatible lane mappings, packed-load vectorization and activation reuse for input dimension 2560.
- [ ] Compare codebook access from existing storage versus LDS staging, counting initialization and barriers.
- [ ] Inspect generated code for VGPR usage, spills, memory operations and occupancy. Interpret HIP launch bounds using AMD semantics.
- [ ] Limit each sweep to at most eight predeclared combinations of block size, output rows/block and staging policy; document resource limits first.
- [ ] Keep single-token and grouped-verification decisions separate; retain only profitable specializations.

Done when: a correctness-passing candidate improves its exposed cost and survives T09. IQ2_S plus IQ3_XXS cover 37 layers, but this is not a predicted speedup.

### T06 — Tune IQ4_NL down projection

Dependencies: T02–T04; may precede T05 if profiling ranks it higher.

- [ ] Target the 640-input down projection used in 39 layers.
- [ ] Compare existing subwave/reduction geometry, output rows/block and vectorized packed loads.
- [ ] Measure shared activation staging against its barriers and occupancy cost; avoid assuming gate/up choices transfer to down.
- [ ] Preserve scale application, routed weighting and combination order required by the baseline.
- [ ] Keep Q2_0 down as an unchanged comparison/fallback; optimize its nine layers only if their measured cost warrants it.

Done when: a correctness-passing down candidate improves overall decode, not merely an isolated microbenchmark.

### T07 — Improve expert reuse during verification

Dependencies: T04 and promising results from T05/T06.

- [ ] Measure how often verified positions share experts and the actual tokens-per-expert distribution.
- [ ] Reuse unpacked weights across positions only where saved work exceeds grouping, staging and register costs.
- [ ] Measure the entire draft/verify/commit cycle, including rejected tokens and fallback execution.
- [ ] Keep width-specific fallback dispatch when a grouped kernel loses at small or fragmented groups.

Done when: committed tokens/second improves at fixed model, sampler and speculation policy without state errors.

## P2 — Optimize the system cost revealed by profiling

### T08 — Placement, CPU work and transfer overlap

Dependencies: T03. Promote to P1 if it dominates exposed decode time.

- [ ] Sweep a small set of expert-residency budgets at fixed context and speculation, respecting measured memory headroom.
- [ ] Tune CPU thread count at fixed placement if CPU expert execution dominates; include RAM bandwidth and contention with GPU copies.
- [ ] Investigate overlap only where traces show avoidable serialized work; verify per-stream scratch, events and buffer lifetime.
- [ ] Measure n-gram lookup locality and paging separately. Count mapped pages/page cache correctly rather than summing them as separate physical allocations.

Done when: the chosen configuration wins on representative prompts without swapping, allocation failures or unacceptable latency spikes.

### T09 — End-to-end acceptance and regression gates

Dependencies: each proposed kernel/configuration candidate; this gate is mandatory regardless of priority label.

- [ ] Pass T04 for every affected format/shape and baseline fallback.
- [ ] Check logits and greedy output against baseline; investigate token divergence rather than dismissing it as floating-point noise.
- [ ] Exercise partial expert residency, context growth, session reset and speculative rollback of recurrent/convolution and attention state where affected.
- [ ] Run interleaved uninstrumented comparisons on held-out coding/general prompts at short and long context, with normal desktop workload represented or explicitly excluded.
- [ ] Require a median gain of at least 2% and larger than observed run-to-run variation as the initial project threshold; revise only before a new experiment and explain why.
- [ ] Reject unexplained correctness differences, OOM, sustained paging or a repeatable regression above 2% on a representative workload; narrow dispatch where justified.
- [ ] If inconclusive, allow one additional batch of five pairs; then record inconclusive and move to the next measured bottleneck.

Done when: improvement is reproducible in committed output throughput with acceptable latency and memory behavior. No universal maximum-performance claim.

### T10 — Secondary GPU work, only after profiling

- [ ] Evaluate long-context QSA/index selection and relevant #2514 work if exposed time is substantial.
- [ ] Evaluate hyperconnection reductions and BF16 projections only for measured hot shapes.
- [ ] Reassess the earlier BF16 experiment independently; do not mix it into expert-kernel A/B runs.
- [ ] Treat WMMA/rocWMMA prefill or larger-batch work as a separate backlog. Preserve prefill performance while optimizing decode.

## P3 — Package a maintainable change

Dependencies: T09.

- [ ] Add minimal architecture/format/shape dispatch with a default fallback and a documented disable mechanism.
- [ ] Review for duplicate unpackers, speculative abstractions, broad environment flags and unrelated changes.
- [ ] Test supported build configurations affected by the diff; disclose unavailable hardware coverage.
- [ ] Prepare a focused PR with pinned baseline, model revision, hardware, commands, correctness evidence, median/spread, memory and known limits.
- [ ] Keep only successful experiments in production; retain rejected results as short notes to prevent repeated dead ends.

## Deferred work

- Sparse P48NVFP4/MoESQ: separate loader, masks, scaling and activation-quantization project; not required for IQ3_S.
- Persistent whole-model megakernel: defer until launch/synchronization cost proves sufficient benefit and gfx1100 correctness is established.
- Requantization, pruning, changed attention budgets or reduced precision: separate model-quality experiments.
- Multi-GPU, new model formats and general framework/plugin refactoring: outside this single-card decode task.

## Experiment record template

| Field | Required value |
| --- | --- |
| ID / task / status | Planned, running, passed, rejected, inconclusive or blocked |
| Hypothesis | Measured bottleneck and proposed causal improvement |
| Fingerprint | Commits, compiler, model hashes, hardware, flags |
| Change | One variable or explicitly bounded parameter set |
| Correctness | Oracle, tolerances, state checks and results |
| Performance | Median/spread, committed tokens/s, kernel time, context, acceptance |
| Resources | Peak VRAM/RAM, transfer bytes, paging, spills |
| Decision | Keep/reject, dispatch scope, evidence and next task |

## Start here

1. Complete T01 on the actual Fedora host and preserve a clean baseline.
2. Complete T02 and T03; identify whether GPU experts or hybrid execution limits decode.
3. Complete T04; then choose T05, T06 or T08 using the profile.
4. Apply T09 after every candidate before combining improvements.

## References and inspection baseline

- [ISTA model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- [IQ3_S tensor allocation](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt)
- [Strata](https://github.com/Niko1221/Strata): locally inspected base 477ac13cbe8e582c20484871ce459d20d82df09d; recheck before implementing.
- [Strata #3024](https://github.com/Niko1221/Strata/pull/3024), [#2997](https://github.com/Niko1221/Strata/pull/2997), [#2514](https://github.com/Niko1221/Strata/pull/2514), [#646](https://github.com/Niko1221/Strata/pull/646), [#783](https://github.com/Niko1221/Strata/pull/783).
- [llama.cpp #26301](https://github.com/ggml-org/llama.cpp/pull/26301); verify applicability against Strata's actual dependency pin.
- [AMD RDNA3 WMMA](https://gpuopen.com/learn/wmma_on_rdna3/).
