# Tasks v2: IQ3_S decode optimization, local implementation with frontier orchestration

Updated: 2026-10-18, revision 2 (ROCm nightly audit and frontier implementation takeover). Target: Strata HIP/ROCm on RX 7900 XTX (gfx1100), Ryzen 9 7900X, 96 GB RAM, Fedora. Confirm actual host configuration before execution.

This document supersedes tasks.md. It is a self-contained execution plan, not a benchmark result or an installed agent harness. The local model performs bounded implementation and checks; a frontier model orchestrates consequential decisions and independently reviews evidence. A harness or human operator must actually relay the work orders and results—this Markdown file does not create model-to-model communication.

## What changed in v2

- Promote the decode execution timeline to a prerequisite for choosing optimizations.
- Add expert fusion, HIP graph coverage, lossless weight repacking, hybrid scheduling and bounded persistence experiments.
- Retain v1's quantized-kernel, placement and validation tasks, with stable T01–T10 identifiers.
- Add small local-model work orders, frontier review gates, durable handoffs and bounded escalation.
- Separate coding-model activity from benchmarking on the same host.
- Add mandatory P0 upstream ROCm nightly/primitives audit before selecting custom kernel changes.
- Assign frontier models direct kernel implementation ownership when local execution encounters a consequential gap.

The motivating principle is to reduce avoidable launches, memory traffic and synchronization across decode. The SemiAnalysis image is inspiration, not a transferable implementation specification or a 1.68x target for this machine.

## Deployment note (2026-10-18) — execution model and current state

This copy is the **live backlog** as of 2026-10-18. It supersedes `tasks.md` (archived as
`tasks-v1-superseded.md`, with its execution history intact). The retained v1 section at the bottom
carries v1's original checkboxes; the table below is the truth where they disagree.

**Execution model change: there is no local implementation model.** In this deployment the DSH
agent holds both v2 roles — orchestrator and implementer — because it has direct filesystem, build,
runner and evidence access; a local LLM would be a manual, lossy relay with no file access, and the
only local models here are the specimen under test (Strata/Qwen on 8081) and the user's 8080
llama.cpp, which the repo rules put off-limits. Using either as a coder would couple the tool to
the object of study and contaminate measurement. What survives from v2 is the part that matters:
**independent review**, now a fresh-context review subagent used at v2's mandatory checkpoints
(bottleneck selection, oracle/tolerances, synchronization/graph/repack design, final acceptance).
The revised roles:

| v2 role | in this deployment |
| --- | --- |
| Frontier orchestrator | DSH agent (selects the bottleneck, issues the work orders, writes code when needed) |
| Local implementation model | **removed** — same DSH agent implements directly, under the same gates |
| Deterministic runner | the existing stop/restore runner scripts with pre-declared resource gates and records |
| Independent review | fresh-context review subagent at the mandatory checkpoints; never the implementer's own sync/repack design |
| Human owner | product/quality trade-offs, publication, any new consequential decision |

**Status reconciliation** (v2 item → actual state):

| item | state | evidence |
| --- | --- | --- |
| T01 pin | done (the retained checkbox is stale) | `fingerprint.json` (shard1+shard2 hashes, pack identity, toolchain) |
| T02 upstream | done | `upstream-pr-sweep-20261018.md`, `upstream.md` |
| T03 baseline | done, superseded by the live-env re-baseline | `record-iq3s-second-queue-20261018.json` |
| T04 harness | done (real rows, 7 pairs, >LLC, graph windows) | campaign records |
| T05/T06/T07 | only EXPDEC tried and rejected; V02/V04 are the successors | window A/B records |
| T08 | measured; route prior **adopted** (CPU/PCIe −60-75 %); #2690 and its BELOW sweep rejected; V06 successor | route-prior / below-sweep records |
| T09 | applied to every candidate | records |
| U01 | **open** — SDK 10.2.0 pinned by path/version, no artifact URL/checksum audit and no component table yet | this backlog |
| V01 | **partial** — stage profile exists (waits 24.78 ms; stage 28.89: HC 5.2, projections 8.6, VRAM hits 4.07, router 2.32, shared 2.06, head 1.06, draft 3.86); the boundary table is the next deliverable | second-queue record |
| V02–V08 | not started; V07 stays DEFERRED (HC no-go: no `s_memrealtime`, R9700 prior no gain) | HC feasibility record |
| P3 | not applicable (no code-level adoption) | campaign log |

**Live production state at adoption:** `0.1.45-8eb3e85d` with `STRATA_ROUTE_PRIOR=0.5`,
`STRATA_GDN_CHUNKED=1`, `STRATA_PF_GEMM=1`, `STRATA_PF_PAD=1` (all output-changing, each with a
config backup). 8080 is never touched.

**First phase after adoption:** complete V01 (boundary table) and U01 (component/applicability
audit) without a new GPU window where the existing records suffice; review both; then select one
experiment (V02, V05 or V03) by V01's measured exposed fraction and run it behind an opt-in.

**Phase 0/1 status (2026-10-18, same day):** V01 and U01 are **done** (V01 revised after review and
calibration; see below):
`artifacts/rocm/iq3s-decode-timeline-20261018.md` (HC/experts/head at HBM bandwidth; the dense
projections are the only material gap, ~26% under, ~2.0 ms/window) and
`artifacts/rocm/iq3s-u01-audit-20261018.md` (no second nightly; CK/rocSHMEM not applicable;
counters blocked by `perf_event_paranoid=2`). V01's recommendation: **V05 scoped to the dense
projection layout, preceded by one discriminating micro-benchmark** (current vs padded vs
interleaved, T=1/T=4; falsifier: <10% gain → occupancy, drop V05 and take the router/plan fusion).
The fresh-context review returned and was folded in; the calibration (q8_gemv_bench + spec-2 vs
spec-8 profiles) then **dropped V05 by its own falsifier** (weights amortized; the inefficiency is
per-call shape/occupancy) and selected **the router-path fusion** (router+plan+bell 2.37 ms/window,
T-flat, expected 0.7-1.0 ms ≈ 2.0-2.9%) as the first experiment, gated by the existing router
parity. Runner-up: batch the small GEMVs. See `iq3s-decode-timeline-20261018.md` (addendum) and
`record-iq3s-v01-review-calibration-20261018.json`.

**First experiment result (2026-10-18): REJECTED.** With the prior on, the router's own top-10 is
dead work; `STRATA_ROUTER_SKIP_TOPK=1` (commit 662d4736, reverted) dropped it, exact by inspection
(the prior reads only logits and `res`). Paired cells: 1K median **+1.31%, 3/5** (a first window's
+2.91%/4/5 did not reproduce), 32K/128K inside the trajectory band — below the ≥+2% gate, not
adopted. The router path is already one projection + one top-10 per window; the remaining redundancy
is one cheap launch per layer. Next candidate: batch the small GEMV calls (z/rec/ab + shared),
bound ~1.7-2.9%. Record: `record-iq3s-router-skip-gate-20261018.json`.

## 1. Operating contract

### Roles

| Role | Responsibilities | Must not do |
| --- | --- | --- |
| Frontier orchestrator | Select next measured bottleneck; issue bounded work orders; review design, synchronization and final evidence; choose keep/reject/inconclusive; write kernel patches during takeover | Invent measurements, waive failed correctness checks, prescribe a rewrite without inspecting existing paths |
| Local implementation model | Inspect code; gather facts; implement one scoped change; run authorized checks; preserve raw evidence; propose next action | Expand scope silently, change acceptance thresholds after seeing results, treat frontier advice as verified fact |
| Deterministic runner | Execute recorded builds/tests/benchmarks; capture exit codes, logs and metrics | Report inferred or model-generated values as measurements |
| Human owner | Supply unavailable access, resolve genuine product/quality tradeoffs, decide publication/deployment outside existing authorization | Need not approve each reversible source edit or routine benchmark |

The frontier model delegates a whole bounded task, not individual keystrokes. The local model proceeds autonomously within the work order. Frontier review is an engineering checkpoint, not a request for repeated user permission. If the frontier model is unavailable, continue assigned work and safe evidence collection; hold unreviewed synchronization/persistence changes at the experimental stage.

### Protect benchmark validity

- Do not run the local coding model on the target GPU during inference benchmarks. Its VRAM, compute and RAM use can invalidate placement and timing.
- The runner must persist results without needing a live local LLM. If resources conflict, checkpoint the coding session, release its resources, benchmark, then resume the model.
- Record other workload, clocks/power/thermal state, desktop activity and background services. Avoid changing unrelated services automatically.
- Prefer a second machine or remote frontier orchestration for control. Do not assume one is available.
- Never benchmark competing candidates concurrently on the same target GPU. Independent read-only reviews can run concurrently if the harness supports them; source edits need explicit ownership.

### Task states and evidence

Use TODO, READY, RUNNING, REVIEW, ACCEPTED, REJECTED, INCONCLUSIVE, BLOCKED. Only one performance experiment is RUNNING on the target GPU at a time. Completing a source edit is not ACCEPTED: the task's exit gate must pass. BLOCKED records the exact missing capability and the cheapest next step.

Keep a lightweight project log using existing repository conventions. If none exist, use:

```text
bench/iq3s-gfx1100/<run-id>/
  manifest.json           # versions, hashes, hardware, flags, fixture IDs
  commands.txt            # exact commands and exit status
  metrics.json            # raw per-run metrics, units, aggregation
  summary.md              # hypothesis, decision, evidence links
  logs/                   # build/test/profiler output
docs/experiments/iq3s-gfx1100-state.md
```

Do not commit model weights, huge traces, credentials or private prompts. Store large evidence outside git with a path/hash and availability note. Keep rejected experiments documented so a resumed agent does not repeat them.

## 2. Priority and dependency map

This order supersedes the original P1/P2 labels in the retained backlog below. All code candidates must pass T09 before production acceptance.

| Order | Work | Owner / frontier checkpoint |
| --- | --- | --- |
| P0 | T01 fingerprint and clean baseline; T02 upstream audit | Local gathers; frontier resolves applicability questions |
| P0 | U01 pinned ROCm nightly/toolchain and primitives audit | Local builds/compares; frontier chooses upstream reuse or custom implementation |
| P0 | T03 baseline measurements + V01 execution timeline | Local measures; frontier chooses bottleneck and hypothesis |
| P0 | T04 existing-test audit and focused correctness harness | Local implements missing coverage; frontier reviews oracle/tolerances |
| P1 | V02 adjacent expert fusion or V03 graph gaps, if overhead is exposed | Frontier approves scope; local implements one candidate |
| P1 | T05 gate/up and T06 IQ4_NL down, if unpacking/compute/memory dominates | Local bounded tuning; frontier reviews evidence |
| P1 conditional | T08 + V06 hybrid scheduling, if CPU/transfers dominate | Frontier reviews dependency and ownership design |
| P2 | T07/V04 verification reuse; V05 lossless repacking | Profile-gated; frontier reviews representation/state contract |
| P3 conditional | V07 bounded persistent subgraph | Requires explicit frontier design review and strong measured justification |
| Every candidate | T09 acceptance + V08 combination check | Frontier makes evidence-backed keep/reject decision |
| Final | Packaging/PR task in retained backlog | Frontier reviews final diff and reproducibility |

Do not implement all directions just because they are listed. Select the lowest-complexity experiment with material addressable cost. For exclusive exposed fraction f and acceleration s, ideal total speedup is 1/((1-f)+f/s); elimination ceiling is 1/(1-f). Use this only when f is defensible from the dependency timeline. Mark unknown rather than substitute summed overlapping kernel durations.

Before implementing T05/T06 or V02–V07, resolve U01’s applicability decision (or record a specific runtime blocker and retain the working stack). Baseline collection and correctness-harness preparation can proceed in parallel with the audit.

## 3. New execution-path tasks

### U01 — P0 upstream ROCm nightly and gfx1100 primitives audit

Dependencies: T01. Run alongside T02 and baseline T03/V01. Status: TODO; source leads found, no nightly installed or benchmarked.

- [ ] Preserve the current working compiler/runtime/libraries as baseline A. Select exactly one dated TheRock nightly as candidate B, installed separately without replacing the system stack.
- [ ] Pin artifact URL, checksum, build/run ID and exact compiler, HIP runtime, math-library and profiler revisions. Resolve revisions from its manifest; do not label the environment merely “ROCm 10 nightly”.
- [ ] Confirm Linux gfx1100 device coverage and host-driver compatibility. Record the host driver/kernel independently; an isolated user-space SDK does not isolate the kernel driver.
- [ ] Verify compiler and loaded shared-library paths in each run. Prevent accidental mixing of baseline and nightly components; retain both build directories and rollback commands.
- [ ] Build unchanged Strata under A and B with matched intended options, artifact and runtime configuration. Run existing correctness tests, representative partial-residency decode, graph replay and supported speculation before custom kernel changes.
- [ ] Compare generated code for measured hot kernels: VGPR/LDS use, spills, vectorized loads, reductions and scheduling. Distinguish compiler effects from runtime/library effects; mixed-component attribution experiments require compatibility evidence.
- [ ] Inspect pinned upstream changes in TheRock, ROCm compiler sources, rocm-systems/HIP, rocm-libraries/CK/CK Tile, rocWMMA and relevant math/profiling components. Follow actual repository pointers for the selected build.
- [ ] For each promising change record commit/PR, original rationale, architecture guard, supported shapes/formats, tests, licensing and applicability to Strata’s executed path.
- [ ] Classify each gap as ALREADY_FIXED_UPSTREAM, REUSABLE_PRIMITIVE, ADAPTABLE_IMPLEMENTATION, CUSTOM_KERNEL_REQUIRED or UNSUPPORTED_HARDWARE. Produce a concrete recommendation and falsifying test.
- [ ] Select the strongest correctness-passing stack for subsequent experiments, documenting compatibility and performance. Nightly failure must not block work on the working stack; record the blocker and continue.
- [ ] Repeat the relevant V01 timeline if the selected stack changes. Compare custom patches against the selected unchanged-source baseline, with the original baseline retained.

Bound: one baseline plus one pinned nightly initially. Do not chase every new daily build. Expand to a targeted bisect only for a reproducible regression or a specific relevant fix.

Exit: frontier-reviewed component/applicability table, two reproducible build/run manifests and a keep/reject/inconclusive decision. Toolchain performance claims use T09’s measurement discipline. If hardware access is absent, finish source audit and mark runtime comparison BLOCKED.

Upstream leads observed on 2026-10-18; recheck exact commits rather than treating development notes as a release guarantee:

| Lead | Investigation | Boundary |
| --- | --- | --- |
| CK Tile packed-layout, wave-size and async-load fixes | Review packing granularity, host/device layout agreement, validity checks and launch-bound assumptions | Some fixes explicitly concern newer GPUs; no automatic gfx1100 speedup |
| HIP host-NUMA VMM | Check whether allocation control addresses a measured host-memory problem | No implied increase in PCIe bandwidth; distinguish Linux from Windows-only features |
| rocSHMEM wave32 tests and gfx1100 memory policy | Study memory visibility and supported synchronization patterns | Not proof of safe arbitrary cross-block barriers; no new dependency without a need |
| rocWMMA/compiler lowering | Verify available gfx1100 primitives and inspect generated code for our shapes | Newer GPU instructions cannot be supplied by a compiler upgrade |
| Profiling/debugging tooling | Confirm traces/counters needed for V01 on the exact GPU and stack | Do not assume every Instinct profiling or device-sanitizer feature exists on Radeon |
| Sparse libraries | Check exact architecture support before proposing P48NVFP4 reuse | Development table lists hipSPARSELt for gfx950/gfx942; IQ3_S remains the target |

The live development notes retrieved during planning identified 10.1.0. Pin the actual artifact rather than assuming that name matches every nightly. This audit does not establish CuTe DSL support for AMD.

### V01 — Audit the decode execution timeline

Dependencies: T01 and baseline portion of T03. Status: TODO. Local owner; frontier ranks results.

- [ ] Trace one ordinary token and one supported verification window under representative partial residency.
- [ ] Map normalization/router, route preparation, resident GPU experts, CPU experts, transfers, gate/up, SwiGLU/quantization, down, combination, attention/recurrent work and host waits.
- [ ] For each boundary record producer, consumer, tensor shape/type/bytes, storage, stream/event, graph membership and synchronization reason.
- [ ] Identify existing fused implementations and the actual runtime path for every IQ3_S format pair. Do not infer dispatch from function names alone.
- [ ] Distinguish host submission gaps, device scheduling, memory stalls and waiting for CPU results; label uncertain attribution.
- [ ] Quantify addressable intermediate traffic and exposed time for each candidate. Capture instrumentation overhead by comparing uninstrumented baseline.

Deliverable: ranked boundary/cost table with evidence links, current dispatch map and one recommended next experiment. Exit: frontier chooses a hypothesis or requests one specific missing measurement; no optimization code before this choice.

### V02 — Extend adjacent expert fusion where missing and profitable

Dependencies: T02, T04, V01. Status: TODO.

Hypothesis: eliminating materialized gate/up/SwiGLU intermediates or an extra quantization launch reduces exposed decode cost.

- [ ] Inspect existing SwiGLU-plus-quantization, fused gate/up, and grouped variants in iq_kernels.cu before proposing a new kernel.
- [ ] Choose one missing or poorly performing IQ3_S format pair and one execution mode first; prioritize measured cost over layer count.
- [ ] Write a design describing ownership of output rows, completion of gate/up reductions, the full activation quantization block, scales and synchronization.
- [ ] Preserve the baseline SwiGLU expression and Q8_1 quantization semantics, including sums/scales used by downstream dot products.
- [ ] Fuse only where all required producer results are safely available. Do not assume partial per-block gate/up output is enough to quantize an entire activation block.
- [ ] Compare bytes saved, launches, barriers, register pressure, spills and occupancy against the current fused baseline.
- [ ] Keep the down boundary separate initially. Any full gate/up-to-down fusion needs a new design review because down consumes the complete intermediate vector.

Bound: one format pair/mode, up to four geometry variants first. Exit: T04 and T09 pass; dispatch remains narrow. Stop if staging/recomputation/spills outweigh avoided traffic or correctness requires unsafe cross-block synchronization.

### V03 — Improve HIP graph coverage and replay stability

Dependencies: T02, V01, relevant T04 checks. Status: TODO.

- [ ] Inventory graph captures/replays, graph updates, operations outside graphs and host synchronizations at the actual commit.
- [ ] Identify the smallest measured gap: repeated allocation, routing metadata setup, avoidable host wait, graph rebuild or uncaptured tiny operations.
- [ ] Fix one cause while retaining dynamic routing, stable pointer lifetimes and valid event dependencies.
- [ ] Validate replay across changed expert routes, verification widths, context growth, fallback and session reset; inspect captured device/function selection.
- [ ] Measure graph setup separately from steady-state replay. Account for graph-cache memory and invalidation if adding variants.

Exit: improved uninstrumented decode with no stale pointers/routes/state. Stop if existing capture already hides the target overhead. Graphs reduce submission overhead; they do not themselves eliminate intermediate global-memory traffic.

### V04 — Reuse packed weights across verified positions

Dependencies: T04 and V01. Links: T07. Status: TODO.

- [ ] Measure tokens-per-expert distribution, repeated versus distinct experts, accepted length and fallback behavior.
- [ ] Reuse unpacked blocks across positions assigned to the same expert, using the existing grouping mechanism where possible.
- [ ] Include grouping cost, extra scratch, LDS/VGPR use and discarded draft work.
- [ ] Preserve position ordering and rollback of recurrent, convolution and attention state; do not move state updates across commit boundaries without proof.

Exit: T09 passes for the complete draft/verify/commit cycle. Keep single-token and fragmented-group fallbacks. Do not expand speculation width to manufacture a kernel win in the same experiment.

### V05 — Lossless gfx1100-native weight repacking

Dependencies: T01, T04, V01, established kernel bottleneck. Status: TODO.

- [ ] Identify costly load alignment, interleaving or address calculations that a storage permutation could remove.
- [ ] Design one format/layout variant using the existing native-pack machinery. Preserve quantized codes, signs, scales and represented values exactly.
- [ ] Add an explicit version/layout identifier and rejection or fallback for incompatible packs; fingerprint original artifact, converter version and layout.
- [ ] Build an inverse transform and require exact round-trip bytes for the represented source data; independently compare decoded tensors and kernel outputs.
- [ ] Measure pack time, disk size, peak conversion RAM, padding, runtime VRAM and resulting expert residency.
- [ ] Avoid retaining duplicate layouts in scarce VRAM by default. Check the CPU path can still consume the needed format without per-token conversion.

Bound: one format first, selected from measured cost. Exit: T09 passes at a fair total-memory budget. Reject a microbenchmark win that reduces residency enough to hurt decode. Requantization is outside this task.

### V06 — Schedule hybrid CPU/GPU work around true dependencies

Dependencies: T03, V01, T08 placement evidence. Status: TODO.

- [ ] Draw the dependency/ownership table for routing, CPU experts, GPU experts, shared expert, copies and final combination.
- [ ] Identify a demonstrated idle interval that independent work can fill. Do not assume more streams mean more throughput.
- [ ] Launch required CPU work promptly when inputs/routing are available; overlap only dependency-independent GPU work.
- [ ] Specify stream-local scratch, BLAS handle ownership, event lifetimes, host-buffer reuse and CPU scheduling.
- [ ] Preserve the required combination order initially. Test mixed CPU/GPU results with varied routes and partial residency.
- [ ] Measure contention for RAM, PCIe and GPU bandwidth, including tail latency and memory pressure.

Frontier design review required before edits that change synchronization or buffer ownership. Exit: T09 passes and trace demonstrates the intended overlap. No speculative expert prediction/prefetch in this task; that needs a separate waste/benefit study.

### V07 — Prototype a bounded persistent GPU subgraph

Dependencies: V01, lower-complexity alternatives assessed, T04, explicit frontier design review. Status: DEFERRED until evidence promotes it to READY.

- [ ] Select a small wholly GPU-resident sequence with substantial measured boundary cost; do not start with the entire model.
- [ ] Compare against the best graph/fused implementation, not an artificially unfused baseline.
- [ ] Prove synchronization scope and forward progress. A global spin barrier can deadlock when waiting blocks occupy resources needed by unscheduled blocks.
- [ ] Check actual gfx1100/runtime support for any cooperative launch, occupancy assumption, fence, atomic or timeout primitive before designing around it.
- [ ] Do not enable Strata's existing HC persistent flag blindly: the previously inspected implementation excluded gfx11/gfx12 due to its timeout dependency. Recheck current code and review #2226.
- [ ] Test bounded work and failure handling without silently consuming partial results. A host timeout is not proof that a stalled GPU kernel has stopped or buffers are safe to reuse.
- [ ] Measure monopolized GPU resources, occupancy, fairness and interference with hybrid overlap.

Exit: independent frontier review of progress/failure behavior, T04 and T09 pass, recoverable failure handling demonstrated on target hardware. Stop if benefit is small or synchronization support is uncertain. Leave this task deferred rather than invent unsafe primitives.

### V08 — Validate composition of accepted optimizations

Dependencies: at least one accepted candidate; full combined gate when multiple exist. Status: TODO.

- [ ] Preserve original baseline and each accepted isolated configuration.
- [ ] Add changes one at a time and measure incremental contribution; do not multiply isolated speedups.
- [ ] Repeat correctness/state checks for affected interactions, including graphs, repacked weights and concurrent paths.
- [ ] Remove redundant experiments and configuration switches; retain a known-good fallback.
- [ ] Confirm prefill, TTFT, peak memory and long-context behavior remain acceptable.

Exit: a minimal integrated patch with reproducible committed decode improvement and documented limits.

## 4. Work orders and frontier review

### Frontier implementation takeover

The frontier model is an implementation owner when required, not only a reviewer. Trigger takeover for a demonstrated missing kernel/format path, a consequential unresolved fusion/synchronization design, or two failed local repair cycles on the same cause. A hard design gap may escalate immediately; do not force two speculative attempts.

1. Local model supplies exact source revision, relevant original code/callers, minimal reproducer, input shapes/formats, compiler diagnostics, generated-code resource reports and available trace/test evidence.
2. Frontier decides whether upstream adoption, adaptation or custom code addresses the gap. It writes a concrete patch and the missing independent reference checks when custom implementation is justified.
3. Local model checks patch applicability, integrates without unrelated edits, compiles on the actual stack, and runs the prescribed correctness tests before timing.
4. Return raw errors/results to the frontier for bounded revision. After two unsuccessful takeover revisions, reassess the hypothesis or mark BLOCKED/INCONCLUSIVE instead of widening scope.
5. Accept only through T09 and V08. Frontier authorship does not replace independent validation; a fresh review pass checks arithmetic, ownership and synchronization against the oracle.

Backend policy: implement gfx1100 candidates in HIP C++ with verified AMD intrinsics where useful. NVIDIA CuTe DSL is a CUDA/NVIDIA backend, not a supported direct gfx1100 target. Any alternative DSL must first demonstrate compilation, execution, integration and relevant feature coverage on the pinned target. Do not translate hardware-specific NVIDIA operations by name or imply missing native instructions can be added in software without cost.

Aggressive fusion is permitted when evidence supports it: adjacent expert operators, an expert-local pipeline including down, then a bounded persistent subgraph. Expand one validated region at a time. A 12,000-line implementation is not an acceptance criterion. Generated variants must retain understandable source/templates, reproducible generation, tested dispatch and resource bounds.

The 1,500-token response default below applies to decision review, not patch delivery. Allocate sufficient context/output or split complete patches into coherent stages; never truncate implementation or correctness evidence to satisfy a review-token budget.


### Work order template: frontier to local model

```text
Task ID / experiment ID:
Goal: one measurable change
Base commit and model/pack fingerprint:
Observed bottleneck: evidence excerpts + accessible paths
Scope: functions/files allowed; expand only with a reasoned handoff
Invariant: arithmetic, state, ownership and fallback requirements
Hypothesis and falsifying observation:
Implementation plan: 3–6 small steps
Tests: existing checks + the specific missing risk coverage
Benchmark: fixtures, commands, resource conditions, repeat/stop budget
Acceptance: predeclared correctness/performance/resource gates
Budget: default one candidate, <=8 predeclared variants, <=2 repair cycles
Return: diff, actual commands/exit codes, raw result paths, concise findings
Next decision: what the frontier must choose after this task
```

Begin each local session by reading repository instructions, current task state and the relevant work order. Inspect source before editing. Do not paste the entire repository into the model context. Use small excerpts sufficient to understand callers, consumers and invariants. Avoid tests that merely restate implementation; reuse existing independent references.

### Local completion packet

```text
Task/experiment/status:
Base and candidate commit or patch hash:
Changed files and why:
Correctness checks: command, exit code, result, limits
Performance: per-run values, median/spread, committed-token definition
Resources: VRAM/RAM, residency, transfers, spills if relevant
Evidence paths: accessible to reviewer; include decisive original excerpts
Unexpected behavior and remaining uncertainty:
Recommendation: keep / reject / narrow dispatch / one discriminating test
Next work order proposed:
```

A frontier model without filesystem access must receive the relevant diff, manifest and decisive log excerpts. A path alone is not evidence it can inspect. Keep summaries alongside original outputs so a mistaken local summary can be challenged.

### Escalation policy

Use the escalate-minimal-questions workflow if installed; the harness must explicitly expose/invoke it. Otherwise follow this embedded policy:

1. Run available deterministic checks before asking the frontier to guess.
2. Escalate when two plausible answers change the action and existing evidence cannot resolve the consequential choice.
3. Send one decisive question, at most three, with constraints, original evidence, options and checks already performed.
4. Ask for a recommendation, uncertainty and one falsifying test. Treat the response as a hypothesis to verify.
5. Allow one targeted follow-up for missing evidence; then mark BLOCKED/INCONCLUSIVE or choose the safer supported fallback.

Default packet budget: about 4,000 input tokens and 1,500 response tokens, adjustable when decisive evidence requires it. Never truncate the key diff or failure trace just to meet a budget. Record escalation reason, tokens/cost if available, outcome and verified lesson. Do not request repeated generic reviews that cannot change the next action.

Mandatory frontier checkpoints: bottleneck selection after V01; correctness oracle before kernel tuning; synchronization/graph-lifetime changes; repack format design; persistent-kernel design; final acceptance and integrated diff. Routine compilation fixes remain local within the assigned budget.

### Resume and stop rules

- After each task or interruption, persist base/candidate hashes, changed files, exact last successful command, outstanding processes, evidence paths and next action.
- On resume, inspect git status and active processes before rerunning commands. Do not overwrite another task's edits or rerun an expensive completed benchmark blindly.
- After two failed repair cycles on the same cause, send a bounded escalation packet instead of speculative edits.
- If hardware/toolchain/model is absent, complete source inspection and harness preparation; mark runtime claims BLOCKED.
- Correctness failure stops performance promotion. Do not change tolerances retrospectively or hide failing fixtures.
- Model-changing behavior, destructive host changes, publication and deployment follow the owner's existing authorization; ask only when a new consequential decision is actually required.

## 5. First orchestrator instruction

```text
You are orchestrating IQ3_S Strata optimization on gfx1100. Read tasks-v2.md
and the repository instructions. Do not start by writing a megakernel.
Issue a bounded local work order for T01, then T02/U01/T03/V01. Preserve the
unchanged baseline and identify actual runtime dispatch. Have the runner
benchmark without the local coding model competing for resources. Review
the dependency timeline and choose one measured bottleneck. Establish T04
correctness coverage and resolve U01's upstream applicability decision, then
assign one narrow experiment. Take over writing kernel code when a demonstrated
gap or bounded local failure requires it; validate through the local runner. Require raw
evidence and T09 before acceptance; persist decisions for the next session.
```

## 6. Retained implementation backlog and acceptance contract

The following retains the v1 tasks so this document can be used alone. Use the v2 priority map above rather than treating its original priority headings as an instruction to execute everything. T09 is mandatory for every candidate, including all V tasks that alter runtime behavior.

## Objective and scope

Improve committed decode tokens/second for ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, **IQ3_S**, on RX 7900 XTX using Strata's HIP backend under ROCm. Preserve the model's represented weights and validated inference behavior. Target the user's Ryzen 9 7900X / 96 GB RAM / Fedora system; confirm its current configuration before measuring.

This is an execution backlog, not a performance claim. No gfx1100 runtime measurements or HIP compilation were completed in this workspace. The earlier BF16 experiment is unvalidated and must remain disabled in the baseline.

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

## Original baseline sequence (apply v2 priority map first)

1. Complete T01 on the actual Fedora host and preserve a clean baseline.
2. Complete T02, U01 and T03/V01; identify whether GPU experts or hybrid execution limits decode.
3. Complete T04; then choose T05, T06 or T08 using the profile.
4. Apply T09 after every candidate before combining improvements.

## References and inspection baseline

- [ISTA model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- [IQ3_S tensor allocation](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt)
- [Strata](https://github.com/Niko1221/Strata): locally inspected base 477ac13cbe8e582c20484871ce459d20d82df09d; recheck before implementing.
- [Strata #3024](https://github.com/Niko1221/Strata/pull/3024), [#2997](https://github.com/Niko1221/Strata/pull/2997), [#2514](https://github.com/Niko1221/Strata/pull/2514), [#646](https://github.com/Niko1221/Strata/pull/646), [#783](https://github.com/Niko1221/Strata/pull/783).
- [llama.cpp #26301](https://github.com/ggml-org/llama.cpp/pull/26301); verify applicability against Strata's actual dependency pin.
- [AMD RDNA3 WMMA](https://gpuopen.com/learn/wmma_on_rdna3/).


## Additional execution-path references

- HIP graphs: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
- Persistent HC discussion: https://github.com/Niko1221/Strata/pull/2226
- Inspiration image: https://x.com/SemiAnalysis_/status/2111627094878790023/photo/1

Recheck source revisions before implementation. Prior source observations are starting evidence, not proof of the current user checkout or of performance on gfx1100.

## Upstream toolchain references

- TheRock release and nightly artifact instructions: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
- AMD development release notes (moving target): https://rocm.docs.amd.com/en/develop/about/release-notes.html
- TheRock GPU support: https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md
- ROCm system components: https://github.com/ROCm/rocm-systems
- ROCm library components: https://github.com/ROCm/rocm-libraries
- CuTe DSL hardware/backend scope: https://docs.nvidia.com/cutlass/latest/overview.html

For every adopted hint, retain a commit-pinned permalink and tested applicability rather than relying solely on these moving URLs.
