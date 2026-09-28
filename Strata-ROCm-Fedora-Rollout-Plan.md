# Strata on Fedora and RX 7900 XTX: automated rollout plan

Prepared: 27 September 2026 · Version 1.0

## 1. Outcome and how to use this document

Build a maintainable AMD HIP backend for Strata, using a local coding agent to edit, compile, test and report. The user should start the task and handle occasional consequential decisions, rather than edit GPU code.

**This is an implementation brief, not a completed port or executable installer.** Commands named `strata-rocm` below are interfaces the agent must implement. No AMD build, numerical validation or speed measurement has been performed for this plan. A successful port is not guaranteed.

### Your starting steps

1. Save this document in a new folder, for example `~/projects/strata-rocm-rollout/`.
2. Open that folder in whichever coding agent you select. The agent must be able to run a terminal on your Fedora PC and access the actual GPU. A cloud-only coding session cannot complete the hardware gates.
3. Paste the starting prompt in section 3, with this document attached or named.
4. Let the agent complete the preparation and small GPU tests. Do not download the full model until the feasibility gate passes.
5. If the agent reports a blocker, use the plain-English action it provides. To switch agents, give the next agent this document and `STATUS.md`.

You do not need to select a particular provider now. Use one lead agent; keep scripts, tests and progress files provider-independent. Do not install a second orchestration framework merely to begin this project.

## 2. Target, evidence and boundaries

| Item | Target or assumption |
| --- | --- |
| OS | Fedora; detect exact release and kernel locally |
| GPU | AMD Radeon RX 7900 XTX, expected HIP target `gfx1100`; verify locally |
| CPU / RAM | Previously reported Ryzen 9 7900X / 96 GB; detect rather than assume |
| Existing service | Preserve any current llama-server, model files, launch scripts and systemd configuration |
| New server | Loopback only; prefer port 8081 if free, otherwise report an available port |
| First model run | Text only, one request, short context, speculative decoding disabled |
| Baseline source | https://github.com/Niko1221/Strata |
| Inspected revision | `6da1f667e86558b152ab128edf3ebf77a80a9e57` |

The earlier source inspection found 44 `.cu` files under `src`, direct CUDA and cuBLAS dependencies, NVIDIA PTX in `native_qsa_score.cu`, a compute-capability check in `device.cu`, CUDA graph/callback scheduling, and an NVIDIA-oriented installer. The top-level build states that the original `tests/` and `bench/micro/` trees were omitted from the published source.

These are observations of the pinned revision. The agent must recheck them against the exact checkout it uses. A newer upstream HIP implementation may reduce the work; inspect it before duplicating effort. Do not silently upgrade the baseline in the middle of a milestone.

GPU support does not establish support for every Fedora/kernel/ROCm combination. Inspect the existing installation first and consult the current AMD compatibility documentation before proposing system changes. A container may isolate userspace dependencies but does not replace a compatible host driver or GPU access.

### Authorized routine work

- Read local system diagnostics, inspect the repository and official technical documentation.
- Clone into a new project directory; edit source, tests and documentation there.
- Create a project-local environment, install unprivileged project dependencies, build and run bounded tests.
- Make local commits containing only reviewed task changes. Preserve existing instructions and unrelated files.
- Proceed through successive stages when their gates pass and the run budget remains available.

### Changes requiring a concrete proposal first

- Driver/kernel changes, package removal, system-wide ROCm replacement, privilege changes or reboot.
- Stopping another service to free GPU memory; never silently stop the existing model server.
- Full-model or other large downloads outside the initial budget; provide file names, source, license, size and free-space requirements first.
- Spending outside the user's selected agent plan/budget, sending local content to an additional provider, publishing code or contacting maintainers.

Do not use `sudo`, disable security controls, spoof the GPU architecture, expose a server publicly or broadly delete files as an automatic repair technique.

## 3. Copy-paste starting prompt

```text
Implement the attached Strata-ROCm-Fedora-Rollout-Plan.md on this Fedora PC.
Target the actual RX 7900 XTX via HIP/gfx1100 after verifying the hardware.

Read applicable AGENTS.md instructions first. Preserve my existing model server,
drivers, model files and unrelated projects. Use a new isolated checkout and a
separate build directory. Do not install or replace system software automatically.

Start by implementing the diagnostics, resumable status records and build/test
wrapper described in this plan. Then proceed through the staged conversion while
the gates pass. Do not stop merely to ask whether to continue routine authorized
work. Do not download the full model before the feasibility gate and download
approval. Do not claim a working port from compilation or plausible text alone.

Use one lead coding agent. Keep all commands, tests and handoffs provider-neutral.
Use HIPIFY for mechanical translation and independent references for numerical
tests. Audit CUDA-dependent .cpp and header files as well as .cu files. Replace
unsupported NVIDIA PTX with a correct HIP reference path before optimizing it.

Use the initial limits in section 7. Save progress and a precise blocker when a
limit is reached; do not loop indefinitely. Never raise tolerances merely to pass.
Report unsupported, untested and failed cases explicitly. If the original parity
suite is unavailable, create meaningful independent tests before claiming parity.

At every milestone update STATUS.md, TASKS.md and machine-readable results with
the exact revision, environment, commands, outcomes and next action. Finish each
session with a short plain-English report: what works, what remains unverified,
what blocked progress, and the exact next command or decision if needed.
```

## 4. Files and commands the agent must implement

Use a rollout directory containing this brief and a separate `Strata/` checkout. If either already exists, inspect and reuse safely; never overwrite a dirty repository or replace its instructions.

| File | Purpose |
| --- | --- |
| `Strata/AGENTS.md` | Concise project rules; merge with existing instructions |
| `Strata/STATUS.md` | Current stage, evidence, blocker and next step |
| `Strata/TASKS.md` | Ordered tasks and acceptance gates |
| `Strata/PORTING.md` | Backend design, unsupported features and numerical decisions |
| `Strata/strata-rocm` | Executable wrapper for repeatable operations |
| `Strata/scripts/rocm/` | Diagnostics, configure/build/test/report scripts |
| `Strata/tests/rocm/` | Independent references, fixtures and regression tests |
| `Strata/artifacts/rocm/` | Ignored local logs, environment manifests and result JSON |
| `Strata/build-hip/` | Ignored HIP build output |

Do not commit model weights, access tokens, environment secrets or large logs. Include a redacted, small representative report in documentation when useful.

### Command contract — these commands must be created

| Command | Required behavior |
| --- | --- |
| `./strata-rocm doctor` | Read-only diagnostics; report ready, blocked or unverified for each prerequisite |
| `./strata-rocm prepare` | Idempotent local setup; preserve dirty files and existing environments |
| `./strata-rocm build` | Configure/build HIP for the detected target; capture complete logs and exit status |
| `./strata-rocm test --stage kernels` | Run independent kernel tests; report failures and skipped coverage |
| `./strata-rocm test --stage model` | Run model-level checks only when approved weights and reference are available |
| `./strata-rocm bench` | Run a bounded benchmark with recorded settings and resource measurements |
| `./strata-rocm status` | Show stage, tested revision, passed/failed/skipped tests and next action |
| `./strata-rocm report` | Produce a redacted Markdown report and machine-readable result file |
| `./strata-rocm run` | Launch the validated configuration on loopback; check memory and port first |

Scripts must use timeouts, return meaningful nonzero status on failure and avoid interactive prompts inside automated tests. A test that is unavailable or skipped is not a pass. Derive build flags from the implemented backend; do not claim that upstream already supports a proposed HIP flag.

## 5. Staged implementation and gates

### Stage 0 — environment and source audit

Detect Fedora/kernel versions, CPU/RAM, GPU identity, free VRAM, disk capacity, ROCm/HIP compiler, CMake and build tools. Use installed diagnostics where available; record missing commands without installing replacements automatically. Check access to GPU device nodes and compile/run a tiny HIP allocation-and-arithmetic smoke test.

Resolve and record the full upstream commit, working-tree status and dependency revisions. Inspect relevant open upstream work without assuming it is correct. Verify license terms before distribution. Inventory every CUDA runtime, BLAS, intrinsic, assembly and graph dependency across the complete source tree.

**Pass:** a small HIP program runs on the correct GPU; source inventory and exact baseline are recorded. **Blocked:** GPU access or toolchain is unavailable. Provide one specific remediation proposal, not a list of speculative reinstalls.

### Stage 1 — automation and CPU reference foundation

Implement the wrapper, reports, budgets and resumable state. Build the available CPU-only components where possible. Determine which original tests actually exist. Implement independent small reference calculations and fixtures for the first ported kernels. Validate the test harness by deliberately injecting a local defect and showing it is detected, then remove that defect.

**Pass:** the harness catches an intentional failure, retains evidence and distinguishes pass/fail/skip. No claim of model correctness yet.

### Stage 2 — basic HIP backend and kernels

Add explicit backend selection while preserving CUDA. Choose a documented shared-source or generated-HIP strategy that avoids divergent duplicate maintenance. Keep NVIDIA flags separate from HIP flags. HIPIFY CUDA-dependent headers and host files as well as device code; inspect the generated changes.

Port device discovery, memory, streams, events, basic dequantization, elementwise operations, normalization, RoPE, routing and matrix-vector kernels. Check quantized signedness, packing, overflow, boundary sizes, subgroup widths and reduction behavior. Preserve artifact formats and CPU expert semantics.

**Pass:** the selected kernel suite runs on gfx1100 and meets predeclared reference tolerances. Record coverage by operation and quantization format; one passing format does not certify all formats.

### Stage 3 — difficult attention and prefill paths

Replace the NVIDIA PTX path in `src/kernels/cuda/native_qsa_score.cu` with a separately testable HIP reference implementation. Do not assume AMD matrix intrinsics accept NVIDIA fragment layouts. Preserve masks, indexing, scaling and accumulation semantics.

Port `src/prefill/gemm.cu` to supported HIP BLAS operations, checking transposes, strides, BF16/FP16 conversion, workspace handling and accumulation precision. Audit custom attention, recurrent/stateful operations and all remaining kernels required for a forward pass.

**Pass:** the required operation suite passes at representative model dimensions and edge cases. Report expected reference-path performance limitations. This is the feasibility gate for proposing the full-model download; it does not guarantee full-model success.

### Stage 4 — first full-model inference

Before downloading, present an exact model manifest and account for source weights, converted artifacts, temporary conversion space and RAM/VRAM workspace. Reuse existing compatible weights if available, with checksums and format verification. Choose one supported quantization format first.

Use a short context, single request and explicit debug scheduling. Add a serial/synchronized execution mode if necessary; do not delete required ordering operations to avoid deadlocks. Keep speculative decoding and vision disabled. A 4K context is a proposed initial configuration, not a measured capacity guarantee.

Compare layer outputs and logits against an independent reference using identical weights, tokenization and inputs. Prefer upstream fixtures or a compatible trusted implementation whose support for the exact architecture has been verified. If no reference is available, label output as exploratory and keep the parity gate blocked; fluent text is insufficient.

**Pass:** documented numerical checks, clean completion and short text generation with the validated configuration. Publish the exact scope of validation.

### Stage 5 — concurrency and repeated use

Restore CPU/GPU expert overlap, pinned-memory paths, stream/event dependencies and callbacks incrementally. Compare each change against the validated debug path. Add graph capture only after ordinary execution is correct. Test timeouts, request cancellation, repeated requests, state reset and memory cleanup. Do not assume CUDA graph behavior translates identically to HIP.

**Pass:** repeated requests and the agreed stress suite complete without hangs, stale state, numerical regressions or growing allocations. Failures remain reproducible from logged seeds/configuration.

### Stage 6 — useful performance and optional speculation

Benchmark the correct baseline before tuning. Measure time to first token, prompt-processing rate, generation rate, end-to-end time, host RAM, VRAM and CPU load. Record cold/warm state, model checksum, context, prompt, output length, thread count, compiler and flags. Use at least three repetitions and report the median and spread.

Tune one demonstrated bottleneck at a time. Larger expert caches must respect actual free VRAM and workspace requirements. Compare against an existing engine only if it supports the same model and relevant settings; otherwise label the comparison as a workflow comparison, not backend parity.

Speculative decoding is optional. Enable it only with tests for acceptance/rejection, recurrent/KV state rollback and equivalence to the validated non-speculative behavior under controlled settings. Do not promise bitwise agreement between vendors.

**Pass:** a measured improvement without correctness regression, or a documented decision to retain the correct simpler path. Do not extrapolate NVIDIA throughput claims to this GPU.

### Stage 7 — local release and handoff

Deliver reproducible build/launch commands, pinned dependencies, known limitations, validation report, troubleshooting notes and rollback instructions. Preserve a working local revision before experimental optimization. Any new service unit is opt-in. Public release or an upstream pull request requires the user's instruction.

**Pass:** the user can launch the tested configuration, stop it and return to the previous setup without editing GPU code.

## 6. Numerical evidence requirements

Record each fixture's provenance and checksum. An independent reference must not simply duplicate the kernel being tested; use a clear CPU mathematical implementation or verified external oracle. Separate quantization error from backend conversion error by comparing the same quantized weights and decoded values.

| Test category | Required checks |
| --- | --- |
| Integer/packing | Exact agreement where arithmetic is defined; signed inputs, zeros, extremes and block boundaries |
| Floating point | Finite outputs, absolute/relative error and error distribution; tolerances justified by datatype and reduction length |
| Attention/state | Masks, lengths, indexing, state initialization and multi-step updates |
| Routing | Expert IDs and scores; explicitly handle near-ties and documented tie-breaking |
| Model | Layer/logit comparisons, top-token margins and task-level sanity checks |
| Scheduling | Debug versus concurrent results, repeated requests, cancellation and cleanup |

Write tolerance decisions before evaluating candidate changes against acceptance. A changed tolerance needs a recorded mathematical reason and review of the failure; it cannot be a repair for unexplained divergence. Skipped or unavailable reference tests keep the relevant claim unverified.

## 7. Initial automation limits and recovery

These are proposed conservative defaults for the first local run, not estimates of total project duration:

| Limit | Initial value |
| --- | --- |
| Repair attempts per distinct failure signature | 3 |
| Execution slice before checkpoint/report | 60 minutes |
| Compile parallelism | Start at 4 jobs; adjust from observed memory pressure |
| Small GPU test timeout | 120 seconds per test unless justified otherwise |
| Individual build timeout | 30 minutes; report timeout separately from compiler failure |
| Unapproved dependency downloads | 2 GB cumulative; no model weights |
| Provider use | Current user-selected agent only; no automatic paid escalation |

A wall-clock or attempt cap is not a hard API spending cap. Use the chosen provider's supported spending controls when available; if usage/cost cannot be measured, report that explicitly and do not invent a dollar total. The wrapper can enforce its subprocess limits; the agent must respect its session limit. Do not claim the wrapper can control an arbitrary external agent's billing.

At a limit, terminate only owned child processes, save logs and checkpoint changes. Provide the smallest unresolved question and a resume instruction. No retry with the same evidence unless something relevant changed. Never reboot, reset the GPU or kill unrelated processes automatically.

### Minimum persisted state

- Exact revision and dirty-tree summary; record uncommitted patches safely.
- Completed gate IDs and evidence paths, including tested code revision.
- Environment and dependency versions.
- Current failure signature, attempts made and outcomes.
- Remaining unverified features and budgets used where measurable.
- Exact next command and whether a user decision is required.

On resume, check that code, configuration and dependencies still match previous evidence. Invalidate only affected results when they change. Avoid rerunning expensive passing tests without a concrete reason.

## 8. Human review and escalation

Each progress report should begin with one of: **working**, **partially working**, or **blocked**. Explain the practical implication first. For example: “Small GPU tests pass. Full-model correctness is unverified because reference fixtures are missing.”

If a stronger model is needed, prepare a compact handoff containing the decision, smallest reproducer, expected/actual results, versions, attempts and acceptance test. Redact secrets and ask before sending to another provider. Do not request hidden reasoning; request a proposed fix, explanation and testable evidence. Preserve verified conclusions in `PORTING.md`.

If the maintainer's omitted tests are needed, prepare a draft request explaining the exact fixtures required. Do not send it without authorization.

## 9. Definition of success

The first useful release is complete only when:

- The HIP backend builds reproducibly for the detected 7900 XTX environment.
- Independent tests cover the actual operations and quantization format used.
- Full-model numerical validation and short generation pass, with limitations stated.
- Repeated requests complete correctly and resource use is measured.
- A normal launch command works locally and preserves the previous model setup.
- The report distinguishes correctness, performance and unsupported features.

Vision, speculative decoding, maximal context, support for every AMD GPU and NVIDIA-equivalent throughput are not prerequisites for that first release. Keep them explicit follow-up milestones.

If a gate cannot be satisfied, a reproducible blocker report and preserved working checkpoint are the honest result. Compilation alone must never be reported as a completed conversion.

## 10. Technical references

These links ground the porting approach. Check live documentation for the installed ROCm version before choosing APIs or installation commands.

- Strata source: https://github.com/Niko1221/Strata
- Inspected revision: https://github.com/Niko1221/Strata/tree/6da1f667e86558b152ab128edf3ebf77a80a9e57
- Build definition: https://github.com/Niko1221/Strata/blob/6da1f667e86558b152ab128edf3ebf77a80a9e57/CMakeLists.txt
- HIP porting guidance: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- HIPIFY: https://github.com/ROCm/HIPIFY
- ROCm compatibility: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html

This plan deliberately makes no conversion-time or AMD-throughput promise. Those depend on the missing test coverage, the manual kernel work and measurements on the target PC.
