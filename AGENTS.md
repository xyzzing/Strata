# Strata ROCm rollout — agent rules

Read `STATUS.md` first (what is actually proven), then `TASKS.md` (ordered work
and gates). `PORTING.md` holds the design decisions and the reasons, including
three Fedora-ROCm traps that will otherwise cost you an hour each.

## Hard constraints

- **The existing model server on port 8080 is not yours.** It is a llama.cpp
  instance holding ~19.6 GiB of the card's 24 GiB. Never stop it, never free its
  memory, never reconfigure it. This rollout's own server targets loopback
  **8081**.
- **No `sudo`, no `dnf`, no driver, kernel or system-wide ROCm changes.** No
  reboots. No package removal. If something seems to need one, write a proposal
  instead.
- **`Strata/` is a pristine checkout.** Do not add, edit or delete anything
  inside it to make the port work. It is the revision every piece of evidence
  names. Exceptions: `.agentic/checks.json`, which the upstream workflow
  instructs the project to configure, and five files an external agent-onboarding
  hook dropped in (`AGENTS.md`, `CLAUDE.md`, `.agentic/install-state.json`,
  `.agentic/verify.py`, `.agentic/workflow.md`) — not upstream source, documented
  here rather than deleted, and never edited by the rollout.
- **`build-hip/` is generated.** Never hand-edit `build-hip/hipify/`. Change a
  declared patch in `scripts/rocm/patches/` and regenerate with
  `./strata-rocm hipify --force`.
- **Do not download model weights** before the Stage 3 feasibility gate passes
  and the user approves a concrete manifest with sizes, checksums and free-space
  numbers.

## Running anything that touches the GPU

The default shell sandbox hides `/dev/kfd` and `/dev/dri` (its own minimal
`/dev`), while `/sys/class/drm` still shows the card. That is a namespace
property, not a driver fault — `./strata-rocm doctor` says so explicitly.

Compiling works inside the sandbox. **Running does not.** GPU runs need the
sandbox widened, and they should be batched: one widened invocation for a whole
group of tests beats one per command.

## Claiming things

- **A build passing is not a port. Plausible text is not evidence.** Every
  correctness claim needs an independent reference and a recorded comparison.
- **Skipped is not passed.** A test that could not run exits 5, and that is not a
  green result. `tests/rocm/unavailable.json` declares which tests may be
  skipped and the exact output signature that earns the skip — a declared test
  that fails for a different reason is reported as a failure.
- **Never raise a tolerance to make a test pass.** If a reference disagrees,
  either the port is wrong or the arithmetic contract is different (see
  `PORTING.md` §7, where the answer was a compiler flag and *no* tolerance
  moved). A changed tolerance needs a mathematical reason recorded in
  `PORTING.md` first.
- **Report unsupported and untested cases explicitly.** An exclusion from the
  build is written into the build summary and the report, never silently dropped.

## Every milestone

Update `STATUS.md`, `TASKS.md` and the machine-readable results
(`artifacts/rocm/results/`) with the exact revision, environment, command,
outcome and next action. `./strata-rocm report` regenerates `REPORT.md` from the
recorded state — use it rather than writing numbers by hand.

End a session with a plain-English summary: what works, what is unverified, what
blocked progress, and the exact next command or decision.

## Command surface

```sh
./strata-rocm doctor     # read-only diagnostics (run outside the sandbox for GPU checks)
./strata-rocm prepare    # vendor HIPIFY, create the venv; idempotent, unprivileged
./strata-rocm hipify     # regenerate build-hip/hipify from the pinned revision
./strata-rocm build      # CMake + ninja, gfx1100 only, keep-going, complete logs
./strata-rocm test --stage kernels|smoke|model
./strata-rocm bench      # kernel-level timing; NOT a model benchmark (no engine yet)
./strata-rocm status     # stage, gates, pass/fail/skip, blocker, next action
./strata-rocm report     # redacted REPORT.md + results/report.json
./strata-rocm run        # launch on loopback; refuses while no engine binary exists
```

Exit codes are distinct on purpose: `0` ok, `1` failure, `3` blocked,
`4` test failure, `5` skipped coverage (not a pass), `124` timeout.
