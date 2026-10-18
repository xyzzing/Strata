# IQ3_S decode campaign — running decision log (2026-10-18)

Backlog: `tasks.md` (verbatim copy of the user's file, sha256 `64ea21d8…`).
Fingerprint: `fingerprint.json`. Upstream recheck: `upstream.md`.
Baseline: production `0.1.45-8eb3e85d`, config pinned in the fingerprint.

## Phase 0 — pin and prepare

- [x] **P0.1** backlog copied verbatim (hash recorded).
- [x] **P0.2** `fingerprint.json`: code/build/model/pack/config/hardware. The pack's
      `native_experts.txt` **confirms the backlog's format table exactly** (gate/up IQ2_S 20,
      IQ3_XXS 17, IQ3_S 10, IQ4_XS 1; down IQ4_NL 39, Q2_0 9; 512 experts; 98,227,200 B per expert
      over 48 layers = 50,292,326,400 B total; 7 distinct format pairs).
- [x] **P0.3** `upstream.md`: #2514 is the only applicable external change; #3024/#2997 are gfx12;
      #646/#783 closed unmerged; llama.cpp #26301 not in the pin and K-quant-only.
- [x] **P0.4 harness readiness**
      - `native_expert_bench` (built) runs on the IQ3_S shard: layers 0/1, G=2, T=1 →
        `bitwise equal`, 38 µs all / 27 µs gate-up / 16 µs down. **Harness works.**
      - `STRATA_EXPDEC=1` **engages on gfx1100**:
        `strata expdec: CUDA0 gate/up type 18: expdec_gu_kernel R 1 MINB 4, 16 rows per block`.
      - The **down is not covered by default** (`STRATA_EXPDEC_DOWN`/`_D20` off): the log says
        `down type 20 (n_embd 2560, n_ff 640) not covered: the default kernel runs`. That is a cell
        for W1.5, not a defect.
      - **Deviation:** `expdec_bench` is `EXCLUDE_FROM_ALL` and guarded by
        `if(STRATA_ENABLE_CUDA)` in CMakeLists (CUDA-only target), so it does not exist in the HIP
        build. Rather than edit the production line's CMake in Phase 0, the equivalent matrix runs
        through `native_expert_bench` with the `STRATA_EXPDEC_*` env per invocation, and bitwise
        correctness through `iq_multi_parity`/`native_expert_parity` under `STRATA_EXPDEC=1`.
        If Phase 2 needs the dedicated variant bench, adding `STRATA_ENABLE_HIP` to that guard is a
        one-line fork commit.
- [x] **P0.5** R30 audit corrected: `STRATA_PROMPT_ATTN_IMMA` is **not compiled** into the gfx1100
      binary (0 strings), while `STRATA_EXPDEC`, `STRATA_QSA_ROWS`, `STRATA_QSA_SPLIT_ATTN`,
      `STRATA_SELECT_SIMT`, `STRATA_SCORES_THREAD`, `STRATA_HC_UPMIX`, `STRATA_PF_F16` are.
- [x] **P0.6** `scripts/rocm/r30-iq3s-decode-window.sh` written, `bash -n` clean.
- [x] **P0.7** campaign goal created; Phase 1 runs without further questions.

## Phase 1 — Window A (2026-10-18 09:41Z, completed, production restored)

- [x] W1.0 gates met (MemAvailable 68,587 MiB, VRAM 2,235 MiB). Strict parity **6/6 PASS**:
      `iq_parity`, `iq_multi_parity`, `s2_expert_grouped_parity` default **and** under
      `STRATA_EXPDEC=1` -> **EXPDEC is bitwise-correct on gfx1100 for every covered pair.**
- [x] W1.1 attribution: the one-shot summary gives the decode split (see below). The
      `STRATA_DECODE_TIMING` / `STRATA_VERIFY_PROFILE` **stage** line does **not** print in the
      one-shot path (docs: serve path) -> window B runs the serve path for the GPU stage profile.
- [ ] W1.2 baseline cells: **not run** (the script skipped the e2e block once EXPDEC lost the
      kernel bench). Deferred to window B.
- [x] W1.3 EXPDEC kernel matrix (real GGUF rows, layers 0,1,2,8,12,17,21,47 = all 7 pairs,
      `0 differing` in every cell; gu sums are 2 runs/layer):
      | shape | def | expdec | expdec-r2 | expdec-minb2 | expdec-r2-minb2 |
      | --- | ---: | ---: | ---: | ---: | ---: |
      | G20 T4 | **922.2 µs** | 991.1 | 953.5 | 999.8 | 960.4 |
      | G8 T1 | 353.6 | 375.7 | 351.9 | 378.0 | **353.4** |
      **gate/up EXPDEC is slower at the grouped shape and ties at single-token -> K1 REJECTED at
      the kernel level** (no e2e needed: a slower kernel cannot clear a +2% e2e gate).
- [x] W1.4 e2e: not run for K1 (rejected above).
- [x] W1.5 down coverage (bench only, window up later): with `STRATA_EXPDEC_DOWN=1 _D20=1`
      the down is **-27% at G34 T1** (41.9 -> 30.3 µs/layer/run), -30% at G48 T1, -19% at
      G20 T2, -2% at G10 T6. That is the one kernel-level win in the window.

**Window A attribution (attr-1k, 128 tokens, one-shot):** `wait for rings 23.410 ms/round`,
pool 3.567, host 0.014, commit 0.014; 49 rounds of 8, 79/102 drafts accepted, 2.61 tok/round;
mtp 4.156 ms/round; adaptive 0.056; CPU experts 1.55 distinct / 2.15 routed per layer; **pcie
1.08 distinct experts/layer = 141/256 of the misses (55%)**; decode 73.83 tok/s with the timing
overhead (baseline without it is ~95). **The decode window is GPU-bound: 66% of it is the host
waiting on the GPU.** Kernel work is therefore the right track, but the grouped experts are only
~15% of the window: at the real envelope (G34 T1, k=10 x ~3.4 rows) the grouped call is
108.6 µs/layer/run = 5.2 ms/round of 35.4 ms, and it runs at ~690 GB/s (HBM-bound). The other
~85% of the GPU wait is the stage profile's job (window B).

**Window A routing trace** (`--dump-routing`, binary record `int32 layer, int32 k, k ids, k
float weights`, parsed by `scripts/rocm/r30-routing_stats.py`): 14,496 records,
**k = 10** for every record, 302 rows/layer over 90 draft rounds (3.36 rows/round), 144,960
expert-entries; this is the shape the bench should use (G ~= 34, T ~= 1), not G20 T4.

## Phase 1 — Window B (2026-10-18 09:45Z, completed, production restored)

- [x] Baseline (production env, no instrumentation): **1K 96.09 tok/s** (5 cells, spread 0.46%),
      **32K 66.32** (2 cells), **128K 61.57** (2 cells, spread 10.6% - one slow cell; paired arms are
      the rule at depth). Acceptance deterministic: 162/166, 70/103, 36/56.
- [x] Down arm e2e: `STRATA_EXPDEC=1 _DOWN=1 _D20=1` = **-3.34% @1K (5/5 negative), -2.85% @32K,
      +3.01% @128K (noise)** -> **K2 REJECTED.** The -21..-34% kernel win does not survive the
      engine; `docs/EXPERTDEC.md` recorded the same CUDA pattern.
- [x] GPU stage profile (serve path; engine stderr in the config's `log`, not server stdout):
      35.07 ms/window staged - **waits 26.9%**, projections+attention 25.3%, **HC 15.6%**,
      expert VRAM hits 11.2%, shared+quant 6.1%, copy+combine+PCIe 5.3%, router 4.4%, head 3.3%;
      MTP draft 4.77 ms/round outside the stage total (11.3%).
      Caveat: the profiler disables sh_fork/branch/fuse_ab/side_mixer - the profiled round is
      42.1 ms vs ~27.2 ms uninstrumented, so the shares are indicative, not absolute.

**Phase 2 gate reading:** the decode window is GPU-busy, but its largest single item is a *wait*
(27%) that tracks the per-layer host CPU-expert serialization (host 5.40 ms/window; CPU experts
2.63 distinct/layer), and the largest true kernel clusters are bandwidth-bound projections (25%)
and the HC reads (15.6%). The grouped-expert kernel path (11%) has now been tried and rejected
(gate/up at the kernel level, down e2e). **No in-binary decode kernel lever can clear the +2% gate.**
The next candidates are: (1) T08 overlap - the `STRATA_VERIFY_DEFER_CPU` family (#2690, open PR,
output-changing) that moves a layer's light CPU rows to the next layer's FFN half; (2) a new gfx11
HC fusion kernel (HC_UPMIX is prefill-only; HC_PERSIST refuses gfx11 on `s_memrealtime`).

## Phase 3 — #2514 at depth (port 2026-10-18)

The port is `ed3606a6` on `refresh-0145`: `STRATA_TOPK_DECODE_REG=1` makes `fit` use `TK_PER_MAX`
for a decode call, so a 262,144-token context's 65,538 blocks take the PER_MAX register kernel
instead of the wide kernel. **The PR's premise does not hold on our base**: `0.1.45` already routes
`reach > fit` to the wide kernel (`campaign3` / `82922f55`), and 131K decode already takes the
register kernel either way. So this is a **wide-vs-register choice at 262K only**, not a ref-kernel
fix, and it is kept **default OFF** (the fork's opt-in rule) pending its own paired cell.
Gate (pre-declared in the script): same ids via `qsa_topk_parity` (both defaults) + median
ms/round ≤ −3 % at 262K + identical acceptance + no pair > +2 %.

**Result (window C, 2026-10-18 10:03Z): parity 4/4 PASS (same ids under both defaults), but the
262K cell is NEUTRAL — pairs +4.45 % (first base cell cold at 54.11), −0.16 %, −0.37 %; median
ms/round **+0.17 %** with a 4.44 spread → **gate FAIL, NOT ADOPTED**. The wide kernel already
covers 262K decode on this card. The port stays in the tree as a **dormant opt-in** (default OFF,
same ids, parity-tested); a 262K run may set `STRATA_TOPK_DECODE_REG=1`, but it buys nothing here.
Side data point, explicitly NOT a like-for-like claim: 262K decode measured **56.5 tok/s** on this
line/config vs the 69.3 tok/s recorded on the 0.1.43.1 sweep — different configs; flagged for the
R31 deep-tier re-measure.

Provenance note: `build-0145/strata` was rebuilt in place at `ed3606a6` (sha256
`b0d443f9…`); it keeps the `0.1.45-8eb3e85d` version string because the stamp is baked at
configure time. The running 8081 process is the pre-port inode until the window restarts it; with
the env unset the port is behaviour-neutral at the configured 131 072 context.

## Phase 3b — IQ4(a): STRATA_VERIFY_DEFER_CPU (#2690) ported and gated (2026-10-18)

Ported as merge `3f04756b` in worktree `strata-r146` (branch `port-2690`), build `build-0146`
green (sha256 `d2080793…`, stamp `0.1.45-ed3606a6-dirty`). Conflicts resolved: keep the fork's
PcieModel and the `hflag_in_h_`/`herr`/`hdrop`/`h_bal` handles (add `h_pflag_` to all), keep
upstream's `lb_+k` in `batch_poll`, keep both docs sections. Composition checked: TAIL_SKIP rows
are zeros; the deferred combine keeps the PCIe share's grouped launch; `late_add` uses the
pending write's `inj2_`/`Rt(tb)` arithmetic.

**Result (window D): quality holds, performance fails.**
- Parity 4/4 on the ported binary (default path).
- Battery A-B-A **7/8 = 7/8 = 7/8**, no prompt regressed; answers byte-identical across arms
  (the one miss, h2o, is a battery regex artifact in every arm). The deferral demonstrably
  engaged (every captured window printed ", deferred").
- Prose drift is noise-dominated: run 1 arm 0.000 vs restart floor 0.748; run 2 arm 0.734 vs
  floor 0.000 — the restart drift itself flips, so the battery + paired cells carry the gate.
- Paired decode: **1K −11.41 % (1/5), 32K −3.57 % (1/2), 128K −3.61 % (0/2)**. The mechanism is
  visible: the output change costs drafter acceptance (1K 162/166 → 146/179; 32K 70/103 →
  57/77), and although the 32K round is faster (−15.5 % ms/round), committed tok/s falls.
- **Verdict: NOT ADOPTED.** Production stayed on `build-0145`, config unchanged, healthy after
  the window. The port stays dormant on `port-2690`; a `BELOW` sweep (0.05/0.02) is the only
  declared follow-up and trades divergence for acceptance.

## Follow-ups approved 2026-10-18 (the six-item queue)

Tracking moved into `tasks.md` (the verbatim backlog copy, now with an execution-status table per
task); records for each item live beside it.

1. **262K deep-tier A/B — done.** 0.1.43.1 vs 0.1.45 on the same 261,888-token workload/config:
   **0.1.45 is +3.96% faster** (pairs +2.70/+3.96/+39.11, the third 0143 cell a stall), prefill
   +3%. The 69.3-vs-56.5 gap is **config/methodology, not a line regression** — the suspected
   deep-tier regression is closed. (`record-iq3s-262k-line-ab-20261018.json`)
2. **ROUTE_PRIOR campaign — DONE, ADOPTED λ=0.5.** Screen + battery (7/8=7/8) + clean confirmation:
   1K +22.20% (5/5), 32K +25.56% (2/2), 128K +0.24% tok/s / −15.4% ms/round; CPU experts/layer
   1.61→0.44-0.73, PCIe 1.19→0.20-0.89. Live in serve-8081.json (backup, health, Paris smoke).
   (`record-iq3s-window-route-prior-20261018.json`)
3. **BELOW sweep — done, NOT ADOPTED.** On top of the live ROUTE_PRIOR=0.5: 1K −2.48% (2/5),
   32K −3.96%, rounds slower at both tiers. The deferral stays dormant.
   (`record-iq3s-window-below-sweep-20261018.json`)
4. **HC fusion — NO-GO (feasibility done).** The default decode read is norm+down_split+down_finish+up
   per half; the only fused form (HC_PERSIST) refuses gfx11 (no `s_memrealtime`) and its closest
   RDNA prior (R9700, #2226) saw no gain; the 15.6% target share is from the instrumented pipeline.
   Reopen only with an uninstrumented HC share >=10% or a barrier-free fusion shape.
   (`record-iq3s-hc-fusion-feasibility-20261018.json`)
5. **Prefill cells — done, two adoptions.** Scorer three-way @128K: fp16 arm select **1,714 ms**
   vs exact **4,320 ms** vs radix **~10,000 ms**; total ties (2,066 vs 2,055 tok/s) — the exact
   scorer is not a gfx1100 win, which closes #1298's question. **GDN chunked +5.85% @32K ADOPTED**
   and **PF_GEMM+PAD +4.41% @128K ADOPTED**, gated by `hip_gdn_chunked` parity + an 8K needle
   answered identically in base/GDN/PF/base (`.bak-prefill-pre`).
   (`record-iq3s-window-prefill-20261018.json`)
6. **Hygiene — done**: refresh-0145 pushed at `dbb0a561`; **all four** disposed PRs closed with
   their numbers (#2224, #1296, #1297, #1298 — the last with the fp16-vs-exact-scorer phase data).

## Second queue (2026-10-18, late) — executed

1. **Re-baseline + re-profile (live env).** 1K 101.8-142.0 across trajectories, 32K 73.5, 128K 52-60
   (noisy), 262K 44-45 with the prior **+27.8 % (3/3)** vs off; prefill 2,337-2,421 @32K/128K and
   2,315 @262K. Profile: **waitB 9.45 → 3.02 ms**, CPU 2.47, GPU-reach wait 24.78; the prior's
   router kernel costs 0.77 ms and saves 6.4 ms.
2. **Tuning:** λ 0.5 is a sharp optimum (0.25 −47.15 %, 0.75 −38.67 %); TAIL_SKIP=0 still +38-43 %;
   **20-prompt battery 20/20 = 20/20**, no regressions.
3. **T08:** reserve 3072 and workers auto stand (2048 −40 %, 4096 −11 %; 9/13 mixed); lookahead
   stats not captured.
4. **Hygiene:** shard2 hash recorded; spec-off impossible by design; >LLC no cliff; replay covered.
5. **Upstream:** v0.1.45 unchanged; #3024 open (gfx12).
6. **Branches:** keep #2514 carried; keep #2690 standalone with re-test triggers.
   (`record-iq3s-second-queue-20261018.json`)

## v2 adoption and Phase 1 (2026-10-18)

v2 installed (`tasks-v2.md`), v1 archived, **local-LLM executor removed** (no relay exists; the
specimen and 8080 are not coders; the DSH agent implements, a fresh-context subagent reviews at the
mandatory checkpoints). V01 and U01 delivered: HC/experts/head at HBM bandwidth (758/730/734 GB/s),
**dense projections the only material gap at 558 GB/s (~2.0 ms/window, ~5.8%)**; U01 pins the stack
and finds no nightly justification (counters blocked, no compiler-shaped symptom). Recommendation:
V05 scoped to the projection layout with a discriminating micro-benchmark first. Review checkpoint
before code. (`record-iq3s-v01-u01-20261018.json`)

## V01 review + calibration (2026-10-18, round 2)

The fresh-context reviewer confirmed the HC/experts arithmetic, challenged the projection byte basis
(xT unlicensed), the assumed 750 GB/s wall, the "column-interleaved copy" arm (a Q8_0 re-encode) and
the head row (660, not 734), and found the 8.60 ms projection row wrong (named GEMV stages sum to
6.25 ms). Calibration: (a) `q8_gemv_bench` — large shapes at the wall (qkv 695, q 769, head 912),
small/medium shapes at 25-60% (gate 426, HC up 289, shared 308-375, k/v 340); (b) two serve profiles
at spec 2 vs 8 — stage total **24.04 → 29.14 ms for +58% rows (+21%)**, so weights are largely
amortized; **router+plan+bell is 2.25 → 2.37 ms, T-flat and overhead-bound**.

Decision: **V05 dropped by its own falsifier** (layout is not the lever). First experiment selected:
**fuse the router path** (router+prior+topk, 2.37 ms/window, expected 0.7-1.0 ms ≈ 2.0-2.9%) behind
an opt-in, gated by the existing router parity. Runner-up: batch the small GEMVs. Incident disclosed:
the bench was once run against the live card via `--help`; production stayed healthy.

## First v2 experiment — the router-select skip (rejected, 2026-10-18)

With `STRATA_ROUTE_PRIOR=0.5` the prior re-picks the top-10 from the same logits and overwrites
ids/weights, so the router's own `native_router_top10_multi` selection is dead work. `STRATA_ROUTER_SKIP_TOPK=1`
(commit 662d4736, since reverted) dropped that one launch per layer; it is **exact by inspection**
(`route_prior_k` reads only logits and `res`; ids/weights are write-only). Paired cells: 1K median
**+1.31% / 3-of-5** (a first window showed +2.91% / 4-of-5, not reproduced), 32K/128K inside the
trajectory band. It misses the pre-declared ≥+2% gate and was **not adopted**; the router path is
already batched, and the remaining redundancy is one cheap launch per layer. The cross-process
routing-trace check is invalid here (greedy restart drift: 99 vs 93 drafts in the two runs).
Runner-up still open: batch the small GEMV calls (z/rec/ab + shared, ~1.7-2.9% bound).

## Decisions

| id | decision | evidence | next |
| --- | --- | --- | --- |
| D0 | The backlog's format table is accepted as verified (pack-native, not inferred). | `native_experts.txt` | use it for harness layer picks `0,1,2,8,12,17,21,47` |
| D1 | `expdec_bench` is skipped on HIP; the matrix uses `native_expert_bench` + parity env runs. | CMake guard + working smoke | revisit only if a dedicated variant sweep is needed |
| D2 | **K1 (EXPDEC gate/up) REJECTED** on gfx1100: slower at G20 T4 (+7.5% def=922 vs expdec=991), tie at G8 T1; bitwise correct (6/6 parity). | window A logs `r30-iq3s-bench-*` | do not e2e; keep the default gate/up kernel |
| D3 | **K2 (EXPDEC down with `_DOWN=1 _D20=1`) is a live candidate**: -27% down at the real G34 T1 envelope, bitwise covered by the same parity suite. | `r30b-bench-expdown-*` (2026-10-18) | e2e in window B at 1K/32K, gate >= +2% |
| D4 | The decode window is **GPU-bound** (wait-for-rings 23.4 of 35.4 ms/round); the grouped experts are ~15% of it and HBM-bound. Next kernel target must come from the stage profile. | attr-1k summary | window B: serve-path `STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1` |
| D5 | **K2 (EXPDEC down) REJECTED e2e**: -3.34% @1K, -2.85% @32K, despite -21..-34% at the kernel level. | `record-iq3s-window-b-20261018.json` | dispatch stays default |
| D6 | **The in-binary kernel space is exhausted for decode**: gate/up rejected, down rejected, QSA_ROWS/QSA_SPLIT_ATTN <=1.4% each by the profile, HC_UPMIX is prefill-only, HC_PERSIST refuses gfx11. Any further kernel win needs new code. | stage profile + windows A/B | next: T08 overlap (#2690) or a gfx11 HC fusion |
| D8 | **#2690 (DEFER_CPU) ported and REJECTED**: quality battery holds (7/8 = 7/8, identical answers) but decode falls −11.4 % @1K, −3.6 % @32K, −3.6 % @128K because the changed output costs drafter acceptance; the 32K round is faster (−15.5 % ms/round) but committed tok/s is not. | `record-iq3s-window-d-defer-cpu-20261018.json` | production unchanged; port dormant on `port-2690`; only a BELOW sweep remains as a declared follow-up |
| D7 | The profiled pipeline is not the production pipeline (fusions off; 42.1 vs 27.2 ms/round). Shares are indicative only; candidates are decided by uninstrumented paired cells. | engine log + baseline | never adopt on a profiled number |
