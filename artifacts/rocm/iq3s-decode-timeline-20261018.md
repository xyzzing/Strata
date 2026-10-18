# V01 — decode execution timeline and boundary table (2026-10-18)

Status: **deliverable complete for the current profile; one open question (projection efficiency
cause)**. Sources: the live-env serve-path profile in
`record-iq3s-second-queue-20261018.json` (45 windows, avg T 3.13), the engine's
`STRATA_DECODE_TIMING` line, `src/core/verify.cpp`'s stamp sequence, and the kernel/geometry
constants in `src/kernels/cuda/fused_gr.cu` (`N=2560`, `HC=4`, `D=10240`, `LR=320`) plus the
expert layout (`H=2560`, `FF=640`, 512 experts, k=10).

No new GPU run was needed: every stage time below is the existing measured profile. Byte counts are
**derived** from the constants and the per-layer-window counters (marked "d"); where a count is not
derivable it is marked "?" and listed under open questions. Throughput is bytes ÷ stage time; the
"HBM" column compares against a ~750 GB/s planning figure, not a benchmark of peak.

## Per-window stage table (T = 3.13 tokens, round = 35.08 ms, stage total = 28.89 ms)

| stage (profile name) | producer → consumer | bytes/window (d) | ms | achieved | HBM | graph / stream |
| --- | --- | ---: | ---: | ---: | --- | --- |
| HC norm+down+up (hc0+hc1) | R → xn → lo → mixed | 3.9 GB | 5.20 | **758 GB/s** | at limit | captured; one stream |
| projections (q8+qkv, q/q-idx, out-proj, z, rec, ab) | residual → GEMVs → residual | 4.8 GB (dense pack / token × T) | 8.60 | **558 GB/s** | **26% under** | captured; one stream |
| expert VRAM hits | plan → grouped gate/up+down | 2.97 GB (28.1 hits × 2.2 MB × 48) | 4.07 | 730 GB/s | at limit | captured; one stream |
| router+plan+bell | residual → logits → top-10 → plan | ~0.4 GB + the prior kernel | 2.32 | overhead-bound | under | captured |
| shared+quant | q8_1 quantization of the MoE input | small | 2.06 | overhead-bound | under | captured |
| head | last residual → vocab GEMV | ~0.7 GB | 1.06 | 734 GB/s | at limit | captured |
| MTP draft | drafter forward | ? | 3.86 | — | — | separate windows |
| waitA+B | GPU waiting for CPU/PCIe rows | — | 3.02 | — | — | after the route-prior adoption (was 9.45) |

The host's GPU-reach wait is 24.78 ms of the 35.08 ms round; the GPU stage total is 28.89 ms, so
the window is **GPU-critical-path** (~82% busy), not submission-bound. Graph coverage is already
the verify windows themselves (every decode stage above is inside a captured 1-8-token graph), so
V03 has no separable submission gap to close in the decode window.

## Findings

1. **HC, experts and head are at HBM bandwidth** (730-758 GB/s). There is no fusion or repack
   headroom without reducing bytes, and reducing bytes means a precision change, which is out of
   scope. V02 (expert fusion) and any HC fusion are therefore **not** recommended: they would
   target clusters that are already at the memory wall.
2. **The dense projections are the only cluster with a material efficiency gap**: 558 GB/s against
   the 730-758 GB/s the neighboring kernels achieve, i.e. ~26% below. Closing it to ~730 GB/s would
   save ~2.0 ms/window = **~5.8% of the round** (f = 24.5%, s ≈ 1.31; ceiling 1/(1-f) = 1.32).
3. **Router+plan (2.32 ms) and shared+quant (2.06 ms) are overhead-bound**, not byte-bound: a fused
   router+topk+prior+plan (or quant folded into the preceding kernel) could plausibly save ~0.5-1 ms
   (~1.5-3%). This is a second, lower-ceiling candidate.
4. The route prior moved the window as recorded: waitB 9.45 → 3.02 ms, CPU experts 1.55-2.63 → 0.93
   per layer, at the cost of its own 0.77 ms router kernel.

## Pre-review correction: the projection weights are quantized

The pack index shows the dense projection tensors stored quantized (e.g. `attn_qkv.weight`
2560×10240 and `ssm_out.weight` 6144×2560, type byte 8 = the Q8_0 family; the decode path calls
`native_mmvq` / `s_gemv`, not a bf16 GEMV). **"558 GB/s" is therefore bytes ÷ time, not a measured
memory-bandwidth ceiling** — it may be decode/issue-bound rather than load-bound. This does not
change the ranking (the cluster is still the only one with a material gap against its byte floor),
but it changes what the discriminator must prove: a layout permutation only helps if the same
decode work moves faster. If it does not, the correct experiment is a kernel change in the
quantized GEMV (wider loads / codebook staging / occupancy), not a pack repack — and V05 should then
be dropped rather than forced.

## Open question and the discriminating test

The projection gap's **cause is not yet established**: low load efficiency (layout/alignment) or low
occupancy (many small GEMV launches, each latency-bound). The two fixes are different (V05 repack vs
a V02-class kernel-shape change), so the next step is one discriminating measurement, not a pack:

- **Discriminator**: a micro-benchmark of the projection read pattern on this card — the dense
  projection weights read by the GEMV shape at T=1 and T=4, current layout vs a 1024-B row-padded
  (and a column-interleaved) copy in VRAM — reporting achieved GB/s. Falsifier: if the padded copy
  does not raise achieved bandwidth by ≥10%, the cause is occupancy, V05 is dropped, and the
  runner-up (router/plan fusion) becomes the experiment.
- Bounded: one benchmark, three layouts, two token counts, on the live stack; no pack change and no
  engine change until it reports.

## Recommendation (v2 formula applied)

**Start V05 with the discriminator above, scoped to the dense projection weights.** It is the
largest defensible exposed fraction (24.5%) with a mechanism (26% bandwidth gap) and the doc's own
layout tool. If the discriminator falsifies the layout hypothesis, stop V05 and move to the
router+plan fusion (f = 6.6%, plausible ~1.9%) rather than widening the pack work.

Do not start V02 (experts at 730 GB/s), V03 (no separable submission gap), V06 (the prior already
bought the wait reduction; waitB is 3.02) or V07 (deferred) on this evidence.


---

# V01 addendum (2026-10-18, after review + calibration) — supersedes the recommendation above

## Review outcome (fresh-context, bounded)

CONFIRMED: the HC arithmetic (3.94 GB / 758 GB/s) and the experts unit (2.97 GB / 729 GB/s).
CHALLENGED, and accepted:
- The projection byte basis' ×T factor was unlicensed; the defensible range is 179–560 GB/s.
- The "HBM wall" is an assumed 750 GB/s figure, not measured.
- "Column-interleaved copy" of a Q8_0 tensor is a re-encode, not a copy; a <10% null rules out one
  layout family only; and there is no pure-read reference arm.
- The head row was inconsistent (0.7 GB / 1.06 ms = 660 GB/s, not 734).
- The **8.60 ms projection row was wrong**: the named GEMV stages sum to **6.25 ms**, not 8.60.
Corrections applied below.

## Calibration measurements (same day, one window each)

**Per-shape achieved bandwidth** (`q8_gemv_bench`, native_mmvq, synthetic Q8_0 at the engine shapes,
1 col / 4 cols): GDN qkv 2560→10240 27.85 MB **695 / 663 GB/s**; QSA q 2560→12288 769 / 689; out
6144→2560 636 / 606; **GDN gate 2560→6144 426 / 472**; **HC down 10240→320 504 / 317**; **HC up
320→10240 289 / 179**; **shared gate/up 375 / 274**; **shared down 308 / 207**; **k/v 2560→512 340 /
236**; head 2560→248320 **912 GB/s**. The large shapes are at the wall; the small/medium ones are
25-60% of it. Footnote: the engine stores some of these tensors at other quant types, so these are
shape/pattern bounds, not the engine's exact byte counts.

**T-scaling of the engine stage** (two serve profiles, same request, spec 2 vs spec 8):
stage total **24.04 ms at avg T 2.00 → 29.14 ms at T 3.17 (+21% for +58% rows)**.
- router+plan+bell: **2.25 → 2.37 ms (+5%) — effectively flat, overhead-bound**.
- HC cluster: 4.43 → 5.02 (+13%) — weights amortized.
- projection GEMVs: 5.26 → 6.25 (+19%) — mostly amortized, partly row-bound.
- expert VRAM hits: 3.20 → 4.05 (+27%) — most row-sensitive.
- waits: 1.42+1.80 → 2.30+1.50.

## Revised finding and recommendation

**V05 is dropped by its own falsifier.** The weight bytes are largely read once per window, so a
storage permutation cannot recover the gap; the inefficiency is per-call shape/occupancy (small
shapes at 25-60% of wall) and per-launch overhead. The reviewer's re-encode objection to the
column-interleave arm reinforces it.

**First experiment (selected): fuse the router path** — `router+plan+bell` is **2.37 ms/window,
T-flat (2.25→2.37 for +58% rows)**, i.e. the largest purely overhead-bound cluster (8.1% of the
stage). A bounded fused router+prior+topk kernel behind an opt-in could plausibly cut it to
1.3-1.7 ms, i.e. save **0.7-1.0 ms/window (2.0-2.9% of the round)**, with the existing router
parity tests as the correctness gate. It touches `verify.cpp` (the thick file) and the router
kernels, so it is scoped to one fused launch, no plan/doorbell restructuring.
Runner-up if the router fusion is rejected at parity or misses the gate: batch the small GEMV calls
(z/rec/ab + shared) into one launch (combined 2.0-2.5 ms at 40-50% efficiency; expected 0.6-1.0 ms).
Not selected: V02 (experts at the wall), HC fusion (amortized), V03 (windows already captured),
V06 (waitB is 3.02 after the prior), V07 (deferred).
