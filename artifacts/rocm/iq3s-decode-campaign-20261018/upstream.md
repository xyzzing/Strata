# T02 — upstream recheck for the IQ3_S decode campaign (2026-10-18)

Base: `v0.1.45` (`477ac13c`, upstream main HEAD, 0 commits after the tag). PR states read from the
GitHub API on 2026-10-18; the 849-open-PR mirror from the R30 sweep is in `.scratch/pr-audit/`.

| item | state | content | verdict for this campaign |
| --- | --- | --- | --- |
| **#2514** | OPEN | HIP decode QSA top-k on the register kernel at 262K; 9 lines in `qsa_select.cu`; 1.7–1.8× kernel at 98K–262K on the 7900 XTX, same ids; `STRATA_TOPK_DECODE_REG=0` off switch | **take**: Phase 3, deep-tier decode only (production context is 131 072, so it pays above ~98K). Not in base. |
| **#3024** | OPEN | EXPDEC routed-expert kernels optimized and enabled by default on **gfx12** | **not directly applicable**: the reported numbers are gfx12/gfx1201; our card is gfx11. It does confirm EXPDEC is the active line of work. Use the in-base EXPDEC (from #2058) for the gfx11 cell. |
| **#2997** | OPEN | fused_gr tsum on gfx12, q8_inject guard, LDS-only barriers | **not applicable**: gfx12. |
| **#646** | CLOSED, unmerged | CUDA zero-doorbell resident verify graph, sub-warp expert packing (+39–72 %) | **closed**: not a base change, and CUDA-specific. Its mechanisms (graph capture, packing) are already covered differently on the native-pack HIP path (no per-layer graphs). No action. |
| **#783** | CLOSED, unmerged | CUDA fused decode/verify/MTP kernels on 0.1.39 | **closed**: superseded by the maintainer's own line; no portable content for HIP. No action. |
| **llama.cpp #26301** | OPEN (upstream llama.cpp) | `ggml-cuda: add dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K` | **not applicable now**: not in the pinned llama.cpp (`3cf03257…`), and K-quant-only — the IQ3_S expert formats are IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate-up and IQ4_NL / Q2_0 down. Recheck only if the expert formats change. |
| in base already | — | EXPDEC kernels (`#2058`, docs/EXPERTDEC.md), `STRATA_GROUPED_V1`, `STRATA_OLD_IQ_MMVQ`, `STRATA_VERIFY_PROFILE` / `STRATA_DECODE_TIMING`, `--dump-routing` | these are the campaign's instruments; no port needed. |

**Confirmed missing / applicable:** #2514 only. Everything else is either in base, gfx12, or
closed. The campaign therefore works the in-base kernels first (EXPDEC, grouped geometry, down)
and treats #2514 as the single external port.

**Recheck trigger:** if upstream merges a gfx11 decode kernel change for these formats, or a new
tag lands, re-read it before Phase 2 code is written.
