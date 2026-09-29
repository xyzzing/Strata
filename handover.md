# handover.md — Strata HIP/gfx1100 rollout

**Written:** 2026-09-28, session 3 (overnight) · **Revision:** `Strata/` at
`c0107c3228f0be4636543d7ed7b16d63c6140506` (upstream tag **v0.1.16**, MIT),
tracked tree clean
**Stage:** 4 of 7 · **State: the port generates text.** First inference done;
claim is qualified (no same-weights logit reference yet).

## 0. The one-line state

"The capital of France is" → " Paris. The capital of Germany is Berlin" —
one-shot greedy on the RX 7900 XTX through the full HIP port (46.84 GiB of
IQ3_S experts mapped, PLE table from shard 2, MTP draft resident), at
32.35 tok/s decode BESIDE the user's own 8080 server; and through upstream's
web UI + OpenAI API on loopback 8081: "Name the capital of France in one
word." → **"Paris"**. Kernel suite: **40 passed, 0 failed, 2 skipped**
(both weight-gated).

## 1. What session 3 did

1. **Merged v0.1.15 → v0.1.16** (no device kernel changed; all 5 declared
   patches apply; upstream is now MIT — S0.6 licensing unblocked).
2. **Closed S3.2b'**: `prefill_mmq_parity` builds IQ2_XXS/IQ2_XS fixtures with
   a uniform importance matrix (ggml's low-level quantizers accept any non-null
   imatrix); all 8 MMQ types now tested, at 0.001 of the derived bound.
3. **Closed the S5.2 graph residuals**: the per-step-pointer hazard is REAL
   (arguments frozen at capture — measured), concurrent replay is safe.
4. **Verified every model artifact** (sha256, `record-s4-artifacts-verified.json`)
   and packed the model: 1.5 GB IQ pack (`packs/iq3_s`, experts stay in the
   GGUF — the engine mmaps them), tokenizer exported, MTP packed + runtime
   files built (`mtp/rt/`).
5. **First inference + web UI serve on 8081** (evidence:
   `logs/first-inference-20260928.log`, `logs/serve-8081-20260928.log`,
   `record-first-inference`, `record-web-ui-serve`).
6. **Found and fixed the port's first forward-path defect**: `fused_gr_read_multi`
   staged 80 KB of dynamic LDS — past gfx1100's 64 KB limit (fine on upstream's
   RTX targets). Declared patch **0005** (TILE 2560→1024, tile loop exact,
   zero math change) — `PORTING.md` §12a.
7. **rocWMMA QSA fast path**: v0.8 vendored unprivileged, compiles for gfx1100
   (probe in `artifacts/rocm/wmma-probe/`); implementation is pending, with the
   fp16 precision-bound requirement recorded in `PORTING.md` §12c.

## 2. The incident to know about

The serve engine's default expert arena PINS ~47 GB of host RAM
(`hipHostRegister`). Under coexistence with the user's 8080 server that drove
global memory pressure; the kernel OOM-killed OUR strata process
(`record-oom-incident`). The user's 8080 llama-server was **not** kernel-killed
but is DOWN after the event — **restarting it is the user's action** (its
previous full command line is preserved in `record-oom-incident` and in
STATUS.md). The serve config (`artifacts/rocm/serve/strata-hip.json`) now
defaults to `--mmap-experts`, which keeps experts in the page cache instead.

## 3. Blockers / decisions

| # | item | state |
| --- | --- | --- |
| 1 | **8080 server restart** | **yours** — see §2; rollout never starts/stops it |
| 2 | S4.3 same-weights logit reference | needs a RAM window with 8080 stopped (your call), or accept the recorded qualified claim |
| 3 | QSA WMMA fast path | feasibility done; implement behind `qsa_score_parity` second bound (PORTING §12c) whenever wanted |

## Session-4 addendum (2026-09-29)

- **Patch 0006 landed and validated**: file-backed IQ experts (`--mmap-experts` + the `--experts-bin` pack) — the 47.9 GiB pin is gone; A/B 16/16 identical greedy tokens vs the arena path (`results/record-expertsource-ab.json`).
- **The server runs at 98304 context** (int8 streaming KV) with ~59 GiB RAM available: `systemctl --user status strata-8081`, http://127.0.0.1:8081. The four OOM kills that ended session 3 are explained and fixed (PORTING §12d–12e).
- Serve recipe is in README §"Run the server"; the pack with `experts.bin` is `/mnt/LINUXWINSHARE/Backup/models/packs/iq3_s-full`.

## 4. Exact next commands

```sh
cd ~/projects/strata-amd && ./strata-rocm status   # 4 of 7, working

# one-request inference (coexists with the 8080 server):
./build-hip/cmake/strata --pack packs/iq3_s --tokens "760,6511,314,9338,369" \
  --native ~/models/692a40d0341816e7a902267bb139f4efd00e047aa132d07490a14d75f8620fb3 \
  --ple-gguf "/mnt/LINUXWINSHARE/Backup/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  --spec 2 --mtp /mnt/LINUXWINSHARE/Backup/models/mtp/rt --prefill 1024 \
  --expert-profile Strata/data/expert-profile.bin --expert-cache auto \
  --max-new 16 --greedy --max-context 4096

# web UI (after the 8080 server is back, or alone): serve config carries --mmap-experts
cd build-hip/hipify && ../../.venv-rocm/bin/python serve/server.py --engine strata \
  --config ../../artifacts/rocm/serve/strata-hip.json --port 8081 --host 127.0.0.1
```

## 5. Constraints carried forward (unchanged)

- **Port 8080 is not ours.** Restarting it is the user's action. Ours binds 8081.
- **No sudo/dnf/driver changes/reboots.** rocWMMA is vendored, not installed.
- **`Strata/` pristine** at v0.1.16; six documented onboarding-hook files.
- **`build-hip/` generated**; change a declared patch, then `hipify --force`.
- **A build passing is not a port; fluent text is not a logit reference.** The
  S4 claim stays qualified until S4.3's comparison exists.
