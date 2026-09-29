# Strata ROCm status report

Generated 20260928T225519Z · revision `d551edf42c157f1c8c20f07f68c5dfe506b1edda` · Fedora / ROCm HIP backend for `gfx1100`

**Scope of this report.** It states what has been executed on this machine and what has not. A build succeeding is not a port; a kernel agreeing with its reference is not a forward pass. Unsupported and unverified items are listed rather than omitted.

## 1. Environment actually used

- `os`: Linux 7.2.7-200.fc44.x86_64 x86_64
- `cpu_model`: AMD Ryzen 9 7900X 12-Core Processor
- `cpu_threads`: 24
- `mem_total_mib`: 96118
- `gpu_name`: Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]
- `gpu_gfx`: gfx1100
- `hip_version`: 7.1.52802-9999
- `cmake_version`: 4.3.0
- `disk_free_mib`: 20457
- checkout: `d551edf42c157f1c8c20f07f68c5dfe506b1edda`, tracked changes: none

## 2. Gates

| gate | status | evidence |
| --- | --- | --- |
| `S0-hardware-hip-smoke` | passed | ~/artifacts/rocm/logs/smoke-20260928T071227Z.log |
| `S0-source-audit` | passed | HIPIFY_REPORT.md |
| `S1-harness-selfcheck` | passed | TASKS.md#harness-self-check |
| `S1-cpu-reference` | partial | artifacts/rocm/results/test-kernels.json |
| `S2.6-iq-fixtures` | passed | artifacts/rocm/iq_fixture |
| `S2.7-core-runtime` | passed | artifacts/rocm/testlogs/strata-device.log |
| `S2-kernel-suite-gfx1100` | passed_with_skips | ~/artifacts/rocm/results/test-kernels.json |
| `S3.2b-mmq-types` | passed | artifacts/rocm/testlogs/prefill_mmq_parity.log |
| `S3.3-attention` | passed | artifacts/rocm/testlogs/native_flash_attn_parity.log |
| `S3.3-gdn-recurrence` | passed | artifacts/rocm/testlogs/native_gdn_parity.log |
| `S3.3-gdn-preprocess` | passed | artifacts/rocm/testlogs/native_gdn_preprocess_parity.log |
| `S3.3-moe-combine` | passed | artifacts/rocm/testlogs/native_moe_combine_parity.log |
| `S3.3-gr-postops` | passed | artifacts/rocm/testlogs/native_gr_postops_parity.log |
| `S3.3-qsa-ops` | passed | artifacts/rocm/testlogs/native_qsa_ops_parity.log |
| `S3.3-qsa-indexer` | passed | artifacts/rocm/testlogs/native_qsa_indexer_parity.log |
| `S3.3-rope` | passed | artifacts/rocm/testlogs/native_rope_parity.log |
| `S3.3-mmvq-q5k` | passed | artifacts/rocm/testlogs/native_mmvq_q5k_parity.log |
| `S3.3-forward-ops` | partial | artifacts/rocm/results/test-kernels.json |
| `S3-attention-prefill` | partial | artifacts/rocm/results/test-kernels.json |
| `S3-feasibility` | passed | TASKS.md |
| `S4-first-inference` | passed | artifacts/rocm/logs/first-inference-20260928.log |
| `S5-concurrency` | partial | artifacts/rocm/testlogs/core_graph_parity.log |
| `S6-performance` | pending |  |
| `S7-release` | pending |  |
| `S3.4-s2-expert-grouped` | passed | artifacts/rocm/testlogs/s2_expert_grouped_parity.log |
| `totally-made-up-gate` | passed |  |
| `S4.1` | passed | artifacts/rocm/results/record-s4-artifacts-verified.json |

## 3. What was run

- **build**: ok, 46 executables, 0 compiler errors, 4 jobs, timeout 1800s
- **HIPIFY**: 129 of 285 files rewritten from `c1e903310f211e6630780c3bd2038778c071c68d`, translator sha256 `bd4fcfd286bc`
- **declared patches after HIPIFY**: shfl-mask-literal-64bit x79, shfl-mask-identifier-64bit x3, hipblas-include-path x1, device-printf-global x514, hipgraph-instantiate-5arg x8
- **GPU smoke test**: pass (device, arch, integer exactness and float agreement recorded in `results/`)

## 4. Numerical evidence

Independent parity tests executed on the card: **41 passed, 0 failed, 2 skipped**.

| test | result | seconds | note |
| --- | --- | ---: | --- |
| `bf16_gemv_parity` | passed | 0.28 |  |
| `core_graph_parity` | passed | 0.23 |  |
| `cvec_parity` | passed | 0.24 |  |
| `dequant_s2_parity` | passed | 0.24 |  |
| `elementwise_parity` | passed | 0.18 |  |
| `fused_gdn_ab_parity` | passed | 0.18 |  |
| `fused_gdn_conv_l2_parity` | passed | 0.18 |  |
| `fused_gdn_parity` | passed | 0.18 |  |
| `gdn_parity` | passed | 0.26 |  |
| `gr_parity` | passed | 0.27 |  |
| `iq_parity` | passed | 0.19 |  |
| `kv_q4_parity` | passed | 0.19 |  |
| `kv_q8_parity` | passed | 0.21 |  |
| `kv_stream_parity` | passed | 29.27 |  |
| `native_expert_parity` | skipped | 0.01 | builds, but its reference compares CPU and GPU experts on real GGUF rows, so it needs a model shard. The Stage 3 feasibility gate has passed; the download is no |
| `native_flash_attn_parity` | passed | 0.19 |  |
| `native_gdn_parity` | passed | 0.21 |  |
| `native_gdn_preprocess_parity` | passed | 0.17 |  |
| `native_gr_postops_parity` | passed | 0.17 |  |
| `native_mmvq_q5k_parity` | passed | 0.23 |  |
| `native_mmvq_quant_parity` | passed | 0.19 |  |
| `native_moe_combine_parity` | passed | 0.18 |  |
| `native_qsa_indexer_parity` | passed | 0.18 |  |
| `native_qsa_ops_parity` | passed | 0.18 |  |
| `native_rope_parity` | passed | 0.19 |  |
| `native_router_parity` | passed | 0.18 |  |
| `ple_parity` | skipped | 0.01 | needs the model's PLE lookup table from the GSQ-RCO weight shards. The Stage 3 feasibility gate has passed; the download is now awaiting the user's decision (MO |
| `prefill_gemm_parity` | passed | 0.25 |  |
| `prefill_mmq_parity` | passed | 1.75 |  |
| `qsa_parity` | passed | 0.69 |  |
| `qsa_score_parity` | passed | 0.19 |  |
| `qsa_score_wmma_parity` | passed | 0.19 |  |
| `quantize_act_parity` | passed | 0.19 |  |
| `rope_parity` | passed | 0.14 |  |
| `router_top10_parity` | passed | 0.15 |  |
| `s2_expert_grouped_parity` | passed | 0.22 |  |
| `s2_gemv_parity` | passed | 0.14 |  |
| `s2_gemv_q8_parity` | passed | 0.15 |  |
| `s_gemv_parity` | passed | 0.16 |  |
| `s_gemv_q8k_parity` | passed | 0.15 |  |
| `sampler_parity` | passed | 0.32 |  |
| `shared_expert_parity` | passed | 0.15 |  |
| `strata-device` | passed | 0.14 |  |

Suite: in-tree parity tests (src/kernels/*_parity.cpp) on gfx1100

> passing here means each kernel agrees with its own in-tree reference. It does not certify the quantization formats whose fixtures are missing, and it says nothing about the shape of a full forward pass.

### Not verified

- native_expert_parity: builds, but its reference compares CPU and GPU experts on real GGUF rows, so it needs a model shard. The Stage 3 feasibility gate has passed; the download is now awaiting the user's decision (MODEL-DOWNLOAD-PROPOSAL.md).
- ple_parity: needs the model's PLE lookup table from the GSQ-RCO weight shards. The Stage 3 feasibility gate has passed; the download is now awaiting the user's decision (MODEL-DOWNLOAD-PROPOSAL.md).

## 5. Unsupported and excluded

None recorded.


## 6. Blockers and next action

- next action: resolve blocked checks, then ./strata-rocm prepare
- pending gates: `S1-cpu-reference`, `S3.3-forward-ops`, `S3-attention-prefill`, `S5-concurrency`, `S6-performance`, `S7-release`

## 7. Resource use

- wall clock recorded: 0 s
- downloads: 40551103.0 bytes (limit for unapproved downloads: 2147483648 bytes)
- repair attempts used: 0 of 3 per failure signature
- model weights downloaded: **none**
