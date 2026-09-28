# HIPIFY report

Generated 20260928T142657Z from `c0107c3228f0be4636543d7ed7b16d63c6140506` with hipify-perl `bd4fcfd286bc`.

- files in the generated tree: 270
- files hipify-perl rewrote: 125
- tree: `build-hip/hipify/` (generated, never hand-edited; `.prehip` files are hipify-perl's before-image)

## Declared textual patches applied after HIPIFY

| patch | hits | what it means |
| --- | ---: | --- |
| `shfl-mask-literal-64bit` | 79 | HIP's lane mask is 64-bit; 0xFFFFFFFF zero-extends to all 32 lanes of a wave32 target and leaves wave64 lanes on their own. |
| `shfl-mask-identifier-64bit` | 3 | Same requirement for a mask held in a 32-bit variable; zero extension is the same value. |
| `hipblas-include-path` | 1 | HIPIFY rewrites <cublas_v2.h> to <hipblas.h>, which does not exist on this install; the header lives at <hipblas/hipblas.h>. |
| `device-printf-global` | 506 | HIP's device printf is the global ::printf; CUDA source uses std::printf. Identical function on the host side. |
| `hipgraph-instantiate-5arg` | 8 | HIP has no 3-argument overload; CUDA's deprecated form is the 5-argument one with no error node and zero flags. |

77 files touched. Rules live in `scripts/rocm/patches/hipify_patches.py`; the forced-include shim for NVIDIA-only intrinsics is `scripts/rocm/patches/hip_nv_intrinsics.h`.

### File patches applied after HIPIFY

| patch | sha256 | status |
| --- | --- | --- |
| `0001-native-qsa-score-portable.patch` | `e372bc6b9a08` | applied |
| `0002-core-amd-arch.patch` | `e315fc9b4a04` | applied |
| `0003-mmq-device-cc.patch` | `baa041824718` | applied |
| `0004-moe-combine-fma.patch` | `6018253fc7b4` | applied |
| `0005-fused-gr-diagnose-guard.patch` | `7cfc86475f2c` | applied |

## Constructs HIPIFY cannot translate

None found.

A row here is an unsupported or unverified path, not a build error: it needs a HIP implementation and an independent test before any claim.
