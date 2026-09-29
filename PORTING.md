# Porting notes: Strata on AMD HIP / gfx1100

Backend design, decisions, unsupported features and numerical policy for the
CUDA → HIP conversion of Strata on this machine.

Target: **Fedora 44, kernel 7.2.7, ROCm 7.1.1 (Fedora RPMs), AMD Radeon RX 7900 XTX
(`gfx1100`)**, Ryzen 9 7900X, 93 GiB RAM.

Everything below is evidence from this machine, not an assumption carried over
from the rollout plan. Where the plan's expectations and the machine disagree,
the machine wins and the disagreement is recorded.

---

## 1. Workspace layout, and two deviations from the plan

The plan's file table prefixes every agent-authored file with `Strata/`. That
would put our scripts, build output and status files *inside* the checkout we
are trying to keep pristine, so that `git status` could no longer distinguish
an upstream change from ours, and so that the port could never be diffed
against the revision it claims. The plan's own opening sentence describes the
rollout directory as containing "this brief and a separate `Strata/` checkout";
we follow that reading.

| Plan path | Actual path | Why |
| --- | --- | --- |
| `<rollout>/Strata-ROCm-Fedora-Rollout-Plan.md` | same | given |
| `Strata/` | `Strata/` | pinned checkout, never written to by us |
| `Strata/strata-rocm` | `strata-rocm` | wrapper, rollout root |
| `Strata/scripts/rocm/` | `scripts/rocm/` | diagnostics and stage scripts |
| `Strata/tests/rocm/` | `tests/rocm/` | independent references and fixtures |
| `Strata/artifacts/rocm/` | `artifacts/rocm/` | logs, manifests, result JSON |
| `Strata/build-hip/` | `build-hip/` | generated HIP tree and build output |
| `Strata/STATUS.md` … | `STATUS.md`, `TASKS.md`, `PORTING.md` | rollout root |

Second deviation: **HIP does not get added to `Strata/CMakeLists.txt`.** That
build hard-rejects any `CMAKE_CUDA_ARCHITECTURES` below 80 and is written for
nvcc; patching it in place would (a) make every upstream merge a conflict and
(b) put the port inside the tree whose revision we quote as evidence. Instead
`scripts/rocm/hip/CMakeLists.txt` builds the **HIPIFY-generated tree** in
`build-hip/hipify/`. Delete `build-hip/` and nothing is lost: it is all
regenerable from the checkout plus `scripts/rocm/`.

## 2. Running GPU work: the sandbox is the first obstacle

The DSH workspace shell runs under `bwrap` with its own minimal `/dev`
(`PID 1: bwrap --ro-bind / / --dev /dev --unshare-pid …`). Observed consequences:

```
$ ls /dev/kfd /dev/dri/renderD128      # inside the sandbox
ls: cannot access '/dev/kfd': No such file or directory
$ ls -l /sys/class/drm/renderD128      # …yet the kernel knows the card
lrwxrwxrwx … renderD128 -> …/0000:03:00.0/drm/renderD128
```

`rocminfo` fails with `Unable to open /dev/kfd`, while `rocm-smi` still reads
the card through sysfs. **This is a namespace property, not a driver fault.**
`./strata-rocm doctor` reports it as `gpu_devices blocked` with the distinction
spelled out, so nobody "fixes" a working driver.

Everything that must touch the GPU — the smoke test, kernel parity tests, the
engine — has to run with the sandbox widened (`danger-full-access` in this
harness). Compilation does not: `hipcc --offload-arch=gfx1100` builds the
smoke test and the kernels inside the sandbox without touching `/dev/kfd`.

## 3. Fedora packaging trap: hipcc does not link the HIP runtime

`hipconfig` reports `ROCM_PATH=/opt/rocm`, which does not exist on this box
(Fedora installs ROCm under `/usr` with libraries in `/usr/lib64`). hipcc
therefore compiles device code successfully and then fails at link:

```
ld.lld: error: undefined symbol: __hipRegisterFatBinary
ld.lld: error: undefined symbol: hipLaunchKernel
clang++: error: linker command failed with exit code 1
```

`libamdhip64.so` is in `/usr/lib64`, which *is* a default link path, so naming
it is the entire fix: **`-lamdhip64`**. Setting `ROCM_PATH=/usr` does *not*
work — hipcc then skips the HIP link step altogether. `HIP_RUNTIME_LIB` in
`scripts/rocm/lib/common.sh` carries this; CMake gets the equivalent through
`hip::host` from `find_package(hip)`.

### 3b. CMake needs three corrections before it will configure at all

`find_package(hip)` and CMake's HIP language modules are all present on Fedora,
but each of these fails in turn, and each failure message points somewhere
unhelpful:

1. **`CMAKE_HIP_COMPILER` must not be `hipcc`.** CMake rejects the wrapper
   outright: *"CMAKE_HIP_COMPILER is set to the hipcc wrapper… Use Clang
   directly."* The ROCm clang driver (`/usr/lib64/rocm/llvm/bin/amdclang++`) is
   named instead — which is also the compiler the ROCm 7.1 device libraries
   match, unlike the Fedora system clang 22.
2. **`find_package(hip)` does not enable the HIP language.** Without
   `enable_language(HIP)` every `.cu` fails at *generate* time with "language HIP
   was requested for compilation but was not enabled".
3. **The ROCm root cannot be discovered.** CMake tries the compiler's
   `-print-targets` output and then `hipconfig --rocmpath`, which answers
   `/opt/rocm` — a directory that does not exist here — so both routes fail with
   "Failed to find HIP root directory". It is derived from where the compiler
   actually lives, and CMake's "ROCm root" is the prefix holding
   `lib64/cmake/hip-lang`, i.e. `/usr`, **not** the `/usr/lib64/rocm` package
   directory.

Also: `.cu` is CUDA's extension and CMake only maps it to HIP when CUDA is
absent, so the kernel sources set `LANGUAGE HIP` explicitly rather than relying
on that.

### 3c. HIP's headers break libstdc++'s `<format>`

Any translation unit that includes a HIP header **before** `<chrono>` fails to
compile, on both g++ 16 and ROCm clang:

```
/usr/include/c++/16/format:4550:30: error: expected an identifier for the
  attribute name [-Wtemplate-body]
    [[__gnu__::__noinline__]]
```

HIP's `host_defines` define `__noinline__` as a macro, and libstdc++ 16 uses that
spelling as an attribute name. `<chrono>` reaches `<format>` through
`bits/chrono_io.h`. The fix is ordering, not editing: `hip_nv_intrinsics.h`
parses `<format>` before including `<hip/hip_runtime.h>`, so the later include
hits its own guard.

## 4. Wavefront width on RDNA3 is 32, not 64

Measured on the card:

```
device_name=AMD Radeon RX 7900 XTX
gcn_arch=gfx1100
warp_size=32
```

HIP reports `warpSize == 32` on gfx1100 because wave32 is the default wavefront
on RDNA. Any kernel or reduction that hard-codes a 64-lane wave — a common
CUDA→HIP porting assumption, and one the rollout plan explicitly warns about
("check … subgroup widths") — is wrong or slow here. This must be checked per
kernel against `__AMDGCN_WAVEFRONT_SIZE` and against an independent reference,
not assumed either way.

## 5. Translation strategy

- **HIPIFY, not hand-porting.** One pinned, vendored `hipify-perl`
  (commit `5038c162dec23ffe39617d0ac07500d3014f791e`, sha256 `bd4fcfd286bc…`,
  MIT) so the translator is fixed by hash rather than by whatever a distro
  ships. `hipify-clang` is not packaged for Fedora.
- **Generated, never forked.** `./strata-rocm hipify` exports a pristine
  `git archive` of the pinned revision into `build-hip/hipify/` and translates
  in place, keeping hipify-perl's `.prehip` files as the before-image. No
  generated file is edited by hand. If a translation needs a fix, it goes in
  `scripts/rocm/patches/` and is applied by the same script, so the change is
  visible and repeatable.
- Headers and host `.cpp` files are translated too, not only `.cu`: 111
  CUDA-dependent files were found by content, and 123 of 260 tracked files were
  rewritten. `HIPIFY_REPORT.md` (regenerated each run) then lists every construct
  HIPIFY cannot translate, with file and stage.
- **Declared patches** run after HIPIFY as data, not as edits to generated files,
  with hit counts in the report:

  | patch | hits | why it is safe |
  | --- | ---: | --- |
  | `shfl-mask-literal-64bit` | 76 | HIP's lane mask is 64-bit and rejects a 32-bit literal with a static_assert; zero extension is the same value on wave32 |
  | `shfl-mask-identifier-64bit` | 3 | the same requirement for a mask held in a `constexpr unsigned` |
  | `device-printf-global` | 504 | device `printf` is the global `::printf` while the CUDA source writes `std::printf`; they are the same function on the host side, so the rewrite is safe anywhere in the file |
  | `hipgraph-instantiate-5arg` | 8 | HIP has no 3-argument `hipGraphInstantiate`; CUDA's deprecated form is the 5-argument one with no error node and zero flags |

- **NVIDIA-only intrinsics** (`__dp4a`, `__vsub4`, `__vcmpne4`, `__vsubss4`,
  `__nanosleep`) have no HIP equivalent and are supplied as *portable reference*
  implementations by `scripts/rocm/patches/hip_nv_intrinsics.h`, forced in with
  `-include` so no generated file is touched. They are deliberately the slow,
  obviously-correct versions; a fast path (`__builtin_amdgcn_sdot4`, byte
  permutes) goes in only once the parity test agrees with the reference.
  `__nanosleep` is the one entry that is not bit-exact, and it is a spin-wait
  backoff rather than arithmetic — said so in the header.
- **Remaining untranslatable inventory: none.** The scanner now strips line
  comments before matching, which matters more than it sounds: it had been
  counting *its own explanations* and leftover comment text as untranslated code.
  With that fixed the report is zero, and the one real gap — the PTX in
  `native_qsa_score.cu` — is closed by patch 0001 (§6b).
- The CUDA graph API itself is **translated** (`cudaGraph*` → `hipGraph*`) and
  therefore no longer appears as a translation gap. It remains a *verification*
  risk, not a translation one: HIP graph capture semantics are not assumed equal
  to CUDA's, and that is tracked at S5.2 rather than being quietly dropped from
  this list.

## 6. Build definition

`scripts/rocm/hip/CMakeLists.txt`, consumed by `./strata-rocm build`.

- `find_package(hip)` + CMake's native HIP language support; `-DCMAKE_HIP_ARCHITECTURES=gfx1100`.
- **The build refuses anything but gfx1100.** Widening it is a decision with
  evidence behind it, not a convenience flag — a binary that runs and is wrong
  about which GPU it was validated on is the worst outcome available.
- `ninja -k 0`: one build reports the whole failure inventory rather than
  hiding forty errors behind the first. That is what makes the coverage claim
  in the report checkable.
- Timeout is separate from compiler failure (`EX_TIMEOUT=124` vs `EX_FAIL=1`)
  and is reported separately, per section 7 of the plan.
- Floating-point contraction is pinned to `off` for HIP translation units, with
  the measurement behind that decision in section 7 below.
- Slice 1 covers the device kernels and their in-tree parity tests: 43 kernel
  translation units, minus one explicit exclusion, plus a small host support
  library (`ngram.cpp`, `ple_reader.cpp`, `direct_file.cpp`, `memory.cpp`) and 21
  parity executables. `src/core`, `src/prefill` and the engine binary are later
  slices, each its own target, so one broken group cannot hide the state of the
  others.
- **Exclusions are data, not omissions.** `STRATA_HIP_UNSUPPORTED` is empty now
  that the PTX file is ported, but the mechanism stays: an excluded translation
  unit is printed by the build and written to `unsupported-sources.txt`, which
  the test runner copies into the coverage section of every result file. While
  `native_qsa_score.cu` was excluded it was listed there rather than quietly
  dropped from the source glob.

## 6b. The QSA scorer: replacing tensor-core PTX with a reference path

`src/kernels/cuda/native_qsa_score.cu` was the one translation unit that
HIPIFY could not translate and that the first build excluded. It hands raw F32
register bits to `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` through an
`ldmatrix`-loaded fragment layout. **AMD matrix intrinsics do not accept NVIDIA
fragment layouts**, and the plan requires a correct reference path before any
optimisation, so the file is replaced wholesale by
`scripts/rocm/patches/0001-native-qsa-score-portable.patch` — a reviewable
unified diff applied by `hipify.sh` after HIPIFY, to the generated tree only.
Patch application is recorded with the patch's sha256 in `HIPIFY_REPORT.md`;
editing a patch regenerates the tree, because the stamp covers the patch set.

The replacement keeps the documented contract from
`include/strata/kernels/native_qsa_score.hpp` exactly: tf32 operand truncation
(what the tensor core sees when given raw F32 bits), F32 accumulation, per-head
ReLU, ordered head addition, the optional F32 block bias, the finite `1e9` bias
on the incomplete tail block, the materialised `+0` on full blocks, the write
range `[0, n_kv)`, the spare-row tolerance and the invalid-count early exit.

It deliberately differs in one respect, and the difference is stated rather than
hidden: the reduction is a single sequential F32 sum over `d`, where the original
splits `d` across two warps and each tensor-core step sums an 8-wide slice
internally. **Bitwise agreement with NVIDIA is therefore not claimed** — the plan
forbids promising it — and the measured deviation is reported instead.

`tests/rocm/qsa_score_parity.cpp` is the independent check, written from the
documented contract rather than from the kernel body. It asserts four separate
things, and the separation matters:

1. **Bit-exact agreement with a host model of the same arithmetic** (same
   truncation, same order). This is what catches an indexing, masking or
   ordering mistake that any tolerance would hide.
2. **Deviation from a float64 ideal stays inside a bound derived per row from
   the data**, not a chosen tolerance: `sum_d |p·q| · 2·2⁻¹¹ + D·2⁻²⁴·sum_d |p·q|`
   plus the `1e9` rounding term where the tail bias applies.
3. **Nothing is written at or beyond `n_kv`** (sentinel-checked).
4. **A step buffer that fails the contract suppresses every write.**

Result on `gfx1100`: 7 cases (full blocks, with and without bias, incomplete
tail, single cell, short blocks, and the invalid-step guard), **0 failures**,
bit-exact against the host model in every case, no writes outside the range, and
a worst deviation of **1.1e-4 of the row's own magnitude** — comfortably inside
the 9.8e-4 the tf32 truncation allows.

One honest note on the test's own history: an earlier revision also asserted a
*relative* error against the result. That is unsatisfiable wherever ReLU plus
cancellation lands the score near zero while the summands are O(1), and it
failed for exactly that reason. It was replaced by the derived bound and a
reported `deviation/scale` statistic, not deleted to make the suite green.

## 6c. The pinned llama.cpp dependency, and scoping the intrinsic shim

`src/prefill/{kernels,moe_mmq}.cu` are derived from ggml-cuda and include its
headers (`common.cuh`, `mmq.cuh`, `quantize.cuh`); `src/kernels/cpu/native_expert.cpp`
includes `ggml-cpu.h`. Upstream does not vendor them -- `setup.py` downloads
llama.cpp at exactly `3cf03257f219afbe7334045ff7c6a06ac68c627d`. `prepare` now
fetches the same archive at the same pin, verifies and records its sha256, and
extracts it into a regenerable directory. MIT, 39.6 MB, well inside the 2 GB
unapproved-download budget.

Reusing those headers rather than rewriting ggml's MMQ path is not laziness for
its own sake: they are **HIP-aware**. `vendors/hip.h` maps the cuBLAS names onto
hipBLAS and implements `__vsub4`/`__vsubss4`/`__vcmpne4` for AMD, and ggml's own
`dp4a` has a HIP branch using `__builtin_amdgcn_sudot4`. Three things were found
the hard way:

- **Their intrinsics are not CUDA's.** ggml's HIP `__vsub4` is implemented as
  `__vsubss4` -- a *saturating signed* byte subtract, where CUDA's `__vsub4`
  wraps. They disagree only where a byte saturates, which is exactly the corner a
  quantization port trips over. The Strata kernels keep the wrapping reference
  implementation in `hip_nv_intrinsics.h`; the ggml-based units deliberately use
  ggml's, because defining both makes every call ambiguous. **This is a known,
  unresolved divergence between the two halves of the tree** and a candidate root
  cause for any future IQ-format mismatch. It is recorded here rather than
  papered over by picking whichever one makes a test pass.
- **A compatibility shim must be scoped, not global.** Force-including the
  intrinsic shim everywhere collided with `vendors/hip.h`. It is now split into
  `hip_preamble.h` (header ordering only, no symbols, safe for every HIP unit) and
  `hip_nv_intrinsics.h` (symbols, Strata kernel units only), and it links
  `PRIVATE` into `strata_hip_kernels` -- a `PUBLIC` link propagates the same
  collision to every consumer.
- **Two integration points.** HIPIFY rewrites `<cublas_v2.h>` to `<hipblas.h>`,
  but the header on this install is `<hipblas/hipblas.h>`; a declared patch rule
  now fixes that. And `common.cuh` selects `vendors/hip.h` only under
  `GGML_USE_HIP`, without which it pulls in `<cuda_fp16.h>` and fails.

## 6d. The architecture gate, and what HIP graph semantics actually are

**The gate.** `src/core/device.cu` refused to run unless `cc_major == 12`. On this
card HIP reports capability **11.0**, so the upstream check throws on the correct
hardware. Patch `0002-core-amd-arch.patch` replaces it with a list of targets this
port has been *tested* on -- currently `gfx1100`, matched as a prefix of
`gcnArchName` so feature suffixes like `gfx1100:sramecc-:xnack-` still match:

```cpp
d.arch = p.gcnArchName;
const bool arch_validated = d.arch.compare(0, 7, "gfx1100") == 0;
if (!arch_validated) throw CudaError(... names the architecture ...);
```

What it deliberately is **not**: a relaxed comparison that accepts everything, or
a widened constant. Spoofing the architecture is forbidden by the plan, and a
binary that runs on an unvalidated card and is wrong about which card it was
measured on is the worst available outcome. Widening the list is a decision with
a passing suite behind it. `DeviceInfo` gained an `arch` field so the architecture
is reported rather than inferred; `strata-device` now prints it.

**Graph semantics.** HIPIFY renames `cudaGraph*` to `hipGraph*` and reports no
untranslated construct -- which says nothing about whether HIP *behaves* like CUDA.
The plan's Stage 5 warning ("do not assume CUDA graph behavior translates
identically to HIP") is therefore a measurement, not a formality. Measured on
gfx1100 by `tests/rocm/core_graph_parity.cpp`, against the three behaviours
`include/strata/core/graph.hpp` bakes in from CUDA measurements:

| assumption | CUDA (upstream) | gfx1100 (measured here) |
| --- | --- | --- |
| capture → instantiate → launch | works | **works**, 1 node, result correct |
| a replay re-reads its input buffers, but not its kernel arguments | measured | **re-reads buffers**: 3 replays with new data in the same address, 0 stale values. This is the one that would silently corrupt an engine, and it holds |
| the registry captures a key once; the body does not re-run | asserted | **asserted and holds**: 4 `record` calls, 1 body run, 1 graph, replay computes on the newest data |
| a capture with zero nodes is an error, not an empty graph | asserted | **holds** |

Still unmeasured on HIP, and therefore still assumed at the engine's peril: the
per-step-pointer hazard (anything that changes per token must be *data*, never a
kernel argument) and replay behaviour under concurrent streams. Both stay open in
TASKS S5.2.

## 6e. What the engine build needed, and what it does not prove

The `strata` binary links the whole stack (core, kernels, engine, prefill, spec).
Three things it needed that are worth recording, because each is a place a
future agent will otherwise re-derive:

- **`-DSTRATA_VERSION="0.1.13"` and a global `-D__HIP_PLATFORM_AMD__=1`.** Upstream
  sets the first in `project()`; the second is what hipcc defines for HIP-language
  units but *not* for plain C++ units that include a `<hip/...>` header, which
  otherwise `#error` out.
- **The AVX-512 flags stay on the SOURCE, not the target.** Upstream is emphatic
  about this and the reason is sound: a translation unit compiled with
  `-mavx512f` may use AVX-512 anywhere in it, so putting the flag on the library
  would silently make the scalar fallback the same bet with worse opt-out.
- **Native (IQ) experts are an upstream OPTION, and this build currently has it
  off.** `STRATA_NATIVE_EXPERTS=ON` adds `native_expert.cpp`, `iq_avx512.cpp` and
  ggml's CPU backend (`ggml-cpu`, `ggml-base`) built from the pinned llama.cpp.
  With it off the engine still builds and links, and the CPU half of an expert
  layer is simply unavailable. That is recorded as an open task (S2.7c), not
  presented as a working CPU path.

**What linking proves: nothing about behaviour.** The binary was run on the card;
it parsed its arguments and refused cleanly at the first missing artifact
(`cannot open <pack>/index.txt`, exit 1). That is evidence the startup path is
intact. It is not a result, no token was produced, and this document does not
treat it as one.

One more thing the link hid: upstream compiles eight explicit MMQ template
instances (`ggml-cuda/template-instances/mmq-instance-{q2_0,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs}.cu`)
and this build does not. The MMQ wrapper links and its symbols are present, but
whether every quantization type dispatches to a real kernel is untested (S2.7d).
"A library linked" is exactly the kind of evidence that must not be upgraded into
"the path works".

## 6f. Native CPU experts, and a gap that turned out not to be one

**Native experts.** Upstream's `STRATA_NATIVE_EXPERTS` (`ON` by default) adds
`native_expert.cpp`, `iq_avx512.cpp` and ggml's CPU backend, so that an IQ expert
weight means exactly what it means in llama.cpp rather than what a reimplementation
thinks it means. It is now on, built through upstream's own mechanism —
`add_subdirectory` on the pinned llama.cpp's `ggml`, linking `ggml-cpu` and
`ggml-base`. The GPU backends are never built: this is a CPU dot-product library,
not a second inference engine.

The AVX-512 flags stay on the *source* (`iq_avx512.cpp`, `expert.cpp`), not the
target, exactly as upstream does and for upstream's reason: a TU compiled with
`-mavx512f` may use AVX-512 anywhere in it, so a target-level flag would make the
scalar fallback the same bet with worse opt-out. `native_experts_available()` is
the runtime gate.

Ordering trap, recorded because it failed silently: the ggml block originally sat
*before* the block that computes `STRATA_HAVE_LLAMA`, so the condition was false,
the option reported itself disabled, and the build simply proceeded without
ggml — cache said `STRATA_NATIVE_EXPERTS:BOOL=ON` while no `libggml*` existed. It
was caught by checking for the libraries rather than by trusting the option.

**A gap that was not a gap.** The previous checkpoint recorded that upstream
compiles eight explicit MMQ template instances
(`ggml-cuda/template-instances/mmq-instance-*.cu`) and we do not, and called that
a possible correctness hole. Inspection answers it: `moe_mmq.cu` calls
`mul_mat_q_case<GGML_TYPE_Q2_0>(...)` and seven siblings with **explicit template
arguments**, so each template is implicitly instantiated in that translation unit.
Upstream's instance files exist to split an enormous template across eight TUs for
compile time. The lesson is worth more than the fix: "we skipped eight files" was
the right thing to *check* and the wrong thing to *assume*, in either direction.
The MMQ path still has no runtime verification (S3.2b) — it links and dispatches,
and that is all that can be said.

**A harness bug the new test exposed.** `native_expert_parity` takes a GGUF shard
as `argv[1]`; the test runner passes `--selftest` to everything, so the flag was
read as a filename and the binary aborted. The runner now supports per-test argv
via `tests/rocm/testargs.json`. Without that, a test that builds correctly would
have been reported as a failure — or worse, someone would have "fixed" it by
removing it from the suite.

## 6g. The MMQ prompt path on AMD: a defect in the shim, found by a test

`src/prefill/moe_mmq.cu` routes prompt-time expert matrices through llama.cpp's
MMQ kernels. It linked, it dispatched, and it had never been executed. The
independent test built for it (`tests/rocm/prefill_mmq_parity.cpp`) failed three
times in three different ways, and the third failure was a real port defect.

**What the test does.** Q2_0 weight blocks are generated here and decoded here
from the documented layout (`{fp16 d; uint8_t qs[16]}` over 64 values, value =
`(code - 1) * d`, low bits first), so the multiply-accumulate is independent of
the kernel's. The activations are chosen so that quantizing them is *exact* —
integer multiples of 2^-5 with a +/-127 multiple in every block, so
`d = amax/127 = 2^-5` exactly and `round(x/d)` recovers the integer. That is what
lets the reference use the original floats without decoding ggml's 128-value
transposed, padded q8_1 layout — avoiding a test that fails because the *test* is
wrong. The assumption is stated first in the test's output for that reason.

**Failure 1 — the test's own bug.** `quantize()` takes a **device** pointer for
the activations (`x`); the test passed host memory, and the GPU faulted on a host
address. Upstream passes its device buffer `m.H`. Fixed in the test.

**Failure 2 — a conclusion from the previous checkpoint that was wrong.** The
missing `mmq-instance-*.cu` files were recorded as *"resolved by inspection: the
templates are instantiated at the call site"*. They are not. `mul_mat_q_case<T>`
is only **declared** in `mmq.cuh`, its definitions live in
`template-instances/`, and `quantize_mmq_q8_1_cuda` lives in `quantize.cu`;
neither was compiled. The test said so the only way a missing symbol can:
it failed to **link**. All nine upstream translation units are now built as
`strata_hip_mmq`. The lesson is in TASKS: *an inspection that concludes "not
needed" needs a test that would have failed without it.*

**Failure 3 — the real defect.** With the symbols present, every launch aborted:

```
ERROR: HIP kernel mul_mat_q has no device code compatible with HIP arch 1300.
```

`1300` is a red herring — it comes from `__CUDA_ARCH__` inside ggml's
`NO_DEVICE_CODE`. The real cause is one line in Strata's own
`src/prefill/ggml_cuda_host.cu`, which supplies the device info ggml's MMQ
kernels read:

```cpp
d.cc = 100 * prop.major + 10 * prop.minor;   // CUDA's encoding -> 1110 for gfx1100
```

ggml keys its MMQ **instance selection** on `cc`. For AMD it does not use
major/minor at all; `ggml-cuda.cu` parses the GCN architecture string:

```cpp
info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);   // gfx1100 -> 0x1000000 + 0x1100
```

With `cc = 1110` no MMQ configuration matches, the template's config table yields
`GGML_TYPE_COUNT`, and the kernel takes the `NO_DEVICE_CODE` path. **The whole
prompt MMQ path was therefore unusable on AMD**, and nothing in the build could
have shown it — it links, it dispatches, and it aborts only when launched.
Patch `0003-mmq-device-cc.patch` copies ggml's parser verbatim into the shim.

**Result.** Three model-shaped cases (gate+up 1280x512, down 256x640, single row),
**0 failures, and bit-exact** — `worst|err| = 0.000e+00` against the float64
reference, not merely inside the derived bound. That is the expected outcome when
the activations quantize exactly and the arithmetic is integer products plus one
fp32 block-scale multiply, and it is a much stronger statement than "close
enough": block layout, per-block scales, q8_1 quantization, type dispatch,
accumulation and destination indexing all agree exactly.

**What it does not cover.** Only Q2_0. The seven i-quant types the path claims
report `supported()` but have no fixture here, and the model's own weights are
IQ3_S. That is the largest remaining gap on the feasibility-gate path, and it is
listed as such rather than implied by "the MMQ test passes".

## 6h. MMQ coverage: six of eight types, and why not eight

`tests/rocm/prefill_mmq_parity.cpp` now covers every type the MMQ path claims.

| type | result | note |
| --- | --- | --- |
| `Q2_0` | **bit-exact** | and a hand-written decoder agrees with ggml's on 256/256 values |
| `IQ3_S` | 0 failures, worst err 1.6e-05, **0.001 of the bound** | the model's own weights |
| `IQ2_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS` | 0 failures, <=2.1e-05, <=0.001 of the bound | |
| `IQ2_XXS`, `IQ2_XS` | **not buildable here** | `ggml_type_traits::from_float_ref` is null for both |

**The oracle.** Weights are generated and decoded with ggml's own
`from_float_ref` / `to_float`: the scalar definition of each format, against which
the GPU kernel is an independent implementation. For Q2_0 the decoder is *also*
written out by hand in the test, and the two agree exactly - so the oracle itself
is not taken on trust where avoiding that is cheap.

**Why the i-quants are not bit-exact and Q2_0 is.** The activations quantize
exactly in both cases, so the residual difference comes from the weight side: the
kernel computes each dequantized weight in-kernel, while the reference uses
ggml's scalar expression. For Q2_0 the expression is literally `(code - 1) * d`
and the two agree bit for bit; for the i-quants the codebook values are built from
grid tables and the two evaluation orders can differ in the last fp32 bit. The
observed 1e-5 is that rounding, and it sits 1000x inside the derived summation
bound. Reporting it as "bit-exact" would be false; reporting it as "close" without
the bound would be unquantified. Both numbers are in the result file.

**Two honest limits.** `IQ2_XXS` and `IQ2_XS` cannot be exercised through ggml's
public reference API at all, because their quantizers take an importance matrix
that comes from calibration data. They stay unverified in the *MMQ* path (S3.2b'), which is a
different statement from "they pass". Their **dequantization and MMVQ** are
covered by `iq_parity`, which now runs against generated fixtures (§6i).

**A small trap worth recording.** ggml's i-quant quantizers abort with
*"forgot to call ggml_quantize_init()?"* unless that is called first - the
codebook maps are built lazily. A test that builds fixtures from ggml must call it
per type.

## 6i. The IQ fixtures: how a skipped test became ten verified formats

`src/kernels/iq_parity.cpp` covers ten quantization formats and was skipped
entirely, because it reads fixtures from a directory and upstream publishes no
generator (its `tools/iq_fixture.py` was in the omitted tree). "Skipped" was the
honest state; it is no longer the current one.

**The format is simple** - `<NAME>.bin` is `{type, rows, cols}` followed by raw
blocks, `<NAME>.f32` is the reference decode - so the fixtures can be built.

**The split, and why it is that way.** gguf-py implements *dequantization* for
nine of the ten types and *quantization* for none: the i-quant encoders only
exist in C. So capability decides the division of labour:

| half | produced by | why |
| --- | --- | --- |
| blocks | ggml's own encoders (`quantize_iq2_xxs`, `quantize_iq3_s`, …) | nothing else can encode an i-quant; gguf-py has no encoder |
| reference values | **gguf-py** (`gguf.quants.dequantize`) | a separate implementation, in a different language, from the C kernels under test |

That is exactly what the plan asks for. The importance matrix handed to the
encoders is uniform (all ones) and that is deliberate rather than a shortcut: the
fixture's job is to carry valid blocks and their exact decoded values, not to
reproduce a particular model's quantization quality - an imatrix only steers which
values are given more precision.

**Result.** `dequant rel 0.00e+00` for all ten types: the ported dequantizers are
**bit-exact** against gguf-py, which is a stronger statement than the 1e-6 the
test demands. The quantized mat-vec lands at 4.3e-3 to 6.9e-3 against the test's
own 2e-2 bound, consistent with q8_1 activation rounding.

**A cross-check that fell out of it.** Where both decoders exist, their values are
compared before the Python one replaces the C one, and they agree **exactly** on
all nine types gguf-py implements. Two independent implementations of the same
formats agreeing bit-for-bit is worth more than either one alone. `Q2_0` is the
exception - gguf-py raises `NotImplementedError` for it - so its `.f32` keeps
ggml's C values, and it is separately cross-checked against a decoder written out
by hand in `prefill_mmq_parity.cpp` (256/256 values identical). The fixture
script prints which reference produced each file, so provenance is stated rather
than assumed.

**Reproducibility.** `./strata-rocm fixtures` builds them idempotently, and the
test stage calls it, so the suite works from a bare checkout instead of depending
on a directory somebody once created. If gguf-py or the venv is missing the script
says so explicitly and warns that the reference is then *not* independent of the
kernels - a weaker fixture that announces itself is better than one that quietly
is.

## 6j. Scoping "the remaining forward-pass operations", and two of them closed

Stage 3 asks for "custom attention, recurrent/stateful ops and all remaining
kernels required for a forward pass". Rather than guess at that list, every kernel
entry point called from `src/core/layer.cpp`, `session.cpp`, `native_head.cpp`,
`native_dense.cpp` and `mtp.cpp` was cross-referenced against every test in
`tests/rocm/` and `src/kernels/*_parity.cpp`. Around forty kernel files have no
test anywhere, which is unsurprising: upstream's `native_*` parity tests lived in
the omitted `bench/micro` tree and compared against a *CUDA* oracle build, which
cannot run here at all. The plan anticipated this - "if the original parity suite
is unavailable, create meaningful independent tests before claiming parity".

Two of the highest-value gaps are now closed.

**Routing (`native_router_parity`).** The plan names routing explicitly and calls
out tie-breaking, and the kernel has a real rule for it: equal computed
probabilities select the **lower expert index**, implemented as an explicit
comparison in the shuffle reduction. Six cases cover ordinary logits, all-equal
logits, a tie inside the top ten, a tie exactly at the 10th/11th boundary (the one
that decides which expert actually runs), a one-ulp near tie under a -900 offset
that a router skipping its max-subtraction would get wrong, and +/-88 extremes.
All ids match a float64 reference using the same rule; weights sum to 1 within
1e-4 and sit at 0.001 of a bound derived from the 512-expert reduction length.

*One correction worth recording.* An earlier revision asserted that an id mismatch
was only meaningful when the raw-logit margin exceeded 1e-6 - which is backwards,
and made the two exact-tie cases fail. An exact tie is precisely where the
documented rule makes the answer **deterministic**. The escape hatch was removed
rather than loosened, because it encoded a misunderstanding, not a tolerance.

**The q8_1 activation quantizer (`native_mmvq_quant_parity`).** Every quantized
mat-vec in the native path feeds on this. The contract has a subtle clause -
"Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
not reconstruct that sum from the quantized integers" - so the test includes a
block chosen such that the two differ, and asserts the former. An implementation
that reconstructed the sum would pass every other check here and fail only that
one.

*And a second correction.* The first revision compared the stored scale against
the fp32 `amax/127` and failed every non-zero block. The scale is stored as
**fp16**, so what comes back is its fp16 rounding; the header states the matching
precondition ("their block scales/sums representable in FP16"), so the test now
checks the fp16 rounding in general and the exact case under a conforming input.
The kernel was right and the test was wrong - worth saying plainly in a project
where the opposite is the more common failure.

## 6k. Attention, and a contract gap worth knowing about

`native_flash_attn_short_step` is on the critical path for every token and had no
test. `tests/rocm/native_flash_attn_parity.cpp` supplies one from the contract in
the header: Q24x256, KV2x256, scale 1/16, additive mask broadcast over heads, and
the documented query-to-KV map `head / 12`.

Seven cases, all passing, worst deviation 0.001-0.061 of a bound derived from the
reduction length:

| case | what it pins |
| --- | --- |
| ordinary, no mask | the base path against a float64 reference |
| additive mask | the mask is *added* to the scaled dot, per cell, broadcast over heads |
| single key | softmax over one element is 1, so the output must be `v[0]` - and is bit-exact |
| full context (256) | the largest supported step |
| **padding poisoned with NaN** (with and without a mask) | the contract says unused k/v rows and mask entries "may contain arbitrary bytes" and are synthesized rather than read. NaNs sit in exactly those bytes and the result is unchanged |
| invalid step | `status = UnsupportedStep` and every one of the 6144 outputs NaN |

The heads are deliberately made distinguishable (the two KV heads carry different
offsets) so that the documented `head / 12` map cannot coincidentally agree with a
wrong `head % 2` - a class of error that a symmetric fixture would hide.

**A gap in the header's contract.** The header documents the geometry, but the
implementation additionally validates `shapes.idx_block == 4` and
`shapes.idx_top_k >= 256`. A caller working from the header alone - which is what
a contract is *for* - gets:

```
std::invalid_argument: native FlashAttention supports only Q24x256/KV2x256,
  capacity>=256 and context1..256 on an explicit stream
```

which names none of the fields that actually failed. Our test hit exactly this.
Nothing is patched upstream for it; the header is upstream's to fix. It is
recorded because it costs a future caller an hour, and because a validation
message that lists the conditions it does *not* check is worse than no message.

## 6l. The GDN recurrence, and two details the header leaves out

`native_gdn_step` is the stateful core of the recurrent layers. Its parity test
was in the omitted `bench/micro` tree, against llama.cpp's CUDA oracle.
`src/kernels/gdn_parity.cpp` does test the *non-native* path against a reference,
but deliberately in a **different state layout** (`(S, S, h_v)`) to catch layout
mix-ups, so it cannot simply be reused; `tests/rocm/native_gdn_parity.cpp`
implements the recurrence in float64 against the layout the native header
documents, `state[(i*h_v + head)*S + j]`.

**Why the multi-step case is the one that matters.** A recurrence feeds its own
output back. An error at step 1 is not a one-off - it is the input to step 2. A
single-step test therefore cannot distinguish a correct recurrence from one that
mishandles the feedback, and both would look right. The central case here runs
**eight consecutive steps over one state buffer**, with the reference iterating the
same eight, and there is a twelve-step case under a decaying gate as well. Worst
relative error 9.4e-07, at 0.003 of a bound derived from the reduction length.

**Two details the header does not state.** The header gives the layout, the
shapes, the "q must not be pre-scaled" rule and the scale position, but not the
order of the decay against the delta update, nor which state the readout reads.
Both were resolved from the implementation:

```
kv    = sum_i state_old[i][head][col] * k[q_head*S + i]     # UN-decayed state
delta = (v[head*S+col] - exp(gate)*kv) * beta
state_new[i] = exp(gate)*state_old[i] + k[q_head*S+i]*delta
out   = (sum_i state_new[i] * q[q_head*S+i]) * (1/sqrt(S))  # UPDATED state
```

That is algebraically the "decay first" reading, with a different rounding path.
The test asserts the algebra rather than the arithmetic order, so an upstream
change to either detail fails the test instead of silently agreeing with something
else. **Recorded as resolved-from-implementation, not as documented** - a future
reader should know which of the two they are relying on.

The query-to-KV head map is `head % h_k` here, which is *not* the `head / 12`
mapping the attention adapter uses. Both are exercised: the head counts are chosen
so `h_k == h_v` (identity) and `h_k < h_v` (a real reduction) both appear.

## 6m. The GDN preprocessing helpers, and where an epsilon asymmetry actually bites

Five small helpers feed the recurrence: `conv_silu`, `l2_norm`, `out_norm`,
`beta_gate`, `gate`. All five now have an independent float64 reference
(`tests/rocm/native_gdn_preprocess_parity.cpp`), worst relative error 1.7e-07, and
the conv history shift is exact.

**The interesting part is an asymmetry the header states only in passing.**
`native_gdn_l2_norm` is documented as `scale(rms_norm(x, epsilon/128), 1/sqrt(128))`
- the epsilon *divided* by 128. `native_gdn_out_norm` is documented as
`rms_norm(output, epsilon) * gamma * sigmoid(z)` - epsilon as given. Two
normalizations that read almost identically in the header are not the same, and a
port that unified them would be wrong in a way no output inspection would catch.

The first attempt to test this asserted "the two must differ measurably" over
ordinary random rows. On those rows the mean square is ~1.3 and the epsilon term
is ~1e-6, so the two agree to **4e-07** - the assertion failed because it was
vacuous, not because anything was wrong. The regime where the distinction means
anything is a **small-magnitude row**, where epsilon dominates the mean square; the
test now runs one, and there the two differ by **90%**. The assertion was replaced
by a case that exercises the difference rather than by a looser assertion.

The same principle is applied to the softplus threshold of 20 in `native_gdn_gate`:
the fixture asserts that 12 of its 96 heads actually cross it, so the `x` branch is
demonstrably taken rather than assumed to be.

**And a test bug worth recording**, because its symptom looked like a kernel error.
The first revision referenced `out_norm` against the *original* input while the
device received the buffer `l2_norm` had already normalized in place. The residual
was 6.3e-05 - a hundred times fp32 rounding, and *just* over the derived bound,
which is exactly the signature of a real bug. It was not one: on data of ordinary
magnitude RMS normalization is scale-invariant, and the whole gap was the epsilon
term applied to two different magnitudes. Fixing the reference dropped
`out_norm` to 1.7e-07. Worth remembering when a bound is missed by 4%: check what
the device was actually handed before doubting the kernel.

## 6n. When a project-wide flag silently rewrote a kernel's contract

Section 7 records the decision to build every HIP translation unit with
`-ffp-contract=off`, made because the in-tree parity suite writes its references
with a `volatile` temporary that blocks fusion and then asserts bit-equality with
the unfused result. That decision was right *for those kernels*. This is what it
did to a different one.

`include/strata/kernels/native_moe.hpp` contracts:

> The first product rounds to F32, following products accumulate with FMA in expert
> order, and optional shared is added once afterward.

That is a statement about rounding, not about mathematics, and `-ffp-contract=off`
made it false: `sum += parts[...] * weights[expert]` compiled to a separately
rounded multiply and add. Measured on gfx1100, **35 to 280 of 256 to 512 elements**
differed from the contracted reference, and the output matched the
separately-rounded order bit for bit.

**Why no value-based test could have found it.** Both accumulation orders land
within **2.7e-07** of the float64 ideal - the difference between them is in the
last bit. A test comparing against a double reference would have passed either
way. `tests/rocm/native_moe_combine_parity.cpp` therefore computes **two exact fp32
simulations** - one contracted, one separately rounded - and reports which the
device matches:

| case | before patch 0004 | after |
| --- | --- | --- |
| k=2 | vs fused 60, vs rounded **0** | vs fused **0**, vs rounded 60 |
| k=8, shared | vs fused 227, vs rounded **0** | vs fused **0**, vs rounded 227 |
| k=15, shared | vs fused 280, vs rounded **0** | vs fused **0**, vs rounded 280 |

**The fix** (`0004-moe-combine-fma.patch`) names `__fmaf_rn` explicitly rather
than relying on the compiler. That is strictly more robust than depending on a
contraction default, which differs between nvcc and clang in the first place -
so on CUDA this kernel got FMA by default while here it did not, from the same
source line.

**The general lesson, which is why this is a section and not a footnote.** A
project-wide compile flag is an invisible edit to every kernel that has an opinion
about arithmetic. The rollout now knows of two kernels with such an opinion:
`native_moe` (confirmed, fixed, tested) and **`native_gr_postops`** (its header
contracts "ordered FMA accumulation"; no test covers it yet, so it is recorded as
a *suspect* rather than as either working or broken).

## 6o. The FMA suspect, cleared - and a test construction that defeated itself

Section 6n recorded `native_gr_postops` as a **suspect**: its header contracts
"ordered FMA accumulation" for the layer path, this build sets
`-ffp-contract=off` project-wide, and a sibling (`native_moe`) had already been
caught silently losing its documented fusion to that flag.

**The suspect is innocent**, and the reason is the interesting part.
`native_gr_postops.cu` writes the fusion explicitly:

```cpp
if constexpr (Fused) sum = __fmaf_rn(x, w, sum);
else                sum = c == 0 ? product : __fadd_rn(sum, product);
```

`native_moe.cu` wrote `sum += parts[...] * weights[expert]` and let the compiler
decide. **Same contract, two different amounts of care** - and only the second one
lost its arithmetic to a project-wide flag. Naming the intrinsic is what makes a
numerical contract survive a build setting you did not choose.

**How the fusion was confirmed without trusting either implementation.** Run both
documented paths on the same random inputs. A contracting fused path and a
separately-rounded head path disagree on most elements - measured here, **125 of
256**. If the fusion had been lost, the fused path would compute exactly what the
head path computes and the two outputs would be *identical*. That check needs no
tolerance, and unlike a bit-exact comparison it is insensitive to host/device
`expf` differences.

**A construction that defeated itself.** The first attempt set `gate = 0`, so that
`sigmoid(0) = 1/2` is exact in any libm, hoping for a bit-exact comparison of the
accumulation *order*. It does give bit-exactness - and it destroys the
discrimination, because multiplying by 1/2 is exact, so `fma(x, w, sum)` and
`sum + (x*w)` become the **same operation**. The assertion "the two orders must
differ" failed for that reason, not because anything was wrong. The test now uses
two constructions, each asserted only for what it can actually establish:

| construction | what it establishes |
| --- | --- |
| `gate = 0` (all operations exact) | each path matches its own simulated order **bit for bit** |
| random gates, both paths on the same inputs | the orders **differ**, so the fusion is real |

This is the second time a fixture has been at fault rather than a kernel (§6m had
the other). Worth stating plainly: in a port like this the *test* is as likely to
be wrong as the code, and a failing assertion is a question, not a verdict.

## 6p. Two QSA sub-ops, and two details a plausible port could get wrong

`native_qsa_rms_norm_weighted` and `native_qsa_gate_apply` run per layer per token.
Neither had a test. Both are small; both have a detail that produces well-formed,
quietly wrong output if it is missed.

**The reduction has two branches.** `n_cols < 1024` launches a 256-thread kernel
and `n_cols >= 1024` a 1024-thread one - different block sizes, different shared
memory staging, different tree shapes. A bug in one branch is invisible to a test
of the other, so both are exercised: measured relative error **7.13e-08** below the
threshold and **7.16e-08** above it, against 1e-06 justified by `rsqrtf`'s ulp plus
the reduction depth.

**The gate lives in the second half.** `native_qsa_gate_apply` reads
`q_full[head*2*head_dim + head_dim + channel]` - the second half of each row. A
first-half read is not a crash and not an obviously wrong shape; it is the same
computation with the wrong multiplicand, so the output looks entirely reasonable
and is wrong everywhere. The fixture therefore **poisons the first half** with
`+12` while the real gate half holds `-12`: the two sigmoids are ~1 versus ~6e-6,
so a first-half read fails by ~1 in absolute terms rather than by a rounding. That
construction is what makes the layout assertion mean something.

Both properties fall out of the same habit used throughout this rollout: make the
fixture able to *distinguish* the correct implementation from a specific plausible
wrong one, rather than only able to agree with the correct one.

**One layout note.** The header writes the shape as `[n_cols, n_rows]`, which reads
as column-major, while the kernel indexes `blockIdx.x * n_cols + col` - row-major,
`n_cols` being both the row width and the reduction axis. This is the second time
this notation has read the opposite way from the code (`pooled` was the first); the
test follows the kernel and records the discrepancy rather than silently picking
one.

## 6q. RoPE, and a tolerance that was justified wrongly

`native_rope_apply` runs on every token of every layer, and its parity test was in
the omitted `bench/micro` tree. Six fixtures now cover it: both head widths
(128/256), two frequency bases, in-place and out-of-place, with and without the
IMRoPE table, plus a pair-0 anchor.

**Three of the checks are not tolerances**, which is what makes them worth
trusting:

| check | why it is not a tolerance |
| --- | --- |
| the copy region (channels from `n_rot` up) | asserted **byte-exact**; it is a copy, and a kernel that rotated the whole row would produce plausible-looking garbage there |
| the rotation itself | asserted **as an isometry** per pair: `|out| = |in|` to 2 ulps. A wrong pairing breaks it immediately, and unlike a component comparison it does not depend on the trig library matching |
| pair 0 | `powf(theta_scale, 0)` is exactly 1 in any implementation, so `theta == position` and the angle itself is pinned rather than compared |

**The tolerance that was wrong.** The first revision compared component-wise
against a float64 reference with a flat relative tolerance and failed by 10-25x -
which looks exactly like a broken kernel. It was not. The isometry held to
**1.6e-07** in every fixture, and the copy region was byte-perfect, so the rotation
was structurally right.

What the flat tolerance missed is where the error actually comes from. The angle is
computed in fp32:

```cpp
const float theta = mrope_pos(mtab, positions[row], pair) * powf(theta_scale, float(pair));
```

`theta_scale = freq_base^(-2/64)` is about 0.65, so it decays *slowly*: for small
`pair` the angle is **thousands of radians**. One ulp of an angle of 2659 radians
is 1.6e-04, and `cos`/`sin` carry that straight into the output. The observable
error therefore scales with the **angle**, not with the output or the inputs - so
no relative-to-output tolerance can be simultaneously tight and correct.

The bound is now derived per element:

```
(|a| + |b|) * (a few ulps of theta)  +  (a few ulps of the output)
```

and the reported `bound excess` is zero in every fixture.

This is the third fixture problem in this rollout (§6m, §6o, here) and the most
instructive of the three, because the failure looked *most* like a real defect: a
10x overshoot, consistent across every case. The structural checks - the byte-exact
copy and the isometry - are what distinguished "the kernel is wrong" from "the
bound is wrong", and they are why this test has three kinds of assertion instead of
one.

## 6r. An audit that over-reported, and the Q5_K gap it hid

The previous checkpoint listed four untested forward-path groups:
`native_qsa_indexer_append`, `native_q5_k_f32`, `native_embed`, `native_mmvq_*`.
That list came from grepping for function names across the test sources. **It was
wrong.**

`native_mmvq` has no test *file* named after it, so the name never appeared - but
`iq_parity.cpp` calls `strata::kernels::native_mmvq(type, ...)` and compares the
result against a float64 matvec for all ten quantization types it covers. The
dispatcher was already tested, and had been since the IQ fixtures landed. The audit
was redone by grepping for **symbols** rather than filenames, and the remaining set
shrank from four groups to two.

The lesson generalises past this instance: **a name-based search can only find
things named the way you guessed.** Both this rollout's false gap
(`native_mmvq`) and its earlier false "resolved by inspection" (`mmq-instance-*.cu`,
§6g) came from the same move - reasoning about coverage instead of measuring it.
The cheap fix is to search for the symbol, which is what the code actually calls.

**What the corrected audit then found** is that the Q5_K path had a genuine gap,
for a different reason: `iq_parity`'s type list is
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_M, IQ4_NL, IQ4_XS, Q2_0, Q3_K. Q5_K is
not in it, and Q5_K is one of the formats the packs use. `native_q5_k_f32` - the
whole decode-step GEMV for that format - had no test at all.

`tests/rocm/native_mmvq_q5k_parity.cpp` now covers it with five fixtures, in the
two constructions that have served throughout this rollout: exact-quantizing
activations, which isolate the GEMV from the quantizer and allow a tight bound, and
ordinary activations, where the Q8_1 rounding dominates and the bound is derived
from what 8 bits per element can cost. `bound excess` is zero in every case.

One presentational note: the test still prints a relative error, and for the
random-activation cases it can read as `1.18e+02` on a case that passes
comfortably. That is not a defect - the reference value itself cancels to near zero
for some rows, so a relative measure is meaningless there. The output now says so
rather than leaving a reader to misread it.

## 6s. Two instruments that failed, and the bug they eventually found

This round produced the last forward-path test and, along the way, two corrections
worth separating from it.

**The audit instrument was wrong three times.** Coverage had been assessed by
grepping the test sources for function names. That produced, in order:

1. a **false gap** - `native_mmvq` looked untested because no file is named after
   it, while `iq_parity` calls the dispatcher for all ten of its types (§6r);
2. a **false positive** - `native_embed` looked like an untested kernel while being
   a host accessor in `src/core/native_head.cpp`, whose actual work is
   `iq_dequant_f32` and is verified to `rel 0.00e+00`;
3. a **list that was mostly not kernels** - `qsa_shapes`, `qsa_states`,
   `kv_plan`, `*_bytes`, `*_enabled`: structs, sizing helpers and config flags.

The instrument is now a per-header declaration sweep: parse each
`include/strata/kernels/*.hpp`, take its declared entry points, and ask whether any
of them appears in any test. That yields a number - **37 of 42 headers covered** -
and a list of five, each of which can be given a disposition. It is still a proxy:
it cannot tell whether a *test* exercises a function properly, only that something
names it. But it is a proxy with a known failure mode, which the name search was
not.

**And then a byte-exact comparison found a real bug in my own helper.** Three of 384
`tail` entries in one indexer case differed by exactly one ulp. The kernel was
right; the hand-rolled `f32_to_f16` copied into four test files was not:

```cpp
mant += 0x1000;                      // round half UP
```

`__float2half_rn` rounds to nearest, **ties to even**. The difference appears only
on an exact tie, and it had been latent through several byte-exact comparisons that
happened to use exact powers of two - until a fixture with 384 compared values per
step across 12 steps hit one. The helper now lives once, in `tests/rocm/f16.h`, with
the tie rule written out and the four copies deleted.

Two things are worth taking from that. A duplicated numeric helper is a bug that
gets fixed zero times, and a byte-exact comparison is what finds it: every
tolerance-based test in this rollout would have passed the wrong converter.

## 6t. The gate, and the difference between "scoped" and "argued away"

The feasibility gate exists to answer one question before 40-55 GB of model weights
are downloaded: *is the ported compute stack plausibly correct on this GPU?*

The evidence at the point of decision: **37 independent tests passing on the RX
7900 XTX**, covering every kernel on the default forward path - attention, the GDN
recurrence and its preprocessing, the QSA ops and key indexer, RoPE (both
variants), the MoE combine, routing, sampling, every quantization format's
dequantization and MMVQ, the prefill GEMM, MMQ for six of eight types, the runtime
core and the graph layer - plus a build in which every translation unit compiles
with no unsupported exclusions.

**Coverage is stated per kernel header: 38 of 42 have a tested entry point.** The
four that do not were each *measured* rather than described:

| header | measured status |
| --- | --- |
| `native_gr_norm` | **not called by the engine at all** - the symbol appears nowhere in `src/core` or `src/program` |
| `fused_gdn` | reachable only behind `native_gdn_enabled()`, which **defaults to false** (`native_gdn.cu:36`), plus `native_bf16_projections` |
| `native_ple_postops` | on the PLE path, which needs the model's PLE table - weight-gated |
| `verify_kernels` | the speculative decoding path, explicitly out of scope for release 1 (plan §9) |

The word "measured" is doing real work there. For two rounds this gate was held on
a list of untested symbols, and that list was wrong in both directions - a false
gap (`native_mmvq`), a false positive (`native_embed`), and a tail of structs and
sizing helpers (§6s). What replaced it asks, for each header, whether *any* of its
declared entry points appears in *any* test, and then for each exception whether the
engine reaches it **by default**. A kernel behind an opt-in flag that defaults to
false is not on the path; a kernel nothing calls is not on any path.

**The gate passes for the default configuration, with three conditions attached** -
and the conditions are the part that matters, because a gate that passes without
naming what it excluded is the same as no gate:

1. **Enabling the `native_gdn` experiment puts untested code on the path.** The
   fused GDN kernels are not covered. Leave the experiment off, or test them first.
2. **The PLE path is unverified** pending weights: `ple_parity` and
   `native_ple_postops` both need the table.
3. **Speculative decoding is out of scope** for release 1 and untested.

Passing this gate authorises *proposing* a download. It does not authorise one, and
it does not claim the port works: no token has been produced, and Stage 4 is where
that claim would have to be earned.

## 7. Floating-point contraction: decided by measurement

This is the one numerical decision the conversion forced, and it was made from
evidence rather than preference.

The in-tree parity suite writes its reference as

```cpp
volatile float product = code * scales[gi];   // the volatile blocks fusion
want[i] = product + offset;
```

and then asserts **both** that the kernel matches that unfused reference
bit-for-bit **and** that fusing would have changed the answer (it *fails* if
`fma_diff == 0`, i.e. if the fixture stopped being sensitive). clang's default
`-ffp-contract=on` fuses inside a statement, so the HIP kernel disagreed with its
own reference:

| build | `elementwise_parity` embedding gather | `quantize_act_parity` Q8_K, small magnitude |
| --- | --- | --- |
| default (`on`) | 10793 bit mismatches / 108 row cases | 2 wrong bytes of 131072 |
| `-ffp-contract=fast` | 10793 bit mismatches | 2 wrong bytes |
| **`-ffp-contract=off`** | **0 mismatches**, `fma_diff` still 8141 | **0 wrong bytes** |

The build therefore sets `-ffp-contract=off`. **No tolerance was changed**, and
the fixture did not get weaker — `fma_diff` still exceeds zero, so the test is
still sensitive to fusion. Matching the arithmetic contract the reference
declares is the opposite of loosening a gate.

Cost, stated plainly: kernels written as `a*b+c` do not fuse, which is a real
throughput loss on a GPU that fuses for free. The upgrade path is to make fusion
*explicit* per kernel with `std::fma` where the reference says so, then relax the
flag with the parity suite as the check. Do not relax it first.

## 8. Numerical evidence policy

The plan's key demand: **a passing build is not a port, and fluent text is not
evidence.** The rules this rollout follows:

1. **Independent reference or no claim.** A kernel is "ported" only when an
   independent implementation agrees with it. The in-tree `src/kernels/*_parity.cpp`
   files carry host and float64 references written from `ref/qsa.py` and from
   ggml's definitions; those are usable as references because they do not share
   code with the kernels. Where no such reference exists, one gets written
   first, in `tests/rocm/ref/`, and the claim stays unverified until it does.
2. **Tolerances are declared in the test source, next to the comparison they
   justify** (e.g. `Strata/src/kernels/iq_parity.cpp:79` for the 2e-2 MMVQ
   bound, the `tol =` derivations in every `tests/rocm/*.cpp`), with the
   datatype and reduction length that justify them. An earlier revision of this
   rule named an `artifacts/rocm/tolerances.json` file that was never created —
   the declaration-in-source form is the one actually used and reviewed. A
   tolerance that changes after a failure is a bug report, not a fix.
3. **Skipped is not passed.** `EX_SKIP=5` exists so a suite that silently
   covered nothing cannot exit 0. Every skip must be declared in
   `tests/rocm/unavailable.json` with the exact output signature that earns it;
   a test's *internal* skip path must exit 5 with a declared signature, never 0.
4. **The harness must be shown to fail.** Session 1 caught a real
   fp-contraction defect (§7). Since session 2 the check also exists in
   standing, re-runnable form: `./strata-rocm selftest` runs `tests/harness/`,
   which injects a failing binary and requires the classifier to report it
   failed (and requires a declared-but-different failure NOT to earn its skip).
5. **Translation error is separated from quantization error** by comparing the
   same decoded values on both paths, as section 6 of the plan requires.

## 9. Unsupported, unverified and out-of-scope (current)

| Item | State |
| --- | --- |
| NVIDIA tensor-core PTX in `native_qsa_score.cu` (`ldmatrix`, `mma.sync.tf32`) | **replaced** by a portable reference path with its own test (see §6b). A fast path is a later, optional optimisation and must not precede that test |
| CUDA graph capture (`cudaGraph*`, 4 headers + `src/core/graph.cpp`) | renamed by HIPIFY; the three behaviours the engine depends on are **measured on gfx1100** (§6d). The per-step-pointer hazard and replay under concurrency remain unmeasured (S5.2) |
| Stream callbacks (`cudaStreamAddCallback`) | unverified (S5) |
| `src/prefill/*.cu` | GEMM **verified**; MMQ **verified for 6 of 8 types** including the model's IQ3_S (§6g, §6h). `IQ2_XXS`/`IQ2_XS` unverified — their reference quantizers need an importance matrix |
| Full-model inference | **not attempted**; weights not on disk yet. The Stage-3 feasibility gate **passed** (see the gate section in STATUS/TASKS); the user is downloading weights to `/mnt/LINUXWINSHARE` and Stage 4 begins when they land and verify |
| `native_expert_parity` | **builds** on the pinned ggml; skipped because it needs a GGUF shard, which is a missing input rather than a missing reference |
| Native (IQ) CPU experts | **built and linked**; not one has been computed, because that needs a shard |
| IQ/QK format coverage | **verified for all ten types** (§6i): dequantization bit-exact against gguf-py, mat-vec inside the declared bound |
| Licence / distribution | no top-level LICENSE file. README credits llama.cpp/ggml (MIT) and ships `third_party/ggml/LICENSE`. **Local use only** until the project's terms are clarified |
| Speculative decoding, vision, >4K context | explicitly out of scope for the first release (plan §9) |
| Performance vs the NVIDIA baseline | **no claim possible**; no measured AMD baseline exists yet |

## 10. What must not be disturbed

An existing llama.cpp server holds port 8080 (`qwen27b-ista-iq3s`, IQ3_S, 27B)
and roughly 19.5 GiB of the card's 24 GiB. Its model files, launch scripts and
any systemd units are preserved. The rollout's own server targets loopback port
**8081**, and `./strata-rocm doctor` reports the existing service by name so a
later session cannot mistake it for leftovers. GPU memory is never freed by
stopping it (`amdgpu.runpm=0` is set in the kernel command line and is left
alone).

## 11. Spending and download limits

Initial limits from section 7 of the plan are recorded in
`artifacts/rocm/state.json` under `limits` and enforced by the wrapper:
3 repair attempts per distinct failure signature, 60-minute execution slice,
4 compile jobs, 120 s small-GPU-test timeout, 30-minute build timeout, 2 GB
cumulative unapproved downloads, no model weights. Downloads so far:
the vendored `hipify-perl` (986704 bytes) and a `numpy` wheel in the project venv
(venv size 79.8 MB, counted separately as `venv_bytes` so that "downloads" means
what it says).
The full model is **not** downloaded, and will not be before the S3
feasibility gate plus explicit approval.
> **Session 3 update:** the gate passed, the user approved by downloading the
> IQ3_S variant themselves, and the rollout fetched mmproj + MTP. All artifacts
> are sha256-verified (`results/record-s4-artifacts-verified.json`); the
> download budget in `state.json` covers rollout-initiated fetches only.
> See §12 for the first inference.

## 12. Session 3 — first inference on the real model (2026-09-28, overnight)

### 12a. Upstream v0.1.16, and patch 0005 (the LDS defect)

Upstream v0.1.15 → v0.1.16 changed no `src/kernels/cuda/*.cu` and no kernel
header; all declared patches re-applied. The first real-model run then found
the port's first forward-path defect, in `fused_gr_read_multi`:

- The multi-token down-projection kernel stages its activation tile in DYNAMIC
  shared memory: `kFusedGrMaxT * TILE * sizeof(float)` = 8 × 2560 × 4 =
  **80 KB**. RDNA3 (gfx1100) caps a workgroup at **64 KB** of LDS; upstream's
  RTX targets allow 80 KB, so this never fails on NVIDIA.
- On gfx1100 `hipFuncSetAttribute(...MaxDynamicSharedMemorySize...)` fails
  (silently — the return value is discarded), and the subsequent launch
  returns `hipErrorInvalidValue`, which the engine surfaces as
  `fused_gr_read_multi: invalid argument` and exits.

Declared patch **0005** (hash-recorded in `results/patches-applied.jsonl`)
changes `TILE` 2560 → **1024**: 32 KB of dynamic LDS, and the kernel's own
`for (base = 0; base < D; base += TILE)` loop is exact for any TILE dividing D
(10240 % 1024 == 0), so the arithmetic — including reduction order per row —
is unchanged; per row the accumulation chunks are the same set in the same order
(below the tile granularity nothing moves — the reviewer verified bitwise-identical
accumulation); only the tile count grows (4 → 10) and the chunks-per-tile shrink
(10 → 4). The patch also makes the two
argument-guard messages print their values, which is what turned the bare
message into a diagnosis. A tolerance was not touched; no reference moved.

### 12b. Serving under coexistence: the pinned-arena lesson

The engine's default expert arena registers ~47 GB of host RAM
(`hipHostRegister`, i.e. pinned, unevictable). One-shot runs map the experts
through the page cache instead (46.84 GiB at 5–6 GiB/s) and coexist fine with
the user's 8080 server. A `--serve` run without `--mmap-experts` pinned that
arena *on top of* the user server's footprint and drove the machine into global
memory pressure; the kernel's OOM killer selected OUR process
(oom_score_adj 200), the user's server was not kernel-killed — but it is
absent from the process table afterwards and restarting it is the user's
action. Consequence, recorded in the serve config: coexistence runs use
`--mmap-experts`. The measured first-inference numbers (32.35 tok/s decode,
524 ms TTFT at 4096 context) are WITH the user's server resident.

### 12c. rocWMMA QSA fast path: feasibility verified, implementation pending

rocWMMA **v0.8** is vendored unprivileged at
`scripts/rocm/vendor/rocwmma-src` (pinned commit `1b59aaad`, MIT), and a probe
translation unit compiles its headers for gfx1100 through the rollout's own
toolchain (`artifacts/rocm/wmma-probe/`). Two facts bound the actual
implementation, which is NOT done yet:

1. **Precision contract.** gfx1100's WMMA has no tf32 operand type; the fast
   path would be fp16 or bf16 inputs with f32 accumulation, against the
   portable path's deliberately-preserved tf32 truncation. That is a changed
   arithmetic contract, so `qsa_score_parity` must gain a SECOND bound derived
   from the fp16 mantissa (2·2⁻¹¹ → 2·2⁻⁸-ish per operand) and recorded HERE
   before any kernel lands — the existing tf32 bound and the bit-exact host
   model stay untouched.
2. **Expected win.** The scorer reads a pooled [n_kv × 128] activation block
   per (row, head) and is likely memory-bound; the portable path is one
   sequential F32 dot per (row, head). The benchmark (`./strata-rocm bench`)
   decides whether a WMMA path is worth its complexity; the portable path
   stays the default until a measured win and a passed second-bound parity
   test both exist.

### 12d. Serving the full IQ model under 96 GB RAM (the three-crash post-mortem)

The symptom: every `--serve` launch of the full IQ3_S model died by OOM (three
times, 23:29/23:41/23:52), the third taking the agent harness down with it.

- **Why the RAM goes.** The IQ engine pins a **47.9 GiB expert arena**
  (`hipHostRegister PORTABLE`) — unevictable anon memory — on top of a ~33 GB
  desktop baseline. Loading additionally streams the 46.8 GiB GGUF through the
  page cache, so the transient peaks near 94 GB on a 96 GB machine.
- **Why `--mmap-experts` does not rescue IQ packs (an upstream gap, now
  analyzed):** `FileExpertSource` (`src/core/expert_source.cpp:38`) demands
  exactly `n_layers × n_expert × cpu::BLOB` bytes — the uniform 1,382,400 B
  Q2_0 relayout blob — while `iq_pack --experts-bin` writes the **variable-size
  GGUF slices** the docstring says IQ requires (50.29 GB, mixed per-layer
  quant types; the engine expects 33.97 GB uniform). There is no file-backed
  expert source for IQ packs upstream. The durable fix is engine work: an IQ
  `ExpertSource` that mmaps `experts.bin` with per-layer offsets from
  `native_experts.txt` (the file the arena path already parses), validated by
  an A/B run against the arena path (same tokens, greedy: identical ids).
- **Why it kept killing *us* specifically.** The engine ran inside the agent
  session's systemd scope with `oom_score_adj=200` — the preferred victim, and
  the whole scope (harness included) was at risk.
- **What serves stably tonight.** A **detached** `systemd-run --user` unit
  (`strata-8081`; adj=0; survives harness crashes), 32768-token context with
  int8 KV entirely in VRAM (no pinned-RAM KV), `--prompt-cache 2`, and the
  engine's stderr finally captured (the config's `log` key was unset, so it
  went to DEVNULL — that silence cost a debug cycle). Steady state:
  83.4 GiB used, ~12.5 GiB available; stable through real completions.
  98304 context *does* reach READY (verified directly) but leaves <12 GiB
  margin; treat it as needing either the mmap engine work, the Coder variant
  (~24 GB arena), or a bigger RAM window.

## 13. A Windows release: feasibility recorded, port deferred

AMD's HIP SDK for Windows **does support this exact GPU** (RX 7900 XTX,
gfx1100; SDK 7.2, Win11), but: AMD marks **CMake's HIP language unsupported**
on Windows (builds go through the SDK's clang + Ninja, llama.cpp-style), which
means rewriting this port's build plumbing rather than reusing it;
`hipHostRegister` of a huge arena is expected to fail there (upstream's own
Windows fallbacks — slice registration, working-set lock — would become the
primary paths, changing the performance story); and **no Windows+RDNA3
environment is reachable from this Fedora box** — a cross-built `strata.exe`
would be an unvalidated artifact, which this project's rules do not call a
release. Upstream's own Windows build is NVIDIA-only. Verdict: the Linux HIP
port is the AMD story for now; a native Windows port is a feasible but real
project (est. 1–2 weeks of on-hardware iteration, e.g. a dual-boot Windows on
this same machine). Start it only with sustained access to that environment.

### 12e. Session 4 — the durable fix landed: patch 0006 (file-backed IQ experts)

§12d's "durable fix" now exists and is validated. `ArenaExpertSource` already
spoke the per-layer blob layout `iq_pack --experts-bin` writes — only
`FileExpertSource` (the old `--mmap-experts` arm) demanded Q2_0-uniform blobs.
Declared patch **0006** adds a file-backed mode: `set_file_backed(true)` mmaps
`experts.bin` and points `base_` at the mapping — same bytes, same `blob()`
offsets, **zero pinned RAM** (unpinned by design: no PCIe device aliases; the
CPU pool computes experts — upstream's documented mmap trade, now available
for IQ packs). `generate.cpp` routes `--mmap-experts` to it for native packs.

**Validation (A/B, same binary, one-shot greedy, 16 tokens):** arena path and
file-backed path produced **identical token streams** —
`11751 13 561 6511 314 9564 369 19241 13 561 6511 314 14898 369 21047 13`
— through completely different expert memory paths
(`results/record-expertsource-ab.json`). RAM: 47.9 GiB pinned → **0 pinned,
41.6 GiB evictable page cache**. Perf: 14.34 tok/s arena vs 7.57 tok/s
file-backed **cold** (upstream documents a 1.79–3.7× warm/cold spread on
their own mmap arm; S6 tuning is later work). Serving at the full **98304**
context now leaves ~59 GiB available — the OOM class is gone.

### 12f. Session 5 — the speed picture, measured end to end

The file-backed arm (§12e) is RAM-safe but its *miss* path reads experts from
wherever `experts.bin` lives — on `/mnt` (a slow volume) that measured
**5.6–8 tok/s**, unusable. The arena arm on the same workload measured
**prompt 33.1 tok/s, decode 25.1 tok/s at 65536 context** (streaming KV
resident 32768; `results/record-serve-speed-65k.json`): the VRAM expert tier
plus pinned-RAM misses are where the speed lives. Two hard config rules fell
out (both recorded in the serve config note): never use all-VRAM KV at ≥64k
(the 1.5 GB dense-weights `hipMalloc` fails → engine exits before READY), and
`--mmap-experts` requires a pack WITH `experts.bin` (it fails fast otherwise).
Durable speed+stability together = the Coder variant (~24 GB arena),
download-gated.

### 12g. The WMMA QSA fast path: probe verdict and the precision contract

**Toolchain verdict, measured.** Vendored rocWMMA **v0.8 cannot compile
`mma_sync` for gfx1100** (no viable overload; the tree emits only CDNA
`__builtin_amdgcn_mfma_*`) — the earlier "compiles for gfx1100" observation
only ever exercised fragment parsing, never codegen. rocWMMA **rocm-7.1.1**
(commit `1ab208f`, pinned shallow clone in `scripts/rocm/vendor/rocwmma-711/`)
has native `ROCWMMA_ARCH_GFX1100` support, is **wave32-only on gfx11**
(`config.hpp:194` — matching this GPU), and its 16×16×16 fp16×fp16→f32
`mma_sync` **compiles and runs correctly on the card**
(`artifacts/rocm/wmma-probe/rocwmma_711_probe.cpp`: outputs within 1 f32 ulp
of the float64 model; the ulp spread is the WMMA unit's internal accumulation
order differing from a sequential host sum — expected, not a defect).

**Precision contract for the WMMA scorer (declared before implementation).**
The scorer computes s[r][h] = Σ_d pooled[r][d]·query[h][d] (D=128), then the
shared epilogue. Pooled rows are RMS-normalized and scaled by `w_k_norm`
(`qsa.cu:211`), so |inputs| are O(1) — fp16 range (max 65504) and subnormal
floor (6.1e-5; flush at ~6e-8) are both far from the data. Conversion to fp16
rounds to 10 explicit mantissa bits — **the same relative magnitude as the
tf32 truncation the portable path simulates** (2·2⁻¹¹ for both operands), so
the input-conversion term of the existing bound carries over unchanged. What
changes is accumulation: the portable path is one sequential f32 sum
(D·2⁻²⁴·Σ|pq|); the WMMA unit's order is unspecified, so the bound must cover
**any** summation order over 128 f32 additions — ≤ (D−1)·2⁻²⁴·Σ|p̃q̃|, plus a
negligible absolute term for subnormal flush (≤ 2·2⁻²⁵·Σ(|p|+|q|)). The
epilogue (per-head ReLU, ordered head sum, bias, 1e9 tail) stays sequential
f32 and is bounded exactly as in the portable test. Net: the WMMA bound is the
portable bound's expression, computed from the same per-row magnitudes, plus
the flush term — recorded here so the test can assert it before any kernel
lands. The bit-exact host model does NOT transfer (different rounding function
and order); the WMMA test uses the float64 oracle under this bound plus a
cross-check against the portable kernel's output within the combined bounds.

### 12h. Patch 0007 built, validated, and — per the predeclared rule — not promoted

The WMMA scorer (declared patch **0007**) is implemented, parity-validated and
benched. **Parity**: 9/9 cases inside the §12g bound (worst excess 0.000e+00),
portable-arm cross-checks pass, guards and tile-boundary cases pass. **Bench**
(`qsa_score_bench`, hipEvent, median of 9): portable 0.027/0.030/0.035 ms vs
WMMA 0.054/0.056/0.100 ms at 8k/32k/65k pooled shapes — the WMMA arm is ~2×
slower at every shape. The scorer streams pooled F32 and is memory-bound; the
fp16 staging round-trip costs more than tensor cores save at K=128. And the
deciding number: the whole scorer is ~0.03 ms of a ~40 ms decode step (~0.1%
of token time), so no kernel-side win here could be visible end to end.

**Promotion decision (the §12c rule, applied): measured, not promoted.** The
portable path stays the default and the engine-wiring question is answered
negatively with numbers. The WMMA arm remains in the tree as an opt-in
(`native_qsa_score_set_wmma`), maintained by its parity test — a working,
validated demonstration that the gfx1100 WMMA stack (rocWMMA rocm-7.1.1,
wave32) is sound for future kernels where the arithmetic actually dominates.

### 12i. Strata vs llama.cpp on the same model, same GPU (2026-09-29)

Regular llama.cpp **can** run this model: mainline has `qwen4exp` upstream
(PR #27742), and the user's own build (`~/src/llama.cpp`, b10723, HIP/gfx1100)
works with the split shards symlinked to llama.cpp's `-0000N-of-000M` naming.
(The ROCmFPX fork binary in `/usr/local/bin` predates the arch and cannot.)

Same machine, same IQ3_S weights, same tokenizer and prompts, one GPU,
sequential runs — **Strata 2.1–2.3× faster decode, ~1.5× faster prompt**:

| | prompt read | decode | notes |
| --- | --- | --- | --- |
| llama.cpp b10723 (`-ngl 99 --n-cpu-moe 48 -c 32768 -fa on`) | 33.0 tok/s | **18.7 tok/s** | ~12 GB anon RAM + page cache; 8.2 GB VRAM |
| Strata HIP v0.1.20 (arena, 65k ctx, int8 KV) | 47.2–55.2 tok/s | **38.8–43.3 tok/s** | 47.9 GB pinned arena + 17.6 GB VRAM expert tier |

Both answer identically and correctly ("Paris"; same sky-explanation opening).
Strata's v0.1.20 numbers include the upstream penalty fix (draft acceptance
56/69 = 81% — the per-drafted-token penalties change visibly helping).
Caveats recorded in `results/record-llamacpp-benchmark.json`: llama.cpp's
config is its standard CPU-MoE layout (an `-ot` tuning could move some experts
into its ~15 GB free VRAM; gains bounded by expert streaming), and it ran 32k
context vs Strata's 65k. The engine gap is architectural: Strata's packed
expert arena + VRAM tier + MTP drafting is built for exactly this model's
512-expert routing; llama.cpp's generic MoE-on-CPU path streams experts from
page cache.

## 14. The CUDA performance playbook, mapped to HIP (2026-09-29)

Upstream's reference numbers (RTX 5070 12 GB + Ryzen 5 7600 + 64 GB, from
`docs/DETAILS.md` and `bench/results/*`): IQ3_S **52 tok/s decode / 1,070 tok/s
prompt-read** (32K prompt, `--prefill auto`). Our HIP baseline before today:
43.6–45.6 tok/s decode on the same variant. The gap decomposes into what the
analysis of their tree found — most of their machinery is *already in this
port* (it's the same hipified source); the real HIP-side deltas are few.

### Already ported AND measured on gfx1100
- **CUDA graph capture/replay** (their ~8× launch-cost win: 0.805 vs 3.63 µs):
  our graph parity suite verified capture/replay/liveness/frozen-args/concurrency (§6d).
- **Pinned arena + CPU pool + VRAM expert tier** (the core design): ported; the
  arena's coexistence cost is §12b/12d (patch 0006 gives the file-backed arm).
- **MMQ int8 expert GEMM, dp4a decode kernels, fused gr/gdn, flash attention**:
  all verified by the parity suite (§6, 41/0/2).
- **PCIe probe / pcie_frac** (v0.1.19): measured on our link — 28.4 GB/s → 0.55 kept.
- **Speculative decoding**: measured on HIP (draft acceptance 81% at spec 2).

### The actual HIP-side backlog (ranked, with evidence)
1. **`-ffp-contract=off` — the largest known HIP-only regression.** nvcc
   contracts `a*b+c` freely (free throughput on every FP kernel); clang's
   default fused it and broke bit-parity (10,793 mismatches, §7), so the whole
   build compiles unfused. Fix path (documented in §7): introduce explicit
   `std::fma` in the hot kernels one at a time, then relax the flag with the
   parity suite as the gate. Candidates: the s2/gdn/gr projection kernels and
   the MMVQ epilogues.
2. **Prefill chunking config — MEASURED, no change warranted at our scale.**
   The tune controller ran `--prefill auto` vs `1024` interleaved at 500- and
   2,400-token prompts: decode equal (±1%); prompt reads 186 tok/s (500) and
   337 tok/s stable vs 485/144 noisy (2,400). The first cycle's spectacular
   "7 → 187 tok/s" was a cold-start artifact that interleaved pairing exists
   to catch — kept as the controller's founding lesson. Upstream's chunking
   wins (383→1,208 tok/s) are at 8K chunks on 32K prompts with the streaming
   ring warm; re-test with `strata-rocm tune` at ≥8K prompts if long-document
   workloads matter. `1024` stays.
3. **Token-graph spin/flush tuning**: `STRATA_TG_FLUSH_US` and the
   per-iteration `hipEventQuery` visibility requirement were characterized on
   the CUDA driver; one measurement pass on HIP (does the doorbell need the
   query? is the flush interval right?) is outstanding.
4. **Doorbell/pool protocol under load**: works (measured decode), but the
   WDDM-specific flush behavior upstream tuned around does not exist on Linux —
   possibly free latency to reclaim.
5. **QSA scorer fast path**: CLOSED as measured-not-promoted (§12h) — ~0.1% of
   decode time. Revisit only if the model's routing share changes.

### Measured-negative on CUDA — do not spend HIP time
cuBLAS grouped GEMM for experts (770 vs 790 in-engine); shared-memory staging
of the GEMV activation (L1/L2 broadcast wins); constant-memory code LUT;
`cp.async`/pipelines (not used upstream at all).

### The honest reference-frame caveat
Their 1,070 tok/s prompt number is a 32K-prompt measurement on PCIe 5.0 with
the streaming ring warm; ours is a ~500-token prompt on first read. The tune
controller's long-prompt workload is the comparable instrument — re-measure at
4K/32K before quoting a ratio.
