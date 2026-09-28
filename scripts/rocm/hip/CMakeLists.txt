# Strata HIP backend build (rollout-owned).
#
# Deliberately NOT a patch to Strata/CMakeLists.txt. That build refuses any
# architecture below sm_80 and its CUDA targets are wired for nvcc; editing it
# in place would make every future upstream merge a conflict and would put our
# port inside the tree we are trying to keep pristine. This project consumes the
# HIPIFY-generated tree instead, so deleting build-hip/ loses nothing.
#
# The generated tree is a build input, never a source of truth: regenerate it
# with `./strata-rocm hipify --force`.
#
# Slice 1 (this file, today): the device kernels and their in-tree parity tests,
# which carry their own host/float64 references. Later slices add src/core,
# src/prefill and the engine binary - each behind its own target so one broken
# group cannot hide the state of the others.

cmake_minimum_required(VERSION 3.24)

project(strata_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The host support objects and the HIP objects are linked into one executable on
# a PIE-defaulting distro; without this the host half produces R_X86_64_32
# relocations that ld.lld refuses ("recompile with -fPIC").
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Same value upstream's project() carries; a host source reports it.
add_compile_definitions(STRATA_VERSION="0.1.13")
# Whole build is AMD. hipcc defines this for HIP-language translation units, but
# plain C++ units that include a HIP header need it too, and without it every
# <hip/...> header #errors out with "Must define exactly one of
# __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__".
add_compile_definitions(__HIP_PLATFORM_AMD__=1)

# Passed by ./strata-rocm build, which owns the paths; the fallback keeps the
# file usable by hand. Never guess from CMAKE_CURRENT_SOURCE_DIR: scripts/rocm/hip
# and build-hip/ are siblings three levels up, and a wrong guess here reads as
# "you forgot to run hipify".
if(NOT DEFINED STRATA_HIP_ROOT)
  set(STRATA_HIP_ROOT "$ENV{ROCM_ROOT}/build-hip/hipify")
endif()

# The marker file, not CMakeLists.txt: the exported upstream tree has its own
# CMakeLists.txt, so that would happily accept a directory hipify never wrote.
if(NOT EXISTS "${STRATA_HIP_ROOT}/.hipify-revision")
  message(FATAL_ERROR
    "No HIPIFY-generated tree at ${STRATA_HIP_ROOT} (no .hipify-revision marker). "
    "Run './strata-rocm hipify' first.")
endif()

find_package(hip REQUIRED)

# find_package(hip) brings in the imported targets but does NOT enable the HIP
# language; without this, every .cu fails at generate time with "language HIP
# was requested for compilation but was not enabled".
# CMake refuses the `hipcc` wrapper outright ("CMAKE_HIP_COMPILER is set to the
# hipcc wrapper... Use Clang directly"), so the ROCm clang driver is named
# explicitly. That is also the compiler the ROCm 7.1 device libraries match;
# the Fedora system clang (22) is a different LLVM major.
if(NOT CMAKE_HIP_COMPILER)
  find_program(STRATA_ROCM_CLANGXX
    NAMES amdclang++ clang++
    HINTS /usr/lib64/rocm/llvm/bin /opt/rocm/llvm/bin
    REQUIRED)
  set(CMAKE_HIP_COMPILER "${STRATA_ROCM_CLANGXX}" CACHE FILEPATH "HIP compiler")
endif()

# CMake finds the ROCm root either from the compiler's own "-print-targets"
# output or from `hipconfig --rocmpath`, which on Fedora answers /opt/rocm - a
# directory that does not exist here - so both routes fail with "Failed to find
# HIP root directory". Derive it from where the compiler really lives instead.
if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
  string(REGEX REPLACE "/bin/[^/]+$" "" _rocm_llvm "${STRATA_ROCM_CLANGXX}")   # /usr/lib64/rocm/llvm
  string(REGEX REPLACE "/llvm$" "" _rocm "${_rocm_llvm}")                      # /usr/lib64/rocm
  # CMake's "ROCm root" is the prefix holding lib64/cmake/hip-lang, which on
  # Fedora is /usr, not the /usr/lib64/rocm package directory.
  if(_rocm MATCHES "^(.*)/lib64/rocm$")
    set(_rocm "${CMAKE_MATCH_1}")
  endif()
  if(EXISTS "${_rocm}/lib64/cmake/hip-lang/hip-lang-config.cmake"
     OR EXISTS "${_rocm}/lib/cmake/hip-lang/hip-lang-config.cmake")
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_rocm}" CACHE PATH "ROCm root")
    message(STATUS "Strata HIP: ROCm root derived from the compiler: ${_rocm}")
  endif()
endif()
enable_language(HIP)

set(CMAKE_HIP_ARCHITECTURES "gfx1100" CACHE STRING "HIP architectures (RX 7900 XTX = gfx1100)")
if(NOT "gfx1100" IN_LIST CMAKE_HIP_ARCHITECTURES)
  message(FATAL_ERROR
    "This build is validated for gfx1100 only; CMAKE_HIP_ARCHITECTURES is "
    "'${CMAKE_HIP_ARCHITECTURES}'. Widening it is a deliberate, tested decision, "
    "not a convenience.")
endif()
message(STATUS "Strata HIP: arch ${CMAKE_HIP_ARCHITECTURES}, tree ${STRATA_HIP_ROOT}")

# Upstream compiles with -Wall -Wextra (and -Werror only on request). Warnings
# are kept: the CUDA->HIP translation is mechanical and a warning is often the
# first sign that a translation changed meaning.
# Floating-point contraction, decided by measurement rather than taste.
#
# The in-tree parity suite writes its reference as `volatile float product =
# code * scales[gi]; want = product + offset;` - the `volatile` is there to stop
# the compiler fusing the multiply into the add - and then asserts BOTH that the
# kernel matches that unfused reference bit-for-bit AND that fusing would have
# changed the answer (it fails if `fma_diff == 0`, i.e. if the fixture stopped
# being sensitive). clang's default `-ffp-contract=on` fuses inside a statement,
# so the kernel disagreed with its own reference:
#
#   default / fast : elementwise embedding gather 10793 bit mismatches; Q8_K
#                    small-magnitude quantization 2 wrong bytes of 131072
#   -ffp-contract=off : 0 mismatches, 0 wrong bytes, fma_diff still 8141 (so the
#                    fixture is still sensitive - the test did not get weaker)
#
# Matching the declared contract is not the same as loosening a tolerance; every
# tolerance in the suite is untouched. The cost is real and recorded: kernels
# written as a*b+c do not fuse. The upgrade path is to make fusion explicit per
# kernel with std::fma where the reference says so, then relax this flag with the
# parity suite as the check. See PORTING.md section 7.
set(STRATA_HIP_FP_CONTRACT "off" CACHE STRING
    "clang -ffp-contract mode; see the comment above before changing it")
add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=${STRATA_HIP_FP_CONTRACT}>)
message(STATUS "Strata HIP: -ffp-contract=${STRATA_HIP_FP_CONTRACT}")

add_library(strata_hip_warnings INTERFACE)
target_compile_options(strata_hip_warnings INTERFACE
  -Wall -Wextra
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)

# ---- the generated headers --------------------------------------------------
add_library(strata_hip_headers INTERFACE)
target_include_directories(strata_hip_headers INTERFACE
  "${STRATA_HIP_ROOT}/include"
  "${STRATA_HIP_ROOT}/third_party/ggml")
target_link_libraries(strata_hip_headers INTERFACE strata_hip_warnings)

# ---- NVIDIA-only intrinsics shim -------------------------------------------
# __dp4a / __vsub4 / __vcmpne4 / __vsubss4 / __nanosleep have no HIP equivalent.
# Forced in as a header so the generated tree stays untouched; see the file for
# why these are reference implementations rather than fast ones.
set(STRATA_HIP_PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../patches")

# Preamble for every HIP translation unit: ordering only, no symbols.
add_library(strata_hip_preamble INTERFACE)
target_compile_options(strata_hip_preamble INTERFACE
  -include${STRATA_HIP_PATCH_DIR}/hip_preamble.h)
target_include_directories(strata_hip_preamble INTERFACE "${STRATA_HIP_PATCH_DIR}")

# NVIDIA-only intrinsics, for the Strata kernel units only. ggml's own HIP
# vendor header supplies its versions of some of these, so forcing this into a
# ggml-based translation unit makes every call ambiguous.
add_library(strata_hip_shims INTERFACE)
target_link_libraries(strata_hip_shims INTERFACE strata_hip_preamble)
target_compile_options(strata_hip_shims INTERFACE
  $<$<COMPILE_LANGUAGE:HIP>:-include${STRATA_HIP_PATCH_DIR}/hip_nv_intrinsics.h>)

# ---- device kernels ---------------------------------------------------------
file(GLOB STRATA_HIP_KERNELS CONFIGURE_DEPENDS
     "${STRATA_HIP_ROOT}/src/kernels/cuda/*.cu")
# Excluded, not forgotten. native_qsa_score.cu is built on NVIDIA tensor-core
# PTX (`ldmatrix`, `mma.sync.tf32`) that has no HIP equivalent; it needs a
# hand-written reference path and its own test (Stage 3). Leaving it in the
# library would fail the link for every other kernel, which would hide the
# state of the 42 that do build. The list is written into the build directory so
# ./strata-rocm build reports it instead of only the console.
# Empty as of the portable-path patch: native_qsa_score.cu now builds. Kept as a
# mechanism, because the next unsupported translation unit must land here (and
# in the report) rather than being quietly dropped from the source glob.
set(STRATA_HIP_UNSUPPORTED "")
foreach(_u IN LISTS STRATA_HIP_UNSUPPORTED)
  list(REMOVE_ITEM STRATA_HIP_KERNELS "${_u}")
  message(STATUS "Strata HIP: EXCLUDED (unsupported, Stage 3): ${_u}")
endforeach()
list(LENGTH STRATA_HIP_KERNELS _n)
message(STATUS "Strata HIP: ${_n} kernel translation units")
if(STRATA_HIP_UNSUPPORTED)
  file(WRITE "${CMAKE_BINARY_DIR}/unsupported-sources.txt"
       "${STRATA_HIP_UNSUPPORTED}\n")
else()
  file(WRITE "${CMAKE_BINARY_DIR}/unsupported-sources.txt" "")
endif()

# .cu is CUDA's extension; CMake only maps it to HIP when the CUDA language is
# absent, which is fragile to rely on. Naming the language makes the kernel
# target's linker language unambiguous.
set_source_files_properties(${STRATA_HIP_KERNELS} PROPERTIES LANGUAGE HIP)
add_library(strata_hip_kernels STATIC ${STRATA_HIP_KERNELS})
# PRIVATE: the intrinsic shims are needed by these translation units, but a
# consumer that also includes ggml's vendors/hip.h must not inherit them, or
# every call to __vsub4/__vcmpne4 becomes ambiguous.
target_link_libraries(strata_hip_kernels PUBLIC strata_hip_headers hip::device)
# Rollout patch 0007: the WMMA QSA arm includes rocWMMA (vendored, pinned
# rocm-7.1.1, header-only). Include-only for one translation unit; harmless to
# the rest of the target.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/rocwmma-711/library/include/rocwmma/rocwmma.hpp")
  target_include_directories(strata_hip_kernels PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/rocwmma-711/library/include")
  message(STATUS "Strata HIP: rocWMMA rocm-7.1.1 available (QSA WMMA arm enabled)")
endif()

# ---- rollout bench tool: portable vs WMMA QSA scorer timing (patch 0007) ----
add_executable(qsa_score_bench "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/rocm/qsa_score_bench.cpp")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/rocm/qsa_score_bench.cpp" PROPERTIES LANGUAGE HIP)
target_link_libraries(qsa_score_bench PRIVATE strata_hip_kernels strata_hip_shims hip::host strata_hip_core)
target_link_libraries(strata_hip_kernels PRIVATE strata_hip_shims)
# The Fedora packaging detail recorded in scripts/rocm/lib/common.sh: hipcc
# does not add the HIP runtime because hipconfig reports a ROCM_PATH that does
# not exist here. hip::host brings the imported target in explicitly.
target_link_libraries(strata_hip_kernels PUBLIC hip::host)

# ---- host-side support ------------------------------------------------------
# Some kernels have a host half that lives in a plain .cpp (the PLE/n-gram table
# reader), so the parity executables need it linked in. Kept as its own target
# so it is obvious which sources are host code and which are device code.
add_library(strata_hip_host STATIC
  "${STRATA_HIP_ROOT}/src/kernels/ngram.cpp"
  "${STRATA_HIP_ROOT}/src/ngram/ple_reader.cpp"
  "${STRATA_HIP_ROOT}/src/platform/direct_file.cpp"
  "${STRATA_HIP_ROOT}/src/platform/memory.cpp")
target_link_libraries(strata_hip_host PUBLIC strata_hip_headers)
# A host translation unit that pulls in <hip/hip_runtime.h> needs the platform
# macro that hipcc would have defined; without it the header #errors out.
target_compile_definitions(strata_hip_host PUBLIC __HIP_PLATFORM_AMD__=1)
# Same <format>/__noinline__ ordering problem: this host code includes HIP
# headers too, so it needs the same pre-include.
target_link_libraries(strata_hip_host PUBLIC strata_hip_preamble)

# ---- llama.cpp (ggml) pinned sources ---------------------------------------
# src/prefill/{kernels,moe_mmq}.cu are derived from ggml-cuda and include its
# headers; upstream does not vendor them and setup.py downloads the same pin.
# The headers are HIP-aware, which is the whole reason to reuse them instead of
# rewriting ggml's MMQ path.
# Defaulted from the same environment variable the HIPIFY tree uses, so the
# project works by hand as well as through ./strata-rocm build.
if(NOT DEFINED STRATA_LLAMA_DIR)
  set(STRATA_LLAMA_DIR "$ENV{ROCM_ROOT}/build-hip/llama.cpp")
endif()
if(NOT EXISTS "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda/common.cuh")
  message(WARNING
    "Pinned llama.cpp sources not found at ${STRATA_LLAMA_DIR}; the prefill MMQ "
    "targets will be skipped. Run './strata-rocm prepare'.")
  set(STRATA_HAVE_LLAMA FALSE)
else()
  set(STRATA_HAVE_LLAMA TRUE)
endif()

# ---- runtime core (+ the strata-device probe) --------------------------------
# Composition mirrors the upstream strata_core target exactly, so "which sources
# are in the core" stays answerable by reading upstream's CMakeLists.
set(STRATA_CORE_DEV
  "${STRATA_HIP_ROOT}/src/core/device.cu"
  "${STRATA_HIP_ROOT}/src/core/pinned.cu")
set(STRATA_CORE_HOST
  "${STRATA_HIP_ROOT}/src/platform/memory.cpp"
  "${STRATA_HIP_ROOT}/src/core/graph.cpp"
  "${STRATA_HIP_ROOT}/src/core/weights.cpp"
  "${STRATA_HIP_ROOT}/src/core/layout.cpp")

set_source_files_properties(${STRATA_CORE_DEV} PROPERTIES LANGUAGE HIP)
add_library(strata_hip_core STATIC ${STRATA_CORE_DEV} ${STRATA_CORE_HOST})
target_link_libraries(strata_hip_core PUBLIC strata_hip_headers hip::host strata_hip_preamble)
target_link_libraries(strata_hip_core PRIVATE strata_hip_shims)
target_compile_definitions(strata_hip_core PUBLIC __HIP_PLATFORM_AMD__=1)

# `strata-device` is the most useful early target: it reports the card and
# exercises the arena without needing a model, so it is a real end-to-end check
# of the core rather than a compile.
add_executable(strata-device "${STRATA_HIP_ROOT}/src/core/device_main.cpp")
set_source_files_properties("${STRATA_HIP_ROOT}/src/core/device_main.cpp" PROPERTIES LANGUAGE HIP)
target_link_libraries(strata-device PRIVATE strata_hip_core)
add_test(NAME strata-device COMMAND strata-device --selftest)
set_tests_properties(strata-device PROPERTIES TIMEOUT 120)

# ---- ggml CPU backend (upstream's STRATA_NATIVE_EXPERTS) ---------------------
# Upstream's own mechanism: add_subdirectory on a llama.cpp checkout's ggml tree
# and link ggml-cpu + ggml-base. Same pinned commit as everything else, MIT.
# Only ggml-cpu and ggml-base are linked, so the GPU backends are never built:
# this is a CPU dot-product library, not a second inference engine.
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
if(STRATA_HAVE_LLAMA AND STRATA_NATIVE_EXPERTS)
  set(GGML_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
  # Native feature detection instead of chip-specific flags: upstream forces
  # AVX/AVX2/FMA/F16C/BMI2 ON, and ggml-cpu has its own runtime dispatch.
  set(GGML_AVX ON CACHE BOOL "" FORCE)
  set(GGML_AVX2 ON CACHE BOOL "" FORCE)
  set(GGML_FMA ON CACHE BOOL "" FORCE)
  set(GGML_F16C ON CACHE BOOL "" FORCE)
  set(GGML_BMI2 ON CACHE BOOL "" FORCE)
  add_subdirectory("${STRATA_LLAMA_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
  message(STATUS "Strata HIP: native experts enabled through ggml-cpu")
else()
  message(STATUS "Strata HIP: native (IQ) CPU experts DISABLED - the CPU half of an expert layer will be absent")
endif()

# ---- CPU expert kernels ------------------------------------------------------
# Upstream's always-on CPU list. `native_expert.cpp` and `iq_avx512.cpp` are the
# upstream STRATA_NATIVE_EXPERTS option, which needs ggml's CPU backend linked in;
# that is a separate, recorded step (TASKS S2.7c) rather than a silent omission.
#
# The AVX-512 / AVX2 flags go on the SOURCE, not the target, exactly as upstream
# does: a translation unit compiled with -mavx512f may use AVX-512 anywhere in it,
# so the scalar pool and the test drivers must not share it, or the fallback stops
# being a fallback.
set(STRATA_KERNELS_CPU
  "${STRATA_HIP_ROOT}/src/kernels/cpu/expert.cpp"
  "${STRATA_HIP_ROOT}/src/kernels/cpu/pool.cpp"
  "${STRATA_HIP_ROOT}/src/kernels/cpu/expert_layout.cpp"
  "${STRATA_HIP_ROOT}/src/kernels/cpu/q2_avx2.cpp")
add_library(strata_hip_kernels_cpu STATIC ${STRATA_KERNELS_CPU})
if(STRATA_NATIVE_EXPERTS AND TARGET ggml-cpu)
  # Upstream's split (as of v0.1.15): iq_avx512 is the AVX-512 IQ path,
  # iq_avx2 is the AVX2 fallback decode (the v0.1.14/15 "AVX2-only CPUs decode
  # rows once per verify window" work) and native_expert calls into both.
  # Each carries its own ISA requirement, set per source below; upstream
  # composes exactly this trio into strata_kernels_cpu.
  target_sources(strata_hip_kernels_cpu PRIVATE
    "${STRATA_HIP_ROOT}/src/kernels/cpu/native_expert.cpp"
    "${STRATA_HIP_ROOT}/src/kernels/cpu/iq_avx512.cpp"
    "${STRATA_HIP_ROOT}/src/kernels/cpu/iq_avx2.cpp")
  target_include_directories(strata_hip_kernels_cpu PRIVATE "${STRATA_LLAMA_DIR}/ggml/include")
  target_link_libraries(strata_hip_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_hip_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  set_source_files_properties("${STRATA_HIP_ROOT}/src/kernels/cpu/iq_avx512.cpp" PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties("${STRATA_HIP_ROOT}/src/kernels/cpu/iq_avx2.cpp" PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
target_link_libraries(strata_hip_kernels_cpu PUBLIC strata_hip_headers strata_hip_preamble)
find_package(Threads REQUIRED)
target_link_libraries(strata_hip_kernels_cpu PUBLIC Threads::Threads)
set_source_files_properties("${STRATA_HIP_ROOT}/src/kernels/cpu/expert.cpp" PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties("${STRATA_HIP_ROOT}/src/kernels/cpu/q2_avx2.cpp" PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")

# ---- prefill (host + device) ------------------------------------------------
if(STRATA_HAVE_LLAMA)
  file(GLOB STRATA_PREFILL_DEV "${STRATA_HIP_ROOT}/src/prefill/*.cu")
  set(STRATA_PREFILL_HOST "${STRATA_HIP_ROOT}/src/prefill/prefill.cpp")

  set_source_files_properties(${STRATA_PREFILL_DEV} PROPERTIES LANGUAGE HIP)
  add_library(strata_hip_prefill STATIC ${STRATA_PREFILL_DEV})
  target_link_libraries(strata_hip_prefill PUBLIC strata_hip_kernels strata_hip_preamble)
  if(TARGET strata_hip_mmq)
    # PUBLIC: consumers of the prefill MMQ call its symbols too.
    target_link_libraries(strata_hip_prefill PUBLIC strata_hip_mmq)
  endif()
  # Directories go on the include path, not through target_link_libraries: CMake
  # warns about directory link items and then does not add them as includes,
  # which shows up as "'common.cuh' file not found".
  # ggml-cuda's common.cuh chooses vendors/hip.h only under GGML_USE_HIP;
  # without it the CUDA branch pulls in <cuda_fp16.h> and fails.
  target_compile_definitions(strata_hip_prefill PUBLIC GGML_USE_HIP=1)
  target_include_directories(strata_hip_prefill PUBLIC
    "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda"
    "${STRATA_LLAMA_DIR}/ggml/include"
    "${STRATA_LLAMA_DIR}/ggml/src")

  # Fedora puts <hipblas.h> under /usr/include/hipblas, and the imported target
  # carries that directory; linking the bare name "hipblas" does not.
  find_package(hipblas QUIET)
  if(hipblas_FOUND AND TARGET roc::hipblas)
    target_link_libraries(strata_hip_prefill PUBLIC roc::hipblas)
    message(STATUS "Strata HIP: hipBLAS found (roc::hipblas); prefill GEMM linked against it")
  elseif(hipblas_FOUND)
    target_link_libraries(strata_hip_prefill PUBLIC ${hipblas_LIBRARIES})
    target_include_directories(strata_hip_prefill PUBLIC ${hipblas_INCLUDE_DIRS} /usr/include/hipblas)
    message(STATUS "Strata HIP: hipBLAS found (plain libraries)")
  else()
    message(WARNING "hipBLAS not found: src/prefill/gemm.cu will not link")
  endif()

  # prefill.cpp is host code that calls the kernels; same platform macro and
  # <format> ordering problem as the other host half.
  set_source_files_properties("${STRATA_PREFILL_HOST}" PROPERTIES LANGUAGE HIP)
  add_library(strata_hip_prefill_host STATIC "${STRATA_PREFILL_HOST}")
  target_link_libraries(strata_hip_prefill_host PUBLIC strata_hip_prefill)
  target_compile_definitions(strata_hip_prefill_host PUBLIC __HIP_PLATFORM_AMD__=1)
  target_link_libraries(strata_hip_prefill_host PUBLIC strata_hip_preamble)
endif()

# ---- composed engine, speculation, and the `strata` binary -------------------
if(STRATA_HAVE_LLAMA)
  set(STRATA_ENGINE_SRC
    "${STRATA_HIP_ROOT}/src/core/layer.cpp"
    "${STRATA_HIP_ROOT}/src/core/session.cpp"
    "${STRATA_HIP_ROOT}/src/core/expert_source.cpp"
    "${STRATA_HIP_ROOT}/src/core/expert_cache.cpp"
    "${STRATA_HIP_ROOT}/src/core/native_head.cpp"
    "${STRATA_HIP_ROOT}/src/core/native_dense.cpp"
    "${STRATA_HIP_ROOT}/src/core/verify.cpp"
    "${STRATA_HIP_ROOT}/src/core/mtp.cpp")
  # Host translation units that call kernels. Same platform macro and preamble as
  # the other host halves.
  set_source_files_properties(${STRATA_ENGINE_SRC} PROPERTIES LANGUAGE HIP)
  add_library(strata_hip_engine STATIC ${STRATA_ENGINE_SRC})
  target_link_libraries(strata_hip_engine PUBLIC
    strata_hip_core strata_hip_kernels strata_hip_kernels_cpu strata_hip_prefill
    strata_hip_prefill_host strata_hip_host strata_hip_preamble)

  set(STRATA_SPEC_SRC
    "${STRATA_HIP_ROOT}/src/spec/suffix_drafter.cpp"
    "${STRATA_HIP_ROOT}/src/spec/controller.cpp"
    "${STRATA_HIP_ROOT}/src/spec/draft_policy.cpp")
  add_library(strata_hip_spec STATIC ${STRATA_SPEC_SRC})
  target_link_libraries(strata_hip_spec PUBLIC strata_hip_headers strata_hip_preamble)

  add_executable(strata "${STRATA_HIP_ROOT}/src/program/generate.cpp")
  set_source_files_properties("${STRATA_HIP_ROOT}/src/program/generate.cpp" PROPERTIES LANGUAGE HIP)
  target_link_libraries(strata PRIVATE strata_hip_engine strata_hip_spec)
endif()

# ---- ggml-cuda MMQ instances (upstream's strata_mmq composition) -------------
# This is S2.7d, and the previous checkpoint was WRONG to close it by inspection.
# `moe_mmq.cu` does not instantiate `mul_mat_q_case<T>` itself: the template is
# only DECLARED in mmq.cuh, and its definitions live in ggml-cuda's
# template-instances/*.cu. `quantize_mmq_q8_1_cuda` lives in quantize.cu. Neither
# was in this build. Upstream compiles exactly these nine files; so do we now, so
# that the symbols demonstrably exist rather than being assumed to.
if(STRATA_HAVE_LLAMA)
  set(_ggcuda "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda")
  set(STRATA_MMQ_SRC "${_ggcuda}/quantize.cu")
  foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs)
    if(EXISTS "${_ggcuda}/template-instances/mmq-instance-${_t}.cu")
      list(APPEND STRATA_MMQ_SRC "${_ggcuda}/template-instances/mmq-instance-${_t}.cu")
    else()
      message(WARNING "MMQ instance missing for ${_t}: that type will not link")
    endif()
  endforeach()
  set_source_files_properties(${STRATA_MMQ_SRC} PROPERTIES LANGUAGE HIP)
  add_library(strata_hip_mmq STATIC ${STRATA_MMQ_SRC})
  target_include_directories(strata_hip_mmq PRIVATE
    "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda"
    "${STRATA_LLAMA_DIR}/ggml/include"
    "${STRATA_LLAMA_DIR}/ggml/src")
  target_compile_definitions(strata_hip_mmq PRIVATE GGML_USE_HIP=1 __HIP_PLATFORM_AMD__=1)
  target_link_libraries(strata_hip_mmq PRIVATE strata_hip_preamble hip::device)
  list(LENGTH STRATA_MMQ_SRC _nmmq)
  message(STATUS "Strata HIP: ${_nmmq} ggml-cuda MMQ translation units")
endif()

# ---- IQ fixture builder ------------------------------------------------------
# `iq_parity` needs fixtures upstream does not publish; this builds them. It is a
# tool, not a test, so it is not run by the suite.
if(TARGET ggml-base)
  add_executable(make_iq_fixtures
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/rocm/make_iq_fixtures.cpp")
  target_link_libraries(make_iq_fixtures PRIVATE ggml-base)
  target_include_directories(make_iq_fixtures PRIVATE
    "${STRATA_LLAMA_DIR}/ggml/include" "${STRATA_LLAMA_DIR}/ggml/src")
  target_compile_definitions(make_iq_fixtures PRIVATE __HIP_PLATFORM_AMD__=1)
endif()

# ---- in-tree parity tests ---------------------------------------------------
enable_testing()
set(_parity_targets "")
# These compare a kernel against a host implementation written from ref/qsa.py
# or from ggml's definition, so they are real evidence rather than a smoke test.
# They are compiled as HIP translation units: they call cudaMalloc-derived HIP
# APIs directly, and keeping them on one compiler avoids a host/device ABI
# mismatch that would only show up at run time.
set(STRATA_HIP_PARITY
    bf16_gemv cvec dequant_s2 elementwise gdn gr iq kv_q4 kv_q8 kv_stream
    ple qsa quantize_act rope router_top10 s2_gemv s2_gemv_q8
    sampler s_gemv s_gemv_q8k shared_expert)
# Deliberately absent: native_expert. Its parity test is the only one that
# includes "ggml-cpu.h" and "ggml.h", and upstream ships only
# third_party/ggml/ggml-common.h. Its reference is therefore unavailable here -
# an unavailable reference keeps the claim unverified, it does not make it pass.
# native_expert_parity was unavailable while ggml-cpu was absent. It now BUILDS,
# so it is a maintained target; it cannot RUN without a GGUF shard, which is a
# missing input rather than a missing reference, and tests/rocm/unavailable.json
# records exactly that distinction.
if(STRATA_NATIVE_EXPERTS AND TARGET ggml-cpu)
  set(_src "${STRATA_HIP_ROOT}/src/kernels/native_expert_parity.cpp")
  set_source_files_properties("${_src}" PROPERTIES LANGUAGE HIP)
  add_executable(native_expert_parity "${_src}")
  target_link_libraries(native_expert_parity PRIVATE
    strata_hip_kernels strata_hip_kernels_cpu strata_hip_host strata_hip_shims
    strata_hip_preamble hip::host ggml-cpu ggml-base)
  target_include_directories(native_expert_parity PRIVATE "${STRATA_LLAMA_DIR}/ggml/include")
  target_compile_definitions(native_expert_parity PRIVATE STRATA_NATIVE_EXPERTS=1)
  list(APPEND _parity_targets native_expert_parity)
endif()

# The reset above used to sit AFTER the native_expert_parity block, silently
# dropping it from this list: declared suite membership must be appended after
# the reset, never before it.
foreach(_p IN LISTS STRATA_HIP_PARITY)
  set(_src "${STRATA_HIP_ROOT}/src/kernels/${_p}_parity.cpp")
  if(NOT EXISTS "${_src}")
    message(WARNING "parity source missing for ${_p}: ${_src}")
    continue()
  endif()
  set(_tgt "${_p}_parity")
  add_executable(${_tgt} "${_src}")
  set_source_files_properties("${_src}" PROPERTIES LANGUAGE HIP)
  target_link_libraries(${_tgt} PRIVATE strata_hip_kernels strata_hip_host strata_hip_shims)
  target_link_libraries(${_tgt} PRIVATE hip::host)
  add_test(NAME ${_tgt} COMMAND ${_tgt} --selftest)
  set_tests_properties(${_tgt} PROPERTIES TIMEOUT 120)
  list(APPEND _parity_targets ${_tgt})
endforeach()

# Rollout-authored tests live outside the checkout, in tests/rocm/. They are
# listed explicitly rather than globbed: a test that silently stops being built
# is worse than one that fails.
set(STRATA_ROCM_TESTS qsa_score_parity qsa_score_wmma_parity prefill_gemm_parity core_graph_parity prefill_mmq_parity
                     native_router_parity native_mmvq_quant_parity native_flash_attn_parity
                     native_gdn_parity native_gdn_preprocess_parity native_moe_combine_parity
                     native_gr_postops_parity native_qsa_ops_parity native_rope_parity
                     native_mmvq_q5k_parity native_qsa_indexer_parity s2_expert_grouped_parity
                     fused_gdn_parity fused_gdn_conv_l2_parity fused_gdn_ab_parity)
foreach(_t IN LISTS STRATA_ROCM_TESTS)
  set(_src "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/rocm/${_t}.cpp")
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "rollout test source missing: ${_src}")
  endif()
  # .cpp by extension, but it calls HIP APIs and links hip::device (which carries
  # -x hip), so the language must be HIP or GCC is handed a flag it rejects.
  set_source_files_properties("${_src}" PROPERTIES LANGUAGE HIP)
  add_executable(${_t} "${_src}")
  # tests/rocm holds the shared test header(s), e.g. f16.h
  target_include_directories(${_t} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/rocm")
  target_link_libraries(${_t} PRIVATE strata_hip_kernels strata_hip_shims)
  # the MMQ test uses ggml's own reference quantizers as its oracle, and the
  # real ggml type ids, so it needs the ggml headers and library
  if(TARGET ggml-base)
    target_link_libraries(${_t} PRIVATE ggml-base)
    target_include_directories(${_t} PRIVATE "${STRATA_LLAMA_DIR}/ggml/include")
  endif()
  target_link_libraries(${_t} PRIVATE hip::host)
  target_link_libraries(${_t} PRIVATE strata_hip_core)
  if(STRATA_HAVE_LLAMA)
    target_link_libraries(${_t} PRIVATE strata_hip_prefill strata_hip_prefill_host)
    target_link_libraries(${_t} PRIVATE strata_hip_preamble)
    if(TARGET ggml-base)
      target_link_libraries(${_t} PRIVATE ggml-base)
    endif()
    if(TARGET strata_hip_mmq)
      target_link_libraries(${_t} PRIVATE strata_hip_mmq)
    endif()
  endif()
  add_test(NAME ${_t} COMMAND ${_t} --selftest)
  set_tests_properties(${_t} PROPERTIES TIMEOUT 120)
  list(APPEND _parity_targets ${_t})
endforeach()

set(STRATA_HIP_PARITY_TARGETS "${_parity_targets}" CACHE INTERNAL "")
message(STATUS "Strata HIP: parity targets ${_parity_targets}")
