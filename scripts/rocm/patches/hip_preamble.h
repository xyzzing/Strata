// scripts/rocm/patches/hip_preamble.h
//
// A forced include for every HIP translation unit in this rollout, including
// the ones built against llama.cpp's ggml-cuda headers. It defines nothing: it
// only fixes header *ordering*.
//
// DO NOT add symbols here. ggml's vendors/hip.h already implements __vsub4,
// __vsubss4, __vcmpne4 and its own dp4a for HIP; defining them again makes
// every call ambiguous. Symbols belong in hip_nv_intrinsics.h, which is
// force-included only into the Strata kernel translation units.

#ifndef STRATA_HIP_PREAMBLE_H
#define STRATA_HIP_PREAMBLE_H

// Order matters, and getting it wrong is a hard build error with a confusing
// message. HIP's host_defines define `__noinline__` as a macro; libstdc++ 16's
// <format> uses the attribute [[__gnu__::__noinline__]], which the macro
// rewrites into nonsense:
//
//   /usr/include/c++/16/format:4550:30: error: expected an identifier for the
//     attribute name [-Wtemplate-body]
//       [[__gnu__::__noinline__]]
//
// <chrono> reaches <format> through bits/chrono_io.h, so any translation unit
// that includes a HIP header before <chrono> fails to compile on this toolchain
// with both g++ 16 and ROCm clang. Parsing <format> first sidesteps it
// entirely: the later include hits its own guard. No source file is edited.
#if defined(__has_include)
#if __has_include(<format>)
#include <format>
#endif
#endif
#include <hip/hip_runtime.h>

#endif  // STRATA_HIP_PREAMBLE_H
