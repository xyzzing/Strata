// scripts/rocm/patches/hip_nv_intrinsics.h
//
// NVIDIA CUDA intrinsics that HIP does not provide, as *portable reference*
// implementations.
//
// These are deliberately the slow, obviously-correct versions. The rollout plan
// is explicit: replace an unsupported NVIDIA path with a correct HIP reference
// path *before* optimising it. Each of these has an exact integer definition, so
// "reference" here means bit-exact, not approximate - a fast path built on
// __builtin_amdgcn_sdot4 or v_perm byte tricks is only allowed in once the
// independent parity test agrees with these.
//
// Forced in with `-include`, so no generated file is edited by hand and the
// whole shim disappears by removing one compile flag.
//
// Semantics taken from the CUDA math API:
//   __dp4a(a,b,c)   four 8-bit products, i32 accumulate, no saturation.
//                   Signed when the arguments are `int`, unsigned for `unsigned int`.
//   __vsub4(a,b)    per-byte wrapping subtract.
//   __vsubss4(a,b)  per-byte signed saturating subtract.
//   __vcmpne4(a,b)  0xFF in each byte where a and b differ, else 0x00.
//
// __nanosleep is the one entry that is NOT bit-exact, and it is not arithmetic:
// it is a spin-wait backoff. AMD's s_sleep takes a cycle count, CUDA's argument
// is nanoseconds; the busy-wait loops that use it poll a flag and would be
// correct with any positive delay. Recorded here rather than hidden.

#ifndef STRATA_HIP_NV_INTRINSICS_H
#define STRATA_HIP_NV_INTRINSICS_H

#include "hip_preamble.h"


namespace strata_hip_shim {

__device__ __forceinline__ int byte_at_signed(int v, int i) {
    return (int) (signed char) ((v >> (8 * i)) & 0xff);
}

__device__ __forceinline__ unsigned int byte_at_unsigned(unsigned int v, int i) {
    return (v >> (8 * i)) & 0xffu;
}

}  // namespace strata_hip_shim

// ---- __dp4a -----------------------------------------------------------------
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    int r = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r += strata_hip_shim::byte_at_signed(a, i) * strata_hip_shim::byte_at_signed(b, i);
    }
    return r;
}

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int r = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r += strata_hip_shim::byte_at_unsigned(a, i) * strata_hip_shim::byte_at_unsigned(b, i);
    }
    return r;
}

// ---- byte-wise helpers ------------------------------------------------------
__device__ __forceinline__ int __vsub4(int a, int b) {
    unsigned int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int d = (unsigned int) ((a >> (8 * i)) - (b >> (8 * i))) & 0xffu;
        r |= d << (8 * i);
    }
    return (int) r;
}

__device__ __forceinline__ int __vsubss4(int a, int b) {
    unsigned int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = strata_hip_shim::byte_at_signed(a, i) - strata_hip_shim::byte_at_signed(b, i);
        if (d > 127) d = 127;
        if (d < -128) d = -128;
        r |= ((unsigned int) (d & 0xff)) << (8 * i);
    }
    return (int) r;
}

__device__ __forceinline__ int __vcmpne4(int a, int b) {
    unsigned int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int differ =
            (((a >> (8 * i)) ^ (b >> (8 * i))) & 0xffu) != 0u ? 0xffu : 0x00u;
        r |= differ << (8 * i);
    }
    return (int) r;
}

// ---- sleep ------------------------------------------------------------------
// Not bit-exact by nature: a spin-wait backoff. AMD's s_sleep argument is in
// ~64-cycle units, not nanoseconds, so the delay differs; the loops that use it
// poll a flag and their result does not depend on the delay length.
__device__ __forceinline__ void __nanosleep(unsigned int ns) {
    // s_sleep takes a constant immediate (~64 cycles per unit), so the count is
    // a loop bound rather than an operand.
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned int units = ns == 0u ? 1u : (ns + 63u) / 64u;
    for (unsigned int i = 0; i < units; ++i) {
        __builtin_amdgcn_s_sleep(1);
    }
#else
    // The host pass still parses this body (__device__ is empty there) and has
    // no AMD builtins, so the body must exist for host too.
    (void) ns;
#endif
}

#endif  // STRATA_HIP_NV_INTRINSICS_H
