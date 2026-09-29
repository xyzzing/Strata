// tests/rocm/f16.h - the fp16 conversion the rollout tests share.
//
// This exists because the helper was copied into four test files, and all four
// copies had the same bug: they rounded half-UP instead of half-to-EVEN.
//
//     mant += 0x1000;              // round half up
//
// `__float2half_rn` rounds to nearest, ties to EVEN. The two agree except on an
// exact tie, which random data hits rarely - so the bug sat latent through several
// byte-exact comparisons that happened to use exact powers of two, until
// `native_qsa_indexer_parity` compared 384 tail entries per step and found three
// that differed by exactly one ulp.
//
// The lesson is the one this rollout keeps relearning: a hand-written numeric
// helper is a place for a silent half-ulp error, and a byte-exact comparison is
// what finds it. One definition, in one file, with the tie rule written out.
//
// The device-side counterpart is `__float2half_rn`; this must agree with it
// exactly, since several tests compare F16-rendered buffers byte for byte.

#pragma once

#include <cstdint>
#include <cstring>

namespace strata_test {

/// IEEE binary16 conversion, round to nearest, ties to even. Matches
/// `__float2half_rn` for every finite input; subnormals flush to zero, which is
/// adequate for the fixtures here and is stated rather than hidden.
inline uint16_t f32_to_f16(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    const uint32_t magnitude = u & 0x7fffffffu;

    if (magnitude >= 0x7f800000u) {                       // inf or NaN
        return (uint16_t)(sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u));
    }
    int32_t exp = (int32_t)((magnitude >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = magnitude & 0x7fffffu;                // 23 bits
    if (exp <= 0) return sign;                            // flush to zero

    // The 13 bits being dropped: one round bit and twelve sticky bits.
    const uint32_t dropped = mant & 0x1fffu;
    mant >>= 13;
    if (dropped > 0x1000u || (dropped == 0x1000u && (mant & 1u))) ++mant;   // ties to even
    if (mant & 0x400u) { mant = 0; ++exp; }               // carry into the exponent
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);     // overflow to infinity
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

/// Exact: every binary16 value is representable in binary32.
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) {
            float z; const uint32_t u = sign; std::memcpy(&z, &u, 4); return z;
        }
        while (!(mant & 0x400u)) { mant <<= 1; --exp; }
        ++exp; mant &= 0x3ffu;
    } else if (exp == 31) {
        float z; const uint32_t u = sign | 0x7f800000u | (mant << 13);
        std::memcpy(&z, &u, 4); return z;
    }
    const uint32_t u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float z; std::memcpy(&z, &u, 4); return z;
}

/// Round-trip through binary16, which is what several kernels do to their inputs.
inline float half_round(float x) { return f16_to_f32(f32_to_f16(x)); }

}  // namespace strata_test
