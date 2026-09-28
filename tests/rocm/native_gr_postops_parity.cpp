// tests/rocm/native_gr_postops_parity.cpp - independent check of the GR
// post-operations, and the answer to a question the last checkpoint left open.
//
// `native_gr_postops.hpp` was recorded as a KNOWN SUSPECT: its header contracts
// "ordered FMA accumulation" for the layer path, this rollout builds every HIP
// translation unit with `-ffp-contract=off`, and a sibling kernel (`native_moe`)
// had already been caught silently losing its documented fusion to that flag. This
// file closes the question - and the answer is that this file is innocent, for a
// reason worth stating: it names `__fmaf_rn` explicitly, so the contraction
// setting cannot touch it. `native_moe` wrote `sum += a*b` and depended on the
// compiler; this one did not. Same contract, two different amounts of care.
//
// THE MEASUREMENT PROBLEM, AND HOW IT IS SOLVED HERE
//
// Both accumulation paths exist ON PURPOSE (`fused_layer` selects between them,
// and the header documents them as different), so the test must tell them apart
// bit-for-bit. It cannot do that by comparing against a float64 ideal - the two
// differ only in the last bit - and it cannot do it through `sigmoid` either,
// because that calls `expf`, whose host and device implementations are not
// guaranteed to agree to the last ulp.
//
// Two constructions are therefore used, each for what it can actually establish:
// `gate = 0` makes `sigmoid` exactly 1/2 and every operation exact, so each path
// can be checked against its own simulated order BIT FOR BIT - but it makes
// `fma(x,w,s)` and `s+(x*w)` the same operation, so it cannot tell the orders
// apart. For that, both paths are run on the SAME random inputs: a contracting
// fused path and a separately-rounded head path disagree on ~half the elements,
// and if they ever agree the fusion is gone. That check needs no tolerance and is
// insensitive to host/device `expf` differences.
//
// Run: native_gr_postops_parity --selftest

#include "strata/kernels/native_gr_postops.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "  *** WRONG *** %s %s\n", what, detail);
    }
}

#define HIPCHK(expr)                                                                   \
    do {                                                                               \
        hipError_t _e = (expr);                                                        \
        if (_e != hipSuccess) {                                                        \
            std::fprintf(stderr, "hip error %s: %s\n", #expr, hipGetErrorString(_e));   \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

struct Rng {
    uint64_t s = 0xc0ac29b7c97c50ddull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// The kernel's own scale-with-zero-bias: an explicit fma against +0, kept so a
// compile-time zero cannot change signed-zero behaviour.
float szb(float x, float s) { return std::fma(s, x, 0.0f); }
float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-(double)v)); }

// ---- down_silu ---------------------------------------------------------------
int run_down_silu(Rng& rng) {
    using namespace strata::kernels;
    const int count = 384, hc = 4;
    std::vector<float> lo(count, 0.0f);
    for (int i = 0; i < count; ++i) lo[i] = rng.uniform(-3.0f, 3.0f);
    lo[0] = 0.0f;                                   // exact zero must stay zero
    std::vector<double> ref(count);
    for (int i = 0; i < count; ++i) {
        const float x = szb(lo[i], 1.0f / (float)hc);
        ref[i] = (double)(x / (1.0f + (float)std::exp(-(double)x)));
    }

    float* d = nullptr;
    HIPCHK(hipMalloc(&d, count * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(d, lo.data(), count * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));
    native_gr_down_silu(d, count, hc, stream);
    HIPCHK(hipStreamSynchronize(stream));
    std::vector<float> got(count);
    HIPCHK(hipMemcpy(got.data(), d, count * 4, hipMemcpyDeviceToHost));

    double worst = 0.0;
    int nonfinite = 0;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(got[i])) { ++nonfinite; continue; }
        worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]));
    }
    std::printf("  %-34s nonfinite=%d  worst|err|=%.3e  (zero stays %.1e)\n",
                "down_silu", nonfinite, worst, (double)got[0]);
    check(nonfinite == 0, "down_silu must be finite");
    check(worst < 1e-6, "down_silu exceeded the tolerance justified by expf");
    check(got[0] == 0.0f, "a zero input must stay exactly zero");
    hipStreamDestroy(stream);
    hipFree(d);
    return 0;
}

// ---- pre_gated, both documented paths ---------------------------------------
//
// THE DISCRIMINATOR, and why the obvious one does not work.
//
// The first attempt made `gate = 0` so that `sigmoid(0) = 1/2` is exact in any
// libm, hoping to get a bit-exact comparison of the accumulation order. It does
// give bit-exactness - and it destroys the discrimination, because multiplying by
// 1/2 is exact, so `fma(x, w, sum)` and `sum + (x*w)` become the same operation.
// The construction removed the very rounding it was meant to observe.
//
// The order can be told apart without controlling the sigmoid at all, by running
// BOTH paths on the same inputs:
//
//   * if the fused path really contracts, `sum = fma(x, w, sum)` differs from
//     `sum = sum + (x*w)` on most elements, so the two paths disagree;
//   * if it does NOT contract - which is what the project-wide
//     `-ffp-contract=off` did to `native_moe` - the fused path computes exactly
//     what the head path computes, and the two outputs are IDENTICAL.
//
// So "the two paths differ" is the tolerance-free evidence that the documented
// fusion is real, and it is insensitive to host/device `expf` differences. The
// gate=0 case is kept alongside it for the checks it *can* make bit-exactly: the
// gate overwrite, and each path matching its own order when every op is exact.
int run_pre_gated(Rng& rng, bool zero_gate) {
    using namespace strata::kernels;
    const int n_embd = 256, hc = 8;

    std::vector<float> xn((size_t)n_embd * hc), gate((size_t)n_embd * hc);
    for (auto& v : xn) v = rng.uniform(-1.5f, 1.5f);
    for (auto& v : gate) v = zero_gate ? 0.0f : rng.uniform(-4.0f, 4.0f);

    // The two documented orders, simulated exactly in fp32.
    std::vector<float> fused_ref((size_t)n_embd), rounded_ref((size_t)n_embd),
        gate_ref((size_t)n_embd * hc);
    for (int d = 0; d < n_embd; ++d) {
        float sum_f = 0.0f, sum_r = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const size_t i = (size_t)c * n_embd + d;
            const float x = xn[i];
            const float w = zero_gate ? 0.5f : sigmoid(gate[i]);
            const float product = x * w;
            gate_ref[i] = product;
            sum_f = std::fma(x, w, sum_f);
            sum_r = (c == 0) ? product : (sum_r + product);
        }
        const float scale = 1.0f / (float)hc;
        fused_ref[d] = scale * sum_f;
        rounded_ref[d] = szb(sum_r, scale);
    }

    // Run BOTH paths on the same inputs. Each needs its own copy of `gate`,
    // because the kernel overwrites it in place.
    std::vector<float> fused_got((size_t)n_embd), head_got((size_t)n_embd),
        fused_gate(gate.size()), head_gate(gate.size());
    float *dxn = nullptr, *dgate = nullptr, *dmixed = nullptr;
    HIPCHK(hipMalloc(&dxn, xn.size() * 4));
    HIPCHK(hipMalloc(&dgate, gate.size() * 4));
    HIPCHK(hipMalloc(&dmixed, (size_t)n_embd * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dxn, xn.data(), xn.size() * 4, hipMemcpyHostToDevice, stream));

    for (int pass = 0; pass < 2; ++pass) {
        const bool fused = pass == 0;
        HIPCHK(hipMemcpyAsync(dgate, gate.data(), gate.size() * 4, hipMemcpyHostToDevice, stream));
        HIPCHK(hipStreamSynchronize(stream));
        native_gr_pre_gated(dxn, dgate, dmixed, n_embd, hc, fused, stream);
        HIPCHK(hipStreamSynchronize(stream));
        std::vector<float>& mixed = fused ? fused_got : head_got;
        std::vector<float>& g = fused ? fused_gate : head_gate;
        HIPCHK(hipMemcpy(mixed.data(), dmixed, (size_t)n_embd * 4, hipMemcpyDeviceToHost));
        HIPCHK(hipMemcpy(g.data(), dgate, gate.size() * 4, hipMemcpyDeviceToHost));
    }

    int fused_vs_ref = 0, head_vs_ref = 0, cross = 0, gate_wrong = 0;
    double worst_fused = 0.0, worst_head = 0.0;
    for (int d = 0; d < n_embd; ++d) {
        if (std::memcmp(&fused_got[d], &fused_ref[d], 4) != 0) ++fused_vs_ref;
        if (std::memcmp(&head_got[d], &rounded_ref[d], 4) != 0) ++head_vs_ref;
        if (std::memcmp(&fused_got[d], &head_got[d], 4) != 0) ++cross;
        worst_fused = std::fmax(worst_fused, std::fabs((double)fused_got[d] - fused_ref[d]));
        worst_head = std::fmax(worst_head, std::fabs((double)head_got[d] - rounded_ref[d]));
    }
    for (size_t i = 0; i < fused_gate.size(); ++i) {
        if (zero_gate) {
            if (fused_gate[i] != gate_ref[i] || head_gate[i] != gate_ref[i]) ++gate_wrong;
        } else if (std::fabs((double)fused_gate[i] - (double)gate_ref[i]) > 1e-6 ||
                   std::fabs((double)head_gate[i] - (double)gate_ref[i]) > 1e-6) {
            ++gate_wrong;
        }
    }

    std::printf("  %-24s gate=%-6s  fused-vs-its-order=%d  head-vs-its-order=%d  "
                "fused-vs-head=%d  gate wrong=%d  worst |dev-ref| = %.2e / %.2e\n",
                "pre_gated both paths", zero_gate ? "zero" : "random",
                fused_vs_ref, head_vs_ref, cross, gate_wrong, worst_fused, worst_head);

    check(gate_wrong == 0, "gate must be overwritten with xn*sigmoid(gate)");
    if (zero_gate) {
        // With sigmoid exactly 1/2, multiplying by it is exact, so fma(x,w,sum) and
        // sum+(x*w) ARE the same operation and the two paths coincide by
        // construction - `fused-vs-head` is 0 here and must be. This case exists for
        // the opposite reason: because every operation is exact, each path can be
        // checked against its own simulated order BIT FOR BIT.
        check(fused_vs_ref == 0, "with exact operations the fused path must match its order exactly");
        check(head_vs_ref == 0, "with exact operations the head path must match its order exactly");
    } else {
        check(worst_fused < 1e-6 && worst_head < 1e-6,
              "with random gates both paths must stay within the expf tolerance");
        // And this is the case that tells the orders apart: with real products, a
        // contracting fused path and a separately-rounded head path disagree. If
        // they agreed, the fused path would not be contracting - exactly the
        // native_moe defect - and no tolerance would reveal it.
        check(cross > n_embd / 4,
              "the two documented accumulation orders must produce DIFFERENT results; if they are "
              "identical the fused path is not contracting - which is exactly the native_moe defect");
    }

    hipStreamDestroy(stream);
    hipFree(dxn); hipFree(dgate); hipFree(dmixed);
    return 0;
}

// ---- post --------------------------------------------------------------------
int run_post(Rng& rng, bool zero_inject, bool alias) {
    using namespace strata::kernels;
    const int n_embd = 128, hc = 4;

    std::vector<float> residual((size_t)n_embd * hc), block((size_t)n_embd),
        inject((size_t)hc);
    for (auto& v : residual) v = rng.uniform(-2.0f, 2.0f);
    for (auto& v : block) v = rng.uniform(-1.0f, 1.0f);
    for (auto& v : inject) v = zero_inject ? 0.0f : rng.uniform(-3.0f, 3.0f);

    std::vector<double> ref(residual.size());
    for (int c = 0; c < hc; ++c) {
        const float w = szb(sigmoid(szb(inject[c], 1.0f / (float)hc)), 2.0f);
        for (int d = 0; d < n_embd; ++d) {
            const size_t i = (size_t)c * n_embd + d;
            ref[i] = (double)std::fma(block[d], w, residual[i]);
        }
    }

    float *dres = nullptr, *dblk = nullptr, *dinj = nullptr, *dout = nullptr;
    HIPCHK(hipMalloc(&dres, residual.size() * 4));
    HIPCHK(hipMalloc(&dblk, block.size() * 4));
    HIPCHK(hipMalloc(&dinj, inject.size() * 4));
    if (!alias) HIPCHK(hipMalloc(&dout, residual.size() * 4));
    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dres, residual.data(), residual.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dblk, block.data(), block.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dinj, inject.data(), inject.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    // `output may equal residual exactly` - one of the few overlaps the contract allows
    native_gr_post(dres, dblk, dinj, alias ? dres : dout, n_embd, hc, stream);
    HIPCHK(hipStreamSynchronize(stream));

    std::vector<float> got(residual.size());
    HIPCHK(hipMemcpy(got.data(), alias ? dres : dout, residual.size() * 4, hipMemcpyDeviceToHost));

    double worst = 0.0;
    int nonfinite = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) { ++nonfinite; continue; }
        worst = std::fmax(worst, std::fabs((double)got[i] - ref[i]));
    }
    std::printf("  %-34s %s alias=%-5s  nonfinite=%d  worst|err|=%.3e\n", "post",
                zero_inject ? "inject=0 " : "inject=rnd", alias ? "yes" : "no", nonfinite, worst);
    check(nonfinite == 0, "post must be finite");
    check(worst < 1e-6, "post exceeded the tolerance justified by expf");

    hipStreamDestroy(stream);
    hipFree(dres); hipFree(dblk); hipFree(dinj); if (dout) hipFree(dout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_gr_postops_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_gr_postops_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: exact fp32 simulations of both documented accumulation orders\n"
                "  (gate=0 makes sigmoid exactly 1/2, so those comparisons are bit-exact)\n");

    Rng rng;
    if (run_down_silu(rng) != 0) check(false, "down_silu aborted");
    // gate = 0: every operation exact, so both paths must match their order bit for bit
    if (run_pre_gated(rng, true) != 0) check(false, "pre_gated exact aborted");
    // and through the real sigmoid, where the cross-path difference is the evidence
    if (run_pre_gated(rng, false) != 0) check(false, "pre_gated random aborted");
    if (run_post(rng, true, false) != 0) check(false, "post exact aborted");
    if (run_post(rng, false, false) != 0) check(false, "post random aborted");
    if (run_post(rng, false, true) != 0) check(false, "post alias aborted");

    std::printf("\nnative_gr_postops: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_gr_postops_parity OK\n");
    (void)selftest;
    return 0;
}
