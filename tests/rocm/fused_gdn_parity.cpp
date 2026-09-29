// tests/rocm/fused_gdn_parity.cpp - the fused GDN step+norm against its own parts.
//
// `fused_gdn.hpp`'s three entry points are the only kernel group a default-config
// run cannot reach: they sit behind `native_gdn_enabled()`, which defaults to false
// (`native_gdn.cu:36`). That is exactly why they are tested here - the feasibility
// gate passed with a condition attached, "enabling the native experiment puts
// untested code on the path", and this removes the condition for the largest of the
// three rather than leaving it as a warning.
//
// HOW A FUSED KERNEL IS TESTED. The header states its arithmetic exactly:
//
//     S <- g S + k (beta (v - g S^T k))^T      (g = exp(gate[head]))
//     o  = (S^T q) / sqrt(128)
//     y  = rmsnorm(o) * gamma * sigmoid(z)
//
// and that is precisely `native_gdn_step` followed by `native_gdn_out_norm` - both
// of which already have independent passing tests (`native_gdn_parity`,
// `native_gdn_preprocess_parity`). So this test does not re-derive the recurrence a
// third time: it runs the composition of the verified parts and requires the fused
// kernel to agree with it. Two independent, already-verified references agreeing
// with a third implementation is stronger evidence than a fourth hand-written
// reference would be, and it is the decomposition the header itself invites.
//
// What can still differ is the REDUCTION ORDER: the fused kernel gives a block one
// whole head and reduces the 128 columns its own way, where the unfused path
// reduces per warp. So the comparison is tolerance-based, with the bound derived
// from the reduction length - and the STATE, which is a recurrence rather than a
// reduction, is checked separately because an error there compounds.
//
// Run: fused_gdn_parity --selftest

#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int S = 128;
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
    uint64_t s = 0x510e527fade682d1ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs((double)a[i] - (double)b[i]);
        den += std::fabs((double)b[i]);
    }
    return num / (den + 1e-30);
}

// Largest single-element disagreement. An aggregate over 65k state elements can
// absorb a handful of fully-wrong ones; this cannot.
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

int run_case(const char* name, int h_k, int h_v, int steps, float gate_lo, float gate_hi, Rng& rng) {
    using namespace strata::kernels;
    const float eps = 1e-5f;

    std::vector<float> q((size_t)h_k * S), k((size_t)h_k * S), v((size_t)h_v * S),
        gate((size_t)h_v), beta((size_t)h_v), z((size_t)h_v * S), gamma(S);
    for (auto& x : q) x = rng.uniform(-1.0f, 1.0f);
    for (auto& x : k) x = rng.uniform(-1.0f, 1.0f);
    for (auto& x : v) x = rng.uniform(-1.5f, 1.5f);
    for (auto& x : gate) x = rng.uniform(gate_lo, gate_hi);
    for (auto& x : beta) x = rng.uniform(0.2f, 1.0f);
    for (auto& x : z) x = rng.uniform(-3.0f, 3.0f);
    for (auto& x : gamma) x = rng.uniform(-1.0f, 1.0f);

    // q and k are normalized, as the GDN contract requires.
    auto l2 = [&](std::vector<float>& x, int heads) {
        for (int h = 0; h < heads; ++h) {
            double n = 0.0;
            for (int i = 0; i < S; ++i) n += (double)x[(size_t)h * S + i] * x[(size_t)h * S + i];
            const float inv = (float)(1.0 / std::sqrt(n + 1e-30));
            for (int i = 0; i < S; ++i) x[(size_t)h * S + i] *= inv;
        }
    };
    l2(q, h_k);
    l2(k, h_k);

    const size_t state_n = (size_t)S * h_v * S;
    std::vector<float> st0(state_n);
    for (auto& x : st0) x = rng.uniform(-0.25f, 0.25f);
    std::vector<float> st_ref = st0, st_fused = st0;

    float *dstate_ref = nullptr, *dstate_fused = nullptr;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dgate = nullptr, *dbeta = nullptr;
    float *dz = nullptr, *dgamma = nullptr, *dout = nullptr, *dy_ref = nullptr, *dy_fused = nullptr;
    HIPCHK(hipMalloc(&dstate_ref, state_n * 4));
    HIPCHK(hipMalloc(&dstate_fused, state_n * 4));
    HIPCHK(hipMalloc(&dq, q.size() * 4));
    HIPCHK(hipMalloc(&dk, k.size() * 4));
    HIPCHK(hipMalloc(&dv, v.size() * 4));
    HIPCHK(hipMalloc(&dgate, gate.size() * 4));
    HIPCHK(hipMalloc(&dbeta, beta.size() * 4));
    HIPCHK(hipMalloc(&dz, z.size() * 4));
    HIPCHK(hipMalloc(&dgamma, gamma.size() * 4));
    HIPCHK(hipMalloc(&dout, (size_t)h_v * S * 4));
    HIPCHK(hipMalloc(&dy_ref, (size_t)h_v * S * 4));
    HIPCHK(hipMalloc(&dy_fused, (size_t)h_v * S * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dstate_ref, st_ref.data(), state_n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dstate_fused, st_fused.data(), state_n * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dgate, gate.data(), gate.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dbeta, beta.data(), beta.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dz, z.data(), z.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dgamma, gamma.data(), gamma.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    GdnShapes shapes;
    shapes.S = S;
    shapes.h_k = h_k;
    shapes.h_v = h_v;

    for (int step = 0; step < steps; ++step) {
        // reference: the two already-verified kernels, composed
        native_gdn_step(dstate_ref, dq, dk, dv, dgate, dbeta, dout, shapes, stream);
        native_gdn_out_norm(dout, dz, dgamma, dy_ref, h_v, S, eps, stream);
        // fused: one kernel
        fused_gdn_step_norm(dstate_fused, dq, dk, dv, dgate, dbeta, dz, dgamma, eps, dy_fused, h_k, h_v, stream);
        HIPCHK(hipStreamSynchronize(stream));
    }

    std::vector<float> y_ref((size_t)h_v * S), y_fused((size_t)h_v * S), s_ref(state_n), s_fused(state_n);
    HIPCHK(hipMemcpy(y_ref.data(), dy_ref, y_ref.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(y_fused.data(), dy_fused, y_fused.size() * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(s_ref.data(), dstate_ref, state_n * 4, hipMemcpyDeviceToHost));
    HIPCHK(hipMemcpy(s_fused.data(), dstate_fused, state_n * 4, hipMemcpyDeviceToHost));

    int nonfinite = 0;
    for (float x : y_fused) if (!std::isfinite(x)) ++nonfinite;
    for (float x : s_fused) if (!std::isfinite(x)) ++nonfinite;

    const double y_err = rel_l1(y_fused, y_ref);
    const double s_err = rel_l1(s_fused, s_ref);
    // Same bound per element, scaled by the largest reference magnitude: the
    // reduction-order argument bounds each element's error proportionally, so a
    // few badly wrong elements must trip this even when the aggregate is small.
    double y_scale = 0.0, s_scale = 0.0;
    for (float v : y_ref) y_scale = std::fmax(y_scale, std::fabs((double)v));
    for (float v : s_ref) s_scale = std::fmax(s_scale, std::fabs((double)v));
    const double y_max = max_abs_diff(y_fused, y_ref);
    const double s_max = max_abs_diff(s_fused, s_ref);
    // Bound: the fused kernel reduces the 128 columns in one block where the
    // unfused path reduces per warp, so the orders differ. `steps` of recurrence
    // compound whatever that costs.
    const double tol = 8.0 * (double)S * std::ldexp(1.0, -24) * (double)steps;
    std::printf("  %-30s h_k=%d h_v=%d steps=%-2d  y rel=%.3e max=%.1e  state rel=%.3e max=%.1e  "
                "tol=%.1e  nonfinite=%d\n",
                name, h_k, h_v, steps, y_err, y_max, s_err, s_max, tol, nonfinite);

    check(nonfinite == 0, "the fused kernel must produce finite output");
    check(y_err < tol, "the fused output exceeded the bound derived from the reduction length");
    check(s_err < tol, "the fused recurrence's state diverged from the composed reference");
    check(y_max < tol * std::fmax(1.0, y_scale),
          "the fused output has an element past the per-element bound");
    check(s_max < tol * std::fmax(1.0, s_scale),
          "the fused state has an element past the per-element bound");

    hipStreamDestroy(stream);
    hipFree(dstate_ref); hipFree(dstate_fused); hipFree(dq); hipFree(dk); hipFree(dv);
    hipFree(dgate); hipFree(dbeta); hipFree(dz); hipFree(dgamma);
    hipFree(dout); hipFree(dy_ref); hipFree(dy_fused);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: fused_gdn_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("fused_gdn_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: native_gdn_step + native_gdn_out_norm, both already independently\n"
                "  verified, composed exactly as the fused header states its arithmetic\n");

    Rng rng;
    run_case("identity head map", 4, 4, 1, -0.7f, 0.0f, rng);
    run_case("grouped head map", 2, 4, 1, -0.7f, 0.0f, rng);
    // the stateful case, as in native_gdn_parity: a recurrence that mishandles its
    // own feedback passes one step and fails several
    run_case("grouped, eight steps", 2, 4, 8, -0.3f, 0.0f, rng);
    run_case("no decay, four steps", 2, 4, 4, 0.0f, 0.0f, rng);
    run_case("single value head", 1, 1, 6, -0.5f, -0.1f, rng);

    std::printf("\nfused_gdn: %d failures\n", failures);
    if (failures) return 1;
    std::printf("fused_gdn_parity OK\n");
    (void)selftest;
    return 0;
}
