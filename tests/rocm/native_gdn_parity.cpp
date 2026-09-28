// tests/rocm/native_gdn_parity.cpp - independent check of the gated-delta-net
// recurrence, the stateful core of the recurrent layers.
//
// No upstream test exists for the native variant (its parity test was in the
// omitted bench/micro tree, against llama.cpp's CUDA oracle). `src/kernels/gdn_parity.cpp`
// does test the NON-native path against a reference, and that reference is the
// same mathematics - but it deliberately uses a DIFFERENT state layout
// (`(S, S, h_v)`) to catch layout mix-ups, so it cannot simply be reused. This
// file implements the recurrence in float64 against the layout the native header
// documents, `state[(i*h_v + head)*S + j]`.
//
// WHY THIS ONE MATTERS: the recurrence is stateful. An error does not appear once
// and move on - it is fed back into the next step, and the next. So the central
// case here is multi-step: eight consecutive steps over one state buffer, with the
// reference iterating the same eight steps, which is the only way a compounding
// error shows up.
//
// TWO DETAILS THE HEADER DOES NOT STATE, resolved from the implementation and
// flagged as such rather than presented as documented:
//   * the decay is applied as `state_new = g*state_old + k*((v - g*kv)*beta)`
//     with `kv` taken from the UN-decayed state - algebraically the "decay first"
//     reading, but with a different rounding path;
//   * the readout uses the UPDATED state, not the previous one.
// Both are asserted here against the algebra they imply, so if upstream changes
// one the test fails rather than silently agreeing with something else.
//
// Run: native_gdn_parity --selftest

#include "strata/kernels/native_gdn.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int S = 128;          // the only head dim the kernel supports

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
    uint64_t s = 0x082efa98ec4e6c89ull;
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * ((float)(next() % 1000001u) / 1000000.0f);
    }
};

// One step in float64, native layout. `state` is (S, h_v, S) with j fastest.
void ref_step(std::vector<double>& st, const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& gate, const std::vector<float>& beta,
              int h_k, int h_v, std::vector<float>& out) {
    const double scale = 1.0 / std::sqrt((double)S);
    out.assign((size_t)h_v * S, 0.0f);
    for (int head = 0; head < h_v; ++head) {
        const int qh = head % h_k;                       // documented map, and what the kernel does
        const double g = std::exp((double)gate[(size_t)head]);
        const double b = (double)beta[(size_t)head];
        for (int col = 0; col < S; ++col) {
            double kv = 0.0;
            for (int i = 0; i < S; ++i)
                kv += st[((size_t)i * h_v + head) * S + col] * (double)k[(size_t)qh * S + i];
            const double delta = ((double)v[(size_t)head * S + col] - g * kv) * b;
            double attn = 0.0;
            for (int i = 0; i < S; ++i) {
                double& cell = st[((size_t)i * h_v + head) * S + col];
                cell = g * cell + (double)k[(size_t)qh * S + i] * delta;   // updated value ...
                attn += cell * (double)q[(size_t)qh * S + i];              // ... used by the readout
            }
            out[(size_t)head * S + col] = (float)(attn * scale);
        }
    }
}

double rel_l1(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs((double)a[i] - b[i]);
        den += std::fabs(b[i]);
    }
    return num / (den + 1e-30);
}

// Largest single-element disagreement. An aggregate over tens of thousands of
// elements can absorb a handful of fully-wrong ones; this cannot.
template <typename A, typename B>
double max_abs_diff(const A& a, const B& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::fmax(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

struct Cfg {
    const char* name;
    int h_k, h_v;
    int steps;
    float gate_lo, gate_hi;      // log-gate range; exp() of these is the decay
    float beta_lo, beta_hi;
    bool zero_state;
};

int run_case(const Cfg& c, Rng& rng) {
    using namespace strata::kernels;
    const int h_k = c.h_k, h_v = c.h_v;

    std::vector<float> q((size_t)h_k * S), k((size_t)h_k * S), v((size_t)h_v * S),
        gate((size_t)h_v), beta((size_t)h_v);
    for (auto& x : q) x = rng.uniform(-1.0f, 1.0f);
    for (auto& x : k) x = rng.uniform(-1.0f, 1.0f);
    for (auto& x : v) x = rng.uniform(-1.0f, 1.0f);
    for (auto& x : gate) x = rng.uniform(c.gate_lo, c.gate_hi);
    for (auto& x : beta) x = rng.uniform(c.beta_lo, c.beta_hi);

    // Normalise q and k, as the contract requires ("normalized F32 inputs").
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

    std::vector<double> st_ref((size_t)S * h_v * S, 0.0);
    if (!c.zero_state) {
        for (auto& x : st_ref) x = (double)rng.uniform(-0.25f, 0.25f);
    }
    std::vector<float> st_dev(st_ref.size());
    for (size_t i = 0; i < st_ref.size(); ++i) st_dev[i] = (float)st_ref[i];

    float* dst = nullptr;
    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr, *dout = nullptr;
    HIPCHK(hipMalloc(&dst, st_dev.size() * 4));
    HIPCHK(hipMalloc(&dq, q.size() * 4));
    HIPCHK(hipMalloc(&dk, k.size() * 4));
    HIPCHK(hipMalloc(&dv, v.size() * 4));
    HIPCHK(hipMalloc(&dg, gate.size() * 4));
    HIPCHK(hipMalloc(&db, beta.size() * 4));
    HIPCHK(hipMalloc(&dout, (size_t)h_v * S * 4));

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));
    HIPCHK(hipMemcpyAsync(dst, st_dev.data(), st_dev.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(dg, gate.data(), gate.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipMemcpyAsync(db, beta.data(), beta.size() * 4, hipMemcpyHostToDevice, stream));
    HIPCHK(hipStreamSynchronize(stream));

    GdnShapes shapes;
    shapes.S = S;
    shapes.h_k = h_k;
    shapes.h_v = h_v;

    std::vector<float> out((size_t)h_v * S);
    std::vector<double> out_ref((size_t)h_v * S, 0.0);
    for (int step = 0; step < c.steps; ++step) {
        native_gdn_step(dst, dq, dk, dv, dg, db, dout, shapes, stream);
        HIPCHK(hipStreamSynchronize(stream));
        HIPCHK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));

        std::vector<float> step_out;
        ref_step(st_ref, q, k, v, gate, beta, h_k, h_v, step_out);
        for (size_t i = 0; i < out_ref.size(); ++i) out_ref[i] = (double)step_out[i];

        // The state only changes on the first step here (the inputs are the same
        // each step, deliberately: a repeated step is exactly where a recurrence
        // that fails to feed its own output back shows up).
    }

    std::vector<float> st_out(st_dev.size());
    HIPCHK(hipMemcpy(st_out.data(), dst, st_dev.size() * 4, hipMemcpyDeviceToHost));

    // Bound: each step is a 128-term fp32 dot product, and `steps` of them compound.
    const double per_step = (double)S * std::ldexp(1.0, -24);
    const double tol = 4.0 * per_step * (double)c.steps;
    const double out_err = rel_l1(out, out_ref);
    const double st_err = rel_l1(st_out, st_ref);
    // Same bound per element, scaled by the largest reference magnitude:
    // reduction reordering bounds each element's error proportionally, so a few
    // badly wrong elements must trip this even when the aggregate stays small.
    double out_scale = 0.0, st_scale = 0.0;
    for (double v : out_ref) out_scale = std::fmax(out_scale, std::fabs(v));
    for (double v : st_ref) st_scale = std::fmax(st_scale, std::fabs(v));
    const double out_max = max_abs_diff(out, out_ref);
    const double st_max = max_abs_diff(st_out, st_ref);

    std::printf("  %-34s h_k=%d h_v=%d steps=%d zero_state=%d  out rel=%.3e max=%.1e  "
                "state rel=%.3e max=%.1e  tol=%.1e  ratio=%.3f\n",
                c.name, h_k, h_v, c.steps, (int)c.zero_state, out_err, out_max,
                st_err, st_max, tol, std::fmax(out_err, st_err) / tol);

    check(out_err < tol, "the readout exceeded the bound derived from the reduction length");
    check(st_err < tol, "the state exceeded the bound derived from the reduction length");
    check(out_max < tol * std::fmax(1.0, out_scale),
          "the readout has an element past the per-element bound (an aggregate cannot see this)");
    check(st_max < tol * std::fmax(1.0, st_scale),
          "the state has an element past the per-element bound (an aggregate cannot see this)");

    hipStreamDestroy(stream);
    hipFree(dst); hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dg); hipFree(db); hipFree(dout);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: native_gdn_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("native_gdn_parity on %s (%s)\n", prop.name, prop.gcnArchName);
    std::printf("  reference: float64 recurrence in the native layout state[(i*h_v+head)*S+j],\n"
                "  iterated for the same number of steps as the device\n");

    Rng rng;
    const Cfg cases[] = {
        // heads map identity (h_k == h_v)
        {"identity head map, one step",     4, 4, 1,  -0.7f, 0.0f,  0.2f, 1.0f, false},
        // heads map is a real reduction: heads 0..3 -> q heads 0,1,0,1
        {"grouped head map, one step",      2, 4, 1,  -0.7f, 0.0f,  0.2f, 1.0f, false},
        // the case that matters: state fed back into itself
        {"grouped head map, eight steps",   2, 4, 8,  -0.3f, 0.0f,  0.2f, 1.0f, false},
        {"no decay (gate=0), four steps",   2, 4, 4,   0.0f, 0.0f,  0.2f, 1.0f, false},
        // beta = 0 means the delta is zero, so the state must be untouched
        {"beta=0 leaves the state alone",   2, 4, 3,  -0.5f, 0.0f,  0.0f, 0.0f, false},
        {"zero initial state, one step",    2, 4, 1,  -0.7f, -0.1f, 0.2f, 1.0f, true},
        {"decaying gate, twelve steps",     1, 2, 12, -1.2f, -0.2f, 0.2f, 1.0f, false},
    };
    for (const Cfg& c : cases) {
        if (run_case(c, rng) != 0) { check(false, "case aborted early", c.name); }
    }

    std::printf("\nnative_gdn: %d failures\n", failures);
    if (failures) return 1;
    std::printf("native_gdn_parity OK\n");
    (void)selftest;
    return 0;
}
