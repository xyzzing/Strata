// tests/rocm/core_graph_parity.cpp - does HIP graph capture behave the way the
// engine's design assumes?
//
// `include/strata/core/graph.hpp` bakes in three behaviours that were MEASURED on
// CUDA, and the rollout plan is explicit that CUDA graph behaviour must not be
// assumed to translate: "Do not assume CUDA graph behavior translates
// identically to HIP" (Stage 5). HIPIFY renames cudaGraph* to hipGraph* and
// reports no untranslated construct, which says nothing at all about semantics.
// So each assumption gets a check here:
//
//   1. capture -> instantiate -> launch works, and a capture that produced ZERO
//      nodes is an error rather than an empty graph to replay forever;
//   2. a graph RE-READS its input buffers at replay - so data is live even
//      though the node was recorded once;
//   3. the registry records a key ONCE; the body must not run again on later
//      calls for the same key.
//
// Session 3 adds the two S5.2 residuals:
//
//   4. (numbered 5 below) the PER-STEP-POINTER HAZARD: kernel arguments -
//      pointer VALUES - are frozen at capture. A graph replayed after its
//      buffers were reallocated silently acts on the old addresses. The engine
//      must re-capture when addresses change; this pins that HIP behaves the
//      way that rule assumes.
//   5. (numbered 6 below) replay under concurrency: one graphExec launched on
//      two streams, and two independent graphs on two streams - the per-layer
//      registry pattern - must both be exact.
//
// Assumption 2 is the one that would silently corrupt an engine: if HIP instead
// snapshotted the buffer contents, every replay after the first would compute
// on stale activations and the output would look plausible.
//
// Run: core_graph_parity --selftest

#include "strata/core/graph.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
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

constexpr int kN = 256;

__global__ void fill_from(const float* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: core_graph_parity [--selftest]\n"); return 2; }
    }

    hipDeviceProp_t prop{};
    HIPCHK(hipGetDeviceProperties(&prop, 0));
    std::printf("core_graph_parity on %s (%s)\n", prop.name, prop.gcnArchName);

    hipStream_t stream = nullptr;
    HIPCHK(hipStreamCreate(&stream));

    float *in = nullptr, *out = nullptr;
    HIPCHK(hipMalloc(&in, kN * 4));
    HIPCHK(hipMalloc(&out, kN * 4));
    std::vector<float> host(kN, 0.0f);

    auto upload = [&](float v) {
        for (int i = 0; i < kN; ++i) host[i] = v + (float)i;
        hipMemcpyAsync(in, host.data(), kN * 4, hipMemcpyHostToDevice, stream);
        hipStreamSynchronize(stream);
    };
    auto download = [&]() {
        std::vector<float> got(kN, -1.0f);
        hipMemcpy(out, got.data(), 0, hipMemcpyDeviceToDevice);   // no-op, keeps the intent explicit
        hipMemcpyAsync(got.data(), out, kN * 4, hipMemcpyDeviceToHost, stream);
        hipStreamSynchronize(stream);
        return got;
    };

    // ---- 1. capture, instantiate, launch ----------------------------------
    {
        strata::core::CapturedGraph g;
        std::string err;
        upload(1.0f);
        const bool began = g.begin(stream, err);
        check(began, "graph capture must start", err.c_str());
        if (began) {
            fill_from<<<(kN + 63) / 64, 64, 0, stream>>>(in, out, kN);
            const bool ended = g.end(stream, err);
            check(ended, "capture must instantiate", err.c_str());
            check(g.nodes() > 0, "a captured body must produce at least one node");
            check(g.valid(), "the captured graph must be launchable");
            if (ended && g.valid()) {
                const bool launched = g.launch(stream, err);
                check(launched, "replay must launch", err.c_str());
                check(g.wait_ms(2000), "wait_ms must observe completion, not time out");
                const std::vector<float> got = download();
                int wrong = 0;
                for (int i = 0; i < kN; ++i) if (got[i] != host[i] * 2.0f + 1.0f) ++wrong;
                check(wrong == 0, "replayed result must equal the recorded computation");
                std::printf("  capture -> launch        nodes=%zu  wrong=%d\n", g.nodes(), wrong);
            }
        }
    }

    // ---- 2. the data path is live across replays --------------------------
    {
        strata::core::CapturedGraph g;
        std::string err;
        upload(1.0f);
        g.begin(stream, err);
        fill_from<<<(kN + 63) / 64, 64, 0, stream>>>(in, out, kN);
        const bool ended = g.end(stream, err);
        check(ended, "second capture must instantiate", err.c_str());
        if (ended && g.valid()) {
            int stale = 0;
            for (int trial = 0; trial < 3; ++trial) {
                const float base = 10.0f * (float)(trial + 1);
                upload(base);                       // same device address, new contents
                g.launch(stream, err);
                g.wait_ms(2000);
                const std::vector<float> got = download();
                for (int i = 0; i < kN; ++i) {
                    if (got[i] != (base + (float)i) * 2.0f + 1.0f) ++stale;
                }
            }
            // If HIP snapshotted the buffers at capture, trials 2 and 3 would
            // still hold trial 1's answer - 2*kN wrong values, not zero.
            check(stale == 0, "a replay must re-read the input buffer, not a snapshot of it");
            std::printf("  data path live          3 replays with new input, stale values=%d\n", stale);
        }
    }

    // ---- 3. the registry records a key once -------------------------------
    {
        strata::core::GraphRegistry reg(stream);
        std::string err;
        upload(1.0f);
        int body_runs = 0;
        for (int call = 0; call < 4; ++call) {
            const bool ok = reg.record(strata::core::LayerType::GDN, 7, [&]() {
                ++body_runs;
                fill_from<<<(kN + 63) / 64, 64, 0, stream>>>(in, out, kN);
            }, err);
            check(ok, "registry record must succeed", err.c_str());
        }
        check(body_runs == 1, "the body must be captured exactly once across four calls");
        check(reg.size() == 1, "one key must produce one graph");
        check(reg.captures() == 1, "captures() must count body captures, not calls");
        const strata::core::CapturedGraph* found = reg.find(strata::core::LayerType::GDN, 7);
        check(found != nullptr && found->valid(), "the recorded key must be findable");
        check(reg.find(strata::core::LayerType::QSA, 7) == nullptr,
              "a different layer type must not hit the same key");

        upload(5.0f);
        const bool launched = reg.launch(strata::core::LayerType::GDN, 7, 2000, err);
        check(launched, "registry launch must succeed", err.c_str());
        const std::vector<float> got = download();
        int wrong = 0;
        for (int i = 0; i < kN; ++i) if (got[i] != (5.0f + (float)i) * 2.0f + 1.0f) ++wrong;
        check(wrong == 0, "a registry replay must compute on the newest data");
        std::printf("  registry                 body_runs=%d size=%zu captures=%zu wrong=%d\n",
                    body_runs, reg.size(), reg.captures(), wrong);
    }

    // ---- 4. an empty capture is an error, not an empty graph --------------
    {
        strata::core::CapturedGraph g;
        std::string err;
        g.begin(stream, err);
        const bool ended = g.end(stream, err);          // nothing launched in between
        check(!ended, "a capture with zero nodes must be reported as an error");
        std::printf("  empty capture            reported as %s (expected an error)\n",
                    ended ? "success" : "error");
    }

    // ---- 5. the per-step-pointer hazard: kernel ARGUMENTS are frozen -------
    // The engine's known trap (TASKS S5.2): a graph re-reads buffer CONTENTS
    // (case 2) but does not re-read POINTER VALUES. If buffers are reallocated
    // (arena reset, a different layer's tensors) and the graph is replayed, it
    // silently operates on the OLD addresses. This pins the semantics so the
    // engine's rule ("re-capture when addresses change") rests on a measurement.
    {
        strata::core::CapturedGraph g;
        std::string err;
        upload(1.0f);
        g.begin(stream, err);
        fill_from<<<(kN + 63) / 64, 64, 0, stream>>>(in, out, kN);
        const bool ended = g.end(stream, err);
        check(ended, "hazard capture must instantiate", err.c_str());
        if (ended && g.valid()) {
            // New buffers at DIFFERENT addresses with different contents.
            float *in2 = nullptr, *out2 = nullptr;
            HIPCHK(hipMalloc(&in2, kN * 4));
            HIPCHK(hipMalloc(&out2, kN * 4));
            std::vector<float> host2(kN);
            for (int i = 0; i < kN; ++i) host2[i] = 100.0f + (float)i;
            HIPCHK(hipMemcpy(in2, host2.data(), kN * 4, hipMemcpyHostToDevice));
            HIPCHK(hipMemset(out2, 0, kN * 4));

            g.launch(stream, err);
            g.wait_ms(2000);

            std::vector<float> got_old(kN, -1.0f), got_new(kN, -1.0f);
            HIPCHK(hipMemcpy(got_old.data(), out, kN * 4, hipMemcpyDeviceToHost));
            HIPCHK(hipMemcpy(got_new.data(), out2, kN * 4, hipMemcpyDeviceToHost));
            int old_updated = 0, new_touched = 0;
            for (int i = 0; i < kN; ++i) {
                if (got_old[i] == (1.0f + (float)i) * 2.0f + 1.0f) ++old_updated;
                if (got_new[i] != 0.0f) ++new_touched;
            }
            // Frozen arguments: the replay read/wrote the ORIGINAL addresses even
            // though newer data sat in in2/out2. (If a future HIP version made
            // graphs pointer-indirect, this check FAILS loudly - which is the
            // point: the engine's re-capture rule would then be optional.)
            check(old_updated == kN, "the replay must act on the captured addresses");
            check(new_touched == 0,
                  "the replay must NOT touch buffers allocated after capture "
                  "(if this fails, HIP graphs are pointer-indirect and S5.2's hazard re-opens)");
            std::printf("  pointer freeze           old_updated=%d/%d new_touched=%d "
                        "(arguments frozen at capture, as the engine assumes)\n",
                        old_updated, kN, new_touched);
            hipFree(in2);
            hipFree(out2);
        }
    }

    // ---- 6. replay under concurrency ---------------------------------------
    // 6a: the SAME graph launched on two streams back-to-back. The question the
    // engine cares about: does a concurrent replay corrupt the buffer, or does
    // HIP serialize executions of one graphExec?
    // 6b: two DIFFERENT graphs on two streams with disjoint buffers - the
    // per-layer registry pattern the engine actually uses.
    {
        strata::core::CapturedGraph g;
        std::string err;
        upload(3.0f);
        g.begin(stream, err);
        fill_from<<<(kN + 63) / 64, 64, 0, stream>>>(in, out, kN);
        const bool ended = g.end(stream, err);
        check(ended, "concurrency capture must instantiate", err.c_str());
        if (ended && g.valid()) {
            hipStream_t s2 = nullptr;
            HIPCHK(hipStreamCreate(&s2));
            upload(3.0f);                       // stable input for both launches
            const bool l1 = g.launch(stream, err);
            const bool l2 = g.launch(s2, err);
            check(l1 && l2, "back-to-back launches on two streams must both be accepted");
            HIPCHK(hipStreamSynchronize(stream));
            HIPCHK(hipStreamSynchronize(s2));
            const std::vector<float> got = download();
            int wrong = 0;
            for (int i = 0; i < kN; ++i)
                if (got[i] != (3.0f + (float)i) * 2.0f + 1.0f) ++wrong;
            // Measures non-corruption under back-to-back two-stream replay; HIP may
            // serialize executions of one graphExec or overlap them idempotently -
            // either way the buffer must hold exactly the expected values.
            check(wrong == 0, "concurrent replay of one graph must not corrupt its buffers");
            std::printf("  same graph, 2 streams    l1=%d l2=%d wrong=%d\n", l1, l2, wrong);
            hipStreamDestroy(s2);
        }
    }
    {
        // Capture is PER-STREAM (a body must launch on the stream the registry
        // is bound to, or the capture sees zero nodes - measured the hard way),
        // so two graphs on two streams means two registries. launch() takes any
        // stream, which is what the engine uses to replay on the active
        // request's stream.
        strata::core::GraphRegistry reg1(stream);
        hipStream_t s2 = nullptr;
        HIPCHK(hipStreamCreate(&s2));
        strata::core::GraphRegistry reg2(s2);
        std::string err;
        // layer 1 graph: in -> out. layer 2 graph: in2 -> out2, disjoint buffers.
        float *in2 = nullptr, *out2 = nullptr;
        HIPCHK(hipMalloc(&in2, kN * 4));
        HIPCHK(hipMalloc(&out2, kN * 4));
        std::vector<float> host2(kN);
        for (int i = 0; i < kN; ++i) host2[i] = 50.0f + (float)i;

        upload(1.0f);
        int runs1 = 0, runs2 = 0;
        const bool ok1 = reg1.record(strata::core::LayerType::GDN, 1, [&]() {
            ++runs1;
            fill_from<<<(kN + 63) / 64, 64, 0, stream>>>(in, out, kN);
        }, err);
        HIPCHK(hipMemcpy(in2, host2.data(), kN * 4, hipMemcpyHostToDevice));
        const bool ok2 = reg2.record(strata::core::LayerType::QSA, 2, [&]() {
            ++runs2;
            fill_from<<<(kN + 63) / 64, 64, 0, s2>>>(in2, out2, kN);
        }, err);
        check(ok1 && ok2, "two-graph registry capture must succeed", err.c_str());
        if (ok1 && ok2) {
            const bool l1 = reg1.launch(strata::core::LayerType::GDN, 1, 2000, err);
            const bool l2 = reg2.launch(strata::core::LayerType::QSA, 2, 2000, err);
            check(l1 && l2, "concurrent registry launches must succeed");
            HIPCHK(hipStreamSynchronize(stream));
            HIPCHK(hipStreamSynchronize(s2));
            std::vector<float> got1(kN, -1.0f), got2(kN, -1.0f);
            HIPCHK(hipMemcpy(got1.data(), out, kN * 4, hipMemcpyDeviceToHost));
            HIPCHK(hipMemcpy(got2.data(), out2, kN * 4, hipMemcpyDeviceToHost));
            int wrong = 0;
            for (int i = 0; i < kN; ++i) {
                if (got1[i] != (1.0f + (float)i) * 2.0f + 1.0f) ++wrong;
                if (got2[i] != (50.0f + (float)i) * 2.0f + 1.0f) ++wrong;
            }
            check(wrong == 0, "two graphs replayed concurrently must both be exact");
            std::printf("  two graphs, 2 streams    runs=%d/%d wrong=%d\n", runs1, runs2, wrong);
        }
        hipStreamDestroy(s2);
        hipFree(in2);
        hipFree(out2);
    }

    hipFree(in);
    hipFree(out);
    hipStreamDestroy(stream);

    std::printf("\ncore_graph: %d failures\n", failures);
    if (failures) return 1;
    std::printf("core_graph_parity OK\n");
    (void)selftest;
    return 0;
}
