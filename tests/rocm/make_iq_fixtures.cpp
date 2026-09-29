// tests/rocm/make_iq_fixtures.cpp - build the fixtures `iq_parity` needs.
//
//     make_iq_fixtures <outdir>
//     python3 scripts/rocm/ggufpy_reference.py <outdir>      # overwrites the .f32
//     ./build-hip/cmake/iq_parity <outdir>
//
// WHY TWO PROGRAMS. The upstream generator (`tools/iq_fixture.py`, not published)
// used gguf-py for both halves. That is not possible here: gguf-py implements
// *dequantization* for nine of the ten types and *quantization* for none - the
// i-quant encoders only exist in C. So the work is split where the capability is:
//
//   this program      produces VALID blocks with ggml's own encoders
//   ggufpy_reference  produces the reference VALUES with gguf-py (Python)
//
// which is the split the rollout plan asks for anyway - the reference must not be
// a second copy of the thing under test, and gguf-py is a genuinely separate
// implementation in a different language. This program writes a `.f32` too, from
// ggml's C `to_float`, as a fallback for types gguf-py cannot decode (Q2_0) and
// as something for the Python step to disagree with if either is wrong.
//
// The importance matrix passed to the encoders is uniform (all ones). That is
// deliberate and does not weaken the fixture: the fixture's job is to carry valid
// blocks and their exact decoded values, not to reproduce a particular model's
// quantization quality. An imatrix only steers *which* values get more precision.

#include "ggml.h"
// The encoders are declared in ggml's private header, not the public API:
// ggml.h exposes types and type traits, while the quantize_* entry points
// live in ggml/src/ggml-quants.h. Same library, different visibility.
#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ggml type ids from the pinned ggml.h, with the name iq_parity looks for.
struct Type {
    const char* name;
    int id;
};

const Type kTypes[] = {
    {"IQ2_XXS", 16}, {"IQ2_XS", 17},  {"IQ2_S", 22}, {"IQ3_XXS", 18},
    {"IQ3_S", 21},   {"IQ1_M", 29},   {"IQ4_NL", 20}, {"IQ4_XS", 23},
    {"Q2_0", 42},    {"Q3_K", 11},
};

uint32_t rng_state = 0x2545F491u;
float frand() {                       // uniform in [-1, 1)
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state % 2000001u) - 1000000.0f) / 1000000.0f;
}

bool write_file(const std::string& path, const void* data, size_t bytes, const char* what) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s for %s\n", path.c_str(), what); return false; }
    const size_t wrote = std::fwrite(data, 1, bytes, f);
    std::fclose(f);
    if (wrote != bytes) { std::fprintf(stderr, "short write to %s\n", path.c_str()); return false; }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "artifacts/rocm/iq_fixture";
    // 512 elements per row keeps every type's superblock (QK_K = 256) aligned and
    // gives the kernels more than one block to get wrong.
    const int rows = 8, cols = 512;

    int written = 0, skipped = 0;
    for (const Type& t : kTypes) {
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) t.id);
        if (tt == nullptr) {
            std::printf("%-8s type %2d: ggml has no traits; skipped\n", t.name, t.id);
            ++skipped;
            continue;
        }
        const int blck = (int) tt->blck_size;
        if (cols % blck != 0) {
            std::printf("%-8s type %2d: %d values per row is not a multiple of the block "
                        "size %d; skipped\n", t.name, t.id, cols, blck);
            ++skipped;
            continue;
        }
        ggml_quantize_init((ggml_type) t.id);

        std::vector<float> x((size_t) rows * cols);
        for (auto& v : x) v = frand();

        const size_t row_bytes = (size_t)(cols / blck) * (size_t) tt->type_size;
        std::vector<uint8_t> blocks(row_bytes * (size_t) rows, 0);

        // A uniform importance matrix: valid input, no steering.
        std::vector<float> imatrix((size_t) cols, 1.0f);
        size_t nbytes = 0;
        switch (t.id) {
            case 16: nbytes = quantize_iq2_xxs(x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 17: nbytes = quantize_iq2_xs (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 22: nbytes = quantize_iq2_s  (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 18: nbytes = quantize_iq3_xxs(x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 21: nbytes = quantize_iq3_s  (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 29: nbytes = quantize_iq1_m  (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 20: nbytes = quantize_iq4_nl (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 23: nbytes = quantize_iq4_xs (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            case 42: nbytes = quantize_q2_0   (x.data(), blocks.data(), rows, cols, imatrix.data()); break;
            default:
                if (tt->from_float_ref == nullptr) {
                    std::printf("%-8s type %2d: no encoder available; skipped\n", t.name, t.id);
                    ++skipped;
                    continue;
                }
                for (int r = 0; r < rows; ++r) {
                    tt->from_float_ref(x.data() + (size_t) r * cols,
                                       blocks.data() + (size_t) r * row_bytes, cols);
                }
                nbytes = blocks.size();
                break;
        }
        if (nbytes != blocks.size()) {
            std::printf("%-8s type %2d: encoder produced %zu bytes, expected %zu; skipped\n",
                        t.name, t.id, nbytes, blocks.size());
            ++skipped;
            continue;
        }

        // ggml's own C decode, as the fallback reference and as a second opinion
        // for the Python one.
        std::vector<float> back((size_t) rows * cols);
        if (tt->to_float == nullptr) {
            std::printf("%-8s type %2d: no decoder in ggml; skipped\n", t.name, t.id);
            ++skipped;
            continue;
        }
        for (int r = 0; r < rows; ++r) {
            tt->to_float(blocks.data() + (size_t) r * row_bytes, back.data() + (size_t) r * cols, cols);
        }

        int hdr[3] = {t.id, rows, cols};
        std::vector<uint8_t> bin;
        bin.resize(sizeof hdr);
        std::memcpy(bin.data(), hdr, sizeof hdr);
        bin.insert(bin.end(), blocks.begin(), blocks.end());

        const std::string base = dir + "/" + t.name;
        if (!write_file(base + ".bin", bin.data(), bin.size(), "blocks") ||
            !write_file(base + ".f32", back.data(), back.size() * 4, "reference") ||
            !write_file(base + ".raw", blocks.data(), blocks.size(), "raw blocks")) {
            return 1;
        }
        std::printf("%-8s type %2d  %4d x %5d  blck %-3d  %zu block bytes  written\n",
                    t.name, t.id, rows, cols, blck, blocks.size());
        ++written;
    }

    std::printf("\nmake_iq_fixtures: %d written, %d skipped, into %s\n", written, skipped, dir.c_str());
    std::printf("note: .f32 currently holds ggml's C decode; run\n"
                "      python3 scripts/rocm/ggufpy_reference.py %s\n"
                "      to replace it with gguf-py's independent values where available.\n",
                dir.c_str());
    return written == 0 ? 1 : 0;
}
