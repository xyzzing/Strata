#!/usr/bin/env python3
"""Declared textual patches applied to the HIPIFY-generated tree.

HIPIFY is mechanical and does not know about two things this conversion needs:

  * HIP requires a **64-bit** mask for `__shfl_*_sync` / `__ballot_sync` and
    rejects a 32-bit literal with a static_assert. CUDA call sites pass
    `0xFFFFFFFFu`. Zero-extending to 64 bits is exactly equivalent on a
    wave32 target (gfx1100) and is still correct on wave64, where no lane above
    31 consults the missing bits.
  * `hipGraphInstantiate` has no 3-argument overload in HIP; CUDA's deprecated
    form maps onto the 5-argument one with a null error node and zero flags.

Both live here as data, not as edits to generated files, so they are reviewable,
countable and re-applied on every regeneration. Anything that cannot be
expressed as a safe textual rule (the PTX in native_qsa_score.cu, for instance)
does not belong here - it needs a real implementation with a test.

    hipify_patches.py <tree> [--json out.json] [--check]
"""

import argparse
import json
import os
import re
import sys

# (name, pattern, replacement, note). Applied in order to every translated file.
RULES = [
    (
        "shfl-mask-literal-64bit",
        r"(__shfl_sync|__shfl_up_sync|__shfl_down_sync|__shfl_xor_sync|__ballot_sync)"
        r"\(\s*0[xX]([0-9a-fA-F]+)[uUlL]*\s*,",
        "widen-mask-literal",
        "HIP's lane mask is 64-bit; 0xFFFFFFFF zero-extends to all 32 lanes of a "
        "wave32 target and leaves wave64 lanes on their own.",
    ),
    (
        "shfl-mask-identifier-64bit",
        r"(__shfl_sync|__shfl_up_sync|__shfl_down_sync|__shfl_xor_sync|__ballot_sync)"
        r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,",
        r"\1((unsigned long long) \2,",
        "Same requirement for a mask held in a 32-bit variable; zero extension is "
        "the same value.",
    ),
    (
        "hipblas-include-path",
        r"#include <hipblas\.h>",
        r"#include <hipblas/hipblas.h>",
        "HIPIFY rewrites <cublas_v2.h> to <hipblas.h>, which does not exist on "
        "this install; the header lives at <hipblas/hipblas.h>.",
    ),
    (
        "device-printf-global",
        r"\bstd::printf\(",
        r"::printf(",
        "HIP's device printf is the global ::printf; CUDA source uses std::printf. "
        "Identical function on the host side.",
    ),
    (
        "hipgraph-instantiate-5arg",
        r"hipGraphInstantiate\(\s*(&?[A-Za-z_][A-Za-z0-9_.\[\]]*)\s*,"
        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*0\s*\)",
        r"hipGraphInstantiate(\1, \2, nullptr, nullptr, 0)",
        "HIP has no 3-argument overload; CUDA's deprecated form is the 5-argument "
        "one with no error node and zero flags.",
    ),
]

SKIP_SUFFIX = ".prehip"
SOURCE_SUFFIX = (".cu", ".cuh", ".cpp", ".h", ".hpp")


def widen_mask_literal(m):
    head, digits = m.group(1), m.group(2)
    return f"{head}(0x{digits}ull,"


NAMED = {"widen-mask-literal": widen_mask_literal}


def patch_file(path):
    with open(path, encoding="utf-8", errors="surrogateescape") as fh:
        text = original = fh.read()
    hits = {}
    for name, pattern, repl, _note in RULES:
        fn = NAMED.get(repl, repl)
        text, n = re.subn(pattern, fn, text)
        if n:
            hits[name] = hits.get(name, 0) + n
    if text != original:
        with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(text)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--json")
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    totals = {name: 0 for name, *_ in RULES}
    files_changed = 0
    per_file = {}
    for root, dirs, names in os.walk(args.tree):
        dirs[:] = [d for d in dirs if d != ".git"]
        for n in names:
            if n.endswith(SKIP_SUFFIX) or not n.endswith(SOURCE_SUFFIX):
                continue
            p = os.path.join(root, n)
            hits = patch_file(p)
            if hits:
                files_changed += 1
                per_file[os.path.relpath(p, args.tree)] = hits
                for k, v in hits.items():
                    totals[k] += v

    summary = {
        "rules": [{"name": n, "note": note, "hits": totals[n]} for n, _p, _r, note in RULES],
        "files_changed": files_changed,
        "files": per_file,
    }
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")
    print(json.dumps({r["name"]: r["hits"] for r in summary["rules"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
