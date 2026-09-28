#!/usr/bin/env python3
"""Replace the C-decoded `.f32` fixtures with gguf-py's independent values.

    python3 scripts/rocm/ggufpy_reference.py <fixture-dir> [--json OUT.json]

`iq_parity` compares the ported kernels against `<NAME>.f32`. Those files are
first written by `make_iq_fixtures` from ggml's C `to_float`; this script rewrites
them from **gguf-py**, the Python implementation shipped with the pinned
llama.cpp. That matters: the plan requires the reference not to be a second copy
of the thing under test, and gguf-py is a separate implementation in a different
language from the C kernels being checked.

Types gguf-py has no decoder for keep ggml's C values, and that is reported
rather than left implicit - a fixture whose provenance is unknown is worse than
one whose provenance is stated.

Where both decoders exist their values are compared, and any disagreement is
reported: if ggml and gguf-py disagree about a format, that is a finding about
the format, not a detail to smooth over.

--json writes the provenance of every fixture (which decoder produced each
.f32) so callers can record reference independence in results/ rather than
leaving it implicit in a console log.
"""

import argparse
import json
import os
import sys

import numpy as np

# ggml type id -> gguf-py name. Only the ids the fixture builder emits.
TYPE_NAMES = {
    16: "IQ2_XXS", 17: "IQ2_XS", 22: "IQ2_S", 18: "IQ3_XXS", 21: "IQ3_S",
    29: "IQ1_M", 20: "IQ4_NL", 23: "IQ4_XS", 42: "Q2_0", 11: "Q3_K",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("fixture_dir")
    ap.add_argument("--json", metavar="OUT.json", default="",
                    help="write per-fixture provenance (gguf-py vs ggml-c) here")
    args = ap.parse_args()
    fixture_dir = args.fixture_dir

    try:
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize as gguf_dequantize
    except Exception as exc:                                   # pragma: no cover
        sys.stderr.write(
            f"gguf-py is not importable ({exc}).\n"
            "Set PYTHONPATH to the pinned llama.cpp's gguf-py, e.g.\n"
            "  PYTHONPATH=build-hip/llama.cpp/gguf-py python3 "
            "scripts/rocm/ggufpy_reference.py <dir>\n"
            "The existing .f32 files (ggml's C decode) are left in place.\n")
        return 3

    replaced, kept, agreed, disagreed, missing = [], [], [], [], []
    kept_names = []
    for type_id, name in sorted(TYPE_NAMES.items(), key=lambda kv: kv[1]):
        base = os.path.join(fixture_dir, name)
        bin_path, f32_path = base + ".bin", base + ".f32"
        if not (os.path.exists(bin_path) and os.path.exists(f32_path)):
            missing.append(name)
            continue

        raw = open(bin_path, "rb").read()
        header = np.frombuffer(raw[:12], dtype=np.int32)
        rows, cols = int(header[1]), int(header[2])
        blocks = np.frombuffer(raw[12:], dtype=np.uint8)

        qtype = GGMLQuantizationType(type_id)
        try:
            values = gguf_dequantize(blocks, qtype).astype(np.float32)
        except Exception as exc:
            kept.append(f"{name} (gguf-py: {type(exc).__name__})")
            kept_names.append(name)
            continue
        if values.size != rows * cols:
            kept.append(f"{name} (gguf-py returned {values.size} values, expected {rows * cols})")
            kept_names.append(name)
            continue

        c_values = np.fromfile(f32_path, dtype=np.float32)
        if c_values.size == values.size:
            worst = float(np.max(np.abs(c_values - values))) if values.size else 0.0
            if worst == 0.0:
                agreed.append(name)
            else:
                # Report, do not fail: a format either has one definition or the two
                # decoders differ, and that is worth knowing loudly.
                rel = worst / (float(np.max(np.abs(values))) + 1e-30)
                disagreed.append(f"{name} (max abs {worst:.3e}, rel {rel:.2e})")

        values.tofile(f32_path)
        replaced.append(name)

    print(f"gguf-py reference: {len(replaced)} of {len(TYPE_NAMES)} types replaced")
    if replaced:
        print("  replaced:", ", ".join(replaced))
    if agreed:
        print("  ggml C decode and gguf-py agree exactly on:", ", ".join(agreed))
    if disagreed:
        print("  *** the two decoders DISAGREE on:", "; ".join(disagreed))
    if kept:
        print("  kept ggml's C values (gguf-py cannot decode these):", ", ".join(kept))
    if missing:
        print("  no fixture present for:", ", ".join(missing))
    print("  provenance: .f32 is gguf-py wherever it is listed as replaced, ggml's C "
          "decoder otherwise")

    if args.json:
        replaced_set = set(replaced)
        provenance = {name: ("gguf-py" if name in replaced_set else "ggml-c")
                      for name in sorted(replaced_set | set(kept_names))}
        out = {
            "replaced": replaced, "kept": kept, "agreed": agreed,
            "disagreed": disagreed, "missing": missing,
            "provenance": provenance,
        }
        parent = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(parent, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
