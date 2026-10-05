#!/usr/bin/env python3
"""tools/qsa_locality.py - analyzer for the engine's STRATA_QSA_DUMP selection dumps (prefill.cpp:2200).

The dump is a flat sequence of records: int32 {qsa_layer, pos0, T, cap} followed by
T*cap int32 selected-cell ids. This script reports per-layer selection volume, unique
cells (cross-query reuse), and the logical vs unique read bytes, so the attention
kernel's achieved bandwidth and locality can be compared against the DRAM peak.

Note (session 23 anomaly, unresolved): records carried 4x T*cap cells against their
header T — the script counts ACTUAL cells read, never the header's claim.
"""
import array
import struct
import sys

BYTES_PER_CELL = 2 * 256 + (256 // 64) * 2 * 2 + 4  # K+V int8 (HD=256) + per-64 fp16 scales + sel_id


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "-"
    f = sys.stdin if path == "-" else open(path, "rb")
    per_layer = {}
    nrec = 0
    while True:
        hdr = f.read(16)
        if len(hdr) < 16:
            break
        layer, pos0, T, cap = struct.unpack("<4i", hdr)
        cells = array.array("i")
        cells.fromfile(f, T * cap)
        d = per_layer.setdefault(layer, {"recs": 0, "cells": 0, "uniq": 0})
        d["recs"] += 1
        d["cells"] += len(cells)
        d["uniq"] += len(set(cells))
        nrec += 1
    print(f"records: {nrec}, BYTES_PER_CELL: {BYTES_PER_CELL}")
    tot_log, tot_uniq = 0, 0
    for layer in sorted(per_layer):
        d = per_layer[layer]
        tot_log += d["cells"] * BYTES_PER_CELL
        tot_uniq += d["uniq"] * BYTES_PER_CELL
        print(f"layer {layer:2d}: recs={d['recs']} selections={d['cells']:,} "
              f"unique_cells={d['uniq']:,} reuse={d['cells'] / max(d['uniq'], 1):.1f}x")
    print(f"logical reads (no reuse):  {tot_log / 2**30:8.2f} GiB")
    print(f"unique-cell reads (floor): {tot_uniq / 2**30:8.2f} GiB")


if __name__ == "__main__":
    main()
