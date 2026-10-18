#!/usr/bin/env python3
"""r30-routing_stats.py - tokens-per-expert statistics from a `--dump-routing` trace.

Record format (src/program/generate.cpp, drive_pool / drive_pool_multi):
    int32 layer, int32 k, k * int32 expert ids, k * float32 weights
one record per (layer, position-or-window-row). The trace is the *routed* set, so
"entries" below are the number of times an expert was routed in the run.

Usage: r30-routing_stats.py <routing.bin>
"""
import collections
import struct
import sys


def main(path):
    per_layer = collections.defaultdict(collections.Counter)
    n_rec = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            layer, k = struct.unpack("<ii", head)
            if not 0 <= layer < 64 or not 0 < k <= 32:
                print(f"bad record at #{n_rec}: layer={layer} k={k} - stopping", file=sys.stderr)
                break
            ids = struct.unpack(f"<{k}i", f.read(4 * k))
            f.read(4 * k)  # weights
            c = per_layer[layer]
            for e in ids:
                c[e] += 1
            n_rec += 1

    if not n_rec:
        print("no records")
        return 1

    tot = sum(sum(c.values()) for c in per_layer.values())
    uniq = sum(len(c) for c in per_layer.values())
    print(f"records {n_rec}, layer-entries {tot}, distinct (layer,expert) {uniq}, "
          f"mean entries/expert {tot / max(1, uniq):.2f}")

    # distribution over (layer, expert) pairs - never pool expert ids across layers
    sizes = sorted(n for c in per_layer.values() for n in c.values())
    if sizes:
        def pct(p):
            return sizes[min(len(sizes) - 1, int(p * len(sizes)))]
        print(f"entries/expert: min {sizes[0]}, p50 {pct(0.50)}, p90 {pct(0.90)}, "
              f"p99 {pct(0.99)}, max {sizes[-1]}")
        top = sizes[-max(1, len(sizes) // 100):]
        print(f"top-1% experts carry {100 * sum(top) / max(1, sum(sizes)):.1f}% of entries")

    for l in sorted(per_layer)[:4]:
        c = per_layer[l]
        top = c.most_common(8)
        print(f"  layer {l}: {len(c)} experts, {sum(c.values())} entries, top {[(e, n) for e, n in top]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) > 1 else main("/dev/stdin"))
