#!/bin/sh
# strata-rocm tune - paired flag tuning with holdout, per the optimization
# contract in the port-strata-hip skill. Stops the resident strata-8081 for the
# duration (one GPU) and restarts it at the end. The only thing it may change
# is the serve config (a .bak is kept); correctness gates are never touched.
#
# Usage: strata-rocm tune --set prefill=auto [--set flag] [--dry-run]
#       [--workload <json>] [--min-gain 0.03] [--pairs 2]

. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/lib/common.sh"

python3 "$SCRIPTS_DIR/lib/tune.py" --config "$ART_DIR/serve/strata-hip.json" "$@"
