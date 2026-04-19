#!/usr/bin/env bash
# Run the rcf-rs side of the external comparison.
# Usage:  ./run_rcf_rs.sh data.csv [trees=100] [sample=256]

set -euo pipefail

CSV="${1:-data.csv}"
TREES="${2:-100}"
SAMPLE="${3:-256}"

cd "$(dirname "$0")/../.."
cargo run --release --example external_bench_driver -- \
    "$CSV" "$TREES" "$SAMPLE"
