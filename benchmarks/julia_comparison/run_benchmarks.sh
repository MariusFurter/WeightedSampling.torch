#!/usr/bin/env bash
# run_benchmarks.sh — Run WeightedSampling.torch vs WeightedSampling.jl benchmarks
#
# Usage:
#   ./benchmarks/julia_comparison/run_benchmarks.sh           # run both
#   ./benchmarks/julia_comparison/run_benchmarks.sh python    # Python only
#   ./benchmarks/julia_comparison/run_benchmarks.sh julia     # Julia only
#
# For a comparison table written to benchmarks/julia_comparison/results.txt, use:
#   python benchmarks/julia_comparison/run_comparison.py

set -euo pipefail
cd "$(dirname "$0")/../.."

LANG="${1:-both}"

header() {
    echo ""
    echo "################################################################"
    echo "# $1"
    echo "################################################################"
    echo ""
}

# ── Python benchmarks ──────────────────────────────────────────────────
run_python() {
    header "WeightedSampling.torch — Eight Schools"
    python benchmarks/julia_comparison/eight_schools.py

    header "WeightedSampling.torch — SSM (Bootstrap PF)"
    python benchmarks/julia_comparison/ssm.py
}

# ── Julia benchmarks ──────────────────────────────────────────────────
run_julia() {
    header "WeightedSampling.jl — Eight Schools"
    julia --project benchmarks/julia_comparison/eight_schools.jl

    header "WeightedSampling.jl — SSM (Bootstrap PF)"
    julia --project benchmarks/julia_comparison/ssm.jl
}

# ── Main ──────────────────────────────────────────────────────────────
case "$LANG" in
    python) run_python ;;
    julia)  run_julia  ;;
    both)
        run_python
        run_julia
        ;;
    *)
        echo "Usage: $0 [python|julia|both]"
        exit 1
        ;;
esac
