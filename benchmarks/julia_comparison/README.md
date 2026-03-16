# WeightedSampling.torch vs WeightedSampling.jl — Benchmark Comparison

This directory contains benchmarks for comparing **WeightedSampling.torch** (Python/PyTorch)
against **WeightedSampling.jl** (Julia).

## Benchmarks

| Benchmark | Variants | Description |
|-----------|----------|-------------|
| **Eight Schools** | Resampling only, With MH moves | Hierarchical normal model (Gelman et al. 2003) |
| **SSM** | Resampling only | Linear-Gaussian state-space model (bootstrap particle filter) |

Each benchmark runs 5 repetitions at 1k, 5k, and 10k particles, reporting the median time.

## Usage

```bash
# Run comparison and write results to results.txt
python benchmarks/julia_comparison/run_comparison.py

# Python benchmarks only
python benchmarks/julia_comparison/run_comparison.py python

# Julia benchmarks only
python benchmarks/julia_comparison/run_comparison.py julia
```

## Requirements

- **Python**: `weighted_sampling` package (this repo, installed)
- **Julia**: [WeightedSampling.jl](https://github.com/MariusFurter/WeightedSampling.jl) installed

## Files

- `eight_schools.py` / `eight_schools.jl` — Eight Schools model benchmarks
- `ssm.py` / `ssm.jl` — State-space model benchmarks
- `run_comparison.py` — Runs all benchmarks and writes `results.txt`
- `run_benchmarks.sh` — Shell script to run benchmarks individually
- `results.txt` — Latest benchmark results (auto-generated)
