# Changelog

## 0.1.2 — 2026-03-22

### Improvements

- **Summary table redesign:** Switched from multi-row block histograms to compact single-row sparkline histograms.
- **Column separators:** Summary table now uses `│` vertical bars to clearly delineate variable columns.

## 0.1.1 — 2026-03-22

### Features

- **`SMCResult` pretty-printing:** `repr(result)` now shows particle count, log-evidence, ESS, and variable shapes.
- **`SMCSummary` class:** `summary()` and `result.summary()` return an `SMCSummary` object that prints as a formatted table with Unicode marginal histograms above each column.
- **`result.print_summary()`** method for explicit tabular output with configurable `num_bins`.
- **`result.norm_weights`** property returning normalized (non-log) importance weights via `torch.softmax`.
- **`summary` and `expectation` are now primary methods** on `SMCResult`; the standalone functions are thin wrappers.

## 0.1.0 — 2026-03-16

Initial public release.

### Features

- **Vectorized SMC inference** via `run_smc` — all particles execute the model simultaneously using PyTorch broadcasting.
- **Primitives:** `sample`, `observe`, `deterministic`, and `move` for defining probabilistic models.
- **ESS-triggered multinomial resampling** with O(1) buffer permutation.
- **Metropolis-Hastings moves:** `RandomWalkProposal` and `AdaptiveProposal` (particle-estimated covariance) for post-resampling rejuvenation.
- **Importance sampling** via `ImportanceSampler` with automatic log-weight accumulation.
- **Discrete models:** `DiscreteConditional` for lazy, memoized conditional probability tables.
- **Analysis utilities:** `expectation` and `summary` for weighted posterior statistics.
- **`@model` decorator** for calling models directly with SMC arguments.
