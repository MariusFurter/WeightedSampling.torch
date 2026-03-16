# Changelog

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
