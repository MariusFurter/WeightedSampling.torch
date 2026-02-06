# Performance Optimization Log

## Initial Baselines (2026-02-06)

### System Info

- OS: macOS
- Machine: (User machine)

### Benchmarks

**1. Gaussian State Space Model (SMC Loop Overhead)**
_Config: 10,000 particles, 100 timesteps_

- **Execution Time:** ~0.23s
- **Throughput:** ~427 steps/sec
- **Metric:** Total Run Time

**2. MH Linear Regression (Move/Replay Overhead)**
_Config: 1,000 particles, 50 moves_

- **Execution Time:** ~0.16s
- **Throughput:** ~316 moves/sec
- **Metric:** Total Run Time

## Identified Optimization Candidates

### 1. Resampling Loop (`context.py`)

**Issue:** `_resample` iterates over `self.trace.items()` in a Python loop to permute each variable.
**Hypothesis:** For models with many variables (long time horizons), this loop is a bottleneck. Using `torch.gather` on a stacked state or optimizing the dictionary iteration could help. Not trivial since variables have different shapes.
**Status:** **Optimized**. Replaced `Categorical` with `torch.multinomial` and optimized loop iterator.
**Result:** ~7% speedup on SSM.

### 2. Distribution Adapter (`distributions.py`)

**Issue:** `sample_with_weight` uses a `try-except` block to handle broadcasting/expanding distributions.
**Hypothesis:** Exceptions are expensive. Checking `batch_shape` or handling expected shapes explicitly could avoid this overhead for common distributions.
**Status:** **Optimized**. Added shape checks to avoid `expand()`/object creation.
**Result:** Additional ~1.5% speedup on SSM.

### 3. Move Replay (`functional.py`)

**Issue:** `move` performs two full replays (`_replay_trace`). It creates a new `ConditionedContext` and re-runs the model twice per move.
**Hypothesis:** Avoiding shallow copies or optimizing the Context creation could shave off time.
**Status:** **Optimized**. Added optional `track_joint` mode to `SMCContext`. This maintains cumulative `log_prob(x)` eliminating need to replay "old" trace in MH steps.
**Result:** Massive ~46% speedup on MH Benchmark (0.1581s -> 0.1083s). Note: Requires `track_joint=True` to avoid overhead in non-MH models.

### 4. Categorical Sampling (`context.py`)

**Issue:** `Categorical(logits=...).sample()` builds a Distribution object.
**Hypothesis:** `torch.multinomial` on probabilities might be faster and avoid object overhead.
**Status:** **Optimized** (Merged into item 1).

## Final Results (2026-02-06)

**1. Gaussian State Space Model (SMC Only)**

- **Before:** 0.2338s
- **After:** 0.2085s
- **Improvement:** ~11% Speedup

**2. MH Linear Regression (SMC + Moves)**

- **Before:** 0.1581s
- **After:** 0.1083s
- **Improvement:** ~46% Speedup
