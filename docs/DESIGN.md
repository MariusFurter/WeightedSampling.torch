Overview: A Pyro-like PPL for weighted sampling / sequential monte carlo using PyTorch. Users should be able to write models as standard python functions like

```python
def model(data):
    x = sample("x", dist(args...))
    y = sample("y", dist(args...))
    observe(data, dist(x, ...))
```

We then run the model by giving it as an argument to a inference function:

```python
run_smc(model, model_args..., num_particles)
```

The inference function should maintain a global state that stores a trace which includes the samples for all named variables, along with a tensor of log-weights.

The `sample` function should sample a new site from a `dist` object, append it to the trace, and return **only the samples**. The weights and trace are updated internally in the global state.

The system uses the **`WeightedDistribution` protocol** to interface with distributions. A `DistributionAdapter` wraps standard `torch.distributions,` handling broadcasting and incremental importance weights (which are 0 when proposal = target). Custom distributions (like `ImportanceSampler`) can be used to implement Importance Sampling by providing non-zero weights.

The `observe` function should simply update the log-weights according to the log-pdf value.

This summary consolidates our discussion into a cohesive architecture for a **Vectorized PyTorch-based PPL**.

### 1. Core Architecture: Vectorized Execution

The system follows a **"Single Instruction, Multiple Data" (SIMD)** paradigm. The user writes a standard Python function (the model), but every variable inside that function represents a vector of particles rather than a scalar.

- **Model:** `def model(): ...` runs exactly **once** from top to bottom.
- **Variables:** A variable `x` is a PyTorch tensor of shape `(N, ...)`.
- **State:** The inference algorithm maintains a global state (context) that tracks these vectors.
- **Control Flow:** Standard Python control flow (`if/else`) cannot be used on sampled variables because they are vectors. Conditional logic must be encoded within the `dist` objects (e.g. Mixture distributions) or via vectorized masking.

### 2. The Data Structure: Global Context Stack

To keep the user API clean (`x = sample(...)`), we use a **Global Context Stack** (via `threading.local`) to "catch" the operations performed inside the model.

**The "Dataframe" (Trace):**

- **Storage:** A dictionary `trace = {'x': tensor_x, 'y': tensor_y}`.
- **Weights:** A separate tensor `log_weights` of shape `(N,)`.
- **Consistency:** The -th row of every tensor in `trace` corresponds to the -th particle's history.

### 3. The Primitives

#### A. `sample(name, dist)`

1. **Vectorize:** Calls the wrapped `dist` to get samples and incremental log-weights.
   - If parameters are scalars, it broadcasts to `(N,)`.
   - If parameters are vectors, it maintains the batch shape.
2. **Update State:**
   - Appends samples to the global `trace`.
   - Adds incremental log-weights to the global `log_weights`.
3. **Check ESS:** Checks the Effective Sample Size (ESS) against the threshold. If low, triggers **Resampling**.
4. **Return:** Returns the sample tensor _only_.

#### B. `observe(value, dist)` (or Weighted Sample)

1. **Score:** Computes `log_w = dist.log_prob(value)`.
2. **Update:** Adds `log_w` to the global `log_weights`.
3. **Check ESS:** Checks ESS against the configurable threshold. If low, triggers **Resampling**.

### 4. The Resampling Mechanism

This is where PyTorch shines over JAX/Generators. Because we have mutable state, resampling is efficient and synchronous.

**When Resampling Triggered:**

1. **Draw Ancestors:** Sample indices based on current weights.
2. **Permute History:** Iterate through **every** tensor in the `trace` dictionary and apply the indices: `tensor[:] = tensor[A]`.
3. **Reset:** Set `log_weights` to zero.

### 5. Why PyTorch?

- **Custom Distributions:** You can subclass `torch.distributions`. PyTorch handles **broadcasting** automatically. If `mu` is a vector and `sigma` is a scalar, `Normal(mu, sigma)` correctly creates a batch of distributions.
- **Mutable State:** Unlike JAX, you can append to the `trace` dictionary in real-time without complex functional state-passing.
- **Performance:** You get GPU acceleration for the heavy math (log-probs, sampling) while keeping the flexibility of Python control flow.

### 6. Implementation Skeleton

```python
import torch
import threading
from torch.distributions import Categorical

# 1. Global Context Storage
_SMC_STACK = threading.local()

class SMCContext:
    def __init__(self, num_particles, ess_threshold):
        self.N = num_particles
        self.ess_threshold = ess_threshold
        # The "Dataframe": columns are Tensors of size N
        self.trace = {}
        self.log_weights = torch.zeros(num_particles)

    def resample_if_needed(self):
         # Check ESS and resample if < self.ess_threshold * self.N
         pass

    def resample(self):
        # ... implementation ...
        pass

def sample(name, dist):
    ctx = _SMC_STACK.active_context

    # Wrapped dist logic would go here
    # x, log_w = dist.sample_and_weight(ctx.N)

    # Placeholder for standard dist behavior:
    if len(dist.batch_shape) == 0:
        x = dist.sample((ctx.N,))
    else:
        x = dist.sample()

    # Update state
    ctx.trace[name] = x
    # ctx.log_weights += log_w

    ctx.resample_if_needed()
    return x

def observe(value, dist):
    ctx = get_active_context()
    # Compute weights
    log_w = dist.log_prob(value)
    ctx.log_weights += log_w

    ctx.resample_if_needed()

def run_smc(model_fn, N=1000, ess_threshold=0.5):
    # Setup
    ctx = SMCContext(N, ess_threshold)
    _SMC_STACK.active_context = ctx

    # Run
    model_fn()

    # Teardown
    del _SMC_STACK.active_context
    return ctx.trace

```

### 7. Feature RFC: Metropolis-Hastings Moves

To improve sample diversity and mitigate degeneracy, we want to add an MCMC move step.

**API:**

```python
move(name="x", proposal_fn=dist)
```

- Effect: Perturbs the existing variable `x` using a Metropolis-Hastings kernel leaving the target distribution invariant.
- User Proposal: The user must provide a proposal distribution, e.g. `x_new ~ Normal(x_old, sigma)`.

**Implementation Challenge: Computing Likelihoods**
The primitive `sample("x")` usually has an importance weight of 0 (proposal=prior). The accumulated `log_weights` only track the observation likelihoods. To compute the MH acceptance ratio, we need the **full joint density** `P(x, y)` for both the old value and the proposed value.

**Proposed Solution: Trace Replay**
Since Python models are imperative, we cannot analytically determine which `observe` statements depend on `x`. We must "replay" the model.

1.  **State Capture**: Save `ctx.model_fn` inside the context.
2.  **Double Replay Algorithm**:
    When `move("x")` is called:
    a. **Propose**: Generate `x_new`.
    b. **Replay Old**: Run the model from the start using a `ConditionedContext` where all `sample` sites return their _current_ trace values. - `sample` sites compute their `log_prob` (giving us `P(x_old)`). - `observe` sites compute `P(y|x_old)`. - Result: `log_prob_old` = `P(x, y)`.
    c. **Replay New**: Run the model again using a `ConditionedContext`. - `sample("x")` assumes `x_new`. - Other sites use their original trace values. - Result: `log_prob_new` = `P(x', y)`.
    d. **Accept/Reject**: Compute acceptance ratio `alpha` including the proposal density `Q(x'|x)` correction. Update `ctx.trace` and `ctx.log_weights` for accepted particles.

**Optimization**:
To avoid infinite recursion, `move` calls are ignored (no-op) during Replay execution.

**Crucial Implementation Detail**:
During replay, the `sample` primitive behaves differently. Standard SMC `sample` calls `dist.sample_with_weight()` which returns importance weights (often 0).
In `ConditionedContext`, `sample` must:

1. Retrieve the fixed value `x` from the trace.
2. Call `dist.log_prob(x)` (not `sample_with_weight`) to compute the actual model density.
3. Accumulate this `log_prob` into the context.
