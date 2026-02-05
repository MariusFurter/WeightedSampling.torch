Goal: Create a Pyro-like PPL for weighted sampling / sequential monte carlo using PyTorch. Users should be able to write models as standard python functions like

```python
def model():
    x = sample("x", dist(args...))
    y = sample("y", dist(args...))
    observe(x, dist(args...))
```

We then run the model by giving it as an argument to a inference function:

```python
run_smc(model, model_args..., num_particles)
```

The inference function should maintain a global state that stores a trace which includes the samples for all named variables, along with a tensor of log-weights.

The `sample` function should sample a new site from a `dist` object, append it to the trace, and return **only the samples**. The weights and trace are updated internally in the global state.

The system will **wrap `torch.distributions`** (or use custom distribution classes). A `dist` object is responsible for generating samples and computing the incremental log-weights (importance weights). For standard distributions (proposal = target), the importance weight is 0. If doing importance sampling, the `dist` implementation handles the proposal logic internally.

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

#### B. `observe(dist, data)` (or Weighted Sample)

1. **Score:** Computes `log_w = dist.log_prob(data)`.
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

def observe(dist, value):
    ctx = _SMC_STACK.active_context
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
