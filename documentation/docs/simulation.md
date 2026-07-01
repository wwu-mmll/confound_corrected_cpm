# Simulating Data

CCCPM ships a **structural-equation-model (SEM) based simulator** that builds
connectome-like data with an *analytically known* ground truth. It lets you dial in
exactly how strong the brain–outcome association is and how much of it is spurious
(driven by a confound), which is what makes it useful for validating a
confound-aware analysis: you know the right answer before you run the model.

The simulator lives in `cccpm.simulation.simulate_sem`. The two entry points are
[`simulate_data_given_R2`](#specifying-r2-directly) and
[`simulate_data_given_kappa`](#specifying-the-spurious-fraction-kappa).

## The generative model

A single latent confound `Z` is a **common cause** of both the connectome `X` and
the outcome `y`:

```
        Z  (confound: e.g. age, head motion, site)
       / \
      v   v
      X    y      X also has a genuine, direct effect on y
       \___^
```

Because `Z` drives both, a naive correlation between `X` and `y` is **inflated** — it
mixes the genuine brain–outcome effect with a spurious path through the confound. The
simulator controls exactly how large each piece is.

Three quantities fully describe the setup:

| Quantity | Meaning |
|----------|---------|
| `R2_X_y` | **Naive** R² of `y ~ X` — the *apparent* (inflated) brain–outcome strength. |
| `R2_X_y_given_Z` | **Unique / deconfounded** R² of `X` after adjusting for `Z` — the *true* effect you hope to recover. |
| `R2_Z_y` | R² of `y ~ Z` — how much the confound alone explains. |

## Feature (edge) classes

The connectome columns are split into four interpretable classes so you can see
*which* edges an analysis keeps or drops:

| Class | Argument | Correlated with `y`? | Correlated with confound? |
|-------|----------|:--:|:--:|
| **mixed** | `n_features_informative` | yes | yes — carries both genuine signal **and** confound leakage |
| **pure-signal** | `n_pure_signal_features` | yes | no — genuine signal, orthogonal to the confound |
| **confound-only** | `n_confound_only_features` | only *through* the confound | yes |
| **noise** | (the remainder) | no | no |

This separation is what exposes the limit of partial-correlation edge selection: it
rejects **confound-only** edges (their partial association with `y` is null) but keeps
**mixed** edges — whose raw values still leak confound variance into the network score.
The ground-truth column indices for each class are returned in the `info` dict
(`mixed_idx`, `pure_signal_idx`, `confound_only_idx`, `noise_idx`).

!!! note "Valid connectome sizes"
    `n_features` must be a valid number of upper-triangular edges of a symmetric
    node-by-node matrix, i.e. `n_features = n_nodes * (n_nodes - 1) / 2`. For example
    435 edges = 30 nodes, 1225 edges = 50 nodes. CCCPM raises a clear error (with the
    nearest valid sizes) if you pass an invalid count.

## What you get back

Every simulator call returns a dict:

| Key | Shape | Meaning |
|-----|-------|---------|
| `X` | `(n_samples, n_features)` | the simulated connectome (feed as `X`) |
| `y` | `(n_samples, 1)` | the continuous outcome (feed as `y`) |
| `Z` | `(n_samples, n_confounds)` | the confounds (feed as `covariates`) |
| `true_X` | `(n_samples, 1)` | the latent predictor behind the mixed edges |
| `info` | dict | ground-truth R² targets, edge-class indices, `corr_true_X_Z`, etc. |

So plugging the simulator into [`CPMAnalysis`](api/cpm_regression.md) is direct:

```python
from cccpm.simulation.simulate_sem import simulate_data_given_kappa

sim = simulate_data_given_kappa(R2_X_y=0.4, kappa=0.3, n_features=435, n_samples=200)
X, y, covariates = sim["X"], sim["y"], sim["Z"]
```

## Specifying R² directly

Use `simulate_data_given_R2` when you want to pin all three R² components yourself.
The simulator solves for the brain–confound coupling that reproduces your targets and
warns (rather than crashing) if an exact match is infeasible:

```python
from cccpm.simulation.simulate_sem import simulate_data_given_R2, compute_r2s

sim = simulate_data_given_R2(
    R2_X_y=0.25,          # apparent (naive) brain–outcome R²
    R2_X_y_given_Z=0.15,  # true, deconfounded R²
    R2_Z_y=0.10,          # confound–outcome R²
    n_features=435,       # 30-node connectome
    n_features_informative=40,
    n_pure_signal_features=20,
    n_confound_only_features=20,
    n_samples=2000,
    random_state=0,
)

# Empirically check the generated data matches the analytic targets:
print(compute_r2s(sim))   # r2_naive ≈ 0.25, r2_unique_X ≈ 0.15, ...
```

## Specifying the spurious fraction (kappa)

Often the cleanest knob is **how much of the apparent signal is fake**. `kappa` is the
fraction of the naive R² that is confound-driven:

```
R2_X_y_given_Z = (1 - kappa) * R2_X_y     # true, deconfounded value
R2_Z_y         =      kappa  * R2_X_y     # spurious part
```

- `kappa = 0` → no confounding (true == naive),
- `kappa = 1` → fully spurious (true effect → 0).

```python
from cccpm.simulation.simulate_sem import simulate_data_given_kappa

sim = simulate_data_given_kappa(
    R2_X_y=0.4,   # apparent brain–outcome R²
    kappa=0.3,    # 30% of it is confound-driven → true R² is 0.7 * 0.4 = 0.28
    n_features=435,
    n_samples=200,
    random_state=42,
)
```

This is the parameterisation used in the two [quickstart examples](examples/regression.md)
and in `examples/confound_inflation_demo.py`.

## Sweeping a grid of scenarios

`generate_confound_grid` builds a whole matrix of scenarios — rows sweep the
brain–outcome strength `R2_X_y`, columns sweep `kappa` — each with an independent,
reproducible seed. This is the backbone of the confound-inflation demonstration:

```python
from cccpm.simulation.simulate_sem import generate_confound_grid, compute_r2s

grid = generate_confound_grid(n_samples=2000, random_state=42)
for (r2, kappa), sim in grid.items():
    r2s = compute_r2s(sim)
    print(r2, kappa, sim["info"]["R2_X_y_given_Z"], r2s["r2_unique_X"])
```

For a full worked example that runs CCCPM across such a grid and shows that
partial-correlation edge selection does **not** remove the inflation (only
residualization does), see `examples/confound_inflation_demo.py`.

## Making the target binary

The SEM simulator produces a continuous outcome. For a classification example, split
it at the median to get a balanced 0/1 label (the confound remains baked into the
connectome):

```python
import numpy as np

y = (sim["y"].ravel() > np.median(sim["y"])).astype(float)
```

## API reference

See the [Simulation API reference](api/simulation.md) for the full signatures and
every parameter.

## See also

- [Regression example](examples/regression.md) — uses `simulate_data_given_kappa`.
- [Classification example](examples/classification.md) — same, with a binarised target.
- [How CCCPM Works](methods.md) — the confound-control methods the simulator exercises.
