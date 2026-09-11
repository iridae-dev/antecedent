# Local targets, outcome distributions, and joint interventions

This walkthrough uses the 1.5.0 Python API. Version 1.5.0 is in release
preparation; run it against the built 1.5 checkout until that release is published.
Run the Python blocks below in order in one interpreter. They use NumPy and
Antecedent, with no optional causal-learning package.

## Prepare once, change the target population

The synthetic effect is `2 + 0.5*z`. Weighting toward larger `z` therefore changes
which population's average effect we estimate. The DAG asserts that `z` suffices
for adjustment; data alone do not establish that assumption.

```python
import numpy as np
import antecedent
from antecedent.estimation import PreparedAnalysis
from antecedent.intervention import Set
from antecedent.query import ExceedanceGrid, InterventionResponse, Quantile

rng = np.random.default_rng(150)
n = 2000
z = rng.normal(size=n)
probability = 1.0 / (1.0 + np.exp(-0.5 * z))
t = (rng.uniform(size=n) < probability).astype(float)
y = (2.0 + 0.5 * z) * t + z + rng.normal(size=n)
data = {"z": z, "t": t, "y": y}
graph = antecedent.Dag.from_edges(
    ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]
)
plan = PreparedAnalysis.prepare(
    data,
    graph=graph,
    query=antecedent.AverageEffect("t", "y"),
    estimator="aipw",
    refute="none",
    bootstrap=0,
)
weights = np.exp(-0.5 * ((z - 0.5) / 0.8) ** 2)
all_observed = plan.retarget(np.ones(n), depends_on=[])
local = plan.retarget(weights, depends_on=["z"])
print("AllObserved ATE:", all_observed.estimate.ate)
print("Weighted ATE:", local.estimate.ate)
print("Weighted analytic SE:", local.estimate.se_analytic)
assert np.isfinite(local.estimate.ate)
```

`retarget` averages frozen cross-fitted scores with the supplied weights. It does
not refit nuisances. Nonconstant weights require declared parents in the certified
adjustment set; treatment and directed descendants are not allowed. This is
standardization within one population, not cross-environment transport or an
individual treatment-effect model. Inference assumes positivity and appropriate
nuisance convergence rates. `refute="none"` keeps this example focused on the
estimator; it does not establish that the causal assumptions hold.

Weights must align with the retained score rows. Here all rows are complete. If
rows were dropped, inspect `plan.estimate(data).estimate.score_table.row_index`
and align weights to those original row indices. `estimate(new_data)` refits on
new data under the prepared identification contract; `refresh(new_data)` also
replaces retained data and scores after success.

## Read a CDF and its supported bands

An exceedance grid estimates the outcome distribution under each treatment arm.
It does not reduce several thresholds to one scalar ATE. The last threshold below
is deliberately beyond all observed outcomes to demonstrate unsupported tails.

```python
cdf_plan = PreparedAnalysis.prepare(
    data,
    graph=graph,
    query=antecedent.AverageEffect(
        "t", "y", outcome_functional=ExceedanceGrid([0.0, 2.0, 1_000_000.0])
    ),
    estimator="aipw",
    refute="none",
    bootstrap=0,
)
cdf_result = cdf_plan.retarget(weights, depends_on=["z"])
estimate = cdf_result.estimate
assert np.isnan(estimate.ate)
assert estimate.score_table is not None
assert estimate.score_inference is not None
assert estimate.exceedance_cdf is not None
bands = estimate.score_inference
for (arm, threshold), projected, raw, lower, upper, supported in zip(
    estimate.score_table.columns,
    estimate.exceedance_cdf,
    bands.raw_means,
    bands.lower,
    bands.upper,
    bands.threshold_supported,
    strict=True,
):
    print(arm, threshold, "CDF:", projected, "raw CDF:", 1.0 - raw)
    if supported:
        print("raw-CDF simultaneous band:", 1.0 - upper, 1.0 - lower)
    else:
        print("band unavailable: insufficient tail or overlap support")
        # Do not present the numerical bounds when support is false.
assert not all(bands.threshold_supported)
```

For this binary query, arm 0 is control and arm 1 is active. Use the column metadata
rather than assuming an array order. `exceedance_cdf` reports `F_a(c)=P(Y(a)≤c)`;
`monotone_rearranged` indicates whether projection changed the raw estimates.
For this AIPW score-table path, `score_inference.raw_means` and its bounds are
**exceedance** coordinates, `P(Y(a)>c)`. Convert a raw mean with `1-mean` and an
interval with `[1-upper, 1-lower]`, as above, to obtain CDF coordinates. Complementing
all coordinates leaves their joint covariance unchanged. The conditional-grid
path already publishes CDF-coordinate bands and must not be complemented again.
Neither path supplies uncertainty for the projected CDF. Always honor
`threshold_supported`: unsupported bounds may be NaN or finite placeholders, and
must not be displayed as confidence intervals.

For a single `Exceedance(c)` AverageEffect query, the scalar effect is the active
minus control exceedance probability. For a quantile effect, use `Quantile(tau)`
with explicit Frequentist AIPW on a DAG or CoDetermined closure. Its uncertainty
conditions on the finite CDF grid and excludes interpolation bias and grid
selection. Inversion refuses unsupported, flat, or projection-altered crossings;
the request must use `refute="none"` because mean-effect refuters do not validate
quantiles. The licensed query families are:

| Query | Quantile meaning | Estimation contract |
| --- | --- | --- |
| `AverageEffect` | Active minus control quantile | Explicit Frequentist `aipw`, DAG or CoDetermined |
| Binary `ConditionalEffect` with one modifier | Difference of quantiles of arm CDFs standardized over the retained modifier distribution | Frequentist DAG, CPDAG, or PAG; aligned atom influences and zero unidentified mass |
| Discrete joint `InterventionResponse` | Quantile level in the requested joint cell | Explicit Frequentist `cell.aipw`, DAG or CoDetermined |

These start from AllObserved. Average and joint score-table plans also support
prepared retargeting; conditional plans do not export retargetable score tables.
The conditional `estimate.functional.quantile_grid` diagnostic lists the actual
thresholds; CDF coordinates are threshold-major, control then active.
Conditional quantiles are not a pointwise modifier surface or an average of
individual conditional quantiles. Graph mixtures combine CDFs before inversion.
Other estimator families remain refused: reusing a mean regression, matching
estimate, IV estimate, or Bayesian mean posterior would not supply the required
CDF influence function and density. Supporting those would require a separate
validated distributional estimator contract.

```python
conditional_median = antecedent.analyze(
    data,
    graph=graph,
    query=antecedent.ConditionalEffect("t", "y", "z", outcome_functional=Quantile(0.5)),
    refute="none",
    bootstrap=0,
)
print("Modifier-standardized median effect:", conditional_median.estimate.ate)
```

## Estimate a non-additive joint response

A joint intervention requests a response level, not an ATE. This model contains
an interaction, so an additive response model would miss part of its structure.
The two treatments below are binary and all four joint cells have observations.
This Python example uses a DAG with a common observed adjustment variable.

```python
t2 = (rng.uniform(size=n) < probability).astype(float)
y_joint = 1.2 * t + 0.8 * t2 + 1.5 * t * t2 + 0.5 * z + rng.normal(size=n)
joint_data = {"z": z, "t1": t, "t2": t2, "y": y_joint}
joint_graph = antecedent.Dag.from_edges(
    ["z", "t1", "t2", "y"],
    [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")],
)
joint = antecedent.analyze(
    joint_data,
    graph=joint_graph,
    query=InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)]),
    estimator="cell.aipw",
    refute="none",
    bootstrap=0,
)
print("E[Y | do(t1=1, t2=1)]:", joint.estimate)
print("Analytic SE:", joint.uncertainty.standard_error)
assert np.isfinite(joint.estimate)
assert joint.uncertainty.standard_error is not None
```

For a joint quantile level, pass the functional on the response query:

```python
joint_median = antecedent.analyze(
    joint_data,
    graph=joint_graph,
    query=InterventionResponse(
        "y", intervention=[Set("t1", 1.0), Set("t2", 1.0)],
        outcome_functional=Quantile(0.5),
    ),
    estimator="cell.aipw",
    refute="none",
    bootstrap=0,
)
print("Median under the joint intervention:", joint_median.estimate)
```

This is a response level in the requested cell, including cells containing a
zero treatment level; it is not an active-minus-control quantile contrast.

The population response in this simulation is 3.5 because `E[z]=0`; a finite
sample estimate need not equal it. `cell.aipw` fits joint-cell propensities and
per-cell outcome models. Empty training cells and inadequate support refuse.
CoDetermined tier backgrounds also license joint cells when their closure is
certified; Unknown tiers do not. `continuous_cell` does not license a point
intervention on a continuous mediator.

See the [release notes](release-notes/v1.5.0.md) for the full scope,
[artifact guide](artifacts.md#package-15-payloads) for saved payload semantics,
and [evidence ledger](v1.5-evidence.md) for the consuming numerical tests.
