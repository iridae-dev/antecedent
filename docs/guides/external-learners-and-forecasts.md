# External learners and forecasts

Use this guide when a model from another package should supply the nuisance functions of an
average-effect estimate (the outcome regressions and the propensity), or should produce
forecasts that you then score. Antecedent runs such a model only as an explicitly registered
**provider** in a `ProviderRegistry` (see [connect scientific components](build/scientific-interfaces.md)), and
keeps the provider's receipt on every result.

| Task | Entry point |
| --- | --- |
| Average effect of a binary treatment with external nuisance models | `antecedent.external_learners.analyze_ate(..., learners=Learners(outcome, treatment), estimator=...)`, which declares the learners on a compiled study (Rust: `StudyBuilder::learner_factories`) |
| Held-out nuisance propensities on the study's fold plan | `propensity` and `fold_ids` of the `analyze_ate` result |
| Rolling-origin forecasts to assess | `antecedent.forecasting.rolling`, then [`predictive_assessment.assess`](../predictive-assessment.md) |

`dml_ate`, `aipw_ate`, `dr_learner_ate`, `causal_forest_ate` and `cross_fit_aipw` run the same
fits without the compiler. They warn with `AntecedentDeprecationWarning`, return the same
numbers as `analyze_ate`, and are removed in 4.0.

## What a provider result establishes

A provider remains externally attested unless independent fixtures verify the exact request.
Verification of one fold, prediction, or forecast origin does not verify a different request.

A provider's declared prior training data is not independently authenticated. For pretrained
models, leave prior training lineage unknown unless you can supply its complete observation
identities. Ordinary shuffled cross-fitting is not appropriate for time-dependent rows: the
external nuisance entry points raise `CausalValueError` for
`FoldPolicy.ORIGIN_SCOPED` (the legacy word `temporal` decodes to it), because
they have no forward-only fitting contract.

## Resources and cancellation

Provider calls through one registry are serialized, and Antecedent admits its bridge data copies;
provider-internal threads, devices, memory, model artifacts, retained receipts, and external
service calls are not bounded by that admission. Pass `execution=Execution(limits=ResourceLimits(memory_bytes=...))`
to the estimation or cross-fitting entry point to bound the native execution's admitted memory.
An already active shared execution budget is preserved; a request for a smaller independent
limit is refused rather than replacing that shared pool.
Cancellation is cooperative: an in-progress provider call must return before the run can stop;
the cancelled run publishes no partial estimate.

## Nuisance learner

A learner provider implements the `CausalProviderSpec` and `execute` interface. Its
`query_family` is `nuisance_learner`; it declares `mean`, a variable `output_shape`, and an
artifact codec. The fit operation returns an empty estimate and an artifact containing the
fitted model. The predict operation receives that artifact as `model_base64` and returns one
point prediction per input row. Providers that support weights declare `sample_weights` in
`data_dependence` and must actually use the supplied weights.

```python
import base64
import numpy as np
from antecedent.extensibility import (
    CausalProviderSpec, ProviderCapability, ProviderExecution, ProviderRegistry,
)
from antecedent.external_learners import ExternalLearner, Learners, analyze_ate

class MeanProvider:
    spec = CausalProviderSpec(
        query_family="nuisance_learner",
        identification_requirements=("identified_adjustment",),
        observed_distributions=("X,Y",), nuisance_functions=("conditional_mean",),
        support_conditions=("finite",), data_dependence=("training_rows", "sample_weights"),
        inference_claims=("point_only",), influence_function=None,
        fold_policy="host_owned", output_shape="one_per_prediction_row",
        uncertainty_semantics="point_only", artifact_codec="ascii_mean_v1",
        deterministic=True, provenance={"package": "example", "version": "1"},
        capabilities=(ProviderCapability.MEAN,),
    )

    def execute(self, request):
        if request["operation"] == "fit":
            mean = np.average(request["y"], weights=request["weights"])
            return ProviderExecution([], None, (), "supported", {"version": "1"}, str(mean).encode())
        mean = float(base64.b64decode(request["model_base64"]))
        predictions = np.full(len(request["x"]), mean)
        return ProviderExecution(predictions, None, (), "supported", {"version": "1"})

registry = ProviderRegistry()
registry.register("mean", MeanProvider())
outcome = ExternalLearner(registry, "mean", "regression", supports_weights=True)
treatment = ExternalLearner(registry, "mean", "binary_probability", supports_weights=True)
t = [float(i % 2) for i in range(40)]
result = analyze_ate(
    {"t": t, "y": [1 + 2 * value for value in t]}, [("t", "y")], "t", "y",
    learners=Learners(outcome, treatment), folds=2,
)
assert abs(result.estimate - 2.0) < 1e-10
# One (id, fit receipt, predict receipt) entry per fold and nuisance model:
# 2 folds x (control outcome, treated outcome, propensity).
assert len(result.receipts) == 6
```

The fixture has no confounder and an exact effect of 2, so `result.estimate` is 2 to
rounding. The provider here fits a constant; it only shows the interface.

### Estimand and assumptions

`analyze_ate` compiles a study over the supplied static DAG, identifies it before fitting
and fits the configured estimator with the declared learners. It estimates the average
treatment effect `E[Y(1) - Y(0)]` of the binary treatment over the supplied rows, adjusting
for the identified set reported in `result.adjustment_set`. The graph is your causal
assumption; the route does not test it. The point also assumes overlap and that the provider's
out-of-fold predictions estimate the conditional outcome means and the propensity.

### Read the estimate

| Field | Meaning |
| --- | --- |
| `estimate` | the cross-fitted point estimate |
| `standard_error` | the IID cross-fitted score standard error; valid only under the declared graph and the usual nuisance-rate conditions, which are not checked for an external learner |
| `adjustment_set`, `row_ids`, `fold_ids` | the identified adjustment columns, the analysed rows and each row's fold |
| `cate`, `cate_standard_errors` | per-row values for the estimators that produce them, otherwise `None` |
| `assumptions` | the assumptions the route records |
| `provenance`, `receipts` | one entry per fold and nuisance model, linking each fit and prediction to its provider receipt |

A standard error is not an interval: the result carries no confidence interval and no
calibration status.

### Estimators and settings

`analyze_ate` supports AIPW and partially linear DML scores (`score="aipw"`, the default, or
`score="partially_linear"`), with `folds=5` by default. `estimator="aipw"` is the five-fold,
untrimmed all-observed AIPW route with IID analytic inference and no bootstrap.
`estimator="dr_learner"` uses the external first-stage models with a built-in linear CATE final
stage. `estimator="causal_forest"` uses them for the forest's marginal ATE; its CATE trees
remain native. A score other than `"aipw"` is accepted only with
`estimator="dml"`. Unsupported settings are refused rather than silently substituted. Sampling weights require both nuisance providers to declare weight support; the
forest route does not support weights. The result's `propensity` holds the held-out
propensities of an AIPW score fit on the fold plan in `fold_ids`.

The Rust learner interface also supports caller factories for the certified trial-to-target
AIPW point through `estimate_trial_aipw_with_factories`; that route requires zero
bootstrap replicates and marks its external nuisances unverified. Rust callers may also use
`estimate_smoothed_dose_with_factories` when both factories export portable predictors; it
refuses serial folds and bootstrap refits. The Python callback adapter does not export a
portable predictor and is not accepted by that route.
Rust callers may use `fit_factorized_joint_cells_with_factories` for joint-cell scores;
that path marks caller factories unverified and is point-only.

Separately installed packages can register a zero-argument factory under the
`antecedent.providers` entry-point group. Call `registry.load_entry_point(name)` explicitly.
The registry never grants native-supported status to a Python extension.

A forecasting provider can also implement `fit` and `predict` as a nuisance learner through
`ForecastNuisanceAdapter`, but only if it declares `iid_row_conditional_mean` in
`data_dependence`. That declaration is an IID information contract, not permission to use
shuffled folds for temporal validation.

## Forecast provider

A forecast provider declares `query_family="forecast"` and the capabilities it supplies. Each
origin runs as a separate registry request, with only that origin's declared history and future
covariates. The provider may retain information across calls; supplied lineage must include any
such state or remain unknown. Outputs can be points, Gaussian parameters, marginal samples, or
joint sample trajectories. A point forecast alone is not turned into a predictive distribution.
Nonempty prior training lineage must include matching observation times no later than the
forecast origin. The declared lineage is bound to the provider request.

```python
from antecedent import forecasting as fc, predictive_assessment as pa
from antecedent.extensibility import CausalProviderSpec, ProviderCapability, ProviderExecution, ProviderRegistry

class ForecastProvider:
    spec = CausalProviderSpec(
        query_family="forecast", identification_requirements=("predictive_only",),
        observed_distributions=("history",), nuisance_functions=(), support_conditions=("history",),
        data_dependence=("origin_history",), inference_claims=("empirical_samples",),
        influence_function=None, fold_policy="origin_scoped", output_shape="draws",
        uncertainty_semantics="empirical_samples", artifact_codec="none", deterministic=True,
        provenance={"package": "example", "version": "1"},
        capabilities=(ProviderCapability.SAMPLE,),
    )

    def execute(self, request):
        last = request["case"]["history_values"][-1][0]
        return ProviderExecution([last - 1, last, last + 1], None, (), "supported", {"version": "1"})

registry = ProviderRegistry()
registry.register("forecast", ForecastProvider())
keys = pa.keys("series", [1, 2, 3, 4])
target = pa.Target("temperature", "site", "C", horizon_steps=1)
origins = [
    fc.ForecastOrigin(i, list(range(1, i + 1)), [[float(j)] for j in range(1, i + 1)], keys[:i])
    for i in range(1, 4)
]
# This example's provider has no prior training or retained state.
forecasts = fc.rolling(registry, "forecast", origins, target=target, kind="samples",
                       prior_training_keys=[[], [], []])
observed = pa.Observations(target, keys[1:], "real", [2.0, 3.0, 4.0])
assessment = pa.assess(observed, [forecasts.predictor("example")], forecasts.validation_design(),
                       scores=["crps"], dependence=pa.Dependence.serial([1, 2, 3], 1),
                       require_held_out=True)
```

Each result in `forecasts.results` retains its own request identity, fit declaration, and provider receipt.
The aggregate assessment source hashes those receipts and is conservatively externally attested;
it does not promote the batch to verified merely because an individual request was verified.
Per-origin fit declarations ensure an observation used at a later origin does not by itself
make an earlier forecast in-sample. Unknown lineage remains unverified.

## Related pages

- [Connect scientific components](build/scientific-interfaces.md): the provider specification and registry rules.
- [Assess a predictive law](../predictive-assessment.md): scores, dependence declarations and held-out requirements for the forecasts produced here.
- [Sampling weights](sampling-weights.md): what a weighted estimate targets.
- [Refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies): the fields a refusal carries.
