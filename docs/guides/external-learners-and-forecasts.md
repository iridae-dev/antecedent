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

If your code still calls `dml_ate`, `aipw_ate`, `dr_learner_ate`, `causal_forest_ate`, or
`cross_fit_aipw`, switch to `analyze_ate`. Those five names still run, and they warn.
They will be removed in 4.0. `cross_fit_aipw` is the one that only returns the nuisance
predictions, for a fold assignment you supply.

## What a provider result establishes

A result that used your model stays externally attested. Antecedent does not treat your
model as checked unless a fixture verified that exact request. A check of one fold, one
prediction, or one forecast origin does not cover a different request.

If the model was already trained, say the training history is unknown unless you can name
every observation it was trained on. Antecedent does not check that history for you.

Do not shuffle rows that depend on time. These calls raise `CausalValueError` for
`FoldPolicy.ORIGIN_SCOPED` (the older name `temporal` means the same thing). They have no
contract for fitting only on the past.

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

`analyze_ate` fits one of three estimators. Five folds is the default.

- `estimator="aipw"` with `score="aipw"` is the default. It is AIPW on every observed row, with no propensity trimming and no bootstrap. The standard error is the analytic standard error of the scores, treating rows as independent.
- `score="partially_linear"` is the partially linear DML score. It is accepted only with `estimator="dml"`.
- `estimator="dr_learner"` uses your models for the first stage and a linear model for how the effect varies across rows.
- `estimator="causal_forest"` uses your models for the forest's average effect. The trees that estimate how the effect varies stay Antecedent's.

Any other combination is refused. The call does not substitute a nearby setting.

Sampling weights are used only when both of your models declare that they accept weights. The forest does not take weights.

On an AIPW result, `propensity` is the held-out chance of treatment for each row, and `fold_ids` says which fold that row was held out of.

### When the question is not an average effect on one table

**Carry a trial to another population.** Call `analyze_trial_aipw` with your outcome model and your model of who was in the trial. There is no bootstrap. You get a point, and the result says Antecedent did not verify your models. In Rust, declare the learners on the study and call `Study::compile_trial_aipw_learners`.

**A smooth curve of dose against outcome.** Both models must be predictors Antecedent can run again later, without calling back into your process. A model that only exists as a live Python callback is refused. The folds have to be able to run together, and there is no bootstrap.

**Several binary treatments at once.** Call `analyze_joint_cells`. You get a point for each joint setting, and no interval. The result says your models were not verified. In Rust, call `Study::compile_joint_cells_learners`.

If your code still calls `estimate_trial_aipw_with_factories` or `fit_factorized_joint_cells_with_factories`, switch to the calls above. The old calls warn.

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
