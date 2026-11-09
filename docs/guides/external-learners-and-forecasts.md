# External learners and forecasts

Antecedent can execute explicitly registered third-party providers as nuisance learners or
forecast producers. A provider remains externally attested unless independent fixtures verify
the exact request. Verification of one fold, prediction, or forecast origin does not verify a
different request. Provider receipts remain attached to the returned result.
Provider calls through one registry are serialized, and Antecedent admits its bridge data copies;
provider-internal threads, devices, memory, model artifacts, retained receipts, and external
service calls are not bounded by that admission. Pass `execution=Execution(limits=ResourceLimits(memory_bytes=...))`
to the estimation or cross-fitting entry point to bound the native execution's admitted memory.
An already active shared execution budget is preserved; a request for a smaller independent
limit is refused rather than replacing that shared pool.
Cancellation is cooperative: an in-progress provider call must return before the run can stop;
the cancelled run publishes no partial estimate.

A provider's declared prior training data is not independently authenticated. For pretrained
models, leave prior training lineage unknown unless you can supply its complete observation
identities. Ordinary shuffled cross-fitting is not appropriate for time-dependent rows: the
external nuisance entry points refuse `fold_policy="temporal"`, because they have no
forward-only fitting contract.

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
from antecedent.external_learners import ExternalLearner, dml_ate

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
result = dml_ate(
    {"t": t, "y": [1 + 2 * value for value in t]}, [("t", "y")], "t", "y",
    outcome=outcome, treatment=treatment, folds=2,
)
assert abs(result.estimate - 2.0) < 1e-10
# One (id, fit receipt, predict receipt) entry per fold and nuisance model:
# 2 folds x (control outcome, treated outcome, propensity).
assert len(result.receipts) == 6
```

`dml_ate` identifies the supplied static DAG before fitting. It supports AIPW and partially
linear DML scores. `aipw_ate` provides the five-fold, untrimmed all-observed AIPW route with
IID analytic inference and no bootstrap. `dr_learner_ate` uses the external first-stage models
with a built-in linear CATE final stage. `causal_forest_ate` uses them for the forest's marginal
ATE; its CATE trees remain native. Unsupported settings are refused rather than silently
substituted. Sampling weights require both nuisance providers to declare weight support; the
forest route does not support weights. `cross_fit_aipw` exposes held-out nuisance predictions
without making a causal estimate.

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
