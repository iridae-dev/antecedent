# Carry an effect to another population

You can ask what an effect measured, or a model fitted, in one population implies for another: a trial population and the people a clinic actually treats, last year's customers and this year's, one region and its neighbour. Antecedent answers this as a transport question. It decides from a declared diagram of what differs between the populations whether the target effect follows from the evidence you hold, and it computes the answer only when it does.

Use this page when you have a causal graph you are willing to declare, evidence from at least one source population, and a named target population whose outcomes under the intervention you have not observed. The terms are defined in the next section; for what an intervention is, see [interventions and worlds](../../concepts/interventions-and-worlds.md).

## Which transport page answers your question

Ten pages cover transport. Each answers one kind of question; start from the row that matches yours.

| Your question | Page |
| --- | --- |
| You have a declared graph and want a target response or effect from source experiments | This page |
| The target is a neighbourhood of your own rows (row weights, or a kernel over covariate distance), not a labelled population, and you want its effect with overlap diagnostics and a refusal when the rows do not cover it | [Ask for the effect in a weighted target population](weighted-target-population.md) |
| The graph is not known: a discovered class of graphs, a PAG with latent variables, or a temporal class | [Transport through uncertain or discovered structure](../discovered-structure-transport.md) |
| You want a path-specific or counterfactual quantity in the target, a counterfactual probability from counts, or the reasons behind a range over scenario graphs | [Transport counterfactual and sampled quantities](../../transport-counterfactuals.md) |
| You hold a retained effect model, a measurement model, a policy, an adaptive trial or a provider's output and want to use it in a decision | [Transport into decisions and measurement models](../transported-decisions-and-measurement.md) |
| You want to carry fitted mechanisms into a composed model, pool transported summaries with native evidence, or carry a surrogate relation | [Transport mechanisms and synthesize evidence](../mechanism-and-synthesis-transport.md) |
| The shared-mechanism premise might fail, or the target, evidence or diagram changed | [Assumption ranges and revalidation](../assumption-range-and-revalidation.md) |
| The result is not identified, support is missing, or the interval is withheld | [When transport fails](../transport-failure.md) |
| You want the theorem scope and size limits of each route (limited experiments, several sources, conditional questions, dose-response) | [Transport limits by route](../transport-scope.md) |
| You want a measured interval for a transported scalar, an effect over time, an uncertain initial state or a test of effect constancy | [Population, time and uncertainty routes](../../population-time-uncertainty.md) |

The pages share one scope rule: a transported value is a claim about the target conditional on the declared diagram, and it is a point unless the page for its route names an interval construction and tells you where to read its calibration status.

## What a transport question needs

A transport question names four things. Antecedent does not infer any of them from the data.

| You declare | Meaning |
| --- | --- |
| Source and target populations | Where the evidence comes from and where the answer is wanted |
| A selection diagram | A causal graph (directed edges, plus bidirected edges for latent confounding) with a selection node pointing at each variable whose mechanism may differ between the populations |
| What the source can experiment on | The variables with randomized or otherwise experimental evidence in the source |
| The target evidence | Usually the target's observational law or covariates; never target outcomes under the intervention |

Declaring which mechanisms differ is a scientific assumption. The data cannot test it. A mechanism with no selection node is assumed to be shared between the populations, and that sharing is the premise every transported value stands on.

## Run a transport question

`antecedent.transport.Transport` wraps an ordinary question (a `ResponseCurve` or an `AverageEffect`) with the target population and the evidence you hold. It runs on the ordinary lifecycle: analyze, inspect, rebind to new data, export. The call below is the one in [`transport_exact.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_exact.py); `data` there is an `ExactTransportData` holding the source experiment's law of `y` under `do(x = 0)` and `do(x = 1)`.

```python
import antecedent as ant

graph = ant.Admg.from_edges(["x", "y"], [("x", "y")])
query = ant.transport.Transport(
    ant.ResponseCurve("x", "y", grid=[0.0, 1.0]),
    target="target",
    evidence=ant.transport.Evidence(
        source=ant.transport.Source(
            "source", kind="experimental", interventions=["x"], sampling="independent"
        ),
        target_sampling="representative_sample",
    ),
)
result = ant.analyze(data, graph=graph, query=query)
result.answer.kind                  # "response"
result.value.response.values        # one row per grid point: E_target[y | do(x)]
report = result.inspect()           # identification, support, uncertainty, assumptions
refreshed = ant.estimate(ant.rebind(result.study, data=new_data))
artifact = result.export()
loaded = ant.load(artifact, expected=artifact.expectation)
```

The quantity is the target population's mean response `E*[Y | do(X = x)]` at each grid value `x`, where `*` marks the target. Variables whose mechanisms differ are named in `Transport(selections=[...])`, or per source in `Source(selections=[...])`; with none declared, as here, every mechanism is assumed shared and the source experiment answers directly. With the example's laws (`P(y = 1)` of 0.5 and 0.8 under the two interventions) the transported curve is `[0.5, 0.8]`.

Three runnable examples use this call on models with known answers:

| Example | Evidence | What it shows |
| --- | --- | --- |
| [`transport_exact.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_exact.py) (Rust: [`transport_exact.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/transport_exact.rs)) | Exact supplied laws | A point with no sampling interval; rebinding to replacement laws |
| [`transport_statistical.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_statistical.py) (Rust: [`transport_statistical.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/transport_statistical.rs)) | One counted source table | The empirical plug-in of the same formula |
| [`transport_meta_grid.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_meta_grid.py) (Rust: [`transport_meta_grid.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/transport_meta_grid.rs)) | Two complementary sources | A curve neither source identifies alone, and unsupported grid points under a partial catalog |

Count-based estimation uses `EmpiricalTable` by default. `LearnedCategorical` and `TrialAipw` change the assumption set and are passed explicitly; see [estimation assumptions](../transport-scope.md#estimation-assumptions).

### Trial-to-target weighting on row data

When the source is a randomized trial with a binary treatment and you hold one table of trial and target rows, `transport.advanced.TransportQuery` estimates the transported response by inverse-probability weighting. It is the transport cell listed in the [support matrix](../../support-matrix.md) and takes an explicit `Admg`. In this snippet `data` holds the trial and target rows and `admg` is an `Admg` over its columns.

```python
import antecedent as ant
from antecedent import transport

query = transport.advanced.TransportQuery(
    ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
    transport.advanced.SelectionDiagram("trial", "target", ["x"]),
    source_experiments=["a"],
    trial="trial",                    # source-trial membership column
    selection_probability="s",        # P(S=1 | X) on every row
    treatment_probability="e",        # P(A=1 | X, S=1) on trial rows
)
result = ant.analyze(data, graph=admg, query=query)
value = result.value
value.estimate.ate                    # transported contrast between the two grid values
value.transport_overlap               # .selection and .treatment diagnostics
updated = ant.estimate(ant.rebind(result.study, data=new_data))
report = result.inspect().to_dict()
loaded = ant.load(result.export())
```

`S` is trial membership, `A` the treatment and `X` the covariates the selection node points into. The estimand is the target mean response at each grid value, identified either directly or by standardizing over `X`, and `estimate.ate` is the contrast between the two grid values. The selection diagram and the trial column bindings freeze when the study is prepared; `ant.rebind` replaces the rows. The two overlap diagnostics, trial selection and within-trial treatment, are reported separately on `transport_overlap` because one combined number would hide which assumption is failing.

## What the identifier does

Identification runs before any number is computed. To run it on its own, call `ant.identify(graph=graph, query=query)` for a `Transport` query or `transport.advanced.identify(graph=admg, query=query)` for a `TransportQuery`. It adds the selection nodes to the graph and tests m-separation in the treatment-mutilated diagram. The checked subset covers direct transport, pre-treatment (S-admissible) standardization over up to 20 candidate covariates, and recursive factorization over singleton c-components. Where the subset ends, Antecedent stops and reports `not_certified`; it does not substitute a heuristic.

The result carries the formula and a certificate with one of five identification statuses: `identified`, `proven_non_transportable`, `missing_evidence`, `not_certified` and `budget_cancel`. They mean different things and call for different next steps; [when transport fails](../transport-failure.md) explains each.

## Read a transported claim

A transported value is a claim about the target conditional on the diagram you declared.

- **Identification.** Check `ident.certificate["identification_status"]` and `ident.inspect().identification` (on a `transport.advanced.identify` result, `ident.identification_status`). `identified` means a checked formula exists from the supplied evidence; it says nothing about whether the selection diagram is right.
- **Support.** `result.support` locates the coordinates where evidence is missing or the target has mass the source never saw. Unsupported coordinates are kept and labelled, not dropped.
- **Uncertainty.** Exact supplied laws carry no sampling interval. Empirical-table routes publish a pointwise percentile bootstrap that is nominal unless a coverage record is bound to that exact construction. Read `result.inspect().calibration` and `result.inspect().uncertainty`; do not infer coverage from the name of the parent estimator.
- **Point only.** Many transport routes return a point with the premises that produced it. A range from an assumption analysis or a structural envelope over scenarios is a different object from a confidence interval, and each result labels which one it is.

Structural transport is separate from prior transfer. `antecedent.transport` asks whether source experiments and target observations identify a response in the target. `antecedent.priors` moves statistical evidence after the causal quantity is already defined.

## Refusals you will meet

| Situation | Reason code | Next step |
| --- | --- | --- |
| The checked subset does not decide the query | `transport_not_certified` | Declare more structure or use a route whose theorem family covers the question; see [when transport fails](../transport-failure.md#not-certified) |
| A checked obstruction shows no formula exists from the declared evidence | `transport_proven_non_transportable` | Add a source or an experiment; see [theorem-scoped impossibility](../transport-failure.md#theorem-scoped-impossibility) |
| The formula needs a law the catalog does not hold | `transport_missing_evidence` | Supply the named regime; see [missing evidence](../transport-failure.md#missing-evidence) |
| Evidence is missing or empty at one grid coordinate | none on a grid: the coordinate is kept and labelled `missing_evidence` or `outside_empirical_support` | Supply evidence at that coordinate or narrow the target; see [local support and positivity](../transport-failure.md#local-support-and-positivity) |
| The identification step or depth budget ran out | `transport.identification_budget` (a resource error, never a verdict) | Raise the limits or simplify the diagram; see [budget exhaustion](../transport-failure.md#budget-exhaustion) |
| `TransportQuery` with a construction outside the weighting cell, such as a transported derivative or a recursive factorization certificate | `construction_not_supported` | Ask for a mean `ResponseCurve` under direct transport or standardization |
| `study.retarget(...)` on a transported study | `population_not_estimable` | State the new target in a new transport question; a design-defined estimand has no score table to reweight |

The two last codes are described under [other registered refusal codes](../../reference/refusal-codes.md#other-registered-refusal-codes); the fields every refusal exposes are in [structured refusals and their remedies](../../reference/refusal-codes.md#structured-refusals-and-their-remedies).

## Unsupported utilities

Two low-level functions sit outside the study lifecycle. Use them to inspect numbers, not to make a claim.

`transport.advanced.estimate_trial_effect` computes the trial IPW point and, from supplied outcome regressions (`mu0`, `mu1`), the augmented IPW point, and returns them with the overlap report in a `TrialTransportEstimate`: no study, export or calibration slot, and no support. It agrees with the `analyze` path on the point value for the construction both support. `transport.advanced.estimate_augmented_response_grid` evaluates caller-supplied regressions and local response weights on a dose grid under a checked identification:

```python
from antecedent.transport import advanced as transport

reading = transport.estimate_augmented_response_grid(
    identification,
    outcome=outcome, trial=trial_membership,
    selection_probability=selection_probability, grid=dose_grid,
    observed_regression=observed_regression,
    target_regression=target_regression, response_weight=response_weight,
)
reading.mean
reading.source_effective_sample_size
reading.selection_overlap
```

`identification` is the result of `transport.identify(graph=..., query=...)` for the `TransportQuery`. `target_regression` and `response_weight` have shape `(grid_size, row_count)`. Antecedent checks the numerical inputs and that the certificate authorizes this equation. The result is point-only; it adds no fitted estimator, interval or double-robustness claim.

## Boundaries

- The trial IPW algebra evaluates direct transport and standardization, not a product of population-labelled factors.
- Multi-source and limited-experiment questions beyond the `Transport` form use the advanced identifiers and the routes in [Transport limits by route](../transport-scope.md).
- Interference between units is a separate question from transport; see [saturation and interference designs](../designs/interference-and-network-exposure.md).

## Related

- [Refusals](../../concepts/refusals.md): why missing evidence, non-certification, support failure and budget exhaustion differ.
- [Interventions and worlds](../../concepts/interventions-and-worlds.md)
- [Support matrix](../../support-matrix.md)
