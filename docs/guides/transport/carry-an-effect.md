# Carry an effect to another population

You can ask what an effect measured, or a model fitted, in one population implies for another: a trial population and the people a clinic actually treats, last year's customers and this year's, one region and its neighbour. Antecedent answers this as a transport question. It decides from a declared diagram of what differs between the populations whether the target effect follows from the evidence you hold, and it computes the answer only when it does.

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

The licensed static route takes a `TransportQuery` over an explicit `Admg` and runs on the ordinary lifecycle: prepare, inspect, estimate, rebind to new rows, export. This snippet is conceptual in its data (`data` holds the trial and target rows, `admg` is an `Admg` over its columns); the calls match `python/tests/`.

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
study = result.study
updated = ant.estimate(ant.rebind(study, data=new_data))
report = result.inspect().to_dict()
loaded = ant.load(result.export())
```

The selection diagram and the trial column bindings freeze when the study is prepared; `ant.rebind` replaces the rows. The transported estimate is `result.estimate.ate`, and the two overlap diagnostics, trial selection and within-trial treatment, are reported separately on `result.transport_overlap` because one combined number would hide which assumption is failing.

Two runnable examples show the exact-law and statistical versions on a known model: [`transport_exact.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_exact.py) and [`transport_statistical.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_statistical.py) (Rust: [`transport_statistical.rs`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/transport_statistical.rs)).

## What the identifier does

`antecedent.transport.advanced.identify` decides whether a sound formula exists before any number is computed. It adds the selection nodes to the graph and tests m-separation in the treatment-mutilated diagram. The checked subset covers direct transport, pre-treatment (S-admissible) standardization over up to 20 candidate covariates, and recursive factorization over singleton c-components. Where the subset ends, Antecedent stops rather than substituting a heuristic.

`identify` returns the formula and a certificate, with one of five identification statuses: `identified`, `proven_non_transportable`, `missing_evidence`, `not_certified` and `budget_cancel`. They mean different things and call for different next steps; [when transport fails](../transport-failure.md) explains each.

## Choose a route

| Your question | Page |
| --- | --- |
| The graph is not known: a discovered class of graphs, a PAG with latent variables, or a temporal class | [Transport through uncertain or discovered structure](../discovered-structure-transport.md) |
| You want a path-specific or counterfactual quantity in the target, a range over several scenario graphs, or a sampled counterfactual probability | [Transport counterfactual and sampled quantities](../../transport-counterfactuals.md) |
| You hold a retained effect model, a measurement model, a policy, an adaptive trial or a provider's output and want to use it in a decision | [Transport into decisions and measurement models](../transported-decisions-and-measurement.md) |
| You want to carry fitted mechanisms into a composed model, pool transported summaries with native evidence, or carry a surrogate relation | [Transport mechanisms and synthesize evidence](../mechanism-and-synthesis-transport.md) |
| The shared-mechanism premise might fail, or the target, evidence or diagram changed | [Assumption ranges and revalidation](../assumption-range-and-revalidation.md) |
| The result is not identified, support is missing, or the interval is withheld | [When transport fails](../transport-failure.md) |
| You want the theorem scope and size limits of each route | [Transport limits by route](../transport-scope.md) |
| Effects over time, an uncertain initial state, effect constancy | [Population, time and uncertainty routes](../../population-time-uncertainty.md) |

## Read a transported claim

A transported value is a claim about the target conditional on the diagram you declared.

- **Identification.** Check `ident.certificate["identification_status"]` and `ident.inspect().identification`. `identified` means a checked formula exists from the supplied evidence; it says nothing about whether the selection diagram is right.
- **Support.** `result.support` locates the coordinates where evidence is missing or the target has mass the source never saw. Unsupported coordinates are kept and labelled, not dropped.
- **Uncertainty.** Exact supplied laws carry no sampling interval. Empirical-table routes publish a pointwise percentile bootstrap that is nominal unless a coverage record is bound to that exact construction. Read `result.inspect().calibration` and `result.inspect().uncertainty`; do not infer coverage from the name of the parent estimator.
- **Point only.** Many transport routes return a point with the premises that produced it. A range from an assumption analysis or a structural envelope over scenarios is a different object from a confidence interval, and each result labels which one it is.

Structural transport is separate from prior transfer. `antecedent.transport` asks whether source experiments and target observations identify a response in the target. `antecedent.priors` moves statistical evidence after the causal quantity is already defined.

## Unlicensed utilities

`transport.advanced.estimate_trial_effect` computes augmented IPW from supplied outcome regressions (`mu0`, `mu1`) and returns a bare number: no study, export or calibration slot, and no licence. It agrees with the `analyze` path on the point value for the construction both support. `transport.advanced.estimate_augmented_response_grid` evaluates caller-supplied regressions and local response weights on a dose grid under a checked certificate:

```python
from antecedent.transport import advanced as transport

reading = transport.estimate_augmented_response_grid(
    certificate,
    outcome=outcome, trial=trial_membership,
    selection_probability=selection_probability, grid=dose_grid,
    observed_regression=observed_regression,
    target_regression=target_regression, response_weight=response_weight,
)
reading.mean
reading.source_effective_sample_size
reading.selection_overlap
```

`target_regression` and `response_weight` have shape `(grid_size, row_count)`. Antecedent checks the numerical inputs and that the certificate authorizes this equation. The result is point-only; it adds no fitted estimator, interval or double-robustness claim.

## Boundaries

- A design-defined estimand has no score table to reweight, so `study.retarget(...)` refuses (`population_not_estimable`).
- A construction outside the licensed cell, such as a transported derivative or a recursive factorization certificate, refuses with `construction_not_licensed`; the trial IPW algebra evaluates direct transport and standardization, not a product of population-labelled factors.
- Multi-source and limited-experiment questions use the advanced identifiers and the routes in [Transport limits by route](../transport-scope.md).
- Interference between units is a separate question from transport; see [saturation and interference designs](../designs/interference-and-network-exposure.md).

## Related

- [Refusals](../../concepts/refusals.md): why missing evidence, non-certification, support failure and budget exhaustion differ.
- [Interventions and worlds](../../concepts/interventions-and-worlds.md)
- [Support matrix](../../support-matrix.md)
