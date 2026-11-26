# Transport mechanisms and synthesize evidence

You can carry a fitted mechanism from a source population into a target population's composed model, pool transported summaries with native evidence, or carry a surrogate relation to another population. Three calls decide what a source-population fit may contribute. Each states which estimates may be used and under which selection premise. None turns that statement into a calibrated claim about the target.

| Route | Rust entry point | Result |
| --- | --- | --- |
| Fitted linear-Gaussian mechanisms into a target composed model | `antecedent_model::mechanism_transport::transport_linear_gaussian` | `TransportedLinearGaussian` with a per-component disposition |
| Fitted nonlinear and non-Gaussian mechanisms (polynomial or expression mean; Gaussian, heteroscedastic, log-normal, Bernoulli, Poisson or gamma response) into a target composed model, with target-specific refits | `antecedent_model::typed_mechanism_transport::transport_typed` | `TransportedTypedModel` with a per-component disposition and consistency test |
| Native plus transported summaries pooled, borrowed or used as a prior | `antecedent_design::transported_synthesis::synthesize_transported` | `TransportedSynthesis`: point and model variance |
| Surrogate bridge carried to another population | `antecedent_design::transported_bridge::TransportedSurrogateBridge` | A bridge contract checked against a selection diagram |

Use this page when what you carry is a fitted mechanism, a summary or a surrogate relation, not a response identified from source experiments. For that question, and for the list of all transport pages, see [Carry an effect to another population](transport/carry-an-effect.md#which-transport-page-answers-your-question).

In Python, the typed call is `antecedent.typed_mechanism_transport` (below). The linear-Gaussian carry is `antecedent.mechanism_composition`. A transported mechanism composition reaches an inverse query as a carried, extrapolative claim (see [inverse queries and decisions](inverse-outcome.md#inverse-queries-on-composed-cohort-and-transported-sources)). A transported response reaches a decision through `antecedent.transported_response`.

## Carry typed mechanisms into a target model

Fit each mechanism in the source. Refit in the target the ones a selection node points into. Ask the transported composition for a contrast. In this sketch `source_data` and `target_data` are column mappings. `source_income` is a source fit made the same way as `source_demand`:

```python
from antecedent import typed_mechanism_transport as tmt

demand = tmt.MechanismSpecification(
    "demand", parents=("price", "income"), family="poisson", mean=tmt.Polynomial(degree=2)
)
source_demand = tmt.fit_typed_mechanism(demand, source_data, population="source")
target_demand = source_demand.refit_in_target(target_data, population="target")

transport = tmt.TypedTransport(
    source_population="source",
    target_population="target",
    selection_nodes=(tmt.SelectionNode("S", ("demand",)),),   # demand's mechanism may differ
    dependence=tmt.TypedDependence.independent_by_design("disjoint samples"),
)
transported = tmt.transport_typed_mechanisms(
    transport, [source_income, source_demand], [target_demand]
)
estimate = transported.mean_contrast(tmt.MeanContrast.mean_at("demand", {"price": 2.0}))
transported.receipt             # which mechanisms were carried and which refitted
```

The contrast is `sum_k w_k E[outcome | do(arm_k)]` in the target composition, here the single mean `E[demand | do(price = 2)]`, computed by Monte Carlo over the structural model. `transported.receipt` lists the carried and the refitted variables. Support is `extrapolative`. Inference is `point_only`. `transported.interval(contrast)` returns the first-order interval with calibration `not_measured`. The rules behind each step are in [nonlinear and non-Gaussian mechanism transport](#nonlinear-and-non-gaussian-mechanism-transport). A runnable use of this call, including the refusal of a shifted mechanism with no target fit, is [`synthetic_users_launch.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/synthetic_users_launch.py).

## Mechanism transport

A fitted additive-noise linear-Gaussian mechanism is carried only when the
declared selection diagram has no selection node pointing into its variable or
into a latent cause it loads on. The diagram dominates. A selection arrow into
a mechanism's variable is the statement that the mechanism differs. An
invariance declaration for the same variable cannot override it. That holds in
the composition checker and in this transport call. An invariance
declaration admits what the diagram is silent about. With no diagram, or a
shared latent the diagram does not mention, a mechanism that loads on a shared
latent cause needs your explicit population-invariance declaration. A
natively refitted target mechanism always replaces a carried one for the same
variable. The two are never averaged. A natively refitted non-Gaussian
target mechanism is reported as an unsupported family rather than carried.
The identity of a transport result covers each mechanism's uncertainty
description, output support, parent domains and conditioning set. A change
of any of them at an unchanged shape is a different transport. Every component of the result records whether it was
carried or refitted. The carried parameter covariance is the source fit's.
The selection assumption adds no variance. The result says so.

Finite tables are decided by the finite counterfactual transport derivation.
That derivation gives an exact obstruction witness. A table here is refused with a pointer
to it. The tests check this call, the composition checker and the finite
derivation against one shared fixture, including a selection into a shared
latent cause. They pin the mapping between their reason codes
(`quantity_semantics_mismatch` for the composition checks,
`transport_proven_non_transportable` for the finite witness).

## Nonlinear and non-Gaussian mechanism transport

In Rust, call `antecedent_model::typed_mechanism_transport`. In Python,
call `antecedent.typed_mechanism_transport`. Both carry typed families under the same
selection-diagram rule. A mechanism has a conditional mean that is a polynomial
of the standardized parents, optionally with pairwise products, or a mechanism
expression you supply. It also has a response family: Gaussian, heteroscedastic Gaussian
(log variance linear in the parents), log-normal, Bernoulli (logit), Poisson
(log link) or gamma (log link, estimated dispersion). A fit returns the
parameter covariance.

The diagram decides each variable separately. A mechanism no selection node
points into is declared invariant and carried from the source fit. A mechanism a
selection node points into is shifted. It is used only from a target fit,
refitted on the source mechanism's design so the two parameter vectors are
comparable. A shifted mechanism with no target fit is refused with
`typed_mechanism_transport.shifted_without_target_evidence`
(`transport_missing_evidence`). Missing evidence never makes a mechanism
invariant.

Where target data exist for a mechanism declared invariant, the call tests the
declaration. The test is a Wald statistic on the full parameter vectors with the summed fit
covariances, against a chi-square law, at the request's level. A rejection
refuses the transport (`typed_mechanism_transport.invariance_contradicted`). A
non-rejection is not evidence of invariance. An invariance with no target
data is an untested premise. The receipt records that premise.

Interventional means are Monte Carlo over the structural model with paired
draws across the arms of a contrast. The outcome enters through its exact
conditional mean. The Monte Carlo standard error is numerical-approximation
uncertainty. The parameter variance of a contrast is first order, `g' Sigma g`.
`g` is the score-function gradient over every involved mechanism's parameters.
`Sigma` is the source covariance for a carried mechanism and the target
covariance for a refit one. They are joined by the declared dependence: independence by
design, an aligned joint covariance, or cross-covariance blocks. Unknown
dependence between more than one involved mechanism is refused. The transport
assumption adds no variance. Support is `extrapolative`. `interval` reports
the first-order interval of a contrast with calibration `not_measured`.

The conformance fixture `conformance/transport/typed_mechanism_shift` holds a
source and a target sample in which income and price are invariant and six
outcome mechanisms (one per family) shift to a quadratic price effect. Its
reference script fits the target sample with numpy and scipy. It evaluates the
target response by quadrature. The tests compare the transported response and
its delta-method standard error with that reference. They compare the response with the
closed-form truth.

## Transported synthesis

A transported summary is `y = theta + b_t + e`, with a transport error `b_t` of
declared variance `tau_t^2` shared by every summary carried through the same
transport `t`. The three calls are generalized least squares under that
covariance, hierarchical normal borrowing across exchangeable studies, and a
transported claim used as the prior of a native likelihood. The evidence ledger
decides overlap before any number is combined. The call is atomic.
The ledger is changed only when the numbers were computed. Every transported
contribution must be carried to one target population.
`synthesize_transported_in_population` also checks the native estimate's
population against that target. The result records the target, the source
populations and the native population. A transported source with no source
population is refused as an incomplete transport identity. Blank assumptions are refused the same way. A source with its target among its own sources is refused the same way. The result is a point and
a model variance. No interval, posterior coverage or calibration is claimed.
The dependence between source and native estimates is your declaration.

## Surrogate bridge premise

The premise is that the relation between the effect on the surrogate and on the
terminal outcome, validated across trials in population `P`, is the same in `Q`.
It holds only when no selection node points into the terminal outcome or into a
declared modifier of the relation. A node into the surrogate does not contradict
it. The call checks the premise against the diagram. It does not discharge it.
The premise stays an unresolved obligation of the derived claim. The bridge remains
point-only.

## What is refused, and what to do

Every detail below, with its reason code and remedy, is listed in [refusal codes: mechanism transport, synthesis and bridge premise](../reference/refusal-codes.md#mechanism-transport-synthesis-and-bridge-premise).

| Refused request | Reason code or detail | Next step |
| --- | --- | --- |
| Selection node into a mechanism or its latent cause | `mechanism_composition.population_selection_arrow_into_mechanism`, `mechanism_transport.selection_arrow_into_latent` | Refit the mechanism in the target |
| Diagram not between the stated source and target | `mechanism_transport.diagram_source_mismatch` / `diagram_target_mismatch` | Declare the diagram for these two populations |
| Declared or externally bound mechanism, or a non-Gaussian family (including a native non-Gaussian target mechanism) | `mechanism_transport.origin_not_fitted`, `family_unsupported` | Fit the mechanism, or use the finite transport route for tables |
| Shifted mechanism (selection node into it) with no target fit | `typed_mechanism_transport.shifted_without_target_evidence` | Fit the mechanism on target data |
| Target data reject a declared invariance | `typed_mechanism_transport.invariance_contradicted` | Declare a selection node into the mechanism and refit it in the target |
| Fits on different designs, or a refit that changes family, mean or parents | `typed_mechanism_transport.design_not_shared`, `target_specification_differs`, `parents_changed` | Refit with `refit_in_target` |
| Contrast variance over several mechanisms with undeclared dependence | `typed_mechanism_transport.parameter_dependence_unknown`, `parameter_dependence_incomplete` | Declare independence by design, a joint covariance or cross-covariance blocks |
| Contributions carried to different targets, or a native estimate for another population | `transported_synthesis.target_population_mismatch`, `native_population_blank` | Carry every contribution to the one target |
| Transported source with no population, blank assumptions or itself as target | `transported_synthesis.transport_identity_incomplete` | Complete the transport identity |
| Transport declared with no uncertainty | `transported_synthesis.transport_variance_undeclared` | Declare `tau_t^2` for the transport |
| One transport serving several claims without a shared covariance | `transported_synthesis.shared_transport_needs_covariance` | Declare the shared transport once |
| Source that would be both native target data and a transported claim, or unknown overlap | `joint_law_required` / ledger refusal | Keep the sources disjoint in the ledger |
| Selection node into the terminal outcome or a modifier | `bridge_assumption_missing` (`scientific_bridge.transport_premise_contradicted_by_selection_diagram`) | Validate the surrogate relation in the target |
| Interval on a synthesis or bridge | none issued | Use the point and model variance as labelled |

## Related

- [Carry an effect to another population](transport/carry-an-effect.md)
- [Transport into decisions and measurement models](transported-decisions-and-measurement.md)
- [Mechanism composition](../composition.md#compose-independently-fitted-mechanisms) and [evidence ledger](../evidence-ledger.md)
