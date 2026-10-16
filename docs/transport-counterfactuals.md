# Transport, scenarios and transported counterfactuals

Three routes answer "what changes when the population changes?" and are
honest about where they stop:

| Route | Python entry | Claim |
| --- | --- | --- |
| Transported path-specific effect, affine-additive class | `antecedent.transported_counterfactual` | point-only under a fully specified structural fit; calibration unmeasured |
| Selection differences and invariances behind each scenario answer | `antecedent.scenario_invariance` | derived report, nothing sealed |
| Learned joint source-target transport | `antecedent.transport.advanced.learned_joint_transport` | measured scalar carrier only at a current, matching record; underlying posterior remains model-conditional |
| General finite counterfactual transport (Rust: `antecedent::prepare_counterfactual_transport`) | `antecedent.transport.advanced.identify_counterfactual` (`python/tests/test_transport_counterfactual.py`) | exact point of a nested or conditional counterfactual probability from exact finite laws; no interval, posterior or calibration |

For the ordinary transport compiler (`transport.Transport(...)`) see
[transport and interference](transport-interference.md) and
[transport failures](guides/transport-failure.md). This page covers the transported-counterfactual and scenario routes.

## Transported path-specific effect

A transported natural direct, natural indirect or total effect: the source
population has a fitted structural model, the target population has only its
covariate law, and the contrast is carried over using the source equations alone.

The class is narrow and declared:

- covariates `Z` are pre-treatment variables whose finite-support joint law may
  differ between source and target;
- the treatment `A` is only ever set;
- mediator and outcome mechanisms are `V = alpha(Z) + sum_p beta_p(Z) * V_p + U`,
  where `alpha` and every `beta_p` are affine in `Z` and the noises `U` are
  additive and independent of `Z` and of each other.

This is the **affine-additive class**. Nonparametric mechanisms, recanting
witnesses and nonlinear additive noise are outside it. The general route
(`antecedent.temporal_counterfactual.transported_path_specific`) stays closed: it
never evaluates and always raises a typed refusal. That refusal names the
successor route for the finite discrete, static, acyclic semi-Markovian scope,
`antecedent.transport.advanced.identify_counterfactual` (nested counterfactual
and edge-route queries over a selection diagram); the closed function is not a
delegation and answers nothing itself.

The snippet is the tested fixture of `python/tests/test_transported_counterfactual.py`:
`M = 0.5 Z + (2 - 0.5 Z) A + U_M`, `Y = (3 + 0.75 Z) A + 4 M + 1.5 Z + U_Y`, with
treated value 2 and control value 0.5.

```python
from antecedent.transported_counterfactual import (
    AdditiveNoiseScm, Affine, CovariateLaw, EdgeAssignment, Mechanism,
    MechanismSelection, Premises, transported_path_specific_effect,
)

scm = AdditiveNoiseScm(
    treatment="a",
    covariates=("z",),
    mechanisms=(
        Mechanism("m", {"z": 0.5, "a": Affine(2.0, {"z": -0.5})}),
        Mechanism("y", {"a": Affine(3.0, {"z": 0.75}), "m": 4.0, "z": 1.5}),
    ),
)
source = CovariateLaw.univariate("z", {0.0: 0.4, 1.0: 0.4, 2.0: 0.2})
target = CovariateLaw.univariate("z", {0.0: 0.25, 1.0: 0.75})

nde = EdgeAssignment.natural_direct("y", treated_value=2.0, control_value=0.5)
premises = Premises(additive_noise=True, noise_laws_shared=True, cross_world_independence=True)

effect = transported_path_specific_effect(
    scm, source, target, MechanismSelection.on("z"), nde, premises=premises
)
effect.target_contrast      # 5.34375 = 4.5 + 1.125 * E_T[z]
effect.source_contrast      # 5.4: what a source-only analysis would report
effect.inference_claim      # "point_only"
effect.derivation.checked   # premises Antecedent verified
effect.derivation.declared  # premises you asserted and it did not verify
```

`EdgeAssignment.natural_indirect("y", ["m"], ...)` and `.total("y", ["m"], ...)`
give the NIE (9.75 on the same fixture) and the total effect (15.09375 = NDE +
NIE). The noise cancels in every unit's contrast, so it is not modelled.

### Checked versus declared

The result separates the premises Antecedent **checked** (acyclic well-formed
model, no selection on a mediator or outcome mechanism, covariates are
pre-treatment roots, target support inside source support, regime evidence
present) from the ones you only **declared** (`Premises`: additive noise, shared
noise laws, cross-world independence). Declarations default to undeclared;
omitting one refuses with the Rust detail
(`transported_counterfactual.nonadditive_mechanism` and the like, code
`cell_not_licensed`). Declaring one records a claim the supplied source fit must
support; it is never checked here. The target law matters exactly as the
arithmetic says: moving mass from `z = 1` to `z = 0` shifts the NDE by the NDE
slope 1.125 times the change in `E z`.

### `MechanismSelection`

A selection node marks a variable whose mechanism (or covariate law) may differ
between populations. `MechanismSelection({label: target})` is a plain label-to-target
mapping for this module's structural model; `MechanismSelection.on("z")` creates
`S_z -> z`. It is not the selection *diagram* graph of
`antecedent.transport.advanced.SelectionDiagram`.

### Refusals and the two-model witness

A selection node on a mediator or outcome mechanism is outside the class, because
two models can agree on the whole source population and differ on the target
contrast. The refusal is a `TransportedCounterfactualRefusal` (a
`StructuredRefusal`):

- `transport_proven_non_transportable` with detail
  `transported_counterfactual.selection_on_mechanism` and a `witness`: two
  target models that share the source model and its source answer but differ on
  the target contrast. In the fixture, selecting on `y` perturbs the `A`
  coefficient of `Y` and moves the NDE by `1.5`; selecting on `m` witnesses only
  for the NIE (the NDE does not depend on the mediator's equation);
- `cell_not_licensed` with no witness when the contrast does not depend on the
  selected mechanism: the class excludes it, and no impossibility is claimed;
- `transport_support_failure` (`overlap_failure`, offending e.g. `z=3`) when the
  target support leaves the source support;
- `transport_missing_evidence` (`factor_missing`, with `missing_factors`) when
  regime evidence is absent;
- `invalid_argument` details `invalid_model`, `invalid_query`, `invalid_law`,
  `invalid_diagram`.

```python
from antecedent.transported_counterfactual import TransportedCounterfactualRefusal

try:
    transported_path_specific_effect(
        scm, source, target, MechanismSelection.on("y"), nde, premises=premises
    )
except TransportedCounterfactualRefusal as refusal:
    refusal.code                      # "transport_proven_non_transportable"
    refusal.detail                    # "transported_counterfactual.selection_on_mechanism"
    w = refusal.witness
    w.target_contrast_b - w.target_contrast_a    # 1.5
```

### Export and replay

`effect.export()` is a checksummed `transported_counterfactual_v1` artifact and
`effect.identity` the digests a consumer retains. `consume_transported_counterfactual_artifact(data,
expected_identity=effect.identity)` replays it and refuses a resealed change to any
field.

## General finite counterfactual transport

The affine-additive route above needs a fitted structural model. The general
finite route needs none: it answers a nested or conditional counterfactual
probability of the **target** population, such as `P*(Y_{x, M_{x'}} = y)`, a
natural direct or indirect effect contrast, or an effect of treatment on the
treated, from *labelled laws* alone.

Declared, and nothing else is accepted:

- finitely many observed variables (at most 8) with declared levels (at most 4
  each) on an acyclic graph with bidirected edges for latent confounding;
- a selection diagram per source population: the variables whose mechanism may
  differ between that source and the target, and the variables the source can
  experiment on;
- evidence regimes labelled by population and intervention set, each an exact
  finite joint law (one law per intervention level). Evidence of one population
  never stands in for another, and separate marginals never create a
  cross-world law.

The decision is a sound, bounded procedure, not a complete one. A formula is
returned only when every factor it needs is either invariant (no selection
target lies in it, so a source law transports) or derivable from the target's
own laws. The outcomes are kept apart:

| Outcome | Meaning | Reason code |
| --- | --- | --- |
| identified | a checked derivation exists from the supplied laws | none |
| proven non-transportable | two finite models agree on every regime of the declared family and differ on the query, re-verified by exact enumeration | `transport_proven_non_transportable` |
| missing evidence | a derivation exists if the named declared regimes were supplied | `transport_missing_evidence` |
| not certified | the bounded search found neither a derivation nor a witness; this is not a proof | `transport_not_certified` |
| out of contract | more than the bounds above, cyclic, continuous, temporal or sampled | `route_not_supported` |
| stopped | operations, depth, memory or cancellation | `transport_budget_cancel` |

```rust
use antecedent::{
    CounterfactualTransportConsumption, CounterfactualTransportOptions,
    CounterfactualTransportPreparation, consume_counterfactual_transport_artifact,
    prepare_counterfactual_transport,
};

// Decide once. `evidence` may be None: bind laws later with `refresh`.
let decided = prepare_counterfactual_transport(
    &problem, &query, &catalog, Some(laws), &CounterfactualTransportOptions::default(), &ctx,
)?;
let CounterfactualTransportPreparation::Identified(prepared) = decided else {
    // a proven obstruction: `report.verify()` re-checks the witness, `report.export_artifact()`
    // seals it
    return Ok(());
};
let result = prepared.estimate(&ctx)?;
let probability = result.probability(); // exact finite-law value

// Changed numbers at the same shape: no search again, the numerical state is rebuilt.
let refreshed = prepared.refresh(new_laws, &ctx)?;

// Export, and replay from bytes plus an identity you hold independently.
let bytes = result.export_artifact()?;
let identity = result.identity()?;
let consumed = consume_counterfactual_transport_artifact(
    &bytes, &identity, Default::default(), &ctx,
)?;
```

An exact law whose conditioning event has zero mass is a support failure, never
an extended value, and a law at an intervention level nobody supplied is a
missing provider, never a guessed table. A counted, empirical or posterior law is
refused (`cell_not_licensed`): sampled counterfactual transport is a separate
route and this one publishes no interval and no posterior.

### Export and replay of the general route

The `counterfactual_transport_v1` artifact keeps the model class, the query with
its selection diagram, the complete evidence catalog and the law tables, the
derivation with its search receipt, and the point with the identity of every law
it read; an obstruction keeps the two-model witness instead. Two digests bind it:
the premises digest (model class, query, names, search and evaluation limits and
the conclusion) and the data digest (catalog and law tables). A consumer that
holds the bytes and an expected identity re-decides under the stored limits,
requires the identical derivation, recomputes the point bit for bit, and
re-verifies a witness by exact enumeration. Another version or required feature,
an unknown field, stored limits above the consumer's, an unexpected identity, or
a resealed query, selection, table or derivation mutation is refused with the
slot that failed named.

Replay detects corruption and resealed edits that change the answer. It does not
detect a producer that seals a wrong graph, query or law on purpose, and it does
not detect an error in the identification theory shared by producer and
consumer; the independent check is the finite-SCM enumeration in the tests.

### Python surface of the general route

`antecedent.transport.advanced.identify_counterfactual` decides the same query from
typed objects: the graph (`Admg`), declared `levels`, worlds that carry hard
interventions and edge routes (`reads` names the world a child reads a parent
from, which is how `Y_{x, M_{x'}}` is written), `CounterfactualEvent` atoms, per
source `selections` and `controllable`, and an `EvidenceCatalog`.

```python
from antecedent.transport import advanced as t

query = t.CounterfactualTransportQuery(
    target="target",
    levels={"Z": (0, 1), "X": (0, 1), "M": (0, 1, 2), "Y": (0, 1)},
    worlds=(
        t.CounterfactualWorld("inner", {"X": 0}),
        t.CounterfactualWorld("outer", {"X": 1}, reads={("M", "Y"): "inner"}),
    ),
    event=(t.CounterfactualEvent("Y", 1, "outer"),),
    sources=(t.CounterfactualTransportSource("source", selections=["Z"]),),
)
decision = t.identify_counterfactual(graph=graph, query=query, catalog=catalog)
decision.status            # identified | proven_non_transportable | missing_evidence | ...
result = decision.derivation.evaluate(evidence=laws)   # exact laws; point only
artifact = result.export()
again = t.consume_counterfactual_transport_artifact(
    artifact,
    expected_premises_digest=result.premises_digest,
    expected_data_digest=result.data_digest,
)
```

A decision that is not identified carries its typed reason (`reason_code`,
`detail`); `missing_regimes` names the experiments that would license a
derivation, and `witness` and `witness_values` give the verified two-model
witness of an obstruction. Preparing it raises the same refusal.

Sampled evidence goes to `decision.derivation.estimate_sampled(tables,
overlap=..., unseen_rule=...)` with `SampledCountTable` values (one per cited
population, regime and intervention world; `cited_regimes` lists them). Overlap
of the units behind the tables is declared with `SamplingOverlap`
(`independent`, `shared_units` or `unknown`; unknown is refused). An unseen cell
is refused, smoothed under `UnseenCellRule.additive_smoothing` (labelled
smoothed) or enveloped, never zero-filled. The estimate keeps four claims in
separate fields, reports the delta-method standard error as
`uncalibrated_standard_error`, and publishes no interval: the interval route
stays closed with `cell_not_licensed` until whole-method coverage is measured.
`consume_sampled_counterfactual_transport_artifact` recomputes the estimate from
the counts in a fresh process.

### What consumes a transported counterfactual probability

The result is a point. The inverse-outcome query and the decision contract do
not take it as a forward source (no API accepts it); a transported point enters
those consumers only as a declared mean source that keeps the point-only label.

The producer row `counterfactual_transport_result` (a `CounterfactualTransportResult`
or its exported artifact) joins [the producer and consumer matrix](producer-consumer-matrix.md)
with the Python producer. Its row exists and every consumer cell is executed: the
decision, inverse-query and recalculation consumers take a caller-declared point
mean as an adapter, and the remaining consumers refuse it with a typed error. The
point is never promoted to a law, an interval or a replayable decision source.

## Scenario invariance report

A transport scenario set says what each selection diagram answered and the range
the answers span. `invariance_report(stage)` says **why**: for every scenario it
reports

- the **selection differences**: the selection targets, the shared mechanisms
  (every other variable) and the directed and bidirected edges the scenario assumes;
- the **invariances the identified formula relies on**: each factor taken from a
  source population, with its variables, conditioning variables, experimental
  regime, district selection targets and the proof rule that produced it. The
  mechanisms of a factor's variables are assumed shared between source and target;
  factors taken from the target population are listed separately because a target
  law is not an invariance;
- for a scenario proven not transportable, the structural **obstruction** (the
  s-hedge forests or the conditional two-model witness) instead of an invariance list.

For each extreme of the structural envelope, the same report of the scenario that
produced it is attached, so a range reads as "0.35 from the scenario selecting on
`z`, which relies on `P_s(y | do(x), z)` and `P*(z)`; 0.56 from the scenario with no
selection, which relies on `P_s(y | do(x))`".

```python
from antecedent.scenario_invariance import invariance_report
from antecedent.transport import advanced as transport

stage = transport.prepare_transport_scenarios(
    scenario_set, outcomes=["y"], treatments=["x"], source="source", target="target",
    catalog=catalog, laws=laws, at={"x": 1.0},
)
stage.estimate()                     # required: an un-estimated stage refuses
report = invariance_report(stage)
print(report.explain())
report.scenario("standardize").report.invariances
report.extreme("y", "lower")
report.to_dict()
```

(`scenario_set`, `catalog` and `laws` are the scenario, evidence-catalog and
exact-law fixtures of `python/tests/test_scenario_invariance.py`.) In that fixture,
selection on `z` relies on one source factor (`y` given `z` under `do(x)`, rule
`transport.pretreatment_standardize`) and one target factor (`z`); no selection
relies on `P_s(y | do(x))` with rule `transport.direct`; selection on `y` is an
s-hedge with an obstruction and no invariance list.

The report is **derived, not stored**: it is a function of the decided scenario set
and the estimated report, so it is recomputed on demand and no artifact byte
changes. Its `identity` is a digest of the canonical text and does not depend on
the order edges or selection targets were supplied or on the scenario's name. Errors
raise `ScenarioInvarianceRefusal`: `scenario_invariance.not_estimated`
(`not_executed`; call `estimate()` first) and `scenario_invariance.wrong_result_type`.

## Learned joint transport and measured source replay

`transport.advanced.learned_joint_transport(...)` uses the original
known-variance conjugate Gaussian polynomial-basis outcome mechanism and original
fixed-DAG transport proof. Its normal measured adapter reports only
`target_effect` through `inference.MeasuredInference`, conditional on current
method-specific evidence.

The measured protocol is quadratic degree 2, one covariate, zero-mean declared
isotropic Gaussian priors of variance 1000, known source variances 1 and 2.25,
disjoint independent source samples with equal sizes 150..600, bounded original
covariates [-1,1] and the fixed target design [-0.25,0.25,0.25,0.55]. The four
measured varying-block/sharing configurations retain the original full joint
covariance. Nominal level is 0.95 with at least 4096 posterior draws; original
native draw/parameter limits still apply. Other bases, prior-bank models,
dependence declarations or target designs cannot borrow these records.

A fresh `MeasuredInference.load(artifact, expected=retained_identity)` consumer
replays the actual source proof, fit, posterior and target effect and resolves
its record again. The underlying posterior artifact stays `unmeasured` outside the named scalar interval. A calibrated
target scalar does not make every coefficient interval calibrated, and source
rows do not authenticate Gaussian correctness or independence. A nonattesting
record or unsupported protocol refuses with `cell_not_licensed`; original
identification, support and evidence-overlap refusals remain intact.

The separately supported learned point-estimate route
`transport.advanced.prepare_learned_continuous` has its own cross-fitted scope
and withheld interval. It does not substitute for this posterior construction.

## Before relying on a transport assumption

A shared mechanism is an assumption. Test it where you can with the
[source-target mechanism discrepancy diagnostic](sensitivity-and-robustness.md),
which informs only a selection node on the tested node and never certifies
invariance, and carry the remaining assumption into the decision as a range with
[sensitivity and robustness](sensitivity-and-robustness.md).

Runnable examples:
[`transported_counterfactual.py`](https://github.com/iridae-dev/antecedent/blob/v2.4.0/examples/python/transported_counterfactual.py) and
[`scenario_invariance.py`](https://github.com/iridae-dev/antecedent/blob/v2.4.0/examples/python/scenario_invariance.py).

Related: [the lifecycle](lifecycle.md),
[transport and interference](transport-interference.md).
