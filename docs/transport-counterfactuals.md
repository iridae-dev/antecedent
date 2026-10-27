# Transport counterfactual and sampled quantities

You can ask "what changes when the population changes?" for quantities beyond an average effect: a natural direct or indirect effect in a target population, a nested counterfactual probability, the reasons behind a range of scenario answers, or a counterfactual probability estimated from counts. Each route below states the class of models it covers and where it stops.

| You want | Python entry | What you get |
| --- | --- | --- |
| A transported natural direct, indirect or total effect from a fitted structural model | `antecedent.transported_counterfactual` | A point under a fully specified fit; the affine-additive class only |
| A nested or conditional counterfactual probability of the target from labelled laws | `antecedent.transport.advanced.identify_counterfactual` | An exact point from exact finite laws |
| The same probability from counted data | `decision.derivation.estimate_sampled` | A plug-in point and an uncalibrated standard error |
| The reasons behind each scenario answer in a range | `antecedent.scenario_invariance` | A derived report; nothing is sealed |
| A target-effect posterior from joint source and target models | `antecedent.transport.advanced.learned_joint_transport` | A measured scalar only at a matching current record; see [Population, time and uncertainty routes](population-time-uncertainty.md) |

For the ordinary transport question see [Carry an effect to another population](guides/transport/carry-an-effect.md) and [when transport fails](guides/transport-failure.md).

## Transport a path-specific effect

A transported natural direct, natural indirect or total effect: the source population has a fitted structural model, the target population has only its covariate law, and the contrast is carried over using the source equations alone.

The class is narrow and declared:

- covariates `Z` are pre-treatment variables whose finite-support joint law may differ between source and target;
- the treatment `A` is only ever set;
- mediator and outcome mechanisms are `V = alpha(Z) + sum_p beta_p(Z) * V_p + U`, where `alpha` and every `beta_p` are affine in `Z` and the noises `U` are additive and independent of `Z` and of each other.

This is the **affine-additive class**. Nonparametric mechanisms, recanting witnesses and nonlinear additive noise are outside it. The general entry `antecedent.temporal_counterfactual.transported_path_specific` is closed: it never evaluates and always raises a typed refusal that names the finite-law route below for the finite discrete, static, acyclic semi-Markovian scope.

The snippet is the tested fixture of `python/tests/test_transported_counterfactual.py`: `M = 0.5 Z + (2 - 0.5 Z) A + U_M`, `Y = (3 + 0.75 Z) A + 4 M + 1.5 Z + U_Y`, with treated value 2 and control value 0.5.

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

`EdgeAssignment.natural_indirect("y", ["m"], ...)` and `.total("y", ["m"], ...)` give the NIE (9.75 on the same fixture) and the total effect (15.09375 = NDE + NIE). The noise cancels in every unit's contrast, so it is not modelled. The target law matters exactly as the arithmetic says: moving mass from `z = 1` to `z = 0` shifts the NDE by the NDE slope 1.125 times the change in `E z`.

The result separates the premises Antecedent **checked** (acyclic well-formed model, no selection on a mediator or outcome mechanism, covariates are pre-treatment roots, target support inside source support, regime evidence present) from the ones you only **declared** (`Premises`: additive noise, shared noise laws, cross-world independence). Declarations default to undeclared; omitting one refuses with `cell_not_licensed` and a detail such as `transported_counterfactual.nonadditive_mechanism`. Declaring one records a claim the supplied source fit must support; it is never checked here.

A selection node marks a variable whose mechanism or covariate law may differ between populations. `MechanismSelection.on("z")` creates `S_z -> z` in this module's structural model; it is not the selection diagram graph of `antecedent.transport.advanced.SelectionDiagram`.

### Refusals and the two-model witness

A selection node on a mediator or outcome mechanism is outside the class, because two models can agree on the whole source population and differ on the target contrast. The refusal is a `TransportedCounterfactualRefusal`:

- `transport_proven_non_transportable` (`transported_counterfactual.selection_on_mechanism`) with a `witness`: two target models that share the source model and its source answer but differ on the target contrast. In the fixture, selecting on `y` moves the NDE by 1.5; selecting on `m` witnesses only the NIE;
- `cell_not_licensed` with no witness when the contrast does not depend on the selected mechanism: the class excludes it and no impossibility is claimed;
- `transport_support_failure` (`overlap_failure`) when the target support leaves the source support;
- `transport_missing_evidence` (`factor_missing`, with `missing_factors`) when regime evidence is absent;
- `invalid_argument` details `invalid_model`, `invalid_query`, `invalid_law`, `invalid_diagram`.

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

`effect.export()` is a checksummed artifact and `effect.identity` the digests a consumer retains. `consume_transported_counterfactual_artifact(data, expected_identity=effect.identity)` replays it and refuses a resealed change to any field. The maintained example is [`transported_counterfactual.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transported_counterfactual.py).

## Transport a nested counterfactual probability from labelled laws

The affine-additive route needs a fitted structural model. The general finite route needs none: it answers a nested or conditional counterfactual probability of the **target** population, such as `P*(Y_{x, M_{x'}} = y)`, a natural direct or indirect effect contrast, or an effect of treatment on the treated, from labelled laws alone.

You declare, and nothing else is accepted:

- finitely many observed variables (at most 8) with declared levels (at most 4 each) on an acyclic graph with bidirected edges for latent confounding;
- a selection diagram per source population: the variables whose mechanism may differ between that source and the target, and the variables the source can experiment on;
- evidence regimes labelled by population and intervention set, each an exact finite joint law (one law per intervention level). Evidence of one population never stands in for another, and separate marginals never create a cross-world law.

The decision is a sound, bounded procedure, not a complete one. A formula is returned only when every factor it needs is either invariant (no selection target lies in it, so a source law transports) or derivable from the target's own laws. The outcomes are kept apart:

| Outcome | Meaning | Reason code |
| --- | --- | --- |
| identified | a checked derivation exists from the supplied laws | none |
| proven non-transportable | two finite models agree on every regime of the declared family and differ on the query, re-verified by exact enumeration | `transport_proven_non_transportable` |
| missing evidence | a derivation exists if the named declared regimes were supplied | `transport_missing_evidence` |
| not certified | the bounded search found neither a derivation nor a witness; this is not a proof | `transport_not_certified` |
| out of contract | more than the bounds above, cyclic, continuous, temporal or sampled | `route_not_supported` |
| stopped | operations, depth, memory or cancellation | `transport_budget_cancel` |

In Python, `identify_counterfactual` decides the query from typed objects: the graph (`Admg`), declared `levels`, worlds that carry hard interventions and edge routes (`reads` names the world a child reads a parent from, which is how `Y_{x, M_{x'}}` is written), `CounterfactualEvent` atoms, per-source `selections` and `controllable`, and an `EvidenceCatalog`. The code below follows `python/tests/test_transport_counterfactual.py`.

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

A decision that is not identified carries its typed reason (`reason_code`, `detail`); `missing_regimes` names the experiments that would license a derivation, and `witness` and `witness_values` give the verified two-model witness of an obstruction. Preparing it raises the same refusal.

In Rust the same route is `antecedent::prepare_counterfactual_transport`:

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

An exact law whose conditioning event has zero mass is a support failure, never an extended value, and a law at an intervention level nobody supplied is a missing provider, never a guessed table. Counted, empirical or posterior laws are refused here (`cell_not_licensed`); counts use the sampled route below.

The `counterfactual_transport_v1` artifact keeps the model class, the query with its selection diagram, the evidence catalog and law tables, the derivation with its search receipt, and the point with the identity of every law it read; an obstruction keeps the two-model witness instead. A premises digest and a data digest bind it. A consumer holding the bytes and an expected identity re-decides under the stored limits, requires the identical derivation, recomputes the point bit for bit and re-verifies a witness by exact enumeration. Replay detects corruption and resealed edits that change the answer; it does not detect a producer that seals a wrong graph, query or law on purpose.

The result is a point. The inverse-outcome query and the decision contract do not take it as a forward source; it enters those consumers only as a declared mean source that keeps the point-only label. See the [producer and consumer matrix](producer-consumer-matrix.md).

## Estimate the probability from counted data

The sampled route evaluates the same stored derivation on laws estimated from counts, keeping the sample's uncertainty apart from the identification. It publishes a point estimate and an uncalibrated standard error and no interval.

```python
estimate = decision.derivation.estimate_sampled(tables, overlap=..., unseen_rule=...)
```

`tables` are `SampledCountTable` values, one per cited population, regime and intervention world (`cited_regimes` lists them), and `overlap` is a `SamplingOverlap`. `consume_sampled_counterfactual_transport_artifact` recomputes the estimate from the counts in a fresh process. The estimate keeps four claims in separate fields:

| Claim | What it says | What it does not say |
| --- | --- | --- |
| Structural identification | the stored derivation identified the query from the declared graph, selection diagram and regime family | anything about the sample |
| Exact finite-law evaluation | the arithmetic of the derivation on the estimated tables is exact | that the tables are the population's laws |
| Point estimate | the plug-in (or smoothed plug-in) value estimates the population probability | a coverage statement |
| Statistical inference | withheld: `cell_not_licensed` (`counterfactual_transport_sampled.interval_not_licensed`) | an interval, posterior or standard-error calibration |

A result is `plug_in_point_estimate` for unsmoothed count shares and `smoothed_plug_in_point_estimate` when any table was smoothed.

**Evidence.** A dense count table per cited population, regime and intervention level over exactly the variables the derivation cites, with a declared sampling design: independent units, or strata whose totals were fixed by design. Declared structural zeros are impossible cells; they must have count zero and are never smoothed. The exact route's refusals carry over: zero conditioning mass is `transport_support_failure`, a cited regime with no table is `transport_missing_provider`, and a table the derivation does not cite is refused.

**Unseen and sparse cells.** An unseen cell is a zero count that is not a structural zero. It is never silently zero-filled. You declare one rule:

| Rule | Result | Label |
| --- | --- | --- |
| refuse | any table with an unseen cell is refused (`transport_support_failure`, `counterfactual_transport_sampled.unseen_cell`) | |
| additive smoothing `a` | `p = (n + a) / (N + K a)` over the `K` non-structural cells (the posterior mean under a symmetric Dirichlet(`a`) prior); `a` is recorded | smoothed |
| unseen-mass envelope `d` | the range of the derivation over the finite family in which each table keeps its observed shares or moves mass `d` onto one unseen cell | structural envelope |

The envelope is a range over that family, not an interval and not a proven outer bound over every placement of the mass.

**Factors that read fewer coordinates.** A district factor is a function of its parents. Read from a sample it differs across the coordinates it does not read by sampling noise, so the sampled route evaluates it as the mass-weighted average over the unread coordinates. The number of averaged factors and any tolerance repair are reported with the result.

**Dependence between tables.** Overlap is declared for every pair of tables. `independent` means disjoint units (two tables declared independent that hold identical observations are refused as `evidence_double_counted`). `shared_units` determines the covariance only when the two tables are the same observations; shared units between different tables are refused (`counterfactual_transport_sampled.shared_units_unlinked`). `unknown`, or nothing declared, is refused (`sampling_dependence_unknown`): independence is never assumed. `overlap_declaration_from_ledger` derives a declaration from an evidence ledger.

**Uncertainty.** The method is the delta method for the multinomial cell proportions of each independent dataset, through the whole functional, with a central-difference gradient. It is deterministic and replayable bit for bit. The standard error is labelled `uncalibrated_standard_error`; at smoothed tables it omits the smoothing bias, and an envelope request has none. It is least reliable near the boundary (sparse tables, small conditioning mass).

**Interval.** The estimate and its artifact never carry an interval. A 95% delta-method Wald interval is available as a separate measured scalar, `SampledCounterfactualTransportEstimate.measure_interval()` in Python, which binds to a coverage record for exactly that construction. The construction is the named queries of the measured designs, independently drawn units, refused unseen cells, unsmoothed tables of equal totals within the measured range (250 to 4,000 units per table) and conditioning mass of at least 0.02. Anything else, including smoothed tables, envelopes, stratified designs, shared datasets, another level or a record that no longer attests, refuses with `cell_not_licensed`, and the point estimate ships labelled.

**Artifact and limits.** `sampled_counterfactual_transport_v1` binds the counts, the derivation and its inputs, the unseen-cell rule, the overlap declaration, the method and the estimate with its four labels; a consumer recomputes the estimate and standard error from the counts bit for bit. At most 64 count tables per request, 65536 cells per table and 65536 envelope members apply, and so do the derivation's bounds (8 variables, 4 levels each, 4 sources). Continuous variables, temporal dependence, clustered or weighted designs and selection into the sample are outside this route.

## Explain a range of scenario answers

A transport scenario set says what each selection diagram answered and the range the answers span. `invariance_report(stage)` says why: for every scenario it reports

- the **selection differences**: the selection targets, the shared mechanisms (every other variable) and the edges the scenario assumes;
- the **invariances the identified formula relies on**: each factor taken from a source population, with its variables, conditioning variables, experimental regime and the proof rule that produced it. Factors taken from the target population are listed separately because a target law is not an invariance;
- for a scenario proven not transportable, the structural **obstruction** instead of an invariance list.

For each extreme of the structural envelope, the report of the scenario that produced it is attached, so a range reads as "0.35 from the scenario selecting on `z`, which relies on `P_s(y | do(x), z)` and `P*(z)`; 0.56 from the scenario with no selection, which relies on `P_s(y | do(x))`".

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

`scenario_set`, `catalog` and `laws` are the fixtures of `python/tests/test_scenario_invariance.py`; the runnable version is [`scenario_invariance.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/scenario_invariance.py). The report is derived, not stored: it is a function of the decided scenario set and the estimated report, so it is recomputed on demand. Calling it before `estimate()` raises `ScenarioInvarianceRefusal` (`scenario_invariance.not_estimated`).

## Learned joint transport

`transport.advanced.learned_joint_transport(...)` fits a known-variance conjugate Gaussian polynomial-basis outcome mechanism jointly across source and target and reports a target effect. Its measured adapter reports the `target_effect` scalar through `inference.MeasuredInference`, conditional on a current record whose protocol matches (a quadratic basis on one covariate, declared isotropic Gaussian priors, disjoint independent source samples, a fixed target design). Other bases, prior-bank models, dependence declarations or target designs cannot borrow that record. The underlying posterior stays model-conditional outside the named scalar, and a calibrated target scalar does not make every coefficient interval calibrated. A fresh `MeasuredInference.load(artifact, expected=retained_identity)` replays the source proof, fit, posterior and target effect. The protocol details are in [Population, time and uncertainty routes](population-time-uncertainty.md#joint-bayesian-transport). The separate point route `transport.advanced.prepare_learned_continuous` has its own cross-fitted scope; see [Transport limits by route](guides/transport-scope.md#learned-continuous-outcome-transport).

## Before relying on a transport assumption

A shared mechanism is an assumption. Test it where you can with the [source-target mechanism discrepancy diagnostic](sensitivity-and-robustness.md), which informs only a selection node on the tested node and never certifies invariance, and carry the remaining assumption into the decision as a range with [assumption ranges and revalidation](guides/assumption-range-and-revalidation.md).

## Related

- [Carry an effect to another population](guides/transport/carry-an-effect.md)
- [Transport limits by route](guides/transport-scope.md)
- [Evaluate actions with a decision contract](decisions.md)
