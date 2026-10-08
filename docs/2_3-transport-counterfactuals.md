# Transport, scenarios and transported counterfactuals (2.3)

Three 2.3 routes answer "what changes when the population changes?" and are
honest about where they stop:

| Route | Python entry | Claim |
| --- | --- | --- |
| Transported path-specific effect, affine-additive class | `antecedent.transported_counterfactual` | point-only under a fully specified structural fit; calibration unmeasured |
| Selection differences and invariances behind each scenario answer | `antecedent.scenario_invariance` | derived report, nothing sealed |
| Learned joint source-target transport | `antecedent.transport.advanced.learned_joint_transport` | closed; calibration unmeasured |

For the ordinary 2.0 transport compiler (`transport.Transport(...)`) see
[transport and interference](transport-interference.md) and
[transport failures](guides/transport-failure.md). This page covers what 2.3 adds.

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
never evaluates and always raises a typed refusal.

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

## Learned joint transport is closed

`transport.advanced.learned_joint_transport(...)` fits a known-variance Bayesian
polynomial-basis outcome mechanism over a fixed DAG and selection assumptions and
would answer the target average effect with a posterior interval. The Rust core
and its artifact replay, but the posterior claims a **calibrated** interval whose
coverage is measured only at the release cut, so the public producer is closed:
the function never returns.

A request is first checked against the row's scope, so you learn the real
obstruction:

| Request | Refusal |
| --- | --- |
| a graph with a bidirected edge, or a graph posterior | `route_not_supported`, `learned_joint_transport.unsupported_graph` |
| basis degree outside 1..6, draws outside 1..100000, more than 256 parameters | `invalid_argument` (`invalid_basis`, `too_many_draws`, `too_many_parameters`) |
| undeclared or overlapping source dependence | `sampling_dependence_unknown` |
| no source, no target law | `transport_missing_evidence`, `joint_law_required` |
| every other request | `cell_not_licensed`, `learned_joint_transport.route_frozen` |

The route-frozen refusal names its **remedy**: use the open learned point-estimate
route `antecedent.transport.advanced.prepare_learned_continuous` (cross-fitted,
interval withheld), or an identified closed-form transport estimate; the learned
joint posterior opens only when its coverage is measured.

```python
from antecedent.errors import CausalUnsupportedError
from antecedent.transport import advanced as transport

try:
    transport.learned_joint_transport(
        sources=[{"id": "s1"}, {"id": "s2"}], target={"x": [0.1, 0.2, 0.3]},
        features=["x"], basis_degree=2, draws=1000, seed=7,
    )
except CausalUnsupportedError as refusal:
    refusal.reason_code     # "cell_not_licensed"
    refusal.remedy          # the open point-estimate route above
```

## Before relying on a transport assumption

A shared mechanism is an assumption. Test it where you can with the
[source-target mechanism discrepancy diagnostic](2_3-sensitivity-and-robustness.md),
which informs only a selection node on the tested node and never certifies
invariance, and carry the remaining assumption into the decision as a range with
[sensitivity and robustness](2_3-sensitivity-and-robustness.md).

Runnable examples:
[`transported_counterfactual.py`](../examples/python/transported_counterfactual.py) and
[`scenario_invariance.py`](../examples/python/scenario_invariance.py).

Related: [the 2.3 lifecycle](2_3-lifecycle.md),
[transport and interference](transport-interference.md).
