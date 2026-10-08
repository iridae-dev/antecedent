# Sensitivity and robustness (2.3)

A decision often rests on something that cannot be tested from the data: that
there is no unmeasured confounding, that two populations share a mechanism, that
the causal graph is the right one. The 2.3 routes on this page carry such an
assumption into the decision as a declared range, report where the best action
switches, and keep the range apart from sampling uncertainty.

Four modules:

| Module | Question | Claim |
| --- | --- | --- |
| `antecedent.msm_sensitivity` | How far can a stratified ATE move if unmeasured confounding shifts the odds of treatment by at most a factor Lambda? | assumption range; calibration unmeasured |
| `antecedent.sensitivity_decision` | Over a declared assumption grid, does the best action change, and where? | point-only; grid points only |
| `antecedent.decision_robust` | Is the choice robust across causal structures, support-robust, graph-dependent or under-claimed? | point-only verdicts over supplied structures |
| `antecedent.mechanism_discrepancy` | Does the mechanism of one node differ between a source and a target population? | asymptotic Wald diagnostic; calibration unmeasured |

Registry authority is `parity/promotion_2_3.toml`; this page restates it. For the
label vocabulary (point-only, no claim, closed pending calibration) see
[decisions, design, repair and estimator breadth](2_3-decisions-breadth.md).

## Three quantities that are never merged

Every route here keeps these apart, and refuses to add them together:

- the **assumption range** is what a quantity could be if the assumption holds
  anywhere inside the declared set. It is not a probability and not a confidence
  interval, so `weights` on a ranged surface refuse;
- the **identified value** is the value with no unmeasured confounding
  (Lambda = 1 for the MSM);
- the **sampling interval** is reported next to the decision, never inside the
  range. The MSM strata are supplied as exact quantities, so no sampling interval
  is reported at all (`sampling interval: not reported`), and asking to compose
  one with the range is refused with `cell_not_licensed`
  (`msm_sensitivity.composition_not_licensed`, or
  `sensitivity_decision_composition.composition_not_licensed` at decide time).

## Marginal sensitivity model: `msm_ate_sensitivity`

The Tan (2006) / Zhao-Small-Bhattacharya (2019) marginal sensitivity model for a
binary treatment, a discrete adjustment set (finite strata) and exact population
inputs. Strata can be built by hand or from a plain table with
`MsmStratum.table`.

```python
from antecedent import msm_sensitivity
from antecedent.msm_sensitivity import MsmStratum, OutcomeLaw

strata = [
    MsmStratum(0.5, 0.50, OutcomeLaw.binary(0.8), OutcomeLaw.binary(0.4)),
    MsmStratum(0.5, 0.25, OutcomeLaw.binary(0.5), OutcomeLaw.binary(0.3)),
]
result = msm_sensitivity.msm_ate_sensitivity(strata, 3.0, decision_threshold=0.0)

result.identified          # 0.3: 0.5 * 0.4 + 0.5 * 0.2
result.assumption_range(2.0)   # (0.04375, 0.4875): sharp ATE bounds at Lambda = 2
result.tipping.bracket     # certified bracket around (2 + sqrt(193)) / 7 = 2.27035
print(result.explain())
```

`MsmStratum.table` reads a mapping of equal-length columns, a sequence of row
mappings or anything with `to_dict("records")` such as a pandas `DataFrame`. A
`treated` or `control` cell may be an `OutcomeLaw`, a `{"values", "probabilities"}`
mapping, a `(values, probabilities)` pair, or a bare number read as `P(Y = 1)`:

```python
columns = {"p_x": [0.5, 0.5], "e": [0.50, 0.25], "y1": [0.8, 0.5], "y0": [0.4, 0.3]}
strata = MsmStratum.table(columns, mass="p_x", propensity="e", treated="y1", control="y0")
assert abs(msm_sensitivity.msm_ate_sensitivity(strata, 3.0).identified - 0.3) < 1e-12
```

Nothing is estimated or normalized: the table must already hold exact stratum
quantities.

What you get back:

- `grid`: sharp lower and upper ATE bounds on Lambda from 1 to `lambda_max`
  (17 equally spaced points by default). The ranges nest, so they only widen.
- `tipping` (when `decision_threshold` is given): the smallest Lambda at which the
  identified-side bound reaches the threshold, found by bisection. Its `status` is
  `bracketed` (a certified bracket: not reached at `bracket[0]`, reached at
  `bracket[1]`), `reached_at_origin` or `not_reached_in_box`.
- `inference_claim == "assumption_range"` and
  `uncertainty.sampling_interval == "sampling interval: not reported"`, always.

Refusals use the `msm_sensitivity.*` namespace: `lambda_below_one`,
`lambda_range_empty` (a `lambda_max` of exactly one), `positivity` (a propensity
of 0 or 1), `stratum_mass`, `outcome_law`, `invalid_threshold`, `invalid_tolerance`
and `bounds_exceeded` (`lambda_max` above 1000). All raise `MsmSensitivityRefusal`,
a `StructuredRefusal`.

## Carrying a range into a decision: `sensitivity_decision`

`MsmResult.to_sensitivity_artifact` turns the Lambda surface into a
`SensitivityArtifact`, and `sensitivity_decision.decide` evaluates a contract
over it. Actions are `SensitivityAction` values whose utilities read surface
quantities through `sensitivity_decision.quantity(variable_id)`. They share the
one `decision.Expr` type with decision contracts, so `sd.const`, `sd.maximum` and
`sd.minimum` are the same builders; each declaration refuses the other's leaf
(`decision.x(i)` for contracts, `sd.quantity(id)` for surfaces).

```python
from antecedent import sensitivity_decision as sd
from antecedent.joint_distribution import ScientificQuantity

ate = ScientificQuantity(
    variable_id="ate", variable_name="ate", role="outcome", units="utils",
    population_id="target", regime_id="do(a=1)", horizon=0, functional_id="msm_ate",
)
result = msm_sensitivity.msm_ate_sensitivity(strata, 2.0, grid_points=3)
artifact = result.to_sensitivity_artifact(
    effect=ate,
    actions=[
        sd.SensitivityAction("treat", sd.quantity("ate")),
        sd.SensitivityAction("skip", sd.const(0.0)),
    ],
    causal_contract_id="checked-contract",
)
decided = sd.decide(artifact.contract(), artifact)
decided.kind                  # "invariant_action": treat leads at every Lambda in [1, 2]
decided.invariant_action      # "treat"
```

`decided.kind` is one of:

- `invariant_action`: one action leads at every evaluated point and every range
  vertex;
- `assumption_dependent`: the leader switches. `decided.switch` gives the
  `from_actions`, `to_actions`, the `bracket` and whether the crossing is `exact`
  (a tie on a grid point) or interpolated, which is labelled an interpolation;
- `no_robust_action`: the best action depends on where inside the range the
  surface lies (`outcome.reason == "mixed_within_range"`);
- `unresolved`.

Only the declared grid points are evaluated; nothing between them is. `range=`
restricts to an inclusive coordinate sub-range. `weights=` are genuine
probabilities, one per grid point, and only for a point surface; weights on a
ranged surface refuse. A `SensitivityArtifact` exports, and
`SensitivityArtifact.consume(data, expected_identity=artifact.identity)` recomputes
it and refuses a resealed or foreign artifact. It can be embedded in a
[composition bundle](2_3-composition.md) as a `sensitivity` node.

A 2.2 joint mechanism sensitivity enters the same way through
`SensitivityArtifact.from_joint_sensitivity(...)`; `from_surface(...)` takes a
hand-supplied surface (the tests derive their oracles by hand: a gamma grid
`{0, 1, 2}` with action A worth `2 - gamma` against B worth 1 leads A, ties at 1
and leads B).

Refusals use `SensitivityRefusal` with detail in
`sensitivity_decision_composition.*`: `composition_not_licensed`, `wrong_contract`
(mixed estimands, unlike units, weights on a ranged surface), `unsupported_coordinate`,
`invalid_surface`, `bounds_exceeded`.

## Robustness across structures: `decision_robust`

When the claims are not one law (several candidate graphs, a support shortfall, an
identified set), `decision_robust` adds admissibility rules and a verdict. Rules
only remove actions where the structure violates them; none becomes a penalty.
Unidentified or unevaluated mass is reported and never renormalized away, a
structural envelope is not a probability law, and CPDAG completion counts are not
probabilities.

```python
from antecedent import decision_robust

contract = decision_robust.admissible_contract(base_contract, None)   # or AdmissibilityRules(...)
claims = [
    decision_robust.Claim.evaluated("s1", law_1, support=decision_robust.Support("supported")),
    decision_robust.Claim.evaluated("s2", law_2, support=decision_robust.Support("supported")),
]
robust = decision_robust.robust(contract, claims, kind="graph_dependent")
robust.verdict.kind       # "structurally_robust", or "graph_dependent_choice" when they disagree
robust.selected           # the action, or None when the choice depends on the structure
print(robust.explain())
```

Verdict kinds include `structurally_robust`, `support_robust`, `support_dependent`,
`graph_dependent_choice`, `unsupported_extrapolation`, `insufficient_claims`,
`no_admissible_action`, `worst_case_choice`, `bayes_choice` and `report_only`.
`finite_scenarios`, `graph_dependent`, `weighted_atoms` and `point_claim` are
shorthand for the claim kinds, and `identified_sets` handles intervals. Bayes over
an identified set has no probability law and is refused. A decision whose value
came from an external callback keeps its attested value and trust limit in the
artifact and is never labelled natively verified. The scenario adapters
(`antecedent.scenario_decision`) feed
[transport scenarios](2_3-transport-counterfactuals.md) into the same verdicts.

## Source-target mechanism discrepancy: `diagnose_mechanism_discrepancy`

A Wald test of the null that the conditional mechanism of one node `V` given its
parents is the same in a source and a target population, from comparable
measurements. Use it before relying on a transport assumption that a node's
mechanism is shared.

```python
from antecedent import mechanism_discrepancy as md

measurement = md.Measurement("V", "mg", parents=[("x", "cm")], protocol_id="protocol-1")
source = md.Sample("source", outcome=[1.0, 3.0, 2.0, 4.0], parents={"x": [0.0, 1.0, 2.0, 3.0]})
target = md.Sample.from_rows(
    "target",
    [(1.5, {"x": 0.0}), (4.0, {"x": 1.0}), (3.5, {"x": 2.0}), (6.0, {"x": 3.0})],
)
result = md.diagnose_mechanism_discrepancy(
    source=source, target=target, measurement=measurement, compare_intercept=True
)
result.statistic, result.degrees_of_freedom, result.p_value
result.conclusion                       # "not_rejected" | "rejected"
result.non_rejection_certifies_invariance   # always False
result.coefficients                     # per-coefficient breakdown, Holm-adjusted
print(result.explain())
```

`Sample` takes columns (`outcome=`, `parents=`), `Sample.from_rows` takes
`(outcome, {parent: value})` records, and `Sample.from_summary` takes sufficient
statistics instead of rows. The statistic is `W = d' (V_s + V_t)^-1 d` over the
compared coefficients, chi-square with `df` equal to their number.

Read it as a diagnostic, not a certificate:

- **Non-rejection never certifies invariance.** `non_rejection_certifies_invariance`
  is always `False`; only differences larger than each coefficient's
  `minimal_detectable_difference` could have been detected.
- It **informs only a selection node on the node itself**
  (`informs_selection_on == ("V",)`): a rejection says that node's mechanism is
  not invariant, so a selection node there cannot be excluded; non-rejection
  leaves it open and says nothing about selection nodes on other variables.
- It is a linear-Gaussian-mean diagnostic: a shift in a nonlinear or higher-moment
  feature may go undetected. Type I error and power are **unmeasured**
  (`calibration == "unmeasured"`, `inference_claim ==
  "asymptotic_wald_calibration_unmeasured"`).
- The test is valid only for **independent** samples. Shared units
  (`dependence="shared_units"` or a common `unit_ids` entry) or an unknown
  dependence refuse as `dependence_unknown`.

The result exports and replays by recomputation:
`MechanismDiscrepancyResult.consume(data, expected_identity=result.identity)`
refuses a resealed mutation (`mechanism_discrepancy.wrong_contract`). `identity`
is a mapping (`measurement_id`, `source_evidence_id`, `target_evidence_id`,
`null`, `design_id`, `digest`).

Refusals use `MechanismDiscrepancyRefusal` with detail
`mechanism_discrepancy.<slot>`: `incomparable_measurements` (a different node,
parent set, unit or protocol id, or a blank declaration; `route_not_supported`),
`dependence_unknown`, `rank_deficient_design`, `degenerate_covariance`,
`sample_too_small`, `non_finite_value`, `row_count_mismatch`,
`inconsistent_summary`, `invalid_alpha`, `invalid_power`.

## What is carried forward, and what is not measured

| Item | Carried forward | Not measured or not claimed |
| --- | --- | --- |
| MSM range | into a `SensitivityArtifact`, a decision, a bundle node | coverage of any interval; a sampling interval (withheld) |
| Tipping Lambda | bracket with a certified bisection tolerance | a probability that the threshold is crossed |
| Assumption-dependent switch | exact tie, or a labelled interpolation | anything between grid points |
| Structural verdict | per-structure values, ranges, unevaluated mass | a probability over structures, unless declared and genuine |
| Mechanism discrepancy | statistic, p-value, minimal detectable differences, `informs_selection_on` | Type I error, power; invariance on non-rejection |

Every standard error, p-value and interval on this page is a diagnostic with
calibration unmeasured until the 2.3 release cut. It is reported so it can be
inspected, not as a confidence statement.

Runnable examples:
[`msm_sensitivity.py`](../examples/python/msm_sensitivity.py) and
[`mechanism_discrepancy.py`](../examples/python/mechanism_discrepancy.py).

Related: [the 2.3 lifecycle](2_3-lifecycle.md),
[transport and counterfactuals](2_3-transport-counterfactuals.md),
[decisions](2_3-decisions.md).
