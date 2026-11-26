# Assumption ranges and revalidation

You can ask how far a transported value moves if its shared-mechanism premise is wrong. You can ask whether a held transported claim is still valid after the target, the evidence, the diagram or the support changes. Every transported value rests on a premise the data cannot test. These two calls say what follows when the premise fails or the situation changes. Neither measures anything.

| Route | Rust surface | Label |
| --- | --- | --- |
| Assumption range for a violated premise | `antecedent_design::transport_sensitivity::transport_assumption_range` | assumption range at declared grid points |
| Revalidation after a change of target, evidence, diagram or support | `antecedent_contract::transport_revalidation::revalidate_transport` | `recompute`, `withdrawn`, `still_valid` |

In Python, the range call is `antecedent.transport.sensitivity.transport_sensitivity`.
You get a `TransportSensitivity`. Its `.decision_source` is a `SensitivityArtifact`.
Revalidation is `antecedent.transport.revalidation`. The names there are
`TransportScene.builder`, `revalidate`, `TransportClaimState.observe`, `Calibration` and `ObservedEffect`.
The range's artifact is decided and exported through `antecedent.sensitivity_decision`.
Both start from a `TransportedResponse` (`antecedent.transported_response`).
Every transport call that reaches a decision produces one. For the other
transport questions, see [which transport page answers your question](transport/carry-an-effect.md#which-transport-page-answers-your-question).

## Assumption range

In this sketch `response` is a `TransportedResponse` from a cohort decision,
and `reference` is the `ScientificQuantity` of the alternative the decision
keeps, in the same population and units:

```python
from antecedent.transport import sensitivity

result = sensitivity.transport_sensitivity(
    response,
    family="additive_shift",       # or "proportional_shift"
    bound=0.5,                     # your declaration of how far the premise may fail
    grid_points=3,                 # shifts 0, 0.25 and 0.5
    reference=reference,
    reference_value=1.75,
    route="cohort_decision",
)
result.lower, result.upper         # the range at each shift in result.grid
result.reference_reached_at        # smallest shift that reaches 1.75, or None within the bound
result.statement                   # the wording that must accompany the range
result.decision_source             # decide through antecedent.sensitivity_decision
```

The violation family has one parameter. The transported quantity differs from
the point value by a shift `s` with `0 <= s <= bound`, additive (in the
quantity's units) or proportional (a fraction of the point value). At shift `s`
the range is `[v - s, v + s]` or `[v - s|v|, v + s|v|]`, evaluated at the
declared grid. It is built into the sensitivity artifact. Tipping
and decision consumption apply to it directly. `reference_reached_at` reports the
shift at which a reference value is reached. The range is built from a
transported response's own point value, coordinate and receipt. The sealed
artifact is bound to that target and quantity. A response whose value is a law
has no single point to shift and is refused
(`transport_sensitivity.law_response_not_scalar`).

It is an assumption range. It is not a confidence or credible interval. No
probability attaches to a shift. The `bound` is your declaration. The data do
not support it. A sampling interval is not composed into it. The calls
cover cohort decisions, statistical and z-transport, path-specific effects,
measurement, mechanism, synthesis, bridge, policy, temporal policy, adaptive
trial source and provider handoff.

## Revalidation

In this sketch `previous_scene` is the scene you held, the identities are
content digests of the data each population supplied, and `response` is the
transported claim:

```python
from antecedent.transport import revalidation as rv

scene = (
    rv.TransportScene.builder("clinic_b", target_data_identity)
    .source("trial", trial_evidence_identity)
    .transported("effect", response, rv.Calibration.point_only())
    .decision("treat_or_wait", decision_identity, quantity_identity, rests_on=["effect"])
    .build()
)
report = rv.revalidate(previous_scene, scene)
report.claim("effect").classification   # "recompute" when the trial's evidence changed
report.claim("effect").reasons          # such as "source_evidence_changed"
report.require_valid("effect")          # raises with a stated remedy otherwise
```

A `TransportScene` declares the target population, sources, selection diagram,
claims (with their calibration standing) and decisions. `revalidate_transport`
compares two scenes through the claim-level graph. The classifications are:

- `recompute`: the target, a source's evidence, the diagram or the claim's own
  derivation changed;
- `withdrawn`: the target or claim is outside its support, a provider's trust does
  not hold, an input was retired, or calibration does not hold at the
  claim's coordinate;
- `still_valid`: nothing it rests on changed.

`TransportRevalidation.export` writes the claim-level verification
(`composition_verification_v1`).
`antecedent.composition_verification.consume_composition_verification_artifact`
replays it. In Rust,
`antecedent_io::transport_revalidation_artifact` also defines
`transport_revalidation_v1`. That artifact carries both scenes, the effects and the
classification. On consume it derives the whole report again, keeping the
transport reasons.

A change of standing is itself a change. A provider's trust moving between
snapshots recomputes the claim. A calibration record replaced recomputes the claim. Support restored recomputes the claim. A provider trust the scene leaves undeclared withdraws it. A refused
refresh leaves the held scene as it was (`TransportClaimState.observe`).

A record held at one target is never read as holding at another. The claim is
withdrawn for calibration. The reason is `calibration_from_another_target`. In Rust, that reason is `CalibrationFromAnotherTarget`. Its point value is left as it was.

## What is refused, and what to do

Refusals raise `TransportedResponseRefusal` with a registered `reason_code`, a
`transport_sensitivity.*` or `transport_revalidation.*` detail and a remedy.
The full list is in [refusal codes: assumption range and revalidation](../reference/refusal-codes.md#assumption-range-and-revalidation).

| Refused request | Next step |
| --- | --- |
| Range for a discovered structure (no scalar family) | Use the structural envelope; declare a graph |
| Missing shift bound, grid or population/quantity bindings | Declare them; none is chosen for you |
| Reading the range as an interval or posterior | Not supported; use as a tipping or sensitivity statement |
| Calibration carried from another target | Withdrawn; measure at the new target |
| Reusing an earlier fit after the diagram changed | `recompute`: derive again |
| A scene that names a claim twice, an unknown claim, a claim about another target or a source with no declared evidence | Correct the scene (`transport_revalidation.duplicate_claim`, `unknown_claim`, `claim_target_mismatch`, `source_undeclared`) |
| A stored revalidation report that does not re-derive | Re-seal from the two scenes; the consumer re-derives (`transport_revalidation.replay_mismatch`, `identity_mismatch`) |

## Related

- [Carry an effect to another population](transport/carry-an-effect.md)
- [Sensitivity and robustness](../sensitivity-and-robustness.md): the source-target mechanism discrepancy diagnostic, which informs a selection node and never certifies invariance
- [Joint mechanism sensitivity](joint-mechanism-sensitivity.md)
