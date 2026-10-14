# Transport assumption ranges and revalidation

Every transported value rests on a premise the data cannot test. Two routes say
what happens when the premise fails or the situation changes. Neither measures
anything.

| Route | Rust surface | Label |
| --- | --- | --- |
| Assumption range for a violated premise | `antecedent_design::transport_sensitivity::transport_assumption_range` | assumption range at declared grid points |
| Revalidation after a change of target, evidence, diagram or support | `antecedent_contract::transport_revalidation::revalidate_transport` | `recompute`, `withdrawn`, `still_valid` |

Python entries: `antecedent.transport.sensitivity.transport_sensitivity`
(a `TransportSensitivity` whose `.artifact` is a `SensitivityArtifact`),
`antecedent.transport.revalidation` (`TransportScene.builder`, `revalidate`,
`TransportClaimState.observe`, `Calibration`, `ObservedEffect`); the range's
artifact is decided and exported through `antecedent.sensitivity_decision`.
Provenance records:
`transport.assumption_range`, `transport.revalidation`.

## Assumption range

The violation family has one parameter: the transported quantity differs from
the point value by a shift `s` with `0 <= s <= bound`, additive (in the
quantity's units) or proportional (a fraction of the point value). At shift `s`
the range is `[v - s, v + s]` or `[v - s|v|, v + s|v|]`, evaluated at the
declared grid. It is built into the existing sensitivity artifact, so tipping
and decision consumption apply unchanged. `reference_reached_at` reports the
shift at which a reference value is reached. The range is built from a
transported response's own point value, coordinate and receipt, so the sealed
artifact is bound to that target and quantity; a response whose value is a law
has no single point to shift and is refused
(`transport_sensitivity.law_response_not_scalar`).

It is an assumption range. It is not a confidence or credible interval, no
probability attaches to a shift, the `bound` is your declaration and the data do
not license it, and a sampling interval is not composed into it. The route set
covers cohort decisions, statistical and z-transport, path-specific effects,
measurement, mechanism, synthesis, bridge, policy, temporal policy, adaptive
trial source and provider handoff.

## Revalidation

A `TransportScene` declares the target population, sources, selection diagram,
claims (with their calibration standing) and decisions. `revalidate_transport`
compares two scenes through the existing claim-level graph:

- `recompute`: the target, a source's evidence, the diagram or the claim's own
  derivation changed;
- `withdrawn`: the target or claim is outside its support, a provider's trust no
  longer holds, an input was retired, or calibration does not hold at the
  claim's coordinate;
- `still_valid`: nothing it rests on changed.

The report exports as the claim-level verification
(`composition_verification_v1`), which a consumer replays. The io-core crate
also defines `transport_revalidation_v1`, which carries both scenes, the
effects and the classification and re-derives the whole report on consume,
keeping the transport reasons; `TransportRevalidation.export` produces it in
Python.

A change of standing is itself a change: a provider's trust moving between
snapshots, a calibration record replaced, or support restored recomputes the
claim, and a provider trust the scene leaves undeclared withdraws it. A refused
refresh leaves the held scene as it was (`TransportClaimState.observe`).

A record held at one target is never read as holding at another: the claim is
withdrawn for calibration with `CalibrationFromAnotherTarget`, and its point
value is left as it was.

## What is refused, and the remedy

| Refused request | Remedy |
| --- | --- |
| Range for a discovered structure (no scalar family) | Use the structural envelope; declare a graph |
| Missing shift bound, grid or population/quantity bindings | Declare them; none is chosen for you |
| Reading the range as an interval or posterior | Not licensed; use as a tipping or sensitivity statement |
| Calibration carried from another target | Withdrawn; measure at the new target |
| Reusing an earlier fit after the diagram changed | `recompute`: derive again |
| A scene that names a claim twice, an unknown claim, a claim about another target or a source with no declared evidence | `transport_revalidation.duplicate_claim`, `unknown_claim`, `claim_target_mismatch`, `source_undeclared` | Correct the scene |
| A stored revalidation report that no longer re-derives | `transport_revalidation.replay_mismatch`, `identity_mismatch` | Re-seal from the two scenes; the consumer re-derives |
