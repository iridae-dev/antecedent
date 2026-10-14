# Transport assumption ranges and revalidation

Every transported value rests on a premise the data cannot test. Two routes say
what happens when the premise fails or the situation changes. Neither measures
anything.

| Route | Rust surface | Label |
| --- | --- | --- |
| Assumption range for a violated premise | `antecedent_design::transport_sensitivity::transport_assumption_range` | assumption range at declared grid points |
| Revalidation after a change of target, evidence, diagram or support | `antecedent_contract::transport_revalidation::revalidate_transport` | `recompute`, `withdrawn`, `still_valid` |

Provenance records: `transport.assumption_range`, `transport.revalidation`.

## Assumption range

The violation family has one parameter: the transported quantity differs from
the point value by a shift `s` with `0 <= s <= bound`, additive (in the
quantity's units) or proportional (a fraction of the point value). At shift `s`
the range is `[v - s, v + s]` or `[v - s|v|, v + s|v|]`, evaluated at the
declared grid. It is built into the existing sensitivity artifact, so tipping
and decision consumption apply unchanged. `reference_reached_at` reports the
shift at which a reference value is reached.

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
