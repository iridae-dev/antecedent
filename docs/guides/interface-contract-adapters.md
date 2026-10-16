# Interface-contract adapters and evidence overlap

Mechanism, bridge and measurement-model declarations are instances of the one
interface-contract schema, decided by the one connection check. Whether two
components share evidence is decided by the evidence ledger's row-level
overlap, not by comparing identity strings.

Rust surface: `antecedent_contract::{bridge, measurement, mechanism_composition,
interface_contract, joint_law_rule}`. Provenance records are the ones of the
routes that consume them; the adapters add no statistic.

## Adapters

| Declaration | Port names | Adapter |
| --- | --- | --- |
| Bridge | `source`, `target` | `BridgeContract::interface_contract` with a `BridgeInterfaceBasis` |
| Measurement model | `indicator`, `construct` | `measurement_interface_contract` with a `MeasurementInterfaceBasis` |
| Mechanism composition | `MECHANISM_OUTPUT_PORT` | `MechanismInterfaceBasis` |

An adapter takes `PortFacts` explicitly: support, estimand identity,
uncertainty meaning and measurement models. Nothing is inferred. An unstated
support is real-valued, no uncertainty meaning is stated, and no measurement
model is claimed. The produced calibration is `unmeasured` when a value carries
uncertainty and `point_only` otherwise; it is never measured by default, and a
bridge never raises a calibration its source did not hold. Licensing remains the
declaration's own check; the projection licenses nothing a connection does not.

Adapted ports are checked by the one connection check, so a population
mismatch is `interface_contract.population_mismatch`, an operation the producer
does not offer (point, mean, law, draws, joint draws, covariance, interval,
posterior; none implies another) is `operation_unavailable`, and a measurement
model the quantity was not established under is
`measurement_assumption_mismatch`.

## Evidence overlap

`check_composition_with_ledger` and `check_propagation_with_ledger` read overlap
from the ledger. The same source, or two distinct sources that share
observations (a subset, a resample, a derived source), refuse as
`interface_contract.shared_evidence`. A source the ledger does not hold or whose
observations are not computable is `dependence_unknown`: unknown is never
disjoint. Without a ledger, `disjoint_requirements_without_ledger` refuses
rather than skipping the requirement.

## Ranking pairing

`check_rows_pairable_across` decides whether the rows of several inputs may be
paired: different alignment identities refuse as `alignment_mismatch`, and an
input with no aligned draws refuses as `index_pairing_refused`. A covariance or a
declared independence is never a route to row pairing.

## What is refused, and the remedy

| Refused request | Detail | Remedy |
| --- | --- | --- |
| Components with shared observations where disjoint evidence is required | `interface_contract.shared_evidence` | Use disjoint sources |
| Overlap not computable | `interface_contract.dependence_unknown` | Register the sources in the ledger |
| Independence declared where overlap is unknown | `joint_law_propagation.evidence_overlap_unknown` | Decide the overlap in the ledger first |
| Inputs with different alignment identities | `alignment_mismatch` | Supply draws from one joint law |
| Bridge value kind that names no operation | port with no operations | Declare a carried value kind |
