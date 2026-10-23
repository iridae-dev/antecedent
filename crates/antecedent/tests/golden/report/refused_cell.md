# Causal analysis report

## Summary

- **Reading:** a contract before execution
- **Query:** PathSpecificEffect
- **Inference:** frequentist
- **Identifier:** path\_specific.natural
- **Estimator:** functional.effect
- **Support status:** refused

- No claim is attached: this report reads the contract before any execution, and no estimate is stated.
- The cell is in the product but is not licensed, and the matrix refuses it.

## Identification

- **Graph class:** Admg
- **Structure source:** explicit
- **Identifier:** path\_specific.natural
- **Estimator:** functional.effect
- **Resolved estimator:** none
- **Discovery algorithm:** none

The identification slot is unavailable: not\_prepared.

## Assumptions

0 obligations are recorded; 0 remain unresolved.

## Diagnostics and evidence

No claim is attached, so no execution evidence or diagnostics are recorded.

## Uncertainty

The uncertainty slot is unavailable: execution\_specific.

### Calibration

Calibration is read from a claim; none is attached.

## Support and licence

- **Support status:** refused
- **Modality:** tabular
- **Inference:** frequentist
- **Query kind:** PathSpecificEffect
- **Rows:** 200
- **Functional:** all\_observed

### Support slot

- **Matrix status:** refused
- **Matrix coordinate:** PathSpecificEffect:Admg:explicit:Frequentist:full
- **Empirical support:** unavailable:not\_evaluated

## Refusals and limits

- **Operation:** inspect
- **Applicability:** unlicensed
- **Readiness:** none
- **Coordinate:** PathSpecificEffect:Admg:explicit:Frequentist:full

No limit is recorded beyond the blockers below.

### Blockers and refusals

| Id | Scientific impossibility | Reason |
| --- | --- | --- |
| support.refused | yes | Path-specific queries execute only on a supplied static Dag; a directly supplied Admg, TemporalAdmg, or Pag hits the same static-Dag requirement (accepted/graph-posterior Admg and Pag are already refused above). |

### Next actions

Next actions are suggestions for the caller; none is taken automatically.

| Action | Description |
| --- | --- |
| consider\_licensed\_neighbor | choose an explicit licensed neighbor; never an automatic fallback |

### Licensed neighbours

A neighbour changes the stated premises; it is not a fallback.

| Coordinate | Changed premises | Required action |
| --- | --- | --- |
| AverageEffect:Admg:explicit:Frequentist:full | query: PathSpecificEffect to AverageEffect | change the query; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:full | query: PathSpecificEffect to InterventionResponse | change the query; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:full | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:full | query: PathSpecificEffect to AverageEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:cheap | query: PathSpecificEffect to AverageEffect; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:none | query: PathSpecificEffect to AverageEffect; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:full | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Frequentist:none | query: PathSpecificEffect to ConditionalEffect; validation: full to none | change the listed premises; do not relabel the graph class |

- **Layers the premises preserve:** target
- **Layers the premises invalidate:** support, identification, results

This report restates the contract. It computes no estimate, grants no licence, and a status recorded here is the status the records carry.

## Identities

- **Target:** d947cc7b8de5b280686f6410de9659f9be3a4afa032d01c120d57a48f132bc1b
- **Identification:** 29e49e714b185af8a7299d26ea3f1427f16b3c4ef30b34868938a1b02cff9ec4
- **Identification product:** none
- **Program:** none
- **Inference binding:** dfbaf4691a9aed9b74147e09af2c59cfaf3d9fd478889631a5173f0bac005f66
- **Observation:** cc659f942a090ca3046701bc26734487d0de84af65e670e49bc5abe48aef151c
- **Data snapshot:** d3f991cc994df9d7757ec98159b5f3dbf5ed8e580cc3eaba9c79184522cf72f2
- **Score reuse:** none
- **Target weights:** none

