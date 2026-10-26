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
- **Matrix coordinate:** PathSpecificEffect:Admg:explicit:Frequentist:cheap
- **Empirical support:** unavailable:not\_evaluated

## Refusals and limits

- **Operation:** inspect
- **Applicability:** unlicensed
- **Readiness:** none
- **Coordinate:** PathSpecificEffect:Admg:explicit:Frequentist:cheap

No limit is recorded beyond the blockers below.

### Blockers and refusals

| Id | Scientific impossibility | Reason |
| --- | --- | --- |
| support.refused | yes | Cheap and full validation of a path-specific effect or an interventional distribution on a supplied or accepted Cpdag, Pag, Admg or TemporalAdmg run the functional refuter state and stay unlicensed. The identified functional executes as at validation none (in every completion of a class); a path-specific effect is refuted by the placebo treatment, subset stability and, under full, the placebo outcome and bootstrap stability of the whole procedure, and an interventional distribution by subset and bootstrap stability of its vector of atom probabilities, each a joint statistic over the whole refitted value. The result keeps support status refused: no evidence row or coverage record names the cell, and the verdicts are diagnostics, not a calibration claim. |

### Next actions

Next actions are suggestions for the caller; none is taken automatically.

| Action | Description |
| --- | --- |
| consider\_licensed\_neighbor | choose an explicit licensed neighbor; never an automatic fallback |

### Licensed neighbours

A neighbour changes the stated premises; it is not a fallback.

| Coordinate | Changed premises | Required action |
| --- | --- | --- |
| AverageEffect:Admg:explicit:Frequentist:cheap | query: PathSpecificEffect to AverageEffect | change the query; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:cheap | query: PathSpecificEffect to InterventionResponse | change the query; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:cheap | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:cheap | query: PathSpecificEffect to AverageEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:full | query: PathSpecificEffect to AverageEffect; validation: cheap to full | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:none | query: PathSpecificEffect to AverageEffect; validation: cheap to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:cheap | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Frequentist:none | query: PathSpecificEffect to ConditionalEffect; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Frequentist:cheap | query: PathSpecificEffect to InterventionResponse; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Bayesian:cheap | query: PathSpecificEffect to InterventionResponse; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:full | query: PathSpecificEffect to InterventionResponse; validation: cheap to full | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:none | query: PathSpecificEffect to InterventionResponse; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Frequentist:cheap | query: PathSpecificEffect to InterventionResponse; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Frequentist:none | query: PathSpecificEffect to InterventionalDistribution; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:explicit:Frequentist:none | query: PathSpecificEffect to ResponseCurve; validation: cheap to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Frequentist:none | query: PathSpecificEffect to TransportQuery; validation: cheap to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Bayesian:cheap | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:full | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted; validation: cheap to full | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:none | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted; validation: cheap to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:full | query: PathSpecificEffect to AverageEffect; inference: Frequentist to Bayesian; validation: cheap to full | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:none | query: PathSpecificEffect to AverageEffect; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Bayesian:cheap | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:full | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior; validation: cheap to full | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:none | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Frequentist:none | query: PathSpecificEffect to ConditionalEffect; structure: explicit to accepted; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Bayesian:none | query: PathSpecificEffect to ConditionalEffect; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Bayesian:cheap | query: PathSpecificEffect to InterventionResponse; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Frequentist:full | query: PathSpecificEffect to InterventionResponse; structure: explicit to accepted; validation: cheap to full | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Frequentist:none | query: PathSpecificEffect to InterventionResponse; structure: explicit to accepted; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Bayesian:full | query: PathSpecificEffect to InterventionResponse; inference: Frequentist to Bayesian; validation: cheap to full | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Bayesian:none | query: PathSpecificEffect to InterventionResponse; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Bayesian:cheap | query: PathSpecificEffect to InterventionResponse; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Frequentist:full | query: PathSpecificEffect to InterventionResponse; structure: explicit to graph\_posterior; validation: cheap to full | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Frequentist:none | query: PathSpecificEffect to InterventionResponse; structure: explicit to graph\_posterior; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Frequentist:none | query: PathSpecificEffect to InterventionalDistribution; structure: explicit to accepted; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Bayesian:none | query: PathSpecificEffect to InterventionalDistribution; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:accepted:Frequentist:none | query: PathSpecificEffect to ResponseCurve; structure: explicit to accepted; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:explicit:Bayesian:none | query: PathSpecificEffect to ResponseCurve; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Frequentist:none | query: PathSpecificEffect to ResponseCurve; structure: explicit to graph\_posterior; validation: cheap to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Bayesian:none | query: PathSpecificEffect to TransportQuery; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Bayesian:full | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to full | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Bayesian:none | query: PathSpecificEffect to AverageEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Bayesian:full | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: cheap to full | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Bayesian:none | query: PathSpecificEffect to AverageEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Bayesian:none | query: PathSpecificEffect to ConditionalEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Bayesian:full | query: PathSpecificEffect to InterventionResponse; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to full | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Bayesian:none | query: PathSpecificEffect to InterventionResponse; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Bayesian:full | query: PathSpecificEffect to InterventionResponse; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: cheap to full | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Bayesian:none | query: PathSpecificEffect to InterventionResponse; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Bayesian:none | query: PathSpecificEffect to InterventionalDistribution; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:accepted:Bayesian:none | query: PathSpecificEffect to ResponseCurve; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Bayesian:none | query: PathSpecificEffect to ResponseCurve; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: cheap to none | change the listed premises; do not relabel the graph class |

- **Layers the premises preserve:** target
- **Layers the premises invalidate:** support, identification, results

This report restates the contract. It computes no estimate, grants no licence, and a status recorded here is the status the records carry.

## Identities

- **Target:** d947cc7b8de5b280686f6410de9659f9be3a4afa032d01c120d57a48f132bc1b
- **Identification:** 29e49e714b185af8a7299d26ea3f1427f16b3c4ef30b34868938a1b02cff9ec4
- **Identification product:** none
- **Program:** none
- **Inference binding:** 1825cf17c03045cb5c87a63ed282be6ed4d3d5e0649ca76a0a270d758eaa63f6
- **Observation:** cc659f942a090ca3046701bc26734487d0de84af65e670e49bc5abe48aef151c
- **Data snapshot:** d3f991cc994df9d7757ec98159b5f3dbf5ed8e580cc3eaba9c79184522cf72f2
- **Score reuse:** none
- **Target weights:** none

