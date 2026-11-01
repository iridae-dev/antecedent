# Causal analysis report

## Summary

- **Reading:** a contract before execution
- **Query:** ChangeAttribution
- **Inference:** frequentist
- **Identifier:** none
- **Estimator:** none
- **Support status:** refused

- No claim is attached: this report reads the contract before any execution, and no estimate is stated.
- The cell is in the product but is not licensed, and the matrix refuses it.

## Identification

- **Graph class:** Admg
- **Structure source:** explicit
- **Identifier:** none
- **Estimator:** none
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
- **Query kind:** ChangeAttribution
- **Rows:** 200
- **Functional:** none

### Support slot

- **Matrix status:** refused
- **Matrix coordinate:** ChangeAttribution:Admg:explicit:Frequentist:full
- **Empirical support:** unavailable:not\_evaluated

## Refusals and limits

- **Operation:** inspect
- **Applicability:** unlicensed
- **Readiness:** none
- **Coordinate:** ChangeAttribution:Admg:explicit:Frequentist:full

No limit is recorded beyond the blockers below.

### Blockers and refusals

| Id | Scientific impossibility | Reason |
| --- | --- | --- |
| support.refused | yes | Cheap and full validation of a change attribution on a supplied or accepted Cpdag, Pag, Admg, TemporalAdmg, TemporalCpdag or TemporalPag is refused. The validation would refit the vector of total change and component contributions on random row subsets and on bootstrap resamples drawn within each of the two populations, and report one joint chi-square statistic against the original. The total change is a difference of two population means and the allocation a Shapley function of mechanism swaps, so a within-population subset or resample leaves the replicate mean at the original whatever heterogeneity is planted: the same derivation holds for every member of a class and every atom of a posterior, which are each attributed by the DAG or noise-block route whose replicates are such subsets and resamples. A verdict that cannot tell a sound attribution from an unsound one is not evidence, so the cell refuses instead of reporting one. Next attempt: a planted-violation design for the allocation (a mechanism that differs between the halves of one population), checked against an exact reference. |

### Next actions

Next actions are suggestions for the caller; none is taken automatically.

| Action | Description |
| --- | --- |
| consider\_licensed\_neighbor | choose an explicit licensed neighbor; never an automatic fallback |

### Licensed neighbours

A neighbour changes the stated premises; it is not a fallback.

| Coordinate | Changed premises | Required action |
| --- | --- | --- |
| AnomalyAttribution:Admg:explicit:Frequentist:full | query: ChangeAttribution to AnomalyAttribution | change the query; do not relabel the graph class |
| AverageDerivative:Admg:explicit:Frequentist:full | query: ChangeAttribution to AverageDerivative | change the query; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:full | query: ChangeAttribution to AverageEffect | change the query; do not relabel the graph class |
| ChangeAttribution:Admg:explicit:Frequentist:none | validation: full to none | change the validation suite |
| ConditionalEffect:Admg:explicit:Frequentist:full | query: ChangeAttribution to ConditionalEffect | change the query; do not relabel the graph class |
| Counterfactual:Admg:explicit:Frequentist:full | query: ChangeAttribution to Counterfactual | change the query; do not relabel the graph class |
| DirectionalDerivative:Admg:explicit:Frequentist:full | query: ChangeAttribution to DirectionalDerivative | change the query; do not relabel the graph class |
| Elasticity:Admg:explicit:Frequentist:full | query: ChangeAttribution to Elasticity | change the query; do not relabel the graph class |
| InterferenceQuery:Admg:explicit:Frequentist:full | query: ChangeAttribution to InterferenceQuery | change the query; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:full | query: ChangeAttribution to InterventionResponse | change the query; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Frequentist:full | query: ChangeAttribution to InterventionalDistribution | change the query; do not relabel the graph class |
| MediationEffect:Admg:explicit:Frequentist:full | query: ChangeAttribution to MediationEffect | change the query; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:explicit:Frequentist:full | query: ChangeAttribution to NestedCounterfactualEffect | change the query; do not relabel the graph class |
| PathSpecificEffect:Admg:explicit:Frequentist:full | query: ChangeAttribution to PathSpecificEffect | change the query; do not relabel the graph class |
| PointDerivative:Admg:explicit:Frequentist:full | query: ChangeAttribution to PointDerivative | change the query; do not relabel the graph class |
| ResponseJacobian:Admg:explicit:Frequentist:full | query: ChangeAttribution to ResponseJacobian | change the query; do not relabel the graph class |
| SemiElasticity:Admg:explicit:Frequentist:full | query: ChangeAttribution to SemiElasticity | change the query; do not relabel the graph class |
| TransportQuery:Admg:explicit:Frequentist:full | query: ChangeAttribution to TransportQuery | change the query; do not relabel the graph class |
| AnomalyAttribution:Admg:accepted:Frequentist:full | query: ChangeAttribution to AnomalyAttribution; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:explicit:Bayesian:full | query: ChangeAttribution to AnomalyAttribution; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to AnomalyAttribution; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:explicit:Frequentist:none | query: ChangeAttribution to AnomalyAttribution; validation: full to none | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to AnomalyAttribution; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:accepted:Frequentist:full | query: ChangeAttribution to AverageDerivative; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:explicit:Bayesian:full | query: ChangeAttribution to AverageDerivative; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to AverageDerivative; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:explicit:Frequentist:none | query: ChangeAttribution to AverageDerivative; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to AverageDerivative; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:full | query: ChangeAttribution to AverageEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:full | query: ChangeAttribution to AverageEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to AverageEffect; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Frequentist:none | query: ChangeAttribution to AverageEffect; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to AverageEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| ChangeAttribution:Admg:accepted:Frequentist:none | structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| ChangeAttribution:Admg:explicit:Bayesian:none | inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ChangeAttribution:Admg:graph\_posterior:Frequentist:none | structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Frequentist:full | query: ChangeAttribution to ConditionalEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Bayesian:full | query: ChangeAttribution to ConditionalEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to ConditionalEffect; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Frequentist:none | query: ChangeAttribution to ConditionalEffect; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to ConditionalEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:accepted:Frequentist:full | query: ChangeAttribution to Counterfactual; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:explicit:Bayesian:full | query: ChangeAttribution to Counterfactual; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to Counterfactual; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:explicit:Frequentist:none | query: ChangeAttribution to Counterfactual; validation: full to none | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to Counterfactual; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:accepted:Frequentist:full | query: ChangeAttribution to DirectionalDerivative; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:explicit:Bayesian:full | query: ChangeAttribution to DirectionalDerivative; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to DirectionalDerivative; validation: full to cheap | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:explicit:Frequentist:none | query: ChangeAttribution to DirectionalDerivative; validation: full to none | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to DirectionalDerivative; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:accepted:Frequentist:full | query: ChangeAttribution to Elasticity; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:explicit:Bayesian:full | query: ChangeAttribution to Elasticity; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to Elasticity; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:explicit:Frequentist:none | query: ChangeAttribution to Elasticity; validation: full to none | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to Elasticity; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:accepted:Frequentist:full | query: ChangeAttribution to InterferenceQuery; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:explicit:Bayesian:full | query: ChangeAttribution to InterferenceQuery; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to InterferenceQuery; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:explicit:Frequentist:none | query: ChangeAttribution to InterferenceQuery; validation: full to none | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to InterferenceQuery; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Frequentist:full | query: ChangeAttribution to InterventionResponse; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Bayesian:full | query: ChangeAttribution to InterventionResponse; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to InterventionResponse; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Frequentist:none | query: ChangeAttribution to InterventionResponse; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to InterventionResponse; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Frequentist:full | query: ChangeAttribution to InterventionalDistribution; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Bayesian:full | query: ChangeAttribution to InterventionalDistribution; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to InterventionalDistribution; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Frequentist:none | query: ChangeAttribution to InterventionalDistribution; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to InterventionalDistribution; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:accepted:Frequentist:full | query: ChangeAttribution to MediationEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:explicit:Bayesian:full | query: ChangeAttribution to MediationEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to MediationEffect; validation: full to cheap | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:explicit:Frequentist:none | query: ChangeAttribution to MediationEffect; validation: full to none | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to MediationEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:accepted:Frequentist:full | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:explicit:Bayesian:full | query: ChangeAttribution to NestedCounterfactualEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to NestedCounterfactualEffect; validation: full to cheap | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:explicit:Frequentist:none | query: ChangeAttribution to NestedCounterfactualEffect; validation: full to none | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:accepted:Frequentist:full | query: ChangeAttribution to PathSpecificEffect; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:explicit:Bayesian:full | query: ChangeAttribution to PathSpecificEffect; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to PathSpecificEffect; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:explicit:Frequentist:none | query: ChangeAttribution to PathSpecificEffect; validation: full to none | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to PathSpecificEffect; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:accepted:Frequentist:full | query: ChangeAttribution to PointDerivative; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:explicit:Bayesian:full | query: ChangeAttribution to PointDerivative; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to PointDerivative; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:explicit:Frequentist:none | query: ChangeAttribution to PointDerivative; validation: full to none | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to PointDerivative; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:explicit:Frequentist:none | query: ChangeAttribution to ResponseCurve; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to ResponseCurve; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:accepted:Frequentist:full | query: ChangeAttribution to ResponseJacobian; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:explicit:Bayesian:full | query: ChangeAttribution to ResponseJacobian; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to ResponseJacobian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:explicit:Frequentist:none | query: ChangeAttribution to ResponseJacobian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to ResponseJacobian; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:accepted:Frequentist:full | query: ChangeAttribution to SemiElasticity; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:explicit:Bayesian:full | query: ChangeAttribution to SemiElasticity; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to SemiElasticity; validation: full to cheap | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:explicit:Frequentist:none | query: ChangeAttribution to SemiElasticity; validation: full to none | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to SemiElasticity; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:accepted:Frequentist:full | query: ChangeAttribution to TransportQuery; structure: explicit to accepted | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Bayesian:full | query: ChangeAttribution to TransportQuery; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Frequentist:cheap | query: ChangeAttribution to TransportQuery; validation: full to cheap | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Frequentist:none | query: ChangeAttribution to TransportQuery; validation: full to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:graph\_posterior:Frequentist:full | query: ChangeAttribution to TransportQuery; structure: explicit to graph\_posterior | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:accepted:Bayesian:full | query: ChangeAttribution to AnomalyAttribution; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to AnomalyAttribution; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:accepted:Frequentist:none | query: ChangeAttribution to AnomalyAttribution; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to AnomalyAttribution; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:explicit:Bayesian:none | query: ChangeAttribution to AnomalyAttribution; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to AnomalyAttribution; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to AnomalyAttribution; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to AnomalyAttribution; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:accepted:Bayesian:full | query: ChangeAttribution to AverageDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to AverageDerivative; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:accepted:Frequentist:none | query: ChangeAttribution to AverageDerivative; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to AverageDerivative; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:explicit:Bayesian:none | query: ChangeAttribution to AverageDerivative; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to AverageDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to AverageDerivative; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to AverageDerivative; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Bayesian:full | query: ChangeAttribution to AverageEffect; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to AverageEffect; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Frequentist:none | query: ChangeAttribution to AverageEffect; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to AverageEffect; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:explicit:Bayesian:none | query: ChangeAttribution to AverageEffect; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to AverageEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to AverageEffect; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to AverageEffect; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| ChangeAttribution:Admg:accepted:Bayesian:none | structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ChangeAttribution:Admg:graph\_posterior:Bayesian:none | structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Bayesian:full | query: ChangeAttribution to ConditionalEffect; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to ConditionalEffect; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Frequentist:none | query: ChangeAttribution to ConditionalEffect; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to ConditionalEffect; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:explicit:Bayesian:none | query: ChangeAttribution to ConditionalEffect; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to ConditionalEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to ConditionalEffect; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to ConditionalEffect; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:accepted:Bayesian:full | query: ChangeAttribution to Counterfactual; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to Counterfactual; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:accepted:Frequentist:none | query: ChangeAttribution to Counterfactual; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to Counterfactual; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:explicit:Bayesian:none | query: ChangeAttribution to Counterfactual; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to Counterfactual; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to Counterfactual; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to Counterfactual; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:accepted:Bayesian:full | query: ChangeAttribution to DirectionalDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to DirectionalDerivative; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:accepted:Frequentist:none | query: ChangeAttribution to DirectionalDerivative; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to DirectionalDerivative; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:explicit:Bayesian:none | query: ChangeAttribution to DirectionalDerivative; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to DirectionalDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to DirectionalDerivative; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to DirectionalDerivative; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:accepted:Bayesian:full | query: ChangeAttribution to Elasticity; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to Elasticity; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:accepted:Frequentist:none | query: ChangeAttribution to Elasticity; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to Elasticity; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:explicit:Bayesian:none | query: ChangeAttribution to Elasticity; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to Elasticity; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to Elasticity; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to Elasticity; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:accepted:Bayesian:full | query: ChangeAttribution to InterferenceQuery; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to InterferenceQuery; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:accepted:Frequentist:none | query: ChangeAttribution to InterferenceQuery; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to InterferenceQuery; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:explicit:Bayesian:none | query: ChangeAttribution to InterferenceQuery; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to InterferenceQuery; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to InterferenceQuery; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to InterferenceQuery; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Bayesian:full | query: ChangeAttribution to InterventionResponse; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to InterventionResponse; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Frequentist:none | query: ChangeAttribution to InterventionResponse; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to InterventionResponse; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:explicit:Bayesian:none | query: ChangeAttribution to InterventionResponse; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to InterventionResponse; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to InterventionResponse; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to InterventionResponse; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Bayesian:full | query: ChangeAttribution to InterventionalDistribution; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to InterventionalDistribution; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Frequentist:none | query: ChangeAttribution to InterventionalDistribution; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to InterventionalDistribution; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:explicit:Bayesian:none | query: ChangeAttribution to InterventionalDistribution; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to InterventionalDistribution; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to InterventionalDistribution; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to InterventionalDistribution; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:accepted:Bayesian:full | query: ChangeAttribution to MediationEffect; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to MediationEffect; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:accepted:Frequentist:none | query: ChangeAttribution to MediationEffect; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to MediationEffect; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:explicit:Bayesian:none | query: ChangeAttribution to MediationEffect; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to MediationEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to MediationEffect; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to MediationEffect; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:accepted:Bayesian:full | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:accepted:Frequentist:none | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to NestedCounterfactualEffect; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:explicit:Bayesian:none | query: ChangeAttribution to NestedCounterfactualEffect; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:accepted:Bayesian:full | query: ChangeAttribution to PathSpecificEffect; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to PathSpecificEffect; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:accepted:Frequentist:none | query: ChangeAttribution to PathSpecificEffect; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to PathSpecificEffect; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:explicit:Bayesian:none | query: ChangeAttribution to PathSpecificEffect; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to PathSpecificEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to PathSpecificEffect; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to PathSpecificEffect; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:accepted:Bayesian:full | query: ChangeAttribution to PointDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to PointDerivative; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:accepted:Frequentist:none | query: ChangeAttribution to PointDerivative; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to PointDerivative; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:explicit:Bayesian:none | query: ChangeAttribution to PointDerivative; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to PointDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to PointDerivative; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to PointDerivative; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:accepted:Frequentist:none | query: ChangeAttribution to ResponseCurve; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:explicit:Bayesian:none | query: ChangeAttribution to ResponseCurve; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to ResponseCurve; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to ResponseCurve; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to ResponseCurve; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:accepted:Bayesian:full | query: ChangeAttribution to ResponseJacobian; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to ResponseJacobian; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:accepted:Frequentist:none | query: ChangeAttribution to ResponseJacobian; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to ResponseJacobian; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:explicit:Bayesian:none | query: ChangeAttribution to ResponseJacobian; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to ResponseJacobian; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to ResponseJacobian; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to ResponseJacobian; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:accepted:Bayesian:full | query: ChangeAttribution to SemiElasticity; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to SemiElasticity; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:accepted:Frequentist:none | query: ChangeAttribution to SemiElasticity; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to SemiElasticity; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:explicit:Bayesian:none | query: ChangeAttribution to SemiElasticity; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to SemiElasticity; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to SemiElasticity; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to SemiElasticity; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:accepted:Bayesian:full | query: ChangeAttribution to TransportQuery; structure: explicit to accepted; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:accepted:Frequentist:cheap | query: ChangeAttribution to TransportQuery; structure: explicit to accepted; validation: full to cheap | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:accepted:Frequentist:none | query: ChangeAttribution to TransportQuery; structure: explicit to accepted; validation: full to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Bayesian:cheap | query: ChangeAttribution to TransportQuery; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:explicit:Bayesian:none | query: ChangeAttribution to TransportQuery; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:graph\_posterior:Bayesian:full | query: ChangeAttribution to TransportQuery; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:graph\_posterior:Frequentist:cheap | query: ChangeAttribution to TransportQuery; structure: explicit to graph\_posterior; validation: full to cheap | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:graph\_posterior:Frequentist:none | query: ChangeAttribution to TransportQuery; structure: explicit to graph\_posterior; validation: full to none | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to AnomalyAttribution; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:accepted:Bayesian:none | query: ChangeAttribution to AnomalyAttribution; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to AnomalyAttribution; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AnomalyAttribution:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to AnomalyAttribution; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to AverageDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:accepted:Bayesian:none | query: ChangeAttribution to AverageDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to AverageDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageDerivative:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to AverageDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to AverageEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:accepted:Bayesian:none | query: ChangeAttribution to AverageEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to AverageEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| AverageEffect:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to AverageEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to ConditionalEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:accepted:Bayesian:none | query: ChangeAttribution to ConditionalEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to ConditionalEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ConditionalEffect:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to ConditionalEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to Counterfactual; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:accepted:Bayesian:none | query: ChangeAttribution to Counterfactual; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to Counterfactual; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Counterfactual:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to Counterfactual; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to DirectionalDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:accepted:Bayesian:none | query: ChangeAttribution to DirectionalDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to DirectionalDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| DirectionalDerivative:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to DirectionalDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to Elasticity; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:accepted:Bayesian:none | query: ChangeAttribution to Elasticity; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to Elasticity; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| Elasticity:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to Elasticity; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to InterferenceQuery; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:accepted:Bayesian:none | query: ChangeAttribution to InterferenceQuery; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to InterferenceQuery; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterferenceQuery:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to InterferenceQuery; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to InterventionResponse; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:accepted:Bayesian:none | query: ChangeAttribution to InterventionResponse; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to InterventionResponse; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionResponse:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to InterventionResponse; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to InterventionalDistribution; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:accepted:Bayesian:none | query: ChangeAttribution to InterventionalDistribution; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to InterventionalDistribution; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| InterventionalDistribution:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to InterventionalDistribution; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to MediationEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:accepted:Bayesian:none | query: ChangeAttribution to MediationEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to MediationEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| MediationEffect:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to MediationEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:accepted:Bayesian:none | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| NestedCounterfactualEffect:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to NestedCounterfactualEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to PathSpecificEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:accepted:Bayesian:none | query: ChangeAttribution to PathSpecificEffect; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to PathSpecificEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PathSpecificEffect:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to PathSpecificEffect; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to PointDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:accepted:Bayesian:none | query: ChangeAttribution to PointDerivative; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to PointDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| PointDerivative:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to PointDerivative; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:accepted:Bayesian:none | query: ChangeAttribution to ResponseCurve; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to ResponseCurve; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseCurve:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to ResponseCurve; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to ResponseJacobian; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:accepted:Bayesian:none | query: ChangeAttribution to ResponseJacobian; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to ResponseJacobian; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| ResponseJacobian:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to ResponseJacobian; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to SemiElasticity; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:accepted:Bayesian:none | query: ChangeAttribution to SemiElasticity; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to SemiElasticity; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| SemiElasticity:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to SemiElasticity; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:accepted:Bayesian:cheap | query: ChangeAttribution to TransportQuery; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:accepted:Bayesian:none | query: ChangeAttribution to TransportQuery; structure: explicit to accepted; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:graph\_posterior:Bayesian:cheap | query: ChangeAttribution to TransportQuery; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to cheap | change the listed premises; do not relabel the graph class |
| TransportQuery:Admg:graph\_posterior:Bayesian:none | query: ChangeAttribution to TransportQuery; structure: explicit to graph\_posterior; inference: Frequentist to Bayesian; validation: full to none | change the listed premises; do not relabel the graph class |

- **Layers the premises preserve:** target
- **Layers the premises invalidate:** support, identification, results

This report restates the contract. It computes no estimate, grants no licence, and a status recorded here is the status the records carry.

## Identities

- **Target:** 24379cbc1d1d911c2949f6b33fb68cd4fcd85925bf14fbc69f511533afc911a0
- **Identification:** 59a49c87168c19e4a1acfbbf01a7d545a4e23e8ccdf9c02ae69fbd078fa62f16
- **Identification product:** none
- **Program:** none
- **Inference binding:** 4526df499d7801943e62b38d43fcdb8c76c39cab692f1d36fbb3fdbbdb59dbc2
- **Observation:** cc659f942a090ca3046701bc26734487d0de84af65e670e49bc5abe48aef151c
- **Data snapshot:** d3f991cc994df9d7757ec98159b5f3dbf5ed8e580cc3eaba9c79184522cf72f2
- **Score reuse:** none
- **Target weights:** none

