# Python API

New to Antecedent? Start with the [Python quickstart](python-workflow.md). For configuration details, use the [workflow reference](python-options.md).

<!-- pdoc adds this sibling directory after MkDocs builds the narrative. -->
<strong><a href="../python/antecedent.html">Browse the Python API</a></strong>

Read the Docs builds this reference from the matching Antecedent release (and
falls back to the checked-out source only before that release reaches PyPI).
The link stays within the documentation version you are reading.

To generate a local reference for the installed package, run:

```bash
python -m pip install antecedent pdoc
python -m pdoc antecedent -o site/python
```

## Task guides

The Python surface is documented by task in these guides:

- [Notebook to system in 2.4](guides/notebook-to-system.md): retained identification, guarded design updates, explicit repreparation, checked mean composition and independent artifact consumption.
- [Python 2.4 migration](migrations/2.4-python-ux.md): immutable scientific readings, named method outputs, session ownership and bytes-first persistence.
- [The lifecycle](lifecycle.md): analyze, bind an external result, decide with `Contract.evaluate(result)`, rank a study, bundle and export, with the structured-refusal pattern.
- [Population, time and uncertainty](population-time-uncertainty.md): exact measured adapter scopes and `inference.MeasuredInference.load` with retained source identity; every scalar needs its own current record.
- [Design ranking](design-ranking.md): the shared `design.rank_designs` entry point for identification, structural sufficiency, information objectives and decision value; `DesignDecision.from_contract`, `design.adapt_prior_to_signal` and `design.consume`.
- [Sensitivity and robustness](sensitivity-and-robustness.md): `msm_sensitivity.msm_ate_sensitivity`, `sensitivity_decision`, `decision_robust` and `mechanism_discrepancy.diagnose_mechanism_discrepancy`.
- [Transport and counterfactuals](transport-counterfactuals.md): the affine-additive transported path-specific effect, `MechanismSelection`, the scenario invariance report and the source-bound learned joint measured adapter.
- [Prepared recalculation](recalculation-capabilities.md): checked family sessions, selective recalculation, bounded external mean callbacks, frozen scores and portable resume.
- [Transported decisions and measurement](guides/transported-decisions-and-measurement.md) and [inverse queries and decisions](guides/inverse-and-decision-reach.md): `transported_response` (`TransportedResponse`, `TransportDeclaration`, `evaluate_transported`) and `inverse_sources` (`composed_forward_source`, `cohort_forward_source`, `transported_forward_source`, `observational_scenarios`).
- [Assumption range and revalidation](guides/assumption-range-and-revalidation.md): `transport.sensitivity.transport_sensitivity`, `transport.revalidation` (`TransportScene`, `revalidate`, `TransportClaimState`).
- [Provider-sourced transport](guides/provider-sourced-transport.md): `transport_handoff.hand_to_callback` and `consume_for_callback`.
- [Finite-SCM discretization](guides/finite-scm-discretization.md): `discretized_composition.discretize`, `PreparedMechanismComposition.discretize`, `consume_discretized_composition`.
- [Composed uncertainty](guides/composed-uncertainty.md): `CrossWorldExecution.whole_path_uncertainty`.
- [What the transport and uncertainty routes do not do](user-facing-limits.md).
- [Refusals](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies): the shared `StructuredRefusal` base with `.code`, `.detail`, `.offending` and `.remedy`.
- [Composition and bundles](composition.md): `composition_bundle.Bundle`, `add_artifact` and `consume_bundle`.

Every stage module is also reachable lazily from the root, for example `antecedent.design`, without a separate import.

For Rust, use `cargo doc -p antecedent --open` for this checkout, or visit the [published crate reference](https://docs.rs/antecedent).

## Python 2.4 continuity

See the [notebook-to-system journey](guides/notebook-to-system.md) and [migration guide](migrations/2.4-python-ux.md) for retained identification, immutable results, typed adaptive records, checked mean adapters and shared bytes-first persistence.
