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

## 2.3 features

The 2.3 Python surface is documented by task in the 2.3 guide:

- [The 2.3 lifecycle](2_3-lifecycle.md): analyze, bind an external result, decide with `Contract.evaluate(result)`, rank a study, bundle and export, with the structured-refusal pattern.
- [Population, time and uncertainty](2_3-population-time-uncertainty.md): exact measured adapter scopes and `inference.MeasuredInference.load` with retained source identity; every scalar needs its own current record.
- [Design ranking](2_3-design-ranking.md): `design.rank_designs` with an explicit `basis` (`identification`, `net_value`, `evsi`), `DesignDecision.from_contract`, `StatePrior.from_distribution` and `design.consume`.
- [Sensitivity and robustness](2_3-sensitivity-and-robustness.md): `msm_sensitivity.msm_ate_sensitivity`, `sensitivity_decision`, `decision_robust` and `mechanism_discrepancy.diagnose_mechanism_discrepancy`.
- [Transport and counterfactuals](2_3-transport-counterfactuals.md): the affine-additive transported path-specific effect, `MechanismSelection`, the scenario invariance report and the source-bound learned joint measured adapter.
- [Prepared recalculation](2_3-recalculation-capabilities.md): selective recalculation, frozen scores and portable resume.
- [Refusals](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies): the shared `StructuredRefusal` base with `.code`, `.detail`, `.offending` and `.remedy`.
- [Composition and bundles](2_3-composition.md): `composition_bundle.Bundle`, `add_artifact` and `consume_bundle`.

Every stage module is also reachable lazily from the root, for example `antecedent.design`, without a separate import.

For Rust, use `cargo doc -p antecedent --open` for this checkout, or visit the [published crate reference](https://docs.rs/antecedent).
