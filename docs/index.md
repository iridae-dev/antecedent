# Antecedent

Antecedent is a causal inference system for turning causal questions and
evidence into checked, executable scientific claims. It preserves what an
answer means—its assumptions, identification status, empirical support,
uncertainty, provenance, and limits—when the analysis is estimated, reused,
combined, saved, transported, or consumed by other software.

> **Antecedent 2.1.0.** Install it with
> `python -m pip install --upgrade antecedent` (Python 3.11+), then begin with
> the [Python quickstart](python-workflow.md). Read the
> [2.1.0 release notes](release-notes/v2.1.0.md) when upgrading from an earlier
> release.

Most causal failures in software are semantic failures at boundaries. Read
[the system model](system-model.md) first: it explains how Antecedent compiles
declared knowledge into a causal contract and why a result is more than a
number.

## Start with the mental model

| Question | Read |
| --- | --- |
| What is Antecedent trying to preserve? | [System model](system-model.md) |
| What does an analysis contract contain? | [The causal contract](causal-contract.md) |
| How should I read a result? | [A result is a claim](result-is-a-claim.md) |
| Why does Antecedent refuse some requests? | [Refusal and partial knowledge](refusal-and-partial-knowledge.md) |
| How do I read graph uncertainty and unidentified mass? | [Graph uncertainty](graph_uncertainty.md) |
| What does “supported” mean? | [Guarantees and support](guarantees.md) |

## Then use it

Start an ordinary analysis with the [Python workflow](python-workflow.md) or
[Rust quickstart](rust-quickstart.md). The same model extends to discovery and
structural uncertainty, response and temporal questions, Bayesian inference,
validation, counterfactuals, learner-backed estimation, transport across
populations, and the 2.1 design-family studies — randomized and factorial
experiments, held-out policy value, quasi-experimental designs, survival, and
longitudinal regimes. Those are different causal programs, not disconnected
products.

## 2.3 features

The 2.3 line connects analysis to decisions. Each page states its claim label and
what it refuses.

| Question | Read |
| --- | --- |
| What is the path from analysis to a decision to a study plan? | [The 2.3 lifecycle](2_3-lifecycle.md) |
| Which study should I run next, and how do I replay the ranking? | [Design ranking](2_3-design-ranking.md) |
| How far could the answer move under an unmeasured assumption? | [Sensitivity and robustness](2_3-sensitivity-and-robustness.md) |
| Can a counterfactual effect be carried to another population? | [Transport and counterfactuals](2_3-transport-counterfactuals.md) |
| How do I recompute a decision without refitting? | [Prepared recalculation](2_3-recalculation-capabilities.md) |
| How do I bind foreign numbers, decide on them and export the whole composition? | [External science](2_3-external-science.md), [decisions](2_3-decisions.md), [composition](2_3-composition.md) |
| How do refusals expose their cause and remedy? | [Structured refusals](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies) |

For exact public boundaries, consult the [support matrix](support-matrix.md).
An implemented capability is not automatically a licensed analysis, and a
licensed analysis does not establish that a real-world causal model is true.

Read `result.calibration`: status `calibrated` requires a coverage record that matches the execution and attests the executing code. See the [2.1.0 release notes](release-notes/v2.1.0.md).
