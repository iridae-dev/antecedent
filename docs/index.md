# Antecedent

Antecedent is a causal inference system for turning causal questions and
evidence into checked, executable scientific claims. It preserves what an
answer means—its assumptions, identification status, empirical support,
uncertainty, provenance, and limits—when the analysis is estimated, reused,
combined, saved, transported, or consumed by other software.

> Install with `python -m pip install --upgrade antecedent` (Python 3.11+),
> then begin with the [Python quickstart](python-workflow.md). The
> [release notes](release-notes/v2.4.0.md) describe what changed in each
> version, with earlier releases listed in the navigation.

Choose [an example](examples.md), use [support lookup](support-lookup.md), or read [the glossary](glossary.md) when a standing label is unfamiliar. For a complete 2.4 workflow, follow [notebook to system](guides/notebook-to-system.md) and [measurement to decision](tutorials/measurement-to-decision.md).

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
populations, and design-family studies — randomized and factorial
experiments, held-out policy value, quasi-experimental designs, survival, and
longitudinal regimes. Those are different causal programs, not disconnected
products.

## Decisions, design, and composition

These pages connect an analysis to a decision. Each page states its claim label and
what it refuses.

| Question | Read |
| --- | --- |
| What is the path from analysis to a decision to a study plan? | [The analysis lifecycle](lifecycle.md) |
| Which study should I run next, and how do I replay the ranking? | [Design ranking](design-ranking.md) |
| How far could the answer move under an unmeasured assumption? | [Sensitivity and robustness](sensitivity-and-robustness.md) |
| Can a counterfactual effect be carried to another population? | [Transport and counterfactuals](transport-counterfactuals.md) |
| How do I recompute a decision without refitting? | [Prepared recalculation](recalculation-capabilities.md) |
| How do I bind foreign numbers, decide on them and export the whole composition? | [External science](external-science.md), [decisions](decisions.md), [composition](composition.md) |
| How do refusals expose their cause and remedy? | [Structured refusals](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies) |

For exact public boundaries, consult the [support matrix](support-matrix.md).
An implemented capability is not automatically a licensed analysis, and a
licensed analysis does not establish that a real-world causal model is true.

Read `result.calibration`: status `calibrated` requires a coverage record that matches the execution and attests the executing code. See the [measured-inference guide](population-time-uncertainty.md).
