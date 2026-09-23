# Antecedent

Antecedent is a causal inference system for turning causal questions and
evidence into checked, executable scientific claims. It preserves what an
answer means—its assumptions, identification status, empirical support,
uncertainty, provenance, and limits—when the analysis is estimated, reused,
combined, saved, transported, or consumed by other software.

> **Antecedent 2.0 is the stable release.** Install it with
> `python -m pip install --upgrade antecedent` (Python 3.11+), then begin with
> the [Python quickstart](python-workflow.md). Read the
> [2.0.0 release notes](release-notes/v2.0.0.md) when upgrading from an earlier
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
validation, counterfactuals, learner-backed estimation, and transport across
populations. Those are different causal programs, not disconnected products.

For exact public boundaries, consult the [support matrix](support-matrix.md).
An implemented capability is not automatically a licensed analysis, and a
licensed analysis does not establish that a real-world causal model is true.

Read `result.calibration`: status `calibrated` requires a coverage record that matches the execution and attests the executing code. See the [2.0 release notes](release-notes/v2.0.0.md).
