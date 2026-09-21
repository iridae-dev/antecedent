# Antecedent

> **Release preparation:** these are the active 2.0 docs. The published package
> remains 1.11.0 until the release metadata and tag advance.

Antecedent is a causal inference system for turning causal questions and
evidence into checked, executable scientific claims. It preserves what an
answer means—its assumptions, identification status, empirical support,
uncertainty, provenance, and limits—when the analysis is estimated, reused,
combined, saved, transported, or consumed by other software.

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

The long-running 2.0 calibration measurement remains planned, not completed.
Read `result.calibration` and the [draft release notes](release-notes/v2.0.0.md)
for the scope of interval evidence.
