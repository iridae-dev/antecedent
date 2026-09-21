# The causal contract

A causal contract is the inspectable result of compiling a question, causal
structure, evidence, and assumptions. It says what the question means, what
is identified, what must be supplied at execution time, and which obligations
or limits travel with the result.

The contract separates scientific semantics from statistical realization and
physical execution. An estimator may fit a nuisance function; it does not get
to choose confounders or declare an effect identified. A compute budget can
stop a search; it does not prove that no solution exists.

## Structure is evidence, not truth

Supplied, discovered, accepted, partial, and posterior graph structures have
different status. DAGs, ADMGs, CPDAGs, PAGs, and temporal forms stay distinct.
When structural uncertainty remains, a point estimate is not substituted for a
bound or partial answer merely to simplify a downstream interface.

## Identity makes changes explicit

The same scalar may arise from different scientific analyses. Antecedent tracks
the target, identification premises and product, program, inference binding,
observation contract, data snapshot, and execution separately. Refreshing with
compatible data, retargeting a population, changing causal structure, and
re-estimating are therefore different operations with different consequences.

This is a form of epistemic type safety: it prevents `estimated = identified`,
`not found = impossible`, or `same number = same analysis` from being silently
accepted.

See [architecture](architecture.md) for the system layers and
[artifacts](artifacts.md) for what survives a process boundary.
