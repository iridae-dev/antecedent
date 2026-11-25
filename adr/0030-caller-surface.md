# ADR 0030: Caller surface for 3.1

- Status: Accepted
- Date: 2026-11-25
- Extends: [0020](0020-support-matrix-and-prepared-workflow.md), [0025](0025-provider-protocol.md), [0028](0028-one-compiled-path.md)
- Inventory: [3.1 baselines](../docs/architecture/3.1-baselines.md)

## Context

3.1 moves writers, learner declarations, resource limits and saved-study
resume. Those batches need one caller-facing shape before their signatures
settle. The names already live in `docs/api_naming.md`: prepare, estimate,
rebind, reconfigure, analyze, load, identify. Several public entries still
sit beside that shape and answer the same question with a different type.

The coverage audit records which of those entries still execute without a
compiled-path row. This ADR chooses the call a caller writes, where the old
entry stays, and the refusal that call returns.

## Decision

### One lifecycle

A caller prepares an immutable study, branches it, and executes each branch.

```python
study = ant.prepare(data, query=query, inference=ant.Frequentist(), execution=ant.Execution(seed=1))
branched = ant.rebind(study, data=new_data)
outcome = ant.estimate(branched)
```

`ant.analyze(...)` is that prepare followed by estimate. It stays. It is the
one-shot form of the same study, and it is the replacement named by the
learner entries below. Rust builds the same study with `StudyBuilder` and
`Study::prepare`, then `estimate`.

Seeds, threads, cancellation and memory belong on `Execution`. Statistical
budgets belong on `Frequentist` or `Bayesian`. A call that still passes those
as loose arguments fails at the call.

### Learners are a study setting

The declaration is `StudyBuilder::learner_factories` in Rust and the learner
argument on `analyze` / `prepare` in Python. `estimator` is the only selector:
`aipw`, `dml` (partially linear score), `dr_learner`, `causal_forest`.

`analyze_ate` keeps today's meaning until batch 3b moves it onto `analyze`:
default `estimator="dml"` with `score="aipw"`, and any score other than
`"aipw"` only with `estimator="dml"`. `dml_ate`, `aipw_ate`, `dr_learner_ate`,
`causal_forest_ate` and `cross_fit_aipw` are already deprecated for 4.0. After
3b each warning names `analyze`. The deprecated entries keep the old
estimator/score reading. The single selector applies to `analyze`.

A fold policy the estimator cannot honour is refused when the learner is
declared. The policy is the enum from batch 2d. The caller does not pass
`iid`, `temporal`, or a host/caller/origin string.

### A decision consumes a compiled result

Python evaluates a contract against aligned joint draws, or against a checked
static average effect or response curve when the utility is an affine mean:

```python
decision = contract.evaluate(joint_distribution)
```

Rust grows the same entry on the compiled result (batch 6b). It does not
re-estimate. A missing authority, a payload that is not a mean the contract
can read, or a failed constraint raises `DecisionRefusal`.

### Opening a saved study is two operations

`ant.load(artifact, expected=outcome.expectation)` returns the recorded
outcome. It recomputes only when the caller passes `replay=True`.
`decision.replay(bytes, contract=, source=)` recomputes a stored decision
under the contract.

Resume of a compiled study is the format 0.6 open in batch 4. It is a
separate call from `load`. Both refuse when the supplied expectation does not
match the artifact. The container-byte digest and the expected program
identity stay distinct checks (batch 2b).

### What the caller sees on refusal

A question the structure cannot answer is an `Outcome` whose refusal carries
the registered reason. `DecisionRefusal` and `ExternalRefusal` stay raised
errors, as `docs/api_naming.md` already specifies. A budget stop is
`SearchReceipt` / `SearchStop` in both languages. Its text names the bound in
force, how much was consumed, and what was not entered. It is never reported
as non-identification, and it never says "at most N" for a fixed problem size.

Memory is declared once, on the execution (`ResourceLimits.memory_bytes`, or
Rust `with_memory_limit` / `with_admission`). Operations and depth are
`SearchLimits`. A route does not grow a second public cap.

### Where a compatibility wrapper lives

A wrapper that still computes a scientific result is `#[deprecated(since = "3.1.0")]`
on the facade item, or a Python function that calls
`_deprecation.direct_path_deprecated` before the old body. Its compiled-path
row names `route:<ProgramCell>`. Kernel methods that a compiled route executes
through stay kernel rows, with a reason, and are not deprecated. Wrappers are
removed in 4.0.

The two public constructors that the compiled-path gate does not see,
`transported_synthesis_response` and `surrogate_bridge_response`, and
`CrossWorldExecutor.execute`, follow this rule once batch 3a gives them a
compiled cell. Until then they stay the entries they are. New work does not
add a third spelling.

## Consequences

Batches 2–6 take the calls in this ADR as the signatures they implement.
Batch 7 walks them in the public examples. A competing name found later is a
wrapper toward one of these calls, or a revision of this ADR.
