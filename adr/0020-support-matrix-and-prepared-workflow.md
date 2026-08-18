# ADR 0020 — support matrix and prepared-workflow contract

- Status: Accepted
- Date: 2026-08-18

## Context

0.5 shipped continuous responses and a staged *spell* (`identify` →
`.estimate`). Inventories (`parity/*.toml`, `docs/capabilities.md`) list
what is implemented. They do not license a cell: query × graph class ×
structure source × inference × validation. `identify_only` is DAG/ADMG
(and bidirected ADMG is AverageEffect-only); graph-posterior analysis
refuses `identify_only`. Response often works only as an `analyze`
sidecar. Missing combinations fail as free-form `Unsupported { message }`
or, worse, return a number.

0.6 is composition and evidence, not new estimands. The claim surface has
to become machine-checkable the way 0.5.1 made parity `evidence_kind`
machine-checkable.

## Decision

### Support matrix

The public license is `parity/support_axes.toml` +
`parity/support_n_a.toml` + `parity/support_licensed.toml`, gated by
`scripts/gate_support_matrix.sh`. Docs are generated from those files.

A cell is exactly one of:

- **`licensed`** — listed in `support_licensed.toml`; `staged = true`;
  `evidence_kind` from the 0.5.1 vocabulary; a fixture or named harness
  that an executing test loads.
- **`n/a`** — a predicate in `support_n_a.toml` (typed impossibility).
- **`refused`** — the default. No `pending` status. Unspecified is a
  gate failure.

Dispatch consults this matrix. `licensed` runs the staged path. `n/a`
and cells listed in `parity/support_closed.toml` raise a **stable error
id**. Remaining default-`refused` cells still run until licensed or
closed. `analyze` / `Study::run` is sugar over identify → prepare →
estimate → optional refute. A combination that only works inside
`analyze` cannot be licensed.

Graph posterior is a `structure` value, not a `GraphClass`. Completing
general sID is out of scope; the implemented subset plus `NotCertified`
is the transport contract.

### Prepared workflow

**Frozen at prepare:** schema (names, types, order); graph /
`AcceptedGraph` or a licensed graph-posterior object; query identity;
identifier; observation / transport / interference assumptions; target
population bindings.

**Estimate click:** same-schema data; estimator numeric knobs, latency,
seeds, bootstrap; `ExecutionContext` budget / cancellation. Must not
re-identify or recompile the logical plan.

**Refute click:** same frozen identification and estimand; schema-gated
data and suite config. `PreparedStudy::refute` stays AverageEffect until
a response-validation cell is licensed.

**Re-prepare required:** any frozen field changes, including schema
mismatch. Changing a frozen field on refresh is an error, not a silent
recompile.

The user-facing handle is `PreparedStudy` / `PreparedAnalysis`. "Ready
plan" is not a public name.

### Graph-posterior × response

Until a scoped mixture that retains unidentified mass as a first-class
axis can be gated, every response-family query × `structure =
graph_posterior` is `n/a` (`SupportRefusal::NotApplicable`, contrast-only).
ATE × graph posterior stays as today (per-graph identify inside
`execute()`, no `identify_only`). Priors do not upgrade identification.

## Consequences

Capabilities and comparison pages are inventories, not licenses. 0.6.0
ships the contract, the gate, and generated docs with an empty licensed
set; cells are promoted only once they sit on the staged handle.
`docs/support-matrix.md` is generated and must stay clean in CI.
