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

## Amendment (2026-08-19): the allowlist, and enforcing default-refused

The original decision left a fifth, unnamed bucket: cells that were
`refused` (not licensed, not n/a, not in `support_closed.toml`) still ran
by default, on the theory that "closed" would catch up to them over time.
It never fully did — auditing that remainder found 167 such cells, and
running each one end-to-end (`Study::build().run()`) split them three ways:
cells that genuinely run today on real data but have no cell-shaped
known-truth fixture to license (e.g. PAG ATE, which re-identifies per
click); cells that die at compile (an identifier/estimator pair the
strategy table rejects) or, worse, that *silently ignore* the claimed axis
value and return a differently-labeled number (`ConditionalEffect` and
`TemporalMediationEffect` under `inference = Bayesian` both hardcode their
Frequentist estimator and never consult inference mode — confirmed
bit-identical output across both labels on the same fixture); and cells
that are structurally unreachable through any public constructor (a `Cpdag`
has no explicit/`IntoGraphInput` path, so `structure = explicit` combined
with `graph_class ∈ {Cpdag, TemporalCpdag, TemporalPag}` can never be built
at all, and several `AverageEffect`/`PulseEffect` collapse or
graph-posterior-stub cells are likewise never actually classified as their
nominal graph class).

A cell is now exactly one of:

- **`licensed`** — unchanged from above.
- **`n/a`** — unchanged from above.
- **`closed`** — a predicate in `support_closed.toml`; fails closed with
  `SupportRefusal::Refused`. Now also covers cells that die at compile and
  cells that silently ignore the axis they claim (dishonest, not merely
  absent — "fail-shut only what cannot return an honest number").
- **`allowed`** — a predicate in the new `parity/support_allowlist.toml`;
  running, unlicensed, and named. Each row carries a `reason` (why it runs,
  and why it is not licensed instead — no staged handle, re-identifies per
  click, or no cell-shaped known-truth fixture) and a `parent` (the
  licensed or keep-running family it rides). An allowed rule must not match
  any licensed, n/a, or closed cell; `scripts/gate_support_matrix.sh`
  enforces that partition.

Any `refused` cell not matched by the allowlist now fails closed with a
single shared, static message
(`crates/antecedent/src/support.rs::UNLICENSED_AND_NOT_ALLOWED`) — the
unnamed fifth bucket no longer exists. Nothing that ran before this
amendment stopped running: every cell a caller could actually reach and
that returned an honest number is on the allowlist; the closed additions
target cells that either never returned a number or returned a
mislabeled one.
