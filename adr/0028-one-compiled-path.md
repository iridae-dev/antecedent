# ADR 0028: One compiled path

- Status: Accepted
- Date: 2026-11-23
- Extends: [0022](0022-causal-compiler-contract.md), [0020](0020-support-matrix-and-prepared-workflow.md), [0027](0027-rust-verification-route.md)
- Amends: [0020](0020-support-matrix-and-prepared-workflow.md), [0022](0022-causal-compiler-contract.md), [0024](0024-transport-architecture.md)

## Context

The causal compiler (Study, compile, prepare, run, lowering through the route
registry) gives a question a typed refusal or a contract, a claim, and evidence
binding. Several capabilities reach a scientific result without it: direct
functions on the facade, re-exported kernel entry points, hidden engines, and
Python wrappers around them. A result from such a path has no refusal taxonomy,
no claim, and no link to the support model or to the calibration evidence that
licenses its interval.

Support status was recorded in more than one place: matrix cells, graphless
rows, per-capability constants, and method lists. Layers that each answer "is
this supported" can disagree, and a hand-written constant can drift from the
route it describes.

Gates that find bypasses by name miss a path that is renamed, re-exported, or
reached through a method of a re-exported type.

Transport (ADR 0024) was scoped as outside the compiler workflow. A transported
effect is a scientific result and has the same need for a contract and claim.

## Decision

### One compiled path

Every user-facing capability that produces a scientific result (an estimate,
interval, posterior, decision, ranking, forecast, test, discovered structure,
transported effect or attribution) runs through the causal compiler. It has
typed refusals, a contract and claim, and evidence binding. A capability that
computes a result outside the compiler is not a supported analysis route.

### One support model

A support unit is the only thing that has a support status. A unit is a matrix
cell, a graphless row, or a capability record that is not an alias.

- Every compiled operation resolves to exactly one unit at lowering.
- Compiled routes that are not matrix cells are bound to their unit in
  `parity/compiled_units.toml`, which generates
  `crates/antecedent/src/support/units_generated.rs`. No hand-written constant
  binds a route to a unit.
- Support decisions never enter program identity (`ScientificProblem::write`).
  Matrix programs keep their legacy supported bit so existing identities do not
  change.

### Kernel API and analysis surface

The kernel crates (`antecedent-estimate`, `antecedent-identify`,
`antecedent-validate`, `antecedent-discovery`, `antecedent-model`,
`antecedent-stats`, `antecedent-graph` and the other non-facade crates) are
kernel API: building blocks the compiler calls. They are not deprecated and are
not a supported analysis surface.

The facade (crate `antecedent`) and the Python package are the analysis surface.
A facade or Python path that reaches a scientific computation without the
compiler is deprecated in favour of its compiled replacement and removed in 4.0.
Rust wrappers carry `#[deprecated(since = "3.1.0")]`; Python paths emit
`AntecedentDeprecationWarning`. The facade re-exports a kernel run entry only as
such a deprecated wrapper. Hidden (`#[doc(hidden)]`) engines are deprecated
shims in 3.1 and move behind the `calibration-internal` feature in 4.0.

### Replacements reproduce the deprecated path

A compiled replacement calls the same kernel function as the path it replaces.
A twin-equality test asserts bit-identical point estimate, standard error and
interval on recorded fixtures. The measured coverage still attaches to the
replacement, through the same calibration key or an alias. Where a direct path
differs from the compiled one (for example the seed derivation for bootstrap
draws), the deprecation note states the difference.

### Bypass gate

The gate classifies by type and method, not by name. These must each be
classified in `parity/compiled_path.toml`:

- every public facade item whose return type is a scientific result type;
- every re-export of a kernel item;
- every method of a re-exported kernel type.

The classes are `compiled`, `deprecated` (with its replacement), `internal`,
`consume` (an artifact consumer that reads a result and computes none),
`not_scientific` (with a reason) and `pending`. Release closure requires no
`pending` row.

### Operation and assurance records

An operation record is a computed view over the layers of a lowered program. It
stores nothing the program does not already determine. Assurance records name
the compiled routes and support units they cover.

### Amendments

- ADR 0024: transport compiles through Study, by a transport identification
  layer and multi-source data identity. This supersedes the clause that
  transport must not grow a second workflow engine or a new `analyze` shape,
  for that purpose. The theorem-family, outcome and catalog contracts of 0024
  are unchanged.
- ADR 0022: multi-source data sets have their own identity domain, and a
  program may be composed over child programs. Existing domains and their
  separation are unchanged.
- ADR 0020: the support model above replaces the per-capability, method-list and
  scenario status layers. Matrix cells keep their definition.

## Consequences

- A result from a deprecated path is reproduced exactly by its replacement, so
  migration changes no number. A difference in the deprecation note is a
  difference in the path, not in the kernel.
- Adding a facade item, a re-export or a method of a re-exported type fails the
  gate until it is classified. A new scientific capability needs a compiled
  route and a unit before it can be released.
- Support status has one source. A route without a unit cannot lower.
- Kernel crates stay free to change signatures; callers outside the facade take
  that risk.
- Removing the deprecated wrappers in 4.0 is a breaking change to the facade and
  the Python package, announced from 3.1 by the compiler and interpreter
  warnings.
