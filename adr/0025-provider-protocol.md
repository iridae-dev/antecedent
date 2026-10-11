# ADR 0025 — Provider protocol and adjustment handoff

- Status: Accepted
- Date: 2026-10-11
- Related: [0020](0020-support-matrix-and-prepared-workflow.md), [0023](0023-learner-substrate.md)

## Context

Antecedent identifies a point-identified backdoor or generalized-adjustment
estimand and can hand the certified set to a caller-owned learner. That
handoff was published as `handoff.econml`, `EconMLSpec`, and
`EconMLProviderAdapter`. The body never imports EconML. `learner` is an
opaque label on an attested receipt.

`CausalProvider`, `CausalProviderSpec`, and `ProviderRegistry` in
`antecedent.extensibility` are the contract for that handoff and for a
separately installed `antecedent.providers` entry point. The registry is
Python-only. Import does not scan plugins. `analyze` does not recast plugin
output as a native point or interval. `ProviderRegistry.verify` checks
caller-supplied fixtures and records a Python report; only an exact verified
request is `verified_extension`.

This is not ADR 0023. ADR 0023 is the native nuisance-learner substrate
(`antecedent-learn`: ridge, forests, and the owned prediction contract). It
does not govern external providers.

`antecedent.external` is a different inbound surface. It binds a foreign
response grid to an identified query (`BoundExternalClaim`) with lineage and
per-coordinate support. The adjustment handoff prepares columns and attaches
an attested external-estimate receipt. It does not bind an `ExternalSpec`.

## Decision

The adjustment handoff is the provider. EconML is one consumer of it.

- `handoff.adjustment(...)` returns an `AdjustmentHandoff`.
- `columns(data, layout="econml"|"named")` defaults to `econml`, the
  `Y` / `T` / `W` / `X` vocabulary. `named` returns `outcome`, `treatment`,
  `confounders`, `modifiers`, `weights`, and `origins`.
- `as_provider(executor)` returns an `AdjustmentProviderAdapter`. The
  executor is a caller-owned callback. This module does not fit a learner,
  assign cross-fit folds, or calibrate an interval.
- `tool` is the provenance adapter identity on the provider spec. It
  defaults to `antecedent.handoff.EconMLSpec`. An omitted tool leaves that
  spec digest unchanged. Registry execution keeps
  `antecedent.handoff.EconMLProviderAdapter` unless the caller sets `tool`.
- `handoff.econml`, `EconMLSpec`, and `EconMLProviderAdapter` are aliases of
  `adjustment`, `AdjustmentHandoff`, and `AdjustmentProviderAdapter`.

Front-door, IV, general identification, partial identification, and
graph-posterior mixtures still refuse. There is no single adjustment set to
hand off.

No CausalML (or other package) implementation is implied. A second learner
uses this handoff and sets `tool`.

## Consequences

- Existing EconML callers keep working, including `columns()` with no layout.
- Spec digests for the default tool match the historical EconML spec provenance.
- Native nuisance learning stays on ADR 0023. Foreign response grids stay on
  `antecedent.external`.
- `parity/estimate.toml` records the shipped handoff as done. In-process
  learner composition stays excluded. `causalml` stays pending and unimplemented.
