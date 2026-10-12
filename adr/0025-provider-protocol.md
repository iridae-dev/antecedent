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

## Interop coherence (2.4.0)

The four inbound surfaces (registry and handoff, `external.bind`,
`recalc_external` callbacks, distribution artifacts) now share three things.

- One verification receipt. `ProviderRegistry.verify` fixtures are sugar for the
  Rust `ProviderVerificationReceipt` that also verifies a typed scientific
  object. The receipt exports through the shared container, loads under a
  retained digest and backs a composition input. Fixture reports are no longer
  a Python-only evidence record, and the registry's trust ladder is the Rust
  one.
- One provider identity. `ProviderRegistry.provider_object` derives the
  `external.ProviderObject` of a registered provider; a distribution identity
  derives its provider and snapshot from it. A spec declares `capabilities` and
  `meaning` once, typed.
- One inbound estimate path. `attach(label="ate")` binds the scalar through
  `external.bind`, and `attach(label="cate")` binds a row-level effect array as
  a column claim: one scientific quantity and the digest, size and support
  counts of the column, not one coordinate per row (a response grid is the
  wrong shape for thousands of rows). Both carry lineage, support labels and a
  provider version. An unlabelled array stays opaque attested evidence.
- One container and one digest. The provider result envelope (`ANTEPROV2`) and
  the callback output (`ANTCALLBACK1`) are written as `provider_result_v1` and
  `external_callback_v1` in the shared sectioned container, load through
  `artifacts.loads`, and the original frames stay readable. Every Python
  identity digest is Rust's `canonical_json_digest` under a named domain;
  nothing in Python hashes `json.dumps` output any more, and a request that has
  no canonical form (an arbitrary object) is refused instead of hashed by its
  `repr`.
- One refusal shape. Registry and handoff refusals are `ProviderRefusal`s with a
  registered reason code, a namespaced detail and a remedy, as `ExternalRefusal`
  is for binding.

## Consequences

- Existing EconML callers keep working, including `columns()` with no layout.
- Spec digests for the default tool match the historical EconML spec provenance.
- Native nuisance learning stays on ADR 0023. Foreign response grids stay on
  `antecedent.external`.
- `parity/estimate.toml` records the shipped handoff as done. In-process
  learner composition stays excluded. `causalml` stays pending and unimplemented.
