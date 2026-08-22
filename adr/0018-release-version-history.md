# ADR 0018 — release version history and artifact format correction

- Status: Accepted
- Date: 2026-07-29
- Supersedes: [0017](0017-release-prep.md)

## Context

ADR 0017 (2026-07-21) decided to keep crate and Python package versions at
**0.1.0** and freeze the artifact format at `FormatVersion { major: 0, minor:
1 }` as part of 1.0 preparation. Four releases have since shipped, each
documented in `CHANGELOG.md`:

- **0.2.0** (2026-07-24) — correctness cut (sandwich/AIPW/HMC publication
  gaps).
- **0.3.0** (2026-07-25) — second correctness cut plus a facade / Python
  binding maintainability pass.
- **0.4.0** (2026-07-29) — Python API-surface freeze (root namespace frozen
  to 41 names; silent hard break, no deprecated aliases).
- **0.4.1** (2026-07-30) — patch: `identify()` accepts an `Admg`, so latent
  confounding fails honestly instead of resolving to an adjustment set the
  study cannot measure.

The artifact format moved to `FormatVersion { major: 0, minor: 2 }`
(`antecedent_io::STABLE_FORMAT`, `crates/antecedent-io/src/migrate.rs`)
during the 0.2.0 cut and has not changed since; `{ major: 0, minor: 1 }`
remains a supported **source** format for migration
(`SUPPORTED_SOURCE_FORMATS`) but is not the frozen target. ADR 0017's
version and format freeze is now stale and, left uncorrected, misleads
anyone who reads it as current guidance — several docs
(`docs/development.md`, `docs/artifacts.md`, `docs/security_review.md`)
independently needed correction to match reality.

## Decision

- Supersede ADR 0017's version-freeze and format-freeze clauses. ADR 0017's
  other decisions (DOT + JSON + GML + NetworkX interchange, the full wheel
  CI matrix, conformance-generated docs, `DESIGN.md` retirement) stand and
  are not reopened by this ADR.
- Record the actual released version history: **0.1.0** (2026-07-23, day-1
  crates.io facade) → **0.2.0** (2026-07-24, correctness cut) → **0.3.0**
  (2026-07-25, second correctness cut + facade/binding maintainability) →
  **0.4.0** (2026-07-30, Python API-surface freeze + algorithmic correctness
  pass: 25 confirmed defects fixed across identification, discovery,
  estimation, attribution, and diagnostics — see the 0.4.0 `CHANGELOG.md`
  entry's *Correctness* section and `docs/release-notes/v0.4.0.md`) →
  **0.4.1** (2026-07-30, patch: identify-only accepts an `Admg`).
- Record the historical artifact format at this decision: `FormatVersion {
  major: 0, minor: 2 }`, unchanged from 0.2.0 through 0.4.1. ADR 0019 later
  advanced response artifacts to 0.3, and ADR 0021 advanced temporal response
  artifacts to the current `antecedent_io::STABLE_FORMAT`, 0.4. Formats
  0.1–0.3 remain supported migration sources, not the frozen target.
- Future version bumps continue to be recorded in `CHANGELOG.md` (breaking
  changes called out per the 0.4.0 entry's convention) rather than requiring
  a new ADR on every release; a new ADR is only needed if the artifact
  format freeze itself changes.

## Consequences

ADR 0017 is marked Superseded and points here for version/format history.
`docs/artifacts.md` follows the live stable-format constant, currently 0.4;
ADR 0019 and ADR 0021 record the two later wire advances. Package releases
after 0.4.1 are recorded in `CHANGELOG.md` and their release notes rather than
retroactively rewriting the historical version list above.
