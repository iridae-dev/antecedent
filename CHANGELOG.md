# Changelog

All notable changes to Antecedent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-23

First crates.io-oriented release of the Rust library graph.

### Added

- Day-1 facade crate **`antecedent`** (`use antecedent::prelude::*`).
- Supporting crates published as **`antecedent-*`** (`antecedent-core`, …).
- Workspace publish metadata (repository, homepage, docs.rs, keywords, categories).
- `scripts/publish_crates.sh` and tag-driven `.github/workflows/publish-crates.yml`.
- `#[non_exhaustive]` on key public result / config types; sealed extension traits
  (`Identifier`, `Estimator`, `DiscoveryAlgorithm`, `Validator`).
- `FromStr` / `Display` for `IdentifierId` and `EstimatorId`.

### Notes

- **`0.1.x` may still introduce breaking changes.** Treat the release as a preview.
- Supporting libraries use **`antecedent-*`** names on crates.io and are **public
  dependencies** of `antecedent` (part of the semver surface). Day-1 usage is
  still only `cargo add antecedent`.
- The Python extension (`antecedent-py` / wheel `antecedent` on PyPI) is
  **not** published to crates.io (`publish = false`).
- `CustomEffectValidator` remains deliberately unsealed so host languages (PyO3)
  can implement the dyn-safe callback path.
- Known 0.1 API debt: many result structs still expose public fields rather than
  getters; prefer constructors (`::new` / `::from_parts`) for cross-crate builds.

[0.1.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.1.0
