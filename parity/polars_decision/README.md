# Polars scope (D09)

Direct Arrow remains the preparation path. `crates/antecedent-data` keeps the off-by-default feature `polars = []`. No Polars crate dependency was added.

The candidate inspected on 2026-10-19 is crates.io `polars` `0.55.2` (tag `rs-0.55.2`, published 2026-08-06). Its crate `Cargo.toml` and workspace `Cargo.toml` do not declare `rust-version`. crates.io publishes no `rust_version` for the crate. The tag toolchain is `nightly-2026-04-01`. That does not show a build on workspace Rust 1.85.

`decision.toml` leaves latency, RSS, spill, build, MSRV, and wheel as `not_measured`.
