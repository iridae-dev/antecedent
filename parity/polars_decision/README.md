# Polars scope

Direct Arrow remains the preparation path. The empty `polars` feature of `crates/antecedent-data` was retired together with its crate-private seam and pool-refusal modules; no Polars crate dependency was added. Parquet and Arrow IPC files are read through the arrow-rs `parquet` and `arrow-ipc` crates behind the `parquet` and `ipc` features (`crates/antecedent-data/src/file_source`).

The candidate inspected on 2026-10-19 is crates.io `polars` `0.55.2` (tag `rs-0.55.2`, published 2026-08-06). Its crate `Cargo.toml` and workspace `Cargo.toml` do not declare `rust-version`. crates.io publishes no `rust_version` for the crate. The tag toolchain is `nightly-2026-04-01`. On 2026-10-20, `cargo build` of that crate with rustc 1.85.1 stopped before compile: transitive crates require Rust 1.88, 1.89, or 1.95.

`decision.toml` leaves latency, RSS, spill, build, MSRV, and wheel as `not_measured`.
