//! Inventory of estimators that execute on a supported cell although the cell's evidence list
//! does not name them.
//!
//! For every supported cell and every `EstimatorId`, a study is built for the cell's coordinate
//! with that estimator explicitly selected, prepared and estimated (no bootstrap, no refuters).
//! In a development build a route the support gate would refuse in a release build runs and
//! records a `would_refuse` entry naming `estimator_evidence:<id>`; that pair is
//! `unlisted_executes`. A build, prepare or estimate refusal is a scientific or compatibility
//! refusal and is skipped. A run with no estimator-evidence entry is `listed`.
//!
//! The pairs classified `unlisted_executes` are written to
//! `<target>/estimator_support_enumeration.json`. The count is printed and never asserted; the
//! test fails only when a study cannot be constructed for a cell.
//!
//! The cell fixtures are shared with `supported_compiler.rs`, which is included as a module, so
//! its own tests also register in this binary: run with
//! `--exact estimator_support_enumeration`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    dead_code,
    unused_imports,
    clippy::too_many_lines,
    reason = "the shared cell fixtures are included whole; this file uses only their setup"
)]

#[path = "supported_compiler.rs"]
mod fixtures;

use std::panic::{AssertUnwindSafe, catch_unwind};

use antecedent::support::classify_estimator;
use antecedent::{CellStatus, EstimatorId, supported_cells};
use antecedent_core::ExecutionContext;
use fixtures::{CellData, CellSetup, cell_setup, parallel_map};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Class {
    /// Ran with a would-refuse entry for this estimator's missing evidence.
    UnlistedExecutes,
    /// Ran with no estimator-evidence would-refuse entry.
    Listed,
    /// Refused by a scientific or compatibility check (build, prepare or estimate), or panicked.
    Skipped,
}

/// Classify one estimator on one cell's study.
fn classify(setup: &CellSetup, estimator: EstimatorId) -> Class {
    let needle = format!("estimator_evidence:{}", estimator.as_str());
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let ctx = ExecutionContext::for_tests(1);
        let built = setup.builder.clone().estimator(estimator).build().ok()?;
        let prepared = built.prepare(&ctx).ok()?;
        let result = match &setup.data {
            CellData::Tabular(data) => prepared.estimate(data, &ctx),
            CellData::Series(data) => prepared.estimate_series(data, &ctx),
        }
        .ok()?;
        let unlisted = result
            .would_refuse()
            .iter()
            .any(|entry| entry.missing.iter().any(|missing| missing == &needle));
        Some(if unlisted { Class::UnlistedExecutes } else { Class::Listed })
    }));
    outcome.ok().flatten().unwrap_or(Class::Skipped)
}

struct CellOutcome {
    coordinate: String,
    unlisted: Vec<&'static str>,
    listed: usize,
    skipped: usize,
    /// Estimators whose observed class contradicts `classify_estimator` for this cell.
    disagreements: Vec<String>,
}

fn enumerate(cell: antecedent::SupportCell) -> Result<CellOutcome, String> {
    let setup = cell_setup(cell)?;
    let mut out = CellOutcome {
        coordinate: setup.expected.clone(),
        unlisted: Vec::new(),
        listed: 0,
        skipped: 0,
        disagreements: Vec::new(),
    };
    for &estimator in EstimatorId::ALL {
        let on_list = classify_estimator(cell, estimator) == CellStatus::Supported;
        match classify(&setup, estimator) {
            Class::UnlistedExecutes => {
                out.unlisted.push(estimator.as_str());
                if on_list {
                    out.disagreements.push(format!(
                        "{}: on the list but recorded would_refuse",
                        estimator.as_str()
                    ));
                }
            }
            Class::Listed => {
                out.listed += 1;
                if !on_list {
                    out.disagreements.push(format!(
                        "{}: off the list but recorded no would_refuse",
                        estimator.as_str()
                    ));
                }
            }
            Class::Skipped => out.skipped += 1,
        }
    }
    Ok(out)
}

#[test]
fn estimator_support_enumeration() {
    let cells: Vec<_> = supported_cells().collect();
    let mut construction_failures = Vec::new();
    let mut pairs = Vec::new();
    let (mut listed, mut skipped, mut disagreements) = (0usize, 0usize, Vec::new());
    for outcome in parallel_map(&cells, |&cell| enumerate(cell)) {
        match outcome {
            Err(error) => construction_failures.push(error),
            Ok(cell) => {
                listed += cell.listed;
                skipped += cell.skipped;
                for estimator in &cell.unlisted {
                    pairs.push(serde_json::json!({
                        "coordinate": cell.coordinate,
                        "estimator": estimator,
                    }));
                }
                disagreements.extend(
                    cell.disagreements.into_iter().map(|d| format!("{}: {d}", cell.coordinate)),
                );
            }
        }
    }
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"),
        std::path::PathBuf::from,
    );
    std::fs::create_dir_all(&target).expect("create target dir");
    let path = target.join("estimator_support_enumeration.json");
    let report = serde_json::json!({
        "cells": cells.len(),
        "estimators": EstimatorId::ALL.len(),
        "unlisted_executes_count": pairs.len(),
        "listed_count": listed,
        "skipped_count": skipped,
        "list_disagreements": disagreements,
        "unlisted_executes": pairs,
    });
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write enumeration report");
    println!(
        "estimator_support_enumeration: {} unlisted_executes, {} listed, {} skipped, {} \
         list disagreements over {} cells x {} estimators -> {}",
        pairs.len(),
        listed,
        skipped,
        disagreements.len(),
        cells.len(),
        EstimatorId::ALL.len(),
        path.display()
    );
    assert!(
        construction_failures.is_empty(),
        "{} supported cells could not be constructed:\n{}",
        construction_failures.len(),
        construction_failures.join("\n")
    );
}
