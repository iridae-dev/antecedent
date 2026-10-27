//! Standing inputs declared by each selective-recalculation session family.
//!
//! The declarations are Rust's: `RecalcSessionKind::standing` states, per family, whether a
//! calibration record, a representation and a coupling enter the stage identities of the law and
//! the decision, are not applicable, or cannot be declared and why. This module only carries the
//! read-only declaration to Python as JSON.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_receipt::{
    RecalcSessionKind, SessionStanding, StandingInputDeclaration,
};
use pyo3::prelude::*;
use serde_json::{Value, json};

fn kind_of(name: &str) -> PyResult<RecalcSessionKind> {
    Ok(match name {
        "receipt" => RecalcSessionKind::Receipt,
        "adjusted" => RecalcSessionKind::Adjusted,
        "cell" => RecalcSessionKind::Cell,
        "design" => RecalcSessionKind::Design,
        "doubly_robust" => RecalcSessionKind::DoublyRobust,
        "static" => RecalcSessionKind::Static,
        "temporal" => RecalcSessionKind::Temporal,
        "bayesian" => RecalcSessionKind::Bayesian,
        "joint_bayesian" => RecalcSessionKind::JointBayesian,
        "external" => RecalcSessionKind::External,
        "composite" => RecalcSessionKind::Composite,
        other => {
            return Err(crate::value_err(format!(
                "unknown recalculation session kind {other:?}; use receipt, adjusted, cell, \
                 design, doubly_robust, static, temporal, bayesian, joint_bayesian, external or \
                 composite"
            )));
        }
    })
}

fn declaration_json(declaration: StandingInputDeclaration) -> Value {
    match declaration {
        StandingInputDeclaration::Declared => json!({ "status": "declared" }),
        StandingInputDeclaration::NotApplicable { reason } => {
            json!({ "status": "not_applicable", "reason": reason })
        }
        StandingInputDeclaration::Undeclared { finding, remedy } => {
            json!({ "status": "undeclared", "finding": finding, "remedy": remedy })
        }
        // The enum is non-exhaustive: a declaration this binding does not know is reported
        // as such rather than guessed.
        _ => json!({ "status": "unrecognized" }),
    }
}

fn standing_json(standing: SessionStanding) -> Value {
    json!({
        "calibration": declaration_json(standing.calibration),
        "representation": declaration_json(standing.representation),
        "coupling": declaration_json(standing.coupling),
    })
}

/// The standing declarations of a session family, as JSON with `calibration`,
/// `representation` and `coupling` entries.
#[pyfunction]
fn recalc_session_standing(kind: &str) -> PyResult<String> {
    let standing = kind_of(kind)?.standing();
    serde_json::to_string(&standing_json(standing)).map_err(|e| crate::value_err(e.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(recalc_session_standing, module)?)?;
    Ok(())
}
