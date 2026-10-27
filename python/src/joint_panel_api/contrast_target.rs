//! Declare an assembled contrast over the worlds of an existing scientific target.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::{InterventionContrast, ScientificTarget, ScientificWorldId, TargetForm};
use antecedent_io::scientific_target_wire::ScientificTargetWire;
use pyo3::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Contrast {
    Linear { terms: Vec<(u32, f64)>, reference: u32 },
    Difference { plus: u32, minus: u32, reference: u32 },
}

/// Re-form a target over the same worlds as one checked assembled contrast.
///
/// The worlds, population, units, inference realization and declared assumptions are kept; only
/// the question changes, so the result carries its own scientific identity. Domain validation
/// refuses a missing, repeated or extra world, a zero or non-finite coefficient and a reference
/// that is not exactly the flagged baseline.
#[pyfunction]
fn joint_target_contrast(target_json: &str, contrast_json: &str) -> PyResult<String> {
    for (text, what) in [(target_json, "scientific target"), (contrast_json, "target contrast")] {
        crate::bridge::check_declared_size(text, crate::bridge::MAX_SPEC_DECLARATION_BYTES, what)?;
    }
    let wire: ScientificTargetWire =
        serde_json::from_str(target_json).map_err(|e| crate::value_err(e.to_string()))?;
    let target = wire.to_domain().map_err(|e| crate::value_err(e.to_string()))?;
    let declaration: Contrast =
        serde_json::from_str(contrast_json).map_err(|e| crate::value_err(e.to_string()))?;
    let worlds = match target.form() {
        TargetForm::Worlds(worlds) => worlds.clone(),
        TargetForm::Contrast(contrast) => contrast.worlds().clone(),
        _ => {
            return Err(crate::value_err(
                "a target contrast needs a collection of intervention worlds",
            ));
        }
    };
    let id = ScientificWorldId::from_raw;
    let contrast = match declaration {
        Contrast::Linear { terms, reference } => InterventionContrast::linear(
            worlds,
            terms.into_iter().map(|(world, coefficient)| (id(world), coefficient)),
            id(reference),
        ),
        Contrast::Difference { plus, minus, reference } => {
            InterventionContrast::difference(worlds, id(plus), id(minus), id(reference))
        }
    }
    .map_err(|e| crate::value_err(e.to_string()))?;
    let contrasted = ScientificTarget::new(
        TargetForm::Contrast(contrast),
        target.assumptions().clone(),
        target.inference(),
    );
    let encoded = ScientificTargetWire::from_domain(&contrasted)
        .map_err(|e| crate::value_err(e.to_string()))?;
    serde_json::to_string(&encoded).map_err(|e| crate::value_err(e.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(joint_target_contrast, module)?)
}
