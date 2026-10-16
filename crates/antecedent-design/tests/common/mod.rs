//! Fixtures shared by the design integration tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{QuantityRole, ScientificQuantity};

/// An observational outcome quantity on the target population; tests override the
/// fields their scenario varies with struct-update syntax.
pub fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}
