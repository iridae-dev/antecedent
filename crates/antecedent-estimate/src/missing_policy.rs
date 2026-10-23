//! Refusals shared by every route that needs a declared missing-data policy.
//!
//! A route refuses a table with missing values until a policy is declared, and names the
//! first missing cell and the policies it can execute, so the caller sees exactly what to
//! declare. A declared policy the route cannot execute is refused the same way.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::{MissingCell, MissingPolicy};

use crate::EstimationError;
use crate::util::refuse;

/// A missing value on a route that has no default for it.
#[must_use]
pub fn policy_required(
    cell: &MissingCell,
    route: &str,
    available: &[MissingPolicy],
) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("missing_policy_required"),
        "missing_data.policy_required",
        &antecedent_data::policy_required_text(cell, route, available),
    )
}

/// A declared policy the route cannot execute for its design.
#[must_use]
pub fn policy_not_applicable(
    declared: MissingPolicy,
    route: &str,
    reason: &str,
    supported: &[MissingPolicy],
) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("missing_policy_not_applicable"),
        "missing_data.policy_not_applicable",
        &format!(
            "{route} cannot execute missing-data policy `{}`: {reason}; use one of: {}",
            declared.name(),
            policy_names(supported),
        ),
    )
}

/// A declared policy the route cannot execute, naming the supported policy to declare next.
#[must_use]
pub fn policy_not_applicable_next(
    declared: MissingPolicy,
    route: &str,
    reason: &str,
    supported: &[MissingPolicy],
    next: MissingPolicy,
) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("missing_policy_not_applicable"),
        "missing_data.policy_not_applicable",
        &format!(
            "{route} cannot execute missing-data policy `{}`: {reason}; use one of: {}; next \
             attempt: declare `{}`",
            declared.name(),
            policy_names(supported),
            next.name(),
        ),
    )
}

/// The declared observation-model covariates are themselves incomplete.
#[must_use]
pub fn covariates_incomplete(cell: &MissingCell, route: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("missing_policy_not_applicable"),
        "missing_data.covariates_incomplete",
        &format!(
            "{route} models the probability of an observed outcome on declared covariates, and \
             covariate `{}` is missing at row {}; the covariates must be complete (a missing \
             covariate would need imputation, which is not declared)",
            cell.variable, cell.row,
        ),
    )
}

fn policy_names(policies: &[MissingPolicy]) -> String {
    policies.iter().map(|policy| policy.name()).collect::<Vec<_>>().join(", ")
}
