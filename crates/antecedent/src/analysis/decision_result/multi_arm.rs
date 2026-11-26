//! Arm means of one multi-arm randomized experiment, as decision coordinates.
//!
//! The published scalar of a multi-arm result is one contrast (the second arm minus the
//! reference). A decision among the arms needs each arm's mean. This reads those means from
//! the result that was estimated, one coordinate per arm, and leaves the variances behind:
//! they are not a joint law. Two outcomes join only when they were estimated under one
//! assignment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CanonicalEncode, CanonicalWriter, CausalQuery, RandomizationDesign, SemanticDigest, VariableId,
};
use antecedent_design::decision_adapters::AdapterError;

use super::{DecisionProjectionError, OUTCOME_NAME, causal_source_identity};
use crate::result::StudyResult;

const NOT_MULTI_ARM: &str = "the result is a multi-arm randomized experiment";
const FINITE_MEANS: &str = "arm means are finite";
const DECLARED_POPULATION: &str = "population is declared, non-empty and at most 256 bytes";
const DECLARED_UNITS: &str = "units are declared for the outcome, non-empty and at most 256 bytes";
const REGIMES: &str = "regimes name each arm once, with a distinct non-empty regime";
const DESIGN_TAG: &str = "antecedent.multi_arm_decision_design.v1";

/// Provider label of the mean source these coordinates belong to.
pub const PROVIDER_ID: &str = "native:multi_arm_means";
/// RNG label of that source. The means are estimated, not drawn.
pub const RNG_ID: &str = "none:arm_means";

/// Arm means and the assignment identity, snapshotted from the result before it crosses a binding.
#[derive(Clone, Debug)]
pub struct MultiArmDecisionAuthority {
    outcome: VariableId,
    labels: Vec<String>,
    means: Vec<f64>,
    design_id: String,
    snapshot_id: String,
    program_id: String,
}

impl MultiArmDecisionAuthority {
    /// Read the multi-arm means and the assignment from the result. `None` when the result is
    /// not a multi-arm randomized experiment with retained arm means and an executed contract.
    #[must_use]
    pub fn from_result(result: &StudyResult) -> Option<Self> {
        let CausalQuery::RandomizedEffect(query) = &result.identification.query else {
            return None;
        };
        if !matches!(query.design, RandomizationDesign::MultiArm { .. }) {
            return None;
        }
        let effect = result.randomized_effect()?;
        let values = effect.multi_arm_values.as_ref();
        if values.is_empty() {
            return None;
        }
        let executed = result.executed_contract()?;
        let mut writer = CanonicalWriter::new(DESIGN_TAG);
        query.design.encode(&mut writer);
        writer.texts(query.assignment_units.as_ref());
        writer.texts(query.outcome_units.as_ref());
        Some(Self {
            outcome: query.outcome,
            labels: values.iter().map(|(label, ..)| label.as_ref().to_owned()).collect(),
            means: values.iter().map(|(_, mean, ..)| *mean).collect(),
            design_id: SemanticDigest::from_bytes(writer.finish()).to_hex(),
            snapshot_id: executed.identities.data_snapshot.to_hex(),
            program_id: causal_source_identity(executed),
        })
    }

    /// Outcome variable the means were estimated for.
    #[must_use]
    pub const fn outcome(&self) -> VariableId {
        self.outcome
    }
}

/// One outcome's arm-mean coordinates, in arm order, with the assignment they were estimated under.
#[derive(Clone, Debug)]
pub struct MultiArmMeanProjection {
    /// Outcome name.
    pub outcome: String,
    /// Identity of the assignment: arms, realized assignment, probabilities and unit labels.
    pub design_id: String,
    /// Data snapshot of this outcome's result.
    pub snapshot_id: String,
    /// Program identity of this outcome's result.
    pub program_id: String,
    /// Arm labels, in the order of `means` and `coordinates`.
    pub labels: Vec<String>,
    /// One mean coordinate per arm.
    pub coordinates: Vec<ScientificQuantity>,
    /// Horvitz-Thompson arm means. Variances are not carried.
    pub means: Vec<f64>,
}

/// Project the snapshotted arm means to one mean coordinate per arm.
///
/// `regimes` maps each arm label to the regime the decision uses. The coordinates use
/// functional `mean` at horizon 0. A caller that needs a joint law is refused later, by the
/// mean-source rule, because these means are not a law.
///
/// # Errors
///
/// The result is not a multi-arm experiment; the outcome name is missing; population, units or
/// regimes are not the declaration this projection accepts; or an arm mean is not finite.
#[allow(
    clippy::result_large_err,
    reason = "ExternalRefusal is the structured refusal of the external contract, returned by value on a cold once-per-decision refusal path"
)]
pub fn project_multi_arm_means(
    authority: Option<&MultiArmDecisionAuthority>,
    population: &str,
    units: &str,
    regimes: &[(&str, &str)],
    outcome: &str,
) -> Result<MultiArmMeanProjection, DecisionProjectionError> {
    let Some(authority) = authority else {
        return Err(invalid(NOT_MULTI_ARM));
    };
    if outcome.is_empty() {
        return Err(DecisionProjectionError::Declaration(OUTCOME_NAME));
    }
    if !declared(population) {
        return Err(invalid(DECLARED_POPULATION));
    }
    if !declared(units) {
        return Err(invalid(DECLARED_UNITS));
    }
    if authority.means.len() != authority.labels.len()
        || authority.means.iter().any(|mean| !mean.is_finite())
    {
        return Err(invalid(FINITE_MEANS));
    }
    let regimes = regime_by_arm(&authority.labels, regimes)?;
    let coordinates = authority
        .labels
        .iter()
        .map(|label| ScientificQuantity {
            variable_id: outcome.to_owned(),
            variable_name: outcome.to_owned(),
            role: QuantityRole::Outcome,
            units: units.to_owned(),
            population_id: population.to_owned(),
            regime_id: regimes[label.as_str()].to_owned(),
            horizon: 0,
            functional_id: "mean".to_owned(),
            conditioning: Vec::new(),
            transform_id: "identity".to_owned(),
        })
        .collect();
    Ok(MultiArmMeanProjection {
        outcome: outcome.to_owned(),
        design_id: authority.design_id.clone(),
        snapshot_id: authority.snapshot_id.clone(),
        program_id: authority.program_id.clone(),
        labels: authority.labels.clone(),
        coordinates,
        means: authority.means.clone(),
    })
}

fn declared(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty() && trimmed.len() == text.len() && text.len() <= 256
}

fn regime_by_arm<'a>(
    labels: &[String],
    regimes: &[(&'a str, &'a str)],
) -> Result<BTreeMap<&'a str, &'a str>, DecisionProjectionError> {
    let mut by_arm = BTreeMap::new();
    for (label, regime) in regimes {
        if by_arm.insert(*label, *regime).is_some() {
            return Err(invalid(REGIMES));
        }
    }
    if by_arm.len() != labels.len()
        || labels.iter().any(|label| !by_arm.contains_key(label.as_str()))
    {
        return Err(invalid(REGIMES));
    }
    let mut seen = BTreeSet::new();
    for regime in by_arm.values() {
        if !declared(regime) || !seen.insert(*regime) {
            return Err(invalid(REGIMES));
        }
    }
    Ok(by_arm)
}

fn invalid(why: &'static str) -> DecisionProjectionError {
    AdapterError::InvalidClaims(why).to_refusal().into()
}
