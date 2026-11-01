//! Compiler-owned joint cluster randomization with declared preassignment adjustment.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{excursion::ExcursionOutcome, experiment::ExperimentExposure};
use std::collections::BTreeSet;

/// A fixed pretreatment adjustment bound to its original observed covariate totals.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedClusterAdjustment {
    /// Name of the outcome coordinate.
    pub component_name: String,
    /// Pretreatment covariate identity.
    pub covariate: String,
    /// Coefficient fixed independently of experimental outcomes.
    pub coefficient: f64,
    /// Aligned pretreatment totals.
    pub pretreatment_totals: Vec<f64>,
    /// True centers the covariate and restores original arm-mean scale.
    pub centered: bool,
    /// Declaration that coefficient and covariate precede assignment/outcomes.
    pub preassignment: String,
}

/// Complete cluster-randomized source with outcome and assignment provenance.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointClusterRecipe {
    /// Source population.
    pub population: String,
    /// Exact active intervention identity.
    pub treated_regime_id: String,
    /// Exact reference intervention identity.
    pub control_regime_id: String,
    /// Decision horizon.
    pub horizon: u32,
    /// Independent assignment cluster identities.
    pub cluster_ids: Vec<String>,
    /// Fixed population sizes per cluster.
    pub cluster_sizes: Vec<usize>,
    /// Complete randomization declaration including fixed treated count.
    pub assignment_law: String,
    /// Prespecified number of treated clusters under complete randomization.
    pub treated_clusters: usize,
    /// Consistency and treatment-version premise.
    pub consistency: String,
    /// Realized cluster assignments.
    pub assignments: Vec<bool>,
    /// Measured outcome totals, one value per cluster.
    pub outcomes: Vec<ExcursionOutcome>,
    /// Declared pretreatment adjustments, computed inside the source owner.
    pub adjustments: Vec<FixedClusterAdjustment>,
    /// Exposure assumptions or bias bounds.
    pub exposure: ExperimentExposure,
}

/// Arm-coordinate means and conservative per-arm Neyman covariance bounds.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointClusterLaw {
    /// Treated arm means.
    pub treated_means: Vec<f64>,
    /// Reference arm means.
    pub control_means: Vec<f64>,
    /// Treated-arm covariance.
    pub treated_covariance: Vec<f64>,
    /// Reference-arm covariance.
    pub control_covariance: Vec<f64>,
    /// Assigned clusters in the treated arm.
    pub treated_clusters: usize,
    /// Assigned clusters in the reference arm.
    pub control_clusters: usize,
    /// Fixed total outcome rows.
    pub outcome_rows: usize,
}

/// Evaluate the existing joint Neyman estimator from the retained source.
/// # Errors
/// Invalid population, source, assignment, adjustments, bounds or cancellation.
pub fn joint_cluster(
    recipe: &JointClusterRecipe,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointClusterLaw, String> {
    let n = recipe.assignments.len();
    let k = recipe.outcomes.len();
    if n == 0
        || k == 0
        || recipe.cluster_ids.len() != n
        || recipe.cluster_sizes.len() != n
        || recipe.cluster_sizes.contains(&0)
        || recipe.treated_clusters != recipe.assignments.iter().filter(|a| **a).count()
        || recipe.cluster_ids.iter().any(|v| v.trim().is_empty())
        || recipe.cluster_ids.iter().collect::<BTreeSet<_>>().len() != n
        || [
            &recipe.population,
            &recipe.treated_regime_id,
            &recipe.control_regime_id,
            &recipe.assignment_law,
            &recipe.consistency,
        ]
        .iter()
        .any(|v| v.trim().is_empty())
        || recipe.treated_regime_id == recipe.control_regime_id
        || recipe.outcomes.iter().any(|o| {
            o.component_name.trim().is_empty()
                || o.variable_id.trim().is_empty()
                || o.units.trim().is_empty()
                || o.values.len() != n
                || o.values.iter().any(|v| !v.is_finite())
        })
        || recipe.outcomes.iter().map(|o| &o.component_name).collect::<BTreeSet<_>>().len() != k
    {
        return Err("invalid cluster-randomized source, assignment or premises".into());
    }
    let check_bounds = |v: &[f64]| v.len() == k && v.iter().all(|x| x.is_finite() && *x >= 0.0);
    let exposure_valid = match &recipe.exposure {
        ExperimentExposure::NoInterference { assumption } => !assumption.trim().is_empty(),
        ExperimentExposure::ContrastBounds { assumption, bounds } => {
            !assumption.trim().is_empty() && check_bounds(bounds)
        }
        ExperimentExposure::ArmBounds { assumption, treated, control } => {
            !assumption.trim().is_empty() && check_bounds(treated) && check_bounds(control)
        }
    };
    if !exposure_valid {
        return Err("invalid exposure assumption or spillover bounds".into());
    }
    let rows = recipe
        .cluster_sizes
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or("cluster-size overflow")?;
    let mut adjusted: Vec<_> = recipe.outcomes.iter().map(|o| o.values.clone()).collect();
    let mut seen = BTreeSet::new();
    for a in &recipe.adjustments {
        if ctx.cancellation.is_cancelled() {
            return Err("joint cluster experiment cancelled".into());
        }
        let index = recipe
            .outcomes
            .iter()
            .position(|o| o.component_name == a.component_name)
            .ok_or("unknown adjusted component")?;
        if !seen.insert(index)
            || a.covariate.trim().is_empty()
            || a.preassignment.trim().is_empty()
            || !a.coefficient.is_finite()
            || a.pretreatment_totals.len() != n
            || a.pretreatment_totals.iter().any(|v| !v.is_finite())
        {
            return Err("invalid or repeated fixed pretreatment adjustment".into());
        }
        adjusted[index] = if a.centered {
            crate::randomized_joint::centered_fixed_cuped_totals(
                &adjusted[index],
                &a.pretreatment_totals,
                &recipe.cluster_sizes,
                a.coefficient,
            )
            .ok_or("invalid centered cluster adjustment")?
        } else {
            adjusted[index]
                .iter()
                .zip(&a.pretreatment_totals)
                .map(|(y, x)| y - a.coefficient * x)
                .collect()
        };
    }
    let refs: Vec<_> = adjusted.iter().map(Vec::as_slice).collect();
    let fit =
        crate::randomized_joint::complete_cluster_joint_contrast(&refs, &recipe.assignments, rows)
            .ok_or("invalid cluster randomization or outcome totals")?;
    if ctx.cancellation.is_cancelled() {
        return Err("joint cluster experiment cancelled".into());
    }
    Ok(JointClusterLaw {
        treated_means: fit.treated_means,
        control_means: fit.control_means,
        treated_covariance: fit.treated_covariance,
        control_covariance: fit.control_covariance,
        treated_clusters: fit.treated_clusters,
        control_clusters: fit.control_clusters,
        outcome_rows: rows,
    })
}
