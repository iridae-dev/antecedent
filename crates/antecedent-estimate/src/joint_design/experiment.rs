//! Compiler-owned joint experimental outcome coordinates.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::excursion::ExcursionOutcome;
use std::collections::{BTreeMap, BTreeSet};

/// Exposure assumption, retained with bounds rather than converted to sampling uncertainty.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExperimentExposure {
    /// No interference between assignment units.
    NoInterference {
        /// Declared assumption identity.
        assumption: String,
    },
    /// Bounds on component contrast bias.
    ContrastBounds {
        /// Declared assumption identity.
        assumption: String,
        /// Component-wise nonnegative bounds.
        bounds: Vec<f64>,
    },
    /// Bounds on both arm means.
    ArmBounds {
        /// Declared assumption identity.
        assumption: String,
        /// Treated-arm bounds.
        treated: Vec<f64>,
        /// Reference-arm bounds.
        control: Vec<f64>,
    },
}

/// Experiment-specific assignment, independent-unit and temporal declarations.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExperimentalDesign {
    /// Independent matched pairs with one treated cluster per pair.
    MatchedPairs {
        /// Stable cluster labels.
        cluster_ids: Vec<String>,
        /// Disjoint index pairs.
        pairs: Vec<(usize, usize)>,
        /// Declared pretreatment matching variables.
        matching_variables: Vec<String>,
        /// Fixed per-cluster treatment probabilities.
        treatment_probabilities: Vec<f64>,
        /// Fixed preassignment population sizes.
        cluster_sizes: Vec<usize>,
        /// Preassignment declaration.
        preassignment: String,
        /// Independent pair assignment declaration.
        independent_pairs: String,
    },
    /// Zero-washout period-randomized direct-effect switchback.
    Switchback {
        /// Independent sequence identity for each row.
        sequence_ids: Vec<String>,
        /// Unique period identity within each sequence.
        period_ids: Vec<String>,
        /// Marginal treatment probabilities.
        treatment_probabilities: Vec<f64>,
        /// No carryover assumption.
        no_carryover: String,
        /// Zero washout rule identity.
        washout_rule: String,
        /// Independent sequence assignment and sampling declaration.
        independent_sequences: String,
    },
}

/// Exact experimental source, outcomes and identifying premises.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointExperimentalRecipe {
    /// Source population.
    pub population: String,
    /// Active intervention identity.
    pub treated_regime_id: String,
    /// Reference intervention identity.
    pub control_regime_id: String,
    /// Decision horizon.
    pub horizon: u32,
    /// Consistency and treatment-version premise.
    pub consistency: String,
    /// Exposure premise.
    pub exposure: ExperimentExposure,
    /// Assignment design.
    pub design: ExperimentalDesign,
    /// Realized assignments.
    pub assignments: Vec<bool>,
    /// Joint measured outcome vectors.
    pub outcomes: Vec<ExcursionOutcome>,
}

/// Joint arm means and covariance with the actual independent-unit count.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointExperimentalLaw {
    /// Treated coordinates followed by reference coordinates.
    pub means: Vec<f64>,
    /// Row-major joint covariance in that order.
    pub covariance: Vec<f64>,
    /// Independent pairs or sequences.
    pub independent_units: usize,
    /// Number of individual outcome rows.
    pub outcome_rows: usize,
}

/// Execute the named experiment's existing joint score owner.
/// # Errors
/// Invalid source, unsupported assignment or exposure, numerical failure or cancellation.
#[allow(
    clippy::float_cmp,
    reason = "exact equality preserves discrete support or boundary semantics"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub fn joint_experiment(
    recipe: &JointExperimentalRecipe,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointExperimentalLaw, String> {
    let n = recipe.assignments.len();
    let k = recipe.outcomes.len();
    if n == 0
        || k == 0
        || [
            &recipe.population,
            &recipe.treated_regime_id,
            &recipe.control_regime_id,
            &recipe.consistency,
        ]
        .iter()
        .any(|x| x.trim().is_empty())
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
        return Err("invalid joint experimental source, outcome or premises".into());
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
    if ctx.cancellation.is_cancelled() {
        return Err("joint experiment cancelled".into());
    }
    let outcome_refs: Vec<_> = recipe.outcomes.iter().map(|o| o.values.as_slice()).collect();
    let width = k.checked_mul(2).ok_or("joint experimental width overflow")?;
    let law = match &recipe.design {
        ExperimentalDesign::MatchedPairs {
            cluster_ids,
            pairs,
            matching_variables,
            treatment_probabilities,
            cluster_sizes,
            preassignment,
            independent_pairs,
        } => {
            if cluster_ids.len() != n
                || pairs.len().checked_mul(2) != Some(n)
                || cluster_sizes.len() != n
                || cluster_sizes.contains(&0)
                || treatment_probabilities.len() != n
                || treatment_probabilities.iter().any(|p| *p != 0.5)
                || cluster_ids.iter().any(|v| v.trim().is_empty())
                || cluster_ids.iter().collect::<BTreeSet<_>>().len() != n
                || matching_variables.is_empty()
                || matching_variables.iter().any(|v| v.trim().is_empty())
                || preassignment.trim().is_empty()
                || independent_pairs.trim().is_empty()
            {
                return Err("invalid matched-pair source or assignment premises".into());
            }
            let rows = cluster_sizes
                .iter()
                .try_fold(0usize, |a, b| a.checked_add(*b))
                .ok_or("cluster size overflow")?;
            let fit = crate::randomized_joint::matched_pair_joint_contrast(
                &outcome_refs,
                &recipe.assignments,
                pairs,
                rows,
            )
            .ok_or("invalid matched-pair assignments or outcomes")?;
            let mut means = fit.treated_means;
            means.extend(fit.control_means);
            JointExperimentalLaw {
                means,
                covariance: fit.arm_covariance,
                independent_units: fit.pairs,
                outcome_rows: rows,
            }
        }
        ExperimentalDesign::Switchback {
            sequence_ids,
            period_ids,
            treatment_probabilities,
            no_carryover,
            washout_rule,
            independent_sequences,
        } => {
            if sequence_ids.len() != n
                || period_ids.len() != n
                || treatment_probabilities.len() != n
                || [no_carryover, washout_rule, independent_sequences]
                    .iter()
                    .any(|v| v.trim().is_empty())
            {
                return Err("invalid switchback schedule or temporal premises".into());
            }
            let mut counts = BTreeMap::<&str, usize>::new();
            let mut seen = BTreeSet::new();
            for ((sequence, period), p) in
                sequence_ids.iter().zip(period_ids).zip(treatment_probabilities)
            {
                if sequence.trim().is_empty()
                    || period.trim().is_empty()
                    || !p.is_finite()
                    || *p < 0.2
                    || *p > 0.8
                    || !seen.insert((sequence, period))
                {
                    return Err(
                        "invalid or repeated switchback sequence-period or probability".into()
                    );
                }
                *counts.entry(sequence).or_default() += 1;
            }
            if counts.len() < 30
                || counts.values().any(|c| *c < 2 || Some(c) != counts.values().next())
            {
                return Err(
                    "switchback requires at least 30 independent equal-length sequences".into()
                );
            }
            let ids: Vec<_> = sequence_ids.iter().map(String::as_str).collect();
            let mut channels = Vec::with_capacity(width);
            for treated in [true, false] {
                for y in &outcome_refs {
                    channels.push(
                        y.iter()
                            .zip(&recipe.assignments)
                            .map(
                                |(v, a)| {
                                    if *a == treated { if treated { *v } else { -*v } } else { 0.0 }
                                },
                            )
                            .collect::<Vec<_>>(),
                    );
                }
            }
            let fit = |y: &[f64]| {
                crate::switchback::switchback_itt(
                    y,
                    &recipe.assignments,
                    treatment_probabilities,
                    &ids,
                )
            };
            let mut means = Vec::with_capacity(width);
            let mut variances = Vec::with_capacity(width);
            for channel in &channels {
                if ctx.cancellation.is_cancelled() {
                    return Err("joint experiment cancelled".into());
                }
                let result = fit(channel).ok_or("switchback estimator refused the source")?;
                if result.interval_95.is_none() {
                    return Err("switchback sequence support is insufficient".into());
                }
                means.push(result.effect);
                variances.push(result.variance);
            }
            let mut covariance =
                vec![0.0; width.checked_mul(width).ok_or("joint covariance overflow")?];
            for i in 0..width {
                covariance[i * width + i] = variances[i];
                for j in 0..i {
                    if ctx.cancellation.is_cancelled() {
                        return Err("joint experiment cancelled".into());
                    }
                    let sum: Vec<_> =
                        channels[i].iter().zip(&channels[j]).map(|(a, b)| a + b).collect();
                    let v = fit(&sum).ok_or("switchback joint covariance unavailable")?.variance;
                    let c = (v - variances[i] - variances[j]) / 2.0;
                    covariance[i * width + j] = c;
                    covariance[j * width + i] = c;
                }
            }
            JointExperimentalLaw {
                means,
                covariance,
                independent_units: counts.len(),
                outcome_rows: n,
            }
        }
    };
    if ctx.cancellation.is_cancelled() {
        return Err("joint experiment cancelled".into());
    }
    if law.means.iter().chain(&law.covariance).any(|v| !v.is_finite()) {
        return Err("joint experimental arithmetic overflow".into());
    }
    Ok(law)
}
