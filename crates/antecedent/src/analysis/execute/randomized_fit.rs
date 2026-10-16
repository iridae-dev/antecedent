//! Per-design point and interval fits for randomized-trial effect queries.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Arc, CausalError, EstimatorId, TableView, TabularData};
use antecedent_core::{RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand};
use std::collections::BTreeMap;

/// Labeled action values of a multi-arm design: label, value, variance bound, support.
#[allow(
    clippy::type_complexity,
    reason = "labeled multi-arm value tuple mirrors the public result field type"
)]
pub(super) type MultiArmValues = Arc<[(Arc<str>, f64, f64, usize)]>;

type Interval = Option<[f64; 2]>;

/// The estimator branch a randomized query takes. Query validation makes
/// receipt adjustment, CUPED and ANCOVA mutually exclusive and Bernoulli-only,
/// so the branch is a function of the query alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RandomizedMethod {
    Ancova,
    TreatmentOnTreated,
    ComplierEffect,
    FixedCuped,
    Bernoulli,
    Complete,
    Cluster,
    Stratified,
    Factorial2x2,
    MultiArm,
    Switchback,
}

impl RandomizedMethod {
    pub(super) fn of(query: &RandomizedEffectQuery) -> Self {
        if !query.ancova_covariates.is_empty() {
            Self::Ancova
        } else if query.received_treatment.is_some() {
            if query.estimand == RandomizedEstimand::TreatmentOnTreated {
                Self::TreatmentOnTreated
            } else {
                Self::ComplierEffect
            }
        } else {
            match &query.design {
                RandomizationDesign::Bernoulli if query.fixed_cuped.is_some() => Self::FixedCuped,
                RandomizationDesign::Bernoulli => Self::Bernoulli,
                RandomizationDesign::Complete { .. } => Self::Complete,
                RandomizationDesign::Cluster { .. } => Self::Cluster,
                RandomizationDesign::Stratified { .. } => Self::Stratified,
                RandomizationDesign::Factorial2x2 { .. } => Self::Factorial2x2,
                RandomizationDesign::MultiArm { .. } => Self::MultiArm,
                RandomizationDesign::Switchback { .. } => Self::Switchback,
            }
        }
    }

    /// Assignment-design tag recorded on the result.
    pub(super) fn design_tag(self) -> &'static str {
        match self {
            Self::Ancova
            | Self::TreatmentOnTreated
            | Self::ComplierEffect
            | Self::FixedCuped
            | Self::Bernoulli => "bernoulli",
            Self::Complete => "complete",
            Self::Cluster => "cluster",
            Self::Stratified => "stratified",
            Self::Factorial2x2 => "factorial_2x2",
            Self::MultiArm => "multi_arm",
            Self::Switchback => "switchback",
        }
    }

    pub(super) fn estimator_id(self) -> EstimatorId {
        match self {
            Self::TreatmentOnTreated | Self::ComplierEffect => EstimatorId::RandomizedWaldCace,
            Self::FixedCuped => EstimatorId::RandomizedFixedCupedHt,
            Self::Ancova => EstimatorId::RandomizedAncova,
            Self::Bernoulli | Self::MultiArm => EstimatorId::RandomizedHt,
            Self::Complete | Self::Stratified | Self::Factorial2x2 | Self::Cluster => {
                EstimatorId::RandomizedNeyman
            }
            Self::Switchback => EstimatorId::RandomizedSwitchbackHt,
        }
    }

    pub(super) fn estimand_label(self) -> &'static str {
        match self {
            Self::TreatmentOnTreated => "treatment_on_treated",
            Self::ComplierEffect => "cace_late",
            Self::Factorial2x2 => "factorial_primary_main_effect",
            Self::MultiArm => "multi_arm_itt",
            _ => "itt",
        }
    }

    pub(super) fn diagnostic_id(self) -> &'static str {
        match self {
            Self::TreatmentOnTreated | Self::ComplierEffect => "estimate.randomized.wald_cace_late",
            Self::FixedCuped => "estimate.randomized.fixed_cuped_ht_itt",
            Self::Ancova => "estimate.randomized.ancova_itt",
            Self::Bernoulli => "estimate.randomized.ht_itt",
            Self::Switchback => "estimate.randomized.switchback_ht_itt",
            Self::MultiArm => "estimate.randomized.multi_arm_ht_itt",
            Self::Complete | Self::Cluster | Self::Stratified | Self::Factorial2x2 => {
                "estimate.randomized.neyman_itt"
            }
        }
    }

    /// Cluster-level dependence is what the published standard error accounts for.
    pub(super) fn clustered_standard_error(self) -> bool {
        matches!(self, Self::Cluster | Self::Switchback)
    }

    pub(super) fn uncertainty(self, interval_published: bool) -> &'static str {
        match (self, interval_published) {
            (Self::Ancova, false) => "bernoulli_ancova_hc0_variance_no_interval",
            (Self::TreatmentOnTreated, false) => {
                "bernoulli_one_sided_tot_influence_variance_no_interval"
            }
            (Self::ComplierEffect, false) => "bernoulli_wald_cace_influence_variance_no_interval",
            (Self::FixedCuped, false) => {
                "bernoulli_fixed_cuped_ht_conservative_variance_no_interval"
            }
            (Self::Bernoulli, false) => "bernoulli_ht_design_variance_no_interval",
            (Self::Complete, false) => "complete_neyman_variance_upper_bound_no_interval",
            (Self::Cluster, false) => "cluster_neyman_variance_upper_bound_no_interval",
            (Self::Stratified, false) => "stratified_neyman_variance_upper_bound_no_interval",
            (Self::Factorial2x2, false) => "factorial_cell_neyman_variance_upper_bound_no_interval",
            (Self::MultiArm, false) => "multi_arm_covariance_free_variance_bound_no_interval",
            (Self::Switchback, false) => {
                "switchback_independent_sequence_sandwich_variance_no_interval"
            }
            (Self::FixedCuped, true) => "bernoulli_fixed_cuped_ht_score_normal_interval",
            (Self::Ancova, true) => "bernoulli_ancova_hc0_normal_interval",
            (Self::TreatmentOnTreated, true) => "bernoulli_one_sided_tot_influence_normal_interval",
            (Self::ComplierEffect, true) => "bernoulli_wald_cace_influence_normal_interval",
            (Self::Bernoulli, true) => "bernoulli_ht_score_normal_interval",
            (Self::Complete, true) => "complete_neyman_normal_interval",
            (Self::Cluster, true) => "cluster_neyman_normal_interval",
            (Self::Stratified, true) => "stratified_neyman_normal_interval",
            (Self::Factorial2x2, true) => "factorial_cell_neyman_pointwise_normal_intervals",
            (Self::MultiArm, true) => "multi_arm_ht_score_pointwise_normal_intervals",
            (Self::Switchback, true) => "switchback_independent_sequence_student_interval",
        }
    }

    pub(super) fn diagnostic(self, interval_published: bool) -> &'static str {
        match (self, interval_published) {
            (Self::Ancova, false) => {
                "OLS ANCOVA with pre-assignment covariates and independent-row HC0 sandwich variance; no confidence interval: the HC0 interval needs 30 observed and 40 expected rows in each arm, a common assignment probability from 0.2 through 0.8 and at most two covariates"
            }
            (Self::TreatmentOnTreated, false) => {
                "Treatment-on-treated among recipients under one-sided noncompliance, exclusion, and independent assignment; Wald point and independent-unit influence variance, no interval"
            }
            (Self::ComplierEffect, false) => {
                "Wald CACE/LATE from randomized encouragement and observed receipt; independent-unit influence variance, no interval"
            }
            (Self::FixedCuped, false) => {
                "Horvitz-Thompson ITT on outcomes adjusted by a declared pre-assignment covariate and externally fixed coefficient; observed-arm upper-bound estimator is conservative in randomization expectation; no interval: the score interval needs 30 observed and 40 expected rows in each arm and assignment probabilities from 0.2 through 0.8"
            }
            (Self::Bernoulli, false) => {
                "Horvitz-Thompson ITT with Bernoulli design variance; no confidence interval: the score interval needs 30 observed and 40 expected rows in each arm and assignment probabilities from 0.2 through 0.8"
            }
            (Self::Complete, false) => {
                "Complete-randomization difference in means with Neyman conservative variance estimate; no confidence interval is reported"
            }
            (Self::Cluster, false) => {
                "Complete cluster-randomization HT ITT over cluster outcome totals, with Neyman conservative variance; no confidence interval is reported"
            }
            (Self::Stratified, false) => {
                "Stratified difference in means weighted by block size, with blockwise Neyman conservative variance estimate; no confidence interval is reported"
            }
            (Self::Factorial2x2, false) => {
                "Fixed-cell 2×2 factorial main effects and interaction from four cell means, with cellwise Neyman conservative variance estimates; no confidence interval is reported"
            }
            (Self::MultiArm, false) => {
                "Independent multi-arm assignment with known action probabilities; all contrasts versus reference retain covariance-free variance bounds, no calibrated interval"
            }
            (Self::Switchback, false) => {
                "Unit-period HT ITT with independent-sequence score sandwich variance; arbitrary within-sequence dependence, no interval"
            }
            (Self::FixedCuped, true) => {
                "Bernoulli ITT with an externally fixed pre-assignment CUPED coefficient and independent-unit score-sandwich pointwise 95% interval"
            }
            (Self::Ancova, true) => {
                "Bernoulli ANCOVA with pre-assignment covariates and independent-unit HC0 pointwise 95% interval"
            }
            (Self::TreatmentOnTreated, true) => {
                "Treatment-on-treated among recipients under one-sided noncompliance and exclusion; Wald ratio with an independent-unit influence-function pointwise 95% interval"
            }
            (Self::ComplierEffect, true) => {
                "Wald CACE/LATE from randomized encouragement and observed receipt under exclusion and monotonicity, with an independent-unit influence-function pointwise 95% interval"
            }
            (Self::Bernoulli, true) => {
                "Bernoulli Horvitz-Thompson ITT with known probabilities and an independent-unit score-sandwich pointwise 95% interval"
            }
            (Self::Complete, true) => {
                "Complete-randomization difference in means with a conservative Neyman variance and pointwise 95% normal interval"
            }
            (Self::Cluster, true) => {
                "Cluster-randomized ITT over cluster outcome totals with a conservative cluster-level Neyman variance and pointwise 95% normal interval"
            }
            (Self::Stratified, true) => {
                "Blocked difference in means with blockwise conservative Neyman variance and pointwise 95% normal interval"
            }
            (Self::Factorial2x2, true) => {
                "Fixed-cell factorial main effects and interaction with separate conservative cellwise variances and pointwise 95% normal intervals; simultaneous coverage is not claimed"
            }
            (Self::MultiArm, true) => {
                "Independent multi-arm HT effects with known probabilities; the retained variance is a covariance-free bound, while separate score-sandwich pointwise 95% intervals cover each action versus reference; simultaneous coverage is not claimed"
            }
            (Self::Switchback, true) => {
                "Unit-period HT ITT with independent-sequence sandwich variance and a pointwise 95% Student interval; no carryover and no between-sequence interference are declared assumptions"
            }
        }
    }
}

/// Secondary factorial contrasts: effect, interaction, and their variance bounds.
#[derive(Clone, Copy)]
pub(super) struct FactorialContrasts {
    pub(super) second_factor_effect: f64,
    pub(super) interaction: f64,
    pub(super) second_factor_variance: f64,
    pub(super) interaction_variance: f64,
}

/// Point estimate, variance and published intervals of one randomized fit.
pub(super) struct TrialFit {
    pub(super) effect: f64,
    pub(super) variance: f64,
    pub(super) control_units: usize,
    pub(super) treatment_units: usize,
    pub(super) blocks: Arc<[Arc<str>]>,
    pub(super) periods: Arc<[Arc<str>]>,
    pub(super) interval_95: Interval,
    pub(super) interval_standard_error: Option<f64>,
    /// Intention-to-treat effect and first stage of a receipt-adjusted fit.
    pub(super) complier_components: Option<(f64, f64)>,
    pub(super) factorial: Option<FactorialContrasts>,
    pub(super) second_factor_interval_95: Interval,
    pub(super) factorial_interaction_interval_95: Interval,
    pub(super) multi_arm_values: MultiArmValues,
    pub(super) multi_arm_intervals_95: Arc<[Interval]>,
    /// What the table's sampling weights did to the fit; `None` for an unweighted fit.
    pub(super) sampling: Option<antecedent_estimate::SamplingWeightRecord>,
}

impl TrialFit {
    fn point(effect: f64, variance: f64, control_units: usize, treatment_units: usize) -> Self {
        Self {
            effect,
            variance,
            control_units,
            treatment_units,
            blocks: Arc::from([]),
            periods: Arc::from([]),
            interval_95: None,
            interval_standard_error: None,
            complier_components: None,
            factorial: None,
            second_factor_interval_95: None,
            factorial_interaction_interval_95: None,
            multi_arm_values: Arc::from([]),
            multi_arm_intervals_95: Arc::from([]),
            sampling: None,
        }
    }

    /// The fit with every interval withheld: a handled missingness has no coverage record.
    pub(super) fn closed(mut self) -> Self {
        self.interval_95 = None;
        self.interval_standard_error = None;
        self.second_factor_interval_95 = None;
        self.factorial_interaction_interval_95 = None;
        self.multi_arm_intervals_95 = self.multi_arm_values.iter().map(|_| None).collect();
        self
    }

    /// Publishes `interval`; its standard error accompanies it only when it exists.
    fn with_interval(mut self, interval: Interval, standard_error: f64) -> Self {
        self.interval_95 = interval;
        if interval.is_some() {
            self.interval_standard_error = Some(standard_error);
        }
        self
    }
}

/// Fits the declared randomized design on (possibly CUPED-adjusted) outcomes.
pub(super) fn fit_randomized(
    query: &RandomizedEffectQuery,
    data: &TabularData,
    outcomes: &[f64],
) -> Result<TrialFit, CausalError> {
    if let Some(weights) = data.storage().weights() {
        return fit_sampling_weighted(query, outcomes, weights);
    }
    match RandomizedMethod::of(query) {
        RandomizedMethod::Ancova => fit_ancova(query, data, outcomes),
        RandomizedMethod::TreatmentOnTreated | RandomizedMethod::ComplierEffect => {
            fit_complier(query, outcomes)
        }
        RandomizedMethod::Bernoulli | RandomizedMethod::FixedCuped => {
            Ok(fit_bernoulli(query, outcomes))
        }
        RandomizedMethod::MultiArm => fit_multi_arm(query, outcomes),
        RandomizedMethod::Complete => fit_complete(query, outcomes),
        RandomizedMethod::Cluster => fit_cluster(query, outcomes),
        RandomizedMethod::Stratified => fit_stratified(query, outcomes),
        RandomizedMethod::Factorial2x2 => fit_factorial(query, outcomes),
        RandomizedMethod::Switchback => fit_switchback(query, outcomes),
    }
}

/// Fits the contrast of a Bernoulli or completely randomized design whose outcomes are
/// missing for some assigned participants, under the declared policy.
///
/// Complete case averages the responders of each arm (outcomes missing completely at random
/// within arm); inverse-probability weighting reweights the responders of each arm by their
/// estimated probability of being observed given the declared covariates. Neither narrows the
/// assignment to the responders, and neither publishes an interval.
pub(super) fn fit_missing_outcomes(
    query: &RandomizedEffectQuery,
    data: &TabularData,
    outcomes: &[f64],
    observed: &[bool],
    declaration: &antecedent_data::MissingDeclaration,
) -> Result<TrialFit, CausalError> {
    use antecedent_data::MissingPolicy;
    let assigned = |group: bool| query.realized_assignment.iter().filter(|a| **a == group).count();
    let (controls, treated) = (assigned(false), assigned(true));
    if outcomes.iter().zip(observed).any(|(y, seen)| *seen && !y.is_finite()) {
        return Err(CausalError::Unsupported {
            message: "randomized ITT requires finite observed outcomes",
        });
    }
    if !matches!(
        RandomizedMethod::of(query),
        RandomizedMethod::Bernoulli | RandomizedMethod::Complete
    ) {
        return fit_design_with_missing_outcomes(query, data, outcomes, observed, declaration);
    }
    match declaration.policy {
        MissingPolicy::CompleteCase => {
            let responders = |group: bool| -> Vec<f64> {
                (0..outcomes.len())
                    .filter(|&i| observed[i] && query.realized_assignment[i] == group)
                    .map(|i| outcomes[i])
                    .collect()
            };
            let fit = antecedent_estimate::randomized_neyman::complete_unit_itt(
                &responders(true),
                &responders(false),
            )
            .ok_or(CausalError::Unsupported {
                message: "complete-case randomized ITT needs at least two responders in each arm",
            })?;
            Ok(TrialFit::point(fit.effect, fit.variance_upper_bound, controls, treated))
        }
        MissingPolicy::InverseProbabilityOfObservation => {
            if let Some(cell) = antecedent_data::first_missing_cell(data, &declaration.covariates)
                .map_err(CausalError::from)?
            {
                return Err(CausalError::Estimate(
                    antecedent_estimate::missing_policy::covariates_incomplete(
                        &cell,
                        "randomized intention-to-treat estimation",
                    ),
                ));
            }
            if declaration.covariates.contains(&query.outcome) {
                return Err(CausalError::Unsupported {
                    message: "the observation model cannot condition on the outcome itself",
                });
            }
            let mut covariates = Vec::with_capacity(declaration.covariates.len() * outcomes.len());
            for &id in declaration.covariates.iter() {
                covariates.extend_from_slice(&data.float64_cow(id).map_err(CausalError::from)?);
            }
            let fit = antecedent_estimate::randomized_missing::responder_weighted_itt(
                outcomes,
                observed,
                &query.realized_assignment,
                &covariates,
                declaration.covariates.len(),
            )?;
            Ok(TrialFit::point(fit.effect, fit.variance, controls, treated))
        }
        _ => Err(CausalError::Unsupported {
            message: "randomized outcomes are missing and the declared policy does not handle them",
        }),
    }
}

/// Cluster, stratified, factorial, multi-arm and switchback designs with missing outcomes:
/// each outcome is replaced by its inverse-probability-of-observation pseudo-outcome (zero for
/// a non-responder), fitted within the unit's assignment group, and the design's own estimator
/// and variance run on every assigned unit. Assignment, blocks, clusters and periods are
/// untouched, and the design's variance (cluster totals, stratum weights, cell contrasts,
/// sequence scores) is applied to the pseudo-outcomes; it ignores that the observation
/// probabilities are estimated, which is conservative.
fn fit_design_with_missing_outcomes(
    query: &RandomizedEffectQuery,
    data: &TabularData,
    outcomes: &[f64],
    observed: &[bool],
    declaration: &antecedent_data::MissingDeclaration,
) -> Result<TrialFit, CausalError> {
    use antecedent_data::MissingPolicy;
    let group: Vec<usize> = match &query.design {
        RandomizationDesign::MultiArm { assignment, .. } => assignment.to_vec(),
        RandomizationDesign::Factorial2x2 { second_factor_assignment, .. } => query
            .realized_assignment
            .iter()
            .zip(second_factor_assignment.iter())
            .map(|(primary, second)| usize::from(*primary) + 2 * usize::from(*second))
            .collect(),
        _ => query.realized_assignment.iter().map(|a| usize::from(*a)).collect(),
    };
    let mut covariates = Vec::new();
    let mut ncov = 0;
    match declaration.policy {
        MissingPolicy::CompleteCase => {}
        MissingPolicy::InverseProbabilityOfObservation => {
            if let Some(cell) = antecedent_data::first_missing_cell(data, &declaration.covariates)
                .map_err(CausalError::from)?
            {
                return Err(CausalError::Estimate(
                    antecedent_estimate::missing_policy::covariates_incomplete(
                        &cell,
                        "randomized intention-to-treat estimation",
                    ),
                ));
            }
            for &id in declaration.covariates.iter() {
                covariates.extend_from_slice(&data.float64_cow(id).map_err(CausalError::from)?);
            }
            ncov = declaration.covariates.len();
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "randomized outcomes are missing and the declared policy does not handle them",
            });
        }
    }
    let pseudo = antecedent_estimate::randomized_missing::responder_pseudo_outcomes(
        outcomes,
        observed,
        &group,
        &covariates,
        ncov,
    )?;
    fit_randomized(query, data, &pseudo)
}

fn fit_ancova(
    query: &RandomizedEffectQuery,
    data: &TabularData,
    outcomes: &[f64],
) -> Result<TrialFit, CausalError> {
    let covariates = query
        .ancova_covariates
        .iter()
        .map(|&id| match data.column(id).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice().to_vec()),
            _ => Err(CausalError::Unsupported {
                message: "ANCOVA pre-assignment covariates must be continuous",
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let refs = covariates.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let fit = antecedent_estimate::ancova::fit_ancova(outcomes, &query.realized_assignment, &refs)
        .map_err(|message| CausalError::Unsupported { message })?;
    let interval = antecedent_estimate::ancova::calibrated_bernoulli_interval_95(
        &fit,
        query.assignment_probabilities[0],
    );
    Ok(TrialFit::point(fit.effect, fit.hc0_variance, fit.control, fit.treated)
        .with_interval(interval, fit.hc0_variance.sqrt()))
}

fn fit_complier(query: &RandomizedEffectQuery, outcomes: &[f64]) -> Result<TrialFit, CausalError> {
    let Some(received) = query.received_treatment.as_ref() else {
        return Err(CausalError::Unsupported {
            message: "receipt-adjusted randomized effects require the received-treatment column",
        });
    };
    let fit = antecedent_estimate::randomized_scores::complier_wald_effect(
        outcomes,
        &query.realized_assignment,
        received,
        &query.assignment_probabilities,
    )
    .ok_or(CausalError::Unsupported {
        message: "receipt-adjusted randomized effects require a finite positive receipt first stage",
    })?;
    let controls = query.realized_assignment.iter().filter(|assigned| !**assigned).count();
    let treated = outcomes.len() - controls;
    let mut out = TrialFit::point(fit.effect, fit.variance, controls, treated)
        .with_interval(fit.interval_95, fit.variance.sqrt());
    out.complier_components = Some((fit.intention_to_treat_effect, fit.first_stage));
    Ok(out)
}

fn fit_multi_arm(query: &RandomizedEffectQuery, outcomes: &[f64]) -> Result<TrialFit, CausalError> {
    let RandomizationDesign::MultiArm { assignment, probabilities, arms } = &query.design else {
        unreachable!("multi-arm method requires a multi-arm design")
    };
    let fit =
        antecedent_estimate::multi_arm::estimate_multi_arm(outcomes, assignment, probabilities)
            .map_err(|message| CausalError::Unsupported { message })?;
    let primary = fit[1].value - fit[0].value;
    let primary_variance = 2.0 * (fit[1].variance_bound + fit[0].variance_bound);
    let mut out = TrialFit::point(
        primary,
        primary_variance,
        fit[0].observed_support,
        fit[1].observed_support,
    );
    out.multi_arm_values = arms
        .iter()
        .zip(fit.iter())
        .map(|(label, arm)| {
            (Arc::clone(label), arm.value, arm.variance_bound, arm.observed_support)
        })
        .collect::<Vec<_>>()
        .into();
    let mut standard_error = None;
    let mut intervals = (0..arms.len())
        .map(|arm| {
            if arm == 0 {
                return None;
            }
            antecedent_estimate::randomized_scores::independent_action_contrast(
                outcomes,
                assignment,
                probabilities,
                0,
                arm,
            )
            .and_then(|contrast| {
                if arm == 1 && contrast.interval_95.is_some() {
                    standard_error = Some(contrast.variance_upper_bound.sqrt());
                }
                contrast.interval_95
            })
        })
        .collect::<Vec<_>>();
    if intervals.iter().skip(1).any(Option::is_none) {
        intervals.fill(None);
        standard_error = None;
    }
    out.interval_95 = intervals[1];
    out.interval_standard_error = standard_error;
    out.multi_arm_intervals_95 = intervals.into();
    Ok(out)
}

fn fit_bernoulli(query: &RandomizedEffectQuery, outcomes: &[f64]) -> TrialFit {
    let n = outcomes.len();
    let cuped = query.fixed_cuped.is_some();
    let mut effect_sum = 0.0;
    let mut variance_sum = 0.0;
    let mut control_units = 0;
    let mut treatment_units = 0;
    // With a fixed pre-assignment adjustment, E[observed y² / assignment²]
    // equals Y(1)²/p + Y(0)²/(1-p). This exceeds the Bernoulli
    // variance by (Y(1)-Y(0))² for each unit. No interval follows
    // from the random bound estimate without calibration.
    for ((&y, &treated), &p) in outcomes
        .iter()
        .zip(query.realized_assignment.iter())
        .zip(query.assignment_probabilities.iter())
    {
        if treated {
            treatment_units += 1;
            effect_sum += y / p;
            variance_sum += if cuped { y * y / (p * p) } else { y * y * (1.0 - p) / (p * p) };
        } else {
            control_units += 1;
            effect_sum -= y / (1.0 - p);
            variance_sum += if cuped {
                y * y / ((1.0 - p) * (1.0 - p))
            } else {
                y * y * p / ((1.0 - p) * (1.0 - p))
            };
        }
    }
    let assignments =
        query.realized_assignment.iter().map(|assigned| usize::from(*assigned)).collect::<Vec<_>>();
    let probabilities =
        query.assignment_probabilities.iter().map(|p| vec![1.0 - p, *p]).collect::<Vec<_>>();
    let mut out = TrialFit::point(
        effect_sum / n as f64,
        variance_sum / (n * n) as f64,
        control_units,
        treatment_units,
    );
    if let Some(contrast) = antecedent_estimate::randomized_scores::independent_action_contrast(
        outcomes,
        &assignments,
        &probabilities,
        0,
        1,
    ) {
        out = out.with_interval(contrast.interval_95, contrast.variance_upper_bound.sqrt());
    }
    out
}

fn fit_complete(query: &RandomizedEffectQuery, outcomes: &[f64]) -> Result<TrialFit, CausalError> {
    let RandomizationDesign::Complete { treated_units } = &query.design else {
        unreachable!("complete method requires a complete design")
    };
    let treated: Vec<f64> = outcomes
        .iter()
        .zip(query.realized_assignment.iter())
        .filter_map(|(&y, &z)| z.then_some(y))
        .collect();
    let control: Vec<f64> = outcomes
        .iter()
        .zip(query.realized_assignment.iter())
        .filter_map(|(&y, &z)| (!z).then_some(y))
        .collect();
    let fit = antecedent_estimate::randomized_neyman::complete_unit_itt(&treated, &control).ok_or(
        CausalError::Unsupported {
            message: "complete-randomized ITT requires finite outcomes and estimable variance",
        },
    )?;
    Ok(TrialFit::point(
        fit.effect,
        fit.variance_upper_bound,
        outcomes.len() - treated_units,
        *treated_units,
    )
    .with_interval(fit.interval_95, fit.variance_upper_bound.sqrt()))
}

fn fit_cluster(query: &RandomizedEffectQuery, outcomes: &[f64]) -> Result<TrialFit, CausalError> {
    let RandomizationDesign::Cluster { treated_clusters } = &query.design else {
        unreachable!("cluster method requires a cluster design")
    };
    let mut cluster_totals: BTreeMap<&str, (bool, f64)> = BTreeMap::new();
    for (i, outcome) in outcomes.iter().enumerate() {
        let entry = cluster_totals
            .entry(query.assignment_units[i].as_ref())
            .or_insert((query.realized_assignment[i], 0.0));
        entry.1 += outcome;
    }
    let treated: Vec<f64> = cluster_totals
        .values()
        .filter_map(|(assigned, total)| assigned.then_some(*total))
        .collect();
    let control: Vec<f64> = cluster_totals
        .values()
        .filter_map(|(assigned, total)| (!assigned).then_some(*total))
        .collect();
    let clusters = cluster_totals.len();
    let fit = antecedent_estimate::randomized_neyman::complete_cluster_itt(
        &treated,
        &control,
        outcomes.len(),
    )
    .ok_or(CausalError::Unsupported {
        message: "cluster-randomized ITT requires finite cluster totals and estimable variance",
    })?;
    Ok(TrialFit::point(
        fit.effect,
        fit.variance_upper_bound,
        clusters - treated_clusters,
        *treated_clusters,
    )
    .with_interval(fit.interval_95, fit.variance_upper_bound.sqrt()))
}

fn fit_stratified(
    query: &RandomizedEffectQuery,
    outcomes: &[f64],
) -> Result<TrialFit, CausalError> {
    let RandomizationDesign::Stratified { blocks, treated_per_row } = &query.design else {
        unreachable!("stratified method requires a stratified design")
    };
    let mut grouped: BTreeMap<&str, (Vec<f64>, Vec<f64>, usize)> = BTreeMap::new();
    for i in 0..outcomes.len() {
        let group = grouped
            .entry(&blocks[i])
            .or_insert_with(|| (Vec::new(), Vec::new(), treated_per_row[i]));
        if query.realized_assignment[i] {
            group.0.push(outcomes[i]);
        } else {
            group.1.push(outcomes[i]);
        }
    }
    let cells = grouped
        .values()
        .map(|(treated, control, _)| (treated.as_slice(), control.as_slice()))
        .collect::<Vec<_>>();
    let fit = antecedent_estimate::randomized_neyman::blocked_unit_itt(&cells).ok_or(
        CausalError::Unsupported {
            message: "blocked-randomized ITT requires finite outcomes and estimable within-block variances",
        },
    )?;
    let treatment_units = cells.iter().map(|(treated, _)| treated.len()).sum();
    let control_units = cells.iter().map(|(_, control)| control.len()).sum();
    let mut out =
        TrialFit::point(fit.effect, fit.variance_upper_bound, control_units, treatment_units)
            .with_interval(fit.interval_95, fit.variance_upper_bound.sqrt());
    out.blocks = Arc::clone(blocks);
    Ok(out)
}

fn fit_factorial(query: &RandomizedEffectQuery, outcomes: &[f64]) -> Result<TrialFit, CausalError> {
    let RandomizationDesign::Factorial2x2 { second_factor_assignment, .. } = &query.design else {
        unreachable!("factorial method requires a factorial design")
    };
    let mut cells: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for i in 0..outcomes.len() {
        let cell = usize::from(query.realized_assignment[i])
            + 2 * usize::from(second_factor_assignment[i]);
        cells[cell].push(outcomes[i]);
    }
    let fit =
        antecedent_estimate::randomized_neyman::factorial_2x2(cells.each_ref().map(Vec::as_slice))
            .ok_or(CausalError::Unsupported {
                message: "factorial ITT requires finite outcomes and estimable cell variances",
            })?;
    let mut out = TrialFit::point(
        fit.primary,
        fit.main_effect_variance_upper_bound,
        cells[0].len() + cells[2].len(),
        cells[1].len() + cells[3].len(),
    )
    .with_interval(fit.primary_interval_95, fit.main_effect_variance_upper_bound.sqrt());
    out.second_factor_interval_95 = fit.secondary_interval_95;
    out.factorial_interaction_interval_95 = fit.interaction_interval_95;
    out.factorial = Some(FactorialContrasts {
        second_factor_effect: fit.secondary,
        interaction: fit.interaction,
        second_factor_variance: fit.main_effect_variance_upper_bound,
        interaction_variance: fit.interaction_variance_upper_bound,
    });
    Ok(out)
}

fn fit_switchback(
    query: &RandomizedEffectQuery,
    outcomes: &[f64],
) -> Result<TrialFit, CausalError> {
    let RandomizationDesign::Switchback { periods } = &query.design else {
        unreachable!("switchback method requires a switchback design")
    };
    let sequence_ids = query.assignment_units.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
    let fit = antecedent_estimate::switchback::switchback_itt(
        outcomes,
        &query.realized_assignment,
        &query.assignment_probabilities,
        &sequence_ids,
    )
    .ok_or(CausalError::Unsupported {
        message: "switchback ITT requires valid outcomes, probabilities, and independent sequences with global arm support",
    })?;
    let mut out = TrialFit::point(
        fit.effect,
        fit.variance,
        query.realized_assignment.iter().filter(|&&a| !a).count(),
        query.realized_assignment.iter().filter(|&&a| a).count(),
    )
    .with_interval(fit.interval_95, fit.variance.sqrt());
    out.periods = Arc::clone(periods);
    Ok(out)
}

/// Exhaustive two-sided Fisher sharp-null p-value and allocation count.
pub(super) fn exact_randomization_test(
    query: &RandomizedEffectQuery,
    outcomes: &[f64],
    effect: f64,
) -> (f64, u64) {
    let RandomizationDesign::Complete { treated_units } = &query.design else {
        unreachable!("validated exact test requires complete randomization")
    };
    let n = outcomes.len();
    let observed = effect.abs();
    let mut extreme = 0_u64;
    let mut allocations = 0_u64;
    let total = outcomes.iter().sum::<f64>();
    for mask in 0..(1_u64 << n) {
        if mask.count_ones() as usize != *treated_units {
            continue;
        }
        let treated_sum = outcomes
            .iter()
            .enumerate()
            .filter_map(|(i, value)| ((mask >> i) & 1 == 1).then_some(*value))
            .sum::<f64>();
        let contrast = treated_sum / *treated_units as f64
            - (total - treated_sum) / (n - *treated_units) as f64;
        allocations += 1;
        if contrast.abs() + 1e-12 >= observed {
            extreme += 1;
        }
    }
    (extreme as f64 / allocations as f64, allocations)
}

/// Smallest assignment probability; `two_sided` folds each Bernoulli
/// probability to `min(p, 1 - p)`.
pub(super) fn assignment_probability_floor(query: &RandomizedEffectQuery, two_sided: bool) -> f64 {
    if let RandomizationDesign::MultiArm { probabilities, .. } = &query.design {
        probabilities.iter().flat_map(|row| row.iter().copied()).fold(f64::INFINITY, f64::min)
    } else {
        query
            .assignment_probabilities
            .iter()
            .copied()
            .map(|p| if two_sided { p.min(1.0 - p) } else { p })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Smaller of the expected treated and expected control rows of an independent Bernoulli design:
/// the arm's summed known assignment probabilities. Zero for every other design, whose licenses
/// state their own arm counts.
pub(super) fn expected_arm_units(query: &RandomizedEffectQuery) -> f64 {
    if !matches!(query.design, RandomizationDesign::Bernoulli) {
        return 0.0;
    }
    let treated: f64 = query.assignment_probabilities.iter().sum();
    treated.min(query.assignment_probabilities.len() as f64 - treated)
}

/// Uncertainty label of a sampling-weighted two-arm fit; no interval is published.
pub(super) const SAMPLING_WEIGHTED_UNCERTAINTY: &str =
    "sampling_weighted_hajek_linearized_variance_no_interval";

/// Diagnostic text of a sampling-weighted two-arm fit.
pub(super) const SAMPLING_WEIGHTED_DIAGNOSTIC: &str = "Sampling-weighted ITT: the difference of the arms' Hajek means in the weighted population the table's sampling weights represent, each row weighted by its sampling weight over the known probability of its assigned arm and weights normalized to mean one; the variance linearizes the two ratio means treating units as independent draws (Neyman's conservative bound for equal weights); rows of weight zero are excluded and counted; no confidence interval is reported";

/// Fits a Bernoulli or completely randomized two-arm design under the table's sampling weights.
///
/// The estimand is the intention-to-treat effect in the weighted population: the difference of
/// the arms' Hajek means with row weights `s_i / P(arm_i)`. Rows of weight zero are excluded
/// and counted. Designs with receipt adjustment, CUPED, ANCOVA, exact randomization inference,
/// clusters, blocks, factors, extra arms or sequences have no sampling-weighted contrast.
fn fit_sampling_weighted(
    query: &RandomizedEffectQuery,
    outcomes: &[f64],
    weights: &[f64],
) -> Result<TrialFit, CausalError> {
    if !matches!(
        RandomizedMethod::of(query),
        RandomizedMethod::Bernoulli | RandomizedMethod::Complete
    ) || query.exact_randomization_test
    {
        return Err(CausalError::Estimate(
            antecedent_estimate::sampling_weights::unsupported_route(
                "sampling weights are read by unadjusted Bernoulli and complete randomization of units; \
             cluster, stratified, factorial, multi-arm, switchback, receipt-adjusted, CUPED, ANCOVA \
             and exact-randomization designs have no sampling-weighted contrast",
            ),
        ));
    }
    let arm_probability: Vec<f64> = query
        .realized_assignment
        .iter()
        .zip(query.assignment_probabilities.iter())
        .map(|(&treated, &p)| if treated { p } else { 1.0 - p })
        .collect();
    let fit = antecedent_estimate::randomized_neyman::weighted_itt(
        outcomes,
        &query.realized_assignment,
        &arm_probability,
        weights,
    )
    .ok_or(CausalError::Unsupported {
        message: "sampling-weighted randomized ITT needs finite outcomes and at least two positive-weight units in each arm",
    })?;
    let rows_used = fit.treated_units + fit.control_units;
    let mut out = TrialFit::point(fit.effect, fit.variance, fit.control_units, fit.treated_units);
    out.sampling = Some(antecedent_estimate::SamplingWeightRecord {
        rows_used: rows_used as u64,
        zero_weight_rows: fit.zero_weight_rows as u64,
        effective_sample_size: fit.effective_sample_size,
        weight_sum: weights.iter().sum(),
    });
    Ok(out)
}
