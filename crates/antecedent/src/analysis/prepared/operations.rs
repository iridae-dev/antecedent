//! Sealed checked operations retained by a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::accepted::GraphClass;
use crate::analysis::builder::RefuteSuite;
use crate::analysis::execute::CheckedStaticCommon;
use crate::analysis::helpers::{
    AssembleArgs, RefutationRun, assemble_result, overlap_diagnostic, overlap_limit_diagnostic,
    project_for_ate_estimate, provenance_pair, stamp_result_context,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::proof_supplies;
use crate::analysis::stage::{
    STAGE_ESTIMATE_POINT, STAGE_IDENTIFY, STAGE_VALIDATE, StageClock, elapsed_ns,
};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, ResponseQuery, TargetPopulation,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    AipwAte, BootstrapAccounting, EffectEstimate, EstimationWorkspace, OverlapPolicy,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;
use std::sync::Arc;
use std::time::Instant;

/// Retained functional-distribution target and program for prepared execution.
#[derive(Clone, Debug)]
pub(crate) struct CheckedDistributionOperation {
    pub(super) query: InterventionalDistributionQuery,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) fitter: antecedent_estimate::FunctionalDistribution,
    /// Producing immutable storage contents (values, validity, mask and weights).
    /// Callers independently check the frozen semantic schema before reuse.
    pub(super) factor_snapshot_digest: [u8; 32],
    pub(super) prepared: antecedent_estimate::PreparedFunctionalDistribution,
    pub(super) common: Arc<CheckedStaticCommon>,
}

/// Retained checked general-ID functional effect for finite-discrete ADMG ATEs.
/// The checked program and data binding are prepared together; estimate clicks
/// rebind that program to compatible data without consulting the original Study.
#[derive(Clone, Debug)]
pub(crate) struct CheckedFunctionalEffectOperation {
    pub(super) query: AverageEffectQuery,
    /// The conditional effect this operation answers when the target is the IDC contrast of a
    /// conditional effect; `query` is then its inner average-effect query.
    pub(super) conditional: Option<antecedent_core::ConditionalEffectQuery>,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) fitter: antecedent_estimate::FunctionalEffect,
    /// Producing immutable storage contents (values, validity, mask and weights).
    /// Callers independently check the frozen semantic schema before reuse.
    pub(super) factor_snapshot_digest: [u8; 32],
    pub(super) prepared: antecedent_estimate::PreparedFunctionalEffect,
    pub(super) common: Arc<CheckedStaticCommon>,
}

/// Checked path-specific natural effect retained with its graph target and program.
#[derive(Clone, Debug)]
pub(crate) struct CheckedPathSpecificEffectOperation {
    pub(super) query: antecedent_core::PathSpecificEffectQuery,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) fitter: antecedent_estimate::FunctionalEffect,
    /// Producing immutable storage contents (values, validity, mask and weights).
    /// Callers independently check the frozen semantic schema before reuse.
    pub(super) factor_snapshot_digest: [u8; 32],
    pub(super) prepared: antecedent_estimate::PreparedFunctionalEffect,
    pub(super) common: Arc<CheckedStaticCommon>,
}

impl CheckedPathSpecificEffectOperation {
    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        matches!(
            self.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) && self.identifier == crate::strategy_table::IdentifierId::PathSpecificNatural
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let prepared = self.prepared.rebind_checked(data).map_err(CausalError::from)?;
        Ok(Self {
            factor_snapshot_digest: data.storage().content_digest(),
            query: self.query.clone(),
            cache: Arc::clone(&self.cache),
            fitter: self.fitter.clone(),
            prepared,
            common: Arc::clone(&self.common),
        })
    }

    pub(super) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.cache.estimand.method_kind().ok()
            != Some(antecedent_expr::EstimandMethod::PathSpecificNatural)
            || self.prepared.estimand.functional != self.cache.estimand.functional
            || self.prepared.program().mapping().source != self.cache.estimand.functional
            || self.query.treatment == self.query.outcome
        {
            return Err(CausalError::Compile {
                message: "prepared path-specific operation disagrees with its checked target"
                    .into(),
            });
        }
        let (estimate, posterior) = crate::analysis::execute::run_functional(
            &self.fitter,
            &self.prepared,
            &self.inference,
            self.cache.identification.status,
            ctx,
        )?;
        let mut diagnostics = self.cache.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.push(crate::analysis::execute::identify_cached_diagnostic());
        if let Some(diagnostic) = crate::analysis::execute::free_variables_averaged(
            &self.cache.identification,
            &self.cache.estimand,
        ) {
            diagnostics.push(diagnostic);
        }
        if let Some(posterior) = posterior.as_ref() {
            diagnostics.extend(crate::analysis::execute::posterior_note_diagnostics([posterior]));
        }
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_path(
                data,
                &self.query,
                &self.cache.identification,
                &self.cache.estimand,
                estimate.ate,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let result = crate::analysis::execute::CheckedResultFinish::new(
            &self.common,
            &self.cache.identification,
            &self.cache.estimand,
            estimate,
            started,
            (self.query.treatment, self.query.outcome),
            (CausalQuery::PathSpecific(self.query.clone()), GraphClass::Dag),
        )
        .with_diagnostics(diagnostics)
        .with_refutations(refutations)
        .with_posterior(posterior)
        .with_bootstrap(Some(self.fitter.bootstrap_replicates), None)
        .finish();
        Ok(result)
    }
}

impl CheckedFunctionalEffectOperation {
    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        matches!(self.graph_class, GraphClass::Admg | GraphClass::TemporalAdmg)
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.identifier == crate::strategy_table::IdentifierId::GeneralId
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && (self.conditional.is_none() || self.refute == RefuteSuite::None)
            && self.custom_validator_names.is_empty()
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let prepared = self.prepared.rebind_checked(data).map_err(CausalError::from)?;
        Ok(Self {
            factor_snapshot_digest: data.storage().content_digest(),
            query: self.query.clone(),
            conditional: self.conditional.clone(),
            cache: Arc::clone(&self.cache),
            fitter: self.fitter.clone(),
            prepared,
            common: Arc::clone(&self.common),
        })
    }

    pub(super) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.cache.estimand.method_kind().ok()
            != Some(antecedent_expr::EstimandMethod::GeneralId)
            || self.prepared.estimand.functional != self.cache.estimand.functional
            || self.prepared.estimand.adjustment_set != self.cache.estimand.adjustment_set
            || self.prepared.program().mapping().source != self.cache.estimand.functional
        {
            return Err(CausalError::Compile {
                message: "prepared functional-effect operation disagrees with its checked target"
                    .into(),
            });
        }
        let (estimate, posterior) = crate::analysis::execute::run_functional(
            &self.fitter,
            &self.prepared,
            &self.inference,
            self.cache.identification.status,
            ctx,
        )?;
        let mut diagnostics = self.cache.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.push(crate::analysis::execute::identify_cached_diagnostic());
        if let Some(diagnostic) = crate::analysis::execute::free_variables_averaged(
            &self.cache.identification,
            &self.cache.estimand,
        ) {
            diagnostics.push(diagnostic);
        }
        if let Some(posterior) = posterior.as_ref() {
            diagnostics.extend(crate::analysis::execute::posterior_note_diagnostics([posterior]));
        }
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra) = if self.refute == RefuteSuite::None {
            (Vec::new(), Vec::new())
        } else {
            RefutationRun {
                data,
                estimand: &self.cache.estimand,
                query: &self.query,
                estimate: &estimate,
                suite: self.refute,
                estimator: self.estimator.as_str(),
                custom: &[],
                temporal: None,
            }
            .reports(&mut refute_ws, None, ctx)?
        };
        diagnostics.extend(extra);
        let certificate_query = self.conditional.as_ref().map_or_else(
            || CausalQuery::AverageEffect(self.query.clone()),
            |conditional| CausalQuery::ConditionalEffect(conditional.clone()),
        );
        let result = crate::analysis::execute::CheckedResultFinish::new(
            &self.common,
            &self.cache.identification,
            &self.cache.estimand,
            estimate,
            started,
            (self.query.treatment, self.query.outcome),
            (certificate_query, self.graph_class),
        )
        .with_diagnostics(diagnostics)
        .with_refutations(refutations)
        .with_posterior(posterior)
        .with_bootstrap(Some(self.fitter.bootstrap_replicates), None)
        .finish();
        Ok(result)
    }
}

/// One ordered, checked intervention point in an ADMG response curve.
#[derive(Clone, Debug)]
pub struct CheckedFunctionalEffectResponseMember {
    pub(super) grid_value: f64,
    pub(super) query: antecedent_core::ResponseQuery,
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) prepared: antecedent_estimate::PreparedFunctionalEffect,
}

impl CheckedFunctionalEffectResponseMember {
    /// Intervention value at this member's original grid position.
    #[must_use]
    pub const fn grid_value(&self) -> f64 {
        self.grid_value
    }
    /// Typed scalar intervention query retained for this member.
    #[must_use]
    pub fn query(&self) -> &antecedent_core::ResponseQuery {
        &self.query
    }
    /// General-ID proof retained for this member.
    #[must_use]
    pub const fn identification(&self) -> &IdentificationResult {
        &self.identification
    }
    /// Identified expression target retained for this member.
    #[must_use]
    pub const fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }
    /// Checked functional program bound to this member.
    #[must_use]
    pub fn program(&self) -> &antecedent_expr::FunctionalProgram {
        self.prepared.program()
    }
}

/// Retained checked program family for an ADMG response curve or scalar
/// intervention response. A scalar response is represented as a one-member
/// family so it shares the same checked identification and provider binding.
#[derive(Clone, Debug)]
pub(crate) struct CheckedAdmgResponseCurveOperation {
    pub(super) factor_snapshot_digest: [u8; 32],
    pub(super) query: antecedent_core::ResponseQuery,
    pub(super) grid: Arc<[f64]>,
    pub(super) members: Arc<[CheckedFunctionalEffectResponseMember]>,
    pub(super) fitter: antecedent_estimate::FunctionalEffect,
    pub(super) common: Arc<CheckedStaticCommon>,
}

/// Per-member results of evaluating an ADMG response grid.
struct MemberEvaluations {
    means: Vec<f64>,
    point_status: Vec<antecedent_core::SupportStatus>,
    member_posteriors: Vec<antecedent_estimate::CausalPosterior>,
    support_status: antecedent_core::SupportStatus,
    support_warnings: Vec<antecedent_core::Diagnostic>,
    freq_bounds: Vec<(f64, f64)>,
    freq_scalar_se: f64,
    freq_replicates_ok: u32,
}

impl CheckedAdmgResponseCurveOperation {
    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        matches!(
            self.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) && self.identifier == crate::strategy_table::IdentifierId::GeneralId
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
            && self.members.len() == self.grid.len()
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let members = self
            .members
            .iter()
            .map(|member| {
                let prepared = member.prepared.rebind_checked(data).map_err(CausalError::from)?;
                Ok(CheckedFunctionalEffectResponseMember {
                    grid_value: member.grid_value,
                    query: member.query.clone(),
                    identification: member.identification.clone(),
                    estimand: member.estimand.clone(),
                    prepared,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?;
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            // Member calls use the same seeded Bayesian-bootstrap stream. Require
            // every member to use every source row so its stream positions name
            // identical rows and therefore one shared row-weight draw.
            for member in &members {
                let snapshot = member.prepared.factor_snapshot().map_err(CausalError::from)?;
                if snapshot.provenance.source_rows != data.row_count()
                    || snapshot.provenance.complete_case_rows != data.row_count()
                {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian ADMG response curves require a common complete-case row set across grid members",
                    });
                }
            }
        }
        Ok(Self {
            factor_snapshot_digest: data.storage().content_digest(),
            query: self.query.clone(),
            grid: Arc::clone(&self.grid),
            members: Arc::from(members),
            fitter: self.fitter.clone(),
            common: Arc::clone(&self.common),
        })
    }

    pub(super) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.members.is_empty() || self.members.len() != self.grid.len() {
            return Err(CausalError::Compile {
                message: "retained ADMG response grid/member mapping is incomplete".into(),
            });
        }
        let MemberEvaluations {
            means,
            point_status,
            member_posteriors,
            support_status,
            support_warnings,
            freq_bounds,
            freq_scalar_se,
            freq_replicates_ok,
        } = self.evaluate_members(ctx)?;
        let n_draws = member_posteriors
            .first()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let posterior = self.joint_posterior(member_posteriors)?;
        let first = self.members.first().ok_or(CausalError::Unsupported {
            message: "a general-identification response needs at least one member",
        })?;
        let mut support = self.response_support(support_status, point_status, &support_warnings)?;
        let scalar_intervention = matches!(
            self.query.functional,
            antecedent_core::ResponseFunctional::InterventionResponse { .. }
        );
        let scalar = if scalar_intervention { means[0] } else { f64::NAN };
        let freq_uncertainty = self.frequentist_uncertainty(
            &mut support,
            &freq_bounds,
            freq_scalar_se,
            freq_replicates_ok,
            means.len(),
            scalar_intervention,
        );
        let frequentist_bootstrap = matches!(self.inference, InferenceMode::Frequentist)
            && self.fitter.bootstrap_replicates > 0;
        let bootstrap_requested =
            if frequentist_bootstrap { self.fitter.bootstrap_replicates } else { 0 };
        let bootstrap_ok = frequentist_bootstrap.then_some(freq_replicates_ok);
        let response = antecedent_core::CausalResponse {
            estimand: self.query.functional.clone(),
            identification_status: first.identification.status,
            estimate: antecedent_core::ResponseIdentification::PointIdentified(
                self.response_value(scalar_intervention, scalar, means),
            ),
            uncertainty: posterior
                .as_ref()
                .and_then(|posterior| credible_response_uncertainty(posterior, scalar_intervention))
                .or(freq_uncertainty)
                .unwrap_or(antecedent_core::ResponseUncertainty::None),
            support,
            assumptions: first.identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        let mut diagnostics = general_id_diagnostics(first, support_warnings);
        let (treatment, outcome) =
            first.query.functional.primary_pair().ok_or(CausalError::Unsupported {
                message: "the response functional names no treatment and outcome pair",
            })?;
        let refutations =
            if scalar_intervention && scalar.is_finite() && self.refute != RefuteSuite::None {
                let (reports, refute_diags) =
                    self.plugin_level_refutations(data, first, scalar, treatment, outcome, ctx)?;
                diagnostics.extend(refute_diags);
                reports
            } else {
                Vec::new()
            };
        let estimate = EffectEstimate::new(
            f64::NAN,
            f64::NAN,
            first.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let result = crate::analysis::execute::CheckedResultFinish::new(
            &self.common,
            &first.identification,
            &first.estimand,
            estimate,
            started,
            (treatment, outcome),
            (CausalQuery::Response(self.query.clone()), GraphClass::Admg),
        )
        .with_estimate_assumptions(first.prepared.assumptions.clone())
        .with_diagnostics(diagnostics)
        .with_refutations(refutations)
        .with_posterior(posterior)
        .with_n_draws(n_draws)
        .with_bootstrap(Some(bootstrap_requested), bootstrap_ok)
        .with_response(response)
        .without_evalue_mirror()
        .with_rebound_interval()
        .finish();
        Ok(result)
    }

    /// Point value of the response: the single plug-in level, or the mean surface over the grid.
    fn response_value(
        &self,
        scalar_intervention: bool,
        scalar: f64,
        means: Vec<f64>,
    ) -> antecedent_core::ResponseValue {
        if scalar_intervention {
            antecedent_core::ResponseValue::Scalar(scalar)
        } else {
            antecedent_core::ResponseValue::Surface {
                grid: Arc::clone(&self.grid),
                dimension: 1,
                mean: Arc::from(means),
            }
        }
    }

    /// Evaluates every grid member under the configured inference mode, in grid order,
    /// retaining per-member posteriors and the frequentist bootstrap bounds.
    fn evaluate_members(&self, ctx: &ExecutionContext) -> Result<MemberEvaluations, CausalError> {
        let mut means = Vec::with_capacity(self.members.len());
        let mut point_status = Vec::with_capacity(self.members.len());
        let mut member_posteriors = Vec::new();
        let mut support_status = antecedent_core::SupportStatus::Supported;
        let mut support_warnings = Vec::new();
        // Frequentist general-ID responses publish a normal interval from the
        // per-member front-door plug-in bootstrap SE (the sibling AverageEffect /
        // InterventionalDistribution ADMG cells use the same resample). One
        // `(lower, upper)` per member; the scalar case keeps its single SE.
        let mut freq_bounds: Vec<(f64, f64)> = Vec::new();
        let mut freq_scalar_se = f64::NAN;
        let mut freq_replicates_ok: u32 = 0;
        for member in self.members.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
            }
            let value = match &self.inference {
                InferenceMode::Frequentist => {
                    match member.prepared.evaluate(&member.prepared.provider) {
                        Ok(value) => value,
                        Err(error) if antecedent_estimate::functional_cell_unevaluable(&error) => {
                            support_status =
                                antecedent_core::SupportStatus::OutsideEmpiricalSupport;
                            support_warnings.push(antecedent_core::Diagnostic::new(
                                "functional.required_cell",
                                antecedent_core::DiagnosticKind::Scientific,
                                antecedent_core::DiagnosticSeverity::Warning,
                                error.to_string(),
                            ));
                            f64::NAN
                        }
                        Err(error) => {
                            return Err(CausalError::from(
                                antecedent_estimate::EstimationError::data_msg(error.to_string()),
                            ));
                        }
                    }
                }
                InferenceMode::Bayesian(_) => {
                    let n_draws =
                        crate::analysis::execute::functional_bayesian_draws(&self.inference)?
                            .ok_or(CausalError::Unsupported {
                                message: "Bayesian inference selects no draw count",
                            })?;
                    let fit = self
                        .fitter
                        .estimate_bayesian(
                            &member.prepared,
                            n_draws,
                            member.identification.status,
                            ctx,
                        )
                        .map_err(CausalError::from)?;
                    let mean = crate::analysis::execute::effect_from_posterior(&fit)?.ate;
                    // `estimate_bayesian` opens the same named RNG stream for each
                    // member. Its Bayesian-bootstrap row weights therefore align
                    // draw-for-draw across the fixed grid; retain every member
                    // column below so the resulting posterior keeps that joint law.
                    member_posteriors.push(fit);
                    mean
                }
            };
            point_status.push(if value.is_finite() {
                antecedent_core::SupportStatus::Supported
            } else {
                antecedent_core::SupportStatus::OutsideEmpiricalSupport
            });
            means.push(value);
            if matches!(self.inference, InferenceMode::Frequentist)
                && value.is_finite()
                && self.fitter.bootstrap_replicates > 0
            {
                let mut ws = antecedent_estimate::FunctionalDistributionWorkspace::default();
                if let Ok(effect) = self.fitter.estimate(&member.prepared, &mut ws, ctx) {
                    if let Some(se) = effect.se_bootstrap() {
                        if se.is_finite() && se > 0.0 {
                            let z = crate::result::reported_se_interval_z();
                            freq_bounds.push((value - z * se, value + z * se));
                            freq_scalar_se = se;
                            freq_replicates_ok =
                                effect.uncertainty.bootstrap.replicates_ok.unwrap_or(0);
                        }
                    }
                }
            }
        }
        Ok(MemberEvaluations {
            means,
            point_status,
            member_posteriors,
            support_status,
            support_warnings,
            freq_bounds,
            freq_scalar_se,
            freq_replicates_ok,
        })
    }

    /// Stacks the per-member posterior effect columns into one joint posterior over the
    /// grid, reusing the first member's posterior as the carrier.
    fn joint_posterior(
        &self,
        mut member_posteriors: Vec<antecedent_estimate::CausalPosterior>,
    ) -> Result<Option<antecedent_estimate::CausalPosterior>, CausalError> {
        if member_posteriors.is_empty() {
            Ok(None)
        } else {
            let n = member_posteriors[0].draws.n_draws;
            let mut columns = Vec::with_capacity(member_posteriors.len());
            for member_posterior in &member_posteriors {
                if member_posterior.draws.n_draws != n {
                    return Err(CausalError::Compile {
                        message:
                            "response curve members did not retain a common posterior draw count"
                                .into(),
                    });
                }
                let quantity =
                    member_posterior.effect_column().ok_or_else(|| CausalError::Compile {
                        message: "response curve member posterior omitted its effect quantity"
                            .into(),
                    })?;
                columns.push(member_posterior.draws.column(quantity)?.to_vec());
            }
            let schema = antecedent_prob::PosteriorSchema {
                quantities: Arc::from(
                    self.grid
                        .iter()
                        .map(|value| antecedent_prob::PosteriorQuantityKind::Scalar {
                            name: Arc::from(format!("response_at_{value}")),
                        })
                        .collect::<Vec<_>>(),
                ),
            };
            let draws = antecedent_prob::PosteriorDraws::from_column_major(
                schema,
                n,
                columns.into_iter().flatten().collect::<Vec<_>>(),
            )?;
            let summaries = draws.summarize();
            let mut joint = member_posteriors.remove(0);
            joint.draws = draws;
            joint.summaries = summaries;
            Ok(Some(joint))
        }
    }

    /// Support report for the grid: the worst member status, per-point status, the grid
    /// range as the query region, and the warnings raised while evaluating members.
    fn response_support(
        &self,
        support_status: antecedent_core::SupportStatus,
        point_status: Vec<antecedent_core::SupportStatus>,
        support_warnings: &[antecedent_core::Diagnostic],
    ) -> Result<antecedent_core::SupportReport, CausalError> {
        let mut support =
            antecedent_estimate::support_from_functional_eval(None).map_err(|error| {
                CausalError::from(antecedent_estimate::EstimationError::data_msg(error.to_string()))
            })?;
        support.status = support_status;
        support.point_status = Some(Arc::from(point_status));
        support.query_region = antecedent_core::SupportRegion {
            minima: Arc::from([self.grid.iter().copied().fold(f64::INFINITY, f64::min)]),
            maxima: Arc::from([self.grid.iter().copied().fold(f64::NEG_INFINITY, f64::max)]),
        };
        support.warnings.extend(support_warnings.iter().cloned());
        Ok(support)
    }

    /// Normal interval from the per-member bootstrap SE, published only when every
    /// member returned one; records the bootstrap-SE support marker when it does.
    fn frequentist_uncertainty(
        &self,
        support: &mut antecedent_core::SupportReport,
        freq_bounds: &[(f64, f64)],
        freq_scalar_se: f64,
        freq_replicates_ok: u32,
        member_count: usize,
        scalar_intervention: bool,
    ) -> Option<antecedent_core::ResponseUncertainty> {
        // Publish the frequentist bootstrap interval only when every member
        // returned a positive-finite SE; otherwise fail open to `None`. The
        // support marker makes the facade report `bootstrap_se` for this band.
        if matches!(self.inference, InferenceMode::Frequentist)
            && !freq_bounds.is_empty()
            && freq_bounds.len() == member_count
        {
            support.diagnostics.push(antecedent_core::SupportDiagnostic {
                id: Arc::from(crate::result::RESPONSE_BOOTSTRAP_SE),
                values: Arc::from([
                    f64::from(self.fitter.bootstrap_replicates),
                    f64::from(freq_replicates_ok),
                ]),
                detail: Arc::from(
                    "general-ID functional.effect response interval is a normal interval from the \
                     front-door plug-in bootstrap SE [requested replicates, successful replicates]",
                ),
                scope: antecedent_core::DiagnosticScope::Global,
            });
            if scalar_intervention {
                let (lower, upper) = freq_bounds[0];
                Some(antecedent_core::ResponseUncertainty::Scalar {
                    standard_error: freq_scalar_se,
                    lower,
                    upper,
                    level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                })
            } else {
                Some(antecedent_core::ResponseUncertainty::PointwiseBand {
                    level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                    lower: Arc::from(freq_bounds.iter().map(|(l, _)| *l).collect::<Vec<_>>()),
                    upper: Arc::from(freq_bounds.iter().map(|(_, u)| *u).collect::<Vec<_>>()),
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                })
            }
        } else {
            None
        }
    }

    /// Cheap/full refutations of the plugin intervention level for the scalar response,
    /// with the diagnostics they raise.
    fn plugin_level_refutations(
        &self,
        data: &TabularData,
        first: &CheckedFunctionalEffectResponseMember,
        scalar: f64,
        treatment: antecedent_core::VariableId,
        outcome: antecedent_core::VariableId,
        ctx: &ExecutionContext,
    ) -> Result<
        (Vec<antecedent_validate::RefutationReport>, Vec<antecedent_core::Diagnostic>),
        CausalError,
    > {
        let mut diagnostics = Vec::new();
        if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "refute.evalue.not_a_contrast",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "contrast-shaped refuters are not licensed for a plugin intervention level; \
                 cheap runs overlap only and full runs overlap plus sampling-stability of the \
                 identified intervention mean",
            ));
        }
        let plugin_estimate = EffectEstimate::new(
            scalar,
            f64::NAN,
            first.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut refute_ws = EstimationWorkspace::default();
        let (reports, refute_diags) = RefutationRun {
            data,
            estimand: &first.estimand,
            query: &AverageEffectQuery::binary_ate(treatment, outcome),
            estimate: &plugin_estimate,
            suite: self.refute,
            estimator: self.estimator.as_str(),
            custom: &[],
            temporal: None,
        }
        .plugin_level(&mut refute_ws, ctx)?;
        diagnostics.extend(refute_diags);
        Ok((reports, diagnostics))
    }
}

/// Identification diagnostics of the first member followed by the grid's support warnings.
fn general_id_diagnostics(
    first: &CheckedFunctionalEffectResponseMember,
    support_warnings: Vec<antecedent_core::Diagnostic>,
) -> Vec<antecedent_core::Diagnostic> {
    let mut diagnostics = first.identification.diagnostics.clone();
    diagnostics.push(antecedent_core::Diagnostic::new(
        "identify.response.general_id",
        antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Info,
        "intervention mean identified by general ID; grid members are evaluated from their retained checked programs",
    ));
    diagnostics.extend(support_warnings);
    diagnostics
}

/// Credible-interval uncertainty of a response posterior, scalar or pointwise by response shape.
fn credible_response_uncertainty(
    posterior: &antecedent_estimate::CausalPosterior,
    scalar_intervention: bool,
) -> Option<antecedent_core::ResponseUncertainty> {
    if scalar_intervention {
        crate::analysis::helpers::credible_scalar_uncertainty(posterior)
    } else {
        crate::analysis::helpers::credible_pointwise_uncertainty(posterior)
    }
}

/// Retained checked family plan for a static DAG, complete observation mean curve.
/// The grid is copied into the plan so execution can detect query tampering before
/// handing the frozen estimand to the response estimator.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticResponseCurve {
    pub(super) query: ResponseQuery,
    pub(super) grid: Arc<[f64]>,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) identifier: crate::strategy_table::IdentifierId,
    pub(super) estimator: crate::strategy_table::EstimatorId,
}

impl CheckedStaticResponseCurve {
    pub(crate) fn checked_route(
        &self,
        query: &ResponseQuery,
    ) -> Result<(&IdentificationResult, &IdentifiedEstimand), CausalError> {
        let values = match &query.functional {
            antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } => {
                treatment.grid.values()?
            }
            _ => Vec::new(),
        };
        if query != &self.query || values.as_slice() != self.grid.as_ref() {
            return Err(CausalError::Compile {
                message: "prepared response curve query or grid changed; re-prepare the study"
                    .into(),
            });
        }
        Ok((&self.cache.identification, &self.cache.estimand))
    }

    pub(crate) const fn procedure(
        &self,
    ) -> (crate::strategy_table::IdentifierId, crate::strategy_table::EstimatorId) {
        (self.identifier, self.estimator)
    }
}

/// Checked binding for the prepared static Bayesian mean ATE g-computation row.
/// It keeps the original query, selected identification claim and inference
/// configuration together so estimate clicks cannot silently select another row.
#[derive(Clone, Debug)]
pub struct CheckedBayesianGcompOperation {
    pub(crate) query: AverageEffectQuery,
    pub(crate) cache: Arc<super::CachedStaticIdentification>,
    pub(crate) inference: InferenceMode,
}

impl CheckedBayesianGcompOperation {
    /// Query retained by the checked Bayesian g-computation route.
    #[must_use]
    pub fn query(&self) -> &AverageEffectQuery {
        &self.query
    }

    /// Selected identification claim retained for this query.
    #[must_use]
    pub fn identification(&self) -> &IdentificationResult {
        &self.cache.identification
    }

    /// Selected causal target retained for this query.
    #[must_use]
    pub fn estimand(&self) -> &IdentifiedEstimand {
        &self.cache.estimand
    }

    /// Bayesian prior and inference configuration bound at prepare time.
    #[must_use]
    pub fn inference(&self) -> &InferenceMode {
        &self.inference
    }
}

/// Selected checked linear front-door procedure and its bound numerical receipt.
/// Keeping the fitter beside the receipt freezes uncertainty choices as well as
/// the causal target across prepared clicks and refresh.
#[derive(Clone, Debug)]
pub(crate) struct CheckedFrontDoorOperation {
    pub(crate) fitter: antecedent_estimate::FrontDoorTwoStage,
    pub(crate) preparation: antecedent_estimate::CheckedFrontDoorPreparation,
}

impl CheckedFrontDoorOperation {
    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self.fitter.rebind_checked(&self.preparation, data)?;
        Ok(Self { fitter: self.fitter.clone(), preparation })
    }
}

/// Selected AIPW procedure and its checked, row-bound numerical receipt.
/// The fitter is retained so estimate clicks cannot recover its nuisance or
/// uncertainty choices from a copied `Study`.
#[derive(Clone, Debug)]
pub(crate) struct CheckedAipwOperation {
    pub(crate) fitter: AipwAte,
    pub(crate) preparation: antecedent_estimate::CheckedAipwPreparation,
    pub(super) query: AverageEffectQuery,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) common: Arc<CheckedStaticCommon>,
    /// Tier closure used to justify adjustment when this is a `CoDetermined` route.
    pub(super) tiered_background: Option<antecedent_graph::TieredBackground>,
}

impl CheckedAipwOperation {
    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        ((self.graph_class == GraphClass::Dag && self.tiered_background.is_none())
            || (self.graph_class == GraphClass::Admg
                && self.tiered_background.as_ref().is_some_and(|background| {
                    background.within_tier == antecedent_graph::WithinTier::CoDetermined
                })))
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::Aipw
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(
                self.refute,
                RefuteSuite::None
                    | RefuteSuite::Cheap
                    | RefuteSuite::PlaceboAndRcc
                    | RefuteSuite::Full
            )
            && self.custom_validator_names.is_empty()
            && matches!(self.fitter.overlap.canonical(), OverlapPolicy::RequireDiagnostics { .. })
            && self.preparation.lowering().overlap == self.fitter.overlap
            && self.preparation.lowering().procedure
                == antecedent_estimate::CheckedAipwProcedure::for_overlap(self.fitter.overlap)
            && self.preparation.lowering().population == TargetPopulation::AllObserved
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self.fitter.rebind_checked(&self.preparation, data)?;
        Ok(Self {
            fitter: self.fitter.clone(),
            preparation,
            query: self.query.clone(),
            cache: Arc::clone(&self.cache),
            common: Arc::clone(&self.common),
            tiered_background: self.tiered_background.clone(),
        })
    }

    pub(super) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_observed(data, ctx, None)
    }

    /// Execute the retained procedure with optional progressive boundaries, using the same
    /// final diagnostics, certificate and provenance as an unobserved checked execution.
    pub(super) fn execute_observed(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
        sink: Option<&Arc<dyn crate::analysis::stage::StageResultSink>>,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
        }
        self.check_retained_target()?;
        let mut clock = StageClock::new();
        if sink.is_some() {
            clock.begin(ctx, STAGE_IDENTIFY, 0.05)?;
            clock.finish(STAGE_IDENTIFY);
            crate::analysis::stage::emit_stage(
                sink,
                &crate::analysis::stage::StageEvent::Identify {
                    identification: self.cache.identification.clone(),
                    estimand: self.cache.estimand.clone(),
                },
            );
            clock.begin(ctx, STAGE_ESTIMATE_POINT, 0.25)?;
        }
        let mut workspace = antecedent_estimate::AipwWorkspace::default();
        let estimate = if sink.is_some() {
            let point = self.fitter.fit_checked_point(&self.preparation, &mut workspace, ctx)?;
            clock.finish(STAGE_ESTIMATE_POINT);
            crate::analysis::stage::emit_stage(
                sink,
                &crate::analysis::stage::StageEvent::Point { estimate: point.clone() },
            );
            clock.begin(ctx, crate::analysis::stage::STAGE_UNCERTAINTY, 0.55)?;
            let filled = self.fitter.attach_bootstrap(
                self.preparation.problem(),
                &mut workspace,
                ctx,
                point,
            )?;
            clock.finish(crate::analysis::stage::STAGE_UNCERTAINTY);
            crate::analysis::stage::emit_uncertainty(sink, &filled);
            filled
        } else {
            self.fitter.fit_checked(&self.preparation, &mut workspace, ctx)?
        };
        let cancelled = estimate.uncertainty.bootstrap.cancelled || ctx.cancellation.is_cancelled();
        if sink.is_some() && !cancelled {
            clock.begin(ctx, STAGE_VALIDATE, 0.8)?;
        }
        let (refutations, refute_diagnostics) = if cancelled || self.refute == RefuteSuite::None {
            (Vec::new(), Vec::new())
        } else {
            let mut estimate_workspace = EstimationWorkspace::default();
            let (reports, diagnostics) = RefutationRun {
                data,
                estimand: &self.cache.estimand,
                query: &self.query,
                estimate: &estimate,
                suite: self.refute,
                estimator: self.estimator.as_str(),
                custom: &[],
                temporal: None,
            }
            .reports(&mut estimate_workspace, Some(&mut workspace.propensity), ctx)?;
            (reports, diagnostics)
        };
        if sink.is_some() && !cancelled {
            clock.finish(STAGE_VALIDATE);
            crate::analysis::stage::emit_stage(
                sink,
                &crate::analysis::stage::StageEvent::Validate {
                    refutations: refutations.clone(),
                    predictive_checks: Vec::new(),
                },
            );
        }
        let diagnostics = self.fit_diagnostics(&estimate, refute_diagnostics);
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (
                identify_artifact,
                identify_operation,
                &[],
                &self.cache.identification.required_assumptions,
            ),
            (
                estimate_artifact,
                estimate_operation,
                &[identify_artifact],
                &estimate.provenance.assumptions,
            ),
        );
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.cache.identification.clone(),
            estimand: self.cache.estimand.clone(),
            estimate,
            posterior: None,
            payload: crate::result::ResultPayload::Scalar,
            refutations,
            diagnostics,
            provenance,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            wall_time_ns: elapsed_ns(started),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: if sink.is_some() { clock.timings() } else { Vec::new() },
            bootstrap_replicates_requested: Some(self.fitter.inference.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
            bayesian: false,
        });
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.cache.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::AverageEffect(self.query.clone()),
            graph_class: self.graph_class,
        });
        stamp_result_context(
            &mut result,
            self.support_status,
            self.structure_source,
            self.population_registry.as_ref(),
            &self.custom_validator_names,
        );
        crate::analysis::helpers::mirror_refuted_evalue(
            &mut result.estimate,
            &result.view.refutations,
        );
        crate::analysis::helpers::attach_tiered_evalue_for_cell(
            &mut result.estimate,
            data,
            self.query.outcome,
            self.query.outcome_functional.quantile_level().is_some(),
            self.tiered_background.as_ref(),
            &mut result.view.diagnostics,
        );
        Ok(result)
    }

    /// Refuses execution when the retained fit or tiered closure no longer matches the
    /// checked target and query.
    #[allow(
        clippy::float_cmp,
        reason = "treatment arms are exact category labels, not computed values"
    )]
    fn check_retained_target(&self) -> Result<(), CausalError> {
        let lowering = self.preparation.lowering();
        let active = match &self.query.active {
            antecedent_core::Intervention::Set { variable, value }
                if *variable == self.query.treatment =>
            {
                value.as_f64()
            }
            _ => None,
        };
        let control = match &self.query.control {
            antecedent_core::Intervention::Set { variable, value }
                if *variable == self.query.treatment =>
            {
                value.as_f64()
            }
            _ => None,
        };
        if self.preparation.target().functional != self.cache.estimand.functional
            || self.preparation.target().adjustment_set != self.cache.estimand.adjustment_set
            || lowering.treatment != self.query.treatment
            || lowering.outcome != self.query.outcome
            || active.is_none()
            || control.is_none()
            || active == control
        {
            return Err(CausalError::Compile {
                message: "prepared AIPW operation disagrees with its checked target or query"
                    .into(),
            });
        }
        if let Some(background) = &self.tiered_background {
            let checked = antecedent_identify::identify_tiered(background, &self.query)?;
            if checked.status != self.cache.identification.status
                || checked.query != self.cache.identification.query
                || !proof_supplies(&checked, &self.cache.estimand)
            {
                return Err(CausalError::Compile {
                    message: "retained CoDetermined AIPW closure no longer justifies its target"
                        .into(),
                });
            }
        }
        Ok(())
    }

    /// Diagnostics for a fitted AIPW estimate, in the order the result reports them.
    fn fit_diagnostics(
        &self,
        estimate: &EffectEstimate,
        refute_diagnostics: Vec<antecedent_core::Diagnostic>,
    ) -> Vec<antecedent_core::Diagnostic> {
        let mut diagnostics = self.cache.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.push(crate::analysis::execute::identify_cached_diagnostic());
        diagnostics.push(if estimate.scores.table.is_some() {
            antecedent_core::Diagnostic::new(
                "estimate.aipw.crossfit_scores",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "cross-fitted AIPW scores φᵢ^a; retarget averages this table. A residualized full-sample AIPW fit is a different object",
            )
        } else {
            antecedent_core::Diagnostic::new(
                "estimate.aipw.full_sample_residualized",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "this AIPW fit is full-sample residualized and has no score table; it is not the cross-fitted φ family that retarget averages. Prepare an AllObserved iid AIPW plan to retarget",
            )
        });
        diagnostics.extend(refute_diagnostics);
        if let Some((tier, configured)) = self.latency_bootstrap_not_applied {
            diagnostics.push(crate::analysis::helpers::latency_bootstrap_not_applied_diagnostic(
                tier, configured,
            ));
        }
        diagnostics
    }
}

// A cache entry is published only after the fit completes. Poison recovery therefore
// retains only the previous complete value; a queued caller can cancel independently.
fn lock_frozen_fit<'a, T>(
    slot: &'a std::sync::Mutex<T>,
    ctx: &ExecutionContext,
) -> Result<std::sync::MutexGuard<'a, T>, CausalError> {
    lock_frozen_fit_with_wait(slot, ctx, || {})
}

fn lock_frozen_fit_with_wait<'a, T>(
    slot: &'a std::sync::Mutex<T>,
    ctx: &ExecutionContext,
    mut on_wait: impl FnMut(),
) -> Result<std::sync::MutexGuard<'a, T>, CausalError> {
    loop {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
        }
        match slot.try_lock() {
            Ok(guard) => {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
                }
                return Ok(guard);
            }
            Err(std::sync::TryLockError::Poisoned(error)) => {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
                }
                return Ok(error.into_inner());
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                on_wait();
                std::thread::park_timeout(std::time::Duration::from_millis(5));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FrozenLinearFit {
    kernel: antecedent_core::KernelPolicy,
    estimate: EffectEstimate,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}

/// Checked linear adjustment target with its selected fit and uncertainty
/// procedure. `default_id` preserves the progressive point/uncertainty stages.
#[derive(Clone)]
pub(crate) struct CheckedLinearOperation {
    pub(crate) frozen_fit: Arc<std::sync::Mutex<Option<FrozenLinearFit>>>,
    pub(crate) compiled: Arc<crate::analysis::compiled_linear::CompiledLinearProgram>,
    /// Immutable table binding of the retained checked design. Equal bindings
    /// need no new projection, missing-data pass, or design construction.
    pub(super) source: TabularData,
    pub(crate) fitter: antecedent_estimate::LinearAdjustmentAte,
    pub(crate) preparation: antecedent_estimate::CheckedLinearAdjustmentAte,
    pub(crate) default_id: bool,
    pub(super) query: AverageEffectQuery,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) stage_sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>,
    pub(super) common: Arc<CheckedStaticCommon>,
}

impl std::fmt::Debug for CheckedLinearOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedLinearOperation")
            .field("query", &self.query)
            .field("estimator", &self.estimator)
            .field("default_id", &self.default_id)
            .field("latency_mode", &self.latency_mode)
            .field("validation", &self.refute)
            .finish_non_exhaustive()
    }
}

/// Immutable result-construction context retained with a checked static operation.
pub(super) struct CheckedStaticResultMetadata<'a> {
    pub(super) source_query: &'a CausalQuery,
    pub(super) query: &'a AverageEffectQuery,
    pub(super) identification: &'a IdentificationResult,
    pub(super) estimand: &'a IdentifiedEstimand,
    pub(super) identifier: crate::strategy_table::IdentifierId,
    pub(super) estimator: crate::strategy_table::EstimatorId,
    pub(super) physical: &'a PhysicalExecutionPlan,
    pub(super) graph_class: GraphClass,
    pub(super) graph_version: u32,
    pub(super) support_status: Option<crate::support::CellStatus>,
    pub(super) structure_source: crate::support::StructureSource,
    pub(super) population_registry: Option<&'a antecedent_core::PopulationRegistry>,
    pub(super) latency_mode: Option<crate::analysis::latency::LatencyMode>,
    /// `(tier replicates, configured replicates)` when a latency tier's count
    /// was set aside for the configured estimator's own count.
    pub(super) latency_bootstrap_not_applied: Option<(u32, u32)>,
    pub(super) refute: RefuteSuite,
    pub(super) custom_validator_names: &'a [Arc<str>],
}

pub(super) fn static_result_metadata<'a>(
    common: &'a CheckedStaticCommon,
    source_query: &'a CausalQuery,
    query: &'a AverageEffectQuery,
    identification: &'a IdentificationResult,
    estimand: &'a IdentifiedEstimand,
) -> CheckedStaticResultMetadata<'a> {
    CheckedStaticResultMetadata {
        source_query,
        query,
        identification,
        estimand,
        identifier: common.identifier,
        estimator: common.estimator,
        physical: &common.physical,
        graph_class: common.graph_class,
        graph_version: common.graph_version,
        support_status: common.support_status,
        structure_source: common.structure_source,
        population_registry: common.population_registry.as_ref(),
        latency_mode: common.latency_mode,
        latency_bootstrap_not_applied: common.latency_bootstrap_not_applied,
        refute: common.refute,
        custom_validator_names: common.custom_validator_names.as_ref(),
    }
}

/// Retained GLM adjustment target, selected configuration, and compiled design.
#[derive(Clone, Debug)]
pub(crate) struct CheckedGlmAdjustmentOperation {
    pub(super) source_query: CausalQuery,
    pub(super) query: AverageEffectQuery,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) fitter: antecedent_estimate::GlmAdjustmentAte,
    pub(super) preparation: antecedent_estimate::PreparedGlmProblem,
    pub(super) common: Arc<CheckedStaticCommon>,
}

impl CheckedGlmAdjustmentOperation {
    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        self.graph_class == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::GlmAdjustment
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(
                self.refute,
                RefuteSuite::None
                    | RefuteSuite::Cheap
                    | RefuteSuite::PlaceboAndRcc
                    | RefuteSuite::Full
            )
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self
            .fitter
            .prepare(data, &self.cache.estimand, &self.query)
            .map_err(CausalError::from)?;
        Ok(Self {
            source_query: self.source_query.clone(),
            query: self.query.clone(),
            cache: Arc::clone(&self.cache),
            fitter: self.fitter.clone(),
            preparation,
            common: Arc::clone(&self.common),
        })
    }
}

/// Retained sharp-RD target, design geometry, and local-linear fit.
#[derive(Clone, Debug)]
pub(crate) struct CheckedRdOperation {
    pub(super) source_query: AverageEffectQuery,
    pub(super) query: AverageEffectQuery,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) fitter: antecedent_estimate::SharpRegressionDiscontinuity,
    pub(super) preparation: antecedent_estimate::CheckedRdPreparation,
    pub(super) common: Arc<CheckedStaticCommon>,
}

macro_rules! deref_checked_static {
    ($($ty:ty),+ $(,)?) => {$(
        impl std::ops::Deref for $ty {
            type Target = CheckedStaticCommon;
            fn deref(&self) -> &Self::Target {
                &self.common
            }
        }
    )+};
}

deref_checked_static!(
    CheckedDistributionOperation,
    CheckedFunctionalEffectOperation,
    CheckedPathSpecificEffectOperation,
    CheckedAdmgResponseCurveOperation,
    CheckedAipwOperation,
    CheckedLinearOperation,
    CheckedGlmAdjustmentOperation,
    CheckedRdOperation,
);

impl CheckedRdOperation {
    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self
            .fitter
            .prepare_checked(data, &self.cache.identification, 0)
            .map_err(CausalError::from)?;
        Ok(Self {
            source_query: self.source_query.clone(),
            query: self.query.clone(),
            cache: Arc::clone(&self.cache),
            fitter: self.fitter.clone(),
            preparation,
            common: Arc::clone(&self.common),
        })
    }
}

impl CheckedLinearOperation {
    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        self.graph_class == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::LinearAdjustmentAte
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let owner = self.physical.linear.as_ref().ok_or_else(|| CausalError::Compile {
            message: "compiled linear owner missing".into(),
        })?;
        self.rebind_with_context(data, &owner.resource_context.context())
    }

    pub(super) fn rebind_with_context(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        if self.source.schema() == data.schema()
            && self.source.storage().content_digest() == data.storage().content_digest()
        {
            self.check_checked_target()?;
            return Ok(self.clone());
        }
        let owner = self.physical.linear.as_ref().ok_or_else(|| CausalError::Compile {
            message: "compiled linear owner missing".into(),
        })?;
        let mut physical = self.physical.as_ref().clone();
        physical.linear = None;
        let product = Arc::new(owner.rebind(data, &self.query, &mut physical, ctx)?);
        physical.linear = Some(Arc::clone(&product));
        let mut common = self.common.as_ref().clone();
        common.physical = Arc::new(physical);
        Ok(Self {
            frozen_fit: Arc::new(std::sync::Mutex::new(None)),
            compiled: Arc::clone(&product.program),
            source: data.clone(),
            fitter: product.fitter.clone(),
            preparation: product.preparation.clone(),
            default_id: product.default_id,
            query: self.query.clone(),
            cache: Arc::clone(&self.cache),
            stage_sink: self.stage_sink.clone(),
            common: Arc::new(common),
        })
    }

    pub(super) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = StageClock::new();
        clock.begin(ctx, STAGE_IDENTIFY, 0.05)?;
        self.check_checked_target()?;
        clock.finish(STAGE_IDENTIFY);
        crate::analysis::stage::emit_stage(
            self.stage_sink.as_ref(),
            &crate::analysis::stage::StageEvent::Identify {
                identification: self.cache.identification.clone(),
                estimand: self.cache.estimand.clone(),
            },
        );

        let mut workspace = EstimationWorkspace::default();
        let (admission, output_bytes) = self.admit_workspace(ctx)?;
        let envelope = Arc::new(
            admission
                .acquire_child(
                    0,
                    crate::analysis::compiled_linear::resources::RESULT_ENVELOPE_BYTES,
                )
                .map_err(|error| CausalError::Resource { message: error.to_string() })?,
        );
        // A deterministic point fit is reusable across downstream utility/world scoring.
        // Stochastic uncertainty still executes under each caller's requested policy.
        let (estimate, output_retention) =
            self.fit_retained(&mut clock, &mut workspace, &admission, output_bytes, ctx)?;
        let cancelled = estimate.uncertainty.bootstrap.cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut extra_diagnostics) =
            self.validate_stage(data, &estimate, cancelled, &mut clock, &mut workspace, ctx)?;
        let mut diagnostics = self.cache.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.push(crate::analysis::execute::identify_cached_diagnostic());
        diagnostics.append(&mut extra_diagnostics);
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (
                identify_artifact,
                identify_operation,
                &[],
                &self.cache.identification.required_assumptions,
            ),
            (
                estimate_artifact,
                estimate_operation,
                &[identify_artifact],
                &estimate.provenance.assumptions,
            ),
        );
        let bootstrap = estimate.uncertainty.bootstrap;
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.cache.identification.clone(),
            estimand: self.cache.estimand.clone(),
            estimate,
            posterior: None,
            payload: crate::result::ResultPayload::Scalar,
            refutations,
            diagnostics,
            provenance,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            wall_time_ns: clock.wall_time_ns(),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: clock.timings(),
            bootstrap_replicates_requested: Some(self.fitter.inference.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap.replicates_ok,
            n_draws: None,
            cancelled,
            early_stopped: bootstrap.early_stopped,
            bayesian: false,
        });
        result.retained_resources = vec![output_retention, envelope];
        Self::record_output_retention(&result);
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.cache.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::AverageEffect(self.query.clone()),
            graph_class: self.graph_class,
        });
        stamp_result_context(
            &mut result,
            self.support_status,
            self.structure_source,
            self.population_registry.as_ref(),
            &self.custom_validator_names,
        );
        crate::analysis::helpers::mirror_refuted_evalue(
            &mut result.estimate,
            &result.view.refutations,
        );
        Ok(result)
    }

    fn admit_workspace(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<antecedent_core::execution::AdmissionLease>, u64), CausalError> {
        let geometry = crate::analysis::compiled_linear::resources::LinearResources::prepared(
            &self.preparation,
            &self.fitter,
            ctx,
        )?;
        geometry.check(ctx)?;
        let owner = self.physical.linear.as_ref().ok_or_else(|| CausalError::Compile {
            message: "compiled linear owner missing".into(),
        })?;
        let pool = ctx.admission();
        let mut bytes = if owner.retention.belongs_to(&pool) {
            owner.retention.observe_current();
            geometry.execution_bytes()?
        } else {
            geometry.peak()?
        };
        let cache = lock_frozen_fit(&self.frozen_fit, ctx)?;
        if let Some(fit) = cache.as_ref().filter(|fit| fit.retention.belongs_to(&pool)) {
            fit.retention.observe_current();
            if fit.kernel == ctx.kernel_policy {
                bytes = bytes.checked_sub(fit.retention.remainder().bytes()).ok_or_else(|| {
                    CausalError::Resource {
                        message: "cached linear output exceeds its allocation quote".into(),
                    }
                })?;
            }
        }
        drop(cache);
        let admission = Arc::new(
            ctx.acquire_admission(0, bytes)
                .map_err(|error| CausalError::Resource { message: error.to_string() })?,
        );
        Ok((
            admission,
            geometry.output - crate::analysis::compiled_linear::resources::RESULT_ENVELOPE_BYTES,
        ))
    }

    fn fit_retained(
        &self,
        clock: &mut StageClock,
        workspace: &mut EstimationWorkspace,
        admission: &Arc<antecedent_core::execution::AdmissionLease>,
        output_bytes: u64,
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Arc<antecedent_core::execution::AdmissionLease>), CausalError>
    {
        let retain = || {
            admission
                .acquire_child(0, output_bytes)
                .map(Arc::new)
                .map_err(|error| CausalError::Resource { message: error.to_string() })
        };
        let estimate = if self.fitter.inference.bootstrap_replicates == 0 {
            let mut frozen = lock_frozen_fit(&self.frozen_fit, ctx)?;
            if let Some(FrozenLinearFit { estimate, retention, .. }) =
                frozen.as_ref().filter(|fit| fit.kernel == ctx.kernel_policy)
            {
                antecedent_core::execution_attempt::record_physical(
                    antecedent_core::execution_attempt::PhysicalEvent::CacheHit,
                    1,
                );
                antecedent_core::execution_attempt::record_physical(
                    antecedent_core::execution_attempt::PhysicalEvent::ModelFit,
                    0,
                );
                (
                    estimate.clone(),
                    if retention.belongs_to(&ctx.admission()) {
                        Arc::clone(retention)
                    } else {
                        retain()?
                    },
                )
            } else {
                let retention = retain()?;
                let estimate = antecedent_core::numerical_admission::with_reservation(
                    Arc::clone(admission),
                    || self.fit_with_stages(clock, workspace, ctx),
                )?;
                if !ctx.cancellation.is_cancelled() {
                    *frozen = Some(FrozenLinearFit {
                        kernel: ctx.kernel_policy,
                        estimate: estimate.clone(),
                        retention: Arc::clone(&retention),
                    });
                }
                (estimate, retention)
            }
        } else {
            let retention = retain()?;
            let estimate = antecedent_core::numerical_admission::with_reservation(
                Arc::clone(admission),
                || self.fit_with_stages(clock, workspace, ctx),
            )?;
            (estimate, retention)
        };
        Ok(estimate)
    }

    fn record_output_retention(result: &StudyResult) {
        use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
        if let Some(bytes) = result.retained_admitted_bytes() {
            record_physical(PhysicalEvent::RetainedFull, 1);
            record_physical(PhysicalEvent::RetainedBytes, bytes);
        }
    }

    /// Refuses execution when the retained fit no longer matches the checked target.
    fn check_checked_target(&self) -> Result<(), CausalError> {
        if self.compiled.source_digest() != self.source.storage().content_digest()
            || self.compiled.identification().assumptions()
                != self.preparation.required_assumptions()
            || self.compiled.target()
                != &crate::planner::linear_gaussian_scientific_problem(
                    &self.query,
                    false,
                    self.source.schema(),
                )?
            || self.preparation.source_functional() != self.cache.estimand.functional
            || self.preparation.target().adjustment_set != self.cache.estimand.adjustment_set
            || self.preparation.lowering().treatment != self.query.treatment
            || self.preparation.lowering().outcome != self.query.outcome
            || self.preparation.lowering().population != self.query.target_population
        {
            return Err(CausalError::Compile {
                message: "prepared linear adjustment operation disagrees with its checked target"
                    .into(),
            });
        }
        Ok(())
    }

    /// Point fit then uncertainty attachment, each as its own timed and emitted stage.
    fn fit_with_stages(
        &self,
        clock: &mut StageClock,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, CausalError> {
        clock.begin(ctx, STAGE_ESTIMATE_POINT, 0.25)?;
        let point = if self.default_id {
            self.fitter
                .fit_point(
                    self.preparation.problem(),
                    workspace,
                    self.preparation.required_assumptions().clone(),
                )
                .map_err(CausalError::from)?
        } else {
            self.fitter.fit_checked(&self.preparation, workspace, ctx).map_err(CausalError::from)?
        };
        clock.finish(STAGE_ESTIMATE_POINT);
        crate::analysis::stage::emit_stage(
            self.stage_sink.as_ref(),
            &crate::analysis::stage::StageEvent::Point { estimate: point.clone() },
        );

        clock.begin(ctx, crate::analysis::stage::STAGE_UNCERTAINTY, 0.55)?;
        let estimate = if self.default_id {
            self.fitter
                .attach_bootstrap(self.preparation.problem(), workspace, ctx, point)
                .map_err(CausalError::from)?
        } else {
            point
        };
        clock.finish(crate::analysis::stage::STAGE_UNCERTAINTY);
        crate::analysis::stage::emit_stage(
            self.stage_sink.as_ref(),
            &crate::analysis::stage::StageEvent::Uncertainty { estimate: estimate.clone() },
        );
        Ok(estimate)
    }

    /// Validation stage: refutes on the target-projected table and returns the reports
    /// with the projection, refutation and latency diagnostics, in report order.
    fn validate_stage(
        &self,
        data: &TabularData,
        estimate: &EffectEstimate,
        cancelled: bool,
        clock: &mut StageClock,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<
        (Vec<antecedent_validate::RefutationReport>, Vec<antecedent_core::Diagnostic>),
        CausalError,
    > {
        if cancelled {
            clock.mark_cancelled();
        } else {
            clock.begin(ctx, STAGE_VALIDATE, 0.8)?;
        }
        // The checked design was prepared against the original semantic
        // columns; validation runs on the table narrowed to the columns the
        // identified target names, as every adjustment route does.
        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, &self.query, &self.cache.estimand)?;
        let projected_cols = data_est.schema().len();
        let (refutations, refute_diagnostics) = if cancelled || self.refute == RefuteSuite::None {
            (Vec::new(), Vec::new())
        } else {
            let mut propensity = antecedent_stats::PropensityWorkspace::default();
            let (reports, diagnostics) = RefutationRun {
                data: &data_est,
                estimand: &estimand_est,
                query: &query_est,
                estimate,
                suite: self.refute,
                estimator: self.estimator.as_str(),
                custom: &[],
                temporal: None,
            }
            .reports(workspace, Some(&mut propensity), ctx)?;
            (reports, diagnostics)
        };
        let mut extra_diagnostics: Vec<antecedent_core::Diagnostic> =
            crate::analysis::helpers::projection_diagnostic(full_cols, projected_cols)
                .into_iter()
                .collect();
        extra_diagnostics.extend(refute_diagnostics);
        if let Some((tier, configured)) = self.latency_bootstrap_not_applied {
            extra_diagnostics.push(
                crate::analysis::helpers::latency_bootstrap_not_applied_diagnostic(
                    tier, configured,
                ),
            );
        }
        if !cancelled {
            clock.finish(STAGE_VALIDATE);
        }
        if !cancelled {
            crate::analysis::stage::emit_stage(
                self.stage_sink.as_ref(),
                &crate::analysis::stage::StageEvent::Validate {
                    refutations: refutations.clone(),
                    predictive_checks: Vec::new(),
                },
            );
        }
        Ok((refutations, extra_diagnostics))
    }
}

#[derive(Clone, Debug)]
pub(crate) enum CheckedIvOperation {
    Wald {
        fitter: antecedent_estimate::WaldIv,
        preparation: antecedent_estimate::CheckedIvPreparation,
    },
    TwoSls {
        fitter: antecedent_estimate::TwoStageLeastSquares,
        preparation: antecedent_estimate::CheckedIvPreparation,
    },
}

impl CheckedIvOperation {
    pub(crate) fn preparation(&self) -> &antecedent_estimate::CheckedIvPreparation {
        match self {
            Self::Wald { preparation, .. } | Self::TwoSls { preparation, .. } => preparation,
        }
    }

    pub(super) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        match self {
            Self::Wald { fitter, preparation } => Ok(Self::Wald {
                fitter: fitter.clone(),
                preparation: fitter.rebind_checked(preparation, data)?,
            }),
            Self::TwoSls { fitter, preparation } => Ok(Self::TwoSls {
                fitter: fitter.clone(),
                preparation: fitter.rebind_checked(preparation, data)?,
            }),
        }
    }
}

impl CheckedDistributionOperation {
    pub(crate) fn query(&self) -> &InterventionalDistributionQuery {
        &self.query
    }

    pub(crate) fn prepared(&self) -> &antecedent_estimate::PreparedFunctionalDistribution {
        &self.prepared
    }

    pub(crate) fn rebind(
        &self,
        data: &TabularData,
    ) -> Result<antecedent_estimate::PreparedFunctionalDistribution, CausalError> {
        self.prepared.rebind_checked(data).map_err(CausalError::from)
    }

    pub(super) fn sealed_for_direct_execution(&self) -> bool {
        matches!(self.graph_class, GraphClass::Dag | GraphClass::Admg | GraphClass::TemporalAdmg)
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalDistribution
            && self.custom_validator_names.is_empty()
            && match self.graph_class {
                GraphClass::Dag => matches!(
                    self.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                ),
                GraphClass::Admg | GraphClass::TemporalAdmg => {
                    self.refute == RefuteSuite::None && self.query.conditioning.is_empty()
                }
                _ => false,
            }
    }

    /// Execute the already rebound provider; the sealed caller binds it once.
    pub(super) fn execute_bound(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if matches!(self.graph_class, GraphClass::Admg | GraphClass::TemporalAdmg)
            && !self.query.conditioning.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "ADMG InterventionalDistribution is licensed for unconditional finite-discrete tables; IDC conditionals are a follow-up",
            });
        }
        let (distribution, posterior) = self.fit_distribution(&self.prepared, ctx)?;
        let bootstrap_ok = distribution.bootstrap_replicates_ok;
        let cancelled = distribution.bootstrap_cancelled;
        let early_stopped = distribution.bootstrap_early_stopped;
        let estimate = EffectEstimate::from_parts(
            distribution.mean,
            distribution.se_analytic,
            BootstrapAccounting {
                se: distribution.se_bootstrap,
                replicates_ok: distribution.bootstrap_replicates_ok,
                replicates_failed: distribution.bootstrap_replicates_failed,
                cancelled: distribution.bootstrap_cancelled,
                early_stopped: distribution.bootstrap_early_stopped,
            },
            distribution.assumptions.clone(),
            distribution.overlap,
            None,
            distribution.retained_memory_bytes,
        );
        let treatment =
            self.query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *self.query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        let diagnostics = self.result_diagnostics(&estimate, &distribution, posterior.as_ref());
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_distribution(
                data,
                &self.query,
                &self.cache.identification,
                &self.cache.estimand,
                &distribution,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let result = crate::analysis::execute::CheckedResultFinish::new(
            &self.common,
            &self.cache.identification,
            &self.cache.estimand,
            estimate,
            started,
            (treatment, outcome),
            (CausalQuery::Distribution(self.query.clone()), self.graph_class),
        )
        .with_diagnostics(diagnostics)
        .with_refutations(refutations)
        .with_distribution(distribution)
        .with_posterior(posterior)
        .with_bootstrap(Some(self.fitter.bootstrap_replicates), bootstrap_ok)
        .with_stop_flags(cancelled, early_stopped)
        .without_evalue_mirror()
        .with_rebound_interval()
        .finish();
        Ok(result)
    }

    /// Point or Bayesian fit of the prepared interventional distribution, chosen by the
    /// inference mode; the Bayesian arm also returns its joint posterior.
    fn fit_distribution(
        &self,
        prepared: &antecedent_estimate::PreparedFunctionalDistribution,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::InterventionalDistributionEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        ),
        CausalError,
    > {
        match &self.inference {
            InferenceMode::Frequentist => {
                let mut workspace = antecedent_estimate::FunctionalDistributionWorkspace::default();
                Ok((
                    self.fitter
                        .estimate(prepared, &[], &mut workspace, ctx)
                        .map_err(CausalError::from)?,
                    None,
                ))
            }
            InferenceMode::Bayesian(_) => {
                let n_draws = crate::analysis::execute::functional_bayesian_draws(&self.inference)?
                    .ok_or(CausalError::Unsupported {
                        message: "Bayesian inference selects no draw count",
                    })?;
                let (distribution, posterior) = self
                    .fitter
                    .estimate_bayesian(
                        prepared,
                        &[],
                        n_draws,
                        self.cache.identification.status,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                Ok((distribution, Some(posterior)))
            }
        }
    }

    /// Diagnostics for a fitted distribution, in the order the result reports them.
    fn result_diagnostics(
        &self,
        estimate: &EffectEstimate,
        distribution: &antecedent_estimate::InterventionalDistributionEstimate,
        posterior: Option<&antecedent_estimate::CausalPosterior>,
    ) -> Vec<antecedent_core::Diagnostic> {
        let mut diagnostics = self.cache.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
        diagnostics.extend(distribution_interval_diagnostics_for_prepared(distribution));
        if let Some(diagnostic) = crate::analysis::execute::free_variables_averaged(
            &self.cache.identification,
            &self.cache.estimand,
        ) {
            diagnostics.push(diagnostic);
        }
        diagnostics.push(crate::analysis::execute::identify_cached_diagnostic());
        if let Some(posterior) = posterior {
            diagnostics.extend(crate::analysis::execute::posterior_note_diagnostics([posterior]));
        }
        diagnostics
    }
}

fn distribution_interval_diagnostics_for_prepared(
    distribution: &antecedent_estimate::InterventionalDistributionEstimate,
) -> Vec<antecedent_core::Diagnostic> {
    let missing: Vec<(usize, &'static str)> = distribution
        .atom_uncertainty
        .iter()
        .enumerate()
        .filter_map(|(index, atom)| match atom.interval {
            antecedent_estimate::ProbabilityInterval::Unavailable(reason) => {
                Some((index, reason.as_str()))
            }
            antecedent_estimate::ProbabilityInterval::Bounded { .. } => None,
        })
        .collect();
    if missing.is_empty() {
        return Vec::new();
    }
    let atoms = missing
        .iter()
        .map(|(index, reason)| format!("{index}:{reason}"))
        .collect::<Vec<_>>()
        .join(",");
    let mut diagnostic = antecedent_core::Diagnostic::new(
        "estimate.distribution.interval_unavailable",
        antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Warning,
        format!(
            "{} of {} interventional probabilities have no bounded interval (atom:reason \
             {atoms}); a plug-in probability of exactly 0 or 1 has no sampling spread to \
             build one from",
            missing.len(),
            distribution.atom_uncertainty.len()
        ),
    );
    diagnostic.fields = Arc::from([(Arc::from("atoms"), Arc::from(atoms.as_str()))]);
    vec![diagnostic]
}

/// Sealed unknown-tier average effect retained by preparation.
#[derive(Clone, Debug)]
pub(crate) struct CheckedUnknownTieredAverageOperation {
    pub(super) query: AverageEffectQuery,
    pub(super) background: antecedent_graph::TieredBackground,
    pub(super) cache: Arc<super::CachedStaticIdentification>,
    pub(super) identifier: crate::strategy_table::IdentifierId,
    pub(super) estimator: crate::strategy_table::EstimatorId,
    pub(super) physical: std::sync::Arc<PhysicalExecutionPlan>,
}

/// Public inspection of the retained two-scenario Unknown-tier ATE operation.
#[derive(Clone, Debug)]
pub struct CheckedUnknownTieredAverageInfo {
    /// Frozen average-effect target.
    pub query: antecedent_core::AverageEffectQuery,
    /// Number of declared tiers.
    pub tier_count: usize,
    /// Canonical identified scenarios retained by the checked operation.
    pub scenario_count: usize,
    /// Selected identifier.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// The prepared physical plan identity.
    pub plan_id: std::sync::Arc<str>,
}

impl Inspect for CheckedUnknownTieredAverageOperation {
    type Info = CheckedUnknownTieredAverageInfo;

    fn info(&self) -> CheckedUnknownTieredAverageInfo {
        CheckedUnknownTieredAverageInfo {
            query: self.query.clone(),
            tier_count: self.background.tiers.len(),
            scenario_count: self.cache.identification.estimands.len(),
            identifier: self.identifier,
            estimator: self.estimator,
            plan_id: Arc::clone(&self.physical.record.plan_id),
        }
    }
}

#[cfg(test)]
#[path = "operations_lock_tests.rs"]
mod cancellable_fit_lock_tests;
