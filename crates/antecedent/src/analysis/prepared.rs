//! Compile-once / re-estimate-many prepared analysis handle.
//!
//! Rediscover policy: structure is frozen at prepare time. Changing bootstrap,
//! prior scale, treatment levels, or latency never re-runs discovery — only an
//! explicit new discover / review → prepare cycle may replace the graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, ExecutionContext, Intervention,
    InterventionalDistributionQuery, MediationQuery, OutcomeFunctional, ResponseQuery,
    TargetPopulation, TemporalEffectQuery, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{PanelData, TableView, TabularData, TemporalIndexer, TimeSeriesData};
use antecedent_discovery::{
    GraphPosterior, dag_from_adjacency_mask, temporal_cpdag_from_dbn_masks,
    temporal_dag_from_dbn_masks, temporal_pag_from_dbn_masks,
};
use antecedent_estimate::{
    AipwAte, CellSaturatedAipw, EffectEstimate, EstimationWorkspace, OverlapPolicy, RetargetResult,
    ScoreTable, crossfit_binary_scores, exceedance_cdf_values,
};

use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use crate::strategy_table::{DEFAULT_ESTIMATOR, EstimatorId, IdentifierId};

use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{Pag, TemporalDag};
use antecedent_identify::{
    IdentificationEnvelope, IdentificationResult, IdentificationStatus, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::CheckedGraphPosteriorEffect;
use super::builder::{DataInput, RefuteSuite};
use super::checked_propensity::{
    CheckedPropensityContext, CheckedPropensityEstimator, CheckedPropensityOperation,
    CheckedPropensityProcedure, CheckedPropensityUncertainty,
};
use super::execute::Study;
use super::helpers::{
    AssembleArgs, assemble_result, overlap_diagnostic, project_for_ate_estimate, provenance_pair,
    run_refuters,
};
use super::stage::{STAGE_ESTIMATE_POINT, STAGE_VALIDATE, StageClock};
use super::{CheckedConditionalOperation, ConditionalProcedure};

/// Prepare-time identification products for the static ATE / response path.
///
/// Everything identification reads — identifier, graph, query, RD config — is
/// frozen when the handle is built, and identification is deterministic, so an
/// estimate click reuses these instead of re-running identification. Results
/// carry an `exec.identify.cached` diagnostic so reuse is observable.
#[derive(Clone, Debug)]
pub struct CachedStaticIdentification {
    /// Identification result computed at prepare time.
    pub identification: IdentificationResult,
    /// Estimand selected for the prepared estimator.
    pub estimand: IdentifiedEstimand,
}

/// Retained functional-distribution target and program for prepared execution.
#[derive(Clone, Debug)]
pub(crate) struct CheckedDistributionOperation {
    query: InterventionalDistributionQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    fitter: antecedent_estimate::FunctionalDistribution,
    prepared: antecedent_estimate::PreparedFunctionalDistribution,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

/// Retained checked general-ID functional effect for finite-discrete ADMG ATEs.
/// The checked program and data binding are prepared together; estimate clicks
/// rebind that program to compatible data without consulting the original Study.
#[derive(Clone, Debug)]
pub(crate) struct CheckedFunctionalEffectOperation {
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    fitter: antecedent_estimate::FunctionalEffect,
    prepared: antecedent_estimate::PreparedFunctionalEffect,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

/// Checked path-specific natural effect retained with its graph target and program.
#[derive(Clone, Debug)]
pub(crate) struct CheckedPathSpecificEffectOperation {
    query: antecedent_core::PathSpecificEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    fitter: antecedent_estimate::FunctionalEffect,
    prepared: antecedent_estimate::PreparedFunctionalEffect,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

impl CheckedPathSpecificEffectOperation {
    fn sealed_for_direct_execution(&self) -> bool {
        matches!(
            self.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) && self.identifier == crate::strategy_table::IdentifierId::PathSpecificNatural
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
    }

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let mut extra = vec![self.query.treatment, self.query.outcome];
        extra.extend(self.query.path_nodes.iter().copied());
        let prepared = self
            .fitter
            .prepare(
                data,
                &self.estimand,
                &self.identification.arena,
                self.identification.required_assumptions.clone(),
                &extra,
            )
            .map_err(CausalError::from)?;
        let mut rebound = self.clone();
        rebound.prepared = prepared;
        Ok(rebound)
    }

    fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.estimand.method_kind().ok()
            != Some(antecedent_expr::EstimandMethod::PathSpecificNatural)
            || self.prepared.estimand.functional != self.estimand.functional
            || self.prepared.program().mapping().source != self.estimand.functional
            || self.query.treatment == self.query.outcome
        {
            return Err(CausalError::Compile {
                message: "prepared path-specific operation disagrees with its checked target"
                    .into(),
            });
        }
        let (estimate, posterior) = match &self.inference {
            InferenceMode::Frequentist => {
                let mut workspace = antecedent_estimate::FunctionalDistributionWorkspace::default();
                (
                    self.fitter
                        .estimate(&self.prepared, &mut workspace, ctx)
                        .map_err(CausalError::from)?,
                    None,
                )
            }
            InferenceMode::Bayesian(config) => {
                if config.prior_artifact.is_some()
                    || config.external_compose.is_some()
                    || config.prior.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "functional Bayesian prior transfer requires a declared functional mapping; a coefficient prior cannot be applied to CPT factors",
                    });
                }
                if config.n_draws < 2 {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian inference requires n_draws >= 2; refusing silent rewrite of 0 or 1",
                    });
                }
                let posterior = self
                    .fitter
                    .estimate_bayesian(
                        &self.prepared,
                        config.n_draws,
                        self.identification.status,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                (super::execute::effect_from_posterior(&posterior)?, Some(posterior))
            }
        };
        let mut diagnostics = self.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        if let Some(diagnostic) =
            super::execute::free_variables_averaged(&self.identification, &self.estimand)
        {
            diagnostics.push(diagnostic);
        }
        if let Some(posterior) = posterior.as_ref() {
            diagnostics.extend(super::execute::posterior_note_diagnostics([posterior]));
        }
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_path(
                data,
                &self.query,
                &self.identification,
                &self.estimand,
                estimate.ate,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &self.identification.required_assumptions),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let n_draws =
            posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX));
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            distribution: None,
            posterior,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.fitter.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws,
            cancelled: false,
            early_stopped: false,
            bayesian: matches!(self.inference, InferenceMode::Bayesian(_)),
        });
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::PathSpecific(self.query.clone()),
            graph_class: GraphClass::Dag,
        });
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        result.population_registry.clone_from(&self.population_registry);
        result.custom_validator_names = self.custom_validator_names.to_vec();
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }
}

impl CheckedFunctionalEffectOperation {
    fn sealed_for_direct_execution(&self) -> bool {
        self.graph_class == GraphClass::Admg
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.identifier == crate::strategy_table::IdentifierId::GeneralId
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
    }

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let prepared = self
            .fitter
            .prepare(
                data,
                &self.estimand,
                &self.identification.arena,
                self.identification.required_assumptions.clone(),
                &[self.query.treatment, self.query.outcome],
            )
            .map_err(CausalError::from)?;
        let mut rebound = self.clone();
        rebound.prepared = prepared;
        Ok(rebound)
    }

    fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::GeneralId)
            || self.prepared.estimand.functional != self.estimand.functional
            || self.prepared.estimand.adjustment_set != self.estimand.adjustment_set
            || self.prepared.program().mapping().source != self.estimand.functional
        {
            return Err(CausalError::Compile {
                message: "prepared functional-effect operation disagrees with its checked target"
                    .into(),
            });
        }
        let (estimate, posterior) = match &self.inference {
            InferenceMode::Frequentist => {
                let mut workspace = antecedent_estimate::FunctionalDistributionWorkspace::default();
                (
                    self.fitter
                        .estimate(&self.prepared, &mut workspace, ctx)
                        .map_err(CausalError::from)?,
                    None,
                )
            }
            InferenceMode::Bayesian(config) => {
                if config.prior_artifact.is_some()
                    || config.external_compose.is_some()
                    || config.prior.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "functional Bayesian prior transfer requires a declared functional mapping; a coefficient prior cannot be applied to CPT factors",
                    });
                }
                if config.n_draws < 2 {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian inference requires n_draws >= 2; refusing silent rewrite of 0 or 1",
                    });
                }
                let posterior = self
                    .fitter
                    .estimate_bayesian(
                        &self.prepared,
                        config.n_draws,
                        self.identification.status,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                (super::execute::effect_from_posterior(&posterior)?, Some(posterior))
            }
        };
        let mut diagnostics = self.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        if let Some(diagnostic) =
            super::execute::free_variables_averaged(&self.identification, &self.estimand)
        {
            diagnostics.push(diagnostic);
        }
        if let Some(posterior) = posterior.as_ref() {
            diagnostics.extend(super::execute::posterior_note_diagnostics([posterior]));
        }
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &self.identification.required_assumptions),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra) = if self.refute == RefuteSuite::None {
            (Vec::new(), Vec::new())
        } else {
            run_refuters(
                data,
                &self.estimand,
                &self.query,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                self.estimator.as_str(),
                &[],
                None,
            )?
        };
        diagnostics.extend(extra);
        let n_draws =
            posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX));
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            distribution: None,
            posterior,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.fitter.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws,
            cancelled: false,
            early_stopped: false,
            bayesian: matches!(self.inference, InferenceMode::Bayesian(_)),
        });
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::AverageEffect(self.query.clone()),
            graph_class: self.graph_class,
        });
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        result.population_registry.clone_from(&self.population_registry);
        result.custom_validator_names = self.custom_validator_names.to_vec();
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }
}

/// One ordered, checked intervention point in an ADMG response curve.
#[derive(Clone, Debug)]
pub struct CheckedFunctionalEffectResponseMember {
    grid_value: f64,
    query: antecedent_core::ResponseQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    prepared: antecedent_estimate::PreparedFunctionalEffect,
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

/// Retained complete ordered program family for an ADMG response curve.
#[derive(Clone, Debug)]
pub(crate) struct CheckedAdmgResponseCurveOperation {
    query: antecedent_core::ResponseQuery,
    grid: Arc<[f64]>,
    members: Arc<[CheckedFunctionalEffectResponseMember]>,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    fitter: antecedent_estimate::FunctionalEffect,
    inference: InferenceMode,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

impl CheckedAdmgResponseCurveOperation {
    fn sealed_for_direct_execution(&self) -> bool {
        matches!(
            self.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) && self.identifier == crate::strategy_table::IdentifierId::GeneralId
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && self.custom_validator_names.is_empty()
            && self.members.len() == self.grid.len()
    }

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let members = self
            .members
            .iter()
            .map(|member| {
                let (treatment, outcome) =
                    member.query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "retained ADMG response member lost its treatment/outcome roles"
                            .into(),
                    })?;
                let prepared = self
                    .fitter
                    .prepare(
                        data,
                        &member.estimand,
                        &member.identification.arena,
                        member.identification.required_assumptions.clone(),
                        &[treatment, outcome],
                    )
                    .map_err(CausalError::from)?;
                Ok(CheckedFunctionalEffectResponseMember { prepared, ..member.clone() })
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
        let mut rebound = self.clone();
        rebound.members = Arc::from(members);
        Ok(rebound)
    }

    fn execute(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.members.is_empty() || self.members.len() != self.grid.len() {
            return Err(CausalError::Compile {
                message: "retained ADMG response grid/member mapping is incomplete".into(),
            });
        }
        let mut means = Vec::with_capacity(self.members.len());
        let mut member_posteriors = Vec::new();
        let mut support_status = antecedent_core::SupportStatus::Supported;
        let mut support_warnings = Vec::new();
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
                InferenceMode::Bayesian(config) => {
                    if config.prior_artifact.is_some()
                        || config.external_compose.is_some()
                        || config.prior.is_some()
                    {
                        return Err(CausalError::Unsupported {
                            message: "functional Bayesian prior transfer requires a declared functional mapping; a coefficient prior cannot be applied to CPT factors",
                        });
                    }
                    if config.n_draws < 2 {
                        return Err(CausalError::Unsupported {
                            message: "Bayesian inference requires n_draws >= 2; refusing silent rewrite of 0 or 1",
                        });
                    }
                    let fit = self
                        .fitter
                        .estimate_bayesian(
                            &member.prepared,
                            config.n_draws,
                            member.identification.status,
                            ctx,
                        )
                        .map_err(CausalError::from)?;
                    let mean = super::execute::effect_from_posterior(&fit)?.ate;
                    // `estimate_bayesian` opens the same named RNG stream for each
                    // member. Its Bayesian-bootstrap row weights therefore align
                    // draw-for-draw across the fixed grid; retain every member
                    // column below so the resulting posterior keeps that joint law.
                    member_posteriors.push(fit);
                    mean
                }
            };
            means.push(value);
        }
        let n_draws = member_posteriors
            .first()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let posterior = if member_posteriors.is_empty() {
            None
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
                columns.push(
                    member_posterior
                        .draws
                        .column(quantity)
                        .map_err(|error| CausalError::Compile { message: error.to_string() })?
                        .to_vec(),
                );
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
            )
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
            let summaries = draws.summarize();
            let mut joint = member_posteriors.remove(0);
            joint.draws = draws;
            joint.summaries = summaries;
            Some(joint)
        };
        let first = self.members.first().expect("nonempty checked member array");
        let mut support =
            antecedent_estimate::support_from_functional_eval(None).map_err(|error| {
                CausalError::from(antecedent_estimate::EstimationError::data_msg(error.to_string()))
            })?;
        support.status = support_status;
        support.query_region = antecedent_core::SupportRegion {
            minima: Arc::from([self.grid.iter().copied().fold(f64::INFINITY, f64::min)]),
            maxima: Arc::from([self.grid.iter().copied().fold(f64::NEG_INFINITY, f64::max)]),
        };
        support.warnings.extend(support_warnings.iter().cloned());
        let response = antecedent_core::CausalResponse {
            estimand: self.query.functional.clone(),
            identification_status: first.identification.status,
            estimate: antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Surface {
                    grid: Arc::clone(&self.grid),
                    dimension: 1,
                    mean: Arc::from(means),
                },
            ),
            uncertainty: antecedent_core::ResponseUncertainty::None,
            support,
            assumptions: first.identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        let mut diagnostics = first.identification.diagnostics.clone();
        diagnostics.push(antecedent_core::Diagnostic::new(
            "identify.response.general_id", antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "intervention mean identified by general ID; grid members are evaluated from their retained checked programs",
        ));
        diagnostics.extend(support_warnings);
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (
                identify_artifact,
                identify_operation,
                &[],
                &first.identification.required_assumptions,
            ),
            (
                estimate_artifact,
                estimate_operation,
                &[identify_artifact],
                &first.prepared.assumptions,
            ),
        );
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: first.identification.clone(),
            estimand: first.estimand.clone(),
            estimate: EffectEstimate::new(
                f64::NAN,
                f64::NAN,
                first.identification.required_assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            ),
            distribution: None,
            posterior,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: first.query.functional.primary_pair().expect("checked member roles").0,
            outcome: first.query.functional.primary_pair().expect("checked member roles").1,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(0),
            bootstrap_replicates_ok: None,
            n_draws,
            cancelled: false,
            early_stopped: false,
            bayesian: matches!(self.inference, InferenceMode::Bayesian(_)),
        });
        result.response = Some(response);
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: first.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::Response(self.query.clone()),
            graph_class: GraphClass::Admg,
        });
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        result.population_registry.clone_from(&self.population_registry);
        result.custom_validator_names = self.custom_validator_names.to_vec();
        Ok(result)
    }
}

/// Retained checked family plan for a static DAG, complete observation mean curve.
/// The grid is copied into the plan so execution can detect query tampering before
/// handing the frozen estimand to the response estimator.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticResponseCurve {
    query: ResponseQuery,
    grid: Arc<[f64]>,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
}

impl CheckedStaticResponseCurve {
    pub(crate) fn checked_route(
        &self,
        query: &ResponseQuery,
    ) -> Result<(&IdentificationResult, &IdentifiedEstimand), CausalError> {
        let values = match &query.functional {
            antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } => treatment
                .grid
                .values()
                .map_err(|e| CausalError::Compile { message: e.to_string() })?,
            _ => Vec::new(),
        };
        if query != &self.query || values.as_slice() != self.grid.as_ref() {
            return Err(CausalError::Compile {
                message: "prepared response curve query or grid changed; re-prepare the study"
                    .into(),
            });
        }
        let _procedure = (self.identifier, self.estimator);
        Ok((&self.identification, &self.estimand))
    }

    pub(crate) const fn procedure(
        &self,
    ) -> (crate::strategy_table::IdentifierId, crate::strategy_table::EstimatorId) {
        (self.identifier, self.estimator)
    }
}

/// Read-only inspection of a retained checked derivative-response plan.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedDerivativeResponseInfo {
    /// Exact response functional and evaluation coordinates.
    pub query: ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Adjustment variables retained by the checked target.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Maximum derivative coordinates the operation may materialize.
    pub max_derivative_cells: usize,
}

/// Read-only inspection of a retained static DAG response operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedStaticDagResponseInfo {
    /// Exact mean-curve or intervention query, including its member values.
    pub query: ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Ordered curve grid. Empty for intervention-response queries.
    pub grid_members: Arc<[f64]>,
}

/// Inspect the sealed joint-cell AIPW response route retained by preparation.
#[derive(Clone, Debug)]
pub struct CheckedCellAipwResponseInfo {
    /// The complete intervention-response query.
    pub query: ResponseQuery,
    /// The checked identifier selected at preparation.
    pub identifier: IdentifierId,
    /// The checked estimator selected at preparation.
    pub estimator: EstimatorId,
    /// Refutation suite carried by the operation.
    pub validation: RefuteSuite,
    /// Whether the graph was explicit or accepted.
    pub origin: super::execute::DagResponseOrigin,
    /// Binary joint-cell index evaluated by this route.
    pub requested_arm: u32,
    /// Covariate roles retained for adjustment.
    pub adjustment_set: Arc<[VariableId]>,
}

/// Read-only target, graph, and procedure of a checked static mediation plan.
#[derive(Clone, Debug)]
pub struct CheckedStaticMediationInfo {
    /// Complete contrast, mediators, intervention levels, and population.
    pub query: MediationQuery,
    /// Prepare-time identifier.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Identified adjustment roles used by the model.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Source and executable functional root identifiers.
    pub functional_roots: (u32, u32),
    /// Canonical supplied DAG edges.
    pub graph_edges: Arc<[(u32, u32)]>,
    /// Bootstrap replicate budget, including zero for point-only execution.
    pub bootstrap_replicates: u32,
    /// Prepare-time validation suite.
    pub validation: RefuteSuite,
}

/// Read-only target and procedure retained by a checked GCM attribution plan.
#[derive(Clone, Debug)]
pub struct CheckedAttributionInfo {
    /// Exact anomaly or distribution-change target.
    pub query: CausalQuery,
    /// Prepare-time identifier and estimator.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Canonical supplied DAG edges.
    pub graph_edges: Arc<[(u32, u32)]>,
    /// Identification status fixed at preparation.
    pub identification_status: antecedent_identify::IdentificationStatus,
}

/// Read-only inspection of a retained TemporalDag response operation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedTemporalDagResponseInfo {
    /// Exact response functional, policy, and requested horizons.
    pub query: ResponseQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Interval procedure and replicate count.
    pub uncertainty: Arc<str>,
    /// Ordered response grid.
    pub grid_members: Arc<[f64]>,
    /// Ordered horizon members.
    pub horizons: Arc<[u32]>,
}

/// Read-only target, proof, and procedure of a prepared TemporalDag effect.
#[derive(Clone, Debug)]
pub struct CheckedTemporalDagEffectInfo {
    /// Exact temporal policy, intervention levels, horizon, and population.
    pub query: TemporalEffectQuery,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Graph-derived adjustment variables in the unfolded proof.
    pub adjustment_set: Arc<[antecedent_core::TemporalNodeKey]>,
    /// Canonical node and edge signature of the prepared temporal graph.
    pub graph_signature: Arc<str>,
}

/// Read-only target and procedure for a checked temporal mediation family.
#[derive(Clone, Debug)]
pub struct CheckedTemporalMediationInfo {
    /// Exact mediation target and requested horizon family.
    pub query: MediationQuery,
    /// Signature of the fixed TemporalDag.
    pub graph_signature: Arc<str>,
    /// Validation suite selected at preparation.
    pub validation: RefuteSuite,
    /// Horizon members retained in order.
    pub horizons: Arc<[u32]>,
    /// Estimator family fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Graph-derived adjustment node keys retained at each horizon.
    pub adjustment_sets: Arc<[(u32, Arc<[antecedent_core::TemporalNodeKey]>)]>,
}

/// Read-only source graph, completion proof, and procedure for a temporal
/// Cpdag/Pag pulse or sustained effect.
#[derive(Clone, Debug)]
pub struct CheckedTemporalClassEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: TemporalEffectQuery,
    /// Source class retained with the completion proof.
    pub graph_class: GraphClass,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: RefuteSuite,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
    /// Number of completion atoms retained by the checked proof.
    pub completion_count: usize,
    /// Maximum number of class completions searched, when configured.
    pub completion_limit: Option<usize>,
    /// Identified and unresolved enumeration mass retained by the proof.
    pub identified_mass: f64,
    /// Unidentified enumeration mass retained by the proof.
    pub unidentified_mass: f64,
    /// Number of completions whose enumeration was cut off.
    pub truncated_completions: usize,
    /// Source graph version sealed with the proof.
    pub graph_version: u64,
}

/// Read-only target and uncertainty binding for a prepared propensity route.
#[derive(Clone, Debug)]
pub struct CheckedPropensityInfo {
    /// The selected weighting or matching estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Adjustment variables in the checked target.
    pub adjustment_set: Arc<[antecedent_core::VariableId]>,
    /// Population targeted by the bound design.
    pub population: TargetPopulation,
    /// Source rows retained by the propensity preparation.
    pub source_rows: Arc<[u32]>,
    /// Named uncertainty procedure.
    pub uncertainty: Arc<str>,
    /// Bootstrap count only when the weighting procedure can use it.
    pub bootstrap_replicates: Option<u32>,
}

/// Read-only checked target and procedure for a frequentist DAG conditional effect.
#[derive(Clone, Debug)]
pub struct CheckedConditionalEffectInfo {
    /// Exact conditional query, including modifiers, arms, and outcome functional.
    pub query: antecedent_core::ConditionalEffectQuery,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Actual scalar or distribution score procedure.
    pub procedure: Arc<str>,
    /// Prepared refutation suite.
    pub validation: RefuteSuite,
    /// Variables in the physical design, retaining their semantic IDs.
    pub design_roles: Arc<[antecedent_core::VariableId]>,
    /// Complete source row indices at preparation or latest refresh.
    pub source_rows: Arc<[u32]>,
}

/// Read-only graph atoms, frozen weights and procedure retained for a
/// frequentist DAG graph-posterior effect.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedGraphPosteriorEffectInfo {
    /// Target query sealed during preparation.
    pub query: AverageEffectQuery,
    /// Static estimator used for every identified atom.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite applied to the atom-wise estimates.
    pub validation: RefuteSuite,
    /// Opaque graph identity per posterior sample.
    pub graph_keys: Arc<[u64]>,
    /// Frozen posterior weight per graph sample.
    pub weights: Arc<[f64]>,
    /// Identification status per graph sample, preserving unknown graph mass.
    pub identified: Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Procedure-specific bootstrap count retained by preparation.
    pub bootstrap_replicates: u32,
}

/// Checked binding for the prepared static Bayesian mean ATE g-computation row.
/// It keeps the original query, selected identification claim and inference
/// configuration together so estimate clicks cannot silently select another row.
#[derive(Clone, Debug)]
pub struct CheckedBayesianGcompOperation {
    pub(crate) query: AverageEffectQuery,
    pub(crate) identification: IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) inference: InferenceMode,
}

/// Inspection receipt for a prepared Bayesian DAG conditional-effect route.
#[derive(Clone, Debug)]
pub struct CheckedBayesianConditionalInfo {
    query: antecedent_core::ConditionalEffectQuery,
    inference: InferenceMode,
    validation: RefuteSuite,
    modifier_roles: Arc<[antecedent_core::VariableId]>,
    identification_status: IdentificationStatus,
    estimand: IdentifiedEstimand,
}

impl CheckedBayesianConditionalInfo {
    /// Conditional causal target retained by preparation.
    #[must_use]
    pub fn query(&self) -> &antecedent_core::ConditionalEffectQuery {
        &self.query
    }
    /// Bayesian prior and inference settings retained by preparation.
    #[must_use]
    pub fn inference(&self) -> &InferenceMode {
        &self.inference
    }
    /// Validation suite retained by preparation.
    #[must_use]
    pub fn validation(&self) -> RefuteSuite {
        self.validation
    }
    /// Semantic modifier variables retained by preparation.
    #[must_use]
    pub fn modifier_roles(&self) -> &[antecedent_core::VariableId] {
        &self.modifier_roles
    }
    /// Identification status retained with the target.
    #[must_use]
    pub fn identification_status(&self) -> IdentificationStatus {
        self.identification_status
    }
    /// Identified target functional and adjustment roles.
    #[must_use]
    pub fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }
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
        &self.identification
    }

    /// Selected causal target retained for this query.
    #[must_use]
    pub fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
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
    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
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
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

impl CheckedAipwOperation {
    fn sealed_for_direct_execution(&self) -> bool {
        self.graph_class == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::Aipw
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
            && self.preparation.lowering().procedure
                == antecedent_estimate::CheckedAipwProcedure::CrossFittedLogisticOls
            && self.preparation.lowering().population == TargetPopulation::AllObserved
    }

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self.fitter.rebind_checked(&self.preparation, data)?;
        let mut rebound = self.clone();
        rebound.preparation = preparation;
        Ok(rebound)
    }

    fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
        }
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
        if self.preparation.target().functional != self.estimand.functional
            || self.preparation.target().adjustment_set != self.estimand.adjustment_set
            || lowering.treatment != self.query.treatment
            || lowering.outcome != self.query.outcome
            || active != Some(1.0)
            || control != Some(0.0)
        {
            return Err(CausalError::Compile {
                message: "prepared AIPW operation disagrees with its checked target or query"
                    .into(),
            });
        }
        let mut workspace = antecedent_estimate::AipwWorkspace::default();
        let estimate = self
            .fitter
            .fit_checked(&self.preparation, &mut workspace, ctx)
            .map_err(CausalError::from)?;
        let cancelled = estimate.bootstrap_cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut refute_diagnostics) = if cancelled || self.refute == RefuteSuite::None
        {
            (Vec::new(), Vec::new())
        } else {
            let mut estimate_workspace = EstimationWorkspace::default();
            let (reports, diagnostics) = run_refuters(
                data,
                &self.estimand,
                &self.query,
                &estimate,
                &mut estimate_workspace,
                Some(&mut workspace.propensity),
                ctx,
                self.refute,
                self.estimator.as_str(),
                &[],
                None,
            )?;
            (reports, diagnostics)
        };
        let mut diagnostics = self.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        diagnostics.push(if estimate.score_table.is_some() {
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
        diagnostics.append(&mut refute_diagnostics);
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &self.identification.required_assumptions),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.fitter.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
            bayesian: false,
        });
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::AverageEffect(self.query.clone()),
            graph_class: self.graph_class,
        });
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        result.population_registry.clone_from(&self.population_registry);
        result.custom_validator_names = self.custom_validator_names.to_vec();
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }
}

/// Checked linear adjustment target with its selected fit and uncertainty
/// procedure. `default_id` preserves the progressive point/uncertainty stages.
#[derive(Clone, Debug)]
pub(crate) struct CheckedLinearOperation {
    pub(crate) fitter: antecedent_estimate::LinearAdjustmentAte,
    pub(crate) preparation: antecedent_estimate::CheckedLinearAdjustmentAte,
    pub(crate) default_id: bool,
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    custom_validator_names: Arc<[Arc<str>]>,
}

/// Immutable result-construction context retained with a checked static operation.
struct CheckedStaticResultMetadata<'a> {
    source_query: &'a CausalQuery,
    query: &'a AverageEffectQuery,
    identification: &'a IdentificationResult,
    estimand: &'a IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: &'a PhysicalExecutionPlan,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<&'a antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    refute: RefuteSuite,
    custom_validator_names: &'a [Arc<str>],
}

/// Retained GLM adjustment target, selected configuration, and compiled design.
#[derive(Clone, Debug)]
pub(crate) struct CheckedGlmAdjustmentOperation {
    source_query: CausalQuery,
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    fitter: antecedent_estimate::GlmAdjustmentAte,
    preparation: antecedent_estimate::PreparedGlmProblem,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

impl CheckedGlmAdjustmentOperation {
    fn sealed_for_direct_execution(&self) -> bool {
        self.graph_class == GraphClass::Dag
            && matches!(
                self.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            && self.estimator == crate::strategy_table::EstimatorId::GlmAdjustment
            && matches!(self.inference, InferenceMode::Frequentist)
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
    }

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation =
            self.fitter.prepare(data, &self.estimand, &self.query).map_err(CausalError::from)?;
        let mut rebound = self.clone();
        rebound.preparation = preparation;
        Ok(rebound)
    }
}

/// Retained sharp-RD target, design geometry, and local-linear fit.
#[derive(Clone, Debug)]
pub(crate) struct CheckedRdOperation {
    source_query: AverageEffectQuery,
    query: AverageEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identifier: crate::strategy_table::IdentifierId,
    estimator: crate::strategy_table::EstimatorId,
    physical: PhysicalExecutionPlan,
    fitter: antecedent_estimate::SharpRegressionDiscontinuity,
    preparation: antecedent_estimate::CheckedRdPreparation,
    inference: InferenceMode,
    refute: RefuteSuite,
    graph_class: GraphClass,
    graph_version: u32,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    population_registry: Option<antecedent_core::PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    custom_validator_names: Arc<[Arc<str>]>,
}

impl CheckedRdOperation {
    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self
            .fitter
            .prepare_checked(data, &self.identification, 0)
            .map_err(CausalError::from)?;
        let mut rebound = self.clone();
        rebound.preparation = preparation;
        Ok(rebound)
    }
}

impl CheckedLinearOperation {
    fn sealed_for_direct_execution(&self) -> bool {
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

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let preparation = self.fitter.rebind_checked(&self.preparation, data)?;
        let mut rebound = self.clone();
        rebound.preparation = preparation;
        Ok(rebound)
    }

    fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
        }
        if self.preparation.source_functional() != self.estimand.functional
            || self.preparation.target().adjustment_set != self.estimand.adjustment_set
            || self.preparation.lowering().treatment != self.query.treatment
            || self.preparation.lowering().outcome != self.query.outcome
            || self.preparation.lowering().population != self.query.target_population
        {
            return Err(CausalError::Compile {
                message: "prepared linear adjustment operation disagrees with its checked target"
                    .into(),
            });
        }

        let mut workspace = EstimationWorkspace::default();
        let point = if self.default_id {
            self.fitter
                .fit_point(
                    self.preparation.problem(),
                    &mut workspace,
                    self.preparation.required_assumptions().clone(),
                )
                .map_err(CausalError::from)?
        } else {
            self.fitter
                .fit_checked(&self.preparation, &mut workspace, ctx)
                .map_err(CausalError::from)?
        };
        let estimate = if self.default_id {
            self.fitter
                .attach_bootstrap(self.preparation.problem(), &mut workspace, ctx, point)
                .map_err(CausalError::from)?
        } else {
            point
        };
        let cancelled = estimate.bootstrap_cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut extra_diagnostics) = if cancelled || self.refute == RefuteSuite::None
        {
            (Vec::new(), Vec::new())
        } else {
            let mut propensity = antecedent_stats::PropensityWorkspace::default();
            let (reports, diagnostics) = run_refuters(
                data,
                &self.estimand,
                &self.query,
                &estimate,
                &mut workspace,
                Some(&mut propensity),
                ctx,
                self.refute,
                self.estimator.as_str(),
                &[],
                None,
            )?;
            (reports, diagnostics)
        };
        let mut diagnostics = self.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        diagnostics.append(&mut extra_diagnostics);
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &self.identification.required_assumptions),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: self.query.treatment,
            outcome: self.query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: None,
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.fitter.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
            bayesian: false,
        });
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::AverageEffect(self.query.clone()),
            graph_class: self.graph_class,
        });
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        result.population_registry.clone_from(&self.population_registry);
        result.custom_validator_names = self.custom_validator_names.to_vec();
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
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

    fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
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

    fn sealed_for_direct_execution(&self) -> bool {
        matches!(self.graph_class, GraphClass::Dag | GraphClass::Admg)
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
                GraphClass::Admg => {
                    self.refute == RefuteSuite::None && self.query.conditioning.is_empty()
                }
                _ => false,
            }
    }

    fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let prepared = self.rebind(data)?;
        if matches!(self.graph_class, GraphClass::Admg) && !self.query.conditioning.is_empty() {
            return Err(CausalError::Unsupported {
                message: "ADMG InterventionalDistribution is licensed for unconditional finite-discrete tables; IDC conditionals are a follow-up",
            });
        }
        let (distribution, posterior) = match &self.inference {
            InferenceMode::Frequentist => {
                let mut workspace = antecedent_estimate::FunctionalDistributionWorkspace::default();
                (
                    self.fitter
                        .estimate(&prepared, &[], &mut workspace, ctx)
                        .map_err(CausalError::from)?,
                    None,
                )
            }
            InferenceMode::Bayesian(config) => {
                if config.prior_artifact.is_some()
                    || config.external_compose.is_some()
                    || config.prior.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "functional Bayesian prior transfer requires a declared functional mapping; a backdoor coefficient artifact cannot be applied as an isotropic CPT prior",
                    });
                }
                if config.n_draws < 2 {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian inference requires n_draws >= 2; refusing silent rewrite of 0 or 1",
                    });
                }
                let (distribution, posterior) = self
                    .fitter
                    .estimate_bayesian(
                        &prepared,
                        &[],
                        config.n_draws,
                        self.identification.status,
                        ctx,
                    )
                    .map_err(CausalError::from)?;
                (distribution, Some(posterior))
            }
        };
        let bootstrap_ok = distribution.bootstrap_replicates_ok;
        let cancelled = distribution.bootstrap_cancelled;
        let early_stopped = distribution.bootstrap_early_stopped;
        let estimate = EffectEstimate::from_parts(
            distribution.mean,
            distribution.se_analytic,
            distribution.se_bootstrap,
            distribution.bootstrap_replicates_ok,
            distribution.bootstrap_replicates_failed,
            distribution.bootstrap_cancelled,
            distribution.bootstrap_early_stopped,
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
        let mut diagnostics = self.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.extend(distribution_interval_diagnostics_for_prepared(&distribution));
        if let Some(diagnostic) =
            super::execute::free_variables_averaged(&self.identification, &self.estimand)
        {
            diagnostics.push(diagnostic);
        }
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        if let Some(posterior) = posterior.as_ref() {
            diagnostics.extend(super::execute::posterior_note_diagnostics([posterior]));
        }
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(self.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(self.estimator);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &self.identification.required_assumptions),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_distribution(
                data,
                &self.query,
                &self.identification,
                &self.estimand,
                &distribution,
                self.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let mut result = assemble_result(AssembleArgs {
            logical: &self.physical.logical.record,
            physical: &self.physical.record,
            identification: self.identification.clone(),
            estimand: self.estimand.clone(),
            estimate,
            distribution: Some(distribution),
            posterior,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.fitter.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws,
            cancelled,
            early_stopped,
            bayesian: matches!(self.inference, InferenceMode::Bayesian(_)),
        });
        result.certificate = Some(crate::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: self.identification.clone(),
                temporal_indexer: None,
                strategy: self.identifier,
                structure_version: self.graph_version,
            },
            query: CausalQuery::Distribution(self.query.clone()),
            graph_class: self.graph_class,
        });
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        result.population_registry.clone_from(&self.population_registry);
        result.custom_validator_names = self.custom_validator_names.to_vec();
        result.rebind_interval(matches!(self.inference, InferenceMode::Bayesian(_)));
        Ok(result)
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

/// Prepare-time generalized-adjustment envelope for a supplied PAG.
#[derive(Clone, Debug)]
pub(crate) struct CachedPagIdentification {
    /// Per-completion identification results and their probability mass.
    pub envelope: IdentificationEnvelope<Pag>,
    /// Public aggregate identification result derived from [`Self::envelope`].
    pub identification: IdentificationResult,
}

/// Prepare-time MEC envelope for a supplied CPDAG.
#[derive(Clone, Debug)]
pub(crate) struct CachedCpdagIdentification {
    /// Per-completion identification results and their probability mass.
    pub envelope: IdentificationEnvelope<antecedent_graph::Dag>,
    /// Public aggregate identification result derived from [`Self::envelope`].
    pub identification: IdentificationResult,
}

/// Prepare-time TemporalCpdag/Pag envelope (completions + unfold indexers).
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassIdentification {
    /// Generalized-adjustment envelope over TemporalDag completions.
    pub envelope: antecedent_identify::TemporalClassEnvelope,
    /// Functional-specific certificates for every requested response or mediation horizon.
    pub by_horizon: Vec<(u32, antecedent_identify::TemporalClassEnvelope)>,
}

/// One identified atom in a prepared static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorAtomIdentification {
    /// Opaque graph key retained by the posterior envelope.
    pub key: u64,
    /// Per-atom selected estimand.
    pub estimand: IdentifiedEstimand,
    /// Full per-atom identification result.
    pub identification: IdentificationResult,
}

/// One completion case inside a CPDAG/PAG posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedClassPosteriorCase {
    /// Enumeration weight of this completion.
    pub weight: f64,
    /// Identification status of the completion.
    pub status: IdentificationStatus,
    /// Selected estimand when the completion is estimable.
    pub estimand: Option<IdentifiedEstimand>,
}

/// Class-envelope identification for one CPDAG/PAG posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedClassPosteriorAtomIdentification {
    /// Opaque graph key retained by the posterior envelope.
    pub key: u64,
    /// Envelope-level identification (status, diagnostics, assumptions).
    pub identification: IdentificationResult,
    /// Shared estimand when every identified completion agrees.
    pub invariant: Option<IdentifiedEstimand>,
    /// Per-completion cases, including unidentified completions.
    pub cases: Arc<[CachedClassPosteriorCase]>,
    /// Envelope identified weight before posterior mixing.
    pub identified_weight: f64,
    /// Completions whose search was capped.
    pub truncated_completions: usize,
}

/// Prepare-time identification for every atom in a static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified DAG atoms, one per distinct graph key, in order of first
    /// appearance. Weight an atom by the combined identified mass of its key in
    /// [`Self::graphs`] (`identified_weight_for_key`), which keeps one entry per
    /// posterior sample. Unidentified atoms remain in [`Self::graphs`] with
    /// [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedGraphPosteriorAtomIdentification]>,
    /// CPDAG/PAG posterior atoms evaluated with the class ATE envelope.
    /// Empty when [`antecedent_discovery::GraphPosterior::atom_kind`] is DAG or ADMG.
    pub class_atoms: Arc<[CachedClassPosteriorAtomIdentification]>,
}

/// One identified atom in a prepared DBN graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedDbnPosteriorAtomIdentification {
    /// Collision-free, position-derived key used by the effect envelope.
    ///
    /// [`GraphPosterior::graph_keys`] default to contemporaneous adjacency
    /// masks, which are not unique for DBN atoms that differ only in lagged
    /// edges. The execution key is therefore local to this frozen cache.
    pub key: u64,
    /// Per-atom selected estimand.
    pub estimand: IdentifiedEstimand,
    /// Full temporal per-atom identification result.
    pub identification: IdentificationResult,
    /// Finite-unfolding indexer produced with [`Self::identification`].
    pub indexer: TemporalIndexer,
    /// Per-horizon `I(h)` for a mediation or temporal-response atom. Contrast
    /// atoms leave this empty and use [`Self::identification`] / [`Self::indexer`]
    /// for the query horizon. A union of these sets across atoms is not a shared
    /// adjustment set.
    pub horizons: Option<CachedTemporalIdentification>,
}

/// Why identification-time DBN atoms were marked unidentified.
///
/// Mass is retained on [`CachedDbnPosteriorIdentification::graphs`]; these
/// counts say *why* so a result does not have to treat unidentified mass as
/// an unexplained residual.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DbnIdentifyDemotion {
    /// `temporal_dag_from_dbn_masks` rejected the atom.
    pub invalid_graph: usize,
    /// Temporal identification returned an error.
    pub identify_failed: usize,
    /// Identification status is not a licensed identified status.
    pub not_identified: usize,
    /// No selected estimand (empty list or selector refusal).
    pub no_estimand: usize,
}

impl DbnIdentifyDemotion {
    pub(crate) fn total(&self) -> usize {
        self.invalid_graph + self.identify_failed + self.not_identified + self.no_estimand
    }

    pub(crate) fn summary(&self, prepare: usize, fit: usize, draws: usize) -> String {
        let estimate = prepare + fit + draws;
        format!(
            "identify_unidentified={} (invalid_graph={} identify_failed={} not_identified={} no_estimand={}); estimate_demoted={estimate} (prepare={prepare} fit={fit} draws={draws})",
            self.total(),
            self.invalid_graph,
            self.identify_failed,
            self.not_identified,
            self.no_estimand,
        )
    }
}

enum DbnAtomOutcome {
    InvalidGraph,
    IdentifyFailed,
    NotIdentified,
    NoEstimand,
    Identified(CachedDbnPosteriorAtomIdentification),
}

/// Prepare-time identification for every atom in a DBN graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedDbnPosteriorIdentification {
    /// Frozen weights and keys; mediation flags describe eligibility at any
    /// cached horizon until projected with `mediation_horizon`.
    pub graphs: WeightedGraphSamples,
    /// Identified atoms, in posterior order. Unidentified atoms remain in
    /// [`Self::graphs`] with [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedDbnPosteriorAtomIdentification]>,
    /// Identification-time demotions for contrasts or a projected mediation
    /// horizon. Unprojected mediation uses `horizon_demotions` instead.
    pub identify_demotion: DbnIdentifyDemotion,
    /// Mediation failure counts for each requested horizon. Empty for contrasts.
    pub horizon_demotions: Arc<[(u32, DbnIdentifyDemotion)]>,
}

impl CachedDbnPosteriorIdentification {
    /// Project the cached union of eligible mediation atoms onto one horizon.
    /// An atom's failure at another horizon never changes this horizon's mass.
    pub fn mediation_horizon(&self, horizon: u32) -> Result<Self, CausalError> {
        let identify_demotion = self
            .horizon_demotions
            .iter()
            .find(|(h, _)| *h == horizon)
            .map(|(_, counts)| counts.clone())
            .ok_or_else(|| CausalError::Compile {
                message: format!("DBN mediation cache missing I({horizon})"),
            })?;
        let atoms: Vec<_> = self
            .atoms
            .iter()
            .filter_map(|atom| {
                let entry = atom.horizons.as_ref()?.get(horizon)?.clone();
                Some(CachedDbnPosteriorAtomIdentification {
                    key: atom.key,
                    estimand: entry.estimand.clone(),
                    identification: entry.identification.clone(),
                    indexer: entry.indexer.clone(),
                    horizons: Some(CachedTemporalIdentification { by_horizon: Arc::from([entry]) }),
                })
            })
            .collect();
        let eligible: std::collections::HashSet<_> = atoms.iter().map(|atom| atom.key).collect();
        let flags: Vec<_> = self
            .graphs
            .graph_keys
            .iter()
            .map(|key| {
                if eligible.contains(key) {
                    GraphIdentFlag::Identified
                } else {
                    GraphIdentFlag::Unidentified
                }
            })
            .collect();
        let graphs = WeightedGraphSamples::new(
            Arc::clone(&self.graphs.weights),
            flags,
            Arc::clone(&self.graphs.graph_keys),
        )
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let horizon_demotions = Arc::from([(horizon, identify_demotion.clone())]);
        Ok(Self { graphs, atoms: atoms.into(), identify_demotion, horizon_demotions })
    }
}

/// Class-envelope identification for one TemporalCpdag/Pag posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassPosteriorAtomIdentification {
    /// Collision-free, position-derived key used by the effect envelope.
    pub key: u64,
    /// Envelope-level identification (status, diagnostics, assumptions).
    pub identification: IdentificationResult,
    /// Shared estimand when every identified completion agrees.
    pub invariant: Option<IdentifiedEstimand>,
    /// Completions + unfold indexers for this class atom.
    pub envelope: antecedent_identify::TemporalClassEnvelope,
    /// Envelope identified weight before posterior mixing.
    pub identified_weight: f64,
    /// Completions whose search was capped.
    pub truncated_completions: usize,
}

/// Prepare-time identification for TemporalCpdag/Pag graph-posterior Pulse/Sustained.
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified class atoms. Unidentified atoms remain in [`Self::graphs`].
    pub class_atoms: Arc<[CachedTemporalClassPosteriorAtomIdentification]>,
}

/// Identify unique adjacency masks under `ctx.parallelism`, then reduce in graph order.
fn identify_unique_adjacency_masks<T, F>(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    identify: F,
) -> Result<std::collections::HashMap<u64, T>, CausalError>
where
    T: Send,
    F: Fn(u64, &ExecutionContext) -> Result<T, CausalError> + Sync,
{
    let mut unique = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for i in 0..posterior.n_graphs {
        let mask = posterior.adjacency[i];
        if seen.insert(mask) {
            unique.push(mask);
        }
    }
    let values = ctx.map_indexed(unique.len(), |i, inner| identify(unique[i], inner))?;
    Ok(unique.into_iter().zip(values).collect())
}

/// One result per posterior position, under `ctx.parallelism`.
fn map_posterior_graphs<T, F>(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    f: F,
) -> Result<Vec<T>, CausalError>
where
    T: Send,
    F: Fn(usize, &ExecutionContext) -> Result<T, CausalError> + Sync,
{
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    ctx.map_indexed(posterior.n_graphs, |i, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        f(i, inner)
    })
}

/// Identify every atom in a static graph posterior and retain its original mass.
///
/// This is shared by fresh execution and [`Study::prepare`]. Calling it for a
/// fresh run performs identification; prepared estimate clicks clone the result
/// stored on the handle instead.
pub(crate) fn build_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{
        DEFAULT_IDENTIFIER_ID, EstimatorId, identify_static, select_estimand,
    };

    // A DBN posterior's contemporaneous masks are valid DAGs, but identifying a
    // static effect on them alone would drop every lagged confounder. That is
    // a different coordinate (temporal Pulse / Sustained), so fail closed.
    if posterior.lag_masks.is_some() || posterior.max_lag.is_some() {
        return Err(CausalError::Unsupported {
            message: "static AverageEffect over a graph posterior requires static DAG atoms; \
                      this posterior carries DBN lag structure, so it needs a temporal \
                      Pulse / single-step Sustained query",
        });
    }
    match posterior.atom_kind {
        antecedent_discovery::GraphPosteriorAtomKind::Dag => {}
        antecedent_discovery::GraphPosteriorAtomKind::Admg => {
            return build_admg_graph_posterior_identification_cache(posterior, query, ctx);
        }
        _ => return build_class_graph_posterior_identification_cache(posterior, query, ctx),
    }
    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(dag) = dag_from_adjacency_mask(mask, posterior.n_vars) else {
                return Ok(None);
            };
            let Ok(identification) = identify_static(DEFAULT_IDENTIFIER_ID, &dag, query) else {
                return Ok(None);
            };
            if !super::execute::identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                return Ok(None);
            }
            let Ok(estimand) = select_estimand(&identification, EstimatorId::LinearAdjustmentAte)
                .or_else(|_| select_estimand(&identification, EstimatorId::BayesianGcomp))
            else {
                return Ok(None);
            };
            Ok(Some((estimand, identification)))
        })()
    })?;
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = by_mask.get(&mask).cloned().flatten();
        // A posterior may list the same graph more than once (one entry per
        // sample). Every entry keeps its own weight and flag in `graphs`, but
        // consumers weight an atom by the combined mass of its key, so each
        // key contributes exactly one atom.
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            if first_for_key {
                atoms.push(CachedGraphPosteriorAtomIdentification {
                    key,
                    estimand,
                    identification,
                });
            }
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

/// Identify every ADMG posterior atom with general ID + functional.effect.
///
/// ADMG atoms are single graphs, not MEC completions. Unidentified mass stays
/// on the atom; there is no completion enumeration to mix.
/// Identify every ADMG posterior atom with general ID + functional.effect for a
/// static response query.
pub(crate) fn build_admg_graph_posterior_response_identification_cache(
    posterior: &GraphPosterior,
    query: &ResponseQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{
        DEFAULT_ADMG_IDENTIFIER_ID, EstimatorId, identify_admg_query, select_estimand,
    };
    use antecedent_discovery::admg_from_adjacency_mask;

    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    // `estimate_admg_posterior_atom_response` reuses this atom's cached
    // identification/estimand verbatim as the *first grid level's* claim (it
    // only re-identifies per level from the second level on). A MeanCurve
    // query identified whole produces one general.id estimand per grid
    // level, and `select_estimand` then has no unique estimator match to
    // pick among them (they all report the same method). Cache the first
    // level's InterventionResponse claim instead, which is what downstream
    // code actually consumes and — being a single intervention level — is
    // exactly what `select_estimand` can disambiguate.
    let causal_query = match &query.functional {
        antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
            let first_level = treatment
                .grid
                .values()
                .map_err(|e| CausalError::Compile { message: e.to_string() })?
                .into_iter()
                .next()
                .ok_or_else(|| CausalError::Compile {
                    message: "MeanCurve response requires a non-empty evaluation grid".into(),
                })?;
            let mut level_query = query.clone();
            level_query.functional = antecedent_core::ResponseFunctional::InterventionResponse {
                outcome: *outcome,
                interventions: Arc::from([Intervention::set(
                    treatment.variable,
                    Value::f64(first_level),
                )]),
            };
            CausalQuery::Response(level_query)
        }
        _ => CausalQuery::Response(query.clone()),
    };
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(admg) = admg_from_adjacency_mask(mask, posterior.n_vars) else {
                return Ok(None);
            };
            let Ok(identification) =
                identify_admg_query(DEFAULT_ADMG_IDENTIFIER_ID, &admg, &causal_query)
            else {
                return Ok(None);
            };
            if !super::execute::identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                return Ok(None);
            }
            let Ok(estimand) = select_estimand(&identification, EstimatorId::FunctionalEffect)
            else {
                return Ok(None);
            };
            Ok(Some((estimand, identification)))
        })()
    })?;
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = by_mask.get(&mask).cloned().flatten();
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            if first_for_key {
                atoms.push(CachedGraphPosteriorAtomIdentification {
                    key,
                    estimand,
                    identification,
                });
            }
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

fn build_admg_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{
        DEFAULT_ADMG_IDENTIFIER_ID, EstimatorId, identify_admg, select_estimand,
    };
    use antecedent_discovery::admg_from_adjacency_mask;

    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(admg) = admg_from_adjacency_mask(mask, posterior.n_vars) else {
                return Ok(None);
            };
            let Ok(identification) = identify_admg(DEFAULT_ADMG_IDENTIFIER_ID, &admg, query) else {
                return Ok(None);
            };
            if !super::execute::identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                return Ok(None);
            }
            let Ok(estimand) = select_estimand(&identification, EstimatorId::FunctionalEffect)
            else {
                return Ok(None);
            };
            Ok(Some((estimand, identification)))
        })()
    })?;
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = by_mask.get(&mask).cloned().flatten();
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            if first_for_key {
                atoms.push(CachedGraphPosteriorAtomIdentification {
                    key,
                    estimand,
                    identification,
                });
            }
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

/// Identify every CPDAG/PAG posterior atom with the existing class ATE envelope.
fn build_class_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER_ID, identify_cpdag, identify_pag};
    use antecedent_discovery::{cpdag_from_adjacency_mask, pag_from_adjacency_mask};

    super::execute::report_identify_compute(ctx);
    let resolved = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let mask = posterior.adjacency[i];
        let mark = posterior.mark_masks.as_ref().map_or(0, |marks| marks[i]);
        let key = posterior.graph_keys[i];
        let cached = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = cpdag_from_adjacency_mask(mask, posterior.n_vars) else {
                    return Ok(None);
                };
                Ok::<_, CausalError>(
                    identify_cpdag(DEFAULT_PAG_IDENTIFIER_ID, &cpdag, query)
                        .ok()
                        .map(|envelope| cache_class_envelope(key, query, envelope)),
                )
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Ok(pag) = pag_from_adjacency_mask(mask, mark, posterior.n_vars) else {
                    return Ok(None);
                };
                Ok(identify_pag(DEFAULT_PAG_IDENTIFIER_ID, &pag, query)
                    .ok()
                    .map(|envelope| cache_class_envelope(key, query, envelope)))
            }
            _ => Ok(None),
        }?;
        Ok(cached)
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut class_atoms = Vec::new();
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for (i, cached) in resolved.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        let Some(cached) = cached else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        // Envelope-level GraphDependent still carries identified completion mass.
        // Only atoms with zero identified completion weight are posterior-unidentified.
        let identified = cached.identified_weight > 0.0;
        if identified {
            flags.push(GraphIdentFlag::Identified);
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
        if first_for_key && identified {
            class_atoms.push(cached);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from([]),
        class_atoms: Arc::from(class_atoms),
    })
}

fn cache_class_envelope<G>(
    key: u64,
    query: &AverageEffectQuery,
    envelope: IdentificationEnvelope<G>,
) -> CachedClassPosteriorAtomIdentification {
    use crate::strategy_table::{EstimatorId, select_estimand};

    let identification = super::execute::envelope_to_identification_result(&envelope, query);
    let cases: Vec<CachedClassPosteriorCase> = envelope
        .cases
        .iter()
        .map(|case| {
            let estimand = if super::execute::identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                select_estimand(&case.result, EstimatorId::LinearAdjustmentAte)
                    .or_else(|_| select_estimand(&case.result, EstimatorId::BayesianGcomp))
                    .ok()
                    .or_else(|| case.result.estimands.first().cloned())
            } else {
                None
            };
            CachedClassPosteriorCase { weight: case.weight.0, status: case.result.status, estimand }
        })
        .collect();
    CachedClassPosteriorAtomIdentification {
        key,
        identification,
        invariant: envelope.invariant,
        cases: Arc::from(cases),
        identified_weight: envelope.identified_weight.0,
        truncated_completions: envelope.truncated_completions,
    }
}

/// Identify every atom in a DBN posterior and retain its original mass.
///
/// The temporal indexer is part of the cached identification product: rebuilding
/// it from refreshed data would silently couple identification to the estimate
/// click even though graph and query are frozen.
pub(crate) fn build_dbn_posterior_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    use crate::strategy_table::select_estimand;

    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    super::execute::report_identify_compute(ctx);
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        // A DBN atom is the pair (contemporaneous mask, lag mask), but the
        // public GraphPosterior constructor keys atoms by contemporaneous mask
        // alone. Use posterior position as a collision-free execution key so
        // lag-distinct atoms keep distinct fits, flags, and weights inside the
        // effect envelope. This key is internal and does not alter the public
        // GraphPosterior representation.
        let key = dbn_envelope_key(i)?;
        let Ok(graph) = temporal_dag_from_dbn_masks(
            posterior.adjacency[i],
            lag_masks[i],
            posterior.n_vars,
            max_lag,
            variables,
        ) else {
            return Ok(DbnAtomOutcome::InvalidGraph);
        };
        let Ok(temporal) = TemporalBackdoorIdentifier::new().identify_temporal(&graph, query)
        else {
            return Ok(DbnAtomOutcome::IdentifyFailed);
        };
        let identification = temporal.result;
        if !super::execute::identification_status_ok_for_case(identification.status) {
            return Ok(DbnAtomOutcome::NotIdentified);
        }
        if identification.estimands.is_empty() {
            return Ok(DbnAtomOutcome::NoEstimand);
        }
        let estimator = dbn_temporal_effect_estimator(query);
        let Ok(estimand) = select_estimand(&identification, estimator) else {
            return Ok(DbnAtomOutcome::NoEstimand);
        };
        Ok(DbnAtomOutcome::Identified(CachedDbnPosteriorAtomIdentification {
            key,
            estimand,
            identification,
            indexer: temporal.indexer,
            horizons: None,
        }))
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for (i, outcome) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        match outcome {
            DbnAtomOutcome::InvalidGraph => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.invalid_graph += 1;
            }
            DbnAtomOutcome::IdentifyFailed => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.identify_failed += 1;
            }
            DbnAtomOutcome::NotIdentified => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.not_identified += 1;
            }
            DbnAtomOutcome::NoEstimand => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.no_estimand += 1;
            }
            DbnAtomOutcome::Identified(atom) => {
                flags.push(GraphIdentFlag::Identified);
                atoms.push(atom);
            }
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        identify_demotion,
        horizon_demotions: Arc::from([]),
    })
}

/// Identify every TemporalCpdag/Pag posterior atom with the class envelope.
///
/// Each atom is reconstructed from adjacency + lag masks (+ mark masks for
/// Pag). Completions that disagree keep envelope status; unidentified class
/// mass stays on [`CachedTemporalClassPosteriorIdentification::graphs`].
/// Completion enumeration is not posterior probability.
pub(crate) fn build_temporal_class_posterior_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &TemporalEffectQuery,
    max_completions: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<CachedTemporalClassPosteriorIdentification, CausalError> {
    use crate::strategy_table::{
        DEFAULT_PAG_IDENTIFIER_ID, identify_temporal_cpdag_configured,
        identify_temporal_pag_configured,
    };

    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior.max_lag.ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing max_lag".into(),
    })?;
    if !matches!(
        posterior.atom_kind,
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag
    ) {
        return Err(CausalError::Compile {
            message: "temporal class posterior requires Cpdag or Pag atom_kind".into(),
        });
    }
    super::execute::report_identify_compute(ctx);
    let mut config = antecedent_identify::GeneralizedAdjustmentConfig::default();
    if let Some(max) = max_completions {
        config.max_completions = max;
    }
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let key = dbn_envelope_key(i)?;
        let mark = posterior.mark_masks.as_ref().map_or(0, |marks| marks[i]);
        let cached = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = temporal_cpdag_from_dbn_masks(
                    posterior.adjacency[i],
                    lag_masks[i],
                    posterior.n_vars,
                    max_lag,
                    variables,
                ) else {
                    return Ok(None);
                };
                Ok::<_, CausalError>(
                    identify_temporal_cpdag_configured(
                        DEFAULT_PAG_IDENTIFIER_ID,
                        &cpdag,
                        query,
                        config.clone(),
                    )
                    .ok()
                    .map(|envelope| cache_temporal_class_atom(key, query, envelope)),
                )
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Ok(pag) = temporal_pag_from_dbn_masks(
                    posterior.adjacency[i],
                    lag_masks[i],
                    mark,
                    posterior.n_vars,
                    max_lag,
                    variables,
                ) else {
                    return Ok(None);
                };
                Ok(identify_temporal_pag_configured(
                    DEFAULT_PAG_IDENTIFIER_ID,
                    &pag,
                    query,
                    config.clone(),
                )
                .ok()
                .map(|envelope| cache_temporal_class_atom(key, query, envelope)))
            }
            _ => Ok(None),
        }?;
        Ok(cached)
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut class_atoms = Vec::new();

    for (i, cached) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        let Some(cached) = cached else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        if cached.identified_weight > 0.0 {
            flags.push(GraphIdentFlag::Identified);
            class_atoms.push(cached);
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedTemporalClassPosteriorIdentification { graphs, class_atoms: Arc::from(class_atoms) })
}

fn cache_temporal_class_atom(
    key: u64,
    query: &TemporalEffectQuery,
    envelope: antecedent_identify::TemporalClassEnvelope,
) -> CachedTemporalClassPosteriorAtomIdentification {
    let identification = super::execute::envelope_to_identification_result_for(
        &envelope.envelope,
        CausalQuery::TemporalEffect(query.clone()),
    );
    CachedTemporalClassPosteriorAtomIdentification {
        key,
        identification,
        invariant: envelope.envelope.invariant.clone(),
        identified_weight: envelope.envelope.identified_weight.0,
        truncated_completions: envelope.envelope.truncated_completions,
        envelope,
    }
}

/// Identify every DBN atom for a temporal [`CausalQuery::Response`].
///
/// Each atom is reconstructed as a [`TemporalDag`] and identified at every
/// requested horizon. An atom that fails any horizon stays unidentified;
/// unidentified mass is retained. Atoms are DAGs: there is no completion
/// enumeration.
pub(crate) fn build_dbn_posterior_response_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &ResponseQuery,
    estimator_id: crate::strategy_table::EstimatorId,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    super::execute::dbn_posterior_response_supported(query)?;
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN-posterior response requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) = query.functional.primary_pair().ok_or_else(|| {
        CausalError::Compile { message: "response query has no treatment/outcome pair".into() }
    })?;
    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    super::execute::report_identify_compute(ctx);
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let key = dbn_envelope_key(i)?;
        let Ok(graph) = temporal_dag_from_dbn_masks(
            posterior.adjacency[i],
            lag_masks[i],
            posterior.n_vars,
            max_lag,
            variables,
        ) else {
            return Ok(DbnAtomOutcome::InvalidGraph);
        };
        let Ok(horizons) = identify_temporal_response_horizons(
            &graph,
            treatment,
            outcome,
            temporal,
            &query.target_population,
            estimator_id,
            None,
            single_step_dose(query).ok().flatten(),
        ) else {
            return Ok(DbnAtomOutcome::IdentifyFailed);
        };
        if horizons.by_horizon.len() != temporal.horizons.len()
            || horizons.by_horizon.iter().any(|entry| {
                !super::execute::identification_status_ok_for_case(entry.identification.status)
                    || entry.identification.estimands.is_empty()
            })
        {
            return Ok(DbnAtomOutcome::NotIdentified);
        }
        let Some(first) = horizons.by_horizon.first() else {
            return Ok(DbnAtomOutcome::NoEstimand);
        };
        Ok(DbnAtomOutcome::Identified(CachedDbnPosteriorAtomIdentification {
            key,
            estimand: first.estimand.clone(),
            identification: first.identification.clone(),
            indexer: first.indexer.clone(),
            horizons: Some(horizons),
        }))
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for (i, outcome) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        match outcome {
            DbnAtomOutcome::InvalidGraph => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.invalid_graph += 1;
            }
            DbnAtomOutcome::IdentifyFailed => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.identify_failed += 1;
            }
            DbnAtomOutcome::NotIdentified => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.not_identified += 1;
            }
            DbnAtomOutcome::NoEstimand => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.no_estimand += 1;
            }
            DbnAtomOutcome::Identified(atom) => {
                flags.push(GraphIdentFlag::Identified);
                atoms.push(atom);
            }
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        identify_demotion,
        horizon_demotions: Arc::from([]),
    })
}

/// Identify every DBN atom for [`CausalQuery::Mediation`] (`TemporalMediationEffect`).
///
/// Each atom gets its own `I(h)` cache from that atom's reconstructed
/// [`TemporalDag`]. Adjustment sets are not unioned across atoms or horizons.
/// Identification failures stay unidentified; priors are not consulted.
pub(crate) fn build_dbn_posterior_mediation_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &MediationQuery,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    build_dbn_mediation_cache_with_identifier(
        posterior,
        variables,
        query,
        ctx,
        |_, _, graph, horizon_query| {
            identify_temporal_mediation_horizons(
                graph,
                horizon_query,
                crate::strategy_table::EstimatorId::BayesianTemporalMediation,
            )
        },
    )
}

fn build_dbn_mediation_cache_with_identifier(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &MediationQuery,
    ctx: &ExecutionContext,
    mut identify: impl FnMut(
        usize,
        u32,
        &TemporalDag,
        &MediationQuery,
    ) -> Result<CachedTemporalIdentification, CausalError>,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
    let mut horizon_demotions: Vec<_> =
        query.horizons.iter().map(|h| (*h, DbnIdentifyDemotion::default())).collect();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        let Ok(graph) = temporal_dag_from_dbn_masks(
            posterior.adjacency[i],
            lag_masks[i],
            posterior.n_vars,
            max_lag,
            variables,
        ) else {
            flags.push(GraphIdentFlag::Unidentified);
            for (_, counts) in &mut horizon_demotions {
                counts.invalid_graph += 1;
            }
            continue;
        };
        let mut entries = Vec::new();
        for (horizon, counts) in &mut horizon_demotions {
            let mut horizon_query = query.clone();
            horizon_query.horizons = Arc::from([*horizon]);
            let Ok(horizons) = identify(i, *horizon, &graph, &horizon_query) else {
                counts.identify_failed += 1;
                continue;
            };
            let Some(entry) = horizons.by_horizon.first() else {
                counts.no_estimand += 1;
                continue;
            };
            if !super::execute::identification_status_ok_for_case(entry.identification.status)
                || entry.identification.estimands.is_empty()
            {
                counts.not_identified += 1;
                continue;
            }
            entries.push(entry.clone());
        }
        let Some(first) = entries.first() else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        flags.push(GraphIdentFlag::Identified);
        atoms.push(CachedDbnPosteriorAtomIdentification {
            key,
            estimand: first.estimand.clone(),
            identification: first.identification.clone(),
            indexer: first.indexer.clone(),
            horizons: Some(CachedTemporalIdentification { by_horizon: entries.into() }),
        });
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: atoms.into(),
        identify_demotion: DbnIdentifyDemotion::default(),
        horizon_demotions: horizon_demotions.into(),
    })
}

fn dbn_temporal_effect_estimator(
    query: &TemporalEffectQuery,
) -> crate::strategy_table::EstimatorId {
    use crate::strategy_table::EstimatorId;
    if query.is_multi_step_sustained() {
        EstimatorId::TemporalSequentialGcomp
    } else {
        EstimatorId::TemporalLinearAdjustment
    }
}

/// Reconstruct the [`TemporalDag`] for a DBN envelope key (posterior index).
pub(crate) fn temporal_dag_from_dbn_atom(
    posterior: &GraphPosterior,
    key: u64,
    variables: &[antecedent_core::VariableId],
) -> Result<TemporalDag, CausalError> {
    let index = usize::try_from(key).map_err(|_| CausalError::Compile {
        message: "DBN envelope key does not fit a posterior index".into(),
    })?;
    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    if index >= posterior.n_graphs || index >= lag_masks.len() {
        return Err(CausalError::Compile { message: "DBN envelope key is out of range".into() });
    }
    temporal_dag_from_dbn_masks(
        posterior.adjacency[index],
        lag_masks[index],
        posterior.n_vars,
        max_lag,
        variables,
    )
    .map_err(|error| CausalError::Compile { message: error.to_string() })
}

fn dbn_envelope_key(index: usize) -> Result<u64, CausalError> {
    u64::try_from(index).map_err(|_| CausalError::Compile {
        message: "DBN posterior has too many atoms for envelope keys".into(),
    })
}

/// Prepare-time identification products for one temporal horizon.
#[derive(Clone, Debug)]
pub struct CachedTemporalHorizonIdentification {
    /// Requested horizon these products were identified for.
    pub horizon: u32,
    /// Unfolded backdoor identification at [`Self::horizon`].
    pub identification: IdentificationResult,
    /// Estimand selected for this horizon.
    pub estimand: IdentifiedEstimand,
    /// Finite-unfolding indexer paired with [`Self::identification`].
    pub indexer: TemporalIndexer,
}

/// Prepare-time temporal-backdoor identification (ADR 0021).
///
/// For temporal [`CausalQuery::Response`], identification + lag indexer are
/// frozen once per unique requested horizon. For scalar
/// [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained), they are
/// frozen for that query's horizon. For [`CausalQuery::Mediation`]
/// (`TemporalMediationEffect`), identification is frozen independently for
/// every requested horizon.
#[derive(Clone, Debug)]
pub struct CachedTemporalIdentification {
    /// One entry per unique requested horizon, in query order.
    pub by_horizon: Arc<[CachedTemporalHorizonIdentification]>,
}

impl CachedTemporalIdentification {
    /// Identification products for `horizon`, if prepared.
    #[must_use]
    pub fn get(&self, horizon: u32) -> Option<&CachedTemporalHorizonIdentification> {
        self.by_horizon.iter().find(|entry| entry.horizon == horizon)
    }
}

/// Data modality a [`PreparedStudy`] was compiled for.
///
/// The physical plan is modality-specific, so a handle prepared on one
/// modality must refuse data of another instead of running a different route
/// under the frozen plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedModality {
    /// Tabular rows: [`PreparedStudy::estimate`] / [`PreparedStudy::refresh`].
    Tabular,
    /// One time series (or event data on a regular grid):
    /// [`PreparedStudy::estimate_series`] / [`PreparedStudy::refresh_series`].
    Series,
    /// Multi-unit panel: [`PreparedStudy::estimate_panel`] / [`PreparedStudy::refresh_panel`].
    Panel,
    /// Multi-environment series: [`PreparedStudy::estimate_multi_env`] /
    /// [`PreparedStudy::refresh_multi_env`].
    MultiEnv,
}

impl PreparedModality {
    const fn label(self) -> &'static str {
        match self {
            Self::Tabular => "tabular",
            Self::Series => "series",
            Self::Panel => "panel",
            Self::MultiEnv => "multi_env",
        }
    }

    const fn estimate_entry(self) -> &'static str {
        match self {
            Self::Tabular => "estimate",
            Self::Series => "estimate_series",
            Self::Panel => "estimate_panel",
            Self::MultiEnv => "estimate_multi_env",
        }
    }
}

/// Durable handle: fixed schema, graph, query, and estimator; swap data and re-estimate.
///
/// Created via [`Study::prepare`]. Discovery / review-required graphs are refused —
/// prepare is for the interactive estimate click path on an already-accepted artifact.
///
/// **Frozen at prepare:** schema (names, types, order); graph / `AcceptedGraph`
/// or supplied graph-posterior atoms and weights; query identity; identifier;
/// observation / transport / interference assumptions; target-population
/// bindings.
///
/// **Estimate click:** same-schema data; estimator numeric knobs, latency,
/// seeds, bootstrap; `ExecutionContext` budget / cancellation. Does not
/// re-identify or recompile the logical plan.
///
/// **Refute click:** same frozen identification and estimand; schema-gated
/// data and suite. Currently [`CausalQuery::AverageEffect`] only.
///
/// **Re-prepare required:** any frozen field change, including schema
/// mismatch. Changing a frozen field on [`Self::refresh`] is an error, not a
/// silent recompile. `analyze` / [`Study::run`] is sugar over identify →
/// prepare → estimate.
#[derive(Clone, Debug)]
pub struct PreparedStudy<S = SampledPreparedState> {
    pub(crate) state: S,
}

/// One sealed execution choice for a prepared tabular click. `LegacyStudyDispatch`
/// names routes that still use the ordinary dispatcher; the checked variants
/// carry the complete operation retained by preparation.
#[derive(Clone, Debug)]
pub(crate) enum PreparedExecution {
    LegacyStudyDispatch,
    FunctionalEffect(CheckedFunctionalEffectOperation),
    PathSpecificEffect(CheckedPathSpecificEffectOperation),
    AdmgResponseCurve(CheckedAdmgResponseCurveOperation),
    CheckedLinear(CheckedLinearOperation),
    CheckedGlmAdjustment(CheckedGlmAdjustmentOperation),
    CheckedRd(CheckedRdOperation),
    CheckedPropensity(CheckedPropensityOperation),
    CheckedConditional(CheckedConditionalOperation),
    BayesianConditional(super::execute::CheckedBayesianConditionalOperation),
    GraphPosteriorEffect(CheckedGraphPosteriorEffect),
    CheckedAipw(CheckedAipwOperation),
    Counterfactual(super::execute::CheckedCounterfactualPlan),
    NestedCounterfactual(crate::gcm::NestedCounterfactualOperation),
    DerivativeResponse(super::execute::CheckedDerivativeResponseOperation),
    CellAipwResponse(super::execute::CheckedCellAipwResponseOperation),
    StaticDagResponse(super::execute::CheckedStaticDagResponseOperation),
    StaticMediation(super::execute::CheckedStaticMediationOperation),
    BayesianStaticMediation(super::execute::CheckedBayesianStaticMediationOperation),
    Attribution(super::execute::CheckedAttributionOperation),
    TemporalDagResponse(super::execute::CheckedTemporalResponseExecution),
    TemporalDagEffect(super::execute::CheckedTemporalEffectExecution),
    TemporalMediation(super::execute::CheckedTemporalMediationOperation),
    TemporalClassEffect(super::execute::CheckedTemporalClassEffectExecution),
    Distribution(CheckedDistributionOperation),
    BayesianGcomp(super::execute::CheckedBayesianDagAteExecution),
    StaticResponseCurve(CheckedStaticResponseCurve),
    FrontDoorLinear(CheckedFrontDoorOperation),
    Iv(CheckedIvOperation),
}

/// Exactly one checked product may supply a prepared result contract. This is
/// deliberately a sum type: passing seven independent optional receipts to
/// contract construction could describe a procedure no prepared plan selected.
#[derive(Clone, Copy)]
pub(crate) enum CheckedProgramBinding<'a> {
    None,
    Aipw(&'a antecedent_estimate::CheckedAipwPreparation),
    FrontDoor(&'a antecedent_estimate::CheckedFrontDoorPreparation),
    Iv(&'a antecedent_estimate::CheckedIvPreparation),
    Linear(&'a CheckedLinearOperation),
    FunctionalEffect(&'a antecedent_expr::FunctionalProgram),
    FunctionalResponse(&'a [CheckedFunctionalEffectResponseMember]),
    Distribution(&'a antecedent_expr::FunctionalProgram),
    NestedCounterfactual(&'a crate::gcm::NestedCounterfactualOperation),
}

impl PreparedExecution {
    pub(crate) fn program_binding(&self) -> CheckedProgramBinding<'_> {
        match self {
            Self::CheckedAipw(operation) => CheckedProgramBinding::Aipw(&operation.preparation),
            Self::FrontDoorLinear(operation) => {
                CheckedProgramBinding::FrontDoor(&operation.preparation)
            }
            Self::Iv(operation) => CheckedProgramBinding::Iv(operation.preparation()),
            Self::CheckedLinear(operation) => CheckedProgramBinding::Linear(operation),
            Self::CheckedGlmAdjustment(_)
            | Self::CheckedRd(_)
            | Self::CheckedPropensity(_)
            | Self::CheckedConditional(_) => CheckedProgramBinding::None,
            Self::FunctionalEffect(operation) => {
                CheckedProgramBinding::FunctionalEffect(operation.prepared.program())
            }
            Self::PathSpecificEffect(operation) => {
                CheckedProgramBinding::FunctionalEffect(operation.prepared.program())
            }
            Self::AdmgResponseCurve(operation) => {
                CheckedProgramBinding::FunctionalResponse(&operation.members)
            }
            Self::Distribution(operation) => {
                CheckedProgramBinding::Distribution(operation.prepared.program())
            }
            Self::NestedCounterfactual(operation) => {
                CheckedProgramBinding::NestedCounterfactual(operation)
            }
            Self::LegacyStudyDispatch
            | Self::Counterfactual(_)
            | Self::DerivativeResponse(_)
            | Self::CellAipwResponse(_)
            | Self::StaticDagResponse(_)
            | Self::StaticMediation(_)
            | Self::BayesianStaticMediation(_)
            | Self::Attribution(_)
            | Self::TemporalDagResponse(_)
            | Self::TemporalDagEffect(_)
            | Self::TemporalMediation(_)
            | Self::TemporalClassEffect(_)
            | Self::BayesianGcomp(_)
            | Self::BayesianConditional(_)
            | Self::StaticResponseCurve(_) => CheckedProgramBinding::None,
            Self::GraphPosteriorEffect(_) => CheckedProgramBinding::None,
        }
    }

    /// Rebind only the checked receipts that still execute through the shared
    /// tabular dispatcher. Direct operations own their rebind at execution.
    /// Keeping the variant selection here prevents a refreshed receipt from
    /// accidentally changing the procedure selected at preparation.
    fn rebind_for_dispatch(&self, data: &TabularData) -> Result<Self, CausalError> {
        match self {
            Self::CheckedLinear(operation) => Ok(Self::CheckedLinear(operation.rebind(data)?)),
            Self::CheckedGlmAdjustment(operation) => {
                Ok(Self::CheckedGlmAdjustment(operation.rebind(data)?))
            }
            Self::CheckedRd(operation) => Ok(Self::CheckedRd(operation.rebind(data)?)),
            Self::CheckedAipw(operation) => Ok(Self::CheckedAipw(operation.rebind(data)?)),
            Self::FrontDoorLinear(operation) => Ok(Self::FrontDoorLinear(operation.rebind(data)?)),
            Self::Iv(operation) => Ok(Self::Iv(operation.rebind(data)?)),
            other => Ok(other.clone()),
        }
    }

    pub(crate) fn checked_linear(
        &self,
    ) -> Option<&antecedent_estimate::CheckedLinearAdjustmentAte> {
        self.linear_operation().map(|operation| &operation.preparation)
    }
    pub(crate) fn linear_operation(&self) -> Option<&CheckedLinearOperation> {
        if let Self::CheckedLinear(value) = self { Some(value) } else { None }
    }
    pub(crate) fn checked_aipw(&self) -> Option<&antecedent_estimate::CheckedAipwPreparation> {
        self.aipw_operation().map(|operation| &operation.preparation)
    }
    pub(crate) fn aipw_operation(&self) -> Option<&CheckedAipwOperation> {
        if let Self::CheckedAipw(value) = self { Some(value) } else { None }
    }
    pub(crate) fn nested_counterfactual(
        &self,
    ) -> Option<&crate::gcm::NestedCounterfactualOperation> {
        if let Self::NestedCounterfactual(value) = self { Some(value) } else { None }
    }
    pub(crate) fn distribution(&self) -> Option<&CheckedDistributionOperation> {
        if let Self::Distribution(value) = self { Some(value) } else { None }
    }
    pub(crate) fn functional_effect_operation(&self) -> Option<&CheckedFunctionalEffectOperation> {
        if let Self::FunctionalEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn path_specific_effect_operation(
        &self,
    ) -> Option<&CheckedPathSpecificEffectOperation> {
        if let Self::PathSpecificEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn admg_response_curve(&self) -> Option<&CheckedAdmgResponseCurveOperation> {
        if let Self::AdmgResponseCurve(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_gcomp(&self) -> Option<&CheckedBayesianGcompOperation> {
        if let Self::BayesianGcomp(value) = self { Some(value.operation()) } else { None }
    }
    pub(crate) fn bayesian_conditional(
        &self,
    ) -> Option<&super::execute::CheckedBayesianConditionalOperation> {
        if let Self::BayesianConditional(value) = self { Some(value) } else { None }
    }
    pub(crate) fn response_curve(&self) -> Option<&CheckedStaticResponseCurve> {
        if let Self::StaticResponseCurve(value) = self { Some(value) } else { None }
    }
    pub(crate) fn static_dag_response(
        &self,
    ) -> Option<&super::execute::CheckedStaticDagResponseOperation> {
        if let Self::StaticDagResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn static_mediation(
        &self,
    ) -> Option<&super::execute::CheckedStaticMediationOperation> {
        if let Self::StaticMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn bayesian_static_mediation(
        &self,
    ) -> Option<&super::execute::CheckedBayesianStaticMediationOperation> {
        if let Self::BayesianStaticMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn attribution(&self) -> Option<&super::execute::CheckedAttributionOperation> {
        if let Self::Attribution(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_dag_response(
        &self,
    ) -> Option<&super::execute::CheckedTemporalResponseExecution> {
        if let Self::TemporalDagResponse(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_dag_effect(
        &self,
    ) -> Option<&super::execute::CheckedTemporalEffectExecution> {
        if let Self::TemporalDagEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_mediation(
        &self,
    ) -> Option<&super::execute::CheckedTemporalMediationOperation> {
        if let Self::TemporalMediation(value) = self { Some(value) } else { None }
    }
    pub(crate) fn temporal_class_effect(
        &self,
    ) -> Option<&super::execute::CheckedTemporalClassEffectExecution> {
        if let Self::TemporalClassEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn propensity(&self) -> Option<&CheckedPropensityOperation> {
        if let Self::CheckedPropensity(value) = self { Some(value) } else { None }
    }
    pub(crate) fn conditional(&self) -> Option<&CheckedConditionalOperation> {
        if let Self::CheckedConditional(value) = self { Some(value) } else { None }
    }
    pub(crate) fn graph_posterior_effect(&self) -> Option<&CheckedGraphPosteriorEffect> {
        if let Self::GraphPosteriorEffect(value) = self { Some(value) } else { None }
    }
    pub(crate) fn frontdoor_linear(&self) -> Option<&CheckedFrontDoorOperation> {
        if let Self::FrontDoorLinear(value) = self { Some(value) } else { None }
    }
    pub(crate) fn iv(&self) -> Option<&CheckedIvOperation> {
        if let Self::Iv(value) = self { Some(value) } else { None }
    }
}

pub(crate) fn checked_aipw_fitter(study: &Study) -> AipwAte {
    match &study.estimator_spec {
        Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => (**config).clone(),
        _ => {
            let mut fitter = AipwAte::new();
            fitter.bootstrap_replicates = study.bootstrap_replicates;
            if let Some(overlap) = study.overlap_policy {
                fitter.overlap = overlap;
            }
            fitter.population_registry = study.population_registry.clone();
            fitter
        }
    }
}

/// Retained state for sampled-data modalities of the common prepared handle.
#[derive(Clone, Debug)]
pub struct SampledPreparedState {
    /// Frozen analysis config (data slot replaced only by checked refresh).
    analysis: Study,
    /// Data-independent contract layers, compiled once per handle state.
    program_cache: std::sync::OnceLock<Arc<super::contract::ProgramPayloads>>,
    /// Ready physical plan from the prepare-time compile (never recompiled on refresh).
    plan: PhysicalExecutionPlan,
    /// Schema fingerprint from prepare-time data.
    schema: CausalSchema,
    /// Data modality frozen at prepare; each estimate / refresh entry point
    /// accepts only its own modality.
    modality: PreparedModality,
    /// Sampling regularity frozen for series and panel prepares (`None` = tabular).
    time_regularity: Option<antecedent_data::SamplingRegularity>,
    /// Cross-fitted AIPW scores frozen at prepare when the cell can export them.
    score_table: Option<antecedent_estimate::ScoreTable>,
    /// Checked causal-to-linear lowering retained independently of the builder.
    /// Sealed route selection and its checked operation, or explicitly named legacy dispatch.
    execution: PreparedExecution,
}

impl std::ops::Deref for PreparedStudy {
    type Target = SampledPreparedState;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
impl std::ops::DerefMut for PreparedStudy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl PreparedStudy {
    pub(crate) fn checked_program_binding(&self) -> CheckedProgramBinding<'_> {
        self.execution.program_binding()
    }

    /// Retained GLM target design for the checked DAG adjustment route.
    #[must_use]
    pub fn checked_glm_adjustment(&self) -> Option<&antecedent_estimate::PreparedGlmProblem> {
        if let PreparedExecution::CheckedGlmAdjustment(operation) = &self.execution {
            Some(&operation.preparation)
        } else {
            None
        }
    }

    /// Retained local boundary and design for the checked sharp RD route.
    #[must_use]
    pub fn checked_rd_preparation(&self) -> Option<&antecedent_estimate::CheckedRdPreparation> {
        if let PreparedExecution::CheckedRd(operation) = &self.execution {
            Some(&operation.preparation)
        } else {
            None
        }
    }

    pub(crate) fn has_sealed_glm_operation(&self) -> bool {
        matches!(&self.execution, PreparedExecution::CheckedGlmAdjustment(op) if op.sealed_for_direct_execution())
    }

    pub(crate) fn has_checked_rd_operation(&self) -> bool {
        matches!(&self.execution, PreparedExecution::CheckedRd(_))
    }

    /// Whether this prepared handle owns the complete operation for a family
    /// whose one-shot facade must execute through preparation as well.
    pub(crate) fn has_complete_program_operation(&self, query: &CausalQuery) -> bool {
        matches!(
            (&self.execution, query),
            (PreparedExecution::Distribution(_), CausalQuery::Distribution(_))
                | (PreparedExecution::PathSpecificEffect(_), CausalQuery::PathSpecific(_))
                | (
                    PreparedExecution::NestedCounterfactual(_),
                    CausalQuery::NestedCounterfactual(_)
                )
        )
    }

    /// Retained checked shared-exogenous natural direct effect operation.
    #[must_use]
    pub fn checked_nested_counterfactual_operation(
        &self,
    ) -> Option<&crate::gcm::NestedCounterfactualOperation> {
        self.execution.nested_counterfactual()
    }

    /// Checked Bayesian g-computation receipt for a static mean ATE, when this
    /// prepared handle uses that licensed route.
    #[must_use]
    pub fn checked_bayesian_gcomp_operation(&self) -> Option<&CheckedBayesianGcompOperation> {
        self.execution.bayesian_gcomp()
    }

    /// Validation procedure frozen with the checked Bayesian DAG effect route.
    #[must_use]
    pub fn checked_bayesian_gcomp_validation(&self) -> Option<RefuteSuite> {
        match &self.execution {
            PreparedExecution::BayesianGcomp(operation) => Some(operation.validation()),
            _ => None,
        }
    }

    /// Checked Bayesian conditional-effect plan, when this handle owns one.
    #[must_use]
    pub fn checked_bayesian_conditional_operation(&self) -> Option<CheckedBayesianConditionalInfo> {
        let operation = self.execution.bayesian_conditional()?;
        Some(CheckedBayesianConditionalInfo {
            query: operation.query().clone(),
            inference: operation.inference().clone(),
            validation: operation.validation(),
            modifier_roles: Arc::from(operation.modifier_roles()),
            identification_status: operation.identification().status,
            estimand: operation.estimand().clone(),
        })
    }
    /// Checked linear adjustment lowering retained by this prepared handle.
    ///
    /// `None` means this route has not migrated to the checked adjustment path;
    /// it does not imply that another estimator is unsupported.
    #[must_use]
    pub fn checked_linear_adjustment(
        &self,
    ) -> Option<&antecedent_estimate::CheckedLinearAdjustmentAte> {
        self.execution.checked_linear()
    }

    /// Checked AIPW lowering retained for the supported back-door mean ATE.
    #[must_use]
    pub fn checked_aipw_ate(&self) -> Option<&antecedent_estimate::CheckedAipwPreparation> {
        self.execution.checked_aipw()
    }

    pub(crate) fn has_sealed_aipw_operation(&self) -> bool {
        self.execution
            .aipw_operation()
            .is_some_and(CheckedAipwOperation::sealed_for_direct_execution)
    }

    pub(crate) fn has_checked_counterfactual_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::Counterfactual(_))
    }

    /// The retained graph and cross-world target for a checked counterfactual route.
    #[must_use]
    pub fn checked_counterfactual_operation(
        &self,
    ) -> Option<&crate::gcm::CheckedCounterfactualOperation> {
        if let PreparedExecution::Counterfactual(plan) = &self.execution {
            Some(plan.operation())
        } else {
            None
        }
    }

    pub(crate) fn has_checked_derivative_response_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::DerivativeResponse(_))
    }

    pub(crate) fn has_checked_static_dag_response_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::StaticDagResponse(_))
    }

    /// Inspect the sealed static DAG mediation target and procedure.
    #[must_use]
    pub fn checked_static_mediation_info(&self) -> Option<CheckedStaticMediationInfo> {
        let (query, estimand, program, bootstrap_replicates, validation, graph) =
            self.execution.static_mediation()?.inspect();
        let mut edges: Vec<_> = graph
            .edges()
            .filter_map(|edge| edge.parent_child())
            .map(|(a, b)| (a.raw(), b.raw()))
            .collect();
        edges.sort_unstable();
        Some(CheckedStaticMediationInfo {
            query: query.clone(),
            identifier: crate::strategy_table::IdentifierId::PathSpecificNatural,
            estimator: crate::strategy_table::EstimatorId::StaticMediationLinear,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            functional_roots: (program.mapping().source.raw(), program.mapping().executable.raw()),
            graph_edges: edges.into(),
            bootstrap_replicates,
            validation,
        })
    }

    /// Inspect the retained Bayesian DAG mediation target, model, and validation.
    #[must_use]
    pub fn checked_bayesian_static_mediation_info(&self) -> Option<CheckedStaticMediationInfo> {
        let operation = self.execution.bayesian_static_mediation()?;
        let (query, estimand, program, validation) = operation.inspect();
        let mut edges: Vec<_> = operation
            .graph()
            .edges()
            .filter_map(|edge| edge.parent_child())
            .map(|(a, b)| (a.raw(), b.raw()))
            .collect();
        edges.sort_unstable();
        Some(CheckedStaticMediationInfo {
            query: query.clone(),
            identifier: crate::strategy_table::IdentifierId::PathSpecificNatural,
            estimator: crate::strategy_table::EstimatorId::StaticMediationLinear,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            functional_roots: (program.mapping().source.raw(), program.mapping().executable.raw()),
            graph_edges: edges.into(),
            bootstrap_replicates: 0,
            validation,
        })
    }

    /// Inspect the sealed frequentist attribution target and supplied DAG.
    #[must_use]
    pub fn checked_attribution_info(&self) -> Option<CheckedAttributionInfo> {
        let operation = self.execution.attribution()?;
        let mut edges: Vec<_> = operation
            .graph()
            .edges()
            .filter_map(|edge| edge.parent_child())
            .map(|(a, b)| (a.raw(), b.raw()))
            .collect();
        edges.sort_unstable();
        let (identifier, estimator) = operation.procedure();
        Some(CheckedAttributionInfo {
            query: operation.query(),
            identifier: identifier.parse().ok()?,
            estimator: estimator.parse().ok()?,
            graph_edges: edges.into(),
            identification_status: operation.identification().status,
        })
    }

    pub(crate) fn has_checked_temporal_dag_response_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::TemporalDagResponse(_))
    }

    pub(crate) fn has_checked_temporal_dag_effect_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::TemporalDagEffect(_))
    }

    pub(crate) fn has_checked_propensity_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::CheckedPropensity(_))
    }

    /// Whether preparation retained the frequentist DAG graph-posterior effect plan.
    #[must_use]
    pub fn has_checked_graph_posterior_effect_operation(&self) -> bool {
        matches!(self.execution, PreparedExecution::GraphPosteriorEffect(_))
    }

    /// Inspect the frozen graph-posterior atoms and estimator procedure.
    #[must_use]
    pub fn checked_graph_posterior_effect_info(&self) -> Option<CheckedGraphPosteriorEffectInfo> {
        let operation = self.execution.graph_posterior_effect()?;
        let (graph_keys, weights, identified) = operation.atom_weights();
        Some(CheckedGraphPosteriorEffectInfo {
            query: operation.query().clone(),
            estimator: operation.estimator(),
            validation: operation.validation(),
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            bootstrap_replicates: operation.bootstrap_replicates(),
        })
    }

    /// Read-only target, row binding, and uncertainty selected for propensity estimation.
    #[must_use]
    pub fn checked_propensity_info(&self) -> Option<CheckedPropensityInfo> {
        let operation = self.execution.propensity()?;
        let lowering = operation.lowering();
        let (estimator, uncertainty, bootstrap_replicates) =
            match (lowering.procedure, lowering.uncertainty) {
                (
                    CheckedPropensityProcedure::HajekWeighting,
                    CheckedPropensityUncertainty::HajekAnalyticAndBootstrap {
                        bootstrap_replicates,
                    },
                ) => (
                    crate::strategy_table::EstimatorId::PropensityWeighting,
                    Arc::from("hajek_analytic_and_optional_bootstrap"),
                    Some(bootstrap_replicates),
                ),
                (
                    CheckedPropensityProcedure::AbadieImbensMatching,
                    CheckedPropensityUncertainty::AbadieImbensAnalytic { .. },
                ) => (
                    crate::strategy_table::EstimatorId::PropensityMatching,
                    Arc::from("abadie_imbens_analytic"),
                    None,
                ),
                _ => return None,
            };
        Some(CheckedPropensityInfo {
            estimator,
            adjustment_set: Arc::clone(&lowering.adjustment),
            population: lowering.population.clone(),
            source_rows: Arc::clone(&lowering.source_rows),
            uncertainty,
            bootstrap_replicates,
        })
    }

    /// Inspect the retained conditional target, procedure, and bound design.
    #[must_use]
    pub fn checked_conditional_effect_info(&self) -> Option<CheckedConditionalEffectInfo> {
        let operation = self.execution.conditional()?;
        Some(CheckedConditionalEffectInfo {
            query: operation.query.clone(),
            identifier: operation.identifier,
            estimator: operation.estimator,
            procedure: Arc::from(match operation.procedure {
                ConditionalProcedure::LinearInteraction => "linear_interaction_plugin",
                ConditionalProcedure::CrossfitAipwDistribution => {
                    "crossfit_aipw_distribution_scores"
                }
            }),
            validation: operation.refute,
            design_roles: Arc::clone(&operation.design_roles),
            source_rows: Arc::clone(&operation.source_rows),
        })
    }

    /// Read-only view of the checked static derivative-response route, when
    /// preparation retained one.
    #[must_use]
    pub fn checked_derivative_response_info(&self) -> Option<CheckedDerivativeResponseInfo> {
        let PreparedExecution::DerivativeResponse(operation) = &self.execution else {
            return None;
        };
        let (_, estimand) = operation.target();
        let (identifier, estimator) = operation.procedure();
        Some(CheckedDerivativeResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            max_derivative_cells: operation.max_derivative_cells(),
        })
    }

    /// Read-only view of the retained checked static DAG response route.
    #[must_use]
    pub fn checked_static_dag_response_info(&self) -> Option<CheckedStaticDagResponseInfo> {
        let operation = self.execution.static_dag_response()?;
        let (identifier, estimator, validation) = operation.procedure();
        Some(CheckedStaticDagResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation,
            grid_members: Arc::from(operation.curve_grid()),
        })
    }

    /// Read-only view of the sealed Cell AIPW intervention-response route.
    #[must_use]
    pub fn checked_cell_aipw_response_info(&self) -> Option<CheckedCellAipwResponseInfo> {
        let PreparedExecution::CellAipwResponse(operation) = &self.execution else {
            return None;
        };
        let (_, estimand) = operation.target();
        let (identifier, estimator) = operation.procedure();
        Some(CheckedCellAipwResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation: operation.validation(),
            origin: operation.origin(),
            requested_arm: operation.requested_arm(),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
        })
    }

    /// Read-only view of the retained TemporalDag response route.
    #[must_use]
    pub fn checked_temporal_dag_response_info(&self) -> Option<CheckedTemporalDagResponseInfo> {
        let operation = self.execution.temporal_dag_response()?.operation();
        let (identifier, estimator, uncertainty, _) = operation.procedure();
        Some(CheckedTemporalDagResponseInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            uncertainty: Arc::from(uncertainty),
            grid_members: Arc::from(operation.grid()),
            horizons: Arc::from(operation.query().temporal.as_ref()?.horizons.as_ref()),
        })
    }

    /// Inspect the retained checked temporal mediation target before execution.
    #[must_use]
    pub fn checked_temporal_mediation_info(&self) -> Option<CheckedTemporalMediationInfo> {
        let operation = self.execution.temporal_mediation()?;
        let (query, graph, validation, horizons) = operation.inspect();
        Some(CheckedTemporalMediationInfo {
            query: query.clone(),
            graph_signature: Arc::from(format!("{:?}", graph)),
            validation,
            horizons: Arc::from(horizons),
            estimator: operation.estimator(),
            adjustment_sets: Arc::from(
                operation
                    .horizon_contracts()
                    .map(|(horizon, _, _, _, design)| {
                        let keys = design
                            .iter()
                            .map(|column| antecedent_core::TemporalNodeKey {
                                variable: column.variable,
                                offset: -i32::try_from(column.lag.raw())
                                    .expect("checked temporal mediation lags fit i32"),
                            })
                            .collect::<Vec<_>>();
                        (horizon, Arc::from(keys))
                    })
                    .collect::<Vec<_>>(),
            ),
        })
    }

    /// Inspect the retained checked temporal effect before execution.
    #[must_use]
    pub fn checked_temporal_dag_effect_info(&self) -> Option<CheckedTemporalDagEffectInfo> {
        let operation = self.execution.temporal_dag_effect()?.operation();
        let (identifier, estimator, validation, bootstrap_replicates) = operation.procedure();
        let adjustment_set = operation
            .estimand()
            .adjustment_set
            .iter()
            .filter_map(|id| operation.indexer().key_of(id.raw()).ok())
            .collect::<Vec<_>>();
        Some(CheckedTemporalDagEffectInfo {
            query: operation.query().clone(),
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            adjustment_set: adjustment_set.into(),
            graph_signature: Arc::from(operation.graph_signature()),
        })
    }

    /// Inspect the retained source graph and complete temporal class envelope.
    #[must_use]
    pub fn checked_temporal_class_effect_info(&self) -> Option<CheckedTemporalClassEffectInfo> {
        let operation = self.execution.temporal_class_effect()?.operation();
        let (identifier, estimator, bootstrap_replicates, _, validation, _) = operation.procedure();
        let envelope = &operation.bundle().envelope.envelope;
        Some(CheckedTemporalClassEffectInfo {
            query: operation.query().clone(),
            graph_class: operation.graph_class(),
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            completion_count: envelope.cases.len(),
            completion_limit: operation.max_completions(),
            identified_mass: envelope.identified_weight.0,
            unidentified_mass: envelope.unidentified_weight.0,
            truncated_completions: envelope.truncated_completions,
            graph_version: u64::from(operation.structure_version()),
        })
    }

    /// Compatibility spelling emphasizing the temporal effect operation family.
    #[must_use]
    pub fn checked_temporal_effect_info(&self) -> Option<CheckedTemporalDagEffectInfo> {
        self.checked_temporal_dag_effect_info()
    }

    /// Compatibility spelling emphasizing the response operation family.
    #[must_use]
    pub fn checked_temporal_response_info(&self) -> Option<CheckedTemporalDagResponseInfo> {
        self.checked_temporal_dag_response_info()
    }

    /// Checked linear two-stage front-door lowering retained by this handle.
    #[must_use]
    pub fn checked_frontdoor_linear(
        &self,
    ) -> Option<&antecedent_estimate::CheckedFrontDoorPreparation> {
        self.execution.frontdoor_linear().map(|operation| &operation.preparation)
    }

    /// Checked Wald or single-binary-instrument 2SLS lowering retained by this handle.
    #[must_use]
    pub fn checked_iv(&self) -> Option<&antecedent_estimate::CheckedIvPreparation> {
        self.execution.iv().map(CheckedIvOperation::preparation)
    }

    /// Checked functional program retained for a prepared distribution query.
    #[must_use]
    pub fn checked_distribution_program(&self) -> Option<&antecedent_expr::FunctionalProgram> {
        self.execution.distribution().map(|operation| operation.prepared.program())
    }

    /// Checked functional program retained for an ADMG general-ID effect.
    #[must_use]
    pub fn checked_functional_effect_program(&self) -> Option<&antecedent_expr::FunctionalProgram> {
        self.execution
            .functional_effect_operation()
            .map(|operation| operation.prepared.program())
            .or_else(|| {
                self.execution
                    .path_specific_effect_operation()
                    .map(|operation| operation.prepared.program())
            })
    }

    /// Ordered checked members for a prepared ADMG mean response curve.
    #[must_use]
    pub fn checked_functional_effect_response_members(
        &self,
    ) -> Option<&[CheckedFunctionalEffectResponseMember]> {
        self.execution.admg_response_curve().map(|operation| operation.members.as_ref())
    }

    /// Export the complete empirical factor laws for the retained effect program.
    pub(crate) fn functional_effect_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        self.ensure_schema_compatible(data)?;
        if let Some(operation) = self.execution.functional_effect_operation() {
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        if let Some(operation) = self.execution.path_specific_effect_operation() {
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        Ok(None)
    }

    /// Rebound empirical factor snapshots, one for each response member in grid order.
    pub(crate) fn functional_effect_response_factor_snapshots(
        &self,
        data: &TabularData,
    ) -> Result<Option<Arc<[antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot]>>, CausalError>{
        let Some(operation) = self.execution.admg_response_curve() else { return Ok(None) };
        self.ensure_schema_compatible(data)?;
        let rebound = operation.rebind(data)?;
        let snapshots = rebound
            .members
            .iter()
            .map(|member| member.prepared.factor_snapshot().map_err(CausalError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Arc::from(snapshots)))
    }

    /// Export the complete provider laws for the retained distribution program,
    /// rebound to `data` after the normal semantic-schema check.
    pub(crate) fn distribution_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        let Some(operation) = self.execution.distribution() else {
            return Ok(None);
        };
        self.ensure_schema_compatible(data)?;
        let prepared = operation.rebind(data)?;
        prepared.factor_snapshot().map(Some).map_err(CausalError::from)
    }

    /// Borrow the caller's frozen query, with original variable ids and query kind.
    #[must_use]
    pub fn query(&self) -> &CausalQuery {
        &self.analysis.query
    }

    /// Crate-visible frozen study.
    pub(crate) fn study(&self) -> &Study {
        &self.analysis
    }

    #[cfg(test)]
    pub(crate) fn study_mut(&mut self) -> &mut Study {
        self.program_cache = std::sync::OnceLock::new();
        if matches!(self.execution, PreparedExecution::CheckedLinear(_)) {
            self.execution = PreparedExecution::LegacyStudyDispatch;
        }
        &mut self.analysis
    }

    /// Replace the frozen study after a successful refresh.
    fn replace_study(&mut self, study: Study) {
        self.program_cache = std::sync::OnceLock::new();
        self.analysis = study;
    }

    pub(crate) fn program_cache(
        &self,
    ) -> &std::sync::OnceLock<Arc<super::contract::ProgramPayloads>> {
        &self.program_cache
    }

    /// Series data in the variant this handle was prepared with (event
    /// studies keep the event modality through estimate and refresh).
    fn series_input(&self, data: TimeSeriesData) -> DataInput {
        match self.analysis.data {
            DataInput::Event(_) => DataInput::Event(data),
            _ => DataInput::Temporal(data),
        }
    }

    /// Stamp the contract `result` was executed under on `data`.
    fn stamp(&self, data: &DataInput, mut result: StudyResult) -> Result<StudyResult, CausalError> {
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &self.analysis.inference,
            data,
        );
        result.executed_contract =
            Some(self.executed_contract(data, self.analysis.refute, None)?);
        Ok(result)
    }

    fn stamp_distribution(
        &self,
        data: &DataInput,
        operation: &CheckedDistributionOperation,
        mut result: StudyResult,
    ) -> Result<StudyResult, CausalError> {
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &operation.inference,
            data,
        );
        result.executed_contract = Some(self.executed_contract(data, operation.refute, None)?);
        Ok(result)
    }

    fn stamp_linear(
        &self,
        data: &DataInput,
        operation: &CheckedLinearOperation,
        mut result: StudyResult,
    ) -> Result<StudyResult, CausalError> {
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &operation.inference,
            data,
        );
        result.executed_contract = Some(self.executed_contract(data, operation.refute, None)?);
        Ok(result)
    }

    fn execute_checked_frontdoor(
        &self,
        data: &TabularData,
        operation: &CheckedFrontDoorOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (query, identification, estimand) = self.checked_static_target()?;
        let lowering = operation.preparation.lowering();
        if operation.preparation.target().functional != estimand.functional
            || operation.preparation.target().adjustment_set != estimand.adjustment_set
            || lowering.treatment != query.treatment
            || lowering.outcome != query.outcome
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
            || !matches!(query.target_population, TargetPopulation::AllObserved)
            || !matches!(query.active, Intervention::Set { variable, ref value } if variable == query.treatment && value.as_f64() == Some(lowering.active))
            || !matches!(query.control, Intervention::Set { variable, ref value } if variable == query.treatment && value.as_f64() == Some(lowering.control))
            || self.plan.logical.record.estimator.as_deref()
                != Some(crate::strategy_table::EstimatorId::FrontDoorTwoStage.as_str())
            || !matches!(self.analysis.inference, InferenceMode::Frequentist)
        {
            return Err(CausalError::Compile {
                message:
                    "retained checked front-door operation no longer matches its query or target"
                        .into(),
            });
        }
        let mut workspace = antecedent_estimate::FrontDoorWorkspace::default();
        let estimate = operation
            .fitter
            .fit_checked(&operation.preparation, &mut workspace, ctx)
            .map_err(CausalError::from)?;
        self.assemble_checked_static_effect(
            data,
            &query,
            identification,
            estimand,
            crate::strategy_table::EstimatorId::FrontDoorTwoStage,
            estimate,
            Some(operation.fitter.bootstrap_replicates),
            started,
            ctx,
        )
    }

    fn execute_checked_iv(
        &self,
        data: &TabularData,
        operation: &CheckedIvOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (query, identification, estimand) = self.checked_static_target()?;
        let preparation = operation.preparation();
        let lowering = preparation.lowering();
        let estimator = match operation {
            CheckedIvOperation::Wald { .. } => crate::strategy_table::EstimatorId::IvWald,
            CheckedIvOperation::TwoSls { .. } => crate::strategy_table::EstimatorId::Iv2Sls,
        };
        if preparation.target().functional != estimand.functional
            || preparation.target().instruments != estimand.instruments
            || preparation.target().adjustment_set != estimand.adjustment_set
            || lowering.treatment != query.treatment
            || lowering.outcome != query.outcome
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
            || !matches!(query.target_population, TargetPopulation::AllObserved)
            || !matches!(query.active, Intervention::Set { variable, ref value } if variable == query.treatment && value.as_f64() == Some(lowering.active))
            || !matches!(query.control, Intervention::Set { variable, ref value } if variable == query.treatment && value.as_f64() == Some(lowering.control))
            || self.plan.logical.record.estimator.as_deref() != Some(estimator.as_str())
            || !matches!(self.analysis.inference, InferenceMode::Frequentist)
        {
            return Err(CausalError::Compile {
                message: "retained checked IV operation no longer matches its query or target"
                    .into(),
            });
        }
        let estimate = match operation {
            CheckedIvOperation::Wald { fitter, preparation } => {
                fitter.fit_checked(preparation, ctx).map_err(CausalError::from)?
            }
            CheckedIvOperation::TwoSls { fitter, preparation } => {
                let mut workspace = antecedent_estimate::TwoStageLeastSquaresWorkspace::default();
                fitter.fit_checked(preparation, &mut workspace, ctx).map_err(CausalError::from)?
            }
        };
        self.assemble_checked_static_effect(
            data,
            &query,
            identification,
            estimand,
            estimator,
            estimate,
            None,
            started,
            ctx,
        )
    }

    fn execute_checked_glm(
        &self,
        data: &TabularData,
        operation: &CheckedGlmAdjustmentOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if operation.query != *operation.identification.average_effect().unwrap_or(&operation.query)
            || operation.estimand.functional
                != operation
                    .identification
                    .estimands
                    .first()
                    .map_or(operation.estimand.functional, |e| e.functional)
            || operation.estimator != crate::strategy_table::EstimatorId::GlmAdjustment
            || operation.physical.logical.record.estimator.as_deref()
                != Some(crate::strategy_table::EstimatorId::GlmAdjustment.as_str())
            || operation.physical.logical.record.identifier.as_deref()
                != Some(operation.identifier.as_str())
            || operation.preparation.method != operation.estimand.method
            || operation.preparation.adjustment_set != operation.estimand.adjustment_set
            || !matches!(operation.query.outcome_functional, OutcomeFunctional::Mean)
            || self.analysis.query != CausalQuery::AverageEffect(operation.query.clone())
            || self.analysis.refute != operation.refute
        {
            return Err(CausalError::Compile {
                message: "retained GLM operation no longer matches its checked target or estimator"
                    .into(),
            });
        }
        let mut workspace = antecedent_estimate::GlmAdjustmentWorkspace::default();
        let estimate = operation
            .fitter
            .fit(
                &operation.preparation,
                &mut workspace,
                ctx,
                operation.identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)?;
        self.assemble_checked_operation_effect(
            data,
            CheckedStaticResultMetadata {
                source_query: &operation.source_query,
                query: &operation.query,
                identification: &operation.identification,
                estimand: &operation.estimand,
                identifier: operation.identifier,
                estimator: operation.estimator,
                physical: &operation.physical,
                graph_class: operation.graph_class,
                graph_version: operation.graph_version,
                support_status: operation.support_status,
                structure_source: operation.structure_source,
                population_registry: operation.population_registry.as_ref(),
                latency_mode: operation.latency_mode,
                refute: operation.refute,
                custom_validator_names: &operation.custom_validator_names,
            },
            estimate,
            Some(operation.fitter.bootstrap_replicates),
            started,
            ctx,
        )
    }

    fn execute_checked_rd(
        &self,
        data: &TabularData,
        operation: &CheckedRdOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let lowering = operation.preparation.lowering();
        if operation.estimator != crate::strategy_table::EstimatorId::RdSharp
            || operation.identifier != crate::strategy_table::IdentifierId::RdSharp
            || operation.physical.logical.record.estimator.as_deref()
                != Some(crate::strategy_table::EstimatorId::RdSharp.as_str())
            || operation.physical.logical.record.identifier.as_deref()
                != Some(crate::strategy_table::IdentifierId::RdSharp.as_str())
            || operation.graph_class != GraphClass::Dag
            || !matches!(
                operation.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !matches!(operation.inference, InferenceMode::Frequentist)
            || lowering.functional != operation.estimand.functional
            || lowering.treatment != operation.query.treatment
            || lowering.outcome != operation.query.outcome
            || operation.preparation.target().functional != operation.estimand.functional
            || !matches!(operation.query.outcome_functional, OutcomeFunctional::Mean)
            || self.analysis.query != CausalQuery::AverageEffect(operation.source_query.clone())
            || self.analysis.refute != operation.refute
        {
            return Err(CausalError::Compile {
                message:
                    "retained sharp-RD operation no longer matches its checked target or design"
                        .into(),
            });
        }
        let mut workspace = antecedent_estimate::RdWorkspace::default();
        let estimate = operation
            .fitter
            .fit_checked(&operation.preparation, &mut workspace, ctx)
            .map_err(CausalError::from)?;
        let source_query = CausalQuery::AverageEffect(operation.source_query.clone());
        self.assemble_checked_operation_effect(
            data,
            CheckedStaticResultMetadata {
                source_query: &source_query,
                query: &operation.query,
                identification: &operation.identification,
                estimand: &operation.estimand,
                identifier: operation.identifier,
                estimator: operation.estimator,
                physical: &operation.physical,
                graph_class: operation.graph_class,
                graph_version: operation.graph_version,
                support_status: operation.support_status,
                structure_source: operation.structure_source,
                population_registry: operation.population_registry.as_ref(),
                latency_mode: operation.latency_mode,
                refute: operation.refute,
                custom_validator_names: &operation.custom_validator_names,
            },
            estimate,
            Some(operation.fitter.bootstrap_replicates),
            started,
            ctx,
        )
    }

    fn execute_checked_propensity(
        &self,
        data: &TabularData,
        operation: &CheckedPropensityOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let retained = operation.context();
        let lowering = operation.lowering();
        if self.analysis.query != retained.source_query
            || self.plan.logical.record.estimator != retained.physical.logical.record.estimator
            || lowering.functional != operation.target().functional
            || lowering.treatment != operation.query().treatment
            || lowering.outcome != operation.query().outcome
            || lowering.adjustment != operation.target().adjustment_set
        {
            return Err(CausalError::Compile {
                message: "retained propensity operation no longer matches its checked target or procedure".into(),
            });
        }
        let estimator = match lowering.procedure {
            CheckedPropensityProcedure::HajekWeighting => {
                crate::strategy_table::EstimatorId::PropensityWeighting
            }
            CheckedPropensityProcedure::AbadieImbensMatching => {
                crate::strategy_table::EstimatorId::PropensityMatching
            }
        };
        let bootstrap_requested = match lowering.uncertainty {
            CheckedPropensityUncertainty::HajekAnalyticAndBootstrap { bootstrap_replicates } => {
                Some(bootstrap_replicates)
            }
            CheckedPropensityUncertainty::AbadieImbensAnalytic { .. } => None,
        };
        let estimate = operation.execute(ctx)?;
        self.assemble_checked_operation_effect(
            data,
            CheckedStaticResultMetadata {
                source_query: &retained.source_query,
                query: operation.query(),
                identification: &retained.identification,
                estimand: operation.target(),
                identifier: retained.identifier,
                estimator,
                physical: &retained.physical,
                graph_class: retained.graph_class,
                graph_version: retained.graph_version,
                support_status: retained.support_status,
                structure_source: retained.structure_source,
                population_registry: retained.population_registry.as_ref(),
                latency_mode: retained.latency_mode,
                refute: retained.refute,
                custom_validator_names: &retained.custom_validator_names,
            },
            estimate,
            bootstrap_requested,
            started,
            ctx,
        )
    }

    fn execute_checked_conditional(
        &self,
        data: &TabularData,
        operation: &CheckedConditionalOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.analysis.query != operation.source_query
            || self.plan.logical.record.estimator != operation.physical.logical.record.estimator
            || operation.program.mapping().source != operation.estimand.functional
        {
            return Err(CausalError::Compile {
                message: "retained conditional operation no longer matches its checked target or procedure".into(),
            });
        }
        let estimate = operation.execute(data, ctx)?;
        let estimator = if operation.procedure == ConditionalProcedure::CrossfitAipwDistribution {
            crate::strategy_table::EstimatorId::Aipw
        } else {
            operation.estimator
        };
        let mut result = self.assemble_checked_operation_effect(
            data,
            CheckedStaticResultMetadata {
                source_query: &operation.source_query,
                query: &operation.query.inner,
                identification: &operation.identification,
                estimand: &operation.estimand,
                identifier: operation.identifier,
                estimator,
                physical: &operation.physical,
                graph_class: GraphClass::Dag,
                graph_version: operation.graph_version,
                support_status: operation.support_status,
                structure_source: operation.structure_source,
                population_registry: operation.population_registry.as_ref(),
                latency_mode: operation.latency_mode,
                refute: operation.refute,
                custom_validator_names: &[],
            },
            estimate,
            None,
            started,
            ctx,
        )?;
        if let Some(diagnostic) = super::helpers::conditional_quantile_grid_diagnostic(
            data,
            &operation.query,
            operation.estimand.adjustment_set.iter().copied(),
        )? {
            result.diagnostics.push(diagnostic);
        }
        result
            .diagnostics
            .extend(super::helpers::conditional_score_estimator_diagnostic(&operation.query));
        if result.estimate.score_inference.is_some() {
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                "estimate.functional.cdf_inference",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "per-arm F_a(c) simultaneous bands describe raw CDF coordinates; rearranged exceedance_cdf values are not mixed with those intervals",
            ));
        }
        if result.estimate.exceedance_cdf.as_ref().is_some_and(|cdf| cdf.len() > 2)
            && !result.estimate.ate.is_finite()
        {
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ));
        }
        Ok(result)
    }

    fn checked_static_target(
        &self,
    ) -> Result<(AverageEffectQuery, IdentificationResult, IdentifiedEstimand), CausalError> {
        let CausalQuery::AverageEffect(query) = &self.analysis.query else {
            return Err(CausalError::Compile {
                message: "checked static operation requires an average-effect query".into(),
            });
        };
        let cache =
            self.analysis.identification_cache.as_deref().ok_or_else(|| CausalError::Compile {
                message: "checked static operation lost its identification product".into(),
            })?;
        Ok((query.clone(), cache.identification.clone(), cache.estimand.clone()))
    }

    fn assemble_checked_static_effect(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        estimator_id: crate::strategy_table::EstimatorId,
        estimate: EffectEstimate,
        bootstrap_requested: Option<u32>,
        started: Instant,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let cancelled = estimate.bootstrap_cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut extra_diagnostics) =
            if cancelled || self.analysis.refute == RefuteSuite::None {
                (Vec::new(), Vec::new())
            } else {
                let mut workspace = EstimationWorkspace::default();
                let mut propensity = antecedent_stats::PropensityWorkspace::default();
                let (reports, diagnostics) = run_refuters(
                    data,
                    &estimand,
                    query,
                    &estimate,
                    &mut workspace,
                    Some(&mut propensity),
                    ctx,
                    self.analysis.refute,
                    estimator_id.as_str(),
                    &self.analysis.custom_validators,
                    None,
                )?;
                (reports, diagnostics)
            };
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the checked prepared operation",
        ));
        diagnostics.append(&mut extra_diagnostics);
        let identifier_id = self
            .plan
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
            .parse()?;
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(identifier_id);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(estimator_id);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &identification.required_assumptions),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let mut result = assemble_result(AssembleArgs {
            logical: &self.plan.logical.record,
            physical: &self.plan.record,
            identification: identification.clone(),
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.analysis.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: bootstrap_requested,
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
            bayesian: false,
        });
        result.certificate = Some(crate::result::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: identification,
                temporal_indexer: None,
                strategy: identifier_id,
                structure_version: self.analysis.graph.version(),
            },
            query: self.analysis.query.clone(),
            graph_class: self.analysis.graph.class(),
        });
        result.support_status = self.analysis.support_status;
        result.structure_source = self.analysis.structure_source;
        result.population_registry.clone_from(&self.analysis.population_registry);
        result.custom_validator_names = self
            .analysis
            .custom_validators
            .iter()
            .map(|validator| Arc::from(validator.name()))
            .collect();
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }

    fn assemble_checked_operation_effect(
        &self,
        data: &TabularData,
        metadata: CheckedStaticResultMetadata<'_>,
        estimate: EffectEstimate,
        bootstrap_requested: Option<u32>,
        started: Instant,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let cancelled = estimate.bootstrap_cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut extra_diagnostics) =
            if cancelled || metadata.refute == RefuteSuite::None {
                (Vec::new(), Vec::new())
            } else {
                let mut workspace = EstimationWorkspace::default();
                let mut propensity = antecedent_stats::PropensityWorkspace::default();
                let (reports, diagnostics) = run_refuters(
                    data,
                    metadata.estimand,
                    metadata.query,
                    &estimate,
                    &mut workspace,
                    Some(&mut propensity),
                    ctx,
                    metadata.refute,
                    metadata.estimator.as_str(),
                    &[],
                    None,
                )?;
                (reports, diagnostics)
            };
        let mut diagnostics = metadata.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the checked prepared operation",
        ));
        diagnostics.append(&mut extra_diagnostics);
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(metadata.identifier);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(metadata.estimator);
        let provenance = provenance_pair(
            (
                identify_artifact,
                identify_operation,
                &[],
                &metadata.identification.required_assumptions,
            ),
            (estimate_artifact, estimate_operation, &[identify_artifact], &estimate.assumptions),
        );
        let mut result = assemble_result(AssembleArgs {
            logical: &metadata.physical.logical.record,
            physical: &metadata.physical.record,
            identification: metadata.identification.clone(),
            estimand: metadata.estimand.clone(),
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: metadata.query.treatment,
            outcome: metadata.query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: metadata.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: bootstrap_requested,
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
            bayesian: false,
        });
        result.certificate = Some(crate::result::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: metadata.identification.clone(),
                temporal_indexer: None,
                strategy: metadata.identifier,
                structure_version: metadata.graph_version,
            },
            query: metadata.source_query.clone(),
            graph_class: metadata.graph_class,
        });
        result.support_status = metadata.support_status;
        result.structure_source = metadata.structure_source;
        result.population_registry = metadata.population_registry.cloned();
        result.custom_validator_names = metadata.custom_validator_names.to_vec();
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }

    fn stamp_aipw(
        &self,
        data: &DataInput,
        operation: &CheckedAipwOperation,
        mut result: StudyResult,
    ) -> Result<StudyResult, CausalError> {
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &operation.inference,
            data,
        );
        result.executed_contract = Some(self.executed_contract(data, operation.refute, None)?);
        Ok(result)
    }

    /// Replace custom validators. Incoming names must match the prepare-time set.
    ///
    /// # Errors
    ///
    /// Name mismatch.
    pub fn rebind_custom_validators(
        &mut self,
        validators: Vec<std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) -> Result<(), CausalError> {
        let expected: std::collections::BTreeSet<&str> =
            self.analysis.custom_validators.iter().map(|v| v.name()).collect();
        let incoming: std::collections::BTreeSet<&str> =
            validators.iter().map(|v| v.name()).collect();
        if expected != incoming {
            return Err(crate::unsupported_reason!(
                "attested_not_reverifiable",
                "rebind_validators names must match attested names"
            ));
        }
        self.analysis.custom_validators = validators;
        Ok(())
    }

    /// Population bindings frozen at prepare, when the query names a predicate
    /// or a custom target distribution.
    #[must_use]
    pub fn population_registry(&self) -> Option<&antecedent_core::PopulationRegistry> {
        self.analysis.population_registry.as_ref()
    }

    /// Names of the caller custom validators frozen at prepare, in order.
    ///
    /// These are the names a claim attests and the names
    /// [`Self::rebind_custom_validators`] requires.
    #[must_use]
    pub fn custom_validator_names(&self) -> Vec<&str> {
        self.analysis.custom_validators.iter().map(|validator| validator.name()).collect()
    }

    /// Stream progressive stages from later estimates of this handle.
    ///
    /// `None` stops streaming. The sink is an in-process execution control: it
    /// is not part of the contract, program, or claim identity.
    pub fn set_stage_sink(&mut self, sink: Option<Arc<dyn super::stage::StageResultSink>>) {
        self.analysis.stage_sink = sink;
    }

    /// Whether estimates of this handle stream identify → estimate_point →
    /// uncertainty → validate stage payloads.
    ///
    /// Stages describe one identification and one scalar effect estimate. They
    /// stream from the single-estimand static executors: a supplied or DAG-coerced
    /// DAG average effect (Frequentist or Bayesian, excluding `rd.sharp` and the
    /// general-ID functional plug-in) and the Bayesian DAG conditional effect.
    #[must_use]
    pub fn streams_stages(&self) -> bool {
        let analysis = &self.analysis;
        if analysis.graph_posterior.is_some()
            || analysis.tiered.is_some()
            || !matches!(analysis.data, DataInput::Tabular(_))
        {
            return false;
        }
        let dag_like = match analysis.graph.class() {
            GraphClass::Dag => true,
            GraphClass::Admg => analysis
                .graph
                .as_admg()
                .is_some_and(|admg| !super::execute::admg_has_bidirected(admg)),
            _ => false,
        };
        let estimator = self.plan.logical.record.estimator.as_deref();
        match &analysis.query {
            CausalQuery::AverageEffect(_) => {
                dag_like && !matches!(estimator, Some("rd.sharp" | "functional.effect"))
            }
            CausalQuery::ConditionalEffect(_) => {
                analysis.graph.class() == GraphClass::Dag
                    && matches!(analysis.inference, InferenceMode::Bayesian(_))
            }
            _ => false,
        }
    }

    /// Frozen horizon-specific identification and its exact unfolded variable namespace.
    ///
    /// A `TemporalDag` prepare caches this directly. A DBN posterior or a TemporalCpdag/Pag
    /// envelope caches a per-atom or per-completion result instead; this projects either
    /// onto the same shape (see [`super::contract::full_temporal_identification`]), so an
    /// exported `analysis_result` artifact validates its identification against the same
    /// namespace the compiled contract already uses.
    #[must_use]
    pub fn temporal_identification(
        &self,
    ) -> Option<std::borrow::Cow<'_, CachedTemporalIdentification>> {
        super::contract::full_temporal_identification(&self.analysis)
    }

    /// Borrow the frozen schema fingerprint.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Matrix structure-source axis frozen at prepare.
    #[must_use]
    pub const fn structure_source(&self) -> crate::support::StructureSource {
        self.state.analysis.structure_source()
    }

    /// Evidence contract frozen at prepare. `None` when the query is off-axis.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.state.analysis.support_status()
    }

    /// Borrow the ready physical plan retained from prepare.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// Borrow the prepare-time AIPW score table, when the cell exported one.
    #[must_use]
    pub fn score_table(&self) -> Option<&ScoreTable> {
        self.score_table.as_ref()
    }

    /// Shared batch design attached when this plan was prepared inside a batch.
    #[must_use]
    pub fn shared_design(&self) -> Option<&super::batch::SharedBatchDesign> {
        self.analysis.shared_batch_design.as_deref()
    }

    /// The prepared score table a retarget reweights.
    ///
    /// # Errors
    ///
    /// `score_table_unavailable` when prepare built none.
    fn retarget_score_table(&self) -> Result<&antecedent_estimate::ScoreTable, CausalError> {
        self.score_table.as_ref().ok_or_else(|| {
            crate::unsupported_reason!(
                "score_table_unavailable",
                "retarget requires a prepared score table on AverageEffect or discrete joint \
                 InterventionResponse"
            )
        })
    }

    /// Whether this handle can perform `intent` at all, before any data or
    /// weights are supplied.
    ///
    /// The one check both [`Self::preview_transform`] and the apply run, so a
    /// preview never authorizes what the apply refuses. Only retarget depends
    /// on the handle (it needs a prepared score table); a compatible-data
    /// refresh is always available, and its schema check needs the new data.
    /// The remaining intents change the question or structure and are
    /// performed by preparing a new study, not applied to this handle.
    ///
    /// # Errors
    ///
    /// The reason-coded refusal the apply raises.
    pub fn transform_capability(
        &self,
        intent: antecedent_core::TransformIntent,
    ) -> Result<(), CausalError> {
        match intent {
            antecedent_core::TransformIntent::Retarget => self.retarget_score_table().map(|_| ()),
            _ => Ok(()),
        }
    }

    /// Estimate `E_Q[μ_a(X)]` from frozen scores. Does not refit or re-identify.
    ///
    /// `weights` must align with the score-table complete-case rows.
    /// `depends_on` is the declared parent set of `w`; it must be a subset of
    /// the certified adjustment set and must not name the treatment, an
    /// intervened coordinate, or a descendant.
    ///
    /// # Errors
    ///
    /// Missing score table, illegal `depends_on`, weight shape, or weighted
    /// overlap failure (support refusal).
    pub fn retarget(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let _ = ctx;
        let table = self.retarget_score_table()?;
        let graph: Option<&dyn antecedent_estimate::DirectedAncestry> = self
            .analysis
            .graph
            .as_dag()
            .map(|g| g as _)
            .or_else(|| self.analysis.graph.as_admg().map(|g| g as _));
        let treatment = score_table_treatment_col(&self.analysis, table);
        let (out, overlap_failed) = antecedent_estimate::retarget(
            table,
            weights,
            depends_on,
            graph,
            treatment.as_deref(),
            None,
        )?;
        if overlap_failed {
            return Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: antecedent_estimate::RetargetRefusal::WeightedOverlap.as_str(),
            });
        }
        let inference = table.inference(Some(weights))?;
        self.retarget_to_result(out, inference, weights, depends_on)
    }

    #[allow(clippy::float_cmp)] // Exact membership in binary intervention levels.
    fn retarget_to_result(
        &self,
        out: RetargetResult,
        inference: antecedent_estimate::scores::ScoreInference,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<StudyResult, CausalError> {
        let cache =
            self.analysis.identification_cache.as_ref().ok_or(CausalError::Unsupported {
                message: "retarget requires prepare-time identification",
            })?;
        let table = self
            .score_table
            .as_ref()
            .ok_or(CausalError::Unsupported { message: "missing frozen scores" })?;
        let n_thresholds = table.distinct_threshold_count();
        let quantile = match &self.analysis.query {
            CausalQuery::AverageEffect(q) => q
                .outcome_functional
                .quantile_level()
                .map(|tau| {
                    antecedent_estimate::quantile::quantile_contrast(table, Some(weights), tau)
                        .map(|q| (q.value, q.influence))
                })
                .transpose()?,
            CausalQuery::Response(q) => q
                .outcome_functional
                .quantile_level()
                .map(|tau| {
                    let arm = super::helpers::requested_joint_arm(q)?;
                    let q = antecedent_estimate::quantile::quantile_arm(
                        table,
                        Some(weights),
                        tau,
                        arm,
                    )?;
                    Ok::<_, CausalError>((q.value, q.influence))
                })
                .transpose()?,
            _ => None,
        };
        let (ate, se) = match &self.analysis.query {
            CausalQuery::AverageEffect(_) | CausalQuery::Response(_) if n_thresholds > 1 => {
                (f64::NAN, f64::NAN)
            }
            CausalQuery::AverageEffect(_) => {
                let c = out
                    .contrast
                    .as_ref()
                    .ok_or(CausalError::Unsupported { message: "missing arm contrast" })?;
                (c.value, c.se)
            }
            CausalQuery::Response(q) => {
                let arm = super::helpers::requested_joint_arm(q)?;
                let col = table
                    .columns
                    .iter()
                    .position(|c| c.arm == arm)
                    .ok_or(CausalError::Unsupported { message: "missing joint cell" })?;
                (out.summary.means[col], out.covariance.se(col))
            }
            _ => return Err(CausalError::Unsupported { message: "unsupported retarget query" }),
        };
        let cdf = self.score_table.as_ref().and_then(|t| exceedance_cdf_values(&out.summary, t));
        let mut estimate = EffectEstimate::new(
            ate,
            se,
            cache.identification.required_assumptions.clone(),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        )
        .with_score_table(self.score_table.clone())
        .with_joint_covariance(Some(out.covariance.clone()))
        .with_exceedance_cdf(cdf)
        .with_monotone_rearranged(out.monotone_rearranged);
        estimate.score_inference = Some(inference);
        if let Some((value, influence)) = quantile.as_ref() {
            estimate.ate = *value;
            estimate.se_analytic =
                antecedent_estimate::joint_influence_covariance(&[influence], None)?.se(0);
            estimate.influence = Some(influence.clone().into());
        }
        let (treatment, outcome) = match &self.analysis.query {
            CausalQuery::AverageEffect(q) => (q.treatment, q.outcome),
            CausalQuery::Response(q) => q
                .functional
                .primary_pair()
                .ok_or(CausalError::Unsupported { message: "retarget response missing pair" })?,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "retarget is licensed for AverageEffect and InterventionResponse",
                });
            }
        };
        let mut diagnostics = out.diagnostics;
        if quantile.is_some() {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "estimate.functional.quantile", antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "weighted piecewise-linear CDF inversion conditional on the frozen grid; grid-selection uncertainty and interpolation bias are excluded",
            ));
        }
        if n_thresholds > 1 && quantile.is_none() {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ));
        }
        diagnostics.push(antecedent_core::Diagnostic::new(
            "estimate.aipw.crossfit_scores",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "retarget averages the prepared cross-fitted φ table; it is not a residualized full-sample AIPW refit, so it can differ from the estimator's own point value under uniform weights",
        ));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "retarget.selection_assumption", antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "weights are caller-declared fixed functions of certified covariates; inference assumes iid sampling, positivity, and nuisance convergence; selection or weight-estimation uncertainty is excluded",
        ));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        let mut result = super::helpers::assemble_result(super::helpers::AssembleArgs {
            logical: &self.plan.logical.record,
            physical: &self.plan.record,
            identification: cache.identification.clone(),
            estimand: cache.estimand.clone(),
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance: antecedent_core::ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: 0,
            latency_mode: None,
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: None,
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
            bayesian: matches!(self.analysis.inference, InferenceMode::Bayesian(_)),
        });
        result.certificate = Some(crate::result::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: cache.identification.clone(),
                temporal_indexer: None,
                strategy: self
                    .plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER_ID),
                structure_version: self.analysis.graph.version(),
            },
            query: self.analysis.query.clone(),
            graph_class: self.analysis.graph.class(),
        });
        if antecedent_estimate::changes_target(weights) {
            let binding = self.row_weights_binding(weights, depends_on)?;
            if let Some(certificate) = &mut result.certificate {
                if let Some(target) = certificate.query.target_population_mut() {
                    *target = binding.population();
                }
            }
            result.row_weights = Some(binding);
        }
        if let CausalQuery::Response(q) = &self.analysis.query {
            result.response = Some(antecedent_core::CausalResponse {
                estimand: q.functional.clone(),
                identification_status: cache.identification.status,
                estimate: antecedent_core::ResponseIdentification::PointIdentified(
                    antecedent_core::ResponseValue::Scalar(result.estimate.ate),
                ),
                uncertainty: antecedent_core::ResponseUncertainty::Scalar {
                    standard_error: result.estimate.se_analytic,
                    lower: result.estimate.ate
                        - crate::result::reported_se_interval_z() * result.estimate.se_analytic,
                    upper: result.estimate.ate
                        + crate::result::reported_se_interval_z() * result.estimate.se_analytic,
                    level: 0.95,
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                },
                support: antecedent_core::SupportReport {
                    status: antecedent_core::SupportStatus::Supported,
                    query_region: antecedent_core::SupportRegion {
                        minima: Arc::from([]),
                        maxima: Arc::from([]),
                    },
                    diagnostics: Vec::new(),
                    warnings: Vec::new(),
                    point_status: None,
                },
                assumptions: cache.identification.required_assumptions.clone(),
                provenance_id: Arc::from("estimate.cell.aipw.retarget"),
                horizon_identification: None,
                interaction_structurally_zero: false,
            });
        }
        result.rebind_interval(matches!(self.analysis.inference, InferenceMode::Bayesian(_)));
        result.support_status = self.analysis.support_status;
        result.structure_source = self.analysis.structure_source;
        let retargeted = result.retarget_population();
        result.executed_contract = Some(self.executed_contract(
            &self.analysis.data,
            self.analysis.refute,
            retargeted.as_ref(),
        )?);
        Ok(result)
    }

    /// Re-estimate on `data` without recompiling the physical plan.
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let shared = self
            .analysis
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(data).map(Arc::new))
            .transpose()?;
        self.estimate_with_shared(data, shared, ctx)
    }

    pub(crate) fn estimate_with_shared(
        &self,
        data: &TabularData,
        shared: Option<Arc<super::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        if let Some(operation) = self.execution.linear_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let result = rebound.execute(data, ctx)?;
                return self.stamp_linear(&DataInput::Tabular(data.clone()), &rebound, result);
            }
        }
        if let Some(operation) = self.execution.admg_response_curve() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                super::execute::push_gaussian_likelihood_disclosure(
                    &mut result,
                    &rebound.inference,
                    &DataInput::Tabular(data.clone()),
                );
                result.executed_contract = Some(self.executed_contract(
                    &DataInput::Tabular(data.clone()),
                    RefuteSuite::None,
                    None,
                )?);
                return Ok(result);
            }
        }
        if let Some(operation) = self.execution.path_specific_effect_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                super::execute::push_gaussian_likelihood_disclosure(
                    &mut result,
                    &rebound.inference,
                    &DataInput::Tabular(data.clone()),
                );
                result.executed_contract = Some(self.executed_contract(
                    &DataInput::Tabular(data.clone()),
                    rebound.refute,
                    None,
                )?);
                return Ok(result);
            }
        }
        if let Some(operation) = self.execution.aipw_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                return self.stamp_aipw(&DataInput::Tabular(data.clone()), &rebound, result);
            }
        }
        if let Some(operation) = self.execution.distribution() {
            if operation.sealed_for_direct_execution() {
                let mut result = operation.execute(data, ctx)?;
                result.custom_validator_names = operation.custom_validator_names.to_vec();
                return self.stamp_distribution(
                    &DataInput::Tabular(data.clone()),
                    operation,
                    result,
                );
            }
        }
        if let Some(operation) = self.execution.functional_effect_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                super::execute::push_gaussian_likelihood_disclosure(
                    &mut result,
                    &rebound.inference,
                    &DataInput::Tabular(data.clone()),
                );
                result.executed_contract = Some(self.executed_contract(
                    &DataInput::Tabular(data.clone()),
                    rebound.refute,
                    None,
                )?);
                return Ok(result);
            }
        }
        if let Some(operation) = self.execution.frontdoor_linear() {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_frontdoor(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let Some(operation) = self.execution.iv() {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_iv(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::Counterfactual(plan) = &self.execution {
            let result = plan.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::DerivativeResponse(operation) = &self.execution {
            let result = self
                .analysis
                .execute_checked_derivative_response(data, &self.plan, ctx, operation)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::StaticDagResponse(operation) = &self.execution {
            let result = self
                .analysis
                .execute_checked_static_dag_response(data, &self.plan, ctx, operation)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::CellAipwResponse(operation) = &self.execution {
            let rebound = operation.refresh(
                self.analysis.graph.as_dag().ok_or_else(|| CausalError::Compile {
                    message: "cell AIPW checked route requires its retained DAG".into(),
                })?,
                data,
            )?;
            let result = self
                .analysis
                .execute_checked_cell_aipw_response(data, &self.plan, ctx, &rebound)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::StaticMediation(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::BayesianStaticMediation(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::Attribution(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::BayesianGcomp(operation) = &self.execution {
            let mut result = operation.execute(data, ctx)?;
            result.custom_validator_names = operation.custom_validator_names();
            super::execute::push_gaussian_likelihood_disclosure(
                &mut result,
                operation.operation().inference(),
                &DataInput::Tabular(data.clone()),
            );
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::BayesianConditional(operation) = &self.execution {
            let mut result = operation.execute(data, ctx)?;
            super::execute::push_gaussian_likelihood_disclosure(
                &mut result,
                operation.inference(),
                &DataInput::Tabular(data.clone()),
            );
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::CheckedGlmAdjustment(operation) = &self.execution {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let result = self.execute_checked_glm(data, &rebound, ctx)?;
                return self.stamp(&DataInput::Tabular(data.clone()), result);
            }
        }
        if let PreparedExecution::CheckedRd(operation) = &self.execution {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_rd(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::CheckedPropensity(operation) = &self.execution {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_propensity(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::CheckedConditional(operation) = &self.execution {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_conditional(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        if let PreparedExecution::GraphPosteriorEffect(operation) = &self.execution {
            let mut click_analysis = self.analysis.clone();
            click_analysis.data = DataInput::Tabular(data.clone());
            let result = click_analysis
                .execute_checked_graph_posterior_frequentist(data, &self.plan, operation, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result);
        }
        let mut click_analysis = self.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.interference =
            click_analysis.interference.as_ref().map(|spec| spec.bound_to(data)).transpose()?;
        click_analysis.shared_batch_design = shared;
        let execution = self.execution.rebind_for_dispatch(data)?;
        let mut result = click_analysis.execute_tabular(data, &self.plan, &execution, ctx)?;
        // `execute_tabular` bypasses `Study::execute_on`, which is where fresh runs
        // record which refutation reports are caller-attested. Without the names,
        // the claim would drop custom-validator evidence from `attested` and from
        // the claim identity.
        result.custom_validator_names = click_analysis
            .custom_validators
            .iter()
            .map(|validator| Arc::from(validator.name()))
            .collect();
        // Only a non-mean functional is read from the frozen scores; a mean click keeps
        // the estimator's own value, so refitting the cross-fit table would be discarded.
        let click_scores = if query_reads_score_table(&self.analysis.query) {
            click_analysis.prepare_score_table(ctx)?
        } else {
            None
        };
        overlay_prepared_score_functional(
            &self.analysis.query,
            click_scores.as_ref(),
            &mut result,
        )?;
        self.stamp(&click_analysis.data, result)
    }

    /// Replace retained data and re-estimate (same semantics as [`Self::estimate`]).
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn refresh(
        &mut self,
        data: TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_schema_compatible(&data)?;
        // The checked static lowering belongs to the prepared handle. Execute
        // through it before replacing the retained data snapshot; otherwise the
        // generic Study refresh route would rederive an unchecked preparation.
        let checked_result = (!matches!(self.execution, PreparedExecution::LegacyStudyDispatch))
            .then(|| self.estimate(&data, ctx))
            .transpose()?;
        let mut refreshed = self.analysis.clone();
        refreshed.shared_batch_design = refreshed
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(&data).map(Arc::new))
            .transpose()?;
        refreshed.interference =
            refreshed.interference.as_ref().map(|spec| spec.bound_to(&data)).transpose()?;
        refreshed.data = DataInput::Tabular(data);
        if let Some(result) = checked_result {
            let rebound_conditional = match (&self.execution, &refreshed.data) {
                (PreparedExecution::CheckedConditional(operation), DataInput::Tabular(data)) => {
                    Some(operation.rebind(data)?)
                }
                _ => None,
            };
            let scores = if matches!(self.execution, PreparedExecution::CheckedAipw(_)) {
                refreshed.prepare_score_table(ctx)?
            } else {
                None
            };
            self.replace_study(refreshed);
            if let PreparedExecution::CellAipwResponse(operation) = &self.execution {
                if let DataInput::Tabular(data) = &self.analysis.data {
                    self.execution = PreparedExecution::CellAipwResponse(operation.refresh(
                        self.analysis.graph.as_dag().ok_or_else(|| CausalError::Compile {
                            message: "cell AIPW checked route requires its retained DAG".into(),
                        })?,
                        data,
                    )?);
                }
            }
            if let Some(operation) = rebound_conditional {
                self.execution = PreparedExecution::CheckedConditional(operation);
            }
            self.score_table = scores;
            return Ok(result);
        }
        let mut result = refreshed.execute(&self.plan, ctx)?;
        let scores = refreshed.prepare_score_table(ctx)?;
        overlay_prepared_score_functional(&refreshed.query, scores.as_ref(), &mut result)?;
        self.replace_study(refreshed);
        self.score_table = scores;
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }

    /// Second-click / background refute: replace validation on a prior estimate.
    ///
    /// Leaves ATE / identification / estimand unchanged. Records `validate` stage timing.
    /// Prefer `suite=PlaceboAndRcc` or `Full` after an interactive first click with
    /// Cheap / None.
    ///
    /// # Errors
    ///
    /// Schema mismatch, missing AverageEffect query, cancel, or validator failures.
    pub fn refute(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        if let CausalQuery::Mediation(query) = &self.analysis.query {
            if prior.treatment != query.treatment
                || prior.outcome != query.outcome
                || prior.identification.query != self.analysis.query
            {
                return Err(CausalError::Compile {
                    message: "refute prior does not match mediation query".into(),
                });
            }
            let graph = self.analysis.graph.as_dag().ok_or(CausalError::Unsupported {
                message: "static mediation refute requires Dag",
            })?;
            let mediation = prior.mediation.as_ref().ok_or(CausalError::Unsupported {
                message: "refute requires prior mediation result",
            })?;
            let mut result = prior.clone();
            result.refutations = if suite == RefuteSuite::None {
                Vec::new()
            } else {
                antecedent_validate::mediation::refute_static_mediation(
                    data,
                    graph,
                    query,
                    mediation,
                    suite == RefuteSuite::Full,
                    ctx,
                )?
            };
            return self.stamp_refuted(prior, data, suite, result);
        }
        let query = match &self.analysis.query {
            CausalQuery::AverageEffect(query) => query.clone(),
            CausalQuery::Response(response)
                if matches!(
                    response.functional,
                    antecedent_core::ResponseFunctional::InterventionResponse { .. }
                ) && prior.estimate.ate.is_finite() =>
            {
                AverageEffectQuery::binary_ate(prior.treatment, prior.outcome)
            }
            _ => {
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "PreparedStudy::refute is licensed for AverageEffect and scalar \
                              InterventionResponse",
                });
            }
        };
        if prior.treatment != query.treatment || prior.outcome != query.outcome {
            return Err(CausalError::Compile {
                message: "refute prior result treatment/outcome does not match prepared query"
                    .into(),
            });
        }
        if matches!(self.analysis.inference, InferenceMode::Bayesian(_)) {
            // Bayesian validation also includes prior/posterior predictive checks
            // (and, for Full, prior sensitivity/MCMC diagnostics). Re-running the
            // frozen physical plan is the only path that constructs those artifacts;
            // retain the prior point/identification while replacing validation state.
            let started = Instant::now();
            let mut analysis = self.analysis.clone();
            analysis.refute = suite;
            let validated =
                analysis.execute_on(&DataInput::Tabular(data.clone()), &self.plan, ctx)?;
            let validate_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let mut out = prior.clone();
            out.refutations = validated.refutations;
            super::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
            out.predictive_checks = validated.predictive_checks;
            out.posterior = validated.posterior;
            out.performance.stage_timings_ns.push((Arc::from(STAGE_VALIDATE), validate_ns));
            out.performance.wall_time_ns =
                Some(out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns));
            out.diagnostics.push(antecedent_core::Diagnostic::new(
                "exec.refute.second_click",
                antecedent_core::DiagnosticKind::Execution,
                antecedent_core::DiagnosticSeverity::Info,
                format!("second-click refute suite={}", suite.diagnostic_label()),
            ));
            return self.stamp_refuted(prior, data, suite, out);
        }
        let estimator = self.plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);

        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, &query, &prior.estimand)?;

        let mut clock = StageClock::new();
        clock.begin(ctx, STAGE_VALIDATE, 0.8)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_VALIDATE });
        }
        let mut workspace = EstimationWorkspace::default();
        let started = Instant::now();
        let (reports, na_diagnostics) = run_refuters(
            &data_est,
            &estimand_est,
            &query_est,
            &prior.estimate,
            &mut workspace,
            None,
            ctx,
            suite,
            estimator,
            &self.analysis.custom_validators,
            None,
        )?;
        clock.finish(STAGE_VALIDATE);
        let validate_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let mut out = prior.clone();
        out.refutations = reports;
        super::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
        out.diagnostics.extend(na_diagnostics);
        out.performance.stage_timings_ns.push((Arc::from(STAGE_VALIDATE), validate_ns));
        out.performance.wall_time_ns =
            Some(out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns));
        let suite_label: Arc<str> = Arc::from(suite.diagnostic_label());
        out.diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.refute.second_click",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            format!("second-click refute suite={suite_label}"),
        ));
        let _ = clock.wall_time_ns();
        self.stamp_refuted(prior, data, suite, out)
    }

    /// Stamp a second-click refute under the refute suite that produced its
    /// refutations. The estimate and the refutations must come from this
    /// handle on the same data snapshot; otherwise the result mixes two
    /// executions and carries no contract stamp (it cannot be exported).
    fn stamp_refuted(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        mut out: StudyResult,
    ) -> Result<StudyResult, CausalError> {
        let input = DataInput::Tabular(data.clone());
        let retargeted = prior.retarget_population();
        let population = retargeted.as_ref();
        let same_execution = match &prior.executed_contract {
            Some(stamp) => *stamp == self.executed_contract(&input, stamp.refute, population)?,
            None => false,
        };
        out.executed_contract = if same_execution {
            Some(self.executed_contract(&input, suite, population)?)
        } else {
            None
        };
        Ok(out)
    }

    /// Refuse data of a modality other than the one the plan was prepared for.
    fn ensure_modality(&self, requested: PreparedModality) -> Result<(), CausalError> {
        if self.modality == requested {
            return Ok(());
        }
        Err(CausalError::Compile {
            message: format!(
                "prepared {prepared} analysis requires {prepared} data; use {entry} \
                 (re-prepare to analyse {requested} data)",
                prepared = self.modality.label(),
                entry = self.modality.estimate_entry(),
                requested = requested.label(),
            ),
        })
    }

    /// Refuse a time index whose sampling regularity differs from prepare time.
    fn ensure_regularity(
        &self,
        regularity: &antecedent_data::SamplingRegularity,
    ) -> Result<(), CausalError> {
        if self.time_regularity.as_ref() == Some(regularity) {
            return Ok(());
        }
        Err(CausalError::Compile {
            message: format!(
                "prepared {} analysis requires the same time-index regularity as \
                 prepare-time data; re-prepare after a time-index change",
                self.modality.label()
            ),
        })
    }

    fn ensure_schema_compatible(&self, data: &TabularData) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::Tabular)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared analysis refresh requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        Ok(())
    }

    fn ensure_series_compatible(&self, data: &TimeSeriesData) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::Series)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared temporal analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        self.ensure_regularity(&data.time_index().regularity)
    }

    /// Re-estimate a prepared temporal response on series data (no re-identify).
    ///
    /// # Errors
    ///
    /// Schema / time-index mismatch, or estimation failures.
    pub fn estimate_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_series_compatible(data)?;
        let input = self.series_input(data.clone());
        if let PreparedExecution::TemporalMediation(operation) = &self.execution {
            let graph =
                self.analysis.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
                    message: "checked temporal mediation lost its prepared TemporalDag".into(),
                })?;
            if !operation.matches_graph(graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal mediation graph differs from the prepared graph"
                        .into(),
                });
            }
            let (DataInput::Temporal(series) | DataInput::Event(series)) = &input else {
                return Err(CausalError::Compile {
                    message: "checked temporal mediation requires series data".into(),
                });
            };
            return self.stamp(&input, operation.execute(series, ctx)?);
        }
        if let PreparedExecution::TemporalDagEffect(operation) = &self.execution {
            let graph =
                self.analysis.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
                    message: "checked temporal effect lost its prepared TemporalDag".into(),
                })?;
            if !operation.operation().matches_graph(graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal effect graph differs from prepared graph".into(),
                });
            }
            return self.stamp(&input, operation.execute(data, ctx)?);
        }
        if let PreparedExecution::TemporalClassEffect(operation) = &self.execution {
            if !operation.operation().matches_source_graph(&self.analysis.graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal class source graph differs from the retained proof"
                        .into(),
                });
            }
            return self.stamp(&input, operation.execute(data, ctx)?);
        }
        if let PreparedExecution::TemporalDagResponse(operation) = &self.execution {
            let graph =
                self.analysis.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
                    message: "checked temporal response lost its prepared TemporalDag".into(),
                })?;
            if !operation.operation().matches_graph(graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal response graph differs from prepared graph".into(),
                });
            }
            return self.stamp(&input, operation.execute(data, ctx)?);
        }
        let result = self.analysis.execute_on(&input, &self.plan, ctx)?;
        self.stamp(&input, result)
    }

    /// Re-estimate a prepared panel Pulse/Sustained analysis (no re-identify).
    ///
    /// Compatible new units and observations are admitted when schema and
    /// sampling regularity match. Incompatible schema, time regularity, or an
    /// empty panel is refused and the retained handle is unchanged.
    ///
    /// # Errors
    ///
    /// Schema / regularity mismatch, empty panel, or estimation failures.
    pub fn estimate_panel(
        &self,
        data: &PanelData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_panel_compatible(data)?;
        let input = DataInput::Panel(data.clone());
        let result = self.analysis.execute_on(&input, &self.plan, ctx)?;
        self.stamp(&input, result)
    }

    /// Replace retained panel data and re-estimate without re-identifying.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate_panel`].
    pub fn refresh_panel(
        &mut self,
        data: PanelData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_panel_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = DataInput::Panel(data);
        let result = refreshed.execute(&self.plan, ctx)?;
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }

    /// Re-estimate a prepared multi-environment temporal analysis (no re-identify).
    ///
    /// # Errors
    ///
    /// Schema / regularity mismatch, or estimation failures.
    pub fn estimate_multi_env(
        &self,
        data: &antecedent_data::MultiEnvironmentData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_multi_env_compatible(data)?;
        let input = DataInput::MultiEnv(data.clone());
        let result = self.analysis.execute_on(&input, &self.plan, ctx)?;
        self.stamp(&input, result)
    }

    /// Replace retained multi-environment data and re-estimate without re-identifying.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate_multi_env`].
    pub fn refresh_multi_env(
        &mut self,
        data: antecedent_data::MultiEnvironmentData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_multi_env_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = DataInput::MultiEnv(data);
        let result = refreshed.execute(&self.plan, ctx)?;
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }

    fn ensure_multi_env_compatible(
        &self,
        data: &antecedent_data::MultiEnvironmentData,
    ) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::MultiEnv)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared multi-env analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        for env in data.environments() {
            self.ensure_regularity(&env.time_index().regularity)?;
        }
        Ok(())
    }

    fn ensure_panel_compatible(&self, data: &PanelData) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::Panel)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared panel analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        if data.unit_count() == 0 {
            return Err(CausalError::Compile {
                message: "prepared panel refresh refuses an empty panel; prior state is retained"
                    .into(),
            });
        }
        for unit in data.units() {
            self.ensure_regularity(&unit.series.time_index().regularity)?;
        }
        Ok(())
    }

    /// Estimate using the data retained by preparation or the latest successful refresh.
    ///
    /// Uses the existing modality-specific executor and preserves identification caches.
    ///
    /// # Errors
    ///
    /// Execution failures, or an unsupported retained data modality.
    pub fn estimate_retained(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        match &self.analysis.data {
            DataInput::Tabular(data) => self.estimate(data, ctx),
            DataInput::Temporal(data) | DataInput::Event(data) => self.estimate_series(data, ctx),
            DataInput::Panel(data) => self.estimate_panel(data, ctx),
            DataInput::MultiEnv(data) => self.estimate_multi_env(data, ctx),
        }
    }

    /// Replace retained series and re-estimate.
    ///
    /// The handle is updated only when estimation succeeds; a refused or
    /// failed refresh leaves the retained series unchanged.
    ///
    /// # Errors
    ///
    /// Schema / time-index mismatch, or estimation failures.
    pub fn refresh_series(
        &mut self,
        data: TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_series_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = self.series_input(data);
        let result = if let PreparedExecution::TemporalMediation(operation) = &self.execution {
            let graph = refreshed.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
                message: "checked temporal mediation refresh lost its TemporalDag".into(),
            })?;
            if !operation.matches_graph(graph) {
                return Err(CausalError::Compile {
                    message:
                        "checked temporal mediation refresh graph differs from its retained proof"
                            .into(),
                });
            }
            let (DataInput::Temporal(series) | DataInput::Event(series)) = &refreshed.data else {
                unreachable!("refresh_series constructs temporal data")
            };
            operation.execute(series, ctx)?
        } else if let PreparedExecution::TemporalDagEffect(operation) = &self.execution {
            let graph = refreshed.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
                message: "checked temporal effect refresh lost its TemporalDag".into(),
            })?;
            if !operation.operation().matches_graph(graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal effect refresh graph differs from prepared graph"
                        .into(),
                });
            }
            let (DataInput::Temporal(series) | DataInput::Event(series)) = &refreshed.data else {
                unreachable!("refresh_series constructs temporal data")
            };
            operation.execute(series, ctx)?
        } else if let PreparedExecution::TemporalClassEffect(operation) = &self.execution {
            if !operation.operation().matches_source_graph(&refreshed.graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal class refresh graph differs from its retained proof"
                        .into(),
                });
            }
            let (DataInput::Temporal(series) | DataInput::Event(series)) = &refreshed.data else {
                unreachable!("refresh_series constructs temporal data")
            };
            operation.execute(series, ctx)?
        } else if let PreparedExecution::TemporalDagResponse(operation) = &self.execution {
            let temporal_graph =
                refreshed.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
                    message: "checked temporal response refresh lost its TemporalDag".into(),
                })?;
            if !operation.operation().matches_graph(temporal_graph) {
                return Err(CausalError::Compile {
                    message: "checked temporal response refresh graph differs from prepared graph"
                        .into(),
                });
            }
            let (DataInput::Temporal(series) | DataInput::Event(series)) = &refreshed.data else {
                unreachable!("refresh_series constructs temporal data")
            };
            operation.execute(series, ctx)?
        } else {
            refreshed.execute(&self.plan, ctx)?
        };
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }
}

/// Sampling regularity shared by every unit of a prepare-time panel.
///
/// Refresh requires every unit to match the frozen regularity, so prepare
/// refuses a panel whose units already disagree rather than freezing one
/// unit's grid and then refusing the same data on refresh.
fn multi_env_regularity(
    multi: &antecedent_data::MultiEnvironmentData,
) -> Result<antecedent_data::SamplingRegularity, CausalError> {
    let first = multi
        .environment(0)
        .map_err(|err| CausalError::Compile { message: err.to_string() })?
        .time_index()
        .regularity
        .clone();
    if multi.environments().iter().any(|env| env.time_index().regularity != first) {
        return Err(CausalError::Compile {
            message: "PreparedStudy requires every multi-env series to share one time-index \
                      regularity; align the environments before preparing"
                .into(),
        });
    }
    Ok(first)
}

impl Study {
    /// Compile once into a durable [`PreparedStudy`] for re-estimate-many.
    ///
    /// Supports:
    /// - tabular [`CausalQuery::AverageEffect`] on a supplied static graph
    ///   ([`GraphClass::Dag`], [`GraphClass::Cpdag`], [`GraphClass::Pag`], or
    ///   [`GraphClass::Admg`])
    /// - tabular [`CausalQuery::AverageEffect`], [`CausalQuery::ConditionalEffect`],
    ///   and static [`CausalQuery::Response`] on a supplied DAG graph posterior
    /// - tabular [`CausalQuery::Response`] and [`CausalQuery::ConditionalEffect`]
    ///   on a supplied [`GraphClass::Dag`], [`GraphClass::Cpdag`],
    ///   [`GraphClass::Pag`], or [`GraphClass::Admg`] (joint Response also on a
    ///   CoDetermined tier closure)
    /// - tabular [`CausalQuery::Distribution`] on a supplied [`GraphClass::Dag`]
    ///   or [`GraphClass::Admg`] (ADMG: unconditional, validation none)
    /// - tabular [`CausalQuery::PathSpecific`], static [`CausalQuery::Mediation`],
    ///   and [`CausalQuery::Counterfactual`] on a supplied [`GraphClass::Dag`]
    /// - series temporal [`CausalQuery::Response`] on a supplied
    ///   [`GraphClass::TemporalDag`], [`GraphClass::TemporalCpdag`], or
    ///   [`GraphClass::TemporalPag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained)
    ///   on a supplied [`GraphClass::TemporalDag`], [`GraphClass::TemporalCpdag`],
    ///   or [`GraphClass::TemporalPag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step or
    ///   multi-step Sustained) on a supplied DBN graph posterior
    /// - series temporal [`CausalQuery::Response`] on a supplied DBN graph
    ///   posterior (TemporalDag atoms; MeanCurve none, one-coordinate
    ///   InterventionResponse none/cheap/full)
    /// - scalar-horizon series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied [`GraphClass::TemporalDag`] or [`GraphClass::TemporalCpdag`]
    ///   (horizon-specific `I(h)`)
    /// - scalar-horizon series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied DBN graph posterior (per-atom `I(h)`, unidentified mass retained)
    /// - panel [`CausalQuery::TemporalEffect`] (Pulse / Sustained) on a supplied
    ///   [`GraphClass::TemporalDag`]; every unit must share one time-index regularity
    ///
    /// The handle records the prepare-time data modality (tabular, series, or
    /// panel) and its estimate / refresh entry points refuse other modalities.
    /// Discovery inputs and review-required compiles are refused.
    ///
    /// # Errors
    ///
    /// Unsupported combination, compile failure, or review-required plan.
    pub fn prepare(&self, ctx: &ExecutionContext) -> Result<PreparedStudy, CausalError> {
        ensure_prepared_supported(self)?;
        let plan = self.compile(ctx)?;
        let counterfactual = match (&self.graph, &self.query) {
            (_, CausalQuery::Counterfactual(query)) => {
                let graph = self
                    .graph
                    .as_dag()
                    .ok_or(CausalError::Unsupported { message: "counterfactual requires Dag" })?;
                Some(crate::gcm::CheckedCounterfactualOperation::compile(
                    graph.clone(),
                    query.clone(),
                )?)
            }
            _ => None,
        };
        let nested_counterfactual = match (&self.graph, &self.query) {
            (_, CausalQuery::NestedCounterfactual(query)) => {
                let graph = self.graph.as_dag().ok_or(CausalError::Unsupported {
                    message: "cross_world_not_identified: natural direct effect requires the licensed three-node DAG",
                })?;
                Some(crate::gcm::NestedCounterfactualOperation::compile(graph.clone(), *query)?)
            }
            _ => None,
        };
        let (schema, modality, time_regularity) = match &self.data {
            DataInput::Tabular(data) => (data.schema().clone(), PreparedModality::Tabular, None),
            DataInput::Temporal(data) | DataInput::Event(data) => (
                data.schema().clone(),
                PreparedModality::Series,
                Some(data.time_index().regularity.clone()),
            ),
            DataInput::Panel(panel) => (
                panel.schema().clone(),
                PreparedModality::Panel,
                Some(super::builder::panel_shared_regularity(panel)?),
            ),
            DataInput::MultiEnv(multi) => (
                multi.schema().clone(),
                PreparedModality::MultiEnv,
                Some(multi_env_regularity(multi)?),
            ),
        };
        let mut analysis = self.clone();
        match (&self.data, &self.query, self.graph_posterior.as_ref()) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, &query.inner, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(posterior))
                if !query.is_temporal() =>
            {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    if matches!(
                        posterior.atom_kind,
                        antecedent_discovery::GraphPosteriorAtomKind::Admg
                    ) {
                        build_admg_graph_posterior_response_identification_cache(
                            posterior, query, ctx,
                        )?
                    } else {
                        let (treatment, outcome) =
                            query.functional.primary_pair().ok_or_else(|| {
                                CausalError::Compile {
                                    message: "response query has no treatment/outcome pair".into(),
                                }
                            })?;
                        build_graph_posterior_identification_cache(
                            posterior,
                            &AverageEffectQuery::binary_ate(treatment, outcome),
                            ctx,
                        )?
                    },
                ));
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    analysis.temporal_class_posterior_identification_cache =
                        Some(Arc::new(build_temporal_class_posterior_identification_cache(
                            posterior,
                            &variables,
                            query,
                            analysis.max_completions,
                            ctx,
                        )?));
                } else {
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_identification_cache(
                            posterior, &variables, query, ctx,
                        )?));
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    let horizon =
                        query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
                            message: "temporal class graph-posterior mediation requires a horizon"
                                .into(),
                        })?;
                    let mut witness =
                        TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
                    witness.horizon_steps = horizon;
                    analysis.temporal_class_posterior_identification_cache =
                        Some(Arc::new(build_temporal_class_posterior_identification_cache(
                            posterior,
                            &variables,
                            &witness,
                            analysis.max_completions,
                            ctx,
                        )?));
                } else {
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_mediation_identification_cache(
                            posterior, &variables, query, ctx,
                        )?));
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Response(query),
                Some(posterior),
            ) if query.is_temporal() => {
                super::execute::dbn_posterior_response_supported(query)?;
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    let (treatment, outcome) =
                        query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                            message: "temporal class graph-posterior response has no \
                                      treatment/outcome pair"
                                .into(),
                        })?;
                    let horizon = query
                        .temporal
                        .as_ref()
                        .and_then(|spec| spec.horizons.first())
                        .copied()
                        .unwrap_or(1);
                    let mut witness = TemporalEffectQuery::pulse(treatment, outcome, 1.0);
                    witness.horizon_steps = horizon;
                    if let Some(temporal) = query.temporal.as_ref() {
                        witness.policy = temporal.policy.clone();
                        witness.max_history_lag = temporal.max_history_lag;
                    }
                    let cache = build_temporal_class_posterior_identification_cache(
                        posterior,
                        &variables,
                        &witness,
                        analysis.max_completions,
                        ctx,
                    )?;
                    if let Some(atom) = cache.class_atoms.first() {
                        analysis.temporal_class_identification_cache =
                            Some(Arc::new(CachedTemporalClassIdentification {
                                envelope: atom.envelope.clone(),
                                by_horizon: vec![(horizon, atom.envelope.clone())],
                            }));
                    }
                    analysis.temporal_class_posterior_identification_cache = Some(Arc::new(cache));
                } else {
                    let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
                        crate::strategy_table::EstimatorId::TemporalResponseBayesian
                    } else {
                        crate::strategy_table::EstimatorId::TemporalResponseGcomp
                    };
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_response_identification_cache(
                            posterior,
                            &variables,
                            query,
                            estimator_id,
                            ctx,
                        )?));
                }
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_), None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_mediation_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query), None) => {
                let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
                    message: "TransportQuery prepare requires a selection diagram",
                })?;
                analysis.transport_identification_cache =
                    Some(Arc::new(super::execute::live_transport_identification(diagram, query)?));
            }
            (DataInput::Tabular(_), _, None) => {
                analysis.identification_cache =
                    self.prepare_static_identification(&plan)?.map(Arc::new);
                analysis.pag_identification_cache =
                    self.prepare_pag_identification(&plan)?.map(Arc::new);
                analysis.cpdag_identification_cache =
                    self.prepare_cpdag_identification(&plan)?.map(Arc::new);
            }
            (
                DataInput::Panel(_),
                CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                None,
            ) => {
                if analysis.graph.class().is_incomplete_temporal() {
                    analysis.temporal_class_identification_cache =
                        self.prepare_temporal_class_identification()?.map(Arc::new);
                } else {
                    analysis.temporal_identification_cache =
                        self.prepare_temporal_identification()?.map(Arc::new);
                }
            }
            (_, _, None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (_, _, Some(_)) => {
                // `ensure_prepared_supported` already refuses this coordinate; a
                // library must still answer with a typed refusal, never a panic.
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "graph_posterior on the prepared handle is licensed only for \
                        tabular AverageEffect/Response/ConditionalEffect and series \
                        TemporalEffect, TemporalMediationEffect, or TemporalDag Response",
                });
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        // The posterior builders report `identify.compute` themselves; the
        // single-graph, PAG, and temporal caches identify above without a
        // progress hook, so report once here when one of them was built. A
        // sharp-RD prepare builds no cache and reports nothing: its clicks do.
        if analysis.identification_cache.is_some()
            || analysis.pag_identification_cache.is_some()
            || analysis.cpdag_identification_cache.is_some()
            || analysis.temporal_identification_cache.is_some()
            || analysis.temporal_class_identification_cache.is_some()
            || analysis.transport_identification_cache.is_some()
        {
            super::execute::report_identify_compute(ctx);
        }
        let checked_linear = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            self.graph.class(),
            self.graph_posterior.is_none(),
            self.tiered.is_none(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                GraphClass::Dag,
                true,
                true,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::LinearAdjustmentAte.as_str()) =>
            {
                let default_id = !matches!(
                    &self.estimator_spec,
                    Some(crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(_))
                );
                let mut fitter = match &self.estimator_spec {
                    Some(crate::estimator_spec::EstimatorSpec::LinearAdjustmentAte(cfg)) => {
                        (**cfg).clone()
                    }
                    _ => antecedent_estimate::LinearAdjustmentAte::new(),
                };
                if default_id {
                    fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                }
                let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ESTIMATOR)
                    .parse()?;
                Some(CheckedLinearOperation {
                    fitter,
                    preparation,
                    default_id,
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier,
                    estimator,
                    physical: plan.clone(),
                    inference: analysis.inference.clone(),
                    refute: analysis.refute,
                    graph_class: analysis.graph.class(),
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|validator| Arc::from(validator.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let checked_glm = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && !matches!(query.active, Intervention::Set { variable, .. } if variable != query.treatment)
                && !matches!(query.control, Intervention::Set { variable, .. } if variable != query.treatment)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::GlmAdjustment.as_str())
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                ) =>
            {
                let mut fitter = match &analysis.estimator_spec {
                    Some(crate::estimator_spec::EstimatorSpec::GlmAdjustment(config)) => {
                        (**config).clone()
                    }
                    _ => {
                        let mut fitter = antecedent_estimate::GlmAdjustmentAte::new();
                        fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                        fitter.population_registry = analysis.population_registry.clone();
                        fitter
                    }
                };
                if fitter.population_registry.is_none() {
                    fitter.population_registry = analysis.population_registry.clone();
                }
                let preparation = fitter.prepare(data, &cache.estimand, query)?;
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ESTIMATOR)
                    .parse()?;
                Some(CheckedGlmAdjustmentOperation {
                    source_query: CausalQuery::AverageEffect(query.clone()),
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier,
                    estimator,
                    physical: plan.clone(),
                    fitter,
                    preparation,
                    inference: analysis.inference.clone(),
                    refute: analysis.refute,
                    graph_class: analysis.graph.class(),
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|v| Arc::from(v.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let checked_rd = match (
            &self.data,
            &self.query,
            analysis.graph.as_dag(),
            analysis.rd,
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(graph),
                Some(config),
                InferenceMode::Frequentist,
            ) if plan.logical.record.estimator.as_deref()
                == Some(crate::strategy_table::EstimatorId::RdSharp.as_str())
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                ) =>
            {
                let identification = antecedent_identify::SharpRdIdentifier::new(
                    antecedent_identify::SharpRdConfig::new(
                        config.running_variable,
                        config.cutoff,
                        config.bandwidth,
                    ),
                )
                .identify_on(graph, CausalQuery::AverageEffect(query.clone()))
                .map_err(CausalError::from)?;
                crate::strategy_table::require_identified(&identification)?;
                let estimand = crate::strategy_table::select_estimand(
                    &identification,
                    crate::strategy_table::EstimatorId::RdSharp,
                )?;
                let identified_query =
                    identification.average_effect().cloned().unwrap_or_else(|| query.clone());
                let mut fitter = antecedent_estimate::SharpRegressionDiscontinuity::new(
                    config.running_variable,
                    config.cutoff,
                    config.bandwidth,
                );
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                fitter.se_kind = config.se_kind;
                let preparation = fitter.prepare_checked(data, &identification, 0)?;
                let identifier = crate::strategy_table::IdentifierId::RdSharp;
                let estimator = crate::strategy_table::EstimatorId::RdSharp;
                Some(CheckedRdOperation {
                    source_query: query.clone(),
                    query: identified_query,
                    identification,
                    estimand,
                    identifier,
                    estimator,
                    physical: plan.clone(),
                    fitter,
                    preparation,
                    inference: analysis.inference.clone(),
                    refute: analysis.refute,
                    graph_class: analysis.graph.class(),
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|v| Arc::from(v.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let checked_conditional = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && query.inner.effect_modifiers.len() == 1
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                )
                && plan.logical.record.estimator.as_deref()
                    == Some(
                        crate::strategy_table::EstimatorId::ConditionalLinearAdjustment.as_str(),
                    ) =>
            {
                Some(CheckedConditionalOperation::prepare(
                    data,
                    query.clone(),
                    cache.identification.clone(),
                    cache.estimand.clone(),
                    plan.clone(),
                    analysis.refute,
                    analysis.graph.version(),
                    analysis.support_status,
                    analysis.structure_source,
                    analysis.population_registry.clone(),
                    analysis.latency_mode,
                )?)
            }
            _ => None,
        };
        let checked_propensity = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.refute,
                    RefuteSuite::None
                        | RefuteSuite::Cheap
                        | RefuteSuite::PlaceboAndRcc
                        | RefuteSuite::Full
                )
                && matches!(
                    plan.logical.record.estimator.as_deref(),
                    Some("propensity.weighting" | "propensity.matching")
                ) =>
            {
                let estimator: crate::strategy_table::EstimatorId = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .expect("checked propensity estimator")
                    .parse()?;
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    .parse()?;
                let estimand_index = cache
                    .identification
                    .estimands
                    .iter()
                    .position(|candidate| {
                        candidate.functional == cache.estimand.functional
                            && candidate.method == cache.estimand.method
                            && candidate.adjustment_set == cache.estimand.adjustment_set
                    })
                    .ok_or_else(|| CausalError::Compile {
                        message:
                            "selected propensity estimand is absent from its identification product"
                                .into(),
                    })?;
                let fitter = match (&analysis.estimator_spec, estimator) {
                    (
                        Some(crate::estimator_spec::EstimatorSpec::PropensityWeighting(config)),
                        crate::strategy_table::EstimatorId::PropensityWeighting,
                    ) => CheckedPropensityEstimator::Weighting((**config).clone()),
                    (
                        Some(crate::estimator_spec::EstimatorSpec::PropensityMatching(config)),
                        crate::strategy_table::EstimatorId::PropensityMatching,
                    ) => CheckedPropensityEstimator::Matching((**config).clone()),
                    (_, crate::strategy_table::EstimatorId::PropensityWeighting) => {
                        let mut fitter = antecedent_estimate::PropensityWeighting::new();
                        fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                        CheckedPropensityEstimator::Weighting(fitter)
                    }
                    (_, crate::strategy_table::EstimatorId::PropensityMatching) => {
                        let mut fitter = antecedent_estimate::PropensityMatching::new();
                        fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                        CheckedPropensityEstimator::Matching(fitter)
                    }
                    _ => unreachable!(),
                };
                let context = CheckedPropensityContext {
                    source_query: analysis.query.clone(),
                    identification: cache.identification.clone(),
                    estimand_index,
                    identifier,
                    physical: plan.clone(),
                    graph_class: analysis.graph.class(),
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    inference: analysis.inference.clone(),
                    refute: analysis.refute,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from([]),
                };
                Some(CheckedPropensityOperation::prepare(data, context, estimator, fitter)?)
            }
            _ => None,
        };
        let checked_aipw = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
            self.graph.class(),
            self.graph_posterior.is_none(),
            self.tiered.is_none(),
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
                GraphClass::Dag,
                true,
                true,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && matches!(&query.active, antecedent_core::Intervention::Set { variable, value }
                        if *variable == query.treatment && value.as_f64() == Some(1.0))
                && matches!(&query.control, antecedent_core::Intervention::Set { variable, value }
                        if *variable == query.treatment && value.as_f64() == Some(0.0))
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::Aipw.as_str()) =>
            {
                let fitter = checked_aipw_fitter(&analysis);
                if matches!(fitter.overlap, OverlapPolicy::RequireDiagnostics { trim: None, .. }) {
                    let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                    let identifier = plan
                        .logical
                        .record
                        .identifier
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                        .parse()?;
                    let estimator = plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_ESTIMATOR)
                        .parse()?;
                    Some(CheckedAipwOperation {
                        fitter,
                        preparation,
                        query: query.clone(),
                        identification: cache.identification.clone(),
                        estimand: cache.estimand.clone(),
                        identifier,
                        estimator,
                        physical: plan.clone(),
                        inference: analysis.inference.clone(),
                        refute: analysis.refute,
                        graph_class: analysis.graph.class(),
                        graph_version: analysis.graph.version(),
                        support_status: analysis.support_status,
                        structure_source: analysis.structure_source,
                        population_registry: analysis.population_registry.clone(),
                        latency_mode: analysis.latency_mode,
                        custom_validator_names: Arc::from(
                            analysis
                                .custom_validators
                                .iter()
                                .map(|validator| Arc::from(validator.name()))
                                .collect::<Vec<_>>(),
                        ),
                    })
                } else {
                    None
                }
            }
            _ => None,
        };
        let checked_frontdoor_linear = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && plan.logical.record.estimator.as_deref()
                    == Some(crate::strategy_table::EstimatorId::FrontDoorTwoStage.as_str()) =>
            {
                let fitter = match &analysis.estimator_spec {
                    Some(crate::estimator_spec::EstimatorSpec::FrontDoorTwoStage(config)) => {
                        (**config).clone()
                    }
                    _ => antecedent_estimate::FrontDoorTwoStage::new(),
                };
                let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                Some(CheckedFrontDoorOperation { fitter, preparation })
            }
            _ => None,
        };
        let checked_iv = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(_),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist,
            ) => {
                let estimator = plan.logical.record.estimator.as_deref().unwrap_or("");
                match (&analysis.estimator_spec, estimator) {
                    (Some(crate::estimator_spec::EstimatorSpec::IvWald(cfg)), "iv.wald") => {
                        Some(CheckedIvOperation::Wald {
                            fitter: (**cfg).clone(),
                            preparation: (**cfg).prepare_checked(data, &cache.identification, 0)?,
                        })
                    }
                    (None, "iv.wald")
                    | (
                        Some(crate::estimator_spec::EstimatorSpec::Default(
                            crate::EstimatorId::IvWald,
                        )),
                        "iv.wald",
                    ) => {
                        let fitter = antecedent_estimate::WaldIv::new();
                        let preparation = fitter.prepare_checked(data, &cache.identification, 0)?;
                        Some(CheckedIvOperation::Wald { fitter, preparation })
                    }
                    (Some(crate::estimator_spec::EstimatorSpec::Iv2Sls(cfg)), "iv.2sls") => {
                        match (**cfg).prepare_checked(data, &cache.identification, 0) {
                            Ok(preparation) => Some(CheckedIvOperation::TwoSls {
                                fitter: (**cfg).clone(),
                                preparation,
                            }),
                            Err(antecedent_estimate::EstimationError::Unsupported {
                                message:
                                    "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                            }) => None,
                            Err(error) => return Err(error.into()),
                        }
                    }
                    (None, "iv.2sls")
                    | (
                        Some(crate::estimator_spec::EstimatorSpec::Default(
                            crate::EstimatorId::Iv2Sls,
                        )),
                        "iv.2sls",
                    ) => {
                        let fitter = antecedent_estimate::TwoStageLeastSquares::new();
                        match fitter.prepare_checked(data, &cache.identification, 0) {
                            Ok(preparation) => {
                                Some(CheckedIvOperation::TwoSls { fitter, preparation })
                            }
                            Err(antecedent_estimate::EstimationError::Unsupported {
                                message:
                                    "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                            }) => None,
                            Err(error) => return Err(error.into()),
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        let distribution_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::Distribution(query),
                Some(cache),
                graph_class,
                inference,
            ) => {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_DISTRIBUTION_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_DISTRIBUTION_ESTIMATOR)
                    .parse()?;
                let mut fitter = antecedent_estimate::FunctionalDistribution::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let prepared = fitter.prepare(
                    data,
                    query,
                    &cache.estimand,
                    &cache.identification.arena,
                    cache.identification.required_assumptions.clone(),
                )?;
                Some(CheckedDistributionOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier,
                    estimator,
                    physical: plan.clone(),
                    fitter,
                    prepared,
                    inference: inference.clone(),
                    refute: analysis.refute,
                    graph_class,
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|validator| Arc::from(validator.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let functional_effect_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(query),
                Some(cache),
                GraphClass::Admg,
                inference,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && cache.estimand.method_kind().ok()
                    == Some(antecedent_expr::EstimandMethod::GeneralId)
                && plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER)
                    == crate::strategy_table::IdentifierId::GeneralId.as_str()
                && plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR)
                    == crate::strategy_table::EstimatorId::FunctionalEffect.as_str()
                && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved) =>
            {
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let prepared = fitter.prepare(
                    data,
                    &cache.estimand,
                    &cache.identification.arena,
                    cache.identification.required_assumptions.clone(),
                    &[query.treatment, query.outcome],
                )?;
                Some(CheckedFunctionalEffectOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier: crate::strategy_table::IdentifierId::GeneralId,
                    estimator: crate::strategy_table::EstimatorId::FunctionalEffect,
                    physical: plan.clone(),
                    fitter,
                    prepared,
                    inference: inference.clone(),
                    refute: analysis.refute,
                    graph_class: GraphClass::Admg,
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|v| Arc::from(v.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let path_specific_effect_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::PathSpecific(query),
                Some(cache),
                GraphClass::Dag,
                inference,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && cache.estimand.method_kind().ok()
                    == Some(antecedent_expr::EstimandMethod::PathSpecificNatural)
                && plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_PATH_IDENTIFIER)
                    == crate::strategy_table::IdentifierId::PathSpecificNatural.as_str()
                && plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_PATH_ESTIMATOR)
                    == crate::strategy_table::EstimatorId::FunctionalEffect.as_str()
                && matches!(query.target_population, TargetPopulation::AllObserved) =>
            {
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let mut extra = vec![query.treatment, query.outcome];
                extra.extend(query.path_nodes.iter().copied());
                let prepared = fitter.prepare(
                    data,
                    &cache.estimand,
                    &cache.identification.arena,
                    cache.identification.required_assumptions.clone(),
                    &extra,
                )?;
                Some(CheckedPathSpecificEffectOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier: crate::strategy_table::IdentifierId::PathSpecificNatural,
                    estimator: crate::strategy_table::EstimatorId::FunctionalEffect,
                    physical: plan.clone(),
                    fitter,
                    prepared,
                    inference: inference.clone(),
                    refute: analysis.refute,
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|v| Arc::from(v.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let admg_response_curve_operation = match (&self.data, &self.query, analysis.graph.class())
        {
            (DataInput::Tabular(data), CausalQuery::Response(query), GraphClass::Admg)
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == TargetPopulation::AllObserved
                    && query.outcome_functional == OutcomeFunctional::Mean
                    && analysis.refute == RefuteSuite::None
                    && plan
                        .logical
                        .record
                        .identifier
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER)
                        == crate::strategy_table::IdentifierId::GeneralId.as_str()
                    && plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR)
                        == crate::strategy_table::EstimatorId::FunctionalEffect.as_str()
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    )
                    && matches!(
                        analysis.inference,
                        InferenceMode::Frequentist | InferenceMode::Bayesian(_)
                    ) =>
            {
                let antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } =
                    &query.functional
                else {
                    unreachable!()
                };
                let grid = treatment
                    .grid
                    .values()
                    .map_err(|error| CausalError::Compile { message: error.to_string() })?;
                if grid.is_empty() {
                    return Err(CausalError::Compile {
                        message: "ADMG response curve has no grid members".into(),
                    });
                }
                let admg = analysis.graph.as_admg().ok_or_else(|| CausalError::Compile {
                    message: "ADMG response operation lacks an ADMG graph".into(),
                })?;
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = analysis.bootstrap_replicates;
                let mut members = Vec::with_capacity(grid.len());
                for level in &grid {
                    let mut member_query = query.clone();
                    member_query.functional =
                        antecedent_core::ResponseFunctional::InterventionResponse {
                            outcome: *outcome,
                            interventions: Arc::from([Intervention::set(
                                treatment.variable,
                                Value::f64(*level),
                            )]),
                        };
                    let member_causal_query = CausalQuery::Response(member_query.clone());
                    let identification = crate::strategy_table::identify_admg_query(
                        crate::strategy_table::IdentifierId::GeneralId,
                        admg,
                        &member_causal_query,
                    )?;
                    let estimand = crate::strategy_table::select_estimand(
                        &identification,
                        crate::strategy_table::EstimatorId::FunctionalEffect,
                    )?;
                    let prepared = fitter.prepare(
                        data,
                        &estimand,
                        &identification.arena,
                        identification.required_assumptions.clone(),
                        &[treatment.variable, *outcome],
                    )?;
                    members.push(CheckedFunctionalEffectResponseMember {
                        grid_value: *level,
                        query: member_query,
                        identification,
                        estimand,
                        prepared,
                    });
                }
                Some(CheckedAdmgResponseCurveOperation {
                    query: query.clone(),
                    grid: Arc::from(grid),
                    members: Arc::from(members),
                    identifier: crate::strategy_table::IdentifierId::GeneralId,
                    estimator: crate::strategy_table::EstimatorId::FunctionalEffect,
                    physical: plan.clone(),
                    fitter,
                    inference: analysis.inference.clone(),
                    graph_version: analysis.graph.version(),
                    support_status: analysis.support_status,
                    structure_source: analysis.structure_source,
                    population_registry: analysis.population_registry.clone(),
                    latency_mode: analysis.latency_mode,
                    custom_validator_names: Arc::from(
                        analysis
                            .custom_validators
                            .iter()
                            .map(|v| Arc::from(v.name()))
                            .collect::<Vec<_>>(),
                    ),
                })
            }
            _ => None,
        };
        let bayesian_gcomp_operation = match (
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                CausalQuery::AverageEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(query.target_population, TargetPopulation::AllObserved)
                && estimator == crate::strategy_table::EstimatorId::BayesianGcomp.as_str() =>
            {
                Some(CheckedBayesianGcompOperation {
                    query: query.clone(),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    inference: analysis.inference.clone(),
                })
            }
            _ => None,
        };
        let bayesian_gcomp_execution = if let Some(operation) = bayesian_gcomp_operation {
            let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                message: "checked Bayesian g-computation requires its retained DAG".into(),
            })?;
            Some(super::execute::CheckedBayesianDagAteExecution::checked(
                graph,
                operation,
                super::execute::IdentifiedResultContext::from_study(&analysis),
                plan.clone(),
                analysis.refute,
                analysis.latency_mode,
                analysis.custom_validators.clone(),
                analysis.stage_sink.clone(),
            )?)
        } else {
            None
        };
        let bayesian_conditional_execution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            &analysis.inference,
            analysis.graph.class(),
            plan.logical.record.estimator.as_deref(),
        ) {
            (
                DataInput::Tabular(_),
                CausalQuery::ConditionalEffect(query),
                Some(cache),
                InferenceMode::Bayesian(_),
                GraphClass::Dag,
                Some(estimator),
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                )
                && estimator
                    == crate::strategy_table::EstimatorId::BayesianConditional.as_str() =>
            {
                let graph = analysis.graph.as_dag().ok_or(CausalError::Compile {
                    message: "checked Bayesian conditional effect requires its retained DAG".into(),
                })?;
                Some(super::execute::CheckedBayesianConditionalOperation::checked(
                    graph,
                    query.clone(),
                    cache.identification.clone(),
                    cache.estimand.clone(),
                    analysis.inference.clone(),
                    analysis.refute,
                    super::execute::IdentifiedResultContext::from_study(&analysis),
                    plan.clone(),
                    analysis.latency_mode,
                    analysis.custom_validators.clone(),
                    analysis.stage_sink.clone(),
                )?)
            }
            _ => None,
        };
        let derivative_response_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && analysis.refute == RefuteSuite::None
                    && analysis.custom_validators.is_empty()
                    && matches!(
                        analysis.inference,
                        InferenceMode::Frequentist | InferenceMode::Bayesian(_)
                    )
                    && query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == TargetPopulation::AllObserved
                    && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::PointDerivative { .. }
                            | antecedent_core::ResponseFunctional::AverageDerivative { .. }
                            | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
                            | antecedent_core::ResponseFunctional::Jacobian { .. }
                    ) =>
            {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                    .parse()?;
                Some(super::execute::CheckedDerivativeResponseOperation::checked(
                    analysis.graph.as_dag().expect("checked Dag response"),
                    query,
                    &cache.identification,
                    &cache.estimand,
                    identifier,
                    estimator,
                    analysis.inference.clone(),
                    analysis.response_options.clone().unwrap_or_default(),
                )?)
            }
            _ => None,
        };
        let static_dag_response_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    )
                    && analysis.custom_validators.is_empty()
                    && matches!(analysis.inference, InferenceMode::Frequentist)
                    && query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == TargetPopulation::AllObserved
                    && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                    && (matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    ) && analysis.refute == RefuteSuite::None
                        || matches!(
                            query.functional,
                            antecedent_core::ResponseFunctional::InterventionResponse { .. }
                        )) =>
            {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                    .parse()?;
                if estimator == crate::strategy_table::EstimatorId::ResponseKennedyDr
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    )
                    || estimator == crate::strategy_table::EstimatorId::ResponseInterventionGcomp
                        && matches!(
                            query.functional,
                            antecedent_core::ResponseFunctional::InterventionResponse { .. }
                        )
                {
                    Some(super::execute::CheckedStaticDagResponseOperation::checked(
                        analysis.graph.as_dag().expect("checked DAG response"),
                        query,
                        &cache.identification,
                        &cache.estimand,
                        identifier,
                        estimator,
                        analysis.response_options.clone().unwrap_or_default(),
                        analysis.refute,
                    )?)
                } else {
                    None
                }
            }
            _ => None,
        };
        let cell_aipw_response_operation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
                if analysis.graph_posterior.is_none()
                    && analysis.tiered.is_none()
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    ) =>
            {
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                    .parse()?;
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                    .parse()?;
                if estimator == crate::strategy_table::EstimatorId::CellAipw
                    && analysis.custom_validators.is_empty()
                    && analysis.shared_batch_design.is_none()
                    && analysis.continuous_cell.is_none()
                    && matches!(analysis.inference, InferenceMode::Frequentist)
                {
                    Some(super::execute::CheckedCellAipwResponseOperation::checked(
                        analysis.graph.as_dag().expect("checked DAG response"),
                        query,
                        &cache.identification,
                        &cache.estimand,
                        identifier,
                        estimator,
                        ctx.rng.master_seed(),
                        analysis.refute,
                        if analysis.structure_source == crate::support::StructureSource::Accepted {
                            super::execute::DagResponseOrigin::Accepted
                        } else {
                            super::execute::DagResponseOrigin::Explicit
                        },
                    )?)
                } else {
                    None
                }
            }
            _ => None,
        };
        let temporal_mediation_operation = match (
            &self.data,
            &self.query,
            analysis.temporal_identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(_),
                Some(cache),
                GraphClass::TemporalDag,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none()
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                ) =>
            {
                Some(super::execute::CheckedTemporalMediationOperation::checked(
                    &analysis, data, &plan, cache,
                )?)
            }
            _ => None,
        };
        let temporal_dag_effect_operation = match (
            &self.data,
            &self.query,
            analysis.temporal_identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(query),
                Some(cache),
                GraphClass::TemporalDag,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.split.is_none()
                && analysis.custom_validators.is_empty()
                && matches!(analysis.inference, InferenceMode::Frequentist)
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                )
                && matches!(
                    query.policy,
                    antecedent_core::TemporalPolicy::Pulse { .. }
                        | antecedent_core::TemporalPolicy::Sustained { .. }
                ) =>
            {
                let identifier = crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded;
                let estimator = if query.is_multi_step_sustained() {
                    crate::strategy_table::EstimatorId::TemporalSequentialGcomp
                } else {
                    crate::strategy_table::EstimatorId::TemporalLinearAdjustment
                };
                if plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .is_some_and(|name| name != identifier.as_str())
                    || plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .is_some_and(|name| name != estimator.as_str())
                {
                    None
                } else {
                    let member =
                        cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                            message: "prepared temporal effect lacks its horizon proof".into(),
                        })?;
                    let operation = super::CheckedTemporalEffectOperation::checked(
                        analysis.graph.as_temporal_dag().expect("guarded TemporalDag"),
                        query,
                        member.identification.clone(),
                        member.estimand.clone(),
                        member.indexer.clone(),
                        identifier,
                        estimator,
                        analysis.bootstrap_replicates,
                        analysis.refute,
                    )?;
                    Some(super::execute::CheckedTemporalEffectExecution::checked(
                        operation,
                        super::execute::IdentifiedResultContext::from_study(&analysis),
                        plan.clone(),
                    )?)
                }
            }
            _ => None,
        };
        let temporal_class_effect_operation = match (
            &self.data,
            &self.query,
            analysis.temporal_class_identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(query),
                Some(cache),
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && matches!(analysis.inference, InferenceMode::Frequentist)
                && matches!(
                    query.policy,
                    antecedent_core::TemporalPolicy::Pulse { .. }
                        | antecedent_core::TemporalPolicy::Sustained { .. }
                ) =>
            {
                let identifier = crate::strategy_table::IdentifierId::GeneralizedAdjustment;
                let estimator = if query.is_multi_step_sustained() {
                    crate::strategy_table::EstimatorId::TemporalSequentialGcomp
                } else {
                    crate::strategy_table::EstimatorId::TemporalLinearAdjustment
                };
                if plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .is_some_and(|name| name != identifier.as_str())
                    || plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .is_some_and(|name| name != estimator.as_str())
                {
                    None
                } else {
                    let operation = super::CheckedTemporalClassEffectOperation::checked(
                        analysis.graph.clone(),
                        query,
                        cache.clone(),
                        analysis.max_completions,
                        identifier,
                        estimator,
                        analysis.bootstrap_replicates,
                        analysis.split.clone(),
                        analysis.refute,
                        analysis.custom_validators.clone(),
                    )?;
                    Some(super::execute::CheckedTemporalClassEffectExecution::checked(
                        operation,
                        super::execute::IdentifiedResultContext::from_study(&analysis),
                        plan.clone(),
                    )?)
                }
            }
            _ => None,
        };
        let temporal_dag_response_operation = match (
            &self.data,
            &self.query,
            analysis.temporal_identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::Response(query),
                Some(cache),
                GraphClass::TemporalDag,
            ) if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty()
                && matches!(analysis.inference, InferenceMode::Frequentist)
                && analysis.refute == RefuteSuite::None
                && query.temporal.is_some()
                && query.temporal.as_ref().is_some_and(|spec| match &spec.policy {
                    antecedent_core::TemporalPolicy::Pulse { .. } => true,
                    antecedent_core::TemporalPolicy::Sustained { from, until } => from == until,
                    antecedent_core::TemporalPolicy::Dynamic { .. } => false,
                    _ => false,
                })
                && query.observation == antecedent_core::ObservationSpec::Complete
                && query.target_population == TargetPopulation::AllObserved
                && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && matches!(
                    query.functional,
                    antecedent_core::ResponseFunctional::MeanCurve { .. }
                ) =>
            {
                let identifier = crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded;
                let estimator = crate::strategy_table::EstimatorId::TemporalResponseGcomp;
                if plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .is_some_and(|selected| selected != identifier.as_str())
                    || plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .is_some_and(|selected| selected != estimator.as_str())
                {
                    None
                } else {
                    let temporal = query.temporal.as_ref().expect("guarded temporal response");
                    let (treatment, outcome) =
                        query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                            message: "temporal response query has no treatment/outcome pair".into(),
                        })?;
                    let dose = single_step_dose(query)?.unwrap_or(1.0);
                    let evidence = cache
                        .by_horizon
                        .iter()
                        .map(|member| {
                            let effect_query = TemporalEffectQuery::pulse(treatment, outcome, dose)
                                .with_policy(temporal.policy.clone())
                                .with_horizon_steps(member.horizon)
                                .with_max_history_lag(temporal.max_history_lag)
                                .with_target_population(query.target_population.clone());
                            super::TemporalResponseHorizonEvidence::checked(
                                member.horizon,
                                effect_query,
                                member.identification.clone(),
                                member.estimand.clone(),
                                member.indexer.clone(),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let operation = super::CheckedTemporalResponseOperation::checked(
                        analysis.graph.as_temporal_dag().expect("checked TemporalDag response"),
                        query,
                        evidence,
                        identifier,
                        estimator,
                        analysis.bootstrap_replicates,
                    )?;
                    let execution = super::execute::CheckedTemporalResponseExecution::checked(
                        operation,
                        super::execute::IdentifiedResultContext::from_study(&analysis),
                        plan.clone(),
                    )?;
                    Some(execution)
                }
            }
            _ => None,
        };
        let checked_response_curve = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(cache), GraphClass::Dag)
                if query.temporal.is_none()
                    && query.observation == antecedent_core::ObservationSpec::Complete
                    && query.target_population == TargetPopulation::AllObserved
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::MeanCurve { .. }
                    )
                    && matches!(analysis.inference, InferenceMode::Frequentist)
                    && plan
                        .logical
                        .record
                        .identifier
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER)
                        == crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER
                    && plan
                        .logical
                        .record
                        .estimator
                        .as_deref()
                        .unwrap_or(crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR)
                        == crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR =>
            {
                let antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } =
                    &query.functional
                else {
                    unreachable!()
                };
                Some(CheckedStaticResponseCurve {
                    query: query.clone(),
                    grid: Arc::from(
                        treatment
                            .grid
                            .values()
                            .map_err(|e| CausalError::Compile { message: e.to_string() })?,
                    ),
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    identifier: crate::strategy_table::DEFAULT_RESPONSE_IDENTIFIER_ID,
                    estimator: crate::strategy_table::DEFAULT_RESPONSE_ESTIMATOR_ID,
                })
            }
            _ => None,
        };
        let checked_counterfactual = if let Some(operation) = counterfactual {
            let cache =
                analysis.identification_cache.as_deref().ok_or_else(|| CausalError::Compile {
                    message: "counterfactual preparation lost its identification product".into(),
                })?;
            let inference = analysis.inference.clone();
            let procedure = super::execute::CounterfactualProcedure::for_inference(&inference)?;
            Some(super::execute::CheckedCounterfactualPlan::new(
                operation,
                cache.identification.clone(),
                cache.estimand.clone(),
                true,
                inference,
                procedure,
                plan.clone(),
                super::execute::IdentifiedResultContext::from_study(&analysis),
            )?)
        } else {
            None
        };
        let checked_graph_posterior_effect = match (
            analysis.graph_posterior.as_ref(),
            &analysis.query,
            analysis.graph_posterior_identification_cache.as_deref(),
            &analysis.inference,
        ) {
            (
                Some(posterior),
                CausalQuery::AverageEffect(query),
                Some(identification),
                InferenceMode::Frequentist,
            ) if posterior.atom_kind == antecedent_discovery::GraphPosteriorAtomKind::Dag
                && matches!(query.outcome_functional, OutcomeFunctional::Mean)
                && query.target_population == TargetPopulation::AllObserved
                && matches!(
                    analysis.refute,
                    RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full
                )
                && analysis.custom_validators.is_empty()
                && matches!(
                    analysis.estimator_spec.as_ref().map(crate::EstimatorSpec::id),
                    None | Some(crate::strategy_table::EstimatorId::LinearAdjustmentAte)
                )
                && matches!(
                    analysis.estimator,
                    None | Some(crate::strategy_table::EstimatorId::LinearAdjustmentAte)
                )
                && plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER)
                    == crate::strategy_table::IdentifierId::BackdoorAdjustment.as_str()
                && plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR)
                    == crate::strategy_table::EstimatorId::LinearAdjustmentAte.as_str() =>
            {
                Some(CheckedGraphPosteriorEffect::prepare(
                    posterior.clone(),
                    query.clone(),
                    identification.clone(),
                    analysis.estimator_spec.clone().unwrap_or(crate::EstimatorSpec::Default(
                        crate::strategy_table::EstimatorId::LinearAdjustmentAte,
                    )),
                    analysis.bootstrap_replicates,
                    analysis.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
                    analysis.population_registry.clone(),
                    analysis.latency_mode,
                    analysis.refute,
                )?)
            }
            _ => None,
        };
        let checked_bayesian_static_mediation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(data),
                CausalQuery::Mediation(_),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Bayesian(_),
            ) => Some(super::execute::CheckedBayesianStaticMediationOperation::checked(
                &analysis, data, &plan, cache,
            )?),
            _ => None,
        };
        let checked_static_mediation = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
        ) {
            (DataInput::Tabular(data), CausalQuery::Mediation(_), Some(cache), GraphClass::Dag)
                if matches!(analysis.inference, InferenceMode::Frequentist)
                    && matches!(
                        analysis.structure_source,
                        crate::support::StructureSource::Explicit
                            | crate::support::StructureSource::Accepted
                    )
                    && analysis.custom_validators.is_empty() =>
            {
                Some(super::execute::CheckedStaticMediationOperation::checked(
                    &analysis, data, &plan, cache,
                )?)
            }
            _ => None,
        };
        let checked_attribution = match (
            &self.data,
            &self.query,
            analysis.identification_cache.as_deref(),
            analysis.graph.class(),
            &analysis.inference,
        ) {
            (
                DataInput::Tabular(_),
                CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
                Some(cache),
                GraphClass::Dag,
                InferenceMode::Frequentist | InferenceMode::Bayesian(_),
            ) if analysis.structure_source == crate::support::StructureSource::Explicit => {
                Some(super::execute::CheckedAttributionOperation::checked(&analysis, &plan, cache)?)
            }
            _ => None,
        };
        let score_table = analysis.prepare_score_table(ctx)?;
        let execution = if let Some(operation) = checked_linear {
            PreparedExecution::CheckedLinear(operation)
        } else if let Some(operation) = checked_glm {
            PreparedExecution::CheckedGlmAdjustment(operation)
        } else if let Some(operation) = checked_rd {
            PreparedExecution::CheckedRd(operation)
        } else if let Some(operation) = checked_propensity {
            PreparedExecution::CheckedPropensity(operation)
        } else if let Some(operation) = checked_conditional {
            PreparedExecution::CheckedConditional(operation)
        } else if let Some(operation) = checked_graph_posterior_effect {
            PreparedExecution::GraphPosteriorEffect(operation)
        } else if let Some(operation) = checked_aipw {
            PreparedExecution::CheckedAipw(operation)
        } else if let Some(operation) = checked_frontdoor_linear {
            PreparedExecution::FrontDoorLinear(operation)
        } else if let Some(operation) = checked_iv {
            PreparedExecution::Iv(operation)
        } else if let Some(operation) = checked_counterfactual {
            PreparedExecution::Counterfactual(operation)
        } else if let Some(operation) = nested_counterfactual {
            PreparedExecution::NestedCounterfactual(operation)
        } else if let Some(operation) = derivative_response_operation {
            PreparedExecution::DerivativeResponse(operation)
        } else if let Some(operation) = cell_aipw_response_operation {
            PreparedExecution::CellAipwResponse(operation)
        } else if let Some(operation) = static_dag_response_operation {
            PreparedExecution::StaticDagResponse(operation)
        } else if let Some(operation) = checked_bayesian_static_mediation {
            PreparedExecution::BayesianStaticMediation(operation)
        } else if let Some(operation) = checked_static_mediation {
            PreparedExecution::StaticMediation(operation)
        } else if let Some(operation) = checked_attribution {
            PreparedExecution::Attribution(operation)
        } else if let Some(operation) = temporal_dag_response_operation {
            PreparedExecution::TemporalDagResponse(operation)
        } else if let Some(operation) = temporal_mediation_operation {
            PreparedExecution::TemporalMediation(operation)
        } else if let Some(operation) = temporal_dag_effect_operation {
            PreparedExecution::TemporalDagEffect(operation)
        } else if let Some(operation) = temporal_class_effect_operation {
            PreparedExecution::TemporalClassEffect(operation)
        } else if let Some(operation) = distribution_operation {
            PreparedExecution::Distribution(operation)
        } else if let Some(operation) = functional_effect_operation {
            PreparedExecution::FunctionalEffect(operation)
        } else if let Some(operation) = path_specific_effect_operation {
            PreparedExecution::PathSpecificEffect(operation)
        } else if let Some(operation) = admg_response_curve_operation {
            PreparedExecution::AdmgResponseCurve(operation)
        } else if let Some(operation) = bayesian_gcomp_execution {
            PreparedExecution::BayesianGcomp(operation)
        } else if let Some(operation) = bayesian_conditional_execution {
            PreparedExecution::BayesianConditional(operation)
        } else if let Some(operation) = checked_response_curve {
            PreparedExecution::StaticResponseCurve(operation)
        } else {
            PreparedExecution::LegacyStudyDispatch
        };
        Ok(PreparedStudy {
            state: SampledPreparedState {
                analysis,
                program_cache: std::sync::OnceLock::new(),
                plan,
                schema,
                modality,
                time_regularity,
                score_table,
                execution,
            },
        })
    }

    /// Compute the static-path identification once at prepare time.
    ///
    /// Mirrors `execute_static`'s (and, for `bayesian.gcomp`, `execute_bayesian`'s)
    /// stage-1 inputs exactly. Sharp RD and PAG envelopes dispatch to separate
    /// preparation paths. Graph posteriors use their per-atom caches.
    fn prepare_static_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{
            DEFAULT_IDENTIFIER, EstimatorId, IdentifierId, identify_static, identify_static_query,
            identify_static_query_with_rd, select_claim, select_estimand,
        };
        if matches!(self.query, CausalQuery::Counterfactual(_)) {
            let graph = self
                .graph
                .as_dag()
                .ok_or(CausalError::Unsupported { message: "counterfactual requires Dag" })?;
            let identification =
                identify_static_query(IdentifierId::GcmParametric, graph, &self.query)?;
            let estimand = identification.estimands[0].clone();
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        if matches!(
            self.query,
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_)
        ) {
            let (treatment, outcome) = super::execute::gcm_query_vars(&self.query)?;
            let (identification, estimand) = super::execute::parametric_scm_identification(
                self.query.clone(),
                treatment,
                outcome,
            );
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        let identifier = plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if matches!(estimator_id, EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear) {
            return Ok(None);
        }
        match &self.query {
            CausalQuery::AverageEffect(query) => {
                if let Some(background) = &self.tiered {
                    let identification = antecedent_identify::identify_tiered(background, query)?;
                    let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                        CausalError::Compile {
                            message: "tiered identification returned no estimand".into(),
                        }
                    })?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                if self.graph.class() == GraphClass::Admg
                    && self.graph.as_admg().is_some_and(super::execute::admg_has_bidirected)
                {
                    use crate::strategy_table::identify_admg;

                    let admg = self.graph.as_admg().ok_or_else(|| CausalError::Compile {
                        message: "ADMG prepare missing supplied graph".into(),
                    })?;
                    let identification = identify_admg(identifier_id, admg, query)?;
                    let estimand = select_estimand(&identification, estimator_id)?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                let graph = match self.graph.class() {
                    GraphClass::Dag => self.graph.as_dag().cloned(),
                    GraphClass::Admg => plan.static_graph().cloned(),
                    GraphClass::Cpdag => return Ok(None),
                    _ => None,
                };
                let Some(graph) = graph else {
                    return Ok(None);
                };
                // `execute_bayesian` never consults `self.rd` (it calls
                // `identify_static`, which is `identify_static_query_with_rd`
                // with `rd: None`); mirror that exactly so the cached
                // identification matches what an uncached Bayesian run would
                // compute, even if a caller set `.rd_config(..)` alongside
                // `bayesian.gcomp`.
                let rd = if matches!(estimator_id, EstimatorId::BayesianGcomp) {
                    None
                } else {
                    self.rd.map(|c| {
                        antecedent_identify::SharpRdConfig::new(
                            c.running_variable,
                            c.cutoff,
                            c.bandwidth,
                        )
                    })
                };
                let identification = identify_static_query_with_rd(
                    identifier_id,
                    &graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    rd,
                )?;
                let (identification, estimand) = select_claim(identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Response(query) => {
                if let Some(background) = &self.tiered {
                    let schema = match &self.data {
                        DataInput::Tabular(data) => data.schema(),
                        _ => {
                            return Err(CausalError::Unsupported {
                                message: "CoDetermined joint prepare requires tabular data",
                            });
                        }
                    };
                    let identification = match self.graph.as_admg() {
                        Some(admg) => {
                            antecedent_identify::identify_tiered_joint_on(background, admg, query)?
                        }
                        None => {
                            antecedent_identify::identify_tiered_joint(background, schema, query)?
                        }
                    };
                    let estimand = identification.estimands.first().cloned().ok_or(
                        CausalError::Unsupported {
                            message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                        },
                    )?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                if self.graph.class() == GraphClass::Admg
                    && self.graph.as_admg().is_some_and(super::execute::admg_has_bidirected)
                {
                    use crate::strategy_table::identify_admg_query;

                    let admg = self.graph.as_admg().ok_or_else(|| CausalError::Compile {
                        message: "ADMG prepare missing supplied graph".into(),
                    })?;
                    // `execute_admg_response` reuses this cache verbatim as the *first grid
                    // level's* claim for a MeanCurve (it only re-identifies per level from the
                    // second level on, mirroring the graph-posterior ADMG response cache). A
                    // MeanCurve identified whole produces one general.id estimand per grid
                    // level, and `select_estimand` then has no unique estimator match to pick
                    // among them (they all report the same method). Cache the first level's
                    // InterventionResponse claim instead, which is what downstream code
                    // actually consumes and — being a single intervention level — is exactly
                    // what `select_estimand` can disambiguate.
                    let causal_query = match &query.functional {
                        antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
                            let first_level = treatment
                                .grid
                                .values()
                                .map_err(|e| CausalError::Compile { message: e.to_string() })?
                                .into_iter()
                                .next()
                                .ok_or_else(|| CausalError::Compile {
                                    message: "MeanCurve response requires a non-empty evaluation \
                                              grid"
                                        .into(),
                                })?;
                            let mut level_query = query.clone();
                            level_query.functional =
                                antecedent_core::ResponseFunctional::InterventionResponse {
                                    outcome: *outcome,
                                    interventions: Arc::from([Intervention::set(
                                        treatment.variable,
                                        Value::f64(first_level),
                                    )]),
                                };
                            CausalQuery::Response(level_query)
                        }
                        _ => CausalQuery::Response(query.clone()),
                    };
                    let identification = identify_admg_query(identifier_id, admg, &causal_query)?;
                    let estimand = select_estimand(&identification, estimator_id)?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                let graph = match (self.graph.as_dag(), plan.static_graph()) {
                    (Some(dag), _) => Some(dag.clone()),
                    (None, Some(dag)) if self.graph.class() == GraphClass::Admg => {
                        Some(dag.clone())
                    }
                    _ => None,
                };
                let Some(graph) = graph else {
                    return Ok(None);
                };
                let identification = identify_static_query(
                    identifier_id,
                    &graph,
                    &CausalQuery::Response(query.clone()),
                )?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "response identifier returned no estimand".into(),
                    }
                })?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Mediation(query) if self.graph.class() == GraphClass::Dag => {
                let graph = self.graph.as_dag().expect("Dag");
                let identification = identify_static_query(
                    IdentifierId::PathSpecificNatural,
                    graph,
                    &CausalQuery::Mediation(query.clone()),
                )?;
                let estimand =
                    select_estimand(&identification, EstimatorId::StaticMediationLinear)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::ConditionalEffect(query) => {
                // Mirrors execute_conditional / execute_bayesian: identifier is
                // builder-selected or defaults to backdoor; estimator follows inference.
                use crate::strategy_table::DEFAULT_CONDITIONAL_IDENTIFIER;
                let Some(graph) = self.graph.as_dag().cloned() else {
                    return Ok(None);
                };
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(DEFAULT_CONDITIONAL_IDENTIFIER);
                let identifier_id: IdentifierId = identifier.parse()?;
                let identification = identify_static(identifier_id, &graph, &query.inner)?;
                let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
                    EstimatorId::BayesianConditional
                } else {
                    EstimatorId::ConditionalLinearAdjustment
                };
                let (identification, estimand) = select_claim(identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::PathSpecific(query) => {
                // Mirrors `execute_path_specific`'s identify+select-estimand step exactly.
                use crate::strategy_table::{DEFAULT_PATH_ESTIMATOR, DEFAULT_PATH_IDENTIFIER};
                let Some(graph) = self.graph.as_dag().cloned() else {
                    return Ok(None);
                };
                let identifier =
                    plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PATH_IDENTIFIER);
                let estimator =
                    plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_PATH_ESTIMATOR);
                let identifier_id: IdentifierId = identifier.parse()?;
                let estimator_id: EstimatorId = estimator.parse()?;
                let cq = CausalQuery::PathSpecific(query.clone());
                let identification = identify_static_query(identifier_id, &graph, &cq)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Distribution(query) => {
                // Mirrors `execute_distribution`.
                use super::execute::DistributionGraph;
                use crate::strategy_table::{
                    DEFAULT_DISTRIBUTION_ESTIMATOR, DEFAULT_DISTRIBUTION_IDENTIFIER,
                };
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER);
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR);
                let identifier_id: IdentifierId = identifier.parse()?;
                let estimator_id: EstimatorId = estimator.parse()?;
                let graph = if let Some(admg) = self.graph.as_admg() {
                    DistributionGraph::Admg(admg)
                } else if let Some(dag) = self.graph.as_dag() {
                    DistributionGraph::Dag(dag)
                } else {
                    return Ok(None);
                };
                let identification = graph.identify(identifier_id, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            _ => Ok(None),
        }
    }

    /// Compute the generalized-adjustment envelope once for a supplied PAG.
    fn prepare_pag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedPagIdentification>, CausalError> {
        use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER, IdentifierId, identify_pag};

        let Some(query) = self.envelope_witness_ate()? else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::Pag {
            return Ok(None);
        }
        let pag = plan.static_pag().ok_or_else(|| CausalError::Compile {
            message: "PAG prepare missing resolved static PAG".into(),
        })?;
        let identifier =
            plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PAG_IDENTIFIER);
        let identifier_id: IdentifierId = identifier.parse()?;
        let envelope = if let CausalQuery::Response(response) = &self.query {
            crate::strategy_table::identify_pag_response(identifier_id, pag, response)?
        } else {
            identify_pag(identifier_id, pag, &query)?
        };
        let identification =
            super::execute::envelope_to_identification_result_for(&envelope, self.query.clone());
        Ok(Some(CachedPagIdentification { envelope, identification }))
    }

    /// Compute the MEC envelope once for a supplied CPDAG.
    fn prepare_cpdag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedCpdagIdentification>, CausalError> {
        use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER, IdentifierId, identify_cpdag};

        let Some(query) = self.envelope_witness_ate()? else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::Cpdag {
            return Ok(None);
        }
        let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
            message: "CPDAG prepare missing supplied graph".into(),
        })?;
        let identifier =
            plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PAG_IDENTIFIER);
        let identifier_id: IdentifierId = identifier.parse()?;
        let envelope = if let CausalQuery::Response(response) = &self.query {
            crate::strategy_table::identify_cpdag_response(identifier_id, cpdag, response)?
        } else {
            identify_cpdag(identifier_id, cpdag, &query)?
        };
        let identification =
            super::execute::envelope_to_identification_result_for(&envelope, self.query.clone());
        Ok(Some(CachedCpdagIdentification { envelope, identification }))
    }

    fn envelope_witness_ate(&self) -> Result<Option<AverageEffectQuery>, CausalError> {
        match &self.query {
            CausalQuery::AverageEffect(query) => Ok(Some(query.clone())),
            CausalQuery::Response(query)
                if super::execute::class_aware_response_supported(query) =>
            {
                Ok(Some(super::execute::response_witness_ate(query)?))
            }
            CausalQuery::ConditionalEffect(query) => Ok(Some(query.inner.clone())),
            _ => Ok(None),
        }
    }

    /// Identify once per requested horizon at prepare for temporal response or TemporalEffect.
    fn prepare_temporal_identification(
        &self,
    ) -> Result<Option<CachedTemporalIdentification>, CausalError> {
        use crate::strategy_table::{EstimatorId, select_estimand};
        if self.graph.class() != GraphClass::TemporalDag {
            return Ok(None);
        }
        let graph = self.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
            message: "temporal prepare requires TemporalDag".into(),
        })?;
        match &self.query {
            CausalQuery::Response(query) => {
                let Some(temporal) = query.temporal.as_ref() else {
                    return Ok(None);
                };
                let (treatment, outcome) =
                    query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "response query has no treatment/outcome pair".into(),
                    })?;
                let schedule = sequence_identification_schedule(query)?;
                Ok(Some(identify_temporal_response_horizons(
                    graph,
                    treatment,
                    outcome,
                    temporal,
                    &query.target_population,
                    if matches!(self.inference, InferenceMode::Bayesian(_)) {
                        EstimatorId::TemporalResponseBayesian
                    } else {
                        EstimatorId::TemporalResponseGcomp
                    },
                    schedule.as_deref(),
                    single_step_dose(query)?,
                )?))
            }
            CausalQuery::TemporalEffect(query) => {
                let id_res = TemporalBackdoorIdentifier::new()
                    .with_parent_adjustment_fallback()
                    .identify_temporal(graph, query)
                    .map_err(CausalError::from)?;
                let estimand = select_estimand(
                    &id_res.result,
                    if query.is_multi_step_sustained() {
                        EstimatorId::TemporalSequentialGcomp
                    } else {
                        EstimatorId::TemporalLinearAdjustment
                    },
                )?;
                Ok(Some(CachedTemporalIdentification {
                    by_horizon: Arc::from([CachedTemporalHorizonIdentification {
                        horizon: query.horizon_steps,
                        identification: id_res.result,
                        estimand,
                        indexer: id_res.indexer,
                    }]),
                }))
            }
            _ => Ok(None),
        }
    }

    fn prepare_temporal_class_identification(
        &self,
    ) -> Result<Option<CachedTemporalClassIdentification>, CausalError> {
        use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER, IdentifierId};

        if !matches!(self.graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag) {
            return Ok(None);
        }
        let query = match &self.query {
            CausalQuery::TemporalEffect(query) => query.clone(),
            CausalQuery::Response(response) => {
                let temporal = response.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                    message: "temporal class response prepare requires TemporalResponseSpec".into(),
                })?;
                let (treatment, outcome) =
                    response.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "temporal class response has no treatment/outcome pair".into(),
                    })?;
                TemporalEffectQuery {
                    treatment,
                    outcome,
                    policy: temporal.policy.clone(),
                    control: Intervention::set(treatment, Value::f64(0.0)),
                    active: Intervention::set(treatment, Value::f64(1.0)),
                    horizon_steps: temporal.horizons.first().copied().unwrap_or(1),
                    max_history_lag: temporal.max_history_lag,
                    target_population: response.target_population.clone(),
                }
            }
            CausalQuery::Mediation(mediation)
                if self.graph.class() == GraphClass::TemporalCpdag =>
            {
                let mut witness =
                    TemporalEffectQuery::pulse(mediation.treatment, mediation.outcome, 1.0);
                witness.horizon_steps = mediation.horizons.first().copied().unwrap_or(1);
                witness
            }
            _ => return Ok(None),
        };
        let identifier = self.identifier.map_or(DEFAULT_PAG_IDENTIFIER, |id| id.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let mut bundle = self.identify_temporal_class(identifier_id, &query)?;
        let horizons: &[u32] = match &self.query {
            CausalQuery::Response(response) => {
                &response.temporal.as_ref().expect("temporal query").horizons
            }
            CausalQuery::Mediation(mediation) => &mediation.horizons,
            _ => &[],
        };
        for &horizon in horizons {
            let mut qh = query.clone();
            qh.horizon_steps = horizon;
            let identified = if horizon == query.horizon_steps {
                bundle.envelope.clone()
            } else {
                self.identify_temporal_class(identifier_id, &qh)?.envelope
            };
            bundle.by_horizon.push((horizon, identified));
        }
        Ok(Some(bundle))
    }

    /// One `I(h)` per requested mediation horizon (path-product + that horizon's backdoor `Z`).
    fn prepare_temporal_mediation_identification(
        &self,
    ) -> Result<Option<CachedTemporalIdentification>, CausalError> {
        use crate::strategy_table::EstimatorId;
        let CausalQuery::Mediation(query) = &self.query else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::TemporalDag {
            return Ok(None);
        }
        let graph = self.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
            message: "temporal mediation prepare requires TemporalDag".into(),
        })?;
        Ok(Some(identify_temporal_mediation_horizons(
            graph,
            query,
            if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::BayesianTemporalMediation
            } else {
                EstimatorId::TemporalMediation
            },
        )?))
    }

    /// Cross-fitted AIPW scores for retarget / exceedance / joint cells.
    pub(crate) fn prepare_score_table(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, CausalError> {
        let fold_seed = ctx.rng.master_seed();
        let DataInput::Tabular(data) = &self.data else {
            return Ok(None);
        };
        // Unknown is two canonical sets / GraphDependent / no single Z.
        // One score table would collapse the envelope; retarget refuses.
        if self
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::Unknown)
            || !matches!(self.inference, InferenceMode::Frequentist)
        {
            return Ok(None);
        }
        match &self.query {
            CausalQuery::AverageEffect(query) => {
                // Score artifacts are an explicit AIPW execution contract; do not
                // silently fit a second estimator for linear/IV/matching plans.
                if self.estimator != Some(crate::strategy_table::EstimatorId::Aipw)
                    || !matches!(query.target_population, TargetPopulation::AllObserved)
                {
                    return Ok(None);
                }
                let Some(cache) = self.identification_cache.as_ref() else {
                    return Ok(None);
                };
                let method = cache.estimand.method.as_ref();
                if !(method.contains("adjustment")
                    || method.contains("backdoor")
                    || method.starts_with("tiered."))
                {
                    return Ok(None);
                }
                let mut est = match self.estimator_spec.as_ref() {
                    Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => *config.clone(),
                    _ => AipwAte::new(),
                };
                if let Some(overlap) = self.overlap_policy {
                    est.overlap = overlap;
                }
                if matches!(est.overlap, OverlapPolicy::RequireDiagnostics { trim: Some(_), .. })
                    || est.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
                {
                    // Frozen cross-fitted scores are untrimmed iid influence
                    // values. A mean estimate under trimming or a non-iid SE is
                    // still estimated by the configured AIPW; it keeps no score
                    // table, so a retarget refuses with `score_table_unavailable`
                    // instead of reweighting scores of a different construction.
                    // A quantile or exceedance functional is estimated from the
                    // scores and has no such fallback.
                    if query.outcome_functional.is_mean() {
                        return Ok(None);
                    }
                    return Err(crate::unsupported_reason!(
                        "option_not_applicable",
                        "prepared AIPW functional scores require iid inference without \
                         propensity trimming"
                    ));
                }
                let estimand = cache.estimand.clone();
                let mut problem = est.prepare(data, &estimand, query)?;
                problem.fold_seed = fold_seed;
                if let Some(shared) = self.shared_batch_design.as_ref() {
                    shared.apply_to_propensity(&mut problem)?;
                }
                let table = if query.outcome_functional.quantile_level().is_some() {
                    let grid = antecedent_estimate::empirical_threshold_grid(&problem.outcome, 19)?;
                    antecedent_estimate::build_binary_scores(
                        &problem,
                        query.treatment,
                        &grid.into_iter().map(Some).collect::<Vec<_>>(),
                        antecedent_estimate::DEFAULT_AIPW_FOLDS,
                        &est.glm_options,
                        est.backend,
                    )?
                } else {
                    crossfit_binary_scores(
                        &problem,
                        query,
                        antecedent_estimate::DEFAULT_AIPW_FOLDS,
                        &est.glm_options,
                        est.backend,
                    )?
                };
                Ok(Some(table))
            }
            CausalQuery::Response(query) => {
                if self.estimator != Some(crate::strategy_table::EstimatorId::CellAipw) {
                    return Ok(None);
                }
                let Some(cache) = self.identification_cache.as_ref() else {
                    return Ok(None);
                };
                let antecedent_core::ResponseFunctional::InterventionResponse {
                    outcome,
                    interventions,
                } = &query.functional
                else {
                    return Ok(None);
                };
                let treatments = discrete_set_treatments(interventions);
                if treatments.len() < 2 {
                    return Ok(None);
                }
                let est = CellSaturatedAipw::new().with_fold_seed(fold_seed);
                let continuous = self.continuous_cell.as_ref().map(|(variable, grid)| {
                    antecedent_estimate::ContinuousCellSpec { variable: *variable, grid }
                });
                // Folds are deliberately not shared here: `fit_scores_with_assignment`
                // leaves them unset so the cell path draws its own cell-stratified
                // `crossfit_fold_plan` (keyed by `est.fold_seed`), reproducing the solo
                // fold plan for this query's own cells bit-for-bit instead of a private,
                // unstratified batch shuffle. See `SharedBatchDesign` docs.
                let fold_ids: Option<Vec<u32>> = None;
                let design = match self.shared_batch_design.as_ref() {
                    Some(shared) => {
                        let mut ids: Vec<_> = treatments
                            .iter()
                            .copied()
                            .chain(std::iter::once(*outcome))
                            .chain(cache.estimand.adjustment_set.iter().copied())
                            .collect();
                        if let Some((variable, _)) = self.continuous_cell.as_ref() {
                            ids.push(*variable);
                        }
                        let ids = data.complete_case_mask(&ids);
                        let row_index = match ids {
                            Ok(mask) => mask
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &keep)| {
                                    keep.then_some(u32::try_from(i).unwrap_or(u32::MAX))
                                })
                                .collect::<Vec<_>>(),
                            Err(_) => Vec::new(),
                        };
                        if row_index.is_empty() {
                            None
                        } else {
                            shared.design_for(&cache.estimand.adjustment_set, &row_index)?
                        }
                    }
                    None => None,
                };
                let table = est.fit_scores_with_assignment(
                    data,
                    &treatments,
                    *outcome,
                    &cache.estimand.adjustment_set,
                    &query.outcome_functional,
                    continuous,
                    fold_ids.as_deref(),
                    design.as_deref(),
                )?;
                Ok(Some(table))
            }
            _ => Ok(None),
        }
    }
}

fn discrete_set_treatments(interventions: &[Intervention]) -> Vec<antecedent_core::VariableId> {
    interventions
        .iter()
        .filter_map(|iv| match iv {
            Intervention::Set { variable, .. } => Some(*variable),
            _ => None,
        })
        .collect()
}

fn score_table_treatment_col(analysis: &Study, table: &ScoreTable) -> Option<Vec<f64>> {
    let DataInput::Tabular(data) = &analysis.data else {
        return None;
    };
    let values = data.float64_values(table.treatment).ok()?;
    let mut col = Vec::with_capacity(table.n_rows);
    for &idx in table.row_index.iter() {
        col.push(*values.get(idx as usize)?);
    }
    Some(col)
}

/// Whether the query's outcome functional is computed from the frozen score table
/// (exceedance, exceedance grid, quantile) rather than by the estimator itself.
fn query_reads_score_table(query: &CausalQuery) -> bool {
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return false,
    };
    matches!(
        functional,
        OutcomeFunctional::Exceedance(_)
            | OutcomeFunctional::ExceedanceGrid(_)
            | OutcomeFunctional::Quantile(_)
    )
}

fn overlay_prepared_score_functional(
    query: &CausalQuery,
    table: Option<&ScoreTable>,
    result: &mut StudyResult,
) -> Result<(), CausalError> {
    let Some(table) = table else {
        return Ok(());
    };
    if matches!(query, CausalQuery::AverageEffect(_))
        && result.logical_plan.estimator.as_deref() != Some("aipw")
    {
        return Ok(());
    }
    if !query_reads_score_table(query) {
        return Ok(());
    }
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return Ok(()),
    };
    let (estimate, diagnostics) = if let Some(tau) = functional.quantile_level() {
        if let CausalQuery::Response(q) = query {
            super::helpers::attach_joint_quantile_from_table(
                result.estimate.clone(),
                table.clone(),
                q,
                tau,
            )?
        } else {
            super::helpers::attach_quantile_from_table(result.estimate.clone(), table.clone(), tau)?
        }
    } else {
        super::helpers::attach_score_functional_grid(result.estimate.clone(), table.clone())?
    };
    if functional.quantile_level().is_some() {
        if let Some(response) = &mut result.response {
            response.estimate = antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            );
            response.uncertainty = antecedent_core::ResponseUncertainty::Scalar {
                standard_error: estimate.se_analytic,
                lower: estimate.ate
                    - crate::result::reported_se_interval_z() * estimate.se_analytic,
                upper: estimate.ate
                    + crate::result::reported_se_interval_z() * estimate.se_analytic,
                level: 0.95,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            };
        }
    }
    result.estimate = estimate;
    result.rebind_interval(result.posterior.is_some());
    result.diagnostics.extend(diagnostics);
    Ok(())
}

/// Identification schedule of a sequence plan: `(variable, lag, optional level)` per step.
type IdentificationSchedule = Vec<(antecedent_core::VariableId, i32, Option<f64>)>;

fn sequence_identification_schedule(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<IdentificationSchedule>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(plan)) if plan.mechanism_overlays().is_some() => {
            let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                message: "sequence schedule requires TemporalResponseSpec".into(),
            })?;
            Ok(Some(plan.identification_schedule(temporal)))
        }
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

/// Requested hard-set dose of a single-step temporal response, when it names one.
pub(crate) fn single_step_dose(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<f64>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(antecedent_estimate::TemporalInterventionPlan::Single { level, .. })) => Ok(level),
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

/// `dose` is the single-step active level (control stays at 0); `None` keeps the unit contrast.
pub(crate) fn identify_temporal_response_horizons(
    graph: &TemporalDag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    temporal: &TemporalResponseSpec,
    target_population: &TargetPopulation,
    estimator_id: crate::strategy_table::EstimatorId,
    schedule: Option<&[(antecedent_core::VariableId, i32, Option<f64>)]>,
    dose: Option<f64>,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    if temporal.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon".into(),
        });
    }
    let origin =
        temporal.treatment_offset().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let sequential = schedule
        .is_some_and(|nodes| nodes.len() != 1 || (nodes[0].0, nodes[0].1) != (treatment, origin));
    let mut by_horizon = Vec::with_capacity(temporal.horizons.len());
    for &horizon in temporal.horizons.iter() {
        let id_res = if sequential {
            let outcome_at = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
            TemporalBackdoorIdentifier::new()
                .identify_temporal_schedule(
                    graph,
                    outcome,
                    outcome_at,
                    schedule.expect("sequential schedule"),
                    temporal.max_history_lag,
                    target_population.clone(),
                )
                .map_err(CausalError::from)?
        } else {
            let id_query = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(dose.unwrap_or(1.0))),
                horizon_steps: horizon,
                max_history_lag: temporal.max_history_lag,
                target_population: target_population.clone(),
            };
            TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, &id_query)
                .map_err(CausalError::from)?
        };
        let estimand = select_estimand(&id_res.result, estimator_id)?;
        by_horizon.push(CachedTemporalHorizonIdentification {
            horizon,
            identification: id_res.result,
            estimand,
            indexer: id_res.indexer,
        });
    }
    Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
}

pub(crate) fn identify_temporal_mediation_horizons(
    graph: &TemporalDag,
    query: &MediationQuery,
    estimator_id: crate::strategy_table::EstimatorId,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if query.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        });
    }
    let ider = TemporalMediationIdentifier {
        allow_natural_controlled_alias: true,
        ..TemporalMediationIdentifier::new()
    };
    let mut by_horizon = Vec::with_capacity(query.horizons.len());
    for &horizon in query.horizons.iter() {
        let (identification, temporal) =
            ider.identify_with_horizon(graph, query, horizon).map_err(CausalError::from)?;
        let estimand = select_estimand(&identification, estimator_id)?;
        by_horizon.push(CachedTemporalHorizonIdentification {
            horizon,
            identification,
            estimand,
            indexer: temporal.indexer,
        });
    }
    Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
}

fn ensure_prepared_supported(analysis: &Study) -> Result<(), CausalError> {
    if analysis.graph_posterior.is_some() {
        // Refuse here what every estimate click would refuse, before the
        // posterior identification cache is built.
        match (&analysis.data, &analysis.query) {
            (DataInput::Tabular(_), CausalQuery::Response(query)) => {
                super::execute::graph_posterior_response_supported(query)?;
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(query))
                if matches!(analysis.inference, InferenceMode::Frequentist)
                    && query.horizons.len() != 1 =>
            {
                return Err(CausalError::Unsupported {
                    message: "Frequentist DBN-posterior mediation is licensed for one horizon; \
                              multi-horizon grids need their own joint uncertainty contract",
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(query)) => {
                super::execute::dbn_posterior_response_supported(query)?;
            }
            _ => {}
        }
        return match (&analysis.data, &analysis.query) {
            (
                DataInput::Tabular(_),
                CausalQuery::AverageEffect(_)
                | CausalQuery::Response(_)
                | CausalQuery::ConditionalEffect(_),
            )
            | (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::Response(_),
            ) => Ok(()),
            _ => Err(crate::support_reason!(
                "data_modality_not_licensed",
                "graph_posterior on the prepared handle is licensed only for tabular \
                 AverageEffect/Response/ConditionalEffect and series TemporalEffect, \
                 TemporalMediationEffect, or temporal Response"
            )),
        };
    }
    match (&analysis.data, &analysis.query) {
        (DataInput::Tabular(_), CausalQuery::AverageEffect(_)) => {
            if !is_supplied_static_graph(analysis.graph.class()) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy requires a static Dag/Cpdag/Pag/Admg structure \
                (temporal classes are not session-refreshable here)",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Response(q)) if !q.is_temporal() => {
            // An Admg is licensed for a joint response with or without a tiered background
            // (the functional-effect estimator serves it), so no tier condition narrows it.
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag | GraphClass::Admg
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ResponseCurve on a supplied Dag, Cpdag, Pag, \
                              or Admg (or CoDetermined joint cells)",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(q))
            if q.is_temporal() =>
        {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports temporal ResponseCurve on TemporalDag, \
                              TemporalCpdag, or TemporalPag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalEffect on TemporalDag, \
                              TemporalCpdag, or TemporalPag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalMediationEffect on TemporalDag or \
                              TemporalCpdag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Counterfactual(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "counterfactual requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::NestedCounterfactual(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "cross_world_not_identified: nested effect requires a supplied DAG",
                });
            }
        }
        (
            DataInput::Tabular(_),
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
        ) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "AnomalyAttribution and ChangeAttribution require a supplied Dag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Transport(_)) => {
            if analysis.graph.class() != GraphClass::Admg {
                return Err(CausalError::Unsupported { message: "TransportQuery requires Admg" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Interference(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "InterferenceQuery requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Mediation(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "static mediation requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ConditionalEffect on a supplied Dag, Cpdag, or Pag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::PathSpecific(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports PathSpecific only on a supplied Dag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Distribution(_)) => {
            if !matches!(analysis.graph.class(), GraphClass::Dag | GraphClass::Admg) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports Distribution on a supplied Dag or Admg",
                });
            }
        }
        (DataInput::MultiEnv(_), CausalQuery::TemporalEffect(_)) => {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports multi-environment TemporalEffect on a TemporalDag"
                ));
            }
        }
        (DataInput::Panel(panel), query) => {
            if !matches!(
                (query, analysis.graph.class()),
                (
                    CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                )
            ) {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports panel Pulse/Sustained and temporal response on \
                     TemporalDag, TemporalCpdag, or TemporalPag"
                ));
            }
            super::builder::refuse_unlicensed_panel_route(
                query,
                analysis.graph.class(),
                &analysis.inference,
                panel,
                analysis.split.as_ref(),
            )?;
        }
        _ => {
            return Err(crate::unsupported_reason!(
                "data_modality_not_licensed",
                "PreparedStudy supports AverageEffect, ResponseCurve, ConditionalEffect, \
                 PathSpecific, Distribution, temporal ResponseCurve, TemporalEffect (Pulse / \
                 single-step Sustained), TemporalMediationEffect, panel Pulse/Sustained, \
                 Counterfactual, AnomalyAttribution, ChangeAttribution, TransportQuery, or \
                 InterferenceQuery"
            ));
        }
    }
    Ok(())
}

fn is_supplied_static_graph(class: GraphClass) -> bool {
    matches!(class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag | GraphClass::Admg)
}

impl PreparedStudy {
    /// Control intervention level frozen on a counterfactual ITE query.
    #[must_use]
    pub fn counterfactual_control_level(&self) -> f64 {
        match self.query() {
            CausalQuery::Counterfactual(q) => match &q.control {
                Intervention::Set { value, .. } => value.as_f64().unwrap_or(0.0),
                _ => 0.0,
            },
            CausalQuery::NestedCounterfactual(q) => q.control_value(),
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{is_supplied_static_graph, query_reads_score_table};
    use crate::accepted::GraphClass;
    use antecedent_core::{AverageEffectQuery, CausalQuery, OutcomeFunctional, VariableId};

    /// A mean click keeps the estimator's own value, so refitting the cross-fit score table
    /// for it would be discarded work; only exceedance, grid and quantile clicks read it.
    #[test]
    fn only_non_mean_functionals_read_the_frozen_score_table() {
        let base = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        assert!(!query_reads_score_table(&CausalQuery::AverageEffect(base.clone())));
        for functional in [
            OutcomeFunctional::exceedance(0.5),
            OutcomeFunctional::exceedance_grid(vec![0.0, 1.0]),
            OutcomeFunctional::quantile(0.5),
        ] {
            let query = base.clone().with_outcome_functional(functional);
            assert!(query_reads_score_table(&CausalQuery::AverageEffect(query)));
        }
    }

    #[test]
    fn supplied_static_graphs_only() {
        assert!(is_supplied_static_graph(GraphClass::Dag));
        assert!(is_supplied_static_graph(GraphClass::Cpdag));
        assert!(is_supplied_static_graph(GraphClass::Pag));
        assert!(is_supplied_static_graph(GraphClass::Admg));
        assert!(!is_supplied_static_graph(GraphClass::TemporalDag));
        assert!(!is_supplied_static_graph(GraphClass::TemporalCpdag));
        assert!(!is_supplied_static_graph(GraphClass::TemporalPag));
    }
}

#[cfg(test)]
mod checked_static_operation_tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::AipwAte;
    use antecedent_estimate::GlmAdjustmentAte;
    use antecedent_graph::Dag;
    use antecedent_stats::GlmFamily;

    use super::PreparedExecution;
    use crate::Study;
    use crate::analysis::builder::RefuteSuite;
    use crate::estimator_spec::EstimatorSpec;
    use crate::strategy_table::EstimatorId;

    fn data() -> TabularData {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let n = 512;
        let mut t = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            let ti = (i % 2) as f64;
            let zi = ((i * 37 % 101) as f64 - 50.0) / 50.0;
            t.push(ti);
            z.push(zi);
            y.push(2.0 * ti + zi + ((i * 13 % 47) as f64 - 23.0) / 100.0);
        }
        let columns = [("t", t), ("y", y), ("z", z)]
            .into_iter()
            .map(|(name, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        schema.id_of(name).unwrap(),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
    }

    #[test]
    fn checked_aipw_click_uses_retained_fitter_after_study_config_changes() {
        let data = data();
        let graph =
            Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
        let query = AverageEffectQuery::with_levels(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
            0.0,
            1.0,
        );
        let context = ExecutionContext::for_tests(23);
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(query)
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let mut prepared = study.prepare(&context).unwrap();
        drop(study);
        let PreparedExecution::CheckedAipw(operation) = &prepared.execution else {
            panic!("expected retained checked AIPW operation")
        };
        assert_eq!(operation.fitter.bootstrap_replicates, 0);
        assert_eq!(operation.preparation.lowering().bootstrap_replicates, 0);
        let first = prepared.estimate(&data, &context).unwrap();
        assert!((first.effect() - 2.0).abs() < 0.2);

        // A copied Study is execution metadata, not the source of the fitter.
        // If a click reconstructed AIPW from it, the changed replicate count
        // would conflict with the checked receipt and reject the click.
        let mut changed_fitter = AipwAte::new();
        changed_fitter.bootstrap_replicates = 7;
        prepared.analysis.estimator_spec = Some(EstimatorSpec::Aipw(Box::new(changed_fitter)));
        let second = prepared.estimate(&data, &context).unwrap();
        assert!((second.effect() - first.effect()).abs() < 1e-10);
        assert!(second.estimate.se_bootstrap.is_none());

        prepared.analysis.estimator_spec = Some(EstimatorSpec::LinearAdjustmentAte(Box::new(
            antecedent_estimate::LinearAdjustmentAte::new(),
        )));
        let third = prepared.estimate(&data, &context).unwrap();
        assert!((third.effect() - first.effect()).abs() < 1e-10);
    }

    #[test]
    fn checked_linear_click_uses_retained_default_uncertainty_after_study_changes() {
        let data = data();
        let graph =
            Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
        let query = AverageEffectQuery::with_levels(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
            0.0,
            1.0,
        );
        let context = ExecutionContext::for_tests(24);
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(query)
            .estimator(EstimatorId::LinearAdjustmentAte)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let mut prepared = study.prepare(&context).unwrap();
        drop(study);
        let PreparedExecution::CheckedLinear(operation) = &prepared.execution else {
            panic!("expected retained checked linear operation")
        };
        assert!(operation.default_id);
        assert_eq!(operation.fitter.bootstrap_replicates, 0);
        let first = prepared.estimate(&data, &context).unwrap();
        assert!((first.effect() - 2.0).abs() < 0.2);
        assert!(first.estimate.se_bootstrap.is_none());

        prepared.analysis.bootstrap_replicates = 9;
        prepared.analysis.estimator_spec = Some(EstimatorSpec::Aipw(Box::new(AipwAte::new())));
        let second = prepared.estimate(&data, &context).unwrap();
        assert!((second.effect() - first.effect()).abs() < 1e-10);
        assert!(second.estimate.se_bootstrap.is_none());

        let graph =
            Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
        let with_bootstrap = Study::tabular(data.clone())
            .graph(graph)
            .query(AverageEffectQuery::with_levels(
                data.schema().id_of("t").unwrap(),
                data.schema().id_of("y").unwrap(),
                0.0,
                1.0,
            ))
            .estimator(EstimatorId::LinearAdjustmentAte)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(8)
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        let with_bootstrap_result = with_bootstrap.estimate(&data, &context).unwrap();
        assert!(with_bootstrap_result.estimate.se_bootstrap.is_some());
    }

    #[test]
    fn checked_glm_adjustment_survives_builder_drop_and_refresh() {
        let data = data();
        let graph =
            Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
        let query = AverageEffectQuery::with_levels(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
            0.0,
            1.0,
        );
        let mut fitter = GlmAdjustmentAte::new().with_family(GlmFamily::GaussianIdentity);
        fitter.bootstrap_replicates = 0;
        let context = ExecutionContext::for_tests(28);
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(query)
            .estimator(fitter)
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let mut prepared = study.prepare(&context).unwrap();
        drop(study);
        let PreparedExecution::CheckedGlmAdjustment(operation) = &prepared.execution else {
            panic!("expected checked GLM operation")
        };
        assert_eq!(operation.estimator, EstimatorId::GlmAdjustment);
        let first = prepared.estimate(&data, &context).unwrap();
        assert!((first.effect() - 2.0).abs() < 0.2);
        let refreshed = prepared.refresh(data.clone(), &context).unwrap();
        assert!((refreshed.effect() - first.effect()).abs() < 1e-10);
    }

    #[test]
    fn checked_rd_retains_cutoff_target_and_bandwidth_on_refresh() {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("r", RoleHint::Context),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let n = 200usize;
        let r: Vec<f64> = (0..n).map(|i| (i as f64 - 99.5) / 100.0).collect();
        let t: Vec<f64> = r.iter().map(|&v| f64::from(v >= 0.0)).collect();
        let y: Vec<f64> = r.iter().zip(&t).map(|(&v, &ti)| 1.0 + 0.4 * v + 2.0 * ti).collect();
        let columns = [("t", t), ("y", y), ("r", r)]
            .into_iter()
            .map(|(name, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        schema.id_of(name).unwrap(),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
        let graph =
            Dag::from_named_edges(data.schema(), &[("r", "t"), ("r", "y"), ("t", "y")]).unwrap();
        let query = AverageEffectQuery::binary_ate(
            data.schema().id_of("t").unwrap(),
            data.schema().id_of("y").unwrap(),
        );
        let context = ExecutionContext::for_tests(29);
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(query)
            .identifier(crate::strategy_table::IdentifierId::RdSharp)
            .estimator(EstimatorId::RdSharp)
            .rd_config(data.schema().id_of("r").unwrap(), 0.0, 0.8)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let mut prepared = study.prepare(&context).unwrap();
        drop(study);
        let PreparedExecution::CheckedRd(operation) = &prepared.execution else {
            panic!("expected checked RD operation")
        };
        assert_eq!(operation.preparation.lowering().bandwidth, 0.8);
        let first = prepared.estimate(&data, &context).unwrap();
        assert!((first.effect() - 2.0).abs() < 1e-10);
        assert_eq!(
            first.identification.average_effect().unwrap().target_population,
            antecedent_core::TargetPopulation::local_at_cutoff(
                data.schema().id_of("r").unwrap(),
                0.0
            )
        );
        let refreshed = prepared.refresh(data, &context).unwrap();
        assert!((refreshed.effect() - first.effect()).abs() < 1e-10);
    }
}

#[cfg(test)]
mod checked_response_curve_tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, MeasurementSpec,
        ResponseFunctional, ResponseQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_graph::Dag;

    use super::Study;

    fn data(shift: f64) -> TabularData {
        let mut schema = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("x", RoleHint::Context),
            ("a", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            schema
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = schema.build().unwrap();
        let n = 120;
        let x: Vec<_> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();
        let a: Vec<_> = (0..n).map(|i| 0.6 * x[i] + (i as f64 * 0.31).cos()).collect();
        let y: Vec<_> = (0..n)
            .map(|i| shift + 1.7 * a[i] + 0.8 * x[i] + (i as f64 * 0.23).sin() * 0.1)
            .collect();
        let cols = [x, a, y]
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
    }

    fn study(data: TabularData, grid: &[f64]) -> Study {
        let graph =
            Dag::from_named_edges(data.schema(), &[("x", "a"), ("x", "y"), ("a", "y")]).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(2),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(1),
                GridSpec::Values(Arc::from(grid)),
            ),
        });
        Study::tabular(data)
            .graph(graph)
            .query(query)
            .bootstrap_replicates(0)
            .refute(super::super::builder::RefuteSuite::None)
            .build()
            .unwrap()
    }

    #[test]
    fn prepared_curve_refresh_matches_independent_numeric_estimate() {
        let grid = [-0.5, 0.0, 0.5];
        let context = ExecutionContext::for_tests(17);
        let mut prepared = study(data(0.0), &grid).prepare(&context).unwrap();
        assert!(matches!(prepared.execution, super::PreparedExecution::StaticDagResponse(_)));
        let refreshed = data(0.4);
        let actual = prepared.refresh(refreshed.clone(), &context).unwrap().response.unwrap();
        let expected = study(refreshed, &grid).run(&context).unwrap().response.unwrap();
        assert_eq!(actual, expected);
        let antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { mean, .. },
        ) = actual.estimate
        else {
            panic!("expected a point-identified curve")
        };
        for (&dose, &estimate) in grid.iter().zip(mean.iter()) {
            let truth = 0.4 + 1.7 * dose;
            assert!(
                (estimate - truth).abs() < 0.15,
                "dose={dose}: estimate={estimate}, truth={truth}"
            );
        }
    }

    #[test]
    fn prepared_curve_refuses_a_tampered_grid() {
        let context = ExecutionContext::for_tests(17);
        let mut prepared = study(data(0.0), &[-0.5, 0.0, 0.5]).prepare(&context).unwrap();
        let changed = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(2),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(1),
                GridSpec::Values(Arc::from([-0.5, 0.25, 0.5])),
            ),
        });
        prepared.study_mut().query = changed.into();
        let err = prepared.estimate(&data(0.2), &context).unwrap_err();
        assert!(err.to_string().contains(
            "prepared static response operation no longer matches the study query or graph"
        ));
    }
}

#[cfg(test)]
mod checked_iv_prepared_tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{AnalyticSeKind, TwoStageLeastSquares, WaldIv};
    use antecedent_graph::Dag;

    use crate::estimator_spec::EstimatorSpec;
    use crate::{Study, analysis::builder::RefuteSuite, strategy_table::IdentifierId};

    use super::{CheckedIvOperation, PreparedExecution};

    fn data(shift: f64) -> TabularData {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let ids =
            [schema.id_of("t").unwrap(), schema.id_of("y").unwrap(), schema.id_of("z").unwrap()];
        let n = 1_600;
        let mut t = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = ((i * 37 % 101) as f64 - 50.0) / 30.0;
            let ti = 0.6 * zi + u;
            z.push(zi);
            t.push(ti);
            y.push(shift + 2.0 * ti + u);
        }
        let columns = [t, y, z]
            .into_iter()
            .zip(ids)
            .map(|(values, id)| {
                OwnedColumn::Float64(
                    Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n))
                        .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
    }

    fn study(
        data: TabularData,
        estimator: impl Into<crate::estimator_spec::EstimatorSpec>,
    ) -> Study {
        let graph = Dag::from_named_edges(data.schema(), &[("z", "t"), ("t", "y")]).unwrap();
        Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                0.0,
                1.0,
            ))
            .identifier(IdentifierId::Auto)
            .estimator(estimator)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
    }

    #[test]
    fn checked_iv_survives_builder_drop_and_refresh_with_selected_fitter() {
        let context = ExecutionContext::for_tests(74);
        for (fitter, is_wald) in [
            (study(data(0.0), WaldIv::new().with_se_kind(AnalyticSeKind::Hc1)), true),
            (
                study(data(0.0), TwoStageLeastSquares::new().with_se_kind(AnalyticSeKind::Hc1)),
                false,
            ),
        ] {
            let mut prepared = fitter.prepare(&context).unwrap();
            match &prepared.execution {
                PreparedExecution::Iv(CheckedIvOperation::Wald { fitter, preparation })
                    if is_wald =>
                {
                    assert_eq!(fitter.se_kind, AnalyticSeKind::Hc1);
                    assert_eq!(
                        preparation.lowering().procedure,
                        antecedent_estimate::CheckedIvProcedure::Wald
                    );
                }
                PreparedExecution::Iv(CheckedIvOperation::TwoSls { fitter, preparation })
                    if !is_wald =>
                {
                    assert_eq!(fitter.se_kind, AnalyticSeKind::Hc1);
                    assert_eq!(
                        preparation.lowering().procedure,
                        antecedent_estimate::CheckedIvProcedure::TwoStageLeastSquares
                    );
                }
                other => panic!("expected retained checked IV operation, got {other:?}"),
            }
            prepared.analysis.estimator_spec = Some(EstimatorSpec::LinearAdjustmentAte(Box::new(
                antecedent_estimate::LinearAdjustmentAte::new(),
            )));
            let result = prepared.estimate(&data(0.0), &context).unwrap();
            assert!((result.estimate.ate - 2.0).abs() < 0.15, "estimate={}", result.estimate.ate);
            assert_eq!(result.estimate.se_kind, Some(AnalyticSeKind::Hc1));
            prepared.study_mut().estimator_spec = Some(EstimatorSpec::LinearAdjustmentAte(
                Box::new(antecedent_estimate::LinearAdjustmentAte::new()),
            ));
            let refreshed = prepared.refresh(data(0.4), &context).unwrap();
            assert!(
                (refreshed.estimate.ate - 2.0).abs() < 0.15,
                "estimate={}",
                refreshed.estimate.ate
            );
            assert_eq!(refreshed.estimate.se_kind, Some(AnalyticSeKind::Hc1));
            assert!(matches!(prepared.execution, PreparedExecution::Iv(_)));
            prepared.study_mut().query =
                crate::CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                    0.0,
                    2.0,
                ));
            assert!(
                prepared.estimate(&data(0.0), &context).is_err(),
                "changed contrast must be refused"
            );
        }
    }
}

#[cfg(test)]
mod checked_frontdoor_prepared_tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_graph::Dag;

    use super::{PreparedExecution, Study};
    use crate::analysis::builder::RefuteSuite;
    use crate::estimator_spec::EstimatorSpec;
    use crate::strategy_table::IdentifierId;

    fn data(y_shift: f64) -> TabularData {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("m", RoleHint::Context),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let n = 2_000;
        let mut treatment = Vec::with_capacity(n);
        let mut mediator = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i % 2) as f64;
            let u = ((i * 37 % 101) as f64 - 50.0) / 100.0;
            let m = 0.7 * t + u;
            treatment.push(t);
            mediator.push(m);
            outcome.push(y_shift + 2.0 * m + ((i * 13 % 47) as f64 - 23.0) / 100.0);
        }
        let columns = [treatment, mediator, outcome]
            .into_iter()
            .enumerate()
            .map(|(index, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(index as u32),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
    }

    #[test]
    fn frontdoor_prepared_click_uses_retained_fit_after_study_config_changes() {
        let base_data = data(0.0);
        let refreshed_data = data(0.4);
        let graph = Dag::from_named_edges(base_data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            0.0,
            1.0,
        );
        let context = ExecutionContext::for_tests(810);
        let study = Study::tabular(base_data.clone())
            .graph(graph)
            .query(query)
            .identifier(IdentifierId::Frontdoor)
            .estimator(antecedent_estimate::FrontDoorTwoStage::new().with_bootstrap_replicates(0))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let mut prepared = study.prepare(&context).unwrap();
        drop(study);
        let PreparedExecution::FrontDoorLinear(operation) = &prepared.execution else {
            panic!("expected retained checked front-door operation")
        };
        assert_eq!(operation.preparation.lowering().treatment, VariableId::from_raw(0));
        assert_eq!(operation.fitter.bootstrap_replicates, 0);
        prepared.analysis.estimator_spec = Some(EstimatorSpec::LinearAdjustmentAte(Box::new(
            antecedent_estimate::LinearAdjustmentAte::new(),
        )));
        let estimate = prepared.estimate(&base_data, &context).unwrap();
        assert!((estimate.effect() - 1.4).abs() < 0.1, "estimate={}", estimate.effect());
        let refreshed = prepared.refresh(refreshed_data, &context).unwrap();
        assert!((refreshed.effect() - estimate.effect()).abs() < 0.05);
        assert!(refreshed.estimate.assumptions.entries.iter().any(|entry| {
            matches!(&entry.assumption, antecedent_core::Assumption::ParametricRestriction(p) if p.id.as_ref() == "frontdoor.linear_path_product")
        }));
    }
}

#[cfg(test)]
mod refresh_tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};

    use super::super::builder::{DataInput, RefuteSuite};
    use crate::analysis::execute::Study;

    #[allow(clippy::cast_precision_loss)]
    fn xy_series(n: usize) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [("x", RoleHint::TreatmentCandidate), ("y", RoleHint::OutcomeCandidate)]
        {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.07).sin();
            y[t] = 0.8 * x[t - 1];
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    fn retained_rows(prepared: &super::PreparedStudy) -> usize {
        match &prepared.analysis.data {
            DataInput::Temporal(data) => data.row_count(),
            _ => panic!("series handle must retain series data"),
        }
    }

    #[test]
    fn failed_series_refresh_leaves_handle_unchanged() {
        let mut graph = TemporalDag::empty();
        let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1));
        let ctx = ExecutionContext::for_tests(2);
        let mut prepared = Study::series(xy_series(160))
            .graph(graph)
            .temporal_query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        assert_eq!(retained_rows(&prepared), 160);
        let original = xy_series(160);
        let before = prepared.contract().unwrap().identities;

        // Same schema and regularity, so the compatibility gate passes and the
        // failure comes from estimation itself.
        assert!(prepared.refresh_series(xy_series(2), &ctx).is_err());
        assert_eq!(retained_rows(&prepared), 160, "failed refresh must not replace data");
        assert_eq!(before, prepared.contract().unwrap().identities);
        let recovered = prepared.estimate_series(&original, &ctx).unwrap();
        assert!(recovered.effect().is_finite(), "old-data estimate must survive a failed refresh");
        assert_eq!(before, prepared.contract().unwrap().identities);

        prepared.refresh_series(xy_series(140), &ctx).unwrap();
        assert_eq!(retained_rows(&prepared), 140);
        let after = prepared.contract().unwrap().identities;
        assert_eq!(before.program, after.program);
        assert_ne!(before.data_snapshot, after.data_snapshot);
    }
}

#[cfg(test)]
#[path = "dbn_mediation_cache_tests.rs"]
mod dbn_mediation_cache_tests;

#[cfg(test)]
mod prepared_frontdoor_tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{AnalyticSeKind, FrontDoorTwoStage};
    use antecedent_graph::Dag;

    use crate::Study;
    use crate::analysis::builder::RefuteSuite;
    use crate::analysis::prepared::PreparedExecution;
    use crate::estimator_spec::EstimatorSpec;
    use crate::strategy_table::{EstimatorId, IdentifierId};

    fn data(outcome_shift: f64) -> TabularData {
        let mut builder = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("m", RoleHint::Context),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(hint),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let treatment_id = schema.id_of("t").unwrap();
        let outcome_id = schema.id_of("y").unwrap();
        let mediator_id = schema.id_of("m").unwrap();
        let n = 1_600;
        let mut treatment = Vec::with_capacity(n);
        let mut mediator = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        for i in 0..(n / 2) {
            let mediator_noise = ((i * 17 % 101) as f64 - 50.0) / 65.0;
            let outcome_noise = ((i * 31 % 97) as f64 - 48.0) / 42.0;
            for t in [0.0, 1.0] {
                let m = 0.8 * t + mediator_noise;
                treatment.push(t);
                mediator.push(m);
                outcome.push(outcome_shift + 2.0 * m + outcome_noise);
            }
        }
        let columns = [(treatment_id, treatment), (outcome_id, outcome), (mediator_id, mediator)];
        let columns = columns
            .into_iter()
            .map(|(id, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n))
                        .unwrap(),
                )
            })
            .collect();
        TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap())
    }

    fn study(data: TabularData, fitter: FrontDoorTwoStage) -> Study {
        let graph = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
        let treatment = data.schema().id_of("t").unwrap();
        let outcome = data.schema().id_of("y").unwrap();
        Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::with_levels(treatment, outcome, 0.0, 1.0))
            .identifier(IdentifierId::Frontdoor)
            .estimator(fitter)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
    }

    #[test]
    fn prepared_frontdoor_keeps_checked_procedure_through_estimate_and_refresh() {
        let context = ExecutionContext::for_tests(91);
        let fitter =
            FrontDoorTwoStage::new().with_bootstrap_replicates(0).with_se_kind(AnalyticSeKind::Hc1);
        // `build` and `prepare` consume their builders; only the retained handle is used below.
        let mut prepared = study(data(0.0), fitter.clone()).prepare(&context).unwrap();
        let PreparedExecution::FrontDoorLinear(operation) = &prepared.execution else {
            panic!("linear front-door must prepare a checked execution variant")
        };
        assert_eq!(operation.fitter.se_kind, AnalyticSeKind::Hc1);
        assert_eq!(
            operation.preparation.lowering().procedure,
            antecedent_estimate::frontdoor::CheckedFrontDoorProcedure::LinearPathProduct
        );
        assert_eq!(
            operation.preparation.program().mapping().source,
            operation.preparation.target().functional
        );

        let first = prepared.estimate(&data(0.0), &context).unwrap();
        assert!((first.estimate.ate - 1.6).abs() < 0.12, "estimate={}", first.estimate.ate);
        assert_eq!(first.estimate.se_kind, Some(AnalyticSeKind::Hc1));
        let first_bytes =
            prepared.encode_contracted_result(&first, "checked-frontdoor", &context).unwrap();
        let first_consumed = antecedent_io::consume_analysis_result(&first_bytes).unwrap();
        assert!(first_consumed.acceptance.accepts_as_verified_program());
        assert_eq!(first_consumed.body.estimate, Some(first.effect()));

        prepared.study_mut().estimator_spec = Some(EstimatorSpec::LinearAdjustmentAte(Box::new(
            antecedent_estimate::LinearAdjustmentAte::new(),
        )));

        let refreshed = prepared.refresh(data(0.35), &context).unwrap();
        assert!((refreshed.estimate.ate - 1.6).abs() < 0.12, "estimate={}", refreshed.estimate.ate);
        assert_eq!(refreshed.estimate.se_kind, Some(AnalyticSeKind::Hc1));
        assert!(matches!(prepared.execution, PreparedExecution::FrontDoorLinear(_)));
        let refreshed_bytes = prepared
            .encode_contracted_result(&refreshed, "checked-frontdoor-refresh", &context)
            .unwrap();
        let refreshed_consumed = antecedent_io::consume_analysis_result(&refreshed_bytes).unwrap();
        assert!(refreshed_consumed.acceptance.accepts_as_verified_program());
        assert_eq!(refreshed_consumed.body.estimate, Some(refreshed.effect()));

        let treatment = prepared.schema.id_of("t").unwrap();
        let outcome = prepared.schema.id_of("y").unwrap();
        prepared.study_mut().query = CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
            treatment, outcome, 0.0, 2.0,
        ));
        let error = prepared.estimate(&data(0.35), &context).unwrap_err();
        assert!(error.to_string().contains("retained checked front-door operation"), "{error}");
    }

    #[test]
    fn default_frontdoor_id_exports_a_verified_checked_result() {
        let data = data(0.0);
        let graph = Dag::from_named_edges(data.schema(), &[("t", "m"), ("m", "y")]).unwrap();
        let treatment = data.schema().id_of("t").unwrap();
        let outcome = data.schema().id_of("y").unwrap();
        let context = ExecutionContext::for_tests(92);
        let prepared = Study::tabular(data.clone())
            .graph(graph)
            .query(AverageEffectQuery::with_levels(treatment, outcome, 0.0, 1.0))
            .identifier(IdentifierId::Frontdoor)
            .estimator(EstimatorId::FrontDoorTwoStage)
            .bootstrap_replicates(0)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&context)
            .unwrap();
        assert!(matches!(prepared.execution, PreparedExecution::FrontDoorLinear(_)));
        let result = prepared.estimate(&data, &context).unwrap();
        let bytes =
            prepared.encode_contracted_result(&result, "default-frontdoor", &context).unwrap();
        assert!(
            antecedent_io::consume_analysis_result(&bytes)
                .unwrap()
                .acceptance
                .accepts_as_verified_program()
        );
    }
}
