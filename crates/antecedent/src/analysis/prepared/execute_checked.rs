//! Stamping and checked-operation execution for prepared estimate clicks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CheckedAipwOperation, CheckedDistributionOperation, CheckedFrontDoorOperation,
    CheckedGlmAdjustmentOperation, CheckedIvOperation, CheckedLinearOperation, CheckedRdOperation,
    CheckedStaticResultMetadata, PreparedStudy, static_result_metadata,
};
use crate::accepted::GraphClass;
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::checked_propensity::{
    CheckedPropensityOperation, CheckedPropensityProcedure, CheckedPropensityUncertainty,
};
use crate::analysis::helpers::{
    AssembleArgs, RefutationRun, assemble_result, overlap_diagnostic, provenance_pair,
    stamp_result_context,
};
use crate::analysis::missing_data::MissingRoute;
use crate::analysis::{CheckedConditionalOperation, ConditionalProcedure};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::RouteDefaults;
use crate::result::StudyResult;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, OutcomeFunctional,
    TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_estimate::{EffectEstimate, EstimationWorkspace};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;
use std::sync::Arc;
use std::time::Instant;

/// What a checked static operation fitted: the estimator that ran, its estimate and timing.
struct CheckedStaticEffectRun {
    estimator_id: crate::strategy_table::EstimatorId,
    estimate: EffectEstimate,
    bootstrap_requested: Option<u32>,
    started: Instant,
}

impl PreparedStudy {
    /// Bind the contract `result` executed under `refute` on `data`, unless the
    /// caller `defer`s it (a refresh binds once, to the state it retains).
    fn bind_executed_contract(
        &self,
        data: &DataInput,
        refute: RefuteSuite,
        defer: bool,
        result: &mut StudyResult,
    ) -> Result<(), CausalError> {
        result.lowered_identity = Some(self.plan.require_lowered(None)?.identity());
        if let (DataInput::Tabular(tabular), Some(MissingRoute::Static)) =
            (data, MissingRoute::of(&self.study().query, &self.study().data))
        {
            self.study().report_static_missing_data(tabular, result)?;
        }
        if !defer {
            result.executed_contract = Some(self.executed_contract(data, refute, None)?);
        }
        Ok(())
    }

    /// Record `inference`'s likelihood disclosure on `result`, then stamp it
    /// as executed on tabular `data`.
    pub(super) fn stamp_with_likelihood(
        &self,
        data: &TabularData,
        inference: &InferenceMode,
        mut result: StudyResult,
        defer: bool,
    ) -> Result<StudyResult, CausalError> {
        let input = DataInput::Tabular(data.clone());
        crate::analysis::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            inference,
            &input,
        );
        self.stamp_deferrable(&input, result, defer)
    }

    /// Stamp the contract `result` was executed under on `data`.
    pub(super) fn stamp(
        &self,
        data: &DataInput,
        result: StudyResult,
    ) -> Result<StudyResult, CausalError> {
        self.stamp_deferrable(data, result, false)
    }

    /// [`Self::stamp`], leaving the contract to the caller when `defer`.
    pub(super) fn stamp_deferrable(
        &self,
        data: &DataInput,
        mut result: StudyResult,
        defer: bool,
    ) -> Result<StudyResult, CausalError> {
        crate::analysis::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &self.analysis.inference,
            data,
        );
        self.bind_executed_contract(data, self.analysis.refute, defer, &mut result)?;
        Ok(result)
    }

    pub(super) fn stamp_distribution(
        &self,
        data: &DataInput,
        operation: &CheckedDistributionOperation,
        mut result: StudyResult,
        defer: bool,
    ) -> Result<StudyResult, CausalError> {
        crate::analysis::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &operation.inference,
            data,
        );
        self.bind_executed_contract(data, operation.refute, defer, &mut result)?;
        Ok(result)
    }

    pub(super) fn stamp_linear(
        &self,
        data: &DataInput,
        operation: &CheckedLinearOperation,
        mut result: StudyResult,
        defer: bool,
    ) -> Result<StudyResult, CausalError> {
        crate::analysis::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &operation.inference,
            data,
        );
        self.bind_executed_contract(data, operation.refute, defer, &mut result)?;
        Ok(result)
    }

    pub(super) fn execute_checked_frontdoor(
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
            || !self
                .plan
                .binding()
                .estimator
                .is(crate::strategy_table::EstimatorId::FrontDoorTwoStage)
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
            CheckedStaticEffectRun {
                estimator_id: crate::strategy_table::EstimatorId::FrontDoorTwoStage,
                estimate,
                bootstrap_requested: Some(operation.fitter.bootstrap_replicates),
                started,
            },
            ctx,
        )
    }

    pub(super) fn execute_checked_iv(
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
            || !self.plan.binding().estimator.is(estimator)
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
            CheckedStaticEffectRun {
                estimator_id: estimator,
                estimate,
                bootstrap_requested: None,
                started,
            },
            ctx,
        )
    }

    pub(super) fn execute_checked_glm(
        &self,
        data: &TabularData,
        operation: &CheckedGlmAdjustmentOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if operation.query
            != *operation.cache.identification.average_effect().unwrap_or(&operation.query)
            || operation.cache.estimand.functional
                != operation
                    .cache
                    .identification
                    .estimands
                    .first()
                    .map_or(operation.cache.estimand.functional, |e| e.functional)
            || operation.estimator != crate::strategy_table::EstimatorId::GlmAdjustment
            || !operation
                .physical
                .binding()
                .names(operation.identifier, crate::strategy_table::EstimatorId::GlmAdjustment)
            || operation.preparation.method != operation.cache.estimand.method
            || operation.preparation.adjustment_set != operation.cache.estimand.adjustment_set
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
                operation.cache.identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)?;
        Self::assemble_checked_operation_effect(
            data,
            &static_result_metadata(
                &operation.common,
                &operation.source_query,
                &operation.query,
                &operation.cache.identification,
                &operation.cache.estimand,
            ),
            estimate,
            Some(operation.fitter.bootstrap_replicates),
            started,
            ctx,
        )
    }

    pub(super) fn execute_checked_rd(
        &self,
        data: &TabularData,
        operation: &CheckedRdOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let lowering = operation.preparation.lowering();
        if operation.estimator != crate::strategy_table::EstimatorId::RdSharp
            || operation.identifier != crate::strategy_table::IdentifierId::RdSharp
            || !operation.physical.binding().names(
                crate::strategy_table::IdentifierId::RdSharp,
                crate::strategy_table::EstimatorId::RdSharp,
            )
            || operation.graph_class != GraphClass::Dag
            || !matches!(
                operation.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !matches!(operation.inference, InferenceMode::Frequentist)
            || lowering.functional != operation.cache.estimand.functional
            || lowering.treatment != operation.query.treatment
            || lowering.outcome != operation.query.outcome
            || operation.preparation.target().functional != operation.cache.estimand.functional
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
        Self::assemble_checked_operation_effect(
            data,
            &static_result_metadata(
                &operation.common,
                &source_query,
                &operation.query,
                &operation.cache.identification,
                &operation.cache.estimand,
            ),
            estimate,
            Some(operation.fitter.bootstrap_replicates),
            started,
            ctx,
        )
    }

    pub(super) fn execute_checked_propensity(
        &self,
        data: &TabularData,
        operation: &CheckedPropensityOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let retained = operation.context();
        let lowering = operation.lowering();
        if self.analysis.query != retained.source_query
            || self.plan.binding().estimator != retained.physical.binding().estimator
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
        let mut metadata = static_result_metadata(
            &retained.common,
            &retained.source_query,
            operation.query(),
            &retained.cache.identification,
            operation.target(),
        );
        metadata.estimator = estimator;
        Self::assemble_checked_operation_effect(
            data,
            &metadata,
            estimate,
            bootstrap_requested,
            started,
            ctx,
        )
    }

    pub(super) fn execute_checked_conditional(
        &self,
        data: &TabularData,
        operation: &CheckedConditionalOperation,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.analysis.query != operation.source_query
            || self.plan.binding().estimator != operation.physical.binding().estimator
            || operation.program.mapping().source != operation.cache.estimand.functional
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
        let mut metadata = static_result_metadata(
            &operation.common,
            &operation.source_query,
            &operation.query.inner,
            &operation.cache.identification,
            &operation.cache.estimand,
        );
        metadata.estimator = estimator;
        metadata.graph_class = operation.graph_class();
        metadata.custom_validator_names = &[];
        let mut result =
            Self::assemble_checked_operation_effect(data, &metadata, estimate, None, started, ctx)?;
        // The checked conditional operation carries the full class proof, but
        // the shared result assembler only knows the selected invariant point
        // identification. Preserve the class envelope in the public result
        // certificate so completion masses and individual cases remain
        // inspectable just as they are for the ordinary class-aware routes.
        result.certificate = match &operation.class_proof {
            Some(crate::analysis::checked_conditional::ConditionalClassProof::Pag {
                cache,
                ..
            }) => Some(crate::result::AnalysisIdentification {
                identification: crate::Identification::Envelope {
                    envelope: (*cache.envelope).clone(),
                    strategy: operation.identifier,
                    structure_version: operation.graph_version,
                },
                query: operation.source_query.clone(),
                graph_class: operation.graph_class(),
            }),
            Some(crate::analysis::checked_conditional::ConditionalClassProof::Cpdag {
                cache,
                ..
            }) => Some(crate::result::AnalysisIdentification {
                identification: crate::Identification::CpdagEnvelope {
                    envelope: (*cache.envelope).clone(),
                    strategy: operation.identifier,
                    structure_version: operation.graph_version,
                },
                query: operation.source_query.clone(),
                graph_class: operation.graph_class(),
            }),
            None => result.certificate,
        };
        if let Some(diagnostic) = crate::analysis::helpers::conditional_quantile_grid_diagnostic(
            data,
            &operation.query,
            operation.cache.estimand.adjustment_set.iter().copied(),
        )? {
            result.diagnostics.push(diagnostic);
        }
        result.diagnostics.extend(
            crate::analysis::helpers::conditional_score_estimator_diagnostic(&operation.query),
        );
        // The exceedance-grid diagnostics (CDF band semantics, unsupported
        // threshold tails, cleared first-threshold scalar) are the ones the
        // shared finisher publishes for the same estimate.
        let mut seen: std::collections::HashSet<Arc<str>> =
            result.diagnostics.iter().map(|diagnostic| Arc::clone(&diagnostic.code)).collect();
        if let Some(effect) = result.estimate.as_effect() {
            crate::analysis::execute::push_grid_scalar_cleared(
                &mut result.diagnostics,
                &mut seen,
                effect,
            );
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
        run: CheckedStaticEffectRun,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let CheckedStaticEffectRun { estimator_id, estimate, bootstrap_requested, started } = run;
        let cancelled = estimate.uncertainty.bootstrap.cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut extra_diagnostics) =
            if cancelled || self.analysis.refute == RefuteSuite::None {
                (Vec::new(), Vec::new())
            } else {
                let mut workspace = EstimationWorkspace::default();
                let mut propensity = antecedent_stats::PropensityWorkspace::default();
                let (reports, diagnostics) = RefutationRun {
                    data,
                    estimand: &estimand,
                    query,
                    estimate: &estimate,
                    suite: self.analysis.refute,
                    estimator: estimator_id.as_str(),
                    custom: &self.analysis.custom_validators,
                    temporal: None,
                }
                .reports(&mut workspace, Some(&mut propensity), ctx)?;
                (reports, diagnostics)
            };
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the checked prepared operation",
        ));
        diagnostics.append(&mut extra_diagnostics);
        let identifier_id = self.plan.binding().identifier_or(RouteDefaults::STATIC_ATE)?;
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;
        let (identify_artifact, identify_operation) =
            crate::strategy_table::identify_provenance_step(identifier_id);
        let (estimate_artifact, estimate_operation) =
            crate::strategy_table::estimate_provenance_step(estimator_id);
        let provenance = provenance_pair(
            (identify_artifact, identify_operation, &[], &identification.required_assumptions),
            (
                estimate_artifact,
                estimate_operation,
                &[identify_artifact],
                &estimate.provenance.assumptions,
            ),
        );
        let mut result = assemble_result(AssembleArgs {
            logical: &self.plan.logical.record,
            physical: &self.plan.record,
            identification: identification.clone(),
            estimand,
            estimate,
            posterior: None,
            payload: crate::result::ResultPayload::Scalar,
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
        let custom_validator_names = self
            .analysis
            .custom_validators
            .iter()
            .map(|validator| Arc::from(validator.name()))
            .collect::<Vec<_>>();
        stamp_result_context(
            &mut result,
            self.analysis.support_status,
            self.analysis.structure_source,
            self.analysis.population_registry.as_ref(),
            &custom_validator_names,
        );
        crate::analysis::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }

    fn assemble_checked_operation_effect(
        data: &TabularData,
        metadata: &CheckedStaticResultMetadata<'_>,
        estimate: EffectEstimate,
        bootstrap_requested: Option<u32>,
        started: Instant,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let cancelled = estimate.uncertainty.bootstrap.cancelled || ctx.cancellation.is_cancelled();
        let (refutations, mut extra_diagnostics) =
            if cancelled || metadata.refute == RefuteSuite::None {
                (Vec::new(), Vec::new())
            } else {
                let mut workspace = EstimationWorkspace::default();
                let mut propensity = antecedent_stats::PropensityWorkspace::default();
                let (reports, diagnostics) = RefutationRun {
                    data,
                    estimand: metadata.estimand,
                    query: metadata.query,
                    estimate: &estimate,
                    suite: metadata.refute,
                    estimator: metadata.estimator.as_str(),
                    custom: &[],
                    temporal: None,
                }
                .reports(&mut workspace, Some(&mut propensity), ctx)?;
                (reports, diagnostics)
            };
        let mut diagnostics = metadata.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        if metadata.identifier != crate::strategy_table::IdentifierId::RdSharp {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "exec.identify.cached",
                antecedent_core::DiagnosticKind::Execution,
                antecedent_core::DiagnosticSeverity::Info,
                "identification reused from the checked prepared operation",
            ));
        }
        diagnostics.append(&mut extra_diagnostics);
        if let Some((tier, configured)) = metadata.latency_bootstrap_not_applied {
            diagnostics.push(crate::analysis::helpers::latency_bootstrap_not_applied_diagnostic(
                tier, configured,
            ));
        }
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;
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
            (
                estimate_artifact,
                estimate_operation,
                &[identify_artifact],
                &estimate.provenance.assumptions,
            ),
        );
        let mut result = assemble_result(AssembleArgs {
            logical: &metadata.physical.logical.record,
            physical: &metadata.physical.record,
            identification: metadata.identification.clone(),
            estimand: metadata.estimand.clone(),
            estimate,
            posterior: None,
            payload: crate::result::ResultPayload::Scalar,
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
        stamp_result_context(
            &mut result,
            metadata.support_status,
            metadata.structure_source,
            metadata.population_registry,
            metadata.custom_validator_names,
        );
        crate::analysis::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        Ok(result)
    }

    pub(super) fn stamp_aipw(
        &self,
        data: &DataInput,
        operation: &CheckedAipwOperation,
        mut result: StudyResult,
        defer: bool,
    ) -> Result<StudyResult, CausalError> {
        crate::analysis::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &operation.inference,
            data,
        );
        self.bind_executed_contract(data, operation.refute, defer, &mut result)?;
        Ok(result)
    }
}
