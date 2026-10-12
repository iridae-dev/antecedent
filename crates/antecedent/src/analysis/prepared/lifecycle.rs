//! Refresh, refute and modality-specific entry points of a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sealed::{SeriesCall, SeriesEntry};
use super::{
    PreparedExecution, PreparedModality, PreparedStudy, overlay_prepared_score_functional,
};
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::execute::Study;
use crate::analysis::helpers::{project_for_ate_estimate, run_refuters};
use crate::analysis::route_guards::series_input;
use crate::analysis::stage::{STAGE_VALIDATE, StageClock};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::StudyResult;
use crate::strategy_table::DEFAULT_ESTIMATOR;
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext};
use antecedent_data::{PanelData, TableView, TabularData, TimeSeriesData};
use antecedent_estimate::EstimationWorkspace;
use std::sync::Arc;
use std::time::Instant;

impl PreparedStudy {
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
        let checked_result = (!matches!(self.execution, PreparedExecution::OrdinaryDispatch))
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
        if let Some(mut result) = checked_result {
            let rebound_conditional = match (&self.execution, &refreshed.data) {
                (PreparedExecution::CheckedConditional(operation), DataInput::Tabular(data)) => {
                    Some(operation.rebind(data)?)
                }
                _ => None,
            };
            // Any prepared route that retained scores before execution must
            // rebuild them against the refreshed rows. Cell-AIPW response
            // operations are sealed too, but their row scores remain the
            // source for retargeting after the click.
            let scores =
                if self.score_table.is_some() { refreshed.prepare_score_table(ctx)? } else { None };
            // Receipts that bind complete-case rows (linear, GLM, RD, AIPW,
            // front-door, IV) must describe the retained snapshot, or the
            // published contract would carry prepare-time row counts.
            let rebound_dispatch = self.rebound_dispatch_execution(&refreshed.data)?;
            self.replace_study(refreshed);
            if let Some(execution) = rebound_dispatch {
                self.install_execution(execution);
            }
            if let PreparedExecution::CellAipwResponse(operation) = &self.execution {
                if let DataInput::Tabular(data) = &self.analysis.data {
                    self.execution = PreparedExecution::CellAipwResponse(operation.refresh(
                        &self.analysis.graph,
                        self.analysis.tiered.as_ref(),
                        data,
                    )?);
                }
            }
            if let Some(operation) = rebound_conditional {
                self.execution = PreparedExecution::CheckedConditional(operation);
            }
            self.score_table = scores;
            // `checked_result` was stamped while this handle still referenced
            // its previous study snapshot. Refresh has now replaced the
            // retained data and any data-bound evidence (such as the fixed
            // interference network), so bind the result to the snapshot the
            // handle actually retains before returning it to callers.
            result.executed_contract =
                Some(self.executed_contract(&self.analysis.data, self.analysis.refute, None)?);
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

    /// Execute the retained discrete ADMG response providers without rebuilding them.
    pub(crate) fn execute_static_functional_retained(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let operation = self.execution.admg_response_curve().ok_or(CausalError::Unsupported {
            message: "prepared static functional response required",
        })?;
        let DataInput::Tabular(data) = &self.analysis.data else {
            return Err(CausalError::Unsupported {
                message: "static functional tabular data required",
            });
        };
        let mut result = operation.execute(data, ctx)?;
        result.executed_contract =
            Some(self.executed_contract(&self.analysis.data, operation.refute, None)?);
        Ok(result)
    }
    /// Rebind only the empirical factor providers of the retained checked static response.
    pub(crate) fn rebind_static_functional_data(
        &mut self,
        data: TabularData,
    ) -> Result<(), CausalError> {
        self.ensure_schema_compatible(&data)?;
        let operation = self
            .execution
            .admg_response_curve()
            .ok_or(CausalError::Unsupported {
                message: "prepared static functional response required",
            })?
            .rebind(&data)?;
        let mut study = self.analysis.clone();
        study.data = DataInput::Tabular(data);
        self.replace_study(study);
        self.execution = PreparedExecution::AdmgResponseCurve(operation);
        Ok(())
    }

    /// Refresh only a DML/DR nuisance configuration, retaining the checked causal query.
    /// The adapter has already checked that graph/query/regime/schema identities are unchanged.
    pub(crate) fn refresh_dr_configuration(
        &mut self,
        data: TabularData,
        spec: crate::estimator_spec::EstimatorSpec,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use crate::strategy_table::EstimatorId;
        if !matches!(spec.id(), EstimatorId::Dml | EstimatorId::DrLearner)
            || self.analysis.estimator != Some(spec.id())
            || !matches!(self.execution, PreparedExecution::OrdinaryDispatch)
            || self.analysis.identification_cache.is_none()
        {
            return Err(CausalError::Unsupported {
                message: "DML/DR configuration refresh requires the same checked estimator family",
            });
        }
        self.analysis.estimator_spec_identity =
            Some(crate::analysis::contract_identity::estimator_spec_identity(&spec));
        self.analysis.overlap_policy = spec.propensity_overlap();
        self.analysis.estimator_spec = Some(spec);
        self.program_cache = std::sync::OnceLock::new();
        self.refresh(data, ctx)
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
            if prior.treatment != Some(query.treatment)
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
                ) && prior.estimate.as_effect().is_some_and(|effect| effect.ate.is_finite()) =>
            {
                AverageEffectQuery::binary_ate(
                    prior.treatment.ok_or(CausalError::Unsupported {
                        message: "scalar response result has no treatment id",
                    })?,
                    prior.outcome,
                )
            }
            _ => {
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "PreparedStudy::refute is licensed for AverageEffect and scalar \
                              InterventionResponse",
                });
            }
        };
        if prior.treatment != Some(query.treatment) || prior.outcome != query.outcome {
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
            crate::analysis::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
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
        let effect = prior.estimate.as_effect().ok_or_else(|| CausalError::Compile {
            message: "second-click refuters require an effect estimate".into(),
        })?;
        let (reports, na_diagnostics) = run_refuters(
            &data_est,
            &estimand_est,
            &query_est,
            effect,
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
        crate::analysis::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
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

    pub(super) fn ensure_schema_compatible(&self, data: &TabularData) -> Result<(), CausalError> {
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
        let input = series_input(&self.analysis.data, data.clone());
        let result =
            match self.run_sealed_series(&self.analysis, &input, ctx, SeriesEntry::Estimate)? {
                Some(result) => result,
                None => self.analysis.execute_on(&input, &self.plan, ctx)?,
            };
        self.stamp(&input, result)
    }

    /// Execute the retained series operation, if any, over `study` and
    /// `input`. `study` is the prepared study (estimate) or its refreshed
    /// clone (refresh); `entry` selects the wording of each refusal.
    /// `None` means no series family is sealed and ordinary dispatch applies.
    #[inline(never)]
    fn run_sealed_series(
        &self,
        study: &Study,
        input: &DataInput,
        ctx: &ExecutionContext,
        entry: SeriesEntry,
    ) -> Result<Option<StudyResult>, CausalError> {
        let (DataInput::Temporal(series) | DataInput::Event(series)) = input else {
            unreachable!("series execution constructs temporal data")
        };
        self.execution.operation().estimate_series(&SeriesCall {
            handle: self,
            study,
            input,
            series,
            ctx,
            entry,
        })
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
        refreshed.data = series_input(&self.analysis.data, data);
        let result =
            match self.run_sealed_series(&refreshed, &refreshed.data, ctx, SeriesEntry::Refresh)? {
                Some(result) => result,
                None => refreshed.execute(&self.plan, ctx)?,
            };
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }
}
