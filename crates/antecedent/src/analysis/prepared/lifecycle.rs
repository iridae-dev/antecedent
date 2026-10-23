//! Refresh, refute and modality-specific entry points of a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::estimate::{ClickNotes, ClickScores, Rebound};
use super::sealed::{SeriesCall, SeriesEntry};
use super::{
    PreparedExecution, PreparedModality, PreparedStudy, overlay_prepared_score_functional,
};
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::execute::Study;
use crate::analysis::helpers::{RefutationRun, project_for_ate_estimate};
use crate::analysis::route_guards::series_input;
use crate::analysis::stage::{STAGE_VALIDATE, StageClock};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::RouteDefaults;
use crate::result::StudyResult;
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext};
use antecedent_data::{PanelData, TableView, TabularData, TimeSeriesData};
use antecedent_estimate::{AipwAte, EstimationWorkspace};
use std::sync::Arc;
use std::time::Instant;

impl PreparedStudy {
    /// Refresh a retained AIPW study after changing only its numerical estimator options.
    ///
    /// The checked graph/query and cached identification are immutable inputs to this
    /// operation. The numerical problem, checked AIPW receipt and score table are rebuilt
    /// from `estimator` and `data`; the identification cache itself remains retained.
    /// Updates are transactional: a failed preparation or refresh leaves `self` unchanged.
    ///
    /// # Errors
    ///
    /// The handle is not a checked AIPW average-effect study, the schema/configuration is
    /// incompatible, or the refreshed estimate cannot be computed.
    pub(crate) fn refresh_aipw_configuration(
        &mut self,
        data: TabularData,
        estimator: AipwAte,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use crate::estimator_spec::EstimatorSpec;
        use crate::strategy_table::EstimatorId;

        if self.analysis.estimator != Some(EstimatorId::Aipw)
            || !matches!(self.analysis.query, CausalQuery::AverageEffect(_))
            || self.analysis.identification_cache.is_none()
            || !matches!(
                self.execution,
                PreparedExecution::OrdinaryDispatch | PreparedExecution::CheckedAipw(_)
            )
        {
            return Err(CausalError::Unsupported {
                message: "AIPW configuration refresh requires the same checked average-effect study",
            });
        }

        self.transform_capability(antecedent_contract::TransformIntent::CompatibleDataReplace)?;
        self.ensure_schema_compatible(&data)?;

        // Work on a candidate so config installation, score rebuilding and execution fail
        // atomically from the session's point of view.
        let mut candidate = self.clone();
        let spec = EstimatorSpec::Aipw(Box::new(estimator.clone()));
        Arc::make_mut(&mut candidate.analysis.config).estimator_spec_identity =
            Some(crate::analysis::contract_identity::estimator_spec_identity(&spec));
        candidate.analysis.estimator_spec = Some(Arc::new(spec));
        candidate.program_cache = std::sync::OnceLock::new();
        candidate.contract_cache = std::sync::OnceLock::new();

        if let PreparedExecution::CheckedAipw(operation) = &mut candidate.execution {
            let identification = &operation.cache.identification;
            let estimand_index = identification
                .estimands
                .iter()
                .position(|estimand| estimand == &operation.cache.estimand)
                .ok_or(CausalError::Unsupported {
                    message: "retained AIPW estimand is absent from cached identification",
                })?;
            let preparation = estimator
                .prepare_checked(&data, identification, estimand_index)
                .map_err(CausalError::from)?;
            operation.fitter = estimator;
            operation.preparation = preparation;
        }

        let result = candidate.refresh(data, ctx)?;
        *self = candidate;
        Ok(result)
    }

    /// Replace retained data and re-estimate (same semantics as [`Self::estimate`]).
    ///
    /// Everything the click derives from the new rows is handed to the
    /// handle rather than derived again: the rebound sealed operation, the
    /// frozen scores of a functional click, and, through one shared
    /// nuisance scope, the cross-fitted nuisances of the click and of the
    /// frozen score table (see [`Self::refresh_scoped`]).
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn refresh(
        &mut self,
        data: TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_contract::TransformIntent::CompatibleDataReplace)?;
        self.ensure_schema_compatible(&data)?;
        // The refreshed result carries the same native observations a fresh estimate does, so
        // its claim identity agrees with the one the retained handle reproduces.
        let mut observed = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            antecedent_estimate::CrossfitNuisanceCache::within_scope(|| {
                self.refresh_scoped(data, ctx)
            })
        });
        if let Ok(result) = &mut observed.result {
            result.performance.native_execution.clone_from(&observed.facts);
        }
        observed.result
    }

    /// The body of [`Self::refresh`], run inside one nuisance scope.
    ///
    /// The click and the frozen score table of the refreshed study fit the
    /// same cross-fitted nuisances: the same design, treatment coding, fold
    /// plan (seeded by the context) and learner options. The scope keys every
    /// fit on all of those inputs, so the second request is served from the
    /// first fit and the table is bit-identical to an unshared build.
    fn refresh_scoped(
        &mut self,
        data: TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let shared = self
            .analysis
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(&data).map(Arc::new))
            .transpose()?;
        // The checked static lowering belongs to the prepared handle. Execute
        // through it before replacing the retained data snapshot; otherwise the
        // generic Study refresh route would rederive an unchecked preparation.
        let mut notes = ClickNotes::deferred();
        let click = if matches!(self.execution, PreparedExecution::OrdinaryDispatch) {
            None
        } else {
            Some(
                self.estimate_with_shared_unnamed(&data, shared.clone(), ctx, &mut notes)
                    .map_err(|error| self.name_stats_refusal(&data, error, ctx))?,
            )
        };
        let mut refreshed = self.analysis.clone();
        refreshed.shared_batch_design = shared;
        refreshed.interference =
            refreshed.interference.as_ref().map(|spec| spec.bound_to(&data)).transpose()?;
        refreshed.data = DataInput::Tabular(data);
        if let Some(mut result) = click {
            let ClickNotes { rebound, scores, .. } = notes;
            // Any prepared route that retained scores before execution must
            // rebuild them against the refreshed rows. A functional click has
            // already built them for exactly this study snapshot and context;
            // cell-AIPW response operations are sealed too, but their row
            // scores remain the source for retargeting after the click.
            let scores = match scores {
                ClickScores::Built(built) => built.filter(|_| self.score_table.is_some()),
                ClickScores::NotBuilt if self.score_table.is_some() => {
                    refreshed.prepare_score_table(ctx)?
                }
                ClickScores::NotBuilt => None,
            };
            self.replace_study(refreshed);
            // Receipts that bind complete-case rows (linear, GLM, RD, AIPW,
            // front-door, IV, conditional, cell-AIPW) must describe the
            // retained snapshot, or the published contract would carry
            // prepare-time row counts.
            if let Some(rebound) = rebound {
                self.install_rebound(rebound);
            }
            self.score_table = scores;
            // The click ran while this handle still referenced its previous
            // study snapshot. Refresh has now replaced the retained data and
            // any data-bound evidence (such as the fixed interference
            // network), so the result is bound once, to the snapshot the
            // handle actually retains.
            result.executed_contract =
                Some(self.executed_contract(&self.analysis.data, self.analysis.refute, None)?);
            return Ok(result);
        }
        let mut result = refreshed.execute_shared(&self.plan, ctx)?;
        let scores = refreshed.prepare_score_table(ctx)?;
        overlay_prepared_score_functional(&refreshed.query, scores.as_ref(), &mut result)?;
        self.replace_study(refreshed);
        self.score_table = scores;
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }

    /// Install a click's rebound operation. Out of line so the refresh frame
    /// never holds a second execution value.
    #[inline(never)]
    fn install_rebound(&mut self, rebound: Rebound) {
        if let Rebound::Linear(operation) = &rebound {
            self.plan = Arc::clone(&operation.common.physical);
        }
        self.execution = rebound.into_execution();
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
        Arc::make_mut(&mut self.analysis.config).estimator_spec_identity =
            Some(crate::analysis::contract_identity::estimator_spec_identity(&spec));
        self.analysis.overlap_policy = spec.propensity_overlap();
        self.analysis.estimator_spec = Some(Arc::new(spec));
        self.program_cache = std::sync::OnceLock::new();
        self.contract_cache = std::sync::OnceLock::new();
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
    /// Schema mismatch, missing `AverageEffect` query, cancel, or validator failures.
    pub fn refute(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        if let CausalQuery::Mediation(query) = &self.analysis.query {
            return self.refute_mediation(prior, data, suite, query, ctx);
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
            return self.refute_bayesian_rerun(prior, data, suite, ctx);
        }
        let estimator = self.plan.binding().estimator_or(RouteDefaults::STATIC_ATE)?;

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
        let (reports, na_diagnostics) = RefutationRun {
            data: &data_est,
            estimand: &estimand_est,
            query: &query_est,
            estimate: effect,
            suite,
            estimator: estimator.as_str(),
            custom: &self.analysis.custom_validators,
            temporal: None,
        }
        .reports(&mut workspace, None, ctx)?;
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
        self.stamp_refuted(prior, data, suite, out)
    }

    /// Second-click refute of a static mediation prior under `suite`.
    fn refute_mediation(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        query: &antecedent_core::MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if prior.treatment != Some(query.treatment)
            || prior.outcome != query.outcome
            || prior.identification.query != self.analysis.query
        {
            return Err(CausalError::Compile {
                message: "refute prior does not match mediation query".into(),
            });
        }
        let graph =
            self.analysis.graph.as_dag().ok_or(CausalError::Unsupported {
                message: "static mediation refute requires Dag",
            })?;
        let mediation = prior.mediation().ok_or(CausalError::Unsupported {
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
        self.stamp_refuted(prior, data, suite, result)
    }

    /// Second-click refute of a Bayesian prior by re-running the frozen plan with `suite`.
    fn refute_bayesian_rerun(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // Bayesian validation also includes prior/posterior predictive checks
        // (and, for Full, prior sensitivity/MCMC diagnostics). Re-running the
        // frozen physical plan is the only path that constructs those artifacts;
        // retain the prior point/identification while replacing validation state.
        let started = Instant::now();
        let mut analysis = self.analysis.clone();
        analysis.refute = suite;
        let validated = analysis.execute_on(&DataInput::Tabular(data.clone()), &self.plan, ctx)?;
        let validate_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let mut out = prior.clone();
        out.refutations = validated.refutations;
        crate::analysis::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
        out.predictive_checks = validated.predictive_checks;
        out.posterior = validated.posterior;
        // A suite whose cell executes unlicensed (the Bayesian intervention-level
        // refuter state) must not inherit the licensed status of the estimate.
        let suite_cell = crate::support::support_cell_named(
            &analysis.query,
            if analysis.graph_posterior.is_some() {
                analysis.graph.class().as_str()
            } else {
                crate::support::matrix_graph_class(
                    &analysis.graph,
                    &analysis.query,
                    analysis.tiered.as_ref(),
                )
            },
            analysis.structure_source,
            &analysis.inference,
            suite,
        );
        if suite_cell.is_some_and(crate::support::executes_unlicensed) {
            out.support_status = Some(crate::support::CellStatus::Refused);
        }
        out.performance.stage_timings_ns.push((Arc::from(STAGE_VALIDATE), validate_ns));
        out.performance.wall_time_ns =
            Some(out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns));
        out.diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.refute.second_click",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            format!("second-click refute suite={}", suite.diagnostic_label()),
        ));
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
        // The second click ran this handle's own program on its own result.
        crate::analysis::target_validation::stamp_route_refutations(&mut out);
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
        let mut result =
            match self.run_sealed_series(&self.analysis, &input, ctx, SeriesEntry::Estimate)? {
                Some(result) => result,
                None => self.analysis.execute_on(&input, &self.plan, ctx)?,
            };
        crate::analysis::execute::push_lag_zero_average_effect_disclosure(
            &self.analysis,
            &mut result,
        );
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
        let scoped = ctx.enter_lowered_dispatch(self.plan.require_lowered(None)?.identity());
        self.execution.operation().estimate_series(&SeriesCall {
            handle: self,
            study,
            input,
            series,
            ctx: &scoped,
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
        self.transform_capability(antecedent_contract::TransformIntent::CompatibleDataReplace)?;
        self.ensure_panel_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = DataInput::Panel(data);
        let result = refreshed.execute_shared(&self.plan, ctx)?;
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
        self.transform_capability(antecedent_contract::TransformIntent::CompatibleDataReplace)?;
        self.ensure_multi_env_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = DataInput::MultiEnv(data);
        let result = refreshed.execute_shared(&self.plan, ctx)?;
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
        self.transform_capability(antecedent_contract::TransformIntent::CompatibleDataReplace)?;
        self.ensure_series_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = series_input(&self.analysis.data, data);
        let result =
            match self.run_sealed_series(&refreshed, &refreshed.data, ctx, SeriesEntry::Refresh)? {
                Some(result) => result,
                None => refreshed.execute_shared(&self.plan, ctx)?,
            };
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }
}
