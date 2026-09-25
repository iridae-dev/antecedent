//! Checked fixed-TemporalDag mediation execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_data::LaggedColumn;
use antecedent_graph::TemporalDag;

#[derive(Clone)]
struct CheckedMediationHorizon {
    horizon: u32,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    adjustment: Arc<[LaggedColumn]>,
    adjustment_keys: Arc<[antecedent_core::TemporalNodeKey]>,
}

#[derive(Clone)]
enum CheckedMediationInference {
    Frequentist { bootstrap_replicates: u32 },
    Bayesian(BayesianConfig),
}

/// Frozen query, per-horizon proof/design, estimator procedure, and validation
/// contract for one fixed temporal mediation DAG.
#[derive(Clone)]
pub(crate) struct CheckedTemporalMediationOperation {
    graph: TemporalDag,
    query: antecedent_core::MediationQuery,
    horizons: Arc<[CheckedMediationHorizon]>,
    inference: CheckedMediationInference,
    validation: RefuteSuite,
    physical: PhysicalExecutionPlan,
    schema: antecedent_core::CausalSchema,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedTemporalMediationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedTemporalMediationOperation")
            .field("query", &self.query)
            .field("graph", &self.graph)
            .field("horizons", &self.horizons.iter().map(|h| h.horizon).collect::<Vec<_>>())
            .field("inference", &self.inference_name())
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalMediationOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TimeSeriesData,
        physical: &PhysicalExecutionPlan,
        cache: &crate::analysis::prepared::CachedTemporalIdentification,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked temporal mediation requires a TemporalMediationEffect target"
                    .into(),
            });
        };
        let graph = study.graph.as_temporal_dag().ok_or(CausalError::Unsupported {
            message: "checked temporal mediation requires a fixed TemporalDag",
        })?;
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if !matches!(
            study.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) || study.split.is_some()
            || study.graph_posterior.is_some()
            || !study.custom_validators.is_empty()
            || !matches!(study.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            || physical.logical.query != study.query
            || physical.logical.record.plan_id.as_ref() != "temporal_mediation"
            || physical.logical.record.identifier.as_deref() != Some("temporal.mediation")
        {
            return Err(CausalError::Compile {
                message: "temporal mediation structure, target, validation, or plan changed after preparation".into(),
            });
        }
        let inference = match &study.inference {
            InferenceMode::Frequentist => {
                if physical.logical.record.estimator.as_deref() != Some("temporal.mediation") {
                    return Err(CausalError::Compile {
                        message:
                            "temporal mediation estimator differs from its checked physical plan"
                                .into(),
                    });
                }
                CheckedMediationInference::Frequentist {
                    bootstrap_replicates: study.bootstrap_replicates,
                }
            }
            InferenceMode::Bayesian(config) => {
                if config.prior.is_some()
                    || config.prior_artifact.is_some()
                    || config.external_compose.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian temporal mediation supports isotropic mechanism priors; shared or transferred coefficient priors are not licensed",
                    });
                }
                if physical.logical.record.estimator.as_deref()
                    != Some("temporal.mediation.bayesian")
                {
                    return Err(CausalError::Compile {
                        message: "Bayesian temporal mediation estimator differs from its checked physical plan".into(),
                    });
                }
                CheckedMediationInference::Bayesian(config.clone())
            }
        };
        let estimator_id = match &inference {
            CheckedMediationInference::Frequentist { .. } => EstimatorId::TemporalMediation,
            CheckedMediationInference::Bayesian(_) => EstimatorId::BayesianTemporalMediation,
        };
        let verified = crate::analysis::prepared::identify_temporal_mediation_horizons(
            graph,
            query,
            estimator_id,
        )?;
        if cache.by_horizon.len() != query.horizons.len()
            || verified.by_horizon.len() != query.horizons.len()
        {
            return Err(CausalError::Compile {
                message:
                    "temporal mediation proof cache does not cover the requested horizon family"
                        .into(),
            });
        }
        let mut horizons = Vec::with_capacity(query.horizons.len());
        for &horizon in query.horizons.iter() {
            let cached = cache.get(horizon).ok_or_else(|| CausalError::Compile {
                message: format!("temporal mediation proof missing horizon {horizon}"),
            })?;
            let checked = verified.get(horizon).ok_or_else(|| CausalError::Compile {
                message: format!("recomputed temporal mediation proof missing horizon {horizon}"),
            })?;
            if cached.identification.query != checked.identification.query
                || cached.identification.status != checked.identification.status
                || cached.estimand.functional != checked.estimand.functional
                || cached.estimand.method != checked.estimand.method
                || cached.estimand.adjustment_set != checked.estimand.adjustment_set
                || cached.estimand.mediators != checked.estimand.mediators
                || cached.indexer.variable_count() != checked.indexer.variable_count()
                || cached.indexer.history() != checked.indexer.history()
                || cached.indexer.horizon() != checked.indexer.horizon()
            {
                return Err(CausalError::Compile {
                    message: format!("temporal mediation proof failed replay at horizon {horizon}"),
                });
            }
            if cached.estimand.mediators.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal mediation supports exactly one identified mediator",
                });
            }
            require_identified(&cached.identification)?;
            let adjustment = adjustment_for_horizon(cached)?;
            let adjustment_keys = cached
                .estimand
                .adjustment_set
                .iter()
                .map(|dense| {
                    cached
                        .indexer
                        .key_of(dense.raw())
                        .map_err(|error| CausalError::Compile { message: error.to_string() })
                })
                .collect::<Result<Vec<_>, _>>()?;
            horizons.push(CheckedMediationHorizon {
                horizon,
                identification: cached.identification.clone(),
                estimand: cached.estimand.clone(),
                indexer: cached.indexer.clone(),
                adjustment,
                adjustment_keys: adjustment_keys.into(),
            });
        }
        Ok(Self {
            graph: graph.clone(),
            query: query.clone(),
            horizons: horizons.into(),
            inference,
            validation: study.refute,
            physical: physical.clone(),
            schema: data.schema().clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn matches_graph(&self, graph: &TemporalDag) -> bool {
        self.graph.nodes() == graph.nodes() && self.graph.edges().eq(graph.edges())
    }

    pub(crate) fn estimator(&self) -> EstimatorId {
        match &self.inference {
            CheckedMediationInference::Frequentist { .. } => EstimatorId::TemporalMediation,
            CheckedMediationInference::Bayesian(_) => EstimatorId::BayesianTemporalMediation,
        }
    }

    fn inference_name(&self) -> &'static str {
        match &self.inference {
            CheckedMediationInference::Frequentist { .. } => "temporal.mediation",
            CheckedMediationInference::Bayesian(_) => "temporal.mediation.bayesian",
        }
    }

    pub(crate) fn inspect(
        &self,
    ) -> (&antecedent_core::MediationQuery, &TemporalDag, RefuteSuite, Vec<u32>) {
        (
            &self.query,
            &self.graph,
            self.validation,
            self.horizons.iter().map(|h| h.horizon).collect(),
        )
    }

    pub(crate) fn horizon_contracts(
        &self,
    ) -> impl Iterator<
        Item = (u32, &IdentificationResult, &IdentifiedEstimand, &TemporalIndexer, &[LaggedColumn]),
    > {
        self.horizons
            .iter()
            .map(|h| (h.horizon, &h.identification, &h.estimand, &h.indexer, h.adjustment.as_ref()))
    }

    pub(crate) fn rebind(&self, data: &TimeSeriesData) -> Result<Self, CausalError> {
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "temporal mediation refresh changed the semantic variable schema".into(),
            });
        }
        Ok(self.clone())
    }

    pub(crate) fn execute(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let bound = self.rebind(data)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let started = Instant::now();
        let bayesian = matches!(bound.inference, CheckedMediationInference::Bayesian(_));
        let estimator_id = if bayesian {
            EstimatorId::BayesianTemporalMediation
        } else {
            EstimatorId::TemporalMediation
        };
        let single = bound.horizons.len() == 1;
        let mut slices = Vec::with_capacity(bound.horizons.len());
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        let mut first_posterior = None;
        let mut bootstrap_ok = None;
        let mut cancelled = false;
        let mut diagnostics = Vec::new();

        let natural_alias = matches!(
            bound.query.contrast,
            antecedent_core::MediationContrast::NaturalDirect
                | antecedent_core::MediationContrast::NaturalIndirect
        );
        if bayesian && natural_alias {
            diagnostics.push(Diagnostic::new(
                "estimate.mediation.bayesian",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; NaturalDirect and NaturalIndirect use the licensed additive linear no-treatment-mediator-interaction alias",
            ));
        }

        for horizon in bound.horizons.iter() {
            let refutation_start = refutations.len();
            let mut qh = bound.query.clone();
            qh.horizons = Arc::from([horizon.horizon]);
            let (mediation, uncertainty, posterior) = match &bound.inference {
                CheckedMediationInference::Frequentist { bootstrap_replicates } => {
                    let est =
                        TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
                    let (mut estimate, block) = est
                        .estimate_with_block_bootstrap(
                            data,
                            &horizon.estimand,
                            &qh,
                            &horizon.adjustment,
                            *bootstrap_replicates,
                            0x4D45_4449_0000_u64.wrapping_add(u64::from(horizon.horizon)),
                            ctx,
                        )
                        .map_err(CausalError::from)?;
                    bootstrap_ok = Some(
                        bootstrap_ok
                            .map_or(block.replicates_ok, |n: u32| n.min(block.replicates_ok)),
                    );
                    cancelled |= estimate.effect.bootstrap_cancelled;
                    if block.replicates_attempted > 0 {
                        estimate.effect = estimate
                            .effect
                            .clone()
                            .with_block_family(antecedent_estimate::CircularBlockFamily::Mediation);
                        diagnostics.extend(super::temporal_path::temporal_dependence_se_diagnostics(
                            antecedent_estimate::CircularBlockFamily::Mediation,
                            block.block_length,
                            block.rows,
                            block.kernel_bias,
                            block.effective_rows,
                            block.replicates_attempted > 0,
                            &format!(
                                "horizon {}: shared circular-block resampling refits all three mediation mechanism regressions, {}/{} replicates",
                                horizon.horizon, block.replicates_ok, block.replicates_attempted,
                            ),
                        ));
                    }
                    if bound.validation != RefuteSuite::None {
                        refutations.extend(
                            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                                data,
                                &horizon.estimand,
                                &qh,
                                &estimate,
                                bound.validation == RefuteSuite::Full,
                                &horizon.adjustment,
                                ctx,
                            )
                            .map_err(CausalError::from)?,
                        );
                    }
                    let uncertainty = if block.replicates_attempted > 0 {
                        antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                            requested: estimate.effect.se_bootstrap,
                            block,
                        }
                    } else {
                        antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
                            standard_error: estimate
                                .effect
                                .se_analytic
                                .is_finite()
                                .then_some(estimate.effect.se_analytic),
                        }
                    };
                    (estimate, uncertainty, None)
                }
                CheckedMediationInference::Bayesian(config) => {
                    let mut estimator = bayesian_gcomp(config, ctx);
                    antecedent_estimate::bayesian_mediation::require_gaussian_mediation(&estimator)
                        .map_err(CausalError::from)?;
                    let preparations = antecedent_estimate::bayesian_mediation::prepare_temporal_mediation_adjusted(
                        data,
                        &horizon.estimand,
                        &qh,
                        &horizon.adjustment,
                        ctx,
                    ).map_err(CausalError::from)?;
                    let posts = preparations
                        .iter()
                        .enumerate()
                        .map(|(i, prep)| {
                            estimator.seed = config_seed(config, ctx, horizon.horizon, i);
                            estimator
                                .fit(
                                    prep,
                                    horizon.identification.status,
                                    &mut BayesianGCompWorkspace::default(),
                                    ctx,
                                )
                                .map_err(CausalError::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut posterior =
                        antecedent_estimate::bayesian_mediation::compose_temporal_mediation(
                            &posts[0],
                            &posts[1],
                            &qh,
                            horizon.identification.status,
                        )
                        .map_err(CausalError::from)?;
                    if natural_alias {
                        posterior
                            .assumptions
                            .push(antecedent_estimate::linear_no_interaction_restriction());
                    }
                    let mut effect = effect_from_posterior(&posterior)?;
                    let estimate = TemporalMediationEstimate {
                        effect: effect.clone(),
                        total: Some(posterior.summaries.mean[1]),
                        direct: Some(posterior.summaries.mean[2]),
                        mediated: Some(posterior.summaries.mean[3]),
                    };
                    if bound.validation != RefuteSuite::None {
                        refutations.extend(
                            antecedent_validate::mediation::refute_temporal_mediation_adjusted(
                                data,
                                &horizon.estimand,
                                &qh,
                                &estimate,
                                bound.validation == RefuteSuite::Full,
                                &horizon.adjustment,
                                ctx,
                            )
                            .map_err(CausalError::from)?,
                        );
                        for (mechanism, (prep, post)) in
                            preparations.iter().zip(posts.iter()).enumerate()
                        {
                            let prior = estimator.prior_in_force(prep.design.ncols);
                            let ppc = PriorPredictiveCheck::for_estimator(&estimator, ctx)
                                .check_with_prior(prep, &prior, ctx)
                                .map_err(CausalError::from)?;
                            let postpc = PosteriorPredictiveCheck::for_estimator(&estimator, ctx)
                                .check(prep, post)
                                .map_err(CausalError::from)?;
                            for report in [ppc, postpc] {
                                let mut refutation = report.to_refutation_report(effect.ate, 0.05);
                                refutation.refuter = Arc::from(format!(
                                    "mediation.mechanism_{mechanism}.{}",
                                    refutation.refuter
                                ));
                                refutations.push(refutation);
                                predictive_checks.push(report);
                            }
                        }
                    }
                    if bound.validation == RefuteSuite::Full {
                        let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
                        let mut means = Vec::new();
                        let mut sds = Vec::new();
                        for &scale in sensitivity.scales.iter() {
                            let scaled = preparations
                                .iter()
                                .enumerate()
                                .map(|(i, prep)| {
                                    let mut scaled_estimator = estimator.clone();
                                    scaled_estimator.prior_scale = scale;
                                    scaled_estimator.seed =
                                        config_seed(config, ctx, horizon.horizon, i);
                                    scaled_estimator
                                        .fit(
                                            prep,
                                            horizon.identification.status,
                                            &mut BayesianGCompWorkspace::default(),
                                            ctx,
                                        )
                                        .map_err(CausalError::from)
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let scaled_posterior = antecedent_estimate::bayesian_mediation::compose_temporal_mediation(
                                &scaled[0], &scaled[1], &qh, horizon.identification.status,
                            ).map_err(CausalError::from)?;
                            means.push(scaled_posterior.summaries.mean[0]);
                            sds.push(scaled_posterior.summaries.sd[0]);
                        }
                        let summary = antecedent_prob::PriorSensitivitySummary {
                            family: antecedent_prob::PriorSensitivityFamily::IsotropicScale,
                            prior_scales: sensitivity.scales.clone(),
                            alphas: Arc::from([]),
                            variance_multipliers: Arc::from([]),
                            effect_means: Arc::from(means),
                            effect_sds: Arc::from(sds),
                        };
                        refutations.push(sensitivity.to_report(&summary, effect.ate));
                        posterior = with_prior_sensitivity(posterior, summary);
                        effect = effect_from_posterior(&posterior)?;
                    }
                    let summary = |column: usize| antecedent_estimate::MediationPosteriorSummary {
                        mean: posterior.summaries.mean[column],
                        standard_deviation: posterior.summaries.sd[column],
                        q025: posterior.summaries.q025[column],
                        q975: posterior.summaries.q975[column],
                    };
                    let uncertainty =
                        antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                            requested: summary(0),
                            total: summary(1),
                            direct: summary(2),
                            mediated: summary(3),
                            n_draws: posterior.draws.n_draws,
                            backend: Arc::clone(&posterior.diagnostics.backend_id),
                        };
                    (
                        TemporalMediationEstimate {
                            effect,
                            total: Some(posterior.summaries.mean[1]),
                            direct: Some(posterior.summaries.mean[2]),
                            mediated: Some(posterior.summaries.mean[3]),
                        },
                        uncertainty,
                        Some(posterior),
                    )
                }
            };
            if !single {
                for report in &mut refutations[refutation_start..] {
                    report.refuter =
                        Arc::from(format!("horizon.{}.{}", horizon.horizon, report.refuter));
                }
            }
            slices.push(antecedent_estimate::TemporalMediationSlice {
                horizon: horizon.horizon,
                identification_status: horizon.identification.status,
                method: Arc::clone(&horizon.estimand.method),
                adjustment: Arc::clone(&horizon.adjustment_keys),
                estimate: mediation,
                uncertainty,
                identified_set: None,
                diagnostics: horizon.identification.diagnostics.clone(),
            });
            if first_posterior.is_none() {
                first_posterior = posterior;
            }
        }

        let first = slices.first().ok_or_else(|| CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        })?;
        let estimate = if single { first.estimate.effect.clone() } else { nan_effect() };
        let mediation = single.then(|| first.estimate.clone());
        let mut identification = bound.horizons[0].identification.clone();
        if !single {
            identification.status = most_conservative_identification_status(
                slices.iter().map(|slice| slice.identification_status),
            );
            diagnostics.push(Diagnostic::new(
                "estimate.temporal_mediation.multi_horizon",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "each horizon retains pointwise mediation results; no joint cross-horizon uncertainty is asserted",
            ));
        }
        let grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from(slices),
            joint_posterior: false,
        };
        let certificate = single.then(|| crate::Identification::Point {
            result: identification.clone(),
            temporal_indexer: Some(bound.horizons[0].indexer.clone()),
            strategy: IdentifierId::Frontdoor,
            structure_version: bound.result_context.graph_version,
        });
        Ok(finish_identified_execute_with_context(
            &bound.result_context,
            None,
            IdentifiedExecuteFinish {
                physical: &bound.physical,
                identification,
                estimand: bound.horizons[0].estimand.clone(),
                estimate,
                identifier_id: IdentifierId::Frontdoor,
                estimator_id,
                treatment: bound.query.treatment,
                outcome: bound.query.outcome,
                identify_cached: true,
                extra_diagnostics: diagnostics,
                refutations,
                distribution: None,
                mediation,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: match &bound.inference {
                    CheckedMediationInference::Frequentist { bootstrap_replicates }
                        if *bootstrap_replicates > 0 =>
                    {
                        bootstrap_ok
                    }
                    _ => None,
                },
                cancelled,
                early_stopped: false,
                extras: IdentifiedExecuteExtras {
                    certificate,
                    mediation_grid: Some(grid),
                    posterior: if single { first_posterior } else { None },
                    predictive_checks,
                    n_draws: if single {
                        if let CheckedMediationInference::Bayesian(config) = &bound.inference {
                            u32::try_from(config.n_draws).ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    },
                    ..Default::default()
                },
            },
        ))
    }
}

fn adjustment_for_horizon(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Result<Arc<[LaggedColumn]>, CausalError> {
    let outcome_offset =
        i32::try_from(entry.horizon.saturating_sub(1)).map_err(|_| CausalError::Compile {
            message: "temporal mediation horizon exceeds supported offset range".into(),
        })?;
    entry
        .estimand
        .adjustment_set
        .iter()
        .map(|&dense| {
            let key = entry
                .indexer
                .key_of(dense.raw())
                .map_err(|error| CausalError::Compile { message: error.to_string() })?;
            let lag =
                outcome_offset.checked_sub(key.offset).ok_or_else(|| CausalError::Compile {
                    message: "temporal mediation adjustment lag overflowed".into(),
                })?;
            if lag < 0 {
                return Err(CausalError::Compile {
                    message: "temporal mediation adjustment requires future information".into(),
                });
            }
            Ok(LaggedColumn {
                variable: key.variable,
                lag: antecedent_core::Lag::from_raw(u32::try_from(lag).map_err(|_| {
                    CausalError::Compile {
                        message: "temporal mediation lag exceeds supported size".into(),
                    }
                })?),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Into::into)
}

fn config_seed(
    config: &BayesianConfig,
    ctx: &ExecutionContext,
    horizon: u32,
    mechanism: usize,
) -> u64 {
    ctx.rng
        .master_seed()
        .wrapping_add(config.n_draws as u64)
        .wrapping_add(u64::from(horizon).wrapping_mul(0x10001))
        .wrapping_add((mechanism as u64).wrapping_mul(0xBA71))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AcceptedGraph, Study};
    use antecedent_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, MediationContrast, RoleHint, SmallRoleSet,
        ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView, TimeIndex,
        ValidityBitmap,
    };
    use antecedent_graph::ensure_lagged;

    fn execute_checked(
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: antecedent_core::MediationQuery,
        inference: InferenceMode,
        validation: RefuteSuite,
        bootstrap_replicates: u32,
    ) -> crate::StudyResult {
        let ctx = ExecutionContext::for_tests(8821);
        let study = Study::series(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Mediation(query.clone()))
            .inference(inference)
            .refute(validation)
            .bootstrap_replicates(bootstrap_replicates)
            .build()
            .unwrap();
        let physical = study.compile(&ctx).unwrap();
        let estimator = if matches!(study.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::BayesianTemporalMediation
        } else {
            EstimatorId::TemporalMediation
        };
        let cache = crate::analysis::prepared::identify_temporal_mediation_horizons(
            graph, &query, estimator,
        )
        .unwrap();
        CheckedTemporalMediationOperation::checked(&study, data, &physical, &cache)
            .unwrap()
            .execute(data, &ctx)
            .unwrap()
    }

    const N: usize = 241;
    const MEDIATED_TRUTH: f64 = 0.8 * 0.55;

    fn fixture() -> (TimeSeriesData, TemporalDag) {
        let mut schema = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("m", RoleHint::Context),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
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
        let z: Vec<f64> =
            (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
        let t: Vec<f64> =
            (0..N).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
        let mut m = vec![0.0; N];
        let mut y = vec![1.0; N];
        for i in 1..N {
            let noise = (i % 5) as f64 - 2.0;
            m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
            y[i] = 1.0 + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
        }
        let columns = [t, m, y, z]
            .into_iter()
            .enumerate()
            .map(|(id, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(id as u32),
                        Arc::from(values),
                        ValidityBitmap::all_valid(N),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let t0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let z1 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        for (a, b) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
            graph.insert_directed(a, b).unwrap();
        }
        (data, graph)
    }

    #[test]
    fn checked_operation_executes_every_fixed_temporal_dag_license_coordinate() {
        let (data, graph) = fixture();
        let ctx = ExecutionContext::for_tests(817);
        for accepted in [false, true] {
            for (inference, bayesian) in [
                (InferenceMode::Frequentist, false),
                (
                    InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(512).prior_scale(1_000.0),
                    ),
                    true,
                ),
            ] {
                for validation in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    let query = antecedent_core::MediationQuery::binary(
                        VariableId::from_raw(0),
                        VariableId::from_raw(2),
                        [VariableId::from_raw(1)],
                        MediationContrast::Mediated,
                    )
                    .with_horizons([1])
                    .unwrap();
                    let base = Study::series(data.clone());
                    let base = if accepted {
                        base.graph(AcceptedGraph::temporal_dag(graph.clone()))
                    } else {
                        base.graph(graph.clone())
                    };
                    let study = base
                        .query(CausalQuery::Mediation(query.clone()))
                        .inference(inference.clone())
                        .refute(validation)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap();
                    let physical = study.compile(&ctx).unwrap();
                    let estimator = if bayesian {
                        EstimatorId::BayesianTemporalMediation
                    } else {
                        EstimatorId::TemporalMediation
                    };
                    let cache = crate::analysis::prepared::identify_temporal_mediation_horizons(
                        &graph, &query, estimator,
                    )
                    .unwrap();
                    let operation = CheckedTemporalMediationOperation::checked(
                        &study, &data, &physical, &cache,
                    )
                    .unwrap();
                    let (retained_query, _, retained_validation, horizons) = operation.inspect();
                    assert_eq!(retained_query, &query);
                    assert_eq!(retained_validation, validation);
                    assert_eq!(horizons, vec![1]);
                    let result = operation.execute(&data, &ctx).unwrap();
                    assert!(
                        (result.estimate.ate - MEDIATED_TRUTH).abs() < 0.12,
                        "accepted={accepted} bayesian={bayesian} validation={validation:?}: {}",
                        result.estimate.ate
                    );
                    let grid = result
                        .mediation_grid
                        .as_ref()
                        .expect("pointwise horizon contract retained");
                    assert_eq!(grid.slices.len(), 1);
                    assert!(!grid.joint_posterior);
                    assert_eq!(result.posterior.is_some(), bayesian);
                    if bayesian {
                        let posterior = result.posterior.as_ref().unwrap();
                        let effect = posterior.effect_column().unwrap();
                        let mut draws = posterior.draws.column(effect).unwrap().to_vec();
                        draws.sort_by(f64::total_cmp);
                        let q = |p: f64| draws[((draws.len() - 1) as f64 * p).round() as usize];
                        assert!(q(0.025) <= MEDIATED_TRUTH && MEDIATED_TRUTH <= q(0.975));
                    }
                    assert_eq!(result.refutations.is_empty(), validation == RefuteSuite::None);
                    let rebound = operation.rebind(&data).unwrap();
                    assert_eq!(rebound.inspect().0, &query);
                }
            }
        }
    }

    #[test]
    fn bayesian_natural_contrasts_retain_the_linear_alias_assumption_and_diagnostic() {
        let (data, graph) = fixture();
        for contrast in [MediationContrast::NaturalDirect, MediationContrast::NaturalIndirect] {
            let mut query = antecedent_core::MediationQuery::binary(
                VariableId::from_raw(0),
                VariableId::from_raw(2),
                [VariableId::from_raw(1)],
                contrast,
            )
            .with_horizons([1])
            .unwrap();
            query.contrast = contrast;
            let result = execute_checked(
                &data,
                &graph,
                query,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96)),
                RefuteSuite::None,
                0,
            );
            assert!(result.posterior.as_ref().unwrap().assumptions.entries.iter().any(|record| {
                matches!(&record.assumption, antecedent_core::Assumption::ParametricRestriction(item)
                    if item.id.as_ref() == "mediation.linear_no_interaction")
            }));
            assert!(
                result.diagnostics.iter().any(|diagnostic| {
                    diagnostic.code.as_ref() == "estimate.mediation.bayesian"
                })
            );
        }
    }

    #[test]
    fn frequentist_bootstrap_publishes_mediation_block_semantics_and_short_series_warning() {
        let (data, graph) = fixture();
        let columns = ["t", "m", "y", "z"]
            .map(|name| data.schema().id_of(name).unwrap())
            .map(|id| data.float64_values(id).unwrap());
        let short = TimeSeriesData::from_f64_columns(
            [
                ("t", &columns[0][..31]),
                ("m", &columns[1][..31]),
                ("y", &columns[2][..31]),
                ("z", &columns[3][..31]),
            ],
            1,
        )
        .unwrap();
        let query = antecedent_core::MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons([1])
        .unwrap();
        let result = execute_checked(
            &short,
            &graph,
            query,
            InferenceMode::Frequentist,
            RefuteSuite::None,
            64,
        );
        assert_eq!(
            result.estimate.block_family,
            Some(antecedent_estimate::CircularBlockFamily::Mediation)
        );
        let interval = result.primary_interval_binding(false);
        assert_eq!(interval.method, antecedent_core::IntervalMethod::CircularBlockSe);
        assert_eq!(interval.dependence, "circular_block:mediation");
        assert!(result.diagnostics.iter().any(|diagnostic| {
            diagnostic.code.as_ref() == "estimate.temporal.circular_block_se.short_series"
        }));
    }

    #[test]
    fn multi_horizon_refutations_are_scoped_and_parent_draw_count_is_suppressed() {
        let (data, graph) = fixture();
        let query = antecedent_core::MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons([1, 2])
        .unwrap();
        let result = execute_checked(
            &data,
            &graph,
            query,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96)),
            RefuteSuite::Cheap,
            0,
        );
        assert!(result.posterior.is_none());
        assert_eq!(result.performance.n_draws, None);
        assert!(!result.refutations.is_empty());
        assert!(result.refutations.iter().all(|report| {
            report.refuter.starts_with("horizon.1.") || report.refuter.starts_with("horizon.2.")
        }));
    }
}
