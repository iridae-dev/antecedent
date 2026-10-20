//! Checked fixed-TemporalDag mediation execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::temporal_mediation_core::{
    MediationInference, TemporalMediationInputs, fit_temporal_mediation,
    require_isotropic_mediation_prior,
};
use super::temporal_path::MediationHorizonClick;
use super::{
    Arc, BayesianConfig, CausalError, CausalQuery, EstimatorId, ExecutionContext,
    IdentificationResult, IdentifiedEstimand, IdentifiedResultContext, InferenceMode, Instant,
    PhysicalExecutionPlan, RefuteSuite, Study, StudyResult, TableView, TemporalIndexer,
    TimeSeriesData, elapsed_ns, finish_identified_execute_with_context, require_identified,
};
use crate::analysis::prepared::Inspect;
use antecedent_data::LaggedColumn;
use antecedent_graph::TemporalDag;

#[allow(
    clippy::large_enum_variant,
    reason = "the Bayesian configuration is declared once per study and moved by value; boxing would change every public pattern match"
)]
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
    horizons: Arc<[MediationHorizonClick]>,
    inference: CheckedMediationInference,
    validation: RefuteSuite,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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

/// The inference a checked mediation was prepared under, refusing an
/// estimator or prior that differs from the physical plan.
fn checked_mediation_inference(
    study: &Study,
    physical: &PhysicalExecutionPlan,
) -> Result<CheckedMediationInference, CausalError> {
    Ok(match &study.inference {
        InferenceMode::Frequentist => {
            if !physical.binding().estimator.is(EstimatorId::TemporalMediation) {
                return Err(CausalError::Compile {
                    message: "temporal mediation estimator differs from its checked physical plan"
                        .into(),
                });
            }
            CheckedMediationInference::Frequentist {
                bootstrap_replicates: study.bootstrap_replicates,
            }
        }
        InferenceMode::Bayesian(config) => {
            require_isotropic_mediation_prior(config)?;
            if !physical.binding().estimator.is(EstimatorId::BayesianTemporalMediation) {
                return Err(CausalError::Compile {
                    message: "Bayesian temporal mediation estimator differs from its checked physical plan".into(),
                });
            }
            CheckedMediationInference::Bayesian(config.clone())
        }
    })
}

/// Replays each requested horizon's cached proof against the recomputed one
/// and collects the identified horizons' mediation clicks.
fn replay_mediation_horizons(
    cache: &crate::analysis::prepared::CachedTemporalIdentification,
    verified: &crate::analysis::prepared::CachedTemporalIdentification,
    query: &antecedent_core::MediationQuery,
) -> Result<Vec<MediationHorizonClick>, CausalError> {
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
        horizons.push(MediationHorizonClick {
            horizon,
            identification: cached.identification.clone(),
            estimand: cached.estimand.clone(),
            indexer: cached.indexer.clone(),
            adjustment,
        });
    }
    Ok(horizons)
}

impl CheckedTemporalMediationOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TimeSeriesData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
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
        query.validate()?;
        if !matches!(
            study.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) || study.split.is_some()
            || study.graph_posterior.is_some()
            || !study.custom_validators.is_empty()
            || !study.refute.is_point()
            || physical.logical.query != study.query
            || physical.logical.record.plan_id.as_ref() != "temporal_mediation"
            || !physical.binding().identifier.is_wire("temporal.mediation")
        {
            return Err(CausalError::Compile {
                message: "temporal mediation structure, target, validation, or plan changed after preparation".into(),
            });
        }
        let inference = checked_mediation_inference(study, physical)?;
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
        let horizons = replay_mediation_horizons(cache, &verified, query)?;
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
        crate::graph_identity::same_temporal_dag_structure(&self.graph, graph)
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
        let started = Instant::now();
        let inference = match &bound.inference {
            CheckedMediationInference::Frequentist { bootstrap_replicates } => {
                MediationInference::Frequentist { bootstrap_replicates: *bootstrap_replicates }
            }
            CheckedMediationInference::Bayesian(config) => MediationInference::Bayesian(config),
        };
        let fit = fit_temporal_mediation(
            &TemporalMediationInputs {
                data,
                query: &bound.query,
                horizons: &bound.horizons,
                inference,
                validation: bound.validation,
                missing: bound.result_context.missing.as_ref(),
            },
            ctx,
        )?;
        let missing_data = fit.missing_data.clone();
        let args = fit.finish_args(
            &bound.physical,
            &bound.query,
            true,
            bound.result_context.graph_version,
            elapsed_ns(started),
        );
        let mut result = finish_identified_execute_with_context(&bound.result_context, None, args);
        result.missing_data = missing_data;
        Ok(result)
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
            let key = entry.indexer.key_of(dense.raw())?;
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

/// Read-only target and procedure for a checked temporal mediation family.
#[derive(Clone, Debug)]
pub struct CheckedTemporalMediationInfo {
    /// Exact mediation target and requested horizon family.
    pub query: antecedent_core::MediationQuery,
    /// Signature of the fixed `TemporalDag`.
    pub graph_signature: std::sync::Arc<str>,
    /// Validation suite selected at preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Horizon members retained in order.
    pub horizons: std::sync::Arc<[u32]>,
    /// Estimator family fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Graph-derived adjustment node keys retained at each horizon.
    pub adjustment_sets:
        std::sync::Arc<[(u32, std::sync::Arc<[antecedent_core::TemporalNodeKey]>)]>,
}

impl Inspect for CheckedTemporalMediationOperation {
    type Info = CheckedTemporalMediationInfo;

    fn info(&self) -> CheckedTemporalMediationInfo {
        let (query, graph, validation, horizons) = self.inspect();
        CheckedTemporalMediationInfo {
            query: query.clone(),
            graph_signature: Arc::from(
                crate::analysis::checked_temporal_effect::temporal_dag_signature(graph),
            ),
            validation,
            horizons: Arc::from(horizons),
            estimator: self.estimator(),
            adjustment_sets: Arc::from(
                self.horizon_contracts()
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AcceptedGraph, Study};
    use antecedent_core::VariableId;
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
        query: &antecedent_core::MediationQuery,
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
            graph, query, estimator,
        )
        .unwrap();
        CheckedTemporalMediationOperation::checked(&study, data, &Arc::new(physical), &cache)
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
                        VariableId::from_raw(u32::try_from(id).unwrap()),
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
                        &study,
                        &data,
                        &Arc::new(physical.clone()),
                        &cache,
                    )
                    .unwrap();
                    let (retained_query, _, retained_validation, horizons) = operation.inspect();
                    assert_eq!(retained_query, &query);
                    assert_eq!(retained_validation, validation);
                    assert_eq!(horizons, vec![1]);
                    let result = operation.execute(&data, &ctx).unwrap();
                    assert!(
                        (result.estimate.as_effect().unwrap().ate - MEDIATED_TRUTH).abs() < 0.12,
                        "accepted={accepted} bayesian={bayesian} validation={validation:?}: {}",
                        result.estimate.as_effect().unwrap().ate
                    );
                    let grid =
                        result.mediation_grid().expect("pointwise horizon contract retained");
                    assert_eq!(grid.slices.len(), 1);
                    assert!(!grid.joint_posterior);
                    assert_eq!(result.posterior.is_some(), bayesian);
                    if bayesian {
                        let posterior = result.posterior.as_ref().unwrap();
                        let effect = posterior.effect_column().unwrap();
                        let mut draws = posterior.draws.column(effect).unwrap().to_vec();
                        draws.sort_by(f64::total_cmp);
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            reason = "a rounded quantile position in [0, len) is a valid index"
                        )]
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
                &query,
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
            &query,
            InferenceMode::Frequentist,
            RefuteSuite::None,
            64,
        );
        assert_eq!(
            result.estimate.as_effect().unwrap().uncertainty.block.family,
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
            &query,
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
