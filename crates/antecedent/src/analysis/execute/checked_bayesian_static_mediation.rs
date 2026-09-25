//! Sealed Bayesian execution for static DAG natural mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_expr::{FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable};

/// Frozen mediation proof, executable functional, mechanism prior configuration,
/// and validation choice for Bayesian static DAG mediation.
#[derive(Clone)]
pub(crate) struct CheckedBayesianStaticMediationOperation {
    graph: Dag,
    query: antecedent_core::MediationQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    program: FunctionalProgram,
    schema: antecedent_core::CausalSchema,
    design_roles: Arc<[VariableId]>,
    source_rows: Arc<[u32]>,
    config: BayesianConfig,
    validation: RefuteSuite,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedBayesianStaticMediationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedBayesianStaticMediationOperation")
            .field("query", &self.query)
            .field("identifier", &IdentifierId::PathSpecificNatural)
            .field("estimator", &EstimatorId::StaticMediationLinear)
            .field("backend", &self.config.backend)
            .field("draws", &self.config.n_draws)
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

impl CheckedBayesianStaticMediationOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        cache: &crate::analysis::prepared::CachedStaticIdentification,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked Bayesian mediation requires a MediationEffect target".into(),
            });
        };
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked Bayesian static mediation requires a supplied DAG",
        })?;
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let InferenceMode::Bayesian(config) = &study.inference else {
            return Err(CausalError::Compile {
                message: "checked Bayesian mediation requires a Bayesian configuration".into(),
            });
        };
        if config.prior.is_some() || config.external_compose.is_some() {
            return Err(CausalError::Unsupported {
                message: "Bayesian mediation supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
            });
        }
        if config.prior_artifact.is_some() && config.prior_mapping.is_none() {
            return Err(CausalError::Unsupported {
                message: "Bayesian mediation prior artifacts require an explicit coefficient mapping",
            });
        }
        if !matches!(
            study.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) || !matches!(study.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            || !study.custom_validators.is_empty()
            || cache.identification.query != study.query
            || !matches!(
                cache.identification.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
            )
            || !cache.identification.estimands.iter().any(|candidate| {
                candidate.functional == cache.estimand.functional
                    && candidate.method == cache.estimand.method
                    && candidate.adjustment_set == cache.estimand.adjustment_set
                    && candidate.mediators == cache.estimand.mediators
            })
            || physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::PathSpecificNatural.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::StaticMediationLinear.as_str())
        {
            return Err(CausalError::Compile {
                message: "Bayesian mediation proof, procedure, structure, or validation changed after preparation".into(),
            });
        }
        let verified =
            identify_static_query(IdentifierId::PathSpecificNatural, graph, &study.query)?;
        let (verified, verified_estimand) =
            select_claim(verified, EstimatorId::StaticMediationLinear)?;
        if verified.query != cache.identification.query
            || verified_estimand.functional != cache.estimand.functional
            || verified_estimand.method != cache.estimand.method
            || verified_estimand.adjustment_set != cache.estimand.adjustment_set
            || verified_estimand.mediators != cache.estimand.mediators
        {
            return Err(CausalError::Compile {
                message: "Bayesian mediation proof is not justified by its retained graph".into(),
            });
        }

        let program_schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        let program = FunctionalProgram::new(
            cache.identification.arena.clone(),
            program_schema,
            cache.estimand.functional,
            cache.estimand.functional,
            ProgramLimits::default(),
        )
        .map_err(|e| CausalError::Compile {
            message: format!("Bayesian mediation functional program: {e}"),
        })?;
        let mut roles = vec![query.treatment, query.outcome];
        roles.extend(query.mediators.iter().copied());
        roles.extend(cache.estimand.adjustment_set.iter().copied());
        roles.sort_unstable();
        roles.dedup();
        let source_rows = bind_mediation_rows(data, &roles)?;
        Ok(Self {
            graph: graph.clone(),
            query: query.clone(),
            identification: cache.identification.clone(),
            estimand: cache.estimand.clone(),
            physical: physical.clone(),
            program,
            schema: data.schema().clone(),
            design_roles: roles.into(),
            source_rows,
            config: config.clone(),
            validation: study.refute,
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn inspect(
        &self,
    ) -> (&antecedent_core::MediationQuery, &IdentifiedEstimand, &FunctionalProgram, RefuteSuite)
    {
        (&self.query, &self.estimand, &self.program, self.validation)
    }

    pub(crate) fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        if data.schema() != &self.schema
            || self.program.mapping().source != self.estimand.functional
        {
            return Err(CausalError::Compile {
                message: "Bayesian mediation refresh changed its semantic schema or target".into(),
            });
        }
        let mut next = self.clone();
        next.source_rows = bind_mediation_rows(data, &self.design_roles)?;
        Ok(next)
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let bound = self.rebind(data)?;
        let started = Instant::now();
        let decoded = crate::inference::decode_prior_hydrate_source(&bound.config)?;
        let bridge = decoded.as_ref().map(|decoded| antecedent_estimate::MediationPriorBridge {
            mapping: &decoded.mapping,
            quantities: &decoded.quantities,
            mean: &decoded.mean,
            sd: &decoded.sd,
            source_contrast: decoded.source_contrast,
        });
        let estimator = bayesian_gcomp(&bound.config, ctx);
        let (mediation, posterior) = antecedent_estimate::estimate_static_mediation_bayesian(
            data,
            &bound.graph,
            &bound.query,
            bound.identification.required_assumptions.clone(),
            &[],
            &estimator,
            bound.identification.status,
            bridge,
            ctx,
        )?;
        let estimate = mediation.effect.clone();
        let refutations = if bound.validation == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_static_mediation(
                data,
                &bound.graph,
                &bound.query,
                &mediation,
                bound.validation == RefuteSuite::Full,
                ctx,
            )?
        };
        let draws = u32::try_from(posterior.draws.n_draws).ok();
        Ok(finish_identified_execute_with_context(
            &bound.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &bound.physical,
                identification: bound.identification.clone(),
                estimand: bound.estimand.clone(),
                bootstrap_replicates_ok: None,
                cancelled: ctx.cancellation.is_cancelled(),
                early_stopped: posterior.early_stopped,
                estimate,
                identifier_id: IdentifierId::PathSpecificNatural,
                estimator_id: EstimatorId::StaticMediationLinear,
                treatment: bound.query.treatment,
                outcome: bound.query.outcome,
                identify_cached: true,
                extra_diagnostics: Vec::new(),
                refutations,
                distribution: None,
                mediation: Some(mediation),
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                extras: IdentifiedExecuteExtras {
                    posterior: Some(posterior),
                    n_draws: draws,
                    ..Default::default()
                },
            },
        ))
    }
}

fn bind_mediation_rows(
    data: &TabularData,
    roles: &[VariableId],
) -> Result<Arc<[u32]>, CausalError> {
    data.complete_case_mask(roles)?
        .iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .map(|(row, _)| {
            u32::try_from(row).map_err(|_| CausalError::Compile {
                message: "Bayesian mediation row index exceeds supported size".into(),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AcceptedGraph, Study};
    use antecedent_core::{Intervention, MediationContrast, MediationQuery, Value};

    fn fixture() -> (TabularData, Dag) {
        let (mut a, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..320 {
            let treatment = ((i as f64) * 0.71).sin();
            let mediator = 2.0 * treatment + ((i as f64) * 1.13).cos();
            let outcome = 3.0 * treatment + 4.0 * mediator + 0.1 * ((i as f64) * 0.31).sin();
            a.push(treatment);
            m.push(mediator);
            y.push(outcome);
        }
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (from, to) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        (data, graph)
    }

    #[test]
    fn checked_bayesian_mediation_retains_and_executes_all_six_supported_coordinates() {
        let (data, graph) = fixture();
        let ctx = ExecutionContext::for_tests(811);
        for accepted in [false, true] {
            for validation in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                for (contrast, truth) in [
                    (MediationContrast::NaturalDirect, 1.8),
                    (MediationContrast::NaturalIndirect, 4.8),
                    (MediationContrast::Total, 6.6),
                ] {
                    let mut query = MediationQuery::binary(
                        VariableId::from_raw(0),
                        VariableId::from_raw(2),
                        [VariableId::from_raw(1)],
                        contrast,
                    );
                    query.control = Intervention::set(query.treatment, Value::f64(0.2));
                    query.active = Intervention::set(query.treatment, Value::f64(0.8));
                    let base = Study::tabular(data.clone());
                    let base = if accepted {
                        base.graph(AcceptedGraph::from(graph.clone()))
                    } else {
                        base.graph(graph.clone())
                    };
                    let study = base
                        .query(CausalQuery::Mediation(query.clone()))
                        .inference(InferenceMode::Bayesian(
                            BayesianConfig::conjugate().n_draws(256).prior_scale(1_000.0),
                        ))
                        .refute(validation)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap();
                    let physical = study.compile(&ctx).unwrap();
                    let identified = identify_static_query(
                        IdentifierId::PathSpecificNatural,
                        &graph,
                        &study.query,
                    )
                    .unwrap();
                    let (identification, estimand) =
                        select_claim(identified, EstimatorId::StaticMediationLinear).unwrap();
                    let cache = crate::analysis::prepared::CachedStaticIdentification {
                        identification,
                        estimand,
                    };
                    let operation = CheckedBayesianStaticMediationOperation::checked(
                        &study, &data, &physical, &cache,
                    )
                    .unwrap();
                    assert_eq!(operation.inspect().0, &query);
                    assert_eq!(operation.inspect().3, validation);
                    let result = operation.execute(&data, &ctx).unwrap();
                    assert!((result.estimate.ate - truth).abs() < 0.18);
                    let posterior = result.posterior.as_ref().expect("posterior retained");
                    assert_eq!(posterior.draws.n_draws, 256);
                    let effect_col = posterior.effect_column().unwrap();
                    let mut draws = posterior.draws.column(effect_col).unwrap().to_vec();
                    draws.sort_by(f64::total_cmp);
                    let at = |q: f64| draws[((draws.len() - 1) as f64 * q).round() as usize];
                    assert!(at(0.05) <= truth && truth <= at(0.95));
                    assert_eq!(result.refutations.is_empty(), validation == RefuteSuite::None);
                    let rebound = operation.rebind(&data).unwrap();
                    assert_eq!(rebound.inspect().0, &query);
                    assert_eq!(rebound.inspect().3, validation);
                }
            }
        }
    }
}
