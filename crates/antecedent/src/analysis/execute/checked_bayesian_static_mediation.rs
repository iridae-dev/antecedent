//! Sealed Bayesian execution for static DAG natural mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianConfig, CausalError, CausalQuery, CheckedStaticMediationInfo, Dag, EstimatorId,
    ExecutionContext, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId, InferenceMode, Instant,
    PhysicalExecutionPlan, RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData,
    VariableId, bayesian_gcomp, elapsed_ns, finish_identified_execute_with_context,
    identify_static_query, select_claim,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::{EstimandClaim, proof_supplies_with_mediators};
use antecedent_expr::{FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable};

/// Frozen mediation proof, executable functional, mechanism prior configuration,
/// and validation choice for Bayesian static DAG mediation.
#[derive(Clone)]
pub(crate) struct CheckedBayesianStaticMediationOperation {
    graph: Dag,
    query: antecedent_core::MediationQuery,
    target: antecedent_core::ScientificTarget,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked Bayesian mediation requires a MediationEffect target".into(),
            });
        };
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked Bayesian static mediation requires a supplied DAG",
        })?;
        query.validate()?;
        let InferenceMode::Bayesian(config) = &study.inference else {
            return Err(CausalError::Compile {
                message: "checked Bayesian mediation requires a Bayesian configuration".into(),
            });
        };
        ensure_mediation_prior_admitted(config, data, query)?;
        verify_mediation_proof(study, physical, cache, graph)?;

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
            target: super::checked_static_mediation::theorem_mediation_target(
                query,
                data.schema(),
                &cache.identification.required_assumptions,
                antecedent_core::InferenceSelection::Bayesian,
                None,
            )?,
            cache: Arc::clone(cache),
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
        (&self.query, &self.cache.estimand, &self.program, self.validation)
    }

    pub(crate) fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        if data.schema() != &self.schema
            || self.program.mapping().source != self.cache.estimand.functional
            || self.target.identity()
                != super::checked_static_mediation::theorem_mediation_target(
                    &self.query,
                    &self.schema,
                    &self.cache.identification.required_assumptions,
                    antecedent_core::InferenceSelection::Bayesian,
                    None,
                )?
                .identity()
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
        let (mediation, posterior) =
            antecedent_estimate::estimate_static_mediation_bayesian_outcome_prior(
                &antecedent_estimate::StaticMediationProblem {
                    data,
                    graph: &bound.graph,
                    query: &bound.query,
                    extra: &[],
                },
                bound.cache.identification.required_assumptions.clone(),
                &estimator,
                bound.cache.identification.status,
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
            IdentifiedExecuteFinish::new(
                &bound.physical,
                bound.cache.identification.clone(),
                bound.cache.estimand.clone(),
                estimate,
                Strategy::new(
                    IdentifierId::PathSpecificNatural,
                    EstimatorId::StaticMediationLinear,
                ),
                bound.query.treatment,
                bound.query.outcome,
            )
            .identify_cached(true)
            .refutations(refutations)
            .mediation(Some(mediation))
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .early_stopped(posterior.early_stopped)
            .extras(IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: draws,
                ..Default::default()
            }),
        ))
    }
}

/// The mechanism-prior configuration Bayesian mediation admits: isotropic priors, or a
/// prior artifact mapped to exactly the outcome mechanism's treatment slope.
fn ensure_mediation_prior_admitted(
    config: &BayesianConfig,
    data: &TabularData,
    query: &antecedent_core::MediationQuery,
) -> Result<(), CausalError> {
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
    if config.prior_artifact.is_some() {
        let treatment_name = data.schema().get(query.treatment).map_err(CausalError::rejected)?;
        let outcome_treatment_slope = format!("coef_{}", treatment_name.name);
        match config.prior_mapping.as_ref() {
            Some(antecedent_io::PriorMapping::IdenticalCoefficientSubspace) => {
                return Err(CausalError::Unsupported {
                    message: "Bayesian mediation prior transfer requires an outcome-mechanism functional or named treatment-slope mapping; identical coefficient-subspace mapping can bind both mechanisms",
                });
            }
            Some(antecedent_io::PriorMapping::NamedParameters { pairs })
                if pairs.len() != 1
                    || pairs[0].0.is_empty()
                    || pairs[0].1 != outcome_treatment_slope =>
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian mediation prior mapping is ambiguous; declare exactly one source quantity for the outcome-mechanism treatment slope",
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// The retained proof, procedure, structure, and validation still match the study, and the
/// natural-path identification re-derives from the retained graph.
fn verify_mediation_proof(
    study: &Study,
    physical: &PhysicalExecutionPlan,
    cache: &crate::analysis::prepared::CachedStaticIdentification,
    graph: &Dag,
) -> Result<(), CausalError> {
    if !matches!(
        study.structure_source,
        crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
    ) || !study.refute.is_point_or_placebo()
        || !study.custom_validators.is_empty()
        || cache.identification.query != study.query
        || !matches!(
            cache.identification.status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
        || !proof_supplies_with_mediators(&cache.identification, &cache.estimand)
        || physical.logical.query != study.query
        || !physical
            .binding()
            .names(IdentifierId::PathSpecificNatural, EstimatorId::StaticMediationLinear)
    {
        return Err(CausalError::Compile {
            message: "Bayesian mediation proof, procedure, structure, or validation changed after preparation".into(),
        });
    }
    let verified = identify_static_query(IdentifierId::PathSpecificNatural, graph, &study.query)?;
    let (verified, verified_estimand) = select_claim(verified, EstimatorId::StaticMediationLinear)?;
    if verified.query != cache.identification.query
        || !verified_estimand.same_claim_with_mediators(&cache.estimand)
    {
        return Err(CausalError::Compile {
            message: "Bayesian mediation proof is not justified by its retained graph".into(),
        });
    }
    Ok(())
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

impl Inspect for CheckedBayesianStaticMediationOperation {
    type Info = CheckedStaticMediationInfo;

    fn info(&self) -> CheckedStaticMediationInfo {
        let (query, estimand, program, validation) = self.inspect();
        let edges = self.graph().directed_edge_pairs_sorted();
        CheckedStaticMediationInfo {
            query: query.clone(),
            scientific_target: self.target.clone(),
            identifier: IdentifierId::PathSpecificNatural,
            estimator: EstimatorId::StaticMediationLinear,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            functional_roots: (program.mapping().source.raw(), program.mapping().executable.raw()),
            graph_edges: edges.into(),
            bootstrap_replicates: 0,
            validation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AcceptedGraph, Study};
    use antecedent_core::{Intervention, MediationContrast, MediationQuery, Value};
    use antecedent_graph::DenseNodeId;

    fn fixture() -> (TabularData, Dag) {
        let (mut a, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..320 {
            let treatment = (f64::from(i) * 0.71).sin();
            let mediator = 2.0 * treatment + (f64::from(i) * 1.13).cos();
            let outcome = 3.0 * treatment + 4.0 * mediator + 0.1 * (f64::from(i) * 0.31).sin();
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
                    let cache = Arc::new(crate::analysis::prepared::CachedStaticIdentification {
                        identification,
                        estimand,
                    });
                    let operation = CheckedBayesianStaticMediationOperation::checked(
                        &study,
                        &data,
                        &Arc::new(physical.clone()),
                        &cache,
                    )
                    .unwrap();
                    assert_eq!(operation.inspect().0, &query);
                    assert_eq!(operation.inspect().3, validation);
                    let info = operation.info();
                    assert!(matches!(
                        info.scientific_target.form(),
                        antecedent_core::TargetForm::Mediation(_)
                    ));
                    assert_eq!(
                        info.scientific_target.inference(),
                        antecedent_core::InferenceSelection::Bayesian
                    );
                    let mut changed = operation.clone();
                    changed.query.active = Intervention::set(query.treatment, Value::f64(0.9));
                    assert!(matches!(changed.rebind(&data), Err(CausalError::Compile { .. })));
                    let result = operation.execute(&data, &ctx).unwrap();
                    assert!((result.estimate.as_effect().unwrap().ate - truth).abs() < 0.18);
                    let posterior = result.posterior.as_ref().expect("posterior retained");
                    assert_eq!(posterior.draws.n_draws, 256);
                    let effect_col = posterior.effect_column().unwrap();
                    let mut draws = posterior.draws.column(effect_col).unwrap().to_vec();
                    draws.sort_by(f64::total_cmp);
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a rounded fraction of a small draw count indexes the sorted draws"
                    )]
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
