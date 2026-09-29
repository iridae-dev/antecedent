//! Checked frequentist linear mediation on a supplied static DAG.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_expr::{FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable};

/// One frozen mediation target, identification, row design, and uncertainty procedure.
#[derive(Clone)]
pub(crate) struct CheckedStaticMediationOperation {
    graph: Dag,
    query: antecedent_core::MediationQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    program: FunctionalProgram,
    schema: antecedent_core::CausalSchema,
    design_roles: Arc<[VariableId]>,
    source_rows: Arc<[u32]>,
    bootstrap_replicates: u32,
    refute: RefuteSuite,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedStaticMediationOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedStaticMediationOperation")
            .field("query", &self.query)
            .field("identifier", &IdentifierId::PathSpecificNatural)
            .field("estimator", &EstimatorId::StaticMediationLinear)
            .field("design_roles", &self.design_roles)
            .field("source_rows", &self.source_rows.len())
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("refute", &self.refute)
            .finish_non_exhaustive()
    }
}

impl CheckedStaticMediationOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        cache: &crate::analysis::prepared::CachedStaticIdentification,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked mediation requires a mediation target".into(),
            });
        };
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked static mediation requires a supplied DAG",
        })?;
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if !matches!(study.inference, InferenceMode::Frequentist)
            || !matches!(
                study.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !matches!(
                study.refute,
                RefuteSuite::None
                    | RefuteSuite::Cheap
                    | RefuteSuite::PlaceboAndRcc
                    | RefuteSuite::Full
            )
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
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::PathSpecificNatural.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::StaticMediationLinear.as_str())
        {
            return Err(CausalError::Compile { message: "mediation identification or estimator lowering does not match the checked target".into() });
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
            message: format!("mediation functional program: {e}"),
        })?;
        let mut roles = vec![query.treatment, query.outcome];
        roles.extend(query.mediators.iter().copied());
        roles.extend(cache.estimand.adjustment_set.iter().copied());
        roles.sort_unstable();
        roles.dedup();
        let source_rows = bind_rows(data, &roles)?;
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
            bootstrap_replicates: study.bootstrap_replicates,
            refute: study.refute,
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        if data.schema() != &self.schema
            || self.program.mapping().source != self.estimand.functional
        {
            return Err(CausalError::Compile {
                message: "mediation refresh changed its semantic schema or target".into(),
            });
        }
        let mut next = self.clone();
        next.source_rows = bind_rows(data, &self.design_roles)?;
        Ok(next)
    }

    pub(crate) fn inspect(
        &self,
    ) -> (
        &antecedent_core::MediationQuery,
        &IdentifiedEstimand,
        &FunctionalProgram,
        u32,
        RefuteSuite,
        &Dag,
    ) {
        (
            &self.query,
            &self.estimand,
            &self.program,
            self.bootstrap_replicates,
            self.refute,
            &self.graph,
        )
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let bound = self.rebind(data)?;
        let started = Instant::now();
        let mediation = antecedent_estimate::estimate_static_mediation(
            data,
            &bound.graph,
            &bound.query,
            bound.identification.required_assumptions.clone(),
            bound.bootstrap_replicates,
            &[],
            ctx,
        )?;
        let estimate = mediation.effect.clone();
        let refutations = if bound.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_static_mediation(
                data,
                &bound.graph,
                &bound.query,
                &mediation,
                bound.refute == RefuteSuite::Full,
                ctx,
            )?
        };
        Ok(finish_identified_execute_with_context(
            &bound.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &bound.physical,
                identification: bound.identification.clone(),
                estimand: bound.estimand.clone(),
                bootstrap_replicates_ok: estimate.bootstrap_replicates_ok,
                cancelled: estimate.bootstrap_cancelled,
                early_stopped: estimate.bootstrap_early_stopped,
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
                extras: IdentifiedExecuteExtras::default(),
            },
        ))
    }
}

fn bind_rows(data: &TabularData, roles: &[VariableId]) -> Result<Arc<[u32]>, CausalError> {
    data.complete_case_mask(roles)?
        .iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .map(|(row, _)| {
            u32::try_from(row).map_err(|_| CausalError::Compile {
                message: "mediation row index exceeds supported size".into(),
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Into::into)
}
