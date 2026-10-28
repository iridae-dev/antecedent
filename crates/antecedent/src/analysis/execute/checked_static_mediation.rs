//! Checked frequentist linear mediation on a supplied static DAG.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalQuery, Dag, EstimatorId, ExecutionContext, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId,
    InferenceMode, Instant, PhysicalExecutionPlan, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TabularData, VariableId, elapsed_ns, finish_identified_execute_with_context,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::proof_supplies_with_mediators;
use antecedent_expr::{FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable};

/// One frozen mediation target, identification, row design, and uncertainty procedure.
#[derive(Clone)]
pub(crate) struct CheckedStaticMediationOperation {
    graph: Dag,
    query: antecedent_core::MediationQuery,
    target: antecedent_core::ScientificTarget,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
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
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::Mediation(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "checked mediation requires a mediation target".into(),
            });
        };
        let graph = study.graph.as_dag().ok_or(CausalError::Unsupported {
            message: "checked static mediation requires a supplied DAG",
        })?;
        query.validate()?;
        if !matches!(study.settings.inference, InferenceMode::Frequentist)
            || !matches!(
                study.structure_source,
                crate::support::StructureSource::Explicit
                    | crate::support::StructureSource::Accepted
            )
            || !study.settings.refute.is_point_or_placebo()
            || !study.declared.custom_validators.is_empty()
            || cache.identification.query != study.query
            || !matches!(
                cache.identification.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
            )
            || !proof_supplies_with_mediators(&cache.identification, &cache.estimand)
            || !physical
                .binding()
                .names(IdentifierId::PathSpecificNatural, EstimatorId::StaticMediationLinear)
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
            target: theorem_mediation_target(
                query,
                data.schema(),
                &cache.identification.required_assumptions,
                antecedent_core::InferenceSelection::Frequentist,
                None,
            )?,
            cache: Arc::clone(cache),
            physical: physical.clone(),
            program,
            schema: data.schema().clone(),
            design_roles: roles.into(),
            source_rows,
            bootstrap_replicates: study.settings.bootstrap_replicates,
            refute: study.settings.refute,
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        if data.schema() != &self.schema
            || self.program.mapping().source != self.cache.estimand.functional
            || self.target.identity()
                != theorem_mediation_target(
                    &self.query,
                    &self.schema,
                    &self.cache.identification.required_assumptions,
                    antecedent_core::InferenceSelection::Frequentist,
                    None,
                )?
                .identity()
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
            &self.cache.estimand,
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
            bound.cache.identification.required_assumptions.clone(),
            // A multiple-imputation run states its own within-table replicates; they are not
            // forced through the study's global setting.
            ctx.imputation_bootstrap().unwrap_or(bound.bootstrap_replicates),
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
        let (replicates_ok, cancelled, early_stopped) = (
            estimate.uncertainty.bootstrap.replicates_ok,
            estimate.uncertainty.bootstrap.cancelled,
            estimate.uncertainty.bootstrap.early_stopped,
        );
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
            .bootstrap_replicates_ok(replicates_ok)
            .cancelled(cancelled)
            .early_stopped(early_stopped),
        ))
    }
}

// The natural-effect theorem target, built once by the lowering for the static and
// temporal mediation executors.
pub(super) use crate::analysis::lowering::theorem_mediation_target;

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

/// Read-only target, graph, and procedure of a checked static mediation plan.
#[derive(Clone, Debug)]
pub struct CheckedStaticMediationInfo {
    /// Complete contrast, mediators, intervention levels, and population.
    pub query: antecedent_core::MediationQuery,
    /// Scientific target retaining the natural-effect theorem's cross-world premises.
    pub scientific_target: antecedent_core::ScientificTarget,
    /// Prepare-time identifier.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Identified adjustment roles used by the model.
    pub adjustment_set: std::sync::Arc<[antecedent_core::VariableId]>,
    /// Source and executable functional root identifiers.
    pub functional_roots: (u32, u32),
    /// Canonical supplied DAG edges.
    pub graph_edges: std::sync::Arc<[(u32, u32)]>,
    /// Bootstrap replicate budget, including zero for point-only execution.
    pub bootstrap_replicates: u32,
    /// Prepare-time validation suite.
    pub validation: crate::analysis::builder::RefuteSuite,
}

impl Inspect for CheckedStaticMediationOperation {
    type Info = CheckedStaticMediationInfo;

    fn info(&self) -> CheckedStaticMediationInfo {
        let (query, estimand, program, bootstrap_replicates, validation, graph) = self.inspect();
        let edges = graph.directed_edge_pairs_sorted();
        CheckedStaticMediationInfo {
            query: query.clone(),
            scientific_target: self.target.clone(),
            identifier: IdentifierId::PathSpecificNatural,
            estimator: EstimatorId::StaticMediationLinear,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            functional_roots: (program.mapping().source.raw(), program.mapping().executable.raw()),
            graph_edges: edges.into(),
            bootstrap_replicates,
            validation,
        }
    }
}
