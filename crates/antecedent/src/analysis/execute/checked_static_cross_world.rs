//! Retained static cross-world execution over mixed graphs and structure classes.
//!
//! The operation freezes its query, accepted structure and exact lowered route.
//! Its click invokes the scientific family executor directly; it does not re-enter
//! the public builder, compile a replacement plan or run the generic dispatcher.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::mediation_mixed::ClassStructure;
use super::{
    CausalError, CausalQuery, CheckedScientificRoute, ExecutionContext, GraphRef,
    PathSpecificMixedSource, PhysicalExecutionPlan, Study, StudyResult, TabularData,
};
use crate::analysis::lowering::RouteCell;

#[derive(Clone, Copy, Debug)]
enum StaticCrossWorldCell {
    CounterfactualClass,
    PathSpecificMixed,
    PathSpecificCpdag,
    PathSpecificPag,
}

impl StaticCrossWorldCell {
    fn checked(cell: RouteCell) -> Result<Self, CausalError> {
        match cell {
            RouteCell::CounterfactualClass => Ok(Self::CounterfactualClass),
            RouteCell::PathSpecificMixed => Ok(Self::PathSpecificMixed),
            RouteCell::PathSpecificCpdag => Ok(Self::PathSpecificCpdag),
            RouteCell::PathSpecificPag => Ok(Self::PathSpecificPag),
            _ => Err(crate::compile_reason!(
                "route_not_lowered",
                "the cell is not a static mixed/class cross-world operation"
            )),
        }
    }
}

#[derive(Clone, Debug)]
struct RetainedFunctional {
    fitter: antecedent_estimate::FunctionalEffect,
    prepared: antecedent_estimate::PreparedFunctionalEffect,
    storage: [u8; 32],
}

/// A query and structure sealed to the exact static cross-world program they execute.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticCrossWorldOperation {
    target: CheckedScientificRoute,
    cell: StaticCrossWorldCell,
    functional: Option<Arc<RetainedFunctional>>,
}

impl CheckedStaticCrossWorldOperation {
    pub(crate) fn checked(
        study: &Study,
        physical: &Arc<PhysicalExecutionPlan>,
        cell: RouteCell,
    ) -> Result<Self, CausalError> {
        let functional = match (&study.query, &study.data, study.identification_cache.as_ref()) {
            (
                CausalQuery::PathSpecific(_) | CausalQuery::NestedCounterfactual(_),
                crate::analysis::builder::DataInput::Tabular(data),
                Some(cache),
            ) if cell == RouteCell::PathSpecificMixed => {
                let source = match &study.query {
                    CausalQuery::PathSpecific(query) => {
                        PathSpecificMixedSource::PathSpecific(query)
                    }
                    CausalQuery::NestedCounterfactual(query) => {
                        PathSpecificMixedSource::Nested(query)
                    }
                    _ => return Err(Self::mismatch()),
                };
                let (_, _, _, roles) = super::static_path::path_specific_source_parts(source);
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                fitter.bootstrap_replicates = study.bootstrap_replicates;
                (fitter.row_resampling, fitter.sampling_design) =
                    study.functional_row_resampling()?;
                let prepared = fitter.prepare(
                    data,
                    &cache.estimand,
                    &cache.identification.arena,
                    cache.identification.required_assumptions.clone(),
                    &roles,
                )?;
                Some(Arc::new(RetainedFunctional {
                    fitter,
                    prepared,
                    storage: data.storage().content_digest(),
                }))
            }
            _ => None,
        };
        Ok(Self {
            target: CheckedScientificRoute::checked(study, physical, Some(cell))?,
            cell: StaticCrossWorldCell::checked(cell)?,
            functional,
        })
    }

    pub(crate) fn require_program(&self, plan: &PhysicalExecutionPlan) -> Result<(), CausalError> {
        self.target.require_program(plan)
    }

    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let functional = self
            .functional
            .as_ref()
            .map(|retained| {
                Ok::<_, CausalError>(Arc::new(RetainedFunctional {
                    fitter: retained.fitter.clone(),
                    prepared: retained.prepared.rebind_checked(data)?,
                    storage: data.storage().content_digest(),
                }))
            })
            .transpose()?;
        Ok(Self { target: self.target.clone(), cell: self.cell, functional })
    }

    pub(crate) fn functional_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        let Some(retained) = self.functional.as_ref() else { return Ok(None) };
        if retained.storage == data.storage().content_digest() {
            return retained.prepared.factor_snapshot().map(Some).map_err(CausalError::from);
        }
        retained
            .prepared
            .rebind_checked(data)?
            .factor_snapshot()
            .map(Some)
            .map_err(CausalError::from)
    }

    /// Run this family's scientific executor on compatible replacement rows.
    pub(crate) fn execute_checked_static_cross_world(
        &self,
        study: &Study,
        data: &TabularData,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.target.require(study, physical)?;
        if let (StaticCrossWorldCell::CounterfactualClass, CausalQuery::Counterfactual(query)) =
            (self.cell, &study.query)
        {
            return study.execute_counterfactual_class(data, query, physical, ctx);
        }
        // A mediation decomposition shares its structural cell with natural
        // path effects, but keeps its joint component estimator and uncertainty.
        if let CausalQuery::Mediation(query) = &study.query {
            return match (self.cell, study.graph.graph_ref()) {
                (StaticCrossWorldCell::PathSpecificMixed, GraphRef::Admg(graph)) => {
                    study.execute_mediation_mixed(data, graph, query, physical, ctx)
                }
                (StaticCrossWorldCell::PathSpecificMixed, GraphRef::TemporalAdmg(graph)) => {
                    let projected = super::temporal_admg_projection(graph)?;
                    study.execute_mediation_mixed(data, &projected, query, physical, ctx)
                }
                (StaticCrossWorldCell::PathSpecificCpdag, GraphRef::Cpdag(graph)) => study
                    .execute_mediation_class(
                        data,
                        ClassStructure::Cpdag(graph),
                        query,
                        physical,
                        ctx,
                    ),
                (StaticCrossWorldCell::PathSpecificPag, GraphRef::Pag(graph)) => study
                    .execute_mediation_class(
                        data,
                        ClassStructure::Pag(graph),
                        query,
                        physical,
                        ctx,
                    ),
                _ => Err(Self::mismatch()),
            };
        }
        let source = match &study.query {
            CausalQuery::PathSpecific(query) => PathSpecificMixedSource::PathSpecific(query),
            CausalQuery::NestedCounterfactual(query) => PathSpecificMixedSource::Nested(query),
            _ => return Err(Self::mismatch()),
        };
        match (self.cell, study.graph.graph_ref()) {
            (StaticCrossWorldCell::PathSpecificMixed, GraphRef::Admg(graph)) => study
                .execute_path_specific_mixed_prepared(
                    data,
                    graph,
                    source,
                    physical,
                    ctx,
                    self.functional.as_ref().map(|held| (&held.fitter, &held.prepared)),
                ),
            (StaticCrossWorldCell::PathSpecificMixed, GraphRef::TemporalAdmg(graph)) => {
                let projected = super::temporal_admg_projection(graph)?;
                study.execute_path_specific_mixed_prepared(
                    data,
                    &projected,
                    source,
                    physical,
                    ctx,
                    self.functional.as_ref().map(|held| (&held.fitter, &held.prepared)),
                )
            }
            (StaticCrossWorldCell::PathSpecificCpdag, GraphRef::Cpdag(graph)) => {
                study.execute_class_path_specific(data, graph, source, physical, ctx)
            }
            (StaticCrossWorldCell::PathSpecificPag, GraphRef::Pag(graph)) => {
                study.execute_pag_path_specific(data, graph, source, physical, ctx)
            }
            _ => Err(Self::mismatch()),
        }
    }

    fn mismatch() -> CausalError {
        crate::compile_reason!(
            "route_not_lowered",
            "the retained static cross-world cell disagrees with its query or graph class"
        )
    }
}
