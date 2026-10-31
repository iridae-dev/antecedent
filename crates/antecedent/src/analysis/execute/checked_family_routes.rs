//! Concrete retained operations for class attribution and temporal mixed graphs.
//!
//! Each operation owns a checked scientific target and calls its family executor
//! directly, preserving the class/window estimator and its uncertainty semantics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{
    CausalError, CausalQuery, CheckedScientificRoute, ExecutionContext, PhysicalExecutionPlan,
    Study, StudyResult, TabularData, TimeSeriesData,
};
use crate::analysis::lowering::RouteCell;

#[derive(Clone, Debug)]
pub(crate) struct CheckedClassAttributionOperation {
    target: CheckedScientificRoute,
}

impl CheckedClassAttributionOperation {
    pub(crate) fn checked(
        study: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
        cell: RouteCell,
    ) -> Result<Self, CausalError> {
        if !matches!(cell, RouteCell::GcmClass | RouteCell::GcmTemporalClass) {
            return Err(family_mismatch());
        }
        Ok(Self { target: CheckedScientificRoute::checked(study, plan, Some(cell))? })
    }

    pub(crate) fn require_program(&self, plan: &PhysicalExecutionPlan) -> Result<(), CausalError> {
        self.target.require_program(plan)
    }

    pub(crate) fn execute_checked_class_attribution_table(
        &self,
        study: &Study,
        data: &TabularData,
        plan: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.target.require(study, plan)?;
        study.execute_class_attribution(data, plan, ctx)
    }

    pub(crate) fn execute_checked_class_attribution_series(
        &self,
        study: &Study,
        data: &TimeSeriesData,
        plan: &Arc<PhysicalExecutionPlan>,
        provenance: Option<&super::TemporalRowProvenance>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.target.require(study, plan)?;
        study.execute_temporal_class_attribution(data, plan, provenance, ctx)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CheckedTemporalAdmgOperation {
    target: CheckedScientificRoute,
}

impl CheckedTemporalAdmgOperation {
    pub(crate) fn checked(
        study: &Study,
        plan: &Arc<PhysicalExecutionPlan>,
        cell: Option<RouteCell>,
    ) -> Result<Self, CausalError> {
        let admitted = match cell {
            Some(
                RouteCell::TemporalEffectAdmg
                | RouteCell::TemporalResponseAdmg
                | RouteCell::TemporalMediationAdmg
                | RouteCell::TemporalMediationPag,
            ) => true,
            None => study
                .graph_posterior
                .as_ref()
                .is_some_and(antecedent_discovery::GraphPosterior::is_lagged_temporal_admg),
            _ => false,
        };
        if !admitted {
            return Err(family_mismatch());
        }
        Ok(Self { target: CheckedScientificRoute::checked(study, plan, cell)? })
    }

    pub(crate) fn require_program(&self, plan: &PhysicalExecutionPlan) -> Result<(), CausalError> {
        self.target.require_program(plan)
    }

    pub(crate) fn execute_checked_temporal_admg(
        &self,
        study: &Study,
        data: &TimeSeriesData,
        plan: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.target.require(study, plan)?;
        if let Some(posterior) = study.graph_posterior.as_ref() {
            return match &study.query {
                CausalQuery::TemporalEffect(query) => {
                    study.execute_temporal_admg_graph_posterior(data, posterior, query, plan, ctx)
                }
                CausalQuery::Mediation(query) => study
                    .execute_temporal_admg_graph_posterior_mediation(
                        data, posterior, query, plan, ctx,
                    ),
                _ => Err(family_mismatch()),
            };
        }
        match &study.query {
            CausalQuery::TemporalEffect(query) => {
                study.execute_temporal_admg_contrast(data, query, plan, ctx)
            }
            CausalQuery::Response(query) => {
                study.execute_temporal_admg_derivative(data, query, plan, ctx)
            }
            CausalQuery::Mediation(query)
                if study.graph.class() == crate::GraphClass::TemporalPag =>
            {
                study.execute_temporal_pag_mediation(data, query, plan, ctx)
            }
            CausalQuery::Mediation(query) => {
                study.execute_temporal_admg_mediation(data, query, plan, ctx)
            }
            _ => Err(family_mismatch()),
        }
    }
}

fn family_mismatch() -> CausalError {
    crate::compile_reason!(
        "route_not_lowered",
        "the retained family operation does not serve this scientific route"
    )
}
