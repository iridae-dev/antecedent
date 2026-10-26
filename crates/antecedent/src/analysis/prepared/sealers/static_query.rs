//! Sealers of the static causal-query routes: counterfactual, mediation and
//! attribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::accepted::GraphClass;
use crate::analysis::builder::DataInput;
use crate::analysis::prepared::PreparedExecution;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use antecedent_core::CausalQuery;
use std::sync::Arc;

pub(super) fn bayesian_static_mediation(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (
        cx.static_mediation_dag(),
        analysis.static_cache(),
        &analysis.inference,
    ) {
        (Some(data), Some(cache), InferenceMode::Bayesian(_)) => Some(Box::new(
            crate::analysis::execute::CheckedBayesianStaticMediationOperation::checked(
                analysis, data, plan, cache, cx.ctx,
            )?,
        )),
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::BayesianStaticMediation))
}

pub(super) fn static_mediation(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match (cx.static_mediation_dag(), analysis.static_cache()) {
        (Some(data), Some(cache))
            if matches!(analysis.inference, InferenceMode::Frequentist)
                && matches!(
                    analysis.structure_source,
                    crate::support::StructureSource::Explicit
                        | crate::support::StructureSource::Accepted
                )
                && analysis.custom_validators.is_empty() =>
        {
            Some(crate::analysis::execute::CheckedStaticMediationOperation::checked(
                analysis, data, plan, cache,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::StaticMediation))
}

pub(super) fn attribution(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.static_cache(),
        analysis.graph.class(),
        &analysis.inference,
    ) {
        (
            DataInput::Tabular(_),
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
            Some(cache),
            GraphClass::Dag,
            InferenceMode::Frequentist | InferenceMode::Bayesian(_),
        ) if analysis.structure_source == crate::support::StructureSource::Explicit => Some(
            crate::analysis::execute::CheckedAttributionOperation::checked(analysis, plan, cache)?,
        ),
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::AnomalyAttribution(_)
            | CausalQuery::ChangeAttribution(_)
            | CausalQuery::Attribute(_)
            | CausalQuery::UnitChange(_)
            | CausalQuery::MechanismChange(_)
            | CausalQuery::Simulate(_),
            Some(cache),
            GraphClass::TemporalDag,
            InferenceMode::Frequentist | InferenceMode::Bayesian(_),
        ) if analysis.structure_source == crate::support::StructureSource::Explicit => Some(
            crate::analysis::execute::CheckedAttributionOperation::checked(analysis, plan, cache)?,
        ),
        // The fitted-model routes and the mechanism- and unit-change studies run on
        // a supplied DAG or on one accepted from a compiled discovery, whose
        // provenance the contract carries.
        (
            DataInput::Tabular(_),
            CausalQuery::MechanismChange(_)
            | CausalQuery::UnitChange(_)
            | CausalQuery::Attribute(_)
            | CausalQuery::Simulate(_),
            Some(cache),
            GraphClass::Dag,
            _,
        ) if matches!(
            analysis.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) =>
        {
            Some(crate::analysis::execute::CheckedAttributionOperation::checked(
                analysis, plan, cache,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Attribution))
}

pub(super) fn counterfactual(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, counterfactual, .. } = *cx;
    let Some(operation) = counterfactual else {
        return Ok(None);
    };
    let cache = analysis.static_cache().ok_or_else(|| CausalError::Compile {
        message: "counterfactual preparation lost its identification product".into(),
    })?;
    let inference = analysis.inference.clone();
    let procedure = crate::analysis::execute::CounterfactualProcedure::for_inference(&inference)?
        .with_bootstrap(analysis.bootstrap_replicates, analysis.sampling_design)?;
    Ok(Some(PreparedExecution::Counterfactual(
        crate::analysis::execute::CheckedCounterfactualPlan::new(
            operation.clone(),
            Arc::clone(cache),
            true,
            inference,
            procedure,
            plan.clone(),
            crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
        )?,
    )))
}

#[allow(
    clippy::unnecessary_wraps,
    reason = "every sealer shares the table's fallible `SealOutcome` signature"
)]
pub(super) fn nested_counterfactual(cx: &SealContext<'_>) -> SealOutcome {
    Ok(cx
        .nested_counterfactual
        .map(|operation| PreparedExecution::NestedCounterfactual(operation.clone())))
}
