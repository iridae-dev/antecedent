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

pub(super) fn bayesian_static_mediation(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.identification_cache.as_deref(),
        analysis.graph.class(),
        &analysis.inference,
    ) {
        (
            DataInput::Tabular(data),
            CausalQuery::Mediation(_),
            Some(cache),
            GraphClass::Dag,
            InferenceMode::Bayesian(_),
        ) => Some(Box::new(
            crate::analysis::execute::CheckedBayesianStaticMediationOperation::checked(
                analysis, data, plan, cache,
            )?,
        )),
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::BayesianStaticMediation))
}

pub(super) fn static_mediation(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (
        &source.data,
        &source.query,
        analysis.identification_cache.as_deref(),
        analysis.graph.class(),
    ) {
        (DataInput::Tabular(data), CausalQuery::Mediation(_), Some(cache), GraphClass::Dag)
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
        analysis.identification_cache.as_deref(),
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
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
            Some(cache),
            GraphClass::TemporalDag,
            InferenceMode::Frequentist | InferenceMode::Bayesian(_),
        ) if analysis.structure_source == crate::support::StructureSource::Explicit => Some(
            crate::analysis::execute::CheckedAttributionOperation::checked(analysis, plan, cache)?,
        ),
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Attribution))
}

pub(super) fn counterfactual(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, counterfactual, .. } = *cx;
    let Some(operation) = counterfactual else {
        return Ok(None);
    };
    let cache = analysis.identification_cache.as_deref().ok_or_else(|| CausalError::Compile {
        message: "counterfactual preparation lost its identification product".into(),
    })?;
    let inference = analysis.inference.clone();
    let procedure = crate::analysis::execute::CounterfactualProcedure::for_inference(&inference)?;
    Ok(Some(PreparedExecution::Counterfactual(
        crate::analysis::execute::CheckedCounterfactualPlan::new(
            operation.clone(),
            cache.identification.clone(),
            cache.estimand.clone(),
            true,
            inference,
            procedure,
            plan.clone(),
            crate::analysis::execute::IdentifiedResultContext::from_study(analysis),
        )?,
    )))
}

pub(super) fn nested_counterfactual(cx: &SealContext<'_>) -> SealOutcome {
    Ok(cx
        .nested_counterfactual
        .map(|operation| PreparedExecution::NestedCounterfactual(operation.clone())))
}
