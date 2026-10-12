//! Sealers of the randomized-design and interference routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::accepted::GraphClass;
use crate::analysis::builder::DataInput;
use crate::analysis::prepared::PreparedExecution;
use antecedent_core::CausalQuery;

pub(super) fn interference(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (DataInput::Tabular(data), CausalQuery::Interference(_), GraphClass::Dag)
            if analysis.structure_source == crate::support::StructureSource::Explicit
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedInterferenceOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Interference))
}

pub(super) fn randomized(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (
            DataInput::Tabular(data),
            CausalQuery::RandomizedEffect(_),
            GraphClass::RandomizedTrial,
        ) if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
            && analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedRandomizedOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Randomized))
}

pub(super) fn policy_value(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (DataInput::Tabular(data), CausalQuery::PolicyValue(_), GraphClass::RandomizedTrial)
            if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedPolicyValueOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::PolicyValue))
}

pub(super) fn continuous_dose(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (
            DataInput::Tabular(data),
            CausalQuery::ContinuousDoseResponse(_),
            GraphClass::RandomizedTrial,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedContinuousDoseOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::ContinuousDoseResponse))
}

pub(super) fn panel_did(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (DataInput::Tabular(data), CausalQuery::PanelDid(_), GraphClass::RandomizedTrial)
            if analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedPanelDidOperation::checked(analysis, data, plan)?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::PanelDid))
}

pub(super) fn synthetic_control(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (
            DataInput::Tabular(data),
            CausalQuery::SyntheticControl(_),
            GraphClass::RandomizedTrial,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedSyntheticControlOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::SyntheticControl))
}

pub(super) fn local_polynomial_ratio(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (
            DataInput::Tabular(data),
            CausalQuery::LocalPolynomialRatio(_),
            GraphClass::RandomizedTrial,
        ) if analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedLocalPolynomialRatioOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::LocalPolynomialRatio))
}

pub(super) fn survival(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (DataInput::Tabular(data), CausalQuery::Survival(_), GraphClass::RandomizedTrial)
            if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedSurvivalOperation::checked(analysis, data, plan)?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Survival))
}

pub(super) fn longitudinal_regime(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { source, analysis, plan, .. } = *cx;
    let sealed = match (&source.data, &source.query, analysis.graph.class()) {
        (
            DataInput::Tabular(data),
            CausalQuery::LongitudinalRegime(_),
            GraphClass::RandomizedTrial,
        ) if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
            && analysis.graph_posterior.is_none()
            && analysis.tiered.is_none()
            && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedLongitudinalRegimeOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::LongitudinalRegime))
}
