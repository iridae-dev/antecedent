//! Sealers of the randomized-design and interference routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::execute::RouteInput;
use crate::analysis::prepared::PreparedExecution;

pub(super) fn interference(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::InterferenceDag { data, .. })
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::RandomizedEffect { data, .. })
            if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::PolicyValue { data, .. })
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::ContinuousDoseResponse { data, .. })
            if analysis.graph_posterior.is_none()
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::PanelDid { data, .. })
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::SyntheticControl { data, .. })
            if analysis.graph_posterior.is_none()
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::LocalPolynomialRatio { data, .. })
            if analysis.graph_posterior.is_none()
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::Survival { data, .. })
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
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::LongitudinalRegime { data, .. })
            if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
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

pub(super) fn history_match(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::HistoryMatch { data, .. })
            if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedHistoryMatchOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::HistoryMatch))
}

pub(super) fn sequential_contrast(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let sealed = match cx.route {
        Some(RouteInput::SequentialContrast { data, .. })
            if analysis.structure_source == crate::support::StructureSource::RandomizedTrial
                && analysis.graph_posterior.is_none()
                && analysis.tiered.is_none()
                && analysis.split.is_none() =>
        {
            Some(crate::analysis::execute::CheckedSequentialContrastOperation::checked(
                analysis, data, plan,
            )?)
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::SequentialContrast))
}
