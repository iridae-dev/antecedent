//! Sealers of the routes whose retained operation is their lowered program:
//! structure discovery, panel and multi-environment temporal effects, temporal
//! cross-world queries and temporal ADMG effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::execute::{
    CheckedAssociationOperation, CheckedDiscoveryOperation, CheckedRouteOperation, RouteInput,
};
use crate::analysis::lowering::RouteCell;
use crate::analysis::prepared::PreparedExecution;
use crate::error::CausalError;

/// Seal the study to the program `plan` carries for `cell`.
fn seal(
    cx: &SealContext<'_>,
    cell: RouteCell,
) -> Result<Option<CheckedRouteOperation>, CausalError> {
    let SealContext { analysis, plan, .. } = *cx;
    if analysis.graph_posterior.is_some() {
        return Ok(None);
    }
    CheckedRouteOperation::checked(analysis, plan, cell).map(Some)
}

pub(super) fn discovery(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let cell = match cx.route {
        Some(RouteInput::DiscoverStatic { .. }) => RouteCell::DiscoverStatic,
        Some(RouteInput::DiscoverTemporal { .. }) => RouteCell::DiscoverTemporal,
        _ => return Ok(None),
    };
    if analysis.graph_posterior.is_some() {
        return Ok(None);
    }
    CheckedDiscoveryOperation::checked(analysis, plan, cell)
        .map(|operation| Some(PreparedExecution::Discovery(operation)))
}

pub(super) fn association(cx: &SealContext<'_>) -> SealOutcome {
    let SealContext { analysis, plan, .. } = *cx;
    let cell = match cx.route {
        Some(RouteInput::AssociationStatic { .. }) => RouteCell::AssociationStatic,
        Some(RouteInput::AssociationTemporal { .. }) => RouteCell::AssociationTemporal,
        _ => return Ok(None),
    };
    if analysis.graph_posterior.is_some() {
        return Ok(None);
    }
    CheckedAssociationOperation::checked(analysis, plan, cell)
        .map(|operation| Some(PreparedExecution::Association(operation)))
}

pub(super) fn panel_temporal(cx: &SealContext<'_>) -> SealOutcome {
    let sealed = match cx.route {
        Some(RouteInput::PanelTemporalEffectDag { .. }) => {
            seal(cx, RouteCell::PanelTemporalEffectDag)?
        }
        Some(RouteInput::PanelTemporalEffectClass { .. }) => {
            seal(cx, RouteCell::PanelTemporalEffectClass)?
        }
        Some(RouteInput::PanelTemporalResponseDag { .. }) => {
            seal(cx, RouteCell::PanelTemporalResponseDag)?
        }
        Some(RouteInput::PanelTemporalResponseClass { .. }) => {
            seal(cx, RouteCell::PanelTemporalResponseClass)?
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::PanelTemporal))
}

pub(super) fn multi_env_temporal(cx: &SealContext<'_>) -> SealOutcome {
    let sealed = match cx.route {
        Some(RouteInput::MultiEnvTemporalEffect { .. }) => {
            seal(cx, RouteCell::MultiEnvTemporalEffect)?
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::MultiEnvTemporal))
}

pub(super) fn temporal_cross_world(cx: &SealContext<'_>) -> SealOutcome {
    let sealed = match cx.route {
        Some(RouteInput::TemporalCrossWorldDag { .. }) => {
            seal(cx, RouteCell::TemporalCrossWorldDag)?
        }
        Some(RouteInput::TemporalCrossWorldClass { .. }) => {
            seal(cx, RouteCell::TemporalCrossWorldClass)?
        }
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalCrossWorld))
}

pub(super) fn temporal_admg_effect(cx: &SealContext<'_>) -> SealOutcome {
    // A graph posterior over lagged temporal-ADMG atoms retains its lowered program: the
    // per-atom window identification of a contrast is frozen in the study's cache, and a
    // mediation identifies each atom's components on its own window at each click.
    let SealContext { analysis, plan, .. } = *cx;
    if analysis
        .graph_posterior
        .as_ref()
        .is_some_and(antecedent_discovery::GraphPosterior::is_lagged_temporal_admg)
        && matches!(
            analysis.data,
            crate::analysis::builder::DataInput::Temporal(_)
                | crate::analysis::builder::DataInput::Event(_)
        )
        && matches!(
            analysis.query,
            antecedent_core::CausalQuery::TemporalEffect(_)
                | antecedent_core::CausalQuery::Mediation(_)
        )
    {
        return crate::analysis::execute::CheckedTemporalAdmgOperation::checked(
            analysis, plan, None,
        )
        .map(|operation| Some(PreparedExecution::TemporalAdmgEffect(operation)));
    }
    let cell = match cx.route {
        Some(RouteInput::TemporalEffectAdmg { .. }) => RouteCell::TemporalEffectAdmg,
        Some(RouteInput::TemporalResponseAdmg { .. }) => RouteCell::TemporalResponseAdmg,
        Some(RouteInput::TemporalMediationAdmg { .. }) => RouteCell::TemporalMediationAdmg,
        Some(RouteInput::TemporalMediationPag { .. }) => RouteCell::TemporalMediationPag,
        _ => return Ok(None),
    };
    if analysis.graph_posterior.is_some() {
        return Ok(None);
    }
    crate::analysis::execute::CheckedTemporalAdmgOperation::checked(analysis, plan, Some(cell))
        .map(|operation| Some(PreparedExecution::TemporalAdmgEffect(operation)))
}

pub(super) fn class_attribution(cx: &SealContext<'_>) -> SealOutcome {
    let cell = match cx.route {
        Some(RouteInput::GcmClass { .. }) => RouteCell::GcmClass,
        Some(RouteInput::GcmTemporalClass { .. }) => RouteCell::GcmTemporalClass,
        _ => return Ok(None),
    };
    if cx.analysis.graph_posterior.is_some() {
        return Ok(None);
    }
    crate::analysis::execute::CheckedClassAttributionOperation::checked(cx.analysis, cx.plan, cell)
        .map(|operation| Some(PreparedExecution::ClassAttribution(operation)))
}

/// Static cross-world cells retain a family operation, not estimator fallback.
pub(super) fn static_cross_world(cx: &SealContext<'_>) -> SealOutcome {
    let cell = match cx.route {
        Some(RouteInput::CounterfactualClass { .. }) => RouteCell::CounterfactualClass,
        Some(RouteInput::PathSpecificMixed { .. }) => RouteCell::PathSpecificMixed,
        Some(RouteInput::PathSpecificCpdag { .. }) => RouteCell::PathSpecificCpdag,
        Some(RouteInput::PathSpecificPag { .. }) => RouteCell::PathSpecificPag,
        _ => return Ok(None),
    };
    if cx.analysis.graph_posterior.is_some() {
        return Ok(None);
    }
    crate::analysis::execute::CheckedStaticCrossWorldOperation::checked(cx.analysis, cx.plan, cell)
        .map(|operation| Some(PreparedExecution::StaticCrossWorld(operation)))
}
