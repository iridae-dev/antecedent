//! Sealers of the routes whose retained operation is their lowered program:
//! structure discovery, panel and multi-environment temporal effects, temporal
//! cross-world queries and temporal ADMG effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SealContext, SealOutcome};
use crate::analysis::execute::{CheckedRouteOperation, RouteInput};
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
    let sealed = match cx.route {
        Some(RouteInput::DiscoverStatic { .. }) => seal(cx, RouteCell::DiscoverStatic)?,
        Some(RouteInput::DiscoverTemporal { .. }) => seal(cx, RouteCell::DiscoverTemporal)?,
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Discovery))
}

pub(super) fn association(cx: &SealContext<'_>) -> SealOutcome {
    let sealed = match cx.route {
        Some(RouteInput::AssociationStatic { .. }) => seal(cx, RouteCell::AssociationStatic)?,
        Some(RouteInput::AssociationTemporal { .. }) => seal(cx, RouteCell::AssociationTemporal)?,
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::Association))
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
    let sealed = match cx.route {
        Some(RouteInput::TemporalEffectAdmg { .. }) => seal(cx, RouteCell::TemporalEffectAdmg)?,
        _ => None,
    };
    Ok(sealed.map(PreparedExecution::TemporalAdmgEffect))
}
