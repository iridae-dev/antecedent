//! Identified estimand types shared by identify and estimate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use causal_core::VariableId;

use crate::ExprId;

/// Typed identification method tag (wire form remains the Display string).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EstimandMethod {
    /// Classic backdoor adjustment.
    BackdoorAdjustment,
    /// Efficient backdoor adjustment set.
    BackdoorEfficient,
    /// Front-door identification.
    FrontDoor,
    /// Instrumental variable.
    Iv,
    /// Sharp regression discontinuity.
    RdSharp,
    /// Temporal backdoor after finite unfolding.
    TemporalBackdoorUnfolded,
    /// Temporal mediation — total effect.
    TemporalMediationTotal,
    /// Temporal mediation — direct effect.
    TemporalMediationDirect,
    /// Temporal mediation — mediated effect.
    TemporalMediationMediated,
    /// General semi-Markovian ID / IDC (Shpitser–Pearl).
    GeneralId,
    /// Path-restricted natural effect (Avin–Shpitser–Pearl).
    PathSpecificNatural,
}

impl EstimandMethod {
    /// Canonical wire / Display string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BackdoorAdjustment => "backdoor.adjustment",
            Self::BackdoorEfficient => "backdoor.efficient",
            Self::FrontDoor => "frontdoor",
            Self::Iv => "iv",
            Self::RdSharp => "rd.sharp",
            Self::TemporalBackdoorUnfolded => "temporal.backdoor.unfolded",
            Self::TemporalMediationTotal => "temporal_mediation.total",
            Self::TemporalMediationDirect => "temporal_mediation.direct",
            Self::TemporalMediationMediated => "temporal_mediation.mediated",
            Self::GeneralId => "general.id",
            Self::PathSpecificNatural => "path_specific.natural",
        }
    }

    /// Whether this is any temporal-mediation variant.
    #[must_use]
    pub const fn is_temporal_mediation(self) -> bool {
        matches!(
            self,
            Self::TemporalMediationTotal
                | Self::TemporalMediationDirect
                | Self::TemporalMediationMediated
        )
    }

    /// Whether this is a backdoor-family adjustment estimand.
    #[must_use]
    pub const fn is_backdoor_family(self) -> bool {
        matches!(self, Self::BackdoorAdjustment | Self::BackdoorEfficient)
    }
}

impl fmt::Display for EstimandMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EstimandMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "backdoor.adjustment" => Self::BackdoorAdjustment,
            "backdoor.efficient" => Self::BackdoorEfficient,
            "frontdoor" => Self::FrontDoor,
            "iv" => Self::Iv,
            "rd.sharp" => Self::RdSharp,
            "temporal.backdoor.unfolded" => Self::TemporalBackdoorUnfolded,
            "temporal_mediation.total" => Self::TemporalMediationTotal,
            "temporal_mediation.direct" => Self::TemporalMediationDirect,
            "temporal_mediation.mediated" => Self::TemporalMediationMediated,
            "general.id" => Self::GeneralId,
            "path_specific.natural" => Self::PathSpecificNatural,
            other => return Err(format!("unknown estimand method `{other}`")),
        })
    }
}

impl From<EstimandMethod> for Arc<str> {
    fn from(value: EstimandMethod) -> Self {
        Arc::from(value.as_str())
    }
}

/// One identified estimand.
///
/// Backdoor estimands use [`Self::adjustment_set`]; IV estimands populate
/// [`Self::instruments`]; front-door estimands populate [`Self::mediators`].
/// Unused role slices are empty.
#[derive(Clone, Debug)]
pub struct IdentifiedEstimand {
    /// Method tag (wire string; parse with [`Self::method_kind`]).
    pub method: Arc<str>,
    /// Adjustment set (dense variable ids). Empty when not an adjustment estimand.
    pub adjustment_set: Arc<[VariableId]>,
    /// Instrument variables (dense ids). Empty unless IV.
    pub instruments: Arc<[VariableId]>,
    /// Mediator variables for front-door / two-stage. Empty unless front-door.
    pub mediators: Arc<[VariableId]>,
    /// Functional expression id in `arena`.
    pub functional: ExprId,
}

impl IdentifiedEstimand {
    /// Parse the method tag into a typed [`EstimandMethod`].
    ///
    /// # Errors
    ///
    /// Unknown method string.
    pub fn method_kind(&self) -> Result<EstimandMethod, String> {
        EstimandMethod::from_str(self.method.as_ref())
    }

    /// Backdoor-style estimand with an adjustment set and empty IV/mediator roles.
    #[must_use]
    pub fn backdoor(
        method: impl Into<Arc<str>>,
        adjustment_set: Arc<[VariableId]>,
        functional: ExprId,
    ) -> Self {
        Self {
            method: method.into(),
            adjustment_set,
            instruments: Arc::from([]),
            mediators: Arc::from([]),
            functional,
        }
    }

    /// IV estimand with instruments and empty adjustment/mediators.
    #[must_use]
    pub fn instrumental(
        method: impl Into<Arc<str>>,
        instruments: Arc<[VariableId]>,
        functional: ExprId,
    ) -> Self {
        Self {
            method: method.into(),
            adjustment_set: Arc::from([]),
            instruments,
            mediators: Arc::from([]),
            functional,
        }
    }

    /// Front-door estimand with mediators and empty adjustment/instruments.
    #[must_use]
    pub fn frontdoor(
        method: impl Into<Arc<str>>,
        mediators: Arc<[VariableId]>,
        functional: ExprId,
    ) -> Self {
        Self {
            method: method.into(),
            adjustment_set: Arc::from([]),
            instruments: Arc::from([]),
            mediators,
            functional,
        }
    }
}
