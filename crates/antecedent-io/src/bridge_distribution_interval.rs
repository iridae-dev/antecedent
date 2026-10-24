//! The interval of a bridged distribution, through the measured-coverage gate of a
//! bridged mean.
//!
//! A bridged distribution is read as an uncalibrated spread. It becomes an
//! interval only where the same construction that publishes the interval of a
//! bridged value is measured: the source's own spread (the standard deviation of
//! its draws) carried through the mapping together with the mapping's own
//! estimation uncertainty, at a bridge kind, mapping, direction and validated
//! trial count a current attesting coverage record covers.
//! [`bridged_distribution_interval`] builds the mean twin of the bridge at the
//! source's mean and asks [`crate::bridge_interval::bridged_claim_interval`] for
//! it, so no coverage statement is made here that the gate does not make.
//!
//! The gate refuses what its records did not measure:
//!
//! * a declared affine map with uncertain coefficients, whose uncertainty is
//!   propagated by draws and not by the delta-method construction any record
//!   measured;
//! * a declared covariance between the source and the mapping, which the records
//!   (measured with independent source and mapping evidence) do not cover;
//! * every mapping outside the measured design, as for a mean.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    BridgeSource, SourceMappingDependence, ValueKind, derive_bridged_claim,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, ReasoningView, SlotAvailability,
};
use antecedent_contract::{DomainStatus, ExternalRefusal};
use antecedent_core::reason_code;
use antecedent_io_core::bridge_distribution::BridgedDistribution;

use crate::bridge_interval::{BridgedClaimInterval, bridged_claim_interval};

const DETAIL_OUTSIDE_MEASURED: &str = "scientific_bridge.interval_scope_outside_measured";

fn outside_measured(message: &str) -> ExternalRefusal {
    ExternalRefusal::new(reason_code!("cell_not_licensed"), "bind", DETAIL_OUTSIDE_MEASURED)
        .with_remedy(
            "ask for the interval inside a measured design: a bridge kind and mapping with a current attesting coverage record, source and mapping evidence declared independent, and level 0.95; read the draws as an uncalibrated spread meanwhile",
        )
        .with_offending(message)
}

/// The interval of a bridged distribution at `level`.
///
/// The source standard error is the standard deviation of the source draws. The
/// interval is that of the bridged value at the source's mean, the construction a
/// coverage record measures, and it is published only where such a record exists.
///
/// # Errors
/// `cell_not_licensed` with `scientific_bridge.interval_scope_outside_measured`
/// for a mapping with uncertain coefficients, a declared source-mapping
/// covariance, a level outside the measured range, or a design no current
/// attesting coverage record measured.
#[allow(clippy::result_large_err, reason = "the refusal is the shared external refusal type")]
pub fn bridged_distribution_interval(
    bridged: &BridgedDistribution,
    level: f64,
) -> Result<BridgedClaimInterval, ExternalRefusal> {
    let claim = bridged.claim();
    let contract = &claim.contract;
    if contract.mapping_uncertainty.is_some() {
        return Err(outside_measured(
            "a declared affine map with uncertain coefficients is propagated by draws; no coverage record measured it",
        ));
    }
    if let Some(SourceMappingDependence::SlopeCovariance { .. }) = &contract.source_dependence {
        return Err(outside_measured(
            "the coverage records were measured with the source and the mapping evidence independent; a declared covariance is outside them",
        ));
    }
    // The mean twin: the same bridge of the source's mean, which is what a record measured.
    let mut twin = contract.clone();
    twin.source_value = ValueKind::Mean;
    twin.target_value = ValueKind::Mean;
    twin.source_dependence = None;
    let template = bridged.source_envelope();
    let source = ClaimEnvelope::new(
        template.claim_id,
        template.identities,
        ClaimKind::Point,
        Some(bridged.source_mean()),
        template.outcome_units.clone(),
        ReasoningView::new(
            SlotAvailability::unavailable("draws_summary"),
            SlotAvailability::unavailable("draws_summary"),
            SlotAvailability::unavailable("draws_summary"),
            SlotAvailability::unavailable("draws_summary"),
        ),
        ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, DomainStatus::Evaluated),
        None,
        Vec::<Arc<str>>::new(),
    );
    let point = derive_bridged_claim(
        &twin,
        &BridgeSource { claim: &source, quantity: &twin.source.quantity },
    )?;
    bridged_claim_interval(&point, Some(bridged.source_variance().sqrt()), level)
}
