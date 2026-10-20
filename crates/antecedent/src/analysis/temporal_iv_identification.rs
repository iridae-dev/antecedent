//! The certified lagged instrument of a single-step temporal effect.
//!
//! The graphical conditions are evaluated on the unfolded window graph by
//! [`antecedent_identify::TemporalIvIdentifier`]; this owner turns its verdict into the retained
//! identification, the selected estimand and the indexer the executors read, and turns an absent
//! instrument into a typed refusal that names the premise that failed and the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::TemporalEffectQuery;
use antecedent_data::{TemporalIndexer, TemporalNodeKey};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::TemporalDag;
use antecedent_identify::TemporalIvIdentifier;
use antecedent_identify::result::{IdentificationResult, IdentificationStatus};

use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, select_claim};

/// The identification of a temporal instrument, with the estimand and unfolding it binds.
#[derive(Clone, Debug)]
pub(crate) struct CertifiedTemporalInstrument {
    /// The identification narrowed to the selected instrument's claim.
    pub(crate) identification: IdentificationResult,
    /// The selected Wald estimand.
    pub(crate) estimand: IdentifiedEstimand,
    /// The finite unfolding its dense ids address.
    pub(crate) indexer: TemporalIndexer,
    /// The selected instrument as a `(variable, offset)` node.
    ///
    /// This is the first valid instrument of the window, nearest in search order.
    pub(crate) instrument: TemporalNodeKey,
}

const NO_INSTRUMENT: &str = "no lagged variable is a valid instrument on the unfolded window \
     graph: each lag is a descendant of the treatment, is independent of it (no first stage), \
     reaches the outcome other than through the treatment (exclusion), or shares an unblocked \
     path with the outcome (exogeneity). Next attempt: identify the effect by back-door \
     adjustment for the confounder, declare a graph in which a lagged cause of the treatment is \
     separated from the outcome, or run an experiment on the treatment";

/// Identify the effect by a lagged instrument.
///
/// # Errors
/// A query or window the identifier refuses, or no valid instrument
/// (`effect_not_identified`, with the failed premise and the next attempt in the message).
pub(crate) fn identify_temporal_instrument(
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
) -> Result<CertifiedTemporalInstrument, CausalError> {
    let found = TemporalIvIdentifier::new()
        .identify_temporal_iv(graph, query)
        .map_err(CausalError::from)?;
    if found.result.status == IdentificationStatus::NotIdentified || found.instruments.is_empty() {
        return Err(CausalError::not_identified(found.result.status, false, NO_INSTRUMENT));
    }
    let (identification, estimand) =
        select_claim(found.result, EstimatorId::TemporalLaggedInstrument)?;
    let instrument = *found.instruments.first().ok_or_else(|| CausalError::Compile {
        message: "temporal instrument identification lost its instrument".into(),
    })?;
    Ok(CertifiedTemporalInstrument { identification, estimand, indexer: found.indexer, instrument })
}
