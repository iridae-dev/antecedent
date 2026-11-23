//! Whether separately estimated outcomes rest on shared observations.
//!
//! Studies whose outcomes are joined into one law are judged by the evidence ledger, the same
//! overlap judgment choice inference and the composition bundle use: each study is a source
//! whose observations are its rows, identified by the content identity of the study's stored
//! table and the row index ([`ObservationId::from_key`]). The content identity binds the
//! table's columns and rows, not its schema roles or the analysis compiled over it, so two
//! studies of one table share every row whatever outcome each targets. Two studies of one table share every row; studies
//! of different tables share none. A study's rows are what its analysis read, so the overlap
//! is known, never assumed.
//!
//! Studies that share observations are not independent evidence. Under distinct evidence they
//! are refused (`evidence_double_counted`). Under shared evidence the caller declares, for
//! every pair that overlaps, the covariance between the two studies' estimates; the
//! propagation chain receives it as the cross-study block of its declared covariance. The
//! within-study blocks are each study's own posterior covariance, known only after fitting,
//! so the chain checks the whole matrix when it draws.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, Overlap, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_core::ExecutionContext;
use antecedent_core::reason_details::{decision_evidence, preference_compile};

use super::outcomes::OutcomeDependence;
use super::{BoundStudy, CANCELLED_STAGE, PreferenceRouteError, STAGE_COMPILE};
use crate::CausalError;

/// Domain of the observation identities a study's rows carry in the ledger.
const ROW_DOMAIN: &str = "antecedent.preference_route.study_rows";

/// The declared covariance between the estimates of two studies that share observations.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossStudyCovariance {
    /// Index of the first study, in declaration order.
    pub first: usize,
    /// Index of the second study, after the first.
    pub second: usize,
    /// Row-major covariance between every coordinate of the first study's response (rows,
    /// in its grid order) and every coordinate of the second's (columns). Every term is
    /// finite and nonzero: the studies share their rows, and the chain reads a shared factor
    /// with a zero term as an undeclared overlap.
    pub covariance: Vec<f64>,
}

/// Separately estimated outcomes whose studies share observations, joined into one law under
/// a declared cross-study covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedEvidence {
    /// Aligned rows of the joined law.
    pub draws: usize,
    /// Seed of the family table.
    pub seed: u64,
    /// The covariance of every pair of studies that share observations.
    pub cross: Vec<CrossStudyCovariance>,
}

fn invalid(offending: String, supplied: impl Into<String>) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        STAGE_COMPILE,
        preference_compile::CROSS_STUDY_COVARIANCE_INVALID,
    )
    .with_offending(offending)
    .with_supplied(supplied)
    .with_remedy(
        "declare one finite, nonzero covariance block per pair of studies that share \
         observations, rows of the first study's coordinates by columns of the second's",
    )
    .into()
}

fn pair(studies: &[BoundStudy], first: usize, second: usize) -> String {
    let name = |index: usize| {
        studies
            .get(index)
            .map_or_else(|| format!("study {index}"), |s| s.binding.outcome_id.clone())
    };
    format!("{} and {}", name(first), name(second))
}

/// Check every declared block's indices, shape and terms.
fn check_blocks(
    studies: &[BoundStudy],
    cross: &[CrossStudyCovariance],
) -> Result<(), PreferenceRouteError> {
    for (index, block) in cross.iter().enumerate() {
        let (Some(first), Some(second)) = (studies.get(block.first), studies.get(block.second))
        else {
            return Err(invalid(
                format!("block {index}"),
                format!("studies {} and {} of {}", block.first, block.second, studies.len()),
            ));
        };
        if block.first >= block.second {
            return Err(invalid(
                pair(studies, block.first, block.second),
                "the first study is not before the second",
            ));
        }
        if cross[..index].iter().any(|b| (b.first, b.second) == (block.first, block.second)) {
            return Err(invalid(pair(studies, block.first, block.second), "a repeated block"));
        }
        let shape = first.coordinates.len() * second.coordinates.len();
        if block.covariance.len() != shape {
            return Err(invalid(
                pair(studies, block.first, block.second),
                format!("{} terms for a {shape}-term block", block.covariance.len()),
            ));
        }
        if !block.covariance.iter().all(|term| term.is_finite() && term.abs() > 0.0) {
            return Err(invalid(
                pair(studies, block.first, block.second),
                "a non-finite or zero term",
            ));
        }
    }
    Ok(())
}

/// The ledger of the studies' rows, one observation set per study.
fn ledger_of(
    studies: &[BoundStudy],
    ctx: &ExecutionContext,
) -> Result<(EvidenceLedger, Vec<SourceId>), PreferenceRouteError> {
    let mut ledger = EvidenceLedger::new();
    let mut sources = Vec::with_capacity(studies.len());
    for study in studies {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        let table = study.observations.as_bytes();
        let rows = (0..study.rows as u64).map(|row| {
            let mut key = Vec::with_capacity(table.len() + 8);
            key.extend_from_slice(table);
            key.extend_from_slice(&row.to_le_bytes());
            ObservationId::from_key(ROW_DOMAIN, &key)
        });
        let declaration =
            SourceDeclaration::new(study.binding.outcome_id.clone(), SourceKind::ObservationSet)
                .with_observations(rows);
        let source = ledger.add_source(declaration).map_err(|error| {
            PreferenceRouteError::from(
                ExternalRefusal::new(error.code, STAGE_COMPILE, error.detail)
                    .with_offending(study.binding.outcome_id.clone())
                    .with_supplied(error.message),
            )
        })?;
        sources.push(source);
    }
    Ok((ledger, sources))
}

/// Judge whether the studies' outcomes may be joined under `dependence`: every pair the
/// ledger finds overlapping is refused under distinct evidence and needs a declared block
/// under shared evidence; a block for a pair that shares nothing is refused.
///
/// # Errors
/// `evidence_double_counted` (`preference_compile.studies_share_evidence`) for studies that
/// share observations without a declared block; `evidence_overlap_unknown` for an overlap the
/// ledger cannot compute; `invalid_argument`
/// (`preference_compile.cross_study_covariance_invalid`) for a malformed block or one naming
/// studies that share nothing; a ledger refusal; cancellation.
pub(super) fn judge(
    studies: &[BoundStudy],
    dependence: OutcomeDependence,
    cross: &[CrossStudyCovariance],
    ctx: &ExecutionContext,
) -> Result<(), PreferenceRouteError> {
    check_blocks(studies, cross)?;
    let (ledger, sources) = ledger_of(studies, ctx)?;
    for (i, a) in sources.iter().enumerate() {
        for (offset, b) in sources[i + 1..].iter().enumerate() {
            let j = i + 1 + offset;
            let declared = cross.iter().any(|block| (block.first, block.second) == (i, j));
            let overlap = ledger.overlap(*a, *b).map_err(|error| {
                PreferenceRouteError::from(
                    ExternalRefusal::new(error.code, STAGE_COMPILE, error.detail)
                        .with_supplied(error.message),
                )
            })?;
            match overlap {
                Overlap::Independent if declared => {
                    return Err(invalid(pair(studies, i, j), "studies that share no observations"));
                }
                Overlap::Independent => {}
                Overlap::SharedObservations { .. }
                    if declared
                        && matches!(dependence, OutcomeDependence::SharedEvidence { .. }) => {}
                Overlap::SharedObservations { shared } => {
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_double_counted"),
                        STAGE_COMPILE,
                        preference_compile::STUDIES_SHARE_EVIDENCE,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_expected("studies that share no observations, or a declared block")
                    .with_supplied(format!("{shared} shared observations"))
                    .with_remedy(
                        "declare the studies' cross-study covariance as shared evidence, or \
                         estimate the outcomes jointly in one design",
                    )
                    .into());
                }
                other => {
                    let reason = match other {
                        Overlap::Unknown { reason } => reason,
                        _ => "the observation overlap is not recognised",
                    };
                    return Err(ExternalRefusal::new(
                        antecedent_core::reason_code!("evidence_overlap_unknown"),
                        STAGE_COMPILE,
                        decision_evidence::DEPENDENCE_UNKNOWN,
                    )
                    .with_offending(pair(studies, i, j))
                    .with_supplied(reason)
                    .into());
                }
            }
        }
    }
    Ok(())
}
