//! What a law was fitted on, and whether the assessed observations overlap it.
//!
//! A held-out assessment is only held out of the data the law was fitted or
//! tuned on. That is decided from row-level identities, never from a label: the
//! facade derives a [`FitDeclaration`] from the evidence ledger (the single
//! runtime observation-overlap model), resolving derivation lineage first, and
//! hands this module plain observation keys.
//!
//! * [`FitDeclaration::FittedOn`] with disjoint keys is `held_out_verified`.
//! * Shared keys make the assessment `in_sample`: labelled, or refused under
//!   [`OverlapPolicy::RequireHeldOut`].
//! * [`FitDeclaration::Undeclared`] (the usual case of a provider-supplied
//!   law) is `unverified`: never promoted to held out.
//! * [`FitDeclaration::NotFitted`] is a law fixed without data (a known
//!   generator, a specified model).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;

use super::law::ObservationKey;
use super::refusal::{PredictiveAssessmentError, detail, refuse};

/// What a predictor's law was fitted or tuned on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FitDeclaration {
    /// Fitted or tuned on exactly these observations (lineage already resolved).
    FittedOn(Vec<ObservationKey>),
    /// The law was fixed without data: a specified model or a known generator.
    NotFitted,
    /// The producer did not say. This is not "held out".
    Undeclared {
        /// Why the fit data is not known.
        reason: String,
    },
}

/// What an assessment does with an overlap it finds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum OverlapPolicy {
    /// Report the overlap status as a label and score anyway. The default.
    #[default]
    LabelOnly,
    /// Refuse unless every predictor is verified held out or not fitted.
    RequireHeldOut,
}

/// How the assessed observations stand against what a law was fitted on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OverlapStatus {
    /// The fit observations are declared and disjoint from the assessed ones.
    HeldOutVerified {
        /// Number of observations the law was fitted on.
        fitted_on: u64,
    },
    /// Some assessed observations were used to fit the law.
    InSample {
        /// Assessed observations also in the fit set.
        shared: u64,
        /// Assessed observations.
        assessed: u64,
    },
    /// What the law was fitted on is not declared.
    Unverified {
        /// Why.
        reason: String,
    },
    /// The law was fixed without data.
    NotFitted,
}

impl OverlapStatus {
    /// Stable `snake_case` label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::HeldOutVerified { .. } => "held_out_verified",
            Self::InSample { .. } => "in_sample",
            Self::Unverified { .. } => "unverified",
            Self::NotFitted => "not_fitted",
        }
    }
}

/// Number of keys in both sorted, distinct slices.
fn shared_count(a: &[ObservationKey], b: &[ObservationKey]) -> u64 {
    let (mut i, mut j, mut shared) = (0, 0, 0_u64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared
}

fn sorted_distinct(keys: &[ObservationKey]) -> Vec<ObservationKey> {
    let mut out = keys.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

/// Number of observations two predictors were both fitted on, when both declare
/// what they were fitted on.
pub(crate) fn fit_overlap(a: &FitDeclaration, b: &FitDeclaration) -> Option<u64> {
    match (a, b) {
        (FitDeclaration::FittedOn(x), FitDeclaration::FittedOn(y)) => {
            Some(shared_count(&sorted_distinct(x), &sorted_distinct(y)))
        }
        (FitDeclaration::NotFitted, FitDeclaration::FittedOn(_) | FitDeclaration::NotFitted)
        | (FitDeclaration::FittedOn(_), FitDeclaration::NotFitted) => Some(0),
        _ => None,
    }
}

/// Decide the overlap status of one predictor against the assessed observations.
///
/// `assessed` is the sorted, distinct set of assessed observation keys.
///
/// # Errors
/// [`OverlapPolicy::RequireHeldOut`] with an in-sample or unverified predictor.
pub(crate) fn decide_overlap(
    predictor: &str,
    fit: &FitDeclaration,
    assessed: &[ObservationKey],
    policy: OverlapPolicy,
) -> Result<OverlapStatus, PredictiveAssessmentError> {
    let status = match fit {
        FitDeclaration::NotFitted => OverlapStatus::NotFitted,
        FitDeclaration::Undeclared { reason } => {
            OverlapStatus::Unverified { reason: reason.clone() }
        }
        FitDeclaration::FittedOn(keys) => {
            let fitted = sorted_distinct(keys);
            let shared = shared_count(&fitted, assessed);
            if shared == 0 && fitted.iter().chain(assessed).any(ObservationKey::is_caller_minted) {
                OverlapStatus::Unverified {
                    reason: "observation identities were minted from caller-chosen names, a \
                             namespace disjoint from the evidence ledger, so disjointness from \
                             ledger rows cannot be verified"
                        .to_owned(),
                }
            } else if shared == 0 {
                OverlapStatus::HeldOutVerified { fitted_on: fitted.len() as u64 }
            } else {
                OverlapStatus::InSample { shared, assessed: assessed.len() as u64 }
            }
        }
    };
    if policy == OverlapPolicy::RequireHeldOut {
        match &status {
            OverlapStatus::InSample { shared, assessed } => {
                return Err(refuse(
                    reason_code!("evidence_double_counted"),
                    detail::IN_SAMPLE,
                    format!(
                        "{shared} of the {assessed} assessed observations were used to fit \
                         predictor `{predictor}`, so the score is in-sample and not a held-out \
                         assessment. Assess on observations outside its fit set, or accept the \
                         `in_sample` label by not requiring held-out observations"
                    ),
                ));
            }
            OverlapStatus::Unverified { reason } => {
                return Err(refuse(
                    reason_code!("evidence_overlap_unknown"),
                    detail::OVERLAP_UNKNOWN,
                    format!(
                        "what predictor `{predictor}` was fitted on is not declared ({reason}); \
                         independence of the assessed observations is never assumed. Declare the \
                         row-level observations it was fitted or tuned on through the evidence \
                         ledger, or accept the `unverified` label"
                    ),
                ));
            }
            OverlapStatus::HeldOutVerified { .. } | OverlapStatus::NotFitted => {}
        }
    }
    Ok(status)
}
