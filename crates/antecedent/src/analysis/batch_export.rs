//! Tidy export of a batch retarget: one row per declared claim and contrast.
//!
//! The export never flattens a partial family into its successes: a failed member is a row
//! with `status = "failed"` and its typed refusal, a point-only member is a row with a value
//! and no standard error, and every row carries the family identity, whether the family is
//! complete, how many members failed, the simultaneous-interval availability and the
//! provenance of the scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::batch_retarget::{BatchRetargetReport, MemberFailure, UncertaintyKind};

/// Kind of a tidy row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TidyKind {
    /// A retargeted claim.
    Claim,
    /// A named linear contrast of claims.
    Contrast,
}

impl TidyKind {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Claim => "claim",
            Self::Contrast => "contrast",
        }
    }
}

/// One tidy row of a batch retarget.
#[derive(Clone, Debug, PartialEq)]
pub struct TidyRow {
    /// Order-invariant identity of the declared family and its snapshot.
    pub family_id: String,
    /// Whether every claim and contrast succeeded and every claim carries covariance.
    pub family_complete: bool,
    /// Number of declared claims and contrasts.
    pub family_size: usize,
    /// Number of failed claims and contrasts.
    pub family_failed: usize,
    /// Claim or contrast.
    pub kind: TidyKind,
    /// Claim or contrast name.
    pub name: String,
    /// The estimand of a claim, or the declared terms of a contrast.
    pub estimand: String,
    /// `ok`, `point_only` or `failed`.
    pub status: &'static str,
    /// The point, when the member succeeded.
    pub value: Option<f64>,
    /// Plug-in standard error, when the member carries covariance.
    pub std_error: Option<f64>,
    /// `plug_in_score_covariance` or `none`.
    pub uncertainty_kind: &'static str,
    /// Family-level simultaneous interval: `max_t` when the family is complete (the band is
    /// available from the report), otherwise `none`.
    pub simultaneous_interval: &'static str,
    /// `supported` (weighted overlap passed), `refused` (weighted overlap failed) or
    /// `not_assessed` (a contrast, or a member that failed before the overlap gate).
    pub support_status: &'static str,
    /// Kish effective sample size of the target weights, for a claim that succeeded.
    pub n_eff: Option<f64>,
    /// Reason code of the refusal, when the member failed with one.
    pub refusal_code: Option<String>,
    /// Stable detail of the refusal, when it carries one.
    pub refusal_detail: Option<String>,
    /// Explanation of the refusal.
    pub refusal_message: Option<String>,
    /// Notes of the member.
    pub diagnostics: Vec<String>,
    /// Batch query index of a claim.
    pub query_index: Option<usize>,
    /// Whether the scores were `prepared` or `estimated`.
    pub scores_source: &'static str,
    /// Identity of the common row snapshot.
    pub snapshot_id: Option<String>,
    /// Canonical estimator-configuration fingerprint of the batch.
    pub estimator_fingerprint: String,
    /// The score table's nuisance provenance, for a claim that succeeded.
    pub nuisance_provenance: Option<String>,
}

fn failure_fields(
    failure: &MemberFailure,
) -> (&'static str, Option<String>, Option<String>, Option<String>) {
    (
        if failure.support_refused { "refused" } else { "not_assessed" },
        failure.reason_code.clone(),
        failure.detail.clone(),
        Some(failure.message.clone()),
    )
}

impl BatchRetargetReport {
    /// One row per declared claim then per declared contrast, failed members included.
    #[must_use]
    pub fn tidy_rows(&self) -> Vec<TidyRow> {
        let size = self.claims.len() + self.contrasts.len();
        let failed = self.failed_members().len();
        let complete = self.complete_family().is_ok();
        let base = |kind: TidyKind, name: &str, estimand: String| TidyRow {
            family_id: self.family_id.clone(),
            family_complete: complete,
            family_size: size,
            family_failed: failed,
            kind,
            name: name.to_string(),
            estimand,
            status: "failed",
            value: None,
            std_error: None,
            uncertainty_kind: UncertaintyKind::None.as_str(),
            simultaneous_interval: if complete { "max_t" } else { "none" },
            support_status: "not_assessed",
            n_eff: None,
            refusal_code: None,
            refusal_detail: None,
            refusal_message: None,
            diagnostics: Vec::new(),
            query_index: None,
            scores_source: self.scores_source.as_str(),
            snapshot_id: self.snapshot_id.clone(),
            estimator_fingerprint: self.estimator_fingerprint.clone(),
            nuisance_provenance: None,
        };
        let mut rows = Vec::with_capacity(size);
        for claim in &self.claims {
            let mut row = base(TidyKind::Claim, &claim.name, claim.estimand.clone());
            row.query_index = Some(claim.query_index);
            match &claim.outcome {
                Ok(point) => {
                    row.status = if point.std_error.is_some() { "ok" } else { "point_only" };
                    row.value = Some(point.value);
                    row.std_error = point.std_error;
                    row.uncertainty_kind = point.uncertainty_kind.as_str();
                    row.support_status = "supported";
                    row.n_eff = Some(point.n_eff);
                    row.diagnostics.clone_from(&point.diagnostics);
                    row.nuisance_provenance = Some(point.nuisance_provenance.clone());
                }
                Err(failure) => {
                    (
                        row.support_status,
                        row.refusal_code,
                        row.refusal_detail,
                        row.refusal_message,
                    ) = failure_fields(failure);
                }
            }
            rows.push(row);
        }
        for contrast in &self.contrasts {
            let terms: Vec<String> =
                contrast.terms.iter().map(|(name, c)| format!("{c:+}*{name}")).collect();
            let mut row = base(TidyKind::Contrast, &contrast.name, terms.join(" "));
            match &contrast.outcome {
                Ok(point) => {
                    row.status = if point.std_error.is_some() { "ok" } else { "point_only" };
                    row.value = Some(point.value);
                    row.std_error = point.std_error;
                    row.uncertainty_kind = point.uncertainty_kind.as_str();
                }
                Err(failure) => {
                    (
                        row.support_status,
                        row.refusal_code,
                        row.refusal_detail,
                        row.refusal_message,
                    ) = failure_fields(failure);
                }
            }
            rows.push(row);
        }
        rows
    }
}
