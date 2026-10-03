//! Batch retarget: the points, joint covariance and named contrasts of a family of
//! retargeted claims over one row snapshot, with the family-level max-t band.
//!
//! Each claim `k` reweights one plan's cross-fitted score table `φ_k` (the AIPW contrast
//! score of an average effect, or the cell score of a joint cell) under caller-declared
//! row weights `w_k`: `θ_k = Σ_r w_kr φ_kr / Σ_r w_kr`. With `a_kr = w_kr / Σ_s w_ks`
//! the weighted influence value of row `r` is
//!
//! ```text
//! ξ_kr = sqrt(n / (n - 1)) · a_kr · (φ_kr − θ_k)
//! ```
//!
//! and the family covariance is the Gram matrix `Σ_kl = Σ_r ξ_kr ξ_lr`. Its diagonal is
//! exactly the plug-in variance a single-claim retarget reports (the `n / (n - 1)` of
//! [`antecedent_estimate::joint_influence_covariance`] sits inside `ξ`), so a one-claim
//! family equals a single `retarget`. The off-diagonal entries are the cross-claim
//! covariance from the shared rows: they exist only because every claim's scores are
//! indexed by the *same* rows, which is why the family must share one row snapshot. `Σ` is
//! symmetric by construction and positive semidefinite (it is a Gram matrix; the test pins
//! the Cauchy–Schwarz and quadratic-form checks within floating-point tolerance).
//!
//! A named linear contrast `Σ_k c_k θ_k` has value `c'θ` and plug-in variance `c'Σc`;
//! "route A minus route B" among declared holders is the contrast `+1·A −1·B` of two claims
//! (the same plan retargeted to two weightings, or two plans).
//!
//! What is claimed: points and a plug-in score covariance under iid rows, fixed declared
//! weights, positivity and nuisance convergence (the single-claim retarget's contract).
//! The family-level simultaneous (max-t) band `θ_k ± c · se_k` is
//! [`BatchRetargetReport::simultaneous_interval`]: a nominal asymptotic construction on `Σ`
//! (plug-in covariance, Monte-Carlo critical value `c`), published for a complete family
//! only. Its coverage is measured by the calibration harness, not asserted here. A
//! penalized-propensity table retargets to a point only and joins the family without
//! covariance. A partial family (any failed or point-only member) is reported member by
//! member and never offered as a complete-family claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, VariableId, reason_code};
use antecedent_estimate::{
    JointCovariance, RefusalFields, ScoreTable, max_t_critical_polled,
    provenance_withholds_interval,
};
use antecedent_io::PayloadDigestWire;

use crate::error::CausalError;

use super::batch::BatchQuery;
use super::prepared::PreparedStudy;

/// What a batch retarget is and is not, carried on every report.
pub const BATCH_RETARGET_SCOPE_NOTE: &str = "Points and plug-in score covariance of retargeted claims over one row snapshot, under iid rows, caller-declared fixed weights, positivity and nuisance convergence; selection or weight-estimation uncertainty is excluded. The family-level simultaneous interval is a nominal asymptotic max-t band on that plug-in covariance, available for a complete family only.";

/// A refusal of a batch retarget, with its registered reason code.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct BatchRetargetError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `batch_retarget.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
    /// The claim or contrast the refusal is about, when it names one.
    pub subject: Option<String>,
}

impl BatchRetargetError {
    /// This refusal about a named claim or contrast.
    #[must_use]
    pub fn about(mut self, subject: &str) -> Self {
        self.subject = Some(subject.to_string());
        self
    }

    /// Structured diagnostics: stage, subject (when it names one), the stable detail as the
    /// reason, and the remedy where one is known. Nothing numeric: a refusal of the
    /// family's declaration measures no fitted quantity.
    #[must_use]
    pub fn refusal_fields(&self) -> RefusalFields {
        RefusalFields {
            stage: Some("batch_retarget".to_string()),
            subject: self.subject.clone(),
            reason: Some(self.detail.to_string()),
            remedy: remedy_of(self.detail).map(str::to_string),
            ..RefusalFields::default()
        }
    }
}

/// What the caller can change for a refusal, by its stable detail; `None` where the
/// refusal names no single change.
fn remedy_of(detail: &str) -> Option<&'static str> {
    Some(match detail {
        "batch_retarget.functional_not_licensed" => {
            "retarget the mean functional (an average effect or a joint cell); keep a quantile or \
             exceedance grid on the single-claim retarget"
        }
        "batch_retarget.incompatible_target" => {
            "declare finite non-negative weights, one per row of the common snapshot"
        }
        "batch_retarget.scores_unavailable"
        | "batch_retarget.scores_unavailable_after_estimate" => {
            "use an AllObserved iid AIPW or cell-AIPW plan and keep its scores"
        }
        "batch_retarget.contrast_member_failed" => {
            "fix or drop the failed claim the contrast reads"
        }
        "batch_retarget.unknown_claim" => {
            "name only claims of this family and queries of this batch"
        }
        "batch_retarget.duplicate_name" => "give every claim and contrast its own nonempty name",
        "batch_retarget.invalid_contrast" => "use finite coefficients on distinct declared claims",
        _ => return None,
    })
}

fn invalid(detail: &'static str, message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("invalid_argument"),
        detail,
        message: message.into(),
        subject: None,
    }
}

fn not_licensed(detail: &'static str, message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("cell_not_licensed"),
        detail,
        message: message.into(),
        subject: None,
    }
}

fn not_supported(detail: &'static str, message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("route_not_supported"),
        detail,
        message: message.into(),
        subject: None,
    }
}

fn mixed_snapshot(message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("row_weights_bound_to_snapshot"),
        detail: "batch_retarget.mixed_snapshot",
        message: message.into(),
        subject: None,
    }
}

fn cancelled() -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("cancelled_no_claim"),
        detail: "batch_retarget.cancelled",
        message: "the batch retarget was cancelled; no family is reported and the stop is not a \
                  verdict on the data"
            .into(),
        subject: None,
    }
}

fn scores_unavailable(source: ScoreSource) -> BatchRetargetError {
    let (detail, message) = match source {
        ScoreSource::Estimated => (
            "batch_retarget.scores_unavailable_after_estimate",
            "this plan's estimate carried no score table (a trimmed or non-iid AIPW, or an \
             estimator that keeps no cross-fitted scores), so there is nothing to retarget; \
             the prepare-time rows are never substituted for the estimate's rows",
        ),
        ScoreSource::Prepared => (
            "batch_retarget.scores_unavailable",
            "this plan prepared no score table; retargeting needs an AllObserved iid AIPW or \
             cell-AIPW plan",
        ),
    };
    BatchRetargetError {
        code: reason_code!("score_table_unavailable"),
        detail,
        message: message.into(),
        subject: None,
    }
}

/// Where the score tables of a [`BatchScores`] came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreSource {
    /// Frozen at prepare time.
    Prepared,
    /// Produced by an estimate on a later table.
    Estimated,
}

impl ScoreSource {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Estimated => "estimated",
        }
    }
}

/// One score-table slot per plan of a prepared batch, with the source they came from.
#[derive(Clone, Debug)]
pub struct BatchScores {
    source: ScoreSource,
    tables: Vec<Option<ScoreTable>>,
}

impl BatchScores {
    pub(crate) fn new(source: ScoreSource, tables: Vec<Option<ScoreTable>>) -> Self {
        Self { source, tables }
    }

    /// Where these tables came from.
    #[must_use]
    pub const fn source(&self) -> ScoreSource {
        self.source
    }

    /// The table of plan `index`, when it has one.
    #[must_use]
    pub fn table(&self, index: usize) -> Option<&ScoreTable> {
        self.tables.get(index).and_then(Option::as_ref)
    }

    /// The complete-case original row index every present table shares: the row alignment of
    /// a retarget's weights. `None` when no plan has a table.
    ///
    /// # Errors
    ///
    /// `row_weights_bound_to_snapshot` (`batch_retarget.mixed_snapshot`) when two tables
    /// were built on different rows.
    pub fn common_rows(&self) -> Result<Option<Arc<[u32]>>, BatchRetargetError> {
        let mut present = self.tables.iter().flatten();
        let Some(first) = present.next() else {
            return Ok(None);
        };
        if present.any(|t| t.n_rows != first.n_rows || t.row_index != first.row_index) {
            return Err(mixed_snapshot(
                "the plans' score tables were built on different complete-case rows, so one \
                 weight vector cannot align with every claim and their covariance is not \
                 defined; estimate the family on one snapshot",
            ));
        }
        Ok(Some(Arc::clone(&first.row_index)))
    }

    /// Identity of the common row snapshot: a digest of the shared row index, or `None` when
    /// no plan has a table.
    ///
    /// # Errors
    ///
    /// As [`Self::common_rows`].
    pub fn snapshot_id(&self) -> Result<Option<String>, BatchRetargetError> {
        Ok(self
            .common_rows()?
            .map(|rows| hex(&PayloadDigestWire::u32s("batch.retarget_rows", &rows).digest)))
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// One declared retargeted claim.
#[derive(Clone, Debug)]
pub struct RetargetClaim {
    /// Unique claim name (also unique against contrast names).
    pub name: String,
    /// Index of the batch query this claim reweights.
    pub query_index: usize,
    /// Target row weights, aligned with [`BatchScores::common_rows`].
    pub weights: Vec<f64>,
    /// Declared parents of the weights; must lie in the certified adjustment set and must
    /// not name the treatment, an intervened coordinate or a descendant.
    pub depends_on: Vec<VariableId>,
}

/// One named linear contrast of claims, `Σ c_k θ_k`.
#[derive(Clone, Debug)]
pub struct RetargetContrast {
    /// Unique contrast name.
    pub name: String,
    /// `(claim name, coefficient)` terms; each claim at most once.
    pub coefficients: Vec<(String, f64)>,
}

/// A declared family: claims, contrasts and, optionally, the snapshot they were declared on.
#[derive(Clone, Debug, Default)]
pub struct BatchRetargetRequest {
    /// The claims.
    pub claims: Vec<RetargetClaim>,
    /// The named contrasts among them.
    pub contrasts: Vec<RetargetContrast>,
    /// When set, the snapshot id the caller declared; a different common snapshot refuses.
    pub expected_snapshot: Option<String>,
}

/// Whether a member's uncertainty is a score covariance or absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UncertaintyKind {
    /// Plug-in score covariance (a standard error, never an interval).
    PlugInScoreCovariance,
    /// A point only.
    None,
}

impl UncertaintyKind {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PlugInScoreCovariance => "plug_in_score_covariance",
            Self::None => "none",
        }
    }
}

/// Why a member could not be retargeted. Absent fields stay absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberFailure {
    /// Registered reason code, when the refusal carries one.
    pub reason_code: Option<String>,
    /// Stable `batch_retarget.*` detail, when the refusal carries one.
    pub detail: Option<String>,
    /// Explanation.
    pub message: String,
    /// Whether the refusal is a weighted-overlap support refusal.
    pub support_refused: bool,
    /// Structured diagnostics of the failing member: stage, the claim or contrast as
    /// subject, the reason and, for a failed overlap gate, the per-arm effective sample
    /// size and propensity range that gate measured. Absent entries stay absent.
    pub fields: Option<Box<RefusalFields>>,
}

impl MemberFailure {
    /// This failure about a named claim or contrast.
    fn about(mut self, subject: &str) -> Self {
        if let Some(fields) = self.fields.as_mut() {
            fields.subject = Some(subject.to_string());
        }
        self
    }
}

impl From<BatchRetargetError> for MemberFailure {
    fn from(error: BatchRetargetError) -> Self {
        let fields = Some(Box::new(error.refusal_fields()));
        Self {
            reason_code: Some(error.code.to_string()),
            detail: Some(error.detail.to_string()),
            message: error.message,
            support_refused: false,
            fields,
        }
    }
}

fn member_failure(error: &CausalError) -> MemberFailure {
    let carried = error.refusal_fields().map(|fields| Box::new(fields.into_owned()));
    if matches!(error.peeled(), CausalError::Support { .. }) {
        let mut failure = MemberFailure::from(not_licensed(
            "batch_retarget.weighted_overlap_failed",
            error.to_string(),
        ));
        failure.support_refused = true;
        // The overlap gate's own figures replace the detail-only defaults.
        if let (Some(fields), Some(carried)) = (failure.fields.as_mut(), carried) {
            fields.arm_ess = carried.arm_ess;
            fields.propensity_min = carried.propensity_min;
            fields.propensity_max = carried.propensity_max;
            if carried.reason.is_some() {
                fields.reason = carried.reason;
            }
            fields.remedy = carried.remedy;
        }
        return failure;
    }
    MemberFailure {
        reason_code: error.reason_code().map(str::to_owned),
        detail: None,
        message: error.to_string(),
        support_refused: false,
        fields: carried,
    }
}

/// A retargeted claim's point and support.
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimPoint {
    /// The retargeted point `θ_k`.
    pub value: f64,
    /// Plug-in standard error `sqrt(Σ_kk)`, absent for a point-only member.
    pub std_error: Option<f64>,
    /// Kind of the uncertainty carried.
    pub uncertainty_kind: UncertaintyKind,
    /// Kish effective sample size of the target weights.
    pub n_eff: f64,
    /// Kish effective sample size per observed arm under the target.
    pub n_eff_by_arm: Vec<f64>,
    /// Range of the raw held-out propensities where the target has mass.
    pub propensity_range: Option<(f64, f64)>,
    /// The score table's nuisance provenance.
    pub nuisance_provenance: String,
    /// Notes (a penalized member's withheld covariance).
    pub diagnostics: Vec<String>,
}

/// One claim of a family report.
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimReport {
    /// Claim name.
    pub name: String,
    /// Batch query index.
    pub query_index: usize,
    /// The estimand, as an average-effect or joint-cell description over variable ids.
    pub estimand: String,
    /// Its point, or why it failed.
    pub outcome: Result<ClaimPoint, MemberFailure>,
}

/// A contrast's value.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastPoint {
    /// `Σ c_k θ_k`.
    pub value: f64,
    /// `sqrt(c'Σc)`, present only when every term carries covariance.
    pub std_error: Option<f64>,
    /// Kind of the uncertainty carried.
    pub uncertainty_kind: UncertaintyKind,
}

/// One contrast of a family report.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastReport {
    /// Contrast name.
    pub name: String,
    /// The `(claim, coefficient)` terms as declared.
    pub terms: Vec<(String, f64)>,
    /// Its value, or why it failed.
    pub outcome: Result<ContrastPoint, MemberFailure>,
}

/// The joint covariance of the covariance-bearing members, in declared claim order.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyCovariance {
    /// Claim names indexing the matrix rows and columns.
    pub names: Vec<String>,
    /// Column-major covariance.
    pub matrix: JointCovariance,
}

impl FamilyCovariance {
    /// Entry for two named claims, when both are in the covariance.
    #[must_use]
    pub fn get(&self, a: &str, b: &str) -> Option<f64> {
        let i = self.names.iter().position(|n| n == a)?;
        let j = self.names.iter().position(|n| n == b)?;
        Some(self.matrix.get(i, j))
    }
}

/// The report of a batch retarget.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchRetargetReport {
    /// Order-invariant identity of the declared family and its snapshot.
    pub family_id: String,
    /// Identity of the common row snapshot, when any plan had scores.
    pub snapshot_id: Option<String>,
    /// Whether the scores were prepared or estimated.
    pub scores_source: ScoreSource,
    /// Canonical estimator-configuration fingerprint of the batch.
    pub estimator_fingerprint: String,
    /// Every declared claim, failed ones included, in declared order.
    pub claims: Vec<ClaimReport>,
    /// Every declared contrast, failed ones included, in declared order.
    pub contrasts: Vec<ContrastReport>,
    /// Joint covariance of the covariance-bearing members.
    pub covariance: Option<FamilyCovariance>,
    /// Scope note.
    pub scope_note: &'static str,
}

impl BatchRetargetReport {
    /// Names of the claims and contrasts that failed.
    #[must_use]
    pub fn failed_members(&self) -> Vec<&str> {
        let claims = self.claims.iter().filter(|c| c.outcome.is_err()).map(|c| c.name.as_str());
        let contrasts =
            self.contrasts.iter().filter(|c| c.outcome.is_err()).map(|c| c.name.as_str());
        claims.chain(contrasts).collect()
    }

    /// Names of the claims reported as a point without covariance.
    #[must_use]
    pub fn point_only_members(&self) -> Vec<&str> {
        self.claims
            .iter()
            .filter(|c| c.outcome.as_ref().is_ok_and(|p| p.std_error.is_none()))
            .map(|c| c.name.as_str())
            .collect()
    }

    /// The complete-family covariance: every declared claim succeeded and carries covariance.
    ///
    /// # Errors
    ///
    /// `cell_not_licensed` naming the failed members (`batch_retarget.partial_family`) or
    /// the point-only members (`batch_retarget.point_only_member`).
    pub fn complete_family(&self) -> Result<&FamilyCovariance, BatchRetargetError> {
        let failed = self.failed_members();
        if !failed.is_empty() {
            return Err(not_licensed(
                "batch_retarget.partial_family",
                format!(
                    "the family is partial: {} failed, so the surviving members are not a \
                     complete-family claim",
                    failed.join(", ")
                ),
            ));
        }
        let point_only = self.point_only_members();
        if !point_only.is_empty() {
            return Err(not_licensed(
                "batch_retarget.point_only_member",
                format!(
                    "{} retarget to a point only (a penalized propensity publishes no \
                     covariance), so the family has no complete joint covariance",
                    point_only.join(", ")
                ),
            ));
        }
        self.covariance.as_ref().ok_or_else(|| {
            not_supported(
                "batch_retarget.covariance_unavailable",
                "the family carries no covariance-bearing member",
            )
        })
    }
}

struct Evaluated {
    point: ClaimPoint,
    /// Weighted influence values `ξ`; present only for a covariance-bearing member.
    influence: Option<Vec<f64>>,
}

fn estimand_label(query: &BatchQuery) -> String {
    match query {
        BatchQuery::Average(q) => format!(
            "average_effect(treatment=v{}, outcome=v{}, active={:?}, control={:?})",
            q.treatment.raw(),
            q.outcome.raw(),
            q.active,
            q.control
        ),
        BatchQuery::Response(q) => {
            let treatments: Vec<String> =
                q.functional.treatment_ids().iter().map(|t| format!("v{}", t.raw())).collect();
            let outcomes: Vec<String> =
                q.functional.outcome_ids().iter().map(|o| format!("v{}", o.raw())).collect();
            format!(
                "joint_cell(treatments=[{}], outcome=[{}])",
                treatments.join(","),
                outcomes.join(",")
            )
        }
    }
}

/// Weighted influence values `ξ_r = sqrt(n/(n-1)) · (w_r / Σw) · (φ_r − θ)`.
fn weighted_influence(scores: &[f64], weights: &[f64], theta: f64) -> Vec<f64> {
    let n = scores.len() as f64;
    let scale = weights.iter().copied().fold(0.0_f64, f64::max);
    let total: f64 = weights.iter().map(|w| w / scale).sum();
    let correction = (n / (n - 1.0)).sqrt();
    scores
        .iter()
        .zip(weights)
        .map(|(phi, w)| correction * ((w / scale) / total) * (phi - theta))
        .collect()
}

fn evaluate_claim(
    plan: &PreparedStudy,
    query: &BatchQuery,
    table: Option<&ScoreTable>,
    claim: &RetargetClaim,
    source: ScoreSource,
) -> Result<Evaluated, MemberFailure> {
    let table = table.ok_or_else(|| MemberFailure::from(scores_unavailable(source)))?;
    if table.distinct_threshold_count() > 0 {
        return Err(MemberFailure::from(not_supported(
            "batch_retarget.functional_not_licensed",
            "a batch retarget covers the mean functional; a quantile or exceedance grid keeps \
             its single-claim retarget",
        )));
    }
    let n = table.n_rows;
    if claim.weights.len() != n || n < 2 || claim.weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return Err(MemberFailure::from(invalid(
            "batch_retarget.incompatible_target",
            format!(
                "claim {:?} declares {} weights for {n} snapshot rows; weights must be finite, \
                 non-negative and aligned with the common rows",
                claim.name,
                claim.weights.len()
            ),
        )));
    }
    let out = plan
        .retarget_table(table, &claim.weights, &claim.depends_on)
        .map_err(|e| member_failure(&e))?;
    let (value, coefficients) = match query {
        BatchQuery::Average(_) => {
            let contrast = out.contrast.as_ref().ok_or_else(|| {
                MemberFailure::from(not_supported(
                    "batch_retarget.functional_not_licensed",
                    "the plan's scores carry no single arm contrast",
                ))
            })?;
            let column =
                |arm: u32| table.columns.iter().position(|c| c.arm == arm && c.threshold.is_none());
            let (Some(control), Some(active)) = (column(0), column(1)) else {
                return Err(MemberFailure::from(not_supported(
                    "batch_retarget.functional_not_licensed",
                    "the plan's scores carry no arm-0 and arm-1 mean columns",
                )));
            };
            let mut coefficients = vec![0.0; table.n_columns()];
            coefficients[control] = -1.0;
            coefficients[active] = 1.0;
            (contrast.value, coefficients)
        }
        BatchQuery::Response(q) => {
            let arm = super::helpers::requested_joint_arm(q).map_err(|e| member_failure(&e))?;
            let Some(column) = table.columns.iter().position(|c| c.arm == arm) else {
                return Err(MemberFailure::from(not_supported(
                    "batch_retarget.functional_not_licensed",
                    "the requested joint cell is not a column of the plan's scores",
                )));
            };
            let mut coefficients = vec![0.0; table.n_columns()];
            coefficients[column] = 1.0;
            (out.summary.means[column], coefficients)
        }
    };
    let scores =
        table.combine_scores(&coefficients).map_err(|e| member_failure(&CausalError::from(e)))?;
    let withheld = provenance_withholds_interval(&table.nuisance_provenance);
    let mut diagnostics = Vec::new();
    if withheld {
        diagnostics.push(
            "the scores come from a penalized propensity: a retargeted point, no covariance"
                .to_string(),
        );
    }
    Ok(Evaluated {
        influence: (!withheld).then(|| weighted_influence(&scores, &claim.weights, value)),
        point: ClaimPoint {
            value,
            std_error: None,
            uncertainty_kind: UncertaintyKind::None,
            n_eff: out.support.n_eff,
            n_eff_by_arm: out.support.n_eff_by_arm.clone(),
            propensity_range: out.support.propensity_range,
            nuisance_provenance: table.nuisance_provenance.to_string(),
            diagnostics,
        },
    })
}

fn validate_request(
    request: &BatchRetargetRequest,
    plans: usize,
) -> Result<(), BatchRetargetError> {
    if request.claims.is_empty() {
        return Err(invalid("batch_retarget.empty_family", "a family declares at least one claim"));
    }
    let mut names = std::collections::BTreeSet::new();
    for name in
        request.claims.iter().map(|c| &c.name).chain(request.contrasts.iter().map(|c| &c.name))
    {
        if name.is_empty() || !names.insert(name.as_str()) {
            return Err(invalid(
                "batch_retarget.duplicate_name",
                format!("claim and contrast names must be nonempty and unique, got {name:?}"),
            )
            .about(name));
        }
    }
    if let Some(claim) = request.claims.iter().find(|c| c.query_index >= plans) {
        return Err(invalid(
            "batch_retarget.unknown_claim",
            format!(
                "claim {:?} names query {} but the batch has {plans} plans",
                claim.name, claim.query_index
            ),
        )
        .about(&claim.name));
    }
    for contrast in &request.contrasts {
        let mut seen = std::collections::BTreeSet::new();
        if contrast.coefficients.is_empty()
            || contrast.coefficients.iter().any(|(_, c)| !c.is_finite())
            || contrast.coefficients.iter().any(|(n, _)| !seen.insert(n.as_str()))
        {
            return Err(invalid(
                "batch_retarget.invalid_contrast",
                format!(
                    "contrast {:?} needs finite coefficients on distinct claims",
                    contrast.name
                ),
            )
            .about(&contrast.name));
        }
        if let Some((missing, _)) =
            contrast.coefficients.iter().find(|(n, _)| !request.claims.iter().any(|c| &c.name == n))
        {
            return Err(invalid(
                "batch_retarget.unknown_claim",
                format!("contrast {:?} names the undeclared claim {missing:?}", contrast.name),
            )
            .about(&contrast.name));
        }
    }
    Ok(())
}

/// Order-invariant identity of the family: claims and contrasts are digested in name order,
/// with every weight and coefficient as exact IEEE bits, beside the snapshot and the scores'
/// source.
fn family_id(
    request: &BatchRetargetRequest,
    snapshot: Option<&str>,
    source: ScoreSource,
    fingerprint: &str,
) -> String {
    let mut bytes = b"antecedent.batch_retarget.v1\0".to_vec();
    let push_str = |bytes: &mut Vec<u8>, s: &str| {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    };
    push_str(&mut bytes, snapshot.unwrap_or(""));
    push_str(&mut bytes, source.as_str());
    push_str(&mut bytes, fingerprint);
    let mut claims: Vec<&RetargetClaim> = request.claims.iter().collect();
    claims.sort_by(|a, b| a.name.cmp(&b.name));
    bytes.extend_from_slice(&(claims.len() as u64).to_le_bytes());
    for claim in claims {
        push_str(&mut bytes, &claim.name);
        bytes.extend_from_slice(&(claim.query_index as u64).to_le_bytes());
        bytes.extend_from_slice(&(claim.weights.len() as u64).to_le_bytes());
        for w in &claim.weights {
            bytes.extend_from_slice(&w.to_bits().to_le_bytes());
        }
        let mut parents: Vec<u32> = claim.depends_on.iter().map(|v| v.raw()).collect();
        parents.sort_unstable();
        bytes.extend_from_slice(&(parents.len() as u64).to_le_bytes());
        for p in parents {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
    }
    let mut contrasts: Vec<&RetargetContrast> = request.contrasts.iter().collect();
    contrasts.sort_by(|a, b| a.name.cmp(&b.name));
    bytes.extend_from_slice(&(contrasts.len() as u64).to_le_bytes());
    for contrast in contrasts {
        push_str(&mut bytes, &contrast.name);
        let mut terms: Vec<&(String, f64)> = contrast.coefficients.iter().collect();
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        bytes.extend_from_slice(&(terms.len() as u64).to_le_bytes());
        for (name, c) in terms {
            push_str(&mut bytes, name);
            bytes.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    hex(&PayloadDigestWire::bytes("batch.retarget_family", &bytes).digest)
}

/// Gram matrix `Σ_kl = Σ_r ξ_kr ξ_lr` of the weighted influence columns, column-major.
fn gram(columns: &[&[f64]]) -> Result<JointCovariance, BatchRetargetError> {
    let dim = columns.len();
    let mut values = vec![0.0; dim * dim];
    for j in 0..dim {
        for i in 0..=j {
            let cov: f64 = columns[i].iter().zip(columns[j]).map(|(a, b)| a * b).sum();
            values[j * dim + i] = cov;
            values[i * dim + j] = cov;
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(not_supported(
            "batch_retarget.covariance_unavailable",
            "the joint score covariance is not finite",
        ));
    }
    Ok(JointCovariance { dim, values: Arc::from(values) })
}

fn contrast_outcome(
    contrast: &RetargetContrast,
    claims: &[ClaimReport],
    covariance: Option<&FamilyCovariance>,
) -> Result<ContrastPoint, MemberFailure> {
    let mut value = 0.0;
    let mut all_bearing = true;
    for (name, c) in &contrast.coefficients {
        let point = claims
            .iter()
            .find(|claim| &claim.name == name)
            .and_then(|claim| claim.outcome.as_ref().ok())
            .ok_or_else(|| {
                MemberFailure::from(not_licensed(
                    "batch_retarget.contrast_member_failed",
                    format!("contrast {:?} reads claim {name:?}, which failed", contrast.name),
                ))
            })?;
        value += c * point.value;
        all_bearing &= point.std_error.is_some();
    }
    if !all_bearing {
        return Ok(ContrastPoint {
            value,
            std_error: None,
            uncertainty_kind: UncertaintyKind::None,
        });
    }
    let covariance = covariance.ok_or_else(|| {
        MemberFailure::from(not_supported(
            "batch_retarget.covariance_unavailable",
            "no joint covariance for the contrast's claims",
        ))
    })?;
    let mut variance = 0.0;
    let mut absolute = 0.0;
    for (a, ca) in &contrast.coefficients {
        for (b, cb) in &contrast.coefficients {
            let term = ca * cb * covariance.get(a, b).unwrap_or(f64::NAN);
            variance += term;
            absolute += term.abs();
        }
    }
    if !variance.is_finite() || variance < -1e-12 * absolute {
        return Err(MemberFailure::from(not_supported(
            "batch_retarget.covariance_unavailable",
            "the contrast variance is not a finite non-negative number",
        )));
    }
    Ok(ContrastPoint {
        value,
        std_error: Some(variance.max(0.0).sqrt()),
        uncertainty_kind: UncertaintyKind::PlugInScoreCovariance,
    })
}

/// Retarget a declared family over `scores` (see [`super::PreparedBatch::retarget`]).
pub(crate) fn retarget_family(
    plans: &[PreparedStudy],
    queries: &[BatchQuery],
    scores: &BatchScores,
    request: &BatchRetargetRequest,
    estimator_fingerprint: &str,
    ctx: &ExecutionContext,
) -> Result<BatchRetargetReport, BatchRetargetError> {
    validate_request(request, plans.len())?;
    if scores.tables.len() != plans.len() {
        return Err(invalid(
            "batch_retarget.scores_mismatch",
            format!(
                "the scores hold {} slots for {} plans; retain the scores of this batch",
                scores.tables.len(),
                plans.len()
            ),
        ));
    }
    let snapshot = scores.snapshot_id()?;
    if let Some(expected) = request.expected_snapshot.as_deref() {
        if snapshot.as_deref() != Some(expected) {
            return Err(mixed_snapshot(format!(
                "the family was declared on snapshot {expected} but the scores are on {}",
                snapshot.as_deref().unwrap_or("no snapshot")
            )));
        }
    }

    let mut evaluated: Vec<Result<Evaluated, MemberFailure>> =
        Vec::with_capacity(request.claims.len());
    for claim in &request.claims {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        evaluated.push(
            evaluate_claim(
                &plans[claim.query_index],
                &queries[claim.query_index],
                scores.table(claim.query_index),
                claim,
                scores.source,
            )
            .map_err(|failure| failure.about(&claim.name)),
        );
    }

    // Joint covariance of the covariance-bearing members, in declared order.
    let bearing: Vec<(usize, &[f64])> = evaluated
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.as_ref().ok().and_then(|e| e.influence.as_deref()).map(|x| (i, x)))
        .collect();
    let covariance = if bearing.is_empty() {
        None
    } else {
        let columns: Vec<&[f64]> = bearing.iter().map(|(_, x)| *x).collect();
        Some(FamilyCovariance {
            names: bearing.iter().map(|(i, _)| request.claims[*i].name.clone()).collect(),
            matrix: gram(&columns)?,
        })
    };

    let claims: Vec<ClaimReport> = request
        .claims
        .iter()
        .zip(evaluated)
        .map(|(claim, result)| {
            let outcome = result.map(|e| {
                let mut point = e.point;
                if e.influence.is_some() {
                    let se = covariance
                        .as_ref()
                        .and_then(|c| c.get(&claim.name, &claim.name))
                        .map_or(f64::NAN, f64::sqrt);
                    point.std_error = Some(se);
                    point.uncertainty_kind = UncertaintyKind::PlugInScoreCovariance;
                }
                point
            });
            ClaimReport {
                name: claim.name.clone(),
                query_index: claim.query_index,
                estimand: estimand_label(&queries[claim.query_index]),
                outcome,
            }
        })
        .collect();
    let contrasts = request
        .contrasts
        .iter()
        .map(|c| ContrastReport {
            name: c.name.clone(),
            terms: c.coefficients.clone(),
            outcome: contrast_outcome(c, &claims, covariance.as_ref())
                .map_err(|failure| failure.about(&c.name)),
        })
        .collect();

    Ok(BatchRetargetReport {
        family_id: family_id(request, snapshot.as_deref(), scores.source, estimator_fingerprint),
        snapshot_id: snapshot,
        scores_source: scores.source,
        estimator_fingerprint: estimator_fingerprint.to_string(),
        claims,
        contrasts,
        covariance,
        scope_note: BATCH_RETARGET_SCOPE_NOTE,
    })
}

/// Fewest Monte-Carlo draws the unpublished max-t evaluator accepts.
pub const MAX_T_MIN_DRAWS: u32 = 1_000;

/// Most Monte-Carlo draws the unpublished max-t evaluator accepts (the draws' maxima are
/// held in memory to take their empirical quantile).
pub const MAX_T_MAX_DRAWS: u32 = 2_000_000;

/// Slack on a correlation matrix's unit diagonal and unit bound.
const CORRELATION_SLACK: f64 = 1e-9;

/// The correlation matrix of a family covariance, column-major with an exactly symmetric
/// off-diagonal.
fn correlation_of(family: &FamilyCovariance) -> Result<JointCovariance, BatchRetargetError> {
    let matrix = &family.matrix;
    let dim = matrix.dim;
    let se: Vec<f64> = (0..dim).map(|i| matrix.se(i)).collect();
    if dim == 0 || se.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return Err(not_supported(
            "batch_retarget.covariance_unavailable",
            "a max-t band needs every claim's plug-in standard error positive and finite",
        ));
    }
    let mut values = vec![1.0; dim * dim];
    for j in 0..dim {
        for i in (j + 1)..dim {
            let r = (matrix.get(i, j) / (se[i] * se[j])).clamp(-1.0, 1.0);
            values[j * dim + i] = r;
            values[i * dim + j] = r;
        }
    }
    Ok(JointCovariance { dim, values: Arc::from(values) })
}

/// Monte-Carlo max-t critical value `c = q_level(max_j |Z_j|)`, `Z ~ N(0, R)`, for a
/// correlation matrix `R`.
///
/// The evaluator of the family-level simultaneous interval, reached through
/// [`BatchRetargetReport::simultaneous_interval`]. It reuses the library's one max-t sampler
/// ([`max_t_critical_polled`]) on a stream seeded from `seed`, so the value is a
/// deterministic function of `(R, level, seed, draws)`; the Monte-Carlo error of the
/// empirical quantile is not part of any interval claim.
///
/// # Errors
///
/// `invalid_argument` (`batch_retarget.max_t_invalid_level`) for a level outside (0, 1);
/// `invalid_argument` (`batch_retarget.max_t_draws_out_of_range`) for fewer than
/// [`MAX_T_MIN_DRAWS`] or more than [`MAX_T_MAX_DRAWS`] draws, or too few for the level's
/// empirical quantile to exist; `route_not_supported`
/// (`batch_retarget.covariance_unavailable`) when `correlation` is not a symmetric positive
/// semidefinite matrix with a unit diagonal; `cancelled_no_claim`
/// (`batch_retarget.cancelled`) when the context is cancelled before every draw is made.
#[doc(hidden)]
pub fn max_t_critical_value(
    correlation: &JointCovariance,
    level: f64,
    seed: u64,
    draws: u32,
    ctx: &ExecutionContext,
) -> Result<f64, BatchRetargetError> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(invalid(
            "batch_retarget.max_t_invalid_level",
            format!("the max-t level must lie strictly between 0 and 1, got {level}"),
        ));
    }
    if !(MAX_T_MIN_DRAWS..=MAX_T_MAX_DRAWS).contains(&draws) {
        return Err(invalid(
            "batch_retarget.max_t_draws_out_of_range",
            format!(
                "the max-t evaluator takes between {MAX_T_MIN_DRAWS} and {MAX_T_MAX_DRAWS} \
                 draws, got {draws}"
            ),
        ));
    }
    let dim = correlation.dim;
    let well_formed = dim > 0
        && correlation.values.len() == dim * dim
        && (0..dim).all(|i| {
            (correlation.get(i, i) - 1.0).abs() <= CORRELATION_SLACK
                && (0..dim).all(|j| correlation.get(i, j).abs() <= 1.0 + CORRELATION_SLACK)
        });
    if !well_formed {
        return Err(not_supported(
            "batch_retarget.covariance_unavailable",
            "the max-t evaluator needs a finite correlation matrix with a unit diagonal and \
             entries in [-1, 1]",
        ));
    }
    let mut rng = CausalRng::from_seed(seed);
    let critical = max_t_critical_polled(correlation, level, draws, &mut rng, &|| {
        ctx.cancellation.is_cancelled()
    })
    .map_err(|e| not_supported("batch_retarget.covariance_unavailable", e.to_string()))?
    .ok_or_else(cancelled)?;
    if !critical.is_finite() {
        return Err(invalid(
            "batch_retarget.max_t_draws_out_of_range",
            format!("{draws} draws are too few for an empirical {level} quantile"),
        ));
    }
    Ok(critical)
}

/// One claim of a [`SimultaneousBand`]: `point ± c · se`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandMember {
    /// Claim name.
    pub name: String,
    /// The retargeted point.
    pub value: f64,
    /// Plug-in standard error `sqrt(Σ_jj)`.
    pub std_error: f64,
    /// `value − c · std_error`.
    pub lower: f64,
    /// `value + c · std_error`.
    pub upper: f64,
}

/// A simultaneous max-t band over a complete claim family: nominal asymptotic, on the
/// plug-in score covariance, with a Monte-Carlo critical value (its error is not part of the
/// band). Measured coverage belongs to the calibration harness.
#[derive(Clone, Debug, PartialEq)]
pub struct SimultaneousBand {
    /// The nominal level the critical value was taken at.
    pub level: f64,
    /// Seed of the Monte-Carlo stream.
    pub seed: u64,
    /// Number of Monte-Carlo draws.
    pub draws: u32,
    /// The max-t critical value `c`.
    pub critical_value: f64,
    /// One entry per claim, in the covariance's claim order.
    pub members: Vec<BandMember>,
}

/// Calibration-wiring entry to [`BatchRetargetReport::simultaneous_interval`] (same
/// evaluator; kept under its original name for the wired coverage harness).
///
/// # Errors
///
/// As [`BatchRetargetReport::simultaneous_interval`].
#[doc(hidden)]
pub fn simultaneous_band_unpublished(
    report: &BatchRetargetReport,
    level: f64,
    seed: u64,
    draws: u32,
    ctx: &ExecutionContext,
) -> Result<SimultaneousBand, BatchRetargetError> {
    report.simultaneous_interval(level, seed, draws, ctx)
}

impl BatchRetargetReport {
    /// The max-t band `point_j ± c · se_j` of a complete family from its plug-in score
    /// covariance, with `c` the `level` quantile of `max_j |Z_j|`, `Z ~ N(0, R)`, `R` the
    /// correlation matrix of the covariance. `c` is a deterministic function of
    /// `(R, level, seed, draws)`.
    ///
    /// Nominal asymptotic: the Gaussian limit of the studentized retargeted points under the
    /// single-claim retarget's conditions (iid rows, fixed declared weights, positivity,
    /// nuisance convergence); selection and weight-estimation uncertainty and the
    /// Monte-Carlo error of `c` are excluded.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::complete_family`] (`batch_retarget.partial_family`,
    /// `batch_retarget.point_only_member`, `batch_retarget.covariance_unavailable`) and of
    /// [`max_t_critical_value`].
    pub fn simultaneous_interval(
        &self,
        level: f64,
        seed: u64,
        draws: u32,
        ctx: &ExecutionContext,
    ) -> Result<SimultaneousBand, BatchRetargetError> {
        let family = self.complete_family()?;
        let critical_value =
            max_t_critical_value(&correlation_of(family)?, level, seed, draws, ctx)?;
        let members = family
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let value = self
                    .claims
                    .iter()
                    .find(|c| c.name == *name)
                    .and_then(|c| c.outcome.as_ref().ok())
                    .map(|p| p.value)
                    .ok_or_else(|| {
                        not_supported(
                            "batch_retarget.covariance_unavailable",
                            format!("covariance claim {name} has no reported point"),
                        )
                    })?;
                let std_error = family.matrix.se(i);
                let half = critical_value * std_error;
                Ok(BandMember {
                    name: name.clone(),
                    value,
                    std_error,
                    lower: value - half,
                    upper: value + half,
                })
            })
            .collect::<Result<Vec<_>, BatchRetargetError>>()?;
        Ok(SimultaneousBand { level, seed, draws, critical_value, members })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    /// Hand calculation. Unweighted, n = 4: φ1 = (1, 2, 3, 6) has θ1 = 3 and deviations
    /// (−2, −1, 0, 3); φ2 = (0, 2, 2, 0) has θ2 = 1 and deviations (−1, 1, 1, −1). With
    /// ξ = sqrt(4/3) · (1/4) · d the covariance is Σ d d' / (n (n − 1)):
    /// Σ11 = 14/12, Σ22 = 4/12, Σ12 = (2 − 1 + 0 − 3)/12 = −2/12.
    #[test]
    fn the_covariance_of_two_claims_matches_a_hand_calculation() {
        let (p1, p2) = ([1.0, 2.0, 3.0, 6.0], [0.0, 2.0, 2.0, 0.0]);
        let ones = [1.0; 4];
        let x1 = weighted_influence(&p1, &ones, 3.0);
        let x2 = weighted_influence(&p2, &ones, 1.0);
        let cov = gram(&[&x1[..], &x2[..]]).unwrap();
        near(cov.get(0, 0), 14.0 / 12.0);
        near(cov.get(1, 1), 4.0 / 12.0);
        near(cov.get(0, 1), -2.0 / 12.0);
        assert_eq!(cov.get(0, 1).to_bits(), cov.get(1, 0).to_bits());
        // The diagonal is the single-claim plug-in variance of the library's joint IF.
        let library =
            antecedent_estimate::joint_influence_covariance(&[&p1[..], &p2[..]], None).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                near(cov.get(i, j), library.get(i, j));
            }
        }
    }

    /// Weighted, n = 4, w = (1, 1, 2, 0): W = 4, θ = (1 + 2 + 6 + 0)/4 = 9/4, deviations
    /// (−5/4, −1/4, 3/4, 15/4), a = w/W = (1/4, 1/4, 1/2, 0), so
    /// Σ = (4/3) · (25/256 + 1/256 + 36/256 + 0) = (4/3) · (31/128) = 31/96.
    #[test]
    fn a_weighted_variance_matches_a_hand_calculation_and_the_library() {
        let phi = [1.0, 2.0, 3.0, 6.0];
        let w = [1.0, 1.0, 2.0, 0.0];
        let xi = weighted_influence(&phi, &w, 2.25);
        let cov = gram(&[&xi[..]]).unwrap();
        near(cov.get(0, 0), 31.0 / 96.0);
        let library =
            antecedent_estimate::joint_influence_covariance(&[&phi[..]], Some(&w[..])).unwrap();
        near(cov.get(0, 0), library.get(0, 0));
    }

    fn table(rows: &[u32]) -> ScoreTable {
        let n = rows.len();
        ScoreTable {
            observed_arm: Arc::from(vec![0; n]),
            propensities: Arc::from(vec![0.5; 2 * n]),
            observed_outcome: Arc::from(vec![0.0; n]),
            n_rows: n,
            row_index: Arc::from(rows.to_vec()),
            fold_ids: Arc::from(vec![0; n]),
            n_folds: 2,
            scores: Arc::from(vec![0.0; 2 * n]),
            columns: Arc::from(vec![
                antecedent_estimate::ScoreColumn { arm: 0, threshold: None },
                antecedent_estimate::ScoreColumn { arm: 1, threshold: None },
            ]),
            adjustment_set: Arc::from(Vec::new()),
            nuisance_provenance: Arc::from("test"),
            propensity_clip: None,
            treatment: VariableId::from_raw(0),
            intervened: Arc::from(Vec::new()),
        }
    }

    #[test]
    fn tables_on_different_rows_are_a_mixed_snapshot() {
        let same = BatchScores::new(
            ScoreSource::Estimated,
            vec![Some(table(&[0, 1, 2])), None, Some(table(&[0, 1, 2]))],
        );
        assert_eq!(&*same.common_rows().unwrap().unwrap(), &[0, 1, 2]);
        assert_eq!(same.snapshot_id().unwrap(), same.snapshot_id().unwrap());
        for other in [table(&[0, 1, 3]), table(&[0, 1])] {
            let mixed =
                BatchScores::new(ScoreSource::Prepared, vec![Some(table(&[0, 1, 2])), Some(other)]);
            let error = mixed.common_rows().unwrap_err();
            assert_eq!(error.code, "row_weights_bound_to_snapshot");
            assert_eq!(error.detail, "batch_retarget.mixed_snapshot");
            assert!(mixed.snapshot_id().is_err());
        }
        let none = BatchScores::new(ScoreSource::Prepared, vec![None]);
        assert!(none.common_rows().unwrap().is_none());
        assert_ne!(
            BatchScores::new(ScoreSource::Prepared, vec![Some(table(&[0, 1, 2]))])
                .snapshot_id()
                .unwrap(),
            BatchScores::new(ScoreSource::Prepared, vec![Some(table(&[0, 1, 4]))])
                .snapshot_id()
                .unwrap()
        );
    }

    fn claim(name: &str, query_index: usize, weights: &[f64]) -> RetargetClaim {
        RetargetClaim {
            name: name.into(),
            query_index,
            weights: weights.to_vec(),
            depends_on: vec![VariableId::from_raw(2), VariableId::from_raw(1)],
        }
    }

    #[test]
    fn the_family_id_is_order_invariant_and_binds_every_input() {
        let contrast = |name: &str, terms: &[(&str, f64)]| RetargetContrast {
            name: name.into(),
            coefficients: terms.iter().map(|(n, c)| ((*n).to_string(), *c)).collect(),
        };
        let request = BatchRetargetRequest {
            claims: vec![claim("a", 0, &[1.0, 2.0]), claim("b", 1, &[2.0, 1.0])],
            contrasts: vec![contrast("d", &[("a", 1.0), ("b", -1.0)])],
            expected_snapshot: None,
        };
        let id = |r: &BatchRetargetRequest| family_id(r, Some("snap"), ScoreSource::Prepared, "f");
        let base = id(&request);
        let mut permuted = request.clone();
        permuted.claims.reverse();
        permuted.contrasts[0].coefficients.reverse();
        permuted.claims[0].depends_on.reverse();
        assert_eq!(id(&permuted), base);
        let mut changed = request.clone();
        changed.claims[1].weights[0] = f64::from_bits(2.0_f64.to_bits() + 1);
        assert_ne!(id(&changed), base);
        let mut renamed = request.clone();
        renamed.contrasts[0].name = "e".into();
        assert_ne!(id(&renamed), base);
        let mut coefficient = request.clone();
        coefficient.contrasts[0].coefficients[0].1 = 2.0;
        assert_ne!(id(&coefficient), base);
        let mut other_query = request.clone();
        other_query.claims[0].query_index = 1;
        assert_ne!(id(&other_query), base);
        assert_ne!(family_id(&request, Some("other"), ScoreSource::Prepared, "f"), base);
        assert_ne!(family_id(&request, Some("snap"), ScoreSource::Estimated, "f"), base);
        assert_ne!(family_id(&request, Some("snap"), ScoreSource::Prepared, "g"), base);
    }

    fn point(value: f64, std_error: Option<f64>) -> ClaimPoint {
        ClaimPoint {
            value,
            std_error,
            uncertainty_kind: if std_error.is_some() {
                UncertaintyKind::PlugInScoreCovariance
            } else {
                UncertaintyKind::None
            },
            n_eff: 10.0,
            n_eff_by_arm: vec![5.0, 5.0],
            propensity_range: None,
            nuisance_provenance: "test".into(),
            diagnostics: Vec::new(),
        }
    }

    fn report(claims: Vec<ClaimReport>) -> BatchRetargetReport {
        let covariance = Some(FamilyCovariance {
            names: vec!["a".into()],
            matrix: JointCovariance { dim: 1, values: Arc::from(vec![0.25]) },
        });
        BatchRetargetReport {
            family_id: "id".into(),
            snapshot_id: Some("snap".into()),
            scores_source: ScoreSource::Prepared,
            estimator_fingerprint: "f".into(),
            claims,
            contrasts: Vec::new(),
            covariance,
            scope_note: BATCH_RETARGET_SCOPE_NOTE,
        }
    }

    fn claim_report(name: &str, outcome: Result<ClaimPoint, MemberFailure>) -> ClaimReport {
        ClaimReport { name: name.into(), query_index: 0, estimand: "e".into(), outcome }
    }

    #[test]
    fn a_complete_family_needs_every_member_with_covariance() {
        let complete = report(vec![claim_report("a", Ok(point(1.0, Some(0.5))))]);
        assert!(complete.complete_family().is_ok());
        assert!(complete.tidy_rows().iter().all(|r| r.family_complete));

        let failed = MemberFailure::from(invalid("batch_retarget.incompatible_target", "x"));
        let partial = report(vec![
            claim_report("a", Ok(point(1.0, Some(0.5)))),
            claim_report("b", Err(failed)),
        ]);
        assert_eq!(partial.complete_family().unwrap_err().detail, "batch_retarget.partial_family");
        let rows = partial.tidy_rows();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|r| r.name == "b" && r.status == "failed"));
        assert!(rows.iter().all(|r| !r.family_complete && r.family_failed == 1));

        let point_only = report(vec![
            claim_report("a", Ok(point(1.0, Some(0.5)))),
            claim_report("p", Ok(point(2.0, None))),
        ]);
        assert_eq!(
            point_only.complete_family().unwrap_err().detail,
            "batch_retarget.point_only_member"
        );
        let rows = point_only.tidy_rows();
        assert_eq!(rows[1].status, "point_only");
        assert!(rows[1].std_error.is_none());
        assert_eq!(rows[1].uncertainty_kind, "none");
    }

    // ---- unpublished max-t evaluator ------------------------------------------------

    /// Draws for the Monte-Carlo oracles: the empirical-quantile error is about 0.005 at
    /// these levels, an order below the 0.04 tolerance.
    const DRAWS: u32 = 200_000;
    const TOL: f64 = 0.04;

    /// Standard normal CDF by the Maclaurin series of erf (independent of the library).
    fn phi(x: f64) -> f64 {
        let t = x / std::f64::consts::SQRT_2;
        let (mut term, mut sum) = (t, t);
        for n in 1..120 {
            let n = f64::from(n);
            term *= -t * t / n;
            sum += term / (2.0 * n + 1.0);
        }
        0.5 * (1.0 + 2.0 / std::f64::consts::PI.sqrt() * sum)
    }

    /// Standard normal quantile by bisection on [`phi`].
    fn inverse_phi(p: f64) -> f64 {
        let (mut lo, mut hi) = (0.0_f64, 6.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if phi(mid) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    fn equicorrelation(k: usize, r: f64) -> JointCovariance {
        let mut values = vec![r; k * k];
        for i in 0..k {
            values[i * k + i] = 1.0;
        }
        JointCovariance { dim: k, values: Arc::from(values) }
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
    }

    fn critical(corr: &JointCovariance, level: f64, seed: u64) -> f64 {
        max_t_critical_value(corr, level, seed, DRAWS, &ctx()).unwrap()
    }

    #[test]
    fn the_normal_oracle_is_the_known_quantile() {
        assert!((inverse_phi(0.975) - 1.959_963_984_540_054).abs() < 1e-6);
        assert!((inverse_phi(0.995) - 2.575_829_303_548_901).abs() < 1e-6);
    }

    #[test]
    fn one_claim_is_the_two_sided_normal_quantile() {
        let one = equicorrelation(1, 1.0);
        for level in [0.80, 0.90, 0.95, 0.99] {
            let c = critical(&one, level, 11);
            let z = inverse_phi(1.0 - (1.0 - level) / 2.0);
            assert!((c - z).abs() < TOL, "level {level}: {c} vs {z}");
        }
    }

    /// Independent claims: `P(max |Z_j| <= c) = (2 Φ(c) − 1)^k = level`.
    #[test]
    fn independent_claims_solve_the_product_equation() {
        for k in 1..=5_usize {
            for level in [0.90, 0.95] {
                let c = critical(&equicorrelation(k, 0.0), level, 21);
                let exponent = i32::try_from(k).unwrap();
                let covered = (2.0 * phi(c) - 1.0).powi(exponent);
                let oracle = inverse_phi((1.0 + level.powf(1.0 / f64::from(exponent))) / 2.0);
                assert!((c - oracle).abs() < TOL, "k {k} level {level}: {c} vs {oracle}");
                assert!((covered - level).abs() < 0.01, "k {k} level {level}: covered {covered}");
            }
        }
    }

    #[test]
    fn perfectly_correlated_claims_reduce_to_one_claim() {
        let single = inverse_phi(0.975);
        for k in [2, 3, 6] {
            let c = critical(&equicorrelation(k, 1.0), 0.95, 31);
            assert!((c - single).abs() < TOL, "k {k}: {c} vs {single}");
        }
    }

    #[test]
    fn the_critical_value_is_monotone_in_the_family_size_and_the_level() {
        let by_k: Vec<f64> =
            (1..=6).map(|k| critical(&equicorrelation(k, 0.0), 0.95, 41)).collect();
        assert!(by_k.windows(2).all(|w| w[0] < w[1]), "{by_k:?}");
        let by_level: Vec<f64> = [0.80, 0.90, 0.95, 0.99]
            .iter()
            .map(|&level| critical(&equicorrelation(2, 0.3), level, 43))
            .collect();
        assert!(by_level.windows(2).all(|w| w[0] < w[1]), "{by_level:?}");
        // Positive correlation can only shrink the family-wise critical value.
        assert!(
            critical(&equicorrelation(4, 0.8), 0.95, 47)
                < critical(&equicorrelation(4, 0.0), 0.95, 47)
        );
    }

    #[test]
    fn the_critical_value_is_deterministic_in_the_seed() {
        let corr = equicorrelation(3, 0.25);
        let a = critical(&corr, 0.95, 5);
        assert_eq!(a.to_bits(), critical(&corr, 0.95, 5).to_bits());
        assert_ne!(a.to_bits(), critical(&corr, 0.95, 6).to_bits());
        // The context's master seed is not an input: only the declared seed is.
        let other = max_t_critical_value(&corr, 0.95, 5, DRAWS, &ExecutionContext::for_tests(99));
        assert_eq!(a.to_bits(), other.unwrap().to_bits());
    }

    #[test]
    fn the_evaluator_refuses_bad_levels_draws_and_matrices() {
        let corr = equicorrelation(2, 0.0);
        let detail = |level: f64, draws: u32, m: &JointCovariance| {
            max_t_critical_value(m, level, 1, draws, &ctx()).unwrap_err().detail
        };
        for level in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
            assert_eq!(detail(level, 5_000, &corr), "batch_retarget.max_t_invalid_level");
        }
        assert_eq!(
            detail(0.95, MAX_T_MIN_DRAWS - 1, &corr),
            "batch_retarget.max_t_draws_out_of_range"
        );
        assert_eq!(
            detail(0.95, MAX_T_MAX_DRAWS + 1, &corr),
            "batch_retarget.max_t_draws_out_of_range"
        );
        // 1000 draws cannot hold a 0.9999 empirical quantile.
        assert_eq!(
            detail(0.9999, MAX_T_MIN_DRAWS, &corr),
            "batch_retarget.max_t_draws_out_of_range"
        );
        let not_unit = JointCovariance { dim: 1, values: Arc::from(vec![2.0]) };
        let not_psd = JointCovariance { dim: 2, values: Arc::from(vec![1.0, 2.0, 2.0, 1.0]) };
        let empty = JointCovariance { dim: 0, values: Arc::from(Vec::<f64>::new()) };
        for bad in [&not_unit, &not_psd, &empty] {
            assert_eq!(detail(0.95, 5_000, bad), "batch_retarget.covariance_unavailable");
        }
    }

    #[test]
    fn a_cancelled_context_stops_the_draws_without_a_value() {
        let corr = equicorrelation(2, 0.0);
        let before = ExecutionContext::for_tests(3);
        before.cancellation.cancel();
        let stopped = max_t_critical_value(&corr, 0.95, 1, DRAWS, &before).unwrap_err();
        assert_eq!(
            (stopped.code, stopped.detail),
            ("cancelled_no_claim", "batch_retarget.cancelled")
        );
        // Cancelled in the middle: the first poll passes, the second (draw 1024) trips.
        let mut mid = ExecutionContext::for_tests(3);
        mid.cancellation = antecedent_core::CancellationToken::cancel_after_checks(1);
        let stopped = max_t_critical_value(&corr, 0.95, 1, DRAWS, &mid).unwrap_err();
        assert_eq!(stopped.detail, "batch_retarget.cancelled");
        // The same call then completes on a live context.
        assert!(max_t_critical_value(&corr, 0.95, 1, DRAWS, &ctx()).is_ok());
    }

    fn two_claim_report() -> BatchRetargetReport {
        // Σ = [[4, 1.2], [1.2, 9]]: se = (2, 3) and correlation 1.2 / (2 · 3) = 0.2.
        let mut report = report(vec![
            claim_report("a", Ok(point(1.5, Some(2.0)))),
            claim_report("b", Ok(point(-2.0, Some(3.0)))),
        ]);
        report.covariance = Some(FamilyCovariance {
            names: vec!["a".into(), "b".into()],
            matrix: JointCovariance { dim: 2, values: Arc::from(vec![4.0, 1.2, 1.2, 9.0]) },
        });
        report
    }

    #[test]
    fn the_correlation_of_a_family_is_a_hand_calculation() {
        let report = two_claim_report();
        let corr = correlation_of(report.covariance.as_ref().unwrap()).unwrap();
        near(corr.get(0, 0), 1.0);
        near(corr.get(1, 1), 1.0);
        near(corr.get(0, 1), 0.2);
        assert_eq!(corr.get(0, 1).to_bits(), corr.get(1, 0).to_bits());
    }

    #[test]
    fn the_band_half_widths_are_the_critical_value_times_each_standard_error() {
        let report = two_claim_report();
        let band = simultaneous_band_unpublished(&report, 0.95, 17, DRAWS, &ctx()).unwrap();
        // The critical value is the evaluator's value on the hand-computed correlation 0.2.
        let expected = critical(&equicorrelation(2, 0.2), 0.95, 17);
        assert_eq!(band.critical_value.to_bits(), expected.to_bits());
        assert_eq!((band.seed, band.draws), (17, DRAWS));
        assert_eq!(band.level.to_bits(), 0.95_f64.to_bits());
        assert_eq!(band.members.len(), 2);
        let (a, b) = (&band.members[0], &band.members[1]);
        assert_eq!((a.name.as_str(), b.name.as_str()), ("a", "b"));
        near(a.std_error, 2.0);
        near(b.std_error, 3.0);
        near(a.lower, 1.5 - expected * 2.0);
        near(a.upper, 1.5 + expected * 2.0);
        near(b.lower, -2.0 - expected * 3.0);
        near(b.upper, -2.0 + expected * 3.0);
        // Wider than the pointwise 1.96 and no wider than the independent two-claim 2.24.
        assert!(band.critical_value > 1.96 && band.critical_value < 2.3);
        // The published route is the same evaluator.
        let published = report.simultaneous_interval(0.95, 17, DRAWS, &ctx()).unwrap();
        assert_eq!(published, band);
    }

    #[test]
    fn the_band_refuses_a_partial_or_point_only_family() {
        let failed = MemberFailure::from(invalid("batch_retarget.incompatible_target", "x"));
        let partial = report(vec![
            claim_report("a", Ok(point(1.0, Some(0.5)))),
            claim_report("b", Err(failed)),
        ]);
        let refused = partial.simultaneous_interval(0.95, 1, DRAWS, &ctx()).unwrap_err();
        assert_eq!(refused.detail, "batch_retarget.partial_family");
        let point_only = report(vec![
            claim_report("a", Ok(point(1.0, Some(0.5)))),
            claim_report("p", Ok(point(2.0, None))),
        ]);
        let refused = point_only.simultaneous_interval(0.95, 1, DRAWS, &ctx()).unwrap_err();
        assert_eq!(refused.detail, "batch_retarget.point_only_member");
    }
}
