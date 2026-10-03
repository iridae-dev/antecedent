//! Batch retarget: the points, joint covariance and named contrasts of a family of
//! retargeted claims over one row snapshot, with the simultaneous interval closed.
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
//! What is not: the family-level simultaneous (max-t) interval. It is a nominal asymptotic
//! interval no coverage record measures, so it is closed with a typed refusal rather than
//! published under `estimator_grid_not_measured`. A penalized-propensity table retargets to
//! a point only and joins the family without covariance. A partial family (any failed or
//! point-only member) is reported member by member and never offered as a complete-family
//! claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId, reason_code};
use antecedent_estimate::{JointCovariance, ScoreTable, provenance_withholds_interval};
use antecedent_io::PayloadDigestWire;

use crate::error::CausalError;

use super::batch::BatchQuery;
use super::prepared::PreparedStudy;

/// What a batch retarget is and is not, carried on every report.
pub const BATCH_RETARGET_SCOPE_NOTE: &str = "Points and plug-in score covariance of retargeted claims over one row snapshot, under iid rows, caller-declared fixed weights, positivity and nuisance convergence; selection or weight-estimation uncertainty is excluded. No interval is published: the family-level simultaneous interval is a nominal asymptotic construction that no coverage record measures, so it is closed.";

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
}

fn invalid(detail: &'static str, message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError { code: reason_code!("invalid_argument"), detail, message: message.into() }
}

fn not_licensed(detail: &'static str, message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError { code: reason_code!("cell_not_licensed"), detail, message: message.into() }
}

fn not_supported(detail: &'static str, message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("route_not_supported"),
        detail,
        message: message.into(),
    }
}

fn mixed_snapshot(message: impl Into<String>) -> BatchRetargetError {
    BatchRetargetError {
        code: reason_code!("row_weights_bound_to_snapshot"),
        detail: "batch_retarget.mixed_snapshot",
        message: message.into(),
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
}

impl From<BatchRetargetError> for MemberFailure {
    fn from(error: BatchRetargetError) -> Self {
        Self {
            reason_code: Some(error.code.to_string()),
            detail: Some(error.detail.to_string()),
            message: error.message,
            support_refused: false,
        }
    }
}

fn member_failure(error: &CausalError) -> MemberFailure {
    if matches!(error, CausalError::Support { .. }) {
        let mut failure = MemberFailure::from(not_licensed(
            "batch_retarget.weighted_overlap_failed",
            error.to_string(),
        ));
        failure.support_refused = true;
        return failure;
    }
    MemberFailure {
        reason_code: error.reason_code().map(str::to_owned),
        detail: None,
        message: error.to_string(),
        support_refused: false,
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

/// The closed family-level simultaneous interval and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Stable detail.
    pub detail: &'static str,
    /// Explanation.
    pub message: String,
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
    /// The simultaneous interval is always closed.
    pub simultaneous_interval: FamilyRefusal,
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
            ));
        }
    }
    if let Some(claim) = request.claims.iter().find(|c| c.query_index >= plans) {
        return Err(invalid(
            "batch_retarget.unknown_claim",
            format!(
                "claim {:?} names query {} but the batch has {plans} plans",
                claim.name, claim.query_index
            ),
        ));
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
            ));
        }
        if let Some((missing, _)) =
            contrast.coefficients.iter().find(|(n, _)| !request.claims.iter().any(|c| &c.name == n))
        {
            return Err(invalid(
                "batch_retarget.unknown_claim",
                format!("contrast {:?} names the undeclared claim {missing:?}", contrast.name),
            ));
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
            return Err(BatchRetargetError {
                code: reason_code!("cancelled_no_claim"),
                detail: "batch_retarget.cancelled",
                message: "the batch retarget was cancelled; no family is reported and the stop \
                          is not a verdict on the data"
                    .into(),
            });
        }
        evaluated.push(evaluate_claim(
            &plans[claim.query_index],
            &queries[claim.query_index],
            scores.table(claim.query_index),
            claim,
            scores.source,
        ));
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
            outcome: contrast_outcome(c, &claims, covariance.as_ref()),
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
        simultaneous_interval: FamilyRefusal {
            code: reason_code!("cell_not_licensed"),
            detail: "batch_retarget.simultaneous_interval_closed",
            message: "a family-level simultaneous (max-t) interval over retargeted claims is a \
                      nominal asymptotic construction that no coverage record measures; \
                      covariance, contrasts and points are published, the interval is closed"
                .to_string(),
        },
        scope_note: BATCH_RETARGET_SCOPE_NOTE,
    })
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
            simultaneous_interval: FamilyRefusal {
                code: "cell_not_licensed",
                detail: "batch_retarget.simultaneous_interval_closed",
                message: String::new(),
            },
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
}
