//! Independent artifact for an autoregressive mechanism refreshed across observation periods.
//!
//! Format version 1. The artifact is the delta ledger of a mechanism kept by least squares
//! and by the conjugate normal-inverse-gamma posterior on the same rows: the stated prior and
//! its digest, every period applied in order (window identity, cadence, initial law, and
//! whether it extended the fit or restarted it), the blocks of rows the live fit used under
//! each snapshot id, and the cached results (posterior and least squares).
//!
//! A consumer trusts nothing but the retained blocks. It checks the seal and the prior
//! digest, re-decides every period with the window-survival rules, refuses a period that
//! overlaps an earlier one or reuses a snapshot, refuses a row or block that enters twice,
//! rebuilds every block's digest, and recomputes the posterior and the least-squares fit from
//! the rows alone with its own closed form. A stored result that the recomputation does not
//! reproduce is refused. The claim is a posterior of the stated model under the stated prior,
//! not a distribution-free interval; the prior alone identifies no causal target.

use crate::IoError;
use crate::temporal_initial_state_artifact::mismatch;
use crate::temporal_refresh_artifact::WindowIdentityWire;
use antecedent_core::{CanonicalWriter, IdentityDomain, reason_code};
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, TemporalWindowIdentity, WindowFactors, WindowProduct,
    refresh_survival,
};
use antecedent_stats::invert_square;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The artifact format this reader writes and accepts.
pub const PERIOD_REFRESH_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const PERIOD_REFRESH_ARTIFACT_FEATURE: &str = "period_refresh_ledger_v1";
/// The artifact kind; a different kind is a different artifact.
pub const PERIOD_REFRESH_ARTIFACT_KIND: &str = crate::artifact_kinds::PERIOD_REFRESH_LEDGER;

const PREFIX: &str = "period_refresh_artifact";
/// Outcome word of the first period.
pub const PERIOD_START: &str = "start";
/// Outcome word of a period that extended the fit.
pub const PERIOD_EXTENDED: &str = "extended";
/// Outcome word of a period that restarted the fit.
pub const PERIOD_RESTARTED: &str = "restarted";
const INVARIANT: &str = "declared_invariant";
const UNDECLARED: &str = "undeclared";
const PRIOR_TAG: &str = "antecedent.io.period_refresh.prior";
const BLOCK_TAG: &str = "antecedent.io.period_refresh.block";
const REUSE_NEXT: &str = "drop the duplicate block, or rebuild the route from the distinct blocks";
const TOLERANCE: f64 = 1e-9;

fn refused(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: reason_code!("invalid_argument"),
        message: format!("period_refresh.{detail}: {message}"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= TOLERANCE * (1.0 + a.abs().max(b.abs()))
}

fn all_close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

/// The stated prior of the mechanism.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PriorWire {
    /// Prior mean over `[intercept, lag coefficients]`.
    pub mean: Vec<f64>,
    /// Row-major prior precision.
    pub precision: Vec<f64>,
    /// Inverse-gamma shape of the innovation variance.
    pub shape: f64,
    /// Inverse-gamma rate of the innovation variance.
    pub rate: f64,
}

impl PriorWire {
    /// Content digest of the prior.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut writer = CanonicalWriter::new(PRIOR_TAG);
        writer.reals(&self.mean).reals(&self.precision).real(self.shape).real(self.rate);
        writer.finish_hex()
    }
}

/// One retained observation row.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RowWire {
    /// Time index of the target value.
    pub id: u64,
    /// Design row `[1, v_{t-1}, ..., v_{t-L}]`.
    pub x: Vec<f64>,
    /// Response `v_t`.
    pub y: f64,
}

/// The rows one snapshot contributed to the live fit.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BlockWire {
    /// Snapshot the rows came from.
    pub snapshot_id: String,
    /// Content digest of the block.
    pub digest: String,
    /// The retained rows.
    pub rows: Vec<RowWire>,
}

impl BlockWire {
    /// A block with its digest.
    #[must_use]
    pub fn new(snapshot_id: String, rows: Vec<RowWire>) -> Self {
        let digest = block_digest(&snapshot_id, &rows);
        Self { snapshot_id, digest, rows }
    }
}

fn block_digest(snapshot_id: &str, rows: &[RowWire]) -> String {
    let mut writer = CanonicalWriter::new(BLOCK_TAG);
    writer.text(snapshot_id).word(rows.len() as u64);
    for row in rows {
        writer.word(row.id).reals(&row.x).real(row.y);
    }
    writer.finish_hex()
}

/// One period applied to the mechanism.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PeriodEventWire {
    /// Window identity of the period.
    pub window: WindowIdentityWire,
    /// Tick cadence the period was declared with.
    pub cadence_tick: u64,
    /// Initial-law id the period was declared with.
    pub initial_law: String,
    /// `start`, `extended` or `restarted`.
    pub outcome: String,
    /// For `extended`: whether the rows spanning the boundary were formed.
    pub contiguous: bool,
    /// For `restarted`: the products the changes invalidate.
    pub invalidated: Vec<String>,
}

/// The cached posterior of the mechanism.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PosteriorWire {
    /// Observations used.
    pub n: u64,
    /// Posterior mean of `[intercept, lag coefficients]`.
    pub mean: Vec<f64>,
    /// Row-major posterior precision.
    pub precision: Vec<f64>,
    /// Inverse-gamma shape.
    pub shape: f64,
    /// Inverse-gamma rate.
    pub rate: f64,
}

/// The cached least-squares result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LeastSquaresWire {
    /// Observations used.
    pub n: u64,
    /// Coefficients.
    pub beta: Vec<f64>,
    /// Residual variance, absent when there are not more rows than columns.
    pub residual_variance: Option<f64>,
}

/// The refresh ledger artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PeriodRefreshArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required features.
    pub required_features: Vec<String>,
    /// Artifact kind.
    pub kind: String,
    /// Identity of the series.
    pub series_id: String,
    /// Autoregressive order.
    pub lags: u32,
    /// `declared_invariant` or `undeclared`.
    pub standing: String,
    /// The prior stated when the first period was fitted.
    pub prior: PriorWire,
    /// Digest of the prior.
    pub prior_digest: String,
    /// Every period applied, in order.
    pub events: Vec<PeriodEventWire>,
    /// Blocks of rows the live fit used, in the order they entered.
    pub blocks: Vec<BlockWire>,
    /// Total rows used.
    pub rows_used: u64,
    /// The cached posterior.
    pub posterior: PosteriorWire,
    /// The cached least squares.
    pub least_squares: LeastSquaresWire,
    /// Seal over every other field.
    pub seal: String,
}

/// What a consumer recomputed.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodRefreshReplay {
    /// The recomputed posterior.
    pub posterior: PosteriorWire,
    /// The recomputed least squares.
    pub least_squares: LeastSquaresWire,
    /// Snapshot ids of the live blocks, in order.
    pub live_snapshots: Vec<String>,
}

/// Standing as its wire word.
#[must_use]
pub const fn standing_label(standing: MechanismStanding) -> &'static str {
    match standing {
        MechanismStanding::DeclaredInvariant => INVARIANT,
        MechanismStanding::Undeclared => UNDECLARED,
    }
}

/// Product as its wire word.
#[must_use]
pub const fn product_label(product: WindowProduct) -> &'static str {
    match product {
        WindowProduct::Identification => "identification",
        WindowProduct::Design => "design",
        WindowProduct::Model => "model",
        WindowProduct::Evidence => "evidence",
    }
}

struct Accumulated {
    n: u64,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
}

fn accumulate(blocks: &[BlockWire], p: usize) -> Result<Accumulated, IoError> {
    let mut total = Accumulated { n: 0, xtx: vec![0.0; p * p], xty: vec![0.0; p], yty: 0.0 };
    for row in blocks.iter().flat_map(|block| block.rows.iter()) {
        if row.x.len() != p || !row.y.is_finite() || row.x.iter().any(|v| !v.is_finite()) {
            return Err(mismatch(PREFIX, "row_shape"));
        }
        for (i, xi) in row.x.iter().enumerate() {
            if let Some(slot) = total.xty.get_mut(i) {
                *slot += xi * row.y;
            }
            for (j, xj) in row.x.iter().enumerate() {
                if let Some(slot) = total.xtx.get_mut(i * p + j) {
                    *slot += xi * xj;
                }
            }
        }
        total.yty += row.y * row.y;
        total.n += 1;
    }
    Ok(total)
}

fn multiply(matrix: &[f64], p: usize, vector: &[f64]) -> Vec<f64> {
    (0..p)
        .map(|i| {
            (0..p)
                .map(|j| {
                    matrix.get(i * p + j).copied().unwrap_or(0.0)
                        * vector.get(j).copied().unwrap_or(0.0)
                })
                .sum()
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Closed-form normal-inverse-gamma posterior of the pooled rows.
fn posterior_of(prior: &PriorWire, total: &Accumulated) -> Result<PosteriorWire, IoError> {
    let p = prior.mean.len();
    let precision: Vec<f64> = prior.precision.iter().zip(&total.xtx).map(|(a, b)| a + b).collect();
    let prior_eta = multiply(&prior.precision, p, &prior.mean);
    let eta: Vec<f64> = prior_eta.iter().zip(&total.xty).map(|(a, b)| a + b).collect();
    let scale = invert_square(&precision, p).ok_or_else(|| mismatch(PREFIX, "singular"))?;
    let mean = multiply(&scale, p, &eta);
    let sum_squares = dot(&prior.mean, &prior_eta) + total.yty;
    let rate = prior.rate + 0.5 * (sum_squares - dot(&mean, &eta));
    if !rate.is_finite() || rate <= 0.0 {
        return Err(mismatch(PREFIX, "rate"));
    }
    Ok(PosteriorWire {
        n: total.n,
        mean,
        precision,
        shape: prior.shape + total.n as f64 / 2.0,
        rate,
    })
}

/// Least squares from the rows, with the residual sum of squares taken row by row.
fn least_squares_of(
    blocks: &[BlockWire],
    total: &Accumulated,
    p: usize,
) -> Result<LeastSquaresWire, IoError> {
    let Some(inverse) = invert_square(&total.xtx, p) else {
        return Err(mismatch(PREFIX, "singular"));
    };
    let beta = multiply(&inverse, p, &total.xty);
    let enough = usize::try_from(total.n).is_ok_and(|n| n > p);
    let residual_variance = enough.then(|| {
        let sse: f64 = blocks
            .iter()
            .flat_map(|block| block.rows.iter())
            .map(|row| (row.y - dot(&row.x, &beta)).powi(2))
            .sum();
        sse / (total.n as f64 - p as f64)
    });
    Ok(LeastSquaresWire { n: total.n, beta, residual_variance })
}

fn optional_close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-7 * (1.0 + x.abs().max(y.abs())),
        (None, None) => true,
        _ => false,
    }
}

fn posterior_close(a: &PosteriorWire, b: &PosteriorWire) -> bool {
    a.n == b.n
        && all_close(&a.mean, &b.mean)
        && all_close(&a.precision, &b.precision)
        && close(a.shape, b.shape)
        && close(a.rate, b.rate)
}

impl PeriodRefreshArtifactWire {
    /// Everything but the seal, sealed.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn compute_seal(&self) -> Result<String, IoError> {
        let mut body = self.clone();
        body.seal = String::new();
        crate::digest_hex(IdentityDomain::TransportCertificate, &("period_refresh_seal_v1", body))
    }

    /// Seal a ledger assembled from its parts and replay it once, so a producer never writes
    /// an artifact its consumer would refuse.
    ///
    /// # Errors
    /// Encoding failure or any refusal of [`Self::verify`].
    pub fn checked(mut self) -> Result<Self, IoError> {
        self.seal = self.compute_seal()?;
        self.verify()?;
        Ok(self)
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version, feature or kind first.
    ///
    /// # Errors
    /// An unsupported version, a decoding failure, or a foreign feature or kind.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let decoded: Result<Self, IoError> = crate::decode_with_version(bytes, |version| {
            if version == PERIOD_REFRESH_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        });
        let wire = match decoded {
            Ok(wire) => wire,
            Err(error) => {
                if PeriodRefreshSummaryWire::decode(bytes).is_ok() {
                    return Err(refused(
                        "summary_only",
                        "this is the summary-only form, which keeps no rows, so it cannot be \
                         read as the full ledger; load it with the summary-only reader, or \
                         export the full ledger from a fit that still retains its rows",
                    ));
                }
                return Err(error);
            }
        };
        if wire.required_features != [PERIOD_REFRESH_ARTIFACT_FEATURE]
            || wire.kind != PERIOD_REFRESH_ARTIFACT_KIND
        {
            return Err(mismatch(PREFIX, "unsupported_semantics"));
        }
        Ok(wire)
    }

    /// Decode and verify.
    ///
    /// # Errors
    /// See [`Self::decode`] and [`Self::verify`].
    pub fn consume(bytes: &[u8]) -> Result<(Self, PeriodRefreshReplay), IoError> {
        let wire = Self::decode(bytes)?;
        let replay = wire.verify()?;
        Ok((wire, replay))
    }

    /// Refuse a ledger whose prior is not the one the caller holds now. A posterior answers
    /// only under the prior it was built with.
    ///
    /// # Errors
    /// `period_refresh.prior_changed`, naming the rebuild from the retained blocks.
    pub fn require_prior(&self, prior: &PriorWire) -> Result<(), IoError> {
        if prior.digest() == self.prior_digest {
            return Ok(());
        }
        Err(refused(
            "prior_changed",
            "a posterior is the prior times the likelihood of the observations used, so a \
             different prior is a different posterior; rebuild from every retained block under \
             the new prior",
        ))
    }

    /// Recompute everything from the retained blocks and compare with what is stored.
    ///
    /// # Errors
    /// A changed seal, prior, block or stored result; a period that overlaps an earlier one or
    /// reuses a snapshot; a row or block that enters twice; an outcome the window rules do not
    /// give; or an extension the standing does not allow.
    pub fn verify(&self) -> Result<PeriodRefreshReplay, IoError> {
        if self.compute_seal()? != self.seal {
            return Err(mismatch(PREFIX, "seal"));
        }
        let p = self.check_prior()?;
        self.check_events()?;
        self.check_blocks(p)?;
        let total = accumulate(&self.blocks, p)?;
        if total.n == 0 || total.n != self.rows_used {
            return Err(mismatch(PREFIX, "rows_used"));
        }
        let posterior = posterior_of(&self.prior, &total)?;
        if !posterior_close(&posterior, &self.posterior) {
            return Err(mismatch(PREFIX, "posterior"));
        }
        let least_squares = least_squares_of(&self.blocks, &total, p)?;
        if least_squares.n != self.least_squares.n
            || !all_close(&least_squares.beta, &self.least_squares.beta)
            || !optional_close(
                least_squares.residual_variance,
                self.least_squares.residual_variance,
            )
        {
            return Err(mismatch(PREFIX, "least_squares"));
        }
        let live_snapshots = self.blocks.iter().map(|b| b.snapshot_id.clone()).collect();
        Ok(PeriodRefreshReplay { posterior, least_squares, live_snapshots })
    }

    fn check_prior(&self) -> Result<usize, IoError> {
        let p = self.prior.mean.len();
        let lags = usize::try_from(self.lags).map_err(|_| mismatch(PREFIX, "lags"))?;
        let finite = self.prior.mean.iter().chain(&self.prior.precision).all(|v| v.is_finite());
        if lags == 0
            || p != lags + 1
            || self.prior.precision.len() != p * p
            || !finite
            || !(self.prior.shape.is_finite() && self.prior.shape > 0.0)
            || !(self.prior.rate.is_finite() && self.prior.rate > 0.0)
        {
            return Err(mismatch(PREFIX, "prior"));
        }
        if self.prior.digest() != self.prior_digest {
            return Err(mismatch(PREFIX, "prior_digest"));
        }
        if self.standing != INVARIANT && self.standing != UNDECLARED {
            return Err(mismatch(PREFIX, "standing"));
        }
        Ok(p)
    }

    fn standing(&self) -> MechanismStanding {
        if self.standing == INVARIANT {
            MechanismStanding::DeclaredInvariant
        } else {
            MechanismStanding::Undeclared
        }
    }

    /// Order, overlap, snapshot reuse and the outcome of every period, re-decided.
    fn check_events(&self) -> Result<(), IoError> {
        let Some(first) = self.events.first() else {
            return Err(mismatch(PREFIX, "no_events"));
        };
        if first.outcome != PERIOD_START {
            return Err(mismatch(PREFIX, "first_event"));
        }
        let mut snapshots = BTreeSet::new();
        let mut previous: Option<(TemporalWindowIdentity, WindowFactors)> = None;
        for event in &self.events {
            let window = event.window.to_identity()?;
            let cadence = ObservationCadence::new(event.cadence_tick)
                .ok_or_else(|| mismatch(PREFIX, "cadence"))?;
            let factors = WindowFactors::try_from_window(&window, cadence, &event.initial_law)
                .map_err(|e| IoError::Convert(e.to_string()))?;
            if !snapshots.insert(window.snapshot_id.clone()) {
                return Err(refused(
                    "block_reused",
                    &format!(
                        "snapshot {} enters the ledger twice, so the same observations would \
                         enter the likelihood twice; {REUSE_NEXT}",
                        window.snapshot_id
                    ),
                ));
            }
            if let Some((old_window, old_factors)) = &previous {
                self.check_transition(event, old_window, old_factors, &window, &factors)?;
            }
            previous = Some((window, factors));
        }
        Ok(())
    }

    fn check_transition(
        &self,
        event: &PeriodEventWire,
        old_window: &TemporalWindowIdentity,
        old_factors: &WindowFactors,
        window: &TemporalWindowIdentity,
        factors: &WindowFactors,
    ) -> Result<(), IoError> {
        if window.period.start < old_window.period.end {
            return Err(refused(
                "overlap",
                "the period overlaps the summarized one, so the same observations would enter \
                 twice; supply a later period of a new snapshot",
            ));
        }
        let survival = refresh_survival(old_window, window, old_factors, factors, self.standing())
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let extend = self.standing() == MechanismStanding::DeclaredInvariant
            && survival.products().retains(WindowProduct::Model);
        if extend {
            let contiguous = window.period.start == old_window.period.end;
            if event.outcome != PERIOD_EXTENDED
                || event.contiguous != contiguous
                || !event.invalidated.is_empty()
            {
                return Err(mismatch(PREFIX, "outcome"));
            }
            return Ok(());
        }
        let invalidated: Vec<&str> = [
            WindowProduct::Identification,
            WindowProduct::Design,
            WindowProduct::Model,
            WindowProduct::Evidence,
        ]
        .into_iter()
        .filter(|product| survival.products().invalidates(*product))
        .map(product_label)
        .collect();
        let stored: Vec<&str> = event.invalidated.iter().map(String::as_str).collect();
        if event.outcome != PERIOD_RESTARTED || event.contiguous || stored != invalidated {
            return Err(mismatch(PREFIX, "outcome"));
        }
        Ok(())
    }

    /// Block digests, row shapes, and no block or row used twice.
    fn check_blocks(&self, p: usize) -> Result<(), IoError> {
        let segment_start =
            self.events.iter().rposition(|e| e.outcome != PERIOD_EXTENDED).unwrap_or(0);
        let event_snapshots: BTreeSet<&str> =
            self.events.iter().skip(segment_start).map(|e| e.window.snapshot_id.as_str()).collect();
        let mut labels = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for block in &self.blocks {
            if !event_snapshots.contains(block.snapshot_id.as_str()) {
                return Err(mismatch(PREFIX, "block_snapshot"));
            }
            if block_digest(&block.snapshot_id, &block.rows) != block.digest {
                return Err(mismatch(PREFIX, "block_digest"));
            }
            if block.rows.is_empty() || block.rows.iter().any(|row| row.x.len() != p) {
                return Err(mismatch(PREFIX, "row_shape"));
            }
            if !labels.insert(block.snapshot_id.as_str()) {
                return Err(refused(
                    "block_reused",
                    &format!(
                        "block {} is listed twice, so the same observations would enter the \
                         likelihood twice; {REUSE_NEXT}",
                        block.snapshot_id
                    ),
                ));
            }
            for row in &block.rows {
                if !ids.insert(row.id) {
                    return Err(refused(
                        "block_reused",
                        &format!(
                            "row {} appears in more than one block, so the same observation \
                             would enter the likelihood twice; {REUSE_NEXT}",
                            row.id
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The summary-only artifact format this reader writes and accepts.
pub const PERIOD_REFRESH_SUMMARY_VERSION: u32 = 1;
/// The feature marker of the summary-only format.
pub const PERIOD_REFRESH_SUMMARY_FEATURE: &str = "period_refresh_summary_v1";
/// The summary-only artifact kind; it is not the kind of the full ledger.
pub const PERIOD_REFRESH_SUMMARY_KIND: &str = crate::artifact_kinds::PERIOD_REFRESH_SUMMARY;
/// The retention word a summary-only artifact states.
pub const PERIOD_REFRESH_SUMMARY_RETENTION: &str = "summary-only";

const SUMMARY_SEAL_TAG: &str = "period_refresh_summary_seal_v1";

/// Row counts of a retained summary.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SummaryLedgerWire {
    /// Rows the summary was built from, masked or not.
    pub rows: u64,
    /// Rows that entered the summary.
    pub live_rows: u64,
    /// Sum of the counts of the live rows; equals `n`.
    pub effective_n: u64,
}

/// The summary-only form of a refreshed least-squares fit.
///
/// It carries the retained summary (Gram, cross-product, squared responses, observation count,
/// coefficients, residual variance), the source digest and the row counts, and states its
/// retention. No row and no block is in it, so a consumer can serve the summary and nothing that
/// needs rows. It is a different kind from the full ledger, so the full reader refuses it and
/// the summary reader refuses a full ledger: neither form is read as the other.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PeriodRefreshSummaryWire {
    /// Format version.
    pub version: u32,
    /// Required features.
    pub required_features: Vec<String>,
    /// Artifact kind.
    pub kind: String,
    /// Retention the producer reduced to; always `summary-only`.
    pub retention: String,
    /// Source the summarized rows belonged to.
    pub source: String,
    /// Number of columns of the design.
    pub ncols: u32,
    /// Observation count.
    pub n: u64,
    /// Row-major Gram.
    pub xtx: Vec<f64>,
    /// Cross-product.
    pub xty: Vec<f64>,
    /// Sum of squared responses.
    pub yty: f64,
    /// Least-squares coefficients, absent when the Gram is singular.
    pub beta: Option<Vec<f64>>,
    /// Residual variance, absent when there are not more rows than columns.
    pub residual_variance: Option<f64>,
    /// Digest of the source and the ordered rows the summary was built from.
    pub source_digest: String,
    /// Row counts.
    pub ledger: SummaryLedgerWire,
    /// Seal over every other field.
    pub seal: String,
}

impl PeriodRefreshSummaryWire {
    /// Everything but the seal, sealed.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn compute_seal(&self) -> Result<String, IoError> {
        let mut body = self.clone();
        body.seal = String::new();
        crate::digest_hex(IdentityDomain::TransportCertificate, &(SUMMARY_SEAL_TAG, body))
    }

    /// Seal a summary assembled from its parts and verify it once, so a producer never writes
    /// an artifact its consumer would refuse.
    ///
    /// # Errors
    /// Encoding failure or any refusal of [`Self::verify`].
    pub fn checked(mut self) -> Result<Self, IoError> {
        self.seal = self.compute_seal()?;
        self.verify()?;
        Ok(self)
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version, feature or kind first.
    ///
    /// # Errors
    /// An unsupported version, a decoding failure (a full ledger does not decode as a summary),
    /// or a foreign feature or kind.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let wire: Self = crate::decode_with_version(bytes, |version| {
            if version == PERIOD_REFRESH_SUMMARY_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        })?;
        if wire.required_features != [PERIOD_REFRESH_SUMMARY_FEATURE]
            || wire.kind != PERIOD_REFRESH_SUMMARY_KIND
        {
            return Err(mismatch(PREFIX, "unsupported_semantics"));
        }
        Ok(wire)
    }

    /// Decode and verify.
    ///
    /// # Errors
    /// See [`Self::decode`] and [`Self::verify`].
    pub fn consume(bytes: &[u8]) -> Result<Self, IoError> {
        let wire = Self::decode(bytes)?;
        wire.verify()?;
        Ok(wire)
    }

    /// Check the seal, the stated retention, the shapes, the ledger, and that the stored
    /// coefficients and residual variance are what the stored Gram and cross-product give.
    ///
    /// # Errors
    /// A changed seal, a retention other than summary-only, a malformed summary, or a stored
    /// result the summary does not reproduce.
    pub fn verify(&self) -> Result<(), IoError> {
        if self.compute_seal()? != self.seal {
            return Err(mismatch(PREFIX, "seal"));
        }
        if self.retention != PERIOD_REFRESH_SUMMARY_RETENTION {
            return Err(mismatch(PREFIX, "retention"));
        }
        let p = usize::try_from(self.ncols).map_err(|_| mismatch(PREFIX, "ncols"))?;
        let finite = self.xtx.iter().chain(&self.xty).all(|v| v.is_finite());
        if p == 0
            || self.xtx.len() != p * p
            || self.xty.len() != p
            || !finite
            || !(self.yty.is_finite() && self.yty >= 0.0)
            || self.source_digest.is_empty()
        {
            return Err(mismatch(PREFIX, "summary_shape"));
        }
        if self.ledger.effective_n != self.n
            || self.ledger.live_rows > self.ledger.rows
            || self.ledger.live_rows > self.n
        {
            return Err(mismatch(PREFIX, "ledger"));
        }
        let expected_beta = if self.n == 0 {
            None
        } else {
            invert_square(&self.xtx, p).map(|inverse| multiply(&inverse, p, &self.xty))
        };
        match (&expected_beta, &self.beta) {
            (None, None) => {}
            (Some(expected), Some(stored)) if all_close(expected, stored) => {}
            _ => return Err(mismatch(PREFIX, "least_squares")),
        }
        let enough = usize::try_from(self.n).is_ok_and(|n| n > p);
        let expected_variance = match (&expected_beta, enough) {
            (Some(beta), true) => {
                let dof = self.n as f64 - p as f64;
                let curvature = dot(beta, &multiply(&self.xtx, p, beta));
                let sse = self.yty - 2.0 * dot(beta, &self.xty) + curvature;
                Some((sse / dof, self.yty / dof))
            }
            _ => None,
        };
        match (expected_variance, self.residual_variance) {
            (None, None) => {}
            (Some((expected, scale)), Some(stored))
                if (expected - stored).abs()
                    <= 1e-6 * (1.0 + expected.abs().max(stored.abs())) + 1e-9 * scale => {}
            _ => return Err(mismatch(PREFIX, "residual_variance")),
        }
        Ok(())
    }
}
