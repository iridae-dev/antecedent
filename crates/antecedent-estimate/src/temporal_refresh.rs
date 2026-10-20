//! Semantic refresh of a prepared horizon-two temporal result onto a new observation period
//! (`new_period_refresh`).
//!
//! # Contract
//!
//! A prepared two-step result is valid for one declared window: its time horizon, the lag
//! alignment of its coordinates, its intervention history, its selection targets and regimes,
//! its graph and proof, its observation period, the units observed, and its data snapshot
//! ([`TemporalWindowIdentity`]). A replacement snapshot with its own explicit period and unit
//! ids is checked against that identity by [`decide_refresh`]:
//!
//! * the proof is **reusable** only when the graph, horizon, lag alignment, intervention
//!   history, selection targets, regimes and proof id are all unchanged and both the period
//!   and the snapshot are new. The result is then re-evaluated on the new period (never
//!   copied) and a [`RefreshReceipt`] binds the old and new identities;
//! * anything else is a typed [`RefreshInvalidation`] with a registered refusal
//!   (`route_not_supported`, `temporal_refresh.*`): an appended third slice or changed horizon
//!   (`horizon_changed`), changed lag alignment (`lag_alignment_changed`), altered
//!   intervention history (`intervention_history_changed`), changed selection targets,
//!   regimes, graph or proof (`premises_changed`), and a replacement that is not a new
//!   snapshot of a new period (`stale_snapshot`).
//!
//! A stale proof, fit or interval never survives a refresh: the refreshed result carries no
//! interval, and the receipt's `interval_invalidated` flag records that one existed for the
//! old window. No new identification theorem is claimed; the claim stays point only.
//!
//! # Products that read a change
//!
//! [`refresh_survival`] names the identification, design, model and evidence products a
//! change invalidates. A product that does not read the change is retained. The prepared
//! route's refusals are unchanged: a horizon, lag, history or premise change still refuses
//! that route, and a population drift does not.
//!
//! * identification reads the horizon, the initial law and the proof premises (graph, proof,
//!   lag alignment, intervention history, selection targets, regimes);
//! * design reads the period, the population, the cadence, the horizon and the selection
//!   targets and regimes;
//! * the model reads the population, the cadence, the horizon, the initial law, the lag
//!   alignment, the intervention history and the regimes. It reads the period only when
//!   mechanism invariance across periods was not declared;
//! * evidence reads the period, the population, the cadence and the regimes.
//!
//! [`TemporalIndex::mechanism_standing`] is always undeclared. A temporal index is not a
//! stationarity, equilibrium or finite-horizon claim, so a period change is not the same
//! mechanism shifted in time, and a horizon change does not reclassify the model. Declared
//! invariance is a caller statement about periods only. It can retain the model across a
//! period change, and it does not retain the model across a population, cadence, horizon or
//! initial-law change. The old fit and any interval are still dropped. This is not a prior
//! refresh: nothing here updates a Bayesian prior.
//!
//! The digests here are non-cryptographic content fingerprints (two FNV-1a lanes over
//! length-prefixed fields); they detect accidental or unresealed change, not an adversary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{CanonicalWriter, reason_code};

use crate::error::EstimationError;
use crate::util::refuse;

/// Inference claim of the route.
pub const TEMPORAL_REFRESH_INFERENCE_CLAIM: &str = "point_only";

/// The horizon every refreshable result has.
pub const TEMPORAL_REFRESH_HORIZON: usize = 2;

/// Half-open observation period `[start, end)` in an integer time index.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ObservationPeriod {
    /// First time index observed.
    pub start: i64,
    /// One past the last time index observed.
    pub end: i64,
}

/// Every input a prepared two-step result's validity rests on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalWindowIdentity {
    /// Identity of the (unrolled) graph the proof was derived on.
    pub graph_id: String,
    /// Time horizon; a refreshable result has [`TEMPORAL_REFRESH_HORIZON`].
    pub horizon: usize,
    /// Lag alignment: each coordinate name with the slice it is aligned to.
    pub lag_alignment: Vec<(String, u8)>,
    /// Ordered action of each step, as labels.
    pub intervention_history: Vec<String>,
    /// Time-indexed selection targets.
    pub selection_targets: BTreeSet<String>,
    /// Regimes of the evidence.
    pub regimes: BTreeSet<String>,
    /// Observation period of the snapshot.
    pub period: ObservationPeriod,
    /// Unit lineage: the repeated-unit ids observed.
    pub unit_ids: BTreeSet<String>,
    /// Identity of the data snapshot.
    pub snapshot_id: String,
    /// Identity of the proof the result was derived under.
    pub proof_id: String,
}

/// Domain tag of [`TemporalWindowIdentity::digest`].
const WINDOW_IDENTITY_TAG: &str = "antecedent.estimate.temporal_refresh.window_identity";
/// Domain tag of [`RefreshReceipt::compute_digest`].
const RECEIPT_TAG: &str = "antecedent.estimate.temporal_refresh.receipt";

/// A tagged item list: tag, item count, then each item, all length-prefixed text.
fn write_list<'a>(
    writer: &mut CanonicalWriter,
    tag: &str,
    items: impl ExactSizeIterator<Item = &'a str>,
) {
    writer.text(tag).word(items.len() as u64);
    for item in items {
        writer.text(item);
    }
}

/// Lowercase hex of a finished canonical digest.
fn hex_digest(writer: &CanonicalWriter) -> String {
    writer.finish_hex()
}

impl TemporalWindowIdentity {
    /// Content fingerprint over every identity input.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut d = CanonicalWriter::new(WINDOW_IDENTITY_TAG);
        d.text(&self.graph_id);
        d.word(self.horizon as u64);
        let lag: Vec<String> =
            self.lag_alignment.iter().map(|(name, slice)| format!("{name}@{slice}")).collect();
        write_list(&mut d, "lag_alignment", lag.iter().map(String::as_str));
        write_list(
            &mut d,
            "intervention_history",
            self.intervention_history.iter().map(String::as_str),
        );
        write_list(&mut d, "selection_targets", self.selection_targets.iter().map(String::as_str));
        write_list(&mut d, "regimes", self.regimes.iter().map(String::as_str));
        d.text(&self.period.start.to_string());
        d.text(&self.period.end.to_string());
        write_list(&mut d, "unit_ids", self.unit_ids.iter().map(String::as_str));
        d.text(&self.snapshot_id);
        d.text(&self.proof_id);
        hex_digest(&d)
    }

    fn validate(&self) -> Result<(), EstimationError> {
        let invalid = |message: &str| {
            refuse(
                reason_code!("invalid_argument"),
                "temporal_refresh.invalid_replacement",
                message,
            )
        };
        if self.period.end <= self.period.start {
            return Err(invalid("the replacement period is empty or reversed"));
        }
        if self.unit_ids.is_empty() {
            return Err(invalid("the replacement carries no unit ids"));
        }
        if self.snapshot_id.trim().is_empty()
            || self.proof_id.trim().is_empty()
            || self.graph_id.trim().is_empty()
            || self.horizon == 0
            || self.unit_ids.iter().any(|id| id.trim().is_empty())
        {
            return Err(invalid("the replacement lacks a snapshot id, proof id or horizon"));
        }
        Ok(())
    }
}

/// Why a refresh invalidates the old result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshInvalidation {
    /// An appended slice or any other change of horizon.
    HorizonChanged,
    /// The lag alignment of the coordinates changed.
    LagAlignmentChanged,
    /// The intervention history was altered.
    InterventionHistoryChanged,
    /// Selection targets, regimes, graph or proof changed.
    PremisesChanged,
    /// The replacement is not a new snapshot of a new period.
    StaleSnapshot,
}

impl RefreshInvalidation {
    /// Namespaced detail literal.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::HorizonChanged => "temporal_refresh.horizon_changed",
            Self::LagAlignmentChanged => "temporal_refresh.lag_alignment_changed",
            Self::InterventionHistoryChanged => "temporal_refresh.intervention_history_changed",
            Self::PremisesChanged => "temporal_refresh.premises_changed",
            Self::StaleSnapshot => "temporal_refresh.stale_snapshot",
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::HorizonChanged => {
                "the replacement changes the horizon (an appended slice needs its own cell); the \
                 old proof, fit and interval are invalidated"
            }
            Self::LagAlignmentChanged => {
                "the lag alignment of the coordinates changed; the old proof, fit and interval \
                 are invalidated"
            }
            Self::InterventionHistoryChanged => {
                "the intervention history changed; the old proof, fit and interval are \
                 invalidated"
            }
            Self::PremisesChanged => {
                "the selection targets, regimes, graph or proof changed; the old proof, fit and \
                 interval are invalidated"
            }
            Self::StaleSnapshot => {
                "the replacement is not a new snapshot of a new observation period; nothing may \
                 be reused as fresh"
            }
        }
    }

    /// The registered refusal for this invalidation.
    #[must_use]
    pub fn into_error(self) -> EstimationError {
        refuse(reason_code!("route_not_supported"), self.detail(), self.message())
    }
}

/// Outcome of comparing an old identity with a replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshDecision {
    /// The proof is reusable; the result must be re-evaluated on the new period.
    Reusable,
    /// The old result is invalidated.
    Invalidated(RefreshInvalidation),
}

/// Compare the old identity with a replacement. Checks run in a fixed order (horizon, lag
/// alignment, intervention history, premises, snapshot/period), each independently
/// sufficient to invalidate the prepared route. Which products that route's change leaves
/// in place is [`refresh_survival`].
#[must_use]
pub fn decide_refresh(
    old: &TemporalWindowIdentity,
    new: &TemporalWindowIdentity,
) -> RefreshDecision {
    let invalidated = RefreshDecision::Invalidated;
    if old.horizon != new.horizon || new.horizon != TEMPORAL_REFRESH_HORIZON {
        return invalidated(RefreshInvalidation::HorizonChanged);
    }
    if old.lag_alignment != new.lag_alignment {
        return invalidated(RefreshInvalidation::LagAlignmentChanged);
    }
    if old.intervention_history != new.intervention_history {
        return invalidated(RefreshInvalidation::InterventionHistoryChanged);
    }
    if old.selection_targets != new.selection_targets
        || old.regimes != new.regimes
        || old.graph_id != new.graph_id
        || old.proof_id != new.proof_id
    {
        return invalidated(RefreshInvalidation::PremisesChanged);
    }
    if old.period == new.period || old.snapshot_id == new.snapshot_id {
        return invalidated(RefreshInvalidation::StaleSnapshot);
    }
    RefreshDecision::Reusable
}

/// Receipt binding the old and new period, snapshot and proof of a refresh.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshReceipt {
    /// Digest of the old identity.
    pub old_identity_digest: String,
    /// Digest of the new identity.
    pub new_identity_digest: String,
    /// Old observation period.
    pub old_period: ObservationPeriod,
    /// New observation period.
    pub new_period: ObservationPeriod,
    /// Old snapshot id.
    pub old_snapshot_id: String,
    /// New snapshot id.
    pub new_snapshot_id: String,
    /// The proof id, unchanged by a reusable refresh.
    pub proof_id: String,
    /// Whether an interval or fit existed for the old window and was invalidated.
    pub interval_invalidated: bool,
    /// Always [`TEMPORAL_REFRESH_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
    /// Seal over every field above.
    pub digest: String,
}

impl RefreshReceipt {
    /// The seal this receipt's fields imply.
    #[must_use]
    pub fn compute_digest(&self) -> String {
        let mut d = CanonicalWriter::new(RECEIPT_TAG);
        d.text(&self.old_identity_digest);
        d.text(&self.new_identity_digest);
        for period in [self.old_period, self.new_period] {
            d.text(&period.start.to_string());
            d.text(&period.end.to_string());
        }
        d.text(&self.old_snapshot_id);
        d.text(&self.new_snapshot_id);
        d.text(&self.proof_id);
        d.text(if self.interval_invalidated { "interval_invalidated" } else { "no_interval" });
        d.text(self.inference_claim);
        hex_digest(&d)
    }

    fn bind(old: &TemporalWindowIdentity, new: &TemporalWindowIdentity, interval: bool) -> Self {
        let mut receipt = Self {
            old_identity_digest: old.digest(),
            new_identity_digest: new.digest(),
            old_period: old.period,
            new_period: new.period,
            old_snapshot_id: old.snapshot_id.clone(),
            new_snapshot_id: new.snapshot_id.clone(),
            proof_id: new.proof_id.clone(),
            interval_invalidated: interval,
            inference_claim: TEMPORAL_REFRESH_INFERENCE_CLAIM,
            digest: String::new(),
        };
        receipt.digest = receipt.compute_digest();
        receipt
    }

    /// Check the seal against the fields.
    ///
    /// # Errors
    /// `route_not_supported` with `temporal_refresh.receipt_digest_mismatch` when a field was
    /// changed without resealing.
    pub fn verify(&self) -> Result<(), EstimationError> {
        if self.digest == self.compute_digest() {
            Ok(())
        } else {
            Err(refuse(
                reason_code!("route_not_supported"),
                "temporal_refresh.receipt_digest_mismatch",
                "the receipt's seal does not match its fields",
            ))
        }
    }
}

/// A prepared result held with the identity it is valid for.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldTemporalResult<T> {
    /// What the result is valid for.
    pub identity: TemporalWindowIdentity,
    /// The evaluated value.
    pub value: T,
    /// Whether an interval or fit exists for this window.
    pub interval_present: bool,
}

/// Refresh a held result onto a replacement identity by re-evaluating it there.
///
/// `evaluate` is called with the replacement identity only after the proof is judged
/// reusable; its value is a fresh evaluation, never the old value. The returned result
/// carries no interval.
///
/// # Errors
/// `invalid_argument` (`temporal_refresh.invalid_replacement`) for an empty period, no units
/// or a missing id; the typed [`RefreshInvalidation`] refusal (`route_not_supported`) when the
/// proof is not reusable; any error of `evaluate`.
pub fn refresh_held<T, F>(
    held: &HeldTemporalResult<T>,
    replacement: TemporalWindowIdentity,
    evaluate: F,
) -> Result<(HeldTemporalResult<T>, RefreshReceipt), EstimationError>
where
    F: FnOnce(&TemporalWindowIdentity) -> Result<T, EstimationError>,
{
    held.identity.validate()?;
    replacement.validate()?;
    if let RefreshDecision::Invalidated(reason) = decide_refresh(&held.identity, &replacement) {
        return Err(reason.into_error());
    }
    let value = evaluate(&replacement)?;
    let receipt = RefreshReceipt::bind(&held.identity, &replacement, held.interval_present);
    Ok((HeldTemporalResult { identity: replacement, value, interval_present: false }, receipt))
}

/// Accept a refreshed result only with a receipt that matches both the held result and the
/// refreshed one, so a resealed or foreign receipt is refused.
///
/// # Errors
/// `temporal_refresh.receipt_digest_mismatch` for an unsealed edit;
/// `temporal_refresh.receipt_mismatch` when the receipt's old identity is not the held result's,
/// its new identity, period or snapshot is not the refreshed result's, or its interval flag
/// disagrees with the held result; the typed invalidation when the pair is not a reusable
/// refresh; and `temporal_refresh.stale_interval` when the refreshed result still carries an
/// interval.
pub fn accept_refresh<T>(
    held: &HeldTemporalResult<T>,
    receipt: &RefreshReceipt,
    refreshed: HeldTemporalResult<T>,
) -> Result<HeldTemporalResult<T>, EstimationError> {
    held.identity.validate()?;
    refreshed.identity.validate()?;
    receipt.verify()?;
    let mismatch = |message: &str| {
        refuse(reason_code!("route_not_supported"), "temporal_refresh.receipt_mismatch", message)
    };
    if receipt.old_identity_digest != held.identity.digest()
        || receipt.old_period != held.identity.period
        || receipt.old_snapshot_id != held.identity.snapshot_id
    {
        return Err(mismatch("the receipt's old identity is not the held result's"));
    }
    if receipt.new_identity_digest != refreshed.identity.digest()
        || receipt.new_period != refreshed.identity.period
        || receipt.new_snapshot_id != refreshed.identity.snapshot_id
        || receipt.proof_id != refreshed.identity.proof_id
    {
        return Err(mismatch("the receipt's new identity is not the refreshed result's"));
    }
    if receipt.inference_claim != TEMPORAL_REFRESH_INFERENCE_CLAIM {
        return Err(mismatch("the receipt changes the point-only inference claim"));
    }
    if receipt.interval_invalidated != held.interval_present {
        return Err(mismatch("the receipt's interval flag disagrees with the held result"));
    }
    if let RefreshDecision::Invalidated(reason) =
        decide_refresh(&held.identity, &refreshed.identity)
    {
        return Err(reason.into_error());
    }
    if refreshed.interval_present {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "temporal_refresh.stale_interval",
            "a refreshed result may not carry an interval from the old window",
        ));
    }
    Ok(refreshed)
}

/// Tick length of the observation clock. Not a stationarity claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ObservationCadence {
    tick: u64,
}

impl ObservationCadence {
    /// A cadence whose tick length is positive.
    #[must_use]
    pub const fn new(tick: u64) -> Option<Self> {
        if tick == 0 { None } else { Some(Self { tick }) }
    }

    /// Tick length in the caller's time unit.
    #[must_use]
    pub const fn tick(self) -> u64 {
        self.tick
    }
}

/// Whether the mechanism may be treated as the same law on another period.
///
/// [`MechanismStanding::DeclaredInvariant`] is a caller statement. A temporal index does not
/// produce it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MechanismStanding {
    /// No invariance across periods was declared.
    Undeclared,
    /// The caller declared the mechanism invariant across the periods being compared.
    DeclaredInvariant,
}

/// Coordinates ordered in time. This is not a stationarity, equilibrium or finite-horizon claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalIndex;

impl TemporalIndex {
    /// The standing a temporal index supplies: none.
    #[must_use]
    pub const fn mechanism_standing() -> MechanismStanding {
        MechanismStanding::Undeclared
    }
}

/// A change that invalidates only the products that read it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FactorChange {
    /// The observation period moved.
    Period,
    /// The observed population drifted.
    Population,
    /// The tick cadence changed.
    Cadence,
    /// The horizon changed, including an appended slice.
    Horizon,
    /// The initial-state law changed.
    InitialLaw,
}

/// A prepared product of one temporal window.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WindowProduct {
    /// Identification of the window's query, including the proof premises and the initial law.
    Identification,
    /// Sampling design: period, population, cadence, horizon, selection and regimes.
    Design,
    /// Mechanism and initial law. Period-bound unless invariance was declared.
    Model,
    /// Observations bound to a period, population, cadence and regime.
    Evidence,
}

impl WindowProduct {
    const ALL: [Self; 4] = [Self::Identification, Self::Design, Self::Model, Self::Evidence];
}

/// Period, population, cadence, horizon and initial law a product may read.
///
/// Period, population and horizon are copied from the window. Cadence and the initial-law id
/// are not part of [`TemporalWindowIdentity`]. An empty initial-law id means no law was
/// supplied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowFactors {
    period: ObservationPeriod,
    population: BTreeSet<String>,
    cadence: ObservationCadence,
    horizon: usize,
    initial_law: String,
}

impl WindowFactors {
    /// Factors for `window` at `cadence` under `initial_law`.
    ///
    /// # Errors
    /// `temporal_refresh.invalid_replacement` when `window` is not a usable replacement;
    /// `temporal_refresh.invalid_initial_law` when the law id is blank or padded.
    pub fn try_from_window(
        window: &TemporalWindowIdentity,
        cadence: ObservationCadence,
        initial_law: &str,
    ) -> Result<Self, EstimationError> {
        window.validate()?;
        if !initial_law.is_empty() && initial_law.trim() != initial_law {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "temporal_refresh.invalid_initial_law",
                "the initial law id is blank or padded",
            ));
        }
        Ok(Self {
            period: window.period,
            population: window.unit_ids.clone(),
            cadence,
            horizon: window.horizon,
            initial_law: initial_law.to_owned(),
        })
    }

    /// Observation period copied from the window.
    #[must_use]
    pub const fn period(&self) -> ObservationPeriod {
        self.period
    }

    /// Unit lineage copied from the window.
    #[must_use]
    pub const fn population(&self) -> &BTreeSet<String> {
        &self.population
    }

    /// Tick cadence supplied beside the window.
    #[must_use]
    pub const fn cadence(&self) -> ObservationCadence {
        self.cadence
    }

    /// Horizon copied from the window.
    #[must_use]
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// Initial-law id. Empty when no law was supplied.
    #[must_use]
    pub fn initial_law(&self) -> &str {
        &self.initial_law
    }
}

/// Which of the four products a set of changes leaves in place.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductDisposition {
    invalidated: BTreeSet<WindowProduct>,
    retained: BTreeSet<WindowProduct>,
}

impl ProductDisposition {
    fn all_retained() -> Self {
        Self { invalidated: BTreeSet::new(), retained: WindowProduct::ALL.into() }
    }

    fn invalidate(&mut self, product: WindowProduct) {
        self.retained.remove(&product);
        self.invalidated.insert(product);
    }

    /// Whether `product` was not among the readers of the changes.
    #[must_use]
    pub fn retains(&self, product: WindowProduct) -> bool {
        self.retained.contains(&product)
    }

    /// Whether `product` reads at least one of the changes.
    #[must_use]
    pub fn invalidates(&self, product: WindowProduct) -> bool {
        self.invalidated.contains(&product)
    }
}

/// Direct reads of a factor change. Premise reads are applied in [`refresh_survival`].
const fn reads(product: WindowProduct, change: FactorChange, standing: MechanismStanding) -> bool {
    match product {
        WindowProduct::Identification => {
            matches!(change, FactorChange::Horizon | FactorChange::InitialLaw)
        }
        WindowProduct::Design => matches!(
            change,
            FactorChange::Period
                | FactorChange::Population
                | FactorChange::Cadence
                | FactorChange::Horizon
        ),
        WindowProduct::Model => match change {
            FactorChange::Population
            | FactorChange::Cadence
            | FactorChange::Horizon
            | FactorChange::InitialLaw => true,
            FactorChange::Period => matches!(standing, MechanismStanding::Undeclared),
        },
        WindowProduct::Evidence => matches!(
            change,
            FactorChange::Period | FactorChange::Population | FactorChange::Cadence
        ),
    }
}

/// Factor changes between two windows. Equal factors produce an empty set.
#[must_use]
pub fn factor_changes(old: &WindowFactors, new: &WindowFactors) -> BTreeSet<FactorChange> {
    let mut changes = BTreeSet::new();
    if old.period != new.period {
        changes.insert(FactorChange::Period);
    }
    if old.population != new.population {
        changes.insert(FactorChange::Population);
    }
    if old.cadence != new.cadence {
        changes.insert(FactorChange::Cadence);
    }
    if old.horizon != new.horizon {
        changes.insert(FactorChange::Horizon);
    }
    if old.initial_law != new.initial_law {
        changes.insert(FactorChange::InitialLaw);
    }
    changes
}

/// Invalidate every product that reads one of `changes`, and retain the rest.
///
/// `standing` affects only the model's read of [`FactorChange::Period`].
#[must_use]
pub fn disposition_for(
    changes: &BTreeSet<FactorChange>,
    standing: MechanismStanding,
) -> ProductDisposition {
    let mut products = ProductDisposition::all_retained();
    for change in changes {
        for product in WindowProduct::ALL {
            if reads(product, *change, standing) {
                products.invalidate(product);
            }
        }
    }
    products
}

/// Prepared-route decision together with the products a refresh leaves in place.
///
/// The old fit does not survive, and neither does an interval from the old window. The proof
/// premise survives only when [`decide_refresh`] returns [`RefreshDecision::Reusable`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshSurvival {
    decision: RefreshDecision,
    products: ProductDisposition,
    /// Always false: the old fit is not the refreshed value.
    fit_retained: bool,
    /// Always false: an old interval is not part of the refreshed window.
    interval_retained: bool,
}

impl RefreshSurvival {
    /// The prepared route's decision. Unchanged by the product ledger.
    #[must_use]
    pub const fn decision(&self) -> RefreshDecision {
        self.decision
    }

    /// Products invalidated or retained by what the two windows read.
    #[must_use]
    pub const fn products(&self) -> &ProductDisposition {
        &self.products
    }

    /// The proof premise survived only when the prepared route is reusable.
    #[must_use]
    pub const fn proof_retained(&self) -> bool {
        matches!(self.decision, RefreshDecision::Reusable)
    }

    /// The old fit does not survive a refresh, reusable or refused.
    #[must_use]
    pub const fn fit_retained(&self) -> bool {
        self.fit_retained
    }

    /// An interval from the old window does not survive a refresh, reusable or refused.
    #[must_use]
    pub const fn interval_retained(&self) -> bool {
        self.interval_retained
    }
}

fn agree(window: &TemporalWindowIdentity, factors: &WindowFactors) -> Result<(), EstimationError> {
    window.validate()?;
    if factors.period != window.period
        || factors.population != window.unit_ids
        || factors.horizon != window.horizon
    {
        return Err(refuse(
            reason_code!("invalid_argument"),
            "temporal_refresh.factor_window_mismatch",
            "the factors' period, population or horizon is not the window's",
        ));
    }
    Ok(())
}

fn apply_premises(
    old: &TemporalWindowIdentity,
    new: &TemporalWindowIdentity,
    products: &mut ProductDisposition,
) {
    let graph = old.graph_id != new.graph_id;
    let proof = old.proof_id != new.proof_id;
    let lag = old.lag_alignment != new.lag_alignment;
    let intervention = old.intervention_history != new.intervention_history;
    let selection = old.selection_targets != new.selection_targets;
    let regimes = old.regimes != new.regimes;
    if graph || proof || lag || intervention || selection || regimes {
        products.invalidate(WindowProduct::Identification);
    }
    if lag || intervention || regimes {
        products.invalidate(WindowProduct::Model);
    }
    if selection || regimes {
        products.invalidate(WindowProduct::Design);
    }
    if regimes {
        products.invalidate(WindowProduct::Evidence);
    }
}

/// Decide the prepared route and which products survive the move from `old_window` to
/// `new_window`.
///
/// `standing` must be the caller's own declaration. Pass [`TemporalIndex::mechanism_standing`]
/// when none was declared; the index never yields [`MechanismStanding::DeclaredInvariant`].
///
/// # Errors
/// `temporal_refresh.invalid_replacement` for a window that cannot be a replacement;
/// `temporal_refresh.factor_window_mismatch` when a factor set's period, population or horizon
/// is not the window it is paired with.
pub fn refresh_survival(
    old_window: &TemporalWindowIdentity,
    new_window: &TemporalWindowIdentity,
    old_factors: &WindowFactors,
    new_factors: &WindowFactors,
    standing: MechanismStanding,
) -> Result<RefreshSurvival, EstimationError> {
    agree(old_window, old_factors)?;
    agree(new_window, new_factors)?;
    let changes = factor_changes(old_factors, new_factors);
    let mut products = disposition_for(&changes, standing);
    apply_premises(old_window, new_window, &mut products);
    Ok(RefreshSurvival {
        decision: decide_refresh(old_window, new_window),
        products,
        fit_retained: false,
        interval_retained: false,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        FactorChange, HeldTemporalResult, MechanismStanding, ObservationCadence, ObservationPeriod,
        RefreshDecision, RefreshInvalidation, RefreshSurvival, TemporalIndex,
        TemporalWindowIdentity, WindowFactors, WindowProduct, decide_refresh, disposition_for,
        factor_changes, refresh_held, refresh_survival,
    };

    fn units(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().copied().map(str::to_owned).collect()
    }

    fn window(
        horizon: usize,
        period: (i64, i64),
        population: &[&str],
        snapshot: &str,
    ) -> TemporalWindowIdentity {
        TemporalWindowIdentity {
            graph_id: "two_slice".into(),
            horizon,
            lag_alignment: vec![("x".into(), 1)],
            intervention_history: vec!["a=1".into()],
            selection_targets: units(&["s"]),
            regimes: units(&["observational"]),
            period: ObservationPeriod { start: period.0, end: period.1 },
            unit_ids: units(population),
            snapshot_id: snapshot.into(),
            proof_id: "proof".into(),
        }
    }

    fn factors(window: &TemporalWindowIdentity, tick: u64, law: &str) -> WindowFactors {
        let cadence = ObservationCadence::new(tick).unwrap();
        WindowFactors::try_from_window(window, cadence, law).unwrap()
    }

    fn survive(
        old: &TemporalWindowIdentity,
        new: &TemporalWindowIdentity,
        old_factors: &WindowFactors,
        new_factors: &WindowFactors,
        standing: MechanismStanding,
    ) -> RefreshSurvival {
        refresh_survival(old, new, old_factors, new_factors, standing).unwrap()
    }

    fn one(change: FactorChange) -> BTreeSet<FactorChange> {
        BTreeSet::from([change])
    }

    #[test]
    fn period_change_keeps_only_identification_when_invariance_is_undeclared() {
        let products = disposition_for(&one(FactorChange::Period), MechanismStanding::Undeclared);
        assert!(products.retains(WindowProduct::Identification));
        assert!(products.invalidates(WindowProduct::Design));
        assert!(products.invalidates(WindowProduct::Model));
        assert!(products.invalidates(WindowProduct::Evidence));
    }

    #[test]
    fn declared_period_invariance_keeps_the_model_and_still_drops_the_fit() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(2, (10, 20), &["u1"], "snap_new");
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            MechanismStanding::DeclaredInvariant,
        );
        assert!(survival.proof_retained());
        assert!(survival.products().retains(WindowProduct::Identification));
        assert!(survival.products().retains(WindowProduct::Model));
        assert!(survival.products().invalidates(WindowProduct::Design));
        assert!(survival.products().invalidates(WindowProduct::Evidence));
        let held = HeldTemporalResult { identity: old, value: 1.0_f64, interval_present: true };
        let (refreshed, receipt) = refresh_held(&held, new, |_| Ok(4.0_f64)).unwrap();
        assert_eq!(refreshed.value.to_bits(), 4.0_f64.to_bits());
        assert!(!refreshed.interval_present);
        assert!(receipt.interval_invalidated);
    }

    #[test]
    fn population_drift_on_a_new_period_keeps_identification() {
        let old = window(2, (0, 10), &["u1", "u2"], "snap_old");
        let new = window(2, (10, 20), &["u1", "u3"], "snap_new");
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            TemporalIndex::mechanism_standing(),
        );
        let changes = factor_changes(&factors(&old, 1, "law"), &factors(&new, 1, "law"));
        assert_eq!(changes, BTreeSet::from([FactorChange::Period, FactorChange::Population]));
        assert_eq!(survival.decision(), RefreshDecision::Reusable);
        assert!(survival.proof_retained());
        assert!(survival.products().retains(WindowProduct::Identification));
        assert!(survival.products().invalidates(WindowProduct::Design));
        assert!(survival.products().invalidates(WindowProduct::Model));
        assert!(survival.products().invalidates(WindowProduct::Evidence));
        assert_eq!(decide_refresh(&old, &new), RefreshDecision::Reusable);
    }

    #[test]
    fn cadence_change_keeps_identification_and_drops_the_clock_readers() {
        let products = disposition_for(&one(FactorChange::Cadence), MechanismStanding::Undeclared);
        assert!(products.retains(WindowProduct::Identification));
        assert!(products.invalidates(WindowProduct::Design));
        assert!(products.invalidates(WindowProduct::Model));
        assert!(products.invalidates(WindowProduct::Evidence));
        let same = window(2, (0, 10), &["u1"], "snap_old");
        let moved = window(2, (10, 20), &["u1"], "snap_new");
        let changes = factor_changes(&factors(&same, 1, "law"), &factors(&moved, 2, "law"));
        assert!(changes.contains(&FactorChange::Cadence));
        assert!(!changes.contains(&FactorChange::Horizon));
    }

    #[test]
    fn horizon_change_keeps_evidence_and_refuses_the_prepared_route() {
        let products = disposition_for(&one(FactorChange::Horizon), MechanismStanding::Undeclared);
        assert!(products.invalidates(WindowProduct::Identification));
        assert!(products.invalidates(WindowProduct::Design));
        assert!(products.invalidates(WindowProduct::Model));
        assert!(products.retains(WindowProduct::Evidence));
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(3, (0, 10), &["u1"], "snap_new");
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            MechanismStanding::Undeclared,
        );
        assert_eq!(
            survival.decision(),
            RefreshDecision::Invalidated(RefreshInvalidation::HorizonChanged)
        );
        assert!(!survival.proof_retained());
        assert!(survival.products().retains(WindowProduct::Evidence));
        assert!(!survival.fit_retained());
        assert!(!survival.interval_retained());
    }

    #[test]
    fn initial_law_change_keeps_design_and_evidence() {
        let products =
            disposition_for(&one(FactorChange::InitialLaw), MechanismStanding::DeclaredInvariant);
        assert!(products.invalidates(WindowProduct::Identification));
        assert!(products.invalidates(WindowProduct::Model));
        assert!(products.retains(WindowProduct::Design));
        assert!(products.retains(WindowProduct::Evidence));
    }

    #[test]
    fn changed_initial_law_drops_identification_while_the_proof_premise_stays() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(2, (10, 20), &["u1"], "snap_new");
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law_a"),
            &factors(&new, 1, "law_b"),
            TemporalIndex::mechanism_standing(),
        );
        assert!(survival.proof_retained());
        assert!(survival.products().invalidates(WindowProduct::Identification));
        assert!(survival.products().invalidates(WindowProduct::Model));
        assert!(survival.products().invalidates(WindowProduct::Design));
        assert!(survival.products().invalidates(WindowProduct::Evidence));
        assert!(!survival.fit_retained());
        assert!(!survival.interval_retained());
    }

    #[test]
    fn temporal_index_does_not_declare_mechanism_invariance() {
        assert_eq!(TemporalIndex::mechanism_standing(), MechanismStanding::Undeclared);
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(2, (10, 20), &["u1"], "snap_new");
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, ""),
            &factors(&new, 1, ""),
            TemporalIndex::mechanism_standing(),
        );
        assert!(survival.products().invalidates(WindowProduct::Model));
        assert!(survival.products().retains(WindowProduct::Identification));
    }

    #[test]
    fn declared_invariance_does_not_keep_the_model_across_other_factor_changes() {
        for change in [
            FactorChange::Population,
            FactorChange::Cadence,
            FactorChange::Horizon,
            FactorChange::InitialLaw,
        ] {
            let products = disposition_for(&one(change), MechanismStanding::DeclaredInvariant);
            assert!(products.invalidates(WindowProduct::Model), "{change:?}");
        }
    }

    #[test]
    fn stale_proof_premise_does_not_survive_a_graph_change() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let mut new = old.clone();
        new.graph_id = "other_graph".into();
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            MechanismStanding::Undeclared,
        );
        assert_eq!(
            survival.decision(),
            RefreshDecision::Invalidated(RefreshInvalidation::PremisesChanged)
        );
        assert!(!survival.proof_retained());
        assert!(survival.products().invalidates(WindowProduct::Identification));
        assert!(survival.products().retains(WindowProduct::Model));
    }

    #[test]
    fn reusable_period_refresh_drops_the_fit_and_interval() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(2, (10, 20), &["u1"], "snap_new");
        let held =
            HeldTemporalResult { identity: old.clone(), value: 1.0_f64, interval_present: true };
        let (refreshed, receipt) = refresh_held(&held, new.clone(), |_| Ok(2.0_f64)).unwrap();
        assert_eq!(refreshed.value.to_bits(), 2.0_f64.to_bits());
        assert!(!refreshed.interval_present);
        assert!(receipt.interval_invalidated);
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            MechanismStanding::Undeclared,
        );
        assert!(survival.proof_retained());
        assert!(!survival.fit_retained());
        assert!(!survival.interval_retained());
        assert!(survival.products().retains(WindowProduct::Identification));
    }

    #[test]
    fn same_period_keeps_unread_products_and_refuses_a_stale_snapshot() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(2, (0, 10), &["u1"], "snap_new");
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            MechanismStanding::Undeclared,
        );
        assert_eq!(
            survival.decision(),
            RefreshDecision::Invalidated(RefreshInvalidation::StaleSnapshot)
        );
        assert!(!survival.proof_retained());
        assert!(survival.products().retains(WindowProduct::Identification));
        assert!(survival.products().retains(WindowProduct::Design));
        assert!(survival.products().retains(WindowProduct::Model));
        assert!(survival.products().retains(WindowProduct::Evidence));
    }

    #[test]
    fn unchanged_window_retains_every_product() {
        let products = disposition_for(&BTreeSet::new(), MechanismStanding::Undeclared);
        for product in WindowProduct::ALL {
            assert!(products.retains(product));
            assert!(!products.invalidates(product));
        }
    }

    #[test]
    fn lag_alignment_change_drops_identification_and_model_only() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let mut new = old.clone();
        new.lag_alignment = vec![("x".into(), 2)];
        let survival = survive(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&new, 1, "law"),
            MechanismStanding::DeclaredInvariant,
        );
        assert_eq!(
            survival.decision(),
            RefreshDecision::Invalidated(RefreshInvalidation::LagAlignmentChanged)
        );
        assert!(!survival.proof_retained());
        assert!(survival.products().invalidates(WindowProduct::Identification));
        assert!(survival.products().invalidates(WindowProduct::Model));
        assert!(survival.products().retains(WindowProduct::Design));
        assert!(survival.products().retains(WindowProduct::Evidence));
    }

    #[test]
    fn blank_initial_law_is_refused() {
        let sample = window(2, (0, 10), &["u1"], "snap_old");
        let error =
            WindowFactors::try_from_window(&sample, ObservationCadence::new(1).unwrap(), "  ")
                .unwrap_err();
        let crate::error::EstimationError::Refused { code, message } = error else {
            panic!("coded refusal");
        };
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("temporal_refresh.invalid_initial_law"), "{message}");
    }

    #[test]
    fn factors_from_another_window_are_refused() {
        let old = window(2, (0, 10), &["u1"], "snap_old");
        let new = window(2, (10, 20), &["u1"], "snap_new");
        let error = refresh_survival(
            &old,
            &new,
            &factors(&old, 1, "law"),
            &factors(&old, 1, "law"),
            MechanismStanding::Undeclared,
        )
        .unwrap_err();
        let crate::error::EstimationError::Refused { message, .. } = error else {
            panic!("coded refusal");
        };
        assert!(message.contains("temporal_refresh.factor_window_mismatch"), "{message}");
    }
}
