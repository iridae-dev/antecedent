//! Refresh of an autoregressive mechanism, frequentist and Bayesian, onto a new period.
//!
//! The mechanism is `v_t = c + φ₁ v_{t-1} + … + φ_L v_{t-L} + ε_t`, fitted by least squares
//! ([`OlsRefresh`]) and, on the same rows, by the conjugate normal-inverse-gamma posterior
//! ([`ConjugateOlsPosterior`]). Both are one target: the lag coefficients and the innovation
//! variance of one mechanism. A new observation period is applied with the products-that-read-it
//! rules of [`antecedent_estimate::temporal_refresh::refresh_survival`]:
//!
//! * the model reads the period unless the caller declared the mechanism invariant across
//!   periods ([`MechanismStanding::DeclaredInvariant`]); it always reads the population, the
//!   cadence, the horizon, the initial law, the lag alignment, the intervention history and the
//!   regimes. When the model survives, the new period's rows extend the summary in place and the
//!   posterior of the earlier periods is the prior of the new one (a prior update). The result
//!   equals a fresh fit of the retained rows ([`PeriodRefresh::check_against_fresh`]) and the
//!   batch posterior of all rows under the stated prior;
//! * otherwise the fit restarts: the old summary and posterior are dropped and the new period
//!   alone is fitted under the originally stated prior. The earlier posterior is never the
//!   prior of a period whose mechanism was not declared the same, because that would assume the
//!   stationarity a temporal index does not give;
//! * a period that overlaps the summarized one, or reuses its snapshot, is refused: the same
//!   observations would enter twice. A period that starts after the summarized one ends
//!   leaves a gap. The gap is not filled and not assumed stationary: the rows that would span
//!   it are not formed, and the new period contributes its own rows only.
//!
//! Next attempt when the mechanism is not invariant across periods: a model with period-level
//! parameters (a hierarchical or random-walk coefficient model). Neither is fitted here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_estimate::temporal_refresh::{
    FactorChange, MechanismStanding, ObservationCadence, TemporalWindowIdentity, WindowFactors,
    WindowProduct, factor_changes, refresh_survival,
};
use antecedent_io::period_refresh_artifact::{
    BlockWire, LeastSquaresWire, PERIOD_EXTENDED, PERIOD_REFRESH_ARTIFACT_FEATURE,
    PERIOD_REFRESH_ARTIFACT_KIND, PERIOD_REFRESH_ARTIFACT_VERSION, PERIOD_REFRESH_SUMMARY_FEATURE,
    PERIOD_REFRESH_SUMMARY_KIND, PERIOD_REFRESH_SUMMARY_RETENTION, PERIOD_REFRESH_SUMMARY_VERSION,
    PERIOD_RESTARTED, PERIOD_START, PeriodEventWire, PeriodRefreshArtifactWire,
    PeriodRefreshSummaryWire, PosteriorWire, PriorWire, RowWire, SummaryLedgerWire, product_label,
    standing_label,
};
use antecedent_io::temporal_refresh_artifact::WindowIdentityWire;
use antecedent_state::retention::{RetainedOperation, RetentionMode};
use antecedent_state::{
    ConjugateOlsPosterior, ConjugateOlsPrior, ConjugateOlsSummary, InvalidationLog,
    LinearOlsSuffStats, OlsLedgerState, OlsRefresh, RefreshOutcome, RefreshRow, StateError,
    TemporalFactor, merge_within_tolerance,
};

/// How a period refresh reached the cached results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PeriodRefreshOutcome {
    /// The new period extended the summary and the posterior in place.
    Extended {
        /// Whether the rows spanning the boundary were formed (no gap).
        contiguous: bool,
    },
    /// The summary and the posterior were dropped and fitted again on the new period alone.
    Restarted {
        /// The products the changes between the two windows invalidate.
        invalidated: Vec<WindowProduct>,
    },
}

/// What is declared about the series and the mechanism when the first period is fitted.
#[derive(Clone, Copy, Debug)]
pub struct PeriodRefreshSpec<'a> {
    /// Identity of the series, kept across its periods.
    pub series_id: &'a str,
    /// Tick cadence of the observation clock.
    pub cadence: ObservationCadence,
    /// Initial-law id (empty when none was supplied).
    pub initial_law: &'a str,
    /// Whether the caller declared the mechanism invariant across periods.
    pub standing: MechanismStanding,
    /// Autoregressive order `L`.
    pub lags: usize,
}

/// The rows one period contributed to the live fit, under the period's snapshot id.
#[derive(Clone, Debug, PartialEq)]
pub struct RefreshBlock {
    /// Snapshot the rows came from; also the label the posterior used the block under.
    pub snapshot_id: String,
    /// The retained rows of the block.
    pub rows: Vec<RefreshRow>,
}

/// One period applied to the mechanism, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct RefreshEvent {
    /// The window of the period.
    pub window: TemporalWindowIdentity,
    /// Tick cadence the period was declared with.
    pub cadence: ObservationCadence,
    /// Initial-law id the period was declared with.
    pub initial_law: String,
    /// How it reached the cached results (`None` for the first period).
    pub outcome: Option<PeriodRefreshOutcome>,
}

/// An autoregressive mechanism kept across observation periods.
#[derive(Clone, Debug)]
pub struct PeriodRefresh {
    blocks: Vec<RefreshBlock>,
    events: Vec<RefreshEvent>,
    lags: usize,
    standing: MechanismStanding,
    window: TemporalWindowIdentity,
    factors: WindowFactors,
    series_id: String,
    source: String,
    tail: Vec<f64>,
    fit: OlsRefresh,
    prior: ConjugateOlsPrior,
    posterior: ConjugateOlsPosterior,
    snapshots: BTreeSet<String>,
    last_invalidated: Vec<WindowProduct>,
    log: InvalidationLog,
}

/// The window factors that differ between two windows, as the state log names them, for a
/// family that carries no cadence or initial-law declaration of its own: period, population
/// and horizon from the identities, plus the premises they carry.
#[must_use]
pub fn window_factor_changes(
    old: &TemporalWindowIdentity,
    new: &TemporalWindowIdentity,
) -> BTreeSet<TemporalFactor> {
    let mut changes = BTreeSet::new();
    if old.period != new.period {
        changes.insert(FactorChange::Period);
    }
    if old.unit_ids != new.unit_ids {
        changes.insert(FactorChange::Population);
    }
    if old.horizon != new.horizon {
        changes.insert(FactorChange::Horizon);
    }
    temporal_factors(old, new, &changes)
}

/// The window factors that differ between two windows, as the state log names them: the
/// factor changes plus the premises the window identity carries.
fn temporal_factors(
    old: &TemporalWindowIdentity,
    new: &TemporalWindowIdentity,
    changes: &BTreeSet<FactorChange>,
) -> BTreeSet<TemporalFactor> {
    let mut out: BTreeSet<TemporalFactor> = changes
        .iter()
        .map(|change| match change {
            FactorChange::Period => TemporalFactor::Period,
            FactorChange::Population => TemporalFactor::Population,
            FactorChange::Cadence => TemporalFactor::Cadence,
            FactorChange::Horizon => TemporalFactor::Horizon,
            FactorChange::InitialLaw => TemporalFactor::InitialLaw,
        })
        .collect();
    if old.lag_alignment != new.lag_alignment
        || old.intervention_history != new.intervention_history
    {
        out.insert(TemporalFactor::History);
    }
    if old.selection_targets != new.selection_targets {
        out.insert(TemporalFactor::Selection);
    }
    if old.regimes != new.regimes {
        out.insert(TemporalFactor::Regime);
    }
    if old.graph_id != new.graph_id || old.proof_id != new.proof_id {
        out.insert(TemporalFactor::Premise);
    }
    out
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::InvalidEvent(message.into())
}

/// Design rows `[1, v_{t-1}, …, v_{t-L}] → v_t` for every target of `series` from index
/// `lags`, with the absolute time index of the first element of `series`.
fn lagged_rows(
    series: &[f64],
    first_time: i64,
    lags: usize,
) -> Result<Vec<RefreshRow>, StateError> {
    let mut rows = Vec::new();
    for target in lags..series.len() {
        let mut x = Vec::with_capacity(lags + 1);
        x.push(1.0);
        for lag in 1..=lags {
            let earlier = target - lag;
            x.push(series.get(earlier).copied().unwrap_or(f64::NAN));
        }
        let y = series.get(target).copied().unwrap_or(f64::NAN);
        let offset = i64::try_from(target).map_err(|_| invalid("series too long"))?;
        let time = first_time.checked_add(offset).ok_or_else(|| invalid("time overflow"))?;
        let id = u64::try_from(time).map_err(|_| invalid("time indices must be nonnegative"))?;
        rows.push(RefreshRow::new(id, x, y));
    }
    Ok(rows)
}

fn window_stats(ncols: usize, rows: &[RefreshRow]) -> Result<LinearOlsSuffStats, StateError> {
    let mut stats = LinearOlsSuffStats::new(ncols)?;
    for row in rows {
        stats.append_row(&row.x, row.y)?;
    }
    Ok(stats)
}

fn estimation(error: &impl std::fmt::Display) -> StateError {
    invalid(error.to_string())
}

fn expected_len(window: &TemporalWindowIdentity) -> Result<usize, StateError> {
    let span = window.period.end.checked_sub(window.period.start).unwrap_or(0);
    usize::try_from(span).map_err(|_| invalid("period span is not a length"))
}

fn event_wire(event: &RefreshEvent) -> PeriodEventWire {
    let (outcome, contiguous, invalidated) = match &event.outcome {
        None => (PERIOD_START, false, Vec::new()),
        Some(PeriodRefreshOutcome::Extended { contiguous }) => {
            (PERIOD_EXTENDED, *contiguous, Vec::new())
        }
        Some(PeriodRefreshOutcome::Restarted { invalidated }) => (
            PERIOD_RESTARTED,
            false,
            invalidated.iter().map(|p| product_label(*p).into()).collect(),
        ),
    };
    PeriodEventWire {
        window: WindowIdentityWire::from_identity(&event.window),
        cadence_tick: event.cadence.tick(),
        initial_law: event.initial_law.clone(),
        outcome: outcome.into(),
        contiguous,
        invalidated,
    }
}

fn tail_of(series: &[f64], lags: usize) -> Vec<f64> {
    let from = series.len().saturating_sub(lags);
    series.get(from..).map(<[f64]>::to_vec).unwrap_or_default()
}

fn summary_close(left: &ConjugateOlsSummary, right: &ConjugateOlsSummary) -> bool {
    let close = |a: &[f64], b: &[f64]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| merge_within_tolerance(*x, *y))
    };
    left.n == right.n
        && close(&left.mean, &right.mean)
        && close(&left.precision, &right.precision)
        && merge_within_tolerance(left.shape, right.shape)
        && merge_within_tolerance(left.rate, right.rate)
}

impl PeriodRefresh {
    /// Fit the first period: one value per time index of `window.period`.
    ///
    /// `prior` is the stated prior over `[intercept, φ₁, …, φ_L]` and the innovation variance.
    ///
    /// # Errors
    /// An unusable window, a series whose length is not the period's, `lags == 0`, a prior of
    /// the wrong width, or fewer values than `lags + 1`.
    pub fn start(
        spec: &PeriodRefreshSpec<'_>,
        window: &TemporalWindowIdentity,
        prior: &ConjugateOlsPrior,
        values: &[f64],
    ) -> Result<Self, StateError> {
        let PeriodRefreshSpec { series_id, cadence, initial_law, standing, lags } = *spec;
        if lags == 0 || prior.ncols() != lags + 1 {
            return Err(invalid("a lag order of at least one and a prior of width lags + 1"));
        }
        let factors = WindowFactors::try_from_window(window, cadence, initial_law)
            .map_err(|e| estimation(&e))?;
        if values.len() != expected_len(window)? {
            return Err(invalid("one value per time index of the period"));
        }
        let rows = lagged_rows(values, window.period.start, lags)?;
        let source = format!("{series_id}@{}", window.snapshot_id);
        let fit = OlsRefresh::from_rows(source.as_str(), lags + 1, rows.clone())?;
        let mut posterior = ConjugateOlsPosterior::new(prior);
        posterior.update(prior, &window.snapshot_id, &window_stats(lags + 1, &rows)?)?;
        let block = RefreshBlock { snapshot_id: window.snapshot_id.clone(), rows };
        Ok(Self {
            blocks: vec![block],
            events: vec![RefreshEvent {
                window: window.clone(),
                cadence,
                initial_law: initial_law.to_owned(),
                outcome: None,
            }],
            lags,
            standing,
            window: window.clone(),
            factors,
            series_id: series_id.to_owned(),
            source,
            tail: tail_of(values, lags),
            fit,
            prior: prior.clone(),
            posterior,
            snapshots: BTreeSet::from([window.snapshot_id.clone()]),
            last_invalidated: Vec::new(),
            log: InvalidationLog::new(),
        })
    }

    /// The least-squares summary and its retained rows.
    #[must_use]
    pub const fn fit(&self) -> &OlsRefresh {
        &self.fit
    }

    /// The posterior of the mechanism over every period it currently covers.
    ///
    /// # Errors
    /// A singular posterior precision.
    pub fn summary(&self) -> Result<ConjugateOlsSummary, StateError> {
        self.posterior.summary(&self.prior)
    }

    /// The prior stated when the first period was fitted.
    #[must_use]
    pub const fn prior(&self) -> &ConjugateOlsPrior {
        &self.prior
    }

    /// Autoregressive order.
    #[must_use]
    pub const fn lags(&self) -> usize {
        self.lags
    }

    /// Whether the mechanism was declared invariant across periods.
    #[must_use]
    pub const fn standing(&self) -> MechanismStanding {
        self.standing
    }

    /// Identity of the series.
    #[must_use]
    pub fn series_id(&self) -> &str {
        &self.series_id
    }

    /// The blocks the live posterior used, in the order they entered.
    #[must_use]
    pub fn blocks(&self) -> &[RefreshBlock] {
        &self.blocks
    }

    /// Every period applied so far, in order.
    #[must_use]
    pub fn events(&self) -> &[RefreshEvent] {
        &self.events
    }

    /// The products the last applied period invalidated, in dependency order: the products
    /// that read what changed between the two windows, whether the mechanism then extended or
    /// restarted. Empty before a second period is applied.
    #[must_use]
    pub fn last_invalidated(&self) -> &[WindowProduct] {
        &self.last_invalidated
    }

    /// The invalidation log: one entry per product each applied period invalidated, at the
    /// version of the event count (the first period is version one).
    #[must_use]
    pub const fn invalidation_log(&self) -> &InvalidationLog {
        &self.log
    }

    /// The window the cached results describe.
    #[must_use]
    pub const fn window(&self) -> &TemporalWindowIdentity {
        &self.window
    }

    /// Apply a new observation period.
    ///
    /// # Errors
    /// An unusable window, a series whose length is not the period's, a period that overlaps
    /// the summarized one or reuses its snapshot, or a refused window change from
    /// [`refresh_survival`]. Nothing changes.
    pub fn refresh(
        &mut self,
        window: &TemporalWindowIdentity,
        cadence: ObservationCadence,
        initial_law: &str,
        values: &[f64],
    ) -> Result<PeriodRefreshOutcome, StateError> {
        let factors = WindowFactors::try_from_window(window, cadence, initial_law)
            .map_err(|e| estimation(&e))?;
        if values.len() != expected_len(window)? {
            return Err(invalid("one value per time index of the period"));
        }
        if window.period.start < self.window.period.end
            || self.snapshots.contains(&window.snapshot_id)
        {
            return Err(invalid(
                "period_refresh.overlap: the period overlaps the summarized one or reuses its \
                 snapshot, so the same observations would enter twice; supply a later period of \
                 a new snapshot",
            ));
        }
        let survival =
            refresh_survival(&self.window, window, &self.factors, &factors, self.standing)
                .map_err(|e| estimation(&e))?;
        let invalidated: Vec<WindowProduct> = [
            WindowProduct::Identification,
            WindowProduct::Design,
            WindowProduct::Model,
            WindowProduct::Evidence,
        ]
        .into_iter()
        .filter(|product| survival.products().invalidates(*product))
        .collect();
        let extend = self.standing == MechanismStanding::DeclaredInvariant
            && survival.products().retains(WindowProduct::Model);
        let outcome = if extend {
            self.extend(window, values)?
        } else {
            self.restart(window, values)?;
            PeriodRefreshOutcome::Restarted { invalidated: invalidated.clone() }
        };
        let changes =
            temporal_factors(&self.window, window, &factor_changes(&self.factors, &factors));
        let version = antecedent_core::StateVersion::from_raw(self.events.len() as u64 + 1);
        self.log.push_temporal_change(
            version,
            &changes,
            self.standing == MechanismStanding::DeclaredInvariant,
            "observation period applied",
        );
        self.last_invalidated = invalidated;
        self.window = window.clone();
        self.factors = factors;
        self.snapshots.insert(window.snapshot_id.clone());
        self.events.push(RefreshEvent {
            window: window.clone(),
            cadence,
            initial_law: initial_law.to_owned(),
            outcome: Some(outcome.clone()),
        });
        Ok(outcome)
    }

    fn extend(
        &mut self,
        window: &TemporalWindowIdentity,
        values: &[f64],
    ) -> Result<PeriodRefreshOutcome, StateError> {
        let contiguous = window.period.start == self.window.period.end;
        let (series, first_time) = if contiguous {
            let mut joined = self.tail.clone();
            joined.extend_from_slice(values);
            let back = i64::try_from(self.tail.len()).map_err(|_| invalid("tail too long"))?;
            (joined, window.period.start - back)
        } else {
            (values.to_vec(), window.period.start)
        };
        let rows = lagged_rows(&series, first_time, self.lags)?;
        let ncols = self.lags + 1;
        if !rows.is_empty() {
            let block = window_stats(ncols, &rows)?;
            let mut posterior = self.posterior.clone();
            posterior.update(&self.prior, &window.snapshot_id, &block)?;
            self.fit.append_window(&self.source, &rows)?;
            self.posterior = posterior;
            self.blocks.push(RefreshBlock { snapshot_id: window.snapshot_id.clone(), rows });
        }
        self.tail =
            if contiguous { tail_of(&series, self.lags) } else { tail_of(values, self.lags) };
        Ok(PeriodRefreshOutcome::Extended { contiguous })
    }

    fn restart(
        &mut self,
        window: &TemporalWindowIdentity,
        values: &[f64],
    ) -> Result<(), StateError> {
        let ncols = self.lags + 1;
        let rows = lagged_rows(values, window.period.start, self.lags)?;
        let block = window_stats(ncols, &rows)?;
        let mut posterior = ConjugateOlsPosterior::new(&self.prior);
        posterior.update(&self.prior, &window.snapshot_id, &block)?;
        let source = format!("{}@{}", self.series_id, window.snapshot_id);
        self.fit.replace_source(source.as_str(), rows.clone())?;
        self.blocks = vec![RefreshBlock { snapshot_id: window.snapshot_id.clone(), rows }];
        self.source = source;
        self.posterior = posterior;
        self.tail = tail_of(values, self.lags);
        Ok(())
    }

    /// The posterior under `prior`. A posterior answers only under the prior it was built
    /// with, so any other prior is refused with the rebuild as its next attempt.
    ///
    /// # Errors
    /// `prior` is not the prior of the posterior (`StaleCache`), or the posterior is singular.
    pub fn summary_under(
        &self,
        prior: &ConjugateOlsPrior,
    ) -> Result<ConjugateOlsSummary, StateError> {
        self.posterior.summary(prior)
    }

    /// Rebuild the posterior from every retained block under `prior`, which replaces the
    /// stated prior. The least-squares summary does not read the prior and is unchanged.
    ///
    /// # Errors
    /// A prior of the wrong width. Nothing changes.
    pub fn reprior(&mut self, prior: &ConjugateOlsPrior) -> Result<(), StateError> {
        if prior.ncols() != self.lags + 1 {
            return Err(invalid("a prior of width lags + 1"));
        }
        self.posterior = self.posterior_of_blocks(&self.blocks, prior)?;
        self.prior = prior.clone();
        Ok(())
    }

    /// Remove one retained row. The least-squares summary and the posterior are rebuilt from the
    /// rows that remain, because no exact downdate is claimed
    /// ([`antecedent_state::route_contract`]).
    ///
    /// # Errors
    /// An unknown id, or the last remaining row. Nothing changes.
    pub fn evict_row(&mut self, id: u64) -> Result<RefreshOutcome, StateError> {
        let mut blocks = self.blocks.clone();
        let mut found = false;
        for block in &mut blocks {
            let before = block.rows.len();
            block.rows.retain(|row| row.id != id);
            found |= block.rows.len() != before;
        }
        blocks.retain(|block| !block.rows.is_empty());
        if !found {
            return Err(StateError::UnknownId(format!("row {id}")));
        }
        if blocks.is_empty() {
            return Err(invalid(
                "the last retained row cannot be evicted; apply a new period instead",
            ));
        }
        let posterior = self.posterior_of_blocks(&blocks, &self.prior)?;
        let outcome = self.fit.evict_row(id)?;
        self.blocks = blocks;
        self.posterior = posterior;
        Ok(outcome)
    }

    fn posterior_of_blocks(
        &self,
        blocks: &[RefreshBlock],
        prior: &ConjugateOlsPrior,
    ) -> Result<ConjugateOlsPosterior, StateError> {
        let mut posterior = ConjugateOlsPosterior::new(prior);
        for block in blocks {
            let stats = window_stats(self.lags + 1, &block.rows)?;
            posterior.update(prior, &block.snapshot_id, &stats)?;
        }
        Ok(posterior)
    }

    /// The delta ledger of this mechanism as an artifact: the stated prior and its digest,
    /// every period applied, the blocks of rows the live fit used, and the cached posterior and
    /// least squares. It is replayed once before it is returned.
    ///
    /// # Errors
    /// A singular posterior or least-squares fit, or an encoding failure.
    pub fn ledger_wire(&self) -> Result<PeriodRefreshArtifactWire, StateError> {
        let summary = self.summary()?;
        let snapshot = self.fit.snapshot();
        let beta = snapshot.beta.ok_or_else(|| invalid("the least-squares fit is singular"))?;
        let prior = PriorWire {
            mean: self.prior.mean().to_vec(),
            precision: self.prior.precision().to_vec(),
            shape: self.prior.shape(),
            rate: self.prior.rate(),
        };
        let events = self.events.iter().map(event_wire).collect();
        let blocks = self
            .blocks
            .iter()
            .map(|block| {
                let rows = block
                    .rows
                    .iter()
                    .map(|row| RowWire { id: row.id, x: row.x.clone(), y: row.y })
                    .collect();
                BlockWire::new(block.snapshot_id.clone(), rows)
            })
            .collect();
        let lags = u32::try_from(self.lags).map_err(|_| invalid("lag order too large"))?;
        PeriodRefreshArtifactWire {
            version: PERIOD_REFRESH_ARTIFACT_VERSION,
            required_features: vec![PERIOD_REFRESH_ARTIFACT_FEATURE.into()],
            kind: PERIOD_REFRESH_ARTIFACT_KIND.into(),
            series_id: self.series_id.clone(),
            lags,
            standing: standing_label(self.standing).into(),
            prior_digest: prior.digest(),
            prior,
            events,
            blocks,
            rows_used: summary.n,
            posterior: PosteriorWire {
                n: summary.n,
                mean: summary.mean,
                precision: summary.precision,
                shape: summary.shape,
                rate: summary.rate,
            },
            least_squares: LeastSquaresWire {
                n: snapshot.n,
                beta,
                residual_variance: snapshot.residual_variance,
            },
            seal: String::new(),
        }
        .checked()
        .map_err(|error| estimation(&error))
    }

    /// Export the ledger as a retained artifact; see [`Self::ledger_wire`].
    ///
    /// # Errors
    /// See [`Self::ledger_wire`].
    pub fn export_ledger(&self) -> Result<Vec<u8>, StateError> {
        self.ledger_wire()?.export().map_err(|error| estimation(&error))
    }

    /// Export the least-squares fit of this mechanism in its summary-only form, which keeps no
    /// rows; see [`SummaryOnlyFit`]. This mechanism keeps its rows and is not changed.
    ///
    /// # Errors
    /// A singular fit, or an encoding failure.
    pub fn export_summary_only(&self) -> Result<Vec<u8>, StateError> {
        SummaryOnlyFit::from_fit(&self.fit.clone().into_summary_only())?.export()
    }

    /// Compare the cached least-squares result with a fresh rebuild from the retained rows, and
    /// the cached posterior with the batch posterior of those rows under the stated prior.
    ///
    /// # Errors
    /// The first field that disagrees.
    pub fn check_against_fresh(&self) -> Result<(), StateError> {
        self.fit.check_against_fresh()?;
        let rows = self.fit.rows();
        let block = window_stats(self.lags + 1, rows)?;
        let mut batch = ConjugateOlsPosterior::new(&self.prior);
        batch.update(&self.prior, "all rows", &block)?;
        if !summary_close(&self.summary()?, &batch.summary(&self.prior)?) {
            return Err(StateError::StaleCache(
                "the cached posterior differs from the batch posterior of the retained rows".into(),
            ));
        }
        Ok(())
    }
}

/// The retained summary of a least-squares fit that stopped retaining its rows, as read from its
/// artifact by a consumer that never held the rows.
///
/// It serves the summary: observation count, coefficients, residual variance, source digest and
/// row counts. Every operation that needs rows (a refit from rows, the eviction of a row, a
/// change of mask or weight, the replay of rows) is refused with the typed retention refusal,
/// which names the mode and the operation. The artifact is a different kind from the full
/// ledger, so a summary is never read as a full ledger.
#[derive(Clone, Debug)]
pub struct SummaryOnlyFit {
    wire: PeriodRefreshSummaryWire,
}

impl SummaryOnlyFit {
    /// The summary-only artifact of `fit`, which must already be reduced with
    /// [`OlsRefresh::into_summary_only`]. A fit that still retains rows is refused rather than
    /// reduced silently.
    ///
    /// # Errors
    /// The fit retains its rows, its Gram is singular, or the summary does not verify.
    pub fn from_fit(fit: &OlsRefresh) -> Result<Self, StateError> {
        if fit.retention() != RetentionMode::SummaryOnly {
            return Err(invalid(format!(
                "{} is not summary-only; reduce the fit with into_summary_only before exporting \
                 its summary",
                fit.retention()
            )));
        }
        let snapshot = fit.snapshot();
        let ncols = u32::try_from(fit.stats().ncols()).map_err(|_| invalid("too many columns"))?;
        let wire = PeriodRefreshSummaryWire {
            version: PERIOD_REFRESH_SUMMARY_VERSION,
            required_features: vec![PERIOD_REFRESH_SUMMARY_FEATURE.into()],
            kind: PERIOD_REFRESH_SUMMARY_KIND.into(),
            retention: PERIOD_REFRESH_SUMMARY_RETENTION.into(),
            source: snapshot.source.to_string(),
            ncols,
            n: snapshot.n,
            xtx: snapshot.xtx,
            xty: snapshot.xty,
            yty: snapshot.yty,
            beta: snapshot.beta,
            residual_variance: snapshot.residual_variance,
            source_digest: snapshot.source_digest,
            ledger: SummaryLedgerWire {
                rows: snapshot.ledger.rows,
                live_rows: snapshot.ledger.live_rows,
                effective_n: snapshot.ledger.effective_n,
            },
            seal: String::new(),
        }
        .checked()
        .map_err(|error| estimation(&error))?;
        Ok(Self { wire })
    }

    /// Encode the artifact.
    ///
    /// # Errors
    /// An encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, StateError> {
        self.wire.export().map_err(|error| estimation(&error))
    }

    /// Decode and verify a summary-only artifact: the seal, the stated retention, the shapes and
    /// that the stored coefficients and residual variance are what the stored summary gives.
    ///
    /// # Errors
    /// Any tamper, a full ledger, or a malformed summary.
    pub fn consume(bytes: &[u8]) -> Result<Self, StateError> {
        PeriodRefreshSummaryWire::consume(bytes)
            .map(|wire| Self { wire })
            .map_err(|error| estimation(&error))
    }

    /// Retention of the result: always summary-only.
    #[must_use]
    pub const fn retention(&self) -> RetentionMode {
        RetentionMode::SummaryOnly
    }

    /// Source the summarized rows belonged to.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.wire.source
    }

    /// Observation count.
    #[must_use]
    pub const fn n(&self) -> u64 {
        self.wire.n
    }

    /// Least-squares coefficients, when the Gram is nonsingular.
    #[must_use]
    pub fn beta(&self) -> Option<&[f64]> {
        self.wire.beta.as_deref()
    }

    /// Residual variance, when there are more rows than columns.
    #[must_use]
    pub const fn residual_variance(&self) -> Option<f64> {
        self.wire.residual_variance
    }

    /// Digest of the source and the ordered rows the summary was built from.
    #[must_use]
    pub fn source_digest(&self) -> &str {
        &self.wire.source_digest
    }

    /// Row counts of the summary.
    #[must_use]
    pub const fn ledger(&self) -> OlsLedgerState {
        OlsLedgerState {
            rows: self.wire.ledger.rows,
            live_rows: self.wire.ledger.live_rows,
            effective_n: self.wire.ledger.effective_n,
        }
    }

    /// Accept `operation` under summary-only retention, or refuse it with the mode and the
    /// operation named.
    ///
    /// # Errors
    /// The operation needs rows.
    pub fn require(&self, operation: RetainedOperation) -> Result<(), StateError> {
        self.retention().require(operation).map_err(|_| {
            invalid(format!(
                "{} cannot serve {operation}; reload the rows of this source and rebuild from them",
                self.retention()
            ))
        })
    }

    /// Fit again from retained rows. Refused: no rows are retained.
    ///
    /// # Errors
    /// Always.
    pub fn refit_from_rows(&self) -> Result<(), StateError> {
        self.require(RetainedOperation::RefitFromRetainedRows)
    }

    /// Evict one row. Refused: no rows are retained.
    ///
    /// # Errors
    /// Always.
    pub fn evict_row(&self, _id: u64) -> Result<(), StateError> {
        self.require(RetainedOperation::ExactRowEviction)
    }

    /// Change the mask or weight of one row. Refused: no rows are retained.
    ///
    /// # Errors
    /// Always.
    pub fn set_mask(&self, _id: u64, _masked: bool) -> Result<(), StateError> {
        self.require(RetainedOperation::InPlaceWeightOrMaskUpdate)
    }

    /// Replay the rows the summary was built from. Refused: no rows are retained.
    ///
    /// # Errors
    /// Always.
    pub fn replay_rows(&self) -> Result<(), StateError> {
        self.require(RetainedOperation::RowReconstruction)
    }
}
