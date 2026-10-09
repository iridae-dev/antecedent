//! Temporal extensions of the finite two-step sequence (2.3A, X5): the uncertain
//! pre-action state, the new-period refresh, and the closed dependent interval.
//!
//! * **Initial state** (`antecedent.transport.temporal_initial_state`, point only,
//!   licensed): the target two-step mean marginalized over a finite **target**
//!   initial-state law, `sum_s0 P_target(s0) R(s0)`. A source law or a single state
//!   cannot answer it, and fixing the state is a different estimand with its own
//!   label. The result exports an independent artifact.
//! * **New-period refresh** (`antecedent.transport.temporal_new_period_refresh`,
//!   point only, licensed): a held result is re-evaluated, never copied, on a
//!   replacement panel of a new period when the graph, horizon, lag alignment,
//!   intervention history, selection targets, regimes and proof are unchanged. Anything
//!   else is a typed invalidation. A stale interval never survives a refresh.
//! * **Dependent interval** (`antecedent.transport.temporal_dependent_interval`): a
//!   calibrated claim whose calibration is measured only at the release cut, so this
//!   route is **closed**: it validates its arguments through the core and then refuses
//!   with `cell_not_licensed` / `temporal_interval.route_frozen`. Its replay artifact
//!   exists at the io layer as an internal artifact.
use antecedent_core::reason_code;
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, INTERVAL_MAX_REPLICATES, INTERVAL_MIN_REPLICATES,
    ObservedStateSequence, TEMPORAL_INTERVAL_TOO_FEW_UNITS, TEMPORAL_INTERVAL_TOO_MANY_REPLICATES,
    TEMPORAL_INTERVAL_UNSUPPORTED_HISTORY, TemporalEstimator, TemporalUnitPanel, UnitHistories,
    route_frozen_refusal,
};
use antecedent_estimate::temporal_initial_state::{
    FixedStateEffect, FixedStateQuery, InitialStateSpec, MarginalizedEffect, MarginalizedQuery,
};
use antecedent_estimate::temporal_refresh::{
    HeldTemporalResult, ObservationPeriod, RefreshReceipt, TEMPORAL_REFRESH_HORIZON,
    TemporalWindowIdentity, accept_refresh, refresh_held,
};
use antecedent_io::IoError;
use antecedent_io::temporal_initial_state_artifact::{
    InitialStateReplay, PanelSummaryWire, TemporalInitialStateArtifactWire,
    TemporalInitialStateConsumeLimits, TemporalPremisesWire,
};
use antecedent_io::temporal_refresh_artifact::{
    RefreshInputs, RefreshReplay, TemporalRefreshArtifactWire, WindowIdentityWire, expected_receipt,
};
use std::collections::BTreeSet;
use std::convert::Infallible;

fn refused(code: &'static str, detail: &str, message: &str) -> IoError {
    IoError::Refused { code, message: format!("{detail}: {message}") }
}

/// A declared observation window of a prepared result.
#[derive(Clone, Debug)]
pub struct TemporalWindow {
    /// Time horizon; a refreshable result has two.
    pub horizon: usize,
    /// Each coordinate name with the slice it is aligned to.
    pub lag_alignment: Vec<(String, u8)>,
    /// Ordered action label of each step.
    pub intervention_history: Vec<String>,
    /// Time-indexed selection targets.
    pub selection_targets: BTreeSet<String>,
    /// Half-open observation period.
    pub period: ObservationPeriod,
    /// Overrides the premises' graph id for this window (a changed graph).
    pub graph_id: Option<String>,
    /// Overrides the premises' proof id for this window (a changed proof).
    pub proof_id: Option<String>,
}

fn identity_for(
    premises: &TemporalPremisesWire,
    window: &TemporalWindow,
    panel: &TemporalUnitPanel,
    sequence: [u32; 2],
) -> Result<TemporalWindowIdentity, IoError> {
    let identity = TemporalWindowIdentity {
        graph_id: window.graph_id.clone().unwrap_or_else(|| premises.graph_id.clone()),
        horizon: window.horizon,
        lag_alignment: window.lag_alignment.clone(),
        intervention_history: window.intervention_history.clone(),
        selection_targets: window.selection_targets.clone(),
        regimes: BTreeSet::from([premises.source_regime.clone(), premises.target_regime.clone()]),
        period: window.period,
        unit_ids: panel.units().iter().map(|u| u.unit_id.to_string()).collect(),
        snapshot_id: panel.snapshot_id().to_owned(),
        proof_id: window.proof_id.clone().unwrap_or_else(|| premises.proof_id.clone()),
    };
    WindowIdentityWire::from_identity(&identity)
        .check_against(&PanelSummaryWire::from_panel(panel, sequence))?;
    Ok(identity)
}

/// What a refresh was made from, kept so it can be exported.
#[derive(Clone, Debug)]
struct RefreshLineage {
    old_identity: TemporalWindowIdentity,
    old_panel: TemporalUnitPanel,
    receipt: RefreshReceipt,
    interval_invalidated: bool,
    previous_value: f64,
}

/// A prepared target-marginal initial-state result, with the window it is valid for.
#[derive(Clone, Debug)]
pub struct TemporalInitialState {
    premises: TemporalPremisesWire,
    sequence: [u32; 2],
    query: MarginalizedQuery,
    fixed_query: Option<FixedStateQuery>,
    panel: TemporalUnitPanel,
    marginalized: MarginalizedEffect,
    fixed: Option<FixedStateEffect>,
    identity: Option<TemporalWindowIdentity>,
    lineage: Option<RefreshLineage>,
}

impl TemporalInitialState {
    /// Evaluate the target-marginal sum on a panel.
    ///
    /// `fixed_state` additionally evaluates the fixed-state estimand, under its own
    /// label. `window` declares the observation window the result is valid for; it is
    /// required to refresh the result later.
    ///
    /// # Errors
    /// `transport_missing_evidence` / `initial_state.target_law_missing` for a point
    /// state or a source law; `transport_support_failure` /
    /// `initial_state.support_gap` for a state without history support;
    /// `invalid_argument` for malformed premises or a window that is not the panel's.
    pub fn prepare(
        premises: TemporalPremisesWire,
        sequence: [u32; 2],
        spec: InitialStateSpec,
        fixed_state: Option<u32>,
        panel: TemporalUnitPanel,
        window: Option<&TemporalWindow>,
    ) -> Result<Self, IoError> {
        premises.validate()?;
        let query = MarginalizedQuery::new(sequence, spec)?;
        let marginalized = query.effect(&panel)?;
        let fixed_query = fixed_state.map(|s0| FixedStateQuery { sequence, s0 });
        let fixed = fixed_query.as_ref().map(|q| q.effect(&panel)).transpose()?;
        let identity = match window {
            Some(w) => {
                let identity = identity_for(&premises, w, &panel, sequence)?;
                if identity.graph_id != premises.graph_id
                    || identity.proof_id != premises.proof_id
                    || identity.horizon != TEMPORAL_REFRESH_HORIZON
                {
                    return Err(refused(
                        reason_code!("invalid_argument"),
                        "temporal_refresh.invalid_replacement",
                        "a prepared result's window must match its premises and a horizon of two",
                    ));
                }
                Some(identity)
            }
            None => None,
        };
        Ok(Self {
            premises,
            sequence,
            query,
            fixed_query,
            panel,
            marginalized,
            fixed,
            identity,
            lineage: None,
        })
    }

    /// The marginalized result.
    #[must_use]
    pub const fn marginalized(&self) -> &MarginalizedEffect {
        &self.marginalized
    }

    /// The fixed-state companion result, when one was requested.
    #[must_use]
    pub const fn fixed(&self) -> Option<&FixedStateEffect> {
        self.fixed.as_ref()
    }

    /// The action sequence.
    #[must_use]
    pub const fn sequence(&self) -> [u32; 2] {
        self.sequence
    }

    /// The premises.
    #[must_use]
    pub const fn premises(&self) -> &TemporalPremisesWire {
        &self.premises
    }

    /// The panel the value was evaluated on.
    #[must_use]
    pub const fn panel(&self) -> &TemporalUnitPanel {
        &self.panel
    }

    /// The window this result is valid for, when one was declared.
    #[must_use]
    pub const fn identity(&self) -> Option<&TemporalWindowIdentity> {
        self.identity.as_ref()
    }

    /// The receipt of the refresh that produced this result, if it was refreshed.
    #[must_use]
    pub fn receipt(&self) -> Option<&RefreshReceipt> {
        self.lineage.as_ref().map(|l| &l.receipt)
    }

    /// The value before the refresh that produced this result, if it was refreshed.
    #[must_use]
    pub fn previous_value(&self) -> Option<f64> {
        self.lineage.as_ref().map(|l| l.previous_value)
    }

    /// Whether the refresh that produced this result invalidated an old interval.
    #[must_use]
    pub fn interval_invalidated(&self) -> bool {
        self.lineage.as_ref().is_some_and(|l| l.interval_invalidated)
    }

    /// Export the independent initial-state artifact.
    ///
    /// # Errors
    /// An encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        TemporalInitialStateArtifactWire::checked(
            self.premises.clone(),
            self.sequence,
            &self.query,
            self.fixed_query.as_ref(),
            &self.panel,
        )?
        .export()
    }

    fn held(
        &self,
        interval_present: bool,
    ) -> Result<HeldTemporalResult<MarginalizedEffect>, IoError> {
        let identity = self.identity.clone().ok_or_else(|| {
            refused(
                reason_code!("invalid_argument"),
                "temporal_refresh.window_missing",
                "the held result declared no observation window, so it has nothing to refresh",
            )
        })?;
        Ok(HeldTemporalResult { identity, value: self.marginalized.clone(), interval_present })
    }

    /// Refresh onto a replacement panel of a new period.
    ///
    /// The value is re-evaluated on the replacement, never copied; the refreshed
    /// result carries no interval, and the receipt records whether
    /// `interval_existed` for the old window.
    ///
    /// # Errors
    /// `invalid_argument` for a missing or invalid window; the typed
    /// `route_not_supported` / `temporal_refresh.*` invalidation when the proof is not
    /// reusable; the support refusals of the evaluation.
    pub fn refresh(
        &self,
        panel: TemporalUnitPanel,
        window: &TemporalWindow,
        interval_existed: bool,
    ) -> Result<Self, IoError> {
        let held = self.held(interval_existed)?;
        let new_identity = identity_for(&self.premises, window, &panel, self.sequence)?;
        let (refreshed, receipt) =
            refresh_held(&held, new_identity, |_| self.query.effect(&panel))?;
        let refreshed = accept_refresh(&held, &receipt, refreshed)?;
        let fixed = self.fixed_query.as_ref().map(|q| q.effect(&panel)).transpose()?;
        Ok(Self {
            premises: self.premises.clone(),
            sequence: self.sequence,
            query: self.query.clone(),
            fixed_query: self.fixed_query,
            panel,
            marginalized: refreshed.value,
            fixed,
            identity: Some(refreshed.identity),
            lineage: Some(RefreshLineage {
                old_identity: held.identity,
                old_panel: self.panel.clone(),
                receipt,
                interval_invalidated: interval_existed,
                previous_value: self.marginalized.value,
            }),
        })
    }

    /// Export the artifact of the refresh that produced this result.
    ///
    /// # Errors
    /// `invalid_argument` when this result was not produced by a refresh, or an
    /// encoding failure.
    pub fn export_refresh(&self) -> Result<Vec<u8>, IoError> {
        let (Some(lineage), Some(new_identity)) = (&self.lineage, &self.identity) else {
            return Err(refused(
                reason_code!("invalid_argument"),
                "temporal_refresh.no_refresh",
                "only a result produced by a refresh has a refresh artifact",
            ));
        };
        TemporalRefreshArtifactWire::checked(&RefreshInputs {
            premises: &self.premises,
            sequence: self.sequence,
            query: &self.query,
            old_identity: &lineage.old_identity,
            old_panel: &lineage.old_panel,
            new_identity,
            new_panel: &self.panel,
            interval_invalidated: lineage.interval_invalidated,
        })?
        .export()
    }

    /// The artifact of a refresh onto `panel` and `window`, whichever way it is
    /// decided: a reusable refresh carries the re-evaluated point and receipt, an
    /// invalidated one carries its typed reason and no result.
    ///
    /// # Errors
    /// `invalid_argument` for a missing or invalid window, or an encoding failure.
    pub fn refresh_record(
        &self,
        panel: &TemporalUnitPanel,
        window: &TemporalWindow,
        interval_existed: bool,
    ) -> Result<Vec<u8>, IoError> {
        let held = self.held(interval_existed)?;
        let new_identity = identity_for(&self.premises, window, panel, self.sequence)?;
        TemporalRefreshArtifactWire::checked(&RefreshInputs {
            premises: &self.premises,
            sequence: self.sequence,
            query: &self.query,
            old_identity: &held.identity,
            old_panel: &self.panel,
            new_identity: &new_identity,
            new_panel: panel,
            interval_invalidated: interval_existed,
        })?
        .export()
    }

    /// The receipt a reusable refresh from this window to `new` would carry.
    #[must_use]
    pub fn expected_receipt(
        old: &TemporalWindowIdentity,
        new: &TemporalWindowIdentity,
        interval_invalidated: bool,
    ) -> RefreshReceipt {
        expected_receipt(old, new, interval_invalidated)
    }
}

/// Independently replay an initial-state artifact.
///
/// # Errors
/// Any seal, law, support, label, identity or value mismatch.
pub fn consume_temporal_initial_state_artifact(
    bytes: &[u8],
    limits: &TemporalInitialStateConsumeLimits,
) -> Result<(TemporalInitialStateArtifactWire, InitialStateReplay), IoError> {
    TemporalInitialStateArtifactWire::consume(bytes, limits)
}

/// Independently re-decide and replay a refresh artifact.
///
/// # Errors
/// Any seal, window, decision, receipt, interval or value mismatch.
pub fn consume_temporal_refresh_artifact(
    bytes: &[u8],
    limits: &TemporalInitialStateConsumeLimits,
) -> Result<(TemporalRefreshArtifactWire, RefreshReplay), IoError> {
    TemporalRefreshArtifactWire::consume(bytes, limits)
}

/// The estimand a dependent interval is requested for.
#[derive(Clone, Debug)]
pub enum IntervalEstimand {
    /// The two-step response over the panel's own observed initial-state law.
    Observed,
    /// The response with the initial state held fixed.
    FixedState(u32),
    /// The response marginalized over a target initial-state law.
    Marginalized(InitialStateSpec),
}

fn validate_interval_design(config: &DependentIntervalConfig, units: usize) -> Result<(), IoError> {
    if config.replicates > INTERVAL_MAX_REPLICATES {
        return Err(refused(
            reason_code!("route_not_supported"),
            TEMPORAL_INTERVAL_TOO_MANY_REPLICATES,
            &format!(
                "{} replicates exceed the cap of {INTERVAL_MAX_REPLICATES}",
                config.replicates
            ),
        ));
    }
    let bad = |message: &str| IoError::Refused {
        code: reason_code!("invalid_argument"),
        message: message.to_owned(),
    };
    if config.replicates < INTERVAL_MIN_REPLICATES {
        return Err(bad("a dependent interval needs at least 20 replicates"));
    }
    if !(config.level > 0.0 && config.level < 1.0) {
        return Err(bad("the interval level must lie strictly between 0 and 1"));
    }
    if !(0.0..1.0).contains(&config.max_failed_fraction) {
        return Err(bad("the failed-replicate fraction must lie in [0, 1)"));
    }
    if units < config.min_units.max(2) {
        return Err(refused(
            reason_code!("too_few_clusters"),
            TEMPORAL_INTERVAL_TOO_FEW_UNITS,
            &format!("{units} repeated units, at least {} required", config.min_units.max(2)),
        ));
    }
    Ok(())
}

fn interval_estimator(
    sequence: [u32; 2],
    estimand: IntervalEstimand,
) -> Result<Box<dyn TemporalEstimator>, IoError> {
    Ok(match estimand {
        IntervalEstimand::Observed => Box::new(ObservedStateSequence { sequence }),
        IntervalEstimand::FixedState(s0) => Box::new(FixedStateQuery { sequence, s0 }),
        IntervalEstimand::Marginalized(spec) => Box::new(MarginalizedQuery::new(sequence, spec)?),
    })
}

/// The dependent-sampling interval route, **closed** while its calibration is
/// unmeasured.
///
/// The arguments are validated for real first: the unit map and snapshot (the core's
/// `temporal_interval.unknown_units` refusals), the resampling design, the unit count
/// and the estimator's support on the original panel. Only then does the route
/// refuse with `cell_not_licensed` / `temporal_interval.route_frozen`.
///
/// # Errors
/// Always: a validation refusal, or the route-frozen refusal.
pub fn temporal_dependent_interval(
    snapshot_id: &str,
    units: Option<Vec<UnitHistories>>,
    sequence: [u32; 2],
    estimand: IntervalEstimand,
    config: &DependentIntervalConfig,
) -> Result<Infallible, IoError> {
    validated_interval(snapshot_id, units, sequence, estimand, config)?;
    Err(route_frozen_refusal().into())
}

fn validated_interval(
    snapshot_id: &str,
    units: Option<Vec<UnitHistories>>,
    sequence: [u32; 2],
    estimand: IntervalEstimand,
    config: &DependentIntervalConfig,
) -> Result<(TemporalUnitPanel, Box<dyn TemporalEstimator>), IoError> {
    let panel = TemporalUnitPanel::new(snapshot_id, units)?;
    validate_interval_design(config, panel.unit_count())?;
    let estimator = interval_estimator(sequence, estimand)?;
    let all = panel.units().iter().collect::<Vec<_>>();
    estimator.estimate(&all).map_err(|error| {
        refused(
            reason_code!("route_not_supported"),
            TEMPORAL_INTERVAL_UNSUPPORTED_HISTORY,
            &format!("the original panel: {error}"),
        )
    })?;
    Ok((panel, estimator))
}

/// Prepare the original whole-unit candidate and independently replay its artifact.
/// Internal acceptance preparation only: no measured or released interval license.
///
/// # Errors
/// Original panel/design/history refusals or failed original engine/artifact replay.
#[doc(hidden)]
#[cfg(feature = "calibration-internal")]
pub fn temporal_dependent_interval_candidate(
    snapshot_id: &str,
    units: Option<Vec<UnitHistories>>,
    sequence: [u32; 2],
    estimand: IntervalEstimand,
    config: &DependentIntervalConfig,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_io::temporal_interval_artifact::TemporalIntervalArtifactWire, IoError> {
    use antecedent_io::temporal_interval_artifact::{
        IntervalEstimatorWire, TemporalIntervalArtifactWire,
    };
    let wire = match &estimand {
        IntervalEstimand::Observed => IntervalEstimatorWire::observed(sequence),
        IntervalEstimand::FixedState(s0) => {
            IntervalEstimatorWire::fixed(&FixedStateQuery { sequence, s0: *s0 })
        }
        IntervalEstimand::Marginalized(spec) => IntervalEstimatorWire::marginalized(
            sequence,
            &MarginalizedQuery::new(sequence, spec.clone())?,
        ),
    };
    let (panel, estimator) = validated_interval(snapshot_id, units, sequence, estimand, config)?;
    let interval = antecedent_estimate::temporal_dependent_interval::dependent_unit_interval(
        &panel,
        &*estimator,
        config,
        ctx,
    )?;
    TemporalIntervalArtifactWire::checked(&panel, wire, config, &interval, None, ctx)
}
