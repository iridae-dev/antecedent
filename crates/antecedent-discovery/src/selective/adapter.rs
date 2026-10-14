//! One static and one temporal selective-discovery adapter.
//!
//! An adapter retains the last successful run: its declared stage inputs, the data identity,
//! the statistics it computed and its outcome. A new request is planned against that state
//! ([`DiscoveryPlan`]) and then executed as a *candidate*; the retained state is replaced
//! only when the candidate completes.
//!
//! # What is reused, exactly
//!
//! * A request whose every stage identity equals the retained one is served from the
//!   retained outcome: no statistic is computed and no search runs.
//! * Otherwise the discovery algorithm runs again, in full, on a test that answers a request
//!   from a retained statistic when (and only when) the request is at an identical input
//!   coordinate. The saved work is the statistics. The search and orientation procedures are
//!   re-executed; a change of threshold, multiplicity control, background knowledge, search
//!   limits or orientation settings therefore reruns decisions and the adaptive test
//!   schedule over the retained statistics, and a change of anything the statistics depend
//!   on (data, test, seed, kernel policy, variable order, mask, lag-frame depth) retains
//!   none.
//! * New data invalidates the statistics: the candidate starts from an empty store. A
//!   previous graph is never used as the start of a search; there is no incremental
//!   discovery here.
//!
//! The reused result is bit-identical to a full run because every retained statistic is the
//! value the test returned for exactly that coordinate. Retaining the *skeleton state*
//! between runs is not offered: the algorithms interleave orientation with the final search
//! passes and expose no resumable state, so that stage is a closed route.
//!
//! # Score-based search
//!
//! Local-score caches are internal to a score-based run and bound to one data fingerprint.
//! No score route is delivered; [`WorkUnits::score_evaluations`] is zero by construction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_data::{TabularData, TimeSeriesData};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::fci::Fci;
use crate::pc::Pc;
use crate::pcmci::Pcmci;
use crate::pcmci_plus::PcmciPlus;

use super::contract::{
    StaticAlgorithm, StaticDiscoveryContract, TemporalAlgorithm, TemporalDiscoveryContract,
    inputs_contract_identity,
};
use super::identity::{DataIdentity, Digest, static_data_identity, temporal_data_identity};
use super::outcome::{DiscoveryOutcome, NativeResult};
use super::portable::PortableDiscoveryState;
use super::refusal::SelectiveDiscoveryError;
use super::stages::{DiscoveryPlan, DiscoveryStage, StageInputs};
use super::store::{SelectiveCi, StatisticStore};
use super::work::{WorkUnits, observe_orientation};

fn invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        remedy,
    )
}

fn statistics_invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("discovery_statistics_invalid"),
        detail,
        remedy,
    )
}

/// What one request did.
#[derive(Clone, Debug)]
pub struct SelectiveRunReport {
    /// Per-stage reuse decisions made before execution.
    pub plan: DiscoveryPlan,
    /// Work observed at the algorithm boundaries during this request.
    pub work: WorkUnits,
    /// Whether the result identity differs from the previously retained result.
    pub outcome_changed: bool,
    /// Statistics retained after the request.
    pub retained_statistics: usize,
}

struct LiveState {
    inputs: StageInputs,
    data: DataIdentity,
    contract_identity: Digest,
    store: Arc<StatisticStore>,
    outcome: DiscoveryOutcome,
    work: WorkUnits,
}

struct Prepared {
    inputs: StageInputs,
    data: DataIdentity,
    contract_identity: Digest,
    test: Arc<dyn antecedent_stats::ConditionalIndependenceTest + Send + Sync>,
    ci_identity: String,
}

fn ensure_seed(expected: u64, ctx: &ExecutionContext) -> Result<(), SelectiveDiscoveryError> {
    if ctx.rng.master_seed() == expected {
        Ok(())
    } else {
        Err(invalid(
            "selective_discovery.rng_mismatch",
            "the execution context's master seed differs from the seed frozen in the contract; \
             build the context from the contract's seed or change the contract",
        ))
    }
}

fn counts_of(native: &NativeResult) -> (u64, u64) {
    let (consumed, passes) = match native {
        NativeResult::StaticCpdag(r) => (r.performance.ci_tests, r.iterations.len()),
        NativeResult::StaticPag(r) => (r.performance.ci_tests, r.iterations.len()),
        NativeResult::TemporalDag(r) => (r.performance.ci_tests, r.iterations.len()),
        NativeResult::TemporalCpdag(r) => (r.performance.ci_tests, r.iterations.len()),
    };
    (consumed, passes as u64)
}

fn map_failure(store: &StatisticStore, error: DiscoveryError) -> SelectiveDiscoveryError {
    if store.replay_missed() {
        statistics_invalid(
            "selective_discovery.replay_miss",
            "the retained statistics do not cover a test this contract requests; resume needs \
             the receipts of a run of this exact contract on this data",
        )
    } else {
        SelectiveDiscoveryError::Discovery(error)
    }
}

fn work_between(
    before: super::store::StoreCounters,
    store: &StatisticStore,
    outcome: &DiscoveryOutcome,
    orientation: super::work::OrientationWork,
) -> WorkUnits {
    let after = store.counters();
    let (consumed, passes) = counts_of(&outcome.native);
    WorkUnits {
        ci_tests_attempted: after.attempted - before.attempted,
        ci_tests_completed: after.completed - before.completed,
        ci_tests_reused: after.reused - before.reused,
        ci_tests_consumed: consumed,
        search_passes: passes,
        orientation_runs: orientation.fixed_point_runs,
        orientation_rounds: orientation.rounds,
        orientations_applied: orientation.orientations_applied,
        score_evaluations: 0,
        algorithm_executions: 1,
    }
}

/// Plan, execute as a candidate, and return the new live state without publishing it.
fn execute_candidate(
    live: Option<&LiveState>,
    prepared: Prepared,
    execute: impl FnOnce(Arc<SelectiveCi>) -> Result<DiscoveryOutcome, DiscoveryError>,
) -> Result<(Option<LiveState>, SelectiveRunReport), SelectiveDiscoveryError> {
    let plan = DiscoveryPlan::between(live.map(|l| &l.inputs), &prepared.inputs);
    if let Some(live) = live.filter(|_| plan.is_pure_reuse()) {
        let report = SelectiveRunReport {
            plan,
            work: WorkUnits::default(),
            outcome_changed: false,
            retained_statistics: live.store.len(),
        };
        return Ok((None, report));
    }
    let statistics_reused = plan.decision(DiscoveryStage::Statistics).is_reused();
    let store = match live {
        Some(live) if statistics_reused => Arc::new(live.store.fork()),
        _ => Arc::new(StatisticStore::new()),
    };
    let ci = Arc::new(SelectiveCi::new(
        prepared.test,
        Arc::clone(&store),
        prepared.ci_identity.as_str(),
    ));
    let before = store.counters();
    let (executed, orientation) = observe_orientation(|| execute(ci));
    let outcome = executed.map_err(|error| map_failure(&store, error))?;
    let work = work_between(before, &store, &outcome, orientation);
    let outcome_changed = live.is_none_or(|l| l.outcome.result_identity != outcome.result_identity);
    let report =
        SelectiveRunReport { plan, work, outcome_changed, retained_statistics: store.len() };
    let state = LiveState {
        inputs: prepared.inputs,
        data: prepared.data,
        contract_identity: prepared.contract_identity,
        store,
        outcome,
        work,
    };
    Ok((Some(state), report))
}

fn export(live: &LiveState) -> Result<PortableDiscoveryState, SelectiveDiscoveryError> {
    PortableDiscoveryState::from_live(
        &live.inputs,
        &live.data,
        live.contract_identity,
        &live.store,
        &live.outcome,
        live.work,
    )
}

fn execute_static(
    contract: &StaticDiscoveryContract,
    data: &TabularData,
    ci: Arc<SelectiveCi>,
    ctx: &ExecutionContext,
) -> Result<DiscoveryOutcome, DiscoveryError> {
    let mut workspace = DiscoveryWorkspace::default();
    match contract.algorithm {
        StaticAlgorithm::Pc => {
            let pc = Pc::new()
                .with_constraints(contract.constraints.clone())
                .with_fdr_adjustment(contract.fdr)
                .with_ci(ci);
            pc.run(data, &contract.variables, &mut workspace, ctx)
                .map(|r| DiscoveryOutcome::from_static_cpdag(&r))
        }
        StaticAlgorithm::Fci => {
            let mut fci = Fci::new()
                .with_constraints(contract.constraints.clone())
                .with_fdr_adjustment(contract.fdr)
                .with_ci(ci);
            fci.pds_max_nodes = contract.possible_d_sep_max_nodes;
            fci.discriminating_path_budget = contract.discriminating_path_budget;
            fci.run(data, &contract.variables, &mut workspace, ctx)
                .map(|r| DiscoveryOutcome::from_static_pag(&r))
        }
    }
}

fn execute_temporal(
    contract: &TemporalDiscoveryContract,
    data: &TimeSeriesData,
    ci: Arc<SelectiveCi>,
    ctx: &ExecutionContext,
) -> Result<DiscoveryOutcome, DiscoveryError> {
    let mut workspace = DiscoveryWorkspace::default();
    match contract.algorithm {
        TemporalAlgorithm::Pcmci => {
            let alg = Pcmci::new()
                .with_constraints(contract.constraints.clone())
                .with_fdr_adjustment(contract.fdr)
                .with_ci(ci);
            alg.run(data, &contract.variables, &mut workspace, ctx)
                .map(|r| DiscoveryOutcome::from_temporal_dag(&r))
        }
        TemporalAlgorithm::PcmciPlus => {
            let alg = PcmciPlus::new()
                .with_constraints(contract.constraints.clone())
                .with_fdr_adjustment(contract.fdr)
                .with_ci(ci);
            alg.run(data, &contract.variables, &mut workspace, ctx)
                .map(|r| DiscoveryOutcome::from_temporal_cpdag(&r))
        }
    }
}

fn prepare_static(
    contract: &StaticDiscoveryContract,
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Result<Prepared, SelectiveDiscoveryError> {
    contract.validate()?;
    ensure_seed(contract.rng_seed, ctx)?;
    let identity = static_data_identity(data, &contract.variables)?;
    let inputs = contract.stage_inputs(&identity, ctx.kernel_policy)?;
    let (test, ci_identity) = contract.ci_test.resolve()?;
    let contract_identity = inputs_contract_identity(&inputs);
    Ok(Prepared { inputs, data: identity, contract_identity, test, ci_identity })
}

fn prepare_temporal(
    contract: &TemporalDiscoveryContract,
    data: &TimeSeriesData,
    ctx: &ExecutionContext,
) -> Result<Prepared, SelectiveDiscoveryError> {
    contract.validate()?;
    ensure_seed(contract.rng_seed, ctx)?;
    let identity = temporal_data_identity(data, &contract.variables)?;
    let inputs = contract.stage_inputs(&identity, ctx.kernel_policy)?;
    let (test, ci_identity) = contract.ci_test.resolve()?;
    let contract_identity = inputs_contract_identity(&inputs);
    Ok(Prepared { inputs, data: identity, contract_identity, test, ci_identity })
}

/// Verify a portable state against the inputs of a request and replay it.
fn resume_candidate(
    prepared: Prepared,
    state: &PortableDiscoveryState,
    execute: impl FnOnce(Arc<SelectiveCi>) -> Result<DiscoveryOutcome, DiscoveryError>,
) -> Result<(LiveState, SelectiveRunReport), SelectiveDiscoveryError> {
    state.verify_binding(&prepared.inputs, &prepared.data)?;
    let replay = Arc::new(StatisticStore::replay_only(&state.receipts)?);
    let ci = Arc::new(SelectiveCi::new(
        Arc::clone(&prepared.test),
        Arc::clone(&replay),
        prepared.ci_identity.as_str(),
    ));
    let before = replay.counters();
    let (executed, orientation) = observe_orientation(|| execute(ci));
    let outcome = executed.map_err(|error| map_failure(&replay, error))?;
    state.verify_replayed(&outcome)?;
    let work = work_between(before, &replay, &outcome, orientation);
    let live_store = Arc::new(StatisticStore::seeded(&state.receipts)?);
    let plan = DiscoveryPlan::between(Some(&prepared.inputs), &prepared.inputs);
    let report = SelectiveRunReport {
        plan,
        work,
        outcome_changed: false,
        retained_statistics: live_store.len(),
    };
    let live = LiveState {
        inputs: prepared.inputs,
        data: prepared.data,
        contract_identity: prepared.contract_identity,
        store: live_store,
        outcome,
        work,
    };
    Ok((live, report))
}

/// The static selective-discovery adapter (PC and FCI over a table).
#[derive(Default)]
pub struct StaticSelectiveDiscovery {
    live: Option<LiveState>,
}

impl std::fmt::Debug for StaticSelectiveDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticSelectiveDiscovery").field("has_state", &self.live.is_some()).finish()
    }
}

impl StaticSelectiveDiscovery {
    /// An adapter with no retained state.
    #[must_use]
    pub const fn new() -> Self {
        Self { live: None }
    }

    /// What `run` would reuse for this request, without executing anything.
    ///
    /// # Errors
    ///
    /// A contract or data the run would refuse.
    pub fn plan(
        &self,
        contract: &StaticDiscoveryContract,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryPlan, SelectiveDiscoveryError> {
        let prepared = prepare_static(contract, data, ctx)?;
        Ok(DiscoveryPlan::between(self.live.as_ref().map(|l| &l.inputs), &prepared.inputs))
    }

    /// Discover on `data` under `contract`, reusing statistics where the plan allows.
    ///
    /// A refusal, an algorithm failure or a cancellation leaves the previously retained
    /// state, outcome and identities exactly as they were.
    ///
    /// # Errors
    ///
    /// A refused contract, a context whose seed is not the contract's, or an algorithm
    /// failure (data, test, graph, resource or cancellation).
    pub fn run(
        &mut self,
        contract: &StaticDiscoveryContract,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<SelectiveRunReport, SelectiveDiscoveryError> {
        let prepared = prepare_static(contract, data, ctx)?;
        let (state, report) = execute_candidate(self.live.as_ref(), prepared, |ci| {
            execute_static(contract, data, ci, ctx)
        })?;
        if let Some(state) = state {
            self.live = Some(state);
        }
        Ok(report)
    }

    /// The retained outcome.
    #[must_use]
    pub fn outcome(&self) -> Option<&DiscoveryOutcome> {
        self.live.as_ref().map(|l| &l.outcome)
    }

    /// The retained stage inputs.
    #[must_use]
    pub fn inputs(&self) -> Option<&StageInputs> {
        self.live.as_ref().map(|l| &l.inputs)
    }

    /// The retained data identity.
    #[must_use]
    pub fn data_identity(&self) -> Option<&DataIdentity> {
        self.live.as_ref().map(|l| &l.data)
    }

    /// Statistics currently retained.
    #[must_use]
    pub fn retained_statistics(&self) -> usize {
        self.live.as_ref().map_or(0, |l| l.store.len())
    }

    /// Export the retained run as a portable state: configuration and stage identities,
    /// statistic receipts and the graph and result identities. The state carries no raw
    /// data.
    ///
    /// # Errors
    ///
    /// Nothing retained, or more statistics than a portable state carries.
    pub fn export_state(&self) -> Result<PortableDiscoveryState, SelectiveDiscoveryError> {
        let live = self.live.as_ref().ok_or_else(|| {
            invalid(
                "selective_discovery.nothing_retained",
                "run discovery before exporting its state",
            )
        })?;
        export(live)
    }

    /// Resume from a portable state: replay the algorithm over the retained statistics only.
    ///
    /// Resuming needs the raw data. Receipts alone cannot be validated, because a statistic
    /// is bound to the content it was computed from, and absent data is refused rather than
    /// trusted. Nothing is computed: a statistic the state does not hold is a refusal, and
    /// the replayed result must have the identity the state recorded.
    ///
    /// # Errors
    ///
    /// Missing raw data, data or configuration that differ from the state's, an invalid or
    /// incomplete receipt set, or a replay that does not reproduce the recorded result.
    pub fn resume(
        contract: &StaticDiscoveryContract,
        data: Option<&TabularData>,
        state: &PortableDiscoveryState,
        ctx: &ExecutionContext,
    ) -> Result<(Self, SelectiveRunReport), SelectiveDiscoveryError> {
        let data = data.ok_or_else(missing_raw_data)?;
        let prepared = prepare_static(contract, data, ctx)?;
        let (live, report) =
            resume_candidate(prepared, state, |ci| execute_static(contract, data, ci, ctx))?;
        Ok((Self { live: Some(live) }, report))
    }
}

fn missing_raw_data() -> SelectiveDiscoveryError {
    statistics_invalid(
        "selective_discovery.raw_data_missing",
        "resume needs the raw data the statistics were computed from; receipts and graph \
         identities remain inspectable without it, but they cannot be validated or extended",
    )
}

/// The temporal selective-discovery adapter (PCMCI and PCMCI+ over a time series).
#[derive(Default)]
pub struct TemporalSelectiveDiscovery {
    live: Option<LiveState>,
}

impl std::fmt::Debug for TemporalSelectiveDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemporalSelectiveDiscovery")
            .field("has_state", &self.live.is_some())
            .finish()
    }
}

impl TemporalSelectiveDiscovery {
    /// An adapter with no retained state.
    #[must_use]
    pub const fn new() -> Self {
        Self { live: None }
    }

    /// What `run` would reuse for this request, without executing anything.
    ///
    /// # Errors
    ///
    /// A contract or data the run would refuse.
    pub fn plan(
        &self,
        contract: &TemporalDiscoveryContract,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryPlan, SelectiveDiscoveryError> {
        let prepared = prepare_temporal(contract, data, ctx)?;
        Ok(DiscoveryPlan::between(self.live.as_ref().map(|l| &l.inputs), &prepared.inputs))
    }

    /// Discover on `data` under `contract`, reusing statistics where the plan allows.
    ///
    /// A refusal, an algorithm failure or a cancellation leaves the previously retained
    /// state, outcome and identities exactly as they were.
    ///
    /// # Errors
    ///
    /// A refused contract, a context whose seed is not the contract's, or an algorithm
    /// failure (data, test, graph, resource or cancellation).
    pub fn run(
        &mut self,
        contract: &TemporalDiscoveryContract,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<SelectiveRunReport, SelectiveDiscoveryError> {
        let prepared = prepare_temporal(contract, data, ctx)?;
        let (state, report) = execute_candidate(self.live.as_ref(), prepared, |ci| {
            execute_temporal(contract, data, ci, ctx)
        })?;
        if let Some(state) = state {
            self.live = Some(state);
        }
        Ok(report)
    }

    /// The retained outcome.
    #[must_use]
    pub fn outcome(&self) -> Option<&DiscoveryOutcome> {
        self.live.as_ref().map(|l| &l.outcome)
    }

    /// The retained stage inputs.
    #[must_use]
    pub fn inputs(&self) -> Option<&StageInputs> {
        self.live.as_ref().map(|l| &l.inputs)
    }

    /// The retained data identity.
    #[must_use]
    pub fn data_identity(&self) -> Option<&DataIdentity> {
        self.live.as_ref().map(|l| &l.data)
    }

    /// Statistics currently retained.
    #[must_use]
    pub fn retained_statistics(&self) -> usize {
        self.live.as_ref().map_or(0, |l| l.store.len())
    }

    /// Export the retained run as a portable state. The state carries no raw data.
    ///
    /// # Errors
    ///
    /// Nothing retained, or more statistics than a portable state carries.
    pub fn export_state(&self) -> Result<PortableDiscoveryState, SelectiveDiscoveryError> {
        let live = self.live.as_ref().ok_or_else(|| {
            invalid(
                "selective_discovery.nothing_retained",
                "run discovery before exporting its state",
            )
        })?;
        export(live)
    }

    /// Resume from a portable state: replay the algorithm over the retained statistics only.
    ///
    /// # Errors
    ///
    /// As [`StaticSelectiveDiscovery::resume`].
    pub fn resume(
        contract: &TemporalDiscoveryContract,
        data: Option<&TimeSeriesData>,
        state: &PortableDiscoveryState,
        ctx: &ExecutionContext,
    ) -> Result<(Self, SelectiveRunReport), SelectiveDiscoveryError> {
        let data = data.ok_or_else(missing_raw_data)?;
        let prepared = prepare_temporal(contract, data, ctx)?;
        let (live, report) =
            resume_candidate(prepared, state, |ci| execute_temporal(contract, data, ci, ctx))?;
        Ok((Self { live: Some(live) }, report))
    }
}
