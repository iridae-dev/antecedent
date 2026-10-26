//! Instrumental-variable identification over a finite unfolding of a stationary temporal DAG.
//!
//! A lagged variable is a valid instrument for a single-step temporal effect exactly when the
//! static graphical conditions hold on the unfolded window graph: the instrument is not a
//! descendant of the treatment, is not d-separated from it (relevance), and is d-separated from
//! the outcome once the treatment's outgoing edges are cut (no directed path except through the
//! treatment, and no unblocked back-door path). The graph supplies exclusion and exogeneity; a
//! time column supplies neither. Relevance is a connection in the graph, not a strength: the
//! first-stage F statistic of the data checks it at estimation.
//!
//! Each candidate is examined in a window grown until the ancestry of the treatment, the outcome
//! and that candidate no longer touches the truncated boundary, so a confounder older than the
//! window cannot hide. A candidate whose ancestry is unbounded (an autoregressive chain into it)
//! is not certified and is not reported valid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery, Intervention, TemporalEffectQuery,
    TemporalPolicy, VariableId,
};
use antecedent_data::{TemporalIndexer, TemporalNodeKey};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{DenseNodeId, TemporalDag, UnfoldedTemporalGraph};

use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
use crate::iv::{
    InstrumentSearchConfig, instrument_assumptions, is_valid_instrument, wald_effect_restriction,
};
use crate::result::{
    DerivationTrace, EstimandClaim, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};
use crate::temporal_backdoor::{required_variable_count, retarget, template_max_lag};
use crate::temporal_window::{GrownWindow, grow_window};

/// Estimand method tag of an instrument identified on the unfolded window.
pub const TEMPORAL_IV_METHOD: &str = "temporal.iv.unfolded";

/// Instrumental-variable identifier for single-step temporal effects.
#[derive(Clone, Debug, Default)]
pub struct TemporalIvIdentifier {
    /// Search configuration.
    pub config: InstrumentSearchConfig,
}

/// Temporal instrument identification with the finite-unfolding context needed to read dense ids
/// as `(variable, offset)` pairs.
#[derive(Clone, Debug)]
pub struct TemporalIvResult {
    /// Identification over the unfolded static DAG. Its ids are dense unfolded node ids.
    pub result: IdentificationResult,
    /// Indexer of the finite unfolding.
    pub indexer: TemporalIndexer,
    /// The unfolded DAG [`Self::indexer`] addresses.
    pub unfolded: UnfoldedTemporalGraph,
    /// Temporal key of the treatment node.
    pub treatment_key: TemporalNodeKey,
    /// Temporal key of the outcome node.
    pub outcome_key: TemporalNodeKey,
    /// Valid instruments, in estimand order.
    pub instruments: Vec<TemporalNodeKey>,
    /// Candidates whose ancestry no finite window certified, so none was reported valid.
    pub uncertified: Vec<TemporalNodeKey>,
}

/// The finite window and keys one query is identified in.
struct Window {
    treatment_key: TemporalNodeKey,
    outcome_key: TemporalNodeKey,
    variable_count: u32,
    span_offset: i32,
    horizon: u32,
    base_history: u32,
    history_cap: u32,
    chain_cap: u32,
}

impl Window {
    fn of(
        template: &TemporalDag,
        query: &TemporalEffectQuery,
    ) -> Result<Self, IdentificationError> {
        query.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid temporal-effect query",
        })?;
        if query.is_multi_step_sustained()
            || !matches!(
                query.policy,
                TemporalPolicy::Pulse { .. } | TemporalPolicy::Sustained { .. }
            )
        {
            return Err(IdentificationError::UnsupportedQuery {
                message: "temporal IV identifies a pulse or one-step sustained effect only",
            });
        }
        let treatment_at =
            query.try_treatment_offset().map_err(|_| IdentificationError::UnsupportedQuery {
                message: "temporal IV needs one treatment offset",
            })?;
        let outcome_at = query.outcome_offset();
        let variable_count = required_variable_count(template, [query.treatment, query.outcome]);
        let max_lag = template_max_lag(template);
        let span = max_lag.max(1);
        let span_offset = i32::try_from(span)
            .map_err(|_| IdentificationError::out_of_range("instrument window span exceeds i32"))?;
        let earliest = treatment_at.min(outcome_at).min(0).saturating_sub(span_offset);
        let horizon = u32::try_from(treatment_at.max(outcome_at).max(0))
            .map_err(|_| IdentificationError::out_of_range("negative horizon"))?
            .saturating_add(1);
        let base_history = earliest.unsigned_abs().max(max_lag);
        let chain_cap =
            variable_count.saturating_mul(max_lag).saturating_add(earliest.unsigned_abs());
        Ok(Self {
            treatment_key: TemporalNodeKey { variable: query.treatment, offset: treatment_at },
            outcome_key: TemporalNodeKey { variable: query.outcome, offset: outcome_at },
            variable_count,
            span_offset,
            horizon,
            base_history,
            history_cap: query.max_history_lag.unwrap_or(chain_cap).max(base_history),
            chain_cap,
        })
    }

    fn grow(
        &self,
        template: &TemporalDag,
        seeds: &[TemporalNodeKey],
    ) -> Result<GrownWindow, IdentificationError> {
        grow_window(
            template,
            seeds,
            self.variable_count,
            self.horizon,
            self.base_history,
            self.history_cap,
            self.chain_cap,
        )
    }

    /// Every lagged candidate: each variable at each offset before the treatment.
    fn candidates(&self) -> Vec<TemporalNodeKey> {
        let mut keys = Vec::new();
        for variable in 0..self.variable_count {
            for back in 1..=self.span_offset {
                let key = TemporalNodeKey {
                    variable: VariableId::from_raw(variable),
                    offset: self.treatment_key.offset - back,
                };
                if key != self.outcome_key {
                    keys.push(key);
                }
            }
        }
        keys
    }
}

/// What the candidate search found.
struct Search {
    valid: Vec<TemporalNodeKey>,
    uncertified: Vec<TemporalNodeKey>,
    examined: u64,
}

impl TemporalIvIdentifier {
    /// Create with the default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Identify `query` by every lagged variable that is a valid instrument on the unfolded
    /// window graph.
    ///
    /// A query with no valid instrument is an `Ok` result with status
    /// [`IdentificationStatus::NotIdentified`] whose derivation names the candidates examined, so
    /// a caller can state the premise that failed and the next attempt.
    ///
    /// # Errors
    /// An invalid or unsupported query (a multi-step schedule, a non-`Set` intervention), a window
    /// whose treatment or outcome ancestry no finite history certifies, an unfolding failure, or
    /// an invariant violation.
    pub fn identify_temporal_iv(
        &self,
        template: &TemporalDag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalIvResult, IdentificationError> {
        let window = Window::of(template, query)?;
        let search = self.search(template, &window)?;
        let mut seeds = vec![window.treatment_key, window.outcome_key];
        seeds.extend(search.valid.iter().copied());
        let grown = window.grow(template, &seeds)?;
        assemble(query, &window, grown, search)
    }

    /// Examine each candidate in a window that certifies its own ancestry.
    fn search(
        &self,
        template: &TemporalDag,
        window: &Window,
    ) -> Result<Search, IdentificationError> {
        let mut workspace = IdentificationWorkspace::default();
        let mut search = Search { valid: Vec::new(), uncertified: Vec::new(), examined: 0 };
        for candidate in window.candidates() {
            search.examined += 1;
            let seeds = [window.treatment_key, window.outcome_key, candidate];
            let grown = match window.grow(template, &seeds) {
                Ok(grown) => grown,
                Err(IdentificationError::NotCertified { .. }) => {
                    search.uncertified.push(candidate);
                    continue;
                }
                Err(error) => return Err(error),
            };
            let &[t, y, z] = grown.dense.as_slice() else {
                return Err(seed_count());
            };
            let instrument = is_valid_instrument(
                &grown.unfolded.dag,
                DenseNodeId::from_raw(z),
                DenseNodeId::from_raw(t),
                DenseNodeId::from_raw(y),
                &mut workspace.graph,
                &mut workspace.dsep,
            )?;
            if instrument {
                search.valid.push(candidate);
                if search.valid.len() >= self.config.max_results {
                    break;
                }
            }
        }
        Ok(search)
    }
}

fn seed_count() -> IdentificationError {
    IdentificationError::InvariantViolated {
        message: "temporal window growth returned a different seed count",
    }
}

/// The identification of the final window: not identified without a valid instrument, otherwise
/// one Wald estimand per instrument.
fn assemble(
    query: &TemporalEffectQuery,
    window: &Window,
    grown: GrownWindow,
    search: Search,
) -> Result<TemporalIvResult, IdentificationError> {
    let GrownWindow { history, unfolded, dense } = grown;
    let (Some(&treatment_dense), Some(&outcome_dense)) = (dense.first(), dense.get(1)) else {
        return Err(seed_count());
    };
    let treatment_var = VariableId::from_raw(treatment_dense);
    let outcome_var = VariableId::from_raw(outcome_dense);
    let ate = AverageEffectQuery::new(
        treatment_var,
        outcome_var,
        Arc::from([]),
        retarget(&query.control, treatment_var)?,
        retarget(&query.active, treatment_var)?,
        query.target_population.clone(),
    );
    let performance = IdentificationPerformanceRecord {
        candidates_examined: search.examined,
        sets_returned: u64::try_from(search.valid.len()).unwrap_or(u64::MAX),
    };
    let mut assumptions = AssumptionSet::new();
    assumptions.push(crate::assumptions::causal_markov("temporal.iv"));
    assumptions.push(stationarity());
    let mut derivation = derivation_header(query, window, history, &search);
    let query_wire = CausalQuery::average_effect(ate.clone());
    let result = if search.valid.is_empty() {
        derivation.push(
            "temporal.iv.none",
            "no lagged variable is a valid instrument: each candidate is a descendant of the \
             treatment, is independent of it, reaches the outcome other than through the \
             treatment, or shares an unblocked back-door path with it",
        );
        IdentificationResult::not_identified(query_wire, derivation, assumptions, performance)
    } else {
        let instruments = InstrumentPlan { ate: &ate, treatment_var, outcome_var, dense: &dense };
        identified(query_wire, &instruments, &search.valid, derivation, assumptions, performance)?
    };
    Ok(TemporalIvResult {
        result,
        indexer: unfolded.indexer.clone(),
        unfolded,
        treatment_key: window.treatment_key,
        outcome_key: window.outcome_key,
        instruments: search.valid,
        uncertified: search.uncertified,
    })
}

fn derivation_header(
    query: &TemporalEffectQuery,
    window: &Window,
    history: u32,
    search: &Search,
) -> DerivationTrace {
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "temporal.iv.criterion",
        "a lagged Z is not a descendant of T, is d-connected to T (relevance), and is \
         d-separated from Y in the unfolded window graph with T's out-edges cut (exclusion and \
         exogeneity); relevance strength is checked by the first-stage F at estimation",
    );
    derivation.push(
        "temporal.unfold",
        format!(
            "finite window history={history} horizon={}; treatment={}@{} outcome={}@{}; {} lagged \
             candidate(s) examined, {} uncertified",
            window.horizon,
            query.treatment,
            window.treatment_key.offset,
            query.outcome,
            window.outcome_key.offset,
            search.examined,
            search.uncertified.len()
        ),
    );
    derivation
}

/// The unfolded identifiers the Wald estimands are written over.
struct InstrumentPlan<'a> {
    ate: &'a AverageEffectQuery,
    treatment_var: VariableId,
    outcome_var: VariableId,
    dense: &'a [u32],
}

fn identified(
    query: CausalQuery,
    plan: &InstrumentPlan<'_>,
    valid: &[TemporalNodeKey],
    mut derivation: DerivationTrace,
    mut assumptions: AssumptionSet,
    performance: IdentificationPerformanceRecord,
) -> Result<IdentificationResult, IdentificationError> {
    assumptions.push(wald_effect_restriction());
    let shared = assumptions.clone();
    let (active, control) = match (&plan.ate.active, &plan.ate.control) {
        (Intervention::Set { value: active, .. }, Intervention::Set { value: control, .. }) => {
            (active.clone(), control.clone())
        }
        _ => {
            return Err(IdentificationError::UnsupportedQuery {
                message: "temporal IV requires Set interventions",
            });
        }
    };
    let mut arena = CausalExprArena::new();
    let mut estimands = Vec::with_capacity(valid.len());
    let mut claims = Vec::with_capacity(valid.len());
    for (index, key) in valid.iter().enumerate() {
        let Some(&dense_z) = plan.dense.get(index + 2) else {
            return Err(seed_count());
        };
        let z_var = VariableId::from_raw(dense_z);
        let own = instrument_assumptions(z_var);
        let mut claim_assumptions = shared.clone();
        claim_assumptions.extend_unique(&own);
        assumptions.extend_unique(&own);
        claims.push(EstimandClaim {
            status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
            required_assumptions: claim_assumptions,
        });
        let functional = arena
            .iv_wald(plan.treatment_var, plan.outcome_var, &[z_var], &active, &control)
            .map_err(|_| IdentificationError::UnsupportedQuery {
                message: "IV requires a single instrument distinct from the treatment",
            })?;
        estimands.push(IdentifiedEstimand::instrumental(
            TEMPORAL_IV_METHOD,
            Arc::from([z_var]),
            functional,
        ));
        derivation
            .push("temporal.iv.instrument", format!("Z={}@{}", key.variable.raw(), key.offset));
    }
    Ok(IdentificationResult::identified_under_parametric_restrictions(
        query,
        estimands,
        arena,
        derivation,
        assumptions,
        performance,
    )
    .with_estimand_claims(claims))
}

fn stationarity() -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::Stationarity,
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("temporal.iv.unfolded") },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}
