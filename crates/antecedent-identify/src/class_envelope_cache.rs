//! Process-wide memo of temporal class identification envelopes.
//!
//! A temporal CPDAG or PAG envelope (completion enumeration, equivalence audit and
//! per-completion identification) is a pure function of the graph, the query and the
//! identifier settings: the completion search runs under a fixed operation budget with no
//! clock, seed or cancellation. Repeated studies over the same graph, such as every
//! replicate of a calibration design or every refit of a prepared study on new data,
//! therefore share one result.
//!
//! The memo is exact (keys compare the graph, query and settings structurally, so a miss is
//! the only way a difference shows), bounded in entry count and in completions per entry,
//! and holds successes only. A hit returns a clone of the stored envelope, identical to a
//! recomputation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use antecedent_core::TemporalEffectQuery;
use antecedent_graph::{TemporalCpdag, TemporalPag};

use crate::generalized::GeneralizedAdjustmentConfig;
use crate::temporal_generalized::TemporalClassEnvelope;

/// Entries kept; the least recently used is evicted first.
const CAPACITY: usize = 16;
/// Envelopes with more completions than this are recomputed rather than retained.
const MAX_RETAINED_COMPLETIONS: usize = 256;

#[derive(PartialEq)]
enum SourceGraph {
    Cpdag(TemporalCpdag),
    Pag(TemporalPag),
}

#[derive(PartialEq)]
struct Key {
    graph: SourceGraph,
    query: TemporalEffectQuery,
    max_completions: usize,
    weight_bits: u64,
    max_candidates: usize,
    max_examinations: u64,
    completion_operations: usize,
}

impl Key {
    fn new(
        graph: SourceGraph,
        query: &TemporalEffectQuery,
        config: &GeneralizedAdjustmentConfig,
    ) -> Self {
        Self {
            graph,
            query: query.clone(),
            max_completions: config.max_completions,
            weight_bits: config.per_completion_weight.to_bits(),
            max_candidates: config.max_candidates,
            max_examinations: config.max_examinations,
            completion_operations: config.completion_operations,
        }
    }
}

type Entries = VecDeque<(Key, Arc<TemporalClassEnvelope>)>;

fn entries() -> MutexGuard<'static, Entries> {
    static ENTRIES: OnceLock<Mutex<Entries>> = OnceLock::new();
    ENTRIES
        .get_or_init(|| Mutex::new(VecDeque::with_capacity(CAPACITY)))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn memoized<E>(
    key: Key,
    compute: impl FnOnce() -> Result<TemporalClassEnvelope, E>,
) -> Result<TemporalClassEnvelope, E> {
    {
        let mut held = entries();
        if let Some(at) = held.iter().position(|(stored, _)| *stored == key) {
            if let Some(hit) = held.remove(at) {
                let envelope = hit.1.as_ref().clone();
                held.push_front(hit);
                return Ok(envelope);
            }
        }
    }
    // The lock is not held across the enumeration: concurrent first calls both compute the
    // same value and the later insert replaces nothing observable.
    let envelope = compute()?;
    #[cfg(test)]
    COMPUTATIONS.with(|count| count.set(count.get() + 1));
    if envelope.envelope.cases.len() <= MAX_RETAINED_COMPLETIONS {
        let mut held = entries();
        if !held.iter().any(|(stored, _)| *stored == key) {
            held.push_front((key, Arc::new(envelope.clone())));
            held.truncate(CAPACITY);
        }
    }
    Ok(envelope)
}

/// Memoize a temporal CPDAG envelope computation.
pub(crate) fn memoized_cpdag<E>(
    cpdag: &TemporalCpdag,
    query: &TemporalEffectQuery,
    config: &GeneralizedAdjustmentConfig,
    compute: impl FnOnce() -> Result<TemporalClassEnvelope, E>,
) -> Result<TemporalClassEnvelope, E> {
    if config.context.is_some() {
        // An envelope computed under one caller's pool and cancellation is never another's.
        return compute();
    }
    memoized(Key::new(SourceGraph::Cpdag(cpdag.clone()), query, config), compute)
}

/// Memoize a temporal PAG envelope computation.
pub(crate) fn memoized_pag<E>(
    pag: &TemporalPag,
    query: &TemporalEffectQuery,
    config: &GeneralizedAdjustmentConfig,
    compute: impl FnOnce() -> Result<TemporalClassEnvelope, E>,
) -> Result<TemporalClassEnvelope, E> {
    if config.context.is_some() {
        return compute();
    }
    memoized(Key::new(SourceGraph::Pag(pag.clone()), query, config), compute)
}

#[cfg(test)]
thread_local! {
    /// Envelope computations that finished on this thread (cache misses).
    static COMPUTATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Lag, TemporalPolicy, VariableId};

    use super::*;
    use crate::generalized::GeneralizedAdjustmentIdentifier;

    fn chain_pag() -> TemporalPag {
        let mut pag = TemporalPag::empty();
        let t = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = pag.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        pag.insert_circle_arrow(t, y).unwrap();
        pag
    }

    fn pulse() -> TemporalEffectQuery {
        let mut query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        query.policy = TemporalPolicy::pulse(-1);
        query
    }

    fn computations() -> usize {
        COMPUTATIONS.with(std::cell::Cell::get)
    }

    #[test]
    fn a_repeated_pag_envelope_is_reused_and_equals_the_recomputation() {
        let pag = chain_pag();
        let query = pulse();
        // A distinct budget keeps this key private to the test within the shared memo.
        let mut config = GeneralizedAdjustmentConfig::default();
        config.completion_operations -= 17;
        let id = GeneralizedAdjustmentIdentifier { config };

        let before = computations();
        let first = id.identify_temporal_pag_envelope(&pag, &query).unwrap();
        assert_eq!(computations(), before + 1);
        let second = id.identify_temporal_pag_envelope(&pag, &query).unwrap();
        assert_eq!(computations(), before + 1, "the second call re-ran the enumeration");

        let uncached = id.compute_temporal_pag_envelope(&pag, &query).unwrap();
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        assert_eq!(format!("{first:?}"), format!("{uncached:?}"));
    }

    #[test]
    fn a_different_budget_or_query_is_a_different_entry() {
        let pag = chain_pag();
        let mut config = GeneralizedAdjustmentConfig::default();
        config.completion_operations -= 29;
        let id = GeneralizedAdjustmentIdentifier { config };
        let before = computations();
        id.identify_temporal_pag_envelope(&pag, &pulse()).unwrap();
        let mut other = pulse();
        other.horizon_steps += 1;
        id.identify_temporal_pag_envelope(&pag, &other).unwrap();
        assert_eq!(computations(), before + 2);
    }
    #[test]
    fn a_warm_envelope_cannot_bypass_the_callers_cancellation_memory_or_cache_policy() {
        let pag = chain_pag();
        let query = pulse();
        let mut config = GeneralizedAdjustmentConfig::default();
        config.completion_operations -= 83;
        let warm = GeneralizedAdjustmentIdentifier { config: config.clone() }
            .identify_temporal_pag_envelope(&pag, &query)
            .unwrap();
        let before = computations();

        let cancelled = antecedent_core::ExecutionContext::for_tests(83);
        cancelled.cancellation.cancel();
        config.context = Some(cancelled);
        assert!(matches!(
            GeneralizedAdjustmentIdentifier { config: config.clone() }
                .identify_temporal_pag_envelope(&pag, &query),
            Err(crate::IdentificationError::Cancelled)
        ));
        config.context =
            Some(antecedent_core::ExecutionContext::for_tests(83).with_memory_limit(Some(1)));
        assert!(matches!(
            GeneralizedAdjustmentIdentifier { config: config.clone() }
                .identify_temporal_pag_envelope(&pag, &query),
            Err(crate::IdentificationError::Budget { budget: crate::IdentificationBudget::Memory })
        ));
        let mut disabled = antecedent_core::ExecutionContext::for_tests(83);
        disabled.cache_policy = antecedent_core::CachePolicy::disabled();
        config.context = Some(disabled);
        let recomputed = GeneralizedAdjustmentIdentifier { config }
            .identify_temporal_pag_envelope(&pag, &query)
            .unwrap();
        assert_eq!(format!("{warm:?}"), format!("{recomputed:?}"));
        assert_eq!(computations(), before, "caller-context attempts never enter the process memo");
    }
}
