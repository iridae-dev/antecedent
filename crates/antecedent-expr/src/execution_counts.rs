//! Cumulative instrumentation of executing finite factor and functional engines.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::Cell;
/// Actual successful construction/compilation/evaluation work; provider calls include hits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StaticWorkCounts {
    /// Empirical CPT row scans completed.
    pub factor_builds: u64,
    /// Expression programs compiled.
    pub program_compilations: u64,
    /// Providers constructed or rebound.
    pub provider_bindings: u64,
    /// Uncached factor values evaluated.
    pub factor_evaluations: u64,
    /// Full functionals or finite distribution means integrated.
    pub integrations: u64,
    /// Calls to numerical factor providers, including cache hits.
    pub provider_calls: u64,
    /// Checked projection of an already evaluated finite law.
    pub law_summaries: u64,
}
thread_local! {static COUNTS:Cell<StaticWorkCounts>=const{Cell::new(StaticWorkCounts{factor_builds:0,program_compilations:0,provider_bindings:0,factor_evaluations:0,integrations:0,provider_calls:0,law_summaries:0})};}
/// Observe executing work including nested scopes. These finite engines execute serially.
pub fn count_static_work<R>(work: impl FnOnce() -> R) -> (R, StaticWorkCounts) {
    let before = COUNTS.with(Cell::get);
    let result = work();
    let after = COUNTS.with(Cell::get);
    (
        result,
        StaticWorkCounts {
            factor_builds: after.factor_builds.saturating_sub(before.factor_builds),
            program_compilations: after
                .program_compilations
                .saturating_sub(before.program_compilations),
            provider_bindings: after.provider_bindings.saturating_sub(before.provider_bindings),
            factor_evaluations: after.factor_evaluations.saturating_sub(before.factor_evaluations),
            integrations: after.integrations.saturating_sub(before.integrations),
            provider_calls: after.provider_calls.saturating_sub(before.provider_calls),
            law_summaries: after.law_summaries.saturating_sub(before.law_summaries),
        },
    )
}
/// Executed engine operation kinds; adapters never increment these counters themselves.
#[derive(Clone, Copy)]
pub enum StaticWork {
    /// Empirical factor scan.
    FactorBuild,
    /// Program compilation.
    ProgramCompilation,
    /// Provider binding.
    ProviderBinding,
    /// Uncached factor evaluation.
    FactorEvaluation,
    /// Functional integration.
    Integration,
    /// Factor provider invocation.
    ProviderCall,
    /// Retained law projection.
    LawSummary,
}
/// Instrument an executed component operation.
pub fn note_static_work(work: StaticWork) {
    COUNTS.with(|counts| {
        let mut counts_value = counts.get();
        let value = match work {
            StaticWork::FactorBuild => &mut counts_value.factor_builds,
            StaticWork::ProgramCompilation => &mut counts_value.program_compilations,
            StaticWork::ProviderBinding => &mut counts_value.provider_bindings,
            StaticWork::FactorEvaluation => &mut counts_value.factor_evaluations,
            StaticWork::Integration => &mut counts_value.integrations,
            StaticWork::ProviderCall => &mut counts_value.provider_calls,
            StaticWork::LawSummary => &mut counts_value.law_summaries,
        };
        *value = value.saturating_add(1);
        counts.set(counts_value);
    });
}
