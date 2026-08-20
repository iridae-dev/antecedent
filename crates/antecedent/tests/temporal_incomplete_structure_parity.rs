//! End-to-end probe for the `TemporalCpdag`/`TemporalPag` × `accepted` support cell.
//!
//! `effective_graph_class` collapses an accepted temporal CPDAG/PAG to
//! `TemporalDag` whenever `try_into_temporal_dag()` succeeds, so this cell is
//! reached exactly when completion *fails*. Both `PulseEffect` and
//! `SustainedEffect` reach it through the same policy-generic
//! `(AnalysisRoute::TemporalEffect, GraphClass::TemporalCpdag/TemporalPag)`
//! arms in `analysis/execute/compile.rs`, so the two must behave identically:
//! `Study::build()` admits the cell and `run()` surfaces the specific
//! `CausalError::Compile` naming the completion failure, never the generic
//! "unlicensed and not allowed" support refusal and never a silent number.
//!
//! This file is the evidence behind the `PulseEffect` and `SustainedEffect`
//! rows for this coordinate in `parity/support_allowlist.toml`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, CausalError, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{MarkedEdge, TemporalCpdag, TemporalCpdagReview, TemporalPag};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);

/// The same two-column `t`/`y` series the licensed `TemporalDag` Pulse/Sustained
/// cells are pinned against (`conformance/response/temporal_dose_horizon`),
/// generated inline so this probe does not depend on the fixture's contract.
fn series() -> TimeSeriesData {
    let n = 128usize;
    let t: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            _ => -1.0,
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();

    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(T, Arc::from(t), ValidityBitmap::all_valid(n)).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(Y, Arc::from(y), ValidityBitmap::all_valid(n)).unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

/// A temporal PAG the accept gate admits (no circle marks) but that cannot
/// complete: the bidirected `t-1 <-> y` mark has no `parent_child()` reduction.
fn incompletable_temporal_pag() -> AcceptedGraph {
    let mut pag = TemporalPag::empty();
    let t1 = pag.add_lagged(T, Lag::from_raw(1)).unwrap();
    let y0 = pag.add_lagged(Y, Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_marked(MarkedEdge::bidirected(t1, y0)).unwrap();
    AcceptedGraph::temporal_pag(pag).expect("no circle marks remain, so the accept gate admits it")
}

/// A temporal CPDAG that reaches `GraphClass::TemporalCpdag` and cannot
/// complete.
///
/// `AcceptedGraph::temporal_cpdag` counts conflict marks and refuses them, so
/// this must come through the discovery path: `TemporalCpdagReview::from_cpdag`
/// sorts edges into pending-directed and pending-undirected and a conflict
/// (`x-x`) mark is neither, so the review reports `is_complete()` with the
/// conflict still in the graph, `AcceptedGraph::accept` admits it, and
/// `try_into_temporal_dag` then refuses that same mark. This is the CPDAG
/// counterpart of the PAG's bidirected mark.
fn incompletable_temporal_cpdag() -> AcceptedGraph {
    let mut cpdag = TemporalCpdag::empty();
    let t1 = cpdag.add_lagged(T, Lag::from_raw(1)).unwrap();
    let y0 = cpdag.add_lagged(Y, Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_marked(MarkedEdge::conflict(t1, y0)).unwrap();
    let review = TemporalCpdagReview::from_cpdag(cpdag, "probe.temporal_cpdag");
    assert!(review.is_complete(), "a conflict mark is on neither pending list");
    assert!(
        review.graph.try_into_temporal_dag().is_err(),
        "the conflict mark still blocks completion"
    );
    AcceptedGraph::accept(review).expect("a complete review is accepted")
}

fn pulse_query() -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(T, Y, 1.0))
}

/// The licensed Sustained form: a single-step window (`from == until`), not the
/// multi-step schedule the estimator refuses.
fn single_step_sustained_query() -> CausalQuery {
    let mut q = TemporalEffectQuery::sustained(T, Y, 0, 1.0);
    assert!(matches!(q.policy, TemporalPolicy::Sustained { from: 0, until: 0, .. }));
    q.policy = TemporalPolicy::sustained(0, 0);
    CausalQuery::TemporalEffect(q)
}

/// `build()` must admit the cell, and `run()` must fail with the specific
/// completion `CausalError::Compile` -- not a support refusal, not a number.
/// Swept across every validation suite, because the allowlist rows for this
/// coordinate leave the validation axis unconstrained.
fn assert_admitted_then_compile_error(graph: &AcceptedGraph, query: &CausalQuery, label: &str) {
    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        let study = Study::series(series())
            .graph(graph.clone())
            .query(query.clone())
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap_or_else(|e| panic!("{label}/{suite:?}: support gate refused build(): {e}"));
        let ctx = ExecutionContext::for_tests(21);
        let err = study.run(&ctx).err().unwrap_or_else(|| {
            panic!("{label}/{suite:?}: incompletable structure must not produce an estimate")
        });
        assert!(
            matches!(err, CausalError::Compile { .. }),
            "{label}/{suite:?}: expected CausalError::Compile, got {err}"
        );
    }
}

#[test]
fn pulse_on_incompletable_accepted_temporal_pag_reaches_compile() {
    assert_admitted_then_compile_error(&incompletable_temporal_pag(), &pulse_query(), "pulse/pag");
}

#[test]
fn pulse_on_incompletable_accepted_temporal_cpdag_reaches_compile() {
    assert_admitted_then_compile_error(
        &incompletable_temporal_cpdag(),
        &pulse_query(),
        "pulse/cpdag",
    );
}

/// The `SustainedEffect` half of the same policy-generic compile arm. Before
/// `parity/support_allowlist.toml` carried a matching `SustainedEffect` row,
/// this refused at `build()` with the generic "neither licensed nor on the
/// named running allowlist" message while the Pulse tests above reached the
/// specific completion error -- an inconsistency between two policies of one
/// query family sharing a single arm.
#[test]
fn sustained_on_incompletable_accepted_temporal_pag_reaches_compile() {
    assert_admitted_then_compile_error(
        &incompletable_temporal_pag(),
        &single_step_sustained_query(),
        "sustained/pag",
    );
}

#[test]
fn sustained_on_incompletable_accepted_temporal_cpdag_reaches_compile() {
    assert_admitted_then_compile_error(
        &incompletable_temporal_cpdag(),
        &single_step_sustained_query(),
        "sustained/cpdag",
    );
}

/// The allowlist row is Frequentist-only, matching its parent family: the whole
/// running `SustainedEffect` family is Frequentist (Bayesian Sustained has no
/// licensed or allowlisted cell anywhere), so the new row must not be broader
/// than the family it rides.
#[test]
fn bayesian_sustained_on_incompletable_accepted_temporal_structures_stays_refused() {
    for graph in [incompletable_temporal_pag(), incompletable_temporal_cpdag()] {
        let err = Study::series(series())
            .graph(graph)
            .query(single_step_sustained_query())
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .expect_err("Bayesian Sustained is outside the allowlist row");
        assert!(
            matches!(err, CausalError::Support { .. }),
            "expected a support refusal, got {err}"
        );
    }
}
