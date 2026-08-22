//! End-to-end probe for the `TemporalCpdag`/`TemporalPag` × `accepted` support cell.
//!
//! `effective_graph_class` collapses an accepted temporal CPDAG/PAG to
//! `TemporalDag` whenever `try_into_temporal_dag()` succeeds, so this cell is
//! reached exactly when completion *fails*. 0.9 refuses that coordinate at
//! `Study::build()` with a named completion reason (`parity/support_closed.toml`),
//! not the generic unlicensed message and not a number. Successful completion
//! still collapses to the `TemporalDag` cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, CausalError, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
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

/// `build()` must refuse with the named completion reason — not a number,
/// not the generic unlicensed message.
fn assert_support_completion_refusal(graph: &AcceptedGraph, query: &CausalQuery, label: &str) {
    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        let err = Study::series(series())
            .graph(graph.clone())
            .query(query.clone())
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .expect_err(&format!("{label}/{suite:?}: incompletable structure must not build"));
        assert!(
            matches!(err, CausalError::Support { .. }),
            "{label}/{suite:?}: expected CausalError::Support, got {err}"
        );
        let msg = err.to_string();
        assert!(msg.starts_with("refused:"), "{label}/{suite:?}: {msg}");
        assert!(
            msg.contains("try_into_temporal_dag") || msg.contains("completion"),
            "{label}/{suite:?}: {msg}"
        );
    }
}

#[test]
fn pulse_on_incompletable_accepted_temporal_pag_reaches_compile() {
    assert_support_completion_refusal(&incompletable_temporal_pag(), &pulse_query(), "pulse/pag");
}

#[test]
fn pulse_on_incompletable_accepted_temporal_cpdag_reaches_compile() {
    assert_support_completion_refusal(
        &incompletable_temporal_cpdag(),
        &pulse_query(),
        "pulse/cpdag",
    );
}

/// The `SustainedEffect` half of the same policy-generic compile arm. Before
/// the incomplete TemporalCpdag/Pag Pulse/Sustained cells were named as
/// completion refusals, this refused at `build()` with the generic "neither
/// licensed nor on the named running allowlist" message while the Pulse tests
/// above reached the specific completion error -- an inconsistency between two
/// policies of one query family sharing a single arm.
#[test]
fn sustained_on_incompletable_accepted_temporal_pag_reaches_compile() {
    assert_support_completion_refusal(
        &incompletable_temporal_pag(),
        &single_step_sustained_query(),
        "sustained/pag",
    );
}

#[test]
fn sustained_on_incompletable_accepted_temporal_cpdag_reaches_compile() {
    assert_support_completion_refusal(
        &incompletable_temporal_cpdag(),
        &single_step_sustained_query(),
        "sustained/cpdag",
    );
}

/// Incomplete TemporalCpdag/Pag still refuse Bayesian Sustained at `build()`.
/// The licensed Bayesian Sustained cell is `TemporalDag` (explicit/accepted);
/// completion failure is not that cell.
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
            .expect_err("Bayesian Sustained on incomplete TemporalCpdag/Pag stays refused");
        assert!(
            matches!(err, CausalError::Support { .. }),
            "expected a support refusal, got {err}"
        );
    }
}
