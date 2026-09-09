//! End-to-end probe for incompletable `TemporalCpdag`/`TemporalPag` structure.
//!
//! Incomplete TemporalCpdag/Pag Pulse / single-step Sustained is licensed.
//! Bidirected / conflict graphs pass the support gate and fail at the
//! completion sampler. Successful `try_into_temporal_dag` still collapses to
//! the `TemporalDag` cell. Bayesian incomplete-class cells stay closed.
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
use antecedent_graph::{MarkedEdge, TemporalCpdag, TemporalPag};

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
    AcceptedGraph::temporal_pag(pag)
}

/// A temporal CPDAG that reaches `GraphClass::TemporalCpdag` and cannot
/// complete.
///
/// `AcceptedGraph::temporal_cpdag` and `AcceptedGraph::accept` refuse conflict
/// marks. `From<TemporalCpdag>` keeps the class so the licensed cell can build;
/// the completion sampler then refuses the `x-x` mark. Counterpart of the
/// PAG's bidirected mark.
fn incompletable_temporal_cpdag() -> AcceptedGraph {
    let mut cpdag = TemporalCpdag::empty();
    let t1 = cpdag.add_lagged(T, Lag::from_raw(1)).unwrap();
    let y0 = cpdag.add_lagged(Y, Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_marked(MarkedEdge::conflict(t1, y0)).unwrap();
    assert!(
        cpdag.try_into_temporal_dag().is_err(),
        "the conflict mark still blocks completion"
    );
    AcceptedGraph::from(cpdag)
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

/// Bidirected / conflict graphs reach the licensed TemporalCpdag/Pag cell, then
/// fail at the completion sampler — not a Support refusal and not a number.
fn assert_identify_completion_refusal(graph: &AcceptedGraph, query: &CausalQuery, label: &str) {
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        let study = Study::series(series())
            .graph(graph.clone())
            .query(query.clone())
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap_or_else(|e| panic!("{label}/{suite:?}: licensed cell must build: {e}"));
        let err = study
            .run(&ctx)
            .expect_err(&format!("{label}/{suite:?}: incompletable structure must not estimate"));
        let msg = err.to_string();
        assert!(
            msg.contains("bidirected")
                || msg.contains("conflict")
                || msg.contains("completion")
                || msg.contains("refuses"),
            "{label}/{suite:?}: {msg}"
        );
    }
}

#[test]
fn pulse_on_incompletable_accepted_temporal_pag_reaches_compile() {
    assert_identify_completion_refusal(&incompletable_temporal_pag(), &pulse_query(), "pulse/pag");
}

#[test]
fn pulse_on_incompletable_accepted_temporal_cpdag_reaches_compile() {
    assert_identify_completion_refusal(
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
    assert_identify_completion_refusal(
        &incompletable_temporal_pag(),
        &single_step_sustained_query(),
        "sustained/pag",
    );
}

#[test]
fn sustained_on_incompletable_accepted_temporal_cpdag_reaches_compile() {
    assert_identify_completion_refusal(
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
