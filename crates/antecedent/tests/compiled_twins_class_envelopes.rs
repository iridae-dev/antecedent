//! The PAG path-specific, class transport and temporal class envelope methods of
//! `GeneralizedAdjustmentIdentifier`, and its MAG candidate enumeration, are the identification
//! the compiled routes publish.
//!
//! Each compiled route configures `GeneralizedAdjustmentIdentifier` with the caller's execution
//! context (and, on the temporal routes, the caller's materialization budget), which bounds
//! storage, work and cancellation and does not change the answer, and calls one method. Each
//! test calls the same method directly on the same structure and query and compares the
//! envelope with the identification certificate the executed route retains: the status, the
//! identified and unidentified weights bit for bit, the completions and their graphs, the
//! truncation count, the critical graph features, the invariant estimand and each
//! completion's status, estimands and functional.
//!
//! - `identify_pag_path_specific_envelope`: the path-restricted effect and the natural direct
//!   effect over a supplied PAG (route cell `PathSpecificQuery` x `Pag`);
//! - `admg_candidates_of_mag`: every ADMG sharing a MAG completion's m-separation model, read
//!   by the PAG path-specific and PAG transport routes; no compiled route publishes the
//!   candidate graphs themselves, so the twin is the count the route's envelope states
//!   (`pag_admg_candidates`), recomputed from the direct enumeration of each completion;
//! - `identify_cpdag_transport_envelope` and `identify_pag_transport_envelope`: the
//!   trial-to-target transport over a supplied CPDAG and PAG (`TransportQuery` x `Cpdag`,
//!   `Pag`);
//! - `identify_temporal_cpdag_envelope` and `identify_temporal_pag_envelope`: the temporal
//!   effect over a supplied temporal CPDAG and PAG (`TemporalEffectQuery` x `TemporalCpdag`,
//!   `TemporalPag`), whose refinement leaves a temporal effect query unchanged;
//! - `identify_temporal_pag_path_specific_envelope`: the natural mediation over the window of a
//!   supplied temporal PAG (`Mediation` x `TemporalPag`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities; weights are compared by their bits"
)]

#[path = "fixtures_common/mod.rs"]
mod common;

use std::fmt::Debug;
use std::sync::Arc;

use antecedent::identify::{GeneralizedAdjustmentIdentifier, IdentificationEnvelope};
use antecedent::{Identification, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent::{IntoGraphInput, TransportTrialSpec};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag, MediationContrast,
    MediationQuery, NestedCounterfactualQuery, PathSpecificEffectQuery, ResponseFunctional,
    ResponseQuery, TemporalEffectQuery, TemporalPolicy, TransportQuery, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Cpdag, DenseNodeId, Pag, TemporalPag};
use antecedent_identify::{TemporalClassEnvelope, temporal_pag_mediation_window};

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread()
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn twin thread")
        .join()
        .expect("twin body panicked");
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn bernoulli(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

/// Canonical text of a case's first functional: the form the class identifiers compare.
fn functional(result: &antecedent::identify::IdentificationResult) -> Option<String> {
    let estimand = result.estimands.first()?;
    ((estimand.functional.raw() as usize) < result.arena.len())
        .then(|| result.arena.pretty(estimand.functional))
}

/// Everything two envelopes state: status, weights by their bits, completions and their
/// graphs, truncation, critical features, the invariant estimand, and each completion's
/// weight, status, estimands and functional.
fn assert_same_envelope<G: Debug>(
    label: &str,
    direct: &IdentificationEnvelope<G>,
    compiled: &IdentificationEnvelope<G>,
) {
    assert_eq!(direct.status, compiled.status, "{label}: status");
    assert_eq!(
        direct.identified_weight.0.to_bits(),
        compiled.identified_weight.0.to_bits(),
        "{label}: identified weight"
    );
    assert_eq!(
        direct.unidentified_weight.0.to_bits(),
        compiled.unidentified_weight.0.to_bits(),
        "{label}: unidentified weight"
    );
    assert_eq!(direct.truncated_completions, compiled.truncated_completions, "{label}: truncated");
    let features = |envelope: &IdentificationEnvelope<G>| -> Vec<(String, String)> {
        envelope
            .critical_graph_features
            .iter()
            .map(|feature| (feature.kind.to_string(), feature.detail.to_string()))
            .collect()
    };
    assert_eq!(features(direct), features(compiled), "{label}: critical graph features");
    assert_eq!(
        direct.invariant.as_ref().map(|estimand| estimand.method.to_string()),
        compiled.invariant.as_ref().map(|estimand| estimand.method.to_string()),
        "{label}: invariant estimand"
    );
    assert_eq!(direct.cases.len(), compiled.cases.len(), "{label}: completions");
    for (index, (left, right)) in direct.cases.iter().zip(&compiled.cases).enumerate() {
        assert_eq!(
            format!("{:?}", left.graph),
            format!("{:?}", right.graph),
            "{label}: case {index} graph"
        );
        assert_eq!(
            left.weight.0.to_bits(),
            right.weight.0.to_bits(),
            "{label}: case {index} weight"
        );
        assert_eq!(left.result.status, right.result.status, "{label}: case {index} status");
        assert_eq!(
            left.result.estimands.len(),
            right.result.estimands.len(),
            "{label}: case {index} estimands"
        );
        for (a, b) in left.result.estimands.iter().zip(&right.result.estimands) {
            assert_eq!(a.method, b.method, "{label}: case {index} method");
            assert_eq!(a.adjustment_set, b.adjustment_set, "{label}: case {index} adjustment set");
        }
        assert_eq!(
            functional(&left.result),
            functional(&right.result),
            "{label}: case {index} functional"
        );
    }
}

fn assert_same_temporal_envelope(
    label: &str,
    direct: &TemporalClassEnvelope,
    compiled: &TemporalClassEnvelope,
) {
    assert_same_envelope(label, &direct.envelope, &compiled.envelope);
    assert_eq!(direct.indexers, compiled.indexers, "{label}: unfold indexers");
    for (index, (left, right)) in
        direct.envelope.cases.iter().zip(&compiled.envelope.cases).enumerate()
    {
        assert_eq!(
            left.graph.fingerprint(),
            right.graph.fingerprint(),
            "{label}: case {index} completion fingerprint"
        );
    }
}

/// The identification certificate an executed route retains.
fn certificate(result: &StudyResult) -> &Identification {
    &result.certificate.as_ref().expect("the class route retains its identification").identification
}

// --- path-restricted effect and natural direct effect over a supplied PAG ----------------

const ROWS: f64 = 100_000.0;

/// Columns from the cells of a joint over `N` binary variables, `ROWS` rows in all.
fn table_from_joint<const N: usize>(
    names: [&str; N],
    joint: impl Fn([usize; N]) -> f64,
) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); N];
    for cell in 0..(1_usize << N) {
        let mut levels = [0_usize; N];
        for (position, level) in levels.iter_mut().enumerate() {
            *level = (cell >> (N - 1 - position)) & 1;
        }
        let count = (ROWS * joint(levels)).round() as usize;
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(level as f64, count));
        }
    }
    TabularData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

fn pag_of(nodes: u32, directed: &[(u32, u32)]) -> Pag {
    let mut pag = Pag::with_variables(nodes);
    for &(from, to) in directed {
        pag.insert_directed(node(from), node(to)).unwrap();
    }
    pag
}

/// `W -> T`; `T -> A`, `T -> B`; `A, B -> Y`: columns `w, t, a, b, y`.
fn parented_data() -> TabularData {
    table_from_joint(["w", "t", "a", "b", "y"], |[w, t, a, b, y]| {
        bernoulli(0.4, w)
            * bernoulli(0.3 + 0.4 * w as f64, t)
            * bernoulli(0.15 + 0.4 * t as f64, a)
            * bernoulli(0.2 + 0.5 * t as f64, b)
            * bernoulli(0.05 + 0.3 * a as f64 + 0.2 * b as f64 + 0.25 * (a * b) as f64, y)
    })
}

/// `C -> T`; `T -> M`, `T -> Y`; `W -> M`; `M -> Y`: columns `c, w, t, m, y`.
fn mediated_data() -> TabularData {
    table_from_joint(["c", "w", "t", "m", "y"], |[c, w, t, m, y]| {
        bernoulli(0.45, c)
            * bernoulli(0.35, w)
            * bernoulli(0.25 + 0.5 * c as f64, t)
            * bernoulli(0.15 + 0.4 * t as f64 + 0.3 * w as f64, m)
            * bernoulli(0.1 + 0.2 * t as f64 + 0.3 * m as f64 + 0.2 * (t * m) as f64, y)
    })
}

/// Every mark decided, every edge into `Y` and out of `T` visible: one functional.
fn parented_pag() -> Pag {
    pag_of(5, &[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
}

/// Every mark decided; the candidates of the MAG identify the natural direct effect by
/// different functionals (the district formula of the candidate with `W <-> M` and the edge
/// formula of the others).
fn mediated_pag() -> Pag {
    pag_of(5, &[(0, 2), (2, 3), (2, 4), (1, 3), (3, 4)])
}

fn path_through_a() -> CausalQuery {
    CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(v(1), v(4)).with_path_nodes([v(2)]))
}

fn natural_direct() -> NestedCounterfactualQuery {
    NestedCounterfactualQuery::new(v(2), v(3), v(4)).unwrap()
}

fn run_class(
    data: TabularData,
    graph: impl IntoGraphInput,
    query: CausalQuery,
    seed: u64,
) -> StudyResult {
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn pag_certificate(result: &StudyResult) -> &IdentificationEnvelope<Pag> {
    let Identification::Envelope { envelope, .. } = certificate(result) else {
        panic!("a PAG route retains its candidate envelope");
    };
    envelope
}

/// The leading counts of the route's `pag_admg_candidates` feature: completions, candidates
/// and distinct functionals.
fn stated_candidates<G>(envelope: &IdentificationEnvelope<G>) -> Vec<usize> {
    let feature = envelope
        .critical_graph_features
        .iter()
        .find(|feature| feature.kind.as_ref() == "pag_admg_candidates")
        .expect("the PAG route states its candidate enumeration");
    feature.detail.split_whitespace().filter_map(|word| word.parse().ok()).collect()
}

/// The MAG completions of a PAG envelope in enumeration order: the cases of one completion
/// are consecutive and share its graph.
fn completions(envelope: &IdentificationEnvelope<Pag>) -> Vec<Pag> {
    let mut mags: Vec<Pag> = Vec::new();
    for case in &envelope.cases {
        if mags.last() != Some(&case.graph) {
            mags.push(case.graph.clone());
        }
    }
    mags
}

/// The candidate count the compiled route states is the direct enumeration of every
/// completion it read.
fn assert_candidates_match(label: &str, envelope: &IdentificationEnvelope<Pag>) -> usize {
    let mags = completions(envelope);
    let identifier = GeneralizedAdjustmentIdentifier::new();
    let mut candidates = 0;
    for (index, mag) in mags.iter().enumerate() {
        let found = identifier.admg_candidates_of_mag(mag).unwrap();
        assert!(!found.bounded, "{label}: completion {index} enumerated within its bound");
        assert!(!found.members.is_empty(), "{label}: completion {index} stands for an ADMG");
        candidates += found.members.len();
    }
    let stated = stated_candidates(envelope);
    assert_eq!(stated[0], mags.len(), "{label}: completions the route read");
    assert_eq!(stated[1], candidates, "{label}: candidates the route read");
    candidates
}

#[test]
fn pag_path_specific_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let query = path_through_a();
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_pag_path_specific_envelope(&parented_pag(), &query)
            .unwrap();
        let result = run_class(parented_data(), parented_pag(), query, 41);
        assert_same_envelope("pag path-specific", &direct, pag_certificate(&result));
    });
}

#[test]
fn pag_natural_direct_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let nested = natural_direct();
        // The route identifies the natural direct effect as the mediation contrast it is.
        let identified = CausalQuery::Mediation(nested.as_mediation_query());
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_pag_path_specific_envelope(&mediated_pag(), &identified)
            .unwrap();
        let result = run_class(
            mediated_data(),
            mediated_pag(),
            CausalQuery::NestedCounterfactual(nested),
            44,
        );
        assert_same_envelope("pag natural direct", &direct, pag_certificate(&result));
    });
}

#[test]
fn admg_candidates_of_each_mag_are_the_candidates_the_compiled_route_read() {
    on_large_stack(|| {
        let result = run_class(parented_data(), parented_pag(), path_through_a(), 41);
        let envelope = pag_certificate(&result);
        assert_candidates_match("parented pag", envelope);
        // One functional per distinct candidate reading.
        assert_eq!(stated_candidates(envelope)[2], envelope.cases.len());

        let result = run_class(
            mediated_data(),
            mediated_pag(),
            CausalQuery::NestedCounterfactual(natural_direct()),
            44,
        );
        let envelope = pag_certificate(&result);
        let candidates = assert_candidates_match("mediated pag", envelope);
        assert!(candidates > 1, "the mediated MAG stands for several ADMGs");
        assert_eq!(stated_candidates(envelope)[2], envelope.cases.len());
    });
}

// --- trial-to-target transport over a supplied CPDAG or PAG ------------------------------

const A: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 5;
const NODES: u32 = 6;

/// Two trial rows with known selection and arm probabilities and a covariate column.
fn transport_data() -> TabularData {
    TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[0.5, 0.5, 0.5, 0.5][..]),
        ("e", &[0.5, 0.5, 0.5, 0.5][..]),
        ("z", &[0.0, 1.0, 0.0, 1.0][..]),
    ])
    .unwrap()
}

fn transport_query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(Y),
        treatment: ContinuousDomain::new(v(A), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    TransportQuery::new(response, "trial", "target", [v(A)])
}

fn run_transport(graph: impl IntoGraphInput) -> StudyResult {
    Study::tabular(transport_data())
        .graph(graph)
        .query(CausalQuery::Transport(transport_query()))
        .selection_targets(Arc::<[VariableId]>::from(vec![v(Z)]))
        .transport_trial(TransportTrialSpec {
            trial: v(2),
            selection_probability: v(3),
            treatment_probability: v(4),
        })
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9_101))
        .unwrap()
}

/// `a - y - z`: three DAG completions; with `z` as the selection target two certify the
/// direct formula and one standardizes over `z`.
fn open_chain() -> Cpdag {
    let mut graph = Cpdag::with_variables(NODES);
    graph.insert_undirected(node(A), node(Y)).unwrap();
    graph.insert_undirected(node(Z), node(Y)).unwrap();
    graph
}

fn circle_pair() -> Pag {
    let mut pag = Pag::with_variables(NODES);
    pag.insert_circle_circle(node(A), node(Y)).unwrap();
    pag
}

#[test]
fn cpdag_transport_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_cpdag_transport_envelope(&open_chain(), &[v(Z)], &transport_query())
            .unwrap();
        assert_eq!(direct.proofs.len(), direct.envelope.cases.len());
        let result = run_transport(open_chain());
        let Identification::CpdagEnvelope { envelope, .. } = certificate(&result) else {
            panic!("a CPDAG transport retains its completion envelope");
        };
        assert!(direct.envelope.cases.len() > 1, "the fixture has several completions");
        assert_same_envelope("cpdag transport", &direct.envelope, envelope);
    });
}

#[test]
fn pag_transport_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_pag_transport_envelope(&circle_pair(), &[v(Z)], &transport_query())
            .unwrap();
        assert_eq!(direct.proofs.len(), direct.envelope.cases.len());
        let result = run_transport(circle_pair());
        let envelope = pag_certificate(&result);
        assert_same_envelope("pag transport", &direct.envelope, envelope);
        assert_candidates_match("pag transport", envelope);
    });
}

// --- temporal effect over a supplied temporal CPDAG or PAG --------------------------------

fn temporal_query(multi_step: bool) -> TemporalEffectQuery {
    let (treatment, outcome) = (v(0), v(1));
    if multi_step {
        TemporalEffectQuery::sustained(treatment, outcome, -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1)
    } else {
        TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
    }
}

fn run_series(
    data: TimeSeriesData,
    graph: impl IntoGraphInput,
    query: CausalQuery,
    replicates: u32,
) -> StudyResult {
    Study::series(data)
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(977))
        .unwrap()
}

fn temporal_certificate(result: &StudyResult) -> &TemporalClassEnvelope {
    let Identification::TemporalEnvelope { envelope, .. } = certificate(result) else {
        panic!("a temporal class route retains its completion envelope");
    };
    envelope
}

#[test]
fn temporal_cpdag_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let graph = common::fixtures::confounded_cpdag();
        for multi_step in [false, true] {
            let query = temporal_query(multi_step);
            let direct = GeneralizedAdjustmentIdentifier::new()
                .identify_temporal_cpdag_envelope(&graph, &query)
                .unwrap();
            let result = run_series(
                common::fixtures::confounded_series(640, 0.0, 0.0, 977),
                graph.clone(),
                CausalQuery::TemporalEffect(query),
                16,
            );
            assert!(direct.envelope.cases.len() > 1, "the fixture has several completions");
            assert_same_temporal_envelope(
                &format!("temporal cpdag (multi-step {multi_step})"),
                &direct,
                temporal_certificate(&result),
            );
        }
    });
}

#[test]
fn temporal_pag_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let graph = common::fixtures::chain_pag();
        for multi_step in [false, true] {
            let query = temporal_query(multi_step);
            let direct = GeneralizedAdjustmentIdentifier::new()
                .identify_temporal_pag_envelope(&graph, &query)
                .unwrap();
            let result = run_series(
                common::fixtures::chain_pag_series(640, 0.0, 977),
                graph.clone(),
                CausalQuery::TemporalEffect(query),
                16,
            );
            assert_same_temporal_envelope(
                &format!("temporal pag (multi-step {multi_step})"),
                &direct,
                temporal_certificate(&result),
            );
        }
    });
}

// --- natural mediation over the window of a supplied temporal PAG -------------------------

const C: u32 = 0;
const X: u32 = 1;
const W: u32 = 2;
const M: u32 = 3;
const YT: u32 = 4;

/// `c(t-1) -> x(t)`, `x -> m`, `w -> m`, `m -> y`, `x -> y` within a period.
fn lagged_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(v(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(v(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(YT), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// Deterministic uniform draws (splitmix64).
struct Draws(u64);

impl Draws {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn flip(&mut self, p: f64) -> f64 {
        f64::from(self.unit() < p)
    }
}

/// A series whose treatment follows the previous period's `c` and whose other variables
/// follow the generating equations within the period.
fn lagged_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut c, mut x, mut w, mut m, mut y) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        c[t] = draws.flip(0.45);
        w[t] = draws.flip(0.35);
        let previous = if t == 0 { 0.0 } else { c[t - 1] };
        x[t] = draws.flip(0.25 + 0.5 * previous);
        m[t] = draws.flip(0.15 + 0.4 * x[t] + 0.3 * w[t]);
        y[t] = draws.flip(0.1 + 0.2 * x[t] + 0.3 * m[t] + 0.2 * x[t] * m[t]);
    }
    TimeSeriesData::from_f64_columns(
        [("c", &c[..]), ("x", &x[..]), ("w", &w[..]), ("m", &m[..]), ("y", &y[..])],
        1,
    )
    .unwrap()
}

#[test]
fn temporal_pag_path_specific_envelope_equals_the_compiled_route() {
    on_large_stack(|| {
        let template = lagged_template();
        for contrast in [MediationContrast::NaturalIndirect, MediationContrast::NaturalDirect] {
            let query = MediationQuery::binary(v(X), v(YT), [v(M)], contrast);
            // The pulse contrast whose window the route reads: same treatment, outcome, levels
            // and horizon.
            let mut effect = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0)
                .with_horizon_steps(1);
            effect.control = query.control.clone();
            effect.active = query.active.clone();
            effect.target_population = query.target_population.clone();
            let window = temporal_pag_mediation_window(&template, &effect, &query, 5)
                .unwrap()
                .with_contrast(contrast);
            let direct = GeneralizedAdjustmentIdentifier::new()
                .identify_temporal_pag_path_specific_envelope(&template, &window)
                .unwrap();
            let result = run_series(
                lagged_series(6_000, 23),
                template.clone(),
                CausalQuery::Mediation(query),
                0,
            );
            assert_same_temporal_envelope(
                &format!("temporal pag mediation ({contrast:?})"),
                &direct,
                temporal_certificate(&result),
            );
        }
    });
}
