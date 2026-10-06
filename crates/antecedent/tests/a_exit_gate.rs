//! 2.2 A exit gate: six end-to-end user stories, one test each.
//!
//! Every story runs start to finish in ONE test through the public facade:
//! it builds its own graph, catalog and evidence from scratch (nothing here is
//! shared with the unit or lifecycle tests), prepares once, estimates, refreshes
//! where the cell supports it, exports the artifact, drops every producer object,
//! and then consumes the artifact bytes alone in a fresh scope. The consumer's
//! recomputed answers must equal the producer's bit for bit and equal the exact
//! enumerated truth where the story has one. Each story also asserts its named
//! refusal or incomplete side.
//!
//! Story 3 checks the public analytic interval. Its coverage is measured separately
//! in `learned_continuous_calibration.rs` under `scripts/gate_calibration.sh`.
//! `scripts/gate_a_exit.sh` verifies the corresponding coverage records.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::cross_world::{
    CrossWorldOptions, check, consume_cross_world_artifact, evaluate_cross_world_effect,
};
use antecedent::gcm::NestedOutcomeMechanism;
use antecedent::{
    StudyBuilder, consume_learned_continuous_artifact, consume_mixed_source_artifact,
    consume_mz_transport_artifact, consume_temporal_transport_artifact,
    consume_transport_scenarios_artifact,
};
use antecedent_core::{
    ContinuousDomain, CrossWorldQuery, DependenceGroup, DistributionAvailability, Environment,
    EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext, GridSpec,
    InterventionAssignment as CoreIntervention, RegimeBinding, RegimeId, RegimeKind,
    ResponseFunctional, ResponseQuery, SamplingDesign, SearchLimits, SearchStop, TransportQuery,
    Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    LearnedContinuousOptions, LearnerSpec, LinearSpec, TrialAipwInput, TrialSampling,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    ExactTransportData, InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::sid::scenarios::{
    ScenarioCoordinate, TransportScenario, TransportScenarioSet, decide_transport_scenarios,
};
use antecedent_identify::sid::temporal_sequence::{TemporalSequenceSpec, TemporalSlots};
use antecedent_identify::{
    ClassicalTransportQuery, MIXED_SOURCE_DEFAULT_LIMITS, MZ_TRANSPORT_DEFAULT_LIMITS,
    MixedSourceDecision, MixedSourceQuery, MzTransportDecision, MzTransportQuery,
    ZTransportSourceSpec, bind_mixed_source_catalog, bind_mz_transport_catalog,
    decide_mixed_source, decide_mz_transport,
};
use antecedent_io::IoError;
use antecedent_io::learned_continuous_artifact::LearnedContinuousConsumeLimits;
use antecedent_io::mixed_source_artifact::MixedSourceConsumeLimits;
use antecedent_io::mz_transport_artifact::{
    MZ_INTERVAL_NOT_LICENSED, MZ_WITHHELD, MzTransportConsumeLimits,
};
use antecedent_io::temporal_transport_artifact::TemporalTransportConsumeLimits;
use antecedent_io::transport_scenario_artifact::TransportScenarioConsumeLimits;

// ---------------------------------------------------------------------------
// A tiny exact structural-model enumerator, private to this gate.
// ---------------------------------------------------------------------------

type Mech = Box<dyn Fn(&[u8], &[u8]) -> u8>;

/// Node `i` takes `f[i](values, exogenous)`; independent exogenous bits with `p`.
struct Scm {
    p: Vec<f64>,
    f: Vec<Mech>,
}

impl Scm {
    /// Exact joint over `measured` (first variable most significant) under `do_`.
    fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; 1 << measured.len()];
        for mask in 0..(1usize << self.p.len()) {
            let exo: Vec<u8> = (0..self.p.len()).map(|b| u8::from((mask >> b) & 1 == 1)).collect();
            let weight: f64 =
                exo.iter().zip(&self.p).map(|(e, p)| if *e == 1 { *p } else { 1.0 - *p }).product();
            let mut values = vec![0u8; self.f.len()];
            for i in 0..self.f.len() {
                values[i] = match do_.iter().find(|(v, _)| *v == i) {
                    Some((_, level)) => *level,
                    None => (self.f[i])(&values, &exo),
                };
            }
            let index = measured.iter().fold(0usize, |a, v| (a << 1) | usize::from(values[*v]));
            out[index] += weight;
        }
        out.into_iter().map(|p| p.clamp(0.0, 1.0)).collect()
    }

    fn risk(&self, do_: &[(usize, u8)], y: usize) -> f64 {
        self.law(do_, &[y])[1]
    }
}

fn vid(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

fn bit(b: bool) -> u8 {
    u8::from(b)
}

fn bool_axis(i: usize) -> DiscreteAxis {
    DiscreteAxis { variable: vid(i), values: Arc::from([Value::Bool(false), Value::Bool(true)]) }
}

fn num_axis(i: usize) -> DiscreteAxis {
    DiscreteAxis { variable: vid(i), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

fn admg(n: u32, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> Admg {
    let mut g = Admg::with_variables(n);
    let d = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap());
    for (a, b) in directed {
        g.insert_directed(d(*a), d(*b)).unwrap();
    }
    for (a, b) in bidirected {
        g.insert_bidirected(d(*a), d(*b)).unwrap();
    }
    g
}

fn binding(regime: RegimeId, snapshot: &str) -> RegimeBinding {
    RegimeBinding {
        dataset_identity: None,
        regime,
        snapshot_identity: Arc::from(snapshot),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    }
}

/// `P(outcome = 1)` of an evaluated binary distribution.
fn risk_of(distribution: &ExactDistribution) -> f64 {
    distribution
        .atoms
        .iter()
        .zip(distribution.probabilities.iter())
        .filter(|(atom, _)| atom[0] == Value::Bool(true))
        .map(|(_, p)| *p)
        .sum()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

// ---------------------------------------------------------------------------
// Story 1 (X1): a multi-source limited-experiment effect requiring complementary sources.
// ---------------------------------------------------------------------------
//
// `z1 -> x -> z2 -> y` with `z1 <-> x`, `z1 <-> z2`, `z1 <-> y`. Source `a` changes the
// `z1` and `z2` mechanisms and can experiment on `z2`; source `b` changes `z1` and `y`
// and can experiment on `z1`. Neither identifies `P*(y | do(x))` alone; together they do.

mod s1 {
    use super::*;

    pub const Z1: usize = 0;
    pub const X: usize = 1;
    pub const Z2: usize = 2;
    pub const Y: usize = 3;
    pub const NAMES: [&str; 4] = ["z1", "x", "z2", "y"];

    fn x_mech() -> Mech {
        Box::new(|v, e| bit(v[Z1] == 1) ^ bit(e[0] == 1 && e[4] == 1))
    }

    pub fn scm(population: &str, p: &[f64]) -> Scm {
        let z1: Mech = match population {
            "a" => Box::new(|_, e| bit(e[3] == 1 || e[1] == 1)),
            "b" => Box::new(|_, e| bit(e[0] == 1 && e[2] == 1)),
            _ => Box::new(|_, e| bit((e[0] == 1 && e[3] == 1) || (e[1] == 1 && e[2] == 1))),
        };
        let z2: Mech = match population {
            "a" => Box::new(|v, e| bit(v[X] == 1) ^ bit(e[1] == 1 && e[5] == 1)),
            _ => Box::new(|v, e| bit((v[X] == 1 && e[5] == 1) || (v[X] == 0 && e[1] == 1))),
        };
        let y: Mech = match population {
            "b" => Box::new(|v, e| bit(v[Z2] == 1) ^ bit(e[2] == 1 && e[6] == 1)),
            _ => Box::new(|v, e| bit((v[Z2] == 1 && e[6] == 1) || (e[2] == 1 && e[6] == 0))),
        };
        Scm { p: p.to_vec(), f: vec![z1, x_mech(), z2, y] }
    }

    pub const P: [f64; 7] = [0.4, 0.55, 0.5, 0.35, 0.65, 0.7, 0.45];
    pub const P_REFRESHED: [f64; 7] = [0.4, 0.55, 0.5, 0.35, 0.65, 0.7, 0.6];

    pub fn graph() -> Admg {
        admg(4, &[(Z1, X), (X, Z2), (Z2, Y)], &[(Z1, X), (Z1, Z2), (Z1, Y)])
    }

    pub fn source(
        population: &str,
        controllable: &[usize],
        selections: &[usize],
    ) -> ZTransportSourceSpec {
        ZTransportSourceSpec {
            population: Arc::from(population),
            controllable: controllable.iter().map(|i| vid(*i)).collect(),
            experiment_assignment: if population == "b" {
                Arc::from([CoreIntervention { variable: vid(Z1), value: Value::Bool(false) }])
            } else {
                Arc::from([])
            },
            selection_targets: selections.iter().map(|i| vid(*i)).collect(),
        }
    }

    pub fn query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
        MzTransportQuery {
            outcomes: Arc::from([vid(Y)]),
            treatments: Arc::from([vid(X)]),
            target: Arc::from("target"),
            sources: sources.into(),
        }
    }

    pub fn both() -> Vec<ZTransportSourceSpec> {
        vec![source("a", &[Z2], &[Z1, Z2]), source("b", &[Z1], &[Z1, Y])]
    }

    /// The target's observational law, `do(z2 = 0/1)` in `a`, `do(z1 = 0)` in `b`.
    pub fn evidence(p: &[f64]) -> (EvidenceCatalog, ExactTransportData) {
        let specs: [(&str, Vec<(usize, u8)>); 4] =
            [("target", vec![]), ("a", vec![(Z2, 0)]), ("a", vec![(Z2, 1)]), ("b", vec![(Z1, 0)])];
        let coordinates: Vec<_> = (0..4)
            .map(|i| VariableCoordinate {
                variable: vid(i),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect();
        let environments: Vec<_> = ["target", "a", "b"]
            .iter()
            .map(|pop| Environment::try_new(*pop, coordinates.clone(), []).unwrap())
            .collect();
        let (mut regimes, mut bindings, mut laws) = (Vec::new(), Vec::new(), Vec::new());
        for (k, (population, do_)) in specs.iter().enumerate() {
            let id = RegimeId::from_raw(u32::try_from(k).unwrap());
            let measured: Vec<usize> =
                (0..4).filter(|i| !do_.iter().any(|(v, _)| v == i)).collect();
            regimes.push(
                EvidenceRegime::try_new(
                    id,
                    if do_.is_empty() {
                        RegimeKind::Observational
                    } else {
                        RegimeKind::Experimental
                    },
                    EvidenceKind::Available,
                    do_.iter().map(|(v, _)| vid(*v)).collect::<Vec<_>>(),
                    do_.iter()
                        .map(|(v, l)| CoreIntervention {
                            variable: vid(*v),
                            value: Value::Bool(*l == 1),
                        })
                        .collect::<Vec<_>>(),
                    measured.iter().map(|i| vid(*i)).collect::<Vec<_>>(),
                    *population,
                    DistributionAvailability::Joint,
                )
                .unwrap(),
            );
            let snapshot = format!("{population}-{k}");
            bindings.push(binding(id, &snapshot));
            laws.push(
                ExactDiscreteLaw::try_new(
                    *population,
                    id,
                    do_.iter()
                        .map(|(v, l)| {
                            InterventionAssignment::concrete(vid(*v), Value::Bool(*l == 1))
                        })
                        .collect::<Vec<_>>(),
                    measured.iter().map(|i| bool_axis(*i)).collect::<Vec<_>>(),
                    scm(population, p).law(do_, &measured),
                    snapshot,
                    LawTolerance::default(),
                )
                .unwrap(),
            );
        }
        (
            EvidenceCatalog::try_new(environments, regimes, bindings, None).unwrap(),
            ExactTransportData::try_new(laws, 4096).unwrap(),
        )
    }

    /// The same laws as counted tables of `n` rows each.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "probabilities are in [0, 1] and the sample sizes fit u64"
    )]
    pub fn counted(data: &ExactTransportData, n: f64) -> ExactTransportData {
        let laws = data
            .laws()
            .iter()
            .map(|law| {
                let counts: Vec<u64> =
                    law.probabilities().iter().map(|p| (p * n).round() as u64).collect();
                let total = counts.iter().sum::<u64>() as f64;
                ExactDiscreteLaw::try_empirical(
                    law.population(),
                    law.regime(),
                    law.interventions().to_vec(),
                    law.axes().to_vec(),
                    counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>(),
                    law.snapshot_identity(),
                    law.tolerance(),
                )
                .unwrap()
                .with_empirical_counts(counts)
                .unwrap()
            })
            .collect::<Vec<_>>();
        ExactTransportData::try_new(laws, data.max_support_rows()).unwrap()
    }

    pub fn request(level: bool) -> Assignment {
        Assignment::from_pairs([(vid(X), Value::Bool(level))])
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one end-to-end story per test")]
fn story_1_multi_source_limited_experiment_effect_requiring_complementary_sources() {
    use s1::*;
    let ctx = ExecutionContext::for_tests(101);
    let truth = |p: &[f64], x: u8| scm("target", p).risk(&[(X, x)], Y);

    // ---- producer scope: everything the producer holds dies at the closing brace.
    let (before, after, counted) = {
        let (catalog, data) = evidence(&P);
        let graph = graph();

        // Each source alone (with an unhelpful second source) is a checked obstruction ...
        let useless = source("u", &[X], &[Z1, X, Z2, Y]);
        for alone in [source("a", &[Z2], &[Z1, Z2]), source("b", &[Z1], &[Z1, Y])] {
            let decision = decide_mz_transport(
                &graph,
                &query(vec![alone, useless.clone()]),
                &catalog,
                MZ_TRANSPORT_DEFAULT_LIMITS,
                &ctx,
            )
            .unwrap();
            assert_eq!(decision.reason_code(), Some("transport_proven_non_transportable"));
        }
        // ... and the two together identify the query.
        let MzTransportDecision::Identified { derivation, cited } = decide_mz_transport(
            &graph,
            &query(both()),
            &catalog,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap() else {
            panic!("the complementary catalog identifies");
        };
        assert_eq!(cited.len(), 3, "the formula cites a's two arms and b's trial");
        let functional = bind_mz_transport_catalog(&graph, &derivation, &catalog).unwrap();
        let prepared = StudyBuilder::mz_transport(
            graph.clone(),
            functional.clone(),
            MZ_TRANSPORT_DEFAULT_LIMITS,
            data.clone(),
            vec![request(false), request(true)],
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let result = prepared.estimate(&ctx).unwrap();
        for x in [0u8, 1] {
            assert!(close(risk_of(&result.distributions()[usize::from(x)]), truth(&P, x)));
        }
        // The naive target conditional is not the answer: the fixture separates them.
        let joint = scm("target", &P).law(&[], &[X, Y]);
        assert!((joint[3] / (joint[2] + joint[3]) - truth(&P, 1)).abs() > 1e-3);
        let contrast = result.contrasts()[0];
        assert!(close(contrast.estimate, truth(&P, 1) - truth(&P, 0)));
        assert_eq!(
            result.uncertainty().status,
            antecedent_io::mz_transport_artifact::MZ_POINT_ONLY
        );
        let names = NAMES.map(String::from);
        let bytes_before = result.export_named(&prepared, &names, &ctx).unwrap();
        assert_eq!(bytes_before, result.export_named(&prepared, &names, &ctx).unwrap());
        let producer_before = (
            result.distributions().iter().map(|d| bits(&d.probabilities)).collect::<Vec<_>>(),
            contrast.estimate.to_bits(),
        );

        // Refresh: same snapshots, new evidence; the proof is kept and the point moves.
        let (_, data2) = evidence(&P_REFRESHED);
        let refreshed = prepared.refresh(data2, &ctx).unwrap();
        let moved = refreshed.estimate(&ctx).unwrap();
        for x in [0u8, 1] {
            assert!(close(risk_of(&moved.distributions()[usize::from(x)]), truth(&P_REFRESHED, x)));
        }
        assert_ne!(moved.contrasts()[0].estimate.to_bits(), contrast.estimate.to_bits());
        let bytes_after = moved.export_named(&refreshed, &names, &ctx).unwrap();
        let producer_after = (
            moved.distributions().iter().map(|d| bits(&d.probabilities)).collect::<Vec<_>>(),
            moved.contrasts()[0].estimate.to_bits(),
        );

        // Counted laws: the point is returned, the unmeasured interval is withheld.
        let counted_prepared = StudyBuilder::mz_transport_empirical(
            graph,
            functional,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            counted(&data, 40_000.0),
            vec![request(true)],
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let counted_result = counted_prepared.estimate(&ctx).unwrap();
        assert!((risk_of(counted_result.distribution()) - truth(&P, 1)).abs() < 1e-2);
        let u = counted_result.uncertainty();
        assert_eq!((u.status.as_str(), u.reason.as_str()), (MZ_WITHHELD, MZ_INTERVAL_NOT_LICENSED));
        assert!(!u.available() && u.mean_intervals.is_empty() && u.seed.is_none());
        let counted_bytes = counted_result.export(&counted_prepared, &ctx).unwrap();
        let counted_producer = bits(&counted_result.distribution().probabilities);
        (
            (bytes_before, producer_before),
            (bytes_after, producer_after),
            (counted_bytes, counted_producer, u.clone()),
        )
    };

    // ---- consumer scope: only bytes.
    for ((bytes, (points, contrast)), p) in [(before, P), (after, P_REFRESHED)] {
        let consumed =
            consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx)
                .unwrap();
        let recomputed: Vec<Vec<u64>> =
            consumed.distributions().iter().map(|d| bits(&d.probabilities)).collect();
        assert_eq!(recomputed, points, "consumer must equal the producer bit for bit");
        assert_eq!(consumed.contrasts()[0].estimate.to_bits(), contrast);
        for x in [0u8, 1] {
            assert!(close(risk_of(&consumed.distributions()[usize::from(x)]), truth(&p, x)));
        }
    }
    let (bytes, points, uncertainty) = counted;
    let consumed =
        consume_mz_transport_artifact(&bytes, MzTransportConsumeLimits::default(), &ctx).unwrap();
    assert_eq!(bits(&consumed.distribution().probabilities), points);
    assert_eq!(consumed.uncertainty(), &uncertainty);
}

// ---------------------------------------------------------------------------
// Story 2 (X2): a finite structural scenario set retaining identified, unidentified and
// unevaluated members without renormalization.
// ---------------------------------------------------------------------------

mod s2 {
    use super::*;

    pub const Z: u32 = 0;
    pub const X: u32 = 1;
    pub const Y: u32 = 2;

    pub fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    fn schema() -> Arc<[ScenarioCoordinate]> {
        [(Z, "z"), (X, "x"), (Y, "y")]
            .into_iter()
            .map(|(id, name)| ScenarioCoordinate {
                variable: v(id),
                name: Arc::from(name),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect()
    }

    pub fn scenario(name: &str, selections: &[u32], weight: Option<f64>) -> TransportScenario {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(Z, X), (Z, Y), (X, Y)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        graph.insert_bidirected(DenseNodeId::from_raw(X), DenseNodeId::from_raw(Y)).unwrap();
        TransportScenario {
            name: Arc::from(name),
            diagram: SelectionDiagram::try_new(
                graph,
                selections.iter().copied().map(v).collect::<Vec<_>>(),
            )
            .unwrap(),
            weight,
            coordinates: schema(),
        }
    }

    /// `standardize` (selection on z), `direct` (none), `outcome_shift` (selection on y).
    pub fn three() -> Vec<TransportScenario> {
        vec![
            scenario("standardize", &[Z], Some(0.3)),
            scenario("direct", &[], Some(0.2)),
            scenario("outcome_shift", &[Y], Some(0.4)),
        ]
    }

    pub fn query() -> ClassicalTransportQuery {
        ClassicalTransportQuery {
            outcomes: Arc::from([v(Y)]),
            treatments: Arc::from([v(X)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        }
    }

    fn regime(id: u32, population: &str, on: &[u32], measured: &[u32]) -> EvidenceRegime {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            on.iter().copied().map(v).collect::<Vec<_>>(),
            [],
            measured.iter().copied().map(v).collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    pub fn catalog() -> EvidenceCatalog {
        EvidenceCatalog::try_new(
            [],
            vec![regime(1, "source", &[X], &[Z, Y]), regime(0, "target", &[], &[Z, X, Y])],
            [],
            None,
        )
        .unwrap()
    }

    fn axis(variable: u32) -> DiscreteAxis {
        DiscreteAxis {
            variable: v(variable),
            values: Arc::from([Value::f64(0.0), Value::f64(1.0)]),
        }
    }

    /// Source `do(x = 1)` over `(z, y)`: `P_s(z = 1) = 0.6`, `P_s(y = 1 | z) = 0.3, 5/6`.
    pub const SOURCE: [f64; 4] = [0.28, 0.12, 0.10, 0.50];
    /// Target observational `(z, x, y)` with `P*(z = 1) = 0.3`.
    pub const TARGET: [f64; 8] = [0.28, 0.14, 0.14, 0.14, 0.06, 0.06, 0.09, 0.09];
    pub const STANDARDIZED: f64 = 0.7 * 0.3 + 0.3 * (0.5 / 0.6);
    pub const DIRECT: f64 = 0.12 + 0.50;

    pub fn laws(source: [f64; 4]) -> ExactTransportData {
        ExactTransportData::try_new(
            vec![
                ExactDiscreteLaw::try_new(
                    "source",
                    RegimeId::from_raw(1),
                    [InterventionAssignment::concrete(v(X), Value::f64(1.0))],
                    [axis(Z), axis(Y)],
                    source,
                    "trial",
                    LawTolerance::default(),
                )
                .unwrap(),
                ExactDiscreteLaw::try_new(
                    "target",
                    RegimeId::from_raw(0),
                    [],
                    [axis(Z), axis(X), axis(Y)],
                    TARGET,
                    "target",
                    LawTolerance::default(),
                )
                .unwrap(),
            ],
            1000,
        )
        .unwrap()
    }

    pub fn request() -> Assignment {
        Assignment::from_pairs([(v(X), Value::f64(1.0))])
    }

    pub fn decided_under(
        operations: usize,
    ) -> antecedent_identify::sid::scenarios::ScenarioSetDecision {
        decide_transport_scenarios(
            &TransportScenarioSet::try_new(three()).unwrap(),
            &query(),
            &catalog(),
            SearchLimits { operations, depth: 256 },
            &ExecutionContext::for_tests(1),
        )
        .unwrap()
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one end-to-end story per test")]
fn story_2_finite_scenario_set_retains_identified_unidentified_and_unevaluated_members() {
    use s2::*;
    let ctx = ExecutionContext::for_tests(202);
    let status = |d: &antecedent_identify::sid::scenarios::ScenarioSetDecision, name: &str| {
        d.decisions.iter().find(|d| &*d.scenario.name == name).unwrap().outcome.status()
    };

    // A shared budget that decides `direct` and `outcome_shift` (canonical name order)
    // and stops before `standardize`.
    let budget = (1..10_000)
        .find(|ops| {
            let d = decided_under(*ops);
            status(&d, "direct") == "identified" && status(&d, "outcome_shift") != "unevaluated"
        })
        .expect("a small graph decides within 10k operations");
    let decided = decided_under(budget);
    assert_eq!(status(&decided, "standardize"), "unevaluated");
    assert_eq!(status(&decided, "outcome_shift"), "structurally_unidentified");

    let (bytes, producer_report) = {
        let set = TransportScenarioSet::try_new(three()).unwrap();
        let prepared = StudyBuilder::transport_scenarios(
            &set,
            query(),
            catalog(),
            SearchLimits { operations: budget, depth: 256 },
            laws(SOURCE),
            request(),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let report = prepared.estimate(&ctx).unwrap();

        // Every member is retained under its own status.
        let get = |name: &str| report.scenarios.iter().find(|s| &*s.name == name).unwrap();
        assert_eq!(report.scenarios.len(), 3);
        assert_eq!(get("direct").status, "identified");
        assert_eq!(get("outcome_shift").status, "structurally_unidentified");
        assert_eq!(get("standardize").status, "unevaluated");
        assert!(
            get("outcome_shift").distribution.is_none()
                && get("standardize").distribution.is_none()
        );
        let direct = get("direct").distribution.as_ref().unwrap().mean(v(Y)).unwrap();
        assert!(close(direct, DIRECT));
        assert_eq!(
            get("standardize").detail.as_deref(),
            Some("scenarios.unevaluated_budget: search.operations")
        );
        let receipt = report.receipt.as_ref().unwrap();
        assert_eq!(receipt.stop, SearchStop::Operations);
        assert_eq!(receipt.unevaluated, ["standardize"]);

        // Declared masses are kept, never renormalized over the survivor.
        let mass = |name: &str| report.masses.iter().find(|m| m.status == name).unwrap();
        assert!(close(mass("identified").mass.unwrap(), 0.2));
        assert!(close(mass("structurally_unidentified").mass.unwrap(), 0.4));
        assert!(close(mass("unevaluated").mass.unwrap(), 0.3));
        assert!(close(report.residual_mass.unwrap(), 0.1));
        let weighted = report.weighted.as_ref().unwrap();
        assert!(close(weighted.identified_mass, 0.2));
        assert!(close(weighted.unaccounted_mass, 0.8), "unevaluated + unidentified + residual");
        let sum = weighted.identified_weighted_sums[0].1;
        assert!(close(sum, 0.2 * DIRECT), "weighted sum {sum} is not renormalized by 0.2");
        assert!((sum - DIRECT).abs() > 0.1, "a renormalized mean would equal the survivor's");
        let (_, lo, hi) = weighted.ranges.as_ref().unwrap()[0];
        assert!(close(lo, 0.2 * DIRECT) && close(hi, 0.2 * DIRECT + 0.8));
        // The envelope ranges over the identified scenarios only and names them.
        assert_eq!(report.envelope.as_ref().unwrap().scenarios.len(), 1);

        // Inference across scenarios is refused, not aggregated.
        assert!(
            prepared
                .aggregate_interval()
                .unwrap_err()
                .to_string()
                .contains("scenario_aggregate_not_licensed")
        );

        // Refresh moves points and keeps every decision.
        let refreshed = prepared.refresh(laws([0.1, 0.3, 0.3, 0.3]), &ctx).unwrap();
        let moved = refreshed.estimate(&ctx).unwrap();
        assert_eq!(
            moved.scenarios.iter().map(|s| s.status).collect::<Vec<_>>(),
            report.scenarios.iter().map(|s| s.status).collect::<Vec<_>>()
        );
        assert!(close(
            moved
                .scenarios
                .iter()
                .find(|s| &*s.name == "direct")
                .unwrap()
                .distribution
                .as_ref()
                .unwrap()
                .mean(v(Y))
                .unwrap(),
            0.6
        ));

        let bytes = prepared.export(&report).unwrap();
        (bytes, report)
    };

    // ---- consumer scope: only bytes.
    let consumed = consume_transport_scenarios_artifact(
        &bytes,
        TransportScenarioConsumeLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(consumed.scenarios.len(), producer_report.scenarios.len());
    for (c, p) in consumed.scenarios.iter().zip(&producer_report.scenarios) {
        assert_eq!((&*c.name, c.status, &c.detail), (&*p.name, p.status, &p.detail));
        let cm = c.distribution.as_ref().map(|d| bits(&d.probabilities));
        let pm = p.distribution.as_ref().map(|d| bits(&d.probabilities));
        assert_eq!(cm, pm, "{}", c.name);
    }
    let (cw, pw) =
        (consumed.weighted.as_ref().unwrap(), producer_report.weighted.as_ref().unwrap());
    assert_eq!(cw.identified_mass.to_bits(), pw.identified_mass.to_bits());
    assert_eq!(cw.unaccounted_mass.to_bits(), pw.unaccounted_mass.to_bits());
    assert_eq!(
        cw.identified_weighted_sums[0].1.to_bits(),
        pw.identified_weighted_sums[0].1.to_bits()
    );
    assert_eq!(
        consumed.residual_mass.unwrap().to_bits(),
        producer_report.residual_mass.unwrap().to_bits()
    );
    assert_eq!(consumed.receipt.as_ref().unwrap().unevaluated, ["standardize"]);
    assert!(close(cw.identified_weighted_sums[0].1, 0.2 * DIRECT));

    // With enough budget the same set is fully evaluated: standardize equals the closed form.
    let full = StudyBuilder::transport_scenarios(
        &TransportScenarioSet::try_new(three()).unwrap(),
        query(),
        catalog(),
        SearchLimits { operations: 100_000, depth: 256 },
        laws(SOURCE),
        request(),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap()
    .estimate(&ctx)
    .unwrap();
    let standardized = full
        .scenarios
        .iter()
        .find(|s| &*s.name == "standardize")
        .unwrap()
        .distribution
        .as_ref()
        .unwrap()
        .mean(v(Y))
        .unwrap();
    assert!(close(standardized, STANDARDIZED));
    assert!(full.receipt.is_none());
}

// ---------------------------------------------------------------------------
// Story 3 (X4): an overlap-supported learned transport estimate for the one chosen
// continuous-outcome cell. Coverage is attested separately.
// ---------------------------------------------------------------------------

mod s3 {
    use super::*;

    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    struct Stream(u64);

    #[expect(clippy::cast_precision_loss, reason = "53-bit uniform draw")]
    impl Stream {
        fn uniform(&mut self) -> f64 {
            self.0 = mix(self.0);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// Trial `z ~ N(0, 1)`, target `z ~ N(shift, 1)`; `y = 1.5 z + a (1 + 0.5 z) + e`.
    /// The target mean contrast is `1 + 0.5 shift`.
    pub fn truth(shift: f64) -> f64 {
        1.0 + 0.5 * shift
    }

    pub fn draw(
        sampling: TrialSampling,
        shift: f64,
        n_trial: usize,
        n_target: usize,
        seed: u64,
    ) -> TrialAipwInput {
        let mut s = Stream(mix(seed) ^ 0xA_E817);
        let total = n_trial + n_target;
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let p = n_trial as f64 / total as f64;
        let participant: Vec<bool> = match sampling {
            TrialSampling::NestedCohort => (0..total).map(|_| s.uniform() < p).collect(),
            TrialSampling::IndependentSamples => (0..total).map(|i| i < n_trial).collect(),
        };
        let (mut z, mut y, mut a) = (Vec::new(), Vec::new(), Vec::new());
        for &in_trial in &participant {
            let cov = if in_trial { s.normal() } else { shift + s.normal() };
            let treated = s.uniform() < 0.5;
            let arm = f64::from(u8::from(treated));
            let outcome = 1.5 * cov + arm * (1.0 + 0.5 * cov) + s.normal();
            z.push(cov);
            y.push(if in_trial { outcome } else { 0.0 });
            a.push(in_trial && treated);
        }
        TrialAipwInput {
            features: vec![0],
            covariates: vec![z],
            outcome: y,
            treatment: a,
            source: participant,
            randomization: vec![0.5; total],
            sampling,
        }
    }

    pub fn graph() -> (SelectionDiagram, TransportQuery, Vec<String>) {
        let diagram = SelectionDiagram::try_new(
            admg(3, &[(0, 2), (1, 2)], &[]),
            vec![VariableId::from_raw(0)],
        )
        .unwrap();
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(2),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(1),
                    GridSpec::Values(Arc::from([0., 1.])),
                ),
            }),
            "trial",
            "target",
            [VariableId::from_raw(1)],
        );
        (diagram, query, ["z", "a", "y"].map(String::from).to_vec())
    }

    pub fn options() -> LearnedContinuousOptions {
        LearnedContinuousOptions {
            outcome: LearnerSpec::Linear(LinearSpec::default()),
            folds: 3,
            ..LearnedContinuousOptions::default()
        }
    }
}

#[test]
fn story_3_learned_continuous_transport_estimate_with_overlap_refusal() {
    use s3::*;
    let ctx = ExecutionContext::for_tests(303);
    for sampling in [TrialSampling::NestedCohort, TrialSampling::IndependentSamples] {
        let (bytes, producer) = {
            let (diagram, query, names) = graph();
            let prepared = StudyBuilder::learned_continuous_transport(
                diagram,
                query,
                draw(sampling, 0.4, 900, 600, 3),
                options(),
                names,
                &ctx,
            )
            .unwrap();
            let result = prepared.estimate(&ctx).unwrap();
            let estimate = result.estimate().clone();
            assert!(
                (estimate.estimate - truth(0.4)).abs() < 0.3,
                "{sampling:?}: {} vs {}",
                estimate.estimate,
                truth(0.4)
            );
            // Point only; overlap of membership and treatment reported apart; provenance recorded.
            assert_eq!(estimate.uncertainty.status, "point_only");
            assert!(estimate.overlap.selection.probability_min > 0.05);
            assert!((estimate.overlap.treatment.probability_min - 0.5).abs() < 1e-12);
            assert_eq!(estimate.provenance.len(), 9);
            assert_eq!(estimate.folds.count, 3);
            let interval = prepared.interval(&ctx).unwrap();
            assert_eq!(interval.wire().version, 2);
            assert!(interval.estimate().uncertainty.available());
            assert!(interval.estimate().standard_error.unwrap() > 0.0);
            let (low, high) = interval.estimate().interval.unwrap();
            assert!(low < interval.estimate().estimate && interval.estimate().estimate < high);
            // Refresh with fresh rows keeps the certificate and moves the point.
            let refreshed = prepared.refresh(draw(sampling, 0.4, 900, 600, 4), &ctx).unwrap();
            assert_eq!(refreshed.identification(), prepared.identification());
            let moved = refreshed.estimate(&ctx).unwrap();
            assert_ne!(moved.identity(), result.identity());
            assert!((moved.estimate().estimate - truth(0.4)).abs() < 0.3);
            (refreshed_export(&moved), moved.estimate().clone())
        };

        // ---- consumer scope: bytes only.
        let consumed =
            consume_learned_continuous_artifact(&bytes, LearnedContinuousConsumeLimits::default())
                .unwrap();
        assert_eq!(consumed.estimate(), &producer);
        assert_eq!(consumed.estimate().estimate.to_bits(), producer.estimate.to_bits());
        assert_eq!(consumed.export().unwrap(), bytes, "consumption re-exports the same bytes");
        assert!((consumed.estimate().estimate - truth(0.4)).abs() < 0.3);
    }

    // Overlap-refusal side: a target the trial cannot represent refuses; it never extrapolates.
    for sampling in [TrialSampling::NestedCohort, TrialSampling::IndependentSamples] {
        let (diagram, query, names) = graph();
        let prepared = StudyBuilder::learned_continuous_transport(
            diagram,
            query,
            draw(sampling, 3.0, 900, 600, 5),
            options(),
            names,
            &ctx,
        )
        .unwrap();
        match prepared.estimate(&ctx) {
            Err(IoError::Refused { code, message }) => {
                assert_eq!(code, "transport_support_failure");
                assert!(message.starts_with("learned_transport.membership_overlap"), "{message}");
            }
            other => panic!("expected an overlap refusal, got {:?}", other.map(|_| ())),
        }
    }
    // Treatment overlap outside the declared bound refuses at preparation.
    let (diagram, query, names) = graph();
    let mut input = draw(TrialSampling::IndependentSamples, 0.4, 300, 200, 6);
    input.randomization[0] = 0.01;
    match StudyBuilder::learned_continuous_transport(diagram, query, input, options(), names, &ctx)
    {
        Err(IoError::Refused { code, message }) => {
            assert_eq!(code, "transport_support_failure");
            assert!(message.starts_with("learned_transport.treatment_overlap"), "{message}");
        }
        other => panic!("expected a treatment-overlap refusal, got {:?}", other.map(|_| ())),
    }
}

fn refreshed_export(result: &antecedent::LearnedContinuousResult) -> Vec<u8> {
    result.export().unwrap()
}

// ---------------------------------------------------------------------------
// Story 4 (X5): a two-step discrete temporal transported intervention with history-support
// refusal.
// ---------------------------------------------------------------------------

mod s4 {
    use super::*;

    pub const B: usize = 0;
    pub const L1: usize = 1;
    pub const A1: usize = 2;
    pub const L2: usize = 3;
    pub const A2: usize = 4;
    pub const Y: usize = 5;
    pub const NAMES: [&str; 6] = ["b", "l1", "a1", "l2", "a2", "y"];

    pub const SOURCE_P: [f64; 8] = [0.3, 0.35, 0.4, 0.55, 0.3, 0.45, 0.65, 0.4];
    pub const TARGET_P: [f64; 8] = [0.75, 0.35, 0.4, 0.55, 0.55, 0.45, 0.65, 0.4];

    fn mechanisms(l2: Mech) -> Vec<Mech> {
        vec![
            Box::new(|_, e| e[0]),
            Box::new(|v, e| v[B] ^ e[1]),
            Box::new(|v, e| e[2] ^ bit(v[L1] == 1 && e[3] == 1)),
            l2,
            Box::new(|v, e| bit(v[L2] == 1 && e[6] == 1) ^ e[5]),
            Box::new(|v, e| {
                bit(v[A1] == 1 && e[2] == 0)
                    ^ bit(v[A2] == 1 && v[L2] == 1)
                    ^ bit(v[B] == 1 && e[7] == 1)
                    ^ bit(e[5] == 1 && v[L1] == 1)
            }),
        ]
    }

    pub fn scm(population: &str, p: &[f64]) -> Scm {
        let l2: Mech = if population == "source" {
            Box::new(|v, e| bit(v[A1] == 1) ^ e[4])
        } else {
            Box::new(|v, e| bit(v[A1] == 1 && v[B] == 0) ^ e[4])
        };
        Scm { p: p.to_vec(), f: mechanisms(l2) }
    }

    pub fn spec() -> TemporalSequenceSpec {
        let directed = [
            (B, L1),
            (L1, A1),
            (B, L2),
            (A1, L2),
            (L2, A2),
            (A1, A2),
            (B, Y),
            (L1, Y),
            (A1, Y),
            (L2, Y),
            (A2, Y),
        ]; // fmt: skip
        let diagram = SelectionDiagram::try_new(
            admg(6, &directed, &[(A1, Y), (A2, Y)]),
            vec![vid(B), vid(L2)],
        )
        .unwrap();
        let slots = TemporalSlots {
            baseline: vec![vid(B)],
            covariates: [vec![vid(L1)], vec![vid(L2)]],
            actions: [vid(A1), vid(A2)],
            outcome: vid(Y),
        };
        let coordinates: Vec<ScenarioCoordinate> = NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| ScenarioCoordinate {
                variable: vid(i),
                name: Arc::from(*name),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect();
        TemporalSequenceSpec::try_new(2, slots, diagram, coordinates).unwrap()
    }

    fn regime(id: u32, population: &str, on: &[usize], measured: &[usize]) -> EvidenceRegime {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            on.iter().copied().map(vid).collect::<Vec<_>>(),
            [],
            measured.iter().copied().map(vid).collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    /// Target observational law over every coordinate; the source's history experiments.
    pub fn catalog() -> EvidenceCatalog {
        let regimes = vec![
            regime(0, "target", &[], &[B, L1, A1, L2, A2, Y]),
            regime(1, "source", &[B, L1, A1, L2, A2], &[Y]),
        ];
        let bindings = regimes
            .iter()
            .map(|r| binding(r.id, &format!("{}-{}", r.population, r.id.raw())))
            .collect::<Vec<_>>();
        EvidenceCatalog::try_new([], regimes, bindings, None).unwrap()
    }

    /// Every law; the source's experiments for all 32 histories except the `(b, l1, l2)`
    /// triples in `skip`.
    pub fn laws(source: &Scm, target: &Scm, skip: &[(u8, u8, u8)]) -> ExactTransportData {
        let mut laws = vec![
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                (0..6).map(num_axis).collect::<Vec<_>>(),
                target.law(&[], &[B, L1, A1, L2, A2, Y]),
                "target-0",
                LawTolerance::default(),
            )
            .unwrap(),
        ];
        let order = [B, L1, A1, L2, A2];
        for h in 0..32u8 {
            let history = [(h >> 4) & 1, (h >> 3) & 1, (h >> 2) & 1, (h >> 1) & 1, h & 1];
            if skip.contains(&(history[0], history[1], history[3])) {
                continue;
            }
            let do_: Vec<(usize, u8)> = order.iter().copied().zip(history).collect();
            laws.push(
                ExactDiscreteLaw::try_new(
                    "source",
                    RegimeId::from_raw(1),
                    do_.iter()
                        .map(|(var, l)| {
                            InterventionAssignment::concrete(vid(*var), Value::f64(f64::from(*l)))
                        })
                        .collect::<Vec<_>>(),
                    [num_axis(Y)],
                    source.law(&do_, &[Y]),
                    "source-1",
                    LawTolerance::default(),
                )
                .unwrap(),
            );
        }
        ExactTransportData::try_new(laws, 4096).unwrap()
    }

    pub fn sequence(a1: f64, a2: f64) -> Vec<Value> {
        vec![Value::f64(a1), Value::f64(a2)]
    }

    pub fn prepare(
        seq: &[Value],
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<antecedent::PreparedTemporalTransport, IoError> {
        StudyBuilder::temporal_transport_sequence(
            &spec(),
            seq,
            "source",
            "target",
            catalog(),
            SearchLimits { operations: 100_000, depth: 256 },
            data,
            ExactEvaluationLimits::default(),
            ctx,
        )
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one end-to-end story per test")]
fn story_4_two_step_temporal_transport_with_history_support_refusal() {
    use s4::*;
    let ctx = ExecutionContext::for_tests(404);
    let truth = |p: &[f64], seq: [u8; 2]| scm("target", p).risk(&[(A1, seq[0]), (A2, seq[1])], Y);
    let mut moved_source = SOURCE_P;
    moved_source[7] = 0.55;
    let mut moved_target = TARGET_P;
    moved_target[7] = 0.55;

    let (artifacts, producers) = {
        let mut artifacts = Vec::new();
        let mut producers = Vec::new();
        for seq in [[1u8, 0], [0, 1]] {
            let prepared = prepare(
                &sequence(f64::from(seq[0]), f64::from(seq[1])),
                laws(&scm("source", &SOURCE_P), &scm("target", &TARGET_P), &[]),
                &ctx,
            )
            .unwrap();
            let report = prepared.estimate(&ctx).unwrap();
            assert!(close(report.mean, truth(&TARGET_P, seq)));
            assert_eq!((report.horizon, report.inference_claim), (2, "point_only"));
            // The mechanism change matters: the source's own answer differs.
            let source_answer = scm("source", &SOURCE_P).risk(&[(A1, seq[0]), (A2, seq[1])], Y);
            assert!(
                (source_answer - report.mean).abs() > 1e-3,
                "{source_answer} vs {}",
                report.mean
            );
            assert!(report.support.rows.iter().all(|r| r.status == "supported"));
            assert_eq!(report.time_varying_confounders, vec![vid(L2)]);
            let exported = prepared.export(&report).unwrap();
            producers.push((
                seq,
                TARGET_P,
                report.mean.to_bits(),
                bits(&report.distribution.probabilities),
            ));
            artifacts.push(exported);

            // Refresh under the same proof: the point follows the new evidence.
            let refreshed = prepared
                .refresh(
                    laws(&scm("source", &moved_source), &scm("target", &moved_target), &[]),
                    &ctx,
                )
                .unwrap();
            let after = refreshed.estimate(&ctx).unwrap();
            assert!(close(after.mean, truth(&moved_target, seq)));
            assert!((after.mean - report.mean).abs() > 1e-4);
            producers.push((
                seq,
                moved_target,
                after.mean.to_bits(),
                bits(&after.distribution.probabilities),
            ));
            artifacts.push(refreshed.export(&after).unwrap());
        }

        // History-support refusal: dropping a source history experiment refuses, typed,
        // and the refusal names exactly the offending history (b, l1, l2).
        let (source, target) = (scm("source", &SOURCE_P), scm("target", &TARGET_P));
        let mut refusals = Vec::new();
        for skip in [(1u8, 0u8, 1u8), (0, 1, 0)] {
            match prepare(&sequence(1.0, 1.0), laws(&source, &target, &[skip]), &ctx) {
                Err(IoError::Refused { code, message }) => {
                    assert_eq!(code, "transport_support_failure");
                    assert!(
                        message.starts_with("temporal_transport.history_outside_support"),
                        "{message}"
                    );
                    assert!(
                        message.contains(&format!("b={}", skip.0))
                            && message.contains(&format!("l1={}", skip.1))
                            && message.contains(&format!("l2={}", skip.2)),
                        "the refusal is local to history {skip:?}: {message}"
                    );
                    refusals.push(message);
                }
                Ok(_) => panic!("history {skip:?} is reached by the target and has no source law"),
                other => panic!("expected a typed refusal, got {:?}", other.map(|_| ())),
            }
        }
        assert_ne!(refusals[0], refusals[1], "each refusal names its own history");
        // An interval request is a typed refusal, not a silent point.
        let prepared = prepare(&sequence(1.0, 0.0), laws(&source, &target, &[]), &ctx).unwrap();
        match prepared.interval() {
            Err(IoError::Refused { code, message }) => {
                assert_eq!(code, "estimator_inference_mismatch");
                assert!(message.starts_with("temporal_transport.interval_requested"), "{message}");
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
        // A third step never prepares.
        let mut three = sequence(1.0, 0.0);
        three.push(Value::f64(1.0));
        match prepare(&three, laws(&source, &target, &[]), &ctx) {
            Err(IoError::Refused { code, message }) => {
                assert_eq!(code, "route_not_supported");
                assert!(message.starts_with("temporal_transport.horizon"), "{message}");
            }
            other => panic!("expected a horizon refusal, got {:?}", other.map(|_| ())),
        }
        (artifacts, producers)
    };

    // ---- consumer scope: bytes only.
    assert_eq!(artifacts.len(), producers.len());
    for (bytes, (seq, p, mean_bits, probability_bits)) in artifacts.iter().zip(&producers) {
        let consumed = consume_temporal_transport_artifact(
            bytes,
            TemporalTransportConsumeLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(consumed.mean.to_bits(), *mean_bits, "consumer equals producer bit for bit");
        assert_eq!(&bits(&consumed.distribution.probabilities), probability_bits);
        assert!(close(consumed.mean, truth(p, *seq)));
        assert_eq!(consumed.sequence.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// Story 5 (X8): a second fixed-DAG cross-world query with shared-abduction semantics and
// typed refusal.
// ---------------------------------------------------------------------------

mod s5 {
    use super::*;

    pub const X: u32 = 0;
    pub const M: u32 = 1;
    pub const Y: u32 = 2;
    pub const EDGES: [(u32, u32); 3] = [(X, M), (X, Y), (M, Y)];

    pub fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    pub fn dag(n: u32, edges: &[(u32, u32)]) -> Dag {
        let mut g = Dag::with_variables(n);
        for &(a, b) in edges {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g
    }

    pub fn mediation() -> Dag {
        dag(3, &EDGES)
    }

    /// `M = -0.6 X + U`, `Y = 2.2 X + 3 M`, with `U` orthogonal to the constant and `X`, so a
    /// least-squares fit recovers every coefficient exactly. Returned as data columns.
    #[expect(clippy::cast_precision_loss, reason = "small row counts")]
    pub fn linear(n: usize) -> TabularData {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.53).sin()).collect();
        let raw: Vec<f64> = (0..n).map(|i| (i as f64 * 1.07).cos()).collect();
        let nn = n as f64;
        let (mn, mx) = (raw.iter().sum::<f64>() / nn, x.iter().sum::<f64>() / nn);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxn: f64 = x.iter().zip(&raw).map(|(a, b)| (a - mx) * (b - mn)).sum();
        let u: Vec<f64> =
            x.iter().zip(&raw).map(|(a, b)| (b - mn) - sxn / sxx * (a - mx)).collect();
        let m: Vec<f64> = x.iter().zip(&u).map(|(x, u)| -0.6 * x + u).collect();
        let y: Vec<f64> = x.iter().zip(&m).map(|(x, m)| 2.2 * x + 3.0 * m).collect();
        TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap()
    }

    /// `M = 0.7 X + U`, `Y = 1.1 X + 0.4 M + 0.8 X M^2`: convex in the mediator and interacting
    /// with the treatment. Returns the data and the exact per-unit mediator disturbance.
    #[expect(clippy::cast_precision_loss, reason = "small row counts")]
    pub fn non_separable(n: usize) -> (TabularData, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.41).sin() * 3.0).collect();
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.83).cos()).collect();
        let m: Vec<f64> = x.iter().zip(&u).map(|(x, u)| 0.7 * x + u).collect();
        let y: Vec<f64> = x.iter().zip(&m).map(|(x, m)| outcome(*x, *m)).collect();
        (
            TabularData::from_f64_columns([
                ("x", x.as_slice()),
                ("m", m.as_slice()),
                ("y", y.as_slice()),
            ])
            .unwrap(),
            u,
        )
    }

    pub fn outcome(x: f64, m: f64) -> f64 {
        1.1 * x + 0.4 * m + 0.8 * x * m * m
    }

    pub fn edge_query(control: f64, active: f64, intervened: &[(u32, u32)]) -> CrossWorldQuery {
        let all = EDGES.map(|(a, b)| (v(a), v(b)));
        let chosen: Vec<_> = intervened.iter().map(|&(a, b)| (v(a), v(b))).collect();
        CrossWorldQuery::path_specific(v(X), v(Y), control, active, &all, &chosen).unwrap()
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one end-to-end story per test")]
fn story_5_second_fixed_dag_cross_world_query_with_shared_abduction_and_typed_refusal() {
    use s5::*;
    let ctx = ExecutionContext::for_tests(505);
    let (control, active) = (-1.0, 2.0);
    let delta = active - control;

    let (linear_bytes, nonsep_bytes, producer) = {
        // Exact linear truth for the indirect edge set {X->M, M->Y}: 3 * (-0.6) * delta.
        let data = linear(240);
        let indirect = edge_query(control, active, &[(X, M), (M, Y)]);
        let effect = evaluate_cross_world_effect(
            mediation(),
            &data,
            &indirect,
            CrossWorldOptions::default(),
            &ctx,
        )
        .unwrap();
        assert!((effect.point - 3.0 * -0.6 * delta).abs() < 1e-9, "{}", effect.point);
        let linear_bytes = effect.export_artifact().unwrap();
        assert_eq!(linear_bytes, effect.export_artifact().unwrap());

        // Shared abduction: the answer reads each unit's own abduced disturbance.
        let (data, u) = non_separable(600);
        let options = CrossWorldOptions {
            mechanism: NestedOutcomeMechanism::NonSeparableBasis,
            interval_requested: false,
        };
        let direct = CrossWorldQuery::natural_direct(v(X), v(M), v(Y), control, active).unwrap();
        let shared =
            evaluate_cross_world_effect(mediation(), &data, &direct, options, &ctx).unwrap();
        #[expect(clippy::cast_precision_loss, reason = "small row counts")]
        let n = u.len() as f64;
        let m0: Vec<f64> = u.iter().map(|u| 0.7 * control + u).collect();
        let per_unit =
            m0.iter().map(|m| outcome(active, *m) - outcome(control, *m)).sum::<f64>() / n;
        let mbar = m0.iter().sum::<f64>() / n;
        let plug_in = outcome(active, mbar) - outcome(control, mbar);
        assert!((per_unit - plug_in).abs() > 1.0, "the fixture must separate the semantics");
        assert!(
            (shared.point - per_unit).abs() < 0.25,
            "{} vs per-unit truth {per_unit}",
            shared.point
        );
        assert!((shared.point - plug_in).abs() > 1.0);
        let separable = evaluate_cross_world_effect(
            mediation(),
            &data,
            &direct,
            CrossWorldOptions::default(),
            &ctx,
        )
        .unwrap();
        assert!(
            (separable.point - per_unit).abs() > 1.0,
            "a separable fit cannot read the interaction"
        );
        let nonsep_bytes = shared.export_artifact().unwrap();

        // Typed refusals: a recanting witness, an interval request, a latent-confounded graph.
        let recanting_edges = [(0, 1), (1, 2), (1, 3), (2, 3)];
        let graph = dag(4, &recanting_edges);
        let edges = recanting_edges.map(|(a, b)| (v(a), v(b)));
        let recanting = CrossWorldQuery::path_specific(
            v(0),
            v(3),
            0.0,
            1.0,
            &edges,
            &[(v(0), v(1)), (v(1), v(3))],
        )
        .unwrap();
        let text = check(&graph, &recanting).unwrap_err().to_string();
        assert!(text.contains("reason=cross_world_not_identified"), "{text}");
        assert!(text.contains("cross_world.recanting_witness"), "{text}");
        let text = evaluate_cross_world_effect(
            mediation(),
            &linear(60),
            &direct,
            CrossWorldOptions { interval_requested: true, ..CrossWorldOptions::default() },
            &ctx,
        )
        .unwrap_err()
        .to_string();
        assert!(text.contains("reason=estimator_inference_mismatch"), "{text}");
        assert!(text.contains("cross_world.interval_requested"), "{text}");
        let text = evaluate_cross_world_effect(
            admg(3, &EDGES.map(|(a, b)| (a as usize, b as usize)), &[(1, 2)]),
            &linear(60),
            &direct,
            CrossWorldOptions::default(),
            &ctx,
        )
        .unwrap_err()
        .to_string();
        assert!(text.contains("reason=cell_not_licensed"), "{text}");
        assert!(text.contains("cross_world.graph_outside_contract"), "{text}");
        // The witness the check returns is what the artifact carries.
        let witness = check(&mediation(), &indirect).unwrap();
        assert_eq!(witness.intervened_edges, vec![(X, M), (M, Y)]);
        assert!(witness.assumptions.iter().any(|a| a == "markovian_no_latent_confounding"));
        (linear_bytes, nonsep_bytes, (effect.point.to_bits(), shared.point.to_bits(), per_unit))
    };

    // ---- consumer scope: bytes only.
    let consumed = consume_cross_world_artifact(&linear_bytes, &ctx).unwrap();
    assert_eq!(consumed.point.to_bits(), producer.0);
    assert_eq!(consumed.query, edge_query(control, active, &[(X, M), (M, Y)]));
    assert!((consumed.point - 3.0 * -0.6 * delta).abs() < 1e-9);
    let consumed = consume_cross_world_artifact(&nonsep_bytes, &ctx).unwrap();
    assert_eq!(consumed.point.to_bits(), producer.1, "shared-abduction point replays bit for bit");
    assert!((consumed.point - producer.2).abs() < 0.25);
    // A single flipped byte never returns a different answer.
    let mut tampered = nonsep_bytes.clone();
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0x01;
    if let Ok(replay) = consume_cross_world_artifact(&tampered, &ctx) {
        assert_eq!(replay.point.to_bits(), producer.1);
    }
}

// ---------------------------------------------------------------------------
// Story 6 (X9): a mixed-source proof found by bounded search, plus an incomplete-search
// case that stays unresolved.
// ---------------------------------------------------------------------------

mod s6 {
    use super::*;

    pub const X: usize = 0;
    pub const Z: usize = 1;
    pub const Y: usize = 2;

    /// Confounded front door `X -> Z -> Y`, `X <-> Y` through the latent bit `e[5]`.
    pub fn frontdoor(p: &[f64]) -> Scm {
        Scm {
            p: p.to_vec(),
            f: vec![
                Box::new(|_, e| e[5] ^ (e[0] & e[1])),
                Box::new(|v, e| if v[X] == 1 { e[1] } else { e[2] }),
                Box::new(|v, e| {
                    (if v[Z] == 1 { e[3] } else { e[4] }) ^ (e[5] & u8::from(v[Z] == 0))
                }),
            ],
        }
    }

    // e[5] is the latent x <-> y bit; it must stay 0.5 so that x is independent of e[1].
    pub const P: [f64; 6] = [0.55, 0.75, 0.25, 0.65, 0.15, 0.5];
    pub const P_REFRESHED: [f64; 6] = [0.55, 0.6, 0.25, 0.65, 0.15, 0.5];

    pub fn graph() -> Admg {
        admg(3, &[(X, Z), (Z, Y)], &[(X, Y)])
    }

    pub fn query() -> MixedSourceQuery {
        MixedSourceQuery {
            outcomes: Arc::from([vid(Y)]),
            treatments: Arc::from([vid(X)]),
            target: Arc::from("target"),
            sources: Arc::from([]),
        }
    }

    /// An observational study of `{x, z}` and a trial of `do(z)` measuring `{x, y}`.
    pub fn evidence(p: &[f64]) -> (EvidenceCatalog, ExactTransportData) {
        let studies: [(&str, Vec<usize>, Vec<usize>); 2] =
            [("observational", vec![], vec![X, Z]), ("trial", vec![Z], vec![X, Y])];
        let coordinates: Vec<_> = (0..3)
            .map(|i| VariableCoordinate {
                variable: vid(i),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect();
        let environment = Environment::try_new("target", coordinates, []).unwrap();
        let scm = frontdoor(p);
        let (mut regimes, mut bindings, mut laws) = (Vec::new(), Vec::new(), Vec::new());
        for (k, (name, on, measured)) in studies.iter().enumerate() {
            let id = RegimeId::from_raw(u32::try_from(k + 1).unwrap());
            let mut regime = EvidenceRegime::try_new(
                id,
                if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
                EvidenceKind::Available,
                on.iter().map(|v| vid(*v)).collect::<Vec<_>>(),
                [],
                measured.iter().map(|v| vid(*v)).collect::<Vec<_>>(),
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap();
            regime.study = Some(Arc::from(*name));
            regimes.push(regime);
            let snapshot = format!("{name}-{k}");
            bindings.push(binding(id, &snapshot));
            for levels in 0..(1usize << on.len()) {
                let world: Vec<(usize, u8)> = on
                    .iter()
                    .enumerate()
                    .map(|(b, v)| (*v, u8::from((levels >> b) & 1 == 1)))
                    .collect();
                laws.push(
                    ExactDiscreteLaw::try_new(
                        "target",
                        id,
                        world
                            .iter()
                            .map(|(v, l)| {
                                InterventionAssignment::concrete(vid(*v), Value::Bool(*l == 1))
                            })
                            .collect::<Vec<_>>(),
                        measured.iter().map(|i| bool_axis(*i)).collect::<Vec<_>>(),
                        scm.law(&world, measured),
                        snapshot.clone(),
                        LawTolerance::default(),
                    )
                    .unwrap(),
                );
            }
        }
        (
            EvidenceCatalog::try_new([environment], regimes, bindings, None).unwrap(),
            ExactTransportData::try_new(laws, 4096).unwrap(),
        )
    }

    pub fn request(level: bool) -> Assignment {
        Assignment::from_pairs([(vid(X), Value::Bool(level))])
    }
}

#[test]
#[expect(clippy::too_many_lines, reason = "one end-to-end story per test")]
fn story_6_mixed_source_proof_by_bounded_search_and_an_incomplete_search_stays_unresolved() {
    use s6::*;
    let ctx = ExecutionContext::for_tests(606);
    let truth = |p: &[f64], x: u8| frontdoor(p).risk(&[(X, x)], Y);

    let (bytes_before, bytes_after, producer) = {
        let (catalog, data) = evidence(&P);
        // No single study identifies it: the search reports it as not certified.
        let single = |name: &str| {
            let (full, _) = evidence(&P);
            let regimes: Vec<_> =
                full.regimes.iter().filter(|r| r.study.as_deref() == Some(name)).cloned().collect();
            let bindings: Vec<_> = full
                .bindings
                .iter()
                .filter(|b| regimes.iter().any(|r| r.id == b.regime))
                .cloned()
                .collect();
            EvidenceCatalog::try_new(full.environments.to_vec(), regimes, bindings, None).unwrap()
        };
        for alone in ["observational", "trial"] {
            let decision = decide_mixed_source(
                &graph(),
                &query(),
                &single(alone),
                MIXED_SOURCE_DEFAULT_LIMITS,
                &ctx,
            )
            .unwrap();
            assert!(matches!(decision, MixedSourceDecision::NotCertified(_)), "{alone}");
        }
        // Together the bounded search finds a checked proof from the supplied distributions.
        let MixedSourceDecision::Identified { derivation, cited, alternatives } =
            decide_mixed_source(&graph(), &query(), &catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
                .unwrap()
        else {
            panic!("the observational study and the surrogate trial together identify");
        };
        assert_eq!(cited.len(), 2);
        assert!(alternatives.is_empty(), "alternative derivations appear only when found");
        let leaves = derivation.leaves();
        assert_eq!(
            leaves.iter().map(|(_, l)| l.study.as_ref()).collect::<Vec<_>>().len(),
            2,
            "every leaf names its source study"
        );
        let functional = bind_mixed_source_catalog(&graph(), &derivation, &catalog).unwrap();
        let prepared = StudyBuilder::mixed_source(
            graph(),
            functional,
            MIXED_SOURCE_DEFAULT_LIMITS,
            data,
            vec![request(false), request(true)],
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let result = prepared.estimate(&ctx).unwrap();
        for x in [0u8, 1] {
            assert!(close(risk_of(&result.distributions()[usize::from(x)]), truth(&P, x)));
        }
        // The confounded observational conditional is not the answer.
        let joint = frontdoor(&P).law(&[], &[X, Y]);
        assert!((joint[3] / (joint[2] + joint[3]) - truth(&P, 1)).abs() > 1e-2);
        let names = ["x", "z", "y"].map(String::from);
        let before = result.export_named(&prepared, &names).unwrap();
        let producer_before =
            result.distributions().iter().map(|d| bits(&d.probabilities)).collect::<Vec<_>>();

        // Refresh under the same proof.
        let (_, data2) = evidence(&P_REFRESHED);
        let refreshed = prepared.refresh(data2, &ctx).unwrap();
        let moved = refreshed.estimate(&ctx).unwrap();
        for x in [0u8, 1] {
            assert!(close(risk_of(&moved.distributions()[usize::from(x)]), truth(&P_REFRESHED, x)));
        }
        let after = moved.export_named(&refreshed, &names).unwrap();
        let producer_after =
            moved.distributions().iter().map(|d| bits(&d.probabilities)).collect::<Vec<_>>();

        // ---- incomplete side: the bow arc is not identifiable and the search says only
        // "not certified", with what it explored; it never claims non-identification.
        let bow = admg(2, &[(0, 1)], &[(0, 1)]);
        let bow_query = MixedSourceQuery {
            outcomes: Arc::from([vid(1)]),
            treatments: Arc::from([vid(0)]),
            target: Arc::from("target"),
            sources: Arc::from([]),
        };
        let bow_catalog = {
            let mut regime = EvidenceRegime::try_new(
                RegimeId::from_raw(1),
                RegimeKind::Observational,
                EvidenceKind::Available,
                Vec::<VariableId>::new(),
                [],
                [vid(0), vid(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap();
            regime.study = Some(Arc::from("study"));
            EvidenceCatalog::try_new(
                [],
                vec![regime],
                vec![binding(RegimeId::from_raw(1), "s-1")],
                None,
            )
            .unwrap()
        };
        let stuck =
            decide_mixed_source(&bow, &bow_query, &bow_catalog, MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
                .unwrap();
        assert_eq!(stuck.reason_code(), Some("transport_not_certified"));
        assert_eq!(stuck.detail_code(), Some("mixed_search.not_certified"));
        let MixedSourceDecision::NotCertified(inspection) = stuck else { panic!("not certified") };
        assert!(inspection.operations > 0 && inspection.generations > 0);
        assert!(!inspection.frontier.is_empty(), "the receipt lists what was explored");
        // A budget stop is a resource outcome with a receipt: nothing about identification.
        let (catalog, _) = evidence(&P);
        let stopped = decide_mixed_source(
            &graph(),
            &query(),
            &catalog,
            SearchLimits { operations: 30, depth: 16 },
            &ctx,
        )
        .unwrap();
        assert_eq!(stopped.reason_code(), Some("transport_budget_cancel"));
        assert_eq!(stopped.detail_code(), Some("mixed_search.budget"));
        let MixedSourceDecision::Exhausted(receipt) = stopped else { panic!("exhausted") };
        assert_eq!(receipt.stop, SearchStop::Operations);
        assert_eq!(receipt.operations_consumed, Some(30));
        assert!(receipt.unevaluated.iter().any(|r| r == "stage:rule_search"), "{receipt:?}");
        (before, after, (producer_before, producer_after))
    };

    // ---- consumer scope: bytes only.
    for (bytes, points, p) in
        [(&bytes_before, &producer.0, P), (&bytes_after, &producer.1, P_REFRESHED)]
    {
        let consumed =
            consume_mixed_source_artifact(bytes, MixedSourceConsumeLimits::default(), &ctx)
                .unwrap();
        let recomputed: Vec<Vec<u64>> =
            consumed.distributions().iter().map(|d| bits(&d.probabilities)).collect();
        assert_eq!(&recomputed, points, "consumer must equal the producer bit for bit");
        for x in [0u8, 1] {
            assert!(close(risk_of(&consumed.distributions()[usize::from(x)]), truth(&p, x)));
        }
    }
}
