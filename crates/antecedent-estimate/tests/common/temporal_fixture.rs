//! The known temporal SCM shared by the two-step sequence tests (2.2A, X5).
//!
//! Coordinates in time order: `b` (baseline), `l1` (step-1 covariate), `a1`,
//! `l2` (step-2 covariate, affected by `a1`, a time-varying confounder of `a2`
//! and `y`), `a2`, `y`. Two latent bits confound `a1` with `y` and `a2` with `y`.
//! The source and target differ at the initial state (`b`, slice 0) and at the
//! step-2 covariate mechanism (`l2`, slice 2); `y` is invariant. Every law is
//! enumerated from each population's structural model, so a formula checked
//! against these laws is checked against the model's own interventional truth.
//!
//! The parameters are chosen so the answers are far apart: `[0,1]` and `[1,0]`
//! differ by about 0.37 in the target, and every sequence's target truth differs
//! from the source's own and from the per-step product by about 0.1 or more (the
//! target-only sequential g-formula, a strong comparator, is off by about 0.1 at
//! its worst sequence).
//!
//! The source supplies its history-indexed outcome experiments: the law of `y`
//! under `do(b, l1, a1, l2, a2)` for every complete history and both actions.
//! The target supplies its observational law over all six coordinates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code)]

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableDomain, VariableId,
};
use antecedent_expr::{
    DiscreteAxis, ExactDiscreteLaw, ExactTransportData, InterventionAssignment, LawTolerance,
};
use antecedent_identify::sid::{
    scenarios::ScenarioCoordinate,
    temporal_sequence::{TemporalSequenceSpec, TemporalSlots},
};

use super::z_scm::{Mechanism, Scm, bit, diagram, vid};

pub const B: usize = 0;
pub const L1: usize = 1;
pub const A1: usize = 2;
pub const L2: usize = 3;
pub const A2: usize = 4;
pub const Y: usize = 5;
pub const NAMES: [&str; 6] = ["b", "l1", "a1", "l2", "a2", "y"];

pub fn v(i: usize) -> VariableId {
    vid(i)
}

pub fn slots() -> TemporalSlots {
    TemporalSlots {
        baseline: vec![v(B)],
        covariates: [vec![v(L1)], vec![v(L2)]],
        actions: [v(A1), v(A2)],
        outcome: v(Y),
    }
}

pub fn coordinates() -> Vec<ScenarioCoordinate> {
    NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| ScenarioCoordinate {
            variable: v(i),
            name: Arc::from(*name),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

const DIRECTED: [(usize, usize); 12] = [
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
    (L1, A2),
];

/// The unrolled selection diagram with `selections` (time-indexed mechanism
/// differences) and the two latent confounders `a1 <-> y`, `a2 <-> y`.
pub fn unrolled(selections: &[usize], directed: &[(usize, usize)]) -> TemporalSequenceSpec {
    let diagram = diagram(6, directed, &[(A1, Y), (A2, Y)], selections);
    TemporalSequenceSpec::try_new(2, slots(), diagram, coordinates()).unwrap()
}

/// The frozen fixture: mechanism differences at the initial state and at step 2.
pub fn spec() -> TemporalSequenceSpec {
    unrolled(&[B, L2], &DIRECTED[..11])
}

/// Exogenous bits: e0 initial state, e1 `l1`, e2 latent `a1 <-> y`, e3 `a1`,
/// e4 `l2`, e5 latent `a2 <-> y`, e6 `a2`, e7 `y`.
fn mechanisms(l2: Mechanism) -> Vec<Mechanism> {
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

pub fn source_scm() -> Scm {
    Scm {
        n: 6,
        exo_p: vec![0.35, 0.5, 0.9, 0.1, 0.25, 0.1, 0.7, 0.3],
        f: mechanisms(Box::new(|v, e| bit(v[A1] == 1) ^ e[4])),
    }
}

pub fn target_scm() -> Scm {
    Scm {
        n: 6,
        exo_p: vec![0.8, 0.5, 0.9, 0.1, 0.9, 0.1, 0.7, 0.3],
        f: mechanisms(Box::new(|v, e| bit(v[A1] == 1 && v[B] == 0) ^ e[4])),
    }
}

fn axis(variable: usize) -> DiscreteAxis {
    DiscreteAxis { variable: v(variable), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

fn regime(
    id: u32,
    population: &str,
    interventions: &[usize],
    measured: &[usize],
) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        interventions.iter().copied().map(v).collect::<Vec<_>>(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

/// The catalog: the target's observational law over all six coordinates
/// (regime 0) and the source's history-indexed outcome experiments (regime 1).
pub fn catalog() -> EvidenceCatalog {
    let regimes = vec![
        regime(0, "target", &[], &[B, L1, A1, L2, A2, Y]),
        regime(1, "source", &[B, L1, A1, L2, A2], &[Y]),
    ];
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("{}-{}", r.population, r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new([], regimes, bindings, None).unwrap()
}

/// Enumeration sums can exceed one by a rounding error; a law's cells are in [0, 1].
fn clamp(probabilities: Vec<f64>) -> Vec<f64> {
    probabilities.into_iter().map(|p| p.clamp(0.0, 1.0)).collect()
}

/// The target observational law of `scm`.
pub fn target_law(scm: &Scm) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        (0..6).map(axis).collect::<Vec<_>>(),
        clamp(scm.law(&[], &[B, L1, A1, L2, A2, Y])),
        "target-0",
        LawTolerance::default(),
    )
    .unwrap()
}

/// The source's outcome law under `do(b, l1, a1, l2, a2)`.
pub fn history_law(scm: &Scm, history: [u8; 5]) -> ExactDiscreteLaw {
    let order = [B, L1, A1, L2, A2];
    let interventions = order
        .iter()
        .zip(history)
        .map(|(var, level)| InterventionAssignment::concrete(v(*var), Value::f64(f64::from(level))))
        .collect::<Vec<_>>();
    let do_ = order.iter().copied().zip(history).collect::<Vec<_>>();
    ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        interventions,
        [axis(Y)],
        clamp(scm.law(&do_, &[Y])),
        "source-1",
        LawTolerance::default(),
    )
    .unwrap()
}

/// Every supplied law: the target's, and the source's for all 32 histories
/// except those `skip` names (`(b, l1, l2)` with any actions).
pub fn laws(source: &Scm, target: &Scm, skip: &[(u8, u8, u8)]) -> ExactTransportData {
    let mut laws = vec![target_law(target)];
    for h in 0..32u8 {
        let history = [(h >> 4) & 1, (h >> 3) & 1, (h >> 2) & 1, (h >> 1) & 1, h & 1];
        if skip.contains(&(history[0], history[1], history[3])) {
            continue;
        }
        laws.push(history_law(source, history));
    }
    ExactTransportData::try_new(laws, 4096).unwrap()
}

/// `P(y = 1 | do(a1, a2))` in the population of `scm`, by exact enumeration.
pub fn truth(scm: &Scm, sequence: [u8; 2]) -> f64 {
    scm.risk(&[(A1, sequence[0]), (A2, sequence[1])], Y)
}

/// The sequence as request values.
pub fn sequence(a1: f64, a2: f64) -> Vec<Value> {
    vec![Value::f64(a1), Value::f64(a2)]
}

/// A source law of the fixture's regime 1 that does not intervene at all: the
/// source's observational law over every coordinate.
pub fn observational_source_law(scm: &Scm) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [],
        (0..6).map(axis).collect::<Vec<_>>(),
        clamp(scm.law(&[], &[B, L1, A1, L2, A2, Y])),
        "source-1",
        LawTolerance::default(),
    )
    .unwrap()
}

/// A source law of regime 1 that intervenes only on the two actions and measures
/// everything else (`b`, `l1`, `l2`, `y`).
pub fn action_law(scm: &Scm, actions: [u8; 2]) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [
            InterventionAssignment::concrete(v(A1), Value::f64(f64::from(actions[0]))),
            InterventionAssignment::concrete(v(A2), Value::f64(f64::from(actions[1]))),
        ],
        [axis(B), axis(L1), axis(L2), axis(Y)],
        clamp(scm.law(&[(A1, actions[0]), (A2, actions[1])], &[B, L1, L2, Y])),
        "source-1",
        LawTolerance::default(),
    )
    .unwrap()
}

/// Exact laws from an explicit list.
pub fn data_of(laws: Vec<ExactDiscreteLaw>) -> ExactTransportData {
    ExactTransportData::try_new(laws, 4096).unwrap()
}

/// A catalog whose source experiment intervenes only on the two actions and
/// measures `b`, `l1`, `l2` and `y`, plus the target's observational law.
pub fn action_catalog() -> EvidenceCatalog {
    catalog_of(&[
        regime(0, "target", &[], &[B, L1, A1, L2, A2, Y]),
        regime(1, "source", &[A1, A2], &[B, L1, L2, Y]),
    ])
}

/// The fixture catalog plus a source observational regime no derivation reads.
pub fn catalog_with_unused_regime() -> EvidenceCatalog {
    catalog_of(&[
        regime(0, "target", &[], &[B, L1, A1, L2, A2, Y]),
        regime(1, "source", &[B, L1, A1, L2, A2], &[Y]),
        regime(2, "source", &[], &[B, L1, A1, L2, A2, Y]),
    ])
}

fn catalog_of(regimes: &[EvidenceRegime]) -> EvidenceCatalog {
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("{}-{}", r.population, r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new([], regimes.to_vec(), bindings, None).unwrap()
}

/// The target model with only the baseline mechanism moved: the shape whose
/// derivation needs the source's action experiment, not complete histories.
pub fn baseline_shift_source() -> Scm {
    let mut scm = target_scm();
    scm.exo_p[0] = 0.35;
    scm
}

/// The unrolled diagram with a selection at the baseline only.
pub fn baseline_only_spec() -> TemporalSequenceSpec {
    unrolled(&[B], &DIRECTED[..11])
}

/// The unrolled diagram with `extra` more independent three-level baseline
/// covariates (ids 6..) and a catalog measuring and intervening on them: a wide
/// history lattice with the same identification.
pub fn wide(extra: usize) -> (TemporalSequenceSpec, EvidenceCatalog) {
    let n = 6 + extra;
    let mut names = coordinates();
    for i in 6..n {
        names.push(ScenarioCoordinate {
            variable: v(i),
            name: Arc::from(format!("x{i}")),
            domain: VariableDomain::Categorical { cardinality: 3 },
            unit: None,
        });
    }
    let mut slots = slots();
    slots.baseline.extend((6..n).map(v));
    let diagram = diagram(n, &DIRECTED[..11], &[(A1, Y), (A2, Y)], &[B, L2]);
    let spec = TemporalSequenceSpec::try_new(2, slots, diagram, names).unwrap();
    let everything = (0..n).collect::<Vec<_>>();
    // The extra covariates do not reach the outcome, so the derivation never
    // cites them in a source experiment: the regime intervenes on the fixture's
    // five coordinates only.
    let catalog = catalog_of(&[
        regime(0, "target", &[], &everything),
        regime(1, "source", &[B, L1, A1, L2, A2], &[Y]),
    ]);
    (spec, catalog)
}

/// A known model whose two actions have three levels each, enumerated over
/// independent exogenous variables (some ternary), with the fixture's structure.
pub mod categorical {
    use super::super::z_scm::diagram;
    use super::{
        A1, A2, B, EvidenceCatalog, ExactDiscreteLaw, ExactTransportData, InterventionAssignment,
        L1, L2, LawTolerance, RegimeId, ScenarioCoordinate, TemporalSequenceSpec, Value,
        VariableDomain, Y, catalog, coordinates, slots, v,
    };
    use antecedent_expr::DiscreteAxis;
    /// Levels of each coordinate `b, l1, a1, l2, a2, y`.
    pub const LEVELS: [usize; 6] = [2, 2, 3, 2, 3, 2];

    type Mechanism = Box<dyn Fn(&[u8], &[u8]) -> u8>;

    /// Exogenous levels: e0 `b`, e1 `l1`, e2 latent `a1 <-> y`, e3 `a1` (3),
    /// e4 `l2`, e5 latent `a2 <-> y`, e6 `a2` (3), e7 `y`.
    const EXO_LEVELS: [usize; 8] = [2, 2, 2, 3, 2, 2, 3, 2];

    /// A structural model over the six coordinates.
    pub struct Model {
        exo: Vec<Vec<f64>>,
        f: Vec<Mechanism>,
    }

    fn model(exo: Vec<Vec<f64>>, shifted: bool) -> Model {
        let l2: Mechanism = if shifted {
            Box::new(|v, e| u8::from(v[A1] == 2 && v[B] == 0) ^ e[4])
        } else {
            Box::new(|v, e| u8::from(v[A1] == 2) ^ e[4])
        };
        Model {
            exo,
            f: vec![
                Box::new(|_, e| e[0]),
                Box::new(|v, e| v[B] ^ e[1]),
                Box::new(|v, e| (e[3] + e[2] + v[L1]) % 3),
                l2,
                Box::new(|v, e| (e[6] + e[5] + v[L2]) % 3),
                Box::new(|v, e| {
                    u8::from(v[A1] == 1 && e[2] == 0)
                        ^ u8::from(v[A2] == 2 && v[L2] == 1)
                        ^ u8::from(v[B] == 1 && e[7] == 1)
                        ^ u8::from(e[5] == 1 && v[L1] == 1)
                        ^ u8::from(v[A1] == 2 && v[A2] == 1)
                }),
            ],
        }
    }

    fn exo(b: f64) -> Vec<Vec<f64>> {
        vec![
            vec![1.0 - b, b],
            vec![0.7, 0.3],
            vec![0.1, 0.9],
            vec![0.3, 0.4, 0.3],
            vec![0.1, 0.9],
            vec![0.85, 0.15],
            vec![0.5, 0.2, 0.3],
            vec![0.55, 0.45],
        ]
    }

    /// The source population: initial state and `l2` mechanism differ.
    pub fn source() -> Model {
        let mut exo = exo(0.35);
        exo[4] = vec![0.75, 0.25];
        model(exo, false)
    }

    /// The target population.
    pub fn target() -> Model {
        model(exo(0.8), true)
    }

    impl Model {
        /// Exact joint law over `measured` (first most significant) under `do_`.
        pub fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
            let cells = measured.iter().map(|m| LEVELS[*m]).product::<usize>();
            let mut out = vec![0.0; cells];
            let mut e = vec![0u8; EXO_LEVELS.len()];
            loop {
                let weight =
                    e.iter().enumerate().map(|(i, l)| self.exo[i][*l as usize]).product::<f64>();
                let mut values = vec![0u8; 6];
                for i in 0..6 {
                    values[i] = match do_.iter().find(|(var, _)| *var == i) {
                        Some((_, level)) => *level,
                        None => (self.f[i])(&values, &e),
                    };
                }
                let index = measured
                    .iter()
                    .fold(0usize, |acc, m| acc * LEVELS[*m] + usize::from(values[*m]));
                out[index] += weight;
                let mut i = 0;
                while i < e.len() {
                    e[i] += 1;
                    if usize::from(e[i]) < EXO_LEVELS[i] {
                        break;
                    }
                    e[i] = 0;
                    i += 1;
                }
                if i == e.len() {
                    return out;
                }
            }
        }

        /// `P(y = 1 | do(a1, a2))`.
        pub fn truth(&self, sequence: [u8; 2]) -> f64 {
            self.law(&[(A1, sequence[0]), (A2, sequence[1])], &[Y])[1]
        }
    }

    fn axis(variable: usize) -> DiscreteAxis {
        DiscreteAxis {
            variable: v(variable),
            values: (0..LEVELS[variable]).map(|l| Value::f64(l as f64)).collect(),
        }
    }

    fn clamp(p: Vec<f64>) -> Vec<f64> {
        p.into_iter().map(|x| x.clamp(0.0, 1.0)).collect()
    }

    /// The unrolled diagram of the fixture with three-level actions.
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
        ];
        let mut names = coordinates();
        for a in [A1, A2] {
            names[a] = ScenarioCoordinate {
                domain: VariableDomain::Categorical { cardinality: 3 },
                ..names[a].clone()
            };
        }
        TemporalSequenceSpec::try_new(
            2,
            slots(),
            diagram(6, &directed, &[(A1, Y), (A2, Y)], &[B, L2]),
            names,
        )
        .unwrap()
    }

    /// The fixture's catalog: the regimes do not depend on the alphabet.
    pub fn evidence() -> EvidenceCatalog {
        catalog()
    }

    /// The target's observational law and the source's history-indexed outcome
    /// experiments for every history and both three-level actions.
    pub fn laws(source: &Model, target: &Model) -> ExactTransportData {
        let mut laws = vec![
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                (0..6).map(axis).collect::<Vec<_>>(),
                clamp(target.law(&[], &[B, L1, A1, L2, A2, Y])),
                "target-0",
                LawTolerance::default(),
            )
            .unwrap(),
        ];
        let order = [B, L1, A1, L2, A2];
        for code in 0..(2 * 2 * 3 * 2 * 3usize) {
            let mut rest = code;
            let mut history = [0u8; 5];
            for (slot, var) in order.iter().enumerate().rev() {
                history[slot] = u8::try_from(rest % LEVELS[*var]).unwrap();
                rest /= LEVELS[*var];
            }
            let interventions = order
                .iter()
                .zip(history)
                .map(|(var, level)| {
                    InterventionAssignment::concrete(v(*var), Value::f64(f64::from(level)))
                })
                .collect::<Vec<_>>();
            let do_ = order.iter().copied().zip(history).collect::<Vec<_>>();
            laws.push(
                ExactDiscreteLaw::try_new(
                    "source",
                    RegimeId::from_raw(1),
                    interventions,
                    [axis(Y)],
                    clamp(source.law(&do_, &[Y])),
                    "source-1",
                    LawTolerance::default(),
                )
                .unwrap(),
            );
        }
        ExactTransportData::try_new(laws, 4096).unwrap()
    }
}

/// The lagged-template route (ADR 0021): a `TemporalDag` template unrolled by
/// `unroll_two_slice` into the two-slice diagram, with a known structural model
/// enumerated over its unrolled coordinates.
///
/// Template variables `L = 0`, `A = 1`, `Y = 2` with `L[t-1] -> L[t]`,
/// `A[t-1] -> L[t]`, `L[t] -> A[t]`, `L[t] -> Y[t]`, `A[t] -> Y[t]`; one baseline
/// covariate affects `L` and `Y` in both periods. Unrolled ids: `L@1 = 0`,
/// `A@1 = 1`, `Y@1 = 2` (a covariate of step 2), `L@2 = 3`, `A@2 = 4`,
/// `Y@2 = 5` (the outcome), baseline `= 6`. Each action shares a latent cause
/// with its own period's `Y`. The baseline and `L@2` mechanisms differ between
/// the populations.
pub mod template {
    use super::super::z_scm::Mechanism;
    use super::{
        DiscreteAxis, EvidenceCatalog, ExactDiscreteLaw, ExactTransportData,
        InterventionAssignment, LawTolerance, RegimeId, ScenarioCoordinate, TemporalSequenceSpec,
        Value, VariableDomain, VariableId, catalog_of, regime, v,
    };
    use antecedent_core::Lag;
    use antecedent_graph::{DenseNodeId, SelectionDiagram, TemporalDag};
    use antecedent_identify::sid::temporal_sequence::{TemplateRoles, unroll_two_slice};
    use std::sync::Arc;

    pub const L1: usize = 0;
    pub const A1: usize = 1;
    pub const Y1: usize = 2;
    pub const L2: usize = 3;
    pub const A2: usize = 4;
    pub const Y2: usize = 5;
    pub const BASE: usize = 6;
    const NAMES: [&str; 7] = ["l1", "a1", "y1", "l2", "a2", "y2", "base"];
    /// Topological order of the unrolled coordinates: the baseline comes first.
    const ORDER: [usize; 7] = [BASE, L1, A1, Y1, L2, A2, Y2];

    /// The template of one period with its lagged parents.
    pub fn template() -> (TemporalDag, TemplateRoles) {
        let mut dag = TemporalDag::empty();
        let l0 = dag.add_lagged(v(0), Lag::CONTEMPORANEOUS).unwrap();
        let l1 = dag.add_lagged(v(0), Lag::from_raw(1)).unwrap();
        let a0 = dag.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
        let a1 = dag.add_lagged(v(1), Lag::from_raw(1)).unwrap();
        let y0 = dag.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
        for (from, to) in [(l1, l0), (a1, l0), (l0, a0), (l0, y0), (a0, y0)] {
            dag.insert_directed(from, to).unwrap();
        }
        (dag, TemplateRoles { covariates: vec![v(0)], action: v(1), outcome: v(2) })
    }

    /// The specification unrolled from the template, with the latent confounders
    /// and the time-indexed mechanism differences added.
    pub fn spec() -> TemporalSequenceSpec {
        let (dag, roles) = template();
        let mut unrolled = unroll_two_slice(&dag, &roles, 1, &[(0, v(0)), (0, v(2))]).unwrap();
        let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap());
        unrolled.admg.insert_bidirected(dense(A1), dense(Y1)).unwrap();
        unrolled.admg.insert_bidirected(dense(A2), dense(Y2)).unwrap();
        let selections: Arc<[VariableId]> = Arc::from([v(BASE), v(L2)]);
        let diagram = SelectionDiagram::try_new(unrolled.admg, selections).unwrap();
        let coordinates = NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| ScenarioCoordinate {
                variable: v(i),
                name: Arc::from(*name),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        TemporalSequenceSpec::try_new(2, unrolled.slots, diagram, coordinates).unwrap()
    }

    /// The target's observational law and the source's history-indexed experiments.
    pub fn catalog() -> EvidenceCatalog {
        catalog_of(&[
            regime(0, "target", &[], &[L1, A1, Y1, L2, A2, Y2, BASE]),
            // `Y@1` never reaches `Y@2`, so the derivation does not cite it in a
            // source experiment: the regime intervenes on the other five coordinates.
            regime(1, "source", &[BASE, L1, A1, L2, A2], &[Y2]),
        ])
    }

    /// A structural model over the unrolled coordinates.
    pub struct Model {
        exo: [f64; 9],
        f: Vec<Mechanism>,
    }

    fn bit(b: bool) -> u8 {
        u8::from(b)
    }

    fn model(exo: [f64; 9], shifted: bool) -> Model {
        let l2: Mechanism = if shifted {
            Box::new(|v, e| bit(v[A1] == 1 && v[BASE] == 0) ^ e[4])
        } else {
            Box::new(|v, e| v[A1] ^ e[4])
        };
        let f: Vec<Mechanism> = vec![
            Box::new(|v, e| v[BASE] ^ e[1]),
            Box::new(|v, e| e[2] ^ bit(v[L1] == 1 && e[3] == 1)),
            Box::new(|v, e| {
                bit(v[A1] == 1 && e[2] == 0)
                    ^ bit(v[BASE] == 1 && e[7] == 1)
                    ^ bit(v[L1] == 1 && v[A1] == 1)
            }),
            l2,
            Box::new(|v, e| e[5] ^ bit(v[L2] == 1 && e[6] == 1)),
            Box::new(|v, e| {
                bit(v[A2] == 1 && e[5] == 0)
                    ^ bit(v[L2] == 1 && v[A2] == 1)
                    ^ bit(v[BASE] == 1 && e[8] == 1)
            }),
            Box::new(|_, e| e[0]),
        ];
        Model { exo, f }
    }

    /// The source population.
    pub fn source() -> Model {
        model([0.35, 0.5, 0.9, 0.1, 0.25, 0.1, 0.7, 0.4, 0.3], false)
    }

    /// The target population.
    pub fn target() -> Model {
        model([0.8, 0.5, 0.9, 0.1, 0.9, 0.1, 0.7, 0.4, 0.3], true)
    }

    impl Model {
        /// Exact joint law over `measured` (first most significant) under `do_`.
        pub fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
            let mut out = vec![0.0; 1 << measured.len()];
            for mask in 0..(1usize << self.exo.len()) {
                let exo =
                    (0..self.exo.len()).map(|b| u8::from((mask >> b) & 1 == 1)).collect::<Vec<_>>();
                let weight = self
                    .exo
                    .iter()
                    .zip(&exo)
                    .map(|(p, e)| if *e == 1 { *p } else { 1.0 - *p })
                    .product::<f64>();
                let mut values = vec![0u8; 7];
                for i in ORDER {
                    values[i] = match do_.iter().find(|(var, _)| *var == i) {
                        Some((_, level)) => *level,
                        None => (self.f[i])(&values, &exo),
                    };
                }
                let index =
                    measured.iter().fold(0usize, |acc, m| (acc << 1) | usize::from(values[*m]));
                out[index] += weight;
            }
            out
        }

        /// `P(y2 = 1 | do(a1, a2))`.
        pub fn truth(&self, sequence: [u8; 2]) -> f64 {
            self.law(&[(A1, sequence[0]), (A2, sequence[1])], &[Y2])[1]
        }
    }

    fn axis(variable: usize) -> DiscreteAxis {
        DiscreteAxis {
            variable: v(variable),
            values: Arc::from([Value::f64(0.0), Value::f64(1.0)]),
        }
    }

    fn clamp(p: Vec<f64>) -> Vec<f64> {
        p.into_iter().map(|x| x.clamp(0.0, 1.0)).collect()
    }

    /// The target's observational law and the source's outcome law under every
    /// complete history and both actions.
    pub fn laws(source: &Model, target: &Model) -> ExactTransportData {
        let all = [L1, A1, Y1, L2, A2, Y2, BASE];
        let mut laws = vec![
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                all.iter().map(|i| axis(*i)).collect::<Vec<_>>(),
                clamp(target.law(&[], &all)),
                "target-0",
                LawTolerance::default(),
            )
            .unwrap(),
        ];
        let intervened = [BASE, L1, A1, L2, A2];
        for code in 0..32u8 {
            let do_ = intervened
                .iter()
                .enumerate()
                .map(|(k, var)| (*var, (code >> (4 - k)) & 1))
                .collect::<Vec<_>>();
            laws.push(
                ExactDiscreteLaw::try_new(
                    "source",
                    RegimeId::from_raw(1),
                    do_.iter()
                        .map(|(var, level)| {
                            InterventionAssignment::concrete(v(*var), Value::f64(f64::from(*level)))
                        })
                        .collect::<Vec<_>>(),
                    [axis(Y2)],
                    clamp(source.law(&do_, &[Y2])),
                    "source-1",
                    LawTolerance::default(),
                )
                .unwrap(),
            );
        }
        ExactTransportData::try_new(laws, 4096).unwrap()
    }
}
