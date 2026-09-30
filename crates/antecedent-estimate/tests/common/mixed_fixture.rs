//! Binary structural-model enumeration and catalog construction shared by the
//! mixed-source known-truth fixtures.
//!
//! Every law a fixture supplies is enumerated exactly from a binary structural
//! model, so a formula that is checked against these laws is checked against the
//! model's own interventional truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code)]

use std::sync::Arc;

use crate::common::z_scm::{Scm, vid};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableCoordinate,
    VariableDomain,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactTransportData,
    InterventionAssignment as ExprInterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::MixedSourceQuery;

/// One study: measured coordinates and the (possibly empty) set it experiments on.
#[derive(Clone)]
pub struct Study {
    pub name: String,
    pub on: Vec<usize>,
    pub measured: Vec<usize>,
}

pub fn study(name: &str, on: &[usize], measured: &[usize]) -> Study {
    Study { name: name.into(), on: on.to_vec(), measured: measured.to_vec() }
}

/// Catalog and exact laws of the target population's studies: one family regime per
/// study, with one law for every level of its intervention set.
pub fn build(scm: &Scm, studies: &[Study]) -> (EvidenceCatalog, ExactTransportData) {
    let n = scm.n;
    let coordinates = (0..n)
        .map(|i| VariableCoordinate {
            variable: vid(i),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let environment = Environment::try_new("target", coordinates, []).unwrap();
    let (mut regimes, mut bindings, mut laws) = (Vec::new(), Vec::new(), Vec::new());
    for (k, study) in studies.iter().enumerate() {
        let id = RegimeId::from_raw(u32::try_from(k + 1).unwrap());
        let mut regime = EvidenceRegime::try_new(
            id,
            if study.on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            study.on.iter().map(|v| vid(*v)).collect::<Vec<_>>(),
            [],
            study.measured.iter().map(|v| vid(*v)).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        regime.study = Some(Arc::from(study.name.as_str()));
        regimes.push(regime);
        let snapshot = format!("{}-{k}", study.name);
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: id,
            snapshot_identity: Arc::from(snapshot.as_str()),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let axes = study
            .measured
            .iter()
            .map(|i| DiscreteAxis {
                variable: vid(*i),
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            })
            .collect::<Vec<_>>();
        for levels in 0..(1usize << study.on.len()) {
            let assignments = study
                .on
                .iter()
                .enumerate()
                .map(|(bit, v)| (*v, u8::from((levels >> bit) & 1 == 1)))
                .collect::<Vec<_>>();
            let interventions = assignments
                .iter()
                .map(|(v, level)| {
                    ExprInterventionAssignment::concrete(vid(*v), Value::Bool(*level == 1))
                })
                .collect::<Vec<_>>();
            laws.push(
                ExactDiscreteLaw::try_new(
                    "target",
                    id,
                    interventions,
                    axes.clone(),
                    scm.law(&assignments, &study.measured),
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

pub fn graph(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut g = Admg::with_variables(n);
    let d = DenseNodeId::from_raw;
    for (a, b) in directed {
        g.insert_directed(d(*a), d(*b)).unwrap();
    }
    for (a, b) in bidirected {
        g.insert_bidirected(d(*a), d(*b)).unwrap();
    }
    g
}

pub fn query(outcome: usize, treatment: usize) -> MixedSourceQuery {
    MixedSourceQuery {
        outcomes: Arc::from([vid(outcome)]),
        treatments: Arc::from([vid(treatment)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    }
}

/// A query for one outcome and any number of treatments.
pub fn query_many(outcome: usize, treatments: &[usize]) -> MixedSourceQuery {
    MixedSourceQuery {
        outcomes: Arc::from([vid(outcome)]),
        treatments: treatments.iter().map(|t| vid(*t)).collect::<Vec<_>>().into(),
        target: Arc::from("target"),
        sources: Arc::from([]),
    }
}

/// A request that binds every treatment; bit `k` of `levels` is treatment `k`'s level.
pub fn request_many(treatments: &[usize], levels: usize) -> Assignment {
    Assignment::from_pairs(
        treatments.iter().enumerate().map(|(k, t)| (vid(*t), Value::Bool((levels >> k) & 1 == 1))),
    )
}

/// The graph's ADMG edges as `u32` pairs.
pub fn edges32(edges: &[(usize, usize)]) -> Vec<(u32, u32)> {
    edges.iter().map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap())).collect()
}

pub fn request(treatment: usize, level: bool) -> Assignment {
    Assignment::from_pairs([(vid(treatment), Value::Bool(level))])
}

/// Markovian chain `X -> Z -> Y` with independent noise.
pub fn chain_scm() -> Scm {
    Scm {
        n: 3,
        exo_p: vec![0.4, 0.8, 0.3, 0.7, 0.2],
        f: vec![
            Box::new(|_, e| e[0]),
            Box::new(|v, e| if v[0] == 1 { e[1] } else { e[2] }),
            Box::new(|v, e| if v[1] == 1 { e[3] } else { e[4] }),
        ],
    }
}

/// Front-door graph `X -> Z -> Y`, `X <-> Y` through the latent bit `e[5]`.
pub fn frontdoor_scm() -> Scm {
    Scm {
        n: 3,
        exo_p: vec![0.5, 0.8, 0.3, 0.7, 0.2, 0.5],
        f: vec![
            Box::new(|_, e| e[5] ^ (e[0] & e[1])),
            Box::new(|v, e| if v[0] == 1 { e[1] } else { e[2] }),
            Box::new(|v, e| {
                let noise = if v[1] == 1 { e[3] } else { e[4] };
                noise ^ (e[5] & u8::from(v[1] == 0))
            }),
        ],
    }
}

pub const X: usize = 0;
pub const Z: usize = 1;
pub const Y: usize = 2;

/// A small deterministic generator (`SplitMix64`).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
    }
    pub fn unit(&mut self) -> f64 {
        f64::from(u32::try_from(self.next() >> 40).unwrap()) / f64::from(1u32 << 24)
    }
}

/// A random binary structural model over `n` nodes consistent with the ADMG: every
/// node is a random function of its parents and the latent bits of its bidirected
/// edges, flipped by independent noise so every configuration has positive mass.
pub fn random_scm(
    rng: &mut Rng,
    n: usize,
    directed: &[(usize, usize)],
    bidirected: &[(usize, usize)],
) -> Scm {
    let mut exo_p: Vec<f64> = (0..n).map(|_| 0.15 + 0.2 * rng.unit()).collect();
    exo_p.extend(bidirected.iter().map(|_| 0.3 + 0.4 * rng.unit()));
    let f = (0..n)
        .map(|i| {
            let parents: Vec<usize> =
                directed.iter().filter(|(_, to)| *to == i).map(|(from, _)| *from).collect();
            let latents: Vec<usize> = bidirected
                .iter()
                .enumerate()
                .filter(|(_, (a, b))| *a == i || *b == i)
                .map(|(k, _)| n + k)
                .collect();
            let width = parents.len() + latents.len();
            let table: Vec<u8> =
                (0..1usize << width).map(|_| u8::from(rng.below(2) == 1)).collect();
            Box::new(move |values: &[u8], exo: &[u8]| {
                let key = parents
                    .iter()
                    .map(|p| values[*p])
                    .chain(latents.iter().map(|l| exo[*l]))
                    .fold(0usize, |acc, bit| (acc << 1) | usize::from(bit));
                table[key] ^ exo[i]
            }) as crate::common::z_scm::Mechanism
        })
        .collect();
    Scm { n, exo_p, f }
}
