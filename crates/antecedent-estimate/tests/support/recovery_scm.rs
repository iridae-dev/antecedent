//! Enumerated binary m-graph SCMs for the X10 recovery tests, independent of the
//! library's recovery code: the joint over every node is enumerated from the
//! mechanisms, the proxies are computed deterministically, and the observed
//! pattern law and the target law are marginals of that joint.
//!
//! Numbering: `X_i = i` (`0..k`), `O_j = k + j`, `R_i = k + m + i`,
//! `X*_i = 2k + m + i`.
#![allow(dead_code, clippy::unused_self)]
#![allow(clippy::cast_possible_truncation, reason = "node counts are at most 11")]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableCoordinate, VariableDomain,
    VariableId,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{ObservationRecoveryQuery, PartiallyObserved};

pub const POPULATION: &str = "clinic";
pub const SNAPSHOT: &str = "snap-observed";

pub fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// A binary m-graph with seeded positive mechanisms.
#[derive(Clone, Debug)]
pub struct MModel {
    pub k: u32,
    pub m: u32,
    pub graph: Admg,
    /// `P(node = 1 | parents)` per parent configuration (graph parent order,
    /// first parent most significant), for every non-proxy node.
    pub mechanisms: BTreeMap<u32, Vec<f64>>,
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [0.1, 0.9].
    pub fn prob(&mut self) -> f64 {
        0.1 + 0.8 * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

impl MModel {
    pub fn x(&self, i: u32) -> u32 {
        i
    }
    pub fn o(&self, j: u32) -> u32 {
        self.k + j
    }
    pub fn r(&self, i: u32) -> u32 {
        self.k + self.m + i
    }
    pub fn proxy(&self, i: u32) -> u32 {
        2 * self.k + self.m + i
    }
    pub fn nodes(&self) -> u32 {
        3 * self.k + self.m
    }

    /// Build from substantive edges (over `0..k+m`) and each response's parents
    /// (substantive ids), with seeded mechanisms.
    pub fn new(
        k: u32,
        m: u32,
        substantive: &[(u32, u32)],
        responses: &[Vec<u32>],
        seed: u64,
    ) -> Option<Self> {
        let mut graph = Admg::with_variables(3 * k + m);
        let d = DenseNodeId::from_raw;
        for &(a, b) in substantive {
            graph.insert_directed(d(a), d(b)).ok()?;
        }
        for (i, parents) in responses.iter().enumerate() {
            let r = k + m + i as u32;
            for p in parents {
                graph.insert_directed(d(*p), d(r)).ok()?;
            }
        }
        for i in 0..k {
            graph.insert_directed(d(i), d(2 * k + m + i)).ok()?;
            graph.insert_directed(d(k + m + i), d(2 * k + m + i)).ok()?;
        }
        let mut rng = Rng::new(seed);
        let mut mechanisms = BTreeMap::new();
        for node in 0..(2 * k + m) {
            let configs = 1usize << graph.parents(d(node)).len();
            mechanisms.insert(node, (0..configs).map(|_| rng.prob()).collect());
        }
        Some(Self { k, m, graph, mechanisms })
    }

    pub fn query(&self) -> ObservationRecoveryQuery {
        ObservationRecoveryQuery {
            population: Arc::from(POPULATION),
            observed_regime: RegimeId::from_raw(7),
            partially_observed: (0..self.k)
                .map(|i| PartiallyObserved {
                    variable: v(self.x(i)),
                    response: v(self.r(i)),
                    proxy: v(self.proxy(i)),
                })
                .collect::<Vec<_>>()
                .into(),
            fully_observed: (0..self.m).map(|j| v(self.o(j))).collect::<Vec<_>>().into(),
        }
    }

    /// Observed coordinates, sorted: O, R, X*.
    pub fn observed_variables(&self) -> Vec<u32> {
        (self.k..self.k + self.m).chain(self.k + self.m..self.nodes()).collect()
    }

    pub fn catalog(&self) -> EvidenceCatalog {
        self.catalog_with(DistributionAvailability::Joint, &self.observed_variables())
    }

    pub fn catalog_with(
        &self,
        availability: DistributionAvailability,
        measured: &[u32],
    ) -> EvidenceCatalog {
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(7),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            measured.iter().map(|r| v(*r)).collect::<Vec<_>>(),
            POPULATION,
            availability,
        )
        .unwrap();
        let mut labelled = regime;
        labelled.label = Some(Arc::from("observed"));
        let coordinates = (0..self.nodes())
            .map(|n| VariableCoordinate {
                variable: v(n),
                domain: if n >= 2 * self.k + self.m {
                    VariableDomain::Categorical { cardinality: 3 }
                } else {
                    VariableDomain::Binary
                },
                unit: None,
            })
            .collect::<Vec<_>>();
        EvidenceCatalog {
            environments: Arc::from([Environment::try_new(POPULATION, coordinates, []).unwrap()]),
            regimes: Arc::from([labelled]),
            bindings: Arc::from([RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(7),
                snapshot_identity: Arc::from(SNAPSHOT),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: antecedent_core::DependenceGroup::IndependentStudies,
            }]),
            target_sampling: None,
        }
    }

    /// Joint mass of every configuration of the non-proxy nodes `0..2k+m`
    /// (bit `n` of the index is node `n`).
    pub fn joint(&self) -> Vec<f64> {
        let n = 2 * self.k + self.m;
        let mut out = vec![0.0; 1 << n];
        for (config, cell) in out.iter_mut().enumerate() {
            let value = |node: u32| (config >> node) & 1;
            let mut p = 1.0;
            for node in 0..n {
                let parents = self.graph.parents(DenseNodeId::from_raw(node));
                let index = parents.iter().fold(0usize, |acc, q| (acc << 1) | value(q.raw()));
                let one = self.mechanisms[&node][index];
                p *= if value(node) == 1 { one } else { 1.0 - one };
            }
            *cell = p;
        }
        out
    }

    /// Target `P(X, O)` over the axes `0..k+m` (sorted), last fastest.
    pub fn truth(&self) -> Vec<f64> {
        let s = self.k + self.m;
        let mut out = vec![0.0; 1 << s];
        for (config, p) in self.joint().iter().enumerate() {
            let index = (0..s).fold(0usize, |acc, node| (acc << 1) | ((config >> node) & 1));
            out[index] += p;
        }
        out
    }

    /// Observed pattern law over O, R, X* (sorted), X* levels `[0, 1, ?]`.
    pub fn observed_law(&self) -> ExactDiscreteLaw {
        let vars = self.observed_variables();
        let proxy_start = 2 * self.k + self.m;
        let cardinality = |node: u32| if node >= proxy_start { 3 } else { 2 };
        let size: usize = vars.iter().map(|n| cardinality(*n)).product();
        let mut table = vec![0.0; size];
        for (config, p) in self.joint().iter().enumerate() {
            let value = |node: u32| (config >> node) & 1;
            let mut index = 0usize;
            for node in &vars {
                let level = if *node >= proxy_start {
                    let i = node - proxy_start;
                    if value(self.r(i)) == 1 { value(self.x(i)) } else { 2 }
                } else {
                    value(*node)
                };
                index = index * cardinality(*node) + level;
            }
            table[index] += p;
        }
        let axes = vars
            .iter()
            .map(|node| DiscreteAxis {
                variable: v(*node),
                values: if *node >= proxy_start {
                    Arc::from([Value::f64(0.0), Value::f64(1.0), Value::Label(Arc::from("?"))])
                } else {
                    Arc::from([Value::f64(0.0), Value::f64(1.0)])
                },
            })
            .collect::<Vec<_>>();
        ExactDiscreteLaw::try_new(
            POPULATION,
            RegimeId::from_raw(7),
            Vec::new(),
            axes,
            table,
            SNAPSHOT,
            LawTolerance::default(),
        )
        .unwrap()
    }

    /// Complete-data law `P(X, O)` as a table the same shape as a recovered law.
    pub fn complete_law(&self) -> ExactDiscreteLaw {
        let axes = (0..self.k + self.m)
            .map(|n| DiscreteAxis {
                variable: v(n),
                values: Arc::from([Value::f64(0.0), Value::f64(1.0)]),
            })
            .collect::<Vec<_>>();
        ExactDiscreteLaw::try_new(
            POPULATION,
            RegimeId::from_raw(7),
            Vec::new(),
            axes,
            self.truth(),
            "complete",
            LawTolerance::default(),
        )
        .unwrap()
    }

    /// `P(y = 1 | do(t = level))` by enumerating the mutilated SCM.
    pub fn interventional(&self, outcome: u32, treatment: u32, level: usize) -> f64 {
        let mut model = self.clone();
        let parents = model.graph.parents(DenseNodeId::from_raw(treatment)).len();
        model.mechanisms.insert(treatment, vec![level as f64; 1 << parents]);
        model
            .joint()
            .iter()
            .enumerate()
            .filter(|(c, _)| (c >> outcome) & 1 == 1)
            .map(|(_, p)| p)
            .sum()
    }
}

/// Every directed acyclic edge set over `n` labelled nodes.
pub fn dags(n: u32) -> Vec<Vec<(u32, u32)>> {
    let pairs: Vec<(u32, u32)> =
        (0..n).flat_map(|a| (0..n).filter(move |b| *b != a).map(move |b| (a, b))).collect();
    let mut out = Vec::new();
    for mask in 0u64..(1 << pairs.len()) {
        let edges: Vec<_> = pairs
            .iter()
            .enumerate()
            .filter(|(i, _)| (mask >> i) & 1 == 1)
            .map(|(_, e)| *e)
            .collect();
        let mut g = Admg::with_variables(n);
        if edges.iter().all(|(a, b)| {
            g.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).is_ok()
        }) {
            out.push(edges);
        }
    }
    out
}
