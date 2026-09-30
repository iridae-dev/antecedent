//! Latent binary SCMs for the 2.2B B1 conditional transport known-truth tests.
//!
//! Each bidirected edge is a binary latent, the target population shifts the
//! mechanism of every selected node, and every law (each source experiment and
//! the target observational joint) and the conditional truth are enumerated.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "binary levels and small graph coordinates"
)]

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, RegimeId, RegimeKind,
    Value, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactTransportData, InterventionAssignment,
    LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, ConditionalTransportQuery};
use std::sync::Arc;

pub fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// A latent SCM over binary nodes `0..n` in topological order.
#[derive(Clone)]
pub struct Scm {
    pub n: usize,
    pub directed: Vec<(usize, usize)>,
    pub bidirected: Vec<(usize, usize)>,
    pub selected: Vec<usize>,
    /// Parameterization index: shifts every weight so wrong formulas differ.
    pub params: usize,
    /// Target nodes forced to level 0 (a zero-mass event at level 1).
    pub target_zero: Vec<usize>,
}

impl Scm {
    pub fn diagram(&self) -> SelectionDiagram {
        let mut graph = Admg::with_variables(self.n as u32);
        for &(a, b) in &self.directed {
            graph
                .insert_directed(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
                .unwrap();
        }
        for &(a, b) in &self.bidirected {
            graph
                .insert_bidirected(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
                .unwrap();
        }
        SelectionDiagram::try_new(
            graph,
            self.selected.iter().map(|s| v(*s as u32)).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// P(node = 1 | its parents' values and latents) in the source or target.
    pub fn p_one(&self, node: usize, values: usize, latent: usize, target: bool) -> f64 {
        if target && self.target_zero.contains(&node) {
            return 0.0;
        }
        let shift = 0.02 * self.params as f64;
        let mut p = 0.08 + 0.05 * node as f64 + shift;
        for &(a, b) in &self.directed {
            if b == node && (values >> a) & 1 == 1 {
                p += 0.12 + 0.03 * a as f64;
            }
        }
        for (e, &(a, b)) in self.bidirected.iter().enumerate() {
            if (a == node || b == node) && (latent >> e) & 1 == 1 {
                p += 0.06 + 0.01 * e as f64;
            }
        }
        if target && self.selected.contains(&node) {
            p += 0.07 - 0.14 * (node % 2) as f64 + 0.1;
        }
        p.clamp(0.02, 0.98)
    }

    /// The law under do(mask = values) over the non-intervened nodes, row-major
    /// over the ascending axes (first axis most significant).
    pub fn law(&self, target: bool, mask: usize, values: usize) -> (Vec<usize>, Vec<f64>) {
        let axes: Vec<usize> = (0..self.n).filter(|i| mask & (1 << i) == 0).collect();
        let mut law = vec![0.0; 1 << axes.len()];
        let latents = 1usize << self.bidirected.len();
        for latent in 0..latents {
            for world in 0..(1usize << self.n) {
                if world & mask != values & mask {
                    continue;
                }
                let mut mass = 1.0 / latents as f64;
                for node in 0..self.n {
                    if mask & (1 << node) != 0 {
                        continue;
                    }
                    let p = self.p_one(node, world, latent, target);
                    mass *= if (world >> node) & 1 == 1 { p } else { 1.0 - p };
                }
                let row = axes.iter().fold(0, |row, i| row * 2 + ((world >> i) & 1));
                law[row] += mass;
            }
        }
        (axes, law)
    }

    pub fn catalog_and_laws(&self) -> (EvidenceCatalog, ExactTransportData) {
        let binary = |i: usize| DiscreteAxis {
            variable: v(i as u32),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        };
        let mut regimes = Vec::new();
        let mut laws = Vec::new();
        let full = 1usize << self.n;
        for mask in 0..full {
            let interventions: Vec<_> = (0..self.n).filter(|i| mask & (1 << i) != 0).collect();
            regimes.push(
                EvidenceRegime::try_new(
                    RegimeId::from_raw(mask as u32),
                    if mask == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
                    EvidenceKind::Available,
                    interventions.iter().map(|i| v(*i as u32)).collect::<Vec<_>>(),
                    [],
                    (0..self.n)
                        .filter(|i| mask & (1 << i) == 0)
                        .map(|i| v(i as u32))
                        .collect::<Vec<_>>(),
                    "source",
                    DistributionAvailability::Joint,
                )
                .unwrap(),
            );
            for values in 0..full {
                if values & !mask != 0 {
                    continue;
                }
                let (axes, mass) = self.law(false, mask, values);
                let assignments = interventions
                    .iter()
                    .map(|i| InterventionAssignment {
                        variable: v(*i as u32),
                        value: Value::Int64(((values >> i) & 1) as i64),
                    })
                    .collect::<Vec<_>>();
                laws.push(
                    ExactDiscreteLaw::try_new(
                        "source",
                        RegimeId::from_raw(mask as u32),
                        assignments,
                        axes.into_iter().map(binary).collect::<Vec<_>>(),
                        mass,
                        "oracle",
                        LawTolerance::default(),
                    )
                    .unwrap(),
                );
            }
        }
        regimes.push(
            EvidenceRegime::try_new(
                RegimeId::from_raw(full as u32),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                (0..self.n).map(|i| v(i as u32)).collect::<Vec<_>>(),
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap(),
        );
        let (axes, mass) = self.law(true, 0, 0);
        laws.push(
            ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(full as u32),
                [],
                axes.into_iter().map(binary).collect::<Vec<_>>(),
                mass,
                "oracle",
                LawTolerance::default(),
            )
            .unwrap(),
        );
        (
            EvidenceCatalog::try_new([], regimes, [], None).unwrap(),
            ExactTransportData::try_new(laws, 1_000_000).unwrap(),
        )
    }

    /// Enumerated target truth of P*(Y = y-atom | do(X = x), W = w), per y-atom
    /// in row-major order over `y`, or `None` when P*(w | do(x)) is zero.
    pub fn truth(&self, y: &[usize], x: &[usize], w: &[usize], level: usize) -> Option<Vec<f64>> {
        let mask: usize = x.iter().map(|i| 1 << i).sum();
        let values: usize = x.iter().filter(|i| (level >> *i) & 1 == 1).map(|i| 1 << i).sum();
        let (axes, law) = self.law(true, mask, values);
        let bit = |row: usize, node: usize| {
            let position = axes.iter().position(|a| *a == node).unwrap();
            (row >> (axes.len() - 1 - position)) & 1
        };
        let mut joint = vec![0.0; 1 << y.len()];
        for (row, mass) in law.iter().enumerate() {
            if w.iter().all(|node| bit(row, *node) == (level >> node) & 1) {
                let atom = y.iter().fold(0, |a, node| a * 2 + bit(row, *node));
                joint[atom] += mass;
            }
        }
        let total: f64 = joint.iter().sum();
        (total > 0.0).then(|| joint.iter().map(|p| p / total).collect())
    }
}

pub fn query(y: &[usize], x: &[usize], w: &[usize]) -> ConditionalTransportQuery {
    let ids = |s: &[usize]| s.iter().map(|i| v(*i as u32)).collect::<Arc<[_]>>();
    ConditionalTransportQuery {
        base: ClassicalTransportQuery {
            outcomes: ids(y),
            treatments: ids(x),
            source: Arc::from("source"),
            target: Arc::from("target"),
        },
        conditioned_on: ids(w),
    }
}

pub fn request(x: &[usize], w: &[usize], level: usize) -> Assignment {
    Assignment::from_pairs(
        x.iter().chain(w).map(|i| (v(*i as u32), Value::Int64(((level >> i) & 1) as i64))),
    )
}
