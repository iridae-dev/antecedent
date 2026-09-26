//! Binary structural-causal-model enumeration and catalog construction shared by
//! the known-truth z-transport fixtures.
//!
//! Every law a fixture supplies is enumerated exactly from the structural
//! equations, so a formula that is checked against these laws is checked
//! against the model's own interventional truth rather than against another
//! rendering of the same formula.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(dead_code)]

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::evaluate_exact_z_transport;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    ExactTransportData, InterventionAssignment as ExprInterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    SidLimits, ZTransportDecision, ZTransportDerivation, ZTransportQuery, bind_z_transport_catalog,
    decide_z_transport_with_catalog,
};

/// A structural equation: node value from earlier node values and exogenous bits.
pub type Mechanism = Box<dyn Fn(&[u8], &[u8]) -> u8>;

/// Node `i` takes `f[i](values, exogenous)`; only earlier nodes are valid inputs.
pub struct Scm {
    pub n: usize,
    /// Probability that each independent exogenous bit is one.
    pub exo_p: Vec<f64>,
    pub f: Vec<Mechanism>,
}

impl Scm {
    /// Exact joint law over `measured` (first variable most significant) under `do_`.
    pub fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; 1 << measured.len()];
        let m = self.exo_p.len();
        for mask in 0..(1usize << m) {
            let exo: Vec<u8> = (0..m).map(|bit| u8::from((mask >> bit) & 1 == 1)).collect();
            let mut weight = 1.0;
            for (bit, p) in self.exo_p.iter().enumerate() {
                weight *= if exo[bit] == 1 { *p } else { 1.0 - *p };
            }
            let mut values = vec![0u8; self.n];
            for i in 0..self.n {
                values[i] = match do_.iter().find(|(v, _)| *v == i) {
                    Some((_, level)) => *level,
                    None => (self.f[i])(&values, &exo),
                };
            }
            let index = measured.iter().fold(0usize, |acc, v| (acc << 1) | usize::from(values[*v]));
            out[index] += weight;
        }
        out
    }

    /// `P(Y = 1 | do(do_))` for outcome `y`.
    pub fn risk(&self, do_: &[(usize, u8)], y: usize) -> f64 {
        self.law(do_, &[y])[1]
    }
}

/// One regime the fixture supplies, enumerated from the named population's model.
pub struct Spec<'a> {
    pub population: &'a str,
    pub kind: RegimeKind,
    pub assignments: Vec<(usize, u8)>,
}

pub fn vid(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

pub fn bit(b: bool) -> u8 {
    u8::from(b)
}

/// Selection diagram over `n` binary nodes.
pub fn diagram(
    n: usize,
    directed: &[(usize, usize)],
    bidirected: &[(usize, usize)],
    selections: &[usize],
) -> SelectionDiagram {
    let mut graph = Admg::with_variables(u32::try_from(n).unwrap());
    let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap());
    for (a, b) in directed {
        graph.insert_directed(dense(*a), dense(*b)).unwrap();
    }
    for (a, b) in bidirected {
        graph.insert_bidirected(dense(*a), dense(*b)).unwrap();
    }
    let selections: Arc<[VariableId]> = selections.iter().map(|i| vid(*i)).collect();
    SelectionDiagram::try_new(graph, selections).unwrap()
}

/// Catalog and exact laws, every law enumerated from `scm_for(population)`.
pub fn build<'s>(
    scm_for: &dyn Fn(&str) -> &'s Scm,
    n: usize,
    specs: &[Spec<'_>],
    populations: &[&str],
) -> (EvidenceCatalog, ExactTransportData) {
    let coordinates: Vec<VariableCoordinate> = (0..n)
        .map(|i| VariableCoordinate {
            variable: vid(i),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect();
    let environments: Vec<Environment> = populations
        .iter()
        .map(|population| Environment::try_new(*population, coordinates.clone(), []).unwrap())
        .collect();
    let mut regimes = Vec::new();
    let mut bindings = Vec::new();
    let mut laws = Vec::new();
    for (k, spec) in specs.iter().enumerate() {
        let id = RegimeId::from_raw(u32::try_from(k).unwrap());
        let measured: Vec<usize> =
            (0..n).filter(|i| !spec.assignments.iter().any(|(v, _)| v == i)).collect();
        let measured_ids: Arc<[VariableId]> = measured.iter().map(|i| vid(*i)).collect();
        let intervention_ids: Vec<VariableId> =
            spec.assignments.iter().map(|(v, _)| vid(*v)).collect();
        let intervention_values: Vec<InterventionAssignment> = spec
            .assignments
            .iter()
            .map(|(v, level)| InterventionAssignment {
                variable: vid(*v),
                value: Value::Bool(*level == 1),
            })
            .collect();
        regimes.push(
            EvidenceRegime::try_new(
                id,
                spec.kind,
                EvidenceKind::Available,
                intervention_ids,
                intervention_values,
                Arc::clone(&measured_ids),
                spec.population,
                DistributionAvailability::Joint,
            )
            .unwrap(),
        );
        let snapshot = format!("{}-{k}", spec.population);
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: id,
            snapshot_identity: Arc::from(snapshot.as_str()),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let axes: Vec<DiscreteAxis> = measured
            .iter()
            .map(|i| DiscreteAxis {
                variable: vid(*i),
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            })
            .collect();
        let probabilities = scm_for(spec.population).law(&spec.assignments, &measured);
        let interventions: Vec<ExprInterventionAssignment> = spec
            .assignments
            .iter()
            .map(|(v, level)| {
                ExprInterventionAssignment::concrete(vid(*v), Value::Bool(*level == 1))
            })
            .collect();
        laws.push(
            ExactDiscreteLaw::try_new(
                spec.population,
                id,
                interventions,
                axes,
                probabilities,
                snapshot,
                LawTolerance::default(),
            )
            .unwrap(),
        );
    }
    let catalog = EvidenceCatalog::try_new(environments, regimes, bindings, None).unwrap();
    let data = ExactTransportData::try_new(laws, 4096).unwrap();
    (catalog, data)
}

/// The full source experiment family over `controllable` (every nonempty subset
/// at every binary assignment) plus the observational laws of `populations`.
pub fn family_specs<'a>(
    source: &'a str,
    controllable: &[usize],
    observational: &[&'a str],
) -> Vec<Spec<'a>> {
    let mut specs = observational
        .iter()
        .map(|population| Spec {
            population,
            kind: RegimeKind::Observational,
            assignments: Vec::new(),
        })
        .collect::<Vec<_>>();
    for mask in 1..(1usize << controllable.len()) {
        let variables: Vec<usize> = controllable
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, v)| *v)
            .collect();
        for levels in 0..(1usize << variables.len()) {
            let assignments = variables
                .iter()
                .enumerate()
                .map(|(bit, v)| (*v, u8::from(levels & (1 << bit) != 0)))
                .collect();
            specs.push(Spec { population: source, kind: RegimeKind::Experimental, assignments });
        }
    }
    specs
}

/// `P(Y = 1)` of an evaluated distribution whose only outcome is binary.
pub fn risk_of(distribution: &ExactDistribution) -> f64 {
    distribution
        .atoms
        .iter()
        .zip(distribution.probabilities.iter())
        .filter(|(atom, _)| atom[0] == Value::Bool(true))
        .map(|(_, p)| *p)
        .sum()
}

/// Decide, bind, and return the checked derivation with its bound functional.
pub fn identify_and_bind(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> (ZTransportDerivation, antecedent_identify::BoundZTransportFunctional) {
    let ctx = ExecutionContext::for_tests(1);
    let decision =
        decide_z_transport_with_catalog(diagram, query, catalog, SidLimits::default(), &ctx)
            .unwrap();
    let ZTransportDecision::Identified(derivation) = decision else {
        panic!("expected an identified z-transport query, got {decision:?}");
    };
    let bound = bind_z_transport_catalog(diagram, query, &derivation, catalog).unwrap();
    (*derivation, bound)
}

/// Evaluate `P(Y = 1 | do(treatment = level))` on exact laws.
pub fn evaluate_risk(
    bound: &antecedent_identify::BoundZTransportFunctional,
    data: &ExactTransportData,
    treatment: usize,
    level: bool,
) -> Result<f64, antecedent_expr::EvalError> {
    let ctx = ExecutionContext::for_tests(1);
    let request = Assignment::from_pairs([(vid(treatment), Value::Bool(level))]);
    evaluate_exact_z_transport(bound, data.clone(), request, ExactEvaluationLimits::default(), &ctx)
        .map(|distribution| risk_of(&distribution))
}

/// A z-transport query with one outcome and one treatment.
pub fn query(
    outcome: usize,
    treatment: usize,
    controllable: &[usize],
    assignment: &[(usize, bool)],
) -> ZTransportQuery {
    ZTransportQuery {
        outcomes: Arc::from([vid(outcome)]),
        treatments: Arc::from([vid(treatment)]),
        controllable: controllable.iter().map(|v| vid(*v)).collect(),
        experiment_assignment: assignment
            .iter()
            .map(|(v, level)| InterventionAssignment {
                variable: vid(*v),
                value: Value::Bool(*level),
            })
            .collect(),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}
