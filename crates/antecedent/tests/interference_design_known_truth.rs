//! Known truth for the design-based interference route under complete randomization, and
//! under Bernoulli assignment with the exposure mappings other than the neighbour count.
//!
//! The test enumerates the whole assignment support itself. It forms every unit's exposure with
//! its own reading of the mapping, the exact exposure probabilities from the support, and, for a
//! realized assignment, the Horvitz-Thompson, Hajek and Young-bound values by their textbook
//! formulas, without calling the estimator. Over the support the Horvitz-Thompson contrast
//! averages to the known contrast of the linear potential outcomes, and the mean Young bound
//! dominates the design variance of that contrast. The compiled route, prepared and estimated on
//! every realized assignment of the support, reproduces those values and refuses exactly the
//! assignments that leave a requested exposure level unobserved.
#![allow(
    clippy::cast_precision_loss,
    reason = "unit indexes and counts below 2^8 convert to f64 exactly"
)]
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::inference::ExposureProbabilityMethod;
use antecedent::support::CellStatus;
use antecedent::{InterferenceSpec, RefuteSuite, Study};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    InterferenceFunctional, InterferenceQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;

/// Units on the ring.
const N: usize = 8;
/// Units treated under complete randomization.
const TREATED: usize = 3;
/// Effect of a unit's own treatment on its outcome.
const OWN_EFFECT: f64 = 2.0;
/// Effect of one unit of the neighbour exposure on the outcome.
const NEIGHBOR_EFFECT: f64 = 3.0;
/// Unit-specific Bernoulli assignment probabilities.
const PROBABILITIES: [f64; N] = [0.3, 0.5, 0.4, 0.6, 0.35, 0.45, 0.55, 0.5];

#[derive(Clone, Copy, Debug)]
enum Design {
    Complete,
    Bernoulli,
}

impl Design {
    fn assignment(self) -> AssignmentDesign {
        match self {
            Self::Complete => AssignmentDesign::CompleteRandomization { treated: TREATED },
            Self::Bernoulli => {
                AssignmentDesign::Bernoulli { probabilities: Arc::from(PROBABILITIES.to_vec()) }
            }
        }
    }

    /// Every assignment the design can draw, with its probability.
    fn support(self) -> Vec<(Vec<bool>, f64)> {
        let mut support = Vec::new();
        for mask in 0_u32..(1 << N) {
            let assignment = (0..N).map(|unit| (mask >> unit) & 1 == 1).collect::<Vec<_>>();
            match self {
                Self::Complete => {
                    if assignment.iter().filter(|&&treated| treated).count() == TREATED {
                        support.push((assignment, 0.0));
                    }
                }
                Self::Bernoulli => {
                    let probability = assignment
                        .iter()
                        .zip(PROBABILITIES)
                        .map(|(&treated, p)| if treated { p } else { 1.0 - p })
                        .product();
                    support.push((assignment, probability));
                }
            }
        }
        if matches!(self, Self::Complete) {
            let each = 1.0 / support.len() as f64;
            for entry in &mut support {
                entry.1 = each;
            }
        }
        support
    }
}

/// Weight of the edge `from -> to`: unequal, so the weighted mapping differs from the fraction.
fn weight(from: usize, to: usize) -> f64 {
    1.0 + 0.5 * ((from + 2 * to) % 3) as f64
}

/// The incoming edges of every unit: its two ring neighbours.
fn incoming() -> Vec<Vec<(usize, f64)>> {
    (0..N)
        .map(|unit| {
            [(unit + N - 1) % N, (unit + 1) % N]
                .into_iter()
                .map(|from| (from, weight(from, unit)))
                .collect()
        })
        .collect()
}

fn edges() -> Vec<NetworkEdge> {
    let incoming = incoming();
    let mut edges = Vec::new();
    for (to, sources) in incoming.iter().enumerate() {
        for &(from, edge_weight) in sources {
            edges.push(NetworkEdge::new(from as u64, to as u64, edge_weight));
        }
    }
    edges
}

/// Every unit's `(own, neighbours)` exposure, read from the mapping's definition.
fn exposures(assignment: &[bool], mapping: &ExposureMapping) -> Vec<(f64, f64)> {
    incoming()
        .iter()
        .enumerate()
        .map(|(unit, sources)| {
            let own = if assignment[unit] { 1.0 } else { 0.0 };
            let treated = sources.iter().filter(|(from, _)| assignment[*from]);
            let neighbors = match mapping {
                ExposureMapping::OwnTreatment => 0.0,
                ExposureMapping::NeighborCount => treated.count() as f64,
                ExposureMapping::NeighborFraction => treated.count() as f64 / sources.len() as f64,
                ExposureMapping::WeightedNeighborExposure => {
                    let total: f64 = sources.iter().map(|(_, w)| w).sum();
                    treated.map(|(_, w)| w).sum::<f64>() / total
                }
                other @ ExposureMapping::Custom(_) => panic!("no reading of mapping {other:?}"),
            };
            (own, neighbors)
        })
        .collect()
}

fn at(exposure: (f64, f64), level: ExposureLevel) -> bool {
    (exposure.0 - level.own).abs() <= 1e-12 && (exposure.1 - level.neighbors).abs() <= 1e-12
}

/// The potential outcome of `unit` at an exposure: linear in the own and neighbour exposure.
fn potential(unit: usize, exposure: (f64, f64)) -> f64 {
    1.0 + 0.3 * unit as f64 + OWN_EFFECT * exposure.0 + NEIGHBOR_EFFECT * exposure.1
}

/// The probability, over the design, that each unit is at `level`.
fn exposure_probabilities(
    support: &[(Vec<bool>, f64)],
    mapping: &ExposureMapping,
    level: ExposureLevel,
) -> Vec<f64> {
    let mut probabilities = vec![0.0; N];
    for (assignment, probability) in support {
        for (unit, &exposure) in exposures(assignment, mapping).iter().enumerate() {
            if at(exposure, level) {
                probabilities[unit] += probability;
            }
        }
    }
    probabilities
}

/// Horvitz-Thompson mean, Hajek mean, Young bound and exposed-unit count at one level.
fn level_mean(
    outcomes: &[f64],
    observed: &[(f64, f64)],
    probabilities: &[f64],
    level: ExposureLevel,
) -> (f64, f64, f64, usize) {
    let (mut weighted, mut weights, mut bound, mut exposed) = (0.0, 0.0, 0.0, 0);
    for ((&outcome, &exposure), &p) in outcomes.iter().zip(observed).zip(probabilities) {
        if at(exposure, level) {
            weighted += outcome / p;
            weights += 1.0 / p;
            bound += (1.0 - p) * outcome * outcome / (p * p);
            exposed += 1;
        }
    }
    let hajek = if exposed == 0 { f64::NAN } else { weighted / weights };
    (weighted / N as f64, hajek, bound / N as f64, exposed)
}

/// The design-based contrast of one realized assignment, by the textbook formulas.
struct Formula {
    horvitz_thompson: f64,
    hajek: f64,
    variance: f64,
    /// Both levels are observed, so the estimator is defined.
    defined: bool,
}

struct Case {
    design: Design,
    mapping: ExposureMapping,
    from: ExposureLevel,
    to: ExposureLevel,
}

impl Case {
    fn new(design: Design, mapping: ExposureMapping, from: (f64, f64), to: (f64, f64)) -> Self {
        Self {
            design,
            mapping,
            from: ExposureLevel { own: from.0, neighbors: from.1 },
            to: ExposureLevel { own: to.0, neighbors: to.1 },
        }
    }

    fn known_contrast(&self) -> f64 {
        OWN_EFFECT * (self.to.own - self.from.own)
            + NEIGHBOR_EFFECT * (self.to.neighbors - self.from.neighbors)
    }

    fn outcomes(&self, assignment: &[bool]) -> Vec<f64> {
        exposures(assignment, &self.mapping)
            .into_iter()
            .enumerate()
            .map(|(unit, exposure)| potential(unit, exposure))
            .collect()
    }

    fn formula(&self, assignment: &[bool], from_p: &[f64], to_p: &[f64]) -> Formula {
        let outcomes = self.outcomes(assignment);
        let observed = exposures(assignment, &self.mapping);
        let (from_ht, from_hajek, from_bound, from_n) =
            level_mean(&outcomes, &observed, from_p, self.from);
        let (to_ht, to_hajek, to_bound, to_n) = level_mean(&outcomes, &observed, to_p, self.to);
        Formula {
            horvitz_thompson: to_ht - from_ht,
            hajek: to_hajek - from_hajek,
            variance: 2.0 * (from_bound + to_bound),
            defined: from_n > 0 && to_n > 0,
        }
    }

    fn query(&self) -> InterferenceQuery {
        InterferenceQuery::new(
            self.design.assignment(),
            self.mapping.clone(),
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: self.from,
                to: self.to,
            },
        )
    }

    /// Prepare and estimate the compiled study on one realized assignment.
    fn compiled(
        &self,
        assignment: &[bool],
    ) -> Result<antecedent::StudyResult, antecedent::CausalError> {
        let outcomes = self.outcomes(assignment);
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(data.clone(), edges()).unwrap();
        let ctx = ExecutionContext::for_tests(17);
        Study::tabular(data.clone())
            .graph(Dag::with_variables(1))
            .query(CausalQuery::Interference(self.query()))
            .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
            .refute(RefuteSuite::None)
            .build()?
            .prepare(&ctx)?
            .estimate(&data, &ctx)
    }

    fn check(&self) {
        let label = format!("{:?} {:?}", self.design, self.mapping);
        let support = self.design.support();
        let mass: f64 = support.iter().map(|(_, p)| p).sum();
        assert!((mass - 1.0).abs() <= 1e-12, "{label}: support mass {mass}");
        let from_p = exposure_probabilities(&support, &self.mapping, self.from);
        let to_p = exposure_probabilities(&support, &self.mapping, self.to);
        assert!(
            from_p.iter().chain(&to_p).all(|&p| p > 0.0),
            "{label}: every unit reaches both levels"
        );

        // Known truth: an undefined realization contributes zero to the Horvitz-Thompson sums,
        // so the contrast is unbiased for the potential-outcome contrast over the whole support,
        // and the mean Young bound dominates the contrast's design variance.
        let (mut mean, mut second, mut mean_bound) = (0.0, 0.0, 0.0);
        for (assignment, probability) in &support {
            let formula = self.formula(assignment, &from_p, &to_p);
            mean += probability * formula.horvitz_thompson;
            second += probability * formula.horvitz_thompson * formula.horvitz_thompson;
            mean_bound += probability * formula.variance;
        }
        let truth = self.known_contrast();
        assert!((mean - truth).abs() <= 1e-10, "{label}: E[HT] {mean} vs known {truth}");
        let design_variance = second - mean * mean;
        assert!(
            mean_bound + 1e-10 >= design_variance,
            "{label}: mean Young bound {mean_bound} below the design variance {design_variance}"
        );

        // The compiled route reproduces the formulas on every realization of the support.
        let minimum = from_p.iter().chain(&to_p).copied().fold(f64::INFINITY, f64::min);
        let mut defined = 0;
        for (assignment, _) in &support {
            let formula = self.formula(assignment, &from_p, &to_p);
            let result = self.compiled(assignment);
            if !formula.defined {
                assert!(result.is_err(), "{label}: {assignment:?} leaves a level unobserved");
                continue;
            }
            defined += 1;
            let result = result.unwrap_or_else(|error| panic!("{label}: {assignment:?}: {error}"));
            assert_eq!(result.support_status(), Some(CellStatus::Supported), "{label}");
            let estimate = result.interference().expect("an interference estimate");
            close(estimate.contrast.horvitz_thompson, formula.horvitz_thompson, &label, "HT");
            close(estimate.contrast.hajek, formula.hajek, &label, "Hajek");
            close(estimate.contrast.conservative_variance, formula.variance, &label, "bound");
            close(estimate.minimum_exposure_probability, minimum, &label, "minimum p");
            assert!(
                matches!(estimate.from_probability_method, ExposureProbabilityMethod::Exact)
                    && matches!(estimate.to_probability_method, ExposureProbabilityMethod::Exact),
                "{label}: exact probabilities"
            );
        }
        assert!(defined > 0, "{label}: some realization observes both levels");
    }
}

fn close(route: f64, formula: f64, label: &str, what: &str) {
    assert!(
        (route - formula).abs() <= 1e-10 * formula.abs().max(1.0),
        "{label} {what}: route {route} vs formula {formula}"
    );
}

#[test]
fn complete_randomization_contrasts_are_unbiased_and_the_route_reproduces_them() {
    for case in [
        Case::new(Design::Complete, ExposureMapping::OwnTreatment, (0.0, 0.0), (1.0, 0.0)),
        Case::new(Design::Complete, ExposureMapping::NeighborCount, (0.0, 1.0), (1.0, 1.0)),
        Case::new(Design::Complete, ExposureMapping::NeighborFraction, (0.0, 0.5), (1.0, 0.5)),
        Case::new(
            Design::Complete,
            ExposureMapping::WeightedNeighborExposure,
            (0.0, 0.0),
            (1.0, 1.0),
        ),
    ] {
        case.check();
    }
}

#[test]
fn bernoulli_contrasts_under_the_other_mappings_are_unbiased_and_the_route_reproduces_them() {
    for case in [
        Case::new(Design::Bernoulli, ExposureMapping::OwnTreatment, (0.0, 0.0), (1.0, 0.0)),
        Case::new(Design::Bernoulli, ExposureMapping::NeighborFraction, (0.0, 0.0), (1.0, 1.0)),
        Case::new(
            Design::Bernoulli,
            ExposureMapping::WeightedNeighborExposure,
            (0.0, 0.0),
            (1.0, 1.0),
        ),
    ] {
        case.check();
    }
}
