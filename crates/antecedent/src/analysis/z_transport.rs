//! Prepared execution for the registered point-only z-transport specialization.
//!
//! This route intentionally has a separate state type from classical transport:
//! its checked proof covers only one surrogate graph and is not a complete sIDz
//! decision procedure.
use super::StudyBuilder;
use antecedent_core::{ExecutionContext, TheoremScope};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{BoundZTransportFunctional, verify_z_transport_derivation};
use antecedent_io::IoError;

use super::transport_common::err;

/// Prepared exact or empirical plug-in evaluation of the registered zTR graph.
#[derive(Clone, Debug)]
pub struct PreparedZTransport {
    diagram: SelectionDiagram,
    functional: BoundZTransportFunctional,
    data: ExactTransportData,
    request: Assignment,
    limits: ExactEvaluationLimits,
    plan: ExactEvaluationPlan,
}

/// Executed point distribution with no sampling interval claim.
#[derive(Clone, Debug)]
pub struct ZTransportResult {
    distribution: ExactDistribution,
}

impl ZTransportResult {
    /// Target interventional outcome distribution.
    #[must_use]
    pub const fn distribution(&self) -> &ExactDistribution {
        &self.distribution
    }

    /// Sampling uncertainty is not reported by this route.
    #[must_use]
    pub const fn interval_type(&self) -> &'static str {
        "no_interval_reported"
    }
}

impl PreparedZTransport {
    /// The explicitly limited graph and theorem scope for this prepared route.
    #[must_use]
    pub fn theorem_scope(&self) -> TheoremScope {
        antecedent_core::TheoremScope::z_transportability()
    }

    /// Re-evaluate the frozen functional against retained source laws.
    ///
    /// Exact and empirical plugin tables share the checked evaluator; empirical
    /// tables remain point-only and do not acquire a sampling interval here.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<ZTransportResult, IoError> {
        let distribution = self.plan.evaluate(ctx).map_err(err)?;
        Ok(ZTransportResult { distribution })
    }

    /// Replace providers after revalidating their catalog bindings and compile a
    /// fresh plan against the same checked query.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let plan = antecedent_estimate::prepare_exact_z_transport(
            &self.functional,
            data.clone(),
            self.request.clone(),
            self.limits,
            ctx,
        )
        .map_err(err)?;
        Ok(Self {
            diagram: self.diagram.clone(),
            functional: self.functional.clone(),
            data,
            request: self.request.clone(),
            limits: self.limits,
            plan,
        })
    }

    /// Return the immutable query, proof and evidence authority used at prepare.
    #[must_use]
    pub fn functional(&self) -> &BoundZTransportFunctional {
        &self.functional
    }

    /// Retained providers for explicit refresh or inspection.
    #[must_use]
    pub fn data(&self) -> &ExactTransportData {
        &self.data
    }
}

impl StudyBuilder {
    /// Prepare the registered graph-specific zTR point estimator.
    ///
    /// This entry point accepts exact laws or empirical plugin tables. It does
    /// not license a general bounded sIDz search, interval estimates, or a
    /// PreparedStudy artifact/replay contract.
    pub fn z_transport(
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedZTransport, IoError> {
        verify_z_transport_derivation(
            &diagram,
            functional.derivation().query(),
            functional.derivation(),
        )
        .map_err(err)?;
        let plan = antecedent_estimate::prepare_exact_z_transport(
            &functional,
            data.clone(),
            request.clone(),
            limits,
            ctx,
        )
        .map_err(err)?;
        Ok(PreparedZTransport { diagram, functional, data, request, limits, plan })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
        RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableCoordinate,
        VariableDomain, VariableId,
    };
    use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance};
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::{
        ZTransportQuery, ZTransportResult as Identified, bind_z_transport_catalog,
        identify_z_transport_surrogate,
    };
    use std::sync::Arc;

    fn fixture(
        empirical: bool,
    ) -> (SelectionDiagram, BoundZTransportFunctional, ExactTransportData, Assignment) {
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let (w, z, x, y) = (
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        );
        let query = ZTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            controllable: Arc::from([z]),
            experiment_assignment: Arc::from([antecedent_core::InterventionAssignment {
                variable: z,
                value: Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let Identified::Identified(proof) =
            identify_z_transport_surrogate(&diagram, &query).unwrap()
        else {
            panic!("fixture proof")
        };
        let environment = Environment::try_new(
            "source",
            [w, z, x, y].map(|variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            }),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let measured: Arc<[VariableId]> = Arc::from([w, z, x, y]);
        let regimes = [false, true].map(|level| {
            antecedent_core::EvidenceRegime::try_new(
                RegimeId::from_raw(u32::from(level)),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [z],
                [antecedent_core::InterventionAssignment {
                    variable: z,
                    value: Value::Bool(level),
                }],
                Arc::clone(&measured),
                "source",
                DistributionAvailability::Joint,
            )
            .unwrap()
        });
        let bindings = [false, true].map(|level| RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(u32::from(level)),
            snapshot_identity: Arc::from(format!("do-z-{}", u8::from(level))),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let catalog = EvidenceCatalog::try_new([environment], regimes, bindings, None).unwrap();
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        let mut probabilities = Vec::with_capacity(8);
        let mut counts = Vec::with_capacity(8);
        for wv in [false, true] {
            for xv in [false, true] {
                for yv in [false, true] {
                    let p = (if wv { 0.25_f64 } else { 0.75 })
                        * (if xv { 0.35_f64 } else { 0.65 })
                        * (if yv == xv { 0.80_f64 } else { 0.20 });
                    probabilities.push(p);
                    counts.push((p * 10_000.0).round() as u64);
                }
            }
        }
        if empirical {
            let total = counts.iter().sum::<u64>() as f64;
            probabilities = counts.iter().map(|count| *count as f64 / total).collect();
        }
        let mut law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(0),
            [InterventionAssignment::concrete(z, Value::Bool(false))],
            [w, x, y].map(|variable| DiscreteAxis {
                variable,
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            }),
            probabilities,
            "do-z-0",
            LawTolerance::default(),
        )
        .unwrap();
        if empirical {
            law = law.with_empirical_counts(counts).unwrap();
        }
        let data = ExactTransportData::try_new([law], 128).unwrap();
        let assignment = Assignment::from_pairs([(x, Value::Bool(false))]);
        (diagram, functional, data, assignment)
    }

    #[test]
    fn prepared_exact_and_empirical_z_routes_match_known_truth_point_only() {
        for empirical in [false, true] {
            let (diagram, functional, data, assignment) = fixture(empirical);
            let prepared = StudyBuilder::z_transport(
                diagram,
                functional,
                data,
                assignment,
                ExactEvaluationLimits::default(),
                &ExecutionContext::for_tests(7),
            )
            .unwrap();
            let result = prepared.estimate(&ExecutionContext::for_tests(7)).unwrap();
            let true_mass = result
                .distribution()
                .atoms
                .iter()
                .zip(result.distribution().probabilities.iter())
                .filter(|(atom, _)| atom[0] == Value::Bool(true))
                .map(|(_, p)| p)
                .sum::<f64>();
            assert!((true_mass - 0.20).abs() < if empirical { 0.002 } else { 1e-12 });
            assert_eq!(result.interval_type(), "no_interval_reported");
            assert_eq!(
                prepared.theorem_scope().outcome_guarantees,
                antecedent_core::query::OutcomeGuarantee::SoundIncomplete
            );
        }
    }

    #[test]
    fn prepared_z_route_refuses_a_request_missing_treatment_binding() {
        let (diagram, functional, data, _) = fixture(false);
        let wrong = Assignment::new();
        assert!(
            StudyBuilder::z_transport(
                diagram,
                functional,
                data,
                wrong,
                ExactEvaluationLimits::default(),
                &ExecutionContext::for_tests(7),
            )
            .is_err()
        );
    }
}
