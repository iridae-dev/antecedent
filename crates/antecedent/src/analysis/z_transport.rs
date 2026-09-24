//! Prepared execution for the registered point-only z-transport specialization.
//!
//! This route intentionally has a separate state type from classical transport:
//! its checked proof covers only one surrogate graph and is not a complete sIDz
//! decision procedure.
use super::StudyBuilder;
use antecedent_core::{ExecutionContext, TheoremScope};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
    FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{BoundZTransportFunctional, verify_z_transport_derivation};
use antecedent_io::IoError;
use std::sync::Arc;

use super::transport_common::err;

/// Prepared exact or empirical plug-in evaluation of the registered zTR graph.
#[derive(Clone, Debug)]
pub struct PreparedZTransport {
    diagram: SelectionDiagram,
    functional: BoundZTransportFunctional,
    program: FunctionalProgram,
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

    /// Export this point result with its checked proof, catalog, and source laws.
    pub fn export(&self, prepared: &PreparedZTransport) -> Result<Vec<u8>, IoError> {
        let wire = antecedent_io::z_transport_artifact::ZTransportArtifactWire::checked(
            &prepared.diagram,
            &prepared.functional,
            &prepared.data,
            &prepared.request,
            prepared.limits,
            &self.distribution,
            &prepared.program,
        )?;
        wire.export()
    }
}

/// Independently consume a z-transport artifact and recompute its point result.
pub fn consume_z_transport_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(SelectionDiagram, ZTransportResult), IoError> {
    let (diagram, distribution) =
        antecedent_io::z_transport_artifact::ZTransportArtifactWire::consume(bytes, ctx)?;
    Ok((diagram, ZTransportResult { distribution }))
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
        verify_plan_program(&self.plan, &self.program)?;
        let distribution = self.plan.evaluate(ctx).map_err(err)?;
        Ok(ZTransportResult { distribution })
    }

    /// Replace providers after revalidating their catalog bindings and compile a
    /// fresh physical plan against the retained checked program.
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
        verify_plan_program(&plan, &self.program)?;
        Ok(Self {
            diagram: self.diagram.clone(),
            functional: self.functional.clone(),
            program: self.program.clone(),
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

    /// Checked owner for the provider-bound executable expression.
    #[must_use]
    pub fn program(&self) -> &FunctionalProgram {
        &self.program
    }

    /// Retained providers for explicit refresh or inspection.
    #[must_use]
    pub fn data(&self) -> &ExactTransportData {
        &self.data
    }
}

fn verify_plan_program(
    plan: &ExactEvaluationPlan,
    program: &FunctionalProgram,
) -> Result<(), IoError> {
    if plan.root() != program.mapping().executable
        || antecedent_io::expr_arena_to_wire(plan.arena())?
            != antecedent_io::expr_arena_to_wire(program.arena())?
    {
        return Err(err("z_transport.physical_plan_program_mismatch"));
    }
    Ok(())
}

fn checked_program(functional: &BoundZTransportFunctional) -> Result<FunctionalProgram, IoError> {
    let variables = functional
        .catalog()
        .environments
        .iter()
        .flat_map(|environment| environment.variables.iter())
        .map(|coordinate| {
            (
                coordinate.variable,
                ProgramVariable { name: Arc::from(format!("v{}", coordinate.variable.raw())) },
            )
        })
        .collect::<Vec<_>>();
    let program = FunctionalProgram::new(
        functional.arena().clone(),
        ProgramSchema::new(variables),
        functional.derivation().root(),
        functional.root(),
        ProgramLimits::default(),
    )
    .map_err(|error| err(format!("z_transport.functional_program: {error}")))?;
    program
        .compile()
        .map_err(|error| err(format!("z_transport.functional_program_compile: {error}")))?;
    Ok(program)
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
        let program = checked_program(&functional)?;
        Ok(PreparedZTransport { diagram, functional, program, data, request, limits, plan })
    }

    /// Prepare an empirical plugin evaluation, requiring every retained law to
    /// carry empirical counts. Results remain point-only.
    pub fn z_transport_empirical(
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedZTransport, IoError> {
        if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
            return Err(err("z_transport.empirical_counts_required"));
        }
        Self::z_transport(diagram, functional, data, request, limits, ctx)
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
            let prepared = if empirical {
                StudyBuilder::z_transport_empirical(
                    diagram,
                    functional,
                    data,
                    assignment,
                    ExactEvaluationLimits::default(),
                    &ExecutionContext::for_tests(7),
                )
            } else {
                StudyBuilder::z_transport(
                    diagram,
                    functional,
                    data,
                    assignment,
                    ExactEvaluationLimits::default(),
                    &ExecutionContext::for_tests(7),
                )
            }
            .unwrap();
            assert_eq!(
                prepared.program().mapping().source,
                prepared.functional().derivation().root()
            );
            assert_eq!(prepared.program().mapping().executable, prepared.functional().root());
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
            let artifact = result.export(&prepared).unwrap();
            let (_diagram, consumed) =
                consume_z_transport_artifact(&artifact, &ExecutionContext::for_tests(8)).unwrap();
            assert_eq!(consumed.distribution().probabilities, result.distribution().probabilities);
            let mut forged: antecedent_io::z_transport_artifact::ZTransportArtifactWire =
                antecedent_io::from_cbor(&artifact).unwrap();
            forged.result.probabilities[0] += 0.01;
            assert!(
                consume_z_transport_artifact(
                    &forged.export().unwrap(),
                    &ExecutionContext::for_tests(8),
                )
                .is_err()
            );
            let mut forged_proof: antecedent_io::z_transport_artifact::ZTransportArtifactWire =
                antecedent_io::from_cbor(&artifact).unwrap();
            forged_proof.proof.root = forged_proof.proof.root.wrapping_add(1);
            assert!(
                consume_z_transport_artifact(
                    &forged_proof.export().unwrap(),
                    &ExecutionContext::for_tests(8),
                )
                .is_err()
            );
            let mut forged_program: antecedent_io::z_transport_artifact::ZTransportArtifactWire =
                antecedent_io::from_cbor(&artifact).unwrap();
            let program = forged_program.program.as_mut().unwrap();
            program.executable = program.executable.wrapping_add(1);
            assert!(
                consume_z_transport_artifact(
                    &forged_program.export().unwrap(),
                    &ExecutionContext::for_tests(8),
                )
                .is_err()
            );
            let mut forged_schema: antecedent_io::z_transport_artifact::ZTransportArtifactWire =
                antecedent_io::from_cbor(&artifact).unwrap();
            forged_schema.program.as_mut().unwrap().variables[0].1 = "forged-coordinate".into();
            assert!(
                consume_z_transport_artifact(
                    &forged_schema.export().unwrap(),
                    &ExecutionContext::for_tests(8),
                )
                .is_err()
            );
            assert_eq!(
                prepared.theorem_scope().outcome_guarantees,
                antecedent_core::query::OutcomeGuarantee::SoundIncomplete
            );
        }
    }

    #[test]
    fn refresh_rebinds_evidence_under_the_original_checked_program() {
        let (diagram, functional, data, request) = fixture(false);
        let prepared = StudyBuilder::z_transport(
            diagram,
            functional,
            data.clone(),
            request,
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap();
        let retained_program = antecedent_io::functional_program_to_wire(prepared.program())
            .unwrap();
        let refreshed = prepared.refresh(data, &ExecutionContext::for_tests(8)).unwrap();
        assert_eq!(
            antecedent_io::functional_program_to_wire(refreshed.program()).unwrap(),
            retained_program
        );
        assert_eq!(
            refreshed.estimate(&ExecutionContext::for_tests(8)).unwrap().distribution().probabilities,
            prepared.estimate(&ExecutionContext::for_tests(7)).unwrap().distribution().probabilities
        );
        let missing = ExactTransportData::try_new(Vec::<ExactDiscreteLaw>::new(), 16).unwrap();
        assert!(prepared.refresh(missing, &ExecutionContext::for_tests(9)).is_err());
    }

    #[test]
    fn direct_joint_two_variable_route_evaluates_and_exports_exact_point() {
        let (a, b, diagram, functional, data) = direct_joint_fixture();
        let request = Assignment::from_pairs([(a, Value::Bool(true)), (b, Value::Bool(false))]);
        let prepared = StudyBuilder::z_transport(
            diagram,
            functional,
            data,
            request,
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(17),
        )
        .unwrap();
        let result = prepared.estimate(&ExecutionContext::for_tests(17)).unwrap();
        assert_eq!(result.distribution().probabilities.as_ref(), &[0.3, 0.7]);
        assert_eq!(result.interval_type(), "no_interval_reported");
        let artifact = result.export(&prepared).unwrap();
        let (_, consumed) =
            consume_z_transport_artifact(&artifact, &ExecutionContext::for_tests(18)).unwrap();
        assert_eq!(consumed.distribution().probabilities, result.distribution().probabilities);

        let (a, b, diagram, functional, data) = direct_joint_fixture();
        assert!(
            StudyBuilder::z_transport(
                diagram,
                functional,
                data,
                Assignment::from_pairs([(a, Value::Bool(false)), (b, Value::Bool(false))]),
                ExactEvaluationLimits::default(),
                &ExecutionContext::for_tests(17),
            )
            .is_err()
        );
    }

    fn direct_joint_fixture()
    -> (VariableId, VariableId, SelectionDiagram, BoundZTransportFunctional, ExactTransportData)
    {
        let (a, b, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let diagram =
            SelectionDiagram::try_new(Admg::with_variables(3), Arc::<[VariableId]>::from([]))
                .unwrap();
        let assignment = [
            antecedent_core::InterventionAssignment { variable: a, value: Value::Bool(true) },
            antecedent_core::InterventionAssignment { variable: b, value: Value::Bool(false) },
        ];
        let query = ZTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([a, b]),
            controllable: Arc::from([a, b]),
            experiment_assignment: Arc::from(assignment.clone()),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let Identified::Identified(proof) =
            identify_z_transport_surrogate(&diagram, &query).unwrap()
        else {
            panic!("direct joint regime should be identified");
        };
        let environment = Environment::try_new(
            "source",
            [a, b, y].map(|variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            }),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let regime = antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(10),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [a, b],
            assignment.clone(),
            [y],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let binding = RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(10),
            snapshot_identity: Arc::from("joint-do-a1-b0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        };
        let catalog = EvidenceCatalog::try_new([environment], [regime], [binding], None).unwrap();
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        let law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(10),
            assignment
                .iter()
                .map(|a| InterventionAssignment::concrete(a.variable, a.value.clone()))
                .collect::<Vec<_>>(),
            [DiscreteAxis {
                variable: y,
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            }],
            [0.3, 0.7],
            "joint-do-a1-b0",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 16).unwrap();
        (a, b, diagram, functional, data)
    }

    #[test]
    fn sensitivity_artifact_replays_and_rejects_tampering() {
        let (diagram, functional, data, assignment) = fixture(false);
        let prepared = StudyBuilder::z_transport(
            diagram,
            functional,
            data,
            assignment,
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(11),
        )
        .unwrap();
        let result = prepared.estimate(&ExecutionContext::for_tests(11)).unwrap();
        let baseline = result.export(&prepared).unwrap();
        let wire = super::super::ZTransportSensitivityArtifactWire::checked(
            baseline,
            0.2,
            Some(0.4),
            &ExecutionContext::for_tests(11),
        )
        .unwrap();
        let bytes = wire.export().unwrap();
        let replayed = super::super::ZTransportSensitivityArtifactWire::consume(
            &bytes,
            &ExecutionContext::for_tests(12),
        )
        .unwrap();
        assert_eq!(replayed.baseline, wire.baseline);
        assert_eq!(replayed.assumption_range, wire.assumption_range);
        assert_eq!(replayed.provider_snapshots, ["do-z-0"]);
        assert!(replayed.tipping_fraction.is_some());

        let mut changed_range = wire.clone();
        changed_range.assumption_range[0] -= 0.01;
        assert!(
            super::super::ZTransportSensitivityArtifactWire::consume(
                &changed_range.export().unwrap(),
                &ExecutionContext::for_tests(12),
            )
            .is_err()
        );
        let mut changed_baseline = wire;
        changed_baseline.baseline_artifact[0] ^= 1;
        assert!(
            super::super::ZTransportSensitivityArtifactWire::consume(
                &changed_baseline.export().unwrap(),
                &ExecutionContext::for_tests(12),
            )
            .is_err()
        );
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
