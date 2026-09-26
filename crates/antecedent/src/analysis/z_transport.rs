//! Prepared execution for the registered z-transport specialization.
//!
//! This route intentionally has a separate state type from classical transport:
//! its checked proof covers only one surrogate graph and is not a complete sIDz
//! decision procedure.
use super::StudyBuilder;
use antecedent_core::{ExecutionContext, TheoremScope};
use antecedent_estimate::ReplicatePolicy;
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
    FunctionalProgram, ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::BoundZTransportFunctional;
use antecedent_io::IoError;
use antecedent_io::z_transport_artifact::{ZTransportArtifactWire, ZTransportConsumeLimits};
use std::sync::Arc;

use super::transport_common::{err, estimate_err};

/// The facade's licensing floor for a nominal-0.95 posterior interval.
const PERCENTILE_FLOOR: ReplicatePolicy =
    ReplicatePolicy::percentile_floor(crate::result::PERCENTILE_95_MIN_REPLICATES);

/// Prepared exact or empirical evaluation of the registered zTR graph.
#[derive(Clone, Debug)]
pub struct PreparedZTransport {
    diagram: SelectionDiagram,
    functional: BoundZTransportFunctional,
    program: FunctionalProgram,
    data: ExactTransportData,
    request: Assignment,
    limits: ExactEvaluationLimits,
    plan: ExactEvaluationPlan,
    /// Prepared as an empirical plug-in: every retained law carries counts, and
    /// a refresh keeps that contract.
    empirical: bool,
}

/// Executed point distribution. An empirical cited table may also carry a
/// nominal percentile interval that is not a coverage claim.
#[derive(Clone, Debug)]
pub struct ZTransportResult {
    distribution: ExactDistribution,
    interval_method: Option<&'static str>,
    interval_reason: &'static str,
    coverage_target: Option<f64>,
    mean_intervals: Vec<(antecedent_core::VariableId, f64, f64)>,
}

impl ZTransportResult {
    /// Target interventional outcome distribution.
    #[must_use]
    pub const fn distribution(&self) -> &ExactDistribution {
        &self.distribution
    }

    /// Interval constructor, or the reason none was published. Callers that
    /// need to tell the two apart read [`Self::interval_method`] and
    /// [`Self::interval_reason`].
    #[must_use]
    pub fn interval_type(&self) -> &'static str {
        self.interval_method.unwrap_or(self.interval_reason)
    }

    /// Interval constructor when an interval is published.
    #[must_use]
    pub const fn interval_method(&self) -> Option<&'static str> {
        self.interval_method
    }

    /// Why the interval is withheld, or why a published interval is uncalibrated.
    #[must_use]
    pub const fn interval_reason(&self) -> &'static str {
        self.interval_reason
    }

    /// Nominal coverage of a published percentile interval.
    #[must_use]
    pub const fn coverage_target(&self) -> Option<f64> {
        self.coverage_target
    }

    /// Pointwise outcome-mean intervals, empty when no interval is published.
    #[must_use]
    pub fn mean_intervals(&self) -> &[(antecedent_core::VariableId, f64, f64)] {
        &self.mean_intervals
    }

    /// Combine two exact component laws whose checked graph components are independent.
    ///
    /// The caller supplies results from separate prepared component proofs. The
    /// output remains point-only; empirical component intervals are refused because
    /// their joint resampling contract is not part of this bounded route.
    ///
    /// # Errors
    /// Returns an error for interval-bearing inputs, overlapping outcomes, malformed
    /// distributions, or a Cartesian product larger than `max_atoms`.
    fn combine_independent_components(
        left: &Self,
        right: &Self,
        max_atoms: usize,
    ) -> Result<Self, IoError> {
        if left.interval_method.is_some() || right.interval_method.is_some() {
            return Err(err("z_transport.cross_source_interval_not_supported"));
        }
        let first = &left.distribution;
        let second = &right.distribution;
        if max_atoms == 0
            || first.outcomes.iter().any(|outcome| second.outcomes.contains(outcome))
            || first.atoms.len() != first.probabilities.len()
            || second.atoms.len() != second.probabilities.len()
            || first.atoms.iter().any(|atom| atom.len() != first.outcomes.len())
            || second.atoms.iter().any(|atom| atom.len() != second.outcomes.len())
            || first.probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
            || second.probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
            || (first.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || (second.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
        {
            return Err(err("z_transport.invalid_independent_component_law"));
        }
        let atom_count = first
            .atoms
            .len()
            .checked_mul(second.atoms.len())
            .filter(|count| *count <= max_atoms)
            .ok_or_else(|| err("z_transport.combination_atom_limit"))?;
        let mut outcomes = first.outcomes.to_vec();
        outcomes.extend(second.outcomes.iter().copied());
        let mut atoms = Vec::with_capacity(atom_count);
        let mut probabilities = Vec::with_capacity(atom_count);
        for (left_atom, left_probability) in first.atoms.iter().zip(first.probabilities.iter()) {
            for (right_atom, right_probability) in
                second.atoms.iter().zip(second.probabilities.iter())
            {
                let mut atom = left_atom.to_vec();
                atom.extend(right_atom.iter().cloned());
                atoms.push(Arc::from(atom));
                probabilities.push(left_probability * right_probability);
            }
        }
        let mut support = first.support.to_vec();
        support.extend(second.support.iter().cloned());
        Ok(Self {
            distribution: ExactDistribution {
                outcomes: outcomes.into(),
                atoms: atoms.into(),
                probabilities: probabilities.into(),
                support: support.into(),
            },
            interval_method: None,
            interval_reason: "no_interval_reported",
            coverage_target: None,
            mean_intervals: Vec::new(),
        })
    }

    /// Export this point result with its checked proof, catalog, and source laws.
    ///
    /// # Errors
    ///
    /// Returns an error if the proof, provider bindings, or artifact cannot be verified.
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
///
/// # Errors
///
/// Returns an error if the artifact is malformed or its checked result cannot be replayed.
pub fn consume_z_transport_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(SelectionDiagram, ZTransportResult), IoError> {
    consume_z_transport_artifact_with_limits(bytes, ZTransportConsumeLimits::default(), ctx)
}

/// [`consume_z_transport_artifact`] under explicit consumer limits; nothing the
/// artifact stores raises them.
///
/// # Errors
///
/// Returns an error if the artifact is malformed, exceeds the limits, or its
/// checked result cannot be replayed.
pub fn consume_z_transport_artifact_with_limits(
    bytes: &[u8],
    limits: ZTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<(SelectionDiagram, ZTransportResult), IoError> {
    let consumed = ZTransportArtifactWire::consume_with_limits(bytes, limits, ctx)?;
    Ok((
        consumed.diagram,
        ZTransportResult {
            distribution: consumed.distribution,
            interval_method: None,
            interval_reason: "no_interval_reported",
            coverage_target: None,
            mean_intervals: Vec::new(),
        },
    ))
}

impl PreparedZTransport {
    /// Evaluate two separately prepared formulas and combine them when the
    /// checked graph places their outcome/treatment sets in disconnected
    /// components. Exact laws produce a point-only product distribution.
    ///
    /// # Errors
    /// Refuses graph mismatches, connected/overlapping components, empirical
    /// interval results, cancellation, or a product larger than `max_atoms`.
    pub fn estimate_independent_components(
        left: &Self,
        right: &Self,
        max_atoms: usize,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportResult, IoError> {
        if !same_causal_graph(&left.diagram, &right.diagram)
            || left.functional.derivation().query().target
                != right.functional.derivation().query().target
            || left.functional.derivation().query().source
                == right.functional.derivation().query().source
            || !queries_are_disconnected_components(
                &left.diagram,
                left.functional.derivation().query(),
                right.functional.derivation().query(),
            )
        {
            return Err(err("z_transport.components_not_independent"));
        }
        let left_result = left.estimate(ctx)?;
        let right_result = right.estimate(ctx)?;
        ZTransportResult::combine_independent_components(&left_result, &right_result, max_atoms)
    }

    /// The explicitly limited graph and theorem scope for this prepared route.
    #[must_use]
    pub fn theorem_scope(&self) -> TheoremScope {
        antecedent_core::TheoremScope::z_transportability()
    }

    /// Re-evaluate the frozen functional against retained source laws.
    ///
    /// Exact laws stay point-only. When every cited law carries empirical
    /// counts, an uncalibrated equal-tail percentile bootstrap is attached.
    ///
    /// # Errors
    ///
    /// Returns an error if the checked plan or its evidence cannot be evaluated.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<ZTransportResult, IoError> {
        self.estimate_with(None, 0, ctx)
    }

    /// Attach a finite-discrete posterior interval instead of the percentile bootstrap.
    ///
    /// The point is still the empirical plug-in. `draws` below the percentile floor
    /// withholds the interval as `insufficient_bootstrap_replicates`.
    ///
    /// # Errors
    ///
    /// Returns an error if the posterior provider cannot evaluate the retained evidence.
    pub fn estimate_bayesian(
        &self,
        provider: antecedent_estimate::BayesianTransportLawProvider,
        draws: u32,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportResult, IoError> {
        self.estimate_with(Some(provider), draws, ctx)
    }

    fn estimate_with(
        &self,
        provider: Option<antecedent_estimate::BayesianTransportLawProvider>,
        draws: u32,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "z-transport estimate").map_err(estimate_err)?;
        verify_plan_program(&self.plan, &self.program)?;
        let distribution = self
            .plan
            .evaluate(ctx)
            .map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))?;
        let point_only = |reason: &'static str| ZTransportResult {
            distribution: distribution.clone(),
            interval_method: None,
            interval_reason: reason,
            coverage_target: None,
            mean_intervals: Vec::new(),
        };
        if !self.empirical {
            return Ok(point_only("no_interval_reported"));
        }
        let options = antecedent_estimate::EmpiricalTableOptions::default();
        let (method, interval) = if let Some(provider) = provider {
            (
                antecedent_estimate::POSTERIOR_EQUAL_TAIL,
                antecedent_estimate::bayesian_z_transport_interval(
                    &self.functional,
                    &self.data,
                    &self.request,
                    self.limits,
                    antecedent_estimate::statistical_transport::BayesianZTransportIntervalOptions {
                        provider,
                        draws,
                        coverage_level: options.coverage_level,
                    },
                    ctx,
                )
                .map_err(estimate_err)?,
            )
        } else {
            (
                antecedent_estimate::PERCENTILE_BOOTSTRAP,
                antecedent_estimate::nominal_z_transport_interval(
                    &self.functional,
                    &self.data,
                    &self.request,
                    self.limits,
                    options.bootstrap_replicates,
                    options.coverage_level,
                    ctx,
                )
                .map_err(estimate_err)?,
            )
        };
        let interval = match interval {
            Ok(interval) => interval,
            Err(reason) => return Ok(point_only(reason)),
        };
        if provider.is_some() {
            if let Err(reason) = PERCENTILE_FLOOR.decide(
                interval.replicates_requested,
                interval.replicates_ok,
                interval.replicates_failed,
            ) {
                return Ok(point_only(reason));
            }
        }
        Ok(ZTransportResult {
            distribution,
            interval_method: Some(method),
            interval_reason: antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED,
            coverage_target: Some(interval.coverage_target),
            mean_intervals: interval.mean_intervals.to_vec(),
        })
    }

    /// Replace providers after revalidating their catalog bindings and compile a
    /// fresh physical plan against the retained checked program.
    ///
    /// # Errors
    ///
    /// Returns an error if the refreshed evidence changes the checked program's bindings.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "z-transport refresh").map_err(estimate_err)?;
        if self.empirical {
            require_empirical_counts(&data)?;
        }
        let plan = antecedent_estimate::prepare_exact_z_transport(
            &self.functional,
            data.clone(),
            self.request.clone(),
            self.limits,
            ctx,
        )
        .map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))?;
        verify_plan_program(&plan, &self.program)?;
        Ok(Self {
            diagram: self.diagram.clone(),
            functional: self.functional.clone(),
            program: self.program.clone(),
            data,
            request: self.request.clone(),
            limits: self.limits,
            plan,
            empirical: self.empirical,
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

fn same_causal_graph(left: &SelectionDiagram, right: &SelectionDiagram) -> bool {
    let left_graph = left.causal_graph();
    let right_graph = right.causal_graph();
    left_graph.nodes() == right_graph.nodes()
        && (0..left_graph.node_count()).all(|raw| {
            let id =
                antecedent_graph::DenseNodeId::from_raw(u32::try_from(raw).expect("graph bound"));
            left_graph.children(id) == right_graph.children(id)
                && left_graph.parents(id) == right_graph.parents(id)
                && left_graph.bidirected_neighbors(id) == right_graph.bidirected_neighbors(id)
        })
}

fn queries_are_disconnected_components(
    diagram: &SelectionDiagram,
    left: &antecedent_identify::ZTransportQuery,
    right: &antecedent_identify::ZTransportQuery,
) -> bool {
    let graph = diagram.causal_graph();
    let mut labels = vec![usize::MAX; graph.node_count()];
    let mut label = 0usize;
    for root in 0..graph.node_count() {
        if labels[root] != usize::MAX {
            continue;
        }
        let root_id =
            antecedent_graph::DenseNodeId::from_raw(u32::try_from(root).expect("graph bound"));
        labels[root] = label;
        let mut stack = vec![root_id];
        while let Some(node) = stack.pop() {
            for neighbor in graph
                .children(node)
                .iter()
                .chain(graph.parents(node))
                .chain(graph.bidirected_neighbors(node))
            {
                if labels[neighbor.as_usize()] == usize::MAX {
                    labels[neighbor.as_usize()] = label;
                    stack.push(*neighbor);
                }
            }
        }
        label += 1;
    }
    let component = |query: &antecedent_identify::ZTransportQuery| -> Option<usize> {
        let variables = query.outcomes.iter().chain(query.treatments.iter()).copied();
        let mut found = None;
        for variable in variables {
            let index = graph.nodes().iter().position(|node| {
                matches!(node, antecedent_graph::NodeRef::Static(candidate) if *candidate == variable)
            })?;
            let current = labels[index];
            if found.is_some_and(|previous| previous != current) {
                return None;
            }
            found = Some(current);
        }
        found
    };
    component(left).zip(component(right)).is_some_and(|(a, b)| a != b)
}

fn verify_plan_program(
    plan: &ExactEvaluationPlan,
    program: &FunctionalProgram,
) -> Result<(), IoError> {
    if plan.root() != program.mapping().executable || plan.arena() != program.arena() {
        return Err(err("z_transport.physical_plan_program_mismatch"));
    }
    Ok(())
}

/// Refuse a law set that is not an empirical plug-in table.
fn require_empirical_counts(data: &ExactTransportData) -> Result<(), IoError> {
    if data.laws().iter().any(|law| law.empirical_counts().is_none()) {
        return Err(err("z_transport.empirical_counts_required"));
    }
    Ok(())
}

fn checked_program(functional: &BoundZTransportFunctional) -> Result<FunctionalProgram, IoError> {
    // Every environment declares the coordinates it shares with the graph; one
    // schema variable per coordinate, however many populations declare it.
    let variables = functional
        .catalog()
        .environments
        .iter()
        .flat_map(|environment| environment.variables.iter())
        .map(|coordinate| {
            (
                coordinate.variable,
                ProgramVariable {
                    name: Arc::from(antecedent_io::z_transport_artifact::program_variable_name(
                        coordinate.variable,
                    )),
                },
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
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
    ///
    /// # Errors
    ///
    /// Returns an error if the theorem proof, program, or provider bindings fail validation.
    pub fn z_transport(
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedZTransport, IoError> {
        // The bound functional was verified when derived; only its input
        // identity is rechecked here.
        functional.derivation().check_inputs(&diagram, functional.derivation().query())?;
        let plan = antecedent_estimate::prepare_exact_z_transport(
            &functional,
            data.clone(),
            request.clone(),
            limits,
            ctx,
        )
        .map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))?;
        let program = checked_program(&functional)?;
        Ok(PreparedZTransport {
            diagram,
            functional,
            program,
            data,
            request,
            limits,
            plan,
            empirical: false,
        })
    }

    /// Prepare an empirical plugin evaluation, requiring every retained law to
    /// carry empirical counts. Results remain point-only.
    ///
    /// # Errors
    ///
    /// Returns an error if a law lacks empirical counts or preparation fails.
    pub fn z_transport_empirical(
        diagram: SelectionDiagram,
        functional: BoundZTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedZTransport, IoError> {
        require_empirical_counts(&data)?;
        let mut prepared = Self::z_transport(diagram, functional, data, request, limits, ctx)?;
        prepared.empirical = true;
        Ok(prepared)
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
        SidLimits, ZTransportQuery, ZTransportResult as Identified, bind_z_transport_catalog,
        decide_z_transport_with_catalog, identify_z_transport,
    };
    use std::sync::Arc;

    fn fixture(
        empirical: bool,
    ) -> (SelectionDiagram, BoundZTransportFunctional, ExactTransportData, Assignment) {
        fixture_with_environments(empirical, false)
    }

    /// `dual_environments` declares the same coordinates for the target
    /// population too, the natural catalog shape.
    fn fixture_with_environments(
        empirical: bool,
        dual_environments: bool,
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
        let Identified::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
            panic!("fixture proof")
        };
        let coordinates = [w, z, x, y].map(|variable| VariableCoordinate {
            variable,
            domain: VariableDomain::Binary,
            unit: None,
        });
        let mut environments = vec![
            Environment::try_new("source", coordinates.clone(), Arc::<[VariableId]>::from([]))
                .unwrap(),
        ];
        if dual_environments {
            environments.push(
                Environment::try_new("target", coordinates, Arc::<[VariableId]>::from([])).unwrap(),
            );
        }
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
        let catalog = EvidenceCatalog::try_new(environments, regimes, bindings, None).unwrap();
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
                    counts.push(quantized_count(p));
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

    fn independent_component(
        graph: &SelectionDiagram,
        population: &str,
        treatment: VariableId,
        outcome: VariableId,
        treatment_level: bool,
        probability_true: f64,
    ) -> PreparedZTransport {
        let coordinates = graph
            .causal_graph()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_graph::NodeRef::Static(variable) => Some(*variable),
                _ => None,
            })
            .map(|variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let regime = antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(u32::from(treatment_level)),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [treatment],
            [antecedent_core::InterventionAssignment {
                variable: treatment,
                value: Value::Bool(treatment_level),
            }],
            [outcome],
            population,
            DistributionAvailability::Joint,
        )
        .unwrap();
        let catalog = EvidenceCatalog::try_new(
            [
                Environment::try_new("alpha", coordinates.clone(), []).unwrap(),
                Environment::try_new("beta", coordinates.clone(), []).unwrap(),
                Environment::try_new("target", coordinates, []).unwrap(),
            ],
            [regime],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(u32::from(treatment_level)),
                snapshot_identity: Arc::from(format!("{population}-joint-{treatment_level}")),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            }],
            None,
        )
        .unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([outcome]),
            treatments: Arc::from([treatment]),
            controllable: Arc::from([treatment]),
            experiment_assignment: Arc::from([antecedent_core::InterventionAssignment {
                variable: treatment,
                value: Value::Bool(treatment_level),
            }]),
            source: Arc::from(population),
            target: Arc::from("target"),
        };
        let antecedent_identify::ZTransportDecision::Identified(proof) =
            decide_z_transport_with_catalog(
                graph,
                &query,
                &catalog,
                SidLimits::default(),
                &ExecutionContext::for_tests(0),
            )
            .unwrap()
        else {
            panic!("component joint intervention must identify");
        };
        let functional = bind_z_transport_catalog(graph, &query, &proof, &catalog).unwrap();
        let law = ExactDiscreteLaw::try_new(
            population,
            RegimeId::from_raw(u32::from(treatment_level)),
            [InterventionAssignment::concrete(treatment, Value::Bool(treatment_level))],
            [DiscreteAxis {
                variable: outcome,
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            }],
            [1.0 - probability_true, probability_true],
            format!("{population}-joint-{treatment_level}"),
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 8).unwrap();
        StudyBuilder::z_transport(
            graph.clone(),
            functional,
            data,
            Assignment::from_pairs([(treatment, Value::Bool(treatment_level))]),
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(0),
        )
        .unwrap()
    }

    #[test]
    fn complementary_source_components_execute_the_exact_joint_target_effect() {
        // Exact disconnected SCM: U1,U2 are independent Uniform(0,1),
        // Z=1[U1 < 0.25 + 0.45 W], Y=1[U2 < 0.60 - 0.20 X].
        // Under the joint 0→1 contrast, E[Z+Y] changes by 0.45-0.20=0.25.
        let mut graph = Admg::with_variables(4);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let alpha_control = independent_component(
            &diagram,
            "alpha",
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            false,
            0.25,
        );
        let beta_control = independent_component(
            &diagram,
            "beta",
            VariableId::from_raw(2),
            VariableId::from_raw(3),
            false,
            0.60,
        );
        let alpha_active = independent_component(
            &diagram,
            "alpha",
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            true,
            0.70,
        );
        let beta_active = independent_component(
            &diagram,
            "beta",
            VariableId::from_raw(2),
            VariableId::from_raw(3),
            true,
            0.40,
        );
        let control = PreparedZTransport::estimate_independent_components(
            &alpha_control,
            &beta_control,
            4,
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
        let active = PreparedZTransport::estimate_independent_components(
            &alpha_active,
            &beta_active,
            4,
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
        assert_eq!(
            control.distribution().outcomes.as_ref(),
            [VariableId::from_raw(1), VariableId::from_raw(3)]
        );
        assert_eq!(control.distribution().atoms.len(), 4);
        let expected = [0.75 * 0.40, 0.75 * 0.60, 0.25 * 0.40, 0.25 * 0.60];
        for (actual, expected) in control.distribution().probabilities.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let active_expected = [0.30 * 0.60, 0.30 * 0.40, 0.70 * 0.60, 0.70 * 0.40];
        for (actual, expected) in active.distribution().probabilities.iter().zip(active_expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let sum_outcomes = |result: &ZTransportResult| {
            result
                .distribution()
                .atoms
                .iter()
                .zip(result.distribution().probabilities.iter())
                .map(|(atom, probability)| {
                    (atom[0].as_f64().unwrap() + atom[1].as_f64().unwrap()) * probability
                })
                .sum::<f64>()
        };
        let target_effect = sum_outcomes(&active) - sum_outcomes(&control);
        assert!((target_effect - 0.25).abs() < 1e-12, "target effect={target_effect}");
        assert_eq!(control.interval_reason(), "no_interval_reported");
        assert_eq!(control.coverage_target(), None);
        assert!(
            PreparedZTransport::estimate_independent_components(
                &alpha_control,
                &beta_control,
                3,
                &ExecutionContext::for_tests(0),
            )
            .is_err()
        );
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
            if empirical {
                assert_eq!(result.interval_type(), antecedent_estimate::PERCENTILE_BOOTSTRAP);
                assert_eq!(
                    result.interval_reason(),
                    antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED
                );
                let (_, lower, upper) = result.mean_intervals()[0];
                assert!(lower <= true_mass && true_mass <= upper);
            } else {
                assert_eq!(result.interval_type(), "no_interval_reported");
            }
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
            let program = &mut forged_program.program;
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
            forged_schema.program.variables[0].1 = "forged-coordinate".into();
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
    fn both_populations_may_declare_the_same_coordinates() {
        let (diagram, functional, data, assignment) = fixture_with_environments(false, true);
        let ctx = ExecutionContext::for_tests(7);
        let prepared = StudyBuilder::z_transport(
            diagram,
            functional,
            data,
            assignment,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(prepared.program().schema().variables().count(), 4);
        let result = prepared.estimate(&ctx).unwrap();
        let true_mass = result
            .distribution()
            .atoms
            .iter()
            .zip(result.distribution().probabilities.iter())
            .filter(|(atom, _)| atom[0] == Value::Bool(true))
            .map(|(_, p)| p)
            .sum::<f64>();
        assert!((true_mass - 0.20).abs() < 1e-12);
        let artifact = result.export(&prepared).unwrap();
        consume_z_transport_artifact(&artifact, &ExecutionContext::for_tests(8)).unwrap();
    }

    #[test]
    fn an_empirical_handle_refuses_a_refresh_with_count_free_laws() {
        let (diagram, functional, counted, assignment) = fixture(true);
        let (_, _, exact, _) = fixture(false);
        let ctx = ExecutionContext::for_tests(7);
        let prepared = StudyBuilder::z_transport_empirical(
            diagram,
            functional,
            counted.clone(),
            assignment,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let error = prepared.refresh(exact, &ctx).unwrap_err();
        assert!(error.to_string().contains("z_transport.empirical_counts_required"), "{error}");
        let refreshed = prepared.refresh(counted, &ctx).unwrap();
        assert_eq!(
            refreshed.estimate(&ctx).unwrap().interval_type(),
            antecedent_estimate::PERCENTILE_BOOTSTRAP
        );
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
        let retained_program =
            antecedent_io::functional_program_to_wire(prepared.program()).unwrap();
        let refreshed = prepared.refresh(data, &ExecutionContext::for_tests(8)).unwrap();
        assert_eq!(
            antecedent_io::functional_program_to_wire(refreshed.program()).unwrap(),
            retained_program
        );
        assert_eq!(
            refreshed
                .estimate(&ExecutionContext::for_tests(8))
                .unwrap()
                .distribution()
                .probabilities,
            prepared
                .estimate(&ExecutionContext::for_tests(7))
                .unwrap()
                .distribution()
                .probabilities
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
        let Identified::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
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
        assert_eq!(replayed.baseline.to_bits(), wire.baseline.to_bits());
        assert_eq!(
            replayed.assumption_range.map(f64::to_bits),
            wire.assumption_range.map(f64::to_bits)
        );
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

    fn quantized_count(probability: f64) -> u64 {
        assert!((0.0..=1.0).contains(&probability));
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "synthetic probability in [0, 1] is scaled by 10000 and rounded to a count"
        )]
        let count = (probability * 10_000.0).round() as u64;
        count
    }

    #[test]
    fn artifact_consumer_refuses_other_versions_unknown_fields_tampered_laws_and_oversize() {
        use antecedent_io::z_transport_artifact::ZTransportArtifactWire;
        let (diagram, functional, data, assignment) = fixture(true);
        let prepared = StudyBuilder::z_transport(
            diagram,
            functional,
            data,
            assignment,
            ExactEvaluationLimits::default(),
            &ExecutionContext::for_tests(11),
        )
        .unwrap();
        let bytes =
            prepared.estimate(&ExecutionContext::for_tests(11)).unwrap().export(&prepared).unwrap();
        let ctx = ExecutionContext::for_tests(12);
        let wire: ZTransportArtifactWire = antecedent_io::from_cbor(&bytes).unwrap();
        consume_z_transport_artifact(&bytes, &ctx).unwrap();

        // Another format version is refused before anything else is decoded.
        let mut other_version = wire.clone();
        other_version.version += 1;
        assert!(matches!(
            consume_z_transport_artifact(&antecedent_io::to_cbor(&other_version).unwrap(), &ctx),
            Err(IoError::UnsupportedVersion { .. })
        ));

        // An unknown top-level field is a schema violation, not an extension.
        let mut map: std::collections::BTreeMap<String, ciborium::Value> =
            ciborium::from_reader(bytes.as_slice()).unwrap();
        map.insert("extra".into(), ciborium::Value::Integer(1.into()));
        let mut extended = Vec::new();
        ciborium::into_writer(&map, &mut extended).unwrap();
        assert!(consume_z_transport_artifact(&extended, &ctx).is_err());

        // A stored law whose probabilities were changed no longer replays.
        let mut tampered = wire.clone();
        let cells = &mut tampered.laws[0].probabilities;
        cells[0] += 0.05;
        cells[1] -= 0.05;
        assert!(
            consume_z_transport_artifact(&antecedent_io::to_cbor(&tampered).unwrap(), &ctx)
                .is_err()
        );

        // A stored result that was edited is refused even though the laws replay.
        let mut forged = wire.clone();
        forged.result.probabilities[0] += 1e-9;
        assert!(
            consume_z_transport_artifact(&antecedent_io::to_cbor(&forged).unwrap(), &ctx).is_err()
        );

        // Consumer limits bind whatever the artifact declared for itself.
        let tight =
            ZTransportConsumeLimits { max_law_cells: 4, ..ZTransportConsumeLimits::default() };
        assert!(consume_z_transport_artifact_with_limits(&bytes, tight, &ctx).is_err());
        let no_laws = ZTransportConsumeLimits { max_laws: 0, ..ZTransportConsumeLimits::default() };
        assert!(consume_z_transport_artifact_with_limits(&bytes, no_laws, &ctx).is_err());
        let mut oversize = wire;
        oversize.max_support_rows = usize::MAX;
        assert!(matches!(
            consume_z_transport_artifact(&antecedent_io::to_cbor(&oversize).unwrap(), &ctx),
            Err(IoError::ZTransport(
                antecedent_io::z_transport_artifact::ZTransportArtifactError::LimitsExceeded(_)
            ))
        ));
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
