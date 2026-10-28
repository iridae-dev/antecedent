//! Selective execution of the fully declared finite SCM rows.
//!
//! An [`FiniteScmExecutor`] retains the abduction of each unit under its
//! canonical dependency identity. A request that only changes the actions
//! (worlds, routes, quantity) finds every unit's abduction retained and
//! reuses it together with the shared exogenous states; a request that changes
//! the factual evidence, a mechanism table or exogenous law, the structure, the
//! population or the history has a different identity and abduces again.
//! Reuse is never a license: the matrix, the closed readings and the joint-law
//! rule are applied to every request before any work, reused or not.
//!
//! The row is exact: nothing is sampled, so `shared_draws` is always zero and no
//! interval or posterior is produced (those readings are closed; see
//! `docs/cross-world-execution.md` for the recorded calibration design that
//! would be required to open a sampled route). Up to the joint-state bound the
//! exogenous posterior is enumerated and retained; past it the same quantity is
//! computed by exact variable elimination ([`super::factorized`]), which retains
//! nothing (its work depends on the worlds and the quantity) and records its plan
//! in [`FiniteExecutionResult::elimination`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading, check_propagation,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::factorized::{EliminationWork, infer_unit};
use super::finite_scm::{
    CompensatedSum, CompiledScm, FactualUnit, FiniteQuantity, FiniteScm, FiniteWorld,
    MAX_ABDUCTION_WORK, MAX_EVALUATION_WORK, Posterior, WorldPlan, abduct_unit, evaluate_worlds,
    validate_units, validate_worlds,
};
use super::matrix::{
    CapabilityStatus, ConditioningKind, EvidenceKind, HistoryKind, InterventionKind, MatrixRefusal,
    MatrixRequest, MatrixRow, ModelClass, PopulationKind, QuantityClass, decide,
};
use super::refusal::{
    CrossWorldExecutionError, from_external, invalid, poll, refuse, refuse_missing,
};
use super::stages::{
    ExecutionStage, PURPOSE_POSTERIOR_STATE, StageReceipt, StageStatus, WorkCounters,
    stream_identity,
};

/// Default most abductions an executor retains; the smallest identity is evicted. A caller
/// with another budget builds the executor with [`FiniteScmExecutor::with_max_retained`].
pub const MAX_RETAINED: usize = 4096;
/// The only claim of this row.
pub const INFERENCE_LABEL: &str = "point_only";

/// The declared coupling of the worlds' exogenous terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoiseCoupling {
    /// Every world reads the one abduced exogenous posterior state of the unit.
    SharedAbducedExogenous,
    /// Each world is declared to draw its exogenous terms independently: a
    /// interventional, not a counterfactual, joint law.
    IndependentWorldNoise,
    /// Nothing declared.
    Undeclared,
}

impl NoiseCoupling {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SharedAbducedExogenous => "shared_abduced_exogenous",
            Self::IndependentWorldNoise => "independent_world_noise",
            Self::Undeclared => "undeclared",
        }
    }
}

/// How the exogenous posterior is obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExogenousInference {
    /// Exact enumeration of the joint exogenous states.
    ExactEnumeration,
    /// Posterior sampling: closed, no coverage record exists.
    PosteriorSampling,
}

/// How an execution obtained the exogenous posterior it evaluated the worlds under.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExogenousMethod {
    /// Every joint exogenous state was enumerated; the posterior is retained per unit.
    JointEnumeration,
    /// The joint space exceeded the enumeration bound: the quantity was computed exactly by
    /// variable elimination over the factor graph, and nothing is retained.
    VariableElimination,
}

impl ExogenousMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::JointEnumeration => "joint_enumeration",
            Self::VariableElimination => "variable_elimination",
        }
    }
}

/// The plan and work of an exact variable-elimination execution, over all its units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EliminationReceipt {
    /// Units eliminated.
    pub units: u64,
    /// Most variables of one unit's factor graph.
    pub max_nodes: u64,
    /// Most factors one unit's graph started with.
    pub max_factors: u64,
    /// Entries of the largest table formed.
    pub peak_table_entries: u64,
    /// Multiply-accumulate operations over all units.
    pub multiplications: u64,
    /// Digest over every unit's elimination order.
    pub plan_digest: String,
}

/// A cross-world request over a fully declared finite SCM.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteExecutionRequest {
    /// The mechanisms, declared for one population and history.
    pub scm: FiniteScm,
    /// History or regime the factual evidence was gathered under.
    pub evidence_history: String,
    /// Factual evidence, one entry per unit.
    pub units: Vec<FactualUnit>,
    /// The named worlds.
    pub worlds: Vec<FiniteWorld>,
    /// The requested quantity.
    pub quantity: FiniteQuantity,
    /// The declared coupling.
    pub coupling: NoiseCoupling,
    /// How the exogenous posterior is obtained.
    pub exogenous: ExogenousInference,
    /// Target population, when different from the mechanisms' population.
    pub target_population: Option<String>,
    /// The reading asked for; only a point is licensed.
    pub reading: RequestedReading,
}

/// One unit's retained abduction.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedAbduction {
    /// The abduction identity.
    pub identity: String,
    /// The unit.
    pub unit: String,
    /// Digest of the mechanisms it was abduced under.
    pub mechanism_digest: String,
    /// Digest of the unit's factual evidence.
    pub evidence_digest: String,
    /// The exogenous posterior (the shared exogenous states).
    pub posterior: Posterior,
    /// Digest of the posterior content with its semantic stream identities.
    pub draws_digest: String,
}

/// One unit's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitAnswer {
    /// The unit.
    pub unit: String,
    /// The quantity for this unit.
    pub value: f64,
    /// `P(evidence)` under the mechanisms.
    pub evidence_probability: f64,
}

/// The identities of one execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReceipt {
    /// Mechanism digest.
    pub mechanism_digest: String,
    /// Digest over every unit's abduction identity (sorted by unit).
    pub abduction_digest: String,
    /// Digest over every unit's shared-exogenous content.
    pub exogenous_digest: String,
    /// Digest of the world definitions in order.
    pub world_digest: String,
    /// Digest of the action-specific prediction inputs.
    pub prediction_digest: String,
    /// Digest of the summary inputs.
    pub summary_digest: String,
    /// Declared coupling name.
    pub coupling: String,
    /// The overall receipt digest.
    pub digest: String,
}

/// The answer of one execution.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteExecutionResult {
    /// The matrix row that licensed the request.
    pub matrix_row: &'static str,
    /// Mean of the unit values (units in id order).
    pub value: f64,
    /// Per-unit values, sorted by unit id.
    pub units: Vec<UnitAnswer>,
    /// Stage identities and statuses.
    pub stages: Vec<StageReceipt>,
    /// Work observed in this execution.
    pub work: WorkCounters,
    /// Identities.
    pub receipt: ExecutionReceipt,
    /// The inference label (`point_only`).
    pub label: &'static str,
    /// How the exogenous posterior was obtained.
    pub method: ExogenousMethod,
    /// The elimination plan and work, when [`Self::method`] is variable elimination.
    pub elimination: Option<EliminationReceipt>,
}

/// A selective executor over the fully declared finite SCM rows.
#[derive(Debug)]
pub struct FiniteScmExecutor {
    retained: BTreeMap<String, Arc<RetainedAbduction>>,
    counters: WorkCounters,
    max_retained: usize,
}

impl Default for FiniteScmExecutor {
    fn default() -> Self {
        Self::with_max_retained(MAX_RETAINED)
    }
}

fn digest(tag: &str) -> CanonicalWriter {
    CanonicalWriter::new(&format!("antecedent.counterfactual.selective.{tag}.v1"))
}

fn abduction_identity(
    mechanism: &str,
    evidence: &str,
    shared_exogenous: bool,
    method: ExogenousMethod,
) -> String {
    let structure =
        if shared_exogenous { "shared_exogenous_joint" } else { "independent_exogenous_markov" };
    let inference = match method {
        ExogenousMethod::JointEnumeration => "exact_enumeration",
        ExogenousMethod::VariableElimination => "exact_variable_elimination",
    };
    let mut w = digest("abduction");
    w.text(mechanism).text(evidence).text(inference).text(structure);
    w.finish_hex()
}

fn draws_digest(identity: &str, scm: &FiniteScm, unit: &str, posterior: &Posterior) -> String {
    let mut w = digest("draws");
    w.text(identity).text(unit);
    w.word(scm.exogenous.len() as u64);
    for exo in &scm.exogenous {
        w.text(&stream_identity(unit, &exo.name, PURPOSE_POSTERIOR_STATE));
    }
    w.word(posterior.states.len() as u64);
    for (state, weight) in &posterior.states {
        w.word(u64::from(*state)).real(*weight);
    }
    w.real(posterior.evidence_probability);
    w.finish_hex()
}

fn classify(request: &FiniteExecutionRequest, involved: &[usize]) -> QuantityClass {
    let routed = involved.iter().any(|w| !request.worlds[*w].routes.is_empty());
    if routed {
        QuantityClass::NaturalPathSpecific
    } else if involved.len() >= 2 && matches!(request.quantity, FiniteQuantity::JointProbability(_))
    {
        QuantityClass::CounterfactualJoint
    } else {
        QuantityClass::CounterfactualMarginal
    }
}

fn matrix_error(refusal: MatrixRefusal) -> CrossWorldExecutionError {
    refuse_missing(
        refusal.code,
        refusal.detail,
        Some(refusal.subject.to_owned()),
        refusal.remedy,
        refusal.missing,
    )
}

impl FiniteScmExecutor {
    /// An executor with nothing retained that keeps at most [`MAX_RETAINED`] abductions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An executor with nothing retained that keeps at most `max_retained` abductions
    /// (the smallest identity is evicted first). Zero retains nothing, so every request
    /// abduces again.
    #[must_use]
    pub fn with_max_retained(max_retained: usize) -> Self {
        Self { retained: BTreeMap::new(), counters: WorkCounters::default(), max_retained }
    }

    /// The most abductions this executor retains.
    #[must_use]
    pub const fn max_retained(&self) -> usize {
        self.max_retained
    }

    /// Cumulative observed work.
    #[must_use]
    pub fn counters(&self) -> WorkCounters {
        self.counters
    }

    /// Number of retained abductions.
    #[must_use]
    pub fn retained_len(&self) -> usize {
        self.retained.len()
    }

    /// The retained abduction of an identity.
    #[must_use]
    pub fn retained(&self, identity: &str) -> Option<&RetainedAbduction> {
        self.retained.get(identity).map(AsRef::as_ref)
    }

    /// Retained abductions in identity order (for export).
    pub fn retained_all(&self) -> impl Iterator<Item = &RetainedAbduction> {
        self.retained.values().map(AsRef::as_ref)
    }

    /// Drop every retained abduction (a process boundary).
    pub fn clear(&mut self) {
        self.retained.clear();
    }

    fn static_checks(request: &FiniteExecutionRequest) -> Result<(), CrossWorldExecutionError> {
        if request.exogenous == ExogenousInference::PosteriorSampling {
            return Err(refuse(
                antecedent_core::reason_code!("cell_not_licensed"),
                "cross_world_execution.sampled_abduction_not_licensed",
                Some("posterior_sampling".to_owned()),
                "this row enumerates the exogenous posterior exactly; a sampled posterior has \
                 no measured coverage record for the whole abduction-action-prediction path, \
                 so request exact enumeration",
            ));
        }
        if matches!(request.reading, RequestedReading::Spread | RequestedReading::Envelope) {
            return Err(refuse(
                antecedent_core::reason_code!("cell_not_licensed"),
                "cross_world_execution.reading_not_licensed",
                Some(format!("{:?}", request.reading).to_lowercase()),
                "this row returns an exact point value only and computes no spread or \
                 envelope; request the point reading",
            ));
        }
        if let Some(target) = &request.target_population {
            if target != &request.scm.population {
                let refused = MatrixRequest {
                    quantity: QuantityClass::TransportedCrossWorld,
                    model: ModelClass::FiniteMarkovianScm,
                    conditioning: ConditioningKind::FactualEvidence,
                    interventions: InterventionKind::Hard,
                    population: PopulationKind::Transported,
                    history: HistoryKind::NoHistory,
                    evidence: EvidenceKind::DeclaredMechanisms,
                };
                return decide(&refused).map(|_| ()).map_err(matrix_error);
            }
        }
        if request.evidence_history != request.scm.history {
            let refused = MatrixRequest {
                quantity: QuantityClass::CounterfactualMarginal,
                model: ModelClass::FiniteMarkovianScm,
                conditioning: ConditioningKind::FactualEvidence,
                interventions: InterventionKind::Hard,
                population: PopulationKind::Single,
                history: HistoryKind::Incompatible,
                evidence: EvidenceKind::DeclaredMechanisms,
            };
            return decide(&refused).map(|_| ()).map_err(matrix_error);
        }
        Ok(())
    }

    fn check_pairing(
        request: &FiniteExecutionRequest,
        row: &MatrixRow,
        involved: &[usize],
        evidence_ids: &str,
        abduction_set: &str,
    ) -> Result<(), CrossWorldExecutionError> {
        let shared = row.needs_shared_exogenous();
        let (operation, inputs, dependence) = if shared {
            let alignment = match request.coupling {
                NoiseCoupling::SharedAbducedExogenous => InputAlignment::Aligned {
                    alignment_id: format!("shared_exogenous:{abduction_set}"),
                },
                _ => InputAlignment::IndependentMarginals,
            };
            let inputs = involved
                .iter()
                .map(|w| PropagationInput {
                    id: request.worlds[*w].name.clone(),
                    evidence_id: evidence_ids.to_owned(),
                    role: InputRole::Estimate,
                    alignment: alignment.clone(),
                })
                .collect();
            let dependence = match request.coupling {
                NoiseCoupling::SharedAbducedExogenous => PropagationDependence::AlignedJointLaw,
                NoiseCoupling::IndependentWorldNoise => PropagationDependence::DeclaredIndependent,
                NoiseCoupling::Undeclared => PropagationDependence::Unknown,
            };
            (PropagationOperation::NonlinearFunctional, inputs, dependence)
        } else {
            let input = PropagationInput {
                id: row.id.to_owned(),
                evidence_id: evidence_ids.to_owned(),
                role: InputRole::Estimate,
                alignment: InputAlignment::Aligned { alignment_id: abduction_set.to_owned() },
            };
            (
                PropagationOperation::LinearTransform,
                vec![input],
                PropagationDependence::AlignedJointLaw,
            )
        };
        check_propagation(&PropagationRequest {
            operation,
            inputs,
            dependence,
            reading: request.reading,
        })
        .map(|_| ())
        .map_err(from_external)
    }

    /// Execute a request, reusing every retained abduction whose identity
    /// matches.
    ///
    /// Scientific checks (readings, population, history, matrix, coupling) run
    /// before any enumeration; a refusal or cancellation leaves the retained
    /// abductions exactly as they were.
    ///
    /// # Errors
    /// A typed refusal ([`CrossWorldExecutionError::Refused`]) or cancellation.
    #[allow(clippy::too_many_lines, reason = "one sequential pipeline of five named stages")]
    pub fn execute(
        &mut self,
        request: &FiniteExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<FiniteExecutionResult, CrossWorldExecutionError> {
        poll(ctx)?;
        Self::static_checks(request)?;
        let compiled = request.scm.compile_layout()?;
        let units = validate_units(&request.scm, &request.units)?;
        validate_worlds(&request.scm, &request.worlds)?;
        request.quantity.validate(&request.scm, request.worlds.len())?;
        let method = if compiled.enumerable() {
            ExogenousMethod::JointEnumeration
        } else {
            ExogenousMethod::VariableElimination
        };
        if method == ExogenousMethod::JointEnumeration {
            if units.len().saturating_mul(compiled.joint_states) > MAX_ABDUCTION_WORK {
                return Err(invalid("cross_world_execution.bounds_exceeded", "abduction_work"));
            }
            if compiled
                .joint_states
                .saturating_mul(request.scm.variables.len())
                .saturating_mul(request.worlds.len())
                > MAX_EVALUATION_WORK
            {
                return Err(invalid("cross_world_execution.bounds_exceeded", "evaluation_work"));
            }
        }
        let involved = request.quantity.involved_worlds(&request.worlds);
        let conditioning = if units.iter().all(|u| u.observed.is_empty()) {
            ConditioningKind::NoEvidence
        } else {
            ConditioningKind::FactualEvidence
        };
        let row = decide(&MatrixRequest {
            quantity: classify(request, &involved),
            model: if compiled.shared_exogenous {
                ModelClass::FiniteSharedExogenousScm
            } else {
                ModelClass::FiniteMarkovianScm
            },
            conditioning,
            interventions: InterventionKind::Hard,
            population: PopulationKind::Single,
            history: HistoryKind::NoHistory,
            evidence: EvidenceKind::DeclaredMechanisms,
        })
        .map_err(matrix_error)?;
        debug_assert_eq!(row.status, CapabilityStatus::Executed);

        let mechanism = request.scm.digest();
        let identities: Vec<(String, String)> = units
            .iter()
            .map(|u| {
                let evidence = u.digest();
                (
                    abduction_identity(&mechanism, &evidence, compiled.shared_exogenous, method),
                    evidence,
                )
            })
            .collect();
        let mut set = digest("abduction_set");
        for (identity, _) in &identities {
            set.text(identity);
        }
        let abduction_set = set.finish_hex();
        let mut evidence_set = digest("evidence_set");
        for (_, evidence) in &identities {
            evidence_set.text(evidence);
        }
        let evidence_ids = evidence_set.finish_hex();
        Self::check_pairing(request, row, &involved, &evidence_ids, &abduction_set)?;
        if method == ExogenousMethod::VariableElimination {
            return self.execute_eliminated(
                request,
                &compiled,
                &units,
                row,
                (&mechanism, &abduction_set),
                &identities,
                ctx,
            );
        }

        // Abduction (stage 2) and exogenous draws (stage 3): reuse by identity.
        let before = self.counters;
        let mut counters = self.counters;
        let mut resolved: Vec<Arc<RetainedAbduction>> = Vec::with_capacity(units.len());
        let mut computed = 0usize;
        for (unit, (identity, evidence)) in units.iter().zip(&identities) {
            poll(ctx)?;
            let reusable = self.retained.get(identity).filter(|r| {
                r.unit == unit.unit
                    && r.mechanism_digest == mechanism
                    && r.evidence_digest == *evidence
                    && r.draws_digest
                        == draws_digest(identity, &request.scm, &unit.unit, &r.posterior)
            });
            if let Some(found) = reusable {
                counters.abductions_reused += 1;
                resolved.push(Arc::clone(found));
                continue;
            }
            let posterior = abduct_unit(&compiled, unit, ctx)?;
            counters.abductions += 1;
            counters.abduction_states_enumerated += compiled.joint_states as u64;
            computed += 1;
            let draws = draws_digest(identity, &request.scm, &unit.unit, &posterior);
            resolved.push(Arc::new(RetainedAbduction {
                identity: identity.clone(),
                unit: unit.unit.clone(),
                mechanism_digest: mechanism.clone(),
                evidence_digest: evidence.clone(),
                posterior,
                draws_digest: draws,
            }));
        }

        // Action-specific prediction (stage 4) and summary (stage 5).
        let plans: Vec<WorldPlan> = (0..request.worlds.len())
            .map(|w| WorldPlan::new(&request.scm, &request.worlds, w))
            .collect();
        let n_vars = request.scm.variables.len();
        let mut values = vec![vec![0u8; n_vars]; request.worlds.len()];
        let mut answers = Vec::with_capacity(units.len());
        let mut total = CompensatedSum::default();
        for (unit, retained) in units.iter().zip(&resolved) {
            poll(ctx)?;
            let mut expectation = CompensatedSum::default();
            for (state, weight) in &retained.posterior.states {
                evaluate_worlds(&compiled, &plans, *state as usize, &mut values);
                expectation.add(weight * request.quantity.at_state(&values));
            }
            let value = expectation.total();
            counters.world_predictions += request.worlds.len() as u64;
            counters.posterior_states_predicted +=
                (retained.posterior.states.len() * request.worlds.len()) as u64;
            total.add(value);
            answers.push(UnitAnswer {
                unit: unit.unit.clone(),
                value,
                evidence_probability: retained.posterior.evidence_probability,
            });
        }
        counters.summaries += 1;
        let mean = total.total() / answers.len() as f64;

        let (receipt, stages) =
            Self::receipts(request, &mechanism, &resolved, computed, &abduction_set);
        // Commit atomically after every stage succeeded.
        for entry in &resolved {
            self.retained.entry(entry.identity.clone()).or_insert_with(|| Arc::clone(entry));
        }
        while self.retained.len() > self.max_retained {
            if let Some(first) = self.retained.keys().next().cloned() {
                self.retained.remove(&first);
            }
        }
        self.counters = counters;
        Ok(FiniteExecutionResult {
            matrix_row: row.id,
            value: mean,
            units: answers,
            stages,
            work: counters.since(before),
            receipt,
            label: INFERENCE_LABEL,
            method,
            elimination: None,
        })
    }

    /// The execution past the enumeration bound: per unit, exact variable elimination of the
    /// factor graph of the exogenous terms and the per-world variables.
    ///
    /// The elimination depends on the worlds and the quantity as well as the evidence, so no
    /// abduction is retained and none is reused; every stage reports `Computed`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the identities one execution has already derived"
    )]
    fn execute_eliminated(
        &mut self,
        request: &FiniteExecutionRequest,
        compiled: &CompiledScm<'_>,
        units: &[FactualUnit],
        row: &MatrixRow,
        (mechanism, abduction_set): (&str, &str),
        identities: &[(String, String)],
        ctx: &ExecutionContext,
    ) -> Result<FiniteExecutionResult, CrossWorldExecutionError> {
        let before = self.counters;
        let mut counters = self.counters;
        // The factual world is one more world, with no interventions and no routes.
        let mut extended = request.worlds.clone();
        extended.push(FiniteWorld {
            name: "factual".to_owned(),
            interventions: Vec::new(),
            routes: Vec::new(),
        });
        let plans: Vec<WorldPlan> =
            (0..extended.len()).map(|w| WorldPlan::new(&request.scm, &extended, w)).collect();
        let mut work = EliminationWork::default();
        let mut resolved: Vec<Arc<RetainedAbduction>> = Vec::with_capacity(units.len());
        let mut answers = Vec::with_capacity(units.len());
        let mut total = CompensatedSum::default();
        let mut plan_digests = digest("elimination_plans");
        let mut elimination = EliminationReceipt {
            units: 0,
            max_nodes: 0,
            max_factors: 0,
            peak_table_entries: 0,
            multiplications: 0,
            plan_digest: String::new(),
        };
        for (unit, (identity, evidence)) in units.iter().zip(identities) {
            poll(ctx)?;
            let inferred = infer_unit(compiled, &plans, &request.quantity, unit, &mut work, ctx)?;
            counters.abductions += 1;
            counters.world_predictions += request.worlds.len() as u64;
            elimination.units += 1;
            elimination.max_nodes = elimination.max_nodes.max(inferred.nodes);
            elimination.max_factors = elimination.max_factors.max(inferred.factors);
            elimination.peak_table_entries =
                elimination.peak_table_entries.max(inferred.peak_table_entries);
            elimination.multiplications =
                elimination.multiplications.saturating_add(inferred.multiplications);
            plan_digests.text(&inferred.plan_digest);
            let posterior = Posterior {
                states: Vec::new(),
                evidence_probability: inferred.evidence_probability,
            };
            let mut draws = digest("draws");
            draws.text(identity).text(&unit.unit).text(&inferred.plan_digest);
            draws.word(request.scm.exogenous.len() as u64);
            for exo in &request.scm.exogenous {
                draws.text(&stream_identity(&unit.unit, &exo.name, PURPOSE_POSTERIOR_STATE));
            }
            draws.real(inferred.evidence_probability);
            resolved.push(Arc::new(RetainedAbduction {
                identity: identity.clone(),
                unit: unit.unit.clone(),
                mechanism_digest: mechanism.to_owned(),
                evidence_digest: evidence.clone(),
                posterior,
                draws_digest: draws.finish_hex(),
            }));
            total.add(inferred.value);
            answers.push(UnitAnswer {
                unit: unit.unit.clone(),
                value: inferred.value,
                evidence_probability: inferred.evidence_probability,
            });
        }
        elimination.plan_digest = plan_digests.finish_hex();
        counters.summaries += 1;
        let mean = total.total() / answers.len() as f64;
        let (receipt, stages) =
            Self::receipts(request, mechanism, &resolved, units.len(), abduction_set);
        self.counters = counters;
        Ok(FiniteExecutionResult {
            matrix_row: row.id,
            value: mean,
            units: answers,
            stages,
            work: counters.since(before),
            receipt,
            label: INFERENCE_LABEL,
            method: ExogenousMethod::VariableElimination,
            elimination: Some(elimination),
        })
    }

    fn receipts(
        request: &FiniteExecutionRequest,
        mechanism: &str,
        resolved: &[Arc<RetainedAbduction>],
        computed: usize,
        abduction_set: &str,
    ) -> (ExecutionReceipt, Vec<StageReceipt>) {
        let mut exo = digest("exogenous_set");
        for r in resolved {
            exo.text(&r.draws_digest);
        }
        let exogenous_digest = exo.finish_hex();
        let mut worlds = digest("world_set");
        worlds.word(request.worlds.len() as u64);
        for world in &request.worlds {
            worlds.text(&world.digest());
        }
        let world_digest = worlds.finish_hex();
        let mut prediction = digest("prediction");
        prediction.text(&exogenous_digest).text(&world_digest);
        let prediction_digest = prediction.finish_hex();
        let mut summary = digest("summary");
        summary
            .text(&prediction_digest)
            .text(&request.quantity.digest())
            .text(request.coupling.name())
            .text(INFERENCE_LABEL);
        let summary_digest = summary.finish_hex();
        let mut overall = digest("receipt");
        overall
            .text(mechanism)
            .text(abduction_set)
            .text(&exogenous_digest)
            .text(&world_digest)
            .text(&prediction_digest)
            .text(&summary_digest);
        let receipt = ExecutionReceipt {
            mechanism_digest: mechanism.to_owned(),
            abduction_digest: abduction_set.to_owned(),
            exogenous_digest: exogenous_digest.clone(),
            world_digest,
            prediction_digest: prediction_digest.clone(),
            summary_digest: summary_digest.clone(),
            coupling: request.coupling.name().to_owned(),
            digest: overall.finish_hex(),
        };
        let reuse = if computed == 0 { StageStatus::Reused } else { StageStatus::Computed };
        let stage = |stage, identity: &str, status| StageReceipt {
            stage,
            identity: identity.to_owned(),
            status,
        };
        let stages = vec![
            stage(ExecutionStage::MechanismFitting, mechanism, StageStatus::Declared),
            stage(ExecutionStage::FactualAbduction, abduction_set, reuse),
            stage(ExecutionStage::ExogenousDraws, &exogenous_digest, reuse),
            stage(ExecutionStage::ActionPrediction, &prediction_digest, StageStatus::Computed),
            stage(ExecutionStage::SummaryDecision, &summary_digest, StageStatus::Computed),
        ];
        (receipt, stages)
    }
}
