//! Exact-law modality of the retained common prepared-study lifecycle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{PreparedStudy, StudyBuilder};
use antecedent_core::{
    AssumptionSlot, AssumptionSource, AssumptionStatus, ExecutionContext, IdentificationSlot,
    IdentificationStatus, IdentityDomain, NodeRef, ObligationKind, ObligationRecord,
    ObligationScope, ReasoningView, SlotAvailability, SupportSlot, TheoremScope, VariableId,
};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{BoundTransportFunctional, ClassicalTransportQuery, SidLimits};
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_proof::TransportProofWire,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::transport_common::{GraphFields, digest, err, rebind_snapshots, rebuild_checked_proof};

/// Distinct identity layers for a prepared exact-law study.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactStudyIdentities {
    /// Concrete target query and intervention assignments.
    pub target: String,
    /// Frozen observation/evidence schema.
    pub observation: String,
    /// Exact numerical settings; no statistical uncertainty commitment.
    pub inference_binding: String,
    /// Graph, selections, theoretical query and evidence contract.
    pub identification: String,
    /// Checked expression and local premises.
    pub identification_product: String,
    /// Concrete target assignments and certified functional.
    pub program: String,
    /// Immutable supplied law contents and declared snapshot identities.
    pub snapshot: String,
    /// Program, snapshot and numerical execution settings.
    pub execution: String,
}

/// Native retained exact-law state. No mutable display object is execution authority.
#[derive(Clone, Debug)]
pub struct ExactPreparedState {
    diagram: SelectionDiagram,
    functional: BoundTransportFunctional,
    data: ExactTransportData,
    request: Assignment,
    limits: ExactEvaluationLimits,
    plan: ExactEvaluationPlan,
    identities: ExactStudyIdentities,
    shape: String,
    legacy_law_identity: bool,
}

/// Exact execution result, without a sampled-data estimate or fabricated uncertainty.
#[derive(Clone, Debug)]
pub struct ExactStudyResult {
    distribution: ExactDistribution,
    identities: ExactStudyIdentities,
    reasoning: ReasoningView,
}
impl ExactStudyResult {
    /// Complete target law, including supplied-law zero atoms.
    #[must_use]
    pub const fn distribution(&self) -> &ExactDistribution {
        &self.distribution
    }
    /// Executed identity layers.
    #[must_use]
    pub const fn identities(&self) -> &ExactStudyIdentities {
        &self.identities
    }
    /// All four reasoning slots, including explicitly unavailable sampling uncertainty.
    #[must_use]
    pub const fn reasoning(&self) -> &ReasoningView {
        &self.reasoning
    }
}

/// One required distribution leaf in the certified bound expression.
#[derive(Clone, Debug)]
pub struct ExactFactorRequirement {
    /// Original expression coordinate.
    pub expression: antecedent_expr::ExprId,
    /// Population and regime.
    pub binding: antecedent_expr::LeafBinding,
    /// Measured outcome axes.
    pub variables: Vec<VariableId>,
    /// Conditioning axes, with assignment-level support checked on execution.
    pub conditioned_on: Vec<VariableId>,
    /// Required intervention coordinates.
    pub interventions: Vec<VariableId>,
}
fn requirements(functional: &BoundTransportFunctional) -> Vec<ExactFactorRequirement> {
    use antecedent_expr::ExprNode;
    let arena = functional.arena();
    let mut pending = vec![functional.root()];
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                population,
                regime,
                ..
            } => out.push(ExactFactorRequirement {
                expression: id,
                binding: antecedent_expr::LeafBinding {
                    population: Arc::from(arena.population(*population)),
                    regime: *regime,
                },
                variables: arena.var_set(*variables).to_vec(),
                conditioned_on: arena.var_set(*conditioned_on).to_vec(),
                interventions: arena.intervention_set(*intervention),
            }),
            ExprNode::Kernel { body, .. } => pending.push(*body),
            ExprNode::Product(list) => pending.extend(arena.list(*list)),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr)
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator])
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
        }
    }
    out.sort_by_key(|factor| factor.expression.raw());
    out
}

/// Metadata-only inspection of exact prepared authority.
#[derive(Clone, Debug)]
pub struct ExactStudyInspection {
    /// Identity layers; no callbacks or evaluation are performed.
    pub identities: ExactStudyIdentities,
    /// Provider-bound formula, separate from its original certified expression.
    pub formula: String,
    /// Required population/regime leaves.
    pub factors: Vec<ExactFactorRequirement>,
    /// Available population, regime and immutable provider snapshot identities.
    pub bindings: Vec<(String, u32, String)>,
    /// Four typed reasoning slots.
    pub reasoning: ReasoningView,
    /// Durable inspect token for the paired classical-complete and catalog-search scopes.
    pub theorem_scope: &'static str,
    /// Classical single-source family actually executed (Figure-5 recursion).
    pub classical_scope: TheoremScope,
    /// Bounded catalog search. Sound and incomplete.
    pub catalog_scope: TheoremScope,
}

impl StudyBuilder {
    /// Prepare the exact-law data modality using a checked native identification.
    /// The returned common handle retains graph, query, evidence and providers.
    ///
    /// # Errors
    /// Changed proof inputs, incompatible provider contract, or exceeded budget.
    pub fn exact_transport(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<ExactPreparedState>, IoError> {
        antecedent_identify::verify_classical_transport(
            &diagram,
            functional.derivation().query(),
            functional.derivation(),
            SidLimits::default(),
            ctx,
        )
        .map_err(err)?;
        PreparedStudy::<ExactPreparedState>::build(diagram, functional, data, request, limits, ctx)
    }
}

fn law_wires(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    ordered_laws(data.laws().iter().map(ExactLawWire::from_law))
}
fn ordered_laws(laws: impl Iterator<Item = ExactLawWire>) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = laws
        .map(|law| {
            Ok((
                antecedent_io::to_cbor(&(law.population.clone(), law.regime, &law.interventions))?,
                law,
            ))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}
// Preserve the pre-T6 wire field order when checking historical identities.
#[derive(Serialize)]
struct LegacyLawIdentity<'a> {
    population: &'a str,
    regime: u32,
    interventions: &'a [(u32, ValueWire)],
    axes: &'a [(u32, Vec<ValueWire>)],
    probabilities: &'a [f64],
    snapshot: &'a str,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}
fn law_digest(
    domain: IdentityDomain,
    laws: &[ExactLawWire],
    legacy: bool,
) -> Result<String, IoError> {
    if legacy {
        let records: Vec<_> = laws
            .iter()
            .map(|law| LegacyLawIdentity {
                population: &law.population,
                regime: law.regime,
                interventions: &law.interventions,
                axes: &law.axes,
                probabilities: &law.probabilities,
                snapshot: &law.snapshot,
                absolute_tolerance: law.absolute_tolerance,
                relative_tolerance: law.relative_tolerance,
            })
            .collect();
        digest(domain, &records)
    } else {
        digest(domain, &laws)
    }
}
fn shape(data: &ExactTransportData, legacy: bool) -> Result<String, IoError> {
    let mut laws = ordered_laws(data.laws().iter().map(ExactLawWire::metadata))?;
    for law in &mut laws {
        law.probabilities.clear();
        law.snapshot.clear();
        law.origin.clear();
        law.absolute_tolerance = 0.0;
        law.relative_tolerance = 0.0;
        law.axes.sort_by_key(|(v, _)| *v);
        for (_, values) in &mut law.axes {
            let mut keyed = values
                .iter()
                .cloned()
                .map(|value| Ok((antecedent_io::to_cbor(&value)?, value)))
                .collect::<Result<Vec<_>, IoError>>()?;
            keyed.sort_by(|a, b| a.0.cmp(&b.0));
            *values = keyed.into_iter().map(|(_, value)| value).collect();
        }
    }
    law_digest(IdentityDomain::Observation, &laws, legacy)
}
fn assignments(request: &Assignment) -> Vec<(u32, ValueWire)> {
    let mut values: Vec<_> =
        request.entries().iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect();
    values.sort_by_key(|(v, _)| *v);
    values
}
impl PreparedStudy<ExactPreparedState> {
    fn build(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        Self::build_with_identity(diagram, functional, data, request, limits, ctx, false)
    }
    // Identity encoding is retained independently of numerical execution settings.
    #[allow(clippy::too_many_arguments)]
    fn build_with_identity(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
        legacy_law_identity: bool,
    ) -> Result<Self, IoError> {
        if data.laws().iter().any(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact) {
            return Err(err("empirical laws require the statistical transport modality"));
        }
        let identity_bytes = data
            .laws()
            .iter()
            .try_fold(0usize, |bytes, law| {
                bytes.checked_add(law.probabilities().len().checked_mul(32)?)
            })
            .ok_or_else(|| err("exact identity memory overflow"))?;
        if ctx
            .memory
            .hard_limit_bytes
            .is_some_and(|limit| u64::try_from(identity_bytes).map_or(true, |bytes| bytes > limit))
        {
            return Err(err("exact identity memory budget exceeded"));
        }
        let plan = antecedent_estimate::prepare_exact_transport(
            &functional,
            data.clone(),
            request.clone(),
            limits,
            ctx,
        )
        .map_err(err)?;
        Self::from_checked_plan(
            diagram,
            functional,
            data,
            request,
            limits,
            plan,
            legacy_law_identity,
        )
    }

    // Only the facade calls this after graph verification and compilation against these inputs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_checked_plan(
        diagram: SelectionDiagram,
        functional: BoundTransportFunctional,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        plan: ExactEvaluationPlan,
        legacy_law_identity: bool,
    ) -> Result<Self, IoError> {
        let proof = TransportProofWire::from_checked(functional.derivation())?;
        let catalog = functional.catalog().canonicalized().map_err(err)?;
        let mut contract = EvidenceCatalogWire::from_catalog(&catalog);
        contract.bindings.clear();
        let shape = shape(&data, legacy_law_identity)?;
        let target = digest(
            IdentityDomain::Target,
            &(
                &proof.proof.outcomes,
                &proof.proof.treatments,
                &proof.proof.target,
                assignments(&request),
            ),
        )?;
        let observation = digest(IdentityDomain::Observation, &(&contract, &shape))?;
        let inference_binding = digest(
            IdentityDomain::InferenceBinding,
            &("exact_law_no_sampling_uncertainty", limits.operations, limits.depth),
        )?;
        let identification = digest(
            IdentityDomain::Identification,
            &(
                &proof.proof.graph_signature,
                &proof.proof.outcomes,
                &proof.proof.treatments,
                &proof.proof.source,
                &proof.proof.target,
                &proof.proof.evidence_setting,
                &observation,
            ),
        )?;
        let identification = if proof.proof.sources.is_empty() {
            identification
        } else {
            digest(IdentityDomain::Identification, &(&identification, &proof.proof.sources))?
        };
        let identification_product = digest(IdentityDomain::IdentificationProduct, &proof)?;
        let program = digest(
            IdentityDomain::Program,
            &(
                &identification,
                &identification_product,
                &target,
                antecedent_io::expr_wire::expr_arena_to_wire(functional.arena())?,
                functional.root().raw(),
            ),
        )?;
        let snapshot =
            law_digest(IdentityDomain::DataSnapshot, &law_wires(&data)?, legacy_law_identity)?;
        let execution =
            digest(IdentityDomain::Execution, &(&program, &snapshot, &inference_binding))?;
        Ok(Self {
            state: ExactPreparedState {
                diagram,
                functional,
                data,
                request,
                limits,
                plan,
                identities: ExactStudyIdentities {
                    target,
                    observation,
                    inference_binding,
                    identification,
                    identification_product,
                    program,
                    snapshot,
                    execution,
                },
                shape,
                legacy_law_identity,
            },
        })
    }
    /// Frozen provider contract.
    #[must_use]
    pub fn evidence_catalog(&self) -> &antecedent_core::EvidenceCatalog {
        self.state.functional.catalog()
    }
    /// Frozen selection diagram.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagram {
        &self.state.diagram
    }
    /// Consumed derivation rules.
    #[must_use]
    pub fn rules(&self) -> Vec<&'static str> {
        self.state.functional.derivation().rules()
    }
    /// Frozen query; edits require preparation again.
    #[must_use]
    pub fn query(&self) -> &ClassicalTransportQuery {
        self.state.functional.derivation().query()
    }
    /// Inspect metadata only, with support checks marked execution-specific.
    #[must_use]
    pub fn inspect(&self) -> ExactStudyInspection {
        ExactStudyInspection {
            identities: self.state.identities.clone(),
            formula: self.state.functional.arena().pretty(self.state.functional.root()),
            factors: requirements(&self.state.functional),
            bindings: self
                .state
                .data
                .laws()
                .iter()
                .map(|law| {
                    (law.population().into(), law.regime().raw(), law.snapshot_identity().into())
                })
                .collect(),
            reasoning: Self::reasoning(false),
            theorem_scope: if self.state.functional.derivation().sources().is_empty() {
                TheoremScope::exact_law_inspect_label()
            } else {
                "classical_meta_all_source_experiments_v1; finite_catalog_search_bounded"
            },
            classical_scope: self.state.functional.derivation().theorem_scope(),
            catalog_scope: TheoremScope::finite_catalog_search(),
        }
    }

    /// Evaluate a declared one-factor mechanism sensitivity analysis against this
    /// prepared study's checked graph, query, and catalog bindings.
    ///
    /// # Errors
    /// The fixed-graph route refuses unsupported graphs, stale proofs or snapshots,
    /// incomplete categorical strata, and invalid kernel probabilities.
    pub fn mechanism_sensitivity(
        &self,
        spec: &antecedent_validate::FixedGraphMechanismSensitivitySpec,
        ctx: &ExecutionContext,
    ) -> Result<
        antecedent_validate::FixedGraphMechanismSensitivityResult,
        antecedent_validate::FixedGraphSensitivityError,
    > {
        antecedent_validate::fixed_graph_mechanism_sensitivity(
            &self.state.diagram,
            self.state.functional.derivation().query(),
            &self.state.functional,
            spec,
            ctx,
        )
    }

    /// Evaluate one-factor outcome-kernel sensitivity restricted to a single
    /// treatment-level slice while holding the other arm fixed.
    pub fn mechanism_sensitivity_at_treatment_level(
        &self,
        spec: &antecedent_validate::FixedGraphMechanismSensitivitySpec,
        treatment_level: usize,
        ctx: &ExecutionContext,
    ) -> Result<
        antecedent_validate::FixedGraphMechanismSensitivityResult,
        antecedent_validate::FixedGraphSensitivityError,
    > {
        antecedent_validate::fixed_graph_treatment_level_sensitivity(
            &self.state.diagram,
            self.state.functional.derivation().query(),
            &self.state.functional,
            spec,
            treatment_level,
            ctx,
        )
    }

    fn reasoning(evaluated: bool) -> ReasoningView {
        ReasoningView::new(
            SlotAvailability::Available(IdentificationSlot::identified_singleton(
                IdentificationStatus::NonparametricallyIdentified,
            )),
            SlotAvailability::Available(SupportSlot::new(
                "stage_contract",
                Some(Arc::from("transport.exact_law")),
                if evaluated {
                    SlotAvailability::Available(Arc::from("exact_factor_support_checked"))
                } else {
                    SlotAvailability::unavailable("execution_specific")
                },
            )),
            SlotAvailability::unavailable("exact_supplied_law_no_sampling_uncertainty"),
            SlotAvailability::Available(AssumptionSlot::new(vec![ObligationRecord::new(
                "transport.selection_diagram",
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::Uncheckable,
                AssumptionStatus::Untestable,
                "Accepted causal graph, mechanism selections, and exact supplied laws describe the declared populations.",
            )])),
        )
    }
    /// Execute the frozen physical plan. The request identity rejects stale UI clicks.
    ///
    /// # Errors
    /// Stale execution identity, support failure, cancellation, or resource exhaustion.
    pub fn estimate_checked(
        &self,
        execution: &str,
        ctx: &ExecutionContext,
    ) -> Result<ExactStudyResult, IoError> {
        if execution != self.state.identities.execution {
            return Err(err("transport.stale_request"));
        }
        self.estimate_retained(ctx)
    }
    /// Estimate using retained exact laws and the already compiled physical plan.
    ///
    /// # Errors
    /// Located support or numerical failure, cancellation, or exhausted budget.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<ExactStudyResult, IoError> {
        self.estimate_retained(ctx)
    }
    /// Execute retained providers without re-identification or re-preparation.
    ///
    /// # Errors
    /// Support failure, cancellation, or resource exhaustion. No partial result escapes.
    pub fn estimate_retained(&self, ctx: &ExecutionContext) -> Result<ExactStudyResult, IoError> {
        Ok(self.result_from_evaluation(self.state.plan.evaluate(ctx).map_err(err)?))
    }
    /// Wrap a distribution already produced by this handle's own plan, so a caller that
    /// evaluated the plan for eligibility does not evaluate it a second time.
    pub(super) fn result_from_evaluation(
        &self,
        distribution: ExactDistribution,
    ) -> ExactStudyResult {
        ExactStudyResult {
            distribution,
            identities: self.state.identities.clone(),
            reasoning: Self::reasoning(true),
        }
    }
    /// Preview using the common transformation/invalidation contract.
    ///
    /// # Errors
    /// Invalid retained identity (never manufactured from editable displays).
    pub fn preview_transform(
        &self,
        intent: antecedent_core::TransformIntent,
    ) -> Result<antecedent_core::TransformationReport, IoError> {
        use antecedent_core::{IdentityRef, SemanticDigest, TransformIntent, TransformationReport};
        let ids = &self.state.identities;
        let inputs = [
            (IdentityDomain::Target, &ids.target),
            (IdentityDomain::Identification, &ids.identification),
            (IdentityDomain::IdentificationProduct, &ids.identification_product),
            (IdentityDomain::Observation, &ids.observation),
            (IdentityDomain::Program, &ids.program),
            (IdentityDomain::InferenceBinding, &ids.inference_binding),
            (IdentityDomain::DataSnapshot, &ids.snapshot),
            (IdentityDomain::Execution, &ids.execution),
        ]
        .into_iter()
        .map(|(domain, value)| {
            Ok(IdentityRef::new(
                domain,
                SemanticDigest::from_bytes(antecedent_io::external_estimate::parse_digest_hex(
                    value,
                )?),
            ))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
        let report = TransformationReport::new(
            intent,
            inputs,
            antecedent_core::intent_effects(intent).iter().cloned(),
            Vec::new(),
        );
        Ok(
            if matches!(
                intent,
                TransformIntent::DisplayPrecision
                    | TransformIntent::FilterDisplay
                    | TransformIntent::CompatibleDataReplace
            ) {
                report
            } else {
                report.refused_on_handle(
                    "transport.reprepare_required: exact structural or execution contract changed",
                )
            },
        )
    }
    /// Metadata-only replacement preview: a different evidence shape requires preparation.
    ///
    /// # Errors
    /// Identity encoding failure.
    pub fn preview_snapshot(&self, data: &ExactTransportData) -> Result<bool, IoError> {
        Ok(data.laws().iter().all(|law| law.origin() == antecedent_expr::LawOrigin::SuppliedExact)
            && shape(data, self.state.legacy_law_identity)? == self.state.shape)
    }
    fn replacement(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if !self.preview_snapshot(&data)? {
            return Err(err("transport.reprepare_required"));
        }
        let catalog = rebind_snapshots(self.state.functional.catalog(), |regime| {
            data.laws()
                .iter()
                .filter(|law| law.regime() == regime)
                .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                .collect()
        })?;
        let functional = self.state.functional.derivation().bind_catalog(&catalog).map_err(err)?;
        Self::build_with_identity(
            self.state.diagram.clone(),
            functional,
            data,
            self.state.request.clone(),
            self.state.limits,
            ctx,
            self.state.legacy_law_identity,
        )
    }
    /// Replace compatible snapshots, invalidating all earlier execution claims.
    /// Reuses the checked derivation. Failure preserves the current handle.
    ///
    /// # Errors
    /// Changed evidence shape, missing providers or exceeded preflight budget.
    pub fn replace_snapshot(
        &mut self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<(), IoError> {
        let candidate = self.replacement(data, ctx)?;
        *self = candidate;
        Ok(())
    }
    /// Explicit atomic refresh: execute a candidate before publishing its state.
    ///
    /// # Errors
    /// Failed preparation/evaluation leaves the previous valid state intact.
    pub fn refresh(
        &mut self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<ExactStudyResult, IoError> {
        let candidate = self.replacement(data, ctx)?;
        let result = candidate.estimate_retained(ctx)?;
        *self = candidate;
        Ok(result)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExactFactorSupportWire {
    expression: u32,
    assignment: Vec<(u32, ValueWire)>,
    status: String,
    denominator: Option<f64>,
}
fn support_wire(result: &ExactStudyResult) -> Vec<ExactFactorSupportWire> {
    result
        .distribution
        .support
        .iter()
        .map(|record| ExactFactorSupportWire {
            expression: record.expression.raw(),
            assignment: record
                .assignment
                .iter()
                .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                .collect(),
            status: record.status.into(),
            denominator: record.denominator,
        })
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactExecutionWire {
    format: String,
    nodes: Vec<u32>,
    directed: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    selections: Vec<u32>,
    proof: TransportProofWire,
    catalog: EvidenceCatalogWire,
    laws: Vec<ExactLawWire>,
    request: Vec<(u32, ValueWire)>,
    operations: usize,
    depth: usize,
    identities: ExactStudyIdentities,
    atoms: Vec<Vec<ValueWire>>,
    probabilities: Vec<f64>,
    reasoning: antecedent_io::ReasoningSectionWire,
    support: Vec<ExactFactorSupportWire>,
}
impl PreparedStudy<ExactPreparedState> {
    /// Export checked proof, providers, identities and all four reasoning slots.
    /// The result must belong to the current immutable snapshot and request.
    ///
    /// # Errors
    /// Stale/substituted result, unsupported coordinates, or encoding failure.
    pub fn export(&self, result: &ExactStudyResult) -> Result<Vec<u8>, IoError> {
        if result.identities != self.state.identities {
            return Err(err("transport.stale_result"));
        }
        let graph = self.state.diagram.causal_graph();
        let edges = antecedent_io::admg_to_wire(graph)?;
        let nodes = graph
            .nodes()
            .iter()
            .map(|node| match node {
                NodeRef::Static(v) => Ok(v.raw()),
                _ => Err(err("transport requires static coordinates")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        antecedent_io::to_cbor(&ExactExecutionWire {
            format: "antecedent.exact_transport.v1".into(),
            nodes,
            directed: edges.directed,
            bidirected: edges.bidirected,
            selections: self.state.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            proof: TransportProofWire::from_checked(self.state.functional.derivation())?,
            catalog: EvidenceCatalogWire::from_catalog(self.state.functional.catalog()),
            laws: {
                let mut laws = law_wires(&self.state.data)?;
                if self.state.legacy_law_identity {
                    for law in &mut laws {
                        law.origin.clear();
                    }
                }
                laws
            },
            request: assignments(&self.state.request),
            operations: self.state.limits.operations,
            depth: self.state.limits.depth,
            identities: self.state.identities.clone(),
            atoms: result
                .distribution
                .atoms
                .iter()
                .map(|row| row.iter().map(ValueWire::from_value).collect())
                .collect(),
            probabilities: result.distribution.probabilities.to_vec(),
            reasoning: super::contract::reasoning_section(&result.reasoning),
            support: support_wire(result),
        })
    }
    /// Independently consume a portable exact execution using only embedded laws.
    /// Verifies proof, binding, identities and numerical claims; never fits or fetches.
    /// Caller limits cap the untrusted artifact's declared numerical limits.
    ///
    /// # Errors
    /// Invalid proof, substituted binding, changed claims, or resource exhaustion.
    pub fn consume(
        bytes: &[u8],
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, ExactStudyResult), IoError> {
        if ctx.cancellation.is_cancelled()
            || ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes.len() as u64 > limit)
        {
            return Err(err("transport artifact budget/cancellation"));
        }
        let wire: ExactExecutionWire = antecedent_io::from_cbor(bytes)?;
        if wire.format != "antecedent.exact_transport.v1"
            || wire.operations > limits.operations
            || wire.depth > limits.depth
        {
            return Err(err("transport artifact format/limits"));
        }
        let (diagram, proof) = rebuild_checked_proof(
            &GraphFields {
                nodes: &wire.nodes,
                directed: &wire.directed,
                bidirected: &wire.bidirected,
                selections: &wire.selections,
            },
            &wire.proof,
            limits,
            ctx,
        )?;
        let functional = proof.bind_catalog(&wire.catalog.to_catalog()?).map_err(err)?;
        let data = ExactTransportData::try_new(
            wire.laws.iter().map(ExactLawWire::to_law).collect::<Result<Vec<_>, _>>()?,
            limits.operations,
        )
        .map_err(err)?;
        let request = Assignment::from_pairs(
            wire.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        if request.entries().len() != wire.request.len() {
            return Err(err("duplicate request coordinate"));
        }
        let prepared = Self::build_with_identity(
            diagram,
            functional,
            data,
            request,
            ExactEvaluationLimits { operations: wire.operations, depth: wire.depth },
            ctx,
            wire.laws.iter().all(|law| law.origin.is_empty()),
        )?;
        if prepared.state.identities != wire.identities {
            return Err(err("transport artifact identity mismatch"));
        }
        let result = prepared.estimate_retained(ctx)?;
        let atoms: Vec<Vec<_>> = result
            .distribution
            .atoms
            .iter()
            .map(|row| row.iter().map(ValueWire::from_value).collect())
            .collect();
        if support_wire(&result) != wire.support
            || atoms != wire.atoms
            || result.distribution.probabilities.as_ref() != wire.probabilities
            || super::contract::reasoning_section(&result.reasoning) != wire.reasoning
        {
            return Err(err("transport artifact execution claim mismatch"));
        }
        Ok((prepared, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, RegimeId,
        RegimeKind, Value,
    };
    use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
    use antecedent_graph::{Admg, DenseNodeId};
    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }
    fn data(snapshot: &str, p: [f64; 4]) -> ExactTransportData {
        ExactTransportData::try_new(
            [ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(0),
                [],
                [0, 1].map(|i| DiscreteAxis {
                    variable: v(i),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                }),
                p,
                snapshot,
                LawTolerance::default(),
            )
            .unwrap()],
            1000,
        )
        .unwrap()
    }
    fn prepared() -> PreparedStudy<ExactPreparedState> {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = ExecutionContext::for_tests(0);
        let antecedent_identify::ClassicalTransportResult::Identified(proof) =
            antecedent_identify::identify_classical_transport(
                &diagram,
                &query,
                SidLimits::default(),
                &ctx,
            )
            .unwrap()
        else {
            panic!("identified");
        };
        let catalog = EvidenceCatalog::try_new(
            [],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0), v(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [],
            None,
        )
        .unwrap();
        StudyBuilder::exact_transport(
            diagram,
            proof.bind_catalog(&catalog).unwrap(),
            data("one", [0.4, 0.1, 0.15, 0.35]),
            Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap()
    }
    #[test]
    fn historical_exact_law_identity_survives_consume_refresh_and_export() {
        let ctx = ExecutionContext::for_tests(0);
        let current = prepared();
        let state = &current.state;
        let old = PreparedStudy::<ExactPreparedState>::build_with_identity(
            state.diagram.clone(),
            state.functional.clone(),
            state.data.clone(),
            state.request.clone(),
            state.limits,
            &ctx,
            true,
        )
        .unwrap();
        assert_ne!(old.state.identities.snapshot, current.state.identities.snapshot);
        let result = old.estimate_retained(&ctx).unwrap();
        let wire: ExactExecutionWire =
            antecedent_io::from_cbor(&old.export(&result).unwrap()).unwrap();
        // Historical records did not contain an origin field at all.
        let mut value = serde_json::to_value(&wire).unwrap();
        for law in value["laws"].as_array_mut().unwrap() {
            law.as_object_mut().unwrap().remove("origin");
        }
        let bytes = antecedent_io::to_cbor(&value).unwrap();
        let (mut loaded, consumed) =
            PreparedStudy::<ExactPreparedState>::consume(&bytes, state.limits, &ctx).unwrap();
        assert_eq!(result.identities(), consumed.identities());
        let (again, _) = PreparedStudy::<ExactPreparedState>::consume(
            &loaded.export(&consumed).unwrap(),
            state.limits,
            &ctx,
        )
        .unwrap();
        assert_eq!(again.state.identities, old.state.identities);
        let refreshed = loaded.refresh(data("new", [0.1, 0.4, 0.4, 0.1]), &ctx).unwrap();
        assert_eq!(refreshed.identities().identification, result.identities().identification);
        assert_ne!(refreshed.identities().snapshot, result.identities().snapshot);
    }
    #[test]
    fn empirical_origin_cannot_claim_exact_supplied_law_uncertainty() {
        let ctx = ExecutionContext::for_tests(0);
        let prepared = prepared();
        let mut wire = ExactLawWire::from_law(&prepared.state.data.laws()[0]);
        wire.origin = "empirical_plugin".into();
        let empirical = ExactTransportData::try_new([wire.to_law().unwrap()], 1000).unwrap();
        assert!(!prepared.preview_snapshot(&empirical).unwrap());
        let state = prepared.state;
        assert!(
            StudyBuilder::exact_transport(
                state.diagram,
                state.functional,
                empirical,
                state.request,
                state.limits,
                &ctx,
            )
            .unwrap_err()
            .to_string()
            .contains("statistical transport modality")
        );
    }
    #[test]
    fn inspect_does_not_require_evaluate() {
        let prepared = prepared();
        let before = prepared.inspect();
        assert!(before.reasoning.identification.is_available());
        assert!(!before.reasoning.uncertainty.is_available());
        assert_eq!(before.identities.execution, prepared.inspect().identities.execution);
    }

    #[test]
    fn exact_common_lifecycle_atomic_refresh_and_independent_consume() {
        let ctx = ExecutionContext::for_tests(0);
        let mut prepared = prepared();
        let before = prepared.inspect();
        assert!(before.reasoning.support.as_ref().unwrap().empirical.as_ref().is_none());
        let result = prepared.estimate_checked(&before.identities.execution, &ctx).unwrap();
        assert!((result.distribution().mean(v(1)).unwrap() - 0.7).abs() < 1e-12);
        assert!(!result.reasoning().uncertainty.is_available());
        let bytes = prepared.export(&result).unwrap();
        let (_, consumed) = PreparedStudy::<ExactPreparedState>::consume(
            &bytes,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(result.identities(), consumed.identities());
        assert!(prepared.refresh(data("bad-support", [0.8, 0.2, 0.0, 0.0]), &ctx).is_err());
        assert_eq!(prepared.inspect().identities, before.identities);
        prepared.replace_snapshot(data("two", [0.1, 0.4, 0.4, 0.1]), &ctx).unwrap();
        let after = prepared.inspect();
        assert_eq!(before.identities.identification, after.identities.identification);
        assert_eq!(before.identities.program, after.identities.program);
        assert_ne!(before.identities.snapshot, after.identities.snapshot);
        assert!(prepared.estimate_checked(&before.identities.execution, &ctx).is_err());
        assert!(prepared.export(&result).is_err());
        let result = prepared.refresh(data("three", [0.2, 0.3, 0.3, 0.2]), &ctx).unwrap();
        assert!((result.distribution().mean(v(1)).unwrap() - 0.4).abs() < 1e-12);
    }
    #[test]
    fn portable_execution_rejects_premise_query_evidence_and_claim_mutations() {
        let ctx = ExecutionContext::for_tests(0);
        let prepared = prepared();
        let result = prepared.estimate_retained(&ctx).unwrap();
        let bytes = prepared.export(&result).unwrap();
        let wire: ExactExecutionWire = antecedent_io::from_cbor(&bytes).unwrap();
        let reject = |mutated: &ExactExecutionWire| {
            assert!(
                PreparedStudy::<ExactPreparedState>::consume(
                    &antecedent_io::to_cbor(mutated).unwrap(),
                    ExactEvaluationLimits::default(),
                    &ctx
                )
                .is_err()
            )
        };
        let mut changed = wire.clone();
        changed.proof.proof.steps[0].output = u32::MAX;
        reject(&changed);
        let mut changed = wire.clone();
        changed.proof.proof.source = "substituted".into();
        reject(&changed);
        let mut changed = wire.clone();
        changed.proof.proof.evidence_setting = "finite_catalog".into();
        reject(&changed);
        let mut changed = wire.clone();
        changed.directed.clear();
        reject(&changed);
        let mut changed = wire.clone();
        changed.probabilities.reverse();
        reject(&changed);
        let mut changed = wire.clone();
        changed.laws[0].snapshot = "substituted".into();
        reject(&changed);
        let mut changed = wire;
        changed.request.push(changed.request[0].clone());
        reject(&changed);
    }
}
