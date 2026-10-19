//! Inspection of a plan from values the caller already holds.
//!
//! [`inspect`] copies those values into a [`PlanInspection`]. It does not fit,
//! scan rows, enumerate graphs, or simulate. A missing required part is an
//! [`InspectionRefusal`], not an empty list, a zero estimate, or a default
//! policy. The scientific identity of the supplied target is the identity the
//! caller already computed for it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::assumption::Assumption;
use crate::execution::{KernelPolicy, MemoryBudget};
use crate::inference_realization::RealizationKind;
use crate::physical_operator::PhysicalOperator;
use crate::scientific_target::ScientificTarget;
use crate::text::is_blank;

/// Why [`inspect`] or a declaration constructor refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum InspectionRefusal {
    /// No scientific target was supplied.
    #[error("inspection refused: scientific target is missing")]
    MissingTarget,
    /// No assumption list was supplied.
    ///
    /// An empty list is a supplied list. Absence is not that list.
    #[error("inspection refused: assumptions are missing")]
    MissingAssumptions,
    /// No inference realization was supplied.
    #[error("inspection refused: realization is missing")]
    MissingRealization,
    /// No evidence-transformation record was supplied.
    #[error("inspection refused: evidence transformations are missing")]
    MissingEvidenceTransformations,
    /// No uncertainty demand was supplied.
    #[error("inspection refused: uncertainty demand is missing")]
    MissingUncertaintyDemand,
    /// No reuse record was supplied.
    #[error("inspection refused: reuse is missing")]
    MissingReuse,
    /// No materialization record was supplied.
    #[error("inspection refused: materialization is missing")]
    MissingMaterialization,
    /// No resource estimate was supplied.
    #[error("inspection refused: estimated resources are missing")]
    MissingEstimatedResources,
    /// No backend policy was supplied.
    #[error("inspection refused: backend policy is missing")]
    MissingBackendPolicy,
    /// No physical-operator list was supplied.
    ///
    /// An empty list is a supplied list. Absence is not that list.
    #[error("inspection refused: physical operators are missing")]
    MissingPhysicalOperators,
    /// A caller-supplied name was empty or whitespace.
    #[error("declaration name must be non-blank")]
    BlankName,
}

/// One evidence transformation the caller already named.
///
/// The name is a label. Constructing it does not apply the transformation.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct EvidenceTransformation {
    /// Caller-supplied label.
    name: Arc<str>,
}

impl EvidenceTransformation {
    /// Transformation labeled `name`.
    ///
    /// # Errors
    ///
    /// [`InspectionRefusal::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank evidence-transformation name is refused"]
    pub fn declared(name: impl Into<Arc<str>>) -> Result<Self, InspectionRefusal> {
        Ok(Self { name: non_blank(name)? })
    }

    /// Caller-supplied label.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Evidence transformations in the order the caller declared them.
///
/// An empty record means the caller declared no transformations. It is not a
/// stand-in for a missing record.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct EvidenceTransformations {
    /// Declared steps, in caller order.
    steps: Box<[EvidenceTransformation]>,
}

impl EvidenceTransformations {
    /// Record `steps` in iterator order. Nothing is applied or sorted.
    #[must_use]
    pub fn declared(steps: impl IntoIterator<Item = EvidenceTransformation>) -> Self {
        Self { steps: owned_slice(steps) }
    }

    /// Declared steps, in caller order.
    #[must_use]
    pub fn steps(&self) -> &[EvidenceTransformation] {
        &self.steps
    }
}

/// Uncertainty the caller requires the plan to carry.
///
/// The level is stored as the bits the caller supplied. It is not estimated,
/// calibrated, or replaced when non-finite.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct UncertaintyDemand {
    /// Whether the demand is joint across the plan's reported quantities.
    joint: bool,
    /// Bits of the caller-supplied level.
    level_bits: u64,
}

impl UncertaintyDemand {
    /// Demand with the caller's `joint` flag and `level` bits.
    #[must_use]
    pub const fn declared(joint: bool, level: f64) -> Self {
        Self { joint, level_bits: level.to_bits() }
    }

    /// Whether the demand is joint.
    #[must_use]
    pub const fn joint(self) -> bool {
        self.joint
    }

    /// Caller-supplied level, from the stored bits.
    #[must_use]
    pub const fn level(self) -> f64 {
        f64::from_bits(self.level_bits)
    }
}

/// Reusable work the caller already named.
///
/// `frozen_fit` records a caller decision. Construction does not look up a
/// cache or freeze a fit.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ReuseSnapshot {
    /// Caller-supplied label.
    name: Arc<str>,
    /// Whether the caller marked a frozen fit reusable.
    frozen_fit: bool,
}

impl ReuseSnapshot {
    /// Reuse labeled `name`.
    ///
    /// # Errors
    ///
    /// [`InspectionRefusal::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank reuse name is refused"]
    pub fn declared(
        name: impl Into<Arc<str>>,
        frozen_fit: bool,
    ) -> Result<Self, InspectionRefusal> {
        Ok(Self { name: non_blank(name)?, frozen_fit })
    }

    /// Caller-supplied label.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the caller marked a frozen fit reusable.
    #[must_use]
    pub const fn frozen_fit(&self) -> bool {
        self.frozen_fit
    }
}

/// Materialization the caller already described.
///
/// `copy_bytes` is the caller's figure. Construction does not copy a column
/// or measure a buffer.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MaterializationSnapshot {
    /// Caller-supplied label.
    name: Arc<str>,
    /// Caller-supplied copy size, in bytes.
    copy_bytes: u64,
}

impl MaterializationSnapshot {
    /// Materialization labeled `name` with the caller's `copy_bytes`.
    ///
    /// # Errors
    ///
    /// [`InspectionRefusal::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank materialization name is refused"]
    pub fn declared(name: impl Into<Arc<str>>, copy_bytes: u64) -> Result<Self, InspectionRefusal> {
        Ok(Self { name: non_blank(name)?, copy_bytes })
    }

    /// Caller-supplied label.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Caller-supplied copy size, in bytes.
    #[must_use]
    pub const fn copy_bytes(&self) -> u64 {
        self.copy_bytes
    }
}

/// Resources the caller already estimated.
///
/// The memory budget and the byte figure are both caller-supplied. They are
/// not summed from operators and not measured.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EstimatedResources {
    /// Caller-supplied memory budget.
    memory: MemoryBudget,
    /// Caller-supplied byte estimate.
    estimated_bytes: u64,
}

impl EstimatedResources {
    /// Resources from the caller's `memory` budget and `estimated_bytes`.
    #[must_use]
    pub const fn declared(memory: MemoryBudget, estimated_bytes: u64) -> Self {
        Self { memory, estimated_bytes }
    }

    /// Caller-supplied memory budget.
    #[must_use]
    pub const fn memory(self) -> MemoryBudget {
        self.memory
    }

    /// Caller-supplied byte estimate.
    #[must_use]
    pub const fn estimated_bytes(self) -> u64 {
        self.estimated_bytes
    }
}

/// Parts [`inspect`] may copy. Every part starts absent.
///
/// Absence is not an empty assumption list, an empty operator list, a zero
/// byte estimate, or a default kernel policy. Pass the value the caller holds,
/// including an empty list when that is the held value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InspectionInputs {
    /// Scientific target, or absence.
    target: Option<ScientificTarget>,
    /// Assumptions in caller order, or absence.
    assumptions: Option<Box<[Assumption]>>,
    /// Inference realization, or absence.
    realization: Option<RealizationKind>,
    /// Evidence transformations, or absence.
    evidence: Option<EvidenceTransformations>,
    /// Uncertainty demand, or absence.
    uncertainty: Option<UncertaintyDemand>,
    /// Reuse record, or absence.
    reuse: Option<ReuseSnapshot>,
    /// Materialization record, or absence.
    materialization: Option<MaterializationSnapshot>,
    /// Estimated resources, or absence.
    resources: Option<EstimatedResources>,
    /// Backend policy, or absence.
    backend_policy: Option<KernelPolicy>,
    /// Physical operators in caller order, or absence.
    operators: Option<Box<[PhysicalOperator]>>,
}

impl Default for InspectionInputs {
    fn default() -> Self {
        Self::new()
    }
}

impl InspectionInputs {
    /// Inputs with every required part absent.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target: None,
            assumptions: None,
            realization: None,
            evidence: None,
            uncertainty: None,
            reuse: None,
            materialization: None,
            resources: None,
            backend_policy: None,
            operators: None,
        }
    }

    /// Supply the scientific target.
    #[must_use]
    pub fn target(mut self, target: ScientificTarget) -> Self {
        self.target = Some(target);
        self
    }

    /// Supply assumptions in iterator order.
    #[must_use]
    pub fn assumptions(mut self, assumptions: impl IntoIterator<Item = Assumption>) -> Self {
        self.assumptions = Some(owned_slice(assumptions));
        self
    }

    /// Supply the inference realization.
    #[must_use]
    pub const fn realization(mut self, realization: RealizationKind) -> Self {
        self.realization = Some(realization);
        self
    }

    /// Supply the evidence transformations.
    #[must_use]
    pub fn evidence(mut self, evidence: EvidenceTransformations) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// Supply the uncertainty demand.
    #[must_use]
    pub const fn uncertainty(mut self, uncertainty: UncertaintyDemand) -> Self {
        self.uncertainty = Some(uncertainty);
        self
    }

    /// Supply the reuse record.
    #[must_use]
    pub fn reuse(mut self, reuse: ReuseSnapshot) -> Self {
        self.reuse = Some(reuse);
        self
    }

    /// Supply the materialization record.
    #[must_use]
    pub fn materialization(mut self, materialization: MaterializationSnapshot) -> Self {
        self.materialization = Some(materialization);
        self
    }

    /// Supply the estimated resources.
    #[must_use]
    pub const fn resources(mut self, resources: EstimatedResources) -> Self {
        self.resources = Some(resources);
        self
    }

    /// Supply the backend policy.
    ///
    /// This is the caller's [`KernelPolicy`]. It is not selected here.
    #[must_use]
    pub const fn backend_policy(mut self, backend_policy: KernelPolicy) -> Self {
        self.backend_policy = Some(backend_policy);
        self
    }

    /// Supply physical operators in iterator order.
    ///
    /// The operators are stored as given. They are not discovered by walking
    /// a graph.
    #[must_use]
    pub fn operators(mut self, operators: impl IntoIterator<Item = PhysicalOperator>) -> Self {
        self.operators = Some(owned_slice(operators));
        self
    }
}

/// Plan record assembled by [`inspect`].
///
/// Each field is a value the caller held. Reading the record does not fit,
/// scan, enumerate, or simulate, and it does not rewrite the target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PlanInspection {
    /// Scientific target, unchanged.
    target: ScientificTarget,
    /// Assumptions in caller order.
    assumptions: Box<[Assumption]>,
    /// Inference realization.
    realization: RealizationKind,
    /// Evidence transformations in caller order.
    evidence: EvidenceTransformations,
    /// Uncertainty demand.
    uncertainty: UncertaintyDemand,
    /// Reuse record.
    reuse: ReuseSnapshot,
    /// Materialization record.
    materialization: MaterializationSnapshot,
    /// Estimated resources.
    resources: EstimatedResources,
    /// Backend policy.
    backend_policy: KernelPolicy,
    /// Physical operators in caller order.
    operators: Box<[PhysicalOperator]>,
}

impl PlanInspection {
    /// Scientific target supplied to [`inspect`].
    #[must_use]
    pub const fn target(&self) -> &ScientificTarget {
        &self.target
    }

    /// Assumptions in the order supplied to [`inspect`].
    #[must_use]
    pub fn assumptions(&self) -> &[Assumption] {
        &self.assumptions
    }

    /// Inference realization supplied to [`inspect`].
    #[must_use]
    pub const fn realization(&self) -> RealizationKind {
        self.realization
    }

    /// Evidence transformations supplied to [`inspect`].
    #[must_use]
    pub const fn evidence(&self) -> &EvidenceTransformations {
        &self.evidence
    }

    /// Uncertainty demand supplied to [`inspect`].
    #[must_use]
    pub const fn uncertainty(&self) -> UncertaintyDemand {
        self.uncertainty
    }

    /// Reuse record supplied to [`inspect`].
    #[must_use]
    pub const fn reuse(&self) -> &ReuseSnapshot {
        &self.reuse
    }

    /// Materialization record supplied to [`inspect`].
    #[must_use]
    pub const fn materialization(&self) -> &MaterializationSnapshot {
        &self.materialization
    }

    /// Estimated resources supplied to [`inspect`].
    #[must_use]
    pub const fn resources(&self) -> EstimatedResources {
        self.resources
    }

    /// Backend policy supplied to [`inspect`].
    #[must_use]
    pub const fn backend_policy(&self) -> KernelPolicy {
        self.backend_policy
    }

    /// Physical operators in the order supplied to [`inspect`].
    #[must_use]
    pub fn operators(&self) -> &[PhysicalOperator] {
        &self.operators
    }
}

/// Copy `inputs` into a [`PlanInspection`].
///
/// The function takes the parts the caller already holds and returns them.
/// It has no callback parameter. It does not fit, scan rows, enumerate graphs,
/// or simulate, and it does not change the target's scientific identity.
///
/// # Errors
///
/// [`InspectionRefusal`] names the first required part that is still absent,
/// in the order target, assumptions, realization, evidence transformations,
/// uncertainty demand, reuse, materialization, estimated resources, backend
/// policy, physical operators. Absence is not filled with a default.
#[must_use = "a refusal is not an inspection"]
pub fn inspect(inputs: InspectionInputs) -> Result<PlanInspection, InspectionRefusal> {
    Ok(PlanInspection {
        target: required(inputs.target, InspectionRefusal::MissingTarget)?,
        assumptions: required(inputs.assumptions, InspectionRefusal::MissingAssumptions)?,
        realization: required(inputs.realization, InspectionRefusal::MissingRealization)?,
        evidence: required(inputs.evidence, InspectionRefusal::MissingEvidenceTransformations)?,
        uncertainty: required(inputs.uncertainty, InspectionRefusal::MissingUncertaintyDemand)?,
        reuse: required(inputs.reuse, InspectionRefusal::MissingReuse)?,
        materialization: required(
            inputs.materialization,
            InspectionRefusal::MissingMaterialization,
        )?,
        resources: required(inputs.resources, InspectionRefusal::MissingEstimatedResources)?,
        backend_policy: required(inputs.backend_policy, InspectionRefusal::MissingBackendPolicy)?,
        operators: required(inputs.operators, InspectionRefusal::MissingPhysicalOperators)?,
    })
}

fn required<T>(value: Option<T>, refusal: InspectionRefusal) -> Result<T, InspectionRefusal> {
    value.ok_or(refusal)
}

fn non_blank(name: impl Into<Arc<str>>) -> Result<Arc<str>, InspectionRefusal> {
    let name = name.into();
    if is_blank(&name) { Err(InspectionRefusal::BlankName) } else { Ok(name) }
}

fn owned_slice<T>(items: impl IntoIterator<Item = T>) -> Box<[T]> {
    items.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::{
        EstimatedResources, EvidenceTransformation, EvidenceTransformations, InspectionInputs,
        InspectionRefusal, MaterializationSnapshot, PlanInspection, ReuseSnapshot,
        UncertaintyDemand, inspect,
    };
    use crate::VariableId;
    use crate::assumption::{Assumption, AssumptionSet};
    use crate::execution::{KernelPolicy, MemoryBudget};
    use crate::inference_realization::RealizationKind;
    use crate::intervention::Intervention;
    use crate::physical_operator::{
        ByteEstimates, PhysicalEffect, PhysicalOperator, SourceDependencyId,
    };
    use crate::query::TargetPopulation;
    use crate::scientific_target::{
        Conditioning, GraphFamily, InferenceSelection, InterventionWorld, ModelAssumptions,
        ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId,
        TargetForm, TimeFrame, WorldAction, WorldBinding,
    };
    use crate::value::Value;

    fn sample_target() -> ScientificTarget {
        let outcome = VariableId::from_raw(1);
        let treatment = VariableId::from_raw(0);
        let binding = WorldBinding::new(
            TargetPopulation::AllObserved,
            ObservationUnit::new("subject").expect("unit"),
            OutcomeQuantity::mean(outcome, QuantityUnits::dimensionless()).expect("quantity"),
            Conditioning::unconditional(),
        )
        .expect("binding");
        let action = WorldAction::at_decision([Intervention::set(treatment, Value::f64(0.0))])
            .expect("action");
        let world = InterventionWorld::new(
            ScientificWorldId::from_raw(0),
            action,
            true,
            binding,
            TimeFrame::static_decision(),
        )
        .expect("world");
        ScientificTarget::new(
            TargetForm::World(world),
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Frequentist,
        )
    }

    fn sample_operator() -> PhysicalOperator {
        PhysicalOperator::declare(PhysicalEffect::Read)
            .source(SourceDependencyId::from_raw(7))
            .estimates(ByteEstimates::declared(1, 2, 3))
            .build()
            .expect("operator")
    }

    fn demand() -> UncertaintyDemand {
        UncertaintyDemand::declared(true, 0.95)
    }

    fn evidence() -> EvidenceTransformations {
        EvidenceTransformations::declared([
            EvidenceTransformation::declared("standardize").expect("evidence name")
        ])
    }

    fn reuse() -> ReuseSnapshot {
        ReuseSnapshot::declared("frozen_nuisance", true).expect("reuse")
    }

    fn materialization() -> MaterializationSnapshot {
        MaterializationSnapshot::declared("design_matrix", 4_096).expect("materialization")
    }

    fn resources() -> EstimatedResources {
        EstimatedResources::declared(
            MemoryBudget { soft_limit_bytes: Some(10), hard_limit_bytes: Some(20) },
            30,
        )
    }

    fn assumptions() -> [Assumption; 2] {
        [Assumption::Positivity, Assumption::Consistency]
    }

    fn held(target: ScientificTarget, uncertainty: Option<UncertaintyDemand>) -> InspectionInputs {
        let mut inputs = InspectionInputs::new()
            .target(target)
            .assumptions(assumptions())
            .realization(RealizationKind::ScoreFrequentist)
            .evidence(evidence())
            .reuse(reuse())
            .materialization(materialization())
            .resources(resources())
            .backend_policy(KernelPolicy::scalar_only())
            .operators([sample_operator()]);
        if let Some(demand) = uncertainty {
            inputs = inputs.uncertainty(demand);
        }
        inputs
    }

    #[test]
    fn complete_inspection_round_trips_fields() {
        let target = sample_target();
        let identity = target.identity();
        let inference = target.inference();
        let expected = target.clone();
        let operator = sample_operator();
        let plan = inspect(held(target, Some(demand()))).expect("inspection");

        assert_eq!(plan.target(), &expected);
        assert_eq!(plan.target().identity(), identity);
        assert_eq!(plan.target().inference(), inference);
        assert_eq!(plan.assumptions(), assumptions().as_slice());
        assert_eq!(plan.realization(), RealizationKind::ScoreFrequentist);
        assert_eq!(plan.evidence(), &evidence());
        assert!(plan.uncertainty().joint());
        assert_eq!(plan.uncertainty().level().to_bits(), demand().level().to_bits());
        assert_eq!(plan.reuse(), &reuse());
        assert_eq!(plan.materialization(), &materialization());
        assert_eq!(plan.resources(), resources());
        assert_eq!(plan.backend_policy(), KernelPolicy::scalar_only());
        assert_eq!(plan.operators(), &[operator]);
    }

    #[test]
    fn missing_uncertainty_demand_is_refused() {
        let inputs = held(sample_target(), None);
        assert_eq!(inspect(inputs), Err(InspectionRefusal::MissingUncertaintyDemand));
    }

    #[test]
    fn inspect_accepts_no_fit_or_scan_callback() {
        // A callback or closure parameter would not coerce to this pointer type.
        let inspect_fn: fn(InspectionInputs) -> Result<PlanInspection, InspectionRefusal> = inspect;
        let plan = inspect_fn(held(sample_target(), Some(demand()))).expect("inspection");
        assert_eq!(plan.realization(), RealizationKind::ScoreFrequentist);
    }
}
