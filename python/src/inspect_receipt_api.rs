//! Python surface for plan inspection and execution-receipt facts.
//!
//! [`inspect`] copies declarations the caller already holds. It has no callback
//! parameter and does not fit, scan, enumerate, or simulate. Execution facts
//! record a run that already happened. Physical identity, the stored scientific
//! digest, the numerical-policy flag, and resumability stay separate. Loading
//! these records does not resume a live study. Family interchange consumers are
//! not this binding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution_facts::{
    ExecutionBackend, ExecutionCounts, ExecutionFactError, ExecutionFactStatus, ExecutionFacts,
    ExecutionResources, MeasuredExecution, OutputRetention,
};
use antecedent_core::inspect::{
    EstimatedResources, EvidenceTransformation, EvidenceTransformations, InspectionInputs,
    InspectionRefusal, MaterializationSnapshot, PlanInspection, ReuseSnapshot, UncertaintyDemand,
};
use antecedent_core::reason_code;
use antecedent_core::{
    Assumption, AssumptionSet, ByteEstimates, Conditioning, GraphFamily, InferenceSelection,
    Intervention, InterventionWorld, KernelPolicy, MemoryBudget, ModelAssumptions, ObservationUnit,
    OutcomeQuantity, PhysicalEffect, PhysicalOperator, QuantityUnits, RealizationKind,
    ScientificTarget, ScientificWorldId, SemanticDigest, SourceDependencyId, TargetForm,
    TargetPopulation, TimeFrame, Value, VariableId, WorldAction, WorldBinding,
};
use pyo3::prelude::*;

use crate::name_errors::unknown_name;

const LIVE_STUDY: &str = concat!(
    "loading inspect and execution-receipt facts does not resume a live study ",
    "and does not invoke callbacks"
);

fn inspection_err(refusal: InspectionRefusal) -> PyErr {
    crate::unsupported_message(refusal.to_string())
}

fn fact_err(error: ExecutionFactError) -> PyErr {
    crate::unsupported_message(error.to_string())
}

fn digest_of(bytes: Option<Vec<u8>>) -> PyResult<Option<SemanticDigest>> {
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let array: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
        crate::value_err_code(
            reason_code!("invalid_argument"),
            "scientific identity must be 32 bytes",
        )
    })?;
    Ok(Some(SemanticDigest::from_bytes(array)))
}

fn digest_bytes(digest: SemanticDigest) -> Vec<u8> {
    digest.as_bytes().to_vec()
}

fn kernel_policy(name: &str) -> PyResult<KernelPolicy> {
    match name {
        "scalar_only" => Ok(KernelPolicy::scalar_only()),
        "default" => Ok(KernelPolicy::default_policy()),
        _ => Err(unknown_name("kernel policy", name)),
    }
}

fn kernel_name(policy: KernelPolicy) -> &'static str {
    if policy == KernelPolicy::scalar_only() {
        "scalar_only"
    } else if policy == KernelPolicy::default_policy() {
        "default"
    } else {
        "custom"
    }
}

fn retention_of(name: &str) -> PyResult<OutputRetention> {
    match name {
        "full_reusable" => Ok(OutputRetention::FullReusable),
        "summary_only" => Ok(OutputRetention::SummaryOnly),
        "disk_backed" => Ok(OutputRetention::DiskBacked),
        _ => Err(unknown_name("output retention", name)),
    }
}

fn status_of(name: &str) -> PyResult<ExecutionFactStatus> {
    match name {
        "completed" => Ok(ExecutionFactStatus::Completed),
        "cancelled" => Ok(ExecutionFactStatus::Cancelled),
        "failed" => Ok(ExecutionFactStatus::Failed),
        _ => Err(unknown_name("execution fact status", name)),
    }
}

fn assumption(name: &str) -> PyResult<Assumption> {
    Ok(match name {
        "causal_markov" => Assumption::CausalMarkov,
        "faithfulness" => Assumption::Faithfulness,
        "causal_sufficiency" => Assumption::CausalSufficiency,
        "consistency" => Assumption::Consistency,
        "positivity" => Assumption::Positivity,
        "no_interference" => Assumption::NoInterference,
        "stationarity" => Assumption::Stationarity,
        "piecewise_stationarity" => Assumption::PiecewiseStationarity,
        "no_selection_bias" => Assumption::NoSelectionBias,
        "monotonicity" => Assumption::Monotonicity,
        _ => return Err(unknown_name("assumption", name)),
    })
}

fn assumption_name(assumption: &Assumption) -> String {
    match assumption {
        Assumption::CausalMarkov => "causal_markov".to_owned(),
        Assumption::Faithfulness => "faithfulness".to_owned(),
        Assumption::CausalSufficiency => "causal_sufficiency".to_owned(),
        Assumption::Consistency => "consistency".to_owned(),
        Assumption::Positivity => "positivity".to_owned(),
        Assumption::NoInterference => "no_interference".to_owned(),
        Assumption::Stationarity => "stationarity".to_owned(),
        Assumption::PiecewiseStationarity => "piecewise_stationarity".to_owned(),
        Assumption::NoSelectionBias => "no_selection_bias".to_owned(),
        Assumption::Monotonicity => "monotonicity".to_owned(),
        Assumption::ExclusionRestriction { .. } => "exclusion_restriction".to_owned(),
        Assumption::ParametricRestriction(_) => "parametric_restriction".to_owned(),
        Assumption::PriorRestriction(_) => "prior_restriction".to_owned(),
        Assumption::Custom { id, .. } => format!("custom:{id}"),
    }
}

fn realization(name: &str) -> PyResult<RealizationKind> {
    match name {
        "ordinary_bayesian_likelihood" => Ok(RealizationKind::OrdinaryBayesianLikelihood),
        "score_frequentist" => Ok(RealizationKind::ScoreFrequentist),
        "generalized_bayes" => Ok(RealizationKind::GeneralizedBayes),
        "partial_identification_envelope" => Ok(RealizationKind::PartialIdentificationEnvelope),
        _ => Err(unknown_name("realization", name)),
    }
}

fn realization_name(kind: RealizationKind) -> &'static str {
    kind.as_str()
}

fn inference_name(selection: InferenceSelection) -> &'static str {
    match selection {
        InferenceSelection::Unspecified => "unspecified",
        InferenceSelection::Frequentist => "frequentist",
        InferenceSelection::Bayesian => "bayesian",
        InferenceSelection::GeneralizedBayesian => "generalized_bayesian",
        _ => "other",
    }
}

fn live_study_refusal() -> PyErr {
    crate::unsupported_message(LIVE_STUDY)
}

/// Static mean world used by the Rust inspection fixture.
///
/// Outcome variable 1, treatment variable 0 set to 0, unit `subject`,
/// dimensionless units, all observed rows, unconditional conditioning, world 0
/// as the reference, a static frame, an empty model-assumption set, a DAG, and
/// frequentist selection. Construction does not identify or fit.
fn static_mean_target() -> PyResult<ScientificTarget> {
    let outcome = VariableId::from_raw(1);
    let treatment = VariableId::from_raw(0);
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("subject")
            .map_err(|err| crate::unsupported_message(err.to_string()))?,
        OutcomeQuantity::mean(outcome, QuantityUnits::dimensionless())
            .map_err(|err| crate::unsupported_message(err.to_string()))?,
        Conditioning::unconditional(),
    )
    .map_err(|err| crate::unsupported_message(err.to_string()))?;
    let action = WorldAction::at_decision([Intervention::set(treatment, Value::f64(0.0))])
        .map_err(|err| crate::unsupported_message(err.to_string()))?;
    let world = InterventionWorld::new(
        ScientificWorldId::from_raw(0),
        action,
        true,
        binding,
        TimeFrame::static_decision(),
    )
    .map_err(|err| crate::unsupported_message(err.to_string()))?;
    Ok(ScientificTarget::new(
        TargetForm::World(world),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Frequentist,
    ))
}

fn read_operators(rows: &[(u64, u64, u64, u64)]) -> PyResult<Vec<PhysicalOperator>> {
    let mut operators = Vec::with_capacity(rows.len());
    for &(source, scratch, retained, output) in rows {
        let operator = PhysicalOperator::declare(PhysicalEffect::Read)
            .source(SourceDependencyId::from_raw(source))
            .estimates(ByteEstimates::declared(scratch, retained, output))
            .build()
            .map_err(|err| crate::unsupported_message(err.to_string()))?;
        operators.push(operator);
    }
    Ok(operators)
}

fn measured_execution(
    py: Python<'_>,
    counts: Py<PyExecutionCounts>,
    retention: &str,
    resources: Py<PyExecutionResources>,
    backend_name: String,
    kernel: &str,
    status: &str,
) -> PyResult<MeasuredExecution> {
    Ok(MeasuredExecution::new(
        counts.borrow(py).inner,
        retention_of(retention)?,
        resources.borrow(py).inner,
        ExecutionBackend::new(backend_name, kernel_policy(kernel)?),
        status_of(status)?,
    ))
}

/// One planned operator: effect, source ids, then declared scratch, retained and output bytes.
type OperatorRow = (String, Vec<u64>, Option<u64>, Option<u64>, Option<u64>);

#[pyclass(name = "ExecutionCounts", frozen)]
struct PyExecutionCounts {
    inner: ExecutionCounts,
}

#[pymethods]
impl PyExecutionCounts {
    #[new]
    fn new(scans: u64, copies: u64, fits: u64, tests: u64, cache_hits: u64) -> Self {
        Self { inner: ExecutionCounts::new(scans, copies, fits, tests, cache_hits) }
    }

    fn scans(&self) -> u64 {
        self.inner.scans()
    }

    fn copies(&self) -> u64 {
        self.inner.copies()
    }

    fn fits(&self) -> u64 {
        self.inner.fits()
    }

    fn tests(&self) -> u64 {
        self.inner.tests()
    }

    fn cache_hits(&self) -> u64 {
        self.inner.cache_hits()
    }

    /// Add `extra`. An overflowing sum is refused and this value is unchanged.
    fn try_add(&self, py: Python<'_>, extra: Py<PyExecutionCounts>) -> PyResult<Self> {
        let extra = extra.borrow(py).inner;
        self.inner.try_add(extra).map(|inner| Self { inner }).map_err(fact_err)
    }
}

#[pyclass(name = "ExecutionResources", frozen)]
struct PyExecutionResources {
    inner: ExecutionResources,
}

#[pymethods]
impl PyExecutionResources {
    #[new]
    fn new(threads: u64, memory_bytes: u64, spill_bytes: u64) -> Self {
        Self { inner: ExecutionResources::new(threads, memory_bytes, spill_bytes) }
    }

    fn threads(&self) -> u64 {
        self.inner.threads()
    }

    fn memory_bytes(&self) -> u64 {
        self.inner.memory_bytes()
    }

    fn spill_bytes(&self) -> u64 {
        self.inner.spill_bytes()
    }

    /// Add `extra`, including spill bytes. Overflow is refused and this value is unchanged.
    fn try_add(&self, py: Python<'_>, extra: Py<PyExecutionResources>) -> PyResult<Self> {
        let extra = extra.borrow(py).inner;
        self.inner.try_add(extra).map(|inner| Self { inner }).map_err(fact_err)
    }
}

/// Recorded measurements. Reading them does not re-execute the run.
#[pyclass(name = "NumericalReplay", frozen)]
struct PyNumericalReplay {
    counts: ExecutionCounts,
    resources: ExecutionResources,
    retention: &'static str,
    backend_name: String,
    kernel: &'static str,
    status: &'static str,
    contract_changed: bool,
    invoked_callback: bool,
    reexecuted: bool,
}

#[pymethods]
impl PyNumericalReplay {
    fn counts(&self) -> PyExecutionCounts {
        PyExecutionCounts { inner: self.counts }
    }

    fn resources(&self) -> PyExecutionResources {
        PyExecutionResources { inner: self.resources }
    }

    fn retention(&self) -> &'static str {
        self.retention
    }

    fn backend_name(&self) -> &str {
        &self.backend_name
    }

    fn kernel(&self) -> &'static str {
        self.kernel
    }

    fn status(&self) -> &'static str {
        self.status
    }

    fn contract_changed(&self) -> bool {
        self.contract_changed
    }

    /// Always false. Reading the record does not invoke a callback.
    fn invoked_callback(&self) -> bool {
        self.invoked_callback
    }

    /// Always false. The record is the run that was measured, not a second execution.
    fn reexecuted(&self) -> bool {
        self.reexecuted
    }
}

/// Integrity, stored scientific digest, numerical record, and resumability.
///
/// Resumability is refused. These facts are not a live study.
#[pyclass(name = "ExecutionFactStanding", frozen)]
struct PyExecutionFactStanding {
    integrity: Vec<u8>,
    semantic: Option<Vec<u8>>,
    replay: PyNumericalReplay,
    resumable: bool,
}

#[pymethods]
impl PyExecutionFactStanding {
    /// Physical digest of the measurements.
    fn integrity(&self) -> &[u8] {
        &self.integrity
    }

    /// Scientific digest stored beside the measurements, if the caller supplied one.
    fn semantic_verification(&self) -> Option<Vec<u8>> {
        self.semantic.clone()
    }

    /// Recorded counts, resources, backend, and numerical-policy flag.
    fn numerical_replay(&self) -> PyNumericalReplay {
        PyNumericalReplay {
            counts: self.replay.counts,
            resources: self.replay.resources,
            retention: self.replay.retention,
            backend_name: self.replay.backend_name.clone(),
            kernel: self.replay.kernel,
            status: self.replay.status,
            contract_changed: self.replay.contract_changed,
            invoked_callback: false,
            reexecuted: false,
        }
    }

    /// Refused. Execution-receipt facts do not reconstruct a live study.
    fn resumability(&self) -> PyResult<()> {
        if self.resumable { Ok(()) } else { Err(live_study_refusal()) }
    }
}

/// Measured execution-receipt facts.
///
/// The scientific digest is stored beside the measurements. Rebuilding the
/// measurements leaves that digest as it was. Kernel policy, counts, retention,
/// spill, and status do not set the numerical-policy flag.
#[pyclass(name = "ExecutionFacts")]
struct PyExecutionFacts {
    inner: ExecutionFacts,
}

#[pymethods]
impl PyExecutionFacts {
    #[new]
    fn new(
        py: Python<'_>,
        counts: Py<PyExecutionCounts>,
        retention: &str,
        resources: Py<PyExecutionResources>,
        backend_name: String,
        kernel: &str,
        status: &str,
        scientific_identity: Option<Vec<u8>>,
        numerical_policy_changes_scientific_contract: bool,
    ) -> PyResult<Self> {
        let measured =
            measured_execution(py, counts, retention, resources, backend_name, kernel, status)?;
        Ok(Self {
            inner: ExecutionFacts::new(
                measured,
                digest_of(scientific_identity)?,
                numerical_policy_changes_scientific_contract,
            ),
        })
    }

    fn counts(&self) -> PyExecutionCounts {
        PyExecutionCounts { inner: self.inner.counts() }
    }

    fn retention(&self) -> &'static str {
        self.inner.retention().as_str()
    }

    fn resources(&self) -> PyExecutionResources {
        PyExecutionResources { inner: self.inner.resources() }
    }

    fn backend_name(&self) -> &str {
        self.inner.backend().name()
    }

    fn kernel(&self) -> &'static str {
        kernel_name(self.inner.backend().kernel())
    }

    fn status(&self) -> &'static str {
        self.inner.status().as_str()
    }

    fn scientific_identity(&self) -> Option<Vec<u8>> {
        self.inner.scientific_identity().map(digest_bytes)
    }

    fn scientific_contract_changed(&self) -> bool {
        self.inner.scientific_contract_changed()
    }

    fn publishes_claim(&self) -> bool {
        self.inner.publishes_claim()
    }

    /// Digest of the measurements. The scientific digest and the contract flag are not inputs.
    fn physical_identity(&self) -> Vec<u8> {
        digest_bytes(self.inner.physical_identity().digest())
    }

    /// Add counts. Overflow leaves the facts, including the scientific digest, unchanged.
    fn try_add_counts(&mut self, py: Python<'_>, extra: Py<PyExecutionCounts>) -> PyResult<()> {
        let extra = extra.borrow(py).inner;
        self.inner.try_add_counts(extra).map_err(fact_err)
    }

    /// Add resources. Overflow leaves the facts, including the scientific digest, unchanged.
    fn try_add_resources(
        &mut self,
        py: Python<'_>,
        extra: Py<PyExecutionResources>,
    ) -> PyResult<()> {
        let extra = extra.borrow(py).inner;
        self.inner.try_add_resources(extra).map_err(fact_err)
    }

    /// Replace the measurements. The stored scientific digest and the contract flag stay.
    fn rebuild(
        &self,
        py: Python<'_>,
        counts: Py<PyExecutionCounts>,
        retention: &str,
        resources: Py<PyExecutionResources>,
        backend_name: String,
        kernel: &str,
        status: &str,
    ) -> PyResult<Self> {
        let measured =
            measured_execution(py, counts, retention, resources, backend_name, kernel, status)?;
        Ok(Self { inner: self.inner.clone().rebuild(measured) })
    }

    fn with_scientific_identity(&self, scientific_identity: Option<Vec<u8>>) -> PyResult<Self> {
        let identity = digest_of(scientific_identity)?;
        Ok(Self { inner: self.inner.clone().with_scientific_identity(identity) })
    }

    fn declare_numerical_policy_changes_scientific_contract(&self, changes: bool) -> Self {
        Self {
            inner: self.inner.clone().declare_numerical_policy_changes_scientific_contract(changes),
        }
    }

    fn standing(&self) -> PyExecutionFactStanding {
        let measured = self.inner.measured();
        PyExecutionFactStanding {
            integrity: digest_bytes(self.inner.physical_identity().digest()),
            semantic: self.inner.scientific_identity().map(digest_bytes),
            replay: PyNumericalReplay {
                counts: measured.counts(),
                resources: measured.resources(),
                retention: measured.retention().as_str(),
                backend_name: measured.backend().name().to_owned(),
                kernel: kernel_name(measured.backend().kernel()),
                status: measured.status().as_str(),
                contract_changed: self.inner.scientific_contract_changed(),
                invoked_callback: false,
                reexecuted: false,
            },
            resumable: false,
        }
    }
}

/// The four readings of a plan inspection.
///
/// Semantic verification is the target identity the caller already held.
/// A payload checksum, a numerical replay, and a live study are not this record.
#[pyclass(name = "InspectionStanding", frozen)]
struct PyInspectionStanding {
    identity: Vec<u8>,
    has_payload_checksum: bool,
    replays_fit: bool,
    resumable: bool,
}

#[pymethods]
impl PyInspectionStanding {
    /// Refused. Inspection does not carry a payload checksum separate from the target.
    fn integrity(&self) -> PyResult<()> {
        if self.has_payload_checksum {
            Ok(())
        } else {
            Err(crate::unsupported_message(concat!(
                "plan inspection has no payload checksum distinct from ",
                "the scientific target identity",
            )))
        }
    }

    /// Scientific identity of the target copied into the inspection.
    fn semantic_verification(&self) -> &[u8] {
        &self.identity
    }

    /// Refused. Inspection copies held declarations and does not replay a fit.
    fn numerical_replay(&self) -> PyResult<()> {
        if self.replays_fit {
            Ok(())
        } else {
            Err(crate::unsupported_message(
                "plan inspection copies held declarations and does not numerically replay a fit",
            ))
        }
    }

    /// Refused. Inspection does not reconstruct a live study.
    fn resumability(&self) -> PyResult<()> {
        if self.resumable { Ok(()) } else { Err(live_study_refusal()) }
    }
}

/// Parts [`PlanInspection`] may copy. Every part starts absent.
///
/// Absence is not an empty list or a default policy. The static mean target is
/// the inspection fixture; supplying it does not identify or fit.
#[pyclass(name = "InspectionInputs", frozen)]
struct PyInspectionInputs {
    inner: InspectionInputs,
    target: Option<ScientificTarget>,
}

impl PyInspectionInputs {
    fn copied(&self, inner: InspectionInputs) -> Self {
        Self { inner, target: self.target.clone() }
    }
}

#[pymethods]
impl PyInspectionInputs {
    #[new]
    fn new() -> Self {
        Self { inner: InspectionInputs::new(), target: None }
    }

    /// Supply the static mean world from the Rust inspection fixture.
    fn with_static_mean_target(&self) -> PyResult<Self> {
        let target = static_mean_target()?;
        Ok(Self { inner: self.inner.clone().target(target.clone()), target: Some(target) })
    }

    fn with_assumptions(&self, names: Vec<String>) -> PyResult<Self> {
        let assumptions =
            names.iter().map(|name| assumption(name)).collect::<PyResult<Vec<_>>>()?;
        Ok(self.copied(self.inner.clone().assumptions(assumptions)))
    }

    fn with_realization(&self, name: &str) -> PyResult<Self> {
        Ok(self.copied(self.inner.clone().realization(realization(name)?)))
    }

    fn with_evidence(&self, names: Vec<String>) -> PyResult<Self> {
        let mut steps = Vec::with_capacity(names.len());
        for name in &names {
            steps.push(EvidenceTransformation::declared(name.as_str()).map_err(inspection_err)?);
        }
        Ok(self.copied(self.inner.clone().evidence(EvidenceTransformations::declared(steps))))
    }

    fn with_uncertainty(&self, joint: bool, level: f64) -> PyResult<Self> {
        if !(level.is_finite() && level > 0.0 && level < 1.0) {
            return Err(crate::value_err_code(
                reason_code!("invalid_argument"),
                format!(
                    "uncertainty level must be finite and strictly between 0 and 1, got {level}"
                ),
            ));
        }
        Ok(self.copied(self.inner.clone().uncertainty(UncertaintyDemand::declared(joint, level))))
    }

    fn with_reuse(&self, name: &str, frozen_fit: bool) -> PyResult<Self> {
        let reuse = ReuseSnapshot::declared(name, frozen_fit).map_err(inspection_err)?;
        Ok(self.copied(self.inner.clone().reuse(reuse)))
    }

    fn with_materialization(&self, name: &str, copy_bytes: u64) -> PyResult<Self> {
        let materialization =
            MaterializationSnapshot::declared(name, copy_bytes).map_err(inspection_err)?;
        Ok(self.copied(self.inner.clone().materialization(materialization)))
    }

    fn with_resources(
        &self,
        soft_limit_bytes: Option<u64>,
        hard_limit_bytes: Option<u64>,
        estimated_bytes: u64,
    ) -> Self {
        let resources = EstimatedResources::declared(
            MemoryBudget { soft_limit_bytes, hard_limit_bytes },
            estimated_bytes,
        );
        self.copied(self.inner.clone().resources(resources))
    }

    fn with_backend(&self, kernel: &str) -> PyResult<Self> {
        Ok(self.copied(self.inner.clone().backend_policy(kernel_policy(kernel)?)))
    }

    /// Supply read operators in caller order. An empty list is a supplied list.
    ///
    /// Each row is source id, scratch bytes, retained bytes, output bytes.
    /// Declaring the operator does not read a column.
    fn with_read_operators(&self, operators: Vec<(u64, u64, u64, u64)>) -> PyResult<Self> {
        let operators = read_operators(&operators)?;
        Ok(self.copied(self.inner.clone().operators(operators)))
    }

    /// Scientific identity of the supplied target, before inspection.
    fn target_identity(&self) -> Option<Vec<u8>> {
        self.target.as_ref().map(|target| digest_bytes(target.identity()))
    }

    /// Copy the held parts. Missing parts are refused. This method takes no callback.
    fn inspect(&self) -> PyResult<PyPlanInspection> {
        antecedent_core::inspect::inspect(self.inner.clone())
            .map(|inner| PyPlanInspection { inner })
            .map_err(inspection_err)
    }
}

/// Plan record assembled from values the caller already held.
#[pyclass(name = "PlanInspection", frozen)]
struct PyPlanInspection {
    inner: PlanInspection,
}

#[pymethods]
impl PyPlanInspection {
    fn target_identity(&self) -> Vec<u8> {
        digest_bytes(self.inner.target().identity())
    }

    fn target_inference(&self) -> &'static str {
        inference_name(self.inner.target().inference())
    }

    fn assumptions(&self) -> Vec<String> {
        self.inner.assumptions().iter().map(assumption_name).collect()
    }

    fn realization(&self) -> &'static str {
        realization_name(self.inner.realization())
    }

    fn evidence(&self) -> Vec<String> {
        self.inner.evidence().steps().iter().map(|step| step.name().to_owned()).collect()
    }

    fn uncertainty_joint(&self) -> bool {
        self.inner.uncertainty().joint()
    }

    fn uncertainty_level(&self) -> f64 {
        self.inner.uncertainty().level()
    }

    fn uncertainty_level_bits(&self) -> u64 {
        self.inner.uncertainty().level().to_bits()
    }

    fn reuse_name(&self) -> &str {
        self.inner.reuse().name()
    }

    fn reuse_frozen_fit(&self) -> bool {
        self.inner.reuse().frozen_fit()
    }

    fn materialization_name(&self) -> &str {
        self.inner.materialization().name()
    }

    fn materialization_copy_bytes(&self) -> u64 {
        self.inner.materialization().copy_bytes()
    }

    fn memory_soft_bytes(&self) -> Option<u64> {
        self.inner.resources().memory().soft_limit_bytes
    }

    fn memory_hard_bytes(&self) -> Option<u64> {
        self.inner.resources().memory().hard_limit_bytes
    }

    fn estimated_bytes(&self) -> u64 {
        self.inner.resources().estimated_bytes()
    }

    fn backend(&self) -> &'static str {
        kernel_name(self.inner.backend_policy())
    }

    /// Effect, source ids, and declared scratch, retained, and output bytes.
    fn operators(&self) -> Vec<OperatorRow> {
        self.inner
            .operators()
            .iter()
            .map(|operator| {
                let estimates = operator.estimates();
                (
                    operator.effect().as_str().to_owned(),
                    operator.sources().iter().map(|source| source.raw()).collect(),
                    estimates.scratch(),
                    estimates.retained(),
                    estimates.output(),
                )
            })
            .collect()
    }

    fn standing(&self) -> PyInspectionStanding {
        PyInspectionStanding {
            identity: digest_bytes(self.inner.target().identity()),
            has_payload_checksum: false,
            replays_fit: false,
            resumable: false,
        }
    }
}

/// Refused. These bindings do not reconstruct a live study or invoke `loaded`.
#[pyfunction]
#[allow(unused_variables)] // the argument is part of the Python signature and is never called
fn resume_live_study(loaded: &Bound<'_, PyAny>) -> PyResult<()> {
    Err(live_study_refusal())
}

/// Refused. Inspect and execution-receipt facts do not complete a family consumer.
#[pyfunction]
fn consume_interchange_family(family: &str) -> PyResult<()> {
    Err(crate::unsupported_message(format!(
        "interchange family `{family}` has no bound consumer; \
         inspect and execution-receipt facts do not complete family interchange"
    )))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExecutionCounts>()?;
    module.add_class::<PyExecutionResources>()?;
    module.add_class::<PyNumericalReplay>()?;
    module.add_class::<PyExecutionFactStanding>()?;
    module.add_class::<PyExecutionFacts>()?;
    module.add_class::<PyInspectionStanding>()?;
    module.add_class::<PyInspectionInputs>()?;
    module.add_class::<PyPlanInspection>()?;
    module.add_function(wrap_pyfunction!(resume_live_study, module)?)?;
    module.add_function(wrap_pyfunction!(consume_interchange_family, module)?)?;
    Ok(())
}
