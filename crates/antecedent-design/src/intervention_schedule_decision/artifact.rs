//! Versioned evidence and topologically replayed terminal for the integrated decision.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{CanonicalWriter, ExecutionContext, blake3_hex, to_hex};
use antecedent_io_core::artifact_codec::{
    ArtifactContainer, decode_sectioned_io_versioned, encode_sectioned_io,
};
use antecedent_io_core::{IoError, RetainedArtifactIdentity, from_cbor, load_expecting, to_cbor};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    IntegratedScheduleDecision, PreparedIntegratedScheduleDecision, ScheduleDecisionDeclaration,
    ScheduleDecisionError, ScheduleSourceFit, ScheduleSourceSubject, ScheduleTargetCohort,
};
use crate::composition_bundle::{
    BundleConsumer, BundleError, BundleLimits, BundleNode, BundleStage, CompositionBundle,
    NodeFailure, NodeKind, NodeSource, NodeVerifier, SuppliedSources, UpstreamNode, VerifiedNode,
};

pub(crate) const EVIDENCE_KIND: &str = "integrated_schedule_evidence_v1";
pub(crate) const DECISION_KIND: &str = "integrated_schedule_decision_v1";
const VERSION: u16 = 1;
const MAX_BYTES: usize = 32 * 1024 * 1024;
const SECTION: &str = "integrated_schedule.body";
const LAYOUT: &str = "integrated schedule artifact layout";
const FACT: &str = "integrated_schedule_evidence";
const ROLES: [&str; 3] = ["source", "target", "declaration"];
const TERMINAL: &str = "decision";
const CHILD_PROGRAM: &str = "schedule_child_program";
const CHECKED_CHILD: &str = "checked_schedule_child";
const BINARY_OBSERVED_ENDPOINTS: &str = "schedule_binary_observed_endpoints";

/// One independently retained input to the schedule decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScheduleEvidence {
    /// Complete independent source subject trajectories.
    Source {
        /// Subject rows, each retaining its treatment and censoring history.
        rows: Vec<ScheduleSourceSubject>,
    },
    /// Separate independent target sample of the transported baseline.
    Target {
        /// Binary predecision baseline marker for each target subject.
        baseline: Vec<bool>,
    },
    /// Finite schedule, structural and preference states.
    Declaration {
        /// Declarations fixed independently of the samples.
        problem: ScheduleDecisionDeclaration,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
enum EvidenceProduct {
    Source { fit: ScheduleSourceFit },
    Target { cohort: ScheduleTargetCohort },
    Declaration { problem: ScheduleDecisionDeclaration },
}

/// One checked child program, compiled over its complete input and the route's premises.
pub struct CompiledScheduleEvidence {
    evidence: ScheduleEvidence,
    snapshot: [u8; 32],
    program: [u8; 32],
    operation: &'static str,
}

impl CompiledScheduleEvidence {
    /// Bind one source fit, independent target fit or fixed joint preference declaration.
    ///
    /// # Errors
    /// The declared evidence is absent or outside the bounded design.
    pub fn compile(evidence: ScheduleEvidence) -> Result<Self, ScheduleDecisionError> {
        let mut raw = CanonicalWriter::new("antecedent.integrated_schedule.evidence.v1");
        let (operation, premises) = match &evidence {
            ScheduleEvidence::Source { rows } => {
                if rows.is_empty() {
                    return Err(ScheduleDecisionError::MissingJointEvidence);
                }
                if rows.len() > super::MAX_SUBJECTS {
                    return Err(ScheduleDecisionError::InvalidDeclaration);
                }
                raw.text("source").word(rows.len() as u64);
                for row in rows {
                    raw.word(u64::from(row.baseline))
                        .word(u64::from(row.actions[0]))
                        .word(u64::from(row.actions[1]))
                        .word(u64::from(row.endpoint.is_some()));
                    if let Some(value) = row.endpoint {
                        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                            return Err(ScheduleDecisionError::InvalidDeclaration);
                        }
                        raw.real(value);
                    }
                }
                (
                    "joint_assignment_observation_endpoint_cells",
                    "iid_source_trajectory;consistency;conditional_exchangeability_by_declared_structure;conditional_observation_exchangeability;positive_support",
                )
            }
            ScheduleEvidence::Target { baseline } => {
                if baseline.is_empty() {
                    return Err(ScheduleDecisionError::MissingJointEvidence);
                }
                if baseline.len() > super::MAX_SUBJECTS {
                    return Err(ScheduleDecisionError::InvalidDeclaration);
                }
                raw.text("target").word(baseline.len() as u64);
                for &marker in baseline {
                    raw.word(u64::from(marker));
                }
                (
                    "independent_target_baseline_standardization_weights",
                    "iid_target_baseline;independent_source_and_target;transport_conditional_schedule_value_within_baseline_and_structure",
                )
            }
            ScheduleEvidence::Declaration { problem } => {
                problem.check()?;
                raw.text("joint_preferences_and_structures").word(problem.schedules.len() as u64);
                for &code in &problem.schedules {
                    raw.word(u64::from(code));
                }
                raw.word(problem.structures.len() as u64);
                for structure in &problem.structures {
                    raw.text(match structure {
                        super::ScheduleStructure::BothActionsCausal => "both_actions_causal",
                        super::ScheduleStructure::FirstActionOnly => "first_action_only",
                    });
                }
                raw.word(problem.preference_costs.len() as u64);
                for costs in &problem.preference_costs {
                    raw.reals(costs);
                }
                raw.text(match problem.calibration_design {
                    Some(super::ScheduleCalibrationDesign::ExampleUniqueOptima) => {
                        "example_unique_optima"
                    }
                    Some(super::ScheduleCalibrationDesign::ExampleTiedOptima) => {
                        "example_tied_optima"
                    }
                    None => "no_source_fixed_calibration_design",
                });
                (
                    "prespecified_structure_preference_worlds",
                    "finite_unweighted_worlds;known_schedule_costs;fixed_before_sampling;retain_all_tied_maximizers",
                )
            }
        };
        let snapshot = raw.finish();
        let mut program = CanonicalWriter::new("antecedent.integrated_schedule.child_program.v1");
        program.text(operation).text(premises).bytes(&snapshot);
        Ok(Self { evidence, snapshot, program: program.finish(), operation })
    }

    /// Identity of the full child operation, premises and source.
    #[must_use]
    pub const fn program_identity(&self) -> [u8; 32] {
        self.program
    }

    /// Identity of the independently retained raw evidence.
    #[must_use]
    pub const fn snapshot_identity(&self) -> [u8; 32] {
        self.snapshot
    }

    fn execute(&self) -> Result<EvidenceBinding, ScheduleDecisionError> {
        let result = match &self.evidence {
            ScheduleEvidence::Source { rows } => {
                EvidenceProduct::Source { fit: ScheduleSourceFit::fit(rows)? }
            }
            ScheduleEvidence::Target { baseline } => {
                EvidenceProduct::Target { cohort: ScheduleTargetCohort::fit(baseline)? }
            }
            ScheduleEvidence::Declaration { problem } => {
                EvidenceProduct::Declaration { problem: problem.clone() }
            }
        };
        Ok(EvidenceBinding {
            snapshot: self.snapshot,
            program: self.program,
            operation: self.operation.into(),
            result,
        })
    }
}

impl EvidenceProduct {
    fn role(&self) -> usize {
        match self {
            Self::Source { .. } => 0,
            Self::Target { .. } => 1,
            Self::Declaration { .. } => 2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceWire<Evidence> {
    version: u16,
    program_identity: [u8; 32],
    evidence: Evidence,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBinding {
    snapshot: [u8; 32],
    program: [u8; 32],
    operation: String,
    result: EvidenceProduct,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TerminalWire {
    version: u16,
    parents: [String; 3],
    program_identity: [u8; 32],
    result: IntegratedScheduleDecision,
}

/// Result of running the sealed decision through its typed evidence children.
pub struct IntegratedScheduleBundle {
    /// Portable bundle containing the evidence routes and terminal result.
    pub bundle: CompositionBundle,
    /// Decision with an explicit joint coverage scope.
    pub decision: IntegratedScheduleDecision,
    /// Semantic identity of the compiled decision program.
    pub program_identity: [u8; 32],
    /// Program identities of the source, target and declared finite worlds, in that order.
    pub child_program_identities: [[u8; 32]; 3],
}

/// Failure to compile, serialize or consume a schedule decision bundle.
#[derive(Debug, Error)]
pub enum ScheduleBundleError {
    /// A scientific declaration or support failure.
    #[error(transparent)]
    Procedure(#[from] ScheduleDecisionError),
    /// Artifact read or write failure.
    #[error(transparent)]
    Io(#[from] IoError),
    /// Bundle identity, graph or dependency failure.
    #[error(transparent)]
    Bundle(#[from] BundleError),
    /// An expected role is absent or supplied as an unresolved reference.
    #[error(
        "embed the verified source trajectories, independent target baseline sample, and joint structural/preference declaration as the three decision parents"
    )]
    MissingParent,
    /// The bundle's program disagrees with independent compilation of its evidence.
    #[error("the retained schedule decision program differs from the recompiled program")]
    ChangedProgram,
    /// The causal compiler declined execution at the support boundary.
    #[error("integrated schedule decision support refused: {0}")]
    Unsupported(String),
    /// The caller's resource limit prevented execution.
    #[error("the integrated decision exceeded the execution admission budget: {0}")]
    Resource(String),
}

fn decode<T: serde::de::DeserializeOwned>(kind: &str, bytes: &[u8]) -> Result<T, IoError> {
    decode_sectioned_io_versioned(kind, MAX_BYTES, SECTION, &[], LAYOUT, VERSION, bytes)
        .map(|(value, _)| value)
}

fn encode<T: Serialize>(kind: &str, artifact_id: &str, value: &T) -> Result<Vec<u8>, IoError> {
    let header = ArtifactContainer {
        kind,
        note: "integrated intervention schedule decision",
        max_bytes: MAX_BYTES,
        artifact_id,
        library_version: env!("CARGO_PKG_VERSION"),
    };
    encode_sectioned_io(&header, SECTION, value, &[], LAYOUT)
}

fn digest<T: Serialize>(value: &T) -> Result<String, IoError> {
    Ok(blake3_hex(&to_cbor(value)?))
}

fn decode_binding(fact: &str) -> Option<EvidenceBinding> {
    if fact.len() > 16_384 || !fact.len().is_multiple_of(2) {
        return None;
    }
    let bytes = fact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            Some(((pair[0] as char).to_digit(16)? * 16 + (pair[1] as char).to_digit(16)?) as u8)
        })
        .collect::<Option<Vec<_>>>()?;
    from_cbor(&bytes).ok()
}

fn verified_evidence(binding: &EvidenceBinding) -> Result<VerifiedNode, IoError> {
    let mut facts = BTreeMap::from([
        (FACT.into(), to_hex(&to_cbor(binding)?)),
        (CHILD_PROGRAM.into(), to_hex(&binding.program)),
        ("schedule_child_operation".into(), binding.operation.clone()),
    ]);
    if let EvidenceProduct::Source { fit } = &binding.result {
        facts.insert(BINARY_OBSERVED_ENDPOINTS.into(), fit.binary_observed_endpoints.to_string());
    }
    Ok(VerifiedNode {
        identity: digest(binding)?,
        facts,
        values: vec![(CHECKED_CHILD.into(), 1.0)],
    })
}

fn verified_terminal(terminal: &TerminalWire) -> Result<VerifiedNode, IoError> {
    let optimum =
        terminal.result.optimum_set.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
    let mut values = vec![("minimum_coverage".into(), terminal.result.minimum_coverage)];
    for (index, scenario) in terminal.result.scenarios.iter().enumerate() {
        values.extend(scenario.point_utilities.iter().map(|&(schedule, value)| {
            (format!("scenario.{index}.schedule.{schedule}.utility"), value)
        }));
    }
    Ok(VerifiedNode {
        identity: digest(terminal)?,
        facts: BTreeMap::from([
            ("optimum_set".into(), optimum),
            ("coverage_scope".into(), terminal.result.coverage_scope.clone()),
            ("calibration_status".into(), terminal.result.calibration().status.as_str().into()),
        ]),
        values,
    })
}

/// Compile all retained evidence children, run them, and seal the resulting terminal in a bundle.
pub fn compile_program_schedule_bundle(
    children: &[CompiledScheduleEvidence; 3],
    program_identity: [u8; 32],
    ctx: &ExecutionContext,
) -> Result<IntegratedScheduleBundle, ScheduleBundleError> {
    let charge = children.iter().fold(0_u64, |bytes, child| {
        bytes.saturating_add(match &child.evidence {
            ScheduleEvidence::Source { rows } => (rows.len() as u64).saturating_mul(128),
            ScheduleEvidence::Target { baseline } => (baseline.len() as u64).saturating_mul(4),
            ScheduleEvidence::Declaration { .. } => 16_384,
        })
    });
    let _lease = ctx
        .acquire_serial_numerical_admission(charge)
        .map_err(|error| ScheduleBundleError::Resource(error.to_string()))?;
    let child_program_identities =
        children.each_ref().map(CompiledScheduleEvidence::program_identity);
    let mut nodes = Vec::with_capacity(4);
    let mut products = Vec::with_capacity(3);
    let mut parents = Vec::with_capacity(3);
    for (index, (role, child)) in ROLES.iter().zip(children).enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(ScheduleDecisionError::Cancelled.into());
        }
        let binding = child.execute()?;
        if binding.result.role() != index {
            return Err(ScheduleBundleError::MissingParent);
        }
        let identity = digest(&binding)?;
        let bytes = encode(
            EVIDENCE_KIND,
            role,
            &EvidenceWire {
                version: VERSION,
                program_identity: child.program,
                evidence: &child.evidence,
            },
        )?;
        nodes.push(BundleNode::embedded(
            role,
            NodeKind::IntegratedScheduleEvidence,
            &identity,
            bytes,
        ));
        parents.push(identity);
        products.push(binding.result);
    }
    let result = run(products, ctx)?;
    let terminal = TerminalWire {
        version: VERSION,
        program_identity,
        parents: parents.try_into().map_err(|_| ScheduleBundleError::MissingParent)?,
        result: result.clone(),
    };
    let identity = digest(&terminal)?;
    nodes.push(BundleNode::embedded(
        TERMINAL,
        NodeKind::IntegratedScheduleDecision,
        &identity,
        encode(DECISION_KIND, TERMINAL, &terminal)?,
    ));
    let edges = ROLES.map(|role| (role.to_owned(), TERMINAL.to_owned()));
    let bundle = CompositionBundle::new(nodes, &edges, &BundleLimits::default())?;
    Ok(IntegratedScheduleBundle {
        bundle,
        decision: result,
        program_identity,
        child_program_identities,
    })
}

/// Raw retained evidence and the semantic program identity required by the terminal.
pub struct ScheduleReplayInputs {
    /// Full source trajectories in their original order.
    pub source: Vec<ScheduleSourceSubject>,
    /// Separately sampled target baseline rows.
    pub target: Vec<bool>,
    /// Prespecified structural alternatives and utility preferences.
    pub problem: ScheduleDecisionDeclaration,
    /// Semantic program identity recorded by the terminal.
    pub program_identity: [u8; 32],
}

fn require_node<'a>(
    bundle: &'a CompositionBundle,
    id: &str,
    kind: NodeKind,
) -> Result<&'a [u8], ScheduleBundleError> {
    let node = bundle
        .nodes()
        .iter()
        .find(|node| node.id == id && node.kind == kind)
        .ok_or(ScheduleBundleError::MissingParent)?;
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(ScheduleBundleError::MissingParent);
    };
    Ok(bytes)
}

/// Read the bounded inputs an independent compiler needs before replay.
///
/// # Errors
/// The expected bundle differs, an evidence role is absent, or its payload does not decode.
pub fn inspect_program_schedule_bundle(
    bytes: &[u8],
    expected_identity: &str,
) -> Result<ScheduleReplayInputs, ScheduleBundleError> {
    let bundle = CompositionBundle::from_bytes(bytes, expected_identity, &BundleLimits::default())?;
    let [source, target, declaration] =
        ROLES.map(|role| require_node(&bundle, role, NodeKind::IntegratedScheduleEvidence));
    let source: EvidenceWire<ScheduleEvidence> = decode(EVIDENCE_KIND, source?)?;
    let target: EvidenceWire<ScheduleEvidence> = decode(EVIDENCE_KIND, target?)?;
    let declaration: EvidenceWire<ScheduleEvidence> = decode(EVIDENCE_KIND, declaration?)?;
    let TerminalWire { program_identity, .. } = decode(
        DECISION_KIND,
        require_node(&bundle, TERMINAL, NodeKind::IntegratedScheduleDecision)?,
    )?;
    let (
        ScheduleEvidence::Source { rows: source },
        ScheduleEvidence::Target { baseline: target },
        ScheduleEvidence::Declaration { problem },
    ) = (source.evidence, target.evidence, declaration.evidence)
    else {
        return Err(ScheduleBundleError::MissingParent);
    };
    Ok(ScheduleReplayInputs { source, target, problem, program_identity })
}

/// Verify the retained byte identity before calling [`inspect_program_schedule_bundle`].
///
/// # Errors
///
/// Returns the binding refusal on a digest mismatch or the underlying reader error.
pub fn inspect_program_schedule_bundle_expecting(
    bytes: &[u8],
    expected_identity: &str,
    retained: &RetainedArtifactIdentity,
) -> Result<ScheduleReplayInputs, ScheduleBundleError> {
    load_expecting(bytes, retained, |bytes| inspect_program_schedule_bundle(bytes, expected_identity))
}

/// Independently replay a retained integrated schedule bundle from its embedded evidence.
///
/// # Errors
/// The expected bundle differs, a scientific input is absent, or a child fails to replay.
pub fn consume_program_schedule_bundle(
    bytes: &[u8],
    expected_identity: &str,
    program_identity: [u8; 32],
    ctx: &ExecutionContext,
) -> Result<IntegratedScheduleDecision, ScheduleBundleError> {
    let _lease = ctx
        .acquire_serial_numerical_admission((bytes.len() as u64).saturating_mul(4))
        .map_err(|error| ScheduleBundleError::Resource(error.to_string()))?;
    let limits = BundleLimits::default();
    let bundle = CompositionBundle::from_bytes(bytes, expected_identity, &limits)?;
    let terminal: TerminalWire = decode(
        DECISION_KIND,
        require_node(&bundle, TERMINAL, NodeKind::IntegratedScheduleDecision)?,
    )?;
    if terminal.program_identity != program_identity {
        return Err(ScheduleBundleError::ChangedProgram);
    }
    let verified = BundleConsumer::new()
        .register(Box::new(EvidenceVerifier))
        .register(Box::new(DecisionVerifier(ctx.clone())))
        .consume(bytes, &limits, expected_identity, &SuppliedSources::default())?;
    verified.require_verified()?;
    if !verified.all_replayed() {
        return Err(ScheduleBundleError::MissingParent);
    }
    Ok(terminal.result)
}

/// Verify the retained byte identity before calling [`consume_program_schedule_bundle`].
///
/// # Errors
///
/// Returns the binding refusal on a digest mismatch or the underlying reader error.
pub fn consume_program_schedule_bundle_expecting(
    bytes: &[u8],
    expected_identity: &str,
    program_identity: [u8; 32],
    ctx: &ExecutionContext,
    retained: &RetainedArtifactIdentity,
) -> Result<IntegratedScheduleDecision, ScheduleBundleError> {
    load_expecting(bytes, retained, |bytes| consume_program_schedule_bundle(bytes, expected_identity, program_identity, ctx))
}

fn run(
    products: Vec<EvidenceProduct>,
    ctx: &ExecutionContext,
) -> Result<IntegratedScheduleDecision, ScheduleBundleError> {
    let [
        EvidenceProduct::Source { fit },
        EvidenceProduct::Target { cohort },
        EvidenceProduct::Declaration { problem },
    ]: [EvidenceProduct; 3] =
        products.try_into().map_err(|_| ScheduleBundleError::MissingParent)?
    else {
        return Err(ScheduleBundleError::MissingParent);
    };
    Ok(PreparedIntegratedScheduleDecision::compile(fit, cohort, problem)?.execute(ctx)?)
}

fn failure(stage: BundleStage, reason: impl ToString) -> NodeFailure {
    NodeFailure { stage, reason: reason.to_string() }
}

pub(crate) fn inspect_evidence(bytes: &[u8]) -> Result<VerifiedNode, NodeFailure> {
    let artifact: EvidenceWire<ScheduleEvidence> = decode(EVIDENCE_KIND, bytes)
        .map_err(|error| failure(BundleStage::SwappedEvidence, error))?;
    let child = CompiledScheduleEvidence::compile(artifact.evidence)
        .map_err(|error| failure(BundleStage::SwappedEvidence, error))?;
    if artifact.program_identity != child.program_identity() {
        return Err(failure(
            BundleStage::SwappedEvidence,
            "integrated schedule evidence differs from its compiled child program",
        ));
    }
    let binding = child.execute().map_err(|error| failure(BundleStage::SwappedEvidence, error))?;
    verified_evidence(&binding).map_err(|error| failure(BundleStage::SwappedEvidence, error))
}

pub(crate) fn inspect_terminal(bytes: &[u8]) -> Result<VerifiedNode, NodeFailure> {
    let artifact: TerminalWire = decode(DECISION_KIND, bytes)
        .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
    verified_terminal(&artifact).map_err(|error| failure(BundleStage::TamperedQuantity, error))
}

pub(crate) struct EvidenceVerifier;

impl NodeVerifier for EvidenceVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::IntegratedScheduleEvidence
    }

    fn verify(
        &self,
        _id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        inspect_evidence(bytes)
    }
}

pub(crate) struct DecisionVerifier(pub(crate) ExecutionContext);

impl NodeVerifier for DecisionVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::IntegratedScheduleDecision
    }

    fn verify(
        &self,
        _id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let terminal: TerminalWire = decode(DECISION_KIND, bytes)
            .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
        if upstream.len() != ROLES.len()
            || upstream.iter().any(|node| node.kind != NodeKind::IntegratedScheduleEvidence)
        {
            return Err(failure(
                BundleStage::CallbackUnavailable,
                ScheduleBundleError::MissingParent,
            ));
        }
        let mut inputs: [Option<EvidenceProduct>; 3] = [None, None, None];
        for parent in upstream {
            if !parent.values.iter().any(|(name, _)| name == CHECKED_CHILD) {
                return Err(failure(
                    BundleStage::CallbackUnavailable,
                    ScheduleBundleError::MissingParent,
                ));
            }
            let mut binding: EvidenceBinding =
                parent.facts.get(FACT).and_then(|fact| decode_binding(fact)).ok_or_else(|| {
                    failure(BundleStage::CallbackUnavailable, ScheduleBundleError::MissingParent)
                })?;
            let role = binding.result.role();
            let identity =
                digest(&binding).map_err(|error| failure(BundleStage::SwappedEvidence, error))?;
            if parent.identity != identity
                || terminal.parents[role] != identity
                || inputs[role].is_some()
                || parent.facts.get(CHILD_PROGRAM) != Some(&to_hex(&binding.program))
            {
                return Err(failure(
                    BundleStage::SwappedEvidence,
                    "schedule decision parent identity changed",
                ));
            }
            if let EvidenceProduct::Source { fit } = &mut binding.result {
                fit.binary_observed_endpoints = parent
                    .facts
                    .get(BINARY_OBSERVED_ENDPOINTS)
                    .is_some_and(|status| status == "true");
            }
            inputs[role] = Some(binding.result);
        }
        if inputs.iter().any(Option::is_none) {
            return Err(failure(
                BundleStage::CallbackUnavailable,
                ScheduleBundleError::MissingParent,
            ));
        }
        let replayed = run(inputs.into_iter().flatten().collect(), &self.0)
            .map_err(|error| failure(BundleStage::TamperedQuantity, error))?;
        if replayed != terminal.result {
            return Err(failure(
                BundleStage::TamperedQuantity,
                "joint schedule optimum set differs on replay",
            ));
        }
        verified_terminal(&terminal).map_err(|error| failure(BundleStage::TamperedQuantity, error))
    }
}
