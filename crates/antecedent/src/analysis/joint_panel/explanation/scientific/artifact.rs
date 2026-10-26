//! Versioned component recipe and numeric replay tied to the original native design.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    ChangeComponent, ExecutionContext, JointPanelError, PanelDecomposition,
    PopulationReferenceBindings, PopulationReferenceDecomposition, ReferenceComponentBindings,
    ScientificConsumptionScope, ScientificReferenceDecomposition, ScientificWorldId,
    SemanticDigest, VariableId, invalid,
};
use serde::{Deserialize, Serialize};
/// Largest reference-decomposition or population-reference artifact under
/// [`ReferenceArtifactLimits::default`].
pub const MAX_BYTES: usize = 24 * 1024 * 1024;

/// Byte bound of a reference-decomposition or population-reference artifact, checked before
/// decoding and after encoding. The embedded original design is bounded by the same value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceArtifactLimits {
    /// Largest artifact body.
    pub max_bytes: usize,
}

impl Default for ReferenceArtifactLimits {
    fn default() -> Self {
        Self { max_bytes: MAX_BYTES }
    }
}

impl ReferenceArtifactLimits {
    fn design(self) -> crate::analysis::joint_panel::designs::DesignArtifactLimits {
        crate::analysis::joint_panel::designs::DesignArtifactLimits { max_bytes: self.max_bytes }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingsWire {
    exposure: Vec<u32>,
    saturation: Vec<u32>,
    mechanism: Vec<u32>,
    hybrids: [u32; 16],
    order: [ChangeComponent; 4],
}
impl BindingsWire {
    fn from_domain(bindings: &ReferenceComponentBindings) -> Self {
        Self {
            exposure: bindings.exposure.iter().map(|id| id.raw()).collect(),
            saturation: bindings.saturation.iter().map(|id| id.raw()).collect(),
            mechanism: bindings.mechanism.iter().map(|id| id.raw()).collect(),
            hybrids: bindings.hybrids.map(ScientificWorldId::raw),
            order: bindings.order,
        }
    }
    fn to_domain(&self) -> ReferenceComponentBindings {
        ReferenceComponentBindings {
            exposure: self.exposure.iter().copied().map(VariableId::from_raw).collect(),
            saturation: self.saturation.iter().copied().map(VariableId::from_raw).collect(),
            mechanism: self.mechanism.iter().copied().map(VariableId::from_raw).collect(),
            hybrids: self.hybrids.map(ScientificWorldId::from_raw),
            order: self.order,
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceArtifact {
    version: u16,
    design: Vec<u8>,
    bindings: BindingsWire,
    identified: bool,
    digest: [u8; 32],
}
fn digest(
    result: &ScientificReferenceDecomposition,
    ctx: &ExecutionContext,
) -> Result<[u8; 32], JointPanelError> {
    numerical_digest(
        &result.decomposition,
        result.covariance.as_ref(),
        result.question_identity,
        ctx,
    )
}
fn numerical_digest(
    decomposition: &PanelDecomposition,
    covariance: Option<&super::super::engine::decomposition::ReferenceDecompositionCovariance>,
    question_identity: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<[u8; 32], JointPanelError> {
    let bytes = decomposition
        .draws
        .len()
        .checked_mul(16 * 8 + 16)
        .and_then(|n| n.checked_add(8192))
        .ok_or_else(|| invalid("reference replay quote overflow"))?;
    let _lease = ctx
        .acquire_admission(0, bytes as u64)
        .map_err(|error| super::super::engine::PanelError::Resource(error.to_string()))?;
    let mut writer = antecedent_core::CanonicalWriter::new(
        "antecedent.scientific_reference_decomposition.values.v1",
    );
    writer.bytes(question_identity.as_bytes());
    for value in decomposition.point.hybrids {
        writer.word(value.to_bits());
    }
    writer.word(decomposition.draws.len() as u64);
    for draw in &decomposition.draws {
        match draw {
            None => {
                writer.word(0);
            }
            Some(draw) => {
                writer.word(1);
                for value in draw.hybrids {
                    writer.word(value.to_bits());
                }
            }
        }
    }
    match &decomposition.weights {
        None => {
            writer.word(0);
        }
        Some(weights) => {
            writer.word(1);
            for weight in weights {
                writer.word(weight.to_bits());
            }
        }
    }
    if let Some(covariance) = covariance {
        writer.word(1);
        writer.word(covariance.total.to_bits());
        for value in covariance
            .shapley
            .iter()
            .flatten()
            .chain(covariance.sequential.iter().flatten())
            .chain(covariance.by_order.iter().flatten().flatten())
        {
            writer.word(value.to_bits());
        }
    } else {
        writer.word(0);
    }
    Ok(writer.finish())
}
fn decode(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: ReferenceArtifactLimits,
) -> Result<(ReferenceArtifact, antecedent_core::execution::AdmissionLease), JointPanelError> {
    if bytes.is_empty() || bytes.len() > limits.max_bytes {
        return Err(invalid("reference artifact exceeds the declared decode size"));
    }
    let quote = (bytes.len() as u64)
        .checked_mul(8)
        .ok_or_else(|| invalid("reference decode quote overflow"))?;
    let lease = ctx
        .acquire_admission(0, quote)
        .map_err(|error| super::super::engine::PanelError::Resource(error.to_string()))?;
    let wire: ReferenceArtifact =
        antecedent_io_core::convert::from_cbor(bytes).map_err(JointPanelError::compile)?;
    if wire.version != 1 {
        return Err(invalid("unsupported reference decomposition artifact version"));
    }
    Ok((wire, lease))
}
impl crate::analysis::joint_panel::designs::JointDesignResult {
    /// Retain the complete original design, explicit component bindings, scope and
    /// reproducible numeric digest. No release calibration or current evidence is inferred.
    /// # Errors
    /// Original decomposition, encoding, size, resources or cancellation refusal.
    pub fn export_reference_decomposition(
        &self,
        bindings: &ReferenceComponentBindings,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointPanelError> {
        self.export_reference_decomposition_with_limits(
            bindings,
            scope,
            ctx,
            ReferenceArtifactLimits::default(),
        )
    }
    /// [`Self::export_reference_decomposition`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::export_reference_decomposition`], with `limits.max_bytes` as the size bound.
    pub fn export_reference_decomposition_with_limits(
        &self,
        bindings: &ReferenceComponentBindings,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
        limits: ReferenceArtifactLimits,
    ) -> Result<Vec<u8>, JointPanelError> {
        let result = self.decompose_reference_components(bindings, scope, ctx)?;
        let wire = ReferenceArtifact {
            version: 1,
            design: self.export_artifact_with_limits(limits.design())?,
            bindings: BindingsWire::from_domain(bindings),
            identified: scope == ScientificConsumptionScope::IdentifiedCausal,
            digest: digest(&result, ctx)?,
        };
        let bytes =
            antecedent_io_core::convert::to_cbor(&wire).map_err(JointPanelError::compile)?;
        if bytes.len() > limits.max_bytes {
            return Err(invalid("reference decomposition artifact exceeds the declared size"));
        }
        Ok(bytes)
    }
    /// Replay against independently retained current design and caller-requested
    /// components/scope. Artifact declarations cannot silently change the current question.
    /// # Errors
    /// Changed source/proof/data, bindings, scope, numeric replay, resources or cancellation.
    pub fn consume_reference_decomposition_current(
        &self,
        bytes: &[u8],
        bindings: &ReferenceComponentBindings,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
    ) -> Result<ScientificReferenceDecomposition, JointPanelError> {
        self.consume_reference_decomposition_current_with_limits(
            bytes,
            bindings,
            scope,
            ctx,
            ReferenceArtifactLimits::default(),
        )
    }
    /// [`Self::consume_reference_decomposition_current`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::consume_reference_decomposition_current`], with `limits.max_bytes` as the
    /// size bound.
    pub fn consume_reference_decomposition_current_with_limits(
        &self,
        bytes: &[u8],
        bindings: &ReferenceComponentBindings,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
        limits: ReferenceArtifactLimits,
    ) -> Result<ScientificReferenceDecomposition, JointPanelError> {
        let (wire, _lease) = decode(bytes, ctx, limits)?;
        if wire.bindings != BindingsWire::from_domain(bindings)
            || wire.identified != (scope == ScientificConsumptionScope::IdentifiedCausal)
        {
            return Err(invalid(
                "reference artifact differs from caller-current component question or scope",
            ));
        }
        let current =
            self.prepared().consume_current_with_limits(&wire.design, ctx, limits.design())?;
        if current.export_artifact_with_limits(limits.design())?
            != self.export_artifact_with_limits(limits.design())?
        {
            return Err(invalid(
                "reference artifact differs from the caller-current statistical execution, prior, draw law or seed",
            ));
        }
        let result = current.decompose_reference_components(bindings, scope, ctx)?;
        if digest(&result, ctx)? != wire.digest {
            return Err(invalid("reference decomposition numeric replay mismatch"));
        }
        Ok(result)
    }
}

/// Independent historical replay, containing numerical outputs and archived question
/// identity. It exposes no scientific source that could authorize current consumption.
#[derive(Clone, Debug)]
pub struct HistoricalReferenceDecomposition {
    /// Replayed original per-slot full-procedure failure diagnostics.
    pub procedure_failures: Vec<Option<String>>,
    /// Reproduced numerical law and component outputs.
    pub decomposition: PanelDecomposition,
    /// Reproduced analytic joint covariance from the native method owner.
    pub covariance: Option<super::super::engine::decomposition::ReferenceDecompositionCovariance>,
    /// Exact archived component declaration.
    pub bindings: ReferenceComponentBindings,
    /// Original full question identity; historical replay supplies no current authority.
    pub question_identity: SemanticDigest,
}
/// Independently rerun the original design-native procedure and reference decomposition.
/// Current consumption additionally requires a caller-retained design and explicit
/// expected bindings/scope through `consume_reference_decomposition_current`.
/// # Errors
/// Invalid schema/recipe, failed numeric replay, changed digest, resources or cancellation.
pub fn replay_reference_decomposition(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<HistoricalReferenceDecomposition, JointPanelError> {
    replay_reference_decomposition_with_limits(bytes, ctx, ReferenceArtifactLimits::default())
}
/// [`replay_reference_decomposition`] under a caller-declared byte limit.
/// # Errors
/// As [`replay_reference_decomposition`], with `limits.max_bytes` as the size bound.
pub fn replay_reference_decomposition_with_limits(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: ReferenceArtifactLimits,
) -> Result<HistoricalReferenceDecomposition, JointPanelError> {
    let (wire, _lease) = decode(bytes, ctx, limits)?;
    let original = crate::analysis::joint_panel::designs::replay_joint_design_execution(
        &wire.design,
        ctx,
        limits.design(),
    )?;
    let bindings = wire.bindings.to_domain();
    let scope = if wire.identified {
        ScientificConsumptionScope::IdentifiedCausal
    } else {
        ScientificConsumptionScope::ModelConditional
    };
    let result = original.decompose_reference_components(&bindings, scope, ctx)?;
    if digest(&result, ctx)? != wire.digest {
        return Err(invalid("historical reference decomposition numeric replay mismatch"));
    }
    Ok(HistoricalReferenceDecomposition {
        procedure_failures: result.procedure_failures,
        decomposition: result.decomposition,
        covariance: result.covariance,
        bindings,
        question_identity: result.question_identity,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PopulationBindingsWire {
    baseline: u32,
    strata: Vec<f64>,
    worlds: Vec<[u32; 16]>,
    before: Vec<f64>,
    after: Vec<f64>,
    components: BindingsWire,
}
impl PopulationBindingsWire {
    fn from_domain(bindings: &PopulationReferenceBindings) -> Self {
        Self {
            baseline: bindings.baseline.raw(),
            strata: bindings.strata.clone(),
            worlds: bindings
                .stratum_worlds
                .iter()
                .map(|worlds| worlds.map(ScientificWorldId::raw))
                .collect(),
            before: bindings.before_weights.clone(),
            after: bindings.after_weights.clone(),
            components: BindingsWire::from_domain(&bindings.components),
        }
    }
    fn to_domain(&self) -> PopulationReferenceBindings {
        PopulationReferenceBindings {
            baseline: VariableId::from_raw(self.baseline),
            strata: self.strata.clone(),
            stratum_worlds: self
                .worlds
                .iter()
                .map(|worlds| worlds.map(ScientificWorldId::from_raw))
                .collect(),
            before_weights: self.before.clone(),
            after_weights: self.after.clone(),
            components: self.components.to_domain(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagramWire {
    graph: antecedent_io_core::wire::AdmgWire,
    selection: Vec<u32>,
}
impl DiagramWire {
    fn from_domain(diagram: &antecedent_graph::SelectionDiagram) -> Result<Self, JointPanelError> {
        Ok(Self {
            graph: antecedent_io::admg_to_wire(diagram.causal_graph())
                .map_err(JointPanelError::compile)?,
            selection: diagram.selection_targets().iter().map(|id| id.raw()).collect(),
        })
    }
    fn to_domain(&self) -> Result<antecedent_graph::SelectionDiagram, JointPanelError> {
        antecedent_graph::SelectionDiagram::try_new(
            antecedent_io::admg_from_wire(&self.graph).map_err(JointPanelError::compile)?,
            self.selection.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(JointPanelError::compile)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PopulationArtifact {
    version: u16,
    design: Vec<u8>,
    bindings: PopulationBindingsWire,
    diagram: DiagramWire,
    identified: bool,
    digest: [u8; 32],
}
fn decode_population(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: ReferenceArtifactLimits,
) -> Result<(PopulationArtifact, antecedent_core::execution::AdmissionLease), JointPanelError> {
    if bytes.is_empty() || bytes.len() > limits.max_bytes {
        return Err(invalid("population-reference artifact exceeds the declared decode size"));
    }
    let quote = (bytes.len() as u64)
        .checked_mul(8)
        .ok_or_else(|| invalid("population-reference decode quote overflow"))?;
    let lease = ctx
        .acquire_admission(0, quote)
        .map_err(|error| super::super::engine::PanelError::Resource(error.to_string()))?;
    let wire: PopulationArtifact =
        antecedent_io_core::convert::from_cbor(bytes).map_err(JointPanelError::compile)?;
    if wire.version != 1 {
        return Err(invalid("unsupported population-reference artifact version"));
    }
    Ok((wire, lease))
}
impl crate::analysis::joint_panel::designs::JointDesignResult {
    /// Versioned native design, component/population recipe and actual selection diagram.
    /// # Errors
    /// Scientific/numerical execution, encoding, size, resources or cancellation refusal.
    pub fn export_population_reference(
        &self,
        bindings: &PopulationReferenceBindings,
        diagram: &antecedent_graph::SelectionDiagram,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, JointPanelError> {
        self.export_population_reference_with_limits(
            bindings,
            diagram,
            scope,
            ctx,
            ReferenceArtifactLimits::default(),
        )
    }
    /// [`Self::export_population_reference`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::export_population_reference`], with `limits.max_bytes` as the size bound.
    pub fn export_population_reference_with_limits(
        &self,
        bindings: &PopulationReferenceBindings,
        diagram: &antecedent_graph::SelectionDiagram,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
        limits: ReferenceArtifactLimits,
    ) -> Result<Vec<u8>, JointPanelError> {
        let result = self.decompose_population_reference(bindings, diagram, scope, ctx)?;
        let wire = PopulationArtifact {
            version: 1,
            design: self.export_artifact_with_limits(limits.design())?,
            bindings: PopulationBindingsWire::from_domain(bindings),
            diagram: DiagramWire::from_domain(diagram)?,
            identified: scope == ScientificConsumptionScope::IdentifiedCausal,
            digest: numerical_digest(
                &result.decomposition,
                result.covariance.as_ref(),
                result.transport.identity(),
                ctx,
            )?,
        };
        let bytes =
            antecedent_io_core::convert::to_cbor(&wire).map_err(JointPanelError::compile)?;
        if bytes.len() > limits.max_bytes {
            return Err(invalid("population reference artifact exceeds the declared size"));
        }
        Ok(bytes)
    }
    /// Recheck archived population transport against caller-current design, diagram,
    /// component bindings, target population laws and consumption scope.
    /// # Errors
    /// Changed current premises, source, recipe, numerical replay, resources or cancellation.
    pub fn consume_population_reference_current(
        &self,
        bytes: &[u8],
        bindings: &PopulationReferenceBindings,
        diagram: &antecedent_graph::SelectionDiagram,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
    ) -> Result<PopulationReferenceDecomposition, JointPanelError> {
        self.consume_population_reference_current_with_limits(
            bytes,
            bindings,
            diagram,
            scope,
            ctx,
            ReferenceArtifactLimits::default(),
        )
    }
    /// [`Self::consume_population_reference_current`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::consume_population_reference_current`], with `limits.max_bytes` as the size
    /// bound.
    pub fn consume_population_reference_current_with_limits(
        &self,
        bytes: &[u8],
        bindings: &PopulationReferenceBindings,
        diagram: &antecedent_graph::SelectionDiagram,
        scope: ScientificConsumptionScope,
        ctx: &ExecutionContext,
        limits: ReferenceArtifactLimits,
    ) -> Result<PopulationReferenceDecomposition, JointPanelError> {
        let (wire, _lease) = decode_population(bytes, ctx, limits)?;
        if wire.bindings != PopulationBindingsWire::from_domain(bindings)
            || wire.diagram != DiagramWire::from_domain(diagram)?
            || wire.identified != (scope == ScientificConsumptionScope::IdentifiedCausal)
        {
            return Err(invalid(
                "population-reference artifact differs from caller-current transport question",
            ));
        }
        let current =
            self.prepared().consume_current_with_limits(&wire.design, ctx, limits.design())?;
        if current.export_artifact_with_limits(limits.design())?
            != self.export_artifact_with_limits(limits.design())?
        {
            return Err(invalid(
                "reference artifact differs from the caller-current statistical execution, prior, draw law or seed",
            ));
        }
        let result = current.decompose_population_reference(bindings, diagram, scope, ctx)?;
        if numerical_digest(
            &result.decomposition,
            result.covariance.as_ref(),
            result.transport.identity(),
            ctx,
        )? != wire.digest
        {
            return Err(invalid("population-reference numeric replay mismatch"));
        }
        Ok(result)
    }
}
/// Historical population-reference replay without a source granting current permission.
#[derive(Clone, Debug)]
pub struct HistoricalPopulationReference {
    /// Replayed original per-slot full-procedure failure diagnostics.
    pub procedure_failures: Vec<Option<String>>,
    /// Complete reproduced numerical decomposition.
    pub decomposition: PanelDecomposition,
    /// Reproduced native joint covariance.
    pub covariance: Option<super::super::engine::decomposition::ReferenceDecompositionCovariance>,
    /// Archived theorem identity; this is not a current evidence check.
    pub transport_identity: SemanticDigest,
}
/// Independently replay every population-mixture premise and numerical operation.
/// # Errors
/// Invalid recipe/diagram, failed replay, resource or cancellation refusal.
pub fn replay_population_reference(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<HistoricalPopulationReference, JointPanelError> {
    replay_population_reference_with_limits(bytes, ctx, ReferenceArtifactLimits::default())
}
/// [`replay_population_reference`] under a caller-declared byte limit.
/// # Errors
/// As [`replay_population_reference`], with `limits.max_bytes` as the size bound.
pub fn replay_population_reference_with_limits(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: ReferenceArtifactLimits,
) -> Result<HistoricalPopulationReference, JointPanelError> {
    let (wire, _lease) = decode_population(bytes, ctx, limits)?;
    let native = crate::analysis::joint_panel::designs::replay_joint_design_execution(
        &wire.design,
        ctx,
        limits.design(),
    )?;
    let scope = if wire.identified {
        ScientificConsumptionScope::IdentifiedCausal
    } else {
        ScientificConsumptionScope::ModelConditional
    };
    let result = native.decompose_population_reference(
        &wire.bindings.to_domain(),
        &wire.diagram.to_domain()?,
        scope,
        ctx,
    )?;
    if numerical_digest(
        &result.decomposition,
        result.covariance.as_ref(),
        result.transport.identity(),
        ctx,
    )? != wire.digest
    {
        return Err(invalid("historical population-reference numeric replay mismatch"));
    }
    Ok(HistoricalPopulationReference {
        procedure_failures: result.procedure_failures,
        decomposition: result.decomposition,
        covariance: result.covariance,
        transport_identity: result.transport.identity(),
    })
}
