//! 2.3.0 A5 transported static path-specific counterfactual artifact
//! (`transported_counterfactual_v1`).
//!
//! A bounded, checksummed container with one CBOR section, `transported_counterfactual_meta`:
//! the version, feature marker and the point-only claim; the **request** (the additive-noise
//! structural equations with their covariate-dependent coefficients, the source covariate law,
//! the target covariate law, the selection diagram, the edge assignment of the contrast, the
//! declared premises and the supplied regime evidence); the **result** (the target contrast,
//! the source contrast, the per-unit contrast at every target support point and the derivation
//! with its checked and declared assumptions); and the identity digests (one per component and
//! one over the whole request, plus one over the result).
//!
//! A consumer trusts none of the stored numbers. It rebuilds the request, runs
//! [`evaluate_transported_path_specific`] again and refuses unless the target contrast, the
//! source contrast, every per-unit contrast and the derivation are bit-identical. The
//! recomputation also re-checks the declared premises, so an artifact whose premises were
//! relabelled false (or one whose selection diagram now points at a mediator or outcome
//! mechanism) is refused with the core's own `transported_counterfactual.*` refusal. The
//! identity digests are recomputed from the same inputs: a changed premise, law, coefficient,
//! selection or assignment is refused even when the artifact was resealed consistently,
//! provided the consumer passes the identity it retained.
//!
//! Artifacts hold successful results only. A refusal (including the two-model
//! non-recoverability witness of a selection on a mediator or outcome mechanism) is returned
//! by [`TransportedCounterfactualArtifact::seal`] and never stored.
//!
//! What this does **not** say: the claim is a point under the narrow class (covariate
//! selection, additive noise, shared noise laws, unit-level cross-world coupling); the
//! additive-noise, shared-noise-law, cross-world and source-fit premises are declared by the
//! caller, not checked; no interval is published; calibration is unmeasured. Unknown major
//! versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::io::Cursor;

use antecedent_counterfactual::transported_gate::{PopulationRole, RegimeFactorKey};
use antecedent_counterfactual::transported_path_specific::{
    AdditiveLinearScm, Affine, CovariateLaw, DeclaredAssumptions, EdgeAssignment, LinearMechanism,
    NonRecoverableWitness, PathSpecificQuery, SelectionDiagram, SelectionNode,
    TransportedPathSpecificInput, TransportedPathSpecificRefusal, TransportedPathSpecificResult,
    evaluate_transported_path_specific,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const TRANSPORTED_COUNTERFACTUAL_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const TRANSPORTED_COUNTERFACTUAL_ARTIFACT_FEATURE: &str = "transported_counterfactual_v1";
/// The only claim of this route.
pub const TRANSPORTED_COUNTERFACTUAL_INFERENCE_CLAIM: &str = "point_only";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_TRANSPORTED_COUNTERFACTUAL_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most covariates a model may declare.
pub const MAX_COVARIATES: usize = 64;
/// Most mechanism nodes a model may declare.
pub const MAX_MECHANISMS: usize = 256;
/// Most support points either covariate law may declare.
pub const MAX_SUPPORT_POINTS: usize = 65_536;
/// Most selection nodes, evidence factors or assignment children a request may declare.
pub const MAX_DECLARATIONS: usize = 256;

const ARTIFACT_KIND: &str = "transported_counterfactual_v1";
const META_SECTION: &str = "transported_counterfactual_meta";

/// The two-model non-recoverability witness of a selection on a sensitive mechanism.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonRecoverableWitnessWire {
    /// The selected mechanism node.
    pub selected_node: String,
    /// The parent whose coefficient differs between the two target models.
    pub perturbed_parent: String,
    /// `None` when the constant part differs, else the covariate whose slope differs.
    pub perturbed_slope_covariate: Option<String>,
    /// The added amount.
    pub perturbation: f64,
    /// The source model, shared by both candidate worlds.
    pub source_model: ModelWire,
    /// The first target model (equal to the source model).
    pub target_model_a: ModelWire,
    /// The second target model.
    pub target_model_b: ModelWire,
    /// The source contrast, identical under both.
    pub source_contrast: f64,
    /// The target contrast under the first model.
    pub target_contrast_a: f64,
    /// The target contrast under the second model.
    pub target_contrast_b: f64,
}

/// The refusal a consumer or producer returns, in the shared structured shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusalWire {
    /// Registered reason code.
    pub code: String,
    /// Refusing stage.
    pub stage: String,
    /// Namespaced detail, `family.slot`.
    pub detail: String,
    /// Offending node, premise, point or factor, when there is one.
    pub offending: Option<String>,
    /// What the caller can change to proceed, when known.
    pub remedy: Option<String>,
    /// The impossibility witness, for a selection on a sensitive mechanism.
    pub witness: Option<NonRecoverableWitnessWire>,
    /// Regime factors (`source:<regime>` / `target:<regime>`) absent from the evidence.
    pub missing_factors: Vec<String>,
}

/// Why a transported counterfactual artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TransportedCounterfactualArtifactError {
    /// The bytes do not decode as this format.
    #[error("transported counterfactual artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported transported counterfactual artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker or claim that is not this format's.
    #[error("unsupported transported counterfactual semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("transported counterfactual consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored declarations are not a well-formed request.
    #[error("malformed transported counterfactual artifact: {0}")]
    Malformed(String),
    /// A stored or retained identity field differs from the one the inputs produce.
    #[error("transported_counterfactual.artifact_changed: {field} changed")]
    IdentityMismatch {
        /// Which identity field differs.
        field: &'static str,
    },
    /// A stored contrast, per-unit contrast or derivation differs from the recomputed one.
    #[error("stored transported counterfactual does not replay: {0}")]
    ResultMismatch(&'static str),
    /// The artifact could not be encoded.
    #[error("transported counterfactual artifact does not encode: {0}")]
    Encode(String),
    /// The core evaluator refused.
    #[error("transported counterfactual refused: {}", .0.detail)]
    Refused(Box<RefusalWire>),
}

impl TransportedCounterfactualArtifactError {
    /// The structured refusal this error carries, for a core refusal and a changed identity.
    #[must_use]
    pub fn refusal(&self) -> Option<RefusalWire> {
        match self {
            Self::Refused(wire) => Some((**wire).clone()),
            Self::IdentityMismatch { field } => Some(RefusalWire {
                code: antecedent_core::reason_code!("route_not_supported").to_owned(),
                stage: "consume".to_owned(),
                detail: "transported_counterfactual.artifact_changed".to_owned(),
                offending: Some((*field).to_owned()),
                remedy: Some(
                    "consume the artifact the producer exported, or retain the identity of the \
                     inputs you intend"
                        .to_owned(),
                ),
                witness: None,
                missing_factors: Vec::new(),
            }),
            _ => None,
        }
    }
}

impl From<crate::IoError> for TransportedCounterfactualArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

type Failure = TransportedCounterfactualArtifactError;

fn invalid(detail: &str, offending: &str) -> Failure {
    Failure::Refused(Box::new(RefusalWire {
        code: antecedent_core::reason_code!("invalid_argument").to_owned(),
        stage: "identify".to_owned(),
        detail: detail.to_owned(),
        offending: Some(offending.to_owned()),
        remedy: None,
        witness: None,
        missing_factors: Vec::new(),
    }))
}

/// An affine function of the covariates on the wire.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AffineWire {
    /// Value at `z = 0`.
    pub constant: f64,
    /// Per-covariate slopes.
    pub covariate_slopes: Vec<(String, f64)>,
}

/// One mediator or outcome mechanism on the wire: `V = intercept(z) + sum coef_p(z) V_p + U`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismWire {
    /// The node it generates.
    pub node: String,
    /// The covariate-dependent intercept.
    pub intercept: AffineWire,
    /// Parents with their covariate-dependent coefficients.
    pub parents: Vec<(String, AffineWire)>,
}

/// The additive-noise structural model on the wire.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelWire {
    /// The treatment `A`.
    pub treatment: String,
    /// Covariate names.
    pub covariates: Vec<String>,
    /// Mediator and outcome mechanisms.
    pub mechanisms: Vec<MechanismWire>,
}

/// One finite-support point of a covariate law.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportPointWire {
    /// Covariate values by name.
    pub values: Vec<(String, f64)>,
    /// Positive probability weight.
    pub weight: f64,
}

/// A finite-support covariate law on the wire.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LawWire {
    /// Support points with weights summing to one.
    pub points: Vec<SupportPointWire>,
}

/// A selection node on the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionWire {
    /// The selection node's label.
    pub label: String,
    /// The variable it points to.
    pub target: String,
}

/// The edge assignment of the contrast on the wire: the added world feeds the treated value
/// along the `plus` children, the subtracted world along the `minus` children.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssignmentWire {
    /// The outcome node.
    pub outcome: String,
    /// The treated value `a1`.
    pub treated_value: f64,
    /// The control value `a0`.
    pub control_value: f64,
    /// Children of the treatment fed the treated value in the added world.
    pub plus: Vec<String>,
    /// Children of the treatment fed the treated value in the subtracted world.
    pub minus: Vec<String>,
}

/// The declared premises on the wire; each is a declaration, never a check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PremisesWire {
    /// Additive noise in the mediator and outcome mechanisms.
    pub additive_noise: bool,
    /// Same noise laws in both populations, independent of covariates.
    pub noise_laws_shared: bool,
    /// One exogenous draw per unit is fed to both worlds of the contrast.
    pub cross_world_independence: bool,
}

/// One supplied regime factor of the evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceWire {
    /// `source` or `target`.
    pub role: String,
    /// Regime name.
    pub regime: String,
    /// Evidence label.
    pub label: String,
}

/// A complete transported path-specific counterfactual request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportedCounterfactualRequestWire {
    /// The source structural fit.
    pub model: ModelWire,
    /// The source covariate law.
    pub source_law: LawWire,
    /// The target covariate law.
    pub target_law: LawWire,
    /// The selection diagram's selection nodes.
    pub selections: Vec<SelectionWire>,
    /// The contrast's edge assignment.
    pub assignment: AssignmentWire,
    /// The declared premises.
    pub premises: PremisesWire,
    /// The supplied source and target regime evidence.
    pub evidence: Vec<EvidenceWire>,
}

/// The target-law contribution of one support point on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitContrastWire {
    /// Covariate values by name.
    pub point: Vec<(String, f64)>,
    /// Its target weight.
    pub target_weight: f64,
    /// The unit-level contrast `G(z)`.
    pub contrast: f64,
}

/// What was checked and what was only declared.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivationWire {
    /// The theorem used.
    pub theorem: String,
    /// Premises checked from the supplied structure.
    pub checked: Vec<String>,
    /// Premises declared by the caller and not checkable here.
    pub declared: Vec<String>,
    /// `point_only`.
    pub claim: String,
}

/// The stored answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportedCounterfactualResultWire {
    /// `sum_z P_T(z) G(z)`.
    pub target_contrast: f64,
    /// `sum_z P_S(z) G(z)`, the answer if the source law were mistaken for the target's.
    pub source_contrast: f64,
    /// `G` at every target support point, in canonical point order.
    pub unit_contrasts: Vec<UnitContrastWire>,
    /// What was checked and declared.
    pub derivation: DerivationWire,
}

/// Identity digests of one transported counterfactual: one per component of the request, one
/// over the whole request and one over the result (BLAKE3, length-prefixed canonical bytes).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportedCounterfactualIdentity {
    /// Structural equations: treatment, covariates and every coefficient.
    pub model_digest: String,
    /// Source covariate law.
    pub source_law_digest: String,
    /// Target covariate law.
    pub target_law_digest: String,
    /// Selection diagram.
    pub selection_digest: String,
    /// Edge assignment, outcome and treatment values.
    pub assignment_digest: String,
    /// Declared premises.
    pub premises_digest: String,
    /// Supplied regime evidence.
    pub evidence_digest: String,
    /// The whole canonical request.
    pub spec_id: String,
    /// The stored result.
    pub result_digest: String,
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportedCounterfactualMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// The canonical request.
    pub request: TransportedCounterfactualRequestWire,
    /// The stored answer.
    pub result: TransportedCounterfactualResultWire,
    /// Identity digests.
    pub identity: TransportedCounterfactualIdentity,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportedCounterfactualReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// The canonical request.
    pub request: TransportedCounterfactualRequestWire,
    /// The answer.
    pub result: TransportedCounterfactualResultWire,
    /// Identity digests.
    pub identity: TransportedCounterfactualIdentity,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

// ---------------------------------------------------------------------------------------------
// Canonical form
// ---------------------------------------------------------------------------------------------

fn pair_order(a: &(String, f64), b: &(String, f64)) -> Ordering {
    a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1))
}

fn canonical_affine(affine: &mut AffineWire) {
    affine.covariate_slopes.sort_by(pair_order);
}

fn point_order(a: &SupportPointWire, b: &SupportPointWire) -> Ordering {
    for (x, y) in a.values.iter().zip(&b.values) {
        let order = pair_order(x, y);
        if order != Ordering::Equal {
            return order;
        }
    }
    a.values.len().cmp(&b.values.len()).then_with(|| a.weight.total_cmp(&b.weight))
}

fn canonical_law(law: &mut LawWire) {
    for point in &mut law.points {
        point.values.sort_by(pair_order);
    }
    law.points.sort_by(point_order);
}

/// The request in canonical order: covariates, mechanisms, parents, slopes, support points,
/// selections, assignment children and evidence all sorted, so the identity and the evaluated
/// result do not depend on the order anything was supplied in.
fn canonical(
    request: &TransportedCounterfactualRequestWire,
) -> TransportedCounterfactualRequestWire {
    let mut out = request.clone();
    out.model.covariates.sort();
    for mechanism in &mut out.model.mechanisms {
        canonical_affine(&mut mechanism.intercept);
        for (_, coefficient) in &mut mechanism.parents {
            canonical_affine(coefficient);
        }
        mechanism.parents.sort_by(|a, b| a.0.cmp(&b.0));
    }
    out.model.mechanisms.sort_by(|a, b| a.node.cmp(&b.node));
    canonical_law(&mut out.source_law);
    canonical_law(&mut out.target_law);
    out.selections.sort_by(|a, b| (&a.label, &a.target).cmp(&(&b.label, &b.target)));
    out.assignment.plus.sort();
    out.assignment.plus.dedup();
    out.assignment.minus.sort();
    out.assignment.minus.dedup();
    out.evidence
        .sort_by(|a, b| (&a.role, &a.regime, &a.label).cmp(&(&b.role, &b.regime, &b.label)));
    out
}

fn check_bounds(request: &TransportedCounterfactualRequestWire) -> Result<(), Failure> {
    if request.model.covariates.len() > MAX_COVARIATES {
        return Err(Failure::LimitsExceeded("covariates"));
    }
    if request.model.mechanisms.len() > MAX_MECHANISMS {
        return Err(Failure::LimitsExceeded("mechanisms"));
    }
    if request.source_law.points.len() > MAX_SUPPORT_POINTS
        || request.target_law.points.len() > MAX_SUPPORT_POINTS
    {
        return Err(Failure::LimitsExceeded("support points"));
    }
    if request.selections.len() > MAX_DECLARATIONS
        || request.evidence.len() > MAX_DECLARATIONS
        || request.assignment.plus.len() > MAX_DECLARATIONS
        || request.assignment.minus.len() > MAX_DECLARATIONS
    {
        return Err(Failure::LimitsExceeded("declarations"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Wire <-> core
// ---------------------------------------------------------------------------------------------

fn affine_of(wire: &AffineWire) -> Affine {
    Affine { constant: wire.constant, covariate_slopes: wire.covariate_slopes.clone() }
}

fn affine_wire(affine: &Affine) -> AffineWire {
    AffineWire { constant: affine.constant, covariate_slopes: affine.covariate_slopes.clone() }
}

fn model_of(wire: &ModelWire) -> AdditiveLinearScm {
    AdditiveLinearScm {
        treatment: wire.treatment.clone(),
        covariates: wire.covariates.clone(),
        mechanisms: wire
            .mechanisms
            .iter()
            .map(|m| {
                (
                    m.node.clone(),
                    LinearMechanism {
                        intercept: affine_of(&m.intercept),
                        parents: m.parents.iter().map(|(p, a)| (p.clone(), affine_of(a))).collect(),
                    },
                )
            })
            .collect(),
    }
}

fn model_wire(model: &AdditiveLinearScm) -> ModelWire {
    ModelWire {
        treatment: model.treatment.clone(),
        covariates: model.covariates.clone(),
        mechanisms: model
            .mechanisms
            .iter()
            .map(|(node, m)| MechanismWire {
                node: node.clone(),
                intercept: affine_wire(&m.intercept),
                parents: m.parents.iter().map(|(p, a)| (p.clone(), affine_wire(a))).collect(),
            })
            .collect(),
    }
}

fn law_of(wire: &LawWire, label: &str) -> Result<CovariateLaw, Failure> {
    let mut points = Vec::with_capacity(wire.points.len());
    for point in &wire.points {
        let mut values = BTreeMap::new();
        for (name, value) in &point.values {
            if values.insert(name.clone(), *value).is_some() {
                return Err(invalid(
                    "transported_counterfactual.invalid_law",
                    &format!("{label}: repeated covariate {name}"),
                ));
            }
        }
        points.push((values, point.weight));
    }
    Ok(CovariateLaw { points })
}

fn witness_wire(witness: &NonRecoverableWitness) -> NonRecoverableWitnessWire {
    NonRecoverableWitnessWire {
        selected_node: witness.selected_node.clone(),
        perturbed_parent: witness.perturbed_parent.clone(),
        perturbed_slope_covariate: witness.perturbed_slope_covariate.clone(),
        perturbation: witness.perturbation,
        source_model: model_wire(&witness.source_model),
        target_model_a: model_wire(&witness.target_model_a),
        target_model_b: model_wire(&witness.target_model_b),
        source_contrast: witness.source_contrast,
        target_contrast_a: witness.target_contrast_a,
        target_contrast_b: witness.target_contrast_b,
    }
}

fn from_core(refusal: &TransportedPathSpecificRefusal) -> Failure {
    Failure::Refused(Box::new(RefusalWire {
        code: refusal.refusal.code.to_owned(),
        stage: refusal.refusal.stage.to_owned(),
        detail: refusal.refusal.detail.clone(),
        offending: refusal.refusal.offending.clone(),
        remedy: refusal.refusal.remedy.map(str::to_owned),
        witness: refusal.witness.as_ref().map(witness_wire),
        missing_factors: refusal.missing_factors.iter().map(RegimeFactorKey::label).collect(),
    }))
}

fn role_of(tag: &str) -> Result<PopulationRole, Failure> {
    match tag {
        "source" => Ok(PopulationRole::Source),
        "target" => Ok(PopulationRole::Target),
        other => Err(invalid("transported_counterfactual.invalid_factor", other)),
    }
}

fn result_wire(result: &TransportedPathSpecificResult) -> TransportedCounterfactualResultWire {
    let owned =
        |items: &[&'static str]| -> Vec<String> { items.iter().map(|s| (*s).to_owned()).collect() };
    TransportedCounterfactualResultWire {
        target_contrast: result.target_contrast,
        source_contrast: result.source_contrast,
        unit_contrasts: result
            .unit_contrasts
            .iter()
            .map(|u| UnitContrastWire {
                point: u.point.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                target_weight: u.target_weight,
                contrast: u.contrast,
            })
            .collect(),
        derivation: DerivationWire {
            theorem: result.derivation.theorem.to_owned(),
            checked: owned(&result.derivation.checked),
            declared: owned(&result.derivation.declared),
            claim: result.derivation.claim.to_owned(),
        },
    }
}

/// Run the core evaluator on the (already canonical) request.
fn evaluate(
    request: &TransportedCounterfactualRequestWire,
) -> Result<TransportedCounterfactualResultWire, Failure> {
    check_bounds(request)?;
    let model = model_of(&request.model);
    let source = law_of(&request.source_law, "source")?;
    let target = law_of(&request.target_law, "target")?;
    let diagram = SelectionDiagram {
        selections: request
            .selections
            .iter()
            .map(|s| SelectionNode { label: s.label.clone(), target: s.target.clone() })
            .collect(),
    };
    let query = PathSpecificQuery {
        outcome: request.assignment.outcome.clone(),
        treated_value: request.assignment.treated_value,
        control_value: request.assignment.control_value,
        plus: EdgeAssignment { treated: request.assignment.plus.iter().cloned().collect() },
        minus: EdgeAssignment { treated: request.assignment.minus.iter().cloned().collect() },
    };
    let mut factors: BTreeMap<RegimeFactorKey, String> = BTreeMap::new();
    for item in &request.evidence {
        let key = RegimeFactorKey { role: role_of(&item.role)?, regime: item.regime.clone() };
        if factors.insert(key.clone(), item.label.clone()).is_some() {
            return Err(invalid(
                "transported_counterfactual.invalid_factor",
                &format!("repeated factor {}", key.label()),
            ));
        }
    }
    let input = TransportedPathSpecificInput {
        model: &model,
        diagram: &diagram,
        source_law: &source,
        target_law: &target,
        query: &query,
        assumptions: DeclaredAssumptions {
            additive_noise: request.premises.additive_noise,
            noise_laws_shared: request.premises.noise_laws_shared,
            cross_world_independence: request.premises.cross_world_independence,
        },
        supplied_factors: &factors,
    };
    evaluate_transported_path_specific(&input)
        .map(|result| result_wire(&result))
        .map_err(|refusal| from_core(&refusal))
}

// ---------------------------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------------------------

/// Length-prefixed canonical byte builder hashed with BLAKE3.
struct Canon(Vec<u8>);

impl Canon {
    fn new(tag: &str) -> Self {
        let mut canon = Self(Vec::new());
        canon.text(tag);
        canon
    }

    fn word(&mut self, word: u64) {
        self.0.extend_from_slice(&word.to_le_bytes());
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        self.0.extend_from_slice(text.as_bytes());
    }

    fn real(&mut self, value: f64) {
        self.word(value.to_bits());
    }

    fn flag(&mut self, value: bool) {
        self.word(u64::from(value));
    }

    fn affine(&mut self, affine: &AffineWire) {
        self.real(affine.constant);
        self.word(affine.covariate_slopes.len() as u64);
        for (name, slope) in &affine.covariate_slopes {
            self.text(name);
            self.real(*slope);
        }
    }

    fn model(&mut self, model: &ModelWire) {
        self.text(&model.treatment);
        self.word(model.covariates.len() as u64);
        for covariate in &model.covariates {
            self.text(covariate);
        }
        self.word(model.mechanisms.len() as u64);
        for mechanism in &model.mechanisms {
            self.text(&mechanism.node);
            self.affine(&mechanism.intercept);
            self.word(mechanism.parents.len() as u64);
            for (parent, coefficient) in &mechanism.parents {
                self.text(parent);
                self.affine(coefficient);
            }
        }
    }

    fn law(&mut self, law: &LawWire) {
        self.word(law.points.len() as u64);
        for point in &law.points {
            self.word(point.values.len() as u64);
            for (name, value) in &point.values {
                self.text(name);
                self.real(*value);
            }
            self.real(point.weight);
        }
    }

    fn selections(&mut self, selections: &[SelectionWire]) {
        self.word(selections.len() as u64);
        for selection in selections {
            self.text(&selection.label);
            self.text(&selection.target);
        }
    }

    fn assignment(&mut self, assignment: &AssignmentWire) {
        self.text(&assignment.outcome);
        self.real(assignment.treated_value);
        self.real(assignment.control_value);
        for children in [&assignment.plus, &assignment.minus] {
            self.word(children.len() as u64);
            for child in children {
                self.text(child);
            }
        }
    }

    fn premises(&mut self, premises: PremisesWire) {
        self.flag(premises.additive_noise);
        self.flag(premises.noise_laws_shared);
        self.flag(premises.cross_world_independence);
    }

    fn evidence(&mut self, evidence: &[EvidenceWire]) {
        self.word(evidence.len() as u64);
        for item in evidence {
            self.text(&item.role);
            self.text(&item.regime);
            self.text(&item.label);
        }
    }

    fn finish(&self) -> String {
        blake3::hash(&self.0).to_hex().to_string()
    }
}

fn strings(canon: &mut Canon, items: &[String]) {
    canon.word(items.len() as u64);
    for item in items {
        canon.text(item);
    }
}

fn result_digest(result: &TransportedCounterfactualResultWire) -> String {
    let mut c = Canon::new("transported_counterfactual_v1.result");
    c.real(result.target_contrast);
    c.real(result.source_contrast);
    c.word(result.unit_contrasts.len() as u64);
    for unit in &result.unit_contrasts {
        c.word(unit.point.len() as u64);
        for (name, value) in &unit.point {
            c.text(name);
            c.real(*value);
        }
        c.real(unit.target_weight);
        c.real(unit.contrast);
    }
    c.text(&result.derivation.theorem);
    strings(&mut c, &result.derivation.checked);
    strings(&mut c, &result.derivation.declared);
    c.text(&result.derivation.claim);
    c.finish()
}

fn identity_of(
    request: &TransportedCounterfactualRequestWire,
    result: &TransportedCounterfactualResultWire,
) -> TransportedCounterfactualIdentity {
    let mut model = Canon::new("transported_counterfactual_v1.model");
    model.model(&request.model);
    let mut source = Canon::new("transported_counterfactual_v1.source_law");
    source.law(&request.source_law);
    let mut target = Canon::new("transported_counterfactual_v1.target_law");
    target.law(&request.target_law);
    let mut selection = Canon::new("transported_counterfactual_v1.selection");
    selection.selections(&request.selections);
    let mut assignment = Canon::new("transported_counterfactual_v1.assignment");
    assignment.assignment(&request.assignment);
    let mut premises = Canon::new("transported_counterfactual_v1.premises");
    premises.premises(request.premises);
    let mut evidence = Canon::new("transported_counterfactual_v1.evidence");
    evidence.evidence(&request.evidence);
    let mut spec = Canon::new("transported_counterfactual_v1.spec");
    spec.model(&request.model);
    spec.law(&request.source_law);
    spec.law(&request.target_law);
    spec.selections(&request.selections);
    spec.assignment(&request.assignment);
    spec.premises(request.premises);
    spec.evidence(&request.evidence);
    TransportedCounterfactualIdentity {
        model_digest: model.finish(),
        source_law_digest: source.finish(),
        target_law_digest: target.finish(),
        selection_digest: selection.finish(),
        assignment_digest: assignment.finish(),
        premises_digest: premises.finish(),
        evidence_digest: evidence.finish(),
        spec_id: spec.finish(),
        result_digest: result_digest(result),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &TransportedCounterfactualIdentity,
    other: &TransportedCounterfactualIdentity,
) -> Option<&'static str> {
    if stored.premises_digest != other.premises_digest {
        Some("premises")
    } else if stored.model_digest != other.model_digest {
        Some("model")
    } else if stored.source_law_digest != other.source_law_digest {
        Some("source_law")
    } else if stored.target_law_digest != other.target_law_digest {
        Some("target_law")
    } else if stored.selection_digest != other.selection_digest {
        Some("selection")
    } else if stored.assignment_digest != other.assignment_digest {
        Some("assignment")
    } else if stored.evidence_digest != other.evidence_digest {
        Some("evidence")
    } else if stored.spec_id != other.spec_id {
        Some("request")
    } else if stored.result_digest != other.result_digest {
        Some("result")
    } else {
        None
    }
}

fn same_real(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// The first stored result field that does not replay bit for bit.
fn result_diff(
    stored: &TransportedCounterfactualResultWire,
    replayed: &TransportedCounterfactualResultWire,
) -> Option<&'static str> {
    if !same_real(stored.target_contrast, replayed.target_contrast) {
        return Some("target contrast");
    }
    if !same_real(stored.source_contrast, replayed.source_contrast) {
        return Some("source contrast");
    }
    let units_replay = stored.unit_contrasts.len() == replayed.unit_contrasts.len()
        && stored.unit_contrasts.iter().zip(&replayed.unit_contrasts).all(|(a, b)| {
            a.point.len() == b.point.len()
                && a.point.iter().zip(&b.point).all(|(x, y)| x.0 == y.0 && same_real(x.1, y.1))
                && same_real(a.target_weight, b.target_weight)
                && same_real(a.contrast, b.contrast)
        });
    if !units_replay {
        return Some("per-unit contrasts");
    }
    if stored.derivation != replayed.derivation {
        return Some("derivation");
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------------------------

/// Encode a metadata section as a checksummed container.
///
/// Hidden: the producer path is [`TransportedCounterfactualArtifact::to_bytes`]; tests use this
/// to build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &TransportedCounterfactualMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, TransportedCounterfactualArtifactError> {
    let encode = |e: crate::IoError| Failure::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(Failure::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(encode)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .map_err(encode)?,
            artifact_id: artifact_id.into(),
            sections: vec![section_descriptor(META_SECTION, "application/cbor", &meta_bytes)],
            provenance: ProvenanceWire {
                note: "transported_counterfactual_covariate_selected_point_only".into(),
            },
        },
        sections: vec![SectionBytes::new(META_SECTION, meta_bytes)],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_TRANSPORTED_COUNTERFACTUAL_ARTIFACT_BYTES {
        return Err(Failure::LimitsExceeded("artifact bytes"));
    }
    Ok(bytes)
}

/// Decode the metadata section of a container, refusing another version before the metadata
/// is interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<TransportedCounterfactualMeta, TransportedCounterfactualArtifactError> {
    if bytes.len() > MAX_TRANSPORTED_COUNTERFACTUAL_ARTIFACT_BYTES {
        return Err(Failure::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 1
        || manifest.sections[0].id != META_SECTION
    {
        return Err(Failure::Malformed("unsupported container layout".into()));
    }
    if manifest.sections[0].uncompressed_size > MAX_TRANSPORTED_COUNTERFACTUAL_ARTIFACT_BYTES as u64
    {
        return Err(Failure::LimitsExceeded("artifact bytes"));
    }
    let section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(section.as_bytes())?;
    if peek.version != TRANSPORTED_COUNTERFACTUAL_ARTIFACT_VERSION {
        return Err(Failure::UnsupportedVersion { version: peek.version });
    }
    from_cbor(section.as_bytes()).map_err(Failure::from)
}

/// A produced or consumed transported counterfactual artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedCounterfactualArtifact {
    meta: TransportedCounterfactualMeta,
}

impl TransportedCounterfactualArtifact {
    /// Evaluate `request` with the core evaluator and seal the answer with its identity.
    ///
    /// The request is canonicalized (every list sorted) before it is evaluated and digested.
    ///
    /// # Errors
    /// The core evaluator's refusals, with the two-model witness retained on a selection on a
    /// sensitive mechanism (`transported_counterfactual.selection_on_mechanism`), the
    /// undeclared-premise refusals (`.nonadditive_mechanism`, `.noise_law_not_shared`,
    /// `.cross_world_independence_missing`), `.factor_missing`, `.overlap_failure` and the
    /// `invalid_argument` family (`.invalid_model`, `.invalid_query`, `.invalid_law`,
    /// `.invalid_diagram`, `.invalid_factor`); or the format's bounds.
    pub fn seal(
        request: &TransportedCounterfactualRequestWire,
    ) -> Result<Self, TransportedCounterfactualArtifactError> {
        let request = canonical(request);
        let result = evaluate(&request)?;
        let identity = identity_of(&request, &result);
        Ok(Self {
            meta: TransportedCounterfactualMeta {
                version: TRANSPORTED_COUNTERFACTUAL_ARTIFACT_VERSION,
                feature: TRANSPORTED_COUNTERFACTUAL_ARTIFACT_FEATURE.to_owned(),
                inference_claim: TRANSPORTED_COUNTERFACTUAL_INFERENCE_CLAIM.to_owned(),
                request,
                result,
                identity,
            },
        })
    }

    /// The metadata section.
    #[must_use]
    pub const fn meta(&self) -> &TransportedCounterfactualMeta {
        &self.meta
    }

    /// The canonical request.
    #[must_use]
    pub const fn request(&self) -> &TransportedCounterfactualRequestWire {
        &self.meta.request
    }

    /// The stored (and replayed) answer.
    #[must_use]
    pub const fn result(&self) -> &TransportedCounterfactualResultWire {
        &self.meta.result
    }

    /// The identity digests.
    #[must_use]
    pub const fn identity(&self) -> &TransportedCounterfactualIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> TransportedCounterfactualReportWire {
        TransportedCounterfactualReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            request: self.meta.request.clone(),
            result: self.meta.result.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, TransportedCounterfactualArtifactError> {
        encode_parts(&self.meta, artifact_id)
    }

    /// Consume an artifact by recomputing the transported contrast.
    ///
    /// `expected` is an identity the consumer retained independently; when given, every field
    /// must match it, so a resealed change of any premise, law, coefficient, selection or
    /// assignment is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a core refusal (including an
    /// artifact whose declared premises are false-labelled), a changed identity, or a stored
    /// contrast, per-unit contrast or derivation that does not replay.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&TransportedCounterfactualIdentity>,
    ) -> Result<Self, TransportedCounterfactualArtifactError> {
        let meta = decode_parts(bytes)?;
        if meta.feature != TRANSPORTED_COUNTERFACTUAL_ARTIFACT_FEATURE {
            return Err(Failure::UnsupportedSemantics("feature marker"));
        }
        if meta.inference_claim != TRANSPORTED_COUNTERFACTUAL_INFERENCE_CLAIM {
            return Err(Failure::UnsupportedSemantics("inference claim"));
        }
        let request = canonical(&meta.request);
        let replayed = evaluate(&request)?;
        let recomputed = identity_of(&request, &replayed);
        if let Some(field) = identity_diff(&meta.identity, &recomputed) {
            // Only the result digest differing means the request is as stored and the stored
            // answer is not the replayed one.
            if field == "result" {
                return Err(Failure::ResultMismatch(
                    result_diff(&meta.result, &replayed).unwrap_or("result digest"),
                ));
            }
            return Err(Failure::IdentityMismatch { field });
        }
        if let Some(what) = result_diff(&meta.result, &replayed) {
            return Err(Failure::ResultMismatch(what));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &recomputed) {
                return Err(Failure::IdentityMismatch { field });
            }
        }
        let verified = TransportedCounterfactualMeta { request, result: replayed, ..meta };
        Ok(Self { meta: verified })
    }
}
