//! Cross-world execution artifact (`cross_world_execution_v1`).
//!
//! A bounded, checksummed, sectioned container holding one finite Markovian
//! cross-world answer and everything a licensed resume needs:
//!
//! * `cross_world_execution_meta` (CBOR): version, feature marker, the point-only
//!   claim, the matrix row, the **declarations** (mechanisms, factual
//!   conditioning, world definitions with routes, quantity, declared coupling,
//!   population and history), the shared exogenous states per unit (state
//!   indices and their stream-coordinate digests) and the identity digests;
//! * `cross_world_execution_values` (little-endian `f64`): the mean, then for
//!   each unit in id order its value, `P(evidence)` and the posterior weights.
//!
//! A consumer trusts none of the stored numbers. It rebuilds the request,
//! abduces every unit again with a fresh executor, evaluates every world under
//! the same exogenous states, and refuses unless the shared states, weights,
//! per-unit values, mean and every identity digest are bit-identical. The
//! verified executor is returned with the consumed artifact, so a fresh process
//! can continue with an action-only change reusing the verified abduction.
//!
//! What this does **not** say: the claim is a point from exact enumeration of a
//! declared model; no interval or posterior is published, mechanism adequacy is
//! a premise, and a transported or latent-confounded request is refused.
//! Unknown major versions refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::finite_scm::{
    MAX_JOINT_STATES, MAX_UNITS, MAX_VARIABLES,
};
use antecedent_counterfactual::selective::{
    CrossWorldExecutionError, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteExecutionResult, FiniteQuantity, FiniteScm, FiniteScmExecutor,
    FiniteVariable, FiniteWorld, NoiseCoupling, WorldAtom, WorldRoute,
};
use serde::{Deserialize, Serialize};

/// The artifact major version this reader writes and accepts.
pub const CROSS_WORLD_EXECUTION_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const CROSS_WORLD_EXECUTION_ARTIFACT_FEATURE: &str = "cross_world_execution_v1";
/// The only claim of this route.
pub const CROSS_WORLD_EXECUTION_INFERENCE_CLAIM: &str = "point_only";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_CROSS_WORLD_EXECUTION_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

const ARTIFACT_KIND: &str = crate::artifact_kinds::CROSS_WORLD_EXECUTION;
const META_SECTION: &str = "cross_world_execution_meta";
const VALUES_SECTION: &str = "cross_world_execution_values";

/// The refusal a consumer or producer returns, in the shared structured shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldExecutionRefusalWire {
    /// Registered reason code.
    pub code: String,
    /// Refusing stage.
    pub stage: String,
    /// Namespaced detail, `family.slot`.
    pub detail: String,
    /// Offending subject, when there is one.
    pub offending: Option<String>,
    /// What would license the request, when known.
    pub remedy: Option<String>,
    /// Named unmet prerequisites.
    pub missing: Vec<String>,
}

crate::sectioned_artifact_enum! {
    /// Why a cross-world execution artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    CrossWorldExecutionArtifactError {
        noun: "cross-world execution",
        identity: "cross_world_execution.artifact_changed",
        variants {
            /// A stored value, state set or digest differs from the recomputed one.
            ResultMismatch(&'static str),
            /// Cooperative cancellation was observed.
            Cancelled,
            /// The executor refused.
            Refused(Box<CrossWorldExecutionRefusalWire>),
        }
        display(f) {
            Self::ResultMismatch(what) => {
                write!(f, "stored cross-world execution does not replay: {what}")
            }
            Self::Cancelled => f.write_str("cancelled during cross-world execution"),
            Self::Refused(wire) => write!(f, "cross-world execution refused: {}", wire.detail),
        }
    }
}

impl CrossWorldExecutionArtifactError {
    /// The structured refusal this error carries.
    #[must_use]
    pub fn refusal(&self) -> Option<CrossWorldExecutionRefusalWire> {
        match self {
            Self::Refused(wire) => Some((**wire).clone()),
            Self::IdentityMismatch { field } => Some(CrossWorldExecutionRefusalWire {
                code: antecedent_core::reason_code!("route_not_supported").to_owned(),
                stage: "consume".to_owned(),
                detail: "cross_world_execution.artifact_changed".to_owned(),
                offending: Some((*field).to_owned()),
                remedy: Some(
                    "consume the artifact the producer exported, or retain the identity of the \
                     declarations you intend"
                        .to_owned(),
                ),
                missing: Vec::new(),
            }),
            _ => None,
        }
    }
}

fn from_core(error: CrossWorldExecutionError) -> CrossWorldExecutionArtifactError {
    match error {
        CrossWorldExecutionError::Cancelled => CrossWorldExecutionArtifactError::Cancelled,
        CrossWorldExecutionError::Refused(refused) => {
            CrossWorldExecutionArtifactError::Refused(Box::new(CrossWorldExecutionRefusalWire {
                code: refused.refusal.code.to_owned(),
                stage: refused.refusal.stage.to_owned(),
                detail: refused.refusal.detail.clone(),
                offending: refused.refusal.offending.clone(),
                remedy: refused.refusal.remedy.map(str::to_owned),
                missing: refused.missing.clone(),
            }))
        }
        other => CrossWorldExecutionArtifactError::Malformed(other.to_string()),
    }
}

/// An exogenous term on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExogenousWire {
    /// Name.
    pub name: String,
    /// State probabilities.
    pub pmf: Vec<f64>,
}

/// A finite variable and its mechanism table on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariableWire {
    /// Name.
    pub name: String,
    /// Number of levels.
    pub levels: u32,
    /// Parent indices.
    pub parents: Vec<u32>,
    /// Exogenous term index.
    pub exogenous: u32,
    /// Mechanism table.
    pub table: Vec<u8>,
}

/// The declared SCM on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScmWire {
    /// Population the mechanisms are declared for.
    pub population: String,
    /// History the mechanisms are declared for.
    pub history: String,
    /// Exogenous terms.
    pub exogenous: Vec<ExogenousWire>,
    /// Variables in topological order.
    pub variables: Vec<VariableWire>,
}

/// One unit's factual conditioning on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitWire {
    /// Unit id.
    pub unit: String,
    /// Observed `(variable, level)` pairs.
    pub observed: Vec<(u32, u8)>,
}

/// A cross-world route on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteWire {
    /// Parent variable.
    pub parent: u32,
    /// Child variable.
    pub child: u32,
    /// Source world.
    pub source: u32,
}

/// A world definition on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldWire {
    /// Name.
    pub name: String,
    /// `do(variable = level)` pairs.
    pub interventions: Vec<(u32, u8)>,
    /// Routes.
    pub routes: Vec<RouteWire>,
}

/// One atom of a quantity on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtomWire {
    /// World index.
    pub world: u32,
    /// Variable index.
    pub variable: u32,
    /// Level.
    pub level: u8,
}

/// The quantity on the wire: `kind` is `probability`, `mean_contrast` or
/// `joint_probability`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantityWire {
    /// Kind.
    pub kind: String,
    /// Atoms (`probability`: one; `joint_probability`: the event).
    pub atoms: Vec<AtomWire>,
    /// `(world, variable)` added by a `mean_contrast`.
    pub plus: Option<(u32, u32)>,
    /// `(world, variable)` subtracted by a `mean_contrast`.
    pub minus: Option<(u32, u32)>,
}

/// The request on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldRequestWire {
    /// Mechanisms.
    pub scm: ScmWire,
    /// History the factual evidence was gathered under.
    pub evidence_history: String,
    /// Factual conditioning per unit.
    pub units: Vec<UnitWire>,
    /// World definitions in index order.
    pub worlds: Vec<WorldWire>,
    /// The quantity.
    pub quantity: QuantityWire,
    /// Declared coupling name.
    pub coupling: String,
    /// Exogenous inference name.
    pub exogenous: String,
    /// Target population, when declared.
    pub target_population: Option<String>,
    /// Reading name.
    pub reading: String,
}

/// One unit's shared exogenous states on the wire (weights are in the values
/// section).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedDrawsWire {
    /// Unit id.
    pub unit: String,
    /// Abduction identity.
    pub identity: String,
    /// Digest of the unit's evidence.
    pub evidence_digest: String,
    /// Posterior joint-state indices.
    pub states: Vec<u32>,
    /// Digest of the posterior content with its semantic stream identities.
    pub draws_digest: String,
}

/// Identity digests of one execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldExecutionIdentity {
    /// Mechanism digest.
    pub mechanism_digest: String,
    /// Abduction-set digest.
    pub abduction_digest: String,
    /// Shared-exogenous digest.
    pub exogenous_digest: String,
    /// World-definition digest.
    pub world_digest: String,
    /// Action-prediction digest.
    pub prediction_digest: String,
    /// Summary digest.
    pub summary_digest: String,
    /// Overall receipt digest.
    pub receipt_digest: String,
    /// Declared coupling.
    pub coupling: String,
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossWorldExecutionMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// The matrix row that licensed the request.
    pub matrix_row: String,
    /// The declarations.
    pub request: CrossWorldRequestWire,
    /// Shared exogenous states per unit (unit-id order).
    pub shared_draws: Vec<SharedDrawsWire>,
    /// Identity digests.
    pub identity: CrossWorldExecutionIdentity,
}

fn u32_of(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn usize_of(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn coupling_name(c: NoiseCoupling) -> &'static str {
    c.name()
}

fn coupling_of(name: &str) -> Result<NoiseCoupling, CrossWorldExecutionArtifactError> {
    [
        NoiseCoupling::SharedAbducedExogenous,
        NoiseCoupling::IndependentWorldNoise,
        NoiseCoupling::Undeclared,
    ]
    .into_iter()
    .find(|c| c.name() == name)
    .ok_or(CrossWorldExecutionArtifactError::UnsupportedSemantics("coupling"))
}

fn exogenous_name(e: ExogenousInference) -> &'static str {
    match e {
        ExogenousInference::ExactEnumeration => "exact_enumeration",
        ExogenousInference::PosteriorSampling => "posterior_sampling",
    }
}

fn exogenous_of(name: &str) -> Result<ExogenousInference, CrossWorldExecutionArtifactError> {
    match name {
        "exact_enumeration" => Ok(ExogenousInference::ExactEnumeration),
        "posterior_sampling" => Ok(ExogenousInference::PosteriorSampling),
        _ => Err(CrossWorldExecutionArtifactError::UnsupportedSemantics("exogenous inference")),
    }
}

fn atom_wire(a: &WorldAtom) -> AtomWire {
    AtomWire { world: u32_of(a.world), variable: u32_of(a.variable), level: a.level }
}

fn atom_of(a: &AtomWire) -> WorldAtom {
    WorldAtom { world: usize_of(a.world), variable: usize_of(a.variable), level: a.level }
}

fn quantity_wire(q: &FiniteQuantity) -> QuantityWire {
    match q {
        FiniteQuantity::Probability(a) => QuantityWire {
            kind: "probability".to_owned(),
            atoms: vec![atom_wire(a)],
            plus: None,
            minus: None,
        },
        FiniteQuantity::MeanContrast { plus, minus } => QuantityWire {
            kind: "mean_contrast".to_owned(),
            atoms: Vec::new(),
            plus: Some((u32_of(plus.0), u32_of(plus.1))),
            minus: Some((u32_of(minus.0), u32_of(minus.1))),
        },
        FiniteQuantity::JointProbability(atoms) => QuantityWire {
            kind: "joint_probability".to_owned(),
            atoms: atoms.iter().map(atom_wire).collect(),
            plus: None,
            minus: None,
        },
    }
}

fn quantity_of(q: &QuantityWire) -> Result<FiniteQuantity, CrossWorldExecutionArtifactError> {
    let bad = || CrossWorldExecutionArtifactError::Malformed("quantity".to_owned());
    match q.kind.as_str() {
        "probability" => match q.atoms.as_slice() {
            [a] => Ok(FiniteQuantity::Probability(atom_of(a))),
            _ => Err(bad()),
        },
        "mean_contrast" => match (q.plus, q.minus) {
            (Some(p), Some(m)) if q.atoms.is_empty() => Ok(FiniteQuantity::MeanContrast {
                plus: (usize_of(p.0), usize_of(p.1)),
                minus: (usize_of(m.0), usize_of(m.1)),
            }),
            _ => Err(bad()),
        },
        "joint_probability" => {
            Ok(FiniteQuantity::JointProbability(q.atoms.iter().map(atom_of).collect()))
        }
        _ => Err(bad()),
    }
}

/// The declarations of a request as their wire form.
#[must_use]
pub fn request_wire(request: &FiniteExecutionRequest) -> CrossWorldRequestWire {
    let mut units: Vec<&FactualUnit> = request.units.iter().collect();
    units.sort_by(|a, b| a.unit.cmp(&b.unit));
    CrossWorldRequestWire {
        scm: ScmWire {
            population: request.scm.population.clone(),
            history: request.scm.history.clone(),
            exogenous: request
                .scm
                .exogenous
                .iter()
                .map(|e| ExogenousWire { name: e.name.clone(), pmf: e.pmf.clone() })
                .collect(),
            variables: request
                .scm
                .variables
                .iter()
                .map(|v| VariableWire {
                    name: v.name.clone(),
                    levels: u32_of(v.levels),
                    parents: v.parents.iter().map(|p| u32_of(*p)).collect(),
                    exogenous: u32_of(v.exogenous),
                    table: v.table.clone(),
                })
                .collect(),
        },
        evidence_history: request.evidence_history.clone(),
        units: units
            .iter()
            .map(|u| {
                let mut observed: Vec<(u32, u8)> =
                    u.observed.iter().map(|(v, l)| (u32_of(*v), *l)).collect();
                observed.sort_unstable();
                UnitWire { unit: u.unit.clone(), observed }
            })
            .collect(),
        worlds: request
            .worlds
            .iter()
            .map(|w| WorldWire {
                name: w.name.clone(),
                interventions: w.interventions.iter().map(|(v, l)| (u32_of(*v), *l)).collect(),
                routes: w
                    .routes
                    .iter()
                    .map(|r| RouteWire {
                        parent: u32_of(r.parent),
                        child: u32_of(r.child),
                        source: u32_of(r.source),
                    })
                    .collect(),
            })
            .collect(),
        quantity: quantity_wire(&request.quantity),
        coupling: coupling_name(request.coupling).to_owned(),
        exogenous: exogenous_name(request.exogenous).to_owned(),
        target_population: request.target_population.clone(),
        reading: request.reading.as_str().to_owned(),
    }
}

/// Rebuild a request from its wire form, refusing unknown names.
///
/// # Errors
/// An unknown coupling, inference or reading name, a malformed quantity or a bound.
pub fn request_of(
    wire: &CrossWorldRequestWire,
) -> Result<FiniteExecutionRequest, CrossWorldExecutionArtifactError> {
    if wire.units.len() > MAX_UNITS {
        return Err(CrossWorldExecutionArtifactError::LimitsExceeded("units"));
    }
    if wire.scm.variables.len() > MAX_VARIABLES || wire.scm.exogenous.len() > MAX_VARIABLES {
        return Err(CrossWorldExecutionArtifactError::LimitsExceeded("variables"));
    }
    let reading = RequestedReading::from_name(&wire.reading)
        .ok_or(CrossWorldExecutionArtifactError::UnsupportedSemantics("reading"))?;
    Ok(FiniteExecutionRequest {
        scm: FiniteScm {
            population: wire.scm.population.clone(),
            history: wire.scm.history.clone(),
            exogenous: wire
                .scm
                .exogenous
                .iter()
                .map(|e| ExogenousVariable { name: e.name.clone(), pmf: e.pmf.clone() })
                .collect(),
            variables: wire
                .scm
                .variables
                .iter()
                .map(|v| FiniteVariable {
                    name: v.name.clone(),
                    levels: usize_of(v.levels),
                    parents: v.parents.iter().map(|p| usize_of(*p)).collect(),
                    exogenous: usize_of(v.exogenous),
                    table: v.table.clone(),
                })
                .collect(),
        },
        evidence_history: wire.evidence_history.clone(),
        units: wire
            .units
            .iter()
            .map(|u| FactualUnit {
                unit: u.unit.clone(),
                observed: u.observed.iter().map(|(v, l)| (usize_of(*v), *l)).collect(),
            })
            .collect(),
        worlds: wire
            .worlds
            .iter()
            .map(|w| FiniteWorld {
                name: w.name.clone(),
                interventions: w.interventions.iter().map(|(v, l)| (usize_of(*v), *l)).collect(),
                routes: w
                    .routes
                    .iter()
                    .map(|r| WorldRoute {
                        parent: usize_of(r.parent),
                        child: usize_of(r.child),
                        source: usize_of(r.source),
                    })
                    .collect(),
            })
            .collect(),
        quantity: quantity_of(&wire.quantity)?,
        coupling: coupling_of(&wire.coupling)?,
        exogenous: exogenous_of(&wire.exogenous)?,
        target_population: wire.target_population.clone(),
        reading,
    })
}

fn identity_of(result: &FiniteExecutionResult) -> CrossWorldExecutionIdentity {
    let r = &result.receipt;
    CrossWorldExecutionIdentity {
        mechanism_digest: r.mechanism_digest.clone(),
        abduction_digest: r.abduction_digest.clone(),
        exogenous_digest: r.exogenous_digest.clone(),
        world_digest: r.world_digest.clone(),
        prediction_digest: r.prediction_digest.clone(),
        summary_digest: r.summary_digest.clone(),
        receipt_digest: r.digest.clone(),
        coupling: r.coupling.clone(),
    }
}

fn identity_diff(
    a: &CrossWorldExecutionIdentity,
    b: &CrossWorldExecutionIdentity,
) -> Option<&'static str> {
    let fields: [(&'static str, bool); 8] = [
        ("mechanism_digest", a.mechanism_digest == b.mechanism_digest),
        ("abduction_digest", a.abduction_digest == b.abduction_digest),
        ("exogenous_digest", a.exogenous_digest == b.exogenous_digest),
        ("world_digest", a.world_digest == b.world_digest),
        ("prediction_digest", a.prediction_digest == b.prediction_digest),
        ("summary_digest", a.summary_digest == b.summary_digest),
        ("receipt_digest", a.receipt_digest == b.receipt_digest),
        ("coupling", a.coupling == b.coupling),
    ];
    fields.iter().find(|(_, same)| !same).map(|(name, _)| *name)
}

/// One execution's retained abductions and the numbers, as sealed.
fn collect(
    executor: &FiniteScmExecutor,
    result: &FiniteExecutionResult,
) -> (Vec<SharedDrawsWire>, Vec<f64>) {
    let mut values = vec![result.value];
    let mut draws = Vec::with_capacity(result.units.len());
    for answer in &result.units {
        values.push(answer.value);
        values.push(answer.evidence_probability);
        let retained = executor.retained_all().find(|r| {
            r.unit == answer.unit && r.mechanism_digest == result.receipt.mechanism_digest
        });
        if let Some(r) = retained {
            values.extend(r.posterior.states.iter().map(|(_, w)| *w));
            draws.push(SharedDrawsWire {
                unit: r.unit.clone(),
                identity: r.identity.clone(),
                evidence_digest: r.evidence_digest.clone(),
                states: r.posterior.states.iter().map(|(s, _)| *s).collect(),
                draws_digest: r.draws_digest.clone(),
            });
        }
    }
    (draws, values)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn encode_parts_impl(
    meta: &CrossWorldExecutionMeta,
    values: &[f64],
    artifact_id: &str,
) -> Result<Vec<u8>, CrossWorldExecutionArtifactError> {
    let values = crate::artifact_codec::f64s_to_le_bytes(values);
    crate::artifact_codec::encode_sectioned(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "cross_world_execution_exact_enumeration_point_only",
            max_bytes: MAX_CROSS_WORLD_EXECUTION_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[crate::OutSection {
            id: VALUES_SECTION,
            content_type: crate::CONTENT_BYTES,
            bytes: &values,
        }],
    )
}

/// Encode a metadata section and a values section as a checksummed container.
///
/// Hidden: the producer path is [`CrossWorldExecutionArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &CrossWorldExecutionMeta,
    values: &[f64],
    artifact_id: &str,
) -> Result<Vec<u8>, CrossWorldExecutionArtifactError> {
    encode_parts_impl(meta, values, artifact_id)
}

/// Decode the two sections of a container.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<(CrossWorldExecutionMeta, Vec<f64>), CrossWorldExecutionArtifactError> {
    CrossWorldExecutionArtifact::decode(bytes, MAX_CROSS_WORLD_EXECUTION_ARTIFACT_BYTES)
}

/// A produced or consumed cross-world execution artifact.
#[derive(Debug)]
pub struct CrossWorldExecutionArtifact {
    meta: CrossWorldExecutionMeta,
    values: Vec<f64>,
    result: FiniteExecutionResult,
    request: FiniteExecutionRequest,
    executor: FiniteScmExecutor,
}

impl CrossWorldExecutionArtifact {
    /// Execute `request` with a fresh executor and seal the answer, the
    /// declarations and the shared exogenous states.
    ///
    /// # Errors
    /// The executor's refusals (matrix, coupling, readings, population,
    /// history, refuted evidence), cancellation and the format's bounds.
    pub fn seal(
        request: &FiniteExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CrossWorldExecutionArtifactError> {
        let mut executor = FiniteScmExecutor::new();
        let result = executor.execute(request, ctx).map_err(from_core)?;
        let (shared_draws, values) = collect(&executor, &result);
        let meta = CrossWorldExecutionMeta {
            version: CROSS_WORLD_EXECUTION_ARTIFACT_VERSION,
            feature: CROSS_WORLD_EXECUTION_ARTIFACT_FEATURE.to_owned(),
            inference_claim: CROSS_WORLD_EXECUTION_INFERENCE_CLAIM.to_owned(),
            matrix_row: result.matrix_row.to_owned(),
            request: request_wire(request),
            shared_draws,
            identity: identity_of(&result),
        };
        Ok(Self { meta, values, result, request: request.clone(), executor })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &CrossWorldExecutionMeta {
        &self.meta
    }

    /// The executor's result.
    #[must_use]
    pub fn result(&self) -> &FiniteExecutionResult {
        &self.result
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &CrossWorldExecutionIdentity {
        &self.meta.identity
    }

    /// The request the artifact declares.
    #[must_use]
    pub fn request(&self) -> &FiniteExecutionRequest {
        &self.request
    }

    /// Take the executor holding the verified abductions.
    #[must_use]
    pub fn into_executor(self) -> FiniteScmExecutor {
        self.executor
    }

    /// The executor holding the verified abductions: an action-only change
    /// executed through it reuses them (the licensed resume).
    pub fn executor_mut(&mut self) -> &mut FiniteScmExecutor {
        &mut self.executor
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, CrossWorldExecutionArtifactError> {
        encode_parts_impl(&self.meta, &self.values, artifact_id)
    }

    fn decode(
        bytes: &[u8],
        max_bytes: usize,
    ) -> Result<(CrossWorldExecutionMeta, Vec<f64>), CrossWorldExecutionArtifactError> {
        let (meta, parts) = crate::artifact_codec::decode_sectioned::<
            CrossWorldExecutionArtifactError,
            CrossWorldExecutionMeta,
        >(
            ARTIFACT_KIND,
            max_bytes,
            META_SECTION,
            &[VALUES_SECTION],
            CROSS_WORLD_EXECUTION_ARTIFACT_VERSION,
            bytes,
        )?;
        let Ok([payload]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(CrossWorldExecutionArtifactError::Malformed(
                "unsupported container layout".into(),
            ));
        };
        if payload.len() % 8 != 0 {
            return Err(CrossWorldExecutionArtifactError::Malformed(
                "values section length".into(),
            ));
        }
        Ok((meta, crate::artifact_codec::f64s_from_le_bytes(&payload)))
    }

    /// Consume an artifact by recomputing it with a fresh executor.
    ///
    /// `expected` is an identity the consumer retained independently; when
    /// given, every field must match, so a resealed change of mechanisms,
    /// conditioning, world definitions or coupling is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed
    /// identity, stored states or numbers that do not replay, cancellation or
    /// an executor refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&CrossWorldExecutionIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CrossWorldExecutionArtifactError> {
        Self::from_bytes_within(bytes, expected, ctx, MAX_CROSS_WORLD_EXECUTION_ARTIFACT_BYTES)
    }

    /// [`Self::from_bytes`] under the caller's byte bound instead of
    /// [`MAX_CROSS_WORLD_EXECUTION_ARTIFACT_BYTES`].
    ///
    /// # Errors
    /// As [`Self::from_bytes`]; a file over `max_bytes` is `LimitsExceeded`.
    pub fn from_bytes_within(
        bytes: &[u8],
        expected: Option<&CrossWorldExecutionIdentity>,
        ctx: &ExecutionContext,
        max_bytes: usize,
    ) -> Result<Self, CrossWorldExecutionArtifactError> {
        let (meta, stored) = Self::decode(bytes, max_bytes)?;
        crate::artifact_codec::require_semantics::<CrossWorldExecutionArtifactError>(
            meta.feature == CROSS_WORLD_EXECUTION_ARTIFACT_FEATURE,
            "feature marker",
        )?;
        if meta.inference_claim != CROSS_WORLD_EXECUTION_INFERENCE_CLAIM {
            return Err(CrossWorldExecutionArtifactError::UnsupportedSemantics("inference claim"));
        }
        let request = request_of(&meta.request)?;
        let mut executor = FiniteScmExecutor::new();
        let result = executor.execute(&request, ctx).map_err(from_core)?;
        if result.matrix_row != meta.matrix_row {
            return Err(CrossWorldExecutionArtifactError::ResultMismatch("matrix row"));
        }
        let recomputed = identity_of(&result);
        if let Some(field) = identity_diff(&meta.identity, &recomputed) {
            return Err(CrossWorldExecutionArtifactError::IdentityMismatch { field });
        }
        let (draws, values) = collect(&executor, &result);
        if draws != meta.shared_draws {
            return Err(CrossWorldExecutionArtifactError::ResultMismatch(
                "shared exogenous states",
            ));
        }
        if bits(&values) != bits(&stored) {
            return Err(CrossWorldExecutionArtifactError::ResultMismatch(
                "values, evidence probabilities or posterior weights",
            ));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &recomputed) {
                return Err(CrossWorldExecutionArtifactError::IdentityMismatch { field });
            }
        }
        if meta.shared_draws.iter().any(|d| d.states.len() > MAX_JOINT_STATES) {
            return Err(CrossWorldExecutionArtifactError::LimitsExceeded("posterior states"));
        }
        Ok(Self { meta, values: stored, result, request, executor })
    }
}
