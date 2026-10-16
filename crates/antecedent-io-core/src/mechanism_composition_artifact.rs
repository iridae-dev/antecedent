//! Replayable artifact for a composition of independently fitted mechanisms.
//!
//! Format version 1, kind `mechanism_composition_v1`. The artifact keeps every
//! premise the composed answer depends on: the mechanisms with their numbers
//! (linear-Gaussian mechanisms reuse [`MechanismSlotWire`]), parent bindings,
//! fitted domains, contexts, provenance, declared shared latents, the declared
//! graph, admissions of mechanisms from other contexts, the declared posterior
//! dependence, the query, and the stored result.
//!
//! A consumer trusts none of it. It recomputes the premises digest over the
//! canonical (listing-order independent) premises and the query, recompiles the
//! composition under the model-level coherence checker, replays the query, and
//! accepts only the identical result bits. A resealed edit that changes any
//! mechanism, binding, admission or declared dependence changes either the
//! coherence verdict or the replayed result; an edit that is not resealed is a
//! premises mismatch.
//!
//! What replay protects against: corruption, edits that are not resealed, and
//! resealed edits that change the answer. What it does not protect against: a
//! producer that seals wrong but coherent premises (the consumer replays what it
//! is given), or a bug shared by producer and consumer (the same model code
//! recomputes the result; the independent oracles are in the test suites).
//! No interval is carried: uncertainty is a parameter-covariance variance of a
//! functional, labelled with how its dependence was licensed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::{
    AdmissionBasis, AdmissionCoordinate, DependenceDeclaration, ExternalRefusal,
    MechanismCompositionPlan, MechanismOriginKind, composition_refusal,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_model::compose::{
    CoherenceFinding, ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CompositionError,
    ContextCoordinate, ContrastArm, CrossCovarianceBlock, Domain, InvarianceDeclaration,
    LatentDependence, MAX_COMPOSED_MECHANISMS, MeanContrast, MechanismContext, MechanismDescriptor,
    MechanismFamily, MechanismForm, MechanismOrigin, MechanismProvenance, MechanismUncertainty,
    ParameterId, ParameterTerm, ParentBinding, PosteriorDependence, SelectionDiagramDeclaration,
    SelectionNodeDeclaration, TabularRow, UncertaintyMeaning,
};
use serde::{Deserialize, Serialize};

use crate::{IoError, MechanismSlotWire};

/// The artifact format this reader writes and accepts.
pub const MECHANISM_COMPOSITION_ARTIFACT_VERSION: u32 = 1;
/// The feature marker (and artifact kind) of the accepted format.
pub const MECHANISM_COMPOSITION_ARTIFACT_FEATURE: &str = "mechanism_composition_v1";
/// Most arms a stored functional may carry.
pub const MECHANISM_COMPOSITION_MAX_ARMS: usize = 16;
/// Most interventions one stored arm may carry.
pub const MECHANISM_COMPOSITION_MAX_INTERVENTIONS: usize = 64;

/// Why a mechanism-composition artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MechanismCompositionArtifactError {
    /// The bytes do not decode as this format.
    #[error("mechanism-composition artifact does not decode: {0}")]
    Decode(String),
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises and query, or
    /// the caller's expected identity.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The artifact could not be encoded.
    #[error("mechanism-composition artifact does not encode: {0}")]
    Encode(String),
    /// A stored field is not a well formed declaration.
    #[error("malformed premises: {0}")]
    Malformed(String),
    /// The stored mechanisms do not establish a coherent joint law. Each
    /// refusal carries a registered code, a namespaced detail and a remedy.
    #[error("mechanisms do not compose: {} refusal(s)", .refusals.len())]
    Incoherent {
        /// Every defect, as structured refusals.
        refusals: Vec<ExternalRefusal>,
    },
    /// The replayed result differs from the stored one.
    #[error("result does not replay")]
    ResultMismatch,
    /// The replay itself failed or was cancelled.
    #[error("replay failed: {0}")]
    Replay(String),
}

impl From<IoError> for MechanismCompositionArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

/// Structured refusals for model-level findings, using the semantic contract's
/// code mapping so a defect has one name in every layer.
#[must_use]
pub fn refusals_for_findings(findings: &[CoherenceFinding]) -> Vec<ExternalRefusal> {
    findings
        .iter()
        .map(|f| {
            composition_refusal(
                f.kind.tag(),
                f.detail,
                &f.subject,
                f.expected.as_deref(),
                f.supplied.as_deref(),
                f.remedy,
            )
        })
        .collect()
}

fn incoherent(error: CompositionError) -> MechanismCompositionArtifactError {
    match error {
        CompositionError::Incoherent { findings } => {
            MechanismCompositionArtifactError::Incoherent {
                refusals: refusals_for_findings(&findings),
            }
        }
        other => MechanismCompositionArtifactError::Replay(other.to_string()),
    }
}

/// A domain on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DomainWire {
    /// The real line.
    Real,
    /// A closed interval.
    Interval {
        /// Lower end.
        lo: f64,
        /// Upper end.
        hi: f64,
    },
    /// Named levels.
    Levels {
        /// The levels.
        levels: Vec<String>,
    },
}

/// A parent binding on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentBindingWire {
    /// Parent variable.
    pub variable: String,
    /// Units.
    pub units: String,
    /// Transform.
    pub transform: String,
    /// Fitted domain.
    pub domain: DomainWire,
}

/// One table row on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TabularRowWire {
    /// `(parent, level)`.
    pub parent_levels: Vec<(String, String)>,
    /// `(level, probability)`.
    pub probabilities: Vec<(String, f64)>,
}

/// The numbers of a mechanism on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MechanismFormWire {
    /// A linear-Gaussian slot. `coeffs` are aligned with the entry's `parents`.
    LinearGaussian {
        /// The reused mechanism slot (must be the `linear_gaussian` variant).
        slot: MechanismSlotWire,
    },
    /// A finite conditional table.
    Tabular {
        /// Levels of the produced variable.
        levels: Vec<String>,
        /// Rows.
        rows: Vec<TabularRowWire>,
    },
}

/// A latent dependence on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentWire {
    /// Latent identity.
    pub latent: String,
    /// Loading, when declared.
    pub loading: Option<f64>,
    /// Other mechanisms sharing the latent.
    pub shared_with: Vec<String>,
}

/// A parameter term on the wire.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TermWire {
    /// The intercept.
    Intercept,
    /// A coefficient on a parent.
    Coefficient {
        /// Parent variable.
        parent: String,
    },
}

/// A mechanism's reported parameter covariance on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyWire {
    /// `posterior` or `sampling_distribution`.
    pub meaning: String,
    /// Parameters the covariance is over.
    pub parameters: Vec<TermWire>,
    /// Row-major covariance.
    pub covariance: Vec<f64>,
}

/// One mechanism on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComposedMechanismWire {
    /// Produced variable.
    pub variable: String,
    /// Numbers.
    pub form: MechanismFormWire,
    /// Parent bindings.
    pub parents: Vec<ParentBindingWire>,
    /// Conditioning set of the fit.
    pub conditioning_set: Vec<String>,
    /// Units produced.
    pub output_units: String,
    /// Transform produced.
    pub output_transform: String,
    /// Support produced.
    pub output_support: DomainWire,
    /// Population fitted in.
    pub population: String,
    /// Regime fitted in.
    pub regime: String,
    /// `fitted`, `external_binding` or `declared`.
    pub origin: String,
    /// Owning fit or binding.
    pub artifact_id: String,
    /// Evidence identity.
    pub evidence_id: String,
    /// Shared latent causes.
    pub latents: Vec<LatentWire>,
    /// Reported parameter covariance.
    pub uncertainty: Option<UncertaintyWire>,
}

/// An invariance declaration on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvarianceWire {
    /// Variable.
    pub variable: String,
    /// `population` or `regime`.
    pub coordinate: String,
    /// Source value.
    pub source: String,
    /// Target value.
    pub target: String,
    /// Assumption identity.
    pub assumption_id: String,
}

/// A selection diagram on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionDiagramWire {
    /// Source population.
    pub source_population: String,
    /// Target population.
    pub target_population: String,
    /// `(node, variables it points into)`.
    pub selection_nodes: Vec<(String, Vec<String>)>,
}

/// A parameter identity on the wire.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterIdWire {
    /// Mechanism variable.
    pub mechanism: String,
    /// Parameter.
    pub term: TermWire,
}

/// One cross-covariance block on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossBlockWire {
    /// First mechanism.
    pub first: String,
    /// Second mechanism.
    pub second: String,
    /// Rows.
    pub first_parameters: Vec<TermWire>,
    /// Columns.
    pub second_parameters: Vec<TermWire>,
    /// Row-major values.
    pub values: Vec<f64>,
}

/// Declared posterior dependence on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DependenceWire {
    /// Nothing declared.
    Unknown,
    /// Declared independent by design.
    IndependentByDesign {
        /// Justification.
        justification: String,
    },
    /// One aligned joint law.
    Joint {
        /// Meaning of the law.
        meaning: String,
        /// Parameters in covariance order.
        parameters: Vec<ParameterIdWire>,
        /// Row-major covariance.
        covariance: Vec<f64>,
    },
    /// Cross-mechanism covariance blocks.
    CrossCovariance {
        /// Blocks.
        blocks: Vec<CrossBlockWire>,
    },
}

/// Everything the composed answer is derived from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismCompositionPremisesWire {
    /// Target population.
    pub target_population: String,
    /// Target regime.
    pub target_regime: String,
    /// Mechanisms in canonical order.
    pub mechanisms: Vec<ComposedMechanismWire>,
    /// Declared edges, canonical.
    pub edges: Vec<(String, String)>,
    /// Declared invariances.
    pub invariances: Vec<InvarianceWire>,
    /// Declared selection diagrams.
    pub selection_diagrams: Vec<SelectionDiagramWire>,
    /// Declared dependence.
    pub dependence: DependenceWire,
    /// Whether declared (not estimated) mechanisms are acknowledged.
    pub admit_declared_mechanisms: bool,
}

/// What is asked of the composition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CompositionQueryWire {
    /// Mean and variance of one outcome of the linear-Gaussian composition
    /// under hard interventions.
    GaussianInterventionLaw {
        /// `(variable, value)`.
        interventions: Vec<(String, f64)>,
        /// Outcome variable.
        outcome: String,
    },
    /// A weighted contrast of interventional means with parameter uncertainty
    /// propagated by the first-order delta method.
    GaussianMeanContrast {
        /// Outcome variable.
        outcome: String,
        /// `(weight, interventions)`.
        arms: Vec<(f64, Vec<(String, f64)>)>,
    },
    /// Exact distribution of one outcome of the finite composition under hard
    /// interventions.
    FiniteInterventionDistribution {
        /// `(variable, level)`.
        interventions: Vec<(String, String)>,
        /// Outcome variable.
        outcome: String,
    },
}

/// The stored result, as IEEE bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CompositionResultWire {
    /// Mean and variance.
    GaussianLaw {
        /// Mean bits.
        mean_bits: u64,
        /// Variance bits.
        variance_bits: u64,
    },
    /// Point and parameter-uncertainty variance of a contrast.
    GaussianFunctional {
        /// Point bits.
        point_bits: u64,
        /// Variance bits (parameter covariance only; not an interval).
        variance_bits: u64,
        /// How the dependence between mechanisms was licensed.
        dependence: String,
        /// Method label.
        method: String,
    },
    /// `(level, probability bits)`.
    FiniteDistribution {
        /// Level probabilities in level order.
        probabilities: Vec<(String, u64)>,
    },
}

/// The versioned artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismCompositionArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Premises.
    pub premises: MechanismCompositionPremisesWire,
    /// The question.
    pub query: CompositionQueryWire,
    /// The stored answer.
    pub result: CompositionResultWire,
    /// Digest of the premises and the query.
    pub premises_digest: String,
}

/// A consumed artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedMechanismComposition {
    /// The identity that was recomputed and verified.
    pub premises_digest: String,
    /// The family the composition executed in.
    pub family: MechanismFamily,
    /// The replayed result (bit-identical to the stored one).
    pub result: CompositionResultWire,
}

// ---------------------------------------------------------------- conversions

fn domain_to_wire(d: &Domain) -> DomainWire {
    match d {
        Domain::Real => DomainWire::Real,
        Domain::Interval { lo, hi } => DomainWire::Interval { lo: *lo, hi: *hi },
        Domain::Levels(l) => DomainWire::Levels { levels: l.clone() },
    }
}

fn domain_from_wire(d: &DomainWire) -> Domain {
    match d {
        DomainWire::Real => Domain::Real,
        DomainWire::Interval { lo, hi } => Domain::Interval { lo: *lo, hi: *hi },
        DomainWire::Levels { levels } => Domain::Levels(levels.clone()),
    }
}

fn term_to_wire(t: &ParameterTerm) -> TermWire {
    match t {
        ParameterTerm::Intercept => TermWire::Intercept,
        ParameterTerm::Coefficient(p) => TermWire::Coefficient { parent: p.clone() },
    }
}

fn term_from_wire(t: &TermWire) -> ParameterTerm {
    match t {
        TermWire::Intercept => ParameterTerm::Intercept,
        TermWire::Coefficient { parent } => ParameterTerm::Coefficient(parent.clone()),
    }
}

fn meaning_tag(m: UncertaintyMeaning) -> &'static str {
    match m {
        UncertaintyMeaning::Posterior => "posterior",
        UncertaintyMeaning::SamplingDistribution => "sampling_distribution",
    }
}

fn meaning_from(tag: &str) -> Result<UncertaintyMeaning, MechanismCompositionArtifactError> {
    match tag {
        "posterior" => Ok(UncertaintyMeaning::Posterior),
        "sampling_distribution" => Ok(UncertaintyMeaning::SamplingDistribution),
        _ => Err(MechanismCompositionArtifactError::Malformed("uncertainty meaning".into())),
    }
}

fn origin_tag(o: MechanismOrigin) -> &'static str {
    match o {
        MechanismOrigin::Fitted => "fitted",
        MechanismOrigin::ExternalBinding => "external_binding",
        MechanismOrigin::Declared => "declared",
    }
}

fn origin_from(tag: &str) -> Result<MechanismOrigin, MechanismCompositionArtifactError> {
    match tag {
        "fitted" => Ok(MechanismOrigin::Fitted),
        "external_binding" => Ok(MechanismOrigin::ExternalBinding),
        "declared" => Ok(MechanismOrigin::Declared),
        _ => Err(MechanismCompositionArtifactError::Malformed("mechanism origin".into())),
    }
}

fn mechanism_to_wire(d: &MechanismDescriptor) -> ComposedMechanismWire {
    let form = match &d.form {
        MechanismForm::LinearGaussian { intercept, sigma, coefficients } => {
            // Coefficients are aligned with the (canonical) parent order.
            MechanismFormWire::LinearGaussian {
                slot: MechanismSlotWire::LinearGaussian {
                    intercept: *intercept,
                    coeffs: coefficients.iter().map(|(_, c)| *c).collect(),
                    sigma: *sigma,
                },
            }
        }
        MechanismForm::Tabular { levels, rows } => MechanismFormWire::Tabular {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|r| TabularRowWire {
                    parent_levels: r.parent_levels.clone(),
                    probabilities: r.probabilities.clone(),
                })
                .collect(),
        },
    };
    ComposedMechanismWire {
        variable: d.variable.clone(),
        form,
        parents: d
            .parents
            .iter()
            .map(|p| ParentBindingWire {
                variable: p.variable.clone(),
                units: p.units.clone(),
                transform: p.transform.clone(),
                domain: domain_to_wire(&p.domain),
            })
            .collect(),
        conditioning_set: d.conditioning_set.clone(),
        output_units: d.output_units.clone(),
        output_transform: d.output_transform.clone(),
        output_support: domain_to_wire(&d.output_support),
        population: d.context.population.clone(),
        regime: d.context.regime.clone(),
        origin: origin_tag(d.provenance.origin).to_owned(),
        artifact_id: d.provenance.artifact_id.clone(),
        evidence_id: d.provenance.evidence_id.clone(),
        latents: d
            .latents
            .iter()
            .map(|l| LatentWire {
                latent: l.latent.clone(),
                loading: l.loading,
                shared_with: l.shared_with.clone(),
            })
            .collect(),
        uncertainty: d.uncertainty.as_ref().map(|u| UncertaintyWire {
            meaning: meaning_tag(u.meaning).to_owned(),
            parameters: u.parameters.iter().map(term_to_wire).collect(),
            covariance: u.covariance.clone(),
        }),
    }
}

fn mechanism_from_wire(
    w: &ComposedMechanismWire,
) -> Result<MechanismDescriptor, MechanismCompositionArtifactError> {
    let malformed = |m: &str| MechanismCompositionArtifactError::Malformed(m.into());
    let form = match &w.form {
        MechanismFormWire::LinearGaussian { slot } => {
            let MechanismSlotWire::LinearGaussian { intercept, coeffs, sigma } = slot else {
                return Err(malformed("linear-Gaussian form carries another slot"));
            };
            if coeffs.len() != w.parents.len() {
                return Err(malformed("coefficients are not aligned with the parents"));
            }
            MechanismForm::LinearGaussian {
                intercept: *intercept,
                sigma: *sigma,
                coefficients: w
                    .parents
                    .iter()
                    .zip(coeffs.iter())
                    .map(|(p, c)| (p.variable.clone(), *c))
                    .collect(),
            }
        }
        MechanismFormWire::Tabular { levels, rows } => MechanismForm::Tabular {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|r| TabularRow {
                    parent_levels: r.parent_levels.clone(),
                    probabilities: r.probabilities.clone(),
                })
                .collect(),
        },
    };
    let uncertainty = match &w.uncertainty {
        None => None,
        Some(u) => Some(MechanismUncertainty {
            meaning: meaning_from(&u.meaning)?,
            parameters: u.parameters.iter().map(term_from_wire).collect(),
            covariance: u.covariance.clone(),
        }),
    };
    Ok(MechanismDescriptor {
        variable: w.variable.clone(),
        form,
        parents: w
            .parents
            .iter()
            .map(|p| ParentBinding {
                variable: p.variable.clone(),
                units: p.units.clone(),
                transform: p.transform.clone(),
                domain: domain_from_wire(&p.domain),
            })
            .collect(),
        conditioning_set: w.conditioning_set.clone(),
        output_units: w.output_units.clone(),
        output_transform: w.output_transform.clone(),
        output_support: domain_from_wire(&w.output_support),
        context: MechanismContext { population: w.population.clone(), regime: w.regime.clone() },
        provenance: MechanismProvenance {
            origin: origin_from(&w.origin)?,
            artifact_id: w.artifact_id.clone(),
            evidence_id: w.evidence_id.clone(),
        },
        latents: w
            .latents
            .iter()
            .map(|l| LatentDependence {
                latent: l.latent.clone(),
                loading: l.loading,
                shared_with: l.shared_with.clone(),
            })
            .collect(),
        uncertainty,
    })
}

impl MechanismCompositionPremisesWire {
    /// Encode a specification in canonical order, so two declarations that
    /// differ only in listing order have one wire form and one digest.
    #[must_use]
    pub fn from_spec(spec: &ComposedModelSpec) -> Self {
        let s = spec.canonical();
        // Coefficients travel aligned with the canonical parent order.
        let mechanisms = s.mechanisms.iter().map(mechanism_to_wire).collect();
        let dependence = match &s.dependence {
            PosteriorDependence::Unknown => DependenceWire::Unknown,
            PosteriorDependence::IndependentByDesign { justification } => {
                DependenceWire::IndependentByDesign { justification: justification.clone() }
            }
            PosteriorDependence::Joint { meaning, parameters, covariance } => {
                DependenceWire::Joint {
                    meaning: meaning_tag(*meaning).to_owned(),
                    parameters: parameters
                        .iter()
                        .map(|p| ParameterIdWire {
                            mechanism: p.mechanism.clone(),
                            term: term_to_wire(&p.term),
                        })
                        .collect(),
                    covariance: covariance.clone(),
                }
            }
            PosteriorDependence::CrossCovariance { blocks } => DependenceWire::CrossCovariance {
                blocks: blocks
                    .iter()
                    .map(|b| CrossBlockWire {
                        first: b.first.clone(),
                        second: b.second.clone(),
                        first_parameters: b.first_parameters.iter().map(term_to_wire).collect(),
                        second_parameters: b.second_parameters.iter().map(term_to_wire).collect(),
                        values: b.values.clone(),
                    })
                    .collect(),
            },
        };
        Self {
            target_population: s.target.population.clone(),
            target_regime: s.target.regime.clone(),
            mechanisms,
            edges: s.edges.clone(),
            invariances: s
                .invariances
                .iter()
                .map(|i| InvarianceWire {
                    variable: i.variable.clone(),
                    coordinate: match i.coordinate {
                        ContextCoordinate::Population => "population".to_owned(),
                        ContextCoordinate::Regime => "regime".to_owned(),
                    },
                    source: i.source.clone(),
                    target: i.target.clone(),
                    assumption_id: i.assumption_id.clone(),
                })
                .collect(),
            selection_diagrams: s
                .selection_diagrams
                .iter()
                .map(|d| SelectionDiagramWire {
                    source_population: d.source_population.clone(),
                    target_population: d.target_population.clone(),
                    selection_nodes: d
                        .selection_nodes
                        .iter()
                        .map(|n| (n.name.clone(), n.points_into.clone()))
                        .collect(),
                })
                .collect(),
            dependence,
            admit_declared_mechanisms: s.admit_declared_mechanisms,
        }
    }

    /// Decode into a model specification.
    ///
    /// # Errors
    /// An unknown tag or a form that disagrees with its parents.
    pub fn to_spec(&self) -> Result<ComposedModelSpec, MechanismCompositionArtifactError> {
        let malformed = |m: &str| MechanismCompositionArtifactError::Malformed(m.into());
        let mechanisms =
            self.mechanisms.iter().map(mechanism_from_wire).collect::<Result<Vec<_>, _>>()?;
        let mut invariances = Vec::with_capacity(self.invariances.len());
        for i in &self.invariances {
            let coordinate = match i.coordinate.as_str() {
                "population" => ContextCoordinate::Population,
                "regime" => ContextCoordinate::Regime,
                _ => return Err(malformed("invariance coordinate")),
            };
            invariances.push(InvarianceDeclaration {
                variable: i.variable.clone(),
                coordinate,
                source: i.source.clone(),
                target: i.target.clone(),
                assumption_id: i.assumption_id.clone(),
            });
        }
        let dependence = match &self.dependence {
            DependenceWire::Unknown => PosteriorDependence::Unknown,
            DependenceWire::IndependentByDesign { justification } => {
                PosteriorDependence::IndependentByDesign { justification: justification.clone() }
            }
            DependenceWire::Joint { meaning, parameters, covariance } => {
                PosteriorDependence::Joint {
                    meaning: meaning_from(meaning)?,
                    parameters: parameters
                        .iter()
                        .map(|p| ParameterId {
                            mechanism: p.mechanism.clone(),
                            term: term_from_wire(&p.term),
                        })
                        .collect(),
                    covariance: covariance.clone(),
                }
            }
            DependenceWire::CrossCovariance { blocks } => PosteriorDependence::CrossCovariance {
                blocks: blocks
                    .iter()
                    .map(|b| CrossCovarianceBlock {
                        first: b.first.clone(),
                        second: b.second.clone(),
                        first_parameters: b.first_parameters.iter().map(term_from_wire).collect(),
                        second_parameters: b.second_parameters.iter().map(term_from_wire).collect(),
                        values: b.values.clone(),
                    })
                    .collect(),
            },
        };
        Ok(ComposedModelSpec {
            target: MechanismContext {
                population: self.target_population.clone(),
                regime: self.target_regime.clone(),
            },
            mechanisms,
            edges: self.edges.clone(),
            invariances,
            selection_diagrams: self
                .selection_diagrams
                .iter()
                .map(|d| SelectionDiagramDeclaration {
                    source_population: d.source_population.clone(),
                    target_population: d.target_population.clone(),
                    selection_nodes: d
                        .selection_nodes
                        .iter()
                        .map(|(name, into)| SelectionNodeDeclaration {
                            name: name.clone(),
                            points_into: into.clone(),
                        })
                        .collect(),
                })
                .collect(),
            dependence,
            admit_declared_mechanisms: self.admit_declared_mechanisms,
        })
    }
}

// ------------------------------------------------------------------ identity

/// Identity of the premises and the query.
///
/// The digest is over the canonical wire encoding under a versioned tag, with
/// the stored result excluded: the result is an output replayed from the
/// premises, not a premise.
///
/// # Errors
/// Encoding failure.
pub fn mechanism_composition_identity(
    premises: &MechanismCompositionPremisesWire,
    query: &CompositionQueryWire,
) -> Result<String, IoError> {
    crate::digest_hex(
        IdentityDomain::Program,
        &("mechanism_composition_v1", MECHANISM_COMPOSITION_ARTIFACT_VERSION, premises, query),
    )
}

// ------------------------------------------------------------------ execution

fn check_query_bounds(
    query: &CompositionQueryWire,
) -> Result<(), MechanismCompositionArtifactError> {
    match query {
        CompositionQueryWire::GaussianInterventionLaw { interventions, .. } => {
            if interventions.len() > MECHANISM_COMPOSITION_MAX_INTERVENTIONS {
                return Err(MechanismCompositionArtifactError::LimitsExceeded("interventions"));
            }
        }
        CompositionQueryWire::GaussianMeanContrast { arms, .. } => {
            if arms.is_empty() || arms.len() > MECHANISM_COMPOSITION_MAX_ARMS {
                return Err(MechanismCompositionArtifactError::LimitsExceeded("arms"));
            }
            if arms.iter().any(|(_, i)| i.len() > MECHANISM_COMPOSITION_MAX_INTERVENTIONS) {
                return Err(MechanismCompositionArtifactError::LimitsExceeded("interventions"));
            }
        }
        CompositionQueryWire::FiniteInterventionDistribution { interventions, .. } => {
            if interventions.len() > MECHANISM_COMPOSITION_MAX_INTERVENTIONS {
                return Err(MechanismCompositionArtifactError::LimitsExceeded("interventions"));
            }
        }
    }
    Ok(())
}

fn check_premises_bounds(
    premises: &MechanismCompositionPremisesWire,
) -> Result<(), MechanismCompositionArtifactError> {
    if premises.mechanisms.is_empty() || premises.mechanisms.len() > MAX_COMPOSED_MECHANISMS {
        return Err(MechanismCompositionArtifactError::LimitsExceeded("mechanisms"));
    }
    if premises.edges.len() > MAX_COMPOSED_MECHANISMS * MAX_COMPOSED_MECHANISMS {
        return Err(MechanismCompositionArtifactError::LimitsExceeded("edges"));
    }
    Ok(())
}

/// Run `query` on `spec`: compile under the coherence checker, then execute.
///
/// # Errors
/// [`MechanismCompositionArtifactError::Incoherent`] with structured refusals
/// when the mechanisms do not compose or an uncertainty claim lacks a joint
/// law, and [`MechanismCompositionArtifactError::Replay`] for an execution
/// failure or cancellation.
pub fn evaluate_composition_query(
    spec: &ComposedModelSpec,
    query: &CompositionQueryWire,
    ctx: &ExecutionContext,
) -> Result<(MechanismFamily, CompositionResultWire), MechanismCompositionArtifactError> {
    check_query_bounds(query)?;
    match query {
        CompositionQueryWire::GaussianInterventionLaw { interventions, outcome } => {
            let model = ComposedLinearGaussian::compile(spec).map_err(incoherent)?;
            let pairs: Vec<(&str, f64)> =
                interventions.iter().map(|(v, x)| (v.as_str(), *x)).collect();
            let law = model.law(&pairs).map_err(incoherent)?;
            let (Some(mean), Some(variance)) = (law.mean_of(outcome), law.variance_of(outcome))
            else {
                return Err(MechanismCompositionArtifactError::Malformed(
                    "unknown outcome variable".into(),
                ));
            };
            Ok((
                MechanismFamily::LinearGaussian,
                CompositionResultWire::GaussianLaw {
                    mean_bits: mean.to_bits(),
                    variance_bits: variance.to_bits(),
                },
            ))
        }
        CompositionQueryWire::GaussianMeanContrast { outcome, arms } => {
            let model = ComposedLinearGaussian::compile(spec).map_err(incoherent)?;
            let contrast = MeanContrast {
                outcome: outcome.clone(),
                arms: arms
                    .iter()
                    .map(|(w, i)| ContrastArm { weight: *w, interventions: i.clone() })
                    .collect(),
            };
            let u = model.mean_contrast_uncertainty(&contrast).map_err(incoherent)?;
            Ok((
                MechanismFamily::LinearGaussian,
                CompositionResultWire::GaussianFunctional {
                    point_bits: u.point.to_bits(),
                    variance_bits: u.variance.to_bits(),
                    dependence: u.dependence.to_owned(),
                    method: u.method.to_owned(),
                },
            ))
        }
        CompositionQueryWire::FiniteInterventionDistribution { interventions, outcome } => {
            let model = ComposedFinite::compile(spec).map_err(incoherent)?;
            let pairs: Vec<(&str, &str)> =
                interventions.iter().map(|(v, l)| (v.as_str(), l.as_str())).collect();
            let law = model.law(&pairs, ctx).map_err(incoherent)?;
            let marginal = law.marginal(outcome).map_err(incoherent)?;
            Ok((
                MechanismFamily::Tabular,
                CompositionResultWire::FiniteDistribution {
                    probabilities: marginal
                        .into_iter()
                        .map(|(level, p)| (level, p.to_bits()))
                        .collect(),
                },
            ))
        }
    }
}

impl MechanismCompositionArtifactWire {
    /// Compile, execute and seal an artifact for `query` on `spec`.
    ///
    /// # Errors
    /// Incoherent mechanisms (structured refusals), a failed execution, or an
    /// encoding failure.
    pub fn seal(
        spec: &ComposedModelSpec,
        query: CompositionQueryWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, MechanismCompositionArtifactError> {
        let premises = MechanismCompositionPremisesWire::from_spec(spec);
        check_premises_bounds(&premises)?;
        let (_, result) = evaluate_composition_query(&premises.to_spec()?, &query, ctx)?;
        let premises_digest = mechanism_composition_identity(&premises, &query)
            .map_err(|e| MechanismCompositionArtifactError::Encode(e.to_string()))?;
        Ok(Self {
            version: MECHANISM_COMPOSITION_ARTIFACT_VERSION,
            required_features: vec![MECHANISM_COMPOSITION_ARTIFACT_FEATURE.to_owned()],
            premises,
            query,
            result,
            premises_digest,
        })
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, MechanismCompositionArtifactError> {
        check_premises_bounds(&self.premises)?;
        check_query_bounds(&self.query)?;
        crate::to_cbor(self).map_err(|e| MechanismCompositionArtifactError::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature before the payload is
    /// interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, MechanismCompositionArtifactError> {
        let wire: Self = crate::decode_with_version(bytes, |version| {
            if version == MECHANISM_COMPOSITION_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(MechanismCompositionArtifactError::UnsupportedVersion { version })
            }
        })?;
        if wire.required_features != [MECHANISM_COMPOSITION_ARTIFACT_FEATURE] {
            return Err(MechanismCompositionArtifactError::UnsupportedSemantics(
                "required features",
            ));
        }
        Ok(wire)
    }

    /// Verify and replay.
    ///
    /// `expected_identity`, when given, is the premises digest the caller holds
    /// independently of the artifact.
    ///
    /// # Errors
    /// Any of the refusals of [`MechanismCompositionArtifactError`].
    pub fn consume(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedMechanismComposition, MechanismCompositionArtifactError> {
        if self.version != MECHANISM_COMPOSITION_ARTIFACT_VERSION {
            return Err(MechanismCompositionArtifactError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.required_features != [MECHANISM_COMPOSITION_ARTIFACT_FEATURE] {
            return Err(MechanismCompositionArtifactError::UnsupportedSemantics(
                "required features",
            ));
        }
        check_premises_bounds(&self.premises)?;
        check_query_bounds(&self.query)?;
        let recomputed = mechanism_composition_identity(&self.premises, &self.query)
            .map_err(|e| MechanismCompositionArtifactError::Encode(e.to_string()))?;
        if recomputed != self.premises_digest {
            return Err(MechanismCompositionArtifactError::PremisesMismatch);
        }
        if let Some(expected) = expected_identity {
            if expected != recomputed {
                return Err(MechanismCompositionArtifactError::PremisesMismatch);
            }
        }
        let spec = self.premises.to_spec()?;
        let (family, result) = evaluate_composition_query(&spec, &self.query, ctx)?;
        if result != self.result {
            return Err(MechanismCompositionArtifactError::ResultMismatch);
        }
        Ok(ConsumedMechanismComposition { premises_digest: recomputed, family, result })
    }
}

// ---------------------------------------------------- semantic plan -> numbers

/// The numbers of one planned mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismNumbers {
    /// The mechanism's numbers.
    pub form: MechanismForm,
    /// Support of the produced variable.
    pub output_support: Domain,
    /// Fitted domain of each parent the plan names.
    pub parent_domains: Vec<(String, Domain)>,
    /// Marginal parameter covariance reported by the fit.
    pub uncertainty: Option<MechanismUncertainty>,
    /// Loading of this mechanism on each shared latent the plan declares for
    /// it. A declared latent with no loading here stays undeclared and the
    /// composition refuses it.
    pub latent_loadings: Vec<(String, f64)>,
}

/// Bind a semantically checked plan to its numbers, producing the model-level
/// specification. Units, transforms, populations, regimes, parents,
/// conditioning sets, provenance, latent declarations, admissions and the
/// dependence kind all come from the plan, so the numbers cannot restate them.
///
/// `dependence` must be the model-level law of the kind the plan declares.
///
/// # Errors
/// A planned mechanism without numbers, a parent without a declared domain, or
/// a dependence law of another kind than the plan declares.
#[allow(
    clippy::too_many_lines,
    reason = "one linear lowering of the plan whose steps share the declared numbers and dependence law"
)]
pub fn lower_plan(
    plan: &MechanismCompositionPlan,
    numbers: &BTreeMap<String, MechanismNumbers>,
    dependence: PosteriorDependence,
) -> Result<ComposedModelSpec, MechanismCompositionArtifactError> {
    let malformed = MechanismCompositionArtifactError::Malformed;
    let mut mechanisms = Vec::with_capacity(plan.mechanisms.len());
    for m in &plan.mechanisms {
        let Some(n) = numbers.get(&m.variable_id) else {
            return Err(malformed(format!("no numbers for mechanism `{}`", m.variable_id)));
        };
        let mut parents = Vec::with_capacity(m.parents.len());
        for p in &m.parents {
            let Some((_, domain)) = n.parent_domains.iter().find(|(v, _)| *v == p.variable_id)
            else {
                return Err(malformed(format!(
                    "no fitted domain for `{}` in the mechanism of `{}`",
                    p.variable_id, m.variable_id
                )));
            };
            parents.push(ParentBinding {
                variable: p.variable_id.clone(),
                units: p.units.clone(),
                transform: p.transform_id.clone(),
                domain: domain.clone(),
            });
        }
        if let Some((id, _)) = n
            .latent_loadings
            .iter()
            .find(|(id, _)| !m.latent_causes.iter().any(|l| l.latent_id == *id))
        {
            return Err(malformed(format!(
                "loading on latent `{id}` which the plan does not declare for `{}`",
                m.variable_id
            )));
        }
        let latents: Vec<LatentDependence> = m
            .latent_causes
            .iter()
            .map(|l| LatentDependence {
                latent: l.latent_id.clone(),
                loading: n
                    .latent_loadings
                    .iter()
                    .find(|(id, _)| *id == l.latent_id)
                    .map(|(_, x)| *x),
                shared_with: l.shared_with_variable_ids.clone(),
            })
            .collect();
        mechanisms.push(MechanismDescriptor {
            variable: m.variable_id.clone(),
            form: n.form.clone(),
            parents,
            conditioning_set: m.conditioning_variable_ids.clone(),
            output_units: m.units.clone(),
            output_transform: m.transform_id.clone(),
            output_support: n.output_support.clone(),
            context: MechanismContext {
                population: m.population_id.clone(),
                regime: m.regime_id.clone(),
            },
            provenance: MechanismProvenance {
                origin: match m.origin {
                    MechanismOriginKind::Fitted => MechanismOrigin::Fitted,
                    MechanismOriginKind::ExternalBinding => MechanismOrigin::ExternalBinding,
                    MechanismOriginKind::Declared => MechanismOrigin::Declared,
                },
                artifact_id: m.artifact_id.clone(),
                evidence_id: m.evidence_id.clone(),
            },
            latents,
            uncertainty: n.uncertainty.clone(),
        });
    }
    let kind_matches = matches!(
        (&plan.dependence, &dependence),
        (DependenceDeclaration::Unknown, PosteriorDependence::Unknown)
            | (
                DependenceDeclaration::IndependentByDesign { .. },
                PosteriorDependence::IndependentByDesign { .. }
            )
            | (DependenceDeclaration::AlignedJointLaw { .. }, PosteriorDependence::Joint { .. })
            | (DependenceDeclaration::CrossCovariance, PosteriorDependence::CrossCovariance { .. })
    );
    if !kind_matches {
        return Err(malformed(
            "the supplied dependence law is not of the kind the plan declares".into(),
        ));
    }
    let mut invariances = Vec::new();
    let mut selection_diagrams = Vec::new();
    for a in &plan.admissions {
        match &a.basis {
            AdmissionBasis::DeclaredInvariance { assumption_id } => {
                invariances.push(InvarianceDeclaration {
                    variable: a.variable_id.clone(),
                    coordinate: match a.coordinate {
                        AdmissionCoordinate::Population => ContextCoordinate::Population,
                        AdmissionCoordinate::Regime => ContextCoordinate::Regime,
                    },
                    source: a.source_id.clone(),
                    target: a.target_id.clone(),
                    assumption_id: assumption_id.clone(),
                });
            }
            AdmissionBasis::SelectionDiagram { targets } => {
                selection_diagrams.push(SelectionDiagramDeclaration {
                    source_population: a.source_id.clone(),
                    target_population: a.target_id.clone(),
                    selection_nodes: vec![SelectionNodeDeclaration {
                        name: "selection".to_owned(),
                        points_into: targets.clone(),
                    }],
                });
            }
        }
    }
    Ok(ComposedModelSpec {
        target: MechanismContext {
            population: plan.target_population_id.clone(),
            regime: plan.target_regime_id.clone(),
        },
        mechanisms,
        edges: plan.edges.clone(),
        invariances,
        selection_diagrams,
        dependence,
        admit_declared_mechanisms: plan.admit_declared_mechanisms,
    })
}
