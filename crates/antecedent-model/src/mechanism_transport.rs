//! Transport of fitted mechanisms into a composed model of a target population.
//!
//! A mechanism fitted in a source population is a component of a
//! target-population composed model only when the declared selection diagram
//! marks it invariant. This module is the bounded executed route for that
//! admission: it takes fitted linear-Gaussian mechanisms from the source,
//! natively refitted mechanisms of the target, and one selection diagram, and
//! either returns an executable composed model whose every component says
//! whether it was carried or refitted, or refuses with the same finding
//! literals the composition checker emits.
//!
//! # Boundary
//!
//! - **Mechanisms**: fitted (not declared, not externally bound),
//!   additive-noise linear-Gaussian. Finite tables are covered by the finite
//!   counterfactual transport derivation, which decides them with an exact
//!   obstruction witness; a table here is refused with a pointer to it.
//! - **Admission**: a mechanism is carried only when no selection node points
//!   into its variable and none into a latent cause it loads on. A mechanism
//!   with a shared latent cause additionally needs the analyst's explicit
//!   population invariance declaration, exactly as the composition checker
//!   requires; the diagram alone does not cover a latent.
//! - **Precedence**: a natively refitted target mechanism is always used in
//!   place of a carried one for the same variable. The carried one is never
//!   averaged with it.
//! - **Uncertainty**: the parameter covariance of a carried mechanism is the
//!   covariance of its *source* fit. The selection-diagram assumption adds no
//!   variance here and the result says so; it is a statement about which
//!   estimates may be used, not a calibrated claim about the target.
//!
//! The finite-table obstruction and the composition refusal of this module are
//! checked against one shared fixture in the crate's integration tests.
//!
//! # Refresh with outcome-only data
//!
//! The composed model is a pure function of the request, so there is no in-place
//! refresh. New outcome-only data for a *source* mechanism is a refit with a new fit
//! identity: carried components and the model identity are recomputed. New data for a
//! target-native mechanism replaces that component and keeps the carried ones. The
//! selection diagram and invariance declarations are kept only by being passed again;
//! data cannot move them. A refit under another population, regime or quantity than the
//! request names is refused (the diagram and context findings), never adopted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::identity::CanonicalWriter;

use crate::compose::coherence::CoherenceFinding;
use crate::compose::{
    CoherenceKind, ComposedLinearGaussian, ComposedModelSpec, CompositionError,
    Domain, FunctionalUncertainty, InvarianceDeclaration, MeanContrast, MechanismContext,
    MechanismDescriptor, MechanismFamily, MechanismForm, MechanismOrigin, ParameterTerm,
    PosteriorDependence,
    SelectionDiagramDeclaration,
};

/// Version of the transport route's identity encoding.
pub const MECHANISM_TRANSPORT_VERSION: u32 = 2;

/// A carried mechanism is not a fitted one.
pub const DETAIL_ORIGIN_NOT_FITTED: &str = "mechanism_transport.origin_not_fitted";
/// A carried mechanism is not additive-noise linear-Gaussian.
pub const DETAIL_FAMILY_UNSUPPORTED: &str = "mechanism_transport.family_unsupported";
/// The diagram does not describe the source population of a carried mechanism.
pub const DETAIL_DIAGRAM_SOURCE_MISMATCH: &str = "mechanism_transport.diagram_source_mismatch";
/// The diagram does not describe the request's target population.
pub const DETAIL_DIAGRAM_TARGET_MISMATCH: &str = "mechanism_transport.diagram_target_mismatch";
/// A supplied target mechanism was not fitted in the target population.
pub const DETAIL_TARGET_NOT_NATIVE: &str = "mechanism_transport.target_mechanism_not_native";
/// A variable has more than one mechanism in one role.
pub const DETAIL_DUPLICATE_MECHANISM: &str = "mechanism_transport.duplicate_mechanism";
/// A selection node points into a latent cause of a carried mechanism.
pub const DETAIL_SELECTION_INTO_LATENT: &str =
    crate::compose::coherence::DETAIL_SELECTION_INTO_LATENT;
/// A selection node points into a carried mechanism's variable (the literal
/// of the composition checker, so one defect has one name).
pub const DETAIL_SELECTION_INTO_MECHANISM: &str =
    "mechanism_composition.population_selection_arrow_into_mechanism";

const TAG_IDENTITY: &str = "antecedent.model.mechanism_transport";

/// Everything a transported composition is built from.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismTransportRequest {
    /// The population and regime the composition is about.
    pub target: MechanismContext,
    /// The selection diagram from the source population to the target.
    pub diagram: SelectionDiagramDeclaration,
    /// Mechanisms fitted in the source population.
    pub source_mechanisms: Vec<MechanismDescriptor>,
    /// Mechanisms fitted in the target population itself.
    pub target_mechanisms: Vec<MechanismDescriptor>,
    /// Declared causal graph as `(parent, child)`.
    pub edges: Vec<(String, String)>,
    /// Explicit analyst invariance declarations (required for a carried
    /// mechanism that loads on a shared latent cause, or to carry across
    /// regimes).
    pub invariances: Vec<InvarianceDeclaration>,
    /// Declared dependence between the mechanisms' parameter estimates.
    pub dependence: PosteriorDependence,
}

/// How one component of the transported composition entered it.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MechanismRole {
    /// Fitted in the source population and admitted by the selection diagram.
    Transported,
    /// Fitted in the target population.
    NativeInTarget,
}

impl MechanismRole {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Transported => "transported",
            Self::NativeInTarget => "native_in_target",
        }
    }
}

/// One component of the transported composition and where it came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismDisposition {
    /// The variable the mechanism produces.
    pub variable: String,
    /// Carried or refitted.
    pub role: MechanismRole,
    /// Population the mechanism was fitted in.
    pub fitted_population: String,
    /// Owning fit.
    pub artifact_id: String,
    /// Evidence the fit used.
    pub evidence_id: String,
}

/// Scope label of the parameter uncertainty a transported contrast carries.
pub const UNCERTAINTY_SCOPE: &str = "fitted_parameters_only_transport_assumption_not_propagated";

/// Parameter uncertainty of a contrast of a transported composition.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedFunctionalUncertainty {
    /// First-order propagation of the declared parameter law.
    pub functional: FunctionalUncertainty,
    /// What the variance does and does not contain ([`UNCERTAINTY_SCOPE`]).
    pub scope: &'static str,
}

/// An executable target-population composition with its transport record.
#[derive(Clone, Debug)]
pub struct TransportedLinearGaussian {
    model: ComposedLinearGaussian,
    dispositions: Vec<MechanismDisposition>,
    identity: [u8; 32],
}

impl TransportedLinearGaussian {
    /// The executable joint model (carried mechanisms keep their source
    /// context in its specification).
    #[must_use]
    pub fn model(&self) -> &ComposedLinearGaussian {
        &self.model
    }

    /// Every component, sorted by variable.
    #[must_use]
    pub fn dispositions(&self) -> &[MechanismDisposition] {
        &self.dispositions
    }

    /// Variables whose mechanism was carried from the source, sorted.
    #[must_use]
    pub fn transported_variables(&self) -> Vec<&str> {
        self.dispositions
            .iter()
            .filter(|d| d.role == MechanismRole::Transported)
            .map(|d| d.variable.as_str())
            .collect()
    }

    /// Identity of the transported composition: the target, the diagram, every
    /// mechanism's numbers and role, the declared graph, invariances and
    /// dependence. Equal requests give equal identities; a changed number at
    /// an unchanged shape changes it.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Lowercase hexadecimal [`Self::identity`].
    #[must_use]
    pub fn identity_hex(&self) -> String {
        antecedent_core::identity::to_hex(&self.identity)
    }

    /// `sum_k weight_k * E[outcome | do(arm_k)]` in the target composition.
    ///
    /// # Errors
    /// As [`ComposedLinearGaussian::mean_contrast`].
    pub fn mean_contrast(&self, contrast: &MeanContrast) -> Result<f64, CompositionError> {
        self.model.mean_contrast(contrast)
    }

    /// First-order parameter uncertainty of a contrast, scoped honestly: the
    /// covariance is the source fit's for a carried mechanism, and the
    /// selection-diagram assumption is not a variance term.
    ///
    /// # Errors
    /// As [`ComposedLinearGaussian::mean_contrast_uncertainty`]: a missing
    /// marginal law or unknown dependence is a refusal, never a default.
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
    ) -> Result<TransportedFunctionalUncertainty, CompositionError> {
        Ok(TransportedFunctionalUncertainty {
            functional: self.model.mean_contrast_uncertainty(contrast)?,
            scope: UNCERTAINTY_SCOPE,
        })
    }
}

/// Variables the diagram marks with a selection arrow (a node pointing into
/// them), sorted and distinct. A latent cause may appear here too.
#[must_use]
pub fn selection_targets(diagram: &SelectionDiagramDeclaration) -> Vec<String> {
    let set: BTreeSet<&str> = diagram
        .selection_nodes
        .iter()
        .flat_map(|n| n.points_into.iter().map(String::as_str))
        .collect();
    set.into_iter().map(str::to_owned).collect()
}

fn finding(
    kind: CoherenceKind,
    detail: &'static str,
    subject: &str,
    remedy: &'static str,
) -> CoherenceFinding {
    CoherenceFinding::new(kind, detail, subject.to_owned(), remedy)
}

/// Carry fitted linear-Gaussian mechanisms into the target population's
/// composed model, or refuse with every finding.
///
/// # Errors
/// [`CompositionError::Incoherent`] with every defect: a carried mechanism
/// that is not fitted or not linear-Gaussian, a diagram for other populations,
/// a selection arrow into a carried mechanism or its latent, a variable with
/// two mechanisms in one role, and everything the composition checker finds in
/// the assembled specification (units, support, parent sets, latents,
/// dependence, regime). A variable with a selection arrow into it and no
/// natively refitted target mechanism is refused; refit it in the target.
pub fn transport_linear_gaussian(
    request: &MechanismTransportRequest,
) -> Result<TransportedLinearGaussian, CompositionError> {
    let targets: BTreeSet<String> = selection_targets(&request.diagram).into_iter().collect();
    let mut findings: Vec<CoherenceFinding> = Vec::new();

    if request.diagram.target_population != request.target.population {
        findings.push(
            finding(
                CoherenceKind::Population,
                DETAIL_DIAGRAM_TARGET_MISMATCH,
                "selection_diagram",
                "declare the selection diagram between the source population and this target population",
            )
            .comparing(
                request.target.population.clone(),
                request.diagram.target_population.clone(),
            ),
        );
    }

    let mut native: BTreeMap<String, &MechanismDescriptor> = BTreeMap::new();
    for d in &request.target_mechanisms {
        if d.context.population != request.target.population {
            findings.push(
                finding(
                    CoherenceKind::Population,
                    DETAIL_TARGET_NOT_NATIVE,
                    &d.variable,
                    "supply as a target mechanism only a mechanism fitted in the target population; carry the others as source mechanisms",
                )
                .comparing(request.target.population.clone(), d.context.population.clone()),
            );
        }
        if native.insert(d.variable.clone(), d).is_some() {
            findings.push(finding(
                CoherenceKind::VariableIdentity,
                DETAIL_DUPLICATE_MECHANISM,
                &d.variable,
                "give each variable exactly one target mechanism",
            ));
        }
    }

    let mut carried: BTreeMap<String, &MechanismDescriptor> = BTreeMap::new();
    for d in &request.source_mechanisms {
        if carried.insert(d.variable.clone(), d).is_some() {
            findings.push(finding(
                CoherenceKind::VariableIdentity,
                DETAIL_DUPLICATE_MECHANISM,
                &d.variable,
                "give each variable exactly one source mechanism",
            ));
        }
    }

    let mut chosen: Vec<MechanismDescriptor> = Vec::new();
    let mut dispositions: Vec<MechanismDisposition> = Vec::new();
    let variables: BTreeSet<&String> = native.keys().chain(carried.keys()).collect();
    for variable in variables {
        let (descriptor, role) = match (native.get(variable), carried.get(variable)) {
            (Some(d), _) => (*d, MechanismRole::NativeInTarget),
            (None, Some(d)) => (*d, MechanismRole::Transported),
            (None, None) => continue,
        };
        if role == MechanismRole::Transported {
            check_carried(request, &targets, descriptor, &mut findings);
        } else if descriptor.form.family() != MechanismFamily::LinearGaussian {
            findings.push(finding(
                CoherenceKind::Family,
                DETAIL_FAMILY_UNSUPPORTED,
                &descriptor.variable,
                "this route executes additive-noise linear-Gaussian mechanisms only; decide a finite table with the finite counterfactual transport derivation",
            ));
        }
        dispositions.push(MechanismDisposition {
            variable: variable.clone(),
            role,
            fitted_population: descriptor.context.population.clone(),
            artifact_id: descriptor.provenance.artifact_id.clone(),
            evidence_id: descriptor.provenance.evidence_id.clone(),
        });
        chosen.push(descriptor.clone());
    }

    if !findings.is_empty() {
        findings.sort_by(|a, b| (a.kind, a.detail, &a.subject).cmp(&(b.kind, b.detail, &b.subject)));
        findings.dedup();
        return Err(CompositionError::Incoherent { findings });
    }

    let spec = ComposedModelSpec {
        target: request.target.clone(),
        mechanisms: chosen,
        edges: request.edges.clone(),
        invariances: request.invariances.clone(),
        selection_diagrams: vec![request.diagram.clone()],
        dependence: request.dependence.clone(),
        admit_declared_mechanisms: false,
    };
    let model = ComposedLinearGaussian::compile(&spec)?;
    let identity = identity_of(&spec, &dispositions);
    Ok(TransportedLinearGaussian { model, dispositions, identity })
}

fn check_carried(
    request: &MechanismTransportRequest,
    targets: &BTreeSet<String>,
    d: &MechanismDescriptor,
    out: &mut Vec<CoherenceFinding>,
) {
    if d.provenance.origin != MechanismOrigin::Fitted {
        out.push(finding(
            CoherenceKind::Provenance,
            DETAIL_ORIGIN_NOT_FITTED,
            &d.variable,
            "only a mechanism estimated from retained evidence is carried; an external or declared mechanism needs its own admission",
        ));
    }
    if d.form.family() != MechanismFamily::LinearGaussian {
        out.push(finding(
            CoherenceKind::Family,
            DETAIL_FAMILY_UNSUPPORTED,
            &d.variable,
            "carry only additive-noise linear-Gaussian mechanisms here; decide a finite table with the finite counterfactual transport derivation",
        ));
    }
    if d.context.population != request.diagram.source_population {
        out.push(
            finding(
                CoherenceKind::Population,
                DETAIL_DIAGRAM_SOURCE_MISMATCH,
                &d.variable,
                "declare the selection diagram between the population this mechanism was fitted in and the target",
            )
            .comparing(
                request.diagram.source_population.clone(),
                d.context.population.clone(),
            ),
        );
    }
    if targets.contains(&d.variable) {
        out.push(
            finding(
                CoherenceKind::Population,
                DETAIL_SELECTION_INTO_MECHANISM,
                &d.variable,
                "a selection node points into this variable, so its mechanism is not invariant; refit it in the target population",
            )
            .comparing(
                request.target.population.clone(),
                d.context.population.clone(),
            ),
        );
    }
    for latent in &d.latents {
        if targets.contains(&latent.latent) {
            out.push(finding(
                CoherenceKind::Population,
                DETAIL_SELECTION_INTO_LATENT,
                &format!("{}:{}", latent.latent, d.variable),
                "a selection node points into a latent cause this mechanism loads on, so the latent's law differs in the target; refit the mechanism in the target population",
            ));
        }
    }
}

fn write_domain(w: &mut CanonicalWriter, domain: &Domain) {
    match domain {
        Domain::Real => {
            w.word(0);
        }
        Domain::Interval { lo, hi } => {
            w.word(1).real(*lo).real(*hi);
        }
        Domain::Levels(levels) => {
            w.word(2).texts(levels);
        }
    }
}

fn identity_of(spec: &ComposedModelSpec, dispositions: &[MechanismDisposition]) -> [u8; 32] {
    let spec = spec.canonical();
    let mut w = CanonicalWriter::new(TAG_IDENTITY);
    w.word(u64::from(MECHANISM_TRANSPORT_VERSION));
    w.text(&spec.target.population).text(&spec.target.regime);
    w.word(spec.mechanisms.len() as u64);
    for m in &spec.mechanisms {
        w.text(&m.variable);
        let role = dispositions
            .iter()
            .find(|d| d.variable == m.variable)
            .map_or(0, |d| match d.role {
                MechanismRole::Transported => 1,
                MechanismRole::NativeInTarget => 2,
            });
        w.word(role);
        match &m.form {
            MechanismForm::LinearGaussian { intercept, sigma, coefficients } => {
                w.word(1).real(*intercept).real(*sigma).word(coefficients.len() as u64);
                for (name, coefficient) in coefficients {
                    w.text(name).real(*coefficient);
                }
            }
            MechanismForm::Tabular { levels, rows } => {
                w.word(2).texts(levels).word(rows.len() as u64);
                for row in rows {
                    w.word(row.parent_levels.len() as u64);
                    for (parent, level) in &row.parent_levels {
                        w.text(parent).text(level);
                    }
                    w.word(row.probabilities.len() as u64);
                    for (level, probability) in &row.probabilities {
                        w.text(level).real(*probability);
                    }
                }
            }
        }
        w.text(&m.output_units).text(&m.output_transform);
        write_domain(&mut w, &m.output_support);
        w.texts(&m.conditioning_set);
        match &m.uncertainty {
            None => {
                w.word(0);
            }
            Some(u) => {
                w.word(1).word(u.meaning as u64).word(u.parameters.len() as u64);
                for parameter in &u.parameters {
                    match parameter {
                        ParameterTerm::Intercept => {
                            w.word(0);
                        }
                        ParameterTerm::Coefficient(name) => {
                            w.word(1).text(name);
                        }
                    }
                }
                w.reals(&u.covariance);
            }
        }
        w.text(&m.context.population).text(&m.context.regime);
        w.text(&m.provenance.artifact_id).text(&m.provenance.evidence_id);
        w.word(m.parents.len() as u64);
        for p in &m.parents {
            w.text(&p.variable).text(&p.units).text(&p.transform);
            write_domain(&mut w, &p.domain);
        }
        w.word(m.latents.len() as u64);
        for l in &m.latents {
            w.text(&l.latent).opt_real(l.loading).texts(&l.shared_with);
        }
    }
    w.word(spec.edges.len() as u64);
    for (parent, child) in &spec.edges {
        w.text(parent).text(child);
    }
    w.word(spec.invariances.len() as u64);
    for i in &spec.invariances {
        w.text(&i.variable).word(i.coordinate as u64).text(&i.source).text(&i.target);
        w.text(&i.assumption_id);
    }
    w.word(spec.selection_diagrams.len() as u64);
    for diagram in &spec.selection_diagrams {
        w.text(&diagram.source_population).text(&diagram.target_population);
        w.word(diagram.selection_nodes.len() as u64);
        for node in &diagram.selection_nodes {
            w.text(&node.name).texts(&node.points_into);
        }
    }
    match &spec.dependence {
        PosteriorDependence::Unknown => {
            w.word(0);
        }
        PosteriorDependence::IndependentByDesign { justification } => {
            w.word(1).text(justification);
        }
        PosteriorDependence::Joint { covariance, parameters, .. } => {
            w.word(2).word(parameters.len() as u64).reals(covariance);
        }
        PosteriorDependence::CrossCovariance { blocks } => {
            w.word(3).word(blocks.len() as u64);
            for b in blocks {
                w.text(&b.first).text(&b.second).reals(&b.values);
            }
        }
    }
    w.finish()
}
