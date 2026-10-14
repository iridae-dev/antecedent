//! Portable form of a discretized composition and its fresh-consumer re-derivation.
//!
//! A [`DiscretizedLinearGaussian`] is a deterministic function of its source
//! composition's premises (mechanism numbers, graph, context, provenance, declared
//! dependence) and its [`DiscretizationPlan`]. The artifact therefore carries those
//! premises and the plan, plus the numbers the producer claims (the identity, and
//! per variable the cell edges, cell values and mean-error diagnostic). A consumer
//! never trusts a claimed number: it rebuilds the discretization from the carried
//! premises with the current code and requires every claimed number to match the
//! re-derived one bit for bit.
//!
//! # Wire form
//!
//! A length-delimited, little-endian byte string: a text tag, a `u32` version
//! ([`DISCRETIZED_SCM_ARTIFACT_VERSION`]), then fixed-order sections. Counts are
//! checked against the bytes remaining before anything is allocated, text is UTF-8
//! with a `u64` length, reals are their IEEE-754 bits (so an infinite interval end
//! or a signed zero survives), and trailing bytes are refused. Never `Debug` output
//! and never a text format.
//!
//! # What a consumer learns
//!
//! The point value, its computed mean-error bound and the point-only inference label
//! are all re-derived; nothing about the discretized value is inherited from the
//! producer. The artifact carries no interval, posterior or calibration, because the
//! bridge has none to carry. Parameter uncertainty declared on the source mechanisms
//! travels with the premises and is still not propagated through the discretization.
//!
//! # Boundary
//!
//! Only additive-noise linear-Gaussian sources are exportable (the only family the
//! bridge accepts). An edit of a premise changes the re-derived identity; an edit of
//! a claimed number, or a resealed identity over edited premises, fails the
//! comparison. A consumer that holds an expected identity passes it so a
//! self-consistent artifact for a different composition is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;

use crate::compose::{
    ComposedModelSpec, ContextCoordinate, CrossCovarianceBlock, Domain, InvarianceDeclaration,
    LatentDependence, MechanismContext, MechanismDescriptor, MechanismForm, MechanismOrigin,
    MechanismProvenance, MechanismUncertainty, ParameterId, ParameterTerm, ParentBinding,
    PosteriorDependence, SelectionDiagramDeclaration, SelectionNodeDeclaration, UncertaintyMeaning,
};
use crate::discretized_scm::{
    BridgeRefusal, DiscretizationMethod, DiscretizationPlan, DiscretizedLinearGaussian,
};

/// Version of the artifact's wire form and of the claimed-number set.
pub const DISCRETIZED_SCM_ARTIFACT_VERSION: u32 = 1;

/// The bytes are not a well-formed artifact (truncated, trailing bytes, invalid
/// text, a count larger than the bytes remaining, an unknown tag).
pub const DETAIL_MALFORMED: &str = "discretized_scm_artifact.malformed";
/// The artifact's version is not one this build reads.
pub const DETAIL_VERSION_UNSUPPORTED: &str = "discretized_scm_artifact.version_unsupported";
/// The source composition is not exportable (not additive-noise linear-Gaussian).
pub const DETAIL_NOT_EXPORTABLE: &str = "discretized_scm_artifact.not_exportable";
/// The carried premises no longer discretize (the composition or plan is refused).
pub const DETAIL_REDERIVATION_REFUSED: &str = "discretized_scm_artifact.rederivation_refused";
/// The re-derived identity is not the identity the artifact claims or the consumer
/// expects.
pub const DETAIL_IDENTITY_MISMATCH: &str = "discretized_scm_artifact.identity_mismatch";
/// A re-derived cell edge, cell value or mean error differs from the claimed one.
pub const DETAIL_CLAIM_MISMATCH: &str = "discretized_scm_artifact.claimed_numbers_differ";

const TAG: &str = "antecedent.model.discretized_scm_artifact";
/// Fewest bytes any section can occupy; bounds counts before allocation.
const MIN_ITEM_BYTES: usize = 1;

/// A typed refusal of export or of consumption.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactRefusal {
    /// Registered reason code.
    pub reason_code: &'static str,
    /// Typed detail literal.
    pub detail: &'static str,
    /// What the refusal is about (a section, a variable, a detail of the inner refusal).
    pub subject: String,
    /// What would license the request.
    pub remedy: &'static str,
}

impl ArtifactRefusal {
    fn new(
        reason_code: &'static str,
        detail: &'static str,
        subject: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { reason_code, detail, subject: subject.into(), remedy }
    }

    fn malformed(subject: &'static str) -> Self {
        Self::new(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_MALFORMED,
            subject,
            "re-export the artifact from the producing composition",
        )
    }
}

impl std::fmt::Display for ArtifactRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.reason_code, self.detail, self.subject)
    }
}

impl std::error::Error for ArtifactRefusal {}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn tag(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn word(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn count(&mut self, value: usize) {
        self.word(value as u64);
    }

    fn real(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn text(&mut self, value: &str) {
        self.count(value.len());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn texts(&mut self, values: &[String]) {
        self.count(values.len());
        for v in values {
            self.text(v);
        }
    }

    fn reals(&mut self, values: &[f64]) {
        self.count(values.len());
        for v in values {
            self.real(*v);
        }
    }

    fn opt_real(&mut self, value: Option<f64>) {
        match value {
            None => self.tag(0),
            Some(v) => {
                self.tag(1);
                self.real(v);
            }
        }
    }

    fn domain(&mut self, domain: &Domain) {
        match domain {
            Domain::Real => self.tag(0),
            Domain::Interval { lo, hi } => {
                self.tag(1);
                self.real(*lo);
                self.real(*hi);
            }
            Domain::Levels(levels) => {
                self.tag(2);
                self.texts(levels);
            }
        }
    }

    fn term(&mut self, term: &ParameterTerm) {
        match term {
            ParameterTerm::Intercept => self.tag(0),
            ParameterTerm::Coefficient(name) => {
                self.tag(1);
                self.text(name);
            }
        }
    }

    fn terms(&mut self, terms: &[ParameterTerm]) {
        self.count(terms.len());
        for t in terms {
            self.term(t);
        }
    }

    fn meaning(&mut self, meaning: UncertaintyMeaning) {
        self.tag(match meaning {
            UncertaintyMeaning::Posterior => 0,
            UncertaintyMeaning::SamplingDistribution => 1,
        });
    }
}

fn write_mechanism(w: &mut Writer, d: &MechanismDescriptor) -> Result<(), ArtifactRefusal> {
    w.text(&d.variable);
    match &d.form {
        MechanismForm::LinearGaussian { intercept, sigma, coefficients } => {
            w.tag(1);
            w.real(*intercept);
            w.real(*sigma);
            w.count(coefficients.len());
            for (name, coefficient) in coefficients {
                w.text(name);
                w.real(*coefficient);
            }
        }
        MechanismForm::Tabular { .. } => {
            return Err(ArtifactRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                DETAIL_NOT_EXPORTABLE,
                d.variable.clone(),
                "only an additive-noise linear-Gaussian composition is discretized, so only it is exported",
            ));
        }
    }
    w.count(d.parents.len());
    for p in &d.parents {
        w.text(&p.variable);
        w.text(&p.units);
        w.text(&p.transform);
        w.domain(&p.domain);
    }
    w.texts(&d.conditioning_set);
    w.text(&d.output_units);
    w.text(&d.output_transform);
    w.domain(&d.output_support);
    w.text(&d.context.population);
    w.text(&d.context.regime);
    w.tag(match d.provenance.origin {
        MechanismOrigin::Fitted => 0,
        MechanismOrigin::ExternalBinding => 1,
        MechanismOrigin::Declared => 2,
    });
    w.text(&d.provenance.artifact_id);
    w.text(&d.provenance.evidence_id);
    w.count(d.latents.len());
    for l in &d.latents {
        w.text(&l.latent);
        w.opt_real(l.loading);
        w.texts(&l.shared_with);
    }
    match &d.uncertainty {
        None => w.tag(0),
        Some(u) => {
            w.tag(1);
            w.meaning(u.meaning);
            w.terms(&u.parameters);
            w.reals(&u.covariance);
        }
    }
    Ok(())
}

fn write_dependence(w: &mut Writer, dependence: &PosteriorDependence) {
    match dependence {
        PosteriorDependence::Unknown => w.tag(0),
        PosteriorDependence::IndependentByDesign { justification } => {
            w.tag(1);
            w.text(justification);
        }
        PosteriorDependence::Joint { meaning, parameters, covariance } => {
            w.tag(2);
            w.meaning(*meaning);
            w.count(parameters.len());
            for p in parameters {
                w.text(&p.mechanism);
                w.term(&p.term);
            }
            w.reals(covariance);
        }
        PosteriorDependence::CrossCovariance { blocks } => {
            w.tag(3);
            w.count(blocks.len());
            for b in blocks {
                w.text(&b.first);
                w.text(&b.second);
                w.terms(&b.first_parameters);
                w.terms(&b.second_parameters);
                w.reals(&b.values);
            }
        }
        // `PosteriorDependence` may grow; an unknown form is never exported as
        // `Unknown`, which would silently weaken a declaration.
        #[allow(unreachable_patterns)]
        _ => w.tag(255),
    }
}

/// Serialize a discretization: its source premises, its plan and the numbers it
/// claims.
///
/// # Errors
/// [`ArtifactRefusal`] with [`DETAIL_NOT_EXPORTABLE`] when a source mechanism is
/// not additive-noise linear-Gaussian or the declared dependence is a form this
/// version cannot carry.
pub fn export_discretized_scm(
    model: &DiscretizedLinearGaussian,
) -> Result<Vec<u8>, ArtifactRefusal> {
    let spec = model.source_model().spec();
    let mut w = Writer::default();
    w.text(TAG);
    w.bytes.extend_from_slice(&DISCRETIZED_SCM_ARTIFACT_VERSION.to_le_bytes());

    // Premises.
    w.text(&spec.target.population);
    w.text(&spec.target.regime);
    w.count(spec.mechanisms.len());
    for d in &spec.mechanisms {
        write_mechanism(&mut w, d)?;
    }
    w.count(spec.edges.len());
    for (parent, child) in &spec.edges {
        w.text(parent);
        w.text(child);
    }
    w.count(spec.invariances.len());
    for i in &spec.invariances {
        w.text(&i.variable);
        w.tag(match i.coordinate {
            ContextCoordinate::Population => 0,
            ContextCoordinate::Regime => 1,
        });
        w.text(&i.source);
        w.text(&i.target);
        w.text(&i.assumption_id);
    }
    w.count(spec.selection_diagrams.len());
    for diagram in &spec.selection_diagrams {
        w.text(&diagram.source_population);
        w.text(&diagram.target_population);
        w.count(diagram.selection_nodes.len());
        for node in &diagram.selection_nodes {
            w.text(&node.name);
            w.texts(&node.points_into);
        }
    }
    write_dependence(&mut w, &spec.dependence);
    w.tag(u8::from(spec.admit_declared_mechanisms));

    // Plan.
    w.tag(match model.method() {
        DiscretizationMethod::EqualProbabilityQuantile => 1,
        #[allow(unreachable_patterns)]
        _ => 255,
    });
    let variables = model.variables();
    w.count(variables.len());
    for v in &variables {
        let levels = model.level_names(v).map_or(0, <[String]>::len);
        w.text(v);
        w.count(levels);
    }

    // Claims.
    w.bytes.extend_from_slice(&model.identity());
    for v in &variables {
        w.reals(model.cell_edges(v).unwrap_or(&[]));
        w.reals(model.level_values(v).unwrap_or(&[]));
        w.real(model.mechanism_mean_error(v).unwrap_or(f64::NAN));
    }
    Ok(w.bytes)
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

type Decoded<T> = Result<T, ArtifactRefusal>;

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Decoded<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.bytes.len());
        let Some(end) = end else {
            return Err(ArtifactRefusal::malformed(what));
        };
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn tag(&mut self, what: &'static str) -> Decoded<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn word(&mut self, what: &'static str) -> Decoded<u64> {
        let b = self.take(8, what)?;
        Ok(u64::from_le_bytes(b.try_into().map_err(|_| ArtifactRefusal::malformed(what))?))
    }

    fn real(&mut self, what: &'static str) -> Decoded<f64> {
        Ok(f64::from_bits(self.word(what)?))
    }

    /// A count that cannot exceed what the remaining bytes could hold.
    fn count(&mut self, what: &'static str) -> Decoded<usize> {
        let n = self.word(what)?;
        let remaining = self.bytes.len() - self.at;
        match usize::try_from(n) {
            Ok(n) if n.saturating_mul(MIN_ITEM_BYTES) <= remaining => Ok(n),
            _ => Err(ArtifactRefusal::malformed(what)),
        }
    }

    fn text(&mut self, what: &'static str) -> Decoded<String> {
        let n = self.count(what)?;
        let raw = self.take(n, what)?;
        String::from_utf8(raw.to_vec()).map_err(|_| ArtifactRefusal::malformed(what))
    }

    fn texts(&mut self, what: &'static str) -> Decoded<Vec<String>> {
        let n = self.count(what)?;
        (0..n).map(|_| self.text(what)).collect()
    }

    fn reals(&mut self, what: &'static str) -> Decoded<Vec<f64>> {
        let n = self.count(what)?;
        (0..n).map(|_| self.real(what)).collect()
    }

    fn opt_real(&mut self, what: &'static str) -> Decoded<Option<f64>> {
        match self.tag(what)? {
            0 => Ok(None),
            1 => Ok(Some(self.real(what)?)),
            _ => Err(ArtifactRefusal::malformed(what)),
        }
    }

    fn domain(&mut self) -> Decoded<Domain> {
        match self.tag("domain")? {
            0 => Ok(Domain::Real),
            1 => Ok(Domain::Interval { lo: self.real("domain")?, hi: self.real("domain")? }),
            2 => Ok(Domain::Levels(self.texts("domain")?)),
            _ => Err(ArtifactRefusal::malformed("domain")),
        }
    }

    fn term(&mut self) -> Decoded<ParameterTerm> {
        match self.tag("parameter")? {
            0 => Ok(ParameterTerm::Intercept),
            1 => Ok(ParameterTerm::Coefficient(self.text("parameter")?)),
            _ => Err(ArtifactRefusal::malformed("parameter")),
        }
    }

    fn terms(&mut self) -> Decoded<Vec<ParameterTerm>> {
        let n = self.count("parameters")?;
        (0..n).map(|_| self.term()).collect()
    }

    fn meaning(&mut self) -> Decoded<UncertaintyMeaning> {
        match self.tag("meaning")? {
            0 => Ok(UncertaintyMeaning::Posterior),
            1 => Ok(UncertaintyMeaning::SamplingDistribution),
            _ => Err(ArtifactRefusal::malformed("meaning")),
        }
    }
}

fn read_mechanism(r: &mut Reader<'_>) -> Decoded<MechanismDescriptor> {
    let variable = r.text("mechanism")?;
    let form = match r.tag("form")? {
        1 => {
            let intercept = r.real("form")?;
            let sigma = r.real("form")?;
            let n = r.count("coefficients")?;
            let mut coefficients = Vec::with_capacity(n);
            for _ in 0..n {
                coefficients.push((r.text("coefficients")?, r.real("coefficients")?));
            }
            MechanismForm::LinearGaussian { intercept, sigma, coefficients }
        }
        _ => return Err(ArtifactRefusal::malformed("form")),
    };
    let n = r.count("parents")?;
    let mut parents = Vec::with_capacity(n);
    for _ in 0..n {
        parents.push(ParentBinding {
            variable: r.text("parents")?,
            units: r.text("parents")?,
            transform: r.text("parents")?,
            domain: r.domain()?,
        });
    }
    let conditioning_set = r.texts("conditioning_set")?;
    let output_units = r.text("output_units")?;
    let output_transform = r.text("output_transform")?;
    let output_support = r.domain()?;
    let context = MechanismContext { population: r.text("context")?, regime: r.text("context")? };
    let origin = match r.tag("origin")? {
        0 => MechanismOrigin::Fitted,
        1 => MechanismOrigin::ExternalBinding,
        2 => MechanismOrigin::Declared,
        _ => return Err(ArtifactRefusal::malformed("origin")),
    };
    let provenance = MechanismProvenance {
        origin,
        artifact_id: r.text("provenance")?,
        evidence_id: r.text("provenance")?,
    };
    let n = r.count("latents")?;
    let mut latents = Vec::with_capacity(n);
    for _ in 0..n {
        latents.push(LatentDependence {
            latent: r.text("latents")?,
            loading: r.opt_real("latents")?,
            shared_with: r.texts("latents")?,
        });
    }
    let uncertainty = match r.tag("uncertainty")? {
        0 => None,
        1 => Some(MechanismUncertainty {
            meaning: r.meaning()?,
            parameters: r.terms()?,
            covariance: r.reals("uncertainty")?,
        }),
        _ => return Err(ArtifactRefusal::malformed("uncertainty")),
    };
    Ok(MechanismDescriptor {
        variable,
        form,
        parents,
        conditioning_set,
        output_units,
        output_transform,
        output_support,
        context,
        provenance,
        latents,
        uncertainty,
    })
}

fn read_dependence(r: &mut Reader<'_>) -> Decoded<PosteriorDependence> {
    match r.tag("dependence")? {
        0 => Ok(PosteriorDependence::Unknown),
        1 => Ok(PosteriorDependence::IndependentByDesign { justification: r.text("dependence")? }),
        2 => {
            let meaning = r.meaning()?;
            let n = r.count("dependence")?;
            let mut parameters = Vec::with_capacity(n);
            for _ in 0..n {
                parameters.push(ParameterId { mechanism: r.text("dependence")?, term: r.term()? });
            }
            Ok(PosteriorDependence::Joint {
                meaning,
                parameters,
                covariance: r.reals("dependence")?,
            })
        }
        3 => {
            let n = r.count("dependence")?;
            let mut blocks = Vec::with_capacity(n);
            for _ in 0..n {
                blocks.push(CrossCovarianceBlock {
                    first: r.text("dependence")?,
                    second: r.text("dependence")?,
                    first_parameters: r.terms()?,
                    second_parameters: r.terms()?,
                    values: r.reals("dependence")?,
                });
            }
            Ok(PosteriorDependence::CrossCovariance { blocks })
        }
        _ => Err(ArtifactRefusal::malformed("dependence")),
    }
}

struct Claimed {
    variable: String,
    edges: Vec<f64>,
    values: Vec<f64>,
    mean_error: f64,
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Rebuild a discretization from an artifact, re-deriving every number.
///
/// `expected_identity` is the identity the consumer already holds for the
/// composition it expects, when it has one.
///
/// # Errors
/// [`ArtifactRefusal`] for malformed bytes, an unknown version
/// ([`DETAIL_VERSION_UNSUPPORTED`]), premises that no longer discretize
/// ([`DETAIL_REDERIVATION_REFUSED`], naming the inner refusal), an identity that
/// differs from the claimed or expected one ([`DETAIL_IDENTITY_MISMATCH`]), or a
/// claimed number that differs from the re-derived one ([`DETAIL_CLAIM_MISMATCH`]).
pub fn consume_discretized_scm(
    bytes: &[u8],
    expected_identity: Option<[u8; 32]>,
    ctx: &ExecutionContext,
) -> Result<DiscretizedLinearGaussian, ArtifactRefusal> {
    let mut r = Reader { bytes, at: 0 };
    if r.text("tag")? != TAG {
        return Err(ArtifactRefusal::malformed("tag"));
    }
    let version = u32::from_le_bytes(
        r.take(4, "version")?.try_into().map_err(|_| ArtifactRefusal::malformed("version"))?,
    );
    if version != DISCRETIZED_SCM_ARTIFACT_VERSION {
        return Err(ArtifactRefusal::new(
            antecedent_core::reason_code!("schema_mismatch"),
            DETAIL_VERSION_UNSUPPORTED,
            format!("version={version}"),
            "re-export the artifact with a producer this build reads",
        ));
    }

    let target = MechanismContext { population: r.text("target")?, regime: r.text("target")? };
    let n = r.count("mechanisms")?;
    let mut mechanisms = Vec::with_capacity(n);
    for _ in 0..n {
        mechanisms.push(read_mechanism(&mut r)?);
    }
    let n = r.count("edges")?;
    let mut edges = Vec::with_capacity(n);
    for _ in 0..n {
        edges.push((r.text("edges")?, r.text("edges")?));
    }
    let n = r.count("invariances")?;
    let mut invariances = Vec::with_capacity(n);
    for _ in 0..n {
        let variable = r.text("invariances")?;
        let coordinate = match r.tag("invariances")? {
            0 => ContextCoordinate::Population,
            1 => ContextCoordinate::Regime,
            _ => return Err(ArtifactRefusal::malformed("invariances")),
        };
        invariances.push(InvarianceDeclaration {
            variable,
            coordinate,
            source: r.text("invariances")?,
            target: r.text("invariances")?,
            assumption_id: r.text("invariances")?,
        });
    }
    let n = r.count("selection_diagrams")?;
    let mut selection_diagrams = Vec::with_capacity(n);
    for _ in 0..n {
        let source_population = r.text("selection_diagrams")?;
        let target_population = r.text("selection_diagrams")?;
        let k = r.count("selection_nodes")?;
        let mut selection_nodes = Vec::with_capacity(k);
        for _ in 0..k {
            selection_nodes.push(SelectionNodeDeclaration {
                name: r.text("selection_nodes")?,
                points_into: r.texts("selection_nodes")?,
            });
        }
        selection_diagrams.push(SelectionDiagramDeclaration {
            source_population,
            target_population,
            selection_nodes,
        });
    }
    let dependence = read_dependence(&mut r)?;
    let admit_declared_mechanisms = match r.tag("admit_declared")? {
        0 => false,
        1 => true,
        _ => return Err(ArtifactRefusal::malformed("admit_declared")),
    };
    let spec = ComposedModelSpec {
        target,
        mechanisms,
        edges,
        invariances,
        selection_diagrams,
        dependence,
        admit_declared_mechanisms,
    };

    let method = match r.tag("method")? {
        1 => DiscretizationMethod::EqualProbabilityQuantile,
        _ => return Err(ArtifactRefusal::malformed("method")),
    };
    let n = r.count("levels")?;
    let mut levels = BTreeMap::new();
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let name = r.text("levels")?;
        order.push(name.clone());
        let count =
            usize::try_from(r.word("levels")?).map_err(|_| ArtifactRefusal::malformed("levels"))?;
        if levels.insert(name, count).is_some() {
            return Err(ArtifactRefusal::malformed("levels"));
        }
    }
    let plan = DiscretizationPlan { method, levels };

    let claimed_identity: [u8; 32] =
        r.take(32, "identity")?.try_into().map_err(|_| ArtifactRefusal::malformed("identity"))?;
    let mut claimed = Vec::with_capacity(plan.levels.len());
    for variable in &order {
        claimed.push(Claimed {
            variable: variable.clone(),
            edges: r.reals("claimed_edges")?,
            values: r.reals("claimed_values")?,
            mean_error: r.real("claimed_mean_error")?,
        });
    }
    if r.at != bytes.len() {
        return Err(ArtifactRefusal::malformed("trailing_bytes"));
    }

    // Re-derive from the carried premises; nothing claimed is trusted.
    let derived = DiscretizedLinearGaussian::from_spec(&spec, &plan, ctx).map_err(
        |inner: BridgeRefusal| {
            ArtifactRefusal::new(
                inner.reason_code,
                DETAIL_REDERIVATION_REFUSED,
                inner.detail,
                "repair the carried composition or plan and re-export",
            )
        },
    )?;
    let mismatch = |subject: String| {
        ArtifactRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_CLAIM_MISMATCH,
            subject,
            "the claimed numbers are not what these premises produce; re-export from the producing composition",
        )
    };
    if derived.identity() != claimed_identity
        || expected_identity.is_some_and(|expected| expected != derived.identity())
    {
        return Err(ArtifactRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_IDENTITY_MISMATCH,
            derived.identity_hex(),
            "the artifact's premises do not produce the identity it claims or the one expected; \
             use the artifact the identity was issued for",
        ));
    }
    for c in &claimed {
        let edges_ok = derived.cell_edges(&c.variable).is_some_and(|e| same_bits(e, &c.edges));
        let values_ok = derived.level_values(&c.variable).is_some_and(|v| same_bits(v, &c.values));
        let error_ok = derived
            .mechanism_mean_error(&c.variable)
            .is_some_and(|e| e.to_bits() == c.mean_error.to_bits());
        if !(edges_ok && values_ok && error_ok) {
            return Err(mismatch(c.variable.clone()));
        }
    }
    Ok(derived)
}
