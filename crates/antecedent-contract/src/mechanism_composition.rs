//! Semantic contract for composing independently fitted causal mechanisms.
//!
//! A mechanism is a conditional law of one scientific quantity given the
//! quantities it consumes. Two mechanisms compose only when the quantity one
//! produces is, coordinate for coordinate, the quantity the other consumes:
//! the same stable variable and role, units, scale, horizon, and the same
//! population and regime, or a *declared* admission of the difference.
//! Nothing here infers an equivalence; a difference is refused.
//!
//! This module checks the semantic coordinates and emits a
//! [`MechanismCompositionPlan`] of plain identities. The numbers of each
//! mechanism, the executable joint law and the numerical checks (support
//! overlap, positive semidefiniteness, closed-form execution) belong to
//! `antecedent-model`; the artifact layer binds the two. Every refusal is an
//! [`ExternalRefusal`] with a registered reason code and a namespaced detail,
//! and the same detail literals are the ones the model-level checker emits, so a
//! defect has one name wherever it is found.
//!
//! The transport theory behind a population or regime admission is owned
//! elsewhere. This module only represents the declaration and refuses a
//! mechanism from another context that has none.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::identity::CanonicalWriter;

use crate::{ExternalRefusal, QuantityMismatch, ScientificQuantity};

/// Functional identity of a quantity used as a mechanism input or output: one
/// realized value of the variable. A mean, risk or other summary cannot be a
/// parent value.
pub const MECHANISM_VALUE_FUNCTIONAL: &str = "realization";
/// Stage reported by every refusal of this module.
pub const MECHANISM_COMPOSITION_STAGE: &str = "compose";
/// Most mechanisms one declaration may hold.
pub const MAX_DECLARED_MECHANISMS: usize = 64;
/// Version of the plan identity encoding.
pub const MECHANISM_PLAN_IDENTITY_VERSION: u64 = 1;

/// Where a mechanism came from.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MechanismOriginKind {
    /// Fitted from retained evidence.
    Fitted,
    /// Produced by an external provider under a binding.
    ExternalBinding,
    /// Written down by the analyst, not estimated.
    Declared,
}

impl MechanismOriginKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fitted => "fitted",
            Self::ExternalBinding => "external_binding",
            Self::Declared => "declared",
        }
    }
}

/// A declared dependence of a mechanism on an unobserved shared cause.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LatentCauseDeclaration {
    /// Stable identity of the latent cause.
    pub latent_id: String,
    /// Variables of the other mechanisms that depend on the same cause.
    pub shared_with_variable_ids: Vec<String>,
}

/// One independently fitted conditional mechanism, in semantic coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismDeclaration {
    /// The quantity the mechanism produces, in the context it was fitted in.
    pub output: ScientificQuantity,
    /// The quantities it consumes, as its fit saw them.
    pub parents: Vec<ScientificQuantity>,
    /// Variables the fit conditioned on (must equal the parents).
    pub conditioning_variable_ids: Vec<String>,
    /// Origin of the mechanism.
    pub origin: MechanismOriginKind,
    /// Identity of the owning fit or external binding.
    pub artifact_id: String,
    /// Identity of the evidence the mechanism was estimated from.
    pub evidence_id: String,
    /// Declared shared latent causes.
    pub latent_causes: Vec<LatentCauseDeclaration>,
}

/// Which context coordinate an admission is about.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AdmissionCoordinate {
    /// Fitted population differs from the target population.
    Population,
    /// Fitted regime differs from the target regime.
    Regime,
}

/// The basis on which a mechanism from another context is admitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionBasis {
    /// A declared selection diagram between the populations. `targets` lists
    /// every variable that carries a selection arrow; the mechanism of a
    /// variable that does not is invariant.
    SelectionDiagram {
        /// Variables with a selection arrow into them.
        targets: Vec<String>,
    },
    /// An explicit invariance assumption owned by the analyst.
    DeclaredInvariance {
        /// Identity of the assumption statement.
        assumption_id: String,
    },
}

/// A declared admission of one mechanism from another population or regime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportAdmission {
    /// Variable whose mechanism is admitted.
    pub variable_id: String,
    /// Which coordinate differs.
    pub coordinate: AdmissionCoordinate,
    /// Where the mechanism was fitted.
    pub source_id: String,
    /// The composition's target.
    pub target_id: String,
    /// Why it is admitted.
    pub basis: AdmissionBasis,
}

/// What is declared about the dependence between mechanism estimates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependenceDeclaration {
    /// Nothing declared; no uncertainty claim over several mechanisms.
    Unknown,
    /// Estimated from disjoint independently generated evidence.
    IndependentByDesign {
        /// Why the estimates are independent.
        justification: String,
    },
    /// One aligned joint parameter law.
    AlignedJointLaw {
        /// Identity of the joint law.
        law_id: String,
    },
    /// Explicit covariance between every pair of mechanisms.
    CrossCovariance,
}

/// A declared composition in semantic coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismCompositionDeclaration {
    /// Target population identity.
    pub target_population_id: String,
    /// Target regime identity.
    pub target_regime_id: String,
    /// The mechanisms.
    pub mechanisms: Vec<MechanismDeclaration>,
    /// Declared causal graph as `(parent variable, child variable)`.
    pub edges: Vec<(String, String)>,
    /// Declared admissions of mechanisms from other contexts.
    pub admissions: Vec<TransportAdmission>,
    /// Declared posterior dependence.
    pub dependence: DependenceDeclaration,
    /// Acknowledge mechanisms that were declared, not estimated.
    pub admit_declared_mechanisms: bool,
}

/// One consumed quantity in plain coordinates.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct PlannedParent {
    /// Stable variable identity.
    pub variable_id: String,
    /// Units.
    pub units: String,
    /// Scale or transform.
    pub transform_id: String,
}

/// One mechanism in plain coordinates, ready to be bound to its numbers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedMechanism {
    /// Stable variable identity.
    pub variable_id: String,
    /// Units produced.
    pub units: String,
    /// Scale or transform produced.
    pub transform_id: String,
    /// Population fitted in.
    pub population_id: String,
    /// Regime fitted in.
    pub regime_id: String,
    /// Consumed quantities, sorted by variable.
    pub parents: Vec<PlannedParent>,
    /// Conditioning variables, sorted.
    pub conditioning_variable_ids: Vec<String>,
    /// Origin.
    pub origin: MechanismOriginKind,
    /// Owning fit or binding.
    pub artifact_id: String,
    /// Evidence identity.
    pub evidence_id: String,
    /// Latent causes, sorted by identity; peers sorted.
    pub latent_causes: Vec<LatentCauseDeclaration>,
}

/// What an uncertainty claim over the composition needs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UncertaintyRoute {
    /// Point-only: the declarations do not license joint parameter uncertainty.
    Closed(ExternalRefusal),
    /// Joint parameter uncertainty can be assembled under the named declaration.
    Declared(&'static str),
}

/// A semantically checked composition, in canonical order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismCompositionPlan {
    /// Target population identity.
    pub target_population_id: String,
    /// Target regime identity.
    pub target_regime_id: String,
    /// Mechanisms sorted by variable.
    pub mechanisms: Vec<PlannedMechanism>,
    /// Declared edges, sorted.
    pub edges: Vec<(String, String)>,
    /// Admissions in canonical order.
    pub admissions: Vec<TransportAdmission>,
    /// Declared dependence.
    pub dependence: DependenceDeclaration,
    /// Whether declared mechanisms were acknowledged.
    pub admit_declared_mechanisms: bool,
}

/// Registered reason code for a coherence-finding kind tag.
///
/// The tags are the model checker's `CoherenceKind::tag()` words plus
/// `quantity`, used only by the semantic checks of this module.
#[must_use]
pub fn reason_code_for_kind(kind_tag: &str) -> &'static str {
    match kind_tag {
        "shape" => antecedent_core::reason_code!("invalid_argument"),
        "variable_identity" | "units" | "transform" | "population" | "regime" | "quantity" => {
            antecedent_core::reason_code!("quantity_semantics_mismatch")
        }
        "cycle" | "causal_parents" => antecedent_core::reason_code!("graph_invalid"),
        "posterior_dependence" => antecedent_core::reason_code!("joint_law_required"),
        _ => antecedent_core::reason_code!("mechanism_composition_incoherent"),
    }
}

/// A structured refusal for one finding of a composition check, whether it was
/// found by this module or by the executable model-level checker.
#[must_use]
pub fn composition_refusal(
    kind_tag: &str,
    detail: &str,
    subject: &str,
    expected: Option<&str>,
    supplied: Option<&str>,
    remedy: &'static str,
) -> ExternalRefusal {
    let mut refusal =
        ExternalRefusal::new(reason_code_for_kind(kind_tag), MECHANISM_COMPOSITION_STAGE, detail)
            .with_offending(subject)
            .with_remedy(remedy);
    if let Some(e) = expected {
        refusal = refusal.with_expected(e);
    }
    if let Some(s) = supplied {
        refusal = refusal.with_supplied(s);
    }
    refusal
}

fn blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn set_text(items: &BTreeSet<&str>) -> String {
    let joined: Vec<&str> = items.iter().copied().collect();
    format!("{{{}}}", joined.join(","))
}

struct Collector {
    refusals: Vec<ExternalRefusal>,
}

impl Collector {
    fn push(&mut self, kind: &str, detail: &'static str, subject: &str, remedy: &'static str) {
        self.refusals.push(composition_refusal(kind, detail, subject, None, None, remedy));
    }

    fn compare(
        &mut self,
        kind: &str,
        detail: &'static str,
        subject: &str,
        expected: &str,
        supplied: &str,
        remedy: &'static str,
    ) {
        self.refusals.push(composition_refusal(
            kind,
            detail,
            subject,
            Some(expected),
            Some(supplied),
            remedy,
        ));
    }
}

fn check_realization(q: &ScientificQuantity, subject: &str, c: &mut Collector) {
    if q.validate().is_err() {
        c.push(
            "quantity",
            "mechanism_composition.quantity_invalid_coordinate",
            subject,
            "give every semantic coordinate a non-blank identity and canonical conditioning",
        );
        return;
    }
    if q.functional_id != MECHANISM_VALUE_FUNCTIONAL {
        c.compare(
            "quantity",
            "mechanism_composition.quantity_not_a_realization",
            subject,
            MECHANISM_VALUE_FUNCTIONAL,
            &q.functional_id,
            "a mechanism consumes and produces realized values; a mean or other summary is not a parent value",
        );
    }
    if !q.conditioning.is_empty() {
        c.push(
            "quantity",
            "mechanism_composition.quantity_conditioned_event",
            subject,
            "a structural mechanism is not conditioned on observed events; use a conditional query downstream",
        );
    }
}

fn check_context(
    decl: &MechanismCompositionDeclaration,
    m: &MechanismDeclaration,
    c: &mut Collector,
) {
    let v = m.output.variable_id.as_str();
    let admissions_for = |coordinate: AdmissionCoordinate, source: &str, target: &str| {
        decl.admissions
            .iter()
            .filter(|a| {
                a.variable_id == v
                    && a.coordinate == coordinate
                    && a.source_id == source
                    && a.target_id == target
            })
            .collect::<Vec<_>>()
    };
    if m.output.population_id != decl.target_population_id {
        let candidates = admissions_for(
            AdmissionCoordinate::Population,
            &m.output.population_id,
            &decl.target_population_id,
        );
        let mut reason: Option<&'static str> = None;
        let mut admitted = false;
        for a in &candidates {
            match &a.basis {
                AdmissionBasis::DeclaredInvariance { assumption_id } => {
                    if blank(assumption_id) {
                        reason = Some("mechanism_composition.population_invariance_unstated");
                    } else {
                        admitted = true;
                    }
                }
                AdmissionBasis::SelectionDiagram { targets } => {
                    if targets.iter().any(|t| t == v) {
                        reason =
                            Some("mechanism_composition.population_selection_arrow_into_mechanism");
                    } else if m.latent_causes.is_empty() {
                        admitted = true;
                    } else {
                        reason = Some(
                            "mechanism_composition.population_latent_blocks_selection_admissibility",
                        );
                    }
                }
            }
        }
        if !admitted {
            let detail = if candidates.is_empty() {
                "mechanism_composition.population_undeclared_transport"
            } else {
                reason.unwrap_or("mechanism_composition.population_undeclared_transport")
            };
            c.compare(
                "population",
                detail,
                v,
                &decl.target_population_id,
                &m.output.population_id,
                "declare a selection diagram with no arrow into this variable, or an explicit invariance assumption, or refit the mechanism in the target population",
            );
        }
    }
    if m.output.regime_id != decl.target_regime_id {
        let candidates = admissions_for(
            AdmissionCoordinate::Regime,
            &m.output.regime_id,
            &decl.target_regime_id,
        );
        let admitted = candidates.iter().any(|a| {
            matches!(&a.basis, AdmissionBasis::DeclaredInvariance { assumption_id } if !blank(assumption_id))
        });
        if !admitted {
            let detail = if candidates.is_empty() {
                "mechanism_composition.regime_undeclared_invariance"
            } else {
                "mechanism_composition.regime_selection_diagram_not_applicable"
            };
            c.compare(
                "regime",
                detail,
                v,
                &decl.target_regime_id,
                &m.output.regime_id,
                "declare that the mechanism is invariant between the fitted and the target regime, or refit it in the target regime",
            );
        }
    }
}

fn check_graph(
    decl: &MechanismCompositionDeclaration,
    by_var: &BTreeMap<&str, &MechanismDeclaration>,
    c: &mut Collector,
) {
    for (p, ch) in &decl.edges {
        if !by_var.contains_key(p.as_str()) || !by_var.contains_key(ch.as_str()) {
            c.push(
                "variable_identity",
                "mechanism_composition.identity_edge_unknown_variable",
                &format!("{p}->{ch}"),
                "declare edges only between variables that have a mechanism",
            );
        }
    }
    for m in by_var.values() {
        let v = m.output.variable_id.as_str();
        let supplied: BTreeSet<&str> = m.parents.iter().map(|p| p.variable_id.as_str()).collect();
        let declared: BTreeSet<&str> =
            decl.edges.iter().filter(|(_, ch)| ch == v).map(|(p, _)| p.as_str()).collect();
        if supplied != declared {
            c.compare(
                "causal_parents",
                "mechanism_composition.parents_declared_graph_mismatch",
                v,
                &set_text(&declared),
                &set_text(&supplied),
                "refit the mechanism on the declared parents, or correct the declared graph",
            );
        }
        let conditioning: BTreeSet<&str> =
            m.conditioning_variable_ids.iter().map(String::as_str).collect();
        if conditioning.difference(&supplied).next().is_some() {
            c.compare(
                "conditioning_set",
                "mechanism_composition.conditioning_extra_conditioner",
                v,
                &set_text(&supplied),
                &set_text(&conditioning),
                "refit without the extra conditioners, or declare them as parents",
            );
        }
        if supplied.difference(&conditioning).next().is_some() {
            c.compare(
                "conditioning_set",
                "mechanism_composition.conditioning_parent_not_conditioned",
                v,
                &set_text(&supplied),
                &set_text(&conditioning),
                "the fit must condition on every declared parent",
            );
        }
    }
    // Kahn over the union of declared edges and mechanism parents.
    let mut edges: BTreeSet<(&str, &str)> = BTreeSet::new();
    for m in by_var.values() {
        for p in &m.parents {
            edges.insert((p.variable_id.as_str(), m.output.variable_id.as_str()));
        }
    }
    for (p, ch) in &decl.edges {
        if by_var.contains_key(p.as_str()) && by_var.contains_key(ch.as_str()) {
            edges.insert((p.as_str(), ch.as_str()));
        }
    }
    let mut indegree: BTreeMap<&str, usize> = by_var.keys().map(|v| (*v, 0)).collect();
    for (p, ch) in &edges {
        if by_var.contains_key(p) {
            if let Some(n) = indegree.get_mut(ch) {
                *n += 1;
            }
        }
    }
    let mut ready: Vec<&str> = indegree.iter().filter(|(_, n)| **n == 0).map(|(v, _)| *v).collect();
    let mut removed = 0_usize;
    while let Some(v) = ready.pop() {
        removed += 1;
        for (p, ch) in &edges {
            if *p == v {
                if let Some(n) = indegree.get_mut(ch) {
                    *n -= 1;
                    if *n == 0 {
                        ready.push(*ch);
                    }
                }
            }
        }
    }
    if removed < indegree.len() {
        let stuck: BTreeSet<&str> =
            indegree.iter().filter(|(_, n)| **n > 0).map(|(v, _)| *v).collect();
        c.push(
            "cycle",
            "mechanism_composition.cycle_directed_cycle",
            &set_text(&stuck),
            "only acyclic compositions are executed; a feedback system needs its own equilibrium model",
        );
    }
}

fn check_latents(by_var: &BTreeMap<&str, &MechanismDeclaration>, c: &mut Collector) {
    for m in by_var.values() {
        for l in &m.latent_causes {
            for other in &l.shared_with_variable_ids {
                let subject = format!("{}:{other}", l.latent_id);
                match by_var.get(other.as_str()) {
                    None => c.push(
                        "shared_latent",
                        "mechanism_composition.latent_shared_with_absent_mechanism",
                        &subject,
                        "the latent is shared with a mechanism that is not in the composition; add it, or the dropped latent would be treated as independent",
                    ),
                    Some(peer) => {
                        if !peer.latent_causes.iter().any(|x| x.latent_id == l.latent_id) {
                            c.push(
                                "shared_latent",
                                "mechanism_composition.latent_sharing_not_reciprocal",
                                &subject,
                                "both mechanisms must declare the shared latent",
                            );
                        }
                    }
                }
            }
        }
    }
}

fn canonical_mechanism(m: &MechanismDeclaration) -> PlannedMechanism {
    let mut parents: Vec<PlannedParent> = m
        .parents
        .iter()
        .map(|p| PlannedParent {
            variable_id: p.variable_id.clone(),
            units: p.units.clone(),
            transform_id: p.transform_id.clone(),
        })
        .collect();
    parents.sort();
    let mut conditioning = m.conditioning_variable_ids.clone();
    conditioning.sort();
    let mut latents: Vec<LatentCauseDeclaration> = m.latent_causes.clone();
    for l in &mut latents {
        l.shared_with_variable_ids.sort();
    }
    latents.sort_by(|a, b| a.latent_id.cmp(&b.latent_id));
    PlannedMechanism {
        variable_id: m.output.variable_id.clone(),
        units: m.output.units.clone(),
        transform_id: m.output.transform_id.clone(),
        population_id: m.output.population_id.clone(),
        regime_id: m.output.regime_id.clone(),
        parents,
        conditioning_variable_ids: conditioning,
        origin: m.origin,
        artifact_id: m.artifact_id.clone(),
        evidence_id: m.evidence_id.clone(),
        latent_causes: latents,
    }
}

/// Check a declared composition in semantic coordinates.
///
/// # Errors
/// Every semantic refusal found, each with a registered reason code and a
/// namespaced detail. An `Ok` plan is a statement about declared coordinates
/// only: numerical support, parameter-law validity and execution are checked by
/// the model-level composition.
#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent semantic checks over the declaration, each collecting its own refusals"
)]
pub fn check_mechanism_composition(
    decl: &MechanismCompositionDeclaration,
) -> Result<MechanismCompositionPlan, Vec<ExternalRefusal>> {
    let mut c = Collector { refusals: Vec::new() };
    if decl.mechanisms.is_empty() || decl.mechanisms.len() > MAX_DECLARED_MECHANISMS {
        c.push(
            "shape",
            "mechanism_composition.shape_mechanism_count",
            "composition",
            "declare between one and the bounded number of mechanisms",
        );
        return Err(c.refusals);
    }
    if blank(&decl.target_population_id) || blank(&decl.target_regime_id) {
        c.push(
            "shape",
            "mechanism_composition.shape_blank_coordinate",
            "target",
            "name the target population and regime",
        );
        return Err(c.refusals);
    }
    let mut by_var: BTreeMap<&str, &MechanismDeclaration> = BTreeMap::new();
    for m in &decl.mechanisms {
        let v = m.output.variable_id.as_str();
        check_realization(&m.output, v, &mut c);
        for p in &m.parents {
            check_realization(p, &format!("{v}<-{}", p.variable_id), &mut c);
        }
        if by_var.insert(v, m).is_some() {
            c.push(
                "variable_identity",
                "mechanism_composition.identity_duplicate_mechanism",
                v,
                "keep one fitted mechanism per variable, or compose the alternatives separately",
            );
        }
        if blank(&m.artifact_id) {
            c.push(
                "provenance",
                "mechanism_composition.provenance_missing_artifact",
                v,
                "bind the mechanism to the identity of the fit or external binding that produced it",
            );
        }
        if blank(&m.evidence_id) {
            c.push(
                "provenance",
                "mechanism_composition.provenance_missing_evidence",
                v,
                "name the evidence the mechanism was estimated from so dependence can be judged",
            );
        }
        if m.origin == MechanismOriginKind::Declared && !decl.admit_declared_mechanisms {
            c.push(
                "provenance",
                "mechanism_composition.provenance_declared_mechanism_unacknowledged",
                v,
                "a declared (not estimated) mechanism is admitted only when the composition acknowledges it",
            );
        }
    }
    if !c.refusals.is_empty() {
        return Err(c.refusals);
    }
    for m in &decl.mechanisms {
        check_context(decl, m, &mut c);
        let consumer = m.output.variable_id.as_str();
        for p in &m.parents {
            let subject = format!("{}->{consumer}", p.variable_id);
            if p.population_id != m.output.population_id {
                c.compare(
                    "population",
                    "mechanism_composition.population_parent_context_differs_from_fit",
                    &subject,
                    &m.output.population_id,
                    &p.population_id,
                    "a fit sees its parents in the population it was fitted in",
                );
            }
            if p.regime_id != m.output.regime_id {
                c.compare(
                    "regime",
                    "mechanism_composition.regime_parent_context_differs_from_fit",
                    &subject,
                    &m.output.regime_id,
                    &p.regime_id,
                    "a fit sees its parents in the regime it was fitted in",
                );
            }
            let Some(producer) = by_var.get(p.variable_id.as_str()) else {
                c.push(
                    "variable_identity",
                    "mechanism_composition.identity_parent_without_mechanism",
                    &subject,
                    "supply a mechanism for the parent (a root needs its own marginal mechanism)",
                );
                continue;
            };
            // Both mechanisms are brought to the target context (or refused
            // above); the remaining coordinates must then agree exactly.
            let to_target = |q: &ScientificQuantity| {
                let mut n = q.clone();
                n.population_id.clone_from(&decl.target_population_id);
                n.regime_id.clone_from(&decl.target_regime_id);
                n
            };
            match to_target(p).require_same_coordinate(&to_target(&producer.output)) {
                Ok(()) => {}
                Err(QuantityMismatch::Units) => c.compare(
                    "units",
                    "mechanism_composition.units_mismatch",
                    &subject,
                    &p.units,
                    &producer.output.units,
                    "convert explicitly and refit one side; no implicit unit conversion is inferred",
                ),
                Err(QuantityMismatch::Transform) => c.compare(
                    "transform",
                    "mechanism_composition.transform_mismatch",
                    &subject,
                    &p.transform_id,
                    &producer.output.transform_id,
                    "declare the transform the consumer was fitted on, or refit one side",
                ),
                Err(QuantityMismatch::Variable) => c.push(
                    "quantity",
                    "mechanism_composition.quantity_role_mismatch",
                    &subject,
                    "the parent and the producer must name the same variable in the same causal role",
                ),
                Err(QuantityMismatch::Horizon) => c.push(
                    "quantity",
                    "mechanism_composition.quantity_horizon_mismatch",
                    &subject,
                    "the parent and the producer must refer to the same time horizon",
                ),
                Err(QuantityMismatch::Functional) => c.push(
                    "quantity",
                    "mechanism_composition.quantity_functional_mismatch",
                    &subject,
                    "the parent and the producer must be the same functional of the variable",
                ),
                Err(QuantityMismatch::Conditioning) => c.push(
                    "quantity",
                    "mechanism_composition.quantity_conditioning_mismatch",
                    &subject,
                    "the parent and the producer must be conditioned on the same events",
                ),
                Err(
                    QuantityMismatch::InvalidCoordinate
                    | QuantityMismatch::Population
                    | QuantityMismatch::Regime,
                ) => c.push(
                    "quantity",
                    "mechanism_composition.quantity_invalid_coordinate",
                    &subject,
                    "give every semantic coordinate a non-blank identity",
                ),
            }
        }
    }
    check_graph(decl, &by_var, &mut c);
    check_latents(&by_var, &mut c);
    if let DependenceDeclaration::IndependentByDesign { justification } = &decl.dependence {
        if blank(justification) {
            c.push(
                "posterior_dependence",
                "mechanism_composition.posterior_dependence_independence_unjustified",
                "composition",
                "state why the estimates are independent (disjoint, independently generated evidence)",
            );
        }
        let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
        for m in &decl.mechanisms {
            if let Some(first) = seen.insert(m.evidence_id.as_str(), m.output.variable_id.as_str())
            {
                c.compare(
                    "posterior_dependence",
                    "mechanism_composition.posterior_dependence_shared_evidence_declared_independent",
                    &format!("{first}|{}", m.output.variable_id),
                    "distinct evidence identities",
                    &m.evidence_id,
                    "mechanisms estimated from the same evidence are dependent; supply a joint law or cross-mechanism covariance",
                );
            }
        }
    }
    if !c.refusals.is_empty() {
        return Err(c.refusals);
    }
    let mut mechanisms: Vec<PlannedMechanism> =
        decl.mechanisms.iter().map(canonical_mechanism).collect();
    mechanisms.sort_by(|a, b| a.variable_id.cmp(&b.variable_id));
    let mut edges = decl.edges.clone();
    edges.sort();
    edges.dedup();
    let mut admissions = decl.admissions.clone();
    for a in &mut admissions {
        if let AdmissionBasis::SelectionDiagram { targets } = &mut a.basis {
            targets.sort();
            targets.dedup();
        }
    }
    admissions.sort_by(|a, b| {
        (&a.variable_id, a.coordinate, &a.source_id, &a.target_id).cmp(&(
            &b.variable_id,
            b.coordinate,
            &b.source_id,
            &b.target_id,
        ))
    });
    Ok(MechanismCompositionPlan {
        target_population_id: decl.target_population_id.clone(),
        target_regime_id: decl.target_regime_id.clone(),
        mechanisms,
        edges,
        admissions,
        dependence: decl.dependence.clone(),
        admit_declared_mechanisms: decl.admit_declared_mechanisms,
    })
}

impl MechanismCompositionPlan {
    /// What a parameter-uncertainty claim over this composition needs.
    ///
    /// Unknown dependence is never converted into independence: with more than
    /// one mechanism it is a closed route stating the joint law required.
    #[must_use]
    pub fn uncertainty_route(&self) -> UncertaintyRoute {
        if self.mechanisms.len() <= 1 {
            return UncertaintyRoute::Declared("single_mechanism");
        }
        match &self.dependence {
            DependenceDeclaration::Unknown => UncertaintyRoute::Closed(composition_refusal(
                "posterior_dependence",
                "mechanism_composition.posterior_dependence_unknown",
                "composition",
                Some("an aligned joint parameter law or explicit cross-mechanism covariance"),
                Some("independently fitted marginals with undeclared dependence"),
                "supply an aligned joint law over all parameters, or an explicit cross-mechanism covariance for every pair; independence must be declared with its justification, never assumed",
            )),
            DependenceDeclaration::IndependentByDesign { .. } => {
                UncertaintyRoute::Declared("declared_independent_by_design")
            }
            DependenceDeclaration::AlignedJointLaw { .. } => {
                UncertaintyRoute::Declared("aligned_joint_law")
            }
            DependenceDeclaration::CrossCovariance => {
                UncertaintyRoute::Declared("declared_cross_covariance")
            }
        }
    }

    /// Canonical identity of the plan: length-delimited canonical encoding of
    /// every semantic field in canonical order, never a formatted rendering.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.mechanism_composition_plan");
        w.word(MECHANISM_PLAN_IDENTITY_VERSION)
            .text(&self.target_population_id)
            .text(&self.target_regime_id)
            .word(u64::from(self.admit_declared_mechanisms))
            .word(self.mechanisms.len() as u64);
        for m in &self.mechanisms {
            w.text(&m.variable_id)
                .text(&m.units)
                .text(&m.transform_id)
                .text(&m.population_id)
                .text(&m.regime_id)
                .word(match m.origin {
                    MechanismOriginKind::Fitted => 1,
                    MechanismOriginKind::ExternalBinding => 2,
                    MechanismOriginKind::Declared => 3,
                })
                .text(&m.artifact_id)
                .text(&m.evidence_id)
                .word(m.parents.len() as u64);
            for p in &m.parents {
                w.text(&p.variable_id).text(&p.units).text(&p.transform_id);
            }
            w.word(m.conditioning_variable_ids.len() as u64);
            for c in &m.conditioning_variable_ids {
                w.text(c);
            }
            w.word(m.latent_causes.len() as u64);
            for l in &m.latent_causes {
                w.text(&l.latent_id).word(l.shared_with_variable_ids.len() as u64);
                for s in &l.shared_with_variable_ids {
                    w.text(s);
                }
            }
        }
        w.word(self.edges.len() as u64);
        for (p, c) in &self.edges {
            w.text(p).text(c);
        }
        w.word(self.admissions.len() as u64);
        for a in &self.admissions {
            w.text(&a.variable_id)
                .word(match a.coordinate {
                    AdmissionCoordinate::Population => 1,
                    AdmissionCoordinate::Regime => 2,
                })
                .text(&a.source_id)
                .text(&a.target_id);
            match &a.basis {
                AdmissionBasis::SelectionDiagram { targets } => {
                    w.word(1).word(targets.len() as u64);
                    for t in targets {
                        w.text(t);
                    }
                }
                AdmissionBasis::DeclaredInvariance { assumption_id } => {
                    w.word(2).text(assumption_id);
                }
            }
        }
        match &self.dependence {
            DependenceDeclaration::Unknown => {
                w.word(0);
            }
            DependenceDeclaration::IndependentByDesign { justification } => {
                w.word(1).text(justification);
            }
            DependenceDeclaration::AlignedJointLaw { law_id } => {
                w.word(2).text(law_id);
            }
            DependenceDeclaration::CrossCovariance => {
                w.word(3);
            }
        }
        w.finish_hex()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantityRole, external_refusal::ExternalRefusal as Refusal};

    fn q(variable: &str, role: QuantityRole) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: format!("label of {variable}"),
            role,
            units: "kg".into(),
            population_id: "target".into(),
            regime_id: "observational".into(),
            horizon: 0,
            functional_id: MECHANISM_VALUE_FUNCTIONAL.into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn mech(
        variable: &str,
        role: QuantityRole,
        parents: &[(&str, QuantityRole)],
    ) -> MechanismDeclaration {
        MechanismDeclaration {
            output: q(variable, role),
            parents: parents.iter().map(|(p, r)| q(p, *r)).collect(),
            conditioning_variable_ids: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
            origin: MechanismOriginKind::Fitted,
            artifact_id: format!("fit:{variable}"),
            evidence_id: format!("data:{variable}"),
            latent_causes: Vec::new(),
        }
    }

    fn decl() -> MechanismCompositionDeclaration {
        MechanismCompositionDeclaration {
            target_population_id: "target".into(),
            target_regime_id: "observational".into(),
            mechanisms: vec![
                mech("x", QuantityRole::Treatment, &[]),
                mech("m", QuantityRole::Mediator, &[("x", QuantityRole::Treatment)]),
                mech(
                    "y",
                    QuantityRole::Outcome,
                    &[("x", QuantityRole::Treatment), ("m", QuantityRole::Mediator)],
                ),
            ],
            edges: vec![
                ("x".into(), "m".into()),
                ("x".into(), "y".into()),
                ("m".into(), "y".into()),
            ],
            admissions: Vec::new(),
            dependence: DependenceDeclaration::Unknown,
            admit_declared_mechanisms: false,
        }
    }

    fn details(result: Result<MechanismCompositionPlan, Vec<Refusal>>) -> Vec<String> {
        let refusals = result.unwrap_err();
        for r in &refusals {
            r.validate().expect("every refusal is a well formed registered refusal");
        }
        refusals.into_iter().map(|r| r.detail).collect()
    }

    #[test]
    fn a_coherent_declaration_yields_a_canonical_plan_with_a_stable_identity() {
        let plan = check_mechanism_composition(&decl()).unwrap();
        assert_eq!(
            plan.mechanisms.iter().map(|m| m.variable_id.as_str()).collect::<Vec<_>>(),
            ["m", "x", "y"]
        );
        let mut shuffled = decl();
        shuffled.mechanisms.reverse();
        shuffled.edges.reverse();
        for m in &mut shuffled.mechanisms {
            m.parents.reverse();
            m.conditioning_variable_ids.reverse();
        }
        let other = check_mechanism_composition(&shuffled).unwrap();
        assert_eq!(plan, other);
        assert_eq!(plan.identity_hex(), other.identity_hex());
        // A display label is not identity.
        let mut relabeled = decl();
        relabeled.mechanisms[0].output.variable_name = "renamed".into();
        assert_eq!(
            check_mechanism_composition(&relabeled).unwrap().identity_hex(),
            plan.identity_hex()
        );
        // Any semantic edit is a different identity.
        let mut edited = decl();
        edited.mechanisms[1].evidence_id = "data:other".into();
        assert_ne!(
            check_mechanism_composition(&edited).unwrap().identity_hex(),
            plan.identity_hex()
        );
        let mut edited = decl();
        edited.dependence = DependenceDeclaration::CrossCovariance;
        assert_ne!(
            check_mechanism_composition(&edited).unwrap().identity_hex(),
            plan.identity_hex()
        );
    }

    #[test]
    fn units_transform_and_role_mismatches_are_refused_without_conversion() {
        let mut d = decl();
        d.mechanisms[2].parents[1].units = "lb".into();
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.units_mismatch"]
        );
        let mut d = decl();
        d.mechanisms[2].parents[1].transform_id = "log".into();
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.transform_mismatch"]
        );
        let mut d = decl();
        d.mechanisms[2].parents[1].role = QuantityRole::Covariate;
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.quantity_role_mismatch"]
        );
        let mut d = decl();
        d.mechanisms[2].parents[1].functional_id = "mean".into();
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.quantity_not_a_realization"]
        );
    }

    #[test]
    fn population_and_regime_differences_need_a_declared_admission() {
        let mut d = decl();
        d.mechanisms[1].output.population_id = "source".into();
        for p in &mut d.mechanisms[1].parents {
            p.population_id = "source".into();
        }
        // y consumes m as seen in the target.
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.population_undeclared_transport"]
        );

        let mut selection = d.clone();
        selection.admissions.push(TransportAdmission {
            variable_id: "m".into(),
            coordinate: AdmissionCoordinate::Population,
            source_id: "source".into(),
            target_id: "target".into(),
            basis: AdmissionBasis::SelectionDiagram { targets: vec!["x".into()] },
        });
        assert!(check_mechanism_composition(&selection).is_ok());
        selection.admissions[0].basis =
            AdmissionBasis::SelectionDiagram { targets: vec!["m".into()] };
        assert_eq!(
            details(check_mechanism_composition(&selection)),
            ["mechanism_composition.population_selection_arrow_into_mechanism"]
        );
        selection.admissions[0].basis =
            AdmissionBasis::DeclaredInvariance { assumption_id: "assumption:m".into() };
        assert!(check_mechanism_composition(&selection).is_ok());

        let mut regime = decl();
        regime.mechanisms[0].output.regime_id = "do(z=1)".into();
        assert_eq!(
            details(check_mechanism_composition(&regime)),
            ["mechanism_composition.regime_undeclared_invariance"]
        );
    }

    #[test]
    fn graph_conditioning_cycle_latent_and_provenance_defects_are_refused() {
        let mut d = decl();
        d.edges.pop();
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.parents_declared_graph_mismatch"]
        );
        let mut d = decl();
        d.mechanisms[2].conditioning_variable_ids.push("w".into());
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.conditioning_extra_conditioner"]
        );
        let cyc = MechanismCompositionDeclaration {
            mechanisms: vec![
                mech("a", QuantityRole::Covariate, &[("b", QuantityRole::Covariate)]),
                mech("b", QuantityRole::Covariate, &[("a", QuantityRole::Covariate)]),
            ],
            edges: vec![("a".into(), "b".into()), ("b".into(), "a".into())],
            ..decl()
        };
        assert_eq!(
            details(check_mechanism_composition(&cyc)),
            ["mechanism_composition.cycle_directed_cycle"]
        );

        let mut d = decl();
        d.mechanisms[1].latent_causes = vec![LatentCauseDeclaration {
            latent_id: "u".into(),
            shared_with_variable_ids: vec!["ghost".into()],
        }];
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.latent_shared_with_absent_mechanism"]
        );

        let mut d = decl();
        d.mechanisms[1].artifact_id = String::new();
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.provenance_missing_artifact"]
        );
        let mut d = decl();
        d.mechanisms[1].origin = MechanismOriginKind::Declared;
        assert_eq!(
            details(check_mechanism_composition(&d)),
            ["mechanism_composition.provenance_declared_mechanism_unacknowledged"]
        );
    }

    #[test]
    fn unknown_dependence_is_a_joint_law_refusal_never_independence() {
        let plan = check_mechanism_composition(&decl()).unwrap();
        match plan.uncertainty_route() {
            UncertaintyRoute::Closed(refusal) => {
                assert_eq!(refusal.code, "joint_law_required");
                assert_eq!(refusal.detail, "mechanism_composition.posterior_dependence_unknown");
                refusal.validate().unwrap();
                assert!(refusal.remedy.is_some());
            }
            UncertaintyRoute::Declared(label) => panic!("opened a route: {label}"),
        }
        let mut independent = decl();
        independent.dependence =
            DependenceDeclaration::IndependentByDesign { justification: "disjoint trials".into() };
        let plan = check_mechanism_composition(&independent).unwrap();
        assert_eq!(
            plan.uncertainty_route(),
            UncertaintyRoute::Declared("declared_independent_by_design")
        );
        // Independence declared over shared evidence is contradicted.
        independent.mechanisms[1].evidence_id = "data:x".into();
        assert_eq!(
            details(check_mechanism_composition(&independent)),
            ["mechanism_composition.posterior_dependence_shared_evidence_declared_independent"]
        );
    }

    #[test]
    fn every_finding_kind_maps_to_a_registered_runtime_refusal_code() {
        for kind in [
            "shape",
            "family",
            "variable_identity",
            "cycle",
            "causal_parents",
            "conditioning_set",
            "units",
            "transform",
            "population",
            "regime",
            "support",
            "shared_latent",
            "uncertainty_alignment",
            "posterior_dependence",
            "provenance",
            "quantity",
        ] {
            let code = reason_code_for_kind(kind);
            assert!(antecedent_core::reason_code::is_runtime_refusal(code), "{kind} -> {code}");
        }
        assert_eq!(reason_code_for_kind("posterior_dependence"), "joint_law_required");
        assert_eq!(reason_code_for_kind("units"), "quantity_semantics_mismatch");
        assert_eq!(reason_code_for_kind("cycle"), "graph_invalid");
    }
}
