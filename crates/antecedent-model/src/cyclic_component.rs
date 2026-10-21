// SPDX-License-Identifier: MIT OR Apache-2.0

//! A certified cyclic model enters composition as one component.
//!
//! Model composition and contract composition stay feed-forward. A cycle among
//! components is refused, and a provenance chain that returns to its source is
//! refused. Either refusal names the next attempt: wrap the cyclic model as one
//! [`CyclicComponent`] with an explicit solve contract, and keep its source
//! evidence outside the chain that would validate it.
//!
//! Admission is bound to the model and to a family certificate. The component
//! stores the model's dimension and a digest of its coefficients and intercept
//! ([`ModelBinding`]), and it is admitted only when the supplied
//! [`FamilyCertificate`] is [`FamilyCertificate::Admitted`] and is the
//! certificate this model yields. Compilation refuses a model that does not
//! match the binding. The binding is an identity check on the supplied model;
//! it is not a proof that the model is the physical system the source describes.
//! The component's own source id is not evidence that its solution holds, and
//! at least one other certifying-evidence id is required.
//!
//! [`SCC_COMPILE_ATTEMPT`] names the compilation of that contract: factor each
//! free block of the admitted component and keep cross-block disturbance
//! covariance. Naming the attempt does not compile the block and does not
//! identify the model.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::linear_solution::{
    FREE_BLOCK_EQUATION, FamilyCertificate, LinearStructuralMap, family_certificate,
};

/// Next attempt named when a component cycle or a provenance cycle is refused.
pub const CYCLE_GATE_NEXT_ATTEMPT: &str = concat!(
    "wrap the cyclic model as one CyclicComponent with an explicit solve contract, ",
    "and keep its source evidence outside the chain that would validate it",
);

/// Next attempt after a cyclic model is admitted as one component.
///
/// The attempt is compilation of the named free-block equation. This constant
/// does not factor a block, does not admit `ρ(|K|) < 1`, and does not identify
/// the model.
pub const SCC_COMPILE_ATTEMPT: &str = concat!(
    "compile the admitted component by strongly connected components, factor each free block, ",
    "and keep the cross-block disturbance covariance",
);

/// Topological evaluation of an acyclic linear equilibrium.
///
/// This names the DAG schedule. It does not factor a cyclic block, admit a
/// model, or identify it. Specified models remain model-conditional;
/// compilation grants no scientific license.
pub const DAG_EQUILIBRIUM_ATTEMPT: &str =
    "evaluate the acyclic linear equilibrium in topological order";

/// One component and the components that must precede it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDependency {
    /// Component name. Unique in one composition.
    pub name: String,
    /// Components this one consumes.
    ///
    /// Each name must be another component of the same composition. The cycle
    /// gate reads an edge from that dependency to this component.
    pub depends_on: Vec<String>,
}

/// One provenance citation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceCitation {
    /// Artifact this citation justifies.
    pub artifact: String,
    /// Upstream artifacts or evidence ids it cites.
    ///
    /// A cited id that is itself an artifact in the chain is an edge. A cited
    /// id that is not an artifact stays outside the chain.
    pub cites: Vec<String>,
}

/// Why a component composition or a provenance chain was refused.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum CycleGateError {
    /// A component name is empty.
    #[error("component name is empty")]
    EmptyComponentName,
    /// The same component name was declared twice.
    #[error("duplicate component `{name}`")]
    DuplicateComponent {
        /// Repeated name.
        name: String,
    },
    /// A dependency name is empty.
    #[error("component `{component}` names an empty dependency")]
    EmptyDependency {
        /// Component that named it.
        component: String,
    },
    /// A dependency does not name a component of this composition.
    #[error("component `{component}` depends on unknown component `{dependency}`")]
    UnknownComponent {
        /// Component that named the dependency.
        component: String,
        /// Name that is not in the composition.
        dependency: String,
    },
    /// The components depend on each other.
    #[error("component cycle `{path}`; next attempt: {next_attempt}")]
    ComponentCycle {
        /// Cycle path `a -> b -> a`, starting at the smallest name.
        path: String,
        /// [`CYCLE_GATE_NEXT_ATTEMPT`].
        next_attempt: &'static str,
    },
    /// A provenance artifact id is empty.
    #[error("provenance artifact id is empty")]
    EmptyArtifact,
    /// A citation names an empty id.
    #[error("provenance artifact `{artifact}` cites an empty id")]
    EmptyCitation {
        /// Artifact that cited it.
        artifact: String,
    },
    /// The same artifact id was declared twice.
    #[error("duplicate provenance artifact `{id}`")]
    DuplicateArtifact {
        /// Repeated artifact id.
        id: String,
    },
    /// Following citations returns to an artifact.
    #[error("provenance cycle through `{id}`; next attempt: {next_attempt}")]
    ProvenanceCycle {
        /// Smallest artifact id on the cycle.
        id: String,
        /// Cycle path `a -> a` or `a -> b -> a`, starting at `id`.
        path: String,
        /// [`CYCLE_GATE_NEXT_ATTEMPT`].
        next_attempt: &'static str,
    },
}

/// Feed-forward order of a component composition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentOrder {
    order: Vec<String>,
}

impl ComponentOrder {
    /// Producers before consumers. Ties break by name.
    #[must_use]
    pub fn order(&self) -> &[String] {
        &self.order
    }
}

/// Citation order of a provenance chain that does not return to its source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvenanceOrder {
    order: Vec<String>,
}

impl ProvenanceOrder {
    /// Cited artifacts before the artifacts that cite them. Ties break by id.
    #[must_use]
    pub fn order(&self) -> &[String] {
        &self.order
    }
}

/// Identity of the model a component was admitted for: its order and a digest of
/// its coefficients and intercept.
///
/// The digest is a 64-bit FNV-1a hash over the exact `f64` bit patterns (with
/// negative zero folded to zero). Two models with equal bindings are equal up to
/// a hash collision; it is an integrity check, not a cryptographic commitment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelBinding {
    dimension: usize,
    digest: u64,
}

impl ModelBinding {
    /// Binding of `model`.
    #[must_use]
    pub fn of(model: &LinearStructuralMap) -> Self {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash = OFFSET;
        let mut mix = |bytes: [u8; 8]| {
            for byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(PRIME);
            }
        };
        mix(u64::try_from(model.dimension()).unwrap_or(u64::MAX).to_le_bytes());
        for value in model.coefficients().iter().chain(model.intercept()) {
            mix((value + 0.0).to_bits().to_le_bytes());
        }
        Self { dimension: model.dimension(), digest: hash }
    }

    /// Order of the bound model.
    #[must_use]
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Digest of the bound model's coefficients and intercept.
    #[must_use]
    pub const fn digest(self) -> u64 {
        self.digest
    }
}

/// Why a cyclic model was not admitted as one component.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum CyclicComponentError {
    /// The component id is empty.
    #[error("a cyclic component needs a non-empty id")]
    EmptyComponentId,
    /// The source id is empty.
    #[error("cyclic component `{component}` needs a non-empty source id")]
    EmptySourceId {
        /// Component that was refused.
        component: String,
    },
    /// A certifying-evidence id is empty.
    #[error("cyclic component `{component}` names empty certifying evidence")]
    EmptyCertifyingEvidence {
        /// Component that was refused.
        component: String,
    },
    /// No certifying evidence was supplied. The source id is not evidence, so a
    /// component with an empty list would rest on nothing.
    #[error("cyclic component `{component}` needs at least one certifying-evidence id")]
    MissingCertifyingEvidence {
        /// Component that was refused.
        component: String,
    },
    /// The family certificate did not establish `ρ(|K|) < 1`.
    #[error(
        "cyclic component `{component}` needs an admitted ρ(|K|) < 1 family certificate; \
         an insufficient bound does not admit a cyclic component"
    )]
    FamilyNotAdmitted {
        /// Component that was refused.
        component: String,
    },
    /// The supplied certificate is not the one this model yields.
    #[error(
        "cyclic component `{component}`: the family certificate does not belong to the supplied model"
    )]
    CertificateModelMismatch {
        /// Component that was refused.
        component: String,
    },
    /// The source id was offered as evidence that certifies the component.
    #[error(
        "source `{source_id}` cannot certify component `{component}`; source evidence must not \
         recursively validate itself"
    )]
    SourceCertifiesItself {
        /// Component that was refused.
        component: String,
        /// Source id that appeared in the certifying evidence.
        source_id: String,
    },
}

/// One certified cyclic model, admitted as a single composition component.
///
/// The internal feedback of the model is not a cycle among components. Callers
/// place this component in an acyclic composition under [`id`](Self::id). There
/// is no sampling method: a draw is not a solution of the named free-block
/// equation, and this type does not produce one.
#[derive(Clone, Debug, PartialEq)]
pub struct CyclicComponent {
    id: String,
    source_id: String,
    certifying_evidence: Vec<String>,
    binding: ModelBinding,
    rho_abs_upper: f64,
}

impl CyclicComponent {
    /// Admit one cyclic `model` under an admitted family certificate.
    ///
    /// `certifying_evidence` must be non-empty and may name artifacts other than
    /// `source_id`. It must not name `source_id`: the source is not the proof of
    /// the solution. `family` must be [`FamilyCertificate::Admitted`] and must
    /// equal the certificate computed from `model` itself, so a certificate from
    /// another model is refused.
    ///
    /// # Errors
    ///
    /// An empty id, an empty source, an empty evidence list, an empty evidence
    /// id, certifying evidence that contains `source_id`, a certificate that is
    /// not admitted, or a certificate that does not belong to `model`.
    #[must_use = "an admission error is the closed source-evidence gate"]
    pub fn try_new(
        id: &str,
        source_id: &str,
        certifying_evidence: &[&str],
        model: &LinearStructuralMap,
        family: &FamilyCertificate,
    ) -> Result<Self, CyclicComponentError> {
        if id.is_empty() {
            return Err(CyclicComponentError::EmptyComponentId);
        }
        if source_id.is_empty() {
            return Err(CyclicComponentError::EmptySourceId { component: id.to_owned() });
        }
        if certifying_evidence.is_empty() {
            return Err(CyclicComponentError::MissingCertifyingEvidence {
                component: id.to_owned(),
            });
        }
        let mut evidence = Vec::with_capacity(certifying_evidence.len());
        for cited in certifying_evidence {
            if cited.is_empty() {
                return Err(CyclicComponentError::EmptyCertifyingEvidence {
                    component: id.to_owned(),
                });
            }
            if *cited == source_id {
                return Err(CyclicComponentError::SourceCertifiesItself {
                    component: id.to_owned(),
                    source_id: source_id.to_owned(),
                });
            }
            evidence.push((*cited).to_owned());
        }
        let FamilyCertificate::Admitted { rho_abs_upper } = family else {
            return Err(CyclicComponentError::FamilyNotAdmitted { component: id.to_owned() });
        };
        let expected = family_certificate(model);
        let belongs = matches!(
            expected,
            FamilyCertificate::Admitted { rho_abs_upper: own }
                if own.to_bits() == rho_abs_upper.to_bits()
        );
        if !belongs {
            return Err(CyclicComponentError::CertificateModelMismatch {
                component: id.to_owned(),
            });
        }
        Ok(Self {
            id: id.to_owned(),
            source_id: source_id.to_owned(),
            certifying_evidence: evidence,
            binding: ModelBinding::of(model),
            rho_abs_upper: *rho_abs_upper,
        })
    }

    /// Component name used when this model is one node of a composition.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Identity of the source evidence the model was read from.
    ///
    /// This id is not a member of [`Self::certifying_evidence`].
    #[must_use]
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// Evidence that may certify the solution. Never contains [`Self::source_id`].
    #[must_use]
    pub fn certifying_evidence(&self) -> &[String] {
        &self.certifying_evidence
    }

    /// [`FREE_BLOCK_EQUATION`], the operator every admitted component solves.
    #[must_use]
    pub const fn free_block_equation(&self) -> &'static str {
        FREE_BLOCK_EQUATION
    }

    /// Upper bound on `ρ(|K|)` from the admitted family certificate, below one.
    #[must_use]
    pub const fn family_rho_abs_upper(&self) -> f64 {
        self.rho_abs_upper
    }

    /// Order and digest of the model this component was admitted for.
    #[must_use]
    pub const fn model_binding(&self) -> ModelBinding {
        self.binding
    }

    /// Compilation attempt for this admitted component.
    ///
    /// The value is [`SCC_COMPILE_ATTEMPT`]. This does not factor a block and
    /// does not identify the model.
    #[must_use]
    pub fn scc_compile_attempt(&self) -> &'static str {
        SCC_COMPILE_ATTEMPT
    }

    /// Whether `evidence` may certify this component.
    ///
    /// The component's source id never may. This does not add `evidence` to
    /// the certifying list and does not treat it as a proof.
    #[must_use]
    pub fn evidence_may_certify(&self, evidence: &str) -> bool {
        evidence != self.source_id
    }
}

/// Refuse a cycle among components.
///
/// Acyclic compositions return producers before consumers. A cycle is refused
/// before any order is produced. The refusal's `next_attempt` is
/// [`CYCLE_GATE_NEXT_ATTEMPT`].
///
/// # Errors
///
/// An empty or duplicate component name, an empty or unknown dependency, or a
/// directed cycle.
#[must_use = "a component cycle is a refusal, not a feed-forward order"]
pub fn check_component_composition(
    components: &[ComponentDependency],
) -> Result<ComponentOrder, CycleGateError> {
    let mut names = BTreeSet::new();
    for component in components {
        if component.name.is_empty() {
            return Err(CycleGateError::EmptyComponentName);
        }
        if !names.insert(component.name.as_str()) {
            return Err(CycleGateError::DuplicateComponent { name: component.name.clone() });
        }
    }
    let mut predecessors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for component in components {
        let mut edges = BTreeSet::new();
        for dependency in &component.depends_on {
            if dependency.is_empty() {
                return Err(CycleGateError::EmptyDependency { component: component.name.clone() });
            }
            if !names.contains(dependency.as_str()) {
                return Err(CycleGateError::UnknownComponent {
                    component: component.name.clone(),
                    dependency: dependency.clone(),
                });
            }
            edges.insert(dependency.as_str());
        }
        predecessors.insert(component.name.as_str(), edges);
    }
    match topological_order(&names, &predecessors) {
        Ok(order) => Ok(ComponentOrder { order: owned_names(&order) }),
        Err(cycle) => Err(CycleGateError::ComponentCycle {
            path: cycle.join(" -> "),
            next_attempt: CYCLE_GATE_NEXT_ATTEMPT,
        }),
    }
}

/// Refuse a provenance chain that returns to its source.
///
/// A citation of an id that is not an artifact in `citations` stays outside
/// the chain. A citation of an artifact in the chain, including a citation of
/// the artifact itself, is an edge. A cycle is refused before any order is
/// produced. The refusal's `next_attempt` is [`CYCLE_GATE_NEXT_ATTEMPT`].
///
/// # Errors
///
/// An empty or duplicate artifact id, an empty cited id, or a directed cycle.
#[must_use = "a provenance cycle is a refusal, not an acyclic chain"]
pub fn check_provenance_chain(
    citations: &[ProvenanceCitation],
) -> Result<ProvenanceOrder, CycleGateError> {
    let mut names = BTreeSet::new();
    for citation in citations {
        if citation.artifact.is_empty() {
            return Err(CycleGateError::EmptyArtifact);
        }
        if !names.insert(citation.artifact.as_str()) {
            return Err(CycleGateError::DuplicateArtifact { id: citation.artifact.clone() });
        }
    }
    let mut predecessors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for citation in citations {
        let mut edges = BTreeSet::new();
        for cited in &citation.cites {
            if cited.is_empty() {
                return Err(CycleGateError::EmptyCitation { artifact: citation.artifact.clone() });
            }
            if names.contains(cited.as_str()) {
                edges.insert(cited.as_str());
            }
        }
        predecessors.insert(citation.artifact.as_str(), edges);
    }
    match topological_order(&names, &predecessors) {
        Ok(order) => Ok(ProvenanceOrder { order: owned_names(&order) }),
        Err(cycle) => Err(CycleGateError::ProvenanceCycle {
            id: cycle.first().copied().unwrap_or("").to_owned(),
            path: cycle.join(" -> "),
            next_attempt: CYCLE_GATE_NEXT_ATTEMPT,
        }),
    }
}

fn owned_names(order: &[&str]) -> Vec<String> {
    order.iter().map(|name| (*name).to_owned()).collect()
}

/// Producers before consumers, or one cycle rotated to the smallest name.
fn topological_order<'a>(
    nodes: &BTreeSet<&'a str>,
    predecessors: &BTreeMap<&'a str, BTreeSet<&'a str>>,
) -> Result<Vec<&'a str>, Vec<&'a str>> {
    let mut indegree: BTreeMap<&str, usize> = nodes.iter().copied().map(|name| (name, 0)).collect();
    let mut successors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (node, preds) in predecessors {
        if !nodes.contains(node) {
            continue;
        }
        for pred in preds.iter().copied() {
            if !nodes.contains(pred) {
                continue;
            }
            successors.entry(pred).or_default().insert(node);
            if let Some(degree) = indegree.get_mut(node) {
                *degree += 1;
            }
        }
    }
    let mut ready: BTreeSet<&str> =
        indegree.iter().filter(|(_, degree)| **degree == 0).map(|(name, _)| *name).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(next) = ready.pop_first() {
        order.push(next);
        let Some(children) = successors.get(next) else {
            continue;
        };
        for child in children.iter().copied() {
            let Some(degree) = indegree.get_mut(child) else {
                continue;
            };
            *degree -= 1;
            if *degree == 0 {
                ready.insert(child);
            }
        }
    }
    if order.len() == nodes.len() {
        return Ok(order);
    }
    Err(cycle_path(nodes, &order, predecessors))
}

fn cycle_path<'a>(
    nodes: &BTreeSet<&'a str>,
    order: &[&'a str],
    predecessors: &BTreeMap<&'a str, BTreeSet<&'a str>>,
) -> Vec<&'a str> {
    let remaining: BTreeSet<&str> =
        nodes.iter().copied().filter(|name| !order.contains(name)).collect();
    let Some(mut current) = remaining.iter().copied().next() else {
        return Vec::new();
    };
    let mut walk = Vec::new();
    while !walk.contains(&current) {
        walk.push(current);
        let previous = predecessors
            .get(current)
            .and_then(|preds| preds.iter().copied().find(|name| remaining.contains(name)));
        if let Some(name) = previous {
            current = name;
        } else {
            break;
        }
    }
    let start = walk.iter().position(|name| *name == current).unwrap_or(0);
    let mut cycle: Vec<&str> = walk[start..].to_vec();
    cycle.reverse();
    if !cycle.is_empty() {
        let mut pivot = 0;
        for index in 1..cycle.len() {
            if cycle[index] < cycle[pivot] {
                pivot = index;
            }
        }
        cycle.rotate_left(pivot);
        if let Some(first) = cycle.first().copied() {
            cycle.push(first);
        }
    }
    cycle
}

#[cfg(test)]
mod tests {
    use super::{
        CYCLE_GATE_NEXT_ATTEMPT, ComponentDependency, CycleGateError, CyclicComponent,
        CyclicComponentError, ModelBinding, ProvenanceCitation, check_component_composition,
        check_provenance_chain,
    };
    use crate::linear_solution::{FREE_BLOCK_EQUATION, HardClamp, LinearStructuralMap, assess};

    const NEXT_ATTEMPT: &str = concat!(
        "wrap the cyclic model as one CyclicComponent with an explicit solve contract, ",
        "and keep its source evidence outside the chain that would validate it",
    );

    #[test]
    fn two_components_that_depend_on_each_other_are_refused() {
        assert_eq!(CYCLE_GATE_NEXT_ATTEMPT, NEXT_ATTEMPT);
        let feed_forward = [
            ComponentDependency { name: "a".to_owned(), depends_on: Vec::new() },
            ComponentDependency { name: "b".to_owned(), depends_on: vec!["a".to_owned()] },
        ];
        let order = check_component_composition(&feed_forward).unwrap();
        assert_eq!(order.order(), ["a".to_owned(), "b".to_owned()].as_slice());

        let mutual = [
            ComponentDependency { name: "a".to_owned(), depends_on: vec!["b".to_owned()] },
            ComponentDependency { name: "b".to_owned(), depends_on: vec!["a".to_owned()] },
        ];
        let err = check_component_composition(&mutual).unwrap_err();
        let text = err.to_string();
        let CycleGateError::ComponentCycle { path, next_attempt } = err else {
            panic!("expected a component cycle, got {text}");
        };
        assert_eq!(path, "a -> b -> a");
        assert_eq!(next_attempt, NEXT_ATTEMPT);
        assert!(text.contains(next_attempt), "{text}");
    }

    #[test]
    fn a_provenance_chain_that_cites_itself_is_refused() {
        let open = [ProvenanceCitation {
            artifact: "report".to_owned(),
            cites: vec!["source".to_owned()],
        }];
        let order = check_provenance_chain(&open).unwrap();
        assert_eq!(order.order(), ["report".to_owned()].as_slice());

        let cites_itself = [ProvenanceCitation {
            artifact: "source".to_owned(),
            cites: vec!["source".to_owned()],
        }];
        let err = check_provenance_chain(&cites_itself).unwrap_err();
        let text = err.to_string();
        let CycleGateError::ProvenanceCycle { id, path, next_attempt } = err else {
            panic!("expected a provenance cycle, got {text}");
        };
        assert_eq!(id, "source");
        assert_eq!(path, "source -> source");
        assert_eq!(next_attempt, NEXT_ATTEMPT);
        assert!(text.contains(next_attempt), "{text}");

        let returns = [
            ProvenanceCitation { artifact: "report".to_owned(), cites: vec!["source".to_owned()] },
            ProvenanceCitation { artifact: "source".to_owned(), cites: vec!["report".to_owned()] },
        ];
        let err = check_provenance_chain(&returns).unwrap_err();
        let CycleGateError::ProvenanceCycle { id, path, next_attempt } = err else {
            panic!("expected a provenance cycle");
        };
        assert_eq!(id, "report");
        assert_eq!(path, "report -> source -> report");
        assert_eq!(next_attempt, NEXT_ATTEMPT);
    }

    fn cycle_model(weight_a: f64, weight_b: f64) -> LinearStructuralMap {
        LinearStructuralMap::try_new(vec![0.0, weight_a, weight_b, 0.0], vec![0.0, 0.0])
            .expect("fixture model")
    }

    #[test]
    fn one_admitted_cyclic_component_bound_to_its_model_is_accepted() {
        let model = cycle_model(0.4, 0.5);
        let family = assess(&model, &HardClamp::none(2).unwrap()).unwrap().family;
        let component = CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note"],
            &model,
            &family,
        )
        .unwrap();
        assert_eq!(component.id(), "feedback");
        assert_eq!(component.source_id(), "source-measurements");
        assert_eq!(component.certifying_evidence(), ["solve-note".to_owned()].as_slice());
        assert_eq!(component.free_block_equation(), FREE_BLOCK_EQUATION);
        // sqrt(0.4 * 0.5) computed by hand.
        assert!((component.family_rho_abs_upper() - 0.2_f64.sqrt()).abs() < 1.0e-12);
        assert_eq!(component.model_binding(), ModelBinding::of(&model));
        assert_eq!(component.model_binding().dimension(), 2);
        assert_ne!(ModelBinding::of(&cycle_model(0.4, 0.6)), component.model_binding());
        assert!(!component.evidence_may_certify(component.source_id()));

        let alone =
            [ComponentDependency { name: component.id().to_owned(), depends_on: Vec::new() }];
        let order = check_component_composition(&alone).unwrap();
        assert_eq!(order.order(), [component.id().to_owned()].as_slice());
    }

    #[test]
    fn empty_certifying_evidence_is_refused() {
        let model = cycle_model(0.4, 0.5);
        let family = assess(&model, &HardClamp::none(2).unwrap()).unwrap().family;
        let err = CyclicComponent::try_new("feedback", "source", &[], &model, &family).unwrap_err();
        assert_eq!(
            err,
            CyclicComponentError::MissingCertifyingEvidence { component: "feedback".to_owned() }
        );
    }

    #[test]
    fn a_family_certificate_that_is_not_admitted_is_refused() {
        // rho(|K|) = sqrt(1.5 * 1.5) = 1.5 is not below one.
        let model = cycle_model(1.5, 1.5);
        let family = assess(&model, &HardClamp::none(2).unwrap()).unwrap().family;
        assert!(!family.is_admitted());
        let err =
            CyclicComponent::try_new("feedback", "source", &["note"], &model, &family).unwrap_err();
        assert_eq!(
            err,
            CyclicComponentError::FamilyNotAdmitted { component: "feedback".to_owned() }
        );
    }

    #[test]
    fn a_certificate_from_another_model_is_refused() {
        let admitted_model = cycle_model(0.4, 0.5);
        let family = assess(&admitted_model, &HardClamp::none(2).unwrap()).unwrap().family;
        // Wrap an unrelated, differently weighted model with the first model's certificate.
        let other = cycle_model(0.6, 0.5);
        let err =
            CyclicComponent::try_new("feedback", "source", &["note"], &other, &family).unwrap_err();
        assert_eq!(
            err,
            CyclicComponentError::CertificateModelMismatch { component: "feedback".to_owned() }
        );
    }

    #[test]
    fn source_id_cannot_appear_as_the_evidence_that_certifies_it() {
        let model = cycle_model(0.4, 0.5);
        let family = assess(&model, &HardClamp::none(2).unwrap()).unwrap().family;
        let err = CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note", "source-measurements"],
            &model,
            &family,
        )
        .unwrap_err();
        let text = err.to_string();
        assert_eq!(
            err,
            CyclicComponentError::SourceCertifiesItself {
                component: "feedback".to_owned(),
                source_id: "source-measurements".to_owned(),
            }
        );
        assert!(text.contains("source-measurements"), "{text}");
        assert!(text.contains("recursively validate itself"), "{text}");
    }
}
