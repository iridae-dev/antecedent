// SPDX-License-Identifier: MIT OR Apache-2.0

//! A certified cyclic model enters composition as one component.
//!
//! Model composition and contract composition stay feed-forward. A cycle among
//! components is refused, and a provenance chain that returns to its source is
//! refused. Either refusal names the next attempt: wrap the cyclic model as one
//! [`CyclicComponent`] with an explicit solve contract, and keep its source
//! evidence outside the chain that would validate it.
//!
//! The component carries that solve contract by name. [`FREE_BLOCK_EQUATION`]
//! and the [`FAMILY_BOUND_REFERENCE`] `ρ(|K|) < 1` are references to the
//! linear-solution contract ([`crate::linear_solution::FamilyCertificate`]).
//! This module does not evaluate the bound, does not issue a new certificate,
//! and does not sample. The component's own source id is not evidence that
//! its solution holds.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::linear_solution::FREE_BLOCK_EQUATION;

/// Next attempt named when a component cycle or a provenance cycle is refused.
pub const CYCLE_GATE_NEXT_ATTEMPT: &str = concat!(
    "wrap the cyclic model as one CyclicComponent with an explicit solve contract, ",
    "and keep its source evidence outside the chain that would validate it",
);

/// Name of the family bound `ρ(|K|) < 1`.
///
/// This is a reference to [`crate::linear_solution::FamilyCertificate`].
/// Carrying the name does not admit the bound and does not compute a radius.
pub const FAMILY_BOUND_REFERENCE: &str = "ρ(|K|) < 1";

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

/// The free-block equation and the family bound, named rather than re-certified.
///
/// [`Self::explicit`] is the only solve contract a [`CyclicComponent`] accepts.
/// It points at [`FREE_BLOCK_EQUATION`] and [`FAMILY_BOUND_REFERENCE`]. It is
/// not a [`crate::linear_solution::FamilyCertificate`] and not a factorization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SolveContractName {
    /// [`FREE_BLOCK_EQUATION`].
    pub free_block_equation: &'static str,
    /// [`FAMILY_BOUND_REFERENCE`].
    pub family_bound: &'static str,
}

impl SolveContractName {
    /// The linear equilibrium contract, by name.
    ///
    /// This does not read a coefficient matrix and does not admit `ρ(|K|) < 1`.
    #[must_use]
    pub const fn explicit() -> Self {
        Self { free_block_equation: FREE_BLOCK_EQUATION, family_bound: FAMILY_BOUND_REFERENCE }
    }

    /// Whether both names are the linear-solution references.
    #[must_use]
    pub fn is_explicit(self) -> bool {
        self.free_block_equation == FREE_BLOCK_EQUATION
            && self.family_bound == FAMILY_BOUND_REFERENCE
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
    /// The solve contract is not [`SolveContractName::explicit`].
    ///
    /// A different string is not a new certificate and is not this family's
    /// free-block equation or `ρ(|K|) < 1` bound.
    #[error(
        "cyclic component `{component}` needs an explicit solve contract naming the free-block \
         equation and the ρ(|K|) < 1 family bound, as references rather than a new certificate"
    )]
    SolveContractNotExplicit {
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CyclicComponent {
    id: String,
    source_id: String,
    certifying_evidence: Vec<String>,
    solve: SolveContractName,
}

impl CyclicComponent {
    /// Admit one cyclic model with an explicit solve contract.
    ///
    /// `certifying_evidence` may name artifacts other than `source_id`. It
    /// must not name `source_id`: the source is not the proof of the solution.
    /// The solve contract is stored by name and is not re-derived from that
    /// evidence.
    ///
    /// # Errors
    ///
    /// An empty id, an empty source, empty evidence, a solve contract other
    /// than [`SolveContractName::explicit`], or certifying evidence that
    /// contains `source_id`.
    #[must_use = "an admission error is the closed source-evidence gate"]
    pub fn try_new(
        id: &str,
        source_id: &str,
        certifying_evidence: &[&str],
        solve: SolveContractName,
    ) -> Result<Self, CyclicComponentError> {
        if id.is_empty() {
            return Err(CyclicComponentError::EmptyComponentId);
        }
        if source_id.is_empty() {
            return Err(CyclicComponentError::EmptySourceId { component: id.to_owned() });
        }
        if !solve.is_explicit() {
            return Err(CyclicComponentError::SolveContractNotExplicit {
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
        Ok(Self {
            id: id.to_owned(),
            source_id: source_id.to_owned(),
            certifying_evidence: evidence,
            solve,
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

    /// Solve contract carried by name.
    #[must_use]
    pub const fn solve_contract(&self) -> SolveContractName {
        self.solve
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
        CyclicComponentError, ProvenanceCitation, SolveContractName, check_component_composition,
        check_provenance_chain,
    };
    use crate::linear_solution::FREE_BLOCK_EQUATION;

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

    #[test]
    fn one_cyclic_component_with_an_explicit_solve_contract_is_accepted() {
        let unnamed = SolveContractName {
            free_block_equation: FREE_BLOCK_EQUATION,
            family_bound: "a computed radius",
        };
        let refused =
            CyclicComponent::try_new("feedback", "source-measurements", &["solve-note"], unnamed)
                .unwrap_err();
        assert!(matches!(
            refused,
            CyclicComponentError::SolveContractNotExplicit { component } if component == "feedback"
        ));

        let component = CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note"],
            SolveContractName::explicit(),
        )
        .unwrap();
        assert_eq!(component.id(), "feedback");
        assert_eq!(component.source_id(), "source-measurements");
        assert_eq!(component.certifying_evidence(), ["solve-note".to_owned()].as_slice());
        assert_eq!(component.solve_contract().free_block_equation, FREE_BLOCK_EQUATION);
        assert_eq!(component.solve_contract().family_bound, "ρ(|K|) < 1");
        assert!(component.solve_contract().is_explicit());
        assert!(!component.evidence_may_certify(component.source_id()));

        let alone =
            [ComponentDependency { name: component.id().to_owned(), depends_on: Vec::new() }];
        let order = check_component_composition(&alone).unwrap();
        assert_eq!(order.order(), [component.id().to_owned()].as_slice());
    }

    #[test]
    fn source_id_cannot_appear_as_the_evidence_that_certifies_it() {
        let err = CyclicComponent::try_new(
            "feedback",
            "source-measurements",
            &["solve-note", "source-measurements"],
            SolveContractName::explicit(),
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
