//! Coherence checking of a declared composition.
//!
//! Individually valid mechanisms do not imply a valid composition. The checker
//! returns every defect it finds, each as a [`CoherenceFinding`] with a closed
//! kind, a namespaced `detail` literal, the offending subject, the expected and
//! supplied semantics when a comparison failed, and the declaration that would
//! license the composition. A composition with no finding establishes one
//! acyclic joint law over the mechanisms' outputs under the declared graph; it
//! establishes nothing about posterior dependence, which is checked when an
//! uncertainty claim is requested.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::descriptor::{
    ComposedModelSpec, Containment, ContextCoordinate, CrossCovarianceBlock, Domain,
    MAX_COMPOSED_LATENTS, MAX_COMPOSED_MECHANISMS, MAX_LEVELS_PER_VARIABLE,
    MAX_PARENTS_PER_MECHANISM, MechanismDescriptor, MechanismFamily, MechanismForm,
    MechanismOrigin, MechanismUncertainty, ParameterId, ParameterTerm, PosteriorDependence,
};

/// Tolerance for symmetry and marginal-consistency comparisons of covariances.
pub const COVARIANCE_TOLERANCE: f64 = 1e-9;

/// The closed set of things a composition is checked for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CoherenceKind {
    /// A declaration is malformed (blank identity, non-finite number, bad table).
    Shape,
    /// Mechanisms of more than one family are mixed.
    Family,
    /// Variable identity: duplicate mechanism, unknown parent or edge endpoint.
    VariableIdentity,
    /// The declared structure has a directed cycle.
    Cycle,
    /// A mechanism's parent set differs from the declared graph.
    CausalParents,
    /// A mechanism's conditioning set differs from its parent set.
    ConditioningSet,
    /// A parent is consumed in other units than it is produced in.
    Units,
    /// A parent is consumed on another scale or transform than it is produced on.
    Transform,
    /// A mechanism was fitted in another population than the target.
    Population,
    /// A mechanism was fitted in another regime than the target.
    Regime,
    /// A producer's output support is not inside the consumer's input domain.
    Support,
    /// A shared latent cause is dropped, unreciprocated or has no loading.
    SharedLatent,
    /// Uncertainty laws are misaligned or invalid.
    UncertaintyAlignment,
    /// Dependence between parameter estimates is unknown or contradicted.
    PosteriorDependence,
    /// Mechanism provenance is missing or not acknowledged.
    Provenance,
}

impl CoherenceKind {
    /// Stable `snake_case` tag. The same words are the keys the semantic
    /// contract maps to registered reason codes.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Shape => "shape",
            Self::Family => "family",
            Self::VariableIdentity => "variable_identity",
            Self::Cycle => "cycle",
            Self::CausalParents => "causal_parents",
            Self::ConditioningSet => "conditioning_set",
            Self::Units => "units",
            Self::Transform => "transform",
            Self::Population => "population",
            Self::Regime => "regime",
            Self::Support => "support",
            Self::SharedLatent => "shared_latent",
            Self::UncertaintyAlignment => "uncertainty_alignment",
            Self::PosteriorDependence => "posterior_dependence",
            Self::Provenance => "provenance",
        }
    }

    /// The registered reason code a finding of this kind carries. It is the
    /// mapping the semantic contract applies by [`Self::tag`]: a malformed
    /// declaration is an invalid argument, a quantity that does not match its
    /// producer is a semantics mismatch, a cycle or a parent set that is not
    /// the declared graph is an invalid graph, unknown or contradicted
    /// dependence needs a joint law, and every other defect is an incoherent
    /// composition.
    #[must_use]
    pub fn reason_code(self) -> &'static str {
        match self {
            Self::Shape => antecedent_core::reason_code!("invalid_argument"),
            Self::VariableIdentity
            | Self::Units
            | Self::Transform
            | Self::Population
            | Self::Regime => antecedent_core::reason_code!("quantity_semantics_mismatch"),
            Self::Cycle | Self::CausalParents => antecedent_core::reason_code!("graph_invalid"),
            Self::PosteriorDependence => antecedent_core::reason_code!("joint_law_required"),
            Self::Family
            | Self::ConditioningSet
            | Self::Support
            | Self::SharedLatent
            | Self::UncertaintyAlignment
            | Self::Provenance => {
                antecedent_core::reason_code!("mechanism_composition_incoherent")
            }
        }
    }

    /// Every kind, in report order.
    pub const ALL: [CoherenceKind; 15] = [
        Self::Shape,
        Self::Family,
        Self::VariableIdentity,
        Self::Cycle,
        Self::CausalParents,
        Self::ConditioningSet,
        Self::Units,
        Self::Transform,
        Self::Population,
        Self::Regime,
        Self::Support,
        Self::SharedLatent,
        Self::UncertaintyAlignment,
        Self::PosteriorDependence,
        Self::Provenance,
    ];
}

/// One defect of a composition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoherenceFinding {
    /// What was checked.
    pub kind: CoherenceKind,
    /// Namespaced detail literal, `family.slot`.
    pub detail: &'static str,
    /// The variable, edge (`parent->child`) or parameter at fault.
    pub subject: String,
    /// Expected semantics, when a comparison failed.
    pub expected: Option<String>,
    /// Supplied semantics, when a comparison failed.
    pub supplied: Option<String>,
    /// The evidence, assumption or declaration that would license the request.
    pub remedy: &'static str,
}

impl CoherenceFinding {
    pub(crate) fn new(
        kind: CoherenceKind,
        detail: &'static str,
        subject: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { kind, detail, subject: subject.into(), expected: None, supplied: None, remedy }
    }

    pub(crate) fn comparing(
        mut self,
        expected: impl Into<String>,
        supplied: impl Into<String>,
    ) -> Self {
        self.expected = Some(expected.into());
        self.supplied = Some(supplied.into());
        self
    }
}

fn list<I: IntoIterator<Item = String>>(items: I) -> String {
    let collected: Vec<String> = items.into_iter().collect();
    format!("{{{}}}", collected.join(","))
}

fn blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Whether a symmetric `n x n` matrix is positive semidefinite up to a
/// relative jitter of `1e-9` of its largest diagonal entry.
#[must_use]
pub fn is_positive_semidefinite(matrix: &[f64], n: usize) -> bool {
    if matrix.len() != n * n || !all_finite(matrix) {
        return false;
    }
    let scale = (0..n).map(|i| matrix[i * n + i].abs()).fold(0.0_f64, f64::max);
    if (0..n).any(|i| matrix[i * n + i] < 0.0) {
        return false;
    }
    if scale == 0.0 {
        return matrix.iter().all(|v| *v == 0.0);
    }
    let tau = 1e-9 * scale;
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = matrix[i * n + j];
            if i == j {
                s += tau;
            }
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    true
}

fn is_symmetric(matrix: &[f64], n: usize) -> bool {
    (0..n).all(|i| {
        (0..i).all(|j| {
            let a = matrix[i * n + j];
            let b = matrix[j * n + i];
            (a - b).abs() <= COVARIANCE_TOLERANCE * (1.0 + a.abs().max(b.abs()))
        })
    })
}

/// The parameters of one linear-Gaussian mechanism in canonical order:
/// intercept, then one coefficient per parent sorted by variable.
#[must_use]
pub fn parameter_terms(descriptor: &MechanismDescriptor) -> Vec<ParameterTerm> {
    let mut terms = vec![ParameterTerm::Intercept];
    let mut parents: Vec<&str> = descriptor.parents.iter().map(|p| p.variable.as_str()).collect();
    parents.sort_unstable();
    terms.extend(parents.into_iter().map(|p| ParameterTerm::Coefficient(p.to_owned())));
    terms
}

/// Every parameter of a canonical specification, in canonical order.
#[must_use]
pub fn parameter_layout(spec: &ComposedModelSpec) -> Vec<ParameterId> {
    let mut out = Vec::new();
    for mechanism in &spec.mechanisms {
        for term in parameter_terms(mechanism) {
            out.push(ParameterId { mechanism: mechanism.variable.clone(), term });
        }
    }
    out
}

#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent structural checks, each pushing its own coherence findings"
)]
fn check_shape(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    use CoherenceKind::Shape;
    if spec.mechanisms.is_empty() {
        out.push(CoherenceFinding::new(
            Shape,
            "mechanism_composition.shape_empty_composition",
            "composition",
            "declare at least one mechanism",
        ));
    }
    if spec.mechanisms.len() > MAX_COMPOSED_MECHANISMS {
        out.push(
            CoherenceFinding::new(
                Shape,
                "mechanism_composition.shape_too_many_mechanisms",
                "composition",
                "split the composition below the declared bound",
            )
            .comparing(
                format!("at most {MAX_COMPOSED_MECHANISMS} mechanisms"),
                spec.mechanisms.len().to_string(),
            ),
        );
    }
    if blank(&spec.target.population) || blank(&spec.target.regime) {
        out.push(CoherenceFinding::new(
            Shape,
            "mechanism_composition.shape_blank_coordinate",
            "target",
            "name the target population and regime",
        ));
    }
    for d in &spec.mechanisms {
        let v = d.variable.as_str();
        if blank(v) {
            out.push(CoherenceFinding::new(
                Shape,
                "mechanism_composition.shape_blank_identity",
                "variable",
                "give every mechanism a stable variable identity",
            ));
            continue;
        }
        if blank(&d.output_units)
            || blank(&d.output_transform)
            || blank(&d.context.population)
            || blank(&d.context.regime)
        {
            out.push(CoherenceFinding::new(
                Shape,
                "mechanism_composition.shape_blank_coordinate",
                v,
                "declare units, transform, population and regime for the mechanism",
            ));
        }
        if let Err(why) = d.output_support.validate() {
            out.push(
                CoherenceFinding::new(
                    Shape,
                    "mechanism_composition.shape_invalid_domain",
                    v,
                    "declare a well formed output support",
                )
                .comparing("a valid domain", why),
            );
        }
        if d.parents.len() > MAX_PARENTS_PER_MECHANISM {
            out.push(CoherenceFinding::new(
                Shape,
                "mechanism_composition.shape_too_many_parents",
                v,
                "reduce the parent set below the declared bound",
            ));
        }
        for p in &d.parents {
            if blank(&p.variable) || blank(&p.units) || blank(&p.transform) {
                out.push(CoherenceFinding::new(
                    Shape,
                    "mechanism_composition.shape_blank_coordinate",
                    format!("{v}<-{}", p.variable),
                    "declare the parent's variable, units and transform",
                ));
            }
            if let Err(why) = p.domain.validate() {
                out.push(
                    CoherenceFinding::new(
                        Shape,
                        "mechanism_composition.shape_invalid_domain",
                        format!("{v}<-{}", p.variable),
                        "declare a well formed input domain",
                    )
                    .comparing("a valid domain", why),
                );
            }
        }
        if d.parents.windows(2).any(|w| w[0].variable == w[1].variable) {
            out.push(CoherenceFinding::new(
                Shape,
                "mechanism_composition.shape_duplicate_parent",
                v,
                "list each parent once",
            ));
        }
        if d.latents.len() > MAX_COMPOSED_LATENTS
            || d.latents.windows(2).any(|w| w[0].latent == w[1].latent)
            || d.latents.iter().any(|l| blank(&l.latent))
        {
            out.push(CoherenceFinding::new(
                Shape,
                "mechanism_composition.shape_invalid_latents",
                v,
                "name each latent once with a non-blank identity",
            ));
        }
        match &d.form {
            MechanismForm::LinearGaussian { intercept, sigma, coefficients } => {
                if !intercept.is_finite()
                    || !sigma.is_finite()
                    || !coefficients.iter().all(|(_, c)| c.is_finite())
                {
                    out.push(CoherenceFinding::new(
                        Shape,
                        "mechanism_composition.shape_non_finite_parameter",
                        v,
                        "refit or correct the mechanism: every parameter must be finite",
                    ));
                }
                if *sigma < 0.0 {
                    out.push(CoherenceFinding::new(
                        Shape,
                        "mechanism_composition.shape_negative_scale",
                        v,
                        "a residual standard deviation cannot be negative",
                    ));
                }
                let parents: Vec<&str> = d.parents.iter().map(|p| p.variable.as_str()).collect();
                let keys: Vec<&str> = coefficients.iter().map(|(k, _)| k.as_str()).collect();
                if parents != keys {
                    out.push(
                        CoherenceFinding::new(
                            Shape,
                            "mechanism_composition.shape_coefficient_parent_mismatch",
                            v,
                            "give exactly one coefficient per declared parent",
                        )
                        .comparing(
                            list(parents.iter().map(|s| (*s).to_owned())),
                            list(keys.iter().map(|s| (*s).to_owned())),
                        ),
                    );
                }
                let numeric = |domain: &Domain| !matches!(domain, Domain::Levels(_));
                if !numeric(&d.output_support) || !d.parents.iter().all(|p| numeric(&p.domain)) {
                    out.push(CoherenceFinding::new(
                        Shape,
                        "mechanism_composition.shape_invalid_domain",
                        v,
                        "a linear-Gaussian mechanism ranges over numeric domains",
                    ));
                }
            }
            MechanismForm::Tabular { levels, rows } => {
                check_table_shape(d, levels, rows, out);
            }
        }
    }
}

fn check_table_shape(
    d: &MechanismDescriptor,
    levels: &[String],
    rows: &[super::descriptor::TabularRow],
    out: &mut Vec<CoherenceFinding>,
) {
    use CoherenceKind::{Shape, Support};
    let v = d.variable.as_str();
    let table_defect = |why: &'static str| {
        CoherenceFinding::new(
            Shape,
            "mechanism_composition.shape_invalid_table",
            v,
            "supply a normalized conditional table over declared levels",
        )
        .comparing("a valid conditional table", why)
    };
    let mut sorted_levels: Vec<&String> = levels.iter().collect();
    sorted_levels.sort();
    if levels.is_empty()
        || levels.len() > MAX_LEVELS_PER_VARIABLE
        || sorted_levels.windows(2).any(|w| w[0] == w[1])
        || levels.iter().any(|l| blank(l))
    {
        out.push(table_defect("levels are empty, blank, duplicated or too many"));
        return;
    }
    match &d.output_support {
        Domain::Levels(support) if support.as_slice() == levels => {}
        _ => out.push(table_defect("output support is not the declared level set")),
    }
    let mut expected_rows = 1_usize;
    for p in &d.parents {
        if let Domain::Levels(l) = &p.domain {
            expected_rows = expected_rows.saturating_mul(l.len());
        } else {
            out.push(table_defect("a parent of a finite mechanism needs a level domain"));
            return;
        }
    }
    let parent_names: Vec<&str> = d.parents.iter().map(|p| p.variable.as_str()).collect();
    let mut seen: BTreeSet<Vec<(String, String)>> = BTreeSet::new();
    for row in rows {
        let keys: Vec<&str> = row.parent_levels.iter().map(|(k, _)| k.as_str()).collect();
        if keys != parent_names {
            out.push(table_defect("a row does not name exactly the declared parents"));
            return;
        }
        for (name, level) in &row.parent_levels {
            let known = d.parents.iter().any(|p| match &p.domain {
                Domain::Levels(l) => p.variable == *name && l.contains(level),
                _ => false,
            });
            if !known {
                out.push(table_defect("a row names a level outside the parent's domain"));
                return;
            }
        }
        let prob_keys: Vec<&str> = row.probabilities.iter().map(|(k, _)| k.as_str()).collect();
        let level_keys: Vec<&str> = levels.iter().map(String::as_str).collect();
        if prob_keys != level_keys {
            out.push(table_defect("a row does not give one probability per level"));
            return;
        }
        let probs: Vec<f64> = row.probabilities.iter().map(|(_, p)| *p).collect();
        let sum: f64 = probs.iter().sum();
        if !all_finite(&probs) || probs.iter().any(|p| *p < 0.0) || (sum - 1.0).abs() > 1e-9 {
            out.push(table_defect("a row is not a probability distribution"));
            return;
        }
        if !seen.insert(row.parent_levels.clone()) {
            out.push(table_defect("a parent configuration appears twice"));
            return;
        }
    }
    if rows.len() < expected_rows {
        out.push(
            CoherenceFinding::new(
                Support,
                "mechanism_composition.support_missing_parent_configuration",
                v,
                "supply a row for every combination of parent levels the producers can emit",
            )
            .comparing(format!("{expected_rows} rows"), rows.len().to_string()),
        );
    }
}

fn check_family(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    let families: BTreeSet<MechanismFamily> =
        spec.mechanisms.iter().map(|d| d.form.family()).collect();
    if families.len() > 1 {
        out.push(
            CoherenceFinding::new(
                CoherenceKind::Family,
                "mechanism_composition.family_mixed",
                "composition",
                "compose one family at a time: all linear-Gaussian or all finite tables",
            )
            .comparing("one mechanism family", format!("{families:?}")),
        );
    }
}

fn check_identity(
    spec: &ComposedModelSpec,
    known: &BTreeSet<&str>,
    out: &mut Vec<CoherenceFinding>,
) {
    use CoherenceKind::VariableIdentity;
    for w in spec.mechanisms.windows(2) {
        if w[0].variable == w[1].variable {
            out.push(CoherenceFinding::new(
                VariableIdentity,
                "mechanism_composition.identity_duplicate_mechanism",
                w[0].variable.clone(),
                "keep one fitted mechanism per variable, or compose the alternatives separately",
            ));
        }
    }
    for d in &spec.mechanisms {
        for p in &d.parents {
            if !known.contains(p.variable.as_str()) {
                out.push(CoherenceFinding::new(
                    VariableIdentity,
                    "mechanism_composition.identity_parent_without_mechanism",
                    format!("{}<-{}", d.variable, p.variable),
                    "supply a mechanism for the parent (a root needs its own marginal mechanism)",
                ));
            }
        }
    }
    for (parent, child) in &spec.edges {
        if !known.contains(parent.as_str()) || !known.contains(child.as_str()) {
            out.push(CoherenceFinding::new(
                VariableIdentity,
                "mechanism_composition.identity_edge_unknown_variable",
                format!("{parent}->{child}"),
                "declare edges only between variables that have a mechanism",
            ));
        }
    }
    for inv in &spec.invariances {
        if !known.contains(inv.variable.as_str()) {
            out.push(CoherenceFinding::new(
                VariableIdentity,
                "mechanism_composition.identity_declaration_unknown_variable",
                inv.variable.clone(),
                "declare invariance only for a variable that has a mechanism",
            ));
        }
    }
}

fn check_cycles(spec: &ComposedModelSpec, known: &BTreeSet<&str>, out: &mut Vec<CoherenceFinding>) {
    let mut edges: BTreeSet<(&str, &str)> = BTreeSet::new();
    for d in &spec.mechanisms {
        for p in &d.parents {
            edges.insert((p.variable.as_str(), d.variable.as_str()));
        }
    }
    for (p, c) in &spec.edges {
        if known.contains(p.as_str()) && known.contains(c.as_str()) {
            edges.insert((p.as_str(), c.as_str()));
        }
    }
    let mut indegree: BTreeMap<&str, usize> = known.iter().map(|v| (*v, 0)).collect();
    for (_, c) in &edges {
        if let Some(n) = indegree.get_mut(c) {
            *n += 1;
        }
    }
    let mut ready: Vec<&str> = indegree.iter().filter(|(_, n)| **n == 0).map(|(v, _)| *v).collect();
    let mut removed = 0_usize;
    while let Some(v) = ready.pop() {
        removed += 1;
        for (p, c) in &edges {
            if *p == v {
                if let Some(n) = indegree.get_mut(c) {
                    *n -= 1;
                    if *n == 0 {
                        ready.push(*c);
                    }
                }
            }
        }
    }
    if removed < indegree.len() {
        let stuck = indegree.iter().filter(|(_, n)| **n > 0).map(|(v, _)| (*v).to_owned());
        out.push(CoherenceFinding::new(
            CoherenceKind::Cycle,
            "mechanism_composition.cycle_directed_cycle",
            list(stuck),
            "only acyclic compositions are executed; a feedback system needs its own equilibrium model",
        ));
    }
}

fn check_parent_and_conditioning_sets(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    for d in &spec.mechanisms {
        let supplied: BTreeSet<String> = d.parents.iter().map(|p| p.variable.clone()).collect();
        let declared: BTreeSet<String> =
            spec.edges.iter().filter(|(_, c)| *c == d.variable).map(|(p, _)| p.clone()).collect();
        if supplied != declared {
            out.push(
                CoherenceFinding::new(
                    CoherenceKind::CausalParents,
                    "mechanism_composition.parents_declared_graph_mismatch",
                    d.variable.clone(),
                    "refit the mechanism on the declared parents, or correct the declared graph",
                )
                .comparing(list(declared), list(supplied.clone())),
            );
        }
        let conditioning: BTreeSet<String> = d.conditioning_set.iter().cloned().collect();
        let extra: Vec<String> = conditioning.difference(&supplied).cloned().collect();
        let missing: Vec<String> = supplied.difference(&conditioning).cloned().collect();
        if !extra.is_empty() {
            out.push(
                CoherenceFinding::new(
                    CoherenceKind::ConditioningSet,
                    "mechanism_composition.conditioning_extra_conditioner",
                    d.variable.clone(),
                    "refit without the extra conditioners (the fit is P(y | parents, extras), not the mechanism), or declare them as parents",
                )
                .comparing(list(supplied.clone()), list(conditioning.clone())),
            );
        }
        if !missing.is_empty() {
            out.push(
                CoherenceFinding::new(
                    CoherenceKind::ConditioningSet,
                    "mechanism_composition.conditioning_parent_not_conditioned",
                    d.variable.clone(),
                    "the fit must condition on every declared parent",
                )
                .comparing(list(supplied), list(conditioning)),
            );
        }
    }
}

fn check_units_transform_support(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    let by_var: BTreeMap<&str, &MechanismDescriptor> =
        spec.mechanisms.iter().map(|d| (d.variable.as_str(), d)).collect();
    for d in &spec.mechanisms {
        for p in &d.parents {
            let Some(producer) = by_var.get(p.variable.as_str()) else { continue };
            let edge = format!("{}->{}", p.variable, d.variable);
            if producer.output_units != p.units {
                out.push(
                    CoherenceFinding::new(
                        CoherenceKind::Units,
                        "mechanism_composition.units_mismatch",
                        edge.clone(),
                        "convert explicitly and refit, or refit the consumer in the producer's units; no implicit conversion is inferred",
                    )
                    .comparing(p.units.clone(), producer.output_units.clone()),
                );
            }
            if producer.output_transform != p.transform {
                out.push(
                    CoherenceFinding::new(
                        CoherenceKind::Transform,
                        "mechanism_composition.transform_mismatch",
                        edge.clone(),
                        "declare the transform the consumer was fitted on, or refit one side",
                    )
                    .comparing(p.transform.clone(), producer.output_transform.clone()),
                );
            }
            match producer.output_support.contained_in(&p.domain) {
                Containment::Contained => {}
                Containment::Partial => out.push(
                    CoherenceFinding::new(
                        CoherenceKind::Support,
                        "mechanism_composition.support_not_contained",
                        edge,
                        "restrict the producer to the consumer's fitted domain, refit the consumer on the producer's range, or declare global validity of the consumer (domain real)",
                    )
                    .comparing(p.domain.describe(), producer.output_support.describe()),
                ),
                Containment::Disjoint => out.push(
                    CoherenceFinding::new(
                        CoherenceKind::Support,
                        "mechanism_composition.support_disjoint",
                        edge,
                        "the producer never emits values the consumer was fitted on; the composition has no overlap",
                    )
                    .comparing(p.domain.describe(), producer.output_support.describe()),
                ),
                Containment::KindMismatch => out.push(
                    CoherenceFinding::new(
                        CoherenceKind::Support,
                        "mechanism_composition.support_kind_mismatch",
                        edge,
                        "a numeric output cannot feed a finite-level input or the reverse",
                    )
                    .comparing(p.domain.describe(), producer.output_support.describe()),
                ),
            }
        }
    }
}

fn check_context(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    for d in &spec.mechanisms {
        if d.context.population != spec.target.population {
            let invariant = spec.invariances.iter().any(|i| {
                i.variable == d.variable
                    && i.coordinate == ContextCoordinate::Population
                    && i.source == d.context.population
                    && i.target == spec.target.population
                    && !blank(&i.assumption_id)
            });
            if !invariant {
                let diagrams: Vec<_> = spec
                    .selection_diagrams
                    .iter()
                    .filter(|s| {
                        s.source_population == d.context.population
                            && s.target_population == spec.target.population
                    })
                    .collect();
                let compare = |f: CoherenceFinding| {
                    f.comparing(spec.target.population.clone(), d.context.population.clone())
                };
                if diagrams.is_empty() {
                    out.push(compare(CoherenceFinding::new(
                        CoherenceKind::Population,
                        "mechanism_composition.population_undeclared_transport",
                        d.variable.clone(),
                        "declare a selection diagram or an invariance assumption for this mechanism, or refit it in the target population",
                    )));
                } else if diagrams.iter().any(|s| {
                    s.selection_nodes.iter().any(|n| n.points_into.iter().any(|v| *v == d.variable))
                }) {
                    out.push(compare(CoherenceFinding::new(
                        CoherenceKind::Population,
                        "mechanism_composition.population_selection_arrow_into_mechanism",
                        d.variable.clone(),
                        "a selection node points into this variable, so its mechanism is not invariant; refit it in the target population",
                    )));
                } else if !d.latents.is_empty() {
                    out.push(compare(CoherenceFinding::new(
                        CoherenceKind::Population,
                        "mechanism_composition.population_latent_blocks_selection_admissibility",
                        d.variable.clone(),
                        "a mechanism with a shared latent cause is admitted only by an explicit invariance declaration",
                    )));
                }
            }
        }
        if d.context.regime != spec.target.regime {
            let invariant = spec.invariances.iter().any(|i| {
                i.variable == d.variable
                    && i.coordinate == ContextCoordinate::Regime
                    && i.source == d.context.regime
                    && i.target == spec.target.regime
                    && !blank(&i.assumption_id)
            });
            if !invariant {
                out.push(
                    CoherenceFinding::new(
                        CoherenceKind::Regime,
                        "mechanism_composition.regime_undeclared_invariance",
                        d.variable.clone(),
                        "declare that the mechanism is invariant between the fitted and target regime, or refit it in the target regime",
                    )
                    .comparing(spec.target.regime.clone(), d.context.regime.clone()),
                );
            }
        }
    }
}

fn check_latents(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    use CoherenceKind::SharedLatent;
    let mut users: BTreeMap<&str, Vec<&MechanismDescriptor>> = BTreeMap::new();
    for d in &spec.mechanisms {
        for l in &d.latents {
            users.entry(l.latent.as_str()).or_default().push(d);
        }
    }
    let known: BTreeSet<&str> = spec.mechanisms.iter().map(|d| d.variable.as_str()).collect();
    for d in &spec.mechanisms {
        for l in &d.latents {
            let subject = format!("{}:{}", l.latent, d.variable);
            if d.form.family() != MechanismFamily::LinearGaussian {
                out.push(CoherenceFinding::new(
                    SharedLatent,
                    "mechanism_composition.latent_family_unsupported",
                    subject.clone(),
                    "finite tables carry no latent loadings; marginalize the latent explicitly in a model that supports it",
                ));
            }
            match l.loading {
                None => out.push(CoherenceFinding::new(
                    SharedLatent,
                    "mechanism_composition.latent_loading_undeclared",
                    subject.clone(),
                    "declare the loading on the shared latent; its residual is otherwise correlated by an unknown amount and is never treated as independent",
                )),
                Some(x) if !x.is_finite() => out.push(CoherenceFinding::new(
                    SharedLatent,
                    "mechanism_composition.latent_loading_undeclared",
                    subject.clone(),
                    "declare a finite loading",
                )),
                Some(_) => {}
            }
            for other in &l.shared_with {
                if !known.contains(other.as_str()) {
                    out.push(CoherenceFinding::new(
                        SharedLatent,
                        "mechanism_composition.latent_shared_with_absent_mechanism",
                        format!("{}:{other}", l.latent),
                        "the latent is shared with a mechanism that is not in the composition; add that mechanism or the dropped latent would be treated as independent",
                    ));
                } else if let Some(peer) = spec.mechanisms.iter().find(|m| m.variable == *other) {
                    if !peer.latents.iter().any(|x| x.latent == l.latent) {
                        out.push(CoherenceFinding::new(
                            SharedLatent,
                            "mechanism_composition.latent_sharing_not_reciprocal",
                            format!("{}:{other}", l.latent),
                            "both mechanisms must declare the shared latent",
                        ));
                    }
                }
            }
        }
    }
    for (latent, ms) in &users {
        if ms.len() > 1 {
            for m in ms {
                let others: BTreeSet<&str> = ms
                    .iter()
                    .filter(|o| o.variable != m.variable)
                    .map(|o| o.variable.as_str())
                    .collect();
                let declared: BTreeSet<&str> = m
                    .latents
                    .iter()
                    .filter(|l| l.latent == *latent)
                    .flat_map(|l| l.shared_with.iter().map(String::as_str))
                    .collect();
                if !others.is_subset(&declared) {
                    out.push(CoherenceFinding::new(
                        SharedLatent,
                        "mechanism_composition.latent_sharing_not_declared",
                        format!("{latent}:{}", m.variable),
                        "list every mechanism that depends on the latent; undeclared sharing would be treated as independent",
                    ));
                }
            }
        }
    }
    let distinct = users.len();
    if distinct > MAX_COMPOSED_LATENTS {
        out.push(CoherenceFinding::new(
            SharedLatent,
            "mechanism_composition.latent_too_many",
            "composition",
            "reduce the number of declared latent causes",
        ));
    }
}

fn check_provenance(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    use CoherenceKind::Provenance;
    for d in &spec.mechanisms {
        if blank(&d.provenance.artifact_id) {
            out.push(CoherenceFinding::new(
                Provenance,
                "mechanism_composition.provenance_missing_artifact",
                d.variable.clone(),
                "bind the mechanism to the identity of the fit or external binding that produced it",
            ));
        }
        if blank(&d.provenance.evidence_id) {
            out.push(CoherenceFinding::new(
                Provenance,
                "mechanism_composition.provenance_missing_evidence",
                d.variable.clone(),
                "name the evidence the mechanism was estimated from so dependence can be judged",
            ));
        }
        if d.provenance.origin == MechanismOrigin::Declared && !spec.admit_declared_mechanisms {
            out.push(CoherenceFinding::new(
                Provenance,
                "mechanism_composition.provenance_declared_mechanism_unacknowledged",
                d.variable.clone(),
                "a declared (not estimated) mechanism is admitted only when the composition acknowledges it",
            ));
        }
    }
}

fn check_uncertainty_marginal(
    d: &MechanismDescriptor,
    u: &MechanismUncertainty,
    out: &mut Vec<CoherenceFinding>,
) {
    use CoherenceKind::UncertaintyAlignment;
    let v = d.variable.clone();
    if d.form.family() != MechanismFamily::LinearGaussian {
        out.push(CoherenceFinding::new(
            UncertaintyAlignment,
            "mechanism_composition.uncertainty_family_unsupported",
            v,
            "parameter uncertainty is propagated for linear-Gaussian mechanisms only; the finite-table route is point-only",
        ));
        return;
    }
    let expected = parameter_terms(d);
    if u.parameters != expected {
        out.push(
            CoherenceFinding::new(
                UncertaintyAlignment,
                "mechanism_composition.uncertainty_parameter_set_mismatch",
                v,
                "report covariance over exactly the intercept and one coefficient per declared parent",
            )
            .comparing(format!("{expected:?}"), format!("{:?}", u.parameters)),
        );
        return;
    }
    let k = expected.len();
    if u.covariance.len() != k * k || !all_finite(&u.covariance) {
        out.push(CoherenceFinding::new(
            UncertaintyAlignment,
            "mechanism_composition.uncertainty_shape",
            v,
            "supply a finite k x k covariance",
        ));
        return;
    }
    if !is_symmetric(&u.covariance, k) {
        out.push(CoherenceFinding::new(
            UncertaintyAlignment,
            "mechanism_composition.uncertainty_not_symmetric",
            v,
            "supply a symmetric covariance",
        ));
    } else if !is_positive_semidefinite(&u.covariance, k) {
        out.push(CoherenceFinding::new(
            UncertaintyAlignment,
            "mechanism_composition.uncertainty_not_psd",
            v,
            "supply a positive semidefinite covariance",
        ));
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent uncertainty-alignment and posterior-dependence checks, each pushing its own findings"
)]
fn check_uncertainty(spec: &ComposedModelSpec, out: &mut Vec<CoherenceFinding>) {
    use CoherenceKind::{PosteriorDependence as Dep, UncertaintyAlignment as Align};
    let mut meanings = BTreeSet::new();
    for d in &spec.mechanisms {
        if let Some(u) = &d.uncertainty {
            meanings.insert(u.meaning);
            check_uncertainty_marginal(d, u, out);
        }
    }
    if let PosteriorDependence::Joint { meaning, .. } = &spec.dependence {
        meanings.insert(*meaning);
    }
    if meanings.len() > 1 {
        out.push(
            CoherenceFinding::new(
                Align,
                "mechanism_composition.uncertainty_meaning_mismatch",
                "composition",
                "a posterior covariance and a sampling covariance are not one law; supply laws of one meaning",
            )
            .comparing("one uncertainty meaning", format!("{meanings:?}")),
        );
    }
    match &spec.dependence {
        PosteriorDependence::Unknown => {}
        PosteriorDependence::IndependentByDesign { justification } => {
            if blank(justification) {
                out.push(CoherenceFinding::new(
                    Dep,
                    "mechanism_composition.posterior_dependence_independence_unjustified",
                    "composition",
                    "state why the estimates are independent (disjoint, independently generated evidence)",
                ));
            }
            let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
            for d in &spec.mechanisms {
                let e = d.provenance.evidence_id.as_str();
                if blank(e) {
                    continue;
                }
                if let Some(first) = seen.insert(e, d.variable.as_str()) {
                    out.push(
                        CoherenceFinding::new(
                            Dep,
                            "mechanism_composition.posterior_dependence_shared_evidence_declared_independent",
                            format!("{first}|{}", d.variable),
                            "mechanisms estimated from the same evidence are dependent; supply a joint law or cross-mechanism covariance",
                        )
                        .comparing("distinct evidence identities", e.to_owned()),
                    );
                }
            }
        }
        PosteriorDependence::Joint { parameters, covariance, .. } => {
            let layout = parameter_layout(spec);
            let k = parameters.len();
            if spec.mechanisms.iter().any(|d| d.form.family() != MechanismFamily::LinearGaussian) {
                out.push(CoherenceFinding::new(
                    Align,
                    "mechanism_composition.uncertainty_family_unsupported",
                    "composition",
                    "parameter uncertainty is propagated for linear-Gaussian mechanisms only; the finite-table route is point-only",
                ));
                return;
            }
            let given: BTreeSet<&ParameterId> = parameters.iter().collect();
            let expected: BTreeSet<&ParameterId> = layout.iter().collect();
            if given.len() != k || given != expected {
                out.push(
                    CoherenceFinding::new(
                        Align,
                        "mechanism_composition.uncertainty_joint_parameter_set_mismatch",
                        "composition",
                        "the joint law must cover exactly every parameter of every mechanism once",
                    )
                    .comparing(layout.len().to_string(), k.to_string()),
                );
            } else if covariance.len() != k * k || !all_finite(covariance) {
                out.push(CoherenceFinding::new(
                    Align,
                    "mechanism_composition.uncertainty_shape",
                    "composition",
                    "supply a finite k x k joint covariance",
                ));
            } else if !is_symmetric(covariance, k) {
                out.push(CoherenceFinding::new(
                    Align,
                    "mechanism_composition.uncertainty_not_symmetric",
                    "composition",
                    "supply a symmetric covariance",
                ));
            } else if !is_positive_semidefinite(covariance, k) {
                out.push(CoherenceFinding::new(
                    Align,
                    "mechanism_composition.uncertainty_not_psd",
                    "composition",
                    "supply a positive semidefinite covariance",
                ));
            } else {
                // The joint law must have each mechanism's reported marginal.
                let index: BTreeMap<&ParameterId, usize> =
                    parameters.iter().enumerate().map(|(i, p)| (p, i)).collect();
                for d in &spec.mechanisms {
                    let Some(u) = &d.uncertainty else { continue };
                    if u.parameters != parameter_terms(d) {
                        continue;
                    }
                    let kk = u.parameters.len();
                    'blocks: for a in 0..kk {
                        for b in 0..kk {
                            let ia = index[&ParameterId {
                                mechanism: d.variable.clone(),
                                term: u.parameters[a].clone(),
                            }];
                            let ib = index[&ParameterId {
                                mechanism: d.variable.clone(),
                                term: u.parameters[b].clone(),
                            }];
                            let joint = covariance[ia * k + ib];
                            let own = u.covariance[a * kk + b];
                            if (joint - own).abs() > COVARIANCE_TOLERANCE * (1.0 + own.abs()) {
                                out.push(
                                    CoherenceFinding::new(
                                        Align,
                                        "mechanism_composition.uncertainty_marginal_inconsistent_with_joint",
                                        d.variable.clone(),
                                        "the joint law and the mechanism's own fit disagree; they are not one posterior",
                                    )
                                    .comparing(own.to_string(), joint.to_string()),
                                );
                                break 'blocks;
                            }
                        }
                    }
                }
            }
        }
        PosteriorDependence::CrossCovariance { blocks } => {
            check_cross_blocks(spec, blocks, out);
        }
    }
}

fn check_cross_blocks(
    spec: &ComposedModelSpec,
    blocks: &[CrossCovarianceBlock],
    out: &mut Vec<CoherenceFinding>,
) {
    use CoherenceKind::{PosteriorDependence as Dep, UncertaintyAlignment as Align};
    let by_var: BTreeMap<&str, &MechanismDescriptor> =
        spec.mechanisms.iter().map(|d| (d.variable.as_str(), d)).collect();
    let mut seen: BTreeSet<(&str, &str)> = BTreeSet::new();
    for b in blocks {
        let subject = format!("{}|{}", b.first, b.second);
        let (Some(first), Some(second)) =
            (by_var.get(b.first.as_str()), by_var.get(b.second.as_str()))
        else {
            out.push(CoherenceFinding::new(
                Align,
                "mechanism_composition.uncertainty_cross_block_unknown_mechanism",
                subject,
                "name mechanisms that are in the composition",
            ));
            continue;
        };
        if b.first == b.second || !seen.insert((b.first.as_str(), b.second.as_str())) {
            out.push(CoherenceFinding::new(
                Dep,
                "mechanism_composition.posterior_dependence_cross_block_duplicate",
                subject,
                "give each pair of distinct mechanisms one block",
            ));
            continue;
        }
        if b.first_parameters != parameter_terms(first)
            || b.second_parameters != parameter_terms(second)
            || b.values.len() != b.first_parameters.len() * b.second_parameters.len()
            || !all_finite(&b.values)
        {
            out.push(CoherenceFinding::new(
                Align,
                "mechanism_composition.uncertainty_cross_block_shape",
                subject,
                "a cross block is over exactly the two mechanisms' parameters, finite, rows x columns",
            ));
        }
    }
}

/// Check a composition and return every defect found, in a fixed order that
/// does not depend on how the specification listed its mechanisms.
///
/// An empty result means the declared mechanisms establish one acyclic joint
/// law over the declared graph. It does not mean parameter uncertainty can be
/// propagated; that needs [`uncertainty_findings`] to be empty as well.
#[must_use]
pub fn check_composition(spec: &ComposedModelSpec) -> Vec<CoherenceFinding> {
    let spec = spec.canonical();
    let mut out = Vec::new();
    check_shape(&spec, &mut out);
    if !out.is_empty() {
        return finish(out);
    }
    check_family(&spec, &mut out);
    let known: BTreeSet<&str> = spec.mechanisms.iter().map(|d| d.variable.as_str()).collect();
    check_identity(&spec, &known, &mut out);
    check_cycles(&spec, &known, &mut out);
    check_parent_and_conditioning_sets(&spec, &mut out);
    check_units_transform_support(&spec, &mut out);
    check_context(&spec, &mut out);
    check_latents(&spec, &mut out);
    check_provenance(&spec, &mut out);
    check_uncertainty(&spec, &mut out);
    finish(out)
}

fn finish(mut findings: Vec<CoherenceFinding>) -> Vec<CoherenceFinding> {
    findings.sort_by(|a, b| (a.kind, a.detail, &a.subject).cmp(&(b.kind, b.detail, &b.subject)));
    findings.dedup();
    findings
}

/// Findings that block a parameter-uncertainty claim on a composition that
/// already compiles: a missing marginal law, or dependence that is unknown.
///
/// Unknown dependence is never converted into independence; with more than one
/// mechanism it is a [`CoherenceKind::PosteriorDependence`] finding.
#[must_use]
pub fn uncertainty_findings(spec: &ComposedModelSpec) -> Vec<CoherenceFinding> {
    let spec = spec.canonical();
    let mut out = Vec::new();
    let provided = |d: &MechanismDescriptor| d.uncertainty.is_some();
    let joint = matches!(spec.dependence, PosteriorDependence::Joint { .. });
    if !joint {
        for d in &spec.mechanisms {
            if !provided(d) {
                out.push(CoherenceFinding::new(
                    CoherenceKind::UncertaintyAlignment,
                    "mechanism_composition.uncertainty_marginal_missing",
                    d.variable.clone(),
                    "the owning fit must report its parameter covariance",
                ));
            }
        }
    }
    if spec.mechanisms.len() > 1 {
        match &spec.dependence {
            PosteriorDependence::Unknown => out.push(CoherenceFinding::new(
                CoherenceKind::PosteriorDependence,
                "mechanism_composition.posterior_dependence_unknown",
                "composition",
                "supply an aligned joint law over all parameters, or an explicit cross-mechanism covariance for every pair; independence must be declared with its justification, never assumed",
            )),
            PosteriorDependence::CrossCovariance { blocks } => {
                let have: BTreeSet<(&str, &str)> =
                    blocks.iter().map(|b| (b.first.as_str(), b.second.as_str())).collect();
                let names: Vec<&str> = spec.mechanisms.iter().map(|d| d.variable.as_str()).collect();
                for i in 0..names.len() {
                    for j in (i + 1)..names.len() {
                        if !have.contains(&(names[i], names[j])) && !have.contains(&(names[j], names[i])) {
                            out.push(CoherenceFinding::new(
                                CoherenceKind::PosteriorDependence,
                                "mechanism_composition.posterior_dependence_cross_covariance_missing",
                                format!("{}|{}", names[i], names[j]),
                                "give every pair of mechanisms an explicit covariance block, a zero block being a declared independence for that pair",
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    finish(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psd_accepts_semidefinite_and_rejects_indefinite() {
        assert!(is_positive_semidefinite(&[1.0, 1.0, 1.0, 1.0], 2));
        assert!(is_positive_semidefinite(&[2.0, 0.5, 0.5, 1.0], 2));
        assert!(!is_positive_semidefinite(&[1.0, 2.0, 2.0, 1.0], 2));
        assert!(!is_positive_semidefinite(&[-1.0], 1));
        assert!(is_positive_semidefinite(&[0.0], 1));
    }

    #[test]
    fn every_kind_has_a_unique_tag() {
        let mut tags: Vec<&str> = CoherenceKind::ALL.iter().map(|k| k.tag()).collect();
        let n = tags.len();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), n);
    }
}
