//! The transported static path-specific counterfactual: derivation and evaluator for
//! the narrowest honest class.
//!
//! # The class
//!
//! Two populations, source `S` and target `T`, share one structural model over
//!
//! * covariates `Z`: pre-treatment variables (non-descendants of the treatment)
//!   whose joint law `P_pi(Z)` is a finite-support law that may differ between
//!   populations;
//! * a treatment `A` that is only ever *set* (its own mechanism is never used);
//! * mechanism nodes `V_1..V_k` (mediators and the outcome `Y`) with
//!   `V_i = alpha_i(Z) + sum_{p in pa_i} beta_{i,p}(Z) * V_p + U_i`, where
//!   `alpha_i` and each `beta_{i,p}` are affine in `Z`, a parent `p` may be a
//!   covariate, `A` or another mechanism node, and the noises `U` are mutually
//!   independent and independent of `Z` (a static, fully observed Markovian DAG).
//!
//! A selection diagram marks, with a selection node, every variable whose
//! mechanism (or covariate law) may differ between `S` and `T`.
//!
//! An *edge assignment* `g` is the set of children of `A` that are fed the treated
//! value `a1`; every other child of `A` is fed the control value `a0`. The world
//! `Y_g(z, u)` evaluates the equations in topological order, each node reading its
//! mechanism parents from this same world except that the edge `A -> c` carries
//! `a1` when `c` is in `g` and `a0` otherwise. The path-specific edge-intervention
//! contrast of two assignments is `theta_pi = E_pi[Y_{g+} - Y_{g-}]` with
//! `(Z, U) ~ P_pi(Z) x P(U)` and the same `u` fed to both worlds (the
//! shared-exogenous coupling of the fixed-population prerequisite). The natural
//! direct effect is `g+ = {Y}`, `g- = {}`; the natural indirect effect is
//! `g+ = {all}`, `g- = {Y}` (for a `Y` that is a child of `A`).
//!
//! # Theorem
//!
//! Assume
//!
//! 1. (checked from the diagram) every selection node points only to a covariate
//!    or to `A`; none points to a mediator or to the outcome;
//! 2. (declared) additive noise: the equations have the stated form with the noise
//!    entering each mechanism additively;
//! 3. (declared) the noise laws `P(U_i)` are the same in `S` and `T`, and the
//!    noises are independent of `Z` and of each other;
//! 4. (declared) the unit-level cross-world coupling: one `u` per unit is fed to
//!    both worlds of the contrast, the shared-exogenous semantics the
//!    fixed-population route uses;
//! 5. (declared, evidenced by the supplied source fit) the supplied coefficients
//!    are the structural coefficients, which in a Markovian additive model is what
//!    regression on the source identifies given source positivity of `A`;
//! 6. (checked) the support of `P_T(Z)` lies in the support of `P_S(Z)`.
//!
//! Then `theta_T = sum_z P_T(z) * G(z)` where `G(z) = Y_{g+}(z, u) - Y_{g-}(z, u)`
//! for any `u`, computed from the source equations alone.
//!
//! ## Proof
//!
//! 1. *The contrast does not depend on `u`.* By induction along the topological
//!    order, each `V_i^g(z, u) = kappa_i^g(z) + sum_j rho_{ij}(z) u_j`, where
//!    `rho_i = e_i + sum_{p mechanism} beta_{i,p}(z) rho_p` is a sum over directed
//!    mechanism paths and involves neither `g` nor `a1, a0`: the assignment changes
//!    only which constant (`a1` or `a0`) the root `A` feeds an edge, and `A` carries
//!    no noise. Hence `Y_{g+} - Y_{g-} = kappa_Y^{g+}(z) - kappa_Y^{g-}(z) = G(z)`,
//!    the same for every `u`. This is the step that uses additivity (affine in the
//!    endogenous parents).
//! 2. *Factorisation.* `(Z, U)` has law `P_pi(Z) x P(U)` (assumption 3 and, for the
//!    target, assumption 1, which keeps `Z` independent of `U`), so
//!    `theta_pi = sum_z P_pi(z) E_U[G(z)] = sum_z P_pi(z) G(z)` by step 1.
//! 3. *Invariance.* `alpha_i, beta_{i,p}` for the mediator and outcome nodes are
//!    the same functions in `S` and `T` (assumption 1: no selection node points at
//!    them), and `U` has the same law (assumption 3). So `G` is one function, and
//!    only `P_pi(Z)` varies with `pi`.
//! 4. *Identification of `G`.* `G` is a polynomial in the structural coefficients
//!    (assumption 5), so it is computed from the supplied source fit; the
//!    coefficient functions of `z` are evidenced only on the source support, so
//!    `G` is licensed on `supp P_T(Z)` exactly when that lies in `supp P_S(Z)`
//!    (assumption 6; extrapolation of the affine form is not licensed).
//! 5. *Composition of the prerequisites.* `G(z)` is the fixed-population,
//!    unit-level path-specific edge-intervention contrast (Pearl 2001 direct and
//!    indirect effects; the edge-set generalisation of Avin, Shpitser and Pearl
//!    2005); `sum_z P_T(z) (.)` is the covariate-selected transport formula
//!    `P*(y | do(x)) = sum_z P(y | do(x), z) P*(z)` of Bareinboim and Pearl for an
//!    S-admissible set `Z`. Neither alone yields the answer; the second applies to
//!    the first because `G` is a function of `z` alone (step 1), which is exactly
//!    what the cross-world coupling would otherwise obstruct. QED.
//!
//! # What is not covered, and why a selection on a mechanism is refused
//!
//! * Nonlinear additive-noise mechanisms need the noise law inside `G` and a second
//!   proof; they are not implemented, and a caller who declares `additive_noise =
//!   false` is refused with `nonadditive_mechanism`.
//! * Nonparametric identification from data (recanting witnesses and the
//!   cross-world independence of an unspecified model) is not claimed: the theorem
//!   is relative to the fully specified structural fit.
//! * A selection node on a mediator or the outcome mechanism is outside the class:
//!   two models can agree on the whole source population and differ on the target
//!   contrast. When the contrast is sensitive to one coefficient of the selected
//!   mechanism the refusal carries the explicit two-model witness
//!   ([`NonRecoverableWitness`]: same source model, two target models that differ in
//!   that one coefficient, contrasts verified by arithmetic) and the code
//!   `transport_proven_non_transportable`; otherwise the refusal is
//!   `cell_not_licensed` with no witness (the class excludes it, no impossibility
//!   is claimed).
//! * The claim is point-only; calibration is unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::StructuredRefusal;

use crate::transported_gate::{
    PopulationRole, RegimeFactorKey, TransportedCounterfactualPrerequisites,
    refuse_transported_counterfactual,
};

const POINT_TOLERANCE: f64 = 1e-12;
const WEIGHT_TOLERANCE: f64 = 1e-9;
const SENSITIVITY_TOLERANCE: f64 = 1e-9;

/// An affine function of the covariates: `constant + sum slope_k * z_k`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Affine {
    /// Value at `z = 0`.
    pub constant: f64,
    /// Per-covariate slopes.
    pub covariate_slopes: Vec<(String, f64)>,
}

impl Affine {
    /// A constant.
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self { constant: value, covariate_slopes: Vec::new() }
    }

    fn at(&self, z: &BTreeMap<String, f64>) -> Result<f64, String> {
        let mut value = self.constant;
        for (name, slope) in &self.covariate_slopes {
            let x = z.get(name).ok_or_else(|| format!("covariate {name} has no value"))?;
            value += slope * x;
        }
        Ok(value)
    }
}

/// `V = intercept(z) + sum_p coefficient_p(z) * V_p + U`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinearMechanism {
    /// The covariate-dependent intercept.
    pub intercept: Affine,
    /// Parents with their covariate-dependent coefficients.
    pub parents: Vec<(String, Affine)>,
}

/// The source structural fit.
#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveLinearScm {
    /// The treatment `A`.
    pub treatment: String,
    /// Covariate names (no mechanism; their law is supplied).
    pub covariates: Vec<String>,
    /// Mediator and outcome mechanisms in any order.
    pub mechanisms: Vec<(String, LinearMechanism)>,
}

/// A finite-support covariate law.
#[derive(Clone, Debug, PartialEq)]
pub struct CovariateLaw {
    /// Support points (covariate name to value) with positive weights summing to one.
    pub points: Vec<(BTreeMap<String, f64>, f64)>,
}

/// A selection node pointing to a variable whose mechanism or law may differ.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionNode {
    /// The selection node's label.
    pub label: String,
    /// The variable it points to.
    pub target: String,
}

/// The finite source/target selection diagram.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SelectionDiagram {
    /// Selection nodes.
    pub selections: Vec<SelectionNode>,
}

/// Which children of the treatment are fed the treated value; the others get the control.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EdgeAssignment {
    /// Children `c` of `A` whose edge `A -> c` carries the treated value.
    pub treated: BTreeSet<String>,
}

impl EdgeAssignment {
    /// Every `A`-edge carries the control value.
    #[must_use]
    pub fn all_control() -> Self {
        Self::default()
    }

    /// The listed children carry the treated value.
    #[must_use]
    pub fn treated_on<I, S>(children: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self { treated: children.into_iter().map(Into::into).collect() }
    }
}

/// One path-specific edge-intervention mean contrast `E[Y_plus - Y_minus]`.
#[derive(Clone, Debug, PartialEq)]
pub struct PathSpecificQuery {
    /// The outcome node.
    pub outcome: String,
    /// The treated value `a1`.
    pub treated_value: f64,
    /// The control value `a0`.
    pub control_value: f64,
    /// The added world's assignment.
    pub plus: EdgeAssignment,
    /// The subtracted world's assignment.
    pub minus: EdgeAssignment,
}

/// The premises the caller declares; each defaults to undeclared.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeclaredAssumptions {
    /// Additive noise in the mediator and outcome mechanisms.
    pub additive_noise: bool,
    /// Same noise laws in both populations, independent of covariates.
    pub noise_laws_shared: bool,
    /// One exogenous draw per unit is fed to both worlds of the contrast.
    pub cross_world_independence: bool,
}

/// Everything the evaluator needs.
#[derive(Clone, Copy, Debug)]
pub struct TransportedPathSpecificInput<'a> {
    /// The source structural fit.
    pub model: &'a AdditiveLinearScm,
    /// The selection diagram.
    pub diagram: &'a SelectionDiagram,
    /// The source covariate law.
    pub source_law: &'a CovariateLaw,
    /// The target covariate law.
    pub target_law: &'a CovariateLaw,
    /// The query.
    pub query: &'a PathSpecificQuery,
    /// Declared premises.
    pub assumptions: DeclaredAssumptions,
    /// Source and target regime evidence, by regime factor.
    pub supplied_factors: &'a BTreeMap<RegimeFactorKey, String>,
}

/// The target-law contribution of one support point.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitContrast {
    /// The covariate point.
    pub point: BTreeMap<String, f64>,
    /// Its target weight.
    pub target_weight: f64,
    /// The unit-level contrast `G(z)`.
    pub contrast: f64,
}

/// The record of what was checked and what was only declared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportedDerivation {
    /// The theorem used, by module-doc name.
    pub theorem: &'static str,
    /// Premises checked from the supplied structure.
    pub checked: Vec<&'static str>,
    /// Premises declared by the caller and not checkable here.
    pub declared: Vec<&'static str>,
    /// `point_only`.
    pub claim: &'static str,
}

/// The target path-specific mean contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedPathSpecificResult {
    /// `sum_z P_T(z) G(z)`.
    pub target_contrast: f64,
    /// `sum_z P_S(z) G(z)`, the answer if the source law were mistaken for the target's.
    pub source_contrast: f64,
    /// `G` at every target support point.
    pub unit_contrasts: Vec<UnitContrast>,
    /// What was checked and declared.
    pub derivation: TransportedDerivation,
}

/// Two target models that agree with the source and differ on the target contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct NonRecoverableWitness {
    /// The selected mechanism node.
    pub selected_node: String,
    /// The parent whose coefficient differs between the two target models.
    pub perturbed_parent: String,
    /// `None` when the constant part differs, else the covariate whose slope differs.
    pub perturbed_slope_covariate: Option<String>,
    /// The added amount.
    pub perturbation: f64,
    /// The source model, shared by both candidate worlds.
    pub source_model: AdditiveLinearScm,
    /// The first target model (equal to the source model).
    pub target_model_a: AdditiveLinearScm,
    /// The second target model.
    pub target_model_b: AdditiveLinearScm,
    /// The source contrast, identical under both.
    pub source_contrast: f64,
    /// The target contrast under the first model.
    pub target_contrast_a: f64,
    /// The target contrast under the second model.
    pub target_contrast_b: f64,
}

/// A typed refusal.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedPathSpecificRefusal {
    /// Registered code, stage and `transported_counterfactual.*` detail.
    pub refusal: StructuredRefusal,
    /// The impossibility witness, for a selection on a sensitive mechanism.
    pub witness: Option<NonRecoverableWitness>,
    /// Regime factors absent from the supplied evidence.
    pub missing_factors: Vec<RegimeFactorKey>,
}

fn refuse_with(
    code: &'static str,
    detail: &str,
    offending: Option<String>,
    remedy: &'static str,
) -> Box<TransportedPathSpecificRefusal> {
    Box::new(TransportedPathSpecificRefusal {
        refusal: StructuredRefusal {
            code,
            stage: "identify",
            detail: detail.to_owned(),
            offending,
            expected: None,
            supplied: None,
            capability: None,
            remedy: Some(remedy),
        },
        witness: None,
        missing_factors: Vec::new(),
    })
}

fn invalid(detail: &str, message: String) -> Box<TransportedPathSpecificRefusal> {
    refuse_with(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        Some(message),
        "supply a well-formed model, query, covariate laws and selection diagram",
    )
}

/// Topological order of the mechanism nodes after structural validation.
fn validate_model(model: &AdditiveLinearScm) -> Result<Vec<String>, String> {
    let mut names: BTreeSet<&str> = BTreeSet::new();
    names.insert(model.treatment.as_str());
    for covariate in &model.covariates {
        if !names.insert(covariate.as_str()) {
            return Err(format!("duplicate variable {covariate}"));
        }
    }
    let mut mechanism_names: BTreeSet<&str> = BTreeSet::new();
    for (name, _) in &model.mechanisms {
        if !names.insert(name.as_str()) {
            return Err(format!("duplicate variable {name}"));
        }
        mechanism_names.insert(name.as_str());
    }
    let covariates: BTreeSet<&str> = model.covariates.iter().map(String::as_str).collect();
    for (name, mechanism) in &model.mechanisms {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for (parent, coefficient) in &mechanism.parents {
            if !names.contains(parent.as_str()) || !seen.insert(parent.as_str()) {
                return Err(format!("mechanism {name} has an unknown or repeated parent {parent}"));
            }
            check_slopes(name, coefficient, &covariates)?;
        }
        check_slopes(name, &mechanism.intercept, &covariates)?;
    }
    topological_order(model, &mechanism_names)
}

fn check_slopes(name: &str, affine: &Affine, covariates: &BTreeSet<&str>) -> Result<(), String> {
    for (covariate, slope) in &affine.covariate_slopes {
        if !covariates.contains(covariate.as_str()) || !slope.is_finite() {
            return Err(format!("mechanism {name} has an invalid slope on {covariate}"));
        }
    }
    if affine.constant.is_finite() {
        Ok(())
    } else {
        Err(format!("mechanism {name} is not finite"))
    }
}

fn topological_order(
    model: &AdditiveLinearScm,
    mechanism_names: &BTreeSet<&str>,
) -> Result<Vec<String>, String> {
    let by_name: BTreeMap<&str, &LinearMechanism> =
        model.mechanisms.iter().map(|(n, m)| (n.as_str(), m)).collect();
    let mut order: Vec<String> = Vec::new();
    let mut done: BTreeSet<&str> = BTreeSet::new();
    while done.len() < mechanism_names.len() {
        let next = by_name.iter().find(|(name, mechanism)| {
            !done.contains(**name)
                && mechanism.parents.iter().all(|(p, _)| {
                    !mechanism_names.contains(p.as_str()) || done.contains(p.as_str())
                })
        });
        let Some((name, _)) = next else {
            return Err("the mechanism graph has a cycle".to_owned());
        };
        done.insert(*name);
        order.push((*name).to_owned());
    }
    Ok(order)
}

fn validate_query(model: &AdditiveLinearScm, query: &PathSpecificQuery) -> Result<(), String> {
    if !model.mechanisms.iter().any(|(n, _)| *n == query.outcome) {
        return Err(format!("outcome {} is not a mechanism node", query.outcome));
    }
    if !query.treated_value.is_finite() || !query.control_value.is_finite() {
        return Err("treatment values must be finite".to_owned());
    }
    for child in query.plus.treated.iter().chain(query.minus.treated.iter()) {
        let is_child = model
            .mechanisms
            .iter()
            .any(|(n, m)| n == child && m.parents.iter().any(|(p, _)| *p == model.treatment));
        if !is_child {
            return Err(format!("{child} is not a child of the treatment"));
        }
    }
    Ok(())
}

fn validate_law(
    label: &str,
    law: &CovariateLaw,
    covariates: &BTreeSet<&str>,
) -> Result<(), String> {
    if law.points.is_empty() {
        return Err(format!("{label} law has no support point"));
    }
    let mut total = 0.0;
    for (point, weight) in &law.points {
        if !(weight.is_finite() && *weight > 0.0) {
            return Err(format!("{label} law has a non-positive or non-finite weight"));
        }
        let complete = point.len() == covariates.len()
            && covariates.iter().all(|c| point.get(*c).is_some_and(|v| v.is_finite()));
        if !complete {
            return Err(format!("{label} law has a point that does not match the covariates"));
        }
        total += weight;
    }
    if (total - 1.0).abs() > WEIGHT_TOLERANCE {
        return Err(format!("{label} law weights sum to {total}, not one"));
    }
    Ok(())
}

fn world_outcome(
    model: &AdditiveLinearScm,
    query: &PathSpecificQuery,
    order: &[String],
    assignment: &EdgeAssignment,
    z: &BTreeMap<String, f64>,
) -> Result<f64, String> {
    let by_name: BTreeMap<&str, &LinearMechanism> =
        model.mechanisms.iter().map(|(n, m)| (n.as_str(), m)).collect();
    let mut values: BTreeMap<&str, f64> = BTreeMap::new();
    for name in order {
        let mechanism =
            by_name.get(name.as_str()).ok_or_else(|| format!("no mechanism for {name}"))?;
        let mut value = mechanism.intercept.at(z)?;
        for (parent, coefficient) in &mechanism.parents {
            let parent_value = if *parent == model.treatment {
                if assignment.treated.contains(name) {
                    query.treated_value
                } else {
                    query.control_value
                }
            } else if let Some(x) = values.get(parent.as_str()) {
                *x
            } else {
                *z.get(parent).ok_or_else(|| format!("{parent} has no value"))?
            };
            value += coefficient.at(z)? * parent_value;
        }
        values.insert(name.as_str(), value);
    }
    values.get(query.outcome.as_str()).copied().ok_or_else(|| "outcome not evaluated".to_owned())
}

fn unit_contrast(
    model: &AdditiveLinearScm,
    query: &PathSpecificQuery,
    order: &[String],
    z: &BTreeMap<String, f64>,
) -> Result<f64, String> {
    let plus = world_outcome(model, query, order, &query.plus, z)?;
    let minus = world_outcome(model, query, order, &query.minus, z)?;
    Ok(plus - minus)
}

fn law_contrast(
    model: &AdditiveLinearScm,
    query: &PathSpecificQuery,
    order: &[String],
    law: &CovariateLaw,
) -> Result<f64, String> {
    let mut total = 0.0;
    for (point, weight) in &law.points {
        total += weight * unit_contrast(model, query, order, point)?;
    }
    Ok(total)
}

fn points_match(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> bool {
    a.len() == b.len()
        && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| (v - w).abs() <= POINT_TOLERANCE))
}

fn describe(point: &BTreeMap<String, f64>) -> String {
    point.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",")
}

/// A perturbed copy of `model`: `mechanism`'s coefficient on `parent` (constant or the
/// slope on `slope`) is raised by `delta`.
fn perturbed(
    model: &AdditiveLinearScm,
    mechanism: &str,
    parent: &str,
    slope: Option<&str>,
    delta: f64,
) -> AdditiveLinearScm {
    let mut copy = model.clone();
    for (name, spec) in &mut copy.mechanisms {
        if name.as_str() != mechanism {
            continue;
        }
        for (p, coefficient) in &mut spec.parents {
            if p.as_str() != parent {
                continue;
            }
            match slope {
                None => coefficient.constant += delta,
                Some(covariate) => {
                    for (c, s) in &mut coefficient.covariate_slopes {
                        if c.as_str() == covariate {
                            *s += delta;
                        }
                    }
                }
            }
        }
    }
    copy
}

/// The first coefficient of `node`'s equation (parents by name, constant before
/// slopes) whose perturbation moves the target contrast.
fn witness_for(
    input: &TransportedPathSpecificInput<'_>,
    order: &[String],
    node: &str,
) -> Result<Option<NonRecoverableWitness>, String> {
    let Some((_, spec)) = input.model.mechanisms.iter().find(|(n, _)| n == node) else {
        return Ok(None);
    };
    let mut parents: Vec<&(String, Affine)> = spec.parents.iter().collect();
    parents.sort_by(|a, b| a.0.cmp(&b.0));
    let base = law_contrast(input.model, input.query, order, input.target_law)?;
    let source = law_contrast(input.model, input.query, order, input.source_law)?;
    for (parent, coefficient) in parents {
        let mut slopes: Vec<Option<&str>> = vec![None];
        let mut named: Vec<&str> =
            coefficient.covariate_slopes.iter().map(|(c, _)| c.as_str()).collect();
        named.sort_unstable();
        slopes.extend(named.into_iter().map(Some));
        for slope in slopes {
            let other = perturbed(input.model, node, parent, slope, 1.0);
            let moved = law_contrast(&other, input.query, order, input.target_law)?;
            if (moved - base).abs() > SENSITIVITY_TOLERANCE {
                return Ok(Some(NonRecoverableWitness {
                    selected_node: node.to_owned(),
                    perturbed_parent: parent.clone(),
                    perturbed_slope_covariate: slope.map(str::to_owned),
                    perturbation: 1.0,
                    source_model: input.model.clone(),
                    target_model_a: input.model.clone(),
                    target_model_b: other,
                    source_contrast: source,
                    target_contrast_a: base,
                    target_contrast_b: moved,
                }));
            }
        }
    }
    Ok(None)
}

fn check_diagram(
    input: &TransportedPathSpecificInput<'_>,
    order: &[String],
) -> Result<(), Box<TransportedPathSpecificRefusal>> {
    let model = input.model;
    let mut mechanism_targets: BTreeSet<&str> = BTreeSet::new();
    for selection in &input.diagram.selections {
        let target = selection.target.as_str();
        let is_mechanism = model.mechanisms.iter().any(|(n, _)| n == target);
        let is_known = is_mechanism
            || target == model.treatment
            || model.covariates.iter().any(|c| c == target);
        if !is_known {
            return Err(invalid(
                "transported_counterfactual.invalid_diagram",
                format!("selection {} points to unknown variable {target}", selection.label),
            ));
        }
        if is_mechanism {
            mechanism_targets.insert(target);
        }
    }
    let Some(first) = mechanism_targets.iter().next().copied() else {
        return Ok(());
    };
    let mut witness = None;
    for node in &mechanism_targets {
        witness = witness_for(input, order, node)
            .map_err(|m| invalid("transported_counterfactual.invalid_model", m))?;
        if witness.is_some() {
            break;
        }
    }
    let code = if witness.is_some() {
        antecedent_core::reason_code!("transport_proven_non_transportable")
    } else {
        antecedent_core::reason_code!("cell_not_licensed")
    };
    let mut refusal = refuse_with(
        code,
        "transported_counterfactual.selection_on_mechanism",
        Some(witness.as_ref().map_or_else(|| first.to_owned(), |w| w.selected_node.clone())),
        "remove the selection node on the mediator or outcome mechanism or choose a class that \
         models the differing mechanism; the covariate-selected theorem does not apply",
    );
    refusal.witness = witness;
    Err(refusal)
}

fn check_assumptions(
    assumptions: DeclaredAssumptions,
) -> Result<(), Box<TransportedPathSpecificRefusal>> {
    let (detail, premise) = if !assumptions.additive_noise {
        ("transported_counterfactual.nonadditive_mechanism", "additive_noise")
    } else if !assumptions.noise_laws_shared {
        ("transported_counterfactual.noise_law_not_shared", "noise_laws_shared")
    } else if !assumptions.cross_world_independence {
        ("transported_counterfactual.cross_world_independence_missing", "cross_world_independence")
    } else {
        return Ok(());
    };
    Err(refuse_with(
        antecedent_core::reason_code!("cell_not_licensed"),
        detail,
        Some(premise.to_owned()),
        "declare the additive-noise, shared-noise-law and cross-world independence premises, \
         each only when the evidence supports it",
    ))
}

fn check_factors(
    supplied: &BTreeMap<RegimeFactorKey, String>,
) -> Result<(), Box<TransportedPathSpecificRefusal>> {
    let required = [
        RegimeFactorKey { role: PopulationRole::Source, regime: "observational".to_owned() },
        RegimeFactorKey { role: PopulationRole::Target, regime: "observational".to_owned() },
    ];
    let passed = TransportedCounterfactualPrerequisites {
        transport_license: true,
        fixed_population_license: true,
        cross_population_assumptions: true,
    };
    let gate = refuse_transported_counterfactual(&passed, &required, supplied);
    if gate.missing_factors.is_empty() {
        return Ok(());
    }
    Err(Box::new(TransportedPathSpecificRefusal {
        refusal: gate.refusal,
        witness: None,
        missing_factors: gate.missing_factors,
    }))
}

fn check_overlap(
    source: &CovariateLaw,
    target: &CovariateLaw,
) -> Result<(), Box<TransportedPathSpecificRefusal>> {
    for (point, _) in &target.points {
        if !source.points.iter().any(|(s, _)| points_match(s, point)) {
            return Err(refuse_with(
                antecedent_core::reason_code!("transport_support_failure"),
                "transported_counterfactual.overlap_failure",
                Some(describe(point)),
                "restrict the target covariate law to the source support or collect source data \
                 there",
            ));
        }
    }
    Ok(())
}

/// Evaluate the transported path-specific mean contrast `theta_T = sum_z P_T(z) G(z)`.
///
/// Checks, in order: model, query and law well-formedness (`invalid_argument`);
/// the selection diagram (`selection_on_mechanism`, with the witness when the
/// contrast is sensitive to the selected mechanism); the declared premises
/// (`nonadditive_mechanism`, `noise_law_not_shared`,
/// `cross_world_independence_missing`); the source and target regime evidence
/// (`factor_missing`, the existing gate); and overlap (`overlap_failure`).
///
/// # Errors
///
/// A boxed typed refusal in the `transported_counterfactual` namespace.
pub fn evaluate_transported_path_specific(
    input: &TransportedPathSpecificInput<'_>,
) -> Result<TransportedPathSpecificResult, Box<TransportedPathSpecificRefusal>> {
    let order = validate_model(input.model)
        .map_err(|m| invalid("transported_counterfactual.invalid_model", m))?;
    validate_query(input.model, input.query)
        .map_err(|m| invalid("transported_counterfactual.invalid_query", m))?;
    let covariates: BTreeSet<&str> = input.model.covariates.iter().map(String::as_str).collect();
    validate_law("source", input.source_law, &covariates)
        .map_err(|m| invalid("transported_counterfactual.invalid_law", m))?;
    validate_law("target", input.target_law, &covariates)
        .map_err(|m| invalid("transported_counterfactual.invalid_law", m))?;
    check_diagram(input, &order)?;
    check_assumptions(input.assumptions)?;
    check_factors(input.supplied_factors)?;
    check_overlap(input.source_law, input.target_law)?;
    let mut unit_contrasts = Vec::with_capacity(input.target_law.points.len());
    for (point, weight) in &input.target_law.points {
        let contrast = unit_contrast(input.model, input.query, &order, point)
            .map_err(|m| invalid("transported_counterfactual.invalid_model", m))?;
        unit_contrasts.push(UnitContrast {
            point: point.clone(),
            target_weight: *weight,
            contrast,
        });
    }
    let target_contrast: f64 = unit_contrasts.iter().map(|u| u.target_weight * u.contrast).sum();
    let source_contrast = law_contrast(input.model, input.query, &order, input.source_law)
        .map_err(|m| invalid("transported_counterfactual.invalid_model", m))?;
    Ok(TransportedPathSpecificResult {
        target_contrast,
        source_contrast,
        unit_contrasts,
        derivation: TransportedDerivation {
            theorem: "covariate_selected_additive_path_specific_transport",
            checked: vec![
                "model_acyclic_and_well_formed",
                "no_selection_on_mediator_or_outcome",
                "covariates_are_pre_treatment_roots",
                "target_support_within_source_support",
                "source_and_target_regime_evidence_present",
            ],
            declared: vec![
                "additive_noise",
                "noise_laws_shared_and_independent_of_covariates",
                "unit_level_cross_world_independence",
                "source_fit_equals_structural_equations",
            ],
            claim: "point_only",
        },
    })
}
