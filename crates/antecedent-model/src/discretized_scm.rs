//! Discretization of a composed linear-Gaussian model into a finite structural
//! model, with a computed error bound against the closed-form Gaussian value.
//!
//! A [`ComposedLinearGaussian`] has a closed-form joint law and interventional
//! means. The finite counterfactual routes (exact enumeration, finite
//! counterfactual transport) work on variables with finitely many levels. This
//! module is the bounded bridge between them: every variable is replaced by a
//! declared number of levels and every mechanism by the conditional table that
//! the discretized parents induce, and the result is an executable
//! [`ComposedFinite`] composition.
//!
//! # Method
//!
//! * **Grid.** [`DiscretizationMethod::EqualProbabilityQuantile`]: the
//!   marginal Gaussian law of each variable under the composition's
//!   observational regime is cut at its `i/n` quantiles into `n` cells of
//!   equal probability. The two outer cells are open: the unbounded support is
//!   kept, never truncated. The value carried by a cell is its conditional
//!   mean, `mu + sigma * n * (phi(z_j) - phi(z_{j+1}))`, so the discretized
//!   marginal reproduces the Gaussian mean exactly.
//! * **Tables.** A child's row for one parent configuration is the Gaussian
//!   law `N(a + sum b_p x_p, s^2)` of the mechanism evaluated at the parents'
//!   cell values and integrated over the child's cells, by differences of the
//!   normal distribution function.
//! * **Intervention.** `do(V = level)` fixes a variable at the *value of that
//!   level*; the closed-form value is computed at the same value, so the two
//!   sides answer the same question.
//!
//! # Computed error bound
//!
//! Let `delta_v(c)` be the difference between the mean of the discretized
//! table of `v` at parent configuration `c` and the Gaussian mechanism mean at
//! the parents' cell values, and `e_v = E|delta_v|` under the intervened
//! discretized law (unreachable configurations carry no weight). In the discretized
//! law `E[V] = a_v + sum_p b_vp E[X_p] + E[delta_v]`, while the closed form has
//! the same recursion without `delta`. Subtracting, the error of the mean of
//! `v` satisfies `|err_v| <= e_v + sum_p |b_vp| |err_p|`, and an intervened
//! variable has zero error because both sides fix it at the same value. The
//! bound is evaluated in topological order from the `e_v` computed here, so
//! it is a computed guarantee for the mean of every variable and for a linear
//! contrast of means, not an asymptotic statement. It says nothing about any
//! other functional of the discretized law (variances, probabilities,
//! counterfactual quantities).
//!
//! # Boundary
//!
//! * Additive-noise linear-Gaussian mechanisms with real support only; a
//!   finite table, a mixed family or a bounded declared support is refused.
//! * Markovian composition only: a mechanism that loads on a shared latent
//!   cause is refused (the finite engine's tables are conditional on parents
//!   alone).
//! * Levels are declared per variable, never defaulted; `2..=64` per variable
//!   and the joint space within the finite-enumeration bound.
//! * Point-only. The discretized value carries no interval, no posterior and
//!   no calibration; parameter uncertainty declared on the source mechanisms
//!   is not propagated through the discretization and the result says so.
//!
//! The parity table at the end of the module records, per transported or
//! derived route, which inferential paradigm each route supports, and refuses
//! the one a request names that the route does not.
//!
//! # Refresh with outcome-only data
//!
//! The bridge is immutable and keyed by its identity, which covers the source
//! composition, the levels and the method. New outcome-only data refits the source
//! mechanisms and therefore changes the source composition: the grid, the tables, the
//! error bound and the identity are all recomputed by calling the bridge again, never
//! patched. Level edges are never reused across a refit, and a prepared composition's
//! refresh does not rebuild a bridge from it silently. The declared levels are kept only by
//! being passed again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;
use antecedent_core::identity::CanonicalWriter;
use antecedent_kernels::special::{norm_cdf, norm_inv, norm_pdf};

use crate::compose::coherence::CoherenceFinding;
use crate::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CompositionError, Domain,
    MAX_FINITE_STATES, MAX_LEVELS_PER_VARIABLE, MechanismDescriptor, MechanismForm,
    MechanismProvenance, ParentBinding, TabularRow,
};

/// Version of the discretization bridge's semantics and identity encoding.
pub const DISCRETIZED_SCM_VERSION: u32 = 1;
/// Fewest levels a variable may be discretized to.
pub const MIN_BRIDGE_LEVELS: usize = 2;
/// Most levels a variable may be discretized to.
pub const MAX_BRIDGE_LEVELS: usize = MAX_LEVELS_PER_VARIABLE;
/// Label of the inference a discretized value carries.
pub const BRIDGE_INFERENCE: &str = "point_only";
/// Label of the calibration standing of a discretized value.
pub const BRIDGE_CALIBRATION: &str = "no_interval_or_calibration_claim";
/// What the error bound covers.
pub const BRIDGE_ERROR_SCOPE: &str = "mean_of_linear_contrast_of_interventional_means_only";

/// The level count of some variable was not declared.
pub const DETAIL_LEVELS_UNDECLARED: &str = "discretized_scm.levels_undeclared";
/// A level count is outside `MIN_BRIDGE_LEVELS..=MAX_BRIDGE_LEVELS`.
pub const DETAIL_LEVELS_OUT_OF_RANGE: &str = "discretized_scm.levels_out_of_range";
/// The plan names a variable the composition does not have.
pub const DETAIL_UNKNOWN_VARIABLE: &str = "discretized_scm.unknown_variable";
/// The joint space exceeds the finite-enumeration bound.
pub const DETAIL_STATE_SPACE_TOO_LARGE: &str = "discretized_scm.state_space_too_large";
/// A mechanism is not additive-noise linear-Gaussian.
pub const DETAIL_NON_GAUSSIAN: &str = "discretized_scm.non_gaussian_mechanism";
/// A mechanism's declared support is not the real line.
pub const DETAIL_SUPPORT_NOT_REAL: &str = "discretized_scm.support_not_real_line";
/// A mechanism loads on a shared latent cause.
pub const DETAIL_LATENT_CONFOUNDING: &str = "discretized_scm.latent_confounding";
/// A variable's marginal law is degenerate, so it has no grid.
pub const DETAIL_DEGENERATE_MARGINAL: &str = "discretized_scm.degenerate_marginal";
/// The composition is not coherent.
pub const DETAIL_COMPOSITION_INCOHERENT: &str = "discretized_scm.composition_incoherent";
/// An intervention names a variable or level that does not exist.
pub const DETAIL_INTERVENTION_INVALID: &str = "discretized_scm.intervention_invalid";
/// An interval, band, posterior or law was requested of a point-only value.
pub const DETAIL_POINT_ONLY: &str = "discretized_scm.point_only";
/// Cancellation was observed.
pub const DETAIL_CANCELLED: &str = "discretized_scm.cancelled";
/// A numerical failure in the discretization.
pub const DETAIL_NUMERICAL: &str = "discretized_scm.numerical";
/// A route's inferential paradigm is closed.
pub const DETAIL_PARADIGM_CLOSED: &str = "paradigm_parity.paradigm_closed";

const TAG_IDENTITY: &str = "antecedent.model.discretized_scm";

/// A typed refusal of the bridge or of a paradigm request.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeRefusal {
    /// Registered reason code.
    pub reason_code: &'static str,
    /// Typed detail literal.
    pub detail: &'static str,
    /// What the refusal is about (a variable, a route).
    pub subject: String,
    /// What would license the request.
    pub remedy: &'static str,
    /// Coherence findings when the composition itself is the defect.
    pub findings: Vec<CoherenceFinding>,
}

impl BridgeRefusal {
    fn new(
        reason_code: &'static str,
        detail: &'static str,
        subject: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { reason_code, detail, subject: subject.into(), remedy, findings: Vec::new() }
    }
}

fn invalid(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> BridgeRefusal {
    BridgeRefusal::new(antecedent_core::reason_code!("invalid_argument"), detail, subject, remedy)
}

fn unsupported(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> BridgeRefusal {
    BridgeRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        detail,
        subject,
        remedy,
    )
}

/// How the cells of a variable are chosen.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum DiscretizationMethod {
    /// `n` cells of equal marginal probability, open outer cells, each carried
    /// by its conditional mean.
    EqualProbabilityQuantile,
}

impl DiscretizationMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EqualProbabilityQuantile => "equal_probability_quantile",
        }
    }
}

/// The declared discretization: one method and a level count per variable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscretizationPlan {
    /// Cell construction.
    pub method: DiscretizationMethod,
    /// Number of levels of every variable of the composition.
    pub levels: BTreeMap<String, usize>,
}

impl DiscretizationPlan {
    /// The same level count for every named variable.
    #[must_use]
    pub fn uniform(method: DiscretizationMethod, variables: &[String], levels: usize) -> Self {
        Self { method, levels: variables.iter().map(|v| (v.clone(), levels)).collect() }
    }
}

#[derive(Clone, Debug)]
struct BridgeNode {
    variable: String,
    parents: Vec<usize>,
    coefficients: Vec<f64>,
    intercept: f64,
    sigma: f64,
    /// Interior cell edges, ascending, `levels - 1` of them.
    edges: Vec<f64>,
    /// Value carried by each cell.
    values: Vec<f64>,
    level_names: Vec<String>,
    /// `max_c |delta_v(c)|` over parent configurations (diagnostic).
    mean_error: f64,
    /// Signed `delta_v(c)` per parent configuration, first parent most
    /// significant.
    delta: Vec<f64>,
}

/// One arm of a contrast on the discretized model: hard interventions given
/// as `(variable, level index)`.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteArm {
    /// Weight of the arm.
    pub weight: f64,
    /// `do(variable = value of level)` pairs.
    pub interventions: Vec<(String, usize)>,
}

/// A linear contrast of interventional means of one outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteContrast {
    /// Outcome variable.
    pub outcome: String,
    /// Arms.
    pub arms: Vec<DiscreteArm>,
}

/// Discretized and closed-form mean of one arm.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmDiagnostic {
    /// Weight of the arm.
    pub weight: f64,
    /// `E[outcome | do(arm)]` in the discretized model.
    pub discretized_mean: f64,
    /// `E[outcome | do(arm)]` in the Gaussian composition at the same values.
    pub closed_form_mean: f64,
    /// Computed bound on `|discretized_mean - closed_form_mean|`.
    pub mean_error_bound: f64,
}

/// A discretized contrast with its closed-form comparison and computed bound.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscretizedContrast {
    /// Discretized value of the contrast.
    pub value: f64,
    /// Closed-form Gaussian value at the same intervention values.
    pub closed_form: f64,
    /// `|value - closed_form|`.
    pub absolute_error: f64,
    /// Computed bound: `sum_k |w_k| bound_k`; `absolute_error` never exceeds it
    /// beyond floating-point rounding.
    pub error_bound: f64,
    /// Per-arm values.
    pub arms: Vec<ArmDiagnostic>,
    /// Method label.
    pub method: &'static str,
    /// Always [`BRIDGE_INFERENCE`].
    pub inference: &'static str,
    /// Always [`BRIDGE_CALIBRATION`].
    pub calibration: &'static str,
    /// What the bound covers ([`BRIDGE_ERROR_SCOPE`]).
    pub error_scope: &'static str,
    /// Whether the source composition declared parameter uncertainty that this
    /// point value does not propagate.
    pub parameter_uncertainty_not_propagated: bool,
}

/// A composed linear-Gaussian model discretized into an executable finite
/// composition.
///
/// The source model and the finite composition are retained and immutable;
/// nothing here borrows from the caller's specification.
#[derive(Clone, Debug)]
pub struct DiscretizedLinearGaussian {
    model: ComposedLinearGaussian,
    finite: ComposedFinite,
    finite_spec: ComposedModelSpec,
    nodes: Vec<BridgeNode>,
    method: DiscretizationMethod,
    identity: [u8; 32],
    declared_uncertainty: bool,
}

fn poll(ctx: &ExecutionContext) -> Result<(), BridgeRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(BridgeRefusal::new(
            antecedent_core::reason_code!("cancelled_no_claim"),
            DETAIL_CANCELLED,
            "discretization",
            "rerun the discretization to completion",
        ));
    }
    Ok(())
}

fn level_name(index: usize) -> String {
    // Zero padded so lexicographic order, which the finite composition sorts
    // by, is level order.
    format!("{index:02}")
}

fn unbounded_real(domain: &Domain) -> bool {
    match domain {
        Domain::Real => true,
        Domain::Interval { lo, hi } => {
            lo.is_infinite() && *lo < 0.0 && hi.is_infinite() && *hi > 0.0
        }
        Domain::Levels(_) => false,
    }
}

/// Interior quantile edges and cell values of `N(mean, sd^2)` at `n` cells.
fn quantile_grid(mean: f64, sd: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let nf = n as f64;
    let z: Vec<f64> = (1..n).map(|i| norm_inv(i as f64 / nf)).collect();
    let edges: Vec<f64> = z.iter().map(|zi| mean + sd * zi).collect();
    let density = |k: usize| -> f64 {
        // phi at the k-th boundary, k in 0..=n; the outer boundaries are +-inf.
        if k == 0 || k == n { 0.0 } else { norm_pdf(z[k - 1]) }
    };
    let values: Vec<f64> = (0..n).map(|j| mean + sd * nf * (density(j) - density(j + 1))).collect();
    (edges, values)
}

/// Probability of each cell under `N(mean, sigma^2)`.
fn cell_probabilities(edges: &[f64], mean: f64, sigma: f64) -> Vec<f64> {
    let n = edges.len() + 1;
    if sigma <= 0.0 {
        let mut out = vec![0.0; n];
        out[edges.partition_point(|e| *e <= mean)] = 1.0;
        return out;
    }
    let mut out = Vec::with_capacity(n);
    let mut below = 0.0;
    for e in edges {
        let cdf = norm_cdf((e - mean) / sigma).max(below);
        out.push(cdf - below);
        below = cdf;
    }
    out.push((1.0 - below).max(0.0));
    out
}

fn from_composition(error: CompositionError) -> BridgeRefusal {
    match error {
        CompositionError::Incoherent { findings } => {
            if findings.iter().any(|f| f.detail == "mechanism_composition.family_mixed") {
                return unsupported(
                    DETAIL_NON_GAUSSIAN,
                    "composition",
                    "discretize a composition whose every mechanism is additive-noise linear-Gaussian",
                );
            }
            let mut refusal = BridgeRefusal::new(
                antecedent_core::reason_code!("mechanism_composition_incoherent"),
                DETAIL_COMPOSITION_INCOHERENT,
                "composition",
                "repair the findings, then discretize",
            );
            refusal.findings = findings;
            refusal
        }
        CompositionError::Unsupported { .. } => unsupported(
            DETAIL_NON_GAUSSIAN,
            "composition",
            "discretize a composition whose every mechanism is additive-noise linear-Gaussian",
        ),
        CompositionError::Cancelled => BridgeRefusal::new(
            antecedent_core::reason_code!("cancelled_no_claim"),
            DETAIL_CANCELLED,
            "discretization",
            "rerun the discretization to completion",
        ),
        _ => BridgeRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_NUMERICAL,
            "composition",
            "inspect the composition's numerical declaration",
        ),
    }
}

/// Per-variable level counts, after the plan and the composition's mechanisms are checked.
fn plan_level_counts(
    model: &ComposedLinearGaussian,
    plan: &DiscretizationPlan,
) -> Result<Vec<usize>, BridgeRefusal> {
    let spec = model.spec();
    let variables = model.variables();
    for name in plan.levels.keys() {
        if !variables.contains(name) {
            return Err(invalid(
                DETAIL_UNKNOWN_VARIABLE,
                name.clone(),
                "declare levels only for variables of the composition",
            ));
        }
    }
    let mut level_counts = Vec::with_capacity(variables.len());
    let mut joint = 1_usize;
    for v in variables {
        let Some(&n) = plan.levels.get(v) else {
            return Err(invalid(
                DETAIL_LEVELS_UNDECLARED,
                v.clone(),
                "declare a level count for every variable; none is defaulted",
            ));
        };
        if !(MIN_BRIDGE_LEVELS..=MAX_BRIDGE_LEVELS).contains(&n) {
            return Err(invalid(
                DETAIL_LEVELS_OUT_OF_RANGE,
                v.clone(),
                "declare at least 2 levels per variable and no more than the bridge level bound",
            ));
        }
        joint = joint.saturating_mul(n);
        level_counts.push(n);
    }
    if joint > MAX_FINITE_STATES {
        return Err(invalid(
            DETAIL_STATE_SPACE_TOO_LARGE,
            "composition",
            "declare fewer levels or fewer variables; the joint space is enumerated exactly",
        ));
    }
    for d in &spec.mechanisms {
        if !unbounded_real(&d.output_support) {
            return Err(unsupported(
                DETAIL_SUPPORT_NOT_REAL,
                d.variable.clone(),
                "a Gaussian mechanism has the real line as support; declare it so or use a finite mechanism",
            ));
        }
        if !d.latents.is_empty() {
            return Err(unsupported(
                DETAIL_LATENT_CONFOUNDING,
                d.variable.clone(),
                "the finite model is Markovian; refit with the shared cause observed or keep the closed-form route",
            ));
        }
    }
    Ok(level_counts)
}

/// Pass one: the quantile grid and the declared coefficients of every mechanism.
fn grid_nodes(
    model: &ComposedLinearGaussian,
    level_counts: &[usize],
    index: &BTreeMap<&str, usize>,
) -> Result<Vec<BridgeNode>, BridgeRefusal> {
    let spec = model.spec();
    let law = model.law(&[]).map_err(from_composition)?;
    let mut nodes: Vec<BridgeNode> = Vec::with_capacity(spec.mechanisms.len());
    for (i, d) in spec.mechanisms.iter().enumerate() {
        let MechanismForm::LinearGaussian { intercept, sigma, coefficients } = &d.form else {
            return Err(unsupported(
                DETAIL_NON_GAUSSIAN,
                d.variable.clone(),
                "discretize a composition whose every mechanism is additive-noise linear-Gaussian",
            ));
        };
        let (Some(mean), Some(variance)) = (law.mean_of(&d.variable), law.variance_of(&d.variable))
        else {
            return Err(BridgeRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_NUMERICAL,
                d.variable.clone(),
                "inspect the composition's numerical declaration",
            ));
        };
        if !mean.is_finite() || !variance.is_finite() || variance <= 0.0 {
            return Err(unsupported(
                DETAIL_DEGENERATE_MARGINAL,
                d.variable.clone(),
                "a variable with no marginal spread has no grid; keep it out of the discretized composition",
            ));
        }
        let n = level_counts[i];
        let (edges, values) = quantile_grid(mean, variance.sqrt(), n);
        let mut parents = Vec::with_capacity(coefficients.len());
        let mut coefs = Vec::with_capacity(coefficients.len());
        for (name, c) in coefficients {
            let Some(&pi) = index.get(name.as_str()) else {
                return Err(BridgeRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    DETAIL_NUMERICAL,
                    d.variable.clone(),
                    "inspect the composition's numerical declaration",
                ));
            };
            parents.push(pi);
            coefs.push(*c);
        }
        nodes.push(BridgeNode {
            variable: d.variable.clone(),
            parents,
            coefficients: coefs,
            intercept: *intercept,
            sigma: *sigma,
            edges,
            values,
            level_names: (0..n).map(level_name).collect(),
            mean_error: 0.0,
            delta: Vec::new(),
        });
    }
    Ok(nodes)
}

/// Pass two: every mechanism's conditional table and its mean discretization error, which is
/// written onto the nodes.
fn fill_tables(
    nodes: &mut [BridgeNode],
    ctx: &ExecutionContext,
) -> Result<Vec<Vec<TabularRow>>, BridgeRefusal> {
    let mut tables: Vec<Vec<TabularRow>> = Vec::with_capacity(nodes.len());
    let mut errors = Vec::with_capacity(nodes.len());
    let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(nodes.len());
    for v in 0..nodes.len() {
        let node = &nodes[v];
        let widths: Vec<usize> = node.parents.iter().map(|&p| nodes[p].values.len()).collect();
        let configs: usize = widths.iter().product();
        let mut rows = Vec::with_capacity(configs);
        let mut worst = 0.0_f64;
        let mut delta = Vec::with_capacity(configs);
        let mut digits = vec![0_usize; widths.len()];
        for c in 0..configs {
            if c % 4096 == 4095 {
                poll(ctx)?;
            }
            let mut rest = c;
            for k in (0..widths.len()).rev() {
                digits[k] = rest % widths[k];
                rest /= widths[k];
            }
            let mut mean = node.intercept;
            for (k, &p) in node.parents.iter().enumerate() {
                mean += node.coefficients[k] * nodes[p].values[digits[k]];
            }
            let probabilities = cell_probabilities(&node.edges, mean, node.sigma);
            let table_mean: f64 = probabilities.iter().zip(&node.values).map(|(p, x)| p * x).sum();
            worst = worst.max((table_mean - mean).abs());
            delta.push(table_mean - mean);
            rows.push(TabularRow {
                parent_levels: node
                    .parents
                    .iter()
                    .enumerate()
                    .map(|(k, &p)| {
                        (nodes[p].variable.clone(), nodes[p].level_names[digits[k]].clone())
                    })
                    .collect(),
                probabilities: node.level_names.iter().cloned().zip(probabilities).collect(),
            });
        }
        tables.push(rows);
        errors.push(worst);
        deltas.push(delta);
    }
    for ((node, e), d) in nodes.iter_mut().zip(errors).zip(deltas) {
        node.mean_error = e;
        node.delta = d;
    }
    Ok(tables)
}

/// The finite (tabular) mechanisms of the discretized composition.
fn finite_mechanisms(
    spec: &ComposedModelSpec,
    nodes: &[BridgeNode],
    tables: &[Vec<TabularRow>],
    index: &BTreeMap<&str, usize>,
    identity_hex: &str,
) -> Vec<MechanismDescriptor> {
    let mut mechanisms = Vec::with_capacity(nodes.len());
    for (i, d) in spec.mechanisms.iter().enumerate() {
        let node = &nodes[i];
        let parents: Vec<ParentBinding> = d
            .parents
            .iter()
            .map(|binding| {
                let p = index[binding.variable.as_str()];
                ParentBinding {
                    variable: binding.variable.clone(),
                    units: binding.units.clone(),
                    transform: binding.transform.clone(),
                    domain: Domain::Levels(nodes[p].level_names.clone()),
                }
            })
            .collect();
        mechanisms.push(MechanismDescriptor {
            variable: d.variable.clone(),
            form: MechanismForm::Tabular {
                levels: node.level_names.clone(),
                rows: tables[i].clone(),
            },
            parents,
            conditioning_set: d.conditioning_set.clone(),
            output_units: d.output_units.clone(),
            output_transform: d.output_transform.clone(),
            output_support: Domain::Levels(node.level_names.clone()),
            context: d.context.clone(),
            provenance: MechanismProvenance {
                origin: d.provenance.origin,
                artifact_id: format!("discretized:{identity_hex}:{}", d.variable),
                // The evidence is the source's: a discretization is not
                // new evidence, so dependence between mechanisms that
                // shared evidence stays visible.
                evidence_id: d.provenance.evidence_id.clone(),
            },
            latents: Vec::new(),
            uncertainty: None,
        });
    }
    mechanisms
}

impl DiscretizedLinearGaussian {
    /// Compile a declared composition and discretize it.
    ///
    /// # Errors
    /// [`BridgeRefusal`] for an incoherent or non-Gaussian composition and for
    /// every refusal of [`Self::new`].
    pub fn from_spec(
        spec: &ComposedModelSpec,
        plan: &DiscretizationPlan,
        ctx: &ExecutionContext,
    ) -> Result<Self, BridgeRefusal> {
        let model = ComposedLinearGaussian::compile(spec).map_err(from_composition)?;
        Self::new(&model, plan, ctx)
    }

    /// Discretize an executable composition.
    ///
    /// # Errors
    /// A level count missing, out of range or naming an unknown variable; a
    /// joint space above the enumeration bound; a declared support other than
    /// the real line; a mechanism loading on a shared latent cause; a
    /// degenerate marginal; cancellation.
    pub fn new(
        model: &ComposedLinearGaussian,
        plan: &DiscretizationPlan,
        ctx: &ExecutionContext,
    ) -> Result<Self, BridgeRefusal> {
        poll(ctx)?;
        let spec = model.spec();
        let variables = model.variables();
        let level_counts = plan_level_counts(model, plan)?;
        let index: BTreeMap<&str, usize> =
            variables.iter().enumerate().map(|(i, v)| (v.as_str(), i)).collect();
        let mut nodes = grid_nodes(model, &level_counts, &index)?;
        let tables = fill_tables(&mut nodes, ctx)?;

        let identity = identity_of(spec, plan.method, &nodes);
        let identity_hex = antecedent_core::identity::to_hex(&identity);

        let mechanisms = finite_mechanisms(spec, &nodes, &tables, &index, &identity_hex);
        let finite_spec = ComposedModelSpec {
            target: spec.target.clone(),
            mechanisms,
            edges: spec.edges.clone(),
            invariances: spec.invariances.clone(),
            selection_diagrams: spec.selection_diagrams.clone(),
            dependence: crate::compose::PosteriorDependence::Unknown,
            admit_declared_mechanisms: spec.admit_declared_mechanisms,
        };
        let finite = ComposedFinite::compile(&finite_spec).map_err(from_composition)?;
        let declared_uncertainty = spec.mechanisms.iter().any(|d| d.uncertainty.is_some());
        Ok(Self {
            model: model.clone(),
            finite,
            finite_spec,
            nodes,
            method: plan.method,
            identity,
            declared_uncertainty,
        })
    }

    /// The source Gaussian composition.
    #[must_use]
    pub fn source_model(&self) -> &ComposedLinearGaussian {
        &self.model
    }

    /// The executable finite composition (the input of exact enumeration and
    /// of the finite counterfactual routes).
    #[must_use]
    pub fn finite_model(&self) -> &ComposedFinite {
        &self.finite
    }

    /// The tabular specification the finite composition was compiled from.
    #[must_use]
    pub fn finite_spec(&self) -> &ComposedModelSpec {
        &self.finite_spec
    }

    /// Variables in canonical order.
    #[must_use]
    pub fn variables(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.variable.as_str()).collect()
    }

    /// Level names of a variable, in level order.
    #[must_use]
    pub fn level_names(&self, variable: &str) -> Option<&[String]> {
        self.node(variable).map(|n| n.level_names.as_slice())
    }

    /// Numeric value carried by each level of a variable, ascending.
    #[must_use]
    pub fn level_values(&self, variable: &str) -> Option<&[f64]> {
        self.node(variable).map(|n| n.values.as_slice())
    }

    /// Interior cell edges of a variable, ascending.
    #[must_use]
    pub fn cell_edges(&self, variable: &str) -> Option<&[f64]> {
        self.node(variable).map(|n| n.edges.as_slice())
    }

    /// `max_c |delta_v(c)|` of one mechanism: the largest gap between its
    /// discretized table mean and its Gaussian mean at the parents' values.
    #[must_use]
    pub fn mechanism_mean_error(&self, variable: &str) -> Option<f64> {
        self.node(variable).map(|n| n.mean_error)
    }

    /// The declared method.
    #[must_use]
    pub fn method(&self) -> DiscretizationMethod {
        self.method
    }

    /// Identity of the discretization: the source mechanisms (numbers,
    /// parents, context, provenance), the method and every level count.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    /// Hex form of [`Self::identity`].
    #[must_use]
    pub fn identity_hex(&self) -> String {
        antecedent_core::identity::to_hex(&self.identity)
    }

    fn node(&self, variable: &str) -> Option<&BridgeNode> {
        self.nodes.iter().find(|n| n.variable == variable)
    }

    fn position(&self, variable: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.variable == variable)
    }

    fn resolve(
        &self,
        interventions: &[(String, usize)],
    ) -> Result<Vec<Option<usize>>, BridgeRefusal> {
        let mut fixed = vec![None; self.nodes.len()];
        for (name, level) in interventions {
            let Some(i) = self.position(name) else {
                return Err(invalid(
                    DETAIL_INTERVENTION_INVALID,
                    name.clone(),
                    "intervene on a variable of the composition",
                ));
            };
            if *level >= self.nodes[i].values.len() || fixed[i].is_some() {
                return Err(invalid(
                    DETAIL_INTERVENTION_INVALID,
                    name.clone(),
                    "name each variable once, at one of its levels",
                ));
            }
            fixed[i] = Some(*level);
        }
        Ok(fixed)
    }

    /// Evaluate a contrast on the discretized model, beside the closed-form
    /// Gaussian value at the same intervention values and the computed bound.
    ///
    /// # Errors
    /// An unknown outcome, variable or level; a repeated intervention;
    /// cancellation.
    pub fn mean_contrast(
        &self,
        contrast: &DiscreteContrast,
        ctx: &ExecutionContext,
    ) -> Result<DiscretizedContrast, BridgeRefusal> {
        poll(ctx)?;
        let Some(outcome) = self.position(&contrast.outcome) else {
            return Err(invalid(
                DETAIL_INTERVENTION_INVALID,
                contrast.outcome.clone(),
                "name an outcome variable of the composition",
            ));
        };
        let mut value = 0.0;
        let mut closed_form = 0.0;
        let mut error_bound = 0.0;
        let mut arms = Vec::with_capacity(contrast.arms.len());
        for arm in &contrast.arms {
            let fixed = self.resolve(&arm.interventions)?;
            let names: Vec<(&str, &str)> = arm
                .interventions
                .iter()
                .map(|(v, l)| {
                    let i = self.position(v).unwrap_or(0);
                    (v.as_str(), self.nodes[i].level_names[*l].as_str())
                })
                .collect();
            let law = self.finite.law(&names, ctx).map_err(from_composition)?;
            let marginal = law.marginal(&contrast.outcome).map_err(from_composition)?;
            let discretized: f64 =
                marginal.iter().zip(&self.nodes[outcome].values).map(|((_, p), x)| p * x).sum();
            let pairs: Vec<(&str, f64)> = arm
                .interventions
                .iter()
                .map(|(v, l)| {
                    let i = self.position(v).unwrap_or(0);
                    (v.as_str(), self.nodes[i].values[*l])
                })
                .collect();
            let gaussian = self.model.law(&pairs).map_err(from_composition)?;
            let closed = gaussian.mean_of(&contrast.outcome).ok_or_else(|| {
                invalid(
                    DETAIL_INTERVENTION_INVALID,
                    contrast.outcome.clone(),
                    "name an outcome variable of the composition",
                )
            })?;
            let bound = self.mean_error_bound(&fixed, law.probabilities())[outcome];
            value += arm.weight * discretized;
            closed_form += arm.weight * closed;
            error_bound += arm.weight.abs() * bound;
            arms.push(ArmDiagnostic {
                weight: arm.weight,
                discretized_mean: discretized,
                closed_form_mean: closed,
                mean_error_bound: bound,
            });
        }
        Ok(DiscretizedContrast {
            value,
            closed_form,
            absolute_error: (value - closed_form).abs(),
            error_bound,
            arms,
            method: self.method.as_str(),
            inference: BRIDGE_INFERENCE,
            calibration: BRIDGE_CALIBRATION,
            error_scope: BRIDGE_ERROR_SCOPE,
            parameter_uncertainty_not_propagated: self.declared_uncertainty,
        })
    }

    /// The computed bound on the error of the mean of every variable under
    /// the given hard interventions, in canonical variable order.
    ///
    /// The mechanism term of a variable is `E|delta_v|` under the intervened
    /// discretized joint law `probabilities` (last variable fastest), which
    /// bounds `|E[delta_v]|`; configurations the intervened law never reaches
    /// contribute nothing.
    fn mean_error_bound(&self, fixed: &[Option<usize>], probabilities: &[f64]) -> Vec<f64> {
        let n = self.nodes.len();
        let widths: Vec<usize> = self.nodes.iter().map(|x| x.values.len()).collect();
        let mut expected = vec![0.0_f64; n];
        let mut digits = vec![0_usize; n];
        for (state, p) in probabilities.iter().enumerate() {
            if *p <= 0.0 {
                continue;
            }
            let mut rest = state;
            for k in (0..n).rev() {
                digits[k] = rest % widths[k];
                rest /= widths[k];
            }
            for (v, node) in self.nodes.iter().enumerate() {
                if fixed[v].is_some() {
                    continue;
                }
                let mut cfg = 0_usize;
                for &q in &node.parents {
                    cfg = cfg * widths[q] + digits[q];
                }
                expected[v] += p * node.delta[cfg].abs();
            }
        }
        let mut bound = vec![0.0_f64; n];
        for &v in self.model.topological_order() {
            if fixed[v].is_some() {
                continue;
            }
            let node = &self.nodes[v];
            let mut b = expected[v];
            for (k, &p) in node.parents.iter().enumerate() {
                b += node.coefficients[k].abs() * bound[p];
            }
            bound[v] = b;
        }
        bound
    }

    /// The refusal for an interval, band, posterior or law-valued request of a
    /// discretized value, whichever paradigm asks.
    #[must_use]
    pub fn uncertainty_refusal(&self, paradigm: InferenceParadigm) -> BridgeRefusal {
        let _ = paradigm;
        BridgeRefusal::new(
            antecedent_core::reason_code!("cell_not_licensed"),
            DETAIL_POINT_ONLY,
            "discretization_bridge",
            "use the closed-form composed uncertainty for an interval on the Gaussian model; a discretized value is a point with a computed mean-error bound",
        )
    }
}

fn identity_of(
    spec: &ComposedModelSpec,
    method: DiscretizationMethod,
    nodes: &[BridgeNode],
) -> [u8; 32] {
    let mut w = CanonicalWriter::new(TAG_IDENTITY);
    w.word(u64::from(DISCRETIZED_SCM_VERSION));
    w.word(match method {
        DiscretizationMethod::EqualProbabilityQuantile => 1,
    });
    w.text(&spec.target.population).text(&spec.target.regime);
    w.word(nodes.len() as u64);
    for (node, d) in nodes.iter().zip(&spec.mechanisms) {
        w.text(&node.variable).word(node.values.len() as u64);
        w.real(node.intercept).real(node.sigma).word(node.parents.len() as u64);
        for (k, &p) in node.parents.iter().enumerate() {
            w.text(&nodes[p].variable).real(node.coefficients[k]);
        }
        w.text(&d.output_units).text(&d.output_transform);
        w.text(&d.context.population).text(&d.context.regime);
        w.text(&d.provenance.artifact_id).text(&d.provenance.evidence_id);
        w.word(d.provenance.origin as u64);
    }
    w.word(u64::from(spec.admit_declared_mechanisms));
    w.finish()
}

// ---------------------------------------------------------------------------
// Paradigm parity
// ---------------------------------------------------------------------------

/// Inferential paradigm of a request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InferenceParadigm {
    /// Sampling-distribution statements about fitted quantities.
    Frequentist,
    /// Posterior statements under a declared prior.
    Bayesian,
}

impl InferenceParadigm {
    /// Stable name, as it appears in the support axes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frequentist => "Frequentist",
            Self::Bayesian => "Bayesian",
        }
    }
}

/// A route whose paradigm support is stated.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ParityRoute {
    /// Fitted effect model carried to a target cohort for a decision.
    TransportedCohortDecision,
    /// Measurement model carried to a target population.
    MeasurementTransport,
    /// Fitted mechanisms carried into a target composition.
    MechanismTransport,
    /// Transport followed by evidence synthesis.
    TransportedSynthesis,
    /// Surrogate-to-terminal bridge carried to a population.
    TransportedBridge,
    /// Supplied policy evaluated in a target population.
    TransportedPolicy,
    /// Adaptive trial as a transport source.
    AdaptiveTrialSource,
    /// Fixed-horizon, time-invariant temporal transport.
    TemporalTransport,
    /// Transport under graph uncertainty (structural envelope).
    DiscoveredStructureTransport,
    /// Provider output admitted as transport source evidence.
    ProviderHandoff,
    /// Composed spread and exceedance at the composed coordinate.
    ComposedSpreadExceedance,
    /// Whole abduction-action-prediction path uncertainty.
    WholePathUncertainty,
    /// Derived uncertainty of ranking and optimality statements.
    DecisionReliabilityUncertainty,
    /// Whole-procedure uncertainty of adaptive experiments.
    AdaptiveProcedureUncertainty,
    /// Composed linear-Gaussian to finite-SCM discretization.
    DiscretizationBridge,
}

impl ParityRoute {
    /// Every route, in table order.
    pub const ALL: [Self; 15] = [
        Self::TransportedCohortDecision,
        Self::MeasurementTransport,
        Self::MechanismTransport,
        Self::TransportedSynthesis,
        Self::TransportedBridge,
        Self::TransportedPolicy,
        Self::AdaptiveTrialSource,
        Self::TemporalTransport,
        Self::DiscoveredStructureTransport,
        Self::ProviderHandoff,
        Self::ComposedSpreadExceedance,
        Self::WholePathUncertainty,
        Self::DecisionReliabilityUncertainty,
        Self::AdaptiveProcedureUncertainty,
        Self::DiscretizationBridge,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TransportedCohortDecision => "transported_cohort_decision",
            Self::MeasurementTransport => "measurement_transport",
            Self::MechanismTransport => "mechanism_transport",
            Self::TransportedSynthesis => "transported_synthesis",
            Self::TransportedBridge => "transported_bridge",
            Self::TransportedPolicy => "transported_policy",
            Self::AdaptiveTrialSource => "adaptive_trial_source",
            Self::TemporalTransport => "temporal_transport",
            Self::DiscoveredStructureTransport => "discovered_structure_transport",
            Self::ProviderHandoff => "provider_handoff",
            Self::ComposedSpreadExceedance => "composed_spread_exceedance",
            Self::WholePathUncertainty => "whole_path_uncertainty",
            Self::DecisionReliabilityUncertainty => "decision_reliability_uncertainty",
            Self::AdaptiveProcedureUncertainty => "adaptive_procedure_uncertainty",
            Self::DiscretizationBridge => "discretization_bridge",
        }
    }
}

/// What one paradigm gets from one route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParadigmCell {
    /// The route executes this paradigm; the text says what the result is.
    Supported(&'static str),
    /// The route is closed to this paradigm; the text says why.
    Closed(&'static str),
}

/// One row of the parity table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParityRow {
    /// The route.
    pub route: ParityRoute,
    /// Frequentist cell.
    pub frequentist: ParadigmCell,
    /// Bayesian cell.
    pub bayesian: ParadigmCell,
}

/// The parity table: every route has a row and every missing paradigm is a
/// closed cell with its reason.
pub const PARITY_TABLE: [ParityRow; 15] = [
    ParityRow {
        route: ParityRoute::TransportedCohortDecision,
        frequentist: ParadigmCell::Supported(
            "point cohort mean of a fitted effect model under a declared shift; no interval",
        ),
        bayesian: ParadigmCell::Closed(
            "a fitted effect model is a point predictor; no posterior over its parameters is carried to the target cohort",
        ),
    },
    ParityRow {
        route: ParityRoute::MeasurementTransport,
        frequentist: ParadigmCell::Supported(
            "corrected value from the target's own replicates under a declared invariance premise",
        ),
        bayesian: ParadigmCell::Closed(
            "an external latent posterior is bound to its source snapshot and request; it is not transported to another population",
        ),
    },
    ParityRow {
        route: ParityRoute::MechanismTransport,
        frequentist: ParadigmCell::Supported(
            "source-fit sampling covariance by the first-order delta method, labelled; transport adds no variance",
        ),
        bayesian: ParadigmCell::Supported(
            "source-fit posterior covariance by the same propagation, label kept; never mixed with a sampling covariance",
        ),
    },
    ParityRow {
        route: ParityRoute::TransportedSynthesis,
        frequentist: ParadigmCell::Supported(
            "fixed-effect generalized least squares over transported summaries with transport variance counted once",
        ),
        bayesian: ParadigmCell::Supported(
            "hierarchical borrowing with a transported claim as prior; posterior of the common mean",
        ),
    },
    ParityRow {
        route: ParityRoute::TransportedBridge,
        frequentist: ParadigmCell::Supported(
            "point trial-level mapping inside its validated surrogate range under a checked transport premise",
        ),
        bayesian: ParadigmCell::Closed(
            "the bridge mapping is a point relation; no posterior over it is derived",
        ),
    },
    ParityRow {
        route: ParityRoute::TransportedPolicy,
        frequentist: ParadigmCell::Supported(
            "point value of a supplied policy in the target population",
        ),
        bayesian: ParadigmCell::Closed(
            "the value is a point; no posterior over the policy value is derived",
        ),
    },
    ParityRow {
        route: ParityRoute::AdaptiveTrialSource,
        frequentist: ParadigmCell::Supported(
            "adaptively weighted source estimate admitted under the frequentist adaptive-weighting paradigm",
        ),
        bayesian: ParadigmCell::Closed(
            "only the frequentist adaptive-weighting paradigm is admitted as a transport source; a posterior of a policy-selected sample is not a transport source",
        ),
    },
    ParityRow {
        route: ParityRoute::TemporalTransport,
        frequentist: ParadigmCell::Supported(
            "point value across a fixed horizon under time-invariant mechanisms",
        ),
        bayesian: ParadigmCell::Closed(
            "the value is a point; no posterior over the period shift is derived",
        ),
    },
    ParityRow {
        route: ParityRoute::DiscoveredStructureTransport,
        frequentist: ParadigmCell::Closed(
            "the envelope is structural over an equivalence class; no sampling statement about edges is derived",
        ),
        bayesian: ParadigmCell::Closed(
            "no posterior over graphs is transported; a graph posterior is not a single estimand across atoms",
        ),
    },
    ParityRow {
        route: ParityRoute::ProviderHandoff,
        frequentist: ParadigmCell::Supported(
            "a provider law or draws offered as source evidence under its stated meaning",
        ),
        bayesian: ParadigmCell::Supported(
            "a provider parameter or causal-functional posterior offered as source evidence under its stated meaning; no calibration",
        ),
    },
    ParityRow {
        route: ParityRoute::ComposedSpreadExceedance,
        frequentist: ParadigmCell::Supported(
            "uncalibrated spread at the composed coordinate under a licensed dependence route",
        ),
        bayesian: ParadigmCell::Closed(
            "exceedance is decided from supplied composed draws against a cutoff; a posterior statement about the composed quantity has no measured calibration",
        ),
    },
    ParityRow {
        route: ParityRoute::WholePathUncertainty,
        frequentist: ParadigmCell::Supported(
            "structural, exogenous-posterior predictive, sampling and numerical components as an uncalibrated spread",
        ),
        bayesian: ParadigmCell::Closed(
            "nothing is fitted on this row, so there is no parameter posterior; a fitted or sampled mechanism row is a different, closed route",
        ),
    },
    ParityRow {
        route: ParityRoute::DecisionReliabilityUncertainty,
        frequentist: ParadigmCell::Supported(
            "Monte Carlo and selection components over draws whose law is the supplied one; derived, unmeasured",
        ),
        bayesian: ParadigmCell::Supported(
            "the same derivation over posterior draws; the source-distribution spread is carried unchanged, derived, unmeasured",
        ),
    },
    ParityRow {
        route: ParityRoute::AdaptiveProcedureUncertainty,
        frequentist: ParadigmCell::Supported(
            "fixed-sample, group-sequential, weighted-score, anytime-valid and test-inversion reports for the declared procedure",
        ),
        bayesian: ParadigmCell::Supported(
            "posterior validity of a declared prior and decision-theoretic stopping, each its own report",
        ),
    },
    ParityRow {
        route: ParityRoute::DiscretizationBridge,
        frequentist: ParadigmCell::Supported(
            "point mean contrast of the discretized model beside the closed-form Gaussian value, with a computed mean-error bound",
        ),
        bayesian: ParadigmCell::Closed(
            "a discretized value is a point; the source composition's posterior covariance is not propagated through the grid",
        ),
    },
];

/// The row of one route.
#[must_use]
pub fn parity_row(route: ParityRoute) -> &'static ParityRow {
    PARITY_TABLE.iter().find(|row| row.route == route).unwrap_or(&PARITY_TABLE[0])
}

/// Admit a paradigm request on a route, or refuse it by its typed closed row.
///
/// # Errors
/// [`BridgeRefusal`] with `cell_not_licensed`, [`DETAIL_PARADIGM_CLOSED`] and
/// the row's reason as the remedy when the route is closed to the paradigm.
pub fn require_paradigm(
    route: ParityRoute,
    paradigm: InferenceParadigm,
) -> Result<&'static str, BridgeRefusal> {
    let row = parity_row(route);
    let cell = match paradigm {
        InferenceParadigm::Frequentist => row.frequentist,
        InferenceParadigm::Bayesian => row.bayesian,
    };
    match cell {
        ParadigmCell::Supported(what) => Ok(what),
        ParadigmCell::Closed(reason) => Err(BridgeRefusal::new(
            antecedent_core::reason_code!("cell_not_licensed"),
            DETAIL_PARADIGM_CLOSED,
            format!("{}:{}", route.as_str(), paradigm.as_str()),
            reason,
        )),
    }
}
