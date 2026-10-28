//! Fitted nonlinear and non-Gaussian mechanisms carried into a target population's
//! composed model under a declared selection diagram, with target-specific refits.
//!
//! [`super::transported_mechanism`] carries fitted additive-noise linear-Gaussian
//! mechanisms. This module is the same selection-diagram route for typed families: a
//! polynomial or expression conditional mean under Gaussian, heteroscedastic Gaussian,
//! log-normal, Bernoulli, Poisson or gamma noise.
//!
//! 1. [`fit_typed_mechanism`] fits one mechanism from one population's retained data and
//!    returns its parameter covariance; [`refit_typed_mechanism_in_target`] refits a source
//!    mechanism's specification on target data on the source's design, so the two fits'
//!    parameters are comparable.
//! 2. [`transport_typed_mechanisms`] reads the selection diagram per variable. A mechanism
//!    no selection node points into is declared invariant and carried from the source; a
//!    mechanism a selection node points into is shifted and used only from a target fit.
//!    **A shifted mechanism with no target fit is refused**
//!    (`typed_mechanism_transport.shifted_without_target_evidence`,
//!    `transport_missing_evidence`): missing evidence never makes a mechanism invariant.
//! 3. A target fit supplied for a mechanism declared invariant tests that declaration with a
//!    Wald comparison of the full parameter vectors; a rejection refuses the transport
//!    (`typed_mechanism_transport.invariance_contradicted`). A declaration with no target
//!    fit is an untested premise and its disposition says so.
//! 4. The result is a [`TypedTransportedComposition`] with a [`TypedTransportReceipt`]:
//!    route, identity, populations, selection nodes, every component's disposition and
//!    consistency test, and the uncertainty scope.
//!
//! # What is and is not claimed
//!
//! The selection diagram is a premise, never tested; a consistency test that does not
//! reject is not evidence of invariance. Parameter uncertainty is first order, the
//! source-fit covariance for a carried mechanism and the target-fit covariance for a refit
//! one; the transport assumption adds no variance ([`UNCERTAINTY_SCOPE`]). Monte Carlo
//! error is numerical-approximation uncertainty and is reported apart. The receipt's
//! inference label is `point_only`, its support label is `extrapolative`; the interval
//! route stays closed ([`TypedTransportedComposition::interval`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::mechanism_composition::MechanismCompositionRefusal;
use super::mechanism_exports::model_crate::typed_mechanism_transport::{
    fit_mechanism, invariance_check, refit_in_target, transport_typed, TransportedTypedModel,
};
pub use super::mechanism_exports::model_crate::typed_mechanism_transport::{
    selection_targets, typed_refusal_code, ContrastEstimate, CrossBlock, FittedMechanism,
    InvarianceCheck, MeanSpec, MechanismRole, MechanismSpec, ParameterDependence, ResponseFamily,
    TypedDisposition, TypedFitError, TypedFunctionalUncertainty, TypedTransportRequest,
    DEFAULT_INVARIANCE_LEVEL, DETAIL_CYCLE, DETAIL_DEPENDENCE_INCOMPLETE,
    DETAIL_DEPENDENCE_INVALID, DETAIL_DEPENDENCE_UNKNOWN, DETAIL_DESIGN_NOT_SHARED,
    DETAIL_INVARIANCE_CONTRADICTED, DETAIL_INVARIANCE_LEVEL, DETAIL_PARENTS_CHANGED,
    DETAIL_PARENT_WITHOUT_MECHANISM, DETAIL_REGIME_NOT_DECLARED, DETAIL_SELECTION_TARGET_UNKNOWN,
    DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE, DETAIL_TARGET_SPEC_DIFFERS, DETAIL_UNKNOWN_VARIABLE,
    MAX_EXPRESSION_PARAMETERS, MAX_POLYNOMIAL_DEGREE, MAX_TYPED_DRAWS, MAX_TYPED_PARENTS,
    TYPED_TRANSPORT_VERSION, UNCERTAINTY_SCOPE,
};
use antecedent_contract::ExternalRefusal;
use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    CompositionError, MeanContrast, MechanismContext, MechanismProvenance,
};
use std::collections::BTreeMap;

/// Route label of this transport in a [`TypedTransportReceipt`].
pub const TYPED_TRANSPORT_ROUTE: &str = "typed_mechanism";
/// Support label of a transported typed composition: never `supported`.
pub const TYPED_TRANSPORT_SUPPORT: &str = "extrapolative";
/// Inference label of a transported typed composition's point.
pub const TYPED_TRANSPORT_INFERENCE: &str = "point_only";
/// The retained transport identity is not the recomputed one.
pub const DETAIL_TYPED_IDENTITY_MISMATCH: &str =
    "typed_mechanism_transport.transport_identity_mismatch";

/// The typed receipt of one transported typed-mechanism composition.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedTransportReceipt {
    /// Always [`TYPED_TRANSPORT_ROUTE`].
    pub route: &'static str,
    /// Identity of the transported composition (lowercase hex).
    pub identity: String,
    /// Population the carried mechanisms were fitted in.
    pub source_population: String,
    /// Population the composition is about.
    pub target_population: String,
    /// Regime the composition is about.
    pub target_regime: String,
    /// The declared selection nodes: name and the variables each points into.
    pub selection_nodes: Vec<(String, Vec<String>)>,
    /// Variables with a selection arrow into them (sorted, distinct): the shifted ones.
    pub selection_targets: Vec<String>,
    /// Every component, sorted by variable: carried or refitted, family, covariance
    /// source and the consistency test of a carried mechanism against target data.
    pub dispositions: Vec<TypedDisposition>,
    /// Level the declared invariances were tested at.
    pub invariance_level: f64,
    /// Always [`UNCERTAINTY_SCOPE`].
    pub uncertainty_scope: &'static str,
    /// Always [`TYPED_TRANSPORT_SUPPORT`].
    pub support: &'static str,
    /// Always [`TYPED_TRANSPORT_INFERENCE`].
    pub inference: &'static str,
    /// Statements that travel with every value.
    pub assumptions: Vec<String>,
}

/// An executable target-population composition of typed mechanisms with its receipt.
#[derive(Clone, Debug)]
pub struct TypedTransportedComposition {
    inner: TransportedTypedModel,
    receipt: TypedTransportReceipt,
}

fn refusal(
    code: &'static str,
    stage: &'static str,
    detail: &str,
    subject: String,
    remedy: &'static str,
) -> MechanismCompositionRefusal {
    MechanismCompositionRefusal {
        refusals: vec![ExternalRefusal::new(code, stage, detail)
            .with_offending(subject)
            .with_remedy(remedy)],
    }
}

fn model_error(error: CompositionError) -> MechanismCompositionRefusal {
    match error {
        CompositionError::Incoherent { findings } => MechanismCompositionRefusal {
            refusals: findings
                .iter()
                .map(|f| {
                    let code = typed_refusal_code(f.detail).unwrap_or(
                        antecedent_core::reason_code!("mechanism_composition_incoherent"),
                    );
                    let mut r = ExternalRefusal::new(code, "compose", f.detail)
                        .with_offending(f.subject.clone())
                        .with_remedy(f.remedy);
                    if let (Some(expected), Some(supplied)) = (&f.expected, &f.supplied) {
                        r = r.with_expected(expected.clone()).with_supplied(supplied.clone());
                    }
                    r
                })
                .collect(),
        },
        CompositionError::Cancelled => refusal(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "compose",
            "mechanism_composition.cancelled",
            "transport".to_owned(),
            "retry; cancellation is not a verdict on the mechanisms",
        ),
        CompositionError::Numerical { message } => refusal(
            antecedent_core::reason_code!("invalid_argument"),
            "compose",
            "mechanism_composition.numerical_failure",
            message,
            "check the mechanisms' numbers",
        ),
        other => refusal(
            antecedent_core::reason_code!("route_not_supported"),
            "compose",
            "mechanism_composition.request_unsupported",
            other.to_string(),
            "use a composition of fitted typed mechanisms and a bounded draw count",
        ),
    }
}

fn fit_error(error: &TypedFitError) -> MechanismCompositionRefusal {
    refusal(
        error.reason_code(),
        "compose",
        "typed_mechanism_transport.fit_refused",
        error.to_string(),
        "check the specification, the data's support and the design; coarsen or drop parents that separate the response",
    )
}

/// Fit one typed mechanism from one population's retained data.
///
/// # Errors
/// A refusal naming the malformed specification or data, the support violation, the
/// singular design or the non-convergence (including complete separation).
pub fn fit_typed_mechanism(
    spec: &MechanismSpec,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
    design_from: Option<&FittedMechanism>,
) -> Result<FittedMechanism, MechanismCompositionRefusal> {
    fit_mechanism(spec, columns, context, provenance, design_from).map_err(|e| fit_error(&e))
}

/// Refit a source mechanism's specification on target data, on the source's design.
///
/// # Errors
/// As [`fit_typed_mechanism`].
pub fn refit_typed_mechanism_in_target(
    source: &FittedMechanism,
    columns: &BTreeMap<String, Vec<f64>>,
    context: MechanismContext,
    provenance: MechanismProvenance,
) -> Result<FittedMechanism, MechanismCompositionRefusal> {
    refit_in_target(source, columns, context, provenance).map_err(|e| fit_error(&e))
}

/// Wald comparison of a source and a target fit of one mechanism.
///
/// # Errors
/// A refusal when the fits do not share a specification or design.
pub fn check_typed_invariance(
    source: &FittedMechanism,
    target: &FittedMechanism,
    level: f64,
) -> Result<InvarianceCheck, MechanismCompositionRefusal> {
    invariance_check(source, target, level).map_err(model_error)
}

fn assumptions_of(request: &TypedTransportRequest, inner: &TransportedTypedModel) -> Vec<String> {
    let mut out = vec![
        "selection_diagram_declared_not_tested".to_owned(),
        format!(
            "selection arrows into {{{}}} only; every other fitted mechanism is declared \
             invariant between {} and {}",
            selection_targets(&request.diagram).join(","),
            request.diagram.source_population,
            request.diagram.target_population
        ),
        "a shifted mechanism is used only from a target fit; missing target evidence refuses \
         and never makes a mechanism invariant"
            .to_owned(),
        format!("parameter uncertainty scope: {UNCERTAINTY_SCOPE}"),
        "monte carlo error of a response is numerical approximation, reported apart from \
         parameter uncertainty"
            .to_owned(),
    ];
    let carried = inner.transported_variables();
    if carried.is_empty() {
        out.push("no mechanism was carried: every component was refitted in the target".to_owned());
    } else {
        out.push(format!("carried from the source fit: {}", carried.join(",")));
    }
    let untested: Vec<&str> = inner
        .dispositions()
        .iter()
        .filter(|d| d.role == MechanismRole::Transported && d.consistency.is_none())
        .map(|d| d.variable.as_str())
        .collect();
    if !untested.is_empty() {
        out.push(format!(
            "invariance of {} is an untested premise: no target data were supplied for it",
            untested.join(",")
        ));
    }
    out
}

/// Carry invariant typed mechanisms from the source and use target refits for shifted ones.
///
/// # Errors
/// A [`MechanismCompositionRefusal`] naming every unmet premise; no model is issued.
pub fn transport_typed_mechanisms(
    request: &TypedTransportRequest,
    ctx: &ExecutionContext,
) -> Result<TypedTransportedComposition, MechanismCompositionRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(model_error(CompositionError::Cancelled));
    }
    let inner = transport_typed(request).map_err(model_error)?;
    let receipt = TypedTransportReceipt {
        route: TYPED_TRANSPORT_ROUTE,
        identity: inner.identity_hex(),
        source_population: request.diagram.source_population.clone(),
        target_population: request.target.population.clone(),
        target_regime: request.target.regime.clone(),
        selection_nodes: request
            .diagram
            .selection_nodes
            .iter()
            .map(|n| (n.name.clone(), n.points_into.clone()))
            .collect(),
        selection_targets: selection_targets(&request.diagram),
        dispositions: inner.dispositions().to_vec(),
        invariance_level: request.invariance_level,
        uncertainty_scope: UNCERTAINTY_SCOPE,
        support: TYPED_TRANSPORT_SUPPORT,
        inference: TYPED_TRANSPORT_INFERENCE,
        assumptions: assumptions_of(request, &inner),
    };
    Ok(TypedTransportedComposition { inner, receipt })
}

impl TypedTransportedComposition {
    /// The library's transported model.
    #[must_use]
    pub const fn model(&self) -> &TransportedTypedModel {
        &self.inner
    }

    /// The typed receipt.
    #[must_use]
    pub const fn receipt(&self) -> &TypedTransportReceipt {
        &self.receipt
    }

    /// Identity of the transported composition as lowercase hex.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        self.inner.identity_hex()
    }

    /// The standing this composition states as a claim in a composition graph: built from
    /// mechanisms the library composed, so no provider trust; the interval route is closed,
    /// so no calibration record is claimed; support `extrapolative`.
    #[must_use]
    pub fn claim_standing(&self) -> super::composition_verification::ClaimStanding {
        super::composition_verification::library_transported_standing(TYPED_TRANSPORT_SUPPORT, &[])
    }

    /// The composition-graph node of this transported composition under `key`.
    #[must_use]
    pub fn composition_node(&self, key: &str) -> super::composition_verification::CompositionNode {
        super::composition_verification::library_transported_node(
            key,
            TYPED_TRANSPORT_ROUTE,
            &self.identity_hex(),
            &self.receipt.target_population,
            TYPED_TRANSPORT_SUPPORT,
            Vec::new(),
        )
    }

    /// Every component, sorted by variable.
    #[must_use]
    pub fn dispositions(&self) -> &[TypedDisposition] {
        self.inner.dispositions()
    }

    /// Variables whose mechanism was carried from the source, sorted.
    #[must_use]
    pub fn transported_variables(&self) -> Vec<&str> {
        self.inner.transported_variables()
    }

    /// `sum_k weight_k * E[outcome | do(arm_k)]` by Monte Carlo over `draws` paired units.
    ///
    /// # Errors
    /// An unknown variable, a malformed contrast, a draw count outside the bound, a
    /// non-finite draw, or cancellation.
    pub fn mean_contrast(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<ContrastEstimate, MechanismCompositionRefusal> {
        self.inner.mean_contrast(contrast, draws, ctx).map_err(model_error)
    }

    /// First-order parameter uncertainty of a contrast, scoped by [`UNCERTAINTY_SCOPE`].
    ///
    /// # Errors
    /// As [`Self::mean_contrast`]; unknown or incomplete parameter dependence between the
    /// involved mechanisms is a refusal, never an independence default.
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<TypedFunctionalUncertainty, MechanismCompositionRefusal> {
        self.inner.mean_contrast_uncertainty(contrast, draws, ctx).map_err(model_error)
    }

    /// An interval for a transported contrast.
    ///
    /// # Errors
    /// Always `cell_not_licensed`: no measured coverage stands behind the variance, and
    /// none includes the transport assumption.
    pub fn interval(&self) -> Result<std::convert::Infallible, MechanismCompositionRefusal> {
        Err(refusal(
            antecedent_core::reason_code!("cell_not_licensed"),
            "compose",
            "typed_mechanism_transport.interval_route_closed",
            "interval".to_owned(),
            "the first-order parameter variance omits the transport assumption and has no \
             measured coverage; the point and the labelled variance are the result",
        ))
    }
}

/// Recompute the transport and require its identity to be the retained one.
///
/// # Errors
/// Whatever [`transport_typed_mechanisms`] refuses, and
/// `typed_mechanism_transport.transport_identity_mismatch` when the recomputed identity is
/// not `expected_identity_hex`.
pub fn verify_typed_transport(
    request: &TypedTransportRequest,
    expected_identity_hex: &str,
    ctx: &ExecutionContext,
) -> Result<TypedTransportedComposition, MechanismCompositionRefusal> {
    let transported = transport_typed_mechanisms(request, ctx)?;
    if transported.identity_hex() != expected_identity_hex {
        return Err(refusal(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            "compose",
            DETAIL_TYPED_IDENTITY_MISMATCH,
            format!("recomputed={} retained={expected_identity_hex}", transported.identity_hex()),
            "re-derive the transport from the retained fits and declaration",
        ));
    }
    Ok(transported)
}
