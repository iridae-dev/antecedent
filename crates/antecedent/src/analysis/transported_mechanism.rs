//! Fitted linear-Gaussian mechanisms carried into a target population's composed model
//! under a declared selection diagram.
//!
//! [`super::mechanism_composition`] composes mechanisms for one declared population and
//! admits a mechanism of another population or regime only through an explicit
//! [`antecedent_contract::TransportAdmission`]. This module is the executed transport
//! route of the same prepared composition: given the selection diagram from the
//! composition's population to a target population, the mechanisms natively refitted in
//! the target, and any explicit invariance declarations, it returns an executable
//! target-population model whose every component says whether it was carried or refitted,
//! or refuses with the named unmet premise.
//!
//! 1. [`PreparedMechanismComposition::transport_to`] reads the prepared composition's
//!    mechanisms as the source mechanisms, builds the target mechanisms from the declared
//!    refits (same variable, parents, units and support; new numbers, evidence and
//!    population), and runs the library route [`transport_linear_gaussian`].
//! 2. **A selection arrow dominates an invariance declaration.** A variable (or a latent
//!    cause it loads on) that a selection node points into is never carried, whatever
//!    invariance is declared for it; it needs a natively refitted target mechanism, and a
//!    refit always replaces the carried one (the two are never averaged).
//! 3. The result is a [`TransportedMechanismComposition`] with a typed
//!    [`TransportedMechanismReceipt`]: route, identity, source plan, populations, the
//!    selection nodes, every component's disposition and the uncertainty scope.
//!
//! # What is and is not claimed
//!
//! The selection diagram and invariance declarations are premises, never tested. The
//! parameter covariance of a carried mechanism is that of its source fit; the transport
//! assumption adds no variance term and the result says so ([`UNCERTAINTY_SCOPE`]). The
//! variance is first-order and uncalibrated and the interval route stays closed
//! ([`TransportedMechanismComposition::interval`]). The receipt's inference label is
//! `point_only`, its support label is `extrapolative`. The route executes fitted
//! additive-noise linear-Gaussian mechanisms only; a finite table is decided by the finite
//! counterfactual transport derivation, and a declared or external mechanism is not
//! carried (`mechanism_transport.origin_not_fitted`).
//!
//! # Export and consumption
//!
//! [`TransportedMechanismComposition::export`] seals the transported specification, with
//! its selection diagram and invariances, into the existing `mechanism_composition`
//! artifact; [`super::mechanism_composition::consume_mechanism_composition_artifact`]
//! recomputes the premises digest, recompiles under the coherence checker (which refuses a
//! selection arrow into a carried mechanism or its latent) and replays the query.
//! [`verify_mechanism_transport`] is the fresh-consumer path for the receipt: it
//! recomputes the transport from the prepared composition and the declaration and refuses
//! unless the retained identity is the recomputed one.
//!
//! # Refresh with outcome-only data
//!
//! The transported model is a pure function of the request. New outcome-only data for a
//! source mechanism is a refit of the source composition
//! ([`PreparedMechanismComposition::refresh_mechanisms`]) and then a new transport; new
//! data for a target-native mechanism replaces that refit in the declaration. The diagram
//! and invariances are kept only by being passed again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use super::mechanism_exports::model_crate::mechanism_transport::{
    DETAIL_DIAGRAM_SOURCE_MISMATCH, DETAIL_DIAGRAM_TARGET_MISMATCH, DETAIL_DUPLICATE_MECHANISM,
    DETAIL_FAMILY_UNSUPPORTED, DETAIL_ORIGIN_NOT_FITTED, DETAIL_SELECTION_INTO_LATENT,
    DETAIL_SELECTION_INTO_MECHANISM, DETAIL_TARGET_NOT_NATIVE, MECHANISM_TRANSPORT_VERSION,
    MechanismDisposition, MechanismRole, MechanismTransportRequest,
    TransportedFunctionalUncertainty, TransportedLinearGaussian, UNCERTAINTY_SCOPE, refusal_code,
    selection_targets, transport_linear_gaussian,
};
use antecedent_contract::ExternalRefusal;
use antecedent_core::ExecutionContext;
use antecedent_io::mechanism_composition_artifact::{
    CompositionQueryWire, MechanismCompositionArtifactWire, refusals_for_findings,
};
use antecedent_model::compose::{
    CompositionError, InvarianceDeclaration, MeanContrast, MechanismContext, MechanismDescriptor,
    MechanismForm, MechanismOrigin, MechanismProvenance, PosteriorDependence,
    SelectionDiagramDeclaration, SelectionNodeDeclaration,
};

use super::mechanism_composition::{
    DependenceInput, MechanismCompositionExport, MechanismCompositionRefusal, MechanismRefit,
    MechanismSource, PreparedMechanismComposition,
};

/// Route label of this transport in a [`TransportedMechanismReceipt`].
pub const MECHANISM_TRANSPORT_ROUTE: &str = "mechanism";
/// Support label of a transported mechanism composition: never `supported`.
pub const MECHANISM_TRANSPORT_SUPPORT: &str = "extrapolative";
/// Inference label of a transported mechanism composition's point.
pub const MECHANISM_TRANSPORT_INFERENCE: &str = "point_only";
/// The retained transport identity is not the recomputed one.
pub const DETAIL_TRANSPORT_IDENTITY_MISMATCH: &str =
    "mechanism_transport.transport_identity_mismatch";
/// A target refit names a variable the composition has no mechanism for.
pub const DETAIL_REFIT_UNKNOWN_VARIABLE: &str =
    "mechanism_composition.identity_declaration_unknown_variable";
/// A target refit changes the family of a mechanism.
pub const DETAIL_REFIT_FAMILY_CHANGED: &str = "mechanism_composition.family_mixed";

/// What a transport of a prepared composition to a target population declares.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismTransportDeclaration {
    /// Population the transported composition is about.
    pub target_population_id: String,
    /// Regime the transported composition is about; `observational` is explicit.
    pub target_regime_id: String,
    /// The selection diagram's nodes: the variables each points into. The diagram runs
    /// from the prepared composition's population to `target_population_id`.
    pub selection_nodes: Vec<SelectionNodeDeclaration>,
    /// Mechanisms natively refitted in the target population. Each replaces the carried
    /// mechanism of its variable.
    pub target_refits: Vec<MechanismRefit>,
    /// Explicit invariance declarations (a carried mechanism that loads on a shared latent
    /// cause, or a carry across regimes). A selection arrow still dominates them.
    pub invariances: Vec<InvarianceDeclaration>,
    /// Declared dependence between the transported mechanisms' parameter estimates.
    pub dependence: DependenceInput,
}

/// The typed receipt of one transported mechanism composition.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedMechanismReceipt {
    /// Always [`MECHANISM_TRANSPORT_ROUTE`].
    pub route: &'static str,
    /// Identity of the transported composition (lowercase hex).
    pub identity: String,
    /// Plan identity of the prepared composition the mechanisms were carried from.
    pub source_plan_identity: String,
    /// Version of the source plan's semantic identity encoding.
    pub source_plan_identity_version: u64,
    /// Population the mechanisms were fitted in.
    pub source_population: String,
    /// Population the composition is about.
    pub target_population: String,
    /// Regime the composition is about.
    pub target_regime: String,
    /// The declared selection nodes: name and the variables each points into.
    pub selection_nodes: Vec<(String, Vec<String>)>,
    /// Variables with a selection arrow into them (sorted, distinct).
    pub selection_targets: Vec<String>,
    /// Every component, sorted by variable: carried or refitted, and from where.
    pub dispositions: Vec<MechanismDisposition>,
    /// Declared invariances: variable and assumption identity.
    pub invariances: Vec<(String, String)>,
    /// Always [`UNCERTAINTY_SCOPE`].
    pub uncertainty_scope: &'static str,
    /// Always [`MECHANISM_TRANSPORT_SUPPORT`].
    pub support: &'static str,
    /// Always [`MECHANISM_TRANSPORT_INFERENCE`].
    pub inference: &'static str,
    /// Statements that travel with every value.
    pub assumptions: Vec<String>,
}

/// An executable target-population composition with its transport record.
#[derive(Clone, Debug)]
pub struct TransportedMechanismComposition {
    inner: TransportedLinearGaussian,
    receipt: TransportedMechanismReceipt,
}

fn model_error(error: CompositionError) -> MechanismCompositionRefusal {
    let one = |code: &'static str, detail: &str, subject: String, remedy: &'static str| {
        MechanismCompositionRefusal {
            refusals: vec![
                ExternalRefusal::new(code, "compose", detail)
                    .with_offending(subject)
                    .with_remedy(remedy),
            ],
        }
    };
    match error {
        CompositionError::Incoherent { findings } => {
            let mut refusals = refusals_for_findings(&findings);
            for refusal in &mut refusals {
                if let Some(code) = refusal_code(&refusal.detail) {
                    refusal.code = code;
                }
            }
            MechanismCompositionRefusal { refusals }
        }
        CompositionError::Cancelled => one(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "mechanism_composition.cancelled",
            "transport".to_owned(),
            "retry; cancellation is not a verdict on the mechanisms",
        ),
        CompositionError::Numerical { message } => one(
            antecedent_core::reason_code!("invalid_argument"),
            "mechanism_composition.numerical_failure",
            message,
            "check the mechanisms' numbers",
        ),
        other => one(
            antecedent_core::reason_code!("route_not_supported"),
            "mechanism_composition.request_unsupported",
            other.to_string(),
            "use a linear-Gaussian composition of fitted mechanisms",
        ),
    }
}

fn form_of(source: &MechanismSource) -> MechanismForm {
    match source {
        MechanismSource::FittedLinear { intercept, sigma, coefficients }
        | MechanismSource::DeclaredLinear { intercept, sigma, coefficients } => {
            MechanismForm::LinearGaussian {
                intercept: *intercept,
                sigma: *sigma,
                coefficients: coefficients.clone(),
            }
        }
        MechanismSource::DeclaredTable { levels, rows } => {
            MechanismForm::Tabular { levels: levels.clone(), rows: rows.clone() }
        }
    }
}

fn origin_of(source: &MechanismSource) -> MechanismOrigin {
    match source {
        MechanismSource::FittedLinear { .. } => MechanismOrigin::Fitted,
        MechanismSource::DeclaredLinear { .. } | MechanismSource::DeclaredTable { .. } => {
            MechanismOrigin::Declared
        }
    }
}

fn dependence_of(input: &DependenceInput) -> PosteriorDependence {
    match input {
        DependenceInput::Unknown => PosteriorDependence::Unknown,
        DependenceInput::IndependentByDesign { justification } => {
            PosteriorDependence::IndependentByDesign { justification: justification.clone() }
        }
        DependenceInput::AlignedJointLaw { meaning, parameters, covariance, .. } => {
            PosteriorDependence::Joint {
                meaning: *meaning,
                parameters: parameters.clone(),
                covariance: covariance.clone(),
            }
        }
        DependenceInput::CrossCovariance { blocks } => {
            PosteriorDependence::CrossCovariance { blocks: blocks.clone() }
        }
    }
}

fn refit_refusal(
    code: &'static str,
    detail: &str,
    subject: String,
    remedy: &'static str,
) -> MechanismCompositionRefusal {
    MechanismCompositionRefusal {
        refusals: vec![
            ExternalRefusal::new(code, "compose", detail)
                .with_offending(subject)
                .with_remedy(remedy),
        ],
    }
}

impl PreparedMechanismComposition {
    /// Carry this composition's fitted mechanisms into the target population's model.
    ///
    /// Order of refusal: cancellation, target refits against the composition (unknown
    /// variable, changed family), then every finding of the library route in canonical
    /// order: a carried mechanism that is not fitted or not linear-Gaussian, a diagram for
    /// other populations, a selection arrow into a carried mechanism or its latent, a
    /// variable with two mechanisms in one role, and everything the composition checker
    /// finds in the assembled specification.
    ///
    /// Takes `&self` and returns a new value, so a refused transport leaves this
    /// composition untouched.
    ///
    /// # Errors
    /// A [`MechanismCompositionRefusal`] naming every unmet premise; no model is issued.
    pub fn transport_to(
        &self,
        declaration: &MechanismTransportDeclaration,
        ctx: &ExecutionContext,
    ) -> Result<TransportedMechanismComposition, MechanismCompositionRefusal> {
        if ctx.cancellation.is_cancelled() {
            return Err(model_error(CompositionError::Cancelled));
        }
        let spec = self.spec();
        let source_context = spec.target.clone();
        let mut target_mechanisms = Vec::with_capacity(declaration.target_refits.len());
        for refit in &declaration.target_refits {
            let Some(template) = spec.mechanisms.iter().find(|m| m.variable == refit.variable_id)
            else {
                return Err(refit_refusal(
                    antecedent_core::reason_code!("quantity_semantics_mismatch"),
                    DETAIL_REFIT_UNKNOWN_VARIABLE,
                    refit.variable_id.clone(),
                    "refit only a variable the composition has a mechanism for",
                ));
            };
            let form = form_of(&refit.source);
            if form.family() != template.form.family() {
                return Err(refit_refusal(
                    antecedent_core::reason_code!("mechanism_composition_incoherent"),
                    DETAIL_REFIT_FAMILY_CHANGED,
                    refit.variable_id.clone(),
                    "a target refit keeps the mechanism's family and parents",
                ));
            }
            let mut descriptor: MechanismDescriptor = template.clone();
            descriptor.form = form;
            descriptor.context = MechanismContext {
                population: declaration.target_population_id.clone(),
                regime: declaration.target_regime_id.clone(),
            };
            descriptor.provenance = MechanismProvenance {
                origin: origin_of(&refit.source),
                artifact_id: refit.artifact_id.clone(),
                evidence_id: refit.evidence_id.clone(),
            };
            descriptor.uncertainty.clone_from(&refit.uncertainty);
            target_mechanisms.push(descriptor);
        }
        let diagram = SelectionDiagramDeclaration {
            source_population: source_context.population.clone(),
            target_population: declaration.target_population_id.clone(),
            selection_nodes: declaration.selection_nodes.clone(),
        };
        let request = MechanismTransportRequest {
            target: MechanismContext {
                population: declaration.target_population_id.clone(),
                regime: declaration.target_regime_id.clone(),
            },
            diagram: diagram.clone(),
            source_mechanisms: spec.mechanisms.clone(),
            target_mechanisms,
            edges: spec.edges.clone(),
            invariances: declaration.invariances.clone(),
            dependence: dependence_of(&declaration.dependence),
        };
        let inner = transport_linear_gaussian(&request).map_err(model_error)?;
        let identity = inner.identity_hex();
        let receipt = TransportedMechanismReceipt {
            route: MECHANISM_TRANSPORT_ROUTE,
            identity,
            source_plan_identity: self.plan_identity_hex(),
            source_plan_identity_version: self.plan_identity_version(),
            source_population: source_context.population,
            target_population: declaration.target_population_id.clone(),
            target_regime: declaration.target_regime_id.clone(),
            selection_nodes: diagram
                .selection_nodes
                .iter()
                .map(|n| (n.name.clone(), n.points_into.clone()))
                .collect(),
            selection_targets: selection_targets(&diagram),
            dispositions: inner.dispositions().to_vec(),
            invariances: declaration
                .invariances
                .iter()
                .map(|i| (i.variable.clone(), i.assumption_id.clone()))
                .collect(),
            uncertainty_scope: UNCERTAINTY_SCOPE,
            support: MECHANISM_TRANSPORT_SUPPORT,
            inference: MECHANISM_TRANSPORT_INFERENCE,
            assumptions: assumptions_of(&diagram, &inner),
        };
        Ok(TransportedMechanismComposition { inner, receipt })
    }
}

fn assumptions_of(
    diagram: &SelectionDiagramDeclaration,
    inner: &TransportedLinearGaussian,
) -> Vec<String> {
    let mut out = vec![
        "selection_diagram_declared_not_tested".to_owned(),
        format!(
            "selection arrows into {{{}}} only; every other fitted mechanism is invariant between \
             {} and {}",
            selection_targets(diagram).join(","),
            diagram.source_population,
            diagram.target_population
        ),
        "a natively refitted target mechanism replaces the carried one and is never averaged \
         with it"
            .to_owned(),
        format!("parameter uncertainty scope: {UNCERTAINTY_SCOPE}"),
    ];
    let carried = inner.transported_variables();
    if carried.is_empty() {
        out.push("no mechanism was carried: every component was refitted in the target".to_owned());
    } else {
        out.push(format!("carried from the source fit: {}", carried.join(",")));
    }
    out
}

impl TransportedMechanismComposition {
    /// The library's transported model.
    #[must_use]
    pub const fn model(&self) -> &TransportedLinearGaussian {
        &self.inner
    }

    /// The typed receipt.
    #[must_use]
    pub const fn receipt(&self) -> &TransportedMechanismReceipt {
        &self.receipt
    }

    /// Identity of the transported composition as lowercase hex.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        self.inner.identity_hex()
    }

    /// The standing this composition states as a claim in a composition graph:
    /// built from mechanisms the library composed, so no provider trust; the
    /// interval route is closed ([`Self::interval`]), so no calibration record is
    /// claimed; support `extrapolative` ([`MECHANISM_TRANSPORT_SUPPORT`]).
    #[must_use]
    pub fn claim_standing(&self) -> super::composition_verification::ClaimStanding {
        super::composition_verification::library_transported_standing(
            MECHANISM_TRANSPORT_SUPPORT,
            &[],
        )
    }

    /// The composition-graph node of this transported composition under `key`,
    /// carrying [`Self::claim_standing`]. Its identity is the composition's own
    /// identity, so a changed source mechanism, selection diagram, refit or
    /// invariance recomputes it.
    #[must_use]
    pub fn composition_node(&self, key: &str) -> super::composition_verification::CompositionNode {
        super::composition_verification::library_transported_node(
            key,
            MECHANISM_TRANSPORT_ROUTE,
            &self.identity_hex(),
            &self.receipt.target_population,
            MECHANISM_TRANSPORT_SUPPORT,
            Vec::new(),
        )
    }

    /// Every component, sorted by variable.
    #[must_use]
    pub fn dispositions(&self) -> &[MechanismDisposition] {
        self.inner.dispositions()
    }

    /// Variables whose mechanism was carried from the source, sorted.
    #[must_use]
    pub fn transported_variables(&self) -> Vec<&str> {
        self.inner.transported_variables()
    }

    /// `sum_k weight_k * E[outcome | do(arm_k)]` in the target composition.
    ///
    /// # Errors
    /// As [`PreparedMechanismComposition::mean_contrast`].
    pub fn mean_contrast(
        &self,
        contrast: &MeanContrast,
    ) -> Result<f64, MechanismCompositionRefusal> {
        self.inner.mean_contrast(contrast).map_err(model_error)
    }

    /// First-order parameter uncertainty of a contrast, scoped by [`UNCERTAINTY_SCOPE`]:
    /// a carried mechanism's covariance is its source fit's, and the selection diagram is
    /// not a variance term. Uncalibrated; no interval follows.
    ///
    /// # Errors
    /// A missing marginal law or unknown dependence (`joint_law_required`).
    pub fn mean_contrast_uncertainty(
        &self,
        contrast: &MeanContrast,
    ) -> Result<TransportedFunctionalUncertainty, MechanismCompositionRefusal> {
        self.inner.mean_contrast_uncertainty(contrast).map_err(model_error)
    }

    /// An interval for a transported contrast.
    ///
    /// # Errors
    /// Always `cell_not_licensed`: no measured coverage stands behind the variance, and
    /// none includes the transport assumption.
    pub fn interval(&self) -> Result<std::convert::Infallible, MechanismCompositionRefusal> {
        Err(refit_refusal(
            antecedent_core::reason_code!("cell_not_licensed"),
            "mechanism_transport.interval_route_closed",
            "interval".to_owned(),
            "the first-order parameter variance omits the transport assumption and has no \
             measured coverage; the point and the labelled variance are the result",
        ))
    }

    /// Seal the transported premises (with the selection diagram and invariances) and one
    /// query into a `mechanism_composition` artifact.
    ///
    /// # Errors
    /// A query the composition cannot answer, and an uncertainty claim without a declared
    /// joint law.
    pub fn export(
        &self,
        query: CompositionQueryWire,
        ctx: &ExecutionContext,
    ) -> Result<MechanismCompositionExport, MechanismCompositionRefusal> {
        if ctx.cancellation.is_cancelled() {
            return Err(model_error(CompositionError::Cancelled));
        }
        let wire = MechanismCompositionArtifactWire::seal(self.inner.model().spec(), query, ctx)
            .map_err(|error| MechanismCompositionRefusal {
                refusals: vec![
                    ExternalRefusal::new(
                        antecedent_core::reason_code!("mechanism_composition_incoherent"),
                        "export",
                        "mechanism_composition.artifact_invalid",
                    )
                    .with_offending(error.to_string()),
                ],
            })?;
        let bytes = wire.export().map_err(|error| MechanismCompositionRefusal {
            refusals: vec![
                ExternalRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    "export",
                    "mechanism_composition.artifact_invalid",
                )
                .with_offending(error.to_string()),
            ],
        })?;
        Ok(MechanismCompositionExport {
            bytes,
            premises_digest: wire.premises_digest.clone(),
            result: wire.result,
        })
    }
}

/// Recompute the transport and require its identity to be the retained one.
///
/// The fresh-consumer path: the receipt is re-derived from the prepared composition and
/// the declaration, never read from a stored copy.
///
/// # Errors
/// Whatever [`PreparedMechanismComposition::transport_to`] refuses, and
/// `mechanism_transport.transport_identity_mismatch` when the recomputed identity is not
/// `expected_identity_hex`.
pub fn verify_mechanism_transport(
    prepared: &PreparedMechanismComposition,
    declaration: &MechanismTransportDeclaration,
    expected_identity_hex: &str,
    ctx: &ExecutionContext,
) -> Result<TransportedMechanismComposition, MechanismCompositionRefusal> {
    let transported = prepared.transport_to(declaration, ctx)?;
    if transported.identity_hex() != expected_identity_hex {
        return Err(refit_refusal(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            DETAIL_TRANSPORT_IDENTITY_MISMATCH,
            format!("recomputed={} retained={expected_identity_hex}", transported.identity_hex()),
            "re-derive the transport from the retained composition and declaration",
        ));
    }
    Ok(transported)
}
