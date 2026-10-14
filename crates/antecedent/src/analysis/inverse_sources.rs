//! Forward sources of an inverse query, and the associational observational route.
//!
//! An inverse query ([`super::inverse_query`]) asks which declared actions meet a
//! target on a forward claim. The claim is only as good as the route that
//! produced it, so each source is adapted with its own refusals and its identity
//! travels into the result and the sealed artifact:
//!
//! * **Composed mechanisms.**
//!   [`PreparedMechanismComposition::inverse_forward_source`] exports each
//!   action's composition query, consumes the artifact again (premises digest
//!   recomputed, premises recompiled, query replayed to identical bits) and
//!   hands the replayed results to [`composed_forward_source`]. A linear-Gaussian
//!   composition supplies a point per action; a finite-table composition the
//!   exact finite law of each action under the caller's numeric coding of the
//!   outcome levels.
//! * **Cohort contrast.** [`cohort_source`]: the weighted mean contrast of a
//!   retained effect model over a declared cohort, a point. A cohort under a
//!   declared covariate shift is a transported response and is read only under
//!   [`TransportedInputPolicy::AcceptsTransported`].
//! * **Transported response.** [`transported_source`]: a value carried to a
//!   target population, as a point or a transported law.
//!
//! Every source is refused an interval region
//! ([`ForwardSourceRefusal::IntervalNotLicensed`]); a point is refused a
//! quantile, a probability, a nonlinear utility and an outcome-law input. A
//! source's own variance is an uncalibrated diagnostic and never becomes a
//! feasibility statement.
//!
//! # Lifecycle
//!
//! * **Evaluate / seal.** [`evaluate`] returns the engine's result beside the
//!   source's [`ForwardOrigin`]; [`seal`] binds the origin into an inverse-query
//!   artifact that [`super::inverse_query::export`] writes and
//!   [`super::inverse_query::consume`] re-derives (the consumer recomputes the
//!   claim digest and refuses an origin that does not describe the embedded
//!   claim). The origin is a declaration of where the numbers came from; a
//!   replay checks the numbers are the ones the origin names, not that the
//!   origin is authentic.
//! * **Refresh.** New outcome-only data refits the source model, and a refit
//!   composition or a recomputed cohort or transported response is a *new
//!   source*: **recompute** the source law, claim digest and receipt; **keep**
//!   the declared query grid and constraints; **refuse** data of another
//!   population, quantity or arm (the coordinate and transport checks refuse it
//!   when the new source is built). A sealed artifact is never refreshed in
//!   place.
//!
//! # Observational scenarios
//!
//! [`evaluate_observational`] answers "which declared covariate scenarios do
//! the units that reach this outcome fall in" by Bayes inversion of a supplied
//! aligned observational law. The result is associational: it ranks no action
//! (`ranks_actions` is `false`), is not an intervention effect and carries no
//! interval. [`verify_observational`] re-derives a stored result from the law
//! and the query. The law is an input; new data means a new law and a new
//! result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::ExecutionContext;
use antecedent_io::cohort_decision::CohortDecisionSource;
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_model::compose::MechanismFamily;

pub use antecedent_contract::provider_transport::TransportedInputPolicy;
pub use antecedent_design::decision_transported_source::{
    TransportDeclaration, TransportReceipt, TransportedRefusal, TransportedResponse,
    TransportedRoute, TransportedValue,
};
pub use antecedent_design::inverse_forward_source::{
    ComposedActionResponse, ComposedFamily, ComposedResponse, FORWARD_ORIGIN_VERSION,
    ForwardOrigin, ForwardRepresentation, ForwardSourceKind, ForwardSourceRefusal, InverseDemand,
    LicensedForwardSource, SourcedInverseResult, forward_claim_digest, query_needs_law,
};
use antecedent_design::inverse_forward_source::{
    cohort_forward_source, composed_forward_source, composed_forward_source_accepting,
    evaluate_inverse_on_source, seal_on_source, seal_per_action_on_source,
    transported_forward_source,
};
pub use antecedent_design::inverse_observational_scenarios::{
    MAX_OBSERVATIONAL_SCENARIOS, OBSERVATIONAL_SCENARIO_SCOPE_NOTE, OBSERVATIONAL_SCENARIO_VERSION,
    ObservationalScenarioError, ObservationalScenarioQuery, ObservationalScenarioResult,
    OutcomeEvent, ScenarioCell, ScenarioCondition, ScenarioReport,
};
use antecedent_design::inverse_observational_scenarios::{
    evaluate_observational_scenarios, verify_observational_scenarios,
};

use super::inverse_query::{InverseQuery, InverseQueryArtifact};
use super::mechanism_composition::{
    CompositionQueryWire, MechanismCompositionRefusal, PreparedMechanismComposition,
    consume_mechanism_composition_artifact,
};

/// One action's composition query and how its outcome is coded.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedActionQuery {
    /// The declared action this query answers.
    pub action_id: String,
    /// The coordinate the action's utility reads (the interventional mean of
    /// the outcome, or the contrast).
    pub coordinate: ScientificQuantity,
    /// The composition query: a Gaussian intervention law or contrast for a
    /// linear-Gaussian composition, a finite intervention distribution for a
    /// finite-table composition.
    pub query: CompositionQueryWire,
    /// Numeric value of each outcome level, for a finite distribution only.
    pub level_values: Vec<(String, f64)>,
}

fn composition_refusal(error: &MechanismCompositionRefusal) -> ForwardSourceRefusal {
    let detail = error.details().first().copied().unwrap_or("mechanism_composition.artifact_invalid");
    ForwardSourceRefusal::Artifact {
        code: error.code(),
        text: format!("{detail}: the mechanism composition refused the action's query"),
    }
}

impl PreparedMechanismComposition {
    /// Replay one query per action through the composition artifact and adapt
    /// the results as a licensed forward source of an inverse query.
    ///
    /// Each action's query is sealed and consumed again before its result is
    /// read, so the numbers handed to the inverse engine are the replayed ones
    /// and carry the verified premises digest. Mechanisms the analyst declared
    /// rather than estimated are carried into the origin as assumed.
    ///
    /// A composition that admitted mechanisms fitted in another population or
    /// regime is a carried (transported) claim: it needs the `transport`
    /// receipt of that carrying and is read only under `AcceptsTransported`; the
    /// origin then carries the receipt and its support label, never `supported`.
    ///
    /// # Errors
    /// A composition refusal of any query (wrapped with its registered code), and
    /// every refusal of [`composed_forward_source`]: no action, a repeated action
    /// or coordinate, a query of another family than the composition, a mean
    /// addressed by an outcome-law coordinate, a finite outcome whose levels are
    /// not coded exactly once.
    pub fn inverse_forward_source(
        &self,
        actions: &[ComposedActionQuery],
        transport: Option<TransportReceipt>,
        accept: TransportedInputPolicy,
        ctx: &ExecutionContext,
    ) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
        if transport.is_none() && !self.admitted_variables().is_empty() {
            return Err(ForwardSourceRefusal::ComposedInvalid(
                "a composition with mechanisms admitted from another context carries the \
                 transport receipt of that carrying",
            ));
        }
        let family = if self.family() == MechanismFamily::LinearGaussian {
            ComposedFamily::LinearGaussian
        } else {
            ComposedFamily::Tabular
        };
        let mut responses = Vec::with_capacity(actions.len());
        for action in actions {
            let export =
                self.export(action.query.clone(), ctx).map_err(|e| composition_refusal(&e))?;
            let consumed = consume_mechanism_composition_artifact(
                &export.bytes,
                Some(&export.premises_digest),
                ctx,
            )
            .map_err(|e| composition_refusal(&e))?;
            responses.push(ComposedActionResponse {
                action_id: action.action_id.clone(),
                coordinate: action.coordinate.clone(),
                premises_digest: consumed.premises_digest,
                query: action.query.clone(),
                result: consumed.result,
                level_values: action.level_values.clone(),
            });
        }
        let response = ComposedResponse {
            family,
            assumed_mechanisms: self.declared_mechanisms().into_iter().map(str::to_owned).collect(),
            transport,
            actions: responses,
        };
        composed_forward_source_accepting(&response, accept)
    }
}

/// A cohort-mean contrast as a forward source.
///
/// # Errors
/// [`ForwardSourceRefusal::TransportedNotAccepted`] for a cohort under a
/// declared covariate shift offered under `NativeTargetOnly`, and every refusal
/// of [`transported_source`] for one accepted.
pub fn cohort_source(
    source: &CohortDecisionSource,
    accept: TransportedInputPolicy,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    cohort_forward_source(source, accept)
}

/// A transported response as a forward source.
///
/// # Errors
/// [`ForwardSourceRefusal::TransportedNotAccepted`] under `NativeTargetOnly`.
pub fn transported_source(
    response: &TransportedResponse,
    accept: TransportedInputPolicy,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    transported_forward_source(response, accept)
}

/// A composed-mechanism response (already replayed) as a forward source.
///
/// # Errors
/// As [`composed_forward_source`].
pub fn composed_source(
    response: &ComposedResponse,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    composed_forward_source(response)
}

/// Evaluate an inverse query on a licensed source; the result carries the
/// source's origin.
///
/// # Errors
/// An interval region (refused for every source), a quantile, probability,
/// nonlinear utility or outcome-law input on a point, and any engine refusal.
pub fn evaluate(
    query: &InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<SourcedInverseResult, ForwardSourceRefusal> {
    evaluate_inverse_on_source(query, source, demand)
}

/// Evaluate and seal a query on a source with one claim for every action,
/// binding the origin into the artifact.
///
/// # Errors
/// [`ForwardSourceRefusal::PerActionNotSealable`] for a source that answers each
/// action separately (use [`seal_per_action`]), and every refusal of [`evaluate`].
pub fn seal(
    query: InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<InverseQueryArtifact, ForwardSourceRefusal> {
    seal_on_source(query, source, demand)
}

/// Evaluate and seal one single-action artifact per answered action.
///
/// # Errors
/// As [`seal`], for a source that answers each action separately.
pub fn seal_per_action(
    query: &InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<Vec<(String, InverseQueryArtifact)>, ForwardSourceRefusal> {
    seal_per_action_on_source(query, source, demand)
}

/// What an inspection of a forward source shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardSourceInspection {
    /// `composed_mechanism`, `cohort_contrast` or `transported_response`.
    pub kind: &'static str,
    /// `point_only` or `law`.
    pub representation: &'static str,
    /// Support label of the claim.
    pub support: String,
    /// Always `not_licensed`: no source supplies an interval region.
    pub interval_region: &'static str,
    /// Whether the claim was carried from another population.
    pub transported: bool,
    /// Populations the claim was learned in, for a transported claim.
    pub source_populations: Vec<String>,
    /// Population the claim was carried to, for a transported claim.
    pub target_population: Option<String>,
    /// Mechanisms the analyst wrote down rather than estimated.
    pub assumed_mechanisms: Vec<String>,
    /// Digest of the claim handed to the engine.
    pub claim_digest: String,
}

/// Plain-data summary of a source's origin.
#[must_use]
pub fn inspect_origin(origin: &ForwardOrigin) -> ForwardSourceInspection {
    ForwardSourceInspection {
        kind: origin.kind.as_str(),
        representation: origin.representation.as_str(),
        support: origin.support.clone(),
        interval_region: "not_licensed",
        transported: origin.transport.is_some(),
        source_populations: origin
            .transport
            .as_ref()
            .map(|t| t.source_populations.clone())
            .unwrap_or_default(),
        target_population: origin.transport.as_ref().map(|t| t.target_population.clone()),
        assumed_mechanisms: origin.assumed_mechanisms.clone(),
        claim_digest: origin.claim_digest.clone(),
    }
}

/// Answer an observational-scenario question on an aligned observational law.
///
/// The result is associational: it ranks no action and is not an intervention
/// effect.
///
/// # Errors
/// An empty, repeated or oversized scenario list, a non-finite threshold, a
/// scenario that conditions on the outcome, a missing or masked coordinate,
/// non-joint alignment, a law that is not observational, a non-finite draw and
/// an event with no mass.
pub fn evaluate_observational(
    query: &ObservationalScenarioQuery,
    law: &DistributionArtifact,
) -> Result<ObservationalScenarioResult, ObservationalScenarioError> {
    evaluate_observational_scenarios(query, law)
}

/// Re-derive a stored observational-scenario result from its law and query.
///
/// # Errors
/// Every refusal of [`evaluate_observational`] and
/// [`ObservationalScenarioError::ResultMismatch`] when any stored value differs.
pub fn verify_observational(
    result: &ObservationalScenarioResult,
    query: &ObservationalScenarioQuery,
    law: &DistributionArtifact,
) -> Result<(), ObservationalScenarioError> {
    verify_observational_scenarios(result, query, law)
}

/// What an inspection of an observational-scenario result shows.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalScenarioInspection {
    /// Always `associational`.
    pub standing: &'static str,
    /// Always `false`.
    pub ranks_actions: bool,
    /// Always `false`.
    pub is_intervention_effect: bool,
    /// `P(E)`.
    pub event_mass: f64,
    /// Mass of the event no declared scenario explains.
    pub uncovered_event_mass: f64,
    /// Whether the arithmetic is exact (no Monte-Carlo error is claimed
    /// otherwise).
    pub exact: bool,
    /// `(scenario, P(E | S), P(S | E))` in declaration order.
    pub scenarios: Vec<(String, Option<f64>, f64)>,
    /// The scope note the result carries.
    pub scope_note: &'static str,
}

/// Plain-data summary of an observational-scenario result.
#[must_use]
pub fn inspect_observational(
    result: &ObservationalScenarioResult,
) -> ObservationalScenarioInspection {
    ObservationalScenarioInspection {
        standing: "associational",
        ranks_actions: result.ranks_actions,
        is_intervention_effect: result.is_intervention_effect,
        event_mass: result.event_mass,
        uncovered_event_mass: result.uncovered_event_mass,
        exact: result.exact,
        scenarios: result
            .scenarios
            .iter()
            .map(|s| (s.id.clone(), s.event_given_scenario, s.scenario_given_event))
            .collect(),
        scope_note: result.scope_note,
    }
}
