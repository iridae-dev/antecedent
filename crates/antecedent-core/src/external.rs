//! Typed declarations for scientific objects supplied by another system.
//!
//! These contracts describe what an object can supply. They do not verify
//! numerical truth, identify a causal estimand, or grant provider trust.

use std::collections::HashSet;

use crate::{DistributionMeaning, QuantityRole, ScientificQuantity};

/// Stable identity for one provider object and exact scientific request.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProviderObjectIdentity {
    /// Provider service or model identity.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider implementation or model version.
    pub version_id: String,
    /// Input data or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint; a different request needs a new check.
    pub request_id: String,
}

/// Operations a provider may declare independently.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalCapability {
    /// Joint or scalar sampling.
    Sample,
    /// Cumulative distribution function at a named threshold.
    Cdf,
    /// Quantile under an explicit convention.
    Quantile,
    /// Log probability or log density.
    LogProbability,
    /// Mean.
    Mean,
    /// Covariance of a declared joint law.
    Covariance,
    /// Conditional law under named conditions.
    Conditional,
    /// Interventional law under a named regime.
    Intervention,
    /// Predictive law integrated over posterior uncertainty.
    PosteriorPredictive,
    /// Posterior update after a possible observation.
    Update,
    /// Exact or empirical factor needed by a causal proof.
    Factor,
    /// Utility for a named action and input tuple.
    EvaluateUtility,
}

/// Which posterior object a provider supplies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalPosteriorKind {
    /// Posterior over model parameters.
    Parameter,
    /// Posterior over a statistical functional.
    Functional,
    /// Posterior over a causally identified functional.
    CausalFunctional,
    /// Posterior over whole structural models or graph atoms.
    StructuralModel,
}

/// Origin of a supplied evidence factor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalEvidenceKind {
    /// Algebraically exact law or factor supplied with a finite truth receipt.
    Exact,
    /// Empirical factor estimated from a named data snapshot.
    Empirical,
}

/// Declared monotonicity of a utility in one input coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtilityMonotonicity {
    /// No monotonicity claim.
    Unspecified,
    /// Nondecreasing in the named input.
    Nondecreasing,
    /// Nonincreasing in the named input.
    Nonincreasing,
}

/// A capability or semantic declaration cannot satisfy the requested object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExternalContractError {
    /// Required stable identity or finite structure is absent.
    InvalidIdentity,
    /// A quantity, meaning, factor or utility declaration is incompatible.
    InvalidDeclaration,
    /// Duplicate quantities, actions or capability declarations are ambiguous.
    DuplicateDeclaration,
    /// The exact requested operation is absent.
    MissingCapability(ExternalCapability),
    /// The request names a different object, version, snapshot or fingerprint.
    RequestMismatch,
}

/// One exact operation requested from one provider object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalCapabilityRequest {
    /// Independently retained object and request identity.
    pub expected_identity: ProviderObjectIdentity,
    /// Operation the consumer actually needs.
    pub operation: ExternalCapability,
}

/// A direct operation checked before a callback is invoked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegotiatedExternalOperation {
    identity: ProviderObjectIdentity,
    operation: ExternalCapability,
}

impl NegotiatedExternalOperation {
    /// Exact provider object for which this operation was checked.
    #[must_use]
    pub fn identity(&self) -> &ProviderObjectIdentity {
        &self.identity
    }

    /// Operation which was actually declared.
    #[must_use]
    pub const fn operation(&self) -> ExternalCapability {
        self.operation
    }

    /// Invoke an external callback only after direct capability negotiation.
    ///
    /// This carries no truth, error or trust certification for the returned
    /// value; object-level verification and result binding remain separate.
    pub fn invoke<T>(&self, callback: impl FnOnce() -> T) -> T {
        callback()
    }
}

/// A declared law, including its randomness meaning and available operations.
#[derive(Clone, Debug, PartialEq)]
pub struct LawProviderContract {
    /// Exact provider object identity.
    pub identity: ProviderObjectIdentity,
    /// Ordered law coordinates; order is meaningful for joint samples.
    pub quantities: Vec<ScientificQuantity>,
    /// What one draw represents.
    pub meaning: DistributionMeaning,
    /// Independently declared supported operations.
    pub capabilities: Vec<ExternalCapability>,
}

/// A parameter, functional, causal-functional or structural-model posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorProviderContract {
    /// Exact provider object identity.
    pub identity: ProviderObjectIdentity,
    /// Posterior family.
    pub kind: ExternalPosteriorKind,
    /// Ordered parameter, functional or model coordinates.
    pub quantities: Vec<ScientificQuantity>,
    /// Independently declared supported operations.
    pub capabilities: Vec<ExternalCapability>,
}

/// A candidate-specific predictive observation and coherent update operation.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalProviderContract {
    /// Exact provider object identity.
    pub identity: ProviderObjectIdentity,
    /// Stable proposed-study candidate identity.
    pub candidate_id: String,
    /// Prior or current-state identity consumed by the update.
    pub prior_id: String,
    /// Possible observation returned by the signal law.
    pub observation: ScientificQuantity,
    /// Independently declared supported operations.
    pub capabilities: Vec<ExternalCapability>,
}

/// A factor required by a checked identification or transport proof.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceProviderContract {
    /// Exact provider object identity.
    pub identity: ProviderObjectIdentity,
    /// Exact factor slot in the proof, distinct from a display name.
    pub factor_id: String,
    /// Whether the value is exact or empirical.
    pub kind: ExternalEvidenceKind,
    /// Ordered factor coordinates.
    pub quantities: Vec<ScientificQuantity>,
    /// Independently declared supported operations.
    pub capabilities: Vec<ExternalCapability>,
}

/// An action-scoped utility with declared inputs, units and bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct UtilityProviderContract {
    /// Exact provider object identity.
    pub identity: ProviderObjectIdentity,
    /// Ordered required input quantities.
    pub inputs: Vec<ScientificQuantity>,
    /// Scalar utility output, including its unit.
    pub output: ScientificQuantity,
    /// Stable semantic action IDs, not labels.
    pub action_ids: Vec<String>,
    /// Whether repeated evaluation at fixed inputs is stochastic.
    pub stochastic: bool,
    /// Optional finite lower/upper declared bounds.
    pub bounds: Option<(f64, f64)>,
    /// Monotonicity for each input, in input order.
    pub monotonicity: Vec<UtilityMonotonicity>,
    /// Independently declared supported operations.
    pub capabilities: Vec<ExternalCapability>,
}

/// One of the five distinct external scientific responsibilities.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalScientificObject {
    /// A declared probability law.
    Law(LawProviderContract),
    /// A declared posterior.
    Posterior(PosteriorProviderContract),
    /// A candidate signal and update.
    Signal(SignalProviderContract),
    /// A proof evidence factor.
    Evidence(EvidenceProviderContract),
    /// An action utility.
    Utility(UtilityProviderContract),
}

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty()
}

fn allowed_capability(object: &ExternalScientificObject, capability: ExternalCapability) -> bool {
    match object {
        ExternalScientificObject::Law(_) => matches!(
            capability,
            ExternalCapability::Sample
                | ExternalCapability::Cdf
                | ExternalCapability::Quantile
                | ExternalCapability::LogProbability
                | ExternalCapability::Mean
                | ExternalCapability::Covariance
                | ExternalCapability::Conditional
                | ExternalCapability::Intervention
                | ExternalCapability::PosteriorPredictive
        ),
        ExternalScientificObject::Posterior(_) => matches!(
            capability,
            ExternalCapability::Sample
                | ExternalCapability::Mean
                | ExternalCapability::Covariance
                | ExternalCapability::Conditional
                | ExternalCapability::PosteriorPredictive
        ),
        ExternalScientificObject::Signal(_) => matches!(
            capability,
            ExternalCapability::Sample
                | ExternalCapability::Cdf
                | ExternalCapability::Quantile
                | ExternalCapability::LogProbability
                | ExternalCapability::Factor
                | ExternalCapability::Update
        ),
        ExternalScientificObject::Evidence(_) => capability == ExternalCapability::Factor,
        ExternalScientificObject::Utility(value) => {
            capability == ExternalCapability::EvaluateUtility
                || value.stochastic && capability == ExternalCapability::Sample
        }
    }
}

fn validate_common(
    identity: &ProviderObjectIdentity,
    quantities: &[ScientificQuantity],
    capabilities: &[ExternalCapability],
) -> Result<(), ExternalContractError> {
    if ![
        &identity.provider_id,
        &identity.object_id,
        &identity.version_id,
        &identity.snapshot_id,
        &identity.request_id,
    ]
    .into_iter()
    .all(|value| nonblank(value))
        || quantities.is_empty()
        || quantities.len() > 1_024
        || capabilities.is_empty()
    {
        return Err(ExternalContractError::InvalidIdentity);
    }
    if quantities.iter().any(|quantity| quantity.validate().is_err()) {
        return Err(ExternalContractError::InvalidDeclaration);
    }
    let mut seen = HashSet::new();
    if quantities.iter().any(|quantity| {
        !seen.insert((
            &quantity.variable_id,
            &quantity.population_id,
            &quantity.regime_id,
            quantity.horizon,
            &quantity.functional_id,
            &quantity.conditioning,
            &quantity.transform_id,
        ))
    }) {
        return Err(ExternalContractError::DuplicateDeclaration);
    }
    let mut ops = HashSet::new();
    if capabilities.iter().any(|operation| !ops.insert(operation)) {
        return Err(ExternalContractError::DuplicateDeclaration);
    }
    Ok(())
}

impl ExternalScientificObject {
    /// Exact provider object identity carried by this declaration.
    #[must_use]
    pub fn identity(&self) -> &ProviderObjectIdentity {
        match self {
            Self::Law(value) => &value.identity,
            Self::Posterior(value) => &value.identity,
            Self::Signal(value) => &value.identity,
            Self::Evidence(value) => &value.identity,
            Self::Utility(value) => &value.identity,
        }
    }

    /// Check common identities and responsibility-specific declarations.
    ///
    /// # Errors
    /// Empty, duplicate, mismatched or ill-formed declarations refuse.
    pub fn validate(&self) -> Result<(), ExternalContractError> {
        match self {
            Self::Law(contract) => {
                validate_common(&contract.identity, &contract.quantities, &contract.capabilities)?;
                if contract.meaning == DistributionMeaning::ParameterPosterior
                    || contract.meaning == DistributionMeaning::CausalFunctionalPosterior
                {
                    return Err(ExternalContractError::InvalidDeclaration);
                }
            }
            Self::Posterior(contract) => {
                validate_common(&contract.identity, &contract.quantities, &contract.capabilities)?;
            }
            Self::Signal(contract) => {
                validate_common(
                    &contract.identity,
                    std::slice::from_ref(&contract.observation),
                    &contract.capabilities,
                )?;
                if !nonblank(&contract.candidate_id) || !nonblank(&contract.prior_id) {
                    return Err(ExternalContractError::InvalidIdentity);
                }
                if !contract.capabilities.contains(&ExternalCapability::Update)
                    || !(contract.capabilities.contains(&ExternalCapability::Sample)
                        || contract.capabilities.contains(&ExternalCapability::Factor))
                {
                    return Err(ExternalContractError::InvalidDeclaration);
                }
            }
            Self::Evidence(contract) => {
                validate_common(&contract.identity, &contract.quantities, &contract.capabilities)?;
                if !nonblank(&contract.factor_id)
                    || !contract.capabilities.contains(&ExternalCapability::Factor)
                {
                    return Err(ExternalContractError::InvalidDeclaration);
                }
            }
            Self::Utility(contract) => {
                validate_common(&contract.identity, &contract.inputs, &contract.capabilities)?;
                contract
                    .output
                    .validate()
                    .map_err(|_| ExternalContractError::InvalidDeclaration)?;
                if contract.output.role != QuantityRole::Utility
                    || contract.action_ids.is_empty()
                    || contract.monotonicity.len() != contract.inputs.len()
                    || !contract.capabilities.contains(&ExternalCapability::EvaluateUtility)
                {
                    return Err(ExternalContractError::InvalidDeclaration);
                }
                let mut actions = HashSet::new();
                if contract.action_ids.iter().any(|id| !nonblank(id) || !actions.insert(id)) {
                    return Err(ExternalContractError::DuplicateDeclaration);
                }
                if contract
                    .bounds
                    .is_some_and(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
                {
                    return Err(ExternalContractError::InvalidDeclaration);
                }
            }
        }
        let capabilities = self.capabilities();
        if capabilities.iter().any(|operation| !allowed_capability(self, *operation)) {
            return Err(ExternalContractError::InvalidDeclaration);
        }
        Ok(())
    }

    /// Operations declared by this exact provider object.
    #[must_use]
    pub fn capabilities(&self) -> &[ExternalCapability] {
        match self {
            Self::Law(value) => &value.capabilities,
            Self::Posterior(value) => &value.capabilities,
            Self::Signal(value) => &value.capabilities,
            Self::Evidence(value) => &value.capabilities,
            Self::Utility(value) => &value.capabilities,
        }
    }

    /// Refuse an operation absent from this exact object declaration.
    ///
    /// # Errors
    /// Invalid declarations and missing exact operations refuse.
    pub fn require_capability(
        &self,
        required: ExternalCapability,
    ) -> Result<(), ExternalContractError> {
        self.validate()?;
        if !self.capabilities().contains(&required) {
            return Err(ExternalContractError::MissingCapability(required));
        }
        Ok(())
    }

    /// Negotiate one direct operation at an independently supplied request ID.
    ///
    /// `Sample` never substitutes for `Cdf`, `Quantile`, or another operation.
    /// A future sampling approximation requires a separate licensed method
    /// and numerical-error/replicate receipt; this direct path grants none.
    ///
    /// # Errors
    /// Invalid declarations, an identity mismatch or a missing operation.
    pub fn negotiate(
        &self,
        request: &ExternalCapabilityRequest,
    ) -> Result<NegotiatedExternalOperation, ExternalContractError> {
        self.validate()?;
        if self.identity() != &request.expected_identity {
            return Err(ExternalContractError::RequestMismatch);
        }
        self.require_capability(request.operation)?;
        Ok(NegotiatedExternalOperation {
            identity: request.expected_identity.clone(),
            operation: request.operation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> ProviderObjectIdentity {
        ProviderObjectIdentity {
            provider_id: "outside-model".into(),
            object_id: "law-1".into(),
            version_id: "v1".into(),
            snapshot_id: "source-snapshot".into(),
            request_id: "exact-request-1".into(),
        }
    }

    fn quantity(role: QuantityRole) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:y".into(),
            variable_name: "Y".into(),
            role,
            units: "count".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    #[test]
    fn sample_and_mean_do_not_implicitly_supply_a_cdf() {
        let law = ExternalScientificObject::Law(LawProviderContract {
            identity: identity(),
            quantities: vec![quantity(QuantityRole::Outcome)],
            meaning: DistributionMeaning::InterventionalPredictive,
            capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
        });
        assert_eq!(law.require_capability(ExternalCapability::Mean), Ok(()));
        assert_eq!(
            law.require_capability(ExternalCapability::Cdf),
            Err(ExternalContractError::MissingCapability(ExternalCapability::Cdf))
        );
        let mut changed = law.clone();
        if let ExternalScientificObject::Law(value) = &mut changed {
            value.meaning = DistributionMeaning::ParameterPosterior;
        }
        assert_eq!(changed.validate(), Err(ExternalContractError::InvalidDeclaration));

        let request = ExternalCapabilityRequest {
            expected_identity: identity(),
            operation: ExternalCapability::Cdf,
        };
        let called = std::cell::Cell::new(false);
        assert_eq!(
            law.negotiate(&request),
            Err(ExternalContractError::MissingCapability(ExternalCapability::Cdf))
        );
        assert!(!called.get());
        let mut wrong = request;
        wrong.operation = ExternalCapability::Mean;
        wrong.expected_identity.snapshot_id = "other-snapshot".into();
        assert_eq!(law.negotiate(&wrong), Err(ExternalContractError::RequestMismatch));
        wrong.expected_identity = identity();
        let negotiated = law.negotiate(&wrong).unwrap();
        assert_eq!(negotiated.operation(), ExternalCapability::Mean);
        assert_eq!(
            negotiated.invoke(|| {
                called.set(true);
                0.25
            }),
            0.25
        );
        assert!(called.get());
    }

    #[test]
    fn signal_needs_observation_law_and_update_for_same_request() {
        let signal = ExternalScientificObject::Signal(SignalProviderContract {
            identity: identity(),
            candidate_id: "study-1".into(),
            prior_id: "prior-1".into(),
            observation: quantity(QuantityRole::Outcome),
            capabilities: vec![ExternalCapability::Factor, ExternalCapability::Update],
        });
        assert_eq!(signal.validate(), Ok(()));
        let mut missing = signal.clone();
        if let ExternalScientificObject::Signal(value) = &mut missing {
            value.capabilities.retain(|item| *item != ExternalCapability::Update);
        }
        assert_eq!(missing.validate(), Err(ExternalContractError::InvalidDeclaration));
    }

    #[test]
    fn evidence_and_utility_keep_distinct_responsibilities() {
        let evidence = ExternalScientificObject::Evidence(EvidenceProviderContract {
            identity: identity(),
            factor_id: "proof:target-z".into(),
            kind: ExternalEvidenceKind::Empirical,
            quantities: vec![quantity(QuantityRole::Covariate)],
            capabilities: vec![ExternalCapability::Factor],
        });
        assert_eq!(evidence.validate(), Ok(()));
        assert_eq!(
            evidence.require_capability(ExternalCapability::EvaluateUtility),
            Err(ExternalContractError::MissingCapability(ExternalCapability::EvaluateUtility))
        );
        let utility = ExternalScientificObject::Utility(UtilityProviderContract {
            identity: identity(),
            inputs: vec![quantity(QuantityRole::Outcome)],
            output: quantity(QuantityRole::Utility),
            action_ids: vec!["do(a=0)".into(), "do(a=1)".into()],
            stochastic: false,
            bounds: Some((0.0, 1.0)),
            monotonicity: vec![UtilityMonotonicity::Nondecreasing],
            capabilities: vec![ExternalCapability::EvaluateUtility],
        });
        assert_eq!(utility.validate(), Ok(()));
        let mut bad = utility;
        if let ExternalScientificObject::Utility(value) = &mut bad {
            value.action_ids[1] = value.action_ids[0].clone();
        }
        assert_eq!(bad.validate(), Err(ExternalContractError::DuplicateDeclaration));
    }
}
