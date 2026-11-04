//! Capability negotiation over every declared capability set, with the accepted subset derived by
//! hand from the negotiation rules.
//!
//! Rules (the record's theorem): an operation is accepted exactly when the declaration is valid,
//! the request names the declaration's own provider object, and the operation is among the
//! capabilities the declaration lists. Nothing is inferred from another operation or label. A
//! provider kind can declare only the operations its responsibility has: a law any law
//! operation, a posterior no CDF/quantile/log-probability, a signal its observation law plus an
//! update, an evidence factor only a factor, a deterministic utility only a utility evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    DistributionMeaning, EvidenceProviderContract, ExternalCapability as C,
    ExternalCapabilityRequest, ExternalContractError as E, ExternalEvidenceKind,
    ExternalPosteriorKind, ExternalScientificObject as Object, LawProviderContract,
    PosteriorProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
    SignalProviderContract, UtilityMonotonicity, UtilityProviderContract,
};

const ALL: [C; 12] = [
    C::Sample,
    C::Cdf,
    C::Quantile,
    C::LogProbability,
    C::Mean,
    C::Covariance,
    C::Conditional,
    C::Intervention,
    C::PosteriorPredictive,
    C::Update,
    C::Factor,
    C::EvaluateUtility,
];

fn identity() -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "obj".into(),
        version_id: "v1".into(),
        snapshot_id: "snap".into(),
        request_id: "req".into(),
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

fn law(capabilities: Vec<C>) -> Object {
    Object::Law(LawProviderContract {
        identity: identity(),
        quantities: vec![quantity(QuantityRole::Outcome)],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities,
    })
}

fn posterior(capabilities: Vec<C>) -> Object {
    Object::Posterior(PosteriorProviderContract {
        identity: identity(),
        kind: ExternalPosteriorKind::Functional,
        quantities: vec![quantity(QuantityRole::Outcome)],
        capabilities,
    })
}

fn signal(capabilities: Vec<C>) -> Object {
    Object::Signal(SignalProviderContract {
        identity: identity(),
        candidate_id: "study-1".into(),
        prior_id: "prior-1".into(),
        observation: quantity(QuantityRole::Outcome),
        capabilities,
    })
}

fn evidence(capabilities: Vec<C>) -> Object {
    Object::Evidence(EvidenceProviderContract {
        identity: identity(),
        factor_id: "proof:z".into(),
        kind: ExternalEvidenceKind::Exact,
        quantities: vec![quantity(QuantityRole::Covariate)],
        capabilities,
    })
}

fn utility(stochastic: bool, capabilities: Vec<C>) -> Object {
    Object::Utility(UtilityProviderContract {
        identity: identity(),
        inputs: vec![quantity(QuantityRole::Outcome)],
        output: quantity(QuantityRole::Utility),
        action_ids: vec!["a".into()],
        stochastic,
        bounds: Some((0.0, 1.0)),
        monotonicity: vec![UtilityMonotonicity::Nondecreasing],
        capabilities,
    })
}

fn ask(operation: C) -> ExternalCapabilityRequest {
    ExternalCapabilityRequest { expected_identity: identity(), operation }
}

/// A law that lists {Sample, Mean}: of the twelve operations exactly those two are accepted,
/// and each of the other ten refuses by naming itself, whatever else is on offer.
#[test]
fn a_two_operation_offer_accepts_exactly_those_two_of_twelve() {
    let offered = [C::Sample, C::Mean];
    let law = law(offered.to_vec());
    let mut accepted = Vec::new();
    let mut refused = Vec::new();
    for operation in ALL {
        match law.negotiate(&ask(operation)) {
            Ok(done) => {
                assert_eq!(done.operation(), operation);
                assert_eq!(done.identity(), &identity());
                accepted.push(operation);
            }
            Err(error) => {
                assert_eq!(error, E::MissingCapability(operation));
                refused.push(operation);
            }
        }
    }
    assert_eq!(accepted, vec![C::Sample, C::Mean]);
    assert_eq!(refused.len(), 10);
    // Declaring the CDF-adjacent operations does not give a CDF: Quantile and LogProbability
    // together still refuse Cdf, and Cdf together refuses Quantile.
    for (offer, absent) in [
        (vec![C::Quantile, C::LogProbability, C::Mean], C::Cdf),
        (vec![C::Cdf, C::Sample], C::Quantile),
        (vec![C::Covariance, C::Conditional, C::Intervention], C::Mean),
    ] {
        assert_eq!(law_with(offer).negotiate(&ask(absent)), Err(E::MissingCapability(absent)));
    }
}

fn law_with(capabilities: Vec<C>) -> Object {
    law(capabilities)
}

struct Kind {
    name: &'static str,
    make: fn(Vec<C>) -> Object,
    minimum: Vec<C>,
    allowed: Vec<C>,
}

/// Every provider kind with its minimum declaration and the operations it may add.
fn provider_kinds() -> Vec<Kind> {
    Vec::from([
        Kind {
            name: "law",
            make: law,
            minimum: vec![C::Mean],
            allowed: vec![
                C::Sample,
                C::Cdf,
                C::Quantile,
                C::LogProbability,
                C::Mean,
                C::Covariance,
                C::Conditional,
                C::Intervention,
                C::PosteriorPredictive,
            ],
        },
        Kind {
            name: "posterior",
            make: posterior,
            minimum: vec![C::Mean],
            allowed: vec![
                C::Sample,
                C::Mean,
                C::Covariance,
                C::Conditional,
                C::PosteriorPredictive,
            ],
        },
        Kind {
            name: "signal",
            make: signal,
            minimum: vec![C::Factor, C::Update],
            allowed: vec![C::Sample, C::Cdf, C::Quantile, C::LogProbability, C::Factor, C::Update],
        },
        Kind {
            name: "evidence",
            make: evidence,
            minimum: vec![C::Factor],
            allowed: vec![C::Factor],
        },
        Kind {
            name: "deterministic utility",
            make: |capabilities| utility(false, capabilities),
            minimum: vec![C::EvaluateUtility],
            allowed: vec![C::EvaluateUtility],
        },
        Kind {
            name: "stochastic utility",
            make: |capabilities| utility(true, capabilities),
            minimum: vec![C::EvaluateUtility],
            allowed: vec![C::EvaluateUtility, C::Sample],
        },
    ])
}

/// Why each kind has the operations it has, from its documented responsibility rather than from
/// the checker:
///
/// * a law answers questions about an outcome distribution: sampling, CDF, quantile, log
///   probability, mean, covariance, conditioning, interventions and posterior-predictive; it
///   cannot update a posterior, supply a proof factor or evaluate a utility;
/// * a posterior object supplies draws and moments of parameters or functionals (sample, mean,
///   covariance, conditioning, predictive integration) and, per the record theorem, never a
///   CDF, quantile, log probability or intervention, which would let it pose as an outcome law;
/// * a signal is an observation law plus a coherent update, so besides its update and factor it
///   may sample or give CDF, quantile and log probability of the possible observation, but it
///   has no mean, covariance, conditioning, intervention, predictive or utility operation;
/// * an evidence provider supplies exactly one thing, the proof factor;
/// * a utility provider evaluates a utility, and only a stochastic one can also be sampled.
///
/// For every provider kind and every extra operation added to the kind's minimum declaration,
/// the declaration is valid exactly when that kind's responsibility includes the operation; in a
/// valid declaration the accepted set is exactly the declared set.
#[test]
fn each_provider_kind_can_declare_only_its_own_operations() {
    let kinds = provider_kinds();
    for kind in kinds {
        // The minimum declaration negotiates every operation it lists and nothing else.
        let base = (kind.make)(kind.minimum.clone());
        for operation in ALL {
            let expected = if kind.minimum.contains(&operation) {
                Ok(operation)
            } else {
                Err(E::MissingCapability(operation))
            };
            assert_eq!(
                base.negotiate(&ask(operation)).map(|done| done.operation()),
                expected,
                "{} minimum / {operation:?}",
                kind.name
            );
        }
        for extra in ALL.into_iter().filter(|op| !kind.minimum.contains(op)) {
            let mut declared = kind.minimum.clone();
            declared.push(extra);
            let object = (kind.make)(declared.clone());
            if kind.allowed.contains(&extra) {
                for operation in ALL {
                    let expected = if declared.contains(&operation) {
                        Ok(operation)
                    } else {
                        Err(E::MissingCapability(operation))
                    };
                    assert_eq!(
                        object.negotiate(&ask(operation)).map(|done| done.operation()),
                        expected,
                        "{} + {extra:?} / {operation:?}",
                        kind.name
                    );
                }
            } else {
                // Not a responsibility of this kind: the declaration itself is refused, so even the
                // operations it lists negotiate nothing.
                assert_eq!(object.validate(), Err(E::InvalidDeclaration), "{}", kind.name);
                assert_eq!(
                    object.negotiate(&ask(kind.minimum[0])).map(|done| done.operation()),
                    Err(E::InvalidDeclaration),
                    "{} + {extra:?}",
                    kind.name
                );
            }
        }
    }
}

/// The negotiation order is declaration, then identity, then operation: an invalid declaration
/// beats a wrong identity, and a wrong identity beats a missing operation. A request for an
/// operation that is declared still refuses when any one identity field differs.
#[test]
fn declaration_then_identity_then_operation_decide_in_that_order() {
    let mut other = identity();
    other.request_id = "other".into();
    let wrong_identity_missing_op =
        ExternalCapabilityRequest { expected_identity: other.clone(), operation: C::Cdf };
    let valid = law(vec![C::Sample, C::Mean]);
    assert_eq!(valid.negotiate(&wrong_identity_missing_op), Err(E::RequestMismatch));
    let invalid = law(vec![C::Update, C::Mean]);
    assert_eq!(invalid.negotiate(&wrong_identity_missing_op), Err(E::InvalidDeclaration));
    let mut duplicate = law(vec![C::Mean, C::Mean]);
    assert_eq!(duplicate.negotiate(&ask(C::Mean)), Err(E::DuplicateDeclaration));
    if let Object::Law(value) = &mut duplicate {
        value.capabilities = vec![];
    }
    assert_eq!(duplicate.negotiate(&ask(C::Mean)), Err(E::InvalidIdentity));

    for change in [
        |i: &mut ProviderObjectIdentity| i.provider_id = "x".into(),
        |i: &mut ProviderObjectIdentity| i.object_id = "x".into(),
        |i: &mut ProviderObjectIdentity| i.version_id = "x".into(),
        |i: &mut ProviderObjectIdentity| i.snapshot_id = "x".into(),
        |i: &mut ProviderObjectIdentity| i.request_id = "x".into(),
    ] {
        let mut asked = ask(C::Mean);
        change(&mut asked.expected_identity);
        assert_eq!(valid.negotiate(&asked), Err(E::RequestMismatch));
    }
}
