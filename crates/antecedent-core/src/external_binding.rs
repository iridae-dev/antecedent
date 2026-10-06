//! Binding of externally supplied scientific results to a checked causal contract.
//!
//! The contract says what was identified; the external value says only what a
//! foreign system computed. Binding checks that they describe the same
//! quantity and never relabels foreign numerical work as native estimation.

use std::collections::HashSet;

use crate::{
    CompositionLink, CompositionStage, DistributionMeaning, ExternalContractError,
    ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState, IdentificationStatus,
    ProvenanceChain, ProvenanceChainError, ProviderObjectIdentity, QuantityMismatch,
    ScientificQuantity, SupportStatus,
};

/// Regime identity of an unintervened conditional law.
pub const OBSERVATIONAL_REGIME: &str = "observational";

/// A separately checked licence to read an observational law as interventional.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedEquivalence {
    /// Graph for which the equivalence was checked.
    pub graph_id: String,
    /// Interventional regime that the observational law may stand for.
    pub interventional_regime_id: String,
    /// Stable identity of the check (for example an adjustment-set proof).
    pub justification_id: String,
}

/// What an already identified causal contract requires of an external result.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedCausalContract {
    /// Identity of the graph the contract was identified on.
    pub graph_id: String,
    /// Structural identification outcome.
    pub identification: IdentificationStatus,
    /// Ordered requested coordinates (estimand, populations, horizons, ...).
    pub estimand: Vec<ScientificQuantity>,
    /// Distribution meanings the consumer may accept.
    pub accepted_meanings: Vec<DistributionMeaning>,
    /// Evidence factor IDs that the result must declare.
    pub required_evidence_ids: Vec<String>,
    /// Assumption IDs that the result must declare.
    pub required_assumption_ids: Vec<String>,
    /// Checked observational-to-interventional equivalences.
    pub equivalences: Vec<CheckedEquivalence>,
}

/// Declarations common to every external result.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalResultHeader {
    /// Provider object that produced the value.
    pub object: ExternalScientificObject,
    /// Graph identity the provider says it used.
    pub graph_id: String,
    /// Ordered coordinates, one per supplied value or marginal.
    pub quantities: Vec<ScientificQuantity>,
    /// Evidence factors the provider used.
    pub evidence_ids: Vec<String>,
    /// Assumptions the provider relied on.
    pub assumption_ids: Vec<String>,
    /// Trust held by the provider object.
    pub trust: ExternalTrustState,
}

/// What a provider means by reported uncertainty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExternalUncertaintyMeaning {
    /// No uncertainty claim.
    None,
    /// A provider-defined claim, never promoted to an Antecedent interval.
    ProviderDeclared {
        /// Stable method identity.
        method_id: String,
    },
}

/// A finite response grid, one value per coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalResponse {
    /// Shared declarations.
    pub header: ExternalResultHeader,
    /// Values in coordinate order.
    pub values: Vec<f64>,
    /// Uncertainty meaning.
    pub uncertainty: ExternalUncertaintyMeaning,
    /// Provider-declared support per coordinate, in coordinate order. Absent
    /// means support was not assessed, so every coordinate is
    /// `MissingEvidence`, never `Supported`.
    pub point_support: Option<Vec<SupportStatus>>,
}

/// A supplied outcome or functional law.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalDistribution {
    /// Shared declarations.
    pub header: ExternalResultHeader,
    /// What one draw means.
    pub meaning: DistributionMeaning,
}

/// A supplied posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalPosterior {
    /// Shared declarations.
    pub header: ExternalResultHeader,
    /// Posterior family; must equal the provider object's declared kind.
    pub kind: ExternalPosteriorKind,
}

/// The typed result presented for binding.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalResult {
    /// Finite response grid.
    Response(ExternalResponse),
    /// Law of an outcome or functional.
    Distribution(ExternalDistribution),
    /// Posterior.
    Posterior(ExternalPosterior),
}

/// Why binding refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExternalBindingError {
    /// The contract is not point or restriction identified.
    ContractNotIdentified(IdentificationStatus),
    /// Contract has no coordinates or blank identities.
    InvalidContract,
    /// The provider declaration is invalid.
    Object(ExternalContractError),
    /// Object kind cannot supply this result type.
    ObjectKindMismatch,
    /// Result graph differs from the contract graph.
    GraphMismatch,
    /// The coordinate count differs from the contract or the value count.
    DimensionMismatch,
    /// The coordinate at this index differs in the given dimension.
    CoordinateMismatch(usize, QuantityMismatch),
    /// An observational law was offered for an interventional request with no
    /// matching checked equivalence.
    UncheckedObservationalLaw(usize),
    /// A supplied value is not finite.
    NonFiniteValue(usize),
    /// Distribution meaning is not accepted by the contract.
    MeaningNotAccepted,
    /// Posterior kind differs from the provider declaration.
    PosteriorKindMismatch,
    /// A required evidence factor is not declared.
    MissingEvidence(String),
    /// A required assumption is not declared.
    MissingAssumption(String),
    /// Trust is native (impossible for a provider) or verification covers another contract.
    TrustMismatch,
}

/// Trust level retained on a bound external claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundTrustLevel {
    /// Supplier assertion only.
    ExternallyAttested,
    /// The exact provider contract passed object-level verification.
    ExactRequestVerified,
}

/// An external result bound to an identified contract.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundExternalClaim {
    graph_id: String,
    identification: IdentificationStatus,
    quantities: Vec<ScientificQuantity>,
    values: Option<Vec<f64>>,
    point_status: Option<Vec<SupportStatus>>,
    meaning: Option<DistributionMeaning>,
    uncertainty: ExternalUncertaintyMeaning,
    execution: ProviderObjectIdentity,
    trust: BoundTrustLevel,
    evidence_ids: Vec<String>,
    assumption_ids: Vec<String>,
    equivalence_ids: Vec<String>,
}

impl BoundExternalClaim {
    /// Graph identity shared by contract and result.
    #[must_use]
    pub fn graph_id(&self) -> &str {
        &self.graph_id
    }
    /// Structural identification inherited from the contract.
    #[must_use]
    pub const fn identification(&self) -> IdentificationStatus {
        self.identification
    }
    /// Contract coordinates the result was checked against.
    #[must_use]
    pub fn quantities(&self) -> &[ScientificQuantity] {
        &self.quantities
    }
    /// Response values, when this is a response grid.
    #[must_use]
    pub fn values(&self) -> Option<&[f64]> {
        self.values.as_deref()
    }
    /// Per-coordinate support of a response grid; `MissingEvidence` where the
    /// provider declared none. Absent for laws and posteriors.
    #[must_use]
    pub fn point_status(&self) -> Option<&[SupportStatus]> {
        self.point_status.as_deref()
    }
    /// Worst per-coordinate support, matching `SupportReport::status`.
    #[must_use]
    pub fn support_status(&self) -> Option<SupportStatus> {
        self.point_status.as_ref()?.iter().copied().max_by_key(|status| status.severity())
    }
    /// Distribution meaning, when this is a law.
    #[must_use]
    pub const fn meaning(&self) -> Option<DistributionMeaning> {
        self.meaning
    }
    /// Provider-declared uncertainty; it grants no Antecedent interval.
    #[must_use]
    pub fn uncertainty(&self) -> &ExternalUncertaintyMeaning {
        &self.uncertainty
    }
    /// External execution identity (provider, object, version, snapshot, request).
    #[must_use]
    pub fn execution(&self) -> &ProviderObjectIdentity {
        &self.execution
    }
    /// Trust level of the provider.
    #[must_use]
    pub const fn trust(&self) -> BoundTrustLevel {
        self.trust
    }
    /// Evidence factors declared by the provider.
    #[must_use]
    pub fn evidence_ids(&self) -> &[String] {
        &self.evidence_ids
    }
    /// Assumptions declared by the provider.
    #[must_use]
    pub fn assumption_ids(&self) -> &[String] {
        &self.assumption_ids
    }
    /// Checked equivalences that licensed an observational law.
    #[must_use]
    pub fn equivalence_ids(&self) -> &[String] {
        &self.equivalence_ids
    }
    /// Always false: foreign numerical work is not native estimation.
    #[must_use]
    pub const fn is_native_estimation(&self) -> bool {
        false
    }
    /// Identity of the reported claim within [`Self::provenance_chain`].
    pub const CLAIM_LINK_ID: &'static str = "claim";

    /// Derivation chain behind the reported numbers: the causal contract, each
    /// evidence factor, each checked equivalence, the exact provider execution
    /// and the claim itself.
    ///
    /// # Errors
    /// A blank contract identity refuses.
    pub fn provenance_chain(
        &self,
        causal_contract_id: &str,
    ) -> Result<ProvenanceChain, ProvenanceChainError> {
        if blank(causal_contract_id) {
            return Err(ProvenanceChainError::InvalidLink);
        }
        let contract = format!("contract:{causal_contract_id}");
        let mut links = vec![CompositionLink {
            id: contract.clone(),
            stage: CompositionStage::CausalContract,
            parents: vec![],
        }];
        let mut claim_parents = vec![contract.clone()];
        let mut provider_parents = Vec::new();
        for id in &self.evidence_ids {
            let link = format!("evidence:{id}");
            provider_parents.push(link.clone());
            links.push(CompositionLink {
                id: link,
                stage: CompositionStage::Evidence,
                parents: vec![contract.clone()],
            });
        }
        for id in &self.equivalence_ids {
            let link = format!("equivalence:{id}");
            claim_parents.push(link.clone());
            links.push(CompositionLink {
                id: link,
                stage: CompositionStage::Transformation,
                parents: vec![contract.clone()],
            });
        }
        let provider = format!("provider:{}", self.provenance_label());
        claim_parents.push(provider.clone());
        links.push(CompositionLink {
            id: provider,
            stage: CompositionStage::ExternalProvider,
            parents: provider_parents,
        });
        links.push(CompositionLink {
            id: Self::CLAIM_LINK_ID.into(),
            stage: CompositionStage::Claim,
            parents: claim_parents,
        });
        ProvenanceChain::new(links)
    }

    /// Stable provenance label naming the external execution.
    #[must_use]
    pub fn provenance_label(&self) -> String {
        format!(
            "external:{}/{}@{}#{}",
            self.execution.provider_id,
            self.execution.object_id,
            self.execution.version_id,
            self.execution.snapshot_id
        )
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn declared(required: &[String], declared: &[String]) -> Option<String> {
    let have: HashSet<_> = declared.iter().collect();
    required.iter().find(|id| !have.contains(id)).cloned()
}

fn check_coordinates(
    contract: &CheckedCausalContract,
    offered: &[ScientificQuantity],
) -> Result<Vec<String>, ExternalBindingError> {
    use ExternalBindingError as E;
    let mut equivalence_ids = Vec::new();
    for (index, (got, wanted)) in offered.iter().zip(&contract.estimand).enumerate() {
        match got.require_same_coordinate(wanted) {
            Ok(()) => {}
            Err(QuantityMismatch::Regime) if got.regime_id == OBSERVATIONAL_REGIME => {
                let mut as_requested = got.clone();
                as_requested.regime_id.clone_from(&wanted.regime_id);
                as_requested
                    .require_same_coordinate(wanted)
                    .map_err(|m| E::CoordinateMismatch(index, m))?;
                let Some(eq) = contract.equivalences.iter().find(|e| {
                    e.graph_id == contract.graph_id
                        && e.interventional_regime_id == wanted.regime_id
                        && !blank(&e.justification_id)
                }) else {
                    return Err(E::UncheckedObservationalLaw(index));
                };
                if !equivalence_ids.contains(&eq.justification_id) {
                    equivalence_ids.push(eq.justification_id.clone());
                }
            }
            Err(m) => return Err(E::CoordinateMismatch(index, m)),
        }
    }
    Ok(equivalence_ids)
}

/// Check an external result against an identified contract and bind it.
///
/// # Errors
/// Refuses unidentified contracts, wrong provider kinds, graph, coordinate,
/// dimension, meaning, evidence, assumption or trust mismatches, non-finite
/// values, and observational laws offered for interventions without a
/// separately checked equivalence.
pub fn bind_external_result(
    contract: &CheckedCausalContract,
    result: &ExternalResult,
) -> Result<BoundExternalClaim, ExternalBindingError> {
    use ExternalBindingError as E;
    if !matches!(
        contract.identification,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    ) {
        return Err(E::ContractNotIdentified(contract.identification));
    }
    if blank(&contract.graph_id)
        || contract.estimand.is_empty()
        || contract.estimand.iter().any(|q| q.validate().is_err())
    {
        return Err(E::InvalidContract);
    }
    let (header, values, meaning, uncertainty) = match result {
        ExternalResult::Response(r) => (&r.header, Some(&r.values), None, r.uncertainty.clone()),
        ExternalResult::Distribution(d) => {
            (&d.header, None, Some(d.meaning), ExternalUncertaintyMeaning::None)
        }
        ExternalResult::Posterior(p) => (&p.header, None, None, ExternalUncertaintyMeaning::None),
    };
    header.object.validate().map_err(E::Object)?;
    match (result, &header.object) {
        (
            ExternalResult::Response(_) | ExternalResult::Distribution(_),
            ExternalScientificObject::Law(law),
        ) => {
            if meaning.is_some_and(|m| law.meaning != m) {
                return Err(E::MeaningNotAccepted);
            }
            // A response grid carries the meaning of the law behind it; an
            // observational predictive grid cannot answer an interventional request.
            if !contract.accepted_meanings.contains(&law.meaning) {
                return Err(E::MeaningNotAccepted);
            }
        }
        (ExternalResult::Posterior(p), ExternalScientificObject::Posterior(post)) => {
            if post.kind != p.kind {
                return Err(E::PosteriorKindMismatch);
            }
        }
        _ => return Err(E::ObjectKindMismatch),
    }
    if header.graph_id != contract.graph_id {
        return Err(E::GraphMismatch);
    }
    let declared_support = match result {
        ExternalResult::Response(r) => r.point_support.as_ref(),
        _ => None,
    };
    if header.quantities.len() != contract.estimand.len()
        || values.is_some_and(|v| v.len() != header.quantities.len())
        || declared_support.is_some_and(|p| p.len() != header.quantities.len())
    {
        return Err(E::DimensionMismatch);
    }
    if let Some(index) = values.and_then(|v| v.iter().position(|x| !x.is_finite())) {
        return Err(E::NonFiniteValue(index));
    }
    if meaning.is_some_and(|m| !contract.accepted_meanings.contains(&m)) {
        return Err(E::MeaningNotAccepted);
    }
    let equivalence_ids = check_coordinates(contract, &header.quantities)?;
    if let Some(id) = declared(&contract.required_evidence_ids, &header.evidence_ids) {
        return Err(E::MissingEvidence(id));
    }
    if let Some(id) = declared(&contract.required_assumption_ids, &header.assumption_ids) {
        return Err(E::MissingAssumption(id));
    }
    let trust = match &header.trust {
        ExternalTrustState::NativeLicensed => return Err(E::TrustMismatch),
        ExternalTrustState::ExternallyAttested { .. } => BoundTrustLevel::ExternallyAttested,
        state @ ExternalTrustState::ExactRequestVerified(_) => {
            if !state.verifies(&header.object) {
                return Err(E::TrustMismatch);
            }
            BoundTrustLevel::ExactRequestVerified
        }
    };
    Ok(BoundExternalClaim {
        graph_id: contract.graph_id.clone(),
        identification: contract.identification,
        quantities: contract.estimand.clone(),
        values: values.cloned(),
        point_status: values.map(|v| {
            declared_support
                .cloned()
                .unwrap_or_else(|| vec![SupportStatus::MissingEvidence; v.len()])
        }),
        meaning,
        uncertainty,
        execution: header.object.identity().clone(),
        trust,
        evidence_ids: header.evidence_ids.clone(),
        assumption_ids: header.assumption_ids.clone(),
        equivalence_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExternalCapability, LawProviderContract, PosteriorProviderContract, QuantityRole};

    fn quantity(regime: &str, horizon: u32) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:y".into(),
            variable_name: "Y".into(),
            role: QuantityRole::Outcome,
            units: "mmHg".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn identity() -> ProviderObjectIdentity {
        ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "curve".into(),
            version_id: "v3".into(),
            snapshot_id: "snap-9".into(),
            request_id: "req-1".into(),
        }
    }

    fn contract() -> CheckedCausalContract {
        CheckedCausalContract {
            graph_id: "graph-1".into(),
            identification: IdentificationStatus::NonparametricallyIdentified,
            estimand: vec![quantity("do(a=1)", 0), quantity("do(a=1)", 1)],
            accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
            required_evidence_ids: vec!["factor:z".into()],
            required_assumption_ids: vec!["ignorability".into()],
            equivalences: vec![],
        }
    }

    fn law(quantities: Vec<ScientificQuantity>) -> ExternalScientificObject {
        ExternalScientificObject::Law(LawProviderContract {
            identity: identity(),
            quantities,
            meaning: DistributionMeaning::InterventionalPredictive,
            capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
        })
    }

    fn response() -> ExternalResponse {
        let quantities = contract().estimand;
        ExternalResponse {
            header: ExternalResultHeader {
                object: law(quantities.clone()),
                graph_id: "graph-1".into(),
                quantities,
                evidence_ids: vec!["factor:z".into()],
                assumption_ids: vec!["ignorability".into()],
                trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
            },
            values: vec![1.5, 2.5],
            uncertainty: ExternalUncertaintyMeaning::ProviderDeclared { method_id: "m".into() },
            point_support: None,
        }
    }

    #[test]
    fn matching_response_binds_with_external_identity_and_is_not_native() {
        let claim =
            bind_external_result(&contract(), &ExternalResult::Response(response())).unwrap();
        assert!(!claim.is_native_estimation());
        assert_eq!(claim.trust(), BoundTrustLevel::ExternallyAttested);
        assert_eq!(claim.values(), Some(&[1.5, 2.5][..]));
        assert_eq!(claim.execution(), &identity());
        assert_eq!(claim.provenance_label(), "external:lab/curve@v3#snap-9");
        assert_eq!(claim.identification(), IdentificationStatus::NonparametricallyIdentified);
        // Support not assessed by the provider is missing evidence, not support.
        assert_eq!(claim.point_status(), Some(&[SupportStatus::MissingEvidence; 2][..]));
        assert_eq!(claim.support_status(), Some(SupportStatus::MissingEvidence));
    }

    #[test]
    fn bound_claim_answers_where_its_numbers_came_from() {
        use crate::CompositionStage as S;
        let mut c = contract();
        c.equivalences.push(CheckedEquivalence {
            graph_id: "graph-1".into(),
            interventional_regime_id: "do(a=1)".into(),
            justification_id: "backdoor:z".into(),
        });
        let mut r = response();
        for q in &mut r.header.quantities {
            q.regime_id = OBSERVATIONAL_REGIME.into();
        }
        let claim = bind_external_result(&c, &ExternalResult::Response(r)).unwrap();
        let chain = claim.provenance_chain("checked-contract").unwrap();
        let ids: Vec<_> = chain
            .lineage(BoundExternalClaim::CLAIM_LINK_ID)
            .unwrap()
            .into_iter()
            .map(|l| l.id.as_str())
            .collect();
        assert_eq!(
            ids,
            [
                "contract:checked-contract",
                "evidence:factor:z",
                "equivalence:backdoor:z",
                "provider:external:lab/curve@v3#snap-9",
                "claim"
            ]
        );
        assert_eq!(
            chain.require_stages(
                BoundExternalClaim::CLAIM_LINK_ID,
                &[S::CausalContract, S::Evidence, S::Transformation, S::ExternalProvider]
            ),
            Ok(())
        );
        assert_eq!(claim.provenance_chain(" "), Err(ProvenanceChainError::InvalidLink));
    }

    #[test]
    fn response_grid_inherits_its_law_meaning_check() {
        let mut r = response();
        if let ExternalScientificObject::Law(law) = &mut r.header.object {
            law.meaning = DistributionMeaning::PosteriorPredictive;
        }
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Response(r)),
            Err(ExternalBindingError::MeaningNotAccepted)
        );
    }

    #[test]
    fn declared_point_support_is_kept_per_coordinate() {
        let mut r = response();
        r.point_support =
            Some(vec![SupportStatus::Supported, SupportStatus::OutsideEmpiricalSupport]);
        let claim =
            bind_external_result(&contract(), &ExternalResult::Response(r.clone())).unwrap();
        assert_eq!(
            claim.point_status(),
            Some(&[SupportStatus::Supported, SupportStatus::OutsideEmpiricalSupport][..])
        );
        assert_eq!(claim.support_status(), Some(SupportStatus::OutsideEmpiricalSupport));
        r.point_support = Some(vec![SupportStatus::Supported]);
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Response(r)),
            Err(ExternalBindingError::DimensionMismatch)
        );
    }

    #[test]
    fn each_mismatch_refuses_before_a_claim_exists() {
        use ExternalBindingError as E;
        let bind = |c: &CheckedCausalContract, r: ExternalResponse| {
            bind_external_result(c, &ExternalResult::Response(r))
        };
        let mut c = contract();
        c.identification = IdentificationStatus::NotIdentified;
        assert_eq!(
            bind(&c, response()),
            Err(E::ContractNotIdentified(IdentificationStatus::NotIdentified))
        );
        let mut r = response();
        r.header.graph_id = "graph-2".into();
        assert_eq!(bind(&contract(), r), Err(E::GraphMismatch));
        let mut r = response();
        r.values.pop();
        assert_eq!(bind(&contract(), r), Err(E::DimensionMismatch));
        let mut r = response();
        r.values[1] = f64::NAN;
        assert_eq!(bind(&contract(), r), Err(E::NonFiniteValue(1)));
        let mut r = response();
        r.header.quantities[1].horizon = 5;
        assert_eq!(bind(&contract(), r), Err(E::CoordinateMismatch(1, QuantityMismatch::Horizon)));
        let mut r = response();
        r.header.quantities[0].units = "kPa".into();
        assert_eq!(bind(&contract(), r), Err(E::CoordinateMismatch(0, QuantityMismatch::Units)));
        let mut r = response();
        r.header.evidence_ids.clear();
        assert_eq!(bind(&contract(), r), Err(E::MissingEvidence("factor:z".into())));
        let mut r = response();
        r.header.assumption_ids.clear();
        assert_eq!(bind(&contract(), r), Err(E::MissingAssumption("ignorability".into())));
        let mut r = response();
        r.header.trust = ExternalTrustState::NativeLicensed;
        assert_eq!(bind(&contract(), r), Err(E::TrustMismatch));
    }

    #[test]
    fn observational_law_needs_a_checked_equivalence() {
        let mut r = response();
        for q in &mut r.header.quantities {
            q.regime_id = OBSERVATIONAL_REGIME.into();
        }
        let bind = |c: &CheckedCausalContract, r: &ExternalResponse| {
            bind_external_result(c, &ExternalResult::Response(r.clone()))
        };
        assert_eq!(bind(&contract(), &r), Err(ExternalBindingError::UncheckedObservationalLaw(0)));
        let mut c = contract();
        c.equivalences.push(CheckedEquivalence {
            graph_id: "other".into(),
            interventional_regime_id: "do(a=1)".into(),
            justification_id: "backdoor:z".into(),
        });
        assert_eq!(bind(&c, &r), Err(ExternalBindingError::UncheckedObservationalLaw(0)));
        c.equivalences[0].graph_id = "graph-1".into();
        let claim = bind(&c, &r).unwrap();
        assert_eq!(claim.equivalence_ids(), ["backdoor:z".to_owned()]);
        // A different population is not rescued by the regime equivalence.
        let mut wrong = r;
        wrong.header.quantities[0].population_id = "source".into();
        assert_eq!(
            bind(&c, &wrong),
            Err(ExternalBindingError::CoordinateMismatch(0, QuantityMismatch::Population))
        );
    }

    #[test]
    fn distribution_and_posterior_check_meaning_and_kind() {
        let quantities = contract().estimand;
        let header = |object| ExternalResultHeader {
            object,
            graph_id: "graph-1".into(),
            quantities: quantities.clone(),
            evidence_ids: vec!["factor:z".into()],
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        };
        let dist = ExternalDistribution {
            header: header(law(quantities.clone())),
            meaning: DistributionMeaning::InterventionalPredictive,
        };
        let claim =
            bind_external_result(&contract(), &ExternalResult::Distribution(dist.clone())).unwrap();
        assert_eq!(claim.meaning(), Some(DistributionMeaning::InterventionalPredictive));
        let mut c = contract();
        c.accepted_meanings = vec![DistributionMeaning::CausalFunctionalPosterior];
        assert_eq!(
            bind_external_result(&c, &ExternalResult::Distribution(dist)),
            Err(ExternalBindingError::MeaningNotAccepted)
        );
        let posterior_object = ExternalScientificObject::Posterior(PosteriorProviderContract {
            identity: identity(),
            kind: ExternalPosteriorKind::CausalFunctional,
            quantities: quantities.clone(),
            capabilities: vec![ExternalCapability::Sample],
        });
        let post = ExternalPosterior {
            header: header(posterior_object),
            kind: ExternalPosteriorKind::Parameter,
        };
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Posterior(post.clone())),
            Err(ExternalBindingError::PosteriorKindMismatch)
        );
        let mut ok = post;
        ok.kind = ExternalPosteriorKind::CausalFunctional;
        assert!(bind_external_result(&contract(), &ExternalResult::Posterior(ok.clone())).is_ok());
        // A posterior offered where the object declares a law is a kind error.
        let mut wrong = ok;
        wrong.header.object = law(quantities);
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Posterior(wrong)),
            Err(ExternalBindingError::ObjectKindMismatch)
        );
    }

    #[test]
    fn verification_of_another_contract_does_not_bind() {
        use crate::{VerificationProbe, VerificationProbeKind as P, verify_external_object};
        let r = response();
        let mut other = r.header.object.clone();
        if let ExternalScientificObject::Law(l) = &mut other {
            l.identity.request_id = "req-2".into();
        }
        let probes: Vec<_> = [P::Shape, P::Support, P::Moments, P::SeededBehavior, P::KnownTruth]
            .into_iter()
            .map(|kind| VerificationProbe { kind, observed: 1.0, expected: 1.0, tolerance: 0.0 })
            .collect();
        let receipt = verify_external_object(&other, &probes).unwrap();
        let mut bad = r.clone();
        bad.header.trust = ExternalTrustState::ExactRequestVerified(Box::new(receipt));
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Response(bad)),
            Err(ExternalBindingError::TrustMismatch)
        );
        let good = verify_external_object(&r.header.object, &probes).unwrap();
        let mut ok = r;
        ok.header.trust = ExternalTrustState::ExactRequestVerified(Box::new(good));
        let claim = bind_external_result(&contract(), &ExternalResult::Response(ok)).unwrap();
        assert_eq!(claim.trust(), BoundTrustLevel::ExactRequestVerified);
    }
}
