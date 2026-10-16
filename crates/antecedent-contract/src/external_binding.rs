//! Binding of externally supplied scientific results to a checked causal contract.
//!
//! The contract says what was identified; the external value says only what a
//! foreign system computed. Binding checks that they describe the same
//! quantity and never relabels foreign numerical work as native estimation.

use std::collections::HashSet;

use crate::{
    CompositionLink, CompositionStage, DistributionMeaning, ExternalCapability,
    ExternalContractError, ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState,
    ProvenanceChain, ProvenanceChainError, ProviderObjectIdentity, QuantityMismatch,
    ScientificQuantity, VerificationProbe,
};
use antecedent_core::text::is_blank;
use antecedent_core::{IdentificationStatus, SupportStatus};

/// Regime identity of an unintervened conditional law.
pub const OBSERVATIONAL_REGIME: &str = "observational";

/// What a checked equivalence lets an observational law stand for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EquivalenceScope {
    /// A coordinate that differs from the request only in its regime.
    Regime {
        /// Interventional regime that the observational law may stand for.
        interventional_regime_id: String,
    },
    /// A conditional law `P(Y | T = v)` standing for `P(Y | do(T = v))`: the
    /// observational coordinate conditions on the treatment, and each
    /// conditioning value maps to exactly the interventional regime it licenses.
    ConditionedTreatment {
        /// Treatment variable whose conditioning is replaced by intervention.
        variable_id: String,
        /// `(conditioning value id, interventional regime id)` pairs.
        value_to_regime: Vec<(String, String)>,
    },
}

/// A separately checked licence to read an observational law as interventional.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckedEquivalence {
    /// Graph for which the equivalence was checked.
    pub graph_id: String,
    /// Stable identity of the check (for example an adjustment-set proof).
    pub justification_id: String,
    /// Which observational coordinates it licenses.
    pub scope: EquivalenceScope,
}

impl CheckedEquivalence {
    /// Whether this equivalence licenses reading `offered` as `wanted`.
    ///
    /// Everything except the regime (and, for a conditioned treatment, the one
    /// conditioning entry on the treatment) must already agree; a different
    /// population, horizon, units or other condition is never rescued.
    fn licenses(
        &self,
        graph_id: &str,
        offered: &ScientificQuantity,
        wanted: &ScientificQuantity,
    ) -> bool {
        if self.graph_id != graph_id || is_blank(&self.justification_id) {
            return false;
        }
        let mut as_requested = offered.clone();
        as_requested.regime_id.clone_from(&wanted.regime_id);
        match &self.scope {
            EquivalenceScope::Regime { interventional_regime_id } => {
                *interventional_regime_id == wanted.regime_id
                    && as_requested.require_same_coordinate(wanted).is_ok()
            }
            EquivalenceScope::ConditionedTreatment { variable_id, value_to_regime } => {
                let mut matched = offered
                    .conditioning
                    .iter()
                    .filter(|condition| condition.variable_id == *variable_id);
                let (Some(treatment), None) = (matched.next(), matched.next()) else {
                    return false;
                };
                let maps = value_to_regime
                    .iter()
                    .filter(|(value, regime)| {
                        *value == treatment.value_id && *regime == wanted.regime_id
                    })
                    .count()
                    == 1;
                as_requested.conditioning.retain(|condition| condition.variable_id != *variable_id);
                maps && as_requested.require_same_coordinate(wanted).is_ok()
            }
        }
    }
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

/// Declared support of a column's rows.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnSupport {
    /// One label for every row.
    Uniform(SupportStatus),
    /// One label per row, in row order.
    PerRow(Vec<SupportStatus>),
}

/// A column of row-level values, such as one conditional effect per unit.
///
/// A response grid names one coordinate per value, which is the right shape
/// for a dose curve and the wrong one for thousands of rows. A column has *one*
/// scientific quantity (the per-row estimand) and a value per row; the claim
/// keeps the column's digest, not its rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalColumn {
    /// Shared declarations; `quantities` holds exactly the one row-level quantity.
    pub header: ExternalResultHeader,
    /// The finite values, one per row.
    pub values: Vec<f64>,
    /// Uncertainty meaning.
    pub uncertainty: ExternalUncertaintyMeaning,
    /// Provider-declared support of the rows. Absent means support was not
    /// assessed, so every row is `MissingEvidence`, never `Supported`.
    pub support: Option<ColumnSupport>,
}

/// What a bound claim retains of a column: its size, digest and support.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColumnClaim {
    rows: usize,
    values_blake3: String,
    support_counts: Vec<(SupportStatus, usize)>,
}

impl ColumnClaim {
    /// Name of the only element type a column holds.
    pub const DTYPE: &'static str = "f64";

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }
    /// Digest of the ordered values, [`external_values_digest`].
    #[must_use]
    pub fn values_blake3(&self) -> &str {
        &self.values_blake3
    }
    /// Rows per support label, weakest label last, labels with no rows omitted.
    #[must_use]
    pub fn support_counts(&self) -> &[(SupportStatus, usize)] {
        &self.support_counts
    }
    /// Whether `values` is exactly the column this claim was bound to.
    #[must_use]
    pub fn covers(&self, values: &[f64]) -> bool {
        values.len() == self.rows && external_values_digest(values) == self.values_blake3
    }
}

/// Digest of ordered response or column values: the length and every value's
/// bit pattern under the shared canonical writer, so a changed value is a
/// changed identity.
#[must_use]
pub fn external_values_digest(values: &[f64]) -> String {
    let mut hasher = antecedent_core::CanonicalWriter::new("antecedent.external_claim.values");
    hasher.word(values.len() as u64);
    for value in values {
        hasher.real(*value);
    }
    hasher.finish_hex()
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
    /// Row-level column under one quantity.
    Column(ExternalColumn),
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
    /// The values are labeled with different coordinates than the provider object declares.
    ObjectQuantityMismatch,
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
    /// A column was labelled with this many quantities, not the one it answers.
    ColumnQuantities(usize),
    /// A column has no rows.
    EmptyColumn,
    /// A column's per-row support has a different length than its values.
    ColumnSupportLength {
        /// Rows in the column.
        rows: usize,
        /// Labels supplied.
        supplied: usize,
    },
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
    column: Option<ColumnClaim>,
    point_status: Option<Vec<SupportStatus>>,
    meaning: Option<DistributionMeaning>,
    uncertainty: ExternalUncertaintyMeaning,
    execution: ProviderObjectIdentity,
    trust: BoundTrustLevel,
    provider_meaning: Option<DistributionMeaning>,
    capabilities: Vec<ExternalCapability>,
    verification: Vec<VerificationProbe>,
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
    /// The bound column, when this is a row-level column rather than a grid.
    #[must_use]
    pub fn column(&self) -> Option<&ColumnClaim> {
        self.column.as_ref()
    }
    /// Per-coordinate support of a response grid, or the worst row label of a
    /// column's one coordinate; `MissingEvidence` where the provider declared
    /// none. Absent for laws and posteriors.
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
    /// Meaning of the law behind a response grid; absent for a posterior object.
    #[must_use]
    pub const fn provider_meaning(&self) -> Option<DistributionMeaning> {
        self.provider_meaning
    }
    /// Operations the provider object declared at this exact request.
    #[must_use]
    pub fn capabilities(&self) -> &[ExternalCapability] {
        &self.capabilities
    }
    /// Probes that passed when trust is exact-request verified; empty otherwise.
    #[must_use]
    pub fn verification(&self) -> &[VerificationProbe] {
        &self.verification
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
        if is_blank(causal_contract_id) {
            return Err(ProvenanceChainError::InvalidLink);
        }
        let contract = format!("contract:{causal_contract_id}");
        let mut links = vec![CompositionLink {
            id: contract.clone(),
            stage: CompositionStage::CausalContract,
            parents: vec![],
            declared_parent_digests: None,
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
                declared_parent_digests: None,
            });
        }
        for id in &self.equivalence_ids {
            let link = format!("equivalence:{id}");
            claim_parents.push(link.clone());
            links.push(CompositionLink {
                id: link,
                stage: CompositionStage::Transformation,
                parents: vec![contract.clone()],
                declared_parent_digests: None,
            });
        }
        let provider = format!("provider:{}", self.provenance_label());
        claim_parents.push(provider.clone());
        links.push(CompositionLink {
            id: provider,
            stage: CompositionStage::ExternalProvider,
            parents: provider_parents,
            declared_parent_digests: None,
        });
        links.push(CompositionLink {
            id: Self::CLAIM_LINK_ID.into(),
            stage: CompositionStage::Claim,
            parents: claim_parents,
            declared_parent_digests: None,
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
                let Some(eq) = contract
                    .equivalences
                    .iter()
                    .find(|e| e.licenses(&contract.graph_id, got, wanted))
                else {
                    // Name a plain mismatch when something besides the regime (and
                    // any treatment conditioning an equivalence could replace)
                    // differs; otherwise only the licence is missing.
                    let mut as_requested = got.clone();
                    as_requested.regime_id.clone_from(&wanted.regime_id);
                    for equivalence in &contract.equivalences {
                        if let EquivalenceScope::ConditionedTreatment { variable_id, .. } =
                            &equivalence.scope
                        {
                            as_requested.conditioning.retain(|c| c.variable_id != *variable_id);
                        }
                    }
                    as_requested
                        .require_same_coordinate(wanted)
                        .map_err(|m| E::CoordinateMismatch(index, m))?;
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

fn check_contract(contract: &CheckedCausalContract) -> Result<(), ExternalBindingError> {
    if !matches!(
        contract.identification,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    ) {
        return Err(ExternalBindingError::ContractNotIdentified(contract.identification));
    }
    if is_blank(&contract.graph_id)
        || contract.estimand.is_empty()
        || contract.estimand.iter().any(|q| q.validate().is_err())
    {
        return Err(ExternalBindingError::InvalidContract);
    }
    Ok(())
}

/// The trust a result may be bound at: native licence is never a provider's.
fn bound_trust(header: &ExternalResultHeader) -> Result<BoundTrustLevel, ExternalBindingError> {
    match &header.trust {
        ExternalTrustState::NativeLicensed => Err(ExternalBindingError::TrustMismatch),
        ExternalTrustState::ExternallyAttested { .. } => Ok(BoundTrustLevel::ExternallyAttested),
        state @ ExternalTrustState::ExactRequestVerified(_) => {
            if state.verifies(&header.object) {
                Ok(BoundTrustLevel::ExactRequestVerified)
            } else {
                Err(ExternalBindingError::TrustMismatch)
            }
        }
    }
}

/// The retained form of a column and the worst label over its rows.
fn check_column(
    column: &ExternalColumn,
) -> Result<(ColumnClaim, SupportStatus), ExternalBindingError> {
    use ExternalBindingError as E;
    let rows = column.values.len();
    if rows == 0 {
        return Err(E::EmptyColumn);
    }
    if let Some(index) = column.values.iter().position(|x| !x.is_finite()) {
        return Err(E::NonFiniteValue(index));
    }
    let mut counts = [0_usize; 5];
    let slot = |status: SupportStatus| usize::from(status.severity());
    match &column.support {
        None => counts[slot(SupportStatus::MissingEvidence)] = rows,
        Some(ColumnSupport::Uniform(status)) => counts[slot(*status)] = rows,
        Some(ColumnSupport::PerRow(labels)) => {
            if labels.len() != rows {
                return Err(E::ColumnSupportLength { rows, supplied: labels.len() });
            }
            for status in labels {
                counts[slot(*status)] += 1;
            }
        }
    }
    let ordered = [
        SupportStatus::Supported,
        SupportStatus::WeakOverlap,
        SupportStatus::Extrapolative,
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::MissingEvidence,
    ];
    let support_counts: Vec<_> = ordered
        .into_iter()
        .map(|status| (status, counts[slot(status)]))
        .filter(|(_, count)| *count > 0)
        .collect();
    let worst = support_counts.last().map_or(SupportStatus::MissingEvidence, |(status, _)| *status);
    Ok((
        ColumnClaim { rows, values_blake3: external_values_digest(&column.values), support_counts },
        worst,
    ))
}

fn check_provider_kind(
    contract: &CheckedCausalContract,
    result: &ExternalResult,
    header: &ExternalResultHeader,
    meaning: Option<DistributionMeaning>,
) -> Result<(), ExternalBindingError> {
    use ExternalBindingError as E;
    match (result, &header.object) {
        (
            ExternalResult::Response(_)
            | ExternalResult::Column(_)
            | ExternalResult::Distribution(_),
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
    Ok(())
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
    check_contract(contract)?;
    let (header, values, meaning, uncertainty) = match result {
        ExternalResult::Response(r) => (&r.header, Some(&r.values), None, r.uncertainty.clone()),
        ExternalResult::Column(c) => (&c.header, None, None, c.uncertainty.clone()),
        ExternalResult::Distribution(d) => {
            (&d.header, None, Some(d.meaning), ExternalUncertaintyMeaning::None)
        }
        ExternalResult::Posterior(p) => (&p.header, None, None, ExternalUncertaintyMeaning::None),
    };
    header.object.validate().map_err(E::Object)?;
    check_provider_kind(contract, result, header, meaning)?;
    if header.graph_id != contract.graph_id {
        return Err(E::GraphMismatch);
    }
    let declared_support = match result {
        ExternalResult::Response(r) => r.point_support.as_ref(),
        _ => None,
    };
    if matches!(result, ExternalResult::Column(_)) && header.quantities.len() != 1 {
        return Err(E::ColumnQuantities(header.quantities.len()));
    }
    if header.quantities.len() != contract.estimand.len()
        || values.is_some_and(|v| v.len() != header.quantities.len())
        || declared_support.is_some_and(|p| p.len() != header.quantities.len())
    {
        return Err(E::DimensionMismatch);
    }
    if let Some(index) = values.and_then(|v| v.iter().position(|x| !x.is_finite())) {
        return Err(E::NonFiniteValue(index));
    }
    let column = match result {
        ExternalResult::Column(c) => Some(check_column(c)?),
        _ => None,
    };
    if meaning.is_some_and(|m| !contract.accepted_meanings.contains(&m)) {
        return Err(E::MeaningNotAccepted);
    }
    let equivalence_ids = check_coordinates(contract, &header.quantities)?;
    let object_quantities = match &header.object {
        ExternalScientificObject::Law(law) => &law.quantities,
        ExternalScientificObject::Posterior(post) => &post.quantities,
        _ => unreachable!("object kind was checked above"),
    };
    if object_quantities.len() != header.quantities.len()
        || object_quantities
            .iter()
            .zip(&header.quantities)
            .any(|(declared, supplied)| declared.require_same_coordinate(supplied).is_err())
    {
        return Err(E::ObjectQuantityMismatch);
    }
    if let Some(id) = declared(&contract.required_evidence_ids, &header.evidence_ids) {
        return Err(E::MissingEvidence(id));
    }
    if let Some(id) = declared(&contract.required_assumption_ids, &header.assumption_ids) {
        return Err(E::MissingAssumption(id));
    }
    let trust = bound_trust(header)?;
    Ok(BoundExternalClaim {
        graph_id: contract.graph_id.clone(),
        identification: contract.identification,
        quantities: contract.estimand.clone(),
        values: values.cloned(),
        point_status: column.as_ref().map_or_else(
            || {
                values.map(|v| {
                    declared_support
                        .cloned()
                        .unwrap_or_else(|| vec![SupportStatus::MissingEvidence; v.len()])
                })
            },
            |(_, worst)| Some(vec![*worst]),
        ),
        column: column.map(|(claim, _)| claim),
        meaning,
        uncertainty,
        execution: header.object.identity().clone(),
        trust,
        provider_meaning: match &header.object {
            ExternalScientificObject::Law(law) => Some(law.meaning),
            _ => None,
        },
        capabilities: header.object.capabilities().to_vec(),
        verification: match &header.trust {
            ExternalTrustState::ExactRequestVerified(receipt) => {
                receipt.probes_for(&header.object).map(<[_]>::to_vec).unwrap_or_default()
            }
            _ => Vec::new(),
        },
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

    fn column(
        values: Vec<f64>,
        support: Option<ColumnSupport>,
    ) -> (CheckedCausalContract, ExternalColumn) {
        let quantity =
            ScientificQuantity { functional_id: "cate".into(), ..quantity("do(a=1)", 0) };
        let mut c = contract();
        c.estimand = vec![quantity.clone()];
        let object = law(vec![quantity.clone()]);
        let column = ExternalColumn {
            header: ExternalResultHeader {
                object,
                graph_id: "graph-1".into(),
                quantities: vec![quantity],
                evidence_ids: vec!["factor:z".into()],
                assumption_ids: vec!["ignorability".into()],
                trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
            },
            values,
            uncertainty: ExternalUncertaintyMeaning::None,
            support,
        };
        (c, column)
    }

    #[test]
    fn a_column_binds_one_quantity_and_keeps_its_digest_and_support_not_its_rows() {
        let rows: Vec<f64> = (0..20_000).map(|i| f64::from(i) * 0.001).collect();
        let labels: Vec<_> = (0..rows.len())
            .map(|i| if i % 4 == 0 { SupportStatus::WeakOverlap } else { SupportStatus::Supported })
            .collect();
        let (c, column) = column(rows.clone(), Some(ColumnSupport::PerRow(labels)));
        let claim = bind_external_result(&c, &ExternalResult::Column(column)).unwrap();
        assert_eq!(claim.values(), None);
        assert_eq!(claim.quantities().len(), 1);
        let kept = claim.column().unwrap();
        assert_eq!(kept.rows(), 20_000);
        assert_eq!(
            kept.support_counts(),
            [(SupportStatus::Supported, 15_000), (SupportStatus::WeakOverlap, 5_000)]
        );
        assert_eq!(claim.point_status(), Some(&[SupportStatus::WeakOverlap][..]));
        assert_eq!(claim.support_status(), Some(SupportStatus::WeakOverlap));
        assert!(kept.covers(&rows));
        let mut changed = rows;
        changed[7] += 1e-9;
        assert!(!kept.covers(&changed));
        // The same lineage and provider identity as a grid.
        assert_eq!(claim.provenance_label(), "external:lab/curve@v3#snap-9");
        assert!(claim.provenance_chain("checked-contract").is_ok());
    }

    #[test]
    fn a_column_without_declared_support_is_missing_evidence_never_supported() {
        let (c, column) = column(vec![1.0, 2.0], None);
        let claim = bind_external_result(&c, &ExternalResult::Column(column)).unwrap();
        assert_eq!(claim.support_status(), Some(SupportStatus::MissingEvidence));
        assert_eq!(claim.column().unwrap().support_counts(), [(SupportStatus::MissingEvidence, 2)]);
    }

    #[test]
    fn a_column_refuses_empty_non_finite_and_mislabelled_rows() {
        use ExternalBindingError as E;
        let (c, empty) = column(vec![], None);
        assert_eq!(bind_external_result(&c, &ExternalResult::Column(empty)), Err(E::EmptyColumn));
        let (c, bad) = column(vec![1.0, f64::NAN], None);
        assert_eq!(
            bind_external_result(&c, &ExternalResult::Column(bad)),
            Err(E::NonFiniteValue(1))
        );
        let (c, short) = column(
            vec![1.0, 2.0, 3.0],
            Some(ColumnSupport::PerRow(vec![SupportStatus::Supported; 2])),
        );
        assert_eq!(
            bind_external_result(&c, &ExternalResult::Column(short)),
            Err(E::ColumnSupportLength { rows: 3, supplied: 2 })
        );
        // A column answers exactly one coordinate: it is neither labelled with a grid's
        // coordinates nor checked against a contract of several.
        let (c3, mut wide) = column(vec![1.0], None);
        wide.header.quantities = contract().estimand;
        assert_eq!(
            bind_external_result(&c3, &ExternalResult::Column(wide)),
            Err(E::ColumnQuantities(2))
        );
        let (_, one) = column(vec![1.0], None);
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Column(one)),
            Err(E::DimensionMismatch)
        );
    }

    #[test]
    fn bound_claim_answers_where_its_numbers_came_from() {
        use crate::CompositionStage as S;
        let mut c = contract();
        c.equivalences.push(CheckedEquivalence {
            graph_id: "graph-1".into(),
            justification_id: "backdoor:z".into(),
            scope: EquivalenceScope::Regime { interventional_regime_id: "do(a=1)".into() },
        });
        let mut r = response();
        for q in &mut r.header.quantities {
            q.regime_id = OBSERVATIONAL_REGIME.into();
        }
        r.header.object = law(r.header.quantities.clone());
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

    fn dose_contract() -> (CheckedCausalContract, Vec<ScientificQuantity>) {
        let mut c = contract();
        c.estimand = vec![quantity("do(a=0)", 0), quantity("do(a=1)", 0)];
        c.equivalences.push(CheckedEquivalence {
            graph_id: "graph-1".into(),
            justification_id: "backdoor:z".into(),
            scope: EquivalenceScope::ConditionedTreatment {
                variable_id: "a".into(),
                value_to_regime: vec![
                    ("0".into(), "do(a=0)".into()),
                    ("1".into(), "do(a=1)".into()),
                ],
            },
        });
        let offered = ["0", "1"]
            .iter()
            .map(|dose| {
                let mut q = quantity(OBSERVATIONAL_REGIME, 0);
                q.conditioning = vec![crate::QuantityCondition {
                    variable_id: "a".into(),
                    value_id: (*dose).into(),
                }];
                q
            })
            .collect();
        (c, offered)
    }

    fn offer(quantities: Vec<ScientificQuantity>) -> ExternalResult {
        let mut r = response();
        r.header.object = law(quantities.clone());
        r.header.quantities = quantities;
        ExternalResult::Response(r)
    }

    #[test]
    fn observational_conditional_law_over_a_dose_grid_binds_under_its_equivalence() {
        let (c, offered) = dose_contract();
        let claim = bind_external_result(&c, &offer(offered.clone())).unwrap();
        assert_eq!(claim.equivalence_ids(), ["backdoor:z".to_owned()]);
        assert_eq!(claim.quantities()[1].regime_id, "do(a=1)");
        assert!(claim.quantities().iter().all(|q| q.conditioning.is_empty()));

        // A dose mapped to a different intervention is refused.
        let mut crossed = c.clone();
        if let EquivalenceScope::ConditionedTreatment { value_to_regime, .. } =
            &mut crossed.equivalences[0].scope
        {
            value_to_regime[1].1 = "do(a=0)".into();
        }
        assert_eq!(
            bind_external_result(&crossed, &offer(offered.clone())),
            Err(ExternalBindingError::UncheckedObservationalLaw(1))
        );
        // Another treatment variable's conditioning is not covered.
        let mut other = c.clone();
        if let EquivalenceScope::ConditionedTreatment { variable_id, .. } =
            &mut other.equivalences[0].scope
        {
            *variable_id = "b".into();
        }
        assert_eq!(
            bind_external_result(&other, &offer(offered.clone())),
            Err(ExternalBindingError::CoordinateMismatch(0, QuantityMismatch::Conditioning))
        );
        // Nothing else is rescued: a stray extra condition or another population.
        let mut extra = offered.clone();
        extra[0]
            .conditioning
            .push(crate::QuantityCondition { variable_id: "z".into(), value_id: "high".into() });
        assert_eq!(
            bind_external_result(&c, &offer(extra)),
            Err(ExternalBindingError::CoordinateMismatch(0, QuantityMismatch::Conditioning))
        );
        let mut wrong_population = offered.clone();
        wrong_population[1].population_id = "source".into();
        assert_eq!(
            bind_external_result(&c, &offer(wrong_population)),
            Err(ExternalBindingError::CoordinateMismatch(1, QuantityMismatch::Population))
        );
        // The equivalence belongs to its graph.
        let mut elsewhere = c;
        elsewhere.equivalences[0].graph_id = "other".into();
        assert_eq!(
            bind_external_result(&elsewhere, &offer(offered)),
            Err(ExternalBindingError::UncheckedObservationalLaw(0))
        );
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
    fn result_cannot_relabel_a_provider_objects_declared_coordinates() {
        let mut r = response();
        if let ExternalScientificObject::Law(law) = &mut r.header.object {
            law.quantities[1].horizon = 2;
        }
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Response(r)),
            Err(ExternalBindingError::ObjectQuantityMismatch)
        );
        let mut r = response();
        if let ExternalScientificObject::Law(law) = &mut r.header.object {
            law.quantities.swap(0, 1);
        }
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Response(r)),
            Err(ExternalBindingError::ObjectQuantityMismatch)
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
        r.header.object = law(r.header.quantities.clone());
        let bind = |c: &CheckedCausalContract, r: &ExternalResponse| {
            bind_external_result(c, &ExternalResult::Response(r.clone()))
        };
        assert_eq!(bind(&contract(), &r), Err(ExternalBindingError::UncheckedObservationalLaw(0)));
        let mut c = contract();
        c.equivalences.push(CheckedEquivalence {
            graph_id: "other".into(),
            justification_id: "backdoor:z".into(),
            scope: EquivalenceScope::Regime { interventional_regime_id: "do(a=1)".into() },
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

    fn passing_probes() -> Vec<crate::VerificationProbe> {
        use crate::VerificationProbeKind as P;
        [P::Shape, P::Support, P::Moments, P::SeededBehavior, P::KnownTruth]
            .into_iter()
            .map(|kind| crate::VerificationProbe {
                kind,
                observed: 1.0,
                expected: 1.0,
                tolerance: 0.0,
            })
            .collect()
    }

    /// R1 carries `verified_extension`, R2 stays externally
    /// attested, and neither becomes native.
    #[test]
    fn attested_stays_attested_and_verified_is_never_native() {
        use crate::verify_external_object;
        let r2 = response();
        let attested =
            bind_external_result(&contract(), &ExternalResult::Response(r2.clone())).unwrap();
        assert_eq!(attested.trust(), BoundTrustLevel::ExternallyAttested);
        assert!(attested.verification().is_empty());

        let mut r1 = r2;
        let receipt = verify_external_object(&r1.header.object, &passing_probes()).unwrap();
        r1.header.trust = ExternalTrustState::ExactRequestVerified(Box::new(receipt));
        let verified = bind_external_result(&contract(), &ExternalResult::Response(r1)).unwrap();
        assert_eq!(verified.trust(), BoundTrustLevel::ExactRequestVerified);
        assert_eq!(verified.verification().len(), 5);
        for claim in [&attested, &verified] {
            assert!(!claim.is_native_estimation());
        }
        let mut native = response();
        native.header.trust = ExternalTrustState::NativeLicensed;
        assert_eq!(
            bind_external_result(&contract(), &ExternalResult::Response(native)),
            Err(ExternalBindingError::TrustMismatch)
        );
    }

    /// Reusing an R1 receipt after a changed quantity,
    /// provider version or snapshot refuses.
    #[test]
    fn receipt_for_one_fingerprint_does_not_survive_changed_quantity_version_or_snapshot() {
        use crate::verify_external_object;
        let r1 = response();
        let receipt = verify_external_object(&r1.header.object, &passing_probes()).unwrap();
        let reused = |edit: &dyn Fn(&mut ExternalScientificObject)| {
            let mut other = r1.clone();
            edit(&mut other.header.object);
            other.header.trust =
                ExternalTrustState::ExactRequestVerified(Box::new(receipt.clone()));
            bind_external_result(&contract(), &ExternalResult::Response(other))
        };
        let edits: [&dyn Fn(&mut ExternalScientificObject); 3] = [
            &|o| {
                if let ExternalScientificObject::Law(l) = o {
                    l.identity.version_id = "v4".into();
                }
            },
            &|o| {
                if let ExternalScientificObject::Law(l) = o {
                    l.identity.snapshot_id = "snap-10".into();
                }
            },
            &|o| {
                if let ExternalScientificObject::Law(l) = o {
                    l.capabilities.push(ExternalCapability::Cdf);
                }
            },
        ];
        for edit in edits {
            assert_eq!(reused(edit), Err(ExternalBindingError::TrustMismatch));
        }
        // A changed coordinate of the law is caught by the quantity check or the
        // receipt, never silently reused.
        let mut changed_quantity = r1.clone();
        if let ExternalScientificObject::Law(l) = &mut changed_quantity.header.object {
            l.quantities[0].units = "kPa".into();
        }
        changed_quantity.header.trust = ExternalTrustState::ExactRequestVerified(Box::new(receipt));
        assert!(
            bind_external_result(&contract(), &ExternalResult::Response(changed_quantity)).is_err()
        );
    }
}
