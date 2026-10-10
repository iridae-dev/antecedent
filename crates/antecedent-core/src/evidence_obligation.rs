//! Machine-readable evidence obligations.
//!
//! A failed causal contract (a transport derivation that cannot bind a factor,
//! a z-transport proof with an unsupplied joint law, a coordinate whose support
//! is missing, a front-door or back-door set that was never measured) owes
//! concrete evidence. An [`EvidenceObligation`] states that debt in a form a
//! planner can test against a proposed study: a stable id, a kind, the variables
//! and population, the regime (interventions, conditioning, joint versus
//! marginal), the reason, the contract slots it fills, and the source proof step
//! it came from.
//!
//! An obligation is a request, never a verdict. [`EvidenceObligation::addressed_by`]
//! is a necessary-condition screen on population, regime and joint law: a study
//! that matches only variable names never addresses it, and a study that passes
//! the screen still repairs identification only when the theorem-specific
//! checker that owns the contract accepts the hypothetical evidence. An
//! [`EvidenceObligationKind::EstablishAssumption`] obligation is never
//! addressed by a study: a proposed study does not prove an assumption.
//!
//! The vocabulary reuses [`crate::ObligationScope`] and the unresolved
//! [`crate::ObligationRecord`] of the existing assumption records; no existing
//! identifier changes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    EvidenceCatalog, FactorNeed, ObligationRecord, ObligationScope, ScientificQuantity, VariableId,
};

/// Declared bound on the variable coordinates of one obligation.
pub const MAX_OBLIGATION_COORDINATES: usize = 1024;

/// Longest contract slot name, in bytes.
const MAX_SLOT_BYTES: usize = 256;

/// What kind of evidence an obligation asks for.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[non_exhaustive]
pub enum EvidenceObligationKind {
    /// Measure variables in a population/regime that already exists.
    Measure,
    /// Run an experiment (hard intervention set) in a population.
    Intervene,
    /// Observe a population that has no available law yet.
    ObservePopulation,
    /// Observe a named environment.
    ObserveEnvironment,
    /// Collect more rows of an existing design.
    IncreaseSample,
    /// Supply one joint law over several variables in one regime.
    ProvideJointLaw,
    /// Supply a conditional law (conditioning coordinates retained).
    ProvideConditionalLaw,
    /// Establish support (positivity / overlap / coverage) at coordinates.
    EstablishSupport,
    /// Establish an assumption. A proposed study never satisfies this.
    EstablishAssumption,
}

impl EvidenceObligationKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 9] = [
        Self::Measure,
        Self::Intervene,
        Self::ObservePopulation,
        Self::ObserveEnvironment,
        Self::IncreaseSample,
        Self::ProvideJointLaw,
        Self::ProvideConditionalLaw,
        Self::EstablishSupport,
        Self::EstablishAssumption,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Measure => "measure",
            Self::Intervene => "intervene",
            Self::ObservePopulation => "observe_population",
            Self::ObserveEnvironment => "observe_environment",
            Self::IncreaseSample => "increase_sample",
            Self::ProvideJointLaw => "provide_joint_law",
            Self::ProvideConditionalLaw => "provide_conditional_law",
            Self::EstablishSupport => "establish_support",
            Self::EstablishAssumption => "establish_assumption",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    /// Whether a proposed study could ever address this kind. Only
    /// [`Self::EstablishAssumption`] cannot: a study produces evidence, it does
    /// not prove an assumption.
    #[must_use]
    pub const fn satisfiable_by_study(self) -> bool {
        !matches!(self, Self::EstablishAssumption)
    }
}

/// The regime an obligation asks evidence for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObligationRegime {
    /// Hard-intervention set; empty for an observational law.
    pub interventions: Arc<[VariableId]>,
    /// Conditioning coordinates the law retains.
    pub conditioned_on: Arc<[VariableId]>,
    /// Whether the variables must come from one joint law. A joint law over
    /// several variables is never implied by separate marginals or studies.
    pub joint: bool,
}

impl ObligationRegime {
    /// Observational, unconditional regime.
    #[must_use]
    pub fn observational(joint: bool) -> Self {
        Self { interventions: Arc::from([]), conditioned_on: Arc::from([]), joint }
    }
}

/// Where an obligation came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObligationProvenance {
    /// Obligation family or theorem route (`transport`, `z_transport`,
    /// `support`, `backdoor`, `planner_delta`, `assumption_record`, ...).
    pub family: Arc<str>,
    /// The artifact, contract or failure the obligation was read from.
    pub source: Arc<str>,
    /// The source proof step (expression leaf, rule, coordinate), when known.
    pub proof_step: Option<Arc<str>>,
}

/// Everything an obligation is made of; the stable id is derived from it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceObligationSpec {
    /// Explicit scientific coordinates keyed by structural variable identity.
    /// Empty means no measurement semantics have been declared; never infer units.
    pub quantities: BTreeMap<VariableId, ScientificQuantity>,
    /// Kind of evidence requested.
    pub kind: EvidenceObligationKind,
    /// Scope reused from the existing obligation vocabulary.
    pub scope: ObligationScope,
    /// Variables of the requested law (empty only for sample and assumption
    /// obligations).
    pub variables: Arc<[VariableId]>,
    /// Population the evidence must be collected in.
    pub population: Option<Arc<str>>,
    /// Regime of the requested law.
    pub regime: ObligationRegime,
    /// Why the evidence is owed.
    pub reason: Arc<str>,
    /// Contract slots the evidence fills (`factor:<id>`, `coordinate:<i>`, ...).
    pub required_slots: Arc<[Arc<str>]>,
    /// Rows that must be added, for [`EvidenceObligationKind::IncreaseSample`].
    pub min_additional_samples: Option<u64>,
    /// Source of the obligation.
    pub provenance: ObligationProvenance,
}

/// A refused obligation: a registered reason code and a stable detail.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct EvidenceObligationError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `evidence_obligations.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl EvidenceObligationError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: crate::reason_code!("invalid_argument"),
            detail: "evidence_obligations.invalid_obligation",
            message: message.into(),
        }
    }

    /// An obligation that a study, regime or contract cannot satisfy as stated,
    /// for example an assumption presented as satisfied by a study.
    #[must_use]
    pub fn wrong_contract(message: impl Into<String>) -> Self {
        Self {
            code: crate::reason_code!("transport_missing_evidence"),
            detail: "evidence_obligations.wrong_contract",
            message: message.into(),
        }
    }
}

/// What a candidate study could produce, as far as an obligation can screen it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceOffer {
    /// Explicit scientific coordinates keyed by structural variable identity.
    /// Empty means no measurement semantics have been declared; never infer units.
    pub quantities: BTreeMap<VariableId, ScientificQuantity>,
    /// Population the evidence would be collected in.
    pub population: Arc<str>,
    /// Hard-intervention set of the evidence.
    pub interventions: Arc<[VariableId]>,
    /// Conditioning coordinates already conditioned on.
    pub conditioned_on: Arc<[VariableId]>,
    /// Jointly or marginally measured variables.
    pub measured: Arc<[VariableId]>,
    /// Whether `measured` is one joint law (separate marginals are `false`).
    pub joint: bool,
    /// Rows the study would add, when it declares a sample size.
    pub additional_samples: Option<u64>,
}

/// One machine-readable evidence request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceObligation {
    /// Explicit scientific coordinates keyed by structural variable identity.
    /// Empty means no measurement semantics have been declared; never infer units.
    pub quantities: BTreeMap<VariableId, ScientificQuantity>,
    /// Stable id: a digest of the canonical (order-independent) content.
    pub id: Arc<str>,
    /// Kind of evidence requested.
    pub kind: EvidenceObligationKind,
    /// Scope reused from the existing obligation vocabulary.
    pub scope: ObligationScope,
    /// Variables of the requested law.
    pub variables: Arc<[VariableId]>,
    /// Population the evidence must be collected in.
    pub population: Option<Arc<str>>,
    /// Regime of the requested law.
    pub regime: ObligationRegime,
    /// Why the evidence is owed.
    pub reason: Arc<str>,
    /// Contract slots the evidence fills.
    pub required_slots: Arc<[Arc<str>]>,
    /// Rows that must be added, for `IncreaseSample`.
    pub min_additional_samples: Option<u64>,
    /// Source proof step and route.
    pub provenance: ObligationProvenance,
}

fn sorted_ids(variables: &[VariableId]) -> Vec<u32> {
    let mut ids: Vec<u32> = variables.iter().map(|v| v.raw()).collect();
    ids.sort_unstable();
    ids
}

fn id_list(variables: &[VariableId]) -> String {
    sorted_ids(variables).iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn distinct(variables: &[VariableId]) -> bool {
    variables.iter().collect::<BTreeSet<_>>().len() == variables.len()
}

fn same_set(a: &[VariableId], b: &[VariableId]) -> bool {
    a.iter().collect::<BTreeSet<_>>() == b.iter().collect::<BTreeSet<_>>()
}

fn scope_label(scope: ObligationScope) -> String {
    match scope {
        ObligationScope::Horizon { horizon } => format!("horizon:{horizon}"),
        other => other.as_str().to_owned(),
    }
}

impl EvidenceObligation {
    /// Build and validate an obligation; its id is derived from the canonical
    /// content, so reordering variables or slots never changes it.
    ///
    /// # Errors
    /// `evidence_obligations.invalid_obligation` for a malformed obligation
    /// (empty reason or provenance, duplicate or too many variables, a missing
    /// population, or a regime that contradicts the kind) and
    /// `evidence_obligations.wrong_contract` for a joint-law obligation that
    /// does not ask for a joint law.
    pub fn try_new(spec: EvidenceObligationSpec) -> Result<Self, EvidenceObligationError> {
        validate_spec(&spec)?;
        let id = derive_id(&spec);
        Ok(Self {
            id,
            quantities: spec.quantities,
            kind: spec.kind,
            scope: spec.scope,
            variables: spec.variables,
            population: spec.population,
            regime: spec.regime,
            reason: spec.reason,
            required_slots: spec.required_slots,
            min_additional_samples: spec.min_additional_samples,
            provenance: spec.provenance,
        })
    }

    /// Bind explicitly declared measurement semantics to a structural obligation.
    /// Every requested variable must have a valid descriptor in the same population.
    /// The returned ID includes units, regime, horizon, functional, conditioning and transform.
    ///
    /// # Errors
    /// Refuses incomplete, invalid or incompatible coordinate declarations.
    pub fn with_quantities(
        self,
        quantities: BTreeMap<VariableId, ScientificQuantity>,
    ) -> Result<Self, EvidenceObligationError> {
        if quantities.len() != self.variables.len() {
            return Err(EvidenceObligationError::invalid(
                "declared scientific coordinates must cover every requested variable",
            ));
        }
        Self::try_new(EvidenceObligationSpec {
            quantities,
            kind: self.kind,
            scope: self.scope,
            variables: self.variables,
            population: self.population,
            regime: self.regime,
            reason: self.reason,
            required_slots: self.required_slots,
            min_additional_samples: self.min_additional_samples,
            provenance: self.provenance,
        })
    }

    /// Whether a proposed study could ever address this obligation.
    #[must_use]
    pub const fn satisfiable_by_study(&self) -> bool {
        self.kind.satisfiable_by_study()
    }

    /// Canonical, order-independent content string the id is a digest of.
    #[must_use]
    pub fn canonical(&self) -> String {
        canonical_content(&EvidenceObligationSpec {
            quantities: self.quantities.clone(),
            kind: self.kind,
            scope: self.scope,
            variables: Arc::clone(&self.variables),
            population: self.population.clone(),
            regime: self.regime.clone(),
            reason: Arc::clone(&self.reason),
            required_slots: Arc::clone(&self.required_slots),
            min_additional_samples: self.min_additional_samples,
            provenance: self.provenance.clone(),
        })
    }

    /// Necessary-condition screen: could evidence of this shape supply what the
    /// obligation asks for?
    ///
    /// The population, the exact intervention set, every needed variable, the
    /// conditioning coordinates and (when several variables are read) one
    /// joint law must all match; variable names alone never do. A match does
    /// not certify identification: the theorem-specific checker still decides.
    /// An [`EvidenceObligationKind::EstablishAssumption`] obligation is never
    /// addressed.
    #[must_use]
    pub fn addressed_by(&self, offer: &EvidenceOffer) -> bool {
        if !self.satisfiable_by_study() {
            return false;
        }
        let Some(population) = self.population.as_deref() else { return false };
        if offer.population.as_ref() != population
            || !same_set(&offer.interventions, &self.regime.interventions)
        {
            return false;
        }
        let needed: BTreeSet<VariableId> =
            self.variables.iter().chain(self.regime.conditioned_on.iter()).copied().collect();
        let measured_or_set =
            |v: &VariableId| offer.measured.contains(v) || offer.interventions.contains(v);
        let conditioning_ok = offer
            .conditioned_on
            .iter()
            .all(|v| self.regime.conditioned_on.contains(v) && !self.variables.contains(v));
        let joint_ok = !(self.regime.joint || needed.len() > 1) || offer.joint;
        let samples_ok = match self.kind {
            EvidenceObligationKind::IncreaseSample => offer
                .additional_samples
                .zip(self.min_additional_samples)
                .is_some_and(|(have, need)| have >= need),
            _ => true,
        };
        let quantities_ok = self.quantities.iter().all(|(variable, quantity)| {
            offer
                .quantities
                .get(variable)
                .is_some_and(|supplied| quantity.require_same_coordinate(supplied).is_ok())
        });
        needed.iter().all(measured_or_set)
            && conditioning_ok
            && joint_ok
            && samples_ok
            && quantities_ok
    }

    /// The obligation of one executable factor that no available catalog regime
    /// satisfies. The kind is read structurally from the catalog and the need:
    ///
    /// * no available population law with exactly the needed intervention set:
    ///   `Intervene` when the need intervenes, else `ObservePopulation`;
    /// * such a regime exists but the need conditions: `ProvideConditionalLaw`;
    /// * otherwise a joint law when more than one variable is read:
    ///   `ProvideJointLaw`; else `Measure`.
    ///
    /// # Errors
    /// The need reads no variable, or too many.
    pub fn from_unmet_factor(
        catalog: &EvidenceCatalog,
        factor_id: &str,
        need: &FactorNeed<'_>,
        provenance: ObligationProvenance,
    ) -> Result<Self, EvidenceObligationError> {
        let needed = need.needed_variables();
        let has_regime = catalog.regimes.iter().any(|regime| {
            regime.supplies_population_law()
                && regime.population.as_ref() == need.population
                && same_set(&regime.interventions, need.interventions)
        });
        let kind = if !has_regime {
            if need.interventions.is_empty() {
                EvidenceObligationKind::ObservePopulation
            } else {
                EvidenceObligationKind::Intervene
            }
        } else if !need.conditioned_on.is_empty() {
            EvidenceObligationKind::ProvideConditionalLaw
        } else if needed.len() > 1 {
            EvidenceObligationKind::ProvideJointLaw
        } else {
            EvidenceObligationKind::Measure
        };
        Self::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind,
            scope: ObligationScope::Factor,
            variables: Arc::from(need.variables),
            population: Some(Arc::from(need.population)),
            regime: ObligationRegime {
                interventions: Arc::from(need.interventions),
                conditioned_on: Arc::from(need.conditioned_on),
                joint: needed.len() > 1,
            },
            reason: Arc::from(format!(
                "transport.missing_evidence: no available regime satisfies factor {factor_id}"
            )),
            required_slots: Arc::from([Arc::from(format!("factor:{factor_id}"))]),
            min_additional_samples: None,
            provenance,
        })
    }

    /// The `EstablishAssumption` obligation of an unresolved assumption record,
    /// or `None` when the record no longer blocks a claim.
    #[must_use]
    pub fn from_unresolved_record(record: &ObligationRecord) -> Option<Self> {
        if !record.is_unresolved() {
            return None;
        }
        Self::try_new(EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind: EvidenceObligationKind::EstablishAssumption,
            scope: record.scope,
            variables: Arc::from([]),
            population: None,
            regime: ObligationRegime::observational(false),
            reason: Arc::clone(&record.description),
            required_slots: Arc::from([Arc::clone(&record.id)]),
            min_additional_samples: None,
            provenance: ObligationProvenance {
                family: Arc::from("assumption_record"),
                source: Arc::clone(&record.id),
                proof_step: record.required_check.clone(),
            },
        })
        .ok()
    }
}

fn validate_spec(spec: &EvidenceObligationSpec) -> Result<(), EvidenceObligationError> {
    use EvidenceObligationKind as K;
    let invalid = EvidenceObligationError::invalid;
    if spec.reason.trim().is_empty()
        || spec.provenance.family.trim().is_empty()
        || spec.provenance.source.trim().is_empty()
        || spec.provenance.proof_step.as_deref().is_some_and(|step| step.trim().is_empty())
    {
        return Err(invalid("an obligation needs a reason and a provenance family and source"));
    }
    if spec.required_slots.iter().any(|slot| slot.trim().is_empty() || slot.len() > MAX_SLOT_BYTES)
    {
        return Err(invalid("a required contract slot is empty or too long"));
    }
    let coordinates =
        spec.variables.len() + spec.regime.interventions.len() + spec.regime.conditioned_on.len();
    if coordinates > MAX_OBLIGATION_COORDINATES
        || !distinct(&spec.variables)
        || !distinct(&spec.regime.interventions)
        || !distinct(&spec.regime.conditioned_on)
    {
        return Err(invalid(
            "obligation variables must be distinct and within the coordinate bound",
        ));
    }
    if spec.quantities.values().map(|quantity| &quantity.variable_id).collect::<BTreeSet<_>>().len()
        != spec.quantities.len()
    {
        return Err(invalid("scientific coordinates need distinct stable variable identities"));
    }
    for (variable, quantity) in &spec.quantities {
        if !spec.variables.contains(variable)
            || quantity.validate().is_err()
            || spec.population.as_deref() != Some(quantity.population_id.as_str())
            || matches!(spec.scope, ObligationScope::Horizon { horizon } if horizon != quantity.horizon)
        {
            return Err(invalid(
                "scientific coordinates must be valid and match obligation variables, population and horizon",
            ));
        }
    }
    if !spec.quantities.is_empty() && spec.quantities.len() != spec.variables.len() {
        return Err(invalid("declared scientific coordinates must cover every requested variable"));
    }
    if spec.kind == K::EstablishAssumption {
        return if spec.population.is_none() && spec.min_additional_samples.is_none() {
            Ok(())
        } else {
            Err(invalid("an assumption obligation names no population or sample size"))
        };
    }
    if spec.population.as_deref().is_none_or(|p| p.trim().is_empty()) {
        return Err(invalid("a study obligation names the population it needs evidence in"));
    }
    let needs_variables = !matches!(spec.kind, K::IncreaseSample);
    if needs_variables && spec.variables.is_empty() {
        return Err(invalid("a law obligation names at least one variable"));
    }
    match spec.kind {
        K::Intervene if spec.regime.interventions.is_empty() => {
            Err(invalid("an intervene obligation names a non-empty intervention set"))
        }
        K::ObservePopulation | K::ObserveEnvironment if !spec.regime.interventions.is_empty() => {
            Err(invalid("an observation obligation carries no hard intervention"))
        }
        K::ProvideJointLaw if spec.variables.len() < 2 => {
            Err(invalid("a joint-law obligation names at least two variables"))
        }
        K::ProvideJointLaw if !spec.regime.joint => Err(EvidenceObligationError::wrong_contract(
            "a joint-law obligation must require one joint law",
        )),
        K::ProvideConditionalLaw if spec.regime.conditioned_on.is_empty() => {
            Err(invalid("a conditional-law obligation names its conditioning coordinates"))
        }
        K::IncreaseSample if spec.min_additional_samples.is_none_or(|n| n == 0) => {
            Err(invalid("an increase-sample obligation names a positive row count"))
        }
        _ => Ok(()),
    }
}

fn canonical_content(spec: &EvidenceObligationSpec) -> String {
    let mut slots: Vec<&str> = spec.required_slots.iter().map(AsRef::as_ref).collect();
    slots.sort_unstable();
    let mut content = format!(
        "kind={};scope={};pop={};vars=[{}];do=[{}];cond=[{}];joint={};n={};slots=[{}];\
         family={};source={};step={}",
        spec.kind.as_str(),
        scope_label(spec.scope),
        spec.population.as_deref().unwrap_or(""),
        id_list(&spec.variables),
        id_list(&spec.regime.interventions),
        id_list(&spec.regime.conditioned_on),
        spec.regime.joint,
        spec.min_additional_samples.map_or_else(String::new, |n| n.to_string()),
        slots.join("|"),
        spec.provenance.family,
        spec.provenance.source,
        spec.provenance.proof_step.as_deref().unwrap_or(""),
    );
    // Preserve existing structural IDs, while binding every declared semantic dimension.
    for (variable, quantity) in &spec.quantities {
        content.push_str(&format!(
            ";quantity:{}:{}",
            variable.raw(),
            quantity.canonical_identity()
        ));
    }
    content
}

fn derive_id(spec: &EvidenceObligationSpec) -> Arc<str> {
    let digest = blake3::hash(canonical_content(spec).as_bytes()).to_hex();
    Arc::from(format!("eo1:{}:{}", spec.kind.as_str(), &digest.as_str()[..32]))
}

/// The `EstablishAssumption` obligations of every unresolved assumption record.
#[must_use]
pub fn unresolved_assumption_obligations(records: &[ObligationRecord]) -> Vec<EvidenceObligation> {
    records.iter().filter_map(EvidenceObligation::from_unresolved_record).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: EvidenceObligationKind) -> EvidenceObligationSpec {
        EvidenceObligationSpec {
            quantities: std::collections::BTreeMap::new(),
            kind,
            scope: ObligationScope::Factor,
            variables: Arc::from([VariableId::from_raw(2), VariableId::from_raw(1)]),
            population: Some(Arc::from("target")),
            regime: ObligationRegime::observational(true),
            reason: Arc::from("owed"),
            required_slots: Arc::from([Arc::from("factor:f")]),
            min_additional_samples: None,
            provenance: ObligationProvenance {
                family: Arc::from("transport"),
                source: Arc::from("contract"),
                proof_step: Some(Arc::from("leaf:1")),
            },
        }
    }

    #[test]
    fn names_round_trip() {
        for kind in EvidenceObligationKind::ALL {
            assert_eq!(EvidenceObligationKind::from_name(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn id_ignores_variable_order() {
        let a = EvidenceObligation::try_new(spec(EvidenceObligationKind::ProvideJointLaw)).unwrap();
        let mut swapped = spec(EvidenceObligationKind::ProvideJointLaw);
        swapped.variables = Arc::from([VariableId::from_raw(1), VariableId::from_raw(2)]);
        let b = EvidenceObligation::try_new(swapped).unwrap();
        assert_eq!(a.id, b.id);
    }
}
