//! The bridge contract: what is bridged, under which declared assumptions, by
//! which mapping, for which population and regime, and with which
//! transportability premise.
//!
//! [`BridgeContract::check_licensing`] runs the checks in a fixed order, so the
//! refusal a request receives is the first defect it has:
//!
//! 1. the declaration is well formed;
//! 2. the class pair is one the kind connects, and the two closed pairs
//!    (a predictive importance as a causal effect, a model score as a
//!    probability) refuse under their own names;
//! 3. the value kinds agree and are a mean;
//! 4. the kind's principal assumption is declared (its absence is the named
//!    refusal of the class pair: an association offered as an intervention, a
//!    proxy as a construct, a stated preference as realized behavior, a
//!    scientific endpoint as a utility);
//! 5. the mapping is one the kind allows and names its evidence;
//! 6. the mapping's own binding (the measurement model's variables, identity
//!    and population) agrees with the contract;
//! 7. the source and target agree in population, regime (where it may not
//!    change), horizon and conditioning. An explicit experimental estimand
//!    translation may change population/horizon under its target transport premise;
//! 8. a bridge whose evidence was established for another population or regime
//!    than the claim's carries a transportability premise for exactly that
//!    transport;
//! 9. an estimand bridge, when one is declared, names the evidence scope and
//!    the target quantity on population and exposure regime. Mapping
//!    uncertainty, source dependence and the estimand translation are part of
//!    the declaration checked in step 1, and part of the identity when any of
//!    them is present.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};

use super::kind::{AssumptionKind, BridgeKind, QuantityClass, ValueKind};
use super::mapping::{BridgeMapping, MeasurementDirection};
use super::refusal::{
    BridgeRefusal, assumption_evidence_missing, blank_field, conditioning_mismatch,
    duplicate_assumption, endpoint_classes, functional_not_supported, horizon_mismatch,
    invalid_quantity, mapping_not_allowed, mapping_scope_mismatch, measurement_model_mismatch,
    missing_principal, model_identity_mismatch, model_score_as_probability, population_mismatch,
    predictive_importance_as_causal_effect, proxy_as_construct, regime_mismatch,
    scientific_endpoint_as_utility, transport_premise_mismatch,
    transport_premise_missing, transport_premise_unused, value_kind_not_carried, wrong_value_kind,
};
use crate::ScientificQuantity;

#[path = "draws.rs"]
mod draws;
#[path = "propagation.rs"]
mod propagation;
#[path = "point_propagation.rs"]
mod point_propagation;

use crate::interface_contract::{
    CalibrationStatus, ContractOperation, DependenceRequirement, EvidenceBasis, InterfaceContract,
    PortFacts,
};
pub use draws::{
    AffineDrawLaw, AffineInnovation, BridgeDrawsSource, BridgedDistributionClaim,
    CorrectionDrawLaw, DrawMapping, DrawsPlan, MappedDrawsSummary, bridged_distribution_claim_id,
};
pub use point_propagation::{
    NonlinearPropagation, POINT_PROPAGATION_CALIBRATION, POINT_PROPAGATION_LABEL, PointPropagation,
    point_propagation_unavailable,
};
pub(super) use point_propagation::propagate_point;
pub use propagation::{
    AnalyticLaw, EstimandBridge, ExperimentalPrior, ExposureSetting, ImportedExperimentalPrior,
    MappingUncertainty, NonlinearMap, NonlinearMoment, PreservedSupport, PropagatedUncertainty,
    SourceMappingDependence, SupportMap, ValueInterval, nonlinear_expectation, preserved_support,
    support_image,
};
pub use propagation::{
    ContractNonlinearPropagation, JointLaw, JointPropagation, PropagationMethod, propagate_joint,
    propagate_nonlinear,
};

/// Version word written into every bridge contract identity. Bump only when
/// the canonical encoding below changes.
pub const BRIDGE_IDENTITY_VERSION: u64 = 1;
/// Default bound on declared assumptions per contract that a decoder of untrusted input applies.
/// Validating a contract is linear in its assumptions, so the contract itself carries no bound.
pub const MAX_BRIDGE_ASSUMPTIONS: usize = 64;

const TAG_CONTRACT: &str = "antecedent.scientific_bridge.contract";

/// Name of the consumed port of a bridge's interface contract.
pub const BRIDGE_SOURCE_PORT: &str = "source";
/// Name of the produced port of a bridge's interface contract.
pub const BRIDGE_TARGET_PORT: &str = "target";

/// What a bridge declaration does not state but its interface contract needs:
/// the support, estimand, uncertainty meaning and measurement models of the
/// two ends, the calibration the target holds and the evidence it rests on.
///
/// The operations of the two ports come from the declared value kinds. A
/// value kind that names no operation (an identified set, a joint outcome
/// law) projects to a port with none and is refused by the connection check
/// as a port with no operations, as the licensing check refuses it as not
/// carried.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeInterfaceBasis {
    /// Facts of the source quantity the bridge consumes.
    pub source: PortFacts,
    /// Facts of the target quantity the bridge produces.
    pub target: PortFacts,
    /// The calibration the target holds; `None` is unmeasured when the target
    /// value carries uncertainty and point-only otherwise. A bridge never
    /// raises a calibration the source did not hold.
    pub calibration: Option<CalibrationStatus>,
    /// The evidence the target rests on (the source evidence and the
    /// bridge's own).
    pub evidence: EvidenceBasis,
}

const fn value_operation(kind: ValueKind) -> Option<ContractOperation> {
    match kind {
        ValueKind::Mean => Some(ContractOperation::Mean),
        ValueKind::PredictiveLaw => Some(ContractOperation::Law),
        ValueKind::Posterior => Some(ContractOperation::Posterior),
        ValueKind::IdentifiedSet | ValueKind::JointOutcomeLaw => None,
    }
}

/// A population and regime.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct BridgeScope {
    /// Stable population identity.
    pub population_id: String,
    /// Regime identity; `observational` is explicit.
    pub regime_id: String,
}

/// One end of a bridge: a scientific quantity and what class of thing it is.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BridgeEndpoint {
    /// The full semantic coordinate of the quantity.
    pub quantity: ScientificQuantity,
    /// What the quantity is.
    pub class: QuantityClass,
}

/// One declared assumption of a bridge, with its own identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BridgeAssumption {
    /// Stable identity, unique within the contract.
    pub id: String,
    /// The statement the assumption makes.
    pub kind: AssumptionKind,
    /// What is assumed, in words.
    pub statement: String,
    /// The study, model, identification or utility function it rests on.
    pub evidence: Option<String>,
}

impl BridgeAssumption {
    /// The measurement-model assumption of a bridge through `model`: it names
    /// the model's own six-part identity as its evidence, so the bridge's
    /// licensing check can require that the mapping uses exactly that model.
    #[must_use]
    pub fn for_measurement_model(
        id: impl Into<String>,
        model: &antecedent_core::measurement_model::MeasurementModel,
    ) -> Self {
        Self {
            id: id.into(),
            kind: AssumptionKind::MeasurementModel,
            statement: format!(
                "recorded `{}` relates to construct `{}` by the declared {} measurement model",
                model.indicator.id,
                model.construct.id,
                model.route.as_str()
            ),
            evidence: Some(model.identities().model.to_hex()),
        }
    }
}

/// The premise under which a relation established in one population and
/// regime is carried to another.
///
/// It is a declaration about the relation the bridge uses: that it is the same
/// in the scope the claim is about as in the scope it was established for. It
/// is not a check; the derived claim carries it as an unresolved obligation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportabilityPremise {
    /// Stable identity.
    pub id: String,
    /// What is assumed, in words.
    pub statement: String,
    /// Where the bridge's evidence was established.
    pub established_in: BridgeScope,
    /// Where the claim being bridged is about.
    pub applies_to: BridgeScope,
    /// The invariance study, selection diagram or argument it rests on.
    pub evidence: Option<String>,
}

/// An explicit statement of how one scientific quantity answers a question
/// about another.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeContract {
    /// Which of the six bridges this is.
    pub kind: BridgeKind,
    /// The quantity the source claim is about.
    pub source: BridgeEndpoint,
    /// The quantity the derived claim is about.
    pub target: BridgeEndpoint,
    /// What kind of object the source value is.
    pub source_value: ValueKind,
    /// What kind of object the consumer requires.
    pub target_value: ValueKind,
    /// The population and regime the bridge's own evidence (its validation
    /// trials, measurement model, linkage study) was established for.
    pub evidence_scope: BridgeScope,
    /// The mapping that carries the value.
    pub mapping: BridgeMapping,
    /// Declared assumptions, in any order.
    pub assumptions: Vec<BridgeAssumption>,
    /// Transportability premise, when the claim is not about the scope the
    /// evidence was established for.
    pub transport: Option<TransportabilityPremise>,
    /// Variance of a declared affine map. Absent when the map's uncertainty
    /// is not part of this contract.
    pub mapping_uncertainty: Option<MappingUncertainty>,
    /// Dependence of the source estimate and the mapping. Absent is not a
    /// declaration of independence.
    pub source_dependence: Option<SourceMappingDependence>,
    /// Translation of an experiment into this contract's target, when the
    /// bridge uses one.
    pub estimand_bridge: Option<EstimandBridge>,
}

/// What a licensed contract determined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Licensing {
    /// The bridge carries evidence established elsewhere than the claim's
    /// population or regime, under a declared premise.
    pub transported: bool,
    /// Direction of a measurement correction (unused by other mappings).
    pub direction: MeasurementDirection,
}

fn check_text(field: &str, text: &str) -> Result<(), BridgeRefusal> {
    if text.trim().is_empty() {
        return Err(blank_field(field));
    }
    Ok(())
}

fn check_scope(prefix: &str, scope: &BridgeScope) -> Result<(), BridgeRefusal> {
    check_text(&format!("{prefix}.population_id"), &scope.population_id)?;
    check_text(&format!("{prefix}.regime_id"), &scope.regime_id)
}

impl BridgeContract {
    /// Check the declaration is well formed: quantities, identities,
    /// statements, unique assumption ids, the mapping, the premise and, when
    /// declared, the mapping uncertainty, the source dependence and the
    /// estimand bridge.
    ///
    /// # Errors
    /// A malformed declaration, as a `scientific_bridge_invalid` refusal.
    pub fn validate(&self) -> Result<(), BridgeRefusal> {
        for (name, endpoint) in [("source", &self.source), ("target", &self.target)] {
            endpoint.quantity.validate().map_err(|_| invalid_quantity(name))?;
        }
        check_scope("evidence_scope", &self.evidence_scope)?;
        let mut seen = BTreeSet::new();
        for assumption in &self.assumptions {
            check_text("assumption.id", &assumption.id)?;
            check_text("assumption.statement", &assumption.statement)?;
            if let Some(evidence) = &assumption.evidence {
                check_text("assumption.evidence", evidence)?;
            }
            if !seen.insert(assumption.id.as_str()) {
                return Err(duplicate_assumption(&assumption.id));
            }
        }
        if let Some(premise) = &self.transport {
            check_text("transport.id", &premise.id)?;
            check_text("transport.statement", &premise.statement)?;
            check_scope("transport.established_in", &premise.established_in)?;
            check_scope("transport.applies_to", &premise.applies_to)?;
            if let Some(evidence) = &premise.evidence {
                check_text("transport.evidence", evidence)?;
            }
        }
        self.mapping.validate()?;
        if let Some(uncertainty) = &self.mapping_uncertainty {
            uncertainty.validate()?;
        }
        if let Some(dependence) = &self.source_dependence {
            dependence.validate()?;
        }
        if let Some(bridge) = &self.estimand_bridge {
            bridge.validate()?;
        }
        Ok(())
    }

    /// Check the contract licenses its bridge, without a source claim.
    ///
    /// # Errors
    /// The first defect, in the order described in the module documentation,
    /// as a coded refusal with the declaration that would license it.
    pub fn check_licensing(&self) -> Result<Licensing, BridgeRefusal> {
        self.check_licensing_carrying(false)
    }

    /// Check the contract licenses its bridge when the source value is a
    /// distribution carried as aligned draws.
    ///
    /// The checks are those of [`Self::check_licensing`] in the same order,
    /// except that the value kinds may be a posterior or a predictive law (and
    /// then must agree), and that the declaration of the mapping's uncertainty
    /// and of its dependence on the source is checked for the draws it
    /// propagates: an uncertain map declares how it depends on the source
    /// estimate, a trial-level line or a measurement correction does not
    /// declare its uncertainty a second time, and an identified set or a joint
    /// outcome law is still not carried.
    ///
    /// # Errors
    /// As [`Self::check_licensing`], and the refusals of the draws declaration.
    pub fn check_draws_licensing(&self) -> Result<Licensing, BridgeRefusal> {
        let licensing = self.check_licensing_carrying(true)?;
        draws::check_draws_declaration(self)?;
        Ok(licensing)
    }

    fn check_licensing_carrying(&self, draws: bool) -> Result<Licensing, BridgeRefusal> {
        self.validate()?;
        let pair = (self.source.class, self.target.class);
        if !self.kind.endpoint_pairs().contains(&pair) {
            return Err(endpoint_classes(self.kind.as_str(), pair.0, pair.1));
        }
        match pair {
            (QuantityClass::PredictiveImportance, QuantityClass::CausalQuantity) => {
                return Err(predictive_importance_as_causal_effect());
            }
            (QuantityClass::ModelScore, QuantityClass::PredictiveQuantity) => {
                return Err(model_score_as_probability());
            }
            _ => {}
        }
        if self.source_value != self.target_value {
            return Err(wrong_value_kind(self.target_value, self.source_value));
        }
        let carried = self.source_value == ValueKind::Mean
            || (draws
                && matches!(self.source_value, ValueKind::Posterior | ValueKind::PredictiveLaw));
        if !carried {
            return Err(value_kind_not_carried(self.source_value));
        }
        let principal_kind = self.kind.principal_assumption();
        let Some(principal) = self.assumptions.iter().find(|a| a.kind == principal_kind) else {
            return Err(missing_principal(pair.0, pair.1, principal_kind));
        };
        if matches!(self.mapping, BridgeMapping::Identity) {
            match pair {
                (QuantityClass::MeasuredOutcome, QuantityClass::LatentConstruct) => {
                    return Err(proxy_as_construct());
                }
                (QuantityClass::ScientificEndpoint, QuantityClass::DecisionUtility) => {
                    return Err(scientific_endpoint_as_utility());
                }
                _ => {}
            }
        }
        if !self.kind.allowed_mappings().contains(&self.mapping.kind()) {
            return Err(mapping_not_allowed(self.mapping.kind().as_str(), self.kind.as_str()));
        }
        if principal_kind.requires_evidence() && principal.evidence.is_none() {
            return Err(assumption_evidence_missing(&principal.id, principal_kind));
        }
        self.check_mapping_binding(principal)?;
        self.mapping.check_ready()?;
        self.check_coordinates()?;
        let transported = self.check_transport()?;
        self.check_estimand_binding()?;
        let direction = if self.source.class == QuantityClass::LatentConstruct {
            MeasurementDirection::LatentToMeasured
        } else {
            MeasurementDirection::MeasuredToLatent
        };
        Ok(Licensing { transported, direction })
    }

    fn check_mapping_binding(&self, principal: &BridgeAssumption) -> Result<(), BridgeRefusal> {
        let BridgeMapping::MeasurementCorrection(model) = &self.mapping else {
            return Ok(());
        };
        let (latent, measured) = if self.source.class == QuantityClass::LatentConstruct {
            (&self.source, &self.target)
        } else {
            (&self.target, &self.source)
        };
        if model.construct.id != latent.quantity.variable_id {
            return Err(measurement_model_mismatch(
                "construct",
                model.construct.id.clone(),
                latent.quantity.variable_id.clone(),
            ));
        }
        if model.indicator.id != measured.quantity.variable_id {
            return Err(measurement_model_mismatch(
                "indicator",
                model.indicator.id.clone(),
                measured.quantity.variable_id.clone(),
            ));
        }
        let model_identity = model.identities().model.to_hex();
        let evidence = principal.evidence.as_deref().unwrap_or_default();
        if evidence != model_identity {
            return Err(model_identity_mismatch(model_identity, evidence));
        }
        if model.binding.population_id != self.evidence_scope.population_id
            || model.binding.regime_id != self.evidence_scope.regime_id
        {
            return Err(mapping_scope_mismatch(
                format!("{}/{}", model.binding.population_id, model.binding.regime_id),
                format!("{}/{}", self.evidence_scope.population_id, self.evidence_scope.regime_id),
            ));
        }
        for endpoint in [&self.source, &self.target] {
            let functional = endpoint.quantity.functional_id.as_str();
            if !matches!(functional, "prevalence" | "risk")
                || endpoint.quantity.transform_id != "identity"
            {
                return Err(functional_not_supported(format!(
                    "{functional} on the {} scale",
                    endpoint.quantity.transform_id
                )));
            }
        }
        Ok(())
    }

    fn check_coordinates(&self) -> Result<(), BridgeRefusal> {
        let (source, target) = (&self.source.quantity, &self.target.quantity);
        if source.population_id != target.population_id
            && !self.estimand_bridge.as_ref().is_some_and(|translation| {
                self.kind == BridgeKind::ParameterInterventionResponse
                    && translation.experiment.population_id == source.population_id
                    && translation.target.population_id == target.population_id
            })
        {
            return Err(population_mismatch(
                source.population_id.clone(),
                target.population_id.clone(),
            ));
        }
        if !self.kind.regime_may_change() && source.regime_id != target.regime_id {
            return Err(regime_mismatch(source.regime_id.clone(), target.regime_id.clone()));
        }
        if source.horizon != target.horizon
            && !(self.kind == BridgeKind::ParameterInterventionResponse
                && self.estimand_bridge.is_some())
        {
            return Err(horizon_mismatch(source.horizon.to_string(), target.horizon.to_string()));
        }
        if source.conditioning != target.conditioning {
            return Err(conditioning_mismatch(
                source.canonical_identity(),
                target.canonical_identity(),
            ));
        }
        Ok(())
    }

    fn check_transport(&self) -> Result<bool, BridgeRefusal> {
        // An explicit experimental translation maps the source experimental
        // estimand into the target response. Its transport premise therefore
        // applies to the target scope, including a changed time horizon. The
        // ordinary six bridge families retain their original coordinate checks.
        let experimental_endpoint_change = self.kind == BridgeKind::ParameterInterventionResponse
            && self.estimand_bridge.is_some()
            && (self.source.quantity.population_id != self.target.quantity.population_id
                || self.source.quantity.horizon != self.target.quantity.horizon);
        let claim = if experimental_endpoint_change {
            &self.target.quantity
        } else {
            &self.source.quantity
        };
        let claim_scope = BridgeScope {
            population_id: claim.population_id.clone(),
            regime_id: claim.regime_id.clone(),
        };
        let moved = claim_scope != self.evidence_scope || experimental_endpoint_change;
        let described = format!(
            "the {} relation established in population `{}` under regime `{}` holds in population `{}` under regime `{}`",
            self.kind.as_str(),
            self.evidence_scope.population_id,
            self.evidence_scope.regime_id,
            claim_scope.population_id,
            claim_scope.regime_id,
        );
        match (&self.transport, moved) {
            (None, false) => Ok(false),
            (None, true) => Err(transport_premise_missing(described)),
            (Some(premise), true) => {
                if premise.established_in == self.evidence_scope
                    && premise.applies_to == claim_scope
                {
                    Ok(true)
                } else {
                    Err(transport_premise_mismatch(described))
                }
            }
            (Some(_), false) => Err(transport_premise_unused()),
        }
    }

    fn check_estimand_binding(&self) -> Result<(), BridgeRefusal> {
        if let Some(bridge) = &self.estimand_bridge {
            propagation::estimand_matches(self, bridge)?;
        }
        Ok(())
    }

    /// Check an explicitly finite joint law of affine mapping and source draws.
    ///
    /// Every original slot is retained; the declared finite-law moments and dependence
    /// must agree. This check grants no posterior or causal permission beyond licensing.
    /// # Errors
    /// Native licensing, missing/failed slots, unsupported mapping or conflicting moments.
    pub fn validate_finite_affine_mapping(
        &self,
        intercepts: &[f64],
        slopes: &[f64],
        source: &[f64],
    ) -> Result<(), BridgeRefusal> {
        self.check_licensing()?;
        propagation::validate_finite_affine_mapping(self, intercepts, slopes, source)
    }

    /// Delta-method mean and variance of this contract's affine map.
    ///
    /// # Errors
    /// The mapping is not affine or identity; mapping uncertainty or source
    /// dependence is missing or not well formed; the source moments are not
    /// usable; the covariance is not feasible.
    pub fn propagate_affine_uncertainty(
        &self,
        source_mean: f64,
        source_variance: f64,
    ) -> Result<PropagatedUncertainty, BridgeRefusal> {
        self.check_licensing()?;
        propagation::propagate_affine(self, source_mean, source_variance)
    }

    /// Image of `source` under `map`, with this contract's assumptions attached
    /// in identity order.
    ///
    /// # Errors
    /// The source interval or the map does not determine a finite image.
    pub fn map_support(
        &self,
        map: SupportMap,
        source: ValueInterval,
    ) -> Result<PreservedSupport, BridgeRefusal> {
        self.check_licensing()?;
        preserved_support(self, map, source)
    }

    /// Store an experimental prior's estimand bridge, and its mapping
    /// uncertainty and source dependence when the prior declares them.
    ///
    /// A prior with no estimand bridge is refused and leaves the contract
    /// unchanged. Uncertainty or dependence already on the contract is kept
    /// when the prior does not declare its own.
    ///
    /// # Errors
    /// The prior has no estimand bridge, the translation is not well formed,
    /// or it does not name this contract's evidence scope and target.
    pub fn import_experimental_prior(
        &mut self,
        prior: &ExperimentalPrior,
    ) -> Result<ImportedExperimentalPrior, BridgeRefusal> {
        let imported = propagation::import_prior(self, prior)?;
        let mut candidate = self.clone();
        candidate.estimand_bridge = Some(imported.estimand_bridge.clone());
        if imported.mapping_uncertainty.is_some() {
            candidate.mapping_uncertainty.clone_from(&imported.mapping_uncertainty);
        }
        if imported.source_dependence.is_some() {
            candidate.source_dependence.clone_from(&imported.source_dependence);
        }
        candidate.check_licensing()?;
        *self = candidate;
        Ok(imported)
    }

    /// The bridge as one instance of the interface-contract schema: the source
    /// quantity as the consumed port and the target quantity as the produced
    /// port, so the one connection check decides what feeds and what is fed by
    /// a bridge.
    ///
    /// Licensing stays [`Self::check_licensing`] (the principal assumption,
    /// the mapping, the transportability premise); the projection states only
    /// the two ports and licenses nothing a connection does not.
    #[must_use]
    pub fn interface_contract(
        &self,
        component: impl Into<String>,
        basis: &BridgeInterfaceBasis,
    ) -> InterfaceContract {
        let consumed_ops: std::collections::BTreeSet<ContractOperation> =
            value_operation(self.source_value).into_iter().collect();
        let produced_ops: std::collections::BTreeSet<ContractOperation> =
            value_operation(self.target_value).into_iter().collect();
        let consumed = basis.source.consumed(
            BRIDGE_SOURCE_PORT,
            &self.source.quantity,
            consumed_ops,
            DependenceRequirement::Unconstrained,
        );
        let produced = basis.target.produced(
            BRIDGE_TARGET_PORT,
            &self.target.quantity,
            produced_ops,
            ["transformation"],
            basis.calibration.clone(),
            basis.evidence.clone(),
        );
        InterfaceContract {
            component: component.into(),
            produces: vec![produced],
            consumes: vec![consumed],
        }
    }

    /// The identity of the contract: kind, both quantities and classes, value
    /// kinds, evidence scope, mapping (every number as its bit pattern),
    /// assumptions (sorted by identity, so listing order is not identity),
    /// the transportability premise and, when any is declared, the mapping
    /// uncertainty, the source dependence and the estimand bridge. A contract
    /// that declares none of those three keeps the version-1 byte stream.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut out = CanonicalWriter::new(TAG_CONTRACT);
        out.word(BRIDGE_IDENTITY_VERSION).word(self.kind.tag());
        for endpoint in [&self.source, &self.target] {
            out.word(endpoint.class.tag());
            endpoint.quantity.write_canonical(&mut out);
        }
        out.word(self.source_value.tag()).word(self.target_value.tag());
        out.text(&self.evidence_scope.population_id).text(&self.evidence_scope.regime_id);
        self.mapping.write_canonical(&mut out);
        let mut assumptions: Vec<&BridgeAssumption> = self.assumptions.iter().collect();
        assumptions.sort_by(|a, b| a.id.cmp(&b.id));
        out.word(assumptions.len() as u64);
        for assumption in assumptions {
            out.text(&assumption.id)
                .word(assumption.kind.tag())
                .text(&assumption.statement)
                .opt_text(assumption.evidence.as_deref());
        }
        match &self.transport {
            None => {
                out.word(0);
            }
            Some(premise) => {
                out.word(1)
                    .text(&premise.id)
                    .text(&premise.statement)
                    .text(&premise.established_in.population_id)
                    .text(&premise.established_in.regime_id)
                    .text(&premise.applies_to.population_id)
                    .text(&premise.applies_to.regime_id)
                    .opt_text(premise.evidence.as_deref());
            }
        }
        propagation::write_extension(
            &mut out,
            self.mapping_uncertainty.as_ref(),
            self.source_dependence.as_ref(),
            self.estimand_bridge.as_ref(),
        );
        SemanticDigest::from_bytes(out.finish())
    }
}
