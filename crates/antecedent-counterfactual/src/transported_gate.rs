//! The closed transported path-specific counterfactual route and its precise gates.
//!
//! The composition of a transport theorem and a fixed-population cross-world
//! theorem is **not** evaluated here: the joint theorem has not passed, so the
//! route stays closed. This module only reports exactly which prerequisite gate is
//! missing, retains the missing regime factors, and returns the live refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::StructuredRefusal;

/// A prerequisite of the transported counterfactual composition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TransportedCounterfactualGate {
    /// The ordinary transport license for the target functional.
    TransportLicenseMissing,
    /// The fixed-population cross-world counterfactual license for the same quantity.
    FixedPopulationTemporalLicenseMissing,
    /// The extra cross-population assumptions are not stated.
    CrossPopulationAssumptionsMissing,
}

impl TransportedCounterfactualGate {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::TransportLicenseMissing => "transport_license_missing",
            Self::FixedPopulationTemporalLicenseMissing => {
                "fixed_population_temporal_license_missing"
            }
            Self::CrossPopulationAssumptionsMissing => "cross_population_assumptions_missing",
        }
    }
}

/// Which population a regime factor belongs to.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PopulationRole {
    /// The source population.
    Source,
    /// The target population.
    Target,
}

/// A regime factor required by the composed proof.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RegimeFactorKey {
    /// Source or target.
    pub role: PopulationRole,
    /// Regime name.
    pub regime: String,
}

impl RegimeFactorKey {
    /// `source:<regime>` or `target:<regime>`.
    #[must_use]
    pub fn label(&self) -> String {
        let role = match self.role {
            PopulationRole::Source => "source",
            PopulationRole::Target => "target",
        };
        format!("{role}:{}", self.regime)
    }
}

/// Which prerequisites have been passed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransportedCounterfactualPrerequisites {
    /// The ordinary transport license exists.
    pub transport_license: bool,
    /// The fixed-population cross-world license exists.
    pub fixed_population_license: bool,
    /// The extra cross-population assumptions are stated.
    pub cross_population_assumptions: bool,
}

impl TransportedCounterfactualPrerequisites {
    /// The missing gates, in a fixed order.
    #[must_use]
    pub fn missing_gates(&self) -> Vec<TransportedCounterfactualGate> {
        let mut gates = Vec::new();
        if !self.transport_license {
            gates.push(TransportedCounterfactualGate::TransportLicenseMissing);
        }
        if !self.fixed_population_license {
            gates.push(TransportedCounterfactualGate::FixedPopulationTemporalLicenseMissing);
        }
        if !self.cross_population_assumptions {
            gates.push(TransportedCounterfactualGate::CrossPopulationAssumptionsMissing);
        }
        gates
    }
}

/// The live refusal of the closed transported-counterfactual route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportedCounterfactualRefusal {
    /// Registered code, stage and detail.
    pub refusal: StructuredRefusal,
    /// Every missing prerequisite gate.
    pub missing_gates: Vec<TransportedCounterfactualGate>,
    /// Every required regime factor absent from the supplied evidence.
    pub missing_factors: Vec<RegimeFactorKey>,
}

/// Refuse the transported path-specific counterfactual route.
///
/// The route is closed whatever is supplied. The refusal is
/// `transported_counterfactual.route_frozen` (`cell_not_licensed`) when any
/// prerequisite gate is missing, and also when all gates pass but every required
/// regime factor is present (the joint theorem itself has not passed). Only when
/// all gates pass and a required source or target regime factor is absent from
/// `supplied` is it `transported_counterfactual.factor_missing`
/// (`transport_missing_evidence`). Both lists are retained either way.
#[must_use]
pub fn refuse_transported_counterfactual(
    prerequisites: &TransportedCounterfactualPrerequisites,
    required_factors: &[RegimeFactorKey],
    supplied: &BTreeMap<RegimeFactorKey, String>,
) -> TransportedCounterfactualRefusal {
    let missing_gates = prerequisites.missing_gates();
    let mut missing_factors: Vec<RegimeFactorKey> =
        required_factors.iter().filter(|k| !supplied.contains_key(*k)).cloned().collect();
    missing_factors.sort();
    missing_factors.dedup();
    let (code, detail, offending, remedy) =
        if missing_gates.is_empty() && !missing_factors.is_empty() {
            (
                antecedent_core::reason_code!("transport_missing_evidence"),
                "transported_counterfactual.factor_missing",
                missing_factors.first().map(RegimeFactorKey::label),
                "supply every source and target regime factor cited by the composed proof",
            )
        } else {
            (
                antecedent_core::reason_code!("cell_not_licensed"),
                "transported_counterfactual.route_frozen",
                missing_gates.first().map(|g| g.name().to_owned()),
                "pass the transport and fixed-population counterfactual licenses and state the \
             cross-population assumptions; the joint theorem is still required",
            )
        };
    TransportedCounterfactualRefusal {
        refusal: StructuredRefusal {
            code,
            stage: "identify",
            detail: detail.to_owned(),
            offending,
            expected: None,
            supplied: None,
            capability: None,
            remedy: Some(remedy),
        },
        missing_gates,
        missing_factors,
    }
}
