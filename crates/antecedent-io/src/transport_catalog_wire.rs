//! Durable supplied-evidence catalogs, including measurement and sampling identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{IoError, query_wire::ValueWire};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceProjection, EvidenceRegime, InterventionAssignment, LicensedWeights, QueryError,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, TargetSampling, VariableCoordinate,
    VariableDomain, VariableId,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Durable catalog. Each regime remains separate across serialization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceCatalogWire {
    /// Named environments and shared coordinates.
    pub environments: Vec<EnvironmentWire>,
    /// Distinct available, manipulable, or proposed regimes.
    pub regimes: Vec<EvidenceRegimeWire>,
    /// Snapshot bindings.
    pub bindings: Vec<RegimeBindingWire>,
    /// Target sampling contract.
    pub target_sampling: Option<String>,
}

/// Population coordinates.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentWire {
    /// Stable population key.
    pub identity: String,
    /// Variable id, domain tag, categorical cardinality, and physical unit.
    pub variables: Vec<(u32, String, Option<u32>, Option<String>)>,
    /// Mechanisms allowed to differ.
    pub selection_targets: Vec<u32>,
}

/// One supplied evidence regime.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceRegimeWire {
    /// Stable regime id.
    pub id: u32,
    /// External regime name, when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Observational or experimental.
    pub kind: String,
    /// Available, manipulable, or proposed.
    pub evidence_kind: String,
    /// Exact intervention set.
    pub interventions: Vec<u32>,
    /// Available concrete assignments.
    pub intervention_values: Vec<(u32, ValueWire)>,
    /// Measured coordinates.
    pub measured: Vec<u32>,
    /// Already conditioned coordinates.
    #[serde(default)]
    pub conditioned_on: Vec<u32>,
    /// Population key.
    pub population: String,
    /// None denotes a joint law; Some lists separately available marginals.
    pub separate_marginals: Option<Vec<u32>>,
}

/// Concrete table provenance.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RegimeBindingWire {
    /// Optional stable identity shared by forwarded aliases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_identity: Option<String>,
    /// Regime id.
    pub regime: u32,
    /// Dataset identity.
    pub snapshot_identity: String,
    /// Ordered schema names.
    pub schema_names: Vec<String>,
    /// Sampling design tag.
    pub sampling: String,
    /// Snapshot identity licensed for weights.
    pub weights_snapshot: Option<String>,
    /// Dependence relationship.
    pub dependence: String,
}

impl EvidenceCatalogWire {
    /// Encode all catalog semantics without flattening experiments or measurements.
    #[must_use]
    pub fn from_catalog(catalog: &EvidenceCatalog) -> Self {
        let ids = |v: &[VariableId]| v.iter().map(|v| v.raw()).collect::<Vec<_>>();
        Self {
            environments: catalog
                .environments
                .iter()
                .map(|e| EnvironmentWire {
                    identity: e.identity.to_string(),
                    selection_targets: ids(&e.selection_targets),
                    variables: e
                        .variables
                        .iter()
                        .map(|c| {
                            let (domain, cardinality) = match c.domain {
                                VariableDomain::Unspecified => ("unspecified", None),
                                VariableDomain::Continuous => ("continuous", None),
                                VariableDomain::Binary => ("binary", None),
                                VariableDomain::Count => ("count", None),
                                VariableDomain::Categorical { cardinality } => {
                                    ("categorical", Some(cardinality))
                                }
                            };
                            (
                                c.variable.raw(),
                                domain.into(),
                                cardinality,
                                c.unit.as_ref().map(ToString::to_string),
                            )
                        })
                        .collect(),
                })
                .collect(),
            regimes: catalog
                .regimes
                .iter()
                .map(|r| EvidenceRegimeWire {
                    id: r.id.raw(),
                    label: r.label.as_ref().map(ToString::to_string),
                    kind: r.kind.as_str().into(),
                    evidence_kind: r.evidence_kind.as_str().into(),
                    interventions: ids(&r.interventions),
                    measured: ids(&r.measured),
                    conditioned_on: ids(&r.conditioned_on),
                    population: r.population.to_string(),
                    intervention_values: r
                        .intervention_values
                        .iter()
                        .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
                        .collect(),
                    separate_marginals: match &r.distribution {
                        DistributionAvailability::Joint => None,
                        DistributionAvailability::SeparateMarginals { variables } => {
                            Some(ids(variables))
                        }
                    },
                })
                .collect(),
            bindings: catalog
                .bindings
                .iter()
                .map(|b| RegimeBindingWire {
                    dataset_identity: b.dataset_identity.as_ref().map(ToString::to_string),
                    regime: b.regime.raw(),
                    snapshot_identity: b.snapshot_identity.to_string(),
                    schema_names: b.schema_names.iter().map(ToString::to_string).collect(),
                    sampling: b.sampling.as_str().into(),
                    dependence: b.dependence.as_str().into(),
                    weights_snapshot: b.weights.as_ref().map(|w| w.snapshot_identity.to_string()),
                })
                .collect(),
            target_sampling: catalog.target_sampling.map(|s| s.as_str().to_owned()),
        }
    }

    /// Decode and validate all contracts.
    ///
    /// # Errors
    /// Invalid tags, coordinates, assignments, or catalog dependencies.
    #[allow(clippy::too_many_lines)] // Explicit exhaustive decoding of the durable schema.
    pub fn to_catalog(&self) -> Result<EvidenceCatalog, IoError> {
        let invalid = || IoError::Convert("invalid transport catalog tag or domain".into());
        let ids = |v: &[u32]| v.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let mut environments = Vec::new();
        for e in &self.environments {
            let mut variables = Vec::new();
            for (id, domain, cardinality, unit) in &e.variables {
                let domain = match (domain.as_str(), cardinality) {
                    ("unspecified", None) => VariableDomain::Unspecified,
                    ("continuous", None) => VariableDomain::Continuous,
                    ("binary", None) => VariableDomain::Binary,
                    ("count", None) => VariableDomain::Count,
                    ("categorical", Some(n)) => VariableDomain::Categorical { cardinality: *n },
                    _ => return Err(invalid()),
                };
                variables.push(VariableCoordinate {
                    variable: VariableId::from_raw(*id),
                    domain,
                    unit: unit.as_deref().map(Arc::from),
                });
            }
            environments.push(
                Environment::try_new(e.identity.as_str(), variables, ids(&e.selection_targets))
                    .map_err(convert)?,
            );
        }
        let mut regimes = Vec::new();
        for r in &self.regimes {
            let kind = match r.kind.as_str() {
                "observational" => RegimeKind::Observational,
                "experimental" => RegimeKind::Experimental,
                _ => return Err(invalid()),
            };
            let evidence_kind = match r.evidence_kind.as_str() {
                "available" => EvidenceKind::Available,
                "manipulable" => EvidenceKind::Manipulable,
                "proposed" => EvidenceKind::Proposed,
                _ => return Err(invalid()),
            };
            let distribution =
                r.separate_marginals.as_ref().map_or(DistributionAvailability::Joint, |v| {
                    DistributionAvailability::SeparateMarginals { variables: ids(v).into() }
                });
            let assignments = r
                .intervention_values
                .iter()
                .map(|(v, x)| InterventionAssignment {
                    variable: VariableId::from_raw(*v),
                    value: x.to_value(),
                })
                .collect::<Vec<_>>();
            let mut regime = EvidenceRegime::try_new(
                RegimeId::from_raw(r.id),
                kind,
                evidence_kind,
                ids(&r.interventions),
                assignments,
                ids(&r.measured),
                r.population.as_str(),
                distribution,
            )
            .map_err(convert)?;
            regime.label = r.label.as_deref().map(Arc::from);
            regime = regime
                .project(EvidenceProjection::Condition { on: ids(&r.conditioned_on).into() })
                .map_err(convert)?;
            regimes.push(regime);
        }
        let mut bindings = Vec::new();
        for b in &self.bindings {
            let sampling = match b.sampling.as_str() {
                "independent" => SamplingDesign::Independent,
                "clustered" => SamplingDesign::Clustered,
                "unknown" => SamplingDesign::Unknown,
                _ => return Err(invalid()),
            };
            let dependence = match b.dependence.as_str() {
                "independent_studies" => DependenceGroup::IndependentStudies,
                "linked_units" => DependenceGroup::LinkedUnits,
                "unknown_dependence" => DependenceGroup::UnknownDependence,
                _ => return Err(invalid()),
            };
            bindings.push(RegimeBinding {
                dataset_identity: b.dataset_identity.as_deref().map(Arc::from),
                regime: RegimeId::from_raw(b.regime),
                snapshot_identity: Arc::from(b.snapshot_identity.as_str()),
                schema_names: b.schema_names.iter().map(|s| Arc::from(s.as_str())).collect(),
                sampling,
                dependence,
                weights: b
                    .weights_snapshot
                    .as_deref()
                    .map(|s| LicensedWeights { snapshot_identity: Arc::from(s) }),
            });
        }
        let target_sampling = match self.target_sampling.as_deref() {
            None => None,
            Some("supplied_population_law") => Some(TargetSampling::SuppliedPopulationLaw),
            Some("representative_sample") => Some(TargetSampling::RepresentativeSample),
            Some("licensed_weighted_design") => Some(TargetSampling::LicensedWeightedDesign),
            Some("convenience_sample") => Some(TargetSampling::ConvenienceSample),
            _ => return Err(invalid()),
        };
        EvidenceCatalog::try_new(environments, regimes, bindings, target_sampling).map_err(convert)
    }
}

#[allow(clippy::needless_pass_by_value)] // Result::map_err consumes its error.
fn convert(error: QueryError) -> IoError {
    IoError::Convert(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::Value;

    #[test]
    fn catalog_round_trip_preserves_restrictions_and_provenance() {
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(7),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(0)],
            [InterventionAssignment { variable: VariableId::from_raw(0), value: Value::f64(1.0) }],
            [VariableId::from_raw(1)],
            "trial",
            DistributionAvailability::SeparateMarginals {
                variables: Arc::from([VariableId::from_raw(1)]),
            },
        )
        .unwrap();
        let catalog = EvidenceCatalog::try_new(
            [],
            [regime],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(7),
                snapshot_identity: Arc::from("snapshot-1"),
                schema_names: Arc::from([Arc::from("a"), Arc::from("y")]),
                sampling: SamplingDesign::Clustered,
                weights: None,
                dependence: DependenceGroup::LinkedUnits,
            }],
            Some(TargetSampling::ConvenienceSample),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&EvidenceCatalogWire::from_catalog(&catalog)).unwrap();
        let wire: EvidenceCatalogWire = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(wire.to_catalog().unwrap(), catalog);
        let mut invalid = wire;
        invalid.regimes[0].intervention_values.push((0, ValueWire::Float64(0.0)));
        assert!(invalid.to_catalog().is_err());
    }
}
