//! Per-distribution descriptors of a supplied evidence catalog.
//!
//! A [`CatalogDistribution`] names everything a proof leaf may cite: the study and
//! population, jointly measured versus separately marginal variables, conditioning,
//! intervention variables and values, sampling selection, snapshot, evidence kind,
//! measured-versus-model origin, and dependence. Projections (marginalize, condition)
//! produce derived descriptors that always keep the catalog entry they came from,
//! so a leaf can be traced back to its original regime.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;
use std::sync::Arc;

use crate::ids::{RegimeId, VariableId};
use crate::value::Value;

use super::error::QueryError;
use super::transport_catalog::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceProjection,
    EvidenceRegime, InterventionAssignment, LawOrigin, RegimeKind, SamplingDesign,
    SamplingSelection, project_law,
};

/// One distribution a catalog supplies, or a licensed projection of one.
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogDistribution {
    /// The catalog entry this distribution is (or was projected from).
    pub source_regime: RegimeId,
    /// Declared study identity; `None` when only the population was supplied.
    pub study: Option<Arc<str>>,
    /// Population the law describes.
    pub population: Arc<str>,
    /// Observational or experimental.
    pub kind: RegimeKind,
    /// Whether results exist.
    pub evidence_kind: EvidenceKind,
    /// Measured law versus model artifact.
    pub origin: LawOrigin,
    /// Sampling selection of the law.
    pub selection: SamplingSelection,
    /// Hard intervention set.
    pub interventions: Arc<[VariableId]>,
    /// Concrete intervention values; empty when the domain is unrestricted.
    pub intervention_values: Arc<[InterventionAssignment]>,
    /// Measured variables after any projection.
    pub measured: Arc<[VariableId]>,
    /// Joint law versus separately supplied marginals.
    pub availability: DistributionAvailability,
    /// Conditioning variables after any projection.
    pub conditioned_on: Arc<[VariableId]>,
    /// Bound snapshot, when a dataset is bound to the source regime.
    pub snapshot: Option<Arc<str>>,
    /// Shared dataset identity of the bound snapshot, if declared.
    pub dataset: Option<Arc<str>>,
    /// Sampling design of the bound snapshot.
    pub sampling: Option<SamplingDesign>,
    /// Declared dependence of the bound snapshot.
    pub dependence: Option<DependenceGroup>,
    /// Licensed projections applied since the catalog entry, in order.
    pub projections: Arc<[EvidenceProjection]>,
}

/// Known relationship between the data behind two catalog distributions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SharedData {
    /// The same snapshot or declared dataset: sampling units are shared.
    SameDataset,
    /// Declared linked units across the two tables.
    LinkedUnits,
    /// Declared independent studies with distinct study identities.
    IndependentStudies,
    /// Nothing declared settles it; independence is never assumed.
    Unknown,
}

impl SharedData {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameDataset => "same_dataset",
            Self::LinkedUnits => "linked_units",
            Self::IndependentStudies => "independent_studies",
            Self::Unknown => "unknown",
        }
    }
}

impl CatalogDistribution {
    fn from_regime(catalog: &EvidenceCatalog, regime: &EvidenceRegime) -> Self {
        let binding = catalog.bindings.iter().find(|binding| binding.regime == regime.id);
        Self {
            source_regime: regime.id,
            study: regime.study.clone(),
            population: Arc::clone(&regime.population),
            kind: regime.kind,
            evidence_kind: regime.evidence_kind,
            origin: regime.origin.clone(),
            selection: regime.selection.clone(),
            interventions: Arc::clone(&regime.interventions),
            intervention_values: Arc::clone(&regime.intervention_values),
            measured: Arc::clone(&regime.measured),
            availability: regime.distribution.clone(),
            conditioned_on: Arc::clone(&regime.conditioned_on),
            snapshot: binding.map(|b| Arc::clone(&b.snapshot_identity)),
            dataset: binding.and_then(|b| b.dataset_identity.clone()),
            sampling: binding.map(|b| b.sampling),
            dependence: binding.map(|b| b.dependence),
            projections: Arc::from([]),
        }
    }

    /// Study identity, falling back to the population when none was declared.
    #[must_use]
    pub fn study_or_population(&self) -> &str {
        self.study.as_deref().unwrap_or(&self.population)
    }

    /// Whether this is a joint law over its measured variables.
    #[must_use]
    pub const fn is_joint(&self) -> bool {
        matches!(self.availability, DistributionAvailability::Joint)
    }

    /// Apply a licensed projection, recording it and keeping the source entry.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] exactly when
    /// [`EvidenceRegime::project`] would refuse the same projection.
    pub fn project(&self, projection: EvidenceProjection) -> Result<Self, QueryError> {
        let law =
            project_law(&self.measured, &self.conditioned_on, &self.availability, &projection)?;
        let mut out = self.clone();
        out.measured = law.measured;
        out.conditioned_on = law.conditioned_on;
        out.availability = law.distribution;
        out.projections = self.projections.iter().cloned().chain([projection]).collect();
        Ok(out)
    }

    /// Canonical, order-independent identity of this distribution.
    ///
    /// Variable sets are sorted, values are encoded bit-exactly, and the projection
    /// trail is kept in application order (projections do not commute in general).
    #[must_use]
    pub fn canonical_identity(&self) -> String {
        let mut out = String::from("catalog_distribution.v1");
        let mut field = |name: &str, value: &str| {
            let _ = write!(out, "|{name}={}", escape(value));
        };
        field("source_regime", &self.source_regime.raw().to_string());
        field("study", self.study.as_deref().unwrap_or(""));
        field("population", &self.population);
        field("kind", self.kind.as_str());
        field("evidence_kind", self.evidence_kind.as_str());
        field(
            "origin",
            &match &self.origin {
                LawOrigin::Measured => "measured".to_owned(),
                LawOrigin::ModelArtifact { artifact } => format!("model_artifact:{artifact}"),
            },
        );
        field(
            "selection",
            &match &self.selection {
                SamplingSelection::Population => "population".to_owned(),
                SamplingSelection::SelectedOn { variables } => {
                    format!("selected_on:{}", sorted(variables))
                }
            },
        );
        field("interventions", &sorted(&self.interventions));
        let mut values = self
            .intervention_values
            .iter()
            .map(|a| format!("{}:{}", a.variable.raw(), canonical_value(&a.value)))
            .collect::<Vec<_>>();
        values.sort_unstable();
        field("values", &values.join(","));
        field("measured", &sorted(&self.measured));
        field(
            "availability",
            &match &self.availability {
                DistributionAvailability::Joint => "joint".to_owned(),
                DistributionAvailability::SeparateMarginals { variables } => {
                    format!("marginals:{}", sorted(variables))
                }
            },
        );
        field("conditioned_on", &sorted(&self.conditioned_on));
        field("snapshot", self.snapshot.as_deref().unwrap_or(""));
        field("dataset", self.dataset.as_deref().unwrap_or(""));
        field("sampling", self.sampling.map_or("", SamplingDesign::as_str));
        field("dependence", self.dependence.map_or("", DependenceGroup::as_str));
        let trail = self
            .projections
            .iter()
            .map(|projection| match projection {
                EvidenceProjection::Marginalize { drop } => format!("marginalize:{}", sorted(drop)),
                EvidenceProjection::Condition { on } => format!("condition:{}", sorted(on)),
                EvidenceProjection::Deintervene { variables } => {
                    format!("deintervene:{}", sorted(variables))
                }
            })
            .collect::<Vec<_>>();
        field("projections", &trail.join(";"));
        out
    }
}

impl EvidenceCatalog {
    /// Every supplied distribution as a descriptor, in regime-id order.
    #[must_use]
    pub fn distributions(&self) -> Vec<CatalogDistribution> {
        let mut out = self
            .regimes
            .iter()
            .map(|regime| CatalogDistribution::from_regime(self, regime))
            .collect::<Vec<_>>();
        out.sort_by_key(|d| d.source_regime);
        out
    }

    /// Descriptor of one catalog entry.
    #[must_use]
    pub fn distribution(&self, regime: RegimeId) -> Option<CatalogDistribution> {
        self.regimes
            .iter()
            .find(|r| r.id == regime)
            .map(|r| CatalogDistribution::from_regime(self, r))
    }

    /// Known shared-data relationship between the data behind two entries.
    ///
    /// Same snapshot or declared dataset shares units; a declared linked-unit
    /// table is linked; two declared independent studies with distinct study
    /// identities are independent. Anything else, including an unbound entry,
    /// is [`SharedData::Unknown`].
    #[must_use]
    pub fn shared_data(&self, a: RegimeId, b: RegimeId) -> SharedData {
        let (Some(left), Some(right)) = (self.distribution(a), self.distribution(b)) else {
            return SharedData::Unknown;
        };
        let (Some(left_snapshot), Some(right_snapshot)) = (&left.snapshot, &right.snapshot) else {
            return SharedData::Unknown;
        };
        let same_dataset = left.dataset.is_some() && left.dataset == right.dataset;
        if a == b || left_snapshot == right_snapshot || same_dataset {
            return SharedData::SameDataset;
        }
        let dependence = [left.dependence, right.dependence];
        if dependence.contains(&Some(DependenceGroup::LinkedUnits)) {
            return SharedData::LinkedUnits;
        }
        if dependence.iter().all(|d| *d == Some(DependenceGroup::IndependentStudies))
            && left.study_or_population() != right.study_or_population()
        {
            return SharedData::IndependentStudies;
        }
        SharedData::Unknown
    }
}

fn sorted(variables: &[VariableId]) -> String {
    let mut raw = variables.iter().map(|v| v.raw()).collect::<Vec<_>>();
    raw.sort_unstable();
    raw.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}

fn canonical_value(value: &Value) -> String {
    match value {
        Value::Float64(v) => format!("f{:016x}", v.to_bits()),
        Value::Int64(v) => format!("i{v}"),
        Value::Bool(v) => format!("b{}", u8::from(*v)),
        Value::Category(v) => format!("c{v}"),
        Value::Label(v) => format!("l{}", escape(v)),
    }
}

/// Escape the identity's delimiters so no field value can forge another field.
fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('|', "\\|").replace('=', "\\=")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::transport_catalog::RegimeBinding;

    fn ids(raw: &[u32]) -> Vec<VariableId> {
        raw.iter().copied().map(VariableId::from_raw).collect()
    }

    fn regime(
        id: u32,
        population: &str,
        interventions: &[u32],
        measured: &[u32],
    ) -> EvidenceRegime {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            if interventions.is_empty() {
                RegimeKind::Observational
            } else {
                RegimeKind::Experimental
            },
            EvidenceKind::Available,
            ids(interventions),
            [],
            ids(measured),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    fn binding(regime: u32, snapshot: &str, dependence: DependenceGroup) -> RegimeBinding {
        RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(regime),
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence,
        }
    }

    #[test]
    fn descriptors_name_every_coordinate_and_bound_snapshot() {
        let mut trial = regime(2, "trial", &[0], &[1, 2]);
        trial.study = Some(Arc::from("study-a"));
        trial.intervention_values = Arc::from([InterventionAssignment {
            variable: VariableId::from_raw(0),
            value: Value::f64(1.0),
        }]);
        let catalog = EvidenceCatalog::try_new(
            [],
            [trial, regime(1, "target", &[], &[0, 1, 2])],
            [binding(2, "snap-a", DependenceGroup::IndependentStudies)],
            None,
        )
        .unwrap();
        let all = catalog.distributions();
        assert_eq!(all.iter().map(|d| d.source_regime.raw()).collect::<Vec<_>>(), [1, 2]);
        let trial = &all[1];
        assert_eq!(trial.study_or_population(), "study-a");
        assert_eq!(all[0].study_or_population(), "target");
        assert_eq!(trial.snapshot.as_deref(), Some("snap-a"));
        assert!(all[0].snapshot.is_none());
        assert!(trial.is_joint());
        assert_eq!(trial.intervention_values.len(), 1);
    }

    #[test]
    fn projections_keep_their_source_entry_and_trail() {
        let catalog =
            EvidenceCatalog::try_new([], [regime(4, "trial", &[0], &[1, 2, 3])], [], None).unwrap();
        let original = catalog.distribution(RegimeId::from_raw(4)).unwrap();
        let projected = original
            .project(EvidenceProjection::Marginalize { drop: ids(&[3]).into() })
            .unwrap()
            .project(EvidenceProjection::Condition { on: ids(&[2]).into() })
            .unwrap();
        assert_eq!(projected.source_regime, RegimeId::from_raw(4));
        assert_eq!(&*projected.measured, ids(&[1, 2]).as_slice());
        assert_eq!(&*projected.conditioned_on, ids(&[2]).as_slice());
        assert_eq!(projected.projections.len(), 2);
        assert_ne!(projected.canonical_identity(), original.canonical_identity());
        // The regime-level and descriptor-level projections agree.
        let via_regime = catalog.regimes[0]
            .project(EvidenceProjection::Marginalize { drop: ids(&[3]).into() })
            .unwrap();
        assert_eq!(
            via_regime.measured,
            original
                .project(EvidenceProjection::Marginalize { drop: ids(&[3]).into() })
                .unwrap()
                .measured
        );
        assert!(
            original
                .project(EvidenceProjection::Deintervene { variables: ids(&[0]).into() })
                .is_err()
        );
    }

    #[test]
    fn canonical_identity_ignores_declaration_order_but_not_semantics() {
        let a = regime(1, "trial", &[0, 5], &[1, 2]);
        let mut b = regime(1, "trial", &[5, 0], &[2, 1]);
        let identity = |r: EvidenceRegime| {
            EvidenceCatalog::try_new([], [r], [], None).unwrap().distributions()[0]
                .canonical_identity()
        };
        assert_eq!(identity(a.clone()), identity(b.clone()));
        b.study = Some(Arc::from("other"));
        assert_ne!(identity(a.clone()), identity(b));
        let mut forged = a.clone();
        forged.study = Some(Arc::from("x|population=target"));
        assert_ne!(identity(forged), identity(a));
    }

    #[test]
    fn selected_samples_and_model_artifacts_never_satisfy_factors() {
        let need_vars = ids(&[1]);
        let need_do = ids(&[0]);
        let need = super::super::transport_catalog::FactorNeed {
            population: "trial",
            variables: &need_vars,
            conditioned_on: &[],
            interventions: &need_do,
        };
        let base = regime(1, "trial", &[0], &[1]);
        assert!(base.satisfies(&need));
        let mut selected = base.clone();
        selected.selection = SamplingSelection::SelectedOn { variables: ids(&[1]).into() };
        assert!(!selected.satisfies(&need));
        assert!(!selected.available_experiment_on("trial", &need_do));
        let mut posterior = base.clone();
        posterior.origin = LawOrigin::ModelArtifact { artifact: Arc::from("posterior-1") };
        assert!(!posterior.satisfies(&need));
        let catalog = EvidenceCatalog::try_new([], [posterior], [], None).unwrap();
        assert!(catalog.source_experiment_variables("trial").is_empty());
        let mut empty_selection = base.clone();
        empty_selection.selection = SamplingSelection::SelectedOn { variables: Arc::from([]) };
        assert!(EvidenceCatalog::try_new([], [empty_selection], [], None).is_err());
        let mut empty_study = base;
        empty_study.study = Some(Arc::from(" "));
        assert!(EvidenceCatalog::try_new([], [empty_study], [], None).is_err());
    }

    #[test]
    fn shared_data_is_derived_from_declarations_and_never_assumed() {
        let mut first = regime(1, "source", &[0], &[1]);
        first.study = Some(Arc::from("study-1"));
        let mut second = regime(2, "source", &[2], &[1]);
        second.study = Some(Arc::from("study-2"));
        let mut third = regime(3, "source", &[3], &[1]);
        third.study = Some(Arc::from("study-1"));
        let catalog = |bindings: Vec<RegimeBinding>| {
            EvidenceCatalog::try_new(
                [],
                [first.clone(), second.clone(), third.clone()],
                bindings,
                None,
            )
            .unwrap()
        };
        let (r1, r2, r3) = (RegimeId::from_raw(1), RegimeId::from_raw(2), RegimeId::from_raw(3));
        let independent = catalog(vec![
            binding(1, "a", DependenceGroup::IndependentStudies),
            binding(2, "b", DependenceGroup::IndependentStudies),
        ]);
        assert_eq!(independent.shared_data(r1, r2), SharedData::IndependentStudies);
        // An unbound entry is unknown, never independent.
        assert_eq!(independent.shared_data(r1, r3), SharedData::Unknown);
        let same = catalog(vec![
            binding(1, "a", DependenceGroup::IndependentStudies),
            binding(2, "a", DependenceGroup::IndependentStudies),
        ]);
        assert_eq!(same.shared_data(r1, r2), SharedData::SameDataset);
        let linked = catalog(vec![
            binding(1, "a", DependenceGroup::LinkedUnits),
            binding(2, "b", DependenceGroup::IndependentStudies),
        ]);
        assert_eq!(linked.shared_data(r1, r2), SharedData::LinkedUnits);
        let undeclared = catalog(vec![
            binding(1, "a", DependenceGroup::UnknownDependence),
            binding(2, "b", DependenceGroup::IndependentStudies),
        ]);
        assert_eq!(undeclared.shared_data(r1, r2), SharedData::Unknown);
        // Independent declarations within one study identity do not certify independence.
        let one_study = catalog(vec![
            binding(1, "a", DependenceGroup::IndependentStudies),
            binding(3, "c", DependenceGroup::IndependentStudies),
        ]);
        assert_eq!(one_study.shared_data(r1, r3), SharedData::Unknown);
        assert_eq!(one_study.shared_data(r3, r3), SharedData::SameDataset);
    }
}
