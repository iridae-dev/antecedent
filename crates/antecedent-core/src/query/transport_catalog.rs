//! Environments, evidence regimes, and supplied transport catalogs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::ids::{RegimeId, VariableId};
use crate::value::Value;

use super::error::QueryError;

/// Shared variable coordinate used by source and target environments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariableCoordinate {
    /// Dense variable id shared across environments.
    pub variable: VariableId,
    /// Value domain. Incompatible domains for the same id are rejected.
    pub domain: VariableDomain,
    /// Optional physical unit. Differing units for the same id are rejected.
    pub unit: Option<Arc<str>>,
}

/// Domain declared on a transport variable coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VariableDomain {
    /// Unspecified; compatible with any later specified domain for the same id.
    Unspecified,
    /// Real-valued continuous measurement.
    Continuous,
    /// Two-level encoding.
    Binary,
    /// Non-negative integer counts.
    Count,
    /// Unordered categorical with a known cardinality.
    Categorical {
        /// Number of levels.
        cardinality: u32,
    },
}

impl VariableDomain {
    #[allow(clippy::float_cmp)] // Domain membership is exact, not approximate equality.
    fn accepts(&self, value: &Value) -> bool {
        let value = value.as_f64();
        match self {
            Self::Unspecified => true,
            Self::Continuous => value.is_some_and(f64::is_finite),
            Self::Binary => value.is_some_and(|v| v == 0.0 || v == 1.0),
            Self::Count => value.is_some_and(|v| v >= 0.0 && v.fract() == 0.0),
            Self::Categorical { cardinality } => {
                value.is_some_and(|v| v >= 0.0 && v < f64::from(*cardinality) && v.fract() == 0.0)
            }
        }
    }

    fn compatible_with(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::Unspecified, _)
                | (_, Self::Unspecified)
                | (Self::Continuous, Self::Continuous)
                | (Self::Binary, Self::Binary)
                | (Self::Count, Self::Count)
        ) || matches!(
            (self, other),
            (Self::Categorical { cardinality: a }, Self::Categorical { cardinality: b }) if a == b
        )
    }
}

/// One population in a transport problem.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Environment {
    /// Population identity. Duplicate identities in one catalog are rejected.
    pub identity: Arc<str>,
    /// Shared variable coordinates.
    pub variables: Arc<[VariableCoordinate]>,
    /// Source-to-target mechanism-selection targets declared on this environment.
    pub selection_targets: Arc<[VariableId]>,
}

impl Environment {
    /// Construct an environment after rejecting empty identity and duplicate coordinates.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] when the identity is empty or a variable
    /// is listed twice.
    pub fn try_new(
        identity: impl Into<Arc<str>>,
        variables: impl Into<Arc<[VariableCoordinate]>>,
        selection_targets: impl Into<Arc<[VariableId]>>,
    ) -> Result<Self, QueryError> {
        let identity = identity.into();
        if identity.trim().is_empty() {
            return Err(QueryError::InvalidTransport(
                "environment identity must be non-empty".into(),
            ));
        }
        let variables = variables.into();
        let mut seen = BTreeSet::new();
        for coordinate in variables.iter() {
            if matches!(coordinate.domain, VariableDomain::Categorical { cardinality: 0 }) {
                return Err(QueryError::InvalidTransport(
                    "categorical cardinality must be positive".into(),
                ));
            }
            if !seen.insert(coordinate.variable.raw()) {
                return Err(QueryError::InvalidTransport(
                    "environment variable coordinates must be unique".into(),
                ));
            }
        }
        let selection_targets = unique_variables(selection_targets.into(), "selection targets")?;
        Ok(Self { identity, variables, selection_targets })
    }
}

/// Whether a regime's results exist, or it is only proposed / manipulable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EvidenceKind {
    /// Results exist and may satisfy an executable factor.
    Available,
    /// The variables are manipulable but no results exist.
    Manipulable,
    /// A proposed future experiment. Not available evidence.
    Proposed,
}

impl EvidenceKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Manipulable => "manipulable",
            Self::Proposed => "proposed",
        }
    }

    /// Whether this kind can satisfy an executable factor.
    #[must_use]
    pub const fn can_satisfy_factor(self) -> bool {
        matches!(self, Self::Available)
    }
}

/// Observational versus experimental regime.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegimeKind {
    /// No hard intervention.
    Observational,
    /// Hard intervention on a declared set. `do(A)` never implies `do(A,B)`.
    Experimental,
}

impl RegimeKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observational => "observational",
            Self::Experimental => "experimental",
        }
    }
}

/// How a regime's measured law is available.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DistributionAvailability {
    /// Joint over the measured variables.
    Joint,
    /// Separate marginals. Never imply a joint measurement.
    SeparateMarginals {
        /// Variables whose marginals exist.
        variables: Arc<[VariableId]>,
    },
}

/// One available intervention assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct InterventionAssignment {
    /// Intervened variable.
    pub variable: VariableId,
    /// Assigned value.
    pub value: Value,
}

/// One evidence regime. Two single-variable experiments never imply a joint.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceRegime {
    /// Stable regime handle.
    pub id: RegimeId,
    /// Optional external regime name, retained in durable catalog identity.
    pub label: Option<Arc<str>>,
    /// Observational or experimental.
    pub kind: RegimeKind,
    /// Whether results exist.
    pub evidence_kind: EvidenceKind,
    /// Intervention set. Empty for observational regimes.
    pub interventions: Arc<[VariableId]>,
    /// Available intervention values. Empty means the domain is unrestricted.
    pub intervention_values: Arc<[InterventionAssignment]>,
    /// Measured variables.
    pub measured: Arc<[VariableId]>,
    /// Coordinates already conditioned on by an admissible projection.
    pub conditioned_on: Arc<[VariableId]>,
    /// Population this regime was collected in.
    pub population: Arc<str>,
    /// Joint versus separate-marginal availability.
    pub distribution: DistributionAvailability,
}

impl EvidenceRegime {
    /// Construct a regime after checking intervention / measurement consistency.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] for empty population, observational
    /// interventions, or experimental regimes with an empty intervention set.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        id: RegimeId,
        kind: RegimeKind,
        evidence_kind: EvidenceKind,
        interventions: impl Into<Arc<[VariableId]>>,
        intervention_values: impl Into<Arc<[InterventionAssignment]>>,
        measured: impl Into<Arc<[VariableId]>>,
        population: impl Into<Arc<str>>,
        distribution: DistributionAvailability,
    ) -> Result<Self, QueryError> {
        let population = population.into();
        if population.trim().is_empty() {
            return Err(QueryError::InvalidTransport("regime population must be non-empty".into()));
        }
        let interventions = unique_variables(interventions.into(), "regime interventions")?;
        let measured = unique_variables(measured.into(), "regime measured variables")?;
        match kind {
            RegimeKind::Observational if !interventions.is_empty() => {
                return Err(QueryError::InvalidTransport(
                    "observational regimes cannot carry hard interventions".into(),
                ));
            }
            RegimeKind::Experimental if interventions.is_empty() => {
                return Err(QueryError::InvalidTransport(
                    "experimental regimes require a non-empty intervention set".into(),
                ));
            }
            _ => {}
        }
        if let DistributionAvailability::SeparateMarginals { variables } = &distribution {
            unique_variables(Arc::clone(variables), "marginal variables")?;
            if variables.iter().any(|v| !measured.contains(v)) {
                return Err(QueryError::InvalidTransport(
                    "marginals require measured variables".into(),
                ));
            }
        }
        let intervention_values = intervention_values.into();
        let mut assigned = BTreeSet::new();
        for assignment in intervention_values.iter() {
            if !interventions.contains(&assignment.variable)
                || !assigned.insert(assignment.variable)
                || assignment.value.as_f64().is_some_and(|v| !v.is_finite())
            {
                return Err(QueryError::InvalidTransport(
                    "invalid or duplicate intervention assignment".into(),
                ));
            }
        }
        Ok(Self {
            id,
            label: None,
            kind,
            evidence_kind,
            interventions,
            intervention_values,
            measured,
            conditioned_on: Arc::from([]),
            population,
            distribution,
        })
    }

    /// Whether this regime is an available experimental assignment of `variables`.
    ///
    /// The intervention set must match exactly. `do(A)` and `do(B)` never
    /// jointly satisfy `do(A,B)`.
    #[must_use]
    pub fn available_experiment_on(&self, population: &str, variables: &[VariableId]) -> bool {
        self.evidence_kind.can_satisfy_factor()
            && self.kind == RegimeKind::Experimental
            && self.population.as_ref() == population
            && same_variable_set(&self.interventions, variables)
    }

    /// Admissible projection: marginalize or condition within a measured regime.
    ///
    /// Removing a hard intervention and treating the law as observational is
    /// refused. Support obligations stay with the caller.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] when the projection is not licensed.
    pub fn project(&self, projection: EvidenceProjection) -> Result<Self, QueryError> {
        match projection {
            EvidenceProjection::Deintervene { .. } => Err(QueryError::InvalidTransport(
                "removing a hard intervention and treating its law as observational is not licensed"
                    .into(),
            )),
            EvidenceProjection::Marginalize { drop } => {
                if drop.iter().any(|v| !self.measured.contains(v) || self.conditioned_on.contains(v)) {
                    return Err(QueryError::InvalidTransport("cannot marginalize an unmeasured variable".into()));
                }
                let measured = self
                    .measured
                    .iter()
                    .copied()
                    .filter(|variable| !drop.contains(variable))
                    .collect::<Vec<_>>();
                let mut out = self.clone();
                out.measured = measured.into();
                if let DistributionAvailability::SeparateMarginals { variables } = &mut out.distribution {
                    *variables = variables.iter().copied().filter(|v| !drop.contains(v)).collect::<Vec<_>>().into();
                }
                Ok(out)
            }
            EvidenceProjection::Condition { on } => {
                if !on.is_empty() && matches!(self.distribution, DistributionAvailability::SeparateMarginals { .. }) {
                    return Err(QueryError::InvalidTransport("conditioning requires a joint law".into()));
                }
                if on.iter().any(|variable| !self.measured.contains(variable)) {
                    return Err(QueryError::InvalidTransport(
                        "conditioning projection requires measured variables".into(),
                    ));
                }
                let mut out = self.clone();
                out.conditioned_on = self.conditioned_on.iter().chain(on.iter()).copied()
                    .collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>().into();
                Ok(out)
            }
        }
    }
}

/// Licensed projection of an evidence regime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceProjection {
    /// Drop measured variables. The intervention set is unchanged.
    Marginalize {
        /// Variables removed from the measured set.
        drop: Arc<[VariableId]>,
    },
    /// Condition on measured variables. Support obligations stay with the caller.
    Condition {
        /// Conditioning variables.
        on: Arc<[VariableId]>,
    },
    /// Explicitly refused: do not strip a hard intervention.
    Deintervene {
        /// Intervention variables the caller asked to remove.
        variables: Arc<[VariableId]>,
    },
}

/// How units in a bound table relate to other studies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DependenceGroup {
    /// Known independent studies.
    IndependentStudies,
    /// Linked units across tables.
    LinkedUnits,
    /// Dependence is unknown.
    UnknownDependence,
}

impl DependenceGroup {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndependentStudies => "independent_studies",
            Self::LinkedUnits => "linked_units",
            Self::UnknownDependence => "unknown_dependence",
        }
    }
}

/// Sampling design recorded on a regime binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SamplingDesign {
    /// Independent draws from the named population.
    Independent,
    /// Clustered or otherwise dependent draws.
    Clustered,
    /// Unknown design.
    Unknown,
}

impl SamplingDesign {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Clustered => "clustered",
            Self::Unknown => "unknown",
        }
    }
}

/// Licensed weights attached to a regime binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LicensedWeights {
    /// Snapshot identity the weights were written for.
    pub snapshot_identity: Arc<str>,
}

/// Bind a dataset / table to one regime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegimeBinding {
    /// Regime this snapshot satisfies.
    pub regime: RegimeId,
    /// Snapshot identity.
    pub snapshot_identity: Arc<str>,
    /// Schema names in column order.
    pub schema_names: Arc<[Arc<str>]>,
    /// Sampling design.
    pub sampling: SamplingDesign,
    /// Optional licensed weights.
    pub weights: Option<LicensedWeights>,
    /// Unit / cluster / dependence group.
    pub dependence: DependenceGroup,
}

/// How the target population law is supplied.
///
/// A convenience sample is not automatically representative.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TargetSampling {
    /// Caller supplied the population law.
    SuppliedPopulationLaw,
    /// Representative target sample.
    RepresentativeSample,
    /// Explicitly licensed weighted design.
    LicensedWeightedDesign,
    /// Convenience sample. Not automatically representative.
    ConvenienceSample,
}

impl TargetSampling {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SuppliedPopulationLaw => "supplied_population_law",
            Self::RepresentativeSample => "representative_sample",
            Self::LicensedWeightedDesign => "licensed_weighted_design",
            Self::ConvenienceSample => "convenience_sample",
        }
    }

    /// Whether this sampling can stand in for the target population law.
    #[must_use]
    pub const fn represents_target_law(self) -> bool {
        matches!(
            self,
            Self::SuppliedPopulationLaw | Self::RepresentativeSample | Self::LicensedWeightedDesign
        )
    }
}

/// One unmet factor dependency, in stable report order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnmetDependency {
    /// Stable factor id.
    pub factor_id: Arc<str>,
    /// Stable reason id.
    pub reason: Arc<str>,
    /// Variables the factor needed.
    pub variables: Arc<[VariableId]>,
}

/// Supplied evidence for one transport query. Multiple sources are first-class.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceCatalog {
    /// Environments. Duplicate identities are rejected.
    pub environments: Arc<[Environment]>,
    /// Regimes. Matching names across sources never imply invariance.
    pub regimes: Arc<[EvidenceRegime]>,
    /// Dataset bindings.
    pub bindings: Arc<[RegimeBinding]>,
    /// Target-population sampling semantics.
    pub target_sampling: Option<TargetSampling>,
}

impl EvidenceCatalog {
    /// Canonical ordering for new semantic identities. Numeric regime identities
    /// and physical schema column order are preserved. Historical artifact readers
    /// deliberately do not call this method before checking their saved digests.
    ///
    /// # Errors
    /// Invalid catalog entries or references.
    pub fn canonicalized(&self) -> Result<Self, QueryError> {
        self.validate()?;
        let mut result = self.clone();
        let environments = Arc::make_mut(&mut result.environments);
        environments.sort_by(|a, b| a.identity.cmp(&b.identity));
        for env in environments {
            Arc::make_mut(&mut env.variables).sort_by_key(|v| v.variable);
            Arc::make_mut(&mut env.selection_targets).sort_unstable();
        }
        let regimes = Arc::make_mut(&mut result.regimes);
        regimes.sort_by_key(|r| r.id);
        for regime in regimes {
            Arc::make_mut(&mut regime.interventions).sort_unstable();
            Arc::make_mut(&mut regime.intervention_values).sort_by_key(|a| a.variable);
            Arc::make_mut(&mut regime.measured).sort_unstable();
            Arc::make_mut(&mut regime.conditioned_on).sort_unstable();
            if let DistributionAvailability::SeparateMarginals { variables } =
                &mut regime.distribution
            {
                Arc::make_mut(variables).sort_unstable();
            }
        }
        Arc::make_mut(&mut result.bindings).sort_by_key(|b| b.regime);
        Ok(result)
    }

    /// Empty catalog: no source experimental evidence.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            environments: Arc::from([]),
            regimes: Arc::from([]),
            bindings: Arc::from([]),
            target_sampling: None,
        }
    }

    /// Construct and validate a catalog.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`] for duplicate identities, incompatible
    /// domains, duplicate regime ids, or bindings that name an unknown regime.
    pub fn try_new(
        environments: impl Into<Arc<[Environment]>>,
        regimes: impl Into<Arc<[EvidenceRegime]>>,
        bindings: impl Into<Arc<[RegimeBinding]>>,
        target_sampling: Option<TargetSampling>,
    ) -> Result<Self, QueryError> {
        let catalog = Self {
            environments: environments.into(),
            regimes: regimes.into(),
            bindings: bindings.into(),
            target_sampling,
        };
        catalog.validate()?;
        Ok(catalog)
    }

    /// Reject duplicate identities, incompatible domains, and dangling bindings.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidTransport`].
    pub fn validate(&self) -> Result<(), QueryError> {
        let mut identities = BTreeSet::new();
        let mut domains: BTreeMap<u32, (&VariableDomain, Option<&Arc<str>>)> = BTreeMap::new();
        for environment in self.environments.iter() {
            Environment::try_new(
                Arc::clone(&environment.identity),
                Arc::clone(&environment.variables),
                Arc::clone(&environment.selection_targets),
            )?;
            if !identities.insert(environment.identity.as_ref()) {
                return Err(QueryError::InvalidTransport(
                    "catalog environments must have distinct identities".into(),
                ));
            }
            for coordinate in environment.variables.iter() {
                if let Some((existing_domain, existing_unit)) =
                    domains.get_mut(&coordinate.variable.raw())
                {
                    if !existing_domain.compatible_with(&coordinate.domain) {
                        return Err(QueryError::InvalidTransport(
                            "catalog environments declare incompatible domains for the same variable"
                                .into(),
                        ));
                    }
                    if *existing_unit != coordinate.unit.as_ref()
                        && existing_unit.is_some()
                        && coordinate.unit.is_some()
                    {
                        return Err(QueryError::InvalidTransport(
                            "catalog environments declare incompatible units for the same variable"
                                .into(),
                        ));
                    }
                    if matches!(existing_domain, VariableDomain::Unspecified) {
                        *existing_domain = &coordinate.domain;
                    }
                    if existing_unit.is_none() {
                        *existing_unit = coordinate.unit.as_ref();
                    }
                } else {
                    domains.insert(
                        coordinate.variable.raw(),
                        (&coordinate.domain, coordinate.unit.as_ref()),
                    );
                }
            }
        }
        let mut regime_ids = BTreeSet::new();
        let mut labels = BTreeSet::new();
        for regime in self.regimes.iter() {
            if let Some(label) = &regime.label {
                if label.trim().is_empty() || !labels.insert(label.as_ref()) {
                    return Err(QueryError::InvalidTransport(
                        "duplicate or empty regime label".into(),
                    ));
                }
            }
            EvidenceRegime::try_new(
                regime.id,
                regime.kind,
                regime.evidence_kind,
                Arc::clone(&regime.interventions),
                Arc::clone(&regime.intervention_values),
                Arc::clone(&regime.measured),
                Arc::clone(&regime.population),
                regime.distribution.clone(),
            )?;
            unique_variables(Arc::clone(&regime.conditioned_on), "conditioning coordinates")?;
            if regime.conditioned_on.iter().any(|v| !regime.measured.contains(v)) {
                return Err(QueryError::InvalidTransport(
                    "conditioning requires measured coordinates".into(),
                ));
            }
            for assignment in regime.intervention_values.iter() {
                if let Some((domain, _)) = domains.get(&assignment.variable.raw()) {
                    let valid = domain.accepts(&assignment.value);
                    if !valid {
                        return Err(QueryError::InvalidTransport(
                            "intervention value violates declared domain".into(),
                        ));
                    }
                }
            }
            if !regime_ids.insert(regime.id.raw()) {
                return Err(QueryError::InvalidTransport(
                    "catalog regime ids must be unique".into(),
                ));
            }
        }
        self.validate_bindings()
    }

    fn validate_bindings(&self) -> Result<(), QueryError> {
        let mut bound_regimes = BTreeSet::new();
        for binding in self.bindings.iter() {
            if !bound_regimes.insert(binding.regime)
                || binding.snapshot_identity.trim().is_empty()
                || binding.schema_names.iter().any(|s| s.trim().is_empty())
                || binding.schema_names.iter().collect::<BTreeSet<_>>().len()
                    != binding.schema_names.len()
                || binding
                    .weights
                    .as_ref()
                    .is_some_and(|w| w.snapshot_identity != binding.snapshot_identity)
            {
                return Err(QueryError::InvalidTransport(
                    "invalid, duplicate, or mismatched regime binding".into(),
                ));
            }
            if !self.regimes.iter().any(|regime| regime.id == binding.regime) {
                return Err(QueryError::InvalidTransport(
                    "regime binding names an unknown regime".into(),
                ));
            }
        }
        Ok(())
    }

    /// Available experimental intervention variables in `population`.
    ///
    /// Each regime contributes its own intervention set. Separate `do(A)` and
    /// `do(B)` regimes yield `{A, B}` as a compatibility view; they still do
    /// not constitute `do(A,B)`.
    #[must_use]
    pub fn source_experiment_variables(&self, population: &str) -> Arc<[VariableId]> {
        let mut variables = BTreeSet::new();
        for regime in self.regimes.iter() {
            if regime.evidence_kind.can_satisfy_factor()
                && regime.kind == RegimeKind::Experimental
                && regime.population.as_ref() == population
            {
                variables.extend(regime.interventions.iter().copied());
            }
        }
        variables.into_iter().collect::<Vec<_>>().into()
    }

    /// Whether an available experimental regime matches `variables` exactly.
    #[must_use]
    pub fn has_available_experiment(&self, population: &str, variables: &[VariableId]) -> bool {
        self.regimes.iter().any(|regime| regime.available_experiment_on(population, variables))
    }

    /// Unmet executable-factor dependencies, sorted by factor id then reason.
    #[must_use]
    pub fn unmet_factor_dependencies(
        &self,
        population: &str,
        needed: &[(Arc<str>, Arc<[VariableId]>)],
    ) -> Arc<[UnmetDependency]> {
        let mut unmet = needed
            .iter()
            .filter(|(_, variables)| !self.has_available_experiment(population, variables))
            .map(|(factor_id, variables)| UnmetDependency {
                factor_id: Arc::clone(factor_id),
                reason: Arc::from("transport.missing_evidence"),
                variables: Arc::clone(variables),
            })
            .collect::<Vec<_>>();
        unmet.sort_by(|a, b| (&*a.factor_id, &*a.reason).cmp(&(&*b.factor_id, &*b.reason)));
        unmet.into()
    }

    /// Compatibility catalog: one available single-variable experiment per id.
    #[must_use]
    pub fn from_source_experiments(
        population: impl Into<Arc<str>>,
        experiments: &[VariableId],
    ) -> Self {
        let population = population.into();
        let regimes = experiments
            .iter()
            .copied()
            .enumerate()
            .map(|(index, variable)| EvidenceRegime {
                id: RegimeId::from_raw(u32::try_from(index).unwrap_or(u32::MAX)),
                label: None,
                kind: RegimeKind::Experimental,
                evidence_kind: EvidenceKind::Available,
                interventions: Arc::from([variable]),
                intervention_values: Arc::from([]),
                measured: Arc::from([]),
                conditioned_on: Arc::from([]),
                population: Arc::clone(&population),
                distribution: DistributionAvailability::Joint,
            })
            .collect::<Vec<_>>();
        Self {
            environments: Arc::from([]),
            regimes: regimes.into(),
            bindings: Arc::from([]),
            target_sampling: None,
        }
    }
}

fn unique_variables(
    variables: Arc<[VariableId]>,
    label: &str,
) -> Result<Arc<[VariableId]>, QueryError> {
    let mut seen = BTreeSet::new();
    for variable in variables.iter() {
        if !seen.insert(variable.raw()) {
            return Err(QueryError::InvalidTransport(format!("{label} must be unique")));
        }
    }
    Ok(variables)
}

fn same_variable_set(left: &[VariableId], right: &[VariableId]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut a = left.to_vec();
    let mut b = right.to_vec();
    a.sort_unstable_by_key(|id| id.raw());
    b.sort_unstable_by_key(|id| id.raw());
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coord(id: u32, domain: VariableDomain) -> VariableCoordinate {
        VariableCoordinate { variable: VariableId::from_raw(id), domain, unit: None }
    }

    fn experimental(id: u32, population: &str, interventions: &[u32]) -> EvidenceRegime {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            interventions.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            [],
            [VariableId::from_raw(2)],
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    #[test]
    fn single_and_joint_experiments_are_distinct() {
        let catalog = EvidenceCatalog::try_new(
            [Environment::try_new("trial", [coord(0, VariableDomain::Binary)], []).unwrap()],
            [experimental(1, "trial", &[0]), experimental(2, "trial", &[1])],
            [],
            None,
        )
        .unwrap();
        assert!(catalog.has_available_experiment("trial", &[VariableId::from_raw(0)]));
        assert!(catalog.has_available_experiment("trial", &[VariableId::from_raw(1)]));
        assert!(!catalog.has_available_experiment(
            "trial",
            &[VariableId::from_raw(0), VariableId::from_raw(1)]
        ));
        let joint =
            EvidenceCatalog::try_new([], [experimental(3, "trial", &[0, 1])], [], None).unwrap();
        assert!(joint.has_available_experiment(
            "trial",
            &[VariableId::from_raw(0), VariableId::from_raw(1)]
        ));
    }

    #[test]
    fn manipulable_is_not_available_evidence() {
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Manipulable,
            [VariableId::from_raw(0)],
            [],
            [],
            "trial",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let catalog = EvidenceCatalog::try_new([], [regime], [], None).unwrap();
        assert!(!catalog.has_available_experiment("trial", &[VariableId::from_raw(0)]));
        assert!(catalog.source_experiment_variables("trial").is_empty());
    }

    #[test]
    fn separate_marginals_never_imply_a_joint() {
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [VariableId::from_raw(0), VariableId::from_raw(1)],
            "target",
            DistributionAvailability::SeparateMarginals {
                variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            },
        )
        .unwrap();
        assert!(matches!(regime.distribution, DistributionAvailability::SeparateMarginals { .. }));
        assert!(!matches!(regime.distribution, DistributionAvailability::Joint));
    }

    #[test]
    fn convenience_sample_is_not_the_target_law() {
        assert!(!TargetSampling::ConvenienceSample.represents_target_law());
        assert!(TargetSampling::SuppliedPopulationLaw.represents_target_law());
        assert!(TargetSampling::RepresentativeSample.represents_target_law());
        assert_ne!(
            TargetSampling::ConvenienceSample.as_str(),
            TargetSampling::SuppliedPopulationLaw.as_str()
        );
    }

    #[test]
    fn same_named_incompatible_domains_are_rejected() {
        let source = Environment::try_new(
            "trial",
            [VariableCoordinate {
                variable: VariableId::from_raw(0),
                domain: VariableDomain::Binary,
                unit: Some(Arc::from("dose")),
            }],
            [],
        )
        .unwrap();
        let target = Environment::try_new(
            "target",
            [VariableCoordinate {
                variable: VariableId::from_raw(0),
                domain: VariableDomain::Continuous,
                unit: Some(Arc::from("mg")),
            }],
            [],
        )
        .unwrap();
        let error = EvidenceCatalog::try_new([source, target], [], [], None).unwrap_err();
        assert!(
            matches!(error, QueryError::InvalidTransport(message) if message.contains("incompatible"))
        );
    }

    #[test]
    fn duplicate_environment_identities_are_rejected() {
        let a = Environment::try_new("trial", [], []).unwrap();
        let b = Environment::try_new("trial", [], []).unwrap();
        assert!(EvidenceCatalog::try_new([a, b], [], [], None).is_err());
    }

    #[test]
    fn deintervening_is_refused() {
        let regime = experimental(1, "trial", &[0]);
        let error = regime
            .project(EvidenceProjection::Deintervene {
                variables: Arc::from([VariableId::from_raw(0)]),
            })
            .unwrap_err();
        assert!(matches!(error, QueryError::InvalidTransport(_)));
    }

    #[test]
    fn unmet_dependencies_are_stable_and_complete() {
        let catalog = EvidenceCatalog::empty();
        let unmet = catalog.unmet_factor_dependencies(
            "trial",
            &[
                (Arc::from("factor.b"), Arc::from([VariableId::from_raw(1)])),
                (Arc::from("factor.a"), Arc::from([VariableId::from_raw(0)])),
            ],
        );
        assert_eq!(unmet.len(), 2);
        assert_eq!(&*unmet[0].factor_id, "factor.a");
        assert_eq!(&*unmet[1].factor_id, "factor.b");
    }
}
