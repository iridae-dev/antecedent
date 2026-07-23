//! Causal variable schemas and role hints.
//!
//! Schema construction assigns dense [`VariableId`](crate::ids::VariableId)s and
//! validates uniqueness once. Algorithmic code receives compact IDs and
//! immutable schema references. Name lookup is allowed at API boundaries and
//! diagnostics, not inside traversal or numerical loops.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::num::NonZeroU32;
use std::sync::Arc;

use crate::error::SchemaError;
use crate::ids::{CategoryDomainId, VariableId};

/// Scalar element type for vector-valued variables.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScalarType {
    /// IEEE-754 binary64.
    Float64,
    /// IEEE-754 binary32.
    Float32,
    /// Signed 64-bit integer.
    Int64,
    /// Signed 32-bit integer.
    Int32,
}

/// Observed value kind for a variable.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ValueType {
    /// Real-valued continuous measurement.
    Continuous,
    /// Non-negative integer counts.
    Count,
    /// Two-level categorical encoded as binary.
    Binary,
    /// Unordered categorical.
    Categorical,
    /// Ordered categorical.
    Ordinal,
    /// Fixed-width vector of scalar elements.
    Vector {
        /// Number of elements (non-zero).
        width: NonZeroU32,
        /// Element scalar type.
        element: ScalarType,
    },
}

impl ValueType {
    /// Whether this value type requires a category domain.
    #[must_use]
    pub const fn requires_category_domain(&self) -> bool {
        matches!(self, Self::Categorical | Self::Ordinal)
    }
}

/// Role hints and constraints; not graph truth.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u16)]
pub enum RoleHint {
    /// Candidate treatment variable.
    TreatmentCandidate = 1 << 0,
    /// Candidate outcome variable.
    OutcomeCandidate = 1 << 1,
    /// Candidate instrument.
    InstrumentCandidate = 1 << 2,
    /// Context / covariate.
    Context = 1 << 3,
    /// Selection indicator.
    Selection = 1 << 4,
    /// Panel / unit identifier.
    UnitId = 1 << 5,
    /// Time index.
    Time = 1 << 6,
    /// Environment label.
    Environment = 1 << 7,
    /// Regime label.
    Regime = 1 << 8,
}

/// Closed role-hint set stored as a bit mask (no heap allocation).
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct SmallRoleSet(u16);

impl SmallRoleSet {
    /// Empty role set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Construct from a single hint.
    #[must_use]
    pub const fn from_hint(hint: RoleHint) -> Self {
        Self(hint as u16)
    }

    /// Construct from an iterator of hints.
    #[must_use]
    pub fn from_hints(hints: impl IntoIterator<Item = RoleHint>) -> Self {
        let mut set = Self::empty();
        for hint in hints {
            set.insert(hint);
        }
        set
    }

    /// Insert a role hint.
    pub fn insert(&mut self, hint: RoleHint) {
        self.0 |= hint as u16;
    }

    /// Remove a role hint.
    pub fn remove(&mut self, hint: RoleHint) {
        self.0 &= !(hint as u16);
    }

    /// Whether the set contains `hint`.
    #[must_use]
    pub const fn contains(self, hint: RoleHint) -> bool {
        self.0 & (hint as u16) != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Raw bit mask (stable for serialization).
    #[must_use]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Reconstruct from a raw bit mask, ignoring unknown bits.
    #[must_use]
    pub const fn from_bits_truncate(bits: u16) -> Self {
        const KNOWN: u16 = (1 << 9) - 1;
        Self(bits & KNOWN)
    }

    /// Iterate known role hints present in the set.
    pub fn iter(self) -> impl Iterator<Item = RoleHint> {
        const ALL: [RoleHint; 9] = [
            RoleHint::TreatmentCandidate,
            RoleHint::OutcomeCandidate,
            RoleHint::InstrumentCandidate,
            RoleHint::Context,
            RoleHint::Selection,
            RoleHint::UnitId,
            RoleHint::Time,
            RoleHint::Environment,
            RoleHint::Regime,
        ];
        ALL.into_iter().filter(move |h| self.contains(*h))
    }
}

impl fmt::Debug for SmallRoleSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// Measurement metadata attached to a variable.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeasurementSpec {
    /// Optional human-readable measurement description.
    pub description: Option<Arc<str>>,
    /// Whether the measurement is considered noisy.
    pub noisy: bool,
}

/// Immutable description of one variable in a causal schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariableSchema {
    /// Dense variable identifier.
    pub id: VariableId,
    /// Stable user-facing name (stored once).
    pub name: Arc<str>,
    /// Observed value kind.
    pub value_type: ValueType,
    /// Role hints (bit mask).
    pub role_hints: SmallRoleSet,
    /// Optional physical unit label.
    pub unit: Option<Arc<str>>,
    /// Category domain when required by `value_type`.
    pub category_domain: Option<CategoryDomainId>,
    /// Measurement metadata.
    pub measurement: MeasurementSpec,
}

/// Immutable causal schema: dense IDs plus a name dictionary.
///
/// After construction, algorithmic code indexes by [`VariableId`]. Name lookup
/// is an O(n) scan intended for API boundaries and diagnostics only—never for
/// hot graph or numerical loops.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CausalSchema {
    variables: Arc<[VariableSchema]>,
}

impl CausalSchema {
    /// Number of variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Whether the schema has no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Borrow the dense variable table.
    #[must_use]
    pub fn variables(&self) -> &[VariableSchema] {
        &self.variables
    }

    /// Look up a variable by dense ID.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::UnknownVariableId`] when `id` is out of range.
    pub fn get(&self, id: VariableId) -> Result<&VariableSchema, SchemaError> {
        self.variables.get(id.as_usize()).ok_or(SchemaError::UnknownVariableId { id: id.raw() })
    }

    /// Resolve a user-facing name to a dense ID (API-boundary use only).
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::UnknownVariableName`] when the name is absent.
    pub fn id_of(&self, name: &str) -> Result<VariableId, SchemaError> {
        self.variables
            .iter()
            .find(|v| &*v.name == name)
            .map(|v| v.id)
            .ok_or_else(|| SchemaError::UnknownVariableName { name: name.to_owned() })
    }

    /// Resolve a name to the variable schema (API-boundary use only).
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::UnknownVariableName`] when the name is absent.
    pub fn get_by_name(&self, name: &str) -> Result<&VariableSchema, SchemaError> {
        let id = self.id_of(name)?;
        self.get(id)
    }
}

/// Builder that assigns dense IDs and validates uniqueness once.
#[derive(Debug, Default)]
pub struct CausalSchemaBuilder {
    pending: Vec<PendingVariable>,
    /// Deferred error from fluent commits (surfaced at [`Self::build`]).
    deferred: Option<SchemaError>,
}

#[derive(Debug)]
struct PendingVariable {
    name: Arc<str>,
    value_type: ValueType,
    role_hints: SmallRoleSet,
    unit: Option<Arc<str>>,
    category_domain: Option<CategoryDomainId>,
    measurement: MeasurementSpec,
}

impl CausalSchemaBuilder {
    /// Create an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a continuous variable; finish with a role method (e.g. [`VariableInProgress::treatment`]).
    #[must_use]
    pub fn continuous(self, name: impl Into<Arc<str>>) -> VariableInProgress {
        self.begin(name, ValueType::Continuous, None)
    }

    /// Start a binary variable; finish with a role method.
    #[must_use]
    pub fn binary(self, name: impl Into<Arc<str>>) -> VariableInProgress {
        self.begin(name, ValueType::Binary, None)
    }

    /// Start a count variable; finish with a role method.
    #[must_use]
    pub fn count(self, name: impl Into<Arc<str>>) -> VariableInProgress {
        self.begin(name, ValueType::Count, None)
    }

    fn begin(
        self,
        name: impl Into<Arc<str>>,
        value_type: ValueType,
        category_domain: Option<CategoryDomainId>,
    ) -> VariableInProgress {
        VariableInProgress {
            builder: self,
            pending: PendingVariable {
                name: name.into(),
                value_type,
                role_hints: SmallRoleSet::empty(),
                unit: None,
                category_domain,
                measurement: MeasurementSpec::default(),
            },
        }
    }

    /// Add a variable declaration.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::DuplicateVariableName`] if `name` was already added.
    /// Returns domain-related errors when `value_type` and `category_domain`
    /// disagree.
    pub fn add_variable(
        &mut self,
        name: impl Into<Arc<str>>,
        value_type: ValueType,
        role_hints: SmallRoleSet,
        unit: Option<Arc<str>>,
        category_domain: Option<CategoryDomainId>,
        measurement: MeasurementSpec,
    ) -> Result<(), SchemaError> {
        let name = name.into();
        if self.pending.iter().any(|p| p.name == name) {
            return Err(SchemaError::DuplicateVariableName { name: name.to_string() });
        }
        validate_domain_consistency(&name, &value_type, category_domain)?;
        self.pending.push(PendingVariable {
            name,
            value_type,
            role_hints,
            unit,
            category_domain,
            measurement,
        });
        Ok(())
    }

    fn push_pending(&mut self, pending: PendingVariable) {
        if self.deferred.is_some() {
            return;
        }
        if self.pending.iter().any(|p| p.name == pending.name) {
            self.deferred =
                Some(SchemaError::DuplicateVariableName { name: pending.name.to_string() });
            return;
        }
        if let Err(e) =
            validate_domain_consistency(&pending.name, &pending.value_type, pending.category_domain)
        {
            self.deferred = Some(e);
            return;
        }
        self.pending.push(pending);
    }

    /// Consume the builder and produce an immutable schema with dense IDs.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaError::TooManyVariables`] if the count exceeds `u32::MAX`,
    /// or a deferred fluent-construction error.
    pub fn build(self) -> Result<CausalSchema, SchemaError> {
        if let Some(err) = self.deferred {
            return Err(err);
        }
        let n_u32 = u32::try_from(self.pending.len()).map_err(|_| SchemaError::TooManyVariables)?;
        let variables: Arc<[VariableSchema]> = self
            .pending
            .into_iter()
            .zip(0..n_u32)
            .map(|(p, i)| VariableSchema {
                id: VariableId::from_raw(i),
                name: p.name,
                value_type: p.value_type,
                role_hints: p.role_hints,
                unit: p.unit,
                category_domain: p.category_domain,
                measurement: p.measurement,
            })
            .collect();
        Ok(CausalSchema { variables })
    }
}

/// Fluent handle for one variable being added to a [`CausalSchemaBuilder`].
///
/// Call a role method (or [`Self::finish`]) to commit the variable and continue building.
#[derive(Debug)]
pub struct VariableInProgress {
    builder: CausalSchemaBuilder,
    pending: PendingVariable,
}

impl VariableInProgress {
    /// Mark as a treatment candidate and commit.
    #[must_use]
    pub fn treatment(mut self) -> CausalSchemaBuilder {
        self.pending.role_hints.insert(RoleHint::TreatmentCandidate);
        self.finish()
    }

    /// Mark as an outcome candidate and commit.
    #[must_use]
    pub fn outcome(mut self) -> CausalSchemaBuilder {
        self.pending.role_hints.insert(RoleHint::OutcomeCandidate);
        self.finish()
    }

    /// Mark as a context / covariate and commit.
    #[must_use]
    pub fn context(mut self) -> CausalSchemaBuilder {
        self.pending.role_hints.insert(RoleHint::Context);
        self.finish()
    }

    /// Mark as an instrument candidate and commit.
    #[must_use]
    pub fn instrument(mut self) -> CausalSchemaBuilder {
        self.pending.role_hints.insert(RoleHint::InstrumentCandidate);
        self.finish()
    }

    /// Optional physical unit label.
    #[must_use]
    pub fn unit(mut self, unit: impl Into<Arc<str>>) -> Self {
        self.pending.unit = Some(unit.into());
        self
    }

    /// Commit with no additional role hints.
    #[must_use]
    pub fn finish(self) -> CausalSchemaBuilder {
        let mut builder = self.builder;
        builder.push_pending(self.pending);
        builder
    }

    /// Commit and build the schema.
    ///
    /// # Errors
    ///
    /// Propagates [`CausalSchemaBuilder::build`] errors.
    pub fn build(self) -> Result<CausalSchema, SchemaError> {
        self.finish().build()
    }
}

fn validate_domain_consistency(
    name: &str,
    value_type: &ValueType,
    category_domain: Option<CategoryDomainId>,
) -> Result<(), SchemaError> {
    match (value_type.requires_category_domain(), category_domain) {
        (true, None) => Err(SchemaError::MissingCategoryDomain { name: name.to_owned() }),
        (false, Some(_)) => Err(SchemaError::UnexpectedCategoryDomain { name: name.to_owned() }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn role_set_is_bitmask_not_heap() {
        assert_eq!(size_of::<SmallRoleSet>(), 2);
        let mut set =
            SmallRoleSet::from_hints([RoleHint::TreatmentCandidate, RoleHint::OutcomeCandidate]);
        assert!(set.contains(RoleHint::TreatmentCandidate));
        assert!(set.contains(RoleHint::OutcomeCandidate));
        assert!(!set.contains(RoleHint::InstrumentCandidate));
        set.remove(RoleHint::TreatmentCandidate);
        assert!(!set.contains(RoleHint::TreatmentCandidate));
    }

    #[test]
    fn schema_assigns_dense_ids_in_order() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema.variables()[0].id.raw(), 0);
        assert_eq!(schema.variables()[1].id.raw(), 1);
        assert_eq!(schema.id_of("y").unwrap().raw(), 1);
        assert_eq!(schema.get(VariableId::from_raw(0)).unwrap().name.as_ref(), "x");
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::empty(),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let err = b
            .add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap_err();
        assert!(matches!(err, SchemaError::DuplicateVariableName { .. }));
    }

    #[test]
    fn categorical_requires_domain() {
        let mut b = CausalSchemaBuilder::new();
        let err = b
            .add_variable(
                "g",
                ValueType::Categorical,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap_err();
        assert!(matches!(err, SchemaError::MissingCategoryDomain { .. }));
    }

    #[test]
    fn continuous_rejects_domain() {
        let mut b = CausalSchemaBuilder::new();
        let err = b
            .add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                Some(CategoryDomainId::from_raw(0)),
                MeasurementSpec::default(),
            )
            .unwrap_err();
        assert!(matches!(err, SchemaError::UnexpectedCategoryDomain { .. }));
    }

    #[test]
    fn unknown_id_and_name_errors() {
        let schema = CausalSchemaBuilder::new().build().unwrap();
        assert!(matches!(
            schema.get(VariableId::from_raw(0)),
            Err(SchemaError::UnknownVariableId { .. })
        ));
        assert!(matches!(schema.id_of("missing"), Err(SchemaError::UnknownVariableName { .. })));
    }

    #[test]
    fn fluent_schema_roles() {
        let schema = CausalSchemaBuilder::new()
            .continuous("t")
            .treatment()
            .continuous("y")
            .outcome()
            .continuous("z")
            .context()
            .build()
            .unwrap();
        assert_eq!(schema.len(), 3);
        assert!(schema.variables()[0].role_hints.contains(RoleHint::TreatmentCandidate));
        assert!(schema.variables()[1].role_hints.contains(RoleHint::OutcomeCandidate));
        assert!(schema.variables()[2].role_hints.contains(RoleHint::Context));
    }
}
