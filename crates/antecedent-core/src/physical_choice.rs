//! Costed choice among physical backends.
//!
//! The alternatives are a direct prepared numeric view, a streamed reduction,
//! a materialized design, and optional lazy preparation. Each backend requires
//! one operator capability and one resource capability. A missing capability
//! refuses that alternative before any cost is compared, so a cheaper quote
//! cannot select a backend whose required capabilities are absent.
//!
//! [`PhysicalChoice::choose`] records the admitted alternative with the lowest
//! declared cost, and every rejected alternative with the reason it lost. Equal
//! costs keep the earlier alternative in [`PhysicalAlternative::ALL`]. A later
//! alternative with the same cost is rejected because its cost is not lower.
//!
//! The costs are declared quotes. [`RouteChoice::select`] derives them from a
//! route's own work and shape: the cells each backend reads and the bytes it
//! holds, with a backend ineligible when the route does not implement it or its
//! resident bytes exceed the route budget. This module does not execute the
//! plan, scan rows, build a design, or run a lazy backend.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::named_variants;
use std::error::Error;
use std::fmt;

/// Physical backend that can carry a prepared numeric plan.
///
/// Declaration order is the tie break: the earliest admitted alternative wins
/// when declared costs are equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PhysicalAlternative {
    /// Read prepared numeric columns directly.
    ///
    /// Requires [`OperatorCapability::PreparedNumericView`] and
    /// [`ResourceCapability::PreparedColumns`].
    DirectPreparedNumericViews,
    /// Reduce a stream without retaining the scanned rows.
    ///
    /// Requires [`OperatorCapability::StreamedReduction`] and
    /// [`ResourceCapability::StreamingScratch`].
    StreamedReduction,
    /// Build a design matrix before the numeric kernel.
    ///
    /// Requires [`OperatorCapability::DesignMaterialization`] and
    /// [`ResourceCapability::DesignMemory`].
    MaterializedDesign,
    /// Optional lazy preparation.
    ///
    /// Requires [`OperatorCapability::LazyPreparation`] and
    /// [`ResourceCapability::LazyBackend`]. When either capability is absent
    /// this backend is rejected and the other alternatives stay available.
    OptionalLazyPreparation,
}

named_variants! {
    PhysicalAlternative {
        DirectPreparedNumericViews => "direct_prepared_numeric_views",
        StreamedReduction => "streamed_reduction",
        MaterializedDesign => "materialized_design",
        OptionalLazyPreparation => "optional_lazy_preparation",
    }
}

impl PhysicalAlternative {
    /// Operator capability this backend requires.
    #[must_use]
    pub const fn required_operator(self) -> OperatorCapability {
        match self {
            Self::DirectPreparedNumericViews => OperatorCapability::PreparedNumericView,
            Self::StreamedReduction => OperatorCapability::StreamedReduction,
            Self::MaterializedDesign => OperatorCapability::DesignMaterialization,
            Self::OptionalLazyPreparation => OperatorCapability::LazyPreparation,
        }
    }

    /// Resource capability this backend requires.
    #[must_use]
    pub const fn required_resource(self) -> ResourceCapability {
        match self {
            Self::DirectPreparedNumericViews => ResourceCapability::PreparedColumns,
            Self::StreamedReduction => ResourceCapability::StreamingScratch,
            Self::MaterializedDesign => ResourceCapability::DesignMemory,
            Self::OptionalLazyPreparation => ResourceCapability::LazyBackend,
        }
    }

    const fn from_index(index: usize) -> Self {
        Self::ALL[index]
    }
}

impl fmt::Display for PhysicalAlternative {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Operator capability a physical backend may require.
///
/// Each variant is independent. A prepared numeric view does not provide a
/// streamed reduction, design materialization, or lazy preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OperatorCapability {
    /// Read a prepared numeric view without rebuilding it.
    PreparedNumericView,
    /// Reduce values while streaming, without retaining the rows.
    StreamedReduction,
    /// Materialize a design matrix.
    DesignMaterialization,
    /// Lower preparation onto an optional lazy backend.
    LazyPreparation,
}

named_variants! {
    OperatorCapability {
        PreparedNumericView => "prepared_numeric_view",
        StreamedReduction => "streamed_reduction",
        DesignMaterialization => "design_materialization",
        LazyPreparation => "lazy_preparation",
    }
}

impl OperatorCapability {
    const fn mask(self) -> u8 {
        match self {
            Self::PreparedNumericView => 1,
            Self::StreamedReduction => 1 << 1,
            Self::DesignMaterialization => 1 << 2,
            Self::LazyPreparation => 1 << 3,
        }
    }
}

impl fmt::Display for OperatorCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Resource capability a physical backend may require.
///
/// A resource is present or absent. Presence is not a byte budget, and an
/// absent resource is not a cost of zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResourceCapability {
    /// Borrowed prepared numeric columns.
    PreparedColumns,
    /// Scratch for one streamed reduction.
    StreamingScratch,
    /// Memory in which a design matrix can be materialized.
    DesignMemory,
    /// The optional lazy backend itself.
    LazyBackend,
}

named_variants! {
    ResourceCapability {
        PreparedColumns => "prepared_columns",
        StreamingScratch => "streaming_scratch",
        DesignMemory => "design_memory",
        LazyBackend => "lazy_backend",
    }
}

impl ResourceCapability {
    const fn mask(self) -> u8 {
        match self {
            Self::PreparedColumns => 1,
            Self::StreamingScratch => 1 << 1,
            Self::DesignMemory => 1 << 2,
            Self::LazyBackend => 1 << 3,
        }
    }
}

impl fmt::Display for ResourceCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct OperatorBits(u8);

impl OperatorBits {
    const fn empty() -> Self {
        Self(0)
    }

    const fn with(self, capability: OperatorCapability) -> Self {
        Self(self.0 | capability.mask())
    }

    const fn without(self, capability: OperatorCapability) -> Self {
        Self(self.0 & !capability.mask())
    }

    const fn contains(self, capability: OperatorCapability) -> bool {
        self.0 & capability.mask() != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ResourceBits(u8);

impl ResourceBits {
    const fn empty() -> Self {
        Self(0)
    }

    const fn with(self, capability: ResourceCapability) -> Self {
        Self(self.0 | capability.mask())
    }

    const fn without(self, capability: ResourceCapability) -> Self {
        Self(self.0 & !capability.mask())
    }

    const fn contains(self, capability: ResourceCapability) -> bool {
        self.0 & capability.mask() != 0
    }
}

/// Operator and resource capabilities that are present for one choice.
///
/// An empty inventory admits nothing. [`Self::all`] is the capability pair of
/// every [`PhysicalAlternative`], not a scan of data.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AvailableCapabilities {
    operators: OperatorBits,
    resources: ResourceBits,
}

impl AvailableCapabilities {
    /// No operator capabilities and no resource capabilities.
    #[must_use]
    pub const fn empty() -> Self {
        Self { operators: OperatorBits::empty(), resources: ResourceBits::empty() }
    }

    /// Operator and resource capabilities required by every alternative.
    #[must_use]
    pub const fn all() -> Self {
        let mut available = Self::empty();
        let mut index = 0usize;
        while index < PhysicalAlternative::ALL.len() {
            let alternative = PhysicalAlternative::from_index(index);
            available = available
                .with_operator(alternative.required_operator())
                .with_resource(alternative.required_resource());
            index += 1;
        }
        available
    }

    /// Record operator `capability` as present.
    #[must_use]
    pub const fn with_operator(self, capability: OperatorCapability) -> Self {
        Self { operators: self.operators.with(capability), resources: self.resources }
    }

    /// Record resource `capability` as present.
    #[must_use]
    pub const fn with_resource(self, capability: ResourceCapability) -> Self {
        Self { operators: self.operators, resources: self.resources.with(capability) }
    }

    /// Record operator `capability` as absent.
    #[must_use]
    pub const fn without_operator(self, capability: OperatorCapability) -> Self {
        Self { operators: self.operators.without(capability), resources: self.resources }
    }

    /// Record resource `capability` as absent.
    #[must_use]
    pub const fn without_resource(self, capability: ResourceCapability) -> Self {
        Self { operators: self.operators, resources: self.resources.without(capability) }
    }

    /// Whether the operator `capability` was recorded present.
    #[must_use]
    pub const fn has_operator(self, capability: OperatorCapability) -> bool {
        self.operators.contains(capability)
    }

    /// Whether the resource `capability` was recorded present.
    #[must_use]
    pub const fn has_resource(self, capability: ResourceCapability) -> bool {
        self.resources.contains(capability)
    }

    /// Refuse `alternative` when a required capability is absent.
    ///
    /// Operator capability is checked first. Cost is not an argument: a missing
    /// capability refuses the alternative before selection.
    const fn refusal(self, alternative: PhysicalAlternative) -> Option<RejectionReason> {
        let operator = alternative.required_operator();
        if !self.has_operator(operator) {
            return Some(RejectionReason::MissingOperatorCapability { capability: operator });
        }
        let resource = alternative.required_resource();
        if !self.has_resource(resource) {
            return Some(RejectionReason::MissingResourceCapability { capability: resource });
        }
        None
    }
}

/// Declared cost of one alternative.
///
/// The value is a quote in cost units. Zero is a declared cost of zero, not a
/// missing quote. Nothing here turns an absent capability into a zero cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DeclaredCost(u64);

impl DeclaredCost {
    /// Cost of `raw` units.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Declared cost units.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for DeclaredCost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Declared cost of each physical alternative.
///
/// Every alternative is quoted. Admission does not read these quotes until the
/// alternative's operator and resource capabilities are both present.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CostQuotes {
    direct_prepared_numeric_views: DeclaredCost,
    streamed_reduction: DeclaredCost,
    materialized_design: DeclaredCost,
    optional_lazy_preparation: DeclaredCost,
}

impl CostQuotes {
    /// Quotes in [`PhysicalAlternative::ALL`] order.
    #[must_use]
    pub const fn new(
        direct_prepared_numeric_views: DeclaredCost,
        streamed_reduction: DeclaredCost,
        materialized_design: DeclaredCost,
        optional_lazy_preparation: DeclaredCost,
    ) -> Self {
        Self {
            direct_prepared_numeric_views,
            streamed_reduction,
            materialized_design,
            optional_lazy_preparation,
        }
    }

    /// Quote for `alternative`.
    #[must_use]
    pub const fn cost(self, alternative: PhysicalAlternative) -> DeclaredCost {
        match alternative {
            PhysicalAlternative::DirectPreparedNumericViews => self.direct_prepared_numeric_views,
            PhysicalAlternative::StreamedReduction => self.streamed_reduction,
            PhysicalAlternative::MaterializedDesign => self.materialized_design,
            PhysicalAlternative::OptionalLazyPreparation => self.optional_lazy_preparation,
        }
    }
}

/// Why an alternative was not selected.
///
/// A missing capability is recorded before costs are compared.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RejectionReason {
    /// The alternative's operator capability is absent.
    MissingOperatorCapability {
        /// Operator capability that was required and absent.
        capability: OperatorCapability,
    },
    /// The alternative's resource capability is absent.
    ///
    /// The operator capability was present. The resource was still required.
    MissingResourceCapability {
        /// Resource capability that was required and absent.
        capability: ResourceCapability,
    },
    /// Both capabilities were present, and the cost was not strictly lower.
    ///
    /// `cost` is this alternative's quote. `selected_cost` is the quote of the
    /// alternative that was selected. Equal costs reject the later alternative.
    NotLowerCost {
        /// This alternative's declared cost.
        cost: DeclaredCost,
        /// Declared cost of the selected alternative.
        selected_cost: DeclaredCost,
    },
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingOperatorCapability { capability } => {
                write!(formatter, "missing operator capability {capability}")
            }
            Self::MissingResourceCapability { capability } => {
                write!(formatter, "missing resource capability {capability}")
            }
            Self::NotLowerCost { cost, selected_cost } => {
                write!(formatter, "cost {cost} is not lower than selected cost {selected_cost}")
            }
        }
    }
}

/// One alternative that was not selected, with the reason.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RejectedAlternative {
    alternative: PhysicalAlternative,
    reason: RejectionReason,
}

impl RejectedAlternative {
    const fn new(alternative: PhysicalAlternative, reason: RejectionReason) -> Self {
        Self { alternative, reason }
    }

    /// Alternative that was refused or outranked.
    #[must_use]
    pub const fn alternative(self) -> PhysicalAlternative {
        self.alternative
    }

    /// Why [`Self::alternative`] was not selected.
    #[must_use]
    pub const fn reason(self) -> RejectionReason {
        self.reason
    }
}

impl fmt::Display for RejectedAlternative {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.alternative, self.reason)
    }
}

/// One selected backend and the three alternatives that were not selected.
///
/// The value is a record of the choice. It does not execute that backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PhysicalChoice {
    selected: PhysicalAlternative,
    selected_cost: DeclaredCost,
    rejected: [RejectedAlternative; 3],
}

impl PhysicalChoice {
    /// Choose the admitted alternative with the lowest declared cost.
    ///
    /// Each alternative is refused first when its operator capability or its
    /// resource capability is absent. Quotes of refused alternatives are not
    /// compared. Among the alternatives that remain, the lowest quote wins.
    /// An equal quote keeps the earlier entry of [`PhysicalAlternative::ALL`].
    ///
    /// # Errors
    ///
    /// [`PhysicalChoiceRefusal`] when every alternative lacks a required
    /// capability. No alternative is selected.
    #[allow(
        clippy::result_large_err,
        reason = "the refusal is the four-alternative report; boxing it would hide that fixed record"
    )]
    #[must_use = "a refusal selects no backend"]
    pub const fn choose(
        quotes: CostQuotes,
        available: AvailableCapabilities,
    ) -> Result<Self, PhysicalChoiceRefusal> {
        let classified = classify(available);
        match cheapest(classified.eligible, quotes) {
            None => Err(PhysicalChoiceRefusal::from_reasons(classified.reasons)),
            Some(selected) => {
                let selected_alternative = PhysicalAlternative::from_index(selected);
                let selected_cost = quotes.cost(selected_alternative);
                let reasons = mark_cost_rejections(classified, quotes, selected);
                Ok(Self {
                    selected: selected_alternative,
                    selected_cost,
                    rejected: rejections_except(selected, reasons),
                })
            }
        }
    }

    /// Alternative that was selected.
    #[must_use]
    pub const fn selected(&self) -> PhysicalAlternative {
        self.selected
    }

    /// Declared cost of [`Self::selected`].
    #[must_use]
    pub const fn selected_cost(&self) -> DeclaredCost {
        self.selected_cost
    }

    /// Rejected alternatives, in [`PhysicalAlternative::ALL`] order.
    #[must_use]
    pub const fn rejected(&self) -> &[RejectedAlternative] {
        &self.rejected
    }

    /// Report naming the selection and every rejection.
    #[must_use = "the report names the selection and every rejection"]
    pub fn report(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for PhysicalChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "selected {} at cost {}", self.selected, self.selected_cost)?;
        for rejection in &self.rejected {
            writeln!(formatter, "rejected {rejection}")?;
        }
        Ok(())
    }
}

/// Every alternative lacked a required capability, so none was selected.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct PhysicalChoiceRefusal {
    rejected: [RejectedAlternative; 4],
}

impl PhysicalChoiceRefusal {
    const fn from_reasons(reasons: [RejectionReason; 4]) -> Self {
        Self {
            rejected: [
                RejectedAlternative::new(
                    PhysicalAlternative::DirectPreparedNumericViews,
                    reasons[0],
                ),
                RejectedAlternative::new(PhysicalAlternative::StreamedReduction, reasons[1]),
                RejectedAlternative::new(PhysicalAlternative::MaterializedDesign, reasons[2]),
                RejectedAlternative::new(PhysicalAlternative::OptionalLazyPreparation, reasons[3]),
            ],
        }
    }

    /// Every alternative, each with the capability that refused it.
    #[must_use]
    pub const fn rejected(&self) -> &[RejectedAlternative] {
        &self.rejected
    }
}

impl fmt::Display for PhysicalChoiceRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("no physical alternative admitted")?;
        for rejected in &self.rejected {
            write!(formatter, "; rejected {rejected}")?;
        }
        formatter.write_str(
            "; next attempt: declare the operator and resource capabilities of one alternative",
        )
    }
}

impl Error for PhysicalChoiceRefusal {}

/// Rows a streamed reduction holds at once.
pub const DEFAULT_STREAM_BATCH_ROWS: u64 = 1024;

/// Cost of reading one source cell relative to loading one resident design cell.
///
/// A source cell is decoded and validated on every read. A resident cell is a
/// plain load.
const SOURCE_CELL_COST: u64 = 2;

/// Work and shape of one route, from which every alternative is quoted.
///
/// `passes` is the number of times the numeric kernel reads the whole design:
/// one for a closed-form reduction, more for each fit, fold or refit that
/// re-reads it. Costs are counted in cells of `element_bytes` bytes: the cells
/// an alternative touches, plus the cells it keeps resident.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RouteWork {
    rows: u64,
    columns: u64,
    passes: u64,
    element_bytes: u64,
    batch_rows: u64,
}

impl RouteWork {
    /// A `rows` by `columns` binary64 design the kernel reads `passes` times.
    /// Zero passes is read as one: a route that fits reads its design.
    #[must_use]
    pub const fn new(rows: u64, columns: u64, passes: u64) -> Self {
        Self {
            rows,
            columns,
            passes: if passes == 0 { 1 } else { passes },
            element_bytes: 8,
            batch_rows: DEFAULT_STREAM_BATCH_ROWS,
        }
    }

    /// Bytes of one design cell. Zero is read as one.
    #[must_use]
    pub const fn element_bytes(mut self, bytes: u64) -> Self {
        self.element_bytes = if bytes == 0 { 1 } else { bytes };
        self
    }

    /// Rows a streamed reduction holds at once. Zero is read as one.
    #[must_use]
    pub const fn batch_rows(mut self, rows: u64) -> Self {
        self.batch_rows = if rows == 0 { 1 } else { rows };
        self
    }

    /// Rows of the design.
    #[must_use]
    pub const fn rows(self) -> u64 {
        self.rows
    }

    /// Columns of the design.
    #[must_use]
    pub const fn columns(self) -> u64 {
        self.columns
    }

    /// Reads of the whole design by the kernel.
    #[must_use]
    pub const fn passes(self) -> u64 {
        self.passes
    }
}

/// Which backends the route has an implementation for.
///
/// This is an operator declaration, not a preference: an alternative that is
/// not declared here is rejected for its missing operator capability whatever
/// it would cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSupport {
    /// Closed set of implemented `PhysicalAlternative` capabilities.
    alternatives: u8,
}

impl std::hash::Hash for BackendSupport {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Preserve the previous four-field hash input order across the private
        // representation change; the set does not change capability identity.
        for alternative in PhysicalAlternative::ALL {
            std::hash::Hash::hash(&self.implements(alternative), state);
        }
    }
}

impl BackendSupport {
    /// No backend implemented.
    #[must_use]
    pub const fn none() -> Self {
        Self { alternatives: 0 }
    }

    /// Only the materialized design is implemented.
    #[must_use]
    pub const fn materialized_only() -> Self {
        Self::none().with(PhysicalAlternative::MaterializedDesign, true)
    }

    /// Declare whether `alternative` is implemented.
    #[must_use]
    pub const fn with(mut self, alternative: PhysicalAlternative, implemented: bool) -> Self {
        let mask = Self::mask(alternative);
        if implemented {
            self.alternatives |= mask;
        } else {
            self.alternatives &= !mask;
        }
        self
    }

    /// Whether `alternative` is implemented.
    #[must_use]
    pub const fn implements(self, alternative: PhysicalAlternative) -> bool {
        self.alternatives & Self::mask(alternative) != 0
    }

    const fn mask(alternative: PhysicalAlternative) -> u8 {
        match alternative {
            PhysicalAlternative::DirectPreparedNumericViews => 1,
            PhysicalAlternative::StreamedReduction => 2,
            PhysicalAlternative::MaterializedDesign => 4,
            PhysicalAlternative::OptionalLazyPreparation => 8,
        }
    }
}

/// Quote of one alternative: the cells it touches and the bytes it keeps resident.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AlternativeQuote {
    alternative: PhysicalAlternative,
    work_cells: u64,
    resident_bytes: u64,
    cost: DeclaredCost,
}

impl AlternativeQuote {
    /// Alternative quoted.
    #[must_use]
    pub const fn alternative(self) -> PhysicalAlternative {
        self.alternative
    }

    /// Cells the alternative reads or writes.
    #[must_use]
    pub const fn work_cells(self) -> u64 {
        self.work_cells
    }

    /// Bytes the alternative holds while it runs.
    #[must_use]
    pub const fn resident_bytes(self) -> u64 {
        self.resident_bytes
    }

    /// Declared cost compared in the choice: work cells plus resident cells.
    #[must_use]
    pub const fn cost(self) -> DeclaredCost {
        self.cost
    }
}

/// A route quote did not fit, or no alternative was admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum RouteChoiceError {
    /// A work or resident quote overflows `u64`.
    #[error(
        "physical alternative quote overflows u64 before allocation; next attempt: lower the \
         rows, columns or passes"
    )]
    Unrepresentable,
    /// Every alternative lacked an implementation or the memory to run in.
    #[error("{0}")]
    NoneAdmitted(PhysicalChoiceRefusal),
}

/// The costed choice of one route, with the quote behind each alternative.
///
/// An alternative is eligible when the route implements it and its resident
/// bytes fit the route budget. Among eligible alternatives the lowest cost
/// wins ([`PhysicalChoice::choose`]); every other alternative is rejected with
/// the reason, and its quote stays readable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RouteChoice {
    choice: PhysicalChoice,
    quotes: [AlternativeQuote; 4],
    budget_bytes: u64,
}

impl RouteChoice {
    /// Quote every alternative for `work` and choose among those `support`
    /// implements and `budget` can hold.
    ///
    /// Direct views read the source once into the final design and then the
    /// resident design on each pass. A materialized design stages the selected
    /// columns before the final design, so it touches and holds a second
    /// copy. A streamed reduction holds one batch and the Gram matrix, and
    /// re-reads the source on every pass. Lazy preparation costs what direct
    /// views cost.
    ///
    /// # Errors
    ///
    /// [`RouteChoiceError::Unrepresentable`] when a quote overflows `u64`.
    /// [`RouteChoiceError::NoneAdmitted`] when no alternative is both
    /// implemented and within the budget; the refusal names the missing
    /// capability of each.
    #[allow(
        clippy::result_large_err,
        reason = "the refusal is the four-alternative report; boxing it would hide that fixed record"
    )]
    pub fn select(
        work: RouteWork,
        support: BackendSupport,
        budget: crate::route_memory::RouteBudget,
    ) -> Result<Self, RouteChoiceError> {
        let overflow = || RouteChoiceError::Unrepresentable;
        let cells = work.rows.checked_mul(work.columns).ok_or_else(overflow)?;
        let design_bytes = cells.checked_mul(work.element_bytes).ok_or_else(overflow)?;
        let resident_reads = cells.checked_mul(work.passes).ok_or_else(overflow)?;
        let gather = |copies: u64| cells.checked_mul(copies).ok_or_else(overflow);
        let batch = work.batch_rows.min(work.rows.max(1));
        let gram_cells = work.columns.checked_mul(work.columns).ok_or_else(overflow)?;
        let streamed_bytes = batch
            .checked_mul(work.columns)
            .and_then(|window| window.checked_add(gram_cells))
            .and_then(|held| held.checked_mul(work.element_bytes))
            .ok_or_else(overflow)?;
        let streamed_work = resident_reads.checked_mul(SOURCE_CELL_COST).ok_or_else(overflow)?;
        let direct_work = gather(1)?.checked_add(resident_reads).ok_or_else(overflow)?;
        let materialized_work = gather(2)?.checked_add(resident_reads).ok_or_else(overflow)?;
        let materialized_bytes = design_bytes.checked_mul(2).ok_or_else(overflow)?;
        let raw = [
            (PhysicalAlternative::DirectPreparedNumericViews, direct_work, design_bytes),
            (PhysicalAlternative::StreamedReduction, streamed_work, streamed_bytes),
            (PhysicalAlternative::MaterializedDesign, materialized_work, materialized_bytes),
            (PhysicalAlternative::OptionalLazyPreparation, direct_work, design_bytes),
        ];
        let [direct, streamed, materialized, lazy] = raw.map(
            |(alternative, work_cells, resident_bytes)| -> Result<AlternativeQuote, RouteChoiceError> {
                let cost = work_cells
                    .checked_add(resident_bytes.div_ceil(work.element_bytes))
                    .ok_or_else(overflow)?;
                Ok(AlternativeQuote {
                    alternative,
                    work_cells,
                    resident_bytes,
                    cost: DeclaredCost::from_raw(cost),
                })
            },
        );
        let quotes = [direct?, streamed?, materialized?, lazy?];
        let mut available = AvailableCapabilities::empty();
        for quote in quotes {
            let alternative = quote.alternative;
            if support.implements(alternative) {
                available = available.with_operator(alternative.required_operator());
            }
            if quote.resident_bytes <= budget.bytes() {
                available = available.with_resource(alternative.required_resource());
            }
        }
        let costs = CostQuotes::new(quotes[0].cost, quotes[1].cost, quotes[2].cost, quotes[3].cost);
        let choice =
            PhysicalChoice::choose(costs, available).map_err(RouteChoiceError::NoneAdmitted)?;
        Ok(Self { choice, quotes, budget_bytes: budget.bytes() })
    }

    /// The choice: selected alternative, its cost, and every rejection.
    #[must_use]
    pub const fn choice(&self) -> &PhysicalChoice {
        &self.choice
    }

    /// Alternative that was selected.
    #[must_use]
    pub const fn selected(&self) -> PhysicalAlternative {
        self.choice.selected()
    }

    /// Quote of every alternative, in [`PhysicalAlternative::ALL`] order.
    #[must_use]
    pub const fn quotes(&self) -> &[AlternativeQuote; 4] {
        &self.quotes
    }

    /// Quote of `alternative`.
    #[must_use]
    pub const fn quote(&self, alternative: PhysicalAlternative) -> AlternativeQuote {
        let mut index = 0usize;
        while index < self.quotes.len() {
            if self.quotes[index].alternative as u8 == alternative as u8 {
                return self.quotes[index];
            }
            index += 1;
        }
        self.quotes[0]
    }

    /// Byte budget the resident bytes were compared with.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

#[derive(Clone, Copy)]
struct Classified {
    eligible: [bool; 4],
    reasons: [RejectionReason; 4],
}

const fn classify(available: AvailableCapabilities) -> Classified {
    let mut eligible = [false; 4];
    let mut reasons = [RejectionReason::MissingOperatorCapability {
        capability: OperatorCapability::PreparedNumericView,
    }; 4];
    let mut index = 0usize;
    while index < PhysicalAlternative::ALL.len() {
        let alternative = PhysicalAlternative::from_index(index);
        if let Some(reason) = available.refusal(alternative) {
            reasons[index] = reason;
        } else {
            eligible[index] = true;
        }
        index += 1;
    }
    Classified { eligible, reasons }
}

const fn cheapest(eligible: [bool; 4], quotes: CostQuotes) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_cost = 0u64;
    let mut index = 0usize;
    while index < eligible.len() {
        if eligible[index] {
            let cost = quotes.cost(PhysicalAlternative::from_index(index)).raw();
            if best.is_none() || cost < best_cost {
                best = Some(index);
                best_cost = cost;
            }
        }
        index += 1;
    }
    best
}

const fn mark_cost_rejections(
    classified: Classified,
    quotes: CostQuotes,
    selected: usize,
) -> [RejectionReason; 4] {
    let eligible = classified.eligible;
    let mut reasons = classified.reasons;
    let selected_cost = quotes.cost(PhysicalAlternative::from_index(selected));
    let mut index = 0usize;
    while index < eligible.len() {
        if eligible[index] && index != selected {
            reasons[index] = RejectionReason::NotLowerCost {
                cost: quotes.cost(PhysicalAlternative::from_index(index)),
                selected_cost,
            };
        }
        index += 1;
    }
    reasons
}

const fn rejections_except(
    selected: usize,
    reasons: [RejectionReason; 4],
) -> [RejectedAlternative; 3] {
    let placeholder = RejectedAlternative::new(
        PhysicalAlternative::DirectPreparedNumericViews,
        RejectionReason::MissingOperatorCapability {
            capability: OperatorCapability::PreparedNumericView,
        },
    );
    let mut packed = [placeholder; 3];
    let mut slot = 0usize;
    let mut index = 0usize;
    while index < reasons.len() {
        if index != selected {
            packed[slot] =
                RejectedAlternative::new(PhysicalAlternative::from_index(index), reasons[index]);
            slot += 1;
        }
        index += 1;
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::{
        AvailableCapabilities, BackendSupport, CostQuotes, DeclaredCost, OperatorCapability,
        PhysicalAlternative, PhysicalChoice, RejectionReason, ResourceCapability, RouteChoice,
        RouteChoiceError, RouteWork,
    };
    use crate::route_memory::RouteBudget;

    fn everything() -> BackendSupport {
        PhysicalAlternative::ALL
            .into_iter()
            .fold(BackendSupport::none(), |support, alternative| support.with(alternative, true))
    }

    fn work(passes: u64) -> RouteWork {
        RouteWork::new(100_000, 4, passes).batch_rows(1024)
    }

    #[test]
    fn a_single_pass_reduction_streams_instead_of_holding_the_design() {
        let routed = RouteChoice::select(work(1), everything(), RouteBudget::new(u64::MAX))
            .expect("admitted");
        assert_eq!(routed.selected(), PhysicalAlternative::StreamedReduction);
        let streamed = routed.quote(PhysicalAlternative::StreamedReduction);
        assert_eq!(streamed.work_cells(), 800_000, "one source scan at the decode cost");
        assert_eq!(streamed.resident_bytes(), (1024 * 4 + 16) * 8, "one batch and the Gram matrix");
        assert_eq!(streamed.cost(), cost(804_112));
        let direct = routed.quote(PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(direct.work_cells(), 800_000);
        assert_eq!(direct.resident_bytes(), 100_000 * 4 * 8);
        assert_eq!(direct.cost(), cost(1_200_000));
        assert_eq!(
            reason(routed.choice(), PhysicalAlternative::DirectPreparedNumericViews),
            RejectionReason::NotLowerCost { cost: cost(1_200_000), selected_cost: cost(804_112) },
        );
        let materialized = routed.quote(PhysicalAlternative::MaterializedDesign);
        assert_eq!(materialized.resident_bytes(), 2 * direct.resident_bytes());
        assert_eq!(materialized.cost(), cost(2_000_000));
        assert_eq!(routed.choice().rejected().len(), 3);
    }

    #[test]
    fn a_route_that_reads_its_design_repeatedly_reads_it_in_place() {
        let routed = RouteChoice::select(work(4), everything(), RouteBudget::new(u64::MAX))
            .expect("admitted");
        assert_eq!(routed.selected(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(routed.choice().selected_cost(), cost(2_400_000));
        assert!(
            routed.quote(PhysicalAlternative::StreamedReduction).cost()
                > routed.quote(PhysicalAlternative::DirectPreparedNumericViews).cost()
        );
    }

    #[test]
    fn an_unimplemented_backend_is_ineligible_however_cheap() {
        let routed = RouteChoice::select(
            work(1),
            BackendSupport::materialized_only(),
            RouteBudget::new(u64::MAX),
        )
        .expect("the materialized design is implemented");
        assert_eq!(routed.selected(), PhysicalAlternative::MaterializedDesign);
        assert!(
            routed.quote(PhysicalAlternative::StreamedReduction).cost()
                < routed.quote(PhysicalAlternative::MaterializedDesign).cost(),
            "streaming would be cheaper"
        );
        assert_missing_operator(routed.choice(), PhysicalAlternative::StreamedReduction);
        assert_missing_operator(routed.choice(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_missing_operator(routed.choice(), PhysicalAlternative::OptionalLazyPreparation);
    }

    #[test]
    fn a_budget_that_cannot_hold_the_design_leaves_only_the_streamed_reduction() {
        let routed = RouteChoice::select(work(4), everything(), RouteBudget::new(1_000_000))
            .expect("the streamed reduction fits");
        assert_eq!(routed.selected(), PhysicalAlternative::StreamedReduction);
        assert_eq!(routed.budget_bytes(), 1_000_000);
        assert_eq!(
            reason(routed.choice(), PhysicalAlternative::DirectPreparedNumericViews),
            RejectionReason::MissingResourceCapability {
                capability: ResourceCapability::PreparedColumns,
            },
        );
        assert_eq!(
            reason(routed.choice(), PhysicalAlternative::MaterializedDesign),
            RejectionReason::MissingResourceCapability {
                capability: ResourceCapability::DesignMemory,
            },
        );
    }

    #[test]
    fn nothing_is_chosen_when_the_only_implemented_backend_does_not_fit() {
        let refused = RouteChoice::select(
            work(1),
            BackendSupport::materialized_only(),
            RouteBudget::new(1_000_000),
        )
        .expect_err("the materialized design does not fit and nothing else is implemented");
        let RouteChoiceError::NoneAdmitted(refusal) = refused else {
            panic!("unexpected refusal: {refused}");
        };
        assert_eq!(refusal.rejected().len(), 4);
        let text = refused.to_string();
        assert!(text.contains("missing resource capability design_memory"), "{text}");
        assert!(text.contains("missing operator capability streamed_reduction"), "{text}");
        assert!(text.contains("next attempt"), "{text}");
    }

    #[test]
    fn a_quote_that_overflows_is_refused() {
        let overflow = RouteChoice::select(
            RouteWork::new(u64::MAX, 2, 1),
            everything(),
            RouteBudget::new(u64::MAX),
        );
        assert_eq!(overflow, Err(RouteChoiceError::Unrepresentable));
    }

    const fn cost(raw: u64) -> DeclaredCost {
        DeclaredCost::from_raw(raw)
    }

    const fn quotes(direct: u64, streamed: u64, materialized: u64, lazy: u64) -> CostQuotes {
        CostQuotes::new(cost(direct), cost(streamed), cost(materialized), cost(lazy))
    }

    const fn capabilities_for(alternative: PhysicalAlternative) -> AvailableCapabilities {
        AvailableCapabilities::empty()
            .with_operator(alternative.required_operator())
            .with_resource(alternative.required_resource())
    }

    fn choose_when_only(alternative: PhysicalAlternative) -> PhysicalChoice {
        let (direct, streamed, materialized, lazy) = match alternative {
            PhysicalAlternative::DirectPreparedNumericViews => (9, 1, 1, 1),
            PhysicalAlternative::StreamedReduction => (1, 9, 1, 1),
            PhysicalAlternative::MaterializedDesign => (1, 1, 9, 1),
            PhysicalAlternative::OptionalLazyPreparation => (1, 1, 1, 9),
        };
        let choice = PhysicalChoice::choose(
            quotes(direct, streamed, materialized, lazy),
            capabilities_for(alternative),
        )
        .expect("the alternative whose capabilities are present is admitted");
        assert_eq!(choice.rejected().len(), 3);
        assert!(choice.rejected().iter().all(|rejected| rejected.alternative() != alternative));
        choice
    }

    fn reason(choice: &PhysicalChoice, alternative: PhysicalAlternative) -> RejectionReason {
        choice
            .rejected()
            .iter()
            .find(|rejected| rejected.alternative() == alternative)
            .expect("alternative is rejected")
            .reason()
    }

    fn assert_missing_operator(choice: &PhysicalChoice, alternative: PhysicalAlternative) {
        assert_ne!(choice.selected(), alternative);
        assert_eq!(
            reason(choice, alternative),
            RejectionReason::MissingOperatorCapability {
                capability: alternative.required_operator(),
            },
        );
    }

    #[test]
    fn direct_prepared_numeric_views_is_chosen_when_its_capabilities_are_present() {
        let choice = choose_when_only(PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(choice.selected(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(choice.selected_cost(), cost(9));
        assert_missing_operator(&choice, PhysicalAlternative::StreamedReduction);
        assert_missing_operator(&choice, PhysicalAlternative::MaterializedDesign);
        assert_missing_operator(&choice, PhysicalAlternative::OptionalLazyPreparation);
    }

    #[test]
    fn streamed_reduction_is_chosen_when_its_capabilities_are_present() {
        let choice = choose_when_only(PhysicalAlternative::StreamedReduction);
        assert_eq!(choice.selected(), PhysicalAlternative::StreamedReduction);
        assert_eq!(choice.selected_cost(), cost(9));
        assert_missing_operator(&choice, PhysicalAlternative::DirectPreparedNumericViews);
        assert_missing_operator(&choice, PhysicalAlternative::MaterializedDesign);
        assert_missing_operator(&choice, PhysicalAlternative::OptionalLazyPreparation);
    }

    #[test]
    fn materialized_design_is_chosen_when_its_capabilities_are_present() {
        let choice = choose_when_only(PhysicalAlternative::MaterializedDesign);
        assert_eq!(choice.selected(), PhysicalAlternative::MaterializedDesign);
        assert_eq!(choice.selected_cost(), cost(9));
        assert_missing_operator(&choice, PhysicalAlternative::DirectPreparedNumericViews);
        assert_missing_operator(&choice, PhysicalAlternative::StreamedReduction);
        assert_missing_operator(&choice, PhysicalAlternative::OptionalLazyPreparation);
    }

    #[test]
    fn optional_lazy_preparation_is_chosen_when_its_capabilities_are_present() {
        let choice = choose_when_only(PhysicalAlternative::OptionalLazyPreparation);
        assert_eq!(choice.selected(), PhysicalAlternative::OptionalLazyPreparation);
        assert_eq!(choice.selected_cost(), cost(9));
        assert_missing_operator(&choice, PhysicalAlternative::DirectPreparedNumericViews);
        assert_missing_operator(&choice, PhysicalAlternative::StreamedReduction);
        assert_missing_operator(&choice, PhysicalAlternative::MaterializedDesign);
    }

    #[test]
    fn absent_capability_is_rejected_and_is_not_selected() {
        let missing_operator =
            AvailableCapabilities::all().without_operator(OperatorCapability::StreamedReduction);
        let choice = PhysicalChoice::choose(quotes(10, 1, 20, 30), missing_operator)
            .expect("direct views remain admitted");
        assert_eq!(choice.selected(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_ne!(choice.selected(), PhysicalAlternative::StreamedReduction);
        assert_eq!(
            reason(&choice, PhysicalAlternative::StreamedReduction),
            RejectionReason::MissingOperatorCapability {
                capability: OperatorCapability::StreamedReduction,
            },
        );
        assert!(missing_operator.has_resource(ResourceCapability::StreamingScratch));

        let missing_resource =
            AvailableCapabilities::all().without_resource(ResourceCapability::PreparedColumns);
        let choice = PhysicalChoice::choose(quotes(1, 10, 20, 30), missing_resource)
            .expect("streamed reduction remains admitted");
        assert_eq!(choice.selected(), PhysicalAlternative::StreamedReduction);
        assert_ne!(choice.selected(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(
            reason(&choice, PhysicalAlternative::DirectPreparedNumericViews),
            RejectionReason::MissingResourceCapability {
                capability: ResourceCapability::PreparedColumns,
            },
        );
        assert!(missing_resource.has_operator(OperatorCapability::PreparedNumericView));
    }

    #[test]
    fn report_names_the_selection_and_every_rejection() {
        let available = AvailableCapabilities::empty()
            .with_operator(OperatorCapability::PreparedNumericView)
            .with_resource(ResourceCapability::PreparedColumns)
            .with_operator(OperatorCapability::DesignMaterialization)
            .with_operator(OperatorCapability::LazyPreparation)
            .with_resource(ResourceCapability::LazyBackend);
        let choice = PhysicalChoice::choose(quotes(5, 1, 1, 9), available)
            .expect("direct views and lazy preparation are admitted");
        assert_eq!(choice.selected(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(
            choice.report(),
            "\
selected direct_prepared_numeric_views at cost 5
rejected streamed_reduction: missing operator capability streamed_reduction
rejected materialized_design: missing resource capability design_memory
rejected optional_lazy_preparation: cost 9 is not lower than selected cost 5
"
        );
        assert_eq!(
            PhysicalAlternative::from_name("direct_prepared_numeric_views"),
            Some(choice.selected())
        );
    }

    #[test]
    fn equal_declared_costs_keep_the_earlier_alternative() {
        let choice = PhysicalChoice::choose(quotes(4, 4, 4, 4), AvailableCapabilities::all())
            .expect("every alternative is admitted");
        assert_eq!(choice.selected(), PhysicalAlternative::DirectPreparedNumericViews);
        assert_eq!(
            reason(&choice, PhysicalAlternative::OptionalLazyPreparation),
            RejectionReason::NotLowerCost { cost: cost(4), selected_cost: cost(4) },
        );
    }

    #[test]
    fn no_alternative_is_selected_when_every_required_capability_is_absent() {
        let refusal = PhysicalChoice::choose(quotes(1, 1, 1, 1), AvailableCapabilities::empty())
            .expect_err("an empty inventory admits no backend");
        assert_eq!(refusal.rejected().len(), 4);
        for alternative in PhysicalAlternative::ALL {
            let rejected = refusal
                .rejected()
                .iter()
                .find(|rejected| rejected.alternative() == alternative)
                .expect("each alternative is rejected");
            assert_eq!(
                rejected.reason(),
                RejectionReason::MissingOperatorCapability {
                    capability: alternative.required_operator(),
                },
            );
        }
        let report = refusal.to_string();
        assert!(report.starts_with("no physical alternative admitted"));
        for alternative in PhysicalAlternative::ALL {
            assert!(report.contains(alternative.as_str()));
        }
        assert!(!report.contains("selected "));
    }
}
