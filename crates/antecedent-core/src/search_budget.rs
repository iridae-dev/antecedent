//! Work and memory limits for exact GPDC, graph search, functional enumeration, and matching.
//!
//! Each estimate is a `u64` byte count and a `u64` work count. Bytes follow the
//! route-memory rule: a declared width is the width that is charged, zero is a
//! declared empty charge, and an absent declaration is not zero. Products that
//! do not fit in `u64` are unrepresentable. They are not reported as zero and
//! they do not select another mode.
//!
//! Admission compares the estimate with a declared limit and returns before the
//! kernel is factored, the search is entered, the cells are visited, or the
//! donors are scanned. A sum equal to the limit is admitted. A larger memory or
//! work count is refused by name. The next attempt is a larger limit or an
//! explicitly requested approximation mode.
//!
//! An approximation is a distinct scientific mode. It carries its own evidence
//! and is not an automatic large-data fallback. Naming that evidence while
//! admitting an exact estimate does not run the approximation, and an exact
//! refusal cannot be converted into one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use thiserror::Error;

/// Declared byte width of one IEEE-754 binary64 value.
const BINARY64_BYTES: u64 = 8;

/// Declared byte width of one 64-bit index.
///
/// Eight here is the declared index width. It is not a widening of a narrower
/// index, and it is not [`usize`]'s pointer width.
const INDEX_BYTES: u64 = 8;

/// Declared byte width of one exact-search score.
const SCORE_BYTES: u64 = BINARY64_BYTES;

/// Leaf size used to charge a balanced exact matching tree.
///
/// A span of at most this many donors is a leaf. The split vector has
/// `2^levels` records, where `levels` is the number of times that span is
/// halved to reach this leaf size.
const MATCHING_LEAF_DONORS: u64 = 16;

/// Bytes of one tree split record: a 32-bit axis plus one binary64 split.
const SPLIT_RECORD_BYTES: u64 = 4 + BINARY64_BYTES;

/// Which exact family an estimate or a limit belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SearchFamily {
    /// Conditional GPDC kernel and its factorization.
    Gpdc,
    /// Exact conditioning-set graph search.
    GraphSearch,
    /// Finite functional enumeration over declared factor cardinalities.
    FunctionalEnumeration,
    /// Exact donor matching.
    Matching,
}

impl SearchFamily {
    /// Stable `snake_case` name used in admission errors.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gpdc => "gpdc",
            Self::GraphSearch => "graph_search",
            Self::FunctionalEnumeration => "functional_enumeration",
            Self::Matching => "matching",
        }
    }
}

impl fmt::Display for SearchFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Declared ceiling for one exact estimate.
///
/// Both axes are present. Zero is a declared ceiling of no bytes or no work,
/// not an absent limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SearchResourceLimit {
    memory_bytes: u64,
    work: u64,
}

impl SearchResourceLimit {
    /// Limit of `memory_bytes` and `work`.
    #[must_use]
    pub const fn new(memory_bytes: u64, work: u64) -> Self {
        Self { memory_bytes, work }
    }

    /// Memory ceiling in bytes.
    #[must_use]
    pub const fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }

    /// Work ceiling.
    #[must_use]
    pub const fn work(self) -> u64 {
        self.work
    }
}

/// Evidence that an approximation was requested as its own mode.
///
/// An empty contract is not evidence. This value is not built from an exact
/// refusal, and holding it does not admit an exact estimate that missed a limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ApproximationEvidence {
    contract: &'static str,
}

impl ApproximationEvidence {
    /// Evidence named by a non-empty approximation contract.
    #[must_use = "an empty contract is not approximation evidence"]
    pub const fn new(contract: &'static str) -> Option<Self> {
        if contract.is_empty() { None } else { Some(Self { contract }) }
    }

    /// Contract name supplied with the request.
    #[must_use]
    pub const fn contract(self) -> &'static str {
        self.contract
    }
}

/// Exact work, or an approximation requested with its own evidence.
///
/// The approximation variant does not mean the exact estimate was too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NumericalMode {
    /// The exact kernel, search, enumeration, or match.
    Exact,
    /// A separate approximation contract.
    Approximation(ApproximationEvidence),
}

impl NumericalMode {
    /// Exact mode.
    #[must_use]
    pub const fn exact() -> Self {
        Self::Exact
    }

    /// Approximation mode for evidence that was requested on its own.
    #[must_use]
    pub const fn approximation(evidence: ApproximationEvidence) -> Self {
        Self::Approximation(evidence)
    }

    /// Whether this is the exact mode.
    #[must_use]
    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }

    /// Evidence, when this is an approximation request.
    #[must_use]
    pub const fn evidence(self) -> Option<ApproximationEvidence> {
        match self {
            Self::Exact => None,
            Self::Approximation(evidence) => Some(evidence),
        }
    }
}

/// Distance charged for an exact match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MatchingDistance {
    /// Absolute difference on one declared scalar.
    Absolute,
    /// Euclidean distance on the declared feature width.
    Euclidean,
}

impl MatchingDistance {
    /// Stable name used in admission errors.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absolute => "absolute",
            Self::Euclidean => "euclidean",
        }
    }
}

impl fmt::Display for MatchingDistance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal from an exact work or memory estimate.
///
/// Estimation and admission allocate nothing and do not start the work.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum SearchBudgetError {
    /// The estimate is larger than the declared limit on memory, work, or both.
    ///
    /// The estimated counts are the computed counts. They are not replaced with zero.
    #[error(
        "{family} estimate of {estimated_bytes} bytes and {estimated_work} work exceeds the \
         {limit_bytes}-byte and {limit_work}-work limit; next attempt: a larger limit or an \
         explicitly requested approximation mode"
    )]
    ExceedsLimit {
        /// Family that was estimated.
        family: SearchFamily,
        /// Estimated bytes. Not zero unless the estimate itself was a declared zero.
        estimated_bytes: u64,
        /// Estimated work.
        estimated_work: u64,
        /// Declared byte limit.
        limit_bytes: u64,
        /// Declared work limit.
        limit_work: u64,
    },
    /// A product does not fit in `u64`.
    ///
    /// The estimate is not reported as zero, and an approximation is not selected.
    #[error(
        "{family} estimate overflows u64 before the work and is not reported as zero; an \
         approximation is not selected; next attempt: a larger limit or an explicitly requested \
         approximation mode"
    )]
    Unrepresentable {
        /// Family whose product overflowed.
        family: SearchFamily,
    },
    /// A conditional GPDC kernel needs rows and a non-empty conditioning set.
    #[error(
        "gpdc kernel factorization requires a positive row count and a non-empty conditioning \
         set; rows {rows}, conditioning columns {conditioning_columns}"
    )]
    GpdcKernelShape {
        /// Declared rows.
        rows: u64,
        /// Declared conditioning columns.
        conditioning_columns: u64,
    },
    /// Exact graph search needs at least one variable.
    #[error("exact graph search requires at least one variable")]
    EmptyGraph,
    /// No factor cardinality was declared.
    ///
    /// Absence is not a one-cell functional.
    #[error(
        "functional enumeration requires a declared factor; an absent factor list is not one cell"
    )]
    MissingFactors,
    /// A factor of cardinality zero is not a product that fits every limit.
    #[error(
        "functional enumeration has a factor of cardinality 0; a zero cardinality is not an \
         empty product that fits the limit"
    )]
    EmptyFactor,
    /// Matching needs a positive feature width, and absolute distance is one-dimensional.
    #[error(
        "exact matching requires a positive dimension, and absolute distance requires one \
         dimension; donors {donors}, queries {queries}, dimensions {dimensions}, distance \
         {distance}"
    )]
    MatchingShape {
        /// Declared donors.
        donors: u64,
        /// Declared queries.
        queries: u64,
        /// Declared feature width.
        dimensions: u64,
        /// Declared distance.
        distance: MatchingDistance,
    },
    /// An exact refusal was turned into an approximation request.
    #[error(
        "switching exact {family} to an approximation because the exact mode does not fit is \
         refused; approximation is a distinct mode with its own evidence and is not a \
         large-data fallback; next attempt: a larger limit or an explicitly requested \
         approximation mode"
    )]
    ApproximationFallbackRefused {
        /// Family whose exact refusal was not converted.
        family: SearchFamily,
    },
    /// Approximation evidence was offered in place of the exact admission.
    #[error(
        "switching exact {family} to approximation `{contract}` is refused; approximation is a \
         distinct mode with its own evidence and is not a large-data fallback; next attempt: a \
         larger limit or an explicitly requested approximation mode"
    )]
    NamedApproximationFallback {
        /// Exact family that was not switched.
        family: SearchFamily,
        /// Contract named by the caller. It was not run.
        contract: &'static str,
    },
}

impl SearchBudgetError {
    /// Refuse to treat this exact-mode refusal as an approximation request.
    ///
    /// # Errors
    ///
    /// Always [`Self::ApproximationFallbackRefused`]. The contract of an
    /// approximation is not created here.
    #[must_use = "an exact refusal is not an approximation request"]
    pub const fn approximation_fallback(self) -> Result<ApproximationEvidence, Self> {
        Err(Self::ApproximationFallbackRefused { family: self.family() })
    }

    const fn family(self) -> SearchFamily {
        match self {
            Self::ExceedsLimit { family, .. }
            | Self::Unrepresentable { family }
            | Self::ApproximationFallbackRefused { family }
            | Self::NamedApproximationFallback { family, .. } => family,
            Self::GpdcKernelShape { .. } => SearchFamily::Gpdc,
            Self::EmptyGraph => SearchFamily::GraphSearch,
            Self::MissingFactors | Self::EmptyFactor => SearchFamily::FunctionalEnumeration,
            Self::MatchingShape { .. } => SearchFamily::Matching,
        }
    }
}

/// Estimated bytes and work of one exact family.
///
/// [`Self::count`] and [`Self::width`] keep the inputs that produced the
/// charge: rows and conditioning columns, nodes and depth, cells and factor
/// width, or donors and feature width.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SearchEstimate {
    family: SearchFamily,
    memory_bytes: u64,
    work: u64,
    count: u64,
    width: u64,
}

impl SearchEstimate {
    /// Family this estimate charges.
    #[must_use]
    pub const fn family(self) -> SearchFamily {
        self.family
    }

    /// Estimated bytes.
    #[must_use]
    pub const fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }

    /// Estimated work.
    #[must_use]
    pub const fn work(self) -> u64 {
        self.work
    }

    /// Primary count: rows, nodes, cells, or donors.
    #[must_use]
    pub const fn count(self) -> u64 {
        self.count
    }

    /// Width: conditioning columns, depth, factor count, or feature dimensions.
    #[must_use]
    pub const fn width(self) -> u64 {
        self.width
    }

    /// Admit this exact estimate under `limit` and `mode`.
    ///
    /// `mode` must be [`NumericalMode::Exact`]. An approximation, including one
    /// named because this estimate would miss `limit`, is refused before the
    /// work. A fitting exact estimate is admitted and still has not been run.
    ///
    /// # Errors
    ///
    /// [`SearchBudgetError::NamedApproximationFallback`] when `mode` is an
    /// approximation. [`SearchBudgetError::ExceedsLimit`] when the exact
    /// estimate is larger than `limit` on either axis. `estimated_bytes` on
    /// that error is this estimate's byte count.
    #[must_use = "a refused admission is not permission to start the work"]
    pub const fn admit(
        self,
        limit: SearchResourceLimit,
        mode: NumericalMode,
    ) -> Result<AdmittedExactSearch, SearchBudgetError> {
        match mode {
            NumericalMode::Exact => {}
            NumericalMode::Approximation(evidence) => {
                return Err(SearchBudgetError::NamedApproximationFallback {
                    family: self.family,
                    contract: evidence.contract(),
                });
            }
        }
        if self.memory_bytes > limit.memory_bytes || self.work > limit.work {
            return Err(SearchBudgetError::ExceedsLimit {
                family: self.family,
                estimated_bytes: self.memory_bytes,
                estimated_work: self.work,
                limit_bytes: limit.memory_bytes,
                limit_work: limit.work,
            });
        }
        Ok(AdmittedExactSearch {
            family: self.family,
            memory_bytes: self.memory_bytes,
            work: self.work,
        })
    }
}

/// Exact work whose estimate fits the declared limit.
///
/// Constructed only by admission. Holding it has not factored a kernel,
/// entered a search, visited a cell, or scanned a donor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmittedExactSearch {
    family: SearchFamily,
    memory_bytes: u64,
    work: u64,
}

impl AdmittedExactSearch {
    /// Family that was admitted.
    #[must_use]
    pub const fn family(self) -> SearchFamily {
        self.family
    }

    /// Admitted byte count.
    #[must_use]
    pub const fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }

    /// Admitted work count.
    #[must_use]
    pub const fn work(self) -> u64 {
        self.work
    }
}

/// Conditional GPDC kernel and factorization.
///
/// Memory is the dense binary64 kernel and its Cholesky factor, both live
/// (`2 × rows² × 8`), plus the standardized conditioning block
/// (`rows × conditioning columns × 8`). Work is the cubic factorization term
/// `rows³` plus the kernel fill `rows² × conditioning columns`. An empty
/// conditioning set is the unconditional distance correlation, which has no
/// kernel and is not this estimate. No row ceiling in this charge selects a
/// low-rank kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GpdcKernel {
    rows: u64,
    conditioning_columns: u64,
}

impl GpdcKernel {
    /// Kernel on `rows` with `conditioning_columns` conditioning columns.
    #[must_use]
    pub const fn new(rows: u64, conditioning_columns: u64) -> Self {
        Self { rows, conditioning_columns }
    }

    /// Byte and work charge of one exact conditional factorization.
    ///
    /// # Errors
    ///
    /// [`SearchBudgetError::GpdcKernelShape`] when `rows` is zero or the
    /// conditioning set is empty. [`SearchBudgetError::Unrepresentable`] when
    /// a product does not fit in `u64`. This does not factor the kernel.
    #[must_use = "an unrepresentable kernel is not a zero-byte factorization"]
    pub const fn estimate(self) -> Result<SearchEstimate, SearchBudgetError> {
        if self.rows == 0 || self.conditioning_columns == 0 {
            return Err(SearchBudgetError::GpdcKernelShape {
                rows: self.rows,
                conditioning_columns: self.conditioning_columns,
            });
        }
        let family = SearchFamily::Gpdc;
        let value_bytes = match layout_bytes::<f64>(family) {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        let Some(rows_squared) = self.rows.checked_mul(self.rows) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(kernel_and_factor) = mul3(rows_squared, 2, value_bytes) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(conditioning) = mul3(self.rows, self.conditioning_columns, value_bytes) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(memory_bytes) = kernel_and_factor.checked_add(conditioning) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(cubic) = rows_squared.checked_mul(self.rows) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(fill) = rows_squared.checked_mul(self.conditioning_columns) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(work) = cubic.checked_add(fill) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        Ok(SearchEstimate {
            family,
            memory_bytes,
            work,
            count: self.rows,
            width: self.conditioning_columns,
        })
    }
}

/// Exact graph search over conditioning sets.
///
/// For every variable, every subset of the other variables of size at most
/// `depth` is charged. Work is that number of sets. Memory is one binary64
/// score per set: dropping sets to fit a byte limit is not this search.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExactGraphSearch {
    nodes: u64,
    depth: u64,
}

impl ExactGraphSearch {
    /// Search on `nodes` variables to conditioning depth `depth`.
    #[must_use]
    pub const fn new(nodes: u64, depth: u64) -> Self {
        Self { nodes, depth }
    }

    /// Byte and work charge of the exact conditioning-set search.
    ///
    /// # Errors
    ///
    /// [`SearchBudgetError::EmptyGraph`] when `nodes` is zero.
    /// [`SearchBudgetError::Unrepresentable`] when a binomial coefficient or
    /// the score bytes do not fit in `u64`. This does not enter the search.
    #[must_use = "an unrepresentable search is not a zero-work search"]
    pub const fn estimate(self) -> Result<SearchEstimate, SearchBudgetError> {
        if self.nodes == 0 {
            return Err(SearchBudgetError::EmptyGraph);
        }
        let family = SearchFamily::GraphSearch;
        let Some(sets_per_node) = conditioning_sets(self.nodes - 1, self.depth) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(work) = self.nodes.checked_mul(sets_per_node) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(memory_bytes) = work.checked_mul(SCORE_BYTES) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        Ok(SearchEstimate { family, memory_bytes, work, count: self.nodes, width: self.depth })
    }
}

/// Finite functional enumeration, one cell per tuple of factor levels.
///
/// The cell count is the product of the declared cardinalities. Memory is one
/// binary64 value per cell. Work is one visit per cell. No factor is not a
/// product of one, and a cardinality of zero is not an empty product.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FunctionalEnumeration {
    factors: u64,
    cells: u64,
    overflow: bool,
    empty_factor: bool,
}

impl Default for FunctionalEnumeration {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalEnumeration {
    /// No factors declared. The product is absent, not one cell.
    #[must_use]
    pub const fn new() -> Self {
        Self { factors: 0, cells: 1, overflow: false, empty_factor: false }
    }

    /// Declare one more factor of `cardinality`.
    ///
    /// Cardinality zero is recorded and refuses the estimate. It does not
    /// collapse the product to zero bytes.
    #[must_use]
    pub const fn factor(mut self, cardinality: u64) -> Self {
        match self.factors.checked_add(1) {
            Some(factors) => self.factors = factors,
            None => self.overflow = true,
        }
        if cardinality == 0 {
            self.empty_factor = true;
            return self;
        }
        if self.empty_factor {
            return self;
        }
        match self.cells.checked_mul(cardinality) {
            Some(cells) => self.cells = cells,
            None => self.overflow = true,
        }
        self
    }

    /// Byte and work charge of visiting every cell.
    ///
    /// # Errors
    ///
    /// [`SearchBudgetError::MissingFactors`] when no factor was declared.
    /// [`SearchBudgetError::EmptyFactor`] when a cardinality is zero.
    /// [`SearchBudgetError::Unrepresentable`] when the product or the byte
    /// charge does not fit in `u64`. This does not visit a cell.
    #[must_use = "an absent factor list is not a one-cell functional"]
    pub const fn estimate(self) -> Result<SearchEstimate, SearchBudgetError> {
        if self.factors == 0 {
            return Err(SearchBudgetError::MissingFactors);
        }
        if self.empty_factor {
            return Err(SearchBudgetError::EmptyFactor);
        }
        let family = SearchFamily::FunctionalEnumeration;
        if self.overflow {
            return Err(SearchBudgetError::Unrepresentable { family });
        }
        let value_bytes = match layout_bytes::<f64>(family) {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        let Some(memory_bytes) = self.cells.checked_mul(value_bytes) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        Ok(SearchEstimate {
            family,
            memory_bytes,
            work: self.cells,
            count: self.cells,
            width: self.factors,
        })
    }
}

/// Exact donor matching.
///
/// Memory retains the binary64 features, one 64-bit donor index, and the
/// search layout. Absolute distance adds a sorted value and a first position
/// per donor. Euclidean distance adds a 64-bit permutation plus the split
/// records of a balanced tree whose leaves hold at most 16 donors. Work is
/// one operation per feature coordinate while the index is built, plus one
/// per coordinate of a full exact scan of every donor for every query. A
/// pruned tree is not a smaller
/// exact charge, and a donor count above a historical cap does not become an
/// approximate neighbor search.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExactMatching {
    donors: u64,
    queries: u64,
    dimensions: u64,
    distance: MatchingDistance,
}

impl ExactMatching {
    /// Match `queries` against `donors` of `dimensions` features.
    #[must_use]
    pub const fn new(
        donors: u64,
        queries: u64,
        dimensions: u64,
        distance: MatchingDistance,
    ) -> Self {
        Self { donors, queries, dimensions, distance }
    }

    /// Byte and work charge of the exact index and the exact scan.
    ///
    /// Zero donors and zero queries are a declared empty match: zero bytes and
    /// zero work, once the distance shape is legal. Absolute distance is legal
    /// only at one dimension.
    ///
    /// # Errors
    ///
    /// [`SearchBudgetError::MatchingShape`] when the dimension is zero or
    /// absolute distance is not one-dimensional.
    /// [`SearchBudgetError::Unrepresentable`] when a product does not fit in
    /// `u64`. This does not scan a donor.
    #[must_use = "an unrepresentable match is not a zero-byte index"]
    pub const fn estimate(self) -> Result<SearchEstimate, SearchBudgetError> {
        let family = SearchFamily::Matching;
        if self.dimensions == 0
            || (matches!(self.distance, MatchingDistance::Absolute) && self.dimensions != 1)
        {
            return Err(SearchBudgetError::MatchingShape {
                donors: self.donors,
                queries: self.queries,
                dimensions: self.dimensions,
                distance: self.distance,
            });
        }
        let value_bytes = match layout_bytes::<f64>(family) {
            Ok(bytes) => bytes,
            Err(error) => return Err(error),
        };
        let Some(features) = mul3(self.donors, self.dimensions, value_bytes) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(row_ids) = self.donors.checked_mul(INDEX_BYTES) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let layout = match self.distance {
            MatchingDistance::Absolute => {
                let Some(layout) = self.donors.checked_mul(value_bytes + INDEX_BYTES) else {
                    return Err(SearchBudgetError::Unrepresentable { family });
                };
                layout
            }
            MatchingDistance::Euclidean => {
                let Some(slots) = euclidean_slots(self.donors) else {
                    return Err(SearchBudgetError::Unrepresentable { family });
                };
                let Some(permutation) = self.donors.checked_mul(INDEX_BYTES) else {
                    return Err(SearchBudgetError::Unrepresentable { family });
                };
                let Some(splits) = slots.checked_mul(SPLIT_RECORD_BYTES) else {
                    return Err(SearchBudgetError::Unrepresentable { family });
                };
                let Some(layout) = permutation.checked_add(splits) else {
                    return Err(SearchBudgetError::Unrepresentable { family });
                };
                layout
            }
        };
        let Some(with_rows) = features.checked_add(row_ids) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(memory_bytes) = with_rows.checked_add(layout) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(coordinates) = self.donors.checked_mul(self.dimensions) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(passes) = self.queries.checked_add(1) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        let Some(work) = coordinates.checked_mul(passes) else {
            return Err(SearchBudgetError::Unrepresentable { family });
        };
        Ok(SearchEstimate {
            family,
            memory_bytes,
            work,
            count: self.donors,
            width: self.dimensions,
        })
    }
}

const fn layout_bytes<T>(family: SearchFamily) -> Result<u64, SearchBudgetError> {
    let bytes = core::mem::size_of::<T>();
    if bytes > 8 {
        return Err(SearchBudgetError::Unrepresentable { family });
    }
    match bytes {
        0 => Ok(0),
        1 => Ok(1),
        2 => Ok(2),
        4 => Ok(4),
        8 => Ok(8),
        _ => Err(SearchBudgetError::Unrepresentable { family }),
    }
}

const fn mul3(a: u64, b: u64, c: u64) -> Option<u64> {
    match a.checked_mul(b) {
        Some(product) => product.checked_mul(c),
        None => None,
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// `C(n, k)`, or `None` when the coefficient does not fit in `u64`.
const fn binom(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let k = if k > n - k { n - k } else { k };
    let mut result = 1u64;
    let mut i = 1u64;
    while i <= k {
        let Some(factor) = (n - k).checked_add(i) else {
            return None;
        };
        let factor_divisor = gcd(factor, i);
        let factor = factor / factor_divisor;
        let i_reduced = i / factor_divisor;
        let result_divisor = gcd(result, i_reduced);
        let result_reduced = result / result_divisor;
        let divisor = i_reduced / result_divisor;
        let Some(product) = result_reduced.checked_mul(factor) else {
            return None;
        };
        if divisor == 0 || product % divisor != 0 {
            return None;
        }
        result = product / divisor;
        i += 1;
    }
    Some(result)
}

/// Subsets of `candidates` items of size at most `depth`.
const fn conditioning_sets(candidates: u64, depth: u64) -> Option<u64> {
    let max_k = if depth < candidates { depth } else { candidates };
    let mut total = 0u64;
    let mut k = 0u64;
    while k <= max_k {
        let Some(term) = binom(candidates, k) else {
            return None;
        };
        let Some(sum) = total.checked_add(term) else {
            return None;
        };
        total = sum;
        k += 1;
    }
    Some(total)
}

/// Heap-ordered slots of a balanced tree over `donors`.
///
/// Zero donors charge no slot. A non-empty tree charges `2^levels` slots,
/// where each level halves the donor span until it is at most
/// [`MATCHING_LEAF_DONORS`].
const fn euclidean_slots(donors: u64) -> Option<u64> {
    if donors == 0 {
        return Some(0);
    }
    let mut size = donors;
    let mut levels = 0u32;
    while size > MATCHING_LEAF_DONORS {
        size = size.div_ceil(2);
        levels += 1;
        if levels >= 63 {
            return None;
        }
    }
    1u64.checked_shl(levels)
}

#[cfg(test)]
mod tests {
    use super::{
        AdmittedExactSearch, ApproximationEvidence, ExactGraphSearch, ExactMatching,
        FunctionalEnumeration, GpdcKernel, MatchingDistance, NumericalMode, SearchBudgetError,
        SearchEstimate, SearchFamily, SearchResourceLimit,
    };

    const NEXT_ATTEMPT: &str =
        "next attempt: a larger limit or an explicitly requested approximation mode";

    fn admit_exact(estimate: SearchEstimate, limit: SearchResourceLimit) -> AdmittedExactSearch {
        estimate.admit(limit, NumericalMode::exact()).expect("exact mode fits")
    }

    fn assert_switch_refused(estimate: SearchEstimate, contract: &'static str) {
        let evidence = ApproximationEvidence::new(contract).expect("named evidence");
        assert!(ApproximationEvidence::new("").is_none());
        assert_eq!(evidence.contract(), contract);
        let mode = NumericalMode::approximation(evidence);
        assert!(!mode.is_exact());
        assert_eq!(mode.evidence().map(ApproximationEvidence::contract), Some(contract));
        let fitting = SearchResourceLimit::new(estimate.memory_bytes(), estimate.work());
        match estimate.admit(fitting, mode) {
            Err(SearchBudgetError::NamedApproximationFallback { family, contract: named }) => {
                assert_eq!(family, estimate.family());
                assert_eq!(named, contract);
                let text =
                    SearchBudgetError::NamedApproximationFallback { family, contract: named }
                        .to_string();
                assert!(text.contains(contract));
                assert!(text.contains("not a large-data fallback"));
                assert!(text.contains(NEXT_ATTEMPT));
            }
            Ok(admitted) => panic!("approximation admitted {} bytes", admitted.memory_bytes()),
            Err(other) => panic!("unexpected switch result: {other}"),
        }
    }

    fn assert_over_limit_refuses_fallback(estimate: SearchEstimate) {
        assert_ne!(estimate.memory_bytes(), 0);
        assert_ne!(estimate.work(), 0);
        let limit = SearchResourceLimit::new(estimate.memory_bytes() - 1, estimate.work());
        match estimate.admit(limit, NumericalMode::exact()) {
            Err(SearchBudgetError::ExceedsLimit {
                family,
                estimated_bytes,
                estimated_work,
                limit_bytes,
                limit_work,
            }) => {
                assert_eq!(family, estimate.family());
                assert_eq!(estimated_bytes, estimate.memory_bytes());
                assert_eq!(estimated_work, estimate.work());
                assert_ne!(estimated_bytes, 0);
                assert_eq!(limit_bytes, limit.memory_bytes());
                assert_eq!(limit_work, limit.work());
                let refusal = SearchBudgetError::ExceedsLimit {
                    family,
                    estimated_bytes,
                    estimated_work,
                    limit_bytes,
                    limit_work,
                };
                let text = refusal.to_string();
                assert!(text.contains(&estimated_bytes.to_string()));
                assert!(text.contains(&estimated_work.to_string()));
                assert!(text.contains(NEXT_ATTEMPT));
                match refusal.approximation_fallback() {
                    Err(SearchBudgetError::ApproximationFallbackRefused { family: refused }) => {
                        assert_eq!(refused, estimate.family());
                        let fallback =
                            SearchBudgetError::ApproximationFallbackRefused { family: refused }
                                .to_string();
                        assert!(
                            fallback.contains("because the exact mode does not fit is refused")
                        );
                        assert!(fallback.contains(NEXT_ATTEMPT));
                    }
                    Ok(_) => panic!("exact refusal produced approximation evidence"),
                    Err(other) => panic!("unexpected fallback: {other}"),
                }
            }
            Ok(admitted) => panic!("admitted {} bytes", admitted.memory_bytes()),
            Err(other) => panic!("unexpected refusal: {other}"),
        }
    }

    #[test]
    fn gpdc_kernel_limit_refuses_before_the_factorization() {
        let kernel = GpdcKernel::new(5, 3).estimate().expect("conditional kernel");
        assert_eq!(kernel.family(), SearchFamily::Gpdc);
        assert_eq!(kernel.count(), 5);
        assert_eq!(kernel.width(), 3);
        let rows_squared = 5 * 5;
        assert_eq!(kernel.memory_bytes(), (2 * rows_squared * 8) + (5 * 3 * 8));
        assert_eq!(kernel.work(), (5 * rows_squared) + (rows_squared * 3));
        let admitted =
            admit_exact(kernel, SearchResourceLimit::new(kernel.memory_bytes(), kernel.work()));
        assert_eq!(admitted.family(), SearchFamily::Gpdc);
        assert_eq!(admitted.memory_bytes(), kernel.memory_bytes());
        assert_eq!(admitted.work(), kernel.work());
        assert_switch_refused(kernel, "gpdc_nystrom");
        assert_over_limit_refuses_fallback(kernel);
        assert_eq!(
            GpdcKernel::new(0, 3).estimate(),
            Err(SearchBudgetError::GpdcKernelShape { rows: 0, conditioning_columns: 3 })
        );
        assert_eq!(
            GpdcKernel::new(4, 0).estimate(),
            Err(SearchBudgetError::GpdcKernelShape { rows: 4, conditioning_columns: 0 })
        );
        assert_eq!(
            GpdcKernel::new(u64::MAX, 1).estimate(),
            Err(SearchBudgetError::Unrepresentable { family: SearchFamily::Gpdc })
        );
    }

    #[test]
    fn exact_graph_search_limit_refuses_before_the_search() {
        let search = ExactGraphSearch::new(5, 2).estimate().expect("depth-2 search");
        assert_eq!(search.family(), SearchFamily::GraphSearch);
        assert_eq!(search.count(), 5);
        assert_eq!(search.width(), 2);
        let sets_per_node = 1 + 4 + 6;
        assert_eq!(search.work(), 5 * sets_per_node);
        assert_eq!(search.memory_bytes(), search.work() * 8);
        let admitted =
            admit_exact(search, SearchResourceLimit::new(search.memory_bytes(), search.work()));
        assert_eq!(admitted.work(), 55);
        assert_eq!(admitted.memory_bytes(), 440);
        let depth_zero = ExactGraphSearch::new(4, 0).estimate().expect("empty sets only");
        assert_eq!(depth_zero.work(), 4);
        assert_eq!(depth_zero.memory_bytes(), 32);
        assert_switch_refused(search, "reduced_graph_search");
        assert_over_limit_refuses_fallback(search);
        assert_eq!(ExactGraphSearch::new(0, 1).estimate(), Err(SearchBudgetError::EmptyGraph));
        let overflow = ExactGraphSearch::new(80, 40).estimate();
        assert_eq!(
            overflow,
            Err(SearchBudgetError::Unrepresentable { family: SearchFamily::GraphSearch })
        );
        let refusal = overflow.expect_err("binomial does not fit");
        assert!(matches!(
            refusal.approximation_fallback(),
            Err(SearchBudgetError::ApproximationFallbackRefused {
                family: SearchFamily::GraphSearch,
            })
        ));
    }

    #[test]
    fn functional_enumeration_limit_refuses_before_the_cells_are_visited() {
        let enumeration = FunctionalEnumeration::new()
            .factor(2)
            .factor(3)
            .factor(5)
            .estimate()
            .expect("2 × 3 × 5 cells");
        assert_eq!(enumeration.family(), SearchFamily::FunctionalEnumeration);
        assert_eq!(enumeration.width(), 3);
        assert_eq!(enumeration.count(), 30);
        assert_eq!(enumeration.work(), 30);
        assert_eq!(enumeration.memory_bytes(), 30 * 8);
        let admitted = admit_exact(
            enumeration,
            SearchResourceLimit::new(enumeration.memory_bytes(), enumeration.work()),
        );
        assert_eq!(admitted.memory_bytes(), 240);
        assert_eq!(admitted.work(), 30);
        assert_switch_refused(enumeration, "monte_carlo_integration");
        assert_over_limit_refuses_fallback(enumeration);
        assert_eq!(FunctionalEnumeration::new().estimate(), Err(SearchBudgetError::MissingFactors));
        assert_eq!(
            FunctionalEnumeration::new().factor(4).factor(0).estimate(),
            Err(SearchBudgetError::EmptyFactor)
        );
        let overflow = FunctionalEnumeration::new().factor(u64::MAX).factor(2).estimate();
        assert_eq!(
            overflow,
            Err(SearchBudgetError::Unrepresentable { family: SearchFamily::FunctionalEnumeration })
        );
        assert!(matches!(
            overflow.expect_err("product does not fit").approximation_fallback(),
            Err(SearchBudgetError::ApproximationFallbackRefused {
                family: SearchFamily::FunctionalEnumeration,
            })
        ));
    }

    #[test]
    fn exact_matching_limit_refuses_before_the_scan() {
        let matching = ExactMatching::new(17, 0, 1, MatchingDistance::Absolute)
            .estimate()
            .expect("scalar index");
        assert_eq!(matching.family(), SearchFamily::Matching);
        assert_eq!(matching.count(), 17);
        assert_eq!(matching.width(), 1);
        assert_eq!(matching.memory_bytes(), (17 * 8) + (17 * 8) + (17 * 16));
        assert_eq!(matching.work(), 17);
        let admitted = admit_exact(
            matching,
            SearchResourceLimit::new(matching.memory_bytes(), matching.work()),
        );
        assert_eq!(admitted.memory_bytes(), 544);
        assert_eq!(admitted.work(), 17);
        let euclidean = ExactMatching::new(17, 3, 2, MatchingDistance::Euclidean)
            .estimate()
            .expect("tree index");
        let slots = 2;
        let features = 17 * 2 * 8;
        let row_ids = 17 * 8;
        let layout = (17 * 8) + (slots * 12);
        assert_eq!(euclidean.memory_bytes(), features + row_ids + layout);
        assert_eq!(euclidean.work(), (17 * 2) + (3 * 17 * 2));
        let empty = ExactMatching::new(0, 0, 1, MatchingDistance::Absolute)
            .estimate()
            .expect("declared empty match");
        assert_eq!(empty.memory_bytes(), 0);
        assert_eq!(empty.work(), 0);
        assert_eq!(
            empty
                .admit(SearchResourceLimit::new(0, 0), NumericalMode::exact())
                .expect("zero fits")
                .work(),
            0
        );
        assert_switch_refused(matching, "approximate_neighbors");
        assert_over_limit_refuses_fallback(matching);
        assert_eq!(
            ExactMatching::new(4, 1, 2, MatchingDistance::Absolute).estimate(),
            Err(SearchBudgetError::MatchingShape {
                donors: 4,
                queries: 1,
                dimensions: 2,
                distance: MatchingDistance::Absolute,
            })
        );
        assert_eq!(
            ExactMatching::new(u64::MAX, 1, 2, MatchingDistance::Euclidean).estimate(),
            Err(SearchBudgetError::Unrepresentable { family: SearchFamily::Matching })
        );
    }
}
