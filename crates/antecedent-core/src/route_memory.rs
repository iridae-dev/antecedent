//! Route memory from declared encodings, copies, and workers.
//!
//! Encoded width is rows times the sum of the declared column widths. A
//! binary32 or 32-bit integer column contributes four bytes; eight bytes is
//! only the width of a declared binary64 or 64-bit column. Masks, design
//! expansion, learner/fold copies, one solver-scratch copy, retained
//! scores/draws, output, and the further scratch copies held by concurrent
//! workers are added to that width. The estimate is their sum.
//!
//! An absent component is not zero. Zero is a declared estimate of no bytes,
//! and only a complete declaration can be admitted. Admission compares that
//! sum with the route budget and returns before any buffer is allocated.
//! A sum that exceeds the budget is refused by name: the next attempt is to
//! lower the declared copies, workers, or retained outputs.
//!
//! Worker count scales solver scratch. One worker costs one scratch copy.
//! Two workers cost two scratch copies: the solver-scratch component is the
//! first copy, and the concurrent-worker component is every copy after it.
//! Those copies are added once.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// Declared encoded width of one column, in bytes.
///
/// [`Self::float64`] and [`Self::int64`] are eight bytes because that encoding
/// was declared. Nothing here widens a narrower width to eight.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EncodedWidth {
    bytes: u64,
}

impl EncodedWidth {
    /// IEEE-754 binary64, eight bytes.
    #[must_use]
    pub const fn float64() -> Self {
        Self { bytes: 8 }
    }

    /// IEEE-754 binary32, four bytes.
    #[must_use]
    pub const fn float32() -> Self {
        Self { bytes: 4 }
    }

    /// Signed 64-bit integer, eight bytes.
    #[must_use]
    pub const fn int64() -> Self {
        Self { bytes: 8 }
    }

    /// Signed 32-bit integer, four bytes.
    #[must_use]
    pub const fn int32() -> Self {
        Self { bytes: 4 }
    }

    /// Width of `bytes`. Zero is rejected: a missing width is not a zero-width column.
    #[must_use = "a zero width is not a declared encoding"]
    pub const fn new(bytes: u64) -> Option<Self> {
        if bytes == 0 { None } else { Some(Self { bytes }) }
    }

    /// Declared width in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Bytes in a `rows` by `columns` table at this width.
    ///
    /// `None` when the product overflows. Overflow is not reported as zero,
    /// and a four-byte width is not widened to eight.
    #[must_use = "an overflowing table is not a zero-byte design"]
    pub const fn table_bytes(self, rows: u64, columns: u64) -> Option<u64> {
        let Some(cells) = rows.checked_mul(columns) else {
            return None;
        };
        cells.checked_mul(self.bytes)
    }
}

impl fmt::Display for EncodedWidth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}B", self.bytes)
    }
}

/// Which route-memory component was not declared.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RouteMemoryComponent {
    /// Rows times the declared column widths.
    EncodedWidth,
    /// Mask bytes.
    Masks,
    /// Expanded design bytes, after categories or splines.
    DesignExpansion,
    /// Learner and fold copies.
    LearnerFoldCopies,
    /// One solver scratch copy.
    SolverScratch,
    /// Retained scores and draws.
    RetainedScoresDraws,
    /// Output bytes.
    Output,
    /// Scratch copies held by workers after the first.
    ConcurrentWorkers,
}

impl RouteMemoryComponent {
    /// Stable `snake_case` name used in admission errors.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EncodedWidth => "encoded_width",
            Self::Masks => "masks",
            Self::DesignExpansion => "design_expansion",
            Self::LearnerFoldCopies => "learner_fold_copies",
            Self::SolverScratch => "solver_scratch",
            Self::RetainedScoresDraws => "retained_scores_draws",
            Self::Output => "output",
            Self::ConcurrentWorkers => "concurrent_workers",
        }
    }
}

impl fmt::Display for RouteMemoryComponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Byte budget for one route.
///
/// Admission treats this as the limit an unavoidable allocation must fit in.
/// It does not read a soft or hard execution limit on its own.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RouteBudget {
    bytes: u64,
}

impl RouteBudget {
    /// Budget of `bytes`.
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self { bytes }
    }

    /// Budget in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Budget from an optional caller limit. No limit is the largest budget,
    /// which admits any sum that is representable.
    #[must_use]
    pub const fn from_limit(limit: Option<u64>) -> Self {
        match limit {
            Some(bytes) => Self::new(bytes),
            None => Self::new(u64::MAX),
        }
    }
}

/// Refusal from route-memory estimation or admission.
///
/// Estimation and admission allocate nothing. A refusal is the decision not
/// to allocate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RouteMemoryError {
    /// A component was not declared. Absence is not a zero-byte component.
    #[error(
        "route memory admission requires a {component} estimate; an absent component is not zero"
    )]
    MissingComponent {
        /// Component that was not declared.
        component: RouteMemoryComponent,
    },
    /// The declared sum is larger than the route budget.
    ///
    /// `estimated_bytes` is that sum. It is not replaced with zero.
    #[error(
        "route memory estimate of {estimated_bytes} bytes exceeds the {budget_bytes}-byte route \
         budget; next attempt: lower the declared copies, workers, or retained outputs"
    )]
    ExceedsBudget {
        /// Sum of the declared components.
        estimated_bytes: u64,
        /// Route budget the sum exceeded.
        budget_bytes: u64,
    },
    /// A product or the component sum does not fit in `u64`.
    ///
    /// The allocation is not reported as zero bytes.
    #[error(
        "route memory estimate overflows u64 before allocation; next attempt: lower the declared \
         copies, workers, or retained outputs"
    )]
    Unrepresentable,
    /// Workers were declared as zero, so scratch copies cannot be charged.
    #[error(
        "route memory requires at least one worker to hold a solver scratch copy; next attempt: \
         declare the workers"
    )]
    ZeroWorkers,
}

/// Declared inputs of a route-memory estimate.
///
/// Every component starts absent. [`Self::estimate`] refuses the first absent
/// component. A declared zero is present. [`Default`] is that absence, not a
/// zero-byte route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RouteMemoryDeclaration {
    rows: Option<u64>,
    column_count: u64,
    width_sum: u64,
    encoded_overflow: bool,
    masks: Option<u64>,
    design_expansion: Option<u64>,
    learner_copies: Option<u64>,
    learner_bytes_per_copy: Option<u64>,
    solver_scratch: Option<u64>,
    retained_scores_draws: Option<u64>,
    output: Option<u64>,
    workers: Option<u64>,
}

impl Default for RouteMemoryDeclaration {
    /// Every component absent. Absence is not a zero-byte estimate.
    fn default() -> Self {
        Self::new()
    }
}

impl RouteMemoryDeclaration {
    /// Declaration with every component absent.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            rows: None,
            column_count: 0,
            width_sum: 0,
            encoded_overflow: false,
            masks: None,
            design_expansion: None,
            learner_copies: None,
            learner_bytes_per_copy: None,
            solver_scratch: None,
            retained_scores_draws: None,
            output: None,
            workers: None,
        }
    }

    /// Row count used to scale encoded width.
    ///
    /// Zero rows is a declared empty table. Leaving this unset leaves encoded
    /// width absent.
    #[must_use]
    pub const fn rows(mut self, rows: u64) -> Self {
        self.rows = Some(rows);
        self
    }

    /// Add one column at its declared encoded width.
    ///
    /// The column contributes [`EncodedWidth::bytes`], not a nominal eight.
    #[must_use]
    pub const fn encoded_column(self, width: EncodedWidth) -> Self {
        self.encoded_columns(width, 1)
    }

    /// Add `count` columns at one declared encoded width.
    ///
    /// Each column contributes [`EncodedWidth::bytes`]. A count that is zero
    /// declares no column, so encoded width stays absent until one is added.
    #[must_use]
    pub const fn encoded_columns(mut self, width: EncodedWidth, count: u64) -> Self {
        let columns = self.column_count.checked_add(count);
        let added = width.bytes().checked_mul(count);
        let sum = match added {
            Some(added) => self.width_sum.checked_add(added),
            None => None,
        };
        match (columns, sum) {
            (Some(columns), Some(sum)) => {
                self.column_count = columns;
                self.width_sum = sum;
            }
            _ => self.encoded_overflow = true,
        }
        self
    }

    /// Declare mask bytes. Zero is a declared empty mask, not an absent one.
    #[must_use]
    pub const fn masks(mut self, bytes: u64) -> Self {
        self.masks = Some(bytes);
        self
    }

    /// Declare expanded-design bytes. Zero is a declared empty expansion.
    ///
    /// Category and spline expansion belong in this count, at the encoded
    /// width of the expanded columns. [`EncodedWidth::table_bytes`] builds
    /// that count without substituting eight.
    #[must_use]
    pub const fn design_expansion(mut self, bytes: u64) -> Self {
        self.design_expansion = Some(bytes);
        self
    }

    /// Declare learner and fold copies.
    ///
    /// The component is `copies × bytes_per_copy`. Calling this declares the
    /// component, including when both arguments are zero.
    #[must_use]
    pub const fn learner_fold_copies(mut self, copies: u64, bytes_per_copy: u64) -> Self {
        self.learner_copies = Some(copies);
        self.learner_bytes_per_copy = Some(bytes_per_copy);
        self
    }

    /// Declare the bytes of one solver scratch copy.
    #[must_use]
    pub const fn solver_scratch(mut self, bytes: u64) -> Self {
        self.solver_scratch = Some(bytes);
        self
    }

    /// Declare retained score and draw bytes.
    #[must_use]
    pub const fn retained_scores_draws(mut self, bytes: u64) -> Self {
        self.retained_scores_draws = Some(bytes);
        self
    }

    /// Declare output bytes.
    #[must_use]
    pub const fn output(mut self, bytes: u64) -> Self {
        self.output = Some(bytes);
        self
    }

    /// Declare concurrent workers.
    ///
    /// Each worker costs one solver scratch copy. Zero is not a worker count.
    #[must_use]
    pub const fn workers(mut self, workers: u64) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Sum the declared components.
    ///
    /// # Errors
    ///
    /// [`RouteMemoryError::MissingComponent`] for the first absent component,
    /// in encoded-width, masks, design-expansion, learner/fold copies,
    /// solver-scratch, retained scores/draws, output, concurrent-workers order.
    /// [`RouteMemoryError::ZeroWorkers`] when workers were declared as zero.
    /// [`RouteMemoryError::Unrepresentable`] when a product or the sum
    /// overflows `u64`. This does not allocate.
    #[must_use = "a missing component refuses the estimate; absence is not zero bytes"]
    pub const fn estimate(self) -> Result<RouteMemoryEstimate, RouteMemoryError> {
        let Some(rows) = self.rows else {
            return Err(missing(RouteMemoryComponent::EncodedWidth));
        };
        if self.column_count == 0 {
            return Err(missing(RouteMemoryComponent::EncodedWidth));
        }
        let Some(masks) = self.masks else {
            return Err(missing(RouteMemoryComponent::Masks));
        };
        let Some(design_expansion) = self.design_expansion else {
            return Err(missing(RouteMemoryComponent::DesignExpansion));
        };
        let Some(copies) = self.learner_copies else {
            return Err(missing(RouteMemoryComponent::LearnerFoldCopies));
        };
        let Some(bytes_per_copy) = self.learner_bytes_per_copy else {
            return Err(missing(RouteMemoryComponent::LearnerFoldCopies));
        };
        let Some(scratch) = self.solver_scratch else {
            return Err(missing(RouteMemoryComponent::SolverScratch));
        };
        let Some(retained_scores_draws) = self.retained_scores_draws else {
            return Err(missing(RouteMemoryComponent::RetainedScoresDraws));
        };
        let Some(output) = self.output else {
            return Err(missing(RouteMemoryComponent::Output));
        };
        let Some(workers) = self.workers else {
            return Err(missing(RouteMemoryComponent::ConcurrentWorkers));
        };
        if workers == 0 {
            return Err(RouteMemoryError::ZeroWorkers);
        }
        if self.encoded_overflow {
            return Err(RouteMemoryError::Unrepresentable);
        }
        let Some(encoded_width) = rows.checked_mul(self.width_sum) else {
            return Err(RouteMemoryError::Unrepresentable);
        };
        let Some(learner_fold_copies) = copies.checked_mul(bytes_per_copy) else {
            return Err(RouteMemoryError::Unrepresentable);
        };
        let extra_workers = workers - 1;
        let Some(concurrent_workers) = extra_workers.checked_mul(scratch) else {
            return Err(RouteMemoryError::Unrepresentable);
        };
        let Some(total) = component_sum([
            encoded_width,
            masks,
            design_expansion,
            learner_fold_copies,
            scratch,
            retained_scores_draws,
            output,
            concurrent_workers,
        ]) else {
            return Err(RouteMemoryError::Unrepresentable);
        };
        Ok(RouteMemoryEstimate {
            encoded_width,
            masks,
            design_expansion,
            learner_fold_copies,
            solver_scratch: scratch,
            retained_scores_draws,
            output,
            concurrent_workers,
            total,
            copies,
            workers,
        })
    }

    /// Estimate, then admit that sum under `budget`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::estimate`], plus [`RouteMemoryError::ExceedsBudget`]
    /// when the sum is larger than `budget`. The estimated byte count on that
    /// error is the sum. Admission returns before any buffer is allocated.
    #[must_use = "a refused admission is not permission to allocate"]
    pub const fn admit(self, budget: RouteBudget) -> Result<AdmittedRouteMemory, RouteMemoryError> {
        match self.estimate() {
            Ok(estimate) => estimate.admit(budget),
            Err(error) => Err(error),
        }
    }
}

/// Fully declared route memory.
///
/// [`Self::total`] is the sum of the eight components. Solver scratch is one
/// copy. Concurrent workers are the remaining copies, so
/// `solver_scratch + concurrent_workers` equals `workers` scratch copies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RouteMemoryEstimate {
    encoded_width: u64,
    masks: u64,
    design_expansion: u64,
    learner_fold_copies: u64,
    solver_scratch: u64,
    retained_scores_draws: u64,
    output: u64,
    concurrent_workers: u64,
    total: u64,
    copies: u64,
    workers: u64,
}

impl RouteMemoryEstimate {
    /// Encoded column bytes: rows times the sum of declared widths.
    #[must_use]
    pub const fn encoded_width(self) -> u64 {
        self.encoded_width
    }

    /// Declared mask bytes.
    #[must_use]
    pub const fn masks(self) -> u64 {
        self.masks
    }

    /// Declared design-expansion bytes.
    #[must_use]
    pub const fn design_expansion(self) -> u64 {
        self.design_expansion
    }

    /// Learner and fold copy bytes: copies times bytes per copy.
    #[must_use]
    pub const fn learner_fold_copies(self) -> u64 {
        self.learner_fold_copies
    }

    /// Bytes of one solver scratch copy.
    #[must_use]
    pub const fn solver_scratch(self) -> u64 {
        self.solver_scratch
    }

    /// Declared retained score and draw bytes.
    #[must_use]
    pub const fn retained_scores_draws(self) -> u64 {
        self.retained_scores_draws
    }

    /// Declared output bytes.
    #[must_use]
    pub const fn output(self) -> u64 {
        self.output
    }

    /// Scratch copies after the first: `(workers - 1) ×` one scratch copy.
    #[must_use]
    pub const fn concurrent_workers(self) -> u64 {
        self.concurrent_workers
    }

    /// The eight components in declaration order, each with its name.
    ///
    /// Their sum is [`Self::total`]. Solver scratch is one copy and the
    /// concurrent-worker component is every copy after it.
    #[must_use]
    pub const fn components(self) -> [(RouteMemoryComponent, u64); 8] {
        [
            (RouteMemoryComponent::EncodedWidth, self.encoded_width),
            (RouteMemoryComponent::Masks, self.masks),
            (RouteMemoryComponent::DesignExpansion, self.design_expansion),
            (RouteMemoryComponent::LearnerFoldCopies, self.learner_fold_copies),
            (RouteMemoryComponent::SolverScratch, self.solver_scratch),
            (RouteMemoryComponent::RetainedScoresDraws, self.retained_scores_draws),
            (RouteMemoryComponent::Output, self.output),
            (RouteMemoryComponent::ConcurrentWorkers, self.concurrent_workers),
        ]
    }

    /// Declared learner and fold copy count.
    #[must_use]
    pub const fn copy_count(self) -> u64 {
        self.copies
    }

    /// Declared worker count. At least one.
    #[must_use]
    pub const fn worker_count(self) -> u64 {
        self.workers
    }

    /// Solver scratch held by every worker.
    ///
    /// This is [`Self::worker_count`] times one scratch copy. Two workers cost
    /// two scratch copies. The first copy is [`Self::solver_scratch`]; the
    /// rest is [`Self::concurrent_workers`]. Their sum fits because the route
    /// total does.
    #[must_use]
    pub const fn scratch_copy_bytes(self) -> u64 {
        self.solver_scratch.saturating_add(self.concurrent_workers)
    }

    /// Sum of the eight components.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.total
    }

    /// Admit this estimate under `budget`.
    ///
    /// A sum equal to the budget is admitted. A larger sum is refused. The
    /// refusal carries the sum and names the next attempt: lower the declared
    /// copies, workers, or retained outputs. This does not allocate.
    ///
    /// # Errors
    ///
    /// [`RouteMemoryError::ExceedsBudget`] when [`Self::total`] is greater
    /// than `budget`. `estimated_bytes` is that total, not zero.
    #[must_use = "a refused admission is not permission to allocate"]
    pub const fn admit(self, budget: RouteBudget) -> Result<AdmittedRouteMemory, RouteMemoryError> {
        if self.total > budget.bytes {
            Err(RouteMemoryError::ExceedsBudget {
                estimated_bytes: self.total,
                budget_bytes: budget.bytes,
            })
        } else {
            Ok(AdmittedRouteMemory { estimate: self })
        }
    }
}

/// Route memory whose sum fits in the route budget.
///
/// Constructed only by admission. Holding this value has not allocated the
/// bytes it describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmittedRouteMemory {
    estimate: RouteMemoryEstimate,
}

impl AdmittedRouteMemory {
    /// Estimate that was admitted.
    #[must_use]
    pub const fn estimate(self) -> RouteMemoryEstimate {
        self.estimate
    }

    /// Admitted byte count. This is the component sum.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.estimate.total
    }
}

/// Structure of one route, from which every memory component is quoted.
///
/// The profile holds what the route will actually hold: the rows and encoded
/// columns of its design, whether the design is gathered through a
/// materialized expansion, the cross-fitting folds and learners, the
/// resampling replicates and posterior draws it retains, the workers it runs,
/// and any byte estimates its physical operators declared. [`Self::declaration`]
/// turns it into a complete [`RouteMemoryDeclaration`], so no component is
/// left absent and none is a nominal row-by-column constant.
///
/// Components are quoted as follows, with `w` the declared element width:
///
/// * encoded width: `rows × columns × w`;
/// * masks: one validity bit per cell, `⌈rows / 8⌉ × columns` bytes;
/// * design expansion: a second `rows × columns × w` when the design is
///   gathered through a materialized expansion, none when it is borrowed;
/// * learner/fold copies: `folds × learners` copies of the training design,
///   `(rows − ⌈rows / folds⌉) × columns × w` bytes each, none without folds;
/// * solver scratch: a factorization copy of the design, the Gram matrix, the
///   right-hand side and the residual, `(rows × columns + columns² + columns + rows) × w`;
/// * retained scores and draws: out-of-fold predictions `rows × learners × w`
///   when cross-fitting, `replicates × w`, and `draws × columns × w`;
/// * output: the per-row scores, coefficients, replicates and draws published,
///   `(rows + columns + replicates + draws) × w`;
/// * workers: the declared count, each holding one scratch copy.
///
/// A byte estimate the physical operators declared raises its component to
/// the declared figure when that figure is larger than the quote. A declared
/// figure never lowers a quoted one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RouteMemoryProfile {
    rows: u64,
    columns: u64,
    width: EncodedWidth,
    expanded: bool,
    folds: u64,
    learners: u64,
    replicates: u64,
    draws: u64,
    workers: u64,
    declared_scratch: Option<u64>,
    declared_retained: Option<u64>,
    declared_output: Option<u64>,
}

impl RouteMemoryProfile {
    /// A design of `rows` by `columns` binary64 columns, borrowed from its
    /// source, with no cross-fitting, resampling or draws, on one worker.
    #[must_use]
    pub const fn design(rows: u64, columns: u64) -> Self {
        Self {
            rows,
            columns,
            width: EncodedWidth::float64(),
            expanded: false,
            folds: 0,
            learners: 0,
            replicates: 0,
            draws: 0,
            workers: 1,
            declared_scratch: None,
            declared_retained: None,
            declared_output: None,
        }
    }

    /// Encoded width of every design column.
    #[must_use]
    pub const fn width(mut self, width: EncodedWidth) -> Self {
        self.width = width;
        self
    }

    /// Whether the design is gathered through a materialized expansion that
    /// holds the selected columns while the final design is assembled.
    #[must_use]
    pub const fn expanded(mut self, expanded: bool) -> Self {
        self.expanded = expanded;
        self
    }

    /// Cross-fit over `folds` folds, fitting `learners` nuisance learners on
    /// each training split. Fewer than two folds is no cross-fitting.
    #[must_use]
    pub const fn cross_fitting(mut self, folds: u64, learners: u64) -> Self {
        self.folds = folds;
        self.learners = learners;
        self
    }

    /// Retain one estimate per resampling replicate.
    #[must_use]
    pub const fn resampling(mut self, replicates: u64) -> Self {
        self.replicates = replicates;
        self
    }

    /// Retain `draws` posterior draws of every design coefficient.
    #[must_use]
    pub const fn posterior_draws(mut self, draws: u64) -> Self {
        self.draws = draws;
        self
    }

    /// Run on `workers` workers, each holding one scratch copy.
    #[must_use]
    pub const fn workers(mut self, workers: u64) -> Self {
        self.workers = workers;
        self
    }

    /// Byte estimates the physical operators declared in total. `None` is an
    /// operator that declared nothing, which leaves the quoted figure alone.
    #[must_use]
    pub const fn declared_operator_bytes(
        mut self,
        scratch: Option<u64>,
        retained: Option<u64>,
        output: Option<u64>,
    ) -> Self {
        self.declared_scratch = scratch;
        self.declared_retained = retained;
        self.declared_output = output;
        self
    }

    /// Rows of the design.
    #[must_use]
    pub const fn rows(self) -> u64 {
        self.rows
    }

    /// Encoded columns of the design.
    #[must_use]
    pub const fn columns(self) -> u64 {
        self.columns
    }

    /// Bytes of the final design: `rows × columns × w`.
    ///
    /// # Errors
    ///
    /// [`RouteMemoryError::Unrepresentable`] when the product overflows.
    pub const fn design_bytes(self) -> Result<u64, RouteMemoryError> {
        match self.width.table_bytes(self.rows, self.columns) {
            Some(bytes) => Ok(bytes),
            None => Err(RouteMemoryError::Unrepresentable),
        }
    }

    /// The complete declaration this profile quotes.
    ///
    /// # Errors
    ///
    /// [`RouteMemoryError::Unrepresentable`] when a product or sum overflows
    /// `u64`. [`RouteMemoryError::MissingComponent`] when the design has no
    /// column, which leaves encoded width absent.
    pub fn declaration(self) -> Result<RouteMemoryDeclaration, RouteMemoryError> {
        let w = self.width.bytes();
        let design = self.design_bytes()?;
        let masks = checked(self.rows.div_ceil(8).checked_mul(self.columns))?;
        let expansion = if self.expanded { design } else { 0 };
        let (copies, per_copy) = if self.folds >= 2 {
            let held_out = self.rows.div_ceil(self.folds);
            let training = self.width.table_bytes(self.rows - held_out, self.columns);
            (checked(self.folds.checked_mul(self.learners))?, checked(training)?)
        } else {
            (0, 0)
        };
        let gram = checked(self.columns.checked_mul(self.columns))?;
        let solver_cells = sum_checked([
            checked(self.rows.checked_mul(self.columns))?,
            gram,
            self.columns,
            self.rows,
        ])?;
        let scratch = checked(solver_cells.checked_mul(w))?;
        let predictions = if self.folds >= 2 {
            checked(self.rows.checked_mul(self.learners).and_then(|cells| cells.checked_mul(w)))?
        } else {
            0
        };
        let draw_cells = checked(self.draws.checked_mul(self.columns))?;
        let retained = sum_checked([
            predictions,
            checked(self.replicates.checked_mul(w))?,
            checked(draw_cells.checked_mul(w))?,
        ])?;
        let output_cells = sum_checked([self.rows, self.columns, self.replicates, self.draws])?;
        let output = checked(output_cells.checked_mul(w))?;
        Ok(RouteMemoryDeclaration::new()
            .rows(self.rows)
            .encoded_columns(self.width, self.columns)
            .masks(masks)
            .design_expansion(expansion)
            .learner_fold_copies(copies, per_copy)
            .solver_scratch(self.declared_scratch.map_or(scratch, |declared| declared.max(scratch)))
            .retained_scores_draws(
                self.declared_retained.map_or(retained, |declared| declared.max(retained)),
            )
            .output(self.declared_output.map_or(output, |declared| declared.max(output)))
            .workers(self.workers))
    }

    /// The estimate of [`Self::declaration`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::declaration`] and [`RouteMemoryDeclaration::estimate`].
    pub fn estimate(self) -> Result<RouteMemoryEstimate, RouteMemoryError> {
        self.declaration()?.estimate()
    }

    /// Estimate, then admit under `budget`. Returns before any buffer is allocated.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::estimate`], and [`RouteMemoryError::ExceedsBudget`].
    pub fn admit(self, budget: RouteBudget) -> Result<AdmittedRouteMemory, RouteMemoryError> {
        self.declaration()?.admit(budget)
    }
}

fn checked(value: Option<u64>) -> Result<u64, RouteMemoryError> {
    value.ok_or(RouteMemoryError::Unrepresentable)
}

fn sum_checked<const N: usize>(parts: [u64; N]) -> Result<u64, RouteMemoryError> {
    parts
        .into_iter()
        .try_fold(0u64, |total, part| total.checked_add(part))
        .ok_or(RouteMemoryError::Unrepresentable)
}

const fn missing(component: RouteMemoryComponent) -> RouteMemoryError {
    RouteMemoryError::MissingComponent { component }
}

const fn component_sum(parts: [u64; 8]) -> Option<u64> {
    let mut total = 0u64;
    let mut index = 0;
    while index < parts.len() {
        match total.checked_add(parts[index]) {
            Some(sum) => total = sum,
            None => return None,
        }
        index += 1;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::{
        AdmittedRouteMemory, EncodedWidth, RouteBudget, RouteMemoryComponent,
        RouteMemoryDeclaration, RouteMemoryError, RouteMemoryEstimate, RouteMemoryProfile,
    };

    fn cross_fitted_profile() -> RouteMemoryProfile {
        RouteMemoryProfile::design(100, 4)
            .expanded(true)
            .cross_fitting(5, 2)
            .resampling(10)
            .posterior_draws(50)
            .workers(3)
    }

    #[test]
    fn profile_quotes_each_component_from_the_route_structure() {
        let estimate = cross_fitted_profile().estimate().expect("complete profile");
        assert_eq!(estimate.encoded_width(), 100 * 4 * 8);
        assert_eq!(estimate.masks(), 13 * 4, "one validity bit per cell");
        assert_eq!(estimate.design_expansion(), 100 * 4 * 8, "materialized expansion");
        assert_eq!(estimate.copy_count(), 5 * 2);
        assert_eq!(estimate.learner_fold_copies(), 10 * (80 * 4 * 8), "training split per copy");
        assert_eq!(estimate.solver_scratch(), (400 + 16 + 4 + 100) * 8);
        assert_eq!(estimate.retained_scores_draws(), 100 * 2 * 8 + 10 * 8 + 50 * 4 * 8);
        assert_eq!(estimate.output(), (100 + 4 + 10 + 50) * 8);
        assert_eq!(estimate.concurrent_workers(), 2 * estimate.solver_scratch());
        assert_eq!(estimate.worker_count(), 3);
        assert_eq!(estimate.total(), 49_124);
        let summed: u64 = estimate.components().iter().map(|(_, bytes)| *bytes).sum();
        assert_eq!(summed, estimate.total());
        assert_eq!(estimate.components()[0].0, RouteMemoryComponent::EncodedWidth);
    }

    #[test]
    fn a_borrowed_design_without_cross_fitting_holds_no_expansion_or_fold_copies() {
        let borrowed = RouteMemoryProfile::design(100, 4).estimate().expect("complete profile");
        assert_eq!(borrowed.design_expansion(), 0);
        assert_eq!(borrowed.learner_fold_copies(), 0);
        assert_eq!(borrowed.copy_count(), 0);
        assert_eq!(borrowed.retained_scores_draws(), 0);
        assert_eq!(borrowed.concurrent_workers(), 0);
        let expanded =
            RouteMemoryProfile::design(100, 4).expanded(true).estimate().expect("complete profile");
        assert_eq!(expanded.total() - borrowed.total(), 100 * 4 * 8);
        let narrow = RouteMemoryProfile::design(100, 4)
            .width(EncodedWidth::float32())
            .estimate()
            .expect("complete profile");
        assert_eq!(narrow.encoded_width(), 100 * 4 * 4, "a binary32 design is not widened");
        assert!(narrow.total() < borrowed.total());
    }

    #[test]
    fn declared_operator_bytes_raise_a_component_and_never_lower_it() {
        let quoted = RouteMemoryProfile::design(100, 4).estimate().expect("complete profile");
        let raised = RouteMemoryProfile::design(100, 4)
            .declared_operator_bytes(Some(1 << 20), Some(1 << 21), Some(1 << 22))
            .estimate()
            .expect("complete profile");
        assert_eq!(raised.solver_scratch(), 1 << 20);
        assert_eq!(raised.retained_scores_draws(), 1 << 21);
        assert_eq!(raised.output(), 1 << 22);
        let lowered = RouteMemoryProfile::design(100, 4)
            .declared_operator_bytes(Some(1), Some(0), Some(1))
            .estimate()
            .expect("complete profile");
        assert_eq!(lowered, quoted, "a declared figure below the quote leaves the quote");
    }

    #[test]
    fn profile_admission_refuses_by_sum_and_a_lower_declared_copy_set_admits() {
        let profile = cross_fitted_profile();
        let estimate = profile.estimate().expect("complete profile");
        let refused = profile.admit(RouteBudget::new(estimate.total() - 1));
        assert_eq!(
            refused,
            Err(RouteMemoryError::ExceedsBudget {
                estimated_bytes: estimate.total(),
                budget_bytes: estimate.total() - 1,
            })
        );
        assert!(profile.admit(RouteBudget::new(estimate.total())).is_ok());
        let fewer_workers = profile.workers(1).estimate().expect("complete profile");
        let fewer_folds = profile.cross_fitting(2, 1).estimate().expect("complete profile");
        let fewer_draws = profile.posterior_draws(0).estimate().expect("complete profile");
        for lower in [fewer_workers, fewer_folds, fewer_draws] {
            assert!(lower.total() < estimate.total());
            assert!(profile.admit(RouteBudget::new(lower.total())).is_err());
        }
        assert_eq!(RouteBudget::from_limit(None).bytes(), u64::MAX);
        assert_eq!(RouteBudget::from_limit(Some(7)).bytes(), 7);
    }

    #[test]
    fn a_design_without_columns_is_not_a_zero_byte_design() {
        assert_eq!(
            RouteMemoryProfile::design(100, 0).estimate(),
            Err(RouteMemoryError::MissingComponent {
                component: RouteMemoryComponent::EncodedWidth,
            })
        );
        assert_eq!(
            RouteMemoryProfile::design(u64::MAX, 2).estimate(),
            Err(RouteMemoryError::Unrepresentable)
        );
    }

    fn fully_declared() -> RouteMemoryEstimate {
        RouteMemoryDeclaration::new()
            .rows(10)
            .encoded_column(EncodedWidth::float32())
            .encoded_column(EncodedWidth::float32())
            .masks(3)
            .design_expansion(7)
            .learner_fold_copies(4, 5)
            .solver_scratch(11)
            .retained_scores_draws(13)
            .output(17)
            .workers(2)
            .estimate()
            .expect("fully declared")
    }

    fn component_total(estimate: RouteMemoryEstimate) -> u64 {
        let parts = [
            estimate.encoded_width(),
            estimate.masks(),
            estimate.design_expansion(),
            estimate.learner_fold_copies(),
            estimate.solver_scratch(),
            estimate.retained_scores_draws(),
            estimate.output(),
            estimate.concurrent_workers(),
        ];
        let mut total = 0u64;
        for part in parts {
            total = total.checked_add(part).expect("component sum fits");
        }
        total
    }

    #[test]
    fn fully_declared_estimate_equals_the_sum() {
        let estimate = fully_declared();
        assert_eq!(EncodedWidth::float32().bytes(), 4);
        assert_eq!(EncodedWidth::int32().bytes(), 4);
        assert_eq!(EncodedWidth::float64().bytes(), 8);
        assert_eq!(EncodedWidth::int64().bytes(), 8);
        assert!(EncodedWidth::new(0).is_none());
        assert_eq!(EncodedWidth::float32().table_bytes(10, 2), Some(80));
        assert_eq!(EncodedWidth::float64().table_bytes(10, 2), Some(160));
        assert_eq!(estimate.encoded_width(), 10 * (4 + 4));
        assert_ne!(estimate.encoded_width(), 10 * 2 * EncodedWidth::float64().bytes());
        assert_eq!(estimate.masks(), 3);
        assert_eq!(estimate.design_expansion(), 7);
        assert_eq!(estimate.learner_fold_copies(), 4 * 5);
        assert_eq!(estimate.copy_count(), 4);
        assert_eq!(estimate.solver_scratch(), 11);
        assert_eq!(estimate.retained_scores_draws(), 13);
        assert_eq!(estimate.output(), 17);
        assert_eq!(estimate.concurrent_workers(), 11);
        assert_eq!(estimate.total(), component_total(estimate));
        assert_eq!(estimate.total(), 80 + 3 + 7 + 20 + 11 + 13 + 17 + 11);
    }

    #[test]
    fn missing_component_refuses() {
        let absent = RouteMemoryDeclaration::new().estimate();
        assert_eq!(
            absent,
            Err(RouteMemoryError::MissingComponent {
                component: RouteMemoryComponent::EncodedWidth,
            })
        );

        let missing_masks = RouteMemoryDeclaration::new()
            .rows(1)
            .encoded_column(EncodedWidth::float64())
            .design_expansion(0)
            .learner_fold_copies(0, 0)
            .solver_scratch(0)
            .retained_scores_draws(0)
            .output(0)
            .workers(1);
        assert_eq!(
            missing_masks.estimate(),
            Err(RouteMemoryError::MissingComponent { component: RouteMemoryComponent::Masks })
        );

        let declared_zero_masks = missing_masks.masks(0).estimate().expect("declared zero mask");
        assert_eq!(declared_zero_masks.masks(), 0);
        assert_eq!(declared_zero_masks.encoded_width(), EncodedWidth::float64().bytes());

        let missing_workers = RouteMemoryDeclaration::new()
            .rows(1)
            .encoded_column(EncodedWidth::int32())
            .masks(0)
            .design_expansion(0)
            .learner_fold_copies(1, 4)
            .solver_scratch(4)
            .retained_scores_draws(0)
            .output(0);
        assert_eq!(
            missing_workers.estimate(),
            Err(RouteMemoryError::MissingComponent {
                component: RouteMemoryComponent::ConcurrentWorkers,
            })
        );
        assert_eq!(missing_workers.workers(0).estimate(), Err(RouteMemoryError::ZeroWorkers));
    }

    #[test]
    fn sum_over_budget_refuses_and_does_not_report_zero_bytes() {
        let estimate = fully_declared();
        assert_ne!(estimate.total(), 0);
        let budget = RouteBudget::new(estimate.total() - 1);
        let refused = estimate.admit(budget);
        match refused {
            Err(RouteMemoryError::ExceedsBudget { estimated_bytes, budget_bytes }) => {
                assert_eq!(estimated_bytes, estimate.total());
                assert_ne!(estimated_bytes, 0);
                assert_eq!(budget_bytes, budget.bytes());
                let text =
                    RouteMemoryError::ExceedsBudget { estimated_bytes, budget_bytes }.to_string();
                assert!(text.contains(&estimated_bytes.to_string()));
                assert!(text.contains(
                    "next attempt: lower the declared copies, workers, or retained outputs"
                ));
            }
            Ok(admitted) => panic!("admitted {} bytes", admitted.total()),
            Err(other) => panic!("unexpected refusal: {other}"),
        }

        let admitted: AdmittedRouteMemory =
            estimate.admit(RouteBudget::new(estimate.total())).expect("exact budget");
        assert_eq!(admitted.total(), estimate.total());
        assert_eq!(admitted.estimate(), estimate);

        let wrapped = RouteMemoryDeclaration::new()
            .rows(1 << 63)
            .encoded_column(EncodedWidth::new(2).expect("two-byte encoding"))
            .masks(0)
            .design_expansion(0)
            .learner_fold_copies(0, 0)
            .solver_scratch(0)
            .retained_scores_draws(0)
            .output(0)
            .workers(1);
        assert_eq!(wrapped.estimate(), Err(RouteMemoryError::Unrepresentable));
        assert_eq!(wrapped.admit(RouteBudget::new(0)), Err(RouteMemoryError::Unrepresentable));
    }

    #[test]
    fn two_workers_cost_two_scratch_copies() {
        let scratch = 64;
        let workers = |count: u64| {
            RouteMemoryDeclaration::new()
                .rows(1)
                .encoded_column(EncodedWidth::float32())
                .masks(0)
                .design_expansion(0)
                .learner_fold_copies(0, 0)
                .solver_scratch(scratch)
                .retained_scores_draws(0)
                .output(0)
                .workers(count)
                .estimate()
                .expect("declared")
        };
        let one = workers(1);
        let two = workers(2);
        assert_eq!(one.worker_count(), 1);
        assert_eq!(one.scratch_copy_bytes(), scratch);
        assert_eq!(two.worker_count(), 2);
        assert_eq!(two.scratch_copy_bytes(), 2 * scratch);
        assert_eq!(two.total() - one.total(), scratch);
        assert_eq!(two.concurrent_workers(), scratch);
    }

    #[test]
    fn worker_scratch_product_overflow_is_refused() {
        let scratch = (u64::MAX / 2) + 1;
        assert_eq!(scratch.wrapping_mul(2), 0);
        let declaration = RouteMemoryDeclaration::new()
            .rows(1)
            .encoded_column(EncodedWidth::float32())
            .masks(0)
            .design_expansion(0)
            .learner_fold_copies(0, 0)
            .solver_scratch(scratch)
            .retained_scores_draws(0)
            .output(0)
            .workers(3);
        assert_eq!(declaration.estimate(), Err(RouteMemoryError::Unrepresentable));
        assert_eq!(
            declaration.admit(RouteBudget::new(u64::MAX)),
            Err(RouteMemoryError::Unrepresentable)
        );
    }
}
