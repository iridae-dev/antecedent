// SPDX-License-Identifier: MIT OR Apache-2.0

//! Moment and constraint assembly for an accepted clamp-experiment regime.
//!
//! Given [`AcceptedRegimeContract`] and one sample per regime, this module
//! estimates centered second moments and stacks the experimental-effect rows of
//! the Hyttinen–Eberhardt–Hoyer clamp recipe. For a regime that intervenes on
//! `J` and passively observes `U`, each probe `from ∈ J` and target `into ∈ U`
//! contributes one row: the direct effect of `from` has coefficient 1, every
//! other intervened source has coefficient 0, every other passive source has
//! coefficient equal to the experimental effect of `from` on that source, and
//! the right-hand side is the experimental effect of `from` on `into`. An
//! experimental effect is the coefficient in the regression of a passive column
//! on the intervened columns of that regime. Each sample records a
//! [`crate::experiment_regime::SourceIdentity`]. Assembly refuses the sample
//! when that identity is not the one the regime declared.
//!
//! The assembled blocks are the constraint system. Column rank is the rank of
//! each block's coefficient matrix ([`constraint_column_report`]). A deficient
//! block returns the null directions of that matrix and no direct effects. A
//! full-column-rank block whose rows agree returns those direct effects. This
//! module does not form a pseudoinverse, does not fill a free direction, and
//! does not recover a disturbance covariance. Weak stability is not checked,
//! and it is not physical dynamical stability.
//!
//! A target is covered when, for every other unit `from`, some regime clamps
//! `from` while the target stays passive. Blocks are assembled for covered
//! targets, and the uncovered ordered pairs are listed beside them
//! ([`AssembledExperimentConstraints::uncovered`]). When no target is covered
//! the assembly is refused. A regime set whose dependence premise, population,
//! or retained units disagree with one shared independent experiment family is
//! refused.
//!
//! A mask may drop a unit only if that unit is clamped in every regime, because
//! the row for a passive unit `u` carries the term `b_iu t_uj`, and a dropped
//! passive unit would silently lose that term. The regime contract refuses a mask
//! that drops a unit any regime clamps, so every dropped unit that a regime
//! declares is refused here as [`MomentAssemblyRefusal::DroppedUnitNotClamped`].
//!
//! The experimental effects are computed on the correlation matrix of the
//! intervened columns, solved by [`antecedent_kernels::gauss_jordan_solve`].
//! The pivot floor is that kernel's equilibrated floor on a matrix whose
//! diagonal is order 1, so it does not depend on the units of measurement.

use std::collections::BTreeSet;
use std::fmt;

use antecedent_core::VariableId;
use antecedent_kernels::{gauss_jordan_solve, numeric::CompensatedSum};

use crate::experiment_regime::{
    AcceptedRegimeContract, DependencePremise, ExperimentRegime, SourceIdentity,
};

/// Bytes in one assembled `f64` entry.
const F64_BYTES: u64 = 8;

/// Largest 1-norm condition number of the equilibrated intervention covariance accepted
/// for solving experimental effects; beyond it fewer than two digits survive in `f64`.
const MAX_INTERVENTION_CONDITION: f64 = 1e14;

/// Relative floor for a pivot of an assembled constraint matrix.
///
/// Multiplied by the largest absolute coefficient. A column that does not clear
/// it is not a pivot. The intervention solve uses
/// [`antecedent_kernels::EQUILIBRATED_PIVOT_FLOOR`] on the correlation matrix.
/// This floor is not that policy and not a test of weak stability.
const CONSTRAINT_COLUMN_PIVOT_FLOOR: f64 = 1e-12;

/// One observed column of a regime sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitColumn<'a> {
    /// Sampling unit this column records.
    pub unit: VariableId,
    /// Observed values, one per row, including randomized intervention draws.
    pub values: &'a [f64],
}

/// Sample attached to one regime.
///
/// Columns are the units that remain after the accepted mask. A masked unit is
/// not a column. Column order does not matter; each retained unit appears once.
/// [`Self::source`] is the sample's source identity. Assembly refuses the sample
/// when it disagrees with the regime's declared [`SourceIdentity`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeSample<'a> {
    /// Columns of this regime.
    pub columns: &'a [UnitColumn<'a>],
    /// Source the columns were drawn from.
    pub source: SourceIdentity,
}

/// Why moment and constraint assembly stopped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MomentAssemblyRefusal {
    /// `samples` was not one sample per accepted regime.
    SampleCount {
        /// Accepted regimes.
        regimes: usize,
        /// Samples offered.
        samples: usize,
    },
    /// The regimes are not one independent family on one population and unit set.
    InconsistentRegime(RegimeInconsistency),
    /// Some ordered pair is never clamped at the source while the target is passive.
    InadequateInterventionCoverage {
        /// Missing pairs `(from, into)`, ordered by source then target.
        missing: Vec<MissingPair>,
    },
    /// A regime sample does not match its retained units.
    SampleUnits {
        /// Index into the accepted regimes.
        regime: usize,
    },
    /// Columns in one regime have different lengths.
    ColumnLength {
        /// Index into the accepted regimes.
        regime: usize,
    },
    /// Fewer than two rows.
    ShortSample {
        /// Index into the accepted regimes.
        regime: usize,
        /// Rows in the sample.
        rows: usize,
    },
    /// A sample entry is not finite.
    NonFiniteSample {
        /// Index into the accepted regimes.
        regime: usize,
        /// Column that contained the entry.
        unit: VariableId,
    },
    /// A centered moment is not finite.
    NonFiniteMoment {
        /// Index into the accepted regimes.
        regime: usize,
    },
    /// The intervened columns do not have a positive-definite sample covariance.
    DegenerateIntervention {
        /// Index into the accepted regimes.
        regime: usize,
    },
    /// The intervened columns' correlation matrix is nonsingular to the pivot floor but so
    /// ill conditioned that the experimental effects would carry fewer than two correct digits.
    IllConditionedIntervention {
        /// Index into the accepted regimes.
        regime: usize,
        /// Base-10 exponent of the estimated 1-norm condition number (rounded down).
        condition_exponent: u32,
    },
    /// The byte estimate overflowed.
    ResourceOverflow,
    /// The byte estimate exceeds the caller's budget. Nothing was allocated.
    BudgetExceeded {
        /// Estimated bytes, including the centered copies of the largest sample.
        requested: u64,
        /// Budget the caller supplied.
        limit: u64,
    },
    /// A mask drops a declared unit that is not clamped in every regime. The dropped
    /// passive unit's term would be missing from the experimental-effect rows.
    DroppedUnitNotClamped {
        /// The dropped unit.
        unit: VariableId,
    },
    /// The sample's source identity is not the regime's declared source.
    SourceMismatch {
        /// Index into the accepted regimes.
        regime: usize,
        /// Source the regime declared.
        declared: SourceIdentity,
        /// Source recorded on the sample.
        sample: SourceIdentity,
    },
}

/// How the accepted regimes disagree with one independent experiment family.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegimeInconsistency {
    /// The regime claims a shared disturbance. Assembly needs a fresh draw.
    DependentDisturbance {
        /// Index into the accepted regimes.
        regime: usize,
    },
    /// Two regimes name different populations.
    PopulationMismatch {
        /// Index of the earlier regime.
        left: usize,
        /// Index of the later regime.
        right: usize,
    },
    /// After the mask, two regimes do not retain the same sampling units.
    UnitSetMismatch {
        /// Index of the earlier regime.
        left: usize,
        /// Index of the later regime.
        right: usize,
    },
    /// The mask retained no sampling unit.
    EmptyRetainedUnits {
        /// Index into the accepted regimes.
        regime: usize,
    },
}

/// An ordered pair that no regime clamps at `from` while `into` stays passive.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MissingPair {
    /// Source that would have to be intervened on.
    pub from: VariableId,
    /// Target that would have to stay passive in that same regime.
    pub into: VariableId,
}

/// Centered second moments of one regime, in the assembly's unit order.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentMoments {
    /// Index into the accepted regimes.
    pub regime: usize,
    /// Row count of the sample.
    pub rows: usize,
    /// Centered means, one per assembled unit.
    pub mean: Vec<f64>,
    /// Row-major centered second moments. Entry `(r, c)` is at `r * units + c`.
    ///
    /// The divisor is the row count, the same divisor for every entry.
    pub covariance: Vec<f64>,
}

/// One experimental-effect row inside a target block.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ConstraintRow {
    /// Regime the row was read from.
    pub regime: usize,
    /// Intervened probe. Its direct effect has coefficient 1.
    pub probe: VariableId,
}

/// Constraint block for the direct effects into one target.
///
/// `coefficients` is row-major with `sources.len()` entries per row. The
/// unknown vector is the direct effects of `sources` into `target`, in that
/// order. This block does not contain a solution of that system.
/// [`constraint_column_report`] reads the column rank from `coefficients`.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetConstraintBlock {
    /// Passive equation these rows constrain.
    pub target: VariableId,
    /// Sources of the direct effects into `target`, in unit order.
    pub sources: Vec<VariableId>,
    /// Which regime and probe produced each row.
    pub rows: Vec<ConstraintRow>,
    /// Row-major experimental-effect coefficients.
    pub coefficients: Vec<f64>,
    /// Experimental effect of each row's probe on `target`.
    pub rhs: Vec<f64>,
}

/// Moments and unsolved constraint blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct AssembledExperimentConstraints {
    /// Retained units, in [`VariableId`] order.
    pub units: Vec<VariableId>,
    /// One moment matrix per accepted regime, in regime order.
    pub moments: Vec<ExperimentMoments>,
    /// One block per covered target, in unit order. A target with an uncovered
    /// ordered pair has no block.
    pub blocks: Vec<TargetConstraintBlock>,
    /// Ordered pairs no regime clamps at the source while the target is passive,
    /// ordered by source then target. Empty when every target has a block.
    pub uncovered: Vec<MissingPair>,
    /// Byte estimate of the moment matrices, the constraint entries, and the
    /// centered copies of the largest sample.
    pub bytes: u64,
}

impl fmt::Display for MomentAssemblyRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SampleCount { regimes, samples } => {
                write!(f, "expected {regimes} regime samples, received {samples}")
            }
            Self::InconsistentRegime(reason) => write!(f, "{reason}"),
            Self::InadequateInterventionCoverage { missing } => {
                write!(
                    f,
                    "intervention coverage misses {} ordered pair(s); constraints were not solved",
                    missing.len()
                )
            }
            Self::SampleUnits { regime } => {
                write!(f, "regime {regime} sample units do not match the retained units")
            }
            Self::ColumnLength { regime } => {
                write!(f, "regime {regime} columns do not share a length")
            }
            Self::ShortSample { regime, rows } => {
                write!(f, "regime {regime} has {rows} rows; a second moment needs two")
            }
            Self::NonFiniteSample { regime, unit } => {
                write!(f, "regime {regime} has a non-finite sample at {unit}")
            }
            Self::NonFiniteMoment { regime } => {
                write!(f, "regime {regime} produced a non-finite moment")
            }
            Self::DegenerateIntervention { regime } => {
                write!(f, "regime {regime} has a singular intervention covariance")
            }
            Self::IllConditionedIntervention { regime, condition_exponent } => {
                write!(
                    f,
                    "regime {regime} has an intervention covariance with condition number \
                     about 1e{condition_exponent}"
                )
            }
            Self::ResourceOverflow => {
                write!(f, "the moment and constraint byte estimate overflowed")
            }
            Self::BudgetExceeded { requested, limit } => {
                write!(f, "the moment and constraint estimate of {requested} bytes exceeds {limit}")
            }
            Self::DroppedUnitNotClamped { unit } => {
                write!(f, "mask drops {unit}, which is not clamped in every regime")
            }
            Self::SourceMismatch { regime, declared, sample } => {
                write!(f, "regime {regime} declares {declared} but the sample is {sample}")
            }
        }
    }
}

impl fmt::Display for RegimeInconsistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DependentDisturbance { regime } => {
                write!(f, "regime {regime} claims a dependent disturbance")
            }
            Self::PopulationMismatch { left, right } => {
                write!(f, "regimes {left} and {right} name different populations")
            }
            Self::UnitSetMismatch { left, right } => {
                write!(f, "regimes {left} and {right} retain different sampling units")
            }
            Self::EmptyRetainedUnits { regime } => {
                write!(f, "regime {regime} retains no sampling units")
            }
        }
    }
}

/// Bytes occupied by `regimes` moment matrices, `constraint_rows` constraint rows, and
/// the centered copies of a sample of `max_rows` rows, on `units` units.
///
/// Each regime contributes `units * (units + 1)` floats (means and the covariance
/// matrix). Each constraint row contributes `units` floats (`units - 1`
/// coefficients and one right-hand side). The centered copies are held for one
/// regime at a time and add `units * max_rows` floats. Index storage is not
/// included. The arithmetic is checked; overflow is
/// [`MomentAssemblyRefusal::ResourceOverflow`].
pub fn experiment_constraint_bytes(
    units: u64,
    regimes: u64,
    constraint_rows: u64,
    max_rows: u64,
) -> Result<u64, MomentAssemblyRefusal> {
    let per_regime = units
        .checked_add(1)
        .and_then(|width| units.checked_mul(width))
        .ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
    let moments = regimes
        .checked_mul(per_regime)
        .and_then(|cells| cells.checked_mul(F64_BYTES))
        .ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
    let constraints = constraint_rows
        .checked_mul(units)
        .and_then(|cells| cells.checked_mul(F64_BYTES))
        .ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
    let centered = units
        .checked_mul(max_rows)
        .and_then(|cells| cells.checked_mul(F64_BYTES))
        .ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
    moments
        .checked_add(constraints)
        .and_then(|total| total.checked_add(centered))
        .ok_or(MomentAssemblyRefusal::ResourceOverflow)
}

/// Assemble centered moments and experimental-effect constraints.
///
/// The returned blocks are not a coefficient estimate. Blocks exist for covered
/// targets only; the uncovered ordered pairs are listed. `max_bytes` bounds the
/// estimate of [`experiment_constraint_bytes`], which is checked before any moment
/// is accumulated. A refusal does not carry a partial block.
#[must_use = "a refusal must not be read as an assembled constraint"]
pub fn assemble_experiment_constraints(
    contract: &AcceptedRegimeContract,
    samples: &[RegimeSample<'_>],
    max_bytes: u64,
) -> Result<AssembledExperimentConstraints, MomentAssemblyRefusal> {
    let regimes = contract.regimes();
    if samples.len() != regimes.len() {
        return Err(MomentAssemblyRefusal::SampleCount {
            regimes: regimes.len(),
            samples: samples.len(),
        });
    }
    let dropped = &contract.mask().dropped;
    if let Some(reason) = inconsistent_regime(regimes, dropped) {
        return Err(MomentAssemblyRefusal::InconsistentRegime(reason));
    }
    if let Some(unit) = unclamped_dropped_unit(regimes, dropped) {
        return Err(MomentAssemblyRefusal::DroppedUnitNotClamped { unit });
    }
    let units = retained_units(&regimes[0], dropped);
    let unit_list: Vec<VariableId> = units.iter().copied().collect();
    let uncovered = missing_pairs(regimes, &units);
    let uncovered_targets: BTreeSet<VariableId> = uncovered.iter().map(|pair| pair.into).collect();
    let covered: Vec<bool> =
        unit_list.iter().map(|unit| !uncovered_targets.contains(unit)).collect();
    if !covered.iter().any(|&flag| flag) {
        return Err(MomentAssemblyRefusal::InadequateInterventionCoverage { missing: uncovered });
    }
    let prepared = prepare_samples(regimes, dropped, samples)?;
    let row_count = constraint_row_count(regimes, &units, &uncovered_targets)?;
    let max_rows = prepared.iter().map(|sample| sample.rows).max().unwrap_or(0);
    let bytes = experiment_constraint_bytes(
        u64_len(unit_list.len())?,
        u64_len(regimes.len())?,
        row_count,
        u64_len(max_rows)?,
    )?;
    if bytes > max_bytes {
        return Err(MomentAssemblyRefusal::BudgetExceeded { requested: bytes, limit: max_bytes });
    }
    let moments = accumulate_moments(&unit_list, &prepared)?;
    let effects = regime_effects(&unit_list, regimes, &moments)?;
    let blocks = assemble_blocks(&unit_list, &covered, &effects);
    Ok(AssembledExperimentConstraints { units: unit_list, moments, blocks, uncovered, bytes })
}

/// Column rank of one assembled experimental-effect block.
///
/// The rank is the number of pivot columns of `block.coefficients`. It is not a
/// [`crate::experimental_recovery::DeclaredEquationRank`] and not
/// [`crate::experimental_recovery::WeakStability::InvertibilityDeclared`].
#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintColumnReport {
    /// The columns span the sources and the rows agree on one direct-effect vector.
    Unique {
        /// Direct effects in [`TargetConstraintBlock::sources`] order.
        coefficients: Vec<f64>,
    },
    /// The coefficient matrix has a nontrivial nullspace. No direct effect was filled in.
    Deficient {
        /// Number of pivot columns.
        column_rank: usize,
        /// One basis vector per free column, in ascending source order.
        ///
        /// `directions[k][j]` is the weight of source `j` in direction `k`. The
        /// free source of each direction has weight 1, and each pivot source has
        /// the weight that cancels that free source in the reduced matrix.
        directions: Vec<Vec<f64>>,
    },
    /// Full column rank, but the rows do not agree on a right-hand side.
    Inconsistent {
        /// Number of pivot columns, equal to the source count.
        column_rank: usize,
    },
    /// The block is not a finite coefficient matrix of the stated shape.
    NotAConstraint,
}

/// Column rank, null directions, or the unique direct effects of `block`.
///
/// Elimination runs on the coefficient matrix the assembly already built. A
/// deficient matrix returns its null directions. No value is invented for a
/// free source, and a minimum-norm completion is not applied.
#[must_use]
pub fn constraint_column_report(block: &TargetConstraintBlock) -> ConstraintColumnReport {
    let columns = block.sources.len();
    let rows = block.rhs.len();
    let Some(cells) = rows.checked_mul(columns) else {
        return ConstraintColumnReport::NotAConstraint;
    };
    if block.coefficients.len() != cells || block.rows.len() != rows {
        return ConstraintColumnReport::NotAConstraint;
    }
    if block.coefficients.iter().chain(block.rhs.iter()).any(|value| !value.is_finite()) {
        return ConstraintColumnReport::NotAConstraint;
    }
    if columns == 0 {
        return vacuous_column_report(block);
    }
    reduce_constraint_columns(&block.coefficients, &block.rhs, rows, columns)
}

fn u64_len(len: usize) -> Result<u64, MomentAssemblyRefusal> {
    u64::try_from(len).map_err(|_| MomentAssemblyRefusal::ResourceOverflow)
}

fn length_product(left: usize, right: usize) -> Result<usize, MomentAssemblyRefusal> {
    let cells = u64_len(left)?
        .checked_mul(u64_len(right)?)
        .ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
    usize::try_from(cells).map_err(|_| MomentAssemblyRefusal::ResourceOverflow)
}

fn retained_units(regime: &ExperimentRegime, dropped: &[VariableId]) -> BTreeSet<VariableId> {
    regime.units.iter().copied().filter(|unit| !dropped.contains(unit)).collect()
}

/// A dropped unit that some regime declares and does not clamp.
fn unclamped_dropped_unit(
    regimes: &[ExperimentRegime],
    dropped: &[VariableId],
) -> Option<VariableId> {
    dropped.iter().copied().find(|unit| {
        let declared = regimes.iter().any(|regime| regime.units.contains(unit));
        let clamped_everywhere =
            regimes.iter().all(|regime| regime.targets.iter().any(|target| target.unit == *unit));
        declared && !clamped_everywhere
    })
}

fn inconsistent_regime(
    regimes: &[ExperimentRegime],
    dropped: &[VariableId],
) -> Option<RegimeInconsistency> {
    let baseline = retained_units(&regimes[0], dropped);
    if baseline.is_empty() {
        return Some(RegimeInconsistency::EmptyRetainedUnits { regime: 0 });
    }
    let population = regimes[0].population;
    for (index, regime) in regimes.iter().enumerate() {
        if regime.premise == DependencePremise::Dependence {
            return Some(RegimeInconsistency::DependentDisturbance { regime: index });
        }
        if regime.population != population {
            return Some(RegimeInconsistency::PopulationMismatch { left: 0, right: index });
        }
        let retained = retained_units(regime, dropped);
        if retained.is_empty() {
            return Some(RegimeInconsistency::EmptyRetainedUnits { regime: index });
        }
        if retained != baseline {
            return Some(RegimeInconsistency::UnitSetMismatch { left: 0, right: index });
        }
    }
    None
}

fn missing_pairs(regimes: &[ExperimentRegime], units: &BTreeSet<VariableId>) -> Vec<MissingPair> {
    let mut missing = Vec::new();
    for &from in units {
        for &into in units {
            if from == into {
                continue;
            }
            let covered = regimes.iter().any(|regime| {
                regime.targets.iter().any(|target| target.unit == from)
                    && units.contains(&into)
                    && regime.targets.iter().all(|target| target.unit != into)
                    && regime.units.contains(&into)
            });
            if !covered {
                missing.push(MissingPair { from, into });
            }
        }
    }
    missing
}

struct PreparedColumn<'a> {
    unit: VariableId,
    values: &'a [f64],
}

struct PreparedSample<'a> {
    columns: Vec<PreparedColumn<'a>>,
    rows: usize,
}

fn prepare_samples<'a>(
    regimes: &[ExperimentRegime],
    dropped: &[VariableId],
    samples: &[RegimeSample<'a>],
) -> Result<Vec<PreparedSample<'a>>, MomentAssemblyRefusal> {
    regimes
        .iter()
        .zip(samples)
        .enumerate()
        .map(|(index, (regime, sample))| prepare_one(index, regime, dropped, sample))
        .collect()
}

fn prepare_one<'a>(
    index: usize,
    regime: &ExperimentRegime,
    dropped: &[VariableId],
    sample: &RegimeSample<'a>,
) -> Result<PreparedSample<'a>, MomentAssemblyRefusal> {
    if sample.source != regime.source {
        return Err(MomentAssemblyRefusal::SourceMismatch {
            regime: index,
            declared: regime.source,
            sample: sample.source,
        });
    }
    let expected = retained_units(regime, dropped);
    let mut seen = BTreeSet::new();
    for column in sample.columns {
        if !expected.contains(&column.unit) || !seen.insert(column.unit) {
            return Err(MomentAssemblyRefusal::SampleUnits { regime: index });
        }
    }
    if seen != expected {
        return Err(MomentAssemblyRefusal::SampleUnits { regime: index });
    }
    let rows = sample.columns.first().map_or(0, |column| column.values.len());
    if rows < 2 {
        return Err(MomentAssemblyRefusal::ShortSample { regime: index, rows });
    }
    for column in sample.columns {
        if column.values.len() != rows {
            return Err(MomentAssemblyRefusal::ColumnLength { regime: index });
        }
        if column.values.iter().any(|value| !value.is_finite()) {
            return Err(MomentAssemblyRefusal::NonFiniteSample {
                regime: index,
                unit: column.unit,
            });
        }
    }
    let mut columns: Vec<PreparedColumn<'a>> = sample
        .columns
        .iter()
        .map(|column| PreparedColumn { unit: column.unit, values: column.values })
        .collect();
    columns.sort_by_key(|column| column.unit);
    Ok(PreparedSample { columns, rows })
}

fn constraint_row_count(
    regimes: &[ExperimentRegime],
    units: &BTreeSet<VariableId>,
    uncovered_targets: &BTreeSet<VariableId>,
) -> Result<u64, MomentAssemblyRefusal> {
    let mut rows = 0_u64;
    for regime in regimes {
        let targeted: BTreeSet<VariableId> = regime
            .targets
            .iter()
            .map(|target| target.unit)
            .filter(|unit| units.contains(unit))
            .collect();
        let covered_passive = units
            .iter()
            .filter(|unit| !targeted.contains(unit) && !uncovered_targets.contains(unit))
            .count();
        let added = u64_len(targeted.len())?
            .checked_mul(u64_len(covered_passive)?)
            .ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
        rows = rows.checked_add(added).ok_or(MomentAssemblyRefusal::ResourceOverflow)?;
    }
    Ok(rows)
}

fn accumulate_moments(
    units: &[VariableId],
    samples: &[PreparedSample<'_>],
) -> Result<Vec<ExperimentMoments>, MomentAssemblyRefusal> {
    samples
        .iter()
        .enumerate()
        .map(|(regime, sample)| accumulate_one(regime, units, sample))
        .collect()
}

fn accumulate_one(
    regime: usize,
    units: &[VariableId],
    sample: &PreparedSample<'_>,
) -> Result<ExperimentMoments, MomentAssemblyRefusal> {
    let n = units.len();
    let mut mean = Vec::with_capacity(n);
    let mut centered: Vec<Vec<f64>> = Vec::with_capacity(n);
    for unit in units {
        let column = sample
            .columns
            .iter()
            .find(|column| column.unit == *unit)
            .ok_or(MomentAssemblyRefusal::SampleUnits { regime })?;
        let column_mean = compensated_mean(column.values);
        if !column_mean.is_finite() {
            return Err(MomentAssemblyRefusal::NonFiniteMoment { regime });
        }
        let mut deviations = Vec::with_capacity(column.values.len());
        for &value in column.values {
            deviations.push(value - column_mean);
        }
        mean.push(column_mean);
        centered.push(deviations);
    }
    let mut covariance = vec![0.0; length_product(n, n)?];
    let divisor = sample.rows as f64;
    for row in 0..n {
        for col in row..n {
            let moment = compensated_dot(&centered[row], &centered[col]) / divisor;
            if !moment.is_finite() {
                return Err(MomentAssemblyRefusal::NonFiniteMoment { regime });
            }
            covariance[row * n + col] = moment;
            covariance[col * n + row] = moment;
        }
    }
    Ok(ExperimentMoments { regime, rows: sample.rows, mean, covariance })
}

/// Probe and passive layout of one regime, with its experimental effects.
///
/// Slot tables are indexed by the position of a unit in the assembly's unit order, so
/// a coefficient lookup is constant time.
struct RegimeEffects {
    /// Unit positions of the regime's clamps, ascending.
    probe_units: Vec<usize>,
    /// Probe slot of each unit position, when that unit is clamped.
    probe_slot: Vec<Option<usize>>,
    /// Passive slot of each unit position, when that unit is not clamped.
    passive_slot: Vec<Option<usize>>,
    /// Row-major `passive.len() * probe_units.len()`.
    effects: Vec<f64>,
}

/// Experimental effects of every regime, each computed once.
fn regime_effects(
    units: &[VariableId],
    regimes: &[ExperimentRegime],
    moments: &[ExperimentMoments],
) -> Result<Vec<RegimeEffects>, MomentAssemblyRefusal> {
    let n = units.len();
    let mut all = Vec::with_capacity(regimes.len());
    for (regime_index, regime) in regimes.iter().enumerate() {
        let targeted: BTreeSet<VariableId> = regime.targets.iter().map(|item| item.unit).collect();
        let mut probe_units = Vec::new();
        let mut passive = Vec::new();
        let mut probe_slot = vec![None; n];
        let mut passive_slot = vec![None; n];
        for (index, unit) in units.iter().enumerate() {
            if targeted.contains(unit) {
                probe_slot[index] = Some(probe_units.len());
                probe_units.push(index);
            } else {
                passive_slot[index] = Some(passive.len());
                passive.push(index);
            }
        }
        let effects = experimental_effects(
            regime_index,
            &moments[regime_index].covariance,
            n,
            &probe_units,
            &passive,
        )?;
        all.push(RegimeEffects { probe_units, probe_slot, passive_slot, effects });
    }
    Ok(all)
}

fn assemble_blocks(
    units: &[VariableId],
    covered: &[bool],
    effects: &[RegimeEffects],
) -> Vec<TargetConstraintBlock> {
    let mut blocks = Vec::new();
    for (target_index, &target) in units.iter().enumerate() {
        if !covered[target_index] {
            continue;
        }
        let source_indices: Vec<usize> =
            (0..units.len()).filter(|index| *index != target_index).collect();
        let sources: Vec<VariableId> = source_indices.iter().map(|&index| units[index]).collect();
        let mut rows = Vec::new();
        let mut coefficients = Vec::new();
        let mut rhs = Vec::new();
        for (regime_index, layout) in effects.iter().enumerate() {
            // A clamped target has no passive equation in this regime.
            let Some(target_slot) = layout.passive_slot[target_index] else {
                continue;
            };
            let width = layout.probe_units.len();
            for (probe_slot, &probe_index) in layout.probe_units.iter().enumerate() {
                rows.push(ConstraintRow { regime: regime_index, probe: units[probe_index] });
                for &source_index in &source_indices {
                    coefficients.push(row_coefficient(
                        source_index,
                        probe_index,
                        probe_slot,
                        layout,
                    ));
                }
                rhs.push(layout.effects[target_slot * width + probe_slot]);
            }
        }
        blocks.push(TargetConstraintBlock { target, sources, rows, coefficients, rhs });
    }
    blocks
}

/// Coefficient of `source` in the row for the clamp `probe`: one for the probe itself,
/// zero for another clamp, and the probe's experimental effect on `source` otherwise.
fn row_coefficient(source: usize, probe: usize, probe_slot: usize, regime: &RegimeEffects) -> f64 {
    if source == probe {
        return 1.0;
    }
    if regime.probe_slot[source].is_some() {
        return 0.0;
    }
    // Every unit is either a clamp or passive in a regime.
    debug_assert!(regime.passive_slot[source].is_some());
    regime.passive_slot[source].map_or(0.0, |passive_slot| {
        regime.effects[passive_slot * regime.probe_units.len() + probe_slot]
    })
}

/// Experimental effects of each intervened column on each passive column.
///
/// Layout is row-major `passive.len() * intervened.len()`. The system
/// `Σ_JJ B = Σ_JU` is solved on the correlation scale: with `D = diag(Σ_JJ)^{1/2}`,
/// `(D⁻¹ Σ_JJ D⁻¹)(D B) = D⁻¹ Σ_JU`, then each row `j` of the solution is divided by
/// `D_jj`. [`gauss_jordan_solve`] applies its equilibrated pivot floor to that
/// correlation matrix, so rescaling an intervened column does not change which
/// regimes are degenerate. A constant or numerically singular intervention is
/// [`MomentAssemblyRefusal::DegenerateIntervention`].
fn experimental_effects(
    regime: usize,
    covariance: &[f64],
    n: usize,
    intervened: &[usize],
    passive: &[usize],
) -> Result<Vec<f64>, MomentAssemblyRefusal> {
    let j = intervened.len();
    let u = passive.len();
    if j == 0 || u == 0 {
        return Ok(Vec::new());
    }
    let mut spread = Vec::with_capacity(j);
    for &index in intervened {
        let variance = covariance[index * n + index];
        if !(variance.is_finite() && variance > 0.0) {
            return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
        }
        let scale = variance.sqrt();
        if !(scale.is_finite() && scale > 0.0) {
            return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
        }
        spread.push(scale);
    }
    let mut factor = vec![0.0; length_product(j, j)?];
    for (row, &left) in intervened.iter().enumerate() {
        for (col, &right) in intervened.iter().enumerate() {
            factor[row * j + col] = covariance[left * n + right] / spread[row] / spread[col];
        }
    }
    // One right-hand side per passive column: `rhs[k * j + row]`, followed by the `j`
    // columns of the identity so the same elimination yields the inverse for a 1-norm
    // condition estimate `‖A‖₁ ‖A⁻¹‖₁`.
    let mut solved = vec![0.0; length_product(u + j, j)?];
    for (passive_slot, &passive_index) in passive.iter().enumerate() {
        for (probe_slot, &probe_index) in intervened.iter().enumerate() {
            solved[passive_slot * j + probe_slot] =
                covariance[probe_index * n + passive_index] / spread[probe_slot];
        }
    }
    for k in 0..j {
        solved[(u + k) * j + k] = 1.0;
    }
    let norm_a = (0..j)
        .map(|col| (0..j).map(|row| factor[row * j + col].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    if !gauss_jordan_solve(&mut factor, &mut solved, j, u + j) {
        return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
    }
    let norm_inverse = (0..j)
        .map(|k| solved[(u + k) * j..(u + k + 1) * j].iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let condition = norm_a * norm_inverse;
    if !condition.is_finite() {
        return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
    }
    if condition > MAX_INTERVENTION_CONDITION {
        // Truncation to the decimal exponent is the documented rounding.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let condition_exponent = condition.log10().floor() as u32;
        return Err(MomentAssemblyRefusal::IllConditionedIntervention {
            regime,
            condition_exponent,
        });
    }
    solved.truncate(u * j);
    for passive_slot in 0..u {
        for (probe_slot, &scale) in spread.iter().enumerate() {
            // Undo the column scaling: the solved row is `D_jj` times the effect.
            let slot = passive_slot * j + probe_slot;
            let effect = solved[slot] / scale;
            if !effect.is_finite() {
                return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
            }
            solved[slot] = effect;
        }
    }
    Ok(solved)
}

fn swap_factor_rows(factor: &mut [f64], width: usize, left: usize, right: usize) {
    for col in 0..width {
        factor.swap(left * width + col, right * width + col);
    }
}

fn vacuous_column_report(block: &TargetConstraintBlock) -> ConstraintColumnReport {
    let scale = block.rhs.iter().fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let floor = CONSTRAINT_COLUMN_PIVOT_FLOOR * scale.max(1.0);
    if block.rhs.iter().any(|value| value.abs() > floor) {
        ConstraintColumnReport::Inconsistent { column_rank: 0 }
    } else {
        ConstraintColumnReport::Unique { coefficients: Vec::new() }
    }
}

/// Rank-revealing Gauss-Jordan on an assembled constraint.
///
/// The intervention covariance keeps a separate reduction: a singular
/// intervention is refused there, while a singular constraint returns null
/// directions here.
fn reduce_constraint_columns(
    coefficients: &[f64],
    rhs: &[f64],
    rows: usize,
    columns: usize,
) -> ConstraintColumnReport {
    let mut global = 0.0_f64;
    for value in coefficients {
        global = global.max(value.abs());
    }
    if global == 0.0 {
        return ConstraintColumnReport::Deficient {
            column_rank: 0,
            directions: identity_directions(columns),
        };
    }
    let mut rhs_scale = 0.0_f64;
    for value in rhs {
        rhs_scale = rhs_scale.max(value.abs());
    }
    let pivot_floor = CONSTRAINT_COLUMN_PIVOT_FLOOR * global;
    let residual_floor = CONSTRAINT_COLUMN_PIVOT_FLOOR * global.max(rhs_scale).max(1.0);
    let mut factor = coefficients.to_vec();
    let mut solved = rhs.to_vec();
    let mut pivot_row_of = vec![None; columns];
    let mut pivot_columns = Vec::new();
    let mut next_row = 0_usize;
    for col in 0..columns {
        if next_row >= rows {
            break;
        }
        let mut best_row = next_row;
        let mut best = factor[next_row * columns + col].abs();
        for row in (next_row + 1)..rows {
            let value = factor[row * columns + col].abs();
            if value > best {
                best = value;
                best_row = row;
            }
        }
        if !best.is_finite() || best <= pivot_floor {
            continue;
        }
        if best_row != next_row {
            swap_factor_rows(&mut factor, columns, next_row, best_row);
            solved.swap(next_row, best_row);
        }
        let pivot = factor[next_row * columns + col];
        for index in 0..columns {
            factor[next_row * columns + index] /= pivot;
        }
        solved[next_row] /= pivot;
        eliminate_pivot_from_other_rows(&mut factor, &mut solved, rows, columns, next_row, col);
        pivot_row_of[col] = Some(next_row);
        pivot_columns.push(col);
        next_row += 1;
    }
    let column_rank = pivot_columns.len();
    if column_rank < columns {
        return null_directions_of(&factor, &pivot_row_of, &pivot_columns, columns, column_rank);
    }
    unique_or_inconsistent(&solved, &pivot_row_of, &pivot_columns, column_rank, residual_floor)
}

fn eliminate_pivot_from_other_rows(
    factor: &mut [f64],
    solved: &mut [f64],
    rows: usize,
    columns: usize,
    pivot_row: usize,
    pivot_col: usize,
) {
    for row in 0..rows {
        if row == pivot_row {
            continue;
        }
        let multiplier = factor[row * columns + pivot_col];
        for index in 0..columns {
            factor[row * columns + index] -= multiplier * factor[pivot_row * columns + index];
        }
        solved[row] -= multiplier * solved[pivot_row];
    }
}

fn null_directions_of(
    factor: &[f64],
    pivot_row_of: &[Option<usize>],
    pivot_columns: &[usize],
    columns: usize,
    column_rank: usize,
) -> ConstraintColumnReport {
    let mut directions = Vec::with_capacity(columns - column_rank);
    for free in 0..columns {
        if pivot_row_of[free].is_some() {
            continue;
        }
        let mut direction = vec![0.0; columns];
        direction[free] = 1.0;
        for &pivot_col in pivot_columns {
            let Some(pivot_row) = pivot_row_of[pivot_col] else {
                return ConstraintColumnReport::NotAConstraint;
            };
            direction[pivot_col] = -factor[pivot_row * columns + free];
        }
        if direction.iter().any(|weight| !weight.is_finite()) {
            return ConstraintColumnReport::NotAConstraint;
        }
        directions.push(direction);
    }
    ConstraintColumnReport::Deficient { column_rank, directions }
}

fn unique_or_inconsistent(
    solved: &[f64],
    pivot_row_of: &[Option<usize>],
    pivot_columns: &[usize],
    column_rank: usize,
    residual_floor: f64,
) -> ConstraintColumnReport {
    for value in solved.iter().skip(column_rank) {
        if !value.is_finite() {
            return ConstraintColumnReport::NotAConstraint;
        }
        if value.abs() > residual_floor {
            return ConstraintColumnReport::Inconsistent { column_rank };
        }
    }
    let mut coefficients = vec![0.0; pivot_row_of.len()];
    for &pivot_col in pivot_columns {
        let Some(pivot_row) = pivot_row_of[pivot_col] else {
            return ConstraintColumnReport::NotAConstraint;
        };
        let value = solved[pivot_row];
        if !value.is_finite() {
            return ConstraintColumnReport::NotAConstraint;
        }
        coefficients[pivot_col] = value;
    }
    ConstraintColumnReport::Unique { coefficients }
}

fn identity_directions(columns: usize) -> Vec<Vec<f64>> {
    (0..columns)
        .map(|free| {
            let mut direction = vec![0.0; columns];
            direction[free] = 1.0;
            direction
        })
        .collect()
}

/// Sum with [`CompensatedSum`]. A running `f64` total drops a unit that sits next
/// to a shift larger than the mantissa; the correction term keeps it.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    CompensatedSum::of(values)
}

fn compensated_mean(values: &[f64]) -> f64 {
    compensated_sum(values.iter().copied()) / values.len() as f64
}

fn compensated_dot(left: &[f64], right: &[f64]) -> f64 {
    compensated_sum(left.iter().zip(right).map(|(l, r)| l * r))
}

#[cfg(test)]
mod tests {
    use super::{
        AssembledExperimentConstraints, ConstraintColumnReport, ConstraintRow, MissingPair,
        MomentAssemblyRefusal, RegimeInconsistency, RegimeSample, TargetConstraintBlock,
        UnitColumn, assemble_experiment_constraints, compensated_sum, constraint_column_report,
        experiment_constraint_bytes,
    };
    use crate::experiment_regime::{
        DependencePremise, ExperimentRegime, PopulationId, RegimeContract, RegimeContractDecision,
        RegimeIntervention, SourceIdentity, UnitMask, validate_regime_contract,
    };
    use antecedent_core::{Value, VariableId};

    fn unit(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn regime(
        units: &[u32],
        targets: &[u32],
        premise: DependencePremise,
        source: u64,
        population: u32,
    ) -> ExperimentRegime {
        ExperimentRegime {
            targets: targets
                .iter()
                .map(|&raw| RegimeIntervention { unit: unit(raw), value: Value::Int64(0) })
                .collect(),
            units: units.iter().copied().map(unit).collect(),
            population: PopulationId::from_raw(population),
            premise,
            source: SourceIdentity::from_raw(source),
        }
    }

    fn accepted(
        regimes: Vec<ExperimentRegime>,
        dropped: &[u32],
    ) -> crate::experiment_regime::AcceptedRegimeContract {
        let decision = validate_regime_contract(&RegimeContract {
            regimes,
            mask: UnitMask { dropped: dropped.iter().copied().map(unit).collect() },
        });
        let RegimeContractDecision::Accepted(contract) = decision else {
            panic!("the fixture regime contract was refused");
        };
        contract
    }

    fn sample<'a>(columns: &'a [UnitColumn<'a>], source: u64) -> RegimeSample<'a> {
        RegimeSample { columns, source: SourceIdentity::from_raw(source) }
    }

    fn assert_close(actual: f64, expected: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!((actual - expected).abs() <= scale * 1e-12, "actual {actual} expected {expected}");
    }

    fn row(block: &TargetConstraintBlock, regime: usize, probe: u32) -> (&[f64], f64) {
        let slot = block
            .rows
            .iter()
            .position(|row| row.regime == regime && row.probe == unit(probe))
            .expect("constraint row");
        let width = block.sources.len();
        (&block.coefficients[slot * width..(slot + 1) * width], block.rhs[slot])
    }

    fn holds(coefficients: &[f64], rhs: f64, direct: &[f64]) {
        let mut total = 0.0;
        for (coefficient, effect) in coefficients.iter().zip(direct) {
            total += coefficient * effect;
        }
        assert_close(total, rhs);
    }

    /// Synthetic three-unit system derived from the clamp constraint.
    ///
    /// `y = 0.5 x` and `z = 0.25 x + 0.5 y`, with `x` exogenous. Residuals are
    /// zero, so every experimental-effect row is satisfied by those direct
    /// effects. Regime 3 intervenes on correlated `(x, y)`.
    fn synthetic_contract() -> crate::experiment_regime::AcceptedRegimeContract {
        let independent = DependencePremise::Independence;
        accepted(
            vec![
                regime(&[0, 1, 2], &[0], independent, 1, 1),
                regime(&[0, 1, 2], &[1, 2], independent, 2, 1),
                regime(&[0, 1, 2], &[0, 2], independent, 3, 1),
                regime(&[0, 1, 2], &[0, 1], independent, 4, 1),
            ],
            &[],
        )
    }

    #[test]
    fn assembles_the_experimental_effect_constraint_without_solving_it() {
        let x0 = [1.0, -1.0];
        let y0 = [0.5, -0.5];
        let z0 = [0.5, -0.5];
        let x1 = [1.0, 1.0, -1.0, -1.0];
        let y1 = [1.0, -1.0, 1.0, -1.0];
        let z1 = [1.0, -1.0, -1.0, 1.0];
        let x2 = [1.0, -1.0, 1.0, -1.0];
        let y2 = [0.5, -0.5, 0.5, -0.5];
        let z2 = [1.0, 1.0, -1.0, -1.0];
        let x3 = [2.0, 0.0, -2.0, 0.0];
        let y3 = [1.0, 1.0, -1.0, -1.0];
        let z3 = [1.0, 0.5, -1.0, -0.5];
        let c0 = [
            UnitColumn { unit: unit(0), values: &x0 },
            UnitColumn { unit: unit(1), values: &y0 },
            UnitColumn { unit: unit(2), values: &z0 },
        ];
        let c1 = [
            UnitColumn { unit: unit(0), values: &x1 },
            UnitColumn { unit: unit(1), values: &y1 },
            UnitColumn { unit: unit(2), values: &z1 },
        ];
        let c2 = [
            UnitColumn { unit: unit(0), values: &x2 },
            UnitColumn { unit: unit(1), values: &y2 },
            UnitColumn { unit: unit(2), values: &z2 },
        ];
        let c3 = [
            UnitColumn { unit: unit(0), values: &x3 },
            UnitColumn { unit: unit(1), values: &y3 },
            UnitColumn { unit: unit(2), values: &z3 },
        ];
        let samples = [sample(&c0, 1), sample(&c1, 2), sample(&c2, 3), sample(&c3, 4)];
        let assembled = assemble_experiment_constraints(&synthetic_contract(), &samples, u64::MAX)
            .expect("covering independent regimes assemble");
        assert_eq!(assembled.units, vec![unit(0), unit(1), unit(2)]);
        assert_eq!(assembled.moments.len(), 4);
        assert_eq!(assembled.blocks.len(), 3);
        assert_close(assembled.moments[3].covariance[0], 2.0);

        let into_z = block(&assembled, 2);
        assert_eq!(into_z.sources, vec![unit(0), unit(1)]);
        let (cross, cross_rhs) = row(into_z, 0, 0);
        assert_eq!(cross, &[1.0, 0.5]);
        assert_close(cross_rhs, 0.5);
        let (probe_x, rhs_x) = row(into_z, 3, 0);
        assert_eq!(probe_x, &[1.0, 0.0]);
        assert_close(rhs_x, 0.25);
        let (probe_y, rhs_y) = row(into_z, 3, 1);
        assert_eq!(probe_y, &[0.0, 1.0]);
        assert_close(rhs_y, 0.5);
        holds(cross, cross_rhs, &[0.25, 0.5]);
        holds(probe_x, rhs_x, &[0.25, 0.5]);
        holds(probe_y, rhs_y, &[0.25, 0.5]);

        let into_y = block(&assembled, 1);
        let (from_x, from_x_rhs) = row(into_y, 0, 0);
        assert_eq!(from_x, &[1.0, 0.5]);
        assert_close(from_x_rhs, 0.5);
        holds(from_x, from_x_rhs, &[0.5, 0.0]);

        let into_x = block(&assembled, 0);
        let (from_y, from_y_rhs) = row(into_x, 1, 1);
        let (from_z, from_z_rhs) = row(into_x, 1, 2);
        holds(from_y, from_y_rhs, &[0.0, 0.0]);
        holds(from_z, from_z_rhs, &[0.0, 0.0]);

        let rows = assembled.blocks.iter().map(|block| block.rows.len()).sum::<usize>();
        // Moments 4 * 3 * 4 floats, constraints 8 rows * 3 floats, and the centered
        // copies of the largest sample, 3 units * 4 rows: (48 + 24 + 12) * 8 bytes.
        let bytes = experiment_constraint_bytes(3, 4, u64::try_from(rows).expect("row count"), 4)
            .expect("byte count");
        assert_eq!(assembled.bytes, bytes);
        assert_eq!(assembled.bytes, 672);
        assert!(assembled.uncovered.is_empty());
    }

    fn block(assembled: &AssembledExperimentConstraints, target: u32) -> &TargetConstraintBlock {
        assembled.blocks.iter().find(|block| block.target == unit(target)).expect("target block")
    }

    #[test]
    fn inadequate_intervention_coverage_is_refused() {
        let x = [1.0, -1.0];
        let y = [0.5, -0.5];
        let z = [0.5, -0.5];
        let columns = [
            UnitColumn { unit: unit(0), values: &x },
            UnitColumn { unit: unit(1), values: &y },
            UnitColumn { unit: unit(2), values: &z },
        ];
        let contract =
            accepted(vec![regime(&[0, 1, 2], &[0], DependencePremise::Independence, 1, 1)], &[]);
        let refusal = assemble_experiment_constraints(&contract, &[sample(&columns, 1)], u64::MAX)
            .expect_err("one clamp does not cover every ordered pair");
        let MomentAssemblyRefusal::InadequateInterventionCoverage { missing } = refusal else {
            panic!("expected inadequate coverage");
        };
        let pairs: Vec<(u32, u32)> =
            missing.iter().map(|pair| (pair.from.raw(), pair.into.raw())).collect();
        assert_eq!(pairs, vec![(1, 0), (1, 2), (2, 0), (2, 1)]);
    }

    #[test]
    fn a_dependent_regime_is_refused() {
        let contract = accepted(
            vec![
                regime(&[0, 1], &[0], DependencePremise::Dependence, 1, 1),
                regime(&[0, 1], &[1], DependencePremise::Dependence, 1, 1),
            ],
            &[],
        );
        let x = [1.0, -1.0];
        let y = [0.5, -0.5];
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let samples = [sample(&columns, 1), sample(&columns, 1)];
        let refusal = assemble_experiment_constraints(&contract, &samples, u64::MAX)
            .expect_err("a dependence premise is not this recipe");
        assert_eq!(
            refusal,
            MomentAssemblyRefusal::InconsistentRegime(RegimeInconsistency::DependentDisturbance {
                regime: 0,
            })
        );
    }

    #[test]
    fn mismatched_population_and_unit_set_are_refused() {
        let populations = accepted(
            vec![
                regime(&[0, 1], &[0], DependencePremise::Independence, 1, 1),
                regime(&[0, 1], &[1], DependencePremise::Independence, 2, 2),
            ],
            &[],
        );
        let x = [1.0, -1.0];
        let y = [1.0, -1.0];
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let samples = [sample(&columns, 1), sample(&columns, 2)];
        assert_eq!(
            assemble_experiment_constraints(&populations, &samples, u64::MAX).unwrap_err(),
            MomentAssemblyRefusal::InconsistentRegime(RegimeInconsistency::PopulationMismatch {
                left: 0,
                right: 1,
            })
        );

        let units = accepted(
            vec![
                regime(&[0, 1], &[0], DependencePremise::Independence, 1, 1),
                regime(&[0, 1, 2], &[1], DependencePremise::Independence, 2, 1),
            ],
            &[],
        );
        let z = [0.0, 0.0];
        let wider = [
            UnitColumn { unit: unit(0), values: &x },
            UnitColumn { unit: unit(1), values: &y },
            UnitColumn { unit: unit(2), values: &z },
        ];
        assert_eq!(
            assemble_experiment_constraints(
                &units,
                &[sample(&columns, 1), sample(&wider, 2)],
                u64::MAX,
            )
            .expect_err("unit sets differ"),
            MomentAssemblyRefusal::InconsistentRegime(RegimeInconsistency::UnitSetMismatch {
                left: 0,
                right: 1,
            })
        );
    }

    #[test]
    fn repeated_covering_rows_are_kept_without_a_rank_test() {
        let x = [1.0, -1.0];
        let y = [0.5, -0.5];
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let contract = accepted(
            vec![
                regime(&[0, 1], &[0], DependencePremise::Independence, 1, 1),
                regime(&[0, 1], &[0], DependencePremise::Independence, 2, 1),
                regime(&[0, 1], &[1], DependencePremise::Independence, 3, 1),
            ],
            &[],
        );
        let samples = [sample(&columns, 1), sample(&columns, 2), sample(&columns, 3)];
        let assembled = assemble_experiment_constraints(&contract, &samples, u64::MAX)
            .expect("duplicate covering rows are still constraints");
        let into_y = block(&assembled, 1);
        assert_eq!(into_y.rows.len(), 2);
        assert_eq!(into_y.rows[0].probe, unit(0));
        assert_eq!(into_y.rows[1].probe, unit(0));
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "Neumaier recovers this shifted column exactly; a running f64 sum does not"
    )]
    fn compensated_sum_keeps_a_shifted_column() {
        let shift = 2.0_f64.powi(53);
        let column = [shift, 1.0, -shift, 1.0];
        assert_eq!(compensated_sum(column.into_iter()), 2.0);
        let mut naive = 0.0_f64;
        for value in column {
            naive += value;
        }
        assert_eq!(naive, 1.0);
    }

    #[test]
    fn shifted_intervention_column_keeps_the_experimental_effect() {
        let shift = 1.0e8;
        let x = [shift + 1.0, shift - 1.0];
        let y = [0.5, -0.5];
        let passive_x = [1.0, -1.0, 1.0, -1.0];
        let intervened_y = [1.0, 1.0, -1.0, -1.0];
        let shifted =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let other = [
            UnitColumn { unit: unit(0), values: &passive_x },
            UnitColumn { unit: unit(1), values: &intervened_y },
        ];
        let contract = accepted(
            vec![
                regime(&[0, 1], &[0], DependencePremise::Independence, 1, 1),
                regime(&[0, 1], &[1], DependencePremise::Independence, 2, 1),
            ],
            &[],
        );
        let assembled = assemble_experiment_constraints(
            &contract,
            &[sample(&shifted, 1), sample(&other, 2)],
            u64::MAX,
        )
        .expect("a shifted intervention column still has a second moment");
        assert_close(assembled.moments[0].mean[0], shift);
        assert_close(assembled.moments[0].covariance[0], 1.0);
        let into_y = block(&assembled, 1);
        let (coefficients, rhs) = row(into_y, 0, 0);
        assert_eq!(coefficients, &[1.0]);
        assert_close(rhs, 0.5);
    }

    /// Nonzero means and a residual, so the through-origin slope is a different number.
    ///
    /// The assembled effect is `Cov(x, y) / Var(x)`. Dropping the intercept and
    /// regressing the raw columns would return `Σxy / Σx²` and fail this check.
    #[test]
    fn centered_effect_keeps_the_intercept() {
        let x = [1.0, 2.0, 4.0, 7.0];
        let y = [1.0, 3.0, 2.0, 6.0];
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let contract =
            accepted(vec![regime(&[0, 1], &[0], DependencePremise::Independence, 1, 1)], &[]);
        let assembled =
            assemble_experiment_constraints(&contract, &[sample(&columns, 1)], u64::MAX)
                .expect("the passive target is covered");
        let into_y = block(&assembled, 1);
        let (coefficients, effect) = row(into_y, 0, 0);
        assert_eq!(coefficients, &[1.0]);
        let mean_x = 3.5;
        let mean_y = 3.0;
        let mut centered = 0.0;
        let mut centered_x = 0.0;
        let mut raw = 0.0;
        let mut raw_x = 0.0;
        for (&xv, yv) in x.iter().zip(y) {
            centered += (xv - mean_x) * (yv - mean_y);
            centered_x += (xv - mean_x) * (xv - mean_x);
            raw += xv * yv;
            raw_x += xv * xv;
        }
        let with_intercept = centered / centered_x;
        let without_intercept = raw / raw_x;
        assert!((with_intercept - without_intercept).abs() > 1e-2);
        assert_close(effect, with_intercept);
        assert_close(assembled.moments[0].mean[0], mean_x);
        assert_close(assembled.moments[0].mean[1], mean_y);
        assert_close(assembled.moments[0].covariance[0], centered_x / x.len() as f64);
    }

    #[test]
    fn byte_estimate_refuses_overflow() {
        assert_eq!(
            experiment_constraint_bytes(u64::MAX, 2, 1, 0),
            Err(MomentAssemblyRefusal::ResourceOverflow)
        );
        assert_eq!(experiment_constraint_bytes(3, 4, 8, 0), Ok(576));
        assert_eq!(experiment_constraint_bytes(3, 4, 8, 4), Ok(672));
    }

    fn three_unit_columns<'a>(x: &'a [f64], y: &'a [f64], z: &'a [f64]) -> [UnitColumn<'a>; 3] {
        [
            UnitColumn { unit: unit(0), values: x },
            UnitColumn { unit: unit(1), values: y },
            UnitColumn { unit: unit(2), values: z },
        ]
    }

    #[test]
    fn covered_targets_keep_their_blocks_and_uncovered_pairs_are_listed() {
        // Without the regime that clamps (x, y), no regime clamps y while z stays
        // passive: the pair (1, 2) is uncovered and target 2 has no block.
        let independent = DependencePremise::Independence;
        let contract = accepted(
            vec![
                regime(&[0, 1, 2], &[0], independent, 1, 1),
                regime(&[0, 1, 2], &[1, 2], independent, 2, 1),
                regime(&[0, 1, 2], &[0, 2], independent, 3, 1),
            ],
            &[],
        );
        let (x0, y0, z0) = ([1.0, -1.0], [0.5, -0.5], [0.5, -0.5]);
        let (x1, y1, z1) = ([1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]);
        let (x2, y2, z2) = ([1.0, -1.0, 1.0, -1.0], [0.5, -0.5, 0.5, -0.5], [1.0, 1.0, -1.0, -1.0]);
        let c0 = three_unit_columns(&x0, &y0, &z0);
        let c1 = three_unit_columns(&x1, &y1, &z1);
        let c2 = three_unit_columns(&x2, &y2, &z2);
        let assembled = assemble_experiment_constraints(
            &contract,
            &[sample(&c0, 1), sample(&c1, 2), sample(&c2, 3)],
            u64::MAX,
        )
        .expect("two targets are covered");
        assert_eq!(assembled.uncovered, vec![MissingPair { from: unit(1), into: unit(2) }]);
        let targets: Vec<u32> = assembled.blocks.iter().map(|block| block.target.raw()).collect();
        assert_eq!(targets, vec![0, 1]);
        // Regime 0 clamps x; y = 0.5 x and z = 0.5 x in that sample.
        let (from_x, rhs) = row(block(&assembled, 1), 0, 0);
        assert_eq!(from_x, &[1.0, 0.5]);
        assert_close(rhs, 0.5);
    }

    #[test]
    fn a_dropped_passive_unit_is_refused_by_name() {
        // The mask drops unit 2, which no regime clamps. The row for target 1 would
        // omit the term b_12 t_2j for the dropped passive unit.
        let independent = DependencePremise::Independence;
        let contract = accepted(
            vec![
                regime(&[0, 1, 2], &[0], independent, 1, 1),
                regime(&[0, 1, 2], &[1], independent, 2, 1),
            ],
            &[2],
        );
        let (x, y) = ([1.0, -1.0], [0.5, -0.5]);
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let refusal = assemble_experiment_constraints(
            &contract,
            &[sample(&columns, 1), sample(&columns, 2)],
            u64::MAX,
        )
        .expect_err("a dropped passive unit loses its term");
        assert_eq!(refusal, MomentAssemblyRefusal::DroppedUnitNotClamped { unit: unit(2) });
    }

    #[test]
    fn the_byte_budget_is_enforced_before_any_moment_is_accumulated() {
        let x0 = [1.0, -1.0];
        let (y0, z0) = ([0.5, -0.5], [0.5, -0.5]);
        let (x1, y1, z1) = ([1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [1.0, -1.0, -1.0, 1.0]);
        let (x2, y2, z2) = ([1.0, -1.0, 1.0, -1.0], [0.5, -0.5, 0.5, -0.5], [1.0, 1.0, -1.0, -1.0]);
        let (x3, y3, z3) = ([2.0, 0.0, -2.0, 0.0], [1.0, 1.0, -1.0, -1.0], [1.0, 0.5, -1.0, -0.5]);
        let c = [
            three_unit_columns(&x0, &y0, &z0),
            three_unit_columns(&x1, &y1, &z1),
            three_unit_columns(&x2, &y2, &z2),
            three_unit_columns(&x3, &y3, &z3),
        ];
        let samples = [sample(&c[0], 1), sample(&c[1], 2), sample(&c[2], 3), sample(&c[3], 4)];
        // 576 bytes of moments and constraints plus 3 * 4 * 8 = 96 bytes of centered copies.
        let refusal = assemble_experiment_constraints(&synthetic_contract(), &samples, 671)
            .expect_err("one byte short");
        assert_eq!(refusal, MomentAssemblyRefusal::BudgetExceeded { requested: 672, limit: 671 });
        assert!(assemble_experiment_constraints(&synthetic_contract(), &samples, 672).is_ok());
    }

    #[test]
    fn clamps_in_units_that_differ_by_seven_orders_are_not_degenerate() {
        // Unit 1 is measured in units 1e7 times smaller than unit 0: Var(x) = 1 and
        // Var(w) = 1e-14, so a floor relative to the largest entry of the intervention
        // covariance (1e-12) would call w constant. On the correlation scale the pair is
        // orthogonal and perfectly conditioned.
        // y = 2 x + 3e7 w, so the experimental effects are 2 and 3e7.
        let a = [1.0, 1.0, -1.0, -1.0];
        let b = [1.0, -1.0, 1.0, -1.0];
        let c = [1.0, -1.0, -1.0, 1.0];
        let w_b: Vec<f64> = b.iter().map(|value| value * 1.0e-7).collect();
        let y_a: Vec<f64> = a.iter().zip(&b).map(|(p, q)| 2.0 * p + 3.0 * q).collect();
        let w_c: Vec<f64> = c.iter().map(|value| value * 1.0e-7).collect();
        let independent = DependencePremise::Independence;
        let contract = accepted(
            vec![
                regime(&[0, 1, 2], &[0, 1], independent, 1, 1),
                regime(&[0, 1, 2], &[2, 1], independent, 2, 1),
                regime(&[0, 1, 2], &[0, 2], independent, 3, 1),
            ],
            &[],
        );
        // Regime 0 clamps (x, w) and observes y. Regime 1 clamps (y, w) and observes x.
        // Regime 2 clamps (x, y) and observes w.
        let first = three_unit_columns(&a, &w_b, &y_a);
        let second = three_unit_columns(&a, &w_b, &c);
        let third = three_unit_columns(&a, &w_c, &b);
        let assembled = assemble_experiment_constraints(
            &contract,
            &[sample(&first, 1), sample(&second, 2), sample(&third, 3)],
            u64::MAX,
        )
        .expect("a badly scaled but orthogonal intervention is not degenerate");
        let into_y = block(&assembled, 2);
        let (probe_x, rhs_x) = row(into_y, 0, 0);
        assert_eq!(probe_x, &[1.0, 0.0]);
        assert!((rhs_x - 2.0).abs() <= 1.0e-9, "effect of x on y: {rhs_x}");
        let (probe_w, rhs_w) = row(into_y, 0, 1);
        assert_eq!(probe_w, &[0.0, 1.0]);
        assert!((rhs_w - 3.0e7).abs() <= 3.0e7 * 1.0e-9, "effect of w on y: {rhs_w}");
    }

    #[test]
    fn rank_deficient_constraint_returns_the_null_directions() {
        // Both regimes observe z. Clamping x and clamping y produce the same row
        // [1, 1] for the direct effects of (x, y) into z, so the column rank is 1.
        let independent = DependencePremise::Independence;
        let contract = accepted(
            vec![
                regime(&[0, 1, 2], &[0], independent, 1, 1),
                regime(&[0, 1, 2], &[1], independent, 2, 1),
            ],
            &[],
        );
        let (x, y, z) = ([1.0, -1.0], [1.0, -1.0], [1.0, -1.0]);
        let columns = three_unit_columns(&x, &y, &z);
        let assembled = assemble_experiment_constraints(
            &contract,
            &[sample(&columns, 1), sample(&columns, 2)],
            u64::MAX,
        )
        .expect("z is covered");
        assert_eq!(assembled.blocks.len(), 1);
        let into_z = &assembled.blocks[0];
        assert_eq!(into_z.sources, vec![unit(0), unit(1)]);
        assert_eq!(into_z.coefficients, vec![1.0, 1.0, 1.0, 1.0]);
        let ConstraintColumnReport::Deficient { column_rank, directions } =
            constraint_column_report(into_z)
        else {
            panic!("parallel constraint rows must not invent direct effects");
        };
        assert_eq!(column_rank, 1);
        assert_eq!(directions, vec![vec![-1.0, 1.0]]);
        let width = into_z.sources.len();
        for row_index in 0..into_z.rhs.len() {
            let mut total = 0.0;
            for (col, weight) in directions[0].iter().enumerate() {
                total += into_z.coefficients[row_index * width + col] * weight;
            }
            assert!(total.abs() <= 1e-12, "null direction missed the row: {total}");
        }
    }

    #[test]
    fn full_column_rank_constraint_returns_the_direct_effects() {
        let block = TargetConstraintBlock {
            target: unit(2),
            sources: vec![unit(0), unit(1)],
            rows: vec![
                ConstraintRow { regime: 0, probe: unit(0) },
                ConstraintRow { regime: 3, probe: unit(0) },
                ConstraintRow { regime: 3, probe: unit(1) },
            ],
            coefficients: vec![1.0, 0.5, 1.0, 0.0, 0.0, 1.0],
            rhs: vec![0.5, 0.25, 0.5],
        };
        let ConstraintColumnReport::Unique { coefficients } = constraint_column_report(&block)
        else {
            panic!("these rows pin both direct effects");
        };
        assert_eq!(coefficients, vec![0.25, 0.5]);
    }

    #[test]
    fn a_sample_source_that_disagrees_with_the_regime_is_refused() {
        let x = [1.0, -1.0];
        let y = [0.5, -0.5];
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let contract =
            accepted(vec![regime(&[0, 1], &[0], DependencePremise::Independence, 4, 1)], &[]);
        let refusal = assemble_experiment_constraints(&contract, &[sample(&columns, 9)], u64::MAX)
            .expect_err("the sample source is not the declared source");
        assert_eq!(
            refusal,
            MomentAssemblyRefusal::SourceMismatch {
                regime: 0,
                declared: SourceIdentity::from_raw(4),
                sample: SourceIdentity::from_raw(9),
            }
        );
    }
}
