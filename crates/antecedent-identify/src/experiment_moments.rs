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
//! on the intervened columns of that regime.
//!
//! The assembled blocks are the constraint system. This module does not solve
//! it, does not form a pseudoinverse, does not test the rank of the constraint
//! matrix, and does not recover a direct-effect matrix or a disturbance
//! covariance. Weak stability is not checked. A set of regimes that misses the
//! pair condition (some ordered pair is never clamped at the source while the
//! target stays passive) is refused, as is a regime set whose dependence
//! premise, population, or retained units disagree with one shared independent
//! experiment family.

use std::collections::BTreeSet;
use std::fmt;

use antecedent_core::VariableId;

use crate::experiment_regime::{AcceptedRegimeContract, DependencePremise, ExperimentRegime};

/// Bytes in one assembled `f64` entry.
const F64_BYTES: u64 = 8;

/// Relative floor for a pivot of the intervention covariance.
///
/// A pivot at or below this fraction of the matrix scale means the intervened
/// columns do not have a positive-definite sample covariance, so the canonical
/// experimental effects are undefined. It is not a rank test of the constraint
/// matrix.
const INTERVENTION_PIVOT_FLOOR: f64 = 1e-12;

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeSample<'a> {
    /// Columns of this regime.
    pub columns: &'a [UnitColumn<'a>],
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
    /// The byte estimate overflowed.
    ResourceOverflow,
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
    /// One block per retained unit, in unit order.
    pub blocks: Vec<TargetConstraintBlock>,
    /// Byte estimate of the moment matrices and the constraint entries.
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
            Self::ResourceOverflow => {
                write!(f, "the moment and constraint byte estimate overflowed")
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

/// Bytes occupied by `regimes` moment matrices and `constraint_rows` constraint rows
/// on `units` units.
///
/// Each regime contributes `units * (units + 1)` floats (means and the covariance
/// matrix). Each constraint row contributes `units` floats (`units - 1`
/// coefficients and one right-hand side). Index storage is not included. The
/// arithmetic is checked; overflow is [`MomentAssemblyRefusal::ResourceOverflow`].
pub fn experiment_constraint_bytes(
    units: u64,
    regimes: u64,
    constraint_rows: u64,
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
    moments.checked_add(constraints).ok_or(MomentAssemblyRefusal::ResourceOverflow)
}

/// Assemble centered moments and experimental-effect constraints.
///
/// The returned blocks are not a coefficient estimate. A refusal does not carry
/// a partial block.
#[must_use = "a refusal must not be read as an assembled constraint"]
pub fn assemble_experiment_constraints(
    contract: &AcceptedRegimeContract,
    samples: &[RegimeSample<'_>],
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
    let units = retained_units(&regimes[0], dropped);
    let unit_list: Vec<VariableId> = units.iter().copied().collect();
    let missing = missing_pairs(regimes, &units);
    if !missing.is_empty() {
        return Err(MomentAssemblyRefusal::InadequateInterventionCoverage { missing });
    }
    let prepared = prepare_samples(regimes, dropped, samples)?;
    let row_count = constraint_row_count(regimes, &units)?;
    let bytes =
        experiment_constraint_bytes(u64_len(unit_list.len())?, u64_len(regimes.len())?, row_count)?;
    let moments = accumulate_moments(&unit_list, &prepared)?;
    let blocks = assemble_blocks(&unit_list, regimes, &units, &moments)?;
    Ok(AssembledExperimentConstraints { units: unit_list, moments, blocks, bytes })
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
) -> Result<u64, MomentAssemblyRefusal> {
    let mut rows = 0_u64;
    for regime in regimes {
        let targeted = regime.targets.iter().filter(|target| units.contains(&target.unit)).count();
        let passive = units.len() - targeted;
        let added = u64_len(targeted)?
            .checked_mul(u64_len(passive)?)
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

fn assemble_blocks(
    units: &[VariableId],
    regimes: &[ExperimentRegime],
    retained: &BTreeSet<VariableId>,
    moments: &[ExperimentMoments],
) -> Result<Vec<TargetConstraintBlock>, MomentAssemblyRefusal> {
    let mut blocks = Vec::with_capacity(units.len());
    for (target_index, &target) in units.iter().enumerate() {
        let sources: Vec<VariableId> =
            units.iter().copied().filter(|unit| *unit != target).collect();
        let mut rows = Vec::new();
        let mut coefficients = Vec::new();
        let mut rhs = Vec::new();
        for (regime_index, regime) in regimes.iter().enumerate() {
            if regime.targets.iter().any(|item| item.unit == target) {
                continue;
            }
            let probes: Vec<VariableId> = regime
                .targets
                .iter()
                .map(|item| item.unit)
                .filter(|unit| retained.contains(unit))
                .collect();
            let mut probe_order = probes;
            probe_order.sort();
            let passive: Vec<usize> = units
                .iter()
                .enumerate()
                .filter(|(_, unit)| !probe_order.contains(unit))
                .map(|(index, _)| index)
                .collect();
            let mut probe_index = Vec::with_capacity(probe_order.len());
            for probe in &probe_order {
                let Some(index) = units.iter().position(|unit| unit == probe) else {
                    return Err(MomentAssemblyRefusal::SampleUnits { regime: regime_index });
                };
                probe_index.push(index);
            }
            let effects = experimental_effects(
                regime_index,
                &moments[regime_index].covariance,
                units.len(),
                &probe_index,
                &passive,
            )?;
            for (probe_slot, &probe) in probe_order.iter().enumerate() {
                rows.push(ConstraintRow { regime: regime_index, probe });
                for &source in &sources {
                    let Some(coefficient) = row_coefficient(
                        source,
                        probe,
                        &probe_order,
                        &passive,
                        units,
                        probe_slot,
                        &effects,
                    ) else {
                        return Err(MomentAssemblyRefusal::SampleUnits { regime: regime_index });
                    };
                    coefficients.push(coefficient);
                }
                let Some(target_slot) = passive.iter().position(|index| *index == target_index)
                else {
                    return Err(MomentAssemblyRefusal::SampleUnits { regime: regime_index });
                };
                rhs.push(effects[target_slot * probe_order.len() + probe_slot]);
            }
        }
        blocks.push(TargetConstraintBlock { target, sources, rows, coefficients, rhs });
    }
    Ok(blocks)
}

fn row_coefficient(
    source: VariableId,
    probe: VariableId,
    probes: &[VariableId],
    passive: &[usize],
    units: &[VariableId],
    probe_slot: usize,
    effects: &[f64],
) -> Option<f64> {
    if source == probe {
        return Some(1.0);
    }
    if probes.contains(&source) {
        return Some(0.0);
    }
    let source_index = units.iter().position(|unit| *unit == source)?;
    let passive_slot = passive.iter().position(|index| *index == source_index)?;
    effects.get(passive_slot * probes.len() + probe_slot).copied()
}

/// Experimental effects of each intervened column on each passive column.
///
/// Layout is row-major `passive.len() * intervened.len()`. A singular
/// intervention covariance is [`MomentAssemblyRefusal::DegenerateIntervention`].
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
    let mut factor = vec![0.0; length_product(j, j)?];
    for (row, &left) in intervened.iter().enumerate() {
        for (col, &right) in intervened.iter().enumerate() {
            factor[row * j + col] = covariance[left * n + right];
        }
    }
    let scale = factor.iter().fold(0.0_f64, |best, value| best.max(value.abs()));
    if !scale.is_finite() {
        return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
    }
    let mut rhs = vec![0.0; length_product(j, u)?];
    for (col, &passive_index) in passive.iter().enumerate() {
        for (row, &probe_index) in intervened.iter().enumerate() {
            rhs[row * u + col] = covariance[probe_index * n + passive_index];
        }
    }
    for col in 0..j {
        let mut pivot_row = col;
        let mut best = factor[col * j + col].abs();
        for row in (col + 1)..j {
            let value = factor[row * j + col].abs();
            if value > best {
                best = value;
                pivot_row = row;
            }
        }
        if best <= scale * INTERVENTION_PIVOT_FLOOR {
            return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
        }
        if pivot_row != col {
            swap_factor_rows(&mut factor, j, col, pivot_row);
            swap_rhs_rows(&mut rhs, u, col, pivot_row);
        }
        let diag = factor[col * j + col];
        for row in (col + 1)..j {
            let multiplier = factor[row * j + col] / diag;
            for k in col..j {
                factor[row * j + k] -= multiplier * factor[col * j + k];
            }
            for k in 0..u {
                rhs[row * u + k] -= multiplier * rhs[col * u + k];
            }
        }
    }
    for col in (0..j).rev() {
        let diag = factor[col * j + col];
        if !diag.is_finite() || diag.abs() <= scale * INTERVENTION_PIVOT_FLOOR {
            return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
        }
        for k in 0..u {
            let mut value = rhs[col * u + k];
            for row in (col + 1)..j {
                value -= factor[col * j + row] * rhs[row * u + k];
            }
            let solved = value / diag;
            if !solved.is_finite() {
                return Err(MomentAssemblyRefusal::DegenerateIntervention { regime });
            }
            rhs[col * u + k] = solved;
        }
    }
    let mut effects = vec![0.0; length_product(u, j)?];
    for passive_slot in 0..u {
        for probe_slot in 0..j {
            effects[passive_slot * j + probe_slot] = rhs[probe_slot * u + passive_slot];
        }
    }
    Ok(effects)
}

fn swap_factor_rows(factor: &mut [f64], width: usize, left: usize, right: usize) {
    for col in 0..width {
        factor.swap(left * width + col, right * width + col);
    }
}

fn swap_rhs_rows(rhs: &mut [f64], width: usize, left: usize, right: usize) {
    for col in 0..width {
        rhs.swap(left * width + col, right * width + col);
    }
}

/// Neumaier compensated sum. A running `f64` total drops a unit that sits next
/// to a shift larger than the mantissa; the correction term keeps it.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let next = total + value;
        correction += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    total + correction
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
        AssembledExperimentConstraints, MomentAssemblyRefusal, RegimeInconsistency, RegimeSample,
        TargetConstraintBlock, UnitColumn, assemble_experiment_constraints, compensated_sum,
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

    fn sample<'a>(columns: &'a [UnitColumn<'a>]) -> RegimeSample<'a> {
        RegimeSample { columns }
    }

    fn assert_close(actual: f64, expected: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!((actual - expected).abs() <= scale * 1e-12, "actual {actual} expected {expected}");
    }

    fn row<'a>(block: &'a TargetConstraintBlock, regime: usize, probe: u32) -> (&'a [f64], f64) {
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
        let samples = [sample(&c0), sample(&c1), sample(&c2), sample(&c3)];
        let assembled = assemble_experiment_constraints(&synthetic_contract(), &samples)
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
        let bytes = experiment_constraint_bytes(3, 4, u64::try_from(rows).expect("row count"))
            .expect("byte count");
        assert_eq!(assembled.bytes, bytes);
        assert_eq!(assembled.bytes, 576);
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
        let refusal = assemble_experiment_constraints(&contract, &[sample(&columns)])
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
                regime(&[0, 1], &[1], DependencePremise::Independence, 2, 1),
            ],
            &[],
        );
        let x = [1.0, -1.0];
        let y = [0.5, -0.5];
        let columns =
            [UnitColumn { unit: unit(0), values: &x }, UnitColumn { unit: unit(1), values: &y }];
        let samples = [sample(&columns), sample(&columns)];
        let refusal = assemble_experiment_constraints(&contract, &samples)
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
        let samples = [sample(&columns), sample(&columns)];
        assert_eq!(
            assemble_experiment_constraints(&populations, &samples).unwrap_err(),
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
            assemble_experiment_constraints(&units, &[sample(&columns), sample(&wider)])
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
        let samples = [sample(&columns), sample(&columns), sample(&columns)];
        let assembled = assemble_experiment_constraints(&contract, &samples)
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
        let assembled =
            assemble_experiment_constraints(&contract, &[sample(&shifted), sample(&other)])
                .expect("a shifted intervention column still has a second moment");
        assert_close(assembled.moments[0].mean[0], shift);
        assert_close(assembled.moments[0].covariance[0], 1.0);
        let into_y = block(&assembled, 1);
        let (coefficients, rhs) = row(into_y, 0, 0);
        assert_eq!(coefficients, &[1.0]);
        assert_close(rhs, 0.5);
    }

    #[test]
    fn byte_estimate_refuses_overflow() {
        assert_eq!(
            experiment_constraint_bytes(u64::MAX, 2, 1),
            Err(MomentAssemblyRefusal::ResourceOverflow)
        );
        assert_eq!(experiment_constraint_bytes(3, 4, 8), Ok(576));
    }
}
