//! Whole-procedure experimental recovery bootstrap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::convert::Infallible;

use antecedent_kernels::unbiased_index;

use super::{
    AcceptedRegimeContract, BOOTSTRAP_REPLICATE_OPERATION, BootstrapSpec, CausalRng,
    CoefficientCovarianceEstimator, DrawStatus, ExecutionContext, ExperimentRows,
    ExperimentalFitError, ExperimentalFitOptions, ExperimentalRecoveryFit,
    RecoveredCoefficientUncertainty, RecoveredDrawOrigin, RecoveredEstimate, RegimeSample,
    ResamplingRule, UnitColumn, accept, failed_draw, fit_core, keyed_stream,
};

/// Validated resampling plan of every regime.
pub(super) struct ResamplingPlan {
    rule: ResamplingRule,
    rows: Vec<usize>,
    groups: Option<Vec<Vec<Vec<usize>>>>,
    block_length: usize,
}

impl ResamplingPlan {
    pub(super) fn new(
        rule: ResamplingRule,
        samples: &[RegimeSample<'_>],
        spec: &BootstrapSpec<'_>,
    ) -> Result<Self, ExperimentalFitError> {
        let rows: Vec<usize> = samples
            .iter()
            .map(|sample| sample.columns.first().map_or(0, |column| column.values.len()))
            .collect();
        let groups = match spec.sampling_units {
            None => None,
            Some(labels) => Some(group_rows(labels, &rows)?),
        };
        let shortest = rows.iter().copied().min().unwrap_or(0);
        let mut block_length = 1;
        match rule {
            ResamplingRule::RowSubjectIid => {}
            ResamplingRule::WholeCluster => {
                if groups.is_none() {
                    return Err(ExperimentalFitError::MissingSamplingUnits);
                }
            }
            ResamplingRule::ContiguousBlock => {
                block_length = spec
                    .block_length
                    .filter(|length| *length > 0 && *length < shortest)
                    .ok_or(ExperimentalFitError::BlockLength { rows: shortest })?;
            }
            ResamplingRule::RowIid => return Err(ExperimentalFitError::RowLevelDraw(rule)),
        }
        Ok(Self { rule, rows, groups, block_length })
    }

    pub(super) fn draw(&self, regime: usize, rng: &mut CausalRng) -> Vec<usize> {
        let rows = self.rows[regime];
        let mut out = Vec::with_capacity(rows);
        match (self.rule, &self.groups) {
            (ResamplingRule::ContiguousBlock, _) => {
                let starts = rows - self.block_length + 1;
                while out.len() < rows {
                    let start = unbiased_index(rng, starts);
                    out.extend(start..start + self.block_length);
                }
                out.truncate(rows);
            }
            (_, Some(groups)) => {
                let units = &groups[regime];
                for _ in 0..units.len() {
                    out.extend_from_slice(&units[unbiased_index(rng, units.len())]);
                }
            }
            (_, None) => {
                for _ in 0..rows {
                    out.push(unbiased_index(rng, rows));
                }
            }
        }
        out
    }
}

fn group_rows(
    labels: &[&[u64]],
    rows: &[usize],
) -> Result<Vec<Vec<Vec<usize>>>, ExperimentalFitError> {
    if labels.len() != rows.len() {
        return Err(ExperimentalFitError::SamplingUnitCount {
            labels: labels.len(),
            regimes: rows.len(),
        });
    }
    labels
        .iter()
        .zip(rows)
        .enumerate()
        .map(|(regime, (regime_labels, &count))| {
            if regime_labels.len() != count {
                return Err(ExperimentalFitError::SamplingUnitShape {
                    regime,
                    labels: regime_labels.len(),
                    rows: count,
                });
            }
            let mut by_label: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            for (row, label) in regime_labels.iter().enumerate() {
                by_label.entry(*label).or_default().push(row);
            }
            Ok(by_label.into_values().collect())
        })
        .collect()
}

fn run_replicate(
    accepted: &AcceptedRegimeContract,
    samples: &[RegimeSample<'_>],
    plan: &ResamplingPlan,
    options: &ExperimentalFitOptions,
    replicate: usize,
    seed: u64,
) -> Result<ExperimentalRecoveryFit, ExperimentalFitError> {
    let mut rng = keyed_stream(BOOTSTRAP_REPLICATE_OPERATION, 0, replicate, seed);
    let data: Vec<Vec<Vec<f64>>> = samples
        .iter()
        .enumerate()
        .map(|(regime, sample)| {
            let rows = plan.draw(regime, &mut rng);
            sample
                .columns
                .iter()
                .map(|column| rows.iter().map(|&row| column.values[row]).collect())
                .collect()
        })
        .collect();
    let columns: Vec<Vec<UnitColumn<'_>>> = data
        .iter()
        .zip(samples)
        .map(|(regime, sample)| {
            regime
                .iter()
                .zip(sample.columns)
                .map(|(values, column)| UnitColumn { unit: column.unit, values: values.as_slice() })
                .collect()
        })
        .collect();
    let resampled: Vec<RegimeSample<'_>> = columns
        .iter()
        .zip(samples)
        .map(|(regime, sample)| RegimeSample { columns: regime.as_slice(), source: sample.source })
        .collect();
    fit_core(accepted, &resampled, options)
}

/// Whole-procedure nonparametric bootstrap of the recovery.
///
/// `spec.design` must allow `spec.rule`
/// ([`DeclaredSamplingDesign::apply_resampling`]): a row-level draw is refused
/// under every design, a subject trajectory or a cluster is drawn whole, and a
/// dependent series is drawn in contiguous blocks. Units are resampled within
/// each regime, with replacement, to the regime's own count, so regimes stay
/// independent and keep their clamp sets. Each replicate re-runs the moment
/// assembly, the rank-aware block solves, and the disturbance estimate. A
/// replicate whose recovery fails is retained as a marked draw with
/// [`DrawStatus::Failed`]; its weight is not renormalized away.
///
/// Replicate `b` opens [`BOOTSTRAP_REPLICATE_OPERATION`] with replicate `b`.
/// Replicates are scheduled in batches of `spec.batch` on `ctx`; neither the
/// batch size nor the worker count changes a draw.
///
/// # Errors
///
/// A refused design or rule, a malformed label or block length, a refused
/// contract, a fit of the original rows that fails, or a law fault.
pub fn bootstrap_experimental_recovery(
    rows: &ExperimentRows<'_>,
    options: &ExperimentalFitOptions,
    spec: &BootstrapSpec<'_>,
    ctx: &ExecutionContext,
) -> Result<RecoveredEstimate, ExperimentalFitError> {
    if spec.replicates == 0 {
        return Err(ExperimentalFitError::DrawCount);
    }
    if spec.batch == 0 {
        return Err(ExperimentalFitError::BatchSize);
    }
    let bound =
        spec.design.apply_resampling(spec.rule).map_err(ExperimentalFitError::SamplingDesign)?;
    let accepted = accept(rows.contract)?;
    let replicate_options = ExperimentalFitOptions {
        covariance: CoefficientCovarianceEstimator::NotComputed,
        ..*options
    };
    let reference = fit_core(&accepted, rows.samples, &replicate_options)?;
    let plan = ResamplingPlan::new(bound.rule(), rows.samples, spec)?;
    let batches = spec.replicates.div_ceil(spec.batch);
    let outcomes = ctx.map_indexed(batches, |batch, _inner| {
        let start = batch * spec.batch;
        let end = (start + spec.batch).min(spec.replicates);
        Ok::<_, Infallible>(
            (start..end)
                .map(|replicate| {
                    run_replicate(
                        &accepted,
                        rows.samples,
                        &plan,
                        &replicate_options,
                        replicate,
                        spec.seed,
                    )
                })
                .collect::<Vec<_>>(),
        )
    });
    let outcomes: Vec<Result<ExperimentalRecoveryFit, ExperimentalFitError>> = match outcomes {
        Ok(batches) => batches.into_iter().flatten().collect(),
        Err(antecedent_core::execution::MapExecutionError::Admission(error)) => {
            return Err(ExperimentalFitError::Admission(error));
        }
        Err(antecedent_core::execution::MapExecutionError::Work(never)) => match never {},
    };
    let weight = 1.0 / spec.replicates as f64;
    let mut inputs = Vec::with_capacity(outcomes.len());
    let mut statuses = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        let drawn = outcome.and_then(|fit| {
            fit.draw_with_weight(weight).map_err(ExperimentalFitError::Uncertainty)
        });
        match drawn {
            Ok(input) => {
                statuses.push(reference.status_of(&input));
                inputs.push(input);
            }
            Err(error) => {
                statuses.push(DrawStatus::Failed { error });
                inputs.push(failed_draw(weight));
            }
        }
    }
    let law = RecoveredCoefficientUncertainty::retain(&reference.nodes, inputs)
        .map_err(ExperimentalFitError::Uncertainty)?;
    Ok(RecoveredEstimate {
        law,
        origin: RecoveredDrawOrigin::Bootstrap {
            replicates: spec.replicates,
            seed: spec.seed,
            design: spec.design,
            rule: spec.rule,
        },
        statuses,
        identified_posteriors: Vec::new(),
    })
}
