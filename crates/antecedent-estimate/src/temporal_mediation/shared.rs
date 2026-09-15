//! Shared circular-block bootstrap for a frozen-weight mixture of temporal
//! mediation atoms fitted on one series (graph-posterior atoms, each with its
//! own horizon-specific adjustment set).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ContrastFit, EstimationError, ExecutionContext, IdentifiedEstimand, LaggedColumn,
    MediationContrast, MediationDesign, MediationQuery, TemporalMediationBlockSe,
    TemporalMediationEstimate, TemporalMediationEstimator, TimeSeriesData,
    mediation_design_max_lag, ols_three_col, ols_two_col,
};
use crate::temporal_block::{
    AlignedRows, aligned_block_bootstrap, common_time_window, dependence_block_length,
    kernel_bias_scale, normal_equation_scores, score_effective_rows,
};

/// One temporal mediation atom prepared once on the original series, so a
/// shared circular-block bootstrap can refit its three mechanism regressions on
/// resampled lag-aligned rows.
pub struct PreparedTemporalMediation {
    estimator: TemporalMediationEstimator,
    design: MediationDesign,
    delta: f64,
    estimate: TemporalMediationEstimate,
    aligned: AlignedRows,
    structural_span: usize,
    contrast_scores: Option<[Vec<f64>; 3]>,
    persistence_probes: Option<[Vec<f64>; 3]>,
    normal_scores: Vec<Vec<f64>>,
}

impl TemporalMediationEstimator {
    /// Prepare one mediation atom for [`shared_mediation_block_bootstrap`]:
    /// the full-sample point fit (same as [`Self::estimate_with_adjustment`]),
    /// its lag-aligned rows, and the estimating scores the block length reads.
    ///
    /// # Errors
    ///
    /// Validation or point-fit failures.
    pub fn prepare_shared(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        adjustment: &[LaggedColumn],
        ctx: &ExecutionContext,
    ) -> Result<PreparedTemporalMediation, EstimationError> {
        let (mediator, delta) = self.validate(estimand, query)?;
        let design = Self::prepare_design(data, mediator, query, adjustment, &[], ctx)?;
        let point = self.fit_design(&design, None, delta)?;
        let estimate = self.estimate_from_fit(query, &point);
        let max_lag = mediation_design_max_lag(query, adjustment) as usize;
        let contrast_scores = design.contrast_scores(self.backend, &point);
        let persistence_probes = design.persistence_probes(self.backend, &point);
        let normal_scores = mechanism_normal_scores(&design, &point);
        Ok(PreparedTemporalMediation {
            estimator: self.clone(),
            aligned: AlignedRows { first_time: max_lag, rows: design.n },
            structural_span: max_lag + 1,
            design,
            delta,
            estimate,
            contrast_scores,
            persistence_probes,
            normal_scores,
        })
    }
}

impl PreparedTemporalMediation {
    /// Full-sample point estimate (no bootstrap SE attached).
    #[must_use]
    pub const fn estimate(&self) -> &TemporalMediationEstimate {
        &self.estimate
    }

    /// Lag-aligned rows of this atom's design.
    #[must_use]
    pub const fn aligned_rows(&self) -> AlignedRows {
        self.aligned
    }

    /// Total, Direct and Mediated refit on design rows `rows` (any count: a
    /// shared window can be shorter than this atom's own rows).
    fn contrasts_on_rows(&self, rows: &[usize]) -> Option<[f64; 3]> {
        let n = self.design.n;
        let gathered: Vec<Vec<f64>> = self
            .design
            .columns
            .chunks_exact(n)
            .map(|column| rows.iter().map(|&r| column.get(r).copied()).collect())
            .collect::<Option<_>>()?;
        let (t, m, y) = (&gathered[0], &gathered[1], &gathered[2]);
        let extras: Vec<&[f64]> = gathered[3..].iter().map(Vec::as_slice).collect();
        let backend = self.estimator.backend;
        let (a, ..) = ols_two_col(backend, t, m, &extras).ok()?;
        let (c_prime, b, ..) = ols_three_col(backend, t, m, y, &extras).ok()?;
        let (c, ..) = ols_two_col(backend, t, y, &extras).ok()?;
        Some([c * self.delta, c_prime * self.delta, a * b * self.delta])
    }
}

/// Every mechanism's normal-equation scores (intercept = residual series).
fn mechanism_normal_scores(design: &MediationDesign, point: &ContrastFit) -> Vec<Vec<f64>> {
    let (m, y) = (design.column(1), design.column(2));
    point
        .designs
        .iter()
        .zip([m, y, y])
        .filter_map(|(matrix, outcome)| {
            normal_equation_scores(matrix, design.n, matrix.len() / design.n, outcome)
        })
        .flatten()
        .collect()
}

const fn contrast_index(contrast: MediationContrast) -> usize {
    match contrast {
        MediationContrast::Total => 0,
        MediationContrast::Direct | MediationContrast::NaturalDirect => 1,
        MediationContrast::Mediated | MediationContrast::NaturalIndirect => 2,
    }
}

/// Shared circular-block SEs of a frozen-weight mediation mixture.
#[derive(Clone, Debug)]
pub struct SharedMediationBlockSe {
    /// Mixture SEs for Total, Direct and Mediated from the same replicates.
    pub block: TemporalMediationBlockSe,
    /// Requested contrast's mixture SE (repeats one of [`Self::block`]'s SEs).
    pub requested: Option<f64>,
    /// Structural lag span (deepest design lag + 1) over every atom.
    pub structural_span: usize,
}

/// Shared circular-block bootstrap of a frozen-weight mixture over mediation atoms.
///
/// Blocks of consecutive series times are resampled over the window where every
/// atom's lag window is available ([`crate::temporal_block::aligned_block_bootstrap`]);
/// each atom's rows keep their original lag windows, every atom's three
/// mechanism regressions are refit on the same resampled times, and Total,
/// Direct and Mediated are mixed with the frozen weights inside each replicate.
/// A replicate that cannot fit every atom is dropped for every contrast. The
/// block length is [`dependence_block_length`] over every atom's contrast and
/// normal-equation scores and the weighted mixture scores on the shared times;
/// SEs carry the fixed-b factor of that length and the kernel-bias factor of
/// the contrast and mixture scores. With one atom this is the same
/// replicate law as [`TemporalMediationEstimator::estimate_with_block_bootstrap`].
///
/// Returns `None` when the atoms share no series time or `weights` do not align.
#[must_use]
pub fn shared_mediation_block_bootstrap(
    atoms: &[&PreparedTemporalMediation],
    weights: &[f64],
    contrast: MediationContrast,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
) -> Option<SharedMediationBlockSe> {
    let total: f64 = weights.iter().sum();
    if atoms.is_empty() || atoms.len() != weights.len() || total.is_nan() || total <= 0.0 {
        return None;
    }
    let designs: Vec<AlignedRows> = atoms.iter().map(|atom| atom.aligned).collect();
    let (start, len) = common_time_window(&designs)?;
    let window = |series: &[f64], design: AlignedRows| -> Option<Vec<f64>> {
        let offset = start - design.first_time;
        series.get(offset..offset + len).map(<[f64]>::to_vec)
    };
    let mut contrast_windows: Vec<Vec<f64>> = Vec::new();
    let mut mixture: Option<[Vec<f64>; 3]> = Some([vec![0.0; len], vec![0.0; len], vec![0.0; len]]);
    let mut other_windows: Vec<Vec<f64>> = Vec::new();
    for (atom, weight) in atoms.iter().zip(weights) {
        match atom.contrast_scores.as_ref() {
            Some(scores) => {
                for (k, score) in scores.iter().enumerate() {
                    let Some(w) = window(score, atom.aligned) else {
                        mixture = None;
                        continue;
                    };
                    if let Some(mix) = mixture.as_mut() {
                        for (slot, value) in mix[k].iter_mut().zip(&w) {
                            *slot += weight / total * value;
                        }
                    }
                    contrast_windows.push(w);
                }
            }
            None => mixture = None,
        }
        other_windows.extend(atom.normal_scores.iter().filter_map(|s| window(s, atom.aligned)));
    }
    let mut target_scores: Vec<&[f64]> = contrast_windows.iter().map(Vec::as_slice).collect();
    if let Some(mix) = mixture.as_ref() {
        target_scores.extend(mix.iter().map(Vec::as_slice));
    }
    let mut block_scores = target_scores.clone();
    block_scores.extend(other_windows.iter().map(Vec::as_slice));
    let structural_span = atoms.iter().map(|atom| atom.structural_span).max().unwrap_or(1);
    // The scan reads every atom's scores; without replicates no interval is
    // published and the rule length is reported instead.
    let block_length = if replicates > 0 {
        dependence_block_length(structural_span, len, &block_scores)
    } else {
        antecedent_data::circular_block_length(structural_span, len)
    };
    let kernel_bias = kernel_bias_scale(&target_scores, block_length);
    // The short-series statistic reads every atom's persistence probes and
    // their weighted mixture, as the single-atom estimator does.
    let mut probe_windows: Vec<Vec<f64>> = Vec::new();
    let mut probe_mixture: Option<[Vec<f64>; 3]> =
        mixture.as_ref().map(|_| [vec![0.0; len], vec![0.0; len], vec![0.0; len]]);
    for (atom, weight) in atoms.iter().zip(weights) {
        let Some(probes) = atom.persistence_probes.as_ref() else {
            probe_mixture = None;
            continue;
        };
        for (k, probe) in probes.iter().enumerate() {
            let Some(w) = window(probe, atom.aligned) else {
                probe_mixture = None;
                continue;
            };
            if let Some(mix) = probe_mixture.as_mut() {
                for (slot, value) in mix[k].iter_mut().zip(&w) {
                    *slot += weight / total * value;
                }
            }
            probe_windows.push(w);
        }
    }
    let effective_rows = match probe_mixture.as_ref() {
        Some(mix) => {
            let mut targets: Vec<&[f64]> = probe_windows.iter().map(Vec::as_slice).collect();
            if atoms.len() > 1 {
                targets.extend(mix.iter().map(Vec::as_slice));
            }
            score_effective_rows(&targets, block_length)
        }
        None => f64::NAN,
    };
    let mut block = TemporalMediationBlockSe {
        total: None,
        direct: None,
        mediated: None,
        replicates_ok: 0,
        replicates_attempted: 0,
        block_length: block_length.clamp(1, len),
        rows: len,
        effective_rows,
        kernel_bias,
    };
    if replicates == 0 {
        return Some(SharedMediationBlockSe { block, requested: None, structural_span });
    }
    let draws =
        aligned_block_bootstrap(&designs, block_length, replicates, stream_base, ctx, |maps| {
            let mut mixed = [0.0; 3];
            for ((atom, rows), weight) in atoms.iter().zip(maps).zip(weights) {
                let values = atom.contrasts_on_rows(rows)?;
                for (slot, value) in mixed.iter_mut().zip(values) {
                    *slot += weight / total * value;
                }
            }
            Some(mixed.to_vec())
        })?;
    let draws = draws.with_kernel_bias(&target_scores);
    let [total_se, direct_se, mediated_se] = [0, 1, 2].map(|c| draws.se_result(c));
    block.total = total_se.se;
    block.direct = direct_se.se;
    block.mediated = mediated_se.se;
    block.replicates_ok = total_se.replicates_ok;
    block.replicates_attempted = draws.attempted;
    block.block_length = draws.block_length;
    let requested = [block.total, block.direct, block.mediated][contrast_index(contrast)];
    Some(SharedMediationBlockSe { block, requested, structural_span })
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };
    use antecedent_expr::CausalExprArena;

    use super::*;

    fn series(n: usize) -> (TimeSeriesData, MediationQuery, IdentifiedEstimand) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "m", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut y = vec![0.0; n];
        for (i, value) in t.iter_mut().enumerate() {
            *value = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
        }
        for i in 1..n {
            m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
            y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
        }
        let cols = [t, m, y]
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let mut arena = CausalExprArena::new();
        let functional = arena.temporal_mediation_ate(
            q.treatment,
            q.outcome,
            &q.mediators,
            antecedent_core::Value::f64(1.0),
            antecedent_core::Value::f64(0.0),
        );
        let estimand = IdentifiedEstimand::temporal_mediation(
            "temporal_mediation.mediated",
            Arc::clone(&q.mediators),
            functional,
        );
        (data, q, estimand)
    }

    /// One atom (or several identical atoms) is the single-design block
    /// bootstrap: same block length, same replicate streams, same SEs.
    #[test]
    fn one_atom_mixture_matches_single_design_block_bootstrap() {
        let (data, q, estimand) = series(240);
        let ctx = ExecutionContext::for_tests(11);
        let est = TemporalMediationEstimator::new();
        let (single, block) =
            est.estimate_with_block_bootstrap(&data, &estimand, &q, &[], 60, 0xABCD, &ctx).unwrap();
        let atom = est.prepare_shared(&data, &estimand, &q, &[], &ctx).unwrap();
        assert!((atom.estimate().effect.ate - single.effect.ate).abs() < 1e-15);
        for atoms in [vec![&atom], vec![&atom, &atom]] {
            let weights = vec![1.0; atoms.len()];
            let shared = shared_mediation_block_bootstrap(
                &atoms,
                &weights,
                MediationContrast::Mediated,
                60,
                0xABCD,
                &ctx,
            )
            .unwrap();
            assert_eq!(shared.block.block_length, block.block_length);
            assert_eq!(shared.block.replicates_ok, block.replicates_ok);
            for (a, b) in [
                (shared.block.total, block.total),
                (shared.block.direct, block.direct),
                (shared.block.mediated, block.mediated),
            ] {
                let (a, b) = (a.unwrap(), b.unwrap());
                assert!(a > 0.0 && (a - b).abs() < 1e-12 * b.max(1.0), "{a} vs {b}");
            }
            assert_eq!(shared.requested, shared.block.mediated);
            assert_eq!(single.effect.se_bootstrap, block.mediated);
        }
    }

    /// Atoms with different lag windows are refit on the same resampled times;
    /// zero replicates report the block but no SE.
    #[test]
    fn heterogeneous_lag_windows_share_one_time_window() {
        let (data, q, estimand) = series(240);
        let ctx = ExecutionContext::for_tests(3);
        let est = TemporalMediationEstimator::new();
        let plain = est.prepare_shared(&data, &estimand, &q, &[], &ctx).unwrap();
        let lagged = est
            .prepare_shared(
                &data,
                &estimand,
                &q,
                &[LaggedColumn {
                    variable: VariableId::from_raw(0),
                    lag: antecedent_core::Lag::from_raw(2),
                }],
                &ctx,
            )
            .unwrap();
        assert_eq!(plain.aligned_rows().first_time, 1);
        assert_eq!(lagged.aligned_rows().first_time, 2);
        let atoms = [&plain, &lagged];
        let none = shared_mediation_block_bootstrap(
            &atoms,
            &[0.6, 0.4],
            MediationContrast::Total,
            0,
            1,
            &ctx,
        )
        .unwrap();
        assert!(none.block.total.is_none() && none.requested.is_none());
        assert_eq!(none.block.rows, 238);
        assert_eq!(none.structural_span, 3);
        let shared = shared_mediation_block_bootstrap(
            &atoms,
            &[0.6, 0.4],
            MediationContrast::Total,
            40,
            1,
            &ctx,
        )
        .unwrap();
        assert!(shared.requested.is_some_and(|se| se > 0.0));
        assert_eq!(shared.requested, shared.block.total);
        assert!(shared.block.effective_rows.is_finite());
    }
}
