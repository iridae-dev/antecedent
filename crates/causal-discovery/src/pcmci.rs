//! Public lagged PCMCI algorithm.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::TimeSeriesData;
use causal_graph::TemporalGraphReview;
use causal_stats::FdrAdjustment;

use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::{graph_evidence_from_scored_with_sepsets, threshold_scored_links};
use crate::pcmci_family::pcmci_family_builders;
use crate::result::{AlgorithmRecord, DagDiscoveryResult};

/// Lagged PCMCI discovery algorithm.
#[derive(Clone, Debug)]
pub struct Pcmci {
    /// Shared PCMCI engine (crate-private; use builders / [`Self::engine`]).
    pub(crate) engine: PcmciEngine,
    /// Multiple-testing adjustment over the MCI family (`None` = off).
    pub fdr: Option<FdrAdjustment>,
}

impl Default for Pcmci {
    fn default() -> Self {
        Self::new()
    }
}

impl Pcmci {
    /// Default PCMCI (BH FDR on, alpha 0.05).
    #[must_use]
    pub fn new() -> Self {
        Self { engine: PcmciEngine::new(), fdr: Some(FdrAdjustment::bh()) }
    }

    pcmci_family_builders!();

    /// Run lagged PCMCI on `variables` in `data`.
    ///
    /// MCI scores the full constrained candidate family (all allowed
    /// `(X_{t−τ}, Y_t)` pairs); PC parent sets supply conditioning only. When
    /// FDR is set, that family is adjusted, then alpha retains links.
    ///
    /// # Errors
    ///
    /// Propagates engine / data failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DagDiscoveryResult, DiscoveryError> {
        let mut result = self.engine.run_pc_mci(data, variables, workspace, ctx)?;
        let alpha = self.engine.constraints.alpha;

        let scored = threshold_scored_links(
            result.evidence.links.iter().copied().collect(),
            self.fdr,
            alpha,
        );

        result.evidence = graph_evidence_from_scored_with_sepsets(scored, &result.sepsets)?;
        result.algorithm = AlgorithmRecord {
            id: Arc::from("pcmci"),
            config: Arc::from(format!(
                "alpha={},max_lag={},fdr={:?}",
                alpha,
                self.engine.constraints.temporal.max_lag.raw(),
                self.fdr
            )),
        };
        result.review = TemporalGraphReview::from_graph(
            result.evidence.graph.clone(),
            result.algorithm.id.clone(),
        );
        result.performance.links_retained = result.evidence.links.len() as u64;
        Ok(result)
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod calibration_tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };

    use crate::constraints::{DiscoveryConstraints, TemporalConstraints};

    fn next_gauss(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = ((*state >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = ((*state >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn independent_noise_series(n_vars: usize, n_obs: usize, seed: u64) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..n_vars {
            b.add_variable(
                format!("v{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut state = seed;
        let owned: Vec<OwnedColumn> = (0..n_vars)
            .map(|i| {
                let vals: Vec<f64> = (0..n_obs).map(|_| next_gauss(&mut state)).collect();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(vals),
                        ValidityBitmap::all_valid(n_obs),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1 },
                length: n_obs,
            },
        )
        .unwrap()
    }

    fn planted_lag1_series(n_obs: usize, seed: u64) -> (TimeSeriesData, Vec<VariableId>) {
        // Y_t = 0.7 X_{t-1} + ε; X_t = η (independent noise).
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut state = seed;
        let mut x = vec![0.0; n_obs];
        let mut y = vec![0.0; n_obs];
        for t in 0..n_obs {
            x[t] = next_gauss(&mut state);
            if t == 0 {
                y[t] = next_gauss(&mut state);
            } else {
                y[t] = 0.7 * x[t - 1] + 0.5 * next_gauss(&mut state);
            }
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n_obs),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n_obs),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1 },
                length: n_obs,
            },
        )
        .unwrap();
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
    }

    /// Independent noise series: PCMCI link retention rate should track α.
    ///
    /// Candidate family size = n_vars² · (max_lag−min_lag+1) − n_vars (skip self@0 when
    /// lag0 allowed; here min_lag=1 so family = n_vars² · n_lags). FDR off so alpha is
    /// the sole threshold. Band: ±4 MC SE of Bernoulli(α) plus floor/ceiling.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn pcmci_null_fpr_near_alpha() {
        const N_VARS: usize = 3;
        const N_OBS: usize = 300;
        const N_SIM: u32 = 40;
        const ALPHA: f64 = 0.05;
        const MAX_LAG: u32 = 1;
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(MAX_LAG),
                min_lag: Lag::from_raw(1),
            },
            alpha: ALPHA,
            max_cond_size: 1,
            ..DiscoveryConstraints::default()
        };
        let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints.clone());
        let vars: Vec<VariableId> =
            (0..N_VARS as u32).map(VariableId::from_raw).collect();
        let mut family = 0u32;
        for &t in &vars {
            family += constraints.candidate_sources(&vars, t).len() as u32;
        }
        let mut retained = 0u32;
        let mut total = 0u32;
        for s in 0..N_SIM {
            let data = independent_noise_series(N_VARS, N_OBS, 11_000 + u64::from(s));
            let mut ws = DiscoveryWorkspace::default();
            let ctx = ExecutionContext::for_tests(200 + u64::from(s));
            let result = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap();
            retained += result.evidence.links.len() as u32;
            total += family;
        }
        let rate = f64::from(retained) / f64::from(total);
        let se = (ALPHA * (1.0 - ALPHA) / f64::from(total)).sqrt();
        // PCMCI MCI can be slightly conservative under iid noise; allow floor 0.01.
        let lo = (ALPHA - 4.0 * se).max(0.01);
        let hi = (ALPHA + 4.0 * se).min(0.12);
        assert!(
            rate >= lo && rate <= hi,
            "PCMCI null link rate={rate:.3} outside [{lo:.3}, {hi:.3}] \
             ({retained}/{total}; α={ALPHA})"
        );
    }

    /// Light power check: planted X→Y@lag1 recovered often under moderate SNR.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn pcmci_planted_lag1_power() {
        const N_SIM: u32 = 30;
        const N_OBS: usize = 250;
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::from_raw(1),
            },
            alpha: 0.05,
            max_cond_size: 1,
            ..DiscoveryConstraints::default()
        };
        let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints);
        let mut hits = 0u32;
        for s in 0..N_SIM {
            let (data, vars) = planted_lag1_series(N_OBS, 12_000 + u64::from(s));
            let mut ws = DiscoveryWorkspace::default();
            let ctx = ExecutionContext::for_tests(300 + u64::from(s));
            let result = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap();
            let has = result.evidence.links.iter().any(|s| {
                s.link.source == VariableId::from_raw(0)
                    && s.link.target == VariableId::from_raw(1)
                    && s.link.source_lag.raw() == 1
            });
            if has {
                hits += 1;
            }
        }
        let power = f64::from(hits) / f64::from(N_SIM);
        // SNR ≈ 0.7 / 0.5; expect high recovery. Floor 0.70 allows MC noise at N=30.
        assert!(
            power >= 0.70,
            "PCMCI planted lag-1 power={power:.2} ({hits}/{N_SIM}); expected ≥0.70"
        );
    }
}
