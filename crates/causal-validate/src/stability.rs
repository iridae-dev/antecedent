//! Discovery stability and parameter sensitivity (DESIGN.md §18.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, VariableId};
use causal_data::{ResamplingPlan, TableView, TimeSeriesData, resample_timeseries};
use causal_discovery::{DiscoveryWorkspace, LaggedLink, Pcmci, ci_from_name};

use crate::error::ValidationError;

/// Stability frequency for one lagged link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkStability {
    /// Link.
    pub link: LaggedLink,
    /// Fraction of bootstrap replicates / grid cells retaining the link.
    pub frequency: f64,
}

/// Report from discovery stability or parameter-sensitivity grids.
///
/// For [`BlockBootstrapStability`], `replicates` is the bootstrap count and
/// `block_size` is the moving-block length. For parameter sweeps
/// ([`AlphaThresholdSensitivity`], [`LagWindowSensitivity`], [`CiTestSensitivity`]),
/// `replicates` is the number of grid cells and `block_size` is `0`.
#[derive(Clone, Debug)]
pub struct DiscoveryStabilityReport {
    /// Per-link frequencies (links seen in ≥1 replicate / grid cell).
    pub frequencies: Arc<[LinkStability]>,
    /// Replicates run (bootstrap) or grid cell count (parameter sweeps).
    pub replicates: u32,
    /// Block size used (`0` for parameter sweeps).
    pub block_size: usize,
}

/// Block-bootstrap stability around a [`Pcmci`] configuration.
#[derive(Clone, Debug)]
pub struct BlockBootstrapStability {
    /// PCMCI configuration to re-run.
    pub pcmci: Pcmci,
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Block length.
    pub block_size: usize,
}

impl Default for BlockBootstrapStability {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockBootstrapStability {
    /// Defaults: 20 replicates, block size 20.
    #[must_use]
    pub fn new() -> Self {
        Self { pcmci: Pcmci::new().with_fdr(false), replicates: 20, block_size: 20 }
    }

    /// Run stability assessment.
    ///
    /// # Errors
    ///
    /// Data or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryStabilityReport, ValidationError> {
        if self.replicates == 0 || self.block_size == 0 {
            return Err(ValidationError::NotApplicable {
                message: "stability requires positive replicates and block_size",
            });
        }
        if self.block_size > data.row_count() {
            return Err(ValidationError::NotApplicable {
                message: "block_size exceeds series length",
            });
        }
        let mut counts: BTreeMap<LaggedLink, u32> = BTreeMap::new();
        let mut rng = ctx.rng.stream(0x57AB_u64);
        let mut index_scratch = Vec::new();
        for _ in 0..self.replicates {
            let boot = resample_timeseries(
                data,
                ResamplingPlan::MovingBlock { length: self.block_size },
                &mut rng,
                &mut index_scratch,
            )
            .map_err(ValidationError::from)?;
            let result =
                self.pcmci.run(&boot, variables, workspace, ctx).map_err(ValidationError::from)?;
            for s in result.evidence.links.iter() {
                *counts.entry(s.link).or_insert(0) += 1;
            }
        }
        Ok(report_from_counts(counts, self.replicates, self.block_size))
    }
}

/// Alpha-threshold sensitivity: re-run PCMCI across an `alpha` grid on the same data.
#[derive(Clone, Debug)]
pub struct AlphaThresholdSensitivity {
    /// Base PCMCI configuration (FDR, CI, max_lag, …).
    pub pcmci: Pcmci,
    /// Significance levels to sweep.
    pub alphas: Arc<[f64]>,
}

impl AlphaThresholdSensitivity {
    /// Build with a base config and alpha grid.
    #[must_use]
    pub fn new(pcmci: Pcmci, alphas: impl Into<Arc<[f64]>>) -> Self {
        Self { pcmci, alphas: alphas.into() }
    }

    /// Run the alpha grid.
    ///
    /// # Errors
    ///
    /// Empty/invalid grid, or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryStabilityReport, ValidationError> {
        if self.alphas.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "alpha sensitivity requires a non-empty alphas grid",
            });
        }
        if self.alphas.iter().any(|&a| !(a > 0.0 && a <= 1.0)) {
            return Err(ValidationError::NotApplicable {
                message: "alpha sensitivity requires alphas in (0, 1]",
            });
        }
        let configs = self.alphas.iter().map(|&alpha| {
            let mut constraints = self.pcmci.engine().constraints.clone();
            constraints.alpha = alpha;
            self.pcmci.clone().with_constraints(constraints)
        });
        run_param_grid(configs, data, variables, workspace, ctx)
    }
}

/// Lag-window sensitivity: re-run PCMCI across a `max_lag` grid on the same data.
#[derive(Clone, Debug)]
pub struct LagWindowSensitivity {
    /// Base PCMCI configuration.
    pub pcmci: Pcmci,
    /// Maximum lags to sweep.
    pub max_lags: Arc<[u32]>,
}

impl LagWindowSensitivity {
    /// Build with a base config and max-lag grid.
    #[must_use]
    pub fn new(pcmci: Pcmci, max_lags: impl Into<Arc<[u32]>>) -> Self {
        Self { pcmci, max_lags: max_lags.into() }
    }

    /// Run the lag-window grid.
    ///
    /// # Errors
    ///
    /// Empty/invalid grid, or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryStabilityReport, ValidationError> {
        if self.max_lags.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "lag-window sensitivity requires a non-empty max_lags grid",
            });
        }
        let min_lag = self.pcmci.engine().constraints.temporal.min_lag.raw();
        if self.max_lags.iter().any(|&m| m < min_lag) {
            return Err(ValidationError::NotApplicable {
                message: "lag-window sensitivity requires max_lag ≥ constraints.min_lag",
            });
        }
        let configs = self.max_lags.iter().map(|&max_lag| {
            let mut constraints = self.pcmci.engine().constraints.clone();
            constraints.temporal.max_lag = Lag::from_raw(max_lag);
            self.pcmci.clone().with_constraints(constraints)
        });
        run_param_grid(configs, data, variables, workspace, ctx)
    }
}

/// CI-test sensitivity: re-run PCMCI across named CI tests on the same data.
#[derive(Clone, Debug)]
pub struct CiTestSensitivity {
    /// Base PCMCI configuration (constraints / FDR fixed).
    pub pcmci: Pcmci,
    /// CI test names resolved via [`ci_from_name`].
    pub ci_names: Arc<[Arc<str>]>,
}

impl CiTestSensitivity {
    /// Build with a base config and CI name grid.
    #[must_use]
    pub fn new(pcmci: Pcmci, ci_names: impl Into<Arc<[Arc<str>]>>) -> Self {
        Self { pcmci, ci_names: ci_names.into() }
    }

    /// Run the CI-test grid.
    ///
    /// # Errors
    ///
    /// Empty grid, unknown CI name, or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryStabilityReport, ValidationError> {
        if self.ci_names.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "CI-test sensitivity requires a non-empty ci_names grid",
            });
        }
        let mut configs = Vec::with_capacity(self.ci_names.len());
        for name in self.ci_names.iter() {
            let ci = ci_from_name(name).map_err(|_e| ValidationError::NotApplicable {
                message: "CI-test sensitivity: unknown or unsupported CI name",
            })?;
            configs.push(self.pcmci.clone().with_ci(ci));
        }
        run_param_grid(configs.into_iter(), data, variables, workspace, ctx)
    }
}

fn run_param_grid(
    configs: impl IntoIterator<Item = Pcmci>,
    data: &TimeSeriesData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<DiscoveryStabilityReport, ValidationError> {
    let mut counts: BTreeMap<LaggedLink, u32> = BTreeMap::new();
    let mut cells = 0u32;
    for pcmci in configs {
        cells = cells.saturating_add(1);
        let result = pcmci.run(data, variables, workspace, ctx).map_err(ValidationError::from)?;
        for s in result.evidence.links.iter() {
            *counts.entry(s.link).or_insert(0) += 1;
        }
    }
    if cells == 0 {
        return Err(ValidationError::NotApplicable {
            message: "parameter sensitivity grid produced zero cells",
        });
    }
    Ok(report_from_counts(counts, cells, 0))
}

fn report_from_counts(
    counts: BTreeMap<LaggedLink, u32>,
    replicates: u32,
    block_size: usize,
) -> DiscoveryStabilityReport {
    let mut frequencies = Vec::with_capacity(counts.len());
    for (link, c) in counts {
        frequencies
            .push(LinkStability { link, frequency: f64::from(c) / f64::from(replicates) });
    }
    frequencies.sort_by(|a, b| {
        b.frequency.partial_cmp(&a.frequency).unwrap_or(std::cmp::Ordering::Equal)
    });
    DiscoveryStabilityReport {
        frequencies: Arc::from(frequencies),
        replicates,
        block_size,
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};

    use super::*;

    fn linked_series() -> (TimeSeriesData, Vec<VariableId>) {
        let n = 300usize;
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
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.02).sin();
            y[t] = 0.9 * x[t - 1];
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
    }

    fn base_pcmci() -> Pcmci {
        Pcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(2),
                min_lag: Lag::from_raw(1),
            },
            max_cond_size: 1,
            alpha: 0.05,
            ..DiscoveryConstraints::default()
        })
    }

    fn true_link_freq(report: &DiscoveryStabilityReport) -> f64 {
        report
            .frequencies
            .iter()
            .find(|f| {
                f.link.source == VariableId::from_raw(0)
                    && f.link.target == VariableId::from_raw(1)
                    && f.link.source_lag.raw() == 1
            })
            .map_or(0.0, |f| f.frequency)
    }

    #[test]
    fn true_link_is_stable() {
        let (data, vars) = linked_series();
        let mut stab = BlockBootstrapStability::new();
        stab.replicates = 8;
        stab.block_size = 25;
        stab.pcmci = base_pcmci();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(5);
        let report = stab.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(
            true_link_freq(&report) > 0.0,
            "expected true link to appear; report={:?}",
            report.frequencies
        );
    }

    #[test]
    fn alpha_threshold_retains_true_link() {
        let (data, vars) = linked_series();
        let sens = AlphaThresholdSensitivity::new(base_pcmci(), Arc::from([0.05f64, 0.1, 0.2]));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(5);
        let report = sens.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(report.replicates, 3);
        assert_eq!(report.block_size, 0);
        assert!(true_link_freq(&report) > 0.0);
    }

    #[test]
    fn lag_window_retains_true_link() {
        let (data, vars) = linked_series();
        let sens = LagWindowSensitivity::new(base_pcmci(), Arc::from([1u32, 2, 3]));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(5);
        let report = sens.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(report.replicates, 3);
        assert!(true_link_freq(&report) > 0.0);
    }

    #[test]
    fn ci_test_retains_true_link() {
        let (data, vars) = linked_series();
        let names: Arc<[Arc<str>]> =
            Arc::from([Arc::<str>::from("parcorr"), Arc::<str>::from("robust_parcorr")]);
        let sens = CiTestSensitivity::new(base_pcmci(), names);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(5);
        let report = sens.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(report.replicates, 2);
        assert!(true_link_freq(&report) > 0.0);
    }

    #[test]
    fn empty_grids_not_applicable() {
        let (data, vars) = linked_series();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(5);
        assert!(matches!(
            AlphaThresholdSensitivity::new(base_pcmci(), Arc::from([]) as Arc<[f64]>)
                .run(&data, &vars, &mut ws, &ctx),
            Err(ValidationError::NotApplicable { .. })
        ));
        assert!(matches!(
            LagWindowSensitivity::new(base_pcmci(), Arc::from([]) as Arc<[u32]>)
                .run(&data, &vars, &mut ws, &ctx),
            Err(ValidationError::NotApplicable { .. })
        ));
        assert!(matches!(
            CiTestSensitivity::new(base_pcmci(), Arc::from([]) as Arc<[Arc<str>]>)
                .run(&data, &vars, &mut ws, &ctx),
            Err(ValidationError::NotApplicable { .. })
        ));
    }
}
