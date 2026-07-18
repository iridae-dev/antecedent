//! Environment holdout validation via J-PCMCI+ (DESIGN.md §18.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeSet;
use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::{EnvHoldoutSplit, MultiEnvironmentData};
use causal_discovery::{DiscoveryWorkspace, JpcmciPlus, LaggedLink};

use crate::error::ValidationError;

/// Report comparing discovery vs holdout environment graphs.
#[derive(Clone, Debug)]
pub struct EnvironmentHoldoutReport {
    /// Links discovered on training environments.
    pub discovery_links: Arc<[LaggedLink]>,
    /// Links discovered on holdout environments.
    pub holdout_links: Arc<[LaggedLink]>,
    /// Fraction of discovery links also present on holdout.
    pub shared_frequency: f64,
    /// Jaccard index of the two link sets.
    pub jaccard: f64,
}

/// Environment-holdout discovery agreement under [`JpcmciPlus`].
#[derive(Clone, Debug)]
pub struct EnvironmentHoldout {
    /// J-PCMCI+ configuration.
    pub jpcmci: JpcmciPlus,
    /// Discovery vs estimation environment indexes.
    pub split: EnvHoldoutSplit,
}

impl EnvironmentHoldout {
    /// Build with a J-PCMCI+ config and holdout split.
    #[must_use]
    pub fn new(jpcmci: JpcmciPlus, split: EnvHoldoutSplit) -> Self {
        Self { jpcmci, split }
    }

    /// Discover independently on train and holdout env subsets; report link overlap.
    ///
    /// # Errors
    ///
    /// Split indexes out of range, empty subsets, or discovery failures.
    pub fn run(
        &self,
        data: &MultiEnvironmentData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<EnvironmentHoldoutReport, ValidationError> {
        let train = subset_envs(data, &self.split.discovery_envs)?;
        let holdout = subset_envs(data, &self.split.estimation_envs)?;
        let train_res = self
            .jpcmci
            .run(&train, variables, workspace, ctx)
            .map_err(ValidationError::from)?;
        let hold_res = self
            .jpcmci
            .run(&holdout, variables, workspace, ctx)
            .map_err(ValidationError::from)?;
        let train_set: BTreeSet<LaggedLink> =
            train_res.evidence.links.iter().map(|s| s.link).collect();
        let hold_set: BTreeSet<LaggedLink> =
            hold_res.evidence.links.iter().map(|s| s.link).collect();
        let shared = train_set.intersection(&hold_set).count();
        let union = train_set.union(&hold_set).count();
        let shared_frequency = if train_set.is_empty() {
            1.0
        } else {
            shared as f64 / train_set.len() as f64
        };
        let jaccard = if union == 0 { 1.0 } else { shared as f64 / union as f64 };
        Ok(EnvironmentHoldoutReport {
            discovery_links: Arc::from(train_set.into_iter().collect::<Vec<_>>()),
            holdout_links: Arc::from(hold_set.into_iter().collect::<Vec<_>>()),
            shared_frequency,
            jaccard,
        })
    }
}

fn subset_envs(
    data: &MultiEnvironmentData,
    idxs: &[usize],
) -> Result<MultiEnvironmentData, ValidationError> {
    if idxs.is_empty() {
        return Err(ValidationError::NotApplicable {
            message: "environment holdout subset is empty",
        });
    }
    let mut envs = Vec::with_capacity(idxs.len());
    for &i in idxs {
        let env = data.environment(i).map_err(ValidationError::from)?;
        envs.push(env.clone());
    }
    MultiEnvironmentData::try_new(envs).map_err(ValidationError::from)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
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
    use std::sync::Arc;

    use super::*;

    fn shared_lag_env(n: usize, seed: f64) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
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
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.4 * x[t - 1] + ((t as f64) * 0.02 + seed).sin() * 0.1;
            y[t] = 0.75 * x[t - 1] + 0.2 * y[t - 1];
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
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn env_holdout_runs_two_envs() {
        let multi = MultiEnvironmentData::try_new([
            shared_lag_env(180, 0.0),
            shared_lag_env(180, 1.0),
        ])
        .unwrap();
        let split = EnvHoldoutSplit::try_prefix(2, 1).unwrap();
        let mut constraints = DiscoveryConstraints::default();
        constraints.temporal = TemporalConstraints {
            max_lag: Lag::from_raw(1),
            min_lag: Lag::from_raw(1),
        };
        constraints.max_cond_size = 1;
        constraints.alpha = 0.15;
        let hold = EnvironmentHoldout::new(
            JpcmciPlus::new().with_fdr(false).with_constraints(constraints),
            split,
        );
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let report = hold.run(&multi, &vars, &mut ws, &ctx).unwrap();
        assert!(report.jaccard >= 0.0 && report.jaccard <= 1.0);
        assert!(report.shared_frequency >= 0.0 && report.shared_frequency <= 1.0);
    }
}
