//! Environment holdout validation via J-PCMCI+.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{EnvHoldoutSplit, MultiEnvironmentData};
use antecedent_discovery::{DiscoveryWorkspace, JpcmciPlus, LaggedLink};

use crate::error::ValidationError;

/// Report comparing discovery vs holdout environment graphs.
#[derive(Clone, Debug)]
pub struct EnvironmentHoldoutReport {
    /// Links discovered on training environments.
    pub discovery_links: Arc<[LaggedLink]>,
    /// Links discovered on holdout environments.
    pub holdout_links: Arc<[LaggedLink]>,
    /// Fraction of discovery links also present on holdout; `NaN` when no link was discovered
    /// (there is nothing to agree with, which is not perfect agreement).
    pub shared_frequency: f64,
    /// Jaccard index of the two link sets; `NaN` when neither environment set has a link.
    pub jaccard: f64,
}

/// `(shared_frequency, jaccard)` of two link sets; `NaN` where the ratio has an empty
/// denominator, so "nothing discovered" is never reported as full agreement.
#[allow(clippy::cast_precision_loss)]
fn agreement(train: &BTreeSet<LaggedLink>, holdout: &BTreeSet<LaggedLink>) -> (f64, f64) {
    let shared = train.intersection(holdout).count();
    let union = train.union(holdout).count();
    let shared_frequency =
        if train.is_empty() { f64::NAN } else { shared as f64 / train.len() as f64 };
    let jaccard = if union == 0 { f64::NAN } else { shared as f64 / union as f64 };
    (shared_frequency, jaccard)
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
        let train_res =
            self.jpcmci.run(&train, variables, workspace, ctx).map_err(ValidationError::from)?;
        let hold_res =
            self.jpcmci.run(&holdout, variables, workspace, ctx).map_err(ValidationError::from)?;
        let train_set: BTreeSet<LaggedLink> =
            train_res.evidence.links.iter().map(|s| s.link).collect();
        let hold_set: BTreeSet<LaggedLink> =
            hold_res.evidence.links.iter().map(|s| s.link).collect();
        let (shared_frequency, jaccard) = agreement(&train_set, &hold_set);
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
    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};
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
        let multi =
            MultiEnvironmentData::try_new([shared_lag_env(180, 0.0), shared_lag_env(180, 1.0)])
                .unwrap();
        let split = EnvHoldoutSplit::try_prefix(2, 1).unwrap();
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
            max_cond_size: 1,
            alpha: 0.15,
            ..Default::default()
        };
        let hold = EnvironmentHoldout::new(
            JpcmciPlus::new().with_fdr(false).with_constraints(constraints),
            split,
        );
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let report = hold.run(&multi, &vars, &mut ws, &ctx).unwrap();
        let train: BTreeSet<_> = report.discovery_links.iter().copied().collect();
        let holdout: BTreeSet<_> = report.holdout_links.iter().copied().collect();
        let (shared_frequency, jaccard) = agreement(&train, &holdout);
        assert_eq!(report.shared_frequency.to_bits(), shared_frequency.to_bits());
        assert_eq!(report.jaccard.to_bits(), jaccard.to_bits());
        if !train.is_empty() {
            assert!((0.0..=1.0).contains(&report.shared_frequency));
        }
    }

    #[test]
    fn nothing_discovered_is_undefined_agreement_not_perfect_agreement() {
        let link = |source, target| LaggedLink {
            source: VariableId::from_raw(source),
            source_lag: Lag::from_raw(1),
            target: VariableId::from_raw(target),
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let empty = BTreeSet::new();
        let (shared, jaccard) = agreement(&empty, &empty);
        assert!(shared.is_nan() && jaccard.is_nan());
        // Empty discovery set against a non-empty holdout: no agreement to report, but the Jaccard
        // index is the honest 0 (none of the one-sided union is shared).
        let one: BTreeSet<_> = [link(0, 1)].into_iter().collect();
        let (shared, jaccard) = agreement(&empty, &one);
        assert!(shared.is_nan());
        assert_eq!(jaccard, 0.0);
        // {a, b} against {b, c}: 1 shared of 2 discovered, Jaccard 1/3.
        let ab: BTreeSet<_> = [link(0, 1), link(1, 0)].into_iter().collect();
        let bc: BTreeSet<_> = [link(1, 0), link(0, 0)].into_iter().collect();
        let (shared, jaccard) = agreement(&ab, &bc);
        assert_eq!(shared, 0.5);
        assert!((jaccard - 1.0 / 3.0).abs() < 1e-15);
    }
}
