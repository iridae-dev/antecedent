//! Selection inference consumes complete-procedure realizations, not marginal
//! intervals or repeated random seeds standing in for independent observations.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{ProgramSearchError, cancelled, invalid};
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
pub mod procedure;

/// One complete aligned estimate/standard-error row for a fixed candidate set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEstimates {
    /// Stable candidate identities, in original candidate order.
    pub ids: Vec<String>,
    /// Utility estimates on the same target/population/time scale.
    pub values: Vec<f64>,
    /// Standard errors with the same sampling-unit/dependence contract.
    pub standard_errors: Vec<f64>,
}
impl CandidateEstimates {
    fn validate(&self) -> Result<(), ProgramSearchError> {
        if self.ids.is_empty()
            || self.ids.len() != self.values.len()
            || self.ids.len() != self.standard_errors.len()
            || self.values.iter().any(|v| !v.is_finite())
            || self.standard_errors.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid("candidate IDs, finite estimates and standard errors must align"));
        }
        let mut seen = std::collections::BTreeSet::new();
        if self.ids.iter().any(|id| id.trim().is_empty() || !seen.insert(id)) {
            return Err(invalid("candidate IDs must be nonempty and unique"));
        }
        Ok(())
    }
}

/// Simultaneous max-t bootstrap intervals for a prespecified finite candidate set.
/// Their validity requires the declared bootstrap to approximate the joint
/// sampling law; this calculation does not itself establish that premise.
#[derive(Clone, Debug)]
pub struct SimultaneousProgramIntervals {
    /// Candidate IDs in the original order.
    pub ids: Vec<String>,
    /// Joint two-sided intervals. Selection among this fixed set preserves the
    /// same simultaneous event, rather than reusing marginal confidence levels.
    pub intervals: Vec<[f64; 2]>,
    /// Requested simultaneous coverage level.
    pub level: f64,
    /// Empirical joint studentized critical value.
    pub critical_value: f64,
    /// Maximum absolute studentized root of every complete realization.
    pub roots: Vec<f64>,
    /// Selected program (ties resolved by lexical candidate ID).
    pub selected_id: String,
    /// Index of that candidate in the complete interval family.
    pub selected_index: usize,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl SimultaneousProgramIntervals {
    /// Retained reservation for roots, candidate IDs and interval outputs.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

/// Construct simultaneous intervals from complete dependence-preserving bootstrap
/// realizations, each repeating all estimation stages for a *fixed* candidate set.
/// Candidate generation using these same data requires a held-out evaluation or
/// a separate valid selective-inference argument; this API refuses changing sets.
///
/// # Errors
/// Missing/failed slots, changed candidate identities, invalid standard errors,
/// too few replicates for the requested tail probability and arithmetic overflow.
pub fn simultaneous_program_intervals(
    observed: &CandidateEstimates,
    bootstrap: &[Option<CandidateEstimates>],
    level: f64,
    ctx: &ExecutionContext,
) -> Result<SimultaneousProgramIntervals, ProgramSearchError> {
    cancelled(ctx)?;
    observed.validate()?;
    if !level.is_finite() || level <= 0.0 || level >= 1.0 || bootstrap.is_empty() {
        return Err(invalid("simultaneous intervals require a level in (0,1) and bootstrap rows"));
    }
    let quote = bootstrap
        .len()
        .checked_mul(16)
        .and_then(|n| n.checked_add(observed.ids.len().checked_mul(64)?))
        .and_then(|n| n.checked_add(observed.ids.iter().map(String::len).sum::<usize>()))
        .ok_or_else(|| invalid("simultaneous interval storage overflow"))?;
    if ctx.memory_limit_bytes().is_some_and(|limit| quote as u64 > limit) {
        return Err(ProgramSearchError::Resource(
            "simultaneous intervals exceed memory limit".into(),
        ));
    }
    let retention = Arc::new(
        ctx.acquire_admission(0, quote as u64)
            .map_err(|error| ProgramSearchError::Resource(error.to_string()))?,
    );
    let mut roots = Vec::with_capacity(bootstrap.len());
    for row in bootstrap {
        cancelled(ctx)?;
        let row = row.as_ref().ok_or_else(|| invalid("failed bootstrap slot cannot be dropped"))?;
        row.validate()?;
        if row.ids != observed.ids {
            return Err(invalid("bootstrap changed the fixed candidate set/order"));
        }
        let mut root = 0.0_f64;
        for ((value, center), se) in
            row.values.iter().zip(&observed.values).zip(&row.standard_errors)
        {
            let difference = (value - center).abs();
            let studentized = if *se == 0.0 {
                if difference != 0.0 {
                    return Err(invalid("nonconstant root with zero bootstrap standard error"));
                }
                0.0
            } else {
                difference / se
            };
            if !studentized.is_finite() {
                return Err(invalid("studentized bootstrap overflow"));
            }
            root = root.max(studentized);
        }
        roots.push(root);
    }
    let mut sorted = roots.clone();
    sorted.sort_by(f64::total_cmp);
    let quantile_rank = level * (bootstrap.len() as f64 + 1.0);
    let rank = (1..=sorted.len())
        .find(|rank| *rank as f64 >= quantile_rank)
        .ok_or_else(|| invalid("insufficient complete bootstrap replicates for requested tail"))?;
    let critical_value = sorted[rank - 1];
    let intervals = observed
        .values
        .iter()
        .zip(&observed.standard_errors)
        .map(|(value, se)| [value - critical_value * se, value + critical_value * se])
        .collect::<Vec<_>>();
    if intervals.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("simultaneous interval overflow"));
    }
    let selected_index = (0..observed.ids.len())
        .max_by(|&i, &j| {
            observed.values[i]
                .total_cmp(&observed.values[j])
                .then_with(|| observed.ids[j].cmp(&observed.ids[i]))
        })
        .ok_or_else(|| invalid("missing selected candidate"))?;
    Ok(SimultaneousProgramIntervals {
        ids: observed.ids.clone(),
        intervals,
        level,
        critical_value,
        roots,
        selected_id: observed.ids[selected_index].clone(),
        selected_index,
        retention,
    })
}

/// Original observational sampling unit and inclusive time span used by a stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationalSpan {
    /// Stable unit/cluster identity, independent of RNG stream or simulation seed.
    pub unit: String,
    /// First observed period used, including lookback and preprocessing.
    pub first: i64,
    /// Last observed period used by any stage.
    pub last: i64,
}

/// Retained data partition establishing observational separation of selection
/// (including all nuisance fitting and candidate generation) and validation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramHoldoutPartition {
    /// Exact spans read by every selection stage.
    pub training: Vec<ObservationalSpan>,
    /// Exact spans reserved for evaluation of the selected program.
    pub validation: Vec<ObservationalSpan>,
    /// Dependence/feature-lookback gap required between spans of the same unit.
    /// A finite gap alone does not prove independence for an arbitrary time series.
    pub required_gap: u64,
}
impl ProgramHoldoutPartition {
    /// Check observational overlap and required time separation. A distinct random
    /// seed is absent from this contract because it cannot separate observed data.
    ///
    /// # Errors
    /// Empty stages, malformed/duplicate spans, overlap or insufficient dependence gap.
    pub fn validate(&self) -> Result<(), ProgramSearchError> {
        for stage in [&self.training, &self.validation] {
            if stage.is_empty() {
                return Err(invalid("both observational holdout stages must be nonempty"));
            }
            let mut seen = std::collections::BTreeSet::new();
            for span in stage {
                if span.unit.trim().is_empty()
                    || span.first > span.last
                    || !seen.insert((&span.unit, span.first, span.last))
                {
                    return Err(invalid("invalid or duplicate observational span"));
                }
            }
        }
        for training in &self.training {
            for validation in &self.validation {
                if training.unit != validation.unit {
                    continue;
                }
                let gap = if training.last < validation.first {
                    i128::from(validation.first) - i128::from(training.last) - 1
                } else if validation.last < training.first {
                    i128::from(training.first) - i128::from(validation.last) - 1
                } else {
                    return Err(invalid("selection and validation share observational data"));
                };
                if gap < i128::from(self.required_gap) {
                    return Err(invalid("observational holdout gap is too short"));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_roots_preserve_dependence_and_selected_candidate_event() {
        let observed = CandidateEstimates {
            ids: vec!["a".into(), "b".into()],
            values: vec![2.0, 2.0],
            standard_errors: vec![1.0, 1.0],
        };
        let draws = (-10..=10)
            .map(|i| {
                Some(CandidateEstimates {
                    ids: observed.ids.clone(),
                    values: vec![2.0 + f64::from(i) / 10.0, 2.0 - f64::from(i) / 10.0],
                    standard_errors: vec![1.0, 1.0],
                })
            })
            .collect::<Vec<_>>();
        let ctx = ExecutionContext::for_tests(1);
        let result = simultaneous_program_intervals(&observed, &draws, 0.9, &ctx).unwrap();
        assert_eq!(result.selected_id, "a");
        assert_eq!(result.critical_value, 1.0);
        assert_eq!(result.intervals, vec![[1.0, 3.0]; 2]);
        let mut failed = draws.clone();
        failed[0] = None;
        assert!(simultaneous_program_intervals(&observed, &failed, 0.9, &ctx).is_err());
        let mut changed = draws;
        changed[0].as_mut().unwrap().ids.swap(0, 1);
        assert!(simultaneous_program_intervals(&observed, &changed, 0.9, &ctx).is_err());
    }
    #[test]
    fn observational_overlap_is_not_cured_by_another_rng_stream() {
        let span = |unit: &str, first, last| ObservationalSpan { unit: unit.into(), first, last };
        let mut partition = ProgramHoldoutPartition {
            training: vec![span("g", 0, 20)],
            validation: vec![span("g", 20, 40)],
            required_gap: 2,
        };
        assert!(partition.validate().is_err());
        partition.validation[0].first = 22;
        assert!(partition.validate().is_err());
        partition.validation[0].first = 23;
        assert!(partition.validate().is_ok());
        partition.validation[0] = span("independent_cluster", 0, 20);
        assert!(partition.validate().is_ok());
    }

    #[test]
    fn correlated_gaussian_selected_value_simulation_exercises_joint_intervals() {
        // A known Gaussian sampling law is an exact parametric bootstrap here.
        // This is an implementation simulation, not a release coverage record.
        let mut covered = 0;
        let mut simultaneous = 0;
        let ctx = ExecutionContext::for_tests(42);
        let truth = [0.0, 0.05, 0.1];
        for seed in 0..400 {
            let mut rng = antecedent_core::CausalRng::from_seed(seed + 490_000);
            let mut noise = || {
                let common = antecedent_kernels::standard_normal(&mut rng) * 0.8;
                std::array::from_fn::<_, 3, _>(|_| {
                    common + antecedent_kernels::standard_normal(&mut rng) * 0.6
                })
            };
            let observed = CandidateEstimates {
                ids: vec!["a".into(), "b".into(), "c".into()],
                values: noise().iter().zip(truth).map(|(z, t)| z + t).collect(),
                standard_errors: vec![1.0; 3],
            };
            let bootstrap = (0..199)
                .map(|_| {
                    Some(CandidateEstimates {
                        ids: observed.ids.clone(),
                        values: noise().iter().zip(&observed.values).map(|(z, m)| z + m).collect(),
                        standard_errors: vec![1.0; 3],
                    })
                })
                .collect::<Vec<_>>();
            let result = simultaneous_program_intervals(&observed, &bootstrap, 0.95, &ctx).unwrap();
            let contains =
                |j: usize| result.intervals[j][0] <= truth[j] && result.intervals[j][1] >= truth[j];
            covered += usize::from(contains(result.selected_index));
            simultaneous += usize::from((0..3).all(contains));
        }
        assert!(covered >= 360, "selected-value coverage smoke count: {covered}/400");
        assert!(simultaneous >= 350, "simultaneous coverage smoke count: {simultaneous}/400");
    }
}
