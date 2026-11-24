//! Independent-unit bootstrap law of the arm means of a randomized design.
//!
//! Randomization alone identifies each arm mean and a conservative covariance bound for every
//! linear combination of them, but not the dependence of the arm-mean estimators across arms:
//! that term depends on the joint distribution of the potential outcomes under both arms, which
//! no assignment reveals. Under the declared premise that the independent units (clusters,
//! matched pairs or switchback sequences) were sampled independently from a larger population,
//! the sampling law of the arm means is identified, and a nonparametric bootstrap over those
//! units estimates it:
//!
//! * a cluster-randomized design resamples clusters within each arm, so both arm sizes stay
//!   as complete randomization fixed them and the two arms are drawn independently;
//! * a matched-pair experiment resamples whole pairs, keeping both members of a pair together;
//! * a switchback experiment resamples whole sequences, keeping every period of a sequence.
//!
//! Every refit reruns the complete estimator on the resampled source, including any fixed
//! pretreatment adjustment centred at the resampled pretreatment mean. A refit the estimator
//! refuses keeps its slot with its reason; nothing is dropped.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cluster_decision::{JointClusterRecipe, joint_cluster};
use super::compiled::JointDesignRecipe;
use super::experiment::{ExperimentalDesign, JointExperimentalRecipe, joint_experiment};
use antecedent_core::{CausalRng, ExecutionContext, RngFactory, StreamDomain};

/// Fewest complete refits the bootstrap accepts.
pub const MIN_REFITS: usize = crate::temporal_design::regime::MIN_REPLICATES;

/// Fewest independent units the bootstrap accepts: clusters in each arm, matched pairs, or
/// switchback sequences.
pub const MIN_INDEPENDENT_UNITS: usize = crate::randomized_joint::MIN_UNITS_PER_ARM_FOR_INTERVAL;

/// Stream salt of the resampling draws, so they never share a stream with another procedure.
const RESAMPLE_SALT: u64 = 0x1d6a_7b0f_0000_0000;

/// The independent unit a design's bootstrap resamples.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndependentUnit {
    /// Clusters of a cluster-randomized design, resampled within each arm.
    Cluster,
    /// Whole matched pairs.
    Pair,
    /// Whole switchback sequences.
    Sequence,
}

impl IndependentUnit {
    /// Stable name of the unit.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cluster => "cluster",
            Self::Pair => "pair",
            Self::Sequence => "sequence",
        }
    }
}

/// Aligned independent-unit bootstrap refits of a design's arm means.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointUnitBootstrap {
    /// The independent unit resampled.
    pub unit: IndependentUnit,
    /// Independent units resampled: for a cluster design the smaller arm's cluster count.
    pub independent_units: usize,
    /// Full-data arm means: every outcome under the treated arm, then under the control arm.
    pub point: Vec<f64>,
    /// One refit per slot in the same order as `point`; a failed refit stays absent in place.
    pub realizations: Vec<Option<Vec<f64>>>,
    /// The estimator's reason for each failed slot.
    pub failures: Vec<Option<String>>,
}

/// The independent unit of a design and how many the bootstrap would resample (for a cluster
/// design, the smaller arm's count); `None` for a design this bootstrap does not read.
#[must_use]
pub fn independent_units(recipe: &JointDesignRecipe) -> Option<(IndependentUnit, usize)> {
    match recipe {
        JointDesignRecipe::DecisionCluster { recipe } => {
            let treated = recipe.assignments.iter().filter(|a| **a).count();
            let control = recipe.assignments.len() - treated;
            Some((IndependentUnit::Cluster, treated.min(control)))
        }
        JointDesignRecipe::DecisionExperiment { recipe } => match &recipe.design {
            ExperimentalDesign::MatchedPairs { pairs, .. } => {
                Some((IndependentUnit::Pair, pairs.len()))
            }
            ExperimentalDesign::Switchback { sequence_ids, .. } => {
                Some((IndependentUnit::Sequence, sequence_groups(sequence_ids).len()))
            }
        },
        _ => None,
    }
}

/// Bootstrap the arm means of a cluster-randomized, matched-pair or switchback design over its
/// independent units under the declared sampling premise `sampling`, with `replicates`
/// complete refits drawn from `seed`.
///
/// The draws depend on `seed` alone, never on the execution context's generator.
///
/// # Errors
/// Another design; an empty sampling premise; fewer than [`MIN_REFITS`] refits; fewer than
/// [`MIN_INDEPENDENT_UNITS`] independent units (in either arm of a cluster design); a source
/// the full-data estimator refuses; allocation overflow; cancellation.
pub fn joint_unit_bootstrap(
    recipe: &JointDesignRecipe,
    replicates: usize,
    seed: u64,
    sampling: &str,
    ctx: &ExecutionContext,
) -> Result<JointUnitBootstrap, String> {
    if sampling.trim().is_empty() {
        return Err("the independent-unit bootstrap needs a declared sampling premise".into());
    }
    if replicates < MIN_REFITS {
        return Err(format!("the independent-unit bootstrap needs at least {MIN_REFITS} refits"));
    }
    let Some((unit, units)) = independent_units(recipe) else {
        return Err("the independent-unit bootstrap reads a cluster-randomized, matched-pair or \
                    switchback design"
            .into());
    };
    if units < MIN_INDEPENDENT_UNITS {
        return Err(format!(
            "the independent-unit bootstrap needs at least {MIN_INDEPENDENT_UNITS} {}s{}; the \
             design has {units}",
            unit.name(),
            if unit == IndependentUnit::Cluster { " in each arm" } else { "" },
        ));
    }
    let rng = RngFactory::from_seed(seed);
    let (point, realizations, failures) = match recipe {
        JointDesignRecipe::DecisionCluster { recipe } => {
            let law = joint_cluster(recipe, ctx)?;
            let point = law.treated_means.iter().chain(&law.control_means).copied().collect();
            let draw = Resampler::cluster(recipe);
            let (realizations, failures) = refit_all(replicates, &rng, ctx, |rng| {
                let law = joint_cluster(&draw.cluster_draw(recipe, rng), ctx)?;
                Ok(law.treated_means.iter().chain(&law.control_means).copied().collect())
            })?;
            (point, realizations, failures)
        }
        JointDesignRecipe::DecisionExperiment { recipe } => {
            let point = joint_experiment(recipe, ctx)?.means;
            let draw = Resampler::experiment(recipe);
            let (realizations, failures) = refit_all(replicates, &rng, ctx, |rng| {
                Ok(joint_experiment(&draw.experiment_draw(recipe, rng), ctx)?.means)
            })?;
            (point, realizations, failures)
        }
        _ => unreachable!("independent_units admits only the cluster and experiment designs"),
    };
    Ok(JointUnitBootstrap { unit, independent_units: units, point, realizations, failures })
}

/// Run `refit` once per slot on its own resampling stream, keeping every failure in place.
#[allow(clippy::type_complexity, reason = "the aligned slot and failure vectors")]
fn refit_all(
    replicates: usize,
    rng: &RngFactory,
    ctx: &ExecutionContext,
    mut refit: impl FnMut(&mut CausalRng) -> Result<Vec<f64>, String>,
) -> Result<(Vec<Option<Vec<f64>>>, Vec<Option<String>>), String> {
    let mut realizations = Vec::with_capacity(replicates);
    let mut failures = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err("independent-unit bootstrap cancelled".into());
        }
        let stream = u64::try_from(replicate).map_err(|_| "bootstrap stream overflow")?;
        let mut draws = rng.stream_for(StreamDomain::Resample, RESAMPLE_SALT ^ stream);
        match refit(&mut draws) {
            Ok(row) => {
                realizations.push(Some(row));
                failures.push(None);
            }
            Err(_) if ctx.cancellation.is_cancelled() => {
                return Err("independent-unit bootstrap cancelled".into());
            }
            Err(error) => {
                realizations.push(None);
                failures.push(Some(error));
            }
        }
    }
    Ok((realizations, failures))
}

/// The rows of each switchback sequence, in first-appearance order.
fn sequence_groups(sequence_ids: &[String]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut index = std::collections::BTreeMap::<&str, usize>::new();
    for (row, id) in sequence_ids.iter().enumerate() {
        let slot = *index.entry(id.as_str()).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[slot].push(row);
    }
    groups
}

/// The resampling strata of one design: the cluster indices of each arm, the pairs, or the
/// rows of each sequence.
enum Resampler {
    Cluster { treated: Vec<usize>, control: Vec<usize> },
    Pairs { pairs: Vec<(usize, usize)> },
    Sequences { groups: Vec<Vec<usize>> },
}

impl Resampler {
    fn cluster(recipe: &JointClusterRecipe) -> Self {
        let (treated, control): (Vec<usize>, Vec<usize>) =
            (0..recipe.assignments.len()).partition(|c| recipe.assignments[*c]);
        Self::Cluster { treated, control }
    }

    fn experiment(recipe: &JointExperimentalRecipe) -> Self {
        match &recipe.design {
            ExperimentalDesign::MatchedPairs { pairs, .. } => Self::Pairs { pairs: pairs.clone() },
            ExperimentalDesign::Switchback { sequence_ids, .. } => {
                Self::Sequences { groups: sequence_groups(sequence_ids) }
            }
        }
    }

    /// Draw every arm's clusters with replacement from that arm and rebuild the source; a
    /// drawn cluster's label gains its slot so labels stay distinct.
    fn cluster_draw(&self, recipe: &JointClusterRecipe, rng: &mut CausalRng) -> JointClusterRecipe {
        let Self::Cluster { treated, control } = self else {
            unreachable!("a cluster design builds a cluster resampler")
        };
        let mut picks = Vec::with_capacity(recipe.assignments.len());
        for arm in [treated, control] {
            for _ in 0..arm.len() {
                picks.push(arm[antecedent_kernels::unbiased_index(rng, arm.len())]);
            }
        }
        let take = |values: &[f64]| picks.iter().map(|c| values[*c]).collect::<Vec<_>>();
        let mut resampled = recipe.clone();
        resampled.cluster_ids = picks
            .iter()
            .enumerate()
            .map(|(slot, c)| format!("{}#{slot}", recipe.cluster_ids[*c]))
            .collect();
        resampled.cluster_sizes = picks.iter().map(|c| recipe.cluster_sizes[*c]).collect();
        resampled.assignments = picks.iter().map(|c| recipe.assignments[*c]).collect();
        for (outcome, original) in resampled.outcomes.iter_mut().zip(&recipe.outcomes) {
            outcome.values = take(&original.values);
        }
        for (adjustment, original) in resampled.adjustments.iter_mut().zip(&recipe.adjustments) {
            adjustment.pretreatment_totals = take(&original.pretreatment_totals);
        }
        resampled
    }

    /// Draw whole pairs or whole sequences with replacement and rebuild the source; a drawn
    /// unit's labels gain its slot so labels stay distinct.
    fn experiment_draw(
        &self,
        recipe: &JointExperimentalRecipe,
        rng: &mut CausalRng,
    ) -> JointExperimentalRecipe {
        let mut resampled = recipe.clone();
        let mut rows: Vec<usize> = Vec::with_capacity(recipe.assignments.len());
        match (self, &mut resampled.design, &recipe.design) {
            (
                Self::Pairs { pairs },
                ExperimentalDesign::MatchedPairs {
                    cluster_ids,
                    pairs: drawn_pairs,
                    treatment_probabilities,
                    cluster_sizes,
                    ..
                },
                ExperimentalDesign::MatchedPairs {
                    cluster_ids: ids,
                    treatment_probabilities: probabilities,
                    cluster_sizes: sizes,
                    ..
                },
            ) => {
                drawn_pairs.clear();
                cluster_ids.clear();
                for slot in 0..pairs.len() {
                    let (first, second) =
                        pairs[antecedent_kernels::unbiased_index(rng, pairs.len())];
                    drawn_pairs.push((rows.len(), rows.len() + 1));
                    for member in [first, second] {
                        rows.push(member);
                        cluster_ids.push(format!("{}#{slot}", ids[member]));
                    }
                }
                *treatment_probabilities = rows.iter().map(|r| probabilities[*r]).collect();
                *cluster_sizes = rows.iter().map(|r| sizes[*r]).collect();
            }
            (
                Self::Sequences { groups },
                ExperimentalDesign::Switchback {
                    sequence_ids,
                    period_ids,
                    treatment_probabilities,
                    ..
                },
                ExperimentalDesign::Switchback {
                    sequence_ids: ids,
                    period_ids: periods,
                    treatment_probabilities: probabilities,
                    ..
                },
            ) => {
                sequence_ids.clear();
                for slot in 0..groups.len() {
                    let group = &groups[antecedent_kernels::unbiased_index(rng, groups.len())];
                    for row in group {
                        rows.push(*row);
                        sequence_ids.push(format!("{}#{slot}", ids[*row]));
                    }
                }
                *period_ids = rows.iter().map(|r| periods[*r].clone()).collect();
                *treatment_probabilities = rows.iter().map(|r| probabilities[*r]).collect();
            }
            _ => unreachable!("an experiment builds the resampler of its own design"),
        }
        resampled.assignments = rows.iter().map(|r| recipe.assignments[*r]).collect();
        for (outcome, original) in resampled.outcomes.iter_mut().zip(&recipe.outcomes) {
            outcome.values = rows.iter().map(|r| original.values[*r]).collect();
        }
        resampled
    }
}

#[cfg(test)]
mod tests {
    use super::super::excursion::ExcursionOutcome;
    use super::super::experiment::ExperimentExposure;
    use super::*;

    fn uniform(mut state: u64) -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        state ^= state >> 31;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn outcome(name: &str, values: Vec<f64>) -> ExcursionOutcome {
        ExcursionOutcome {
            component_name: name.into(),
            variable_id: format!("{name}_id"),
            units: "points".into(),
            values,
        }
    }

    fn no_interference() -> ExperimentExposure {
        ExperimentExposure::NoInterference { assumption: "no interference".into() }
    }

    /// `per_arm` clusters in each arm, two rows each, two correlated outcomes.
    fn cluster_design(per_arm: usize) -> JointDesignRecipe {
        let n = 2 * per_arm;
        let assignments: Vec<bool> = (0..n).map(|c| c % 2 == 0).collect();
        let base: Vec<f64> = (0..n).map(|c| uniform(c as u64) * 4.0).collect();
        let first: Vec<f64> =
            (0..n).map(|c| base[c] + if assignments[c] { 1.0 } else { 0.0 }).collect();
        let second: Vec<f64> =
            (0..n).map(|c| 0.5 * base[c] + uniform(1000 + c as u64) * 2.0).collect();
        JointDesignRecipe::DecisionCluster {
            recipe: JointClusterRecipe {
                population: "sites".into(),
                treated_regime_id: "treated".into(),
                control_regime_id: "control".into(),
                horizon: 1,
                cluster_ids: (0..n).map(|c| format!("site{c}")).collect(),
                cluster_sizes: vec![2; n],
                assignment_law: "complete randomization".into(),
                treated_clusters: per_arm,
                consistency: "one version".into(),
                assignments,
                outcomes: vec![outcome("benefit", first), outcome("cost", second)],
                adjustments: Vec::new(),
                exposure: no_interference(),
            },
        }
    }

    fn pair_design(pairs: usize) -> JointDesignRecipe {
        let n = 2 * pairs;
        let assignments: Vec<bool> = (0..n).map(|c| (c % 2 == 0) == (c % 4 < 2)).collect();
        let level: Vec<f64> = (0..pairs).map(|p| uniform(5000 + p as u64) * 3.0).collect();
        let first: Vec<f64> = (0..n)
            .map(|c| level[c / 2] + uniform(c as u64) + if assignments[c] { 0.5 } else { 0.0 })
            .collect();
        let second: Vec<f64> =
            (0..n).map(|c| 2.0 * level[c / 2] - uniform(7000 + c as u64)).collect();
        JointDesignRecipe::DecisionExperiment {
            recipe: JointExperimentalRecipe {
                population: "sites".into(),
                treated_regime_id: "treated".into(),
                control_regime_id: "control".into(),
                horizon: 1,
                consistency: "one version".into(),
                exposure: no_interference(),
                design: ExperimentalDesign::MatchedPairs {
                    cluster_ids: (0..n).map(|c| format!("site{c}")).collect(),
                    pairs: (0..pairs).map(|p| (2 * p, 2 * p + 1)).collect(),
                    matching_variables: vec!["size".into()],
                    treatment_probabilities: vec![0.5; n],
                    cluster_sizes: vec![1; n],
                    preassignment: "pairs formed before assignment".into(),
                    independent_pairs: "pairs assigned independently".into(),
                },
                assignments,
                outcomes: vec![outcome("benefit", first), outcome("cost", second)],
            },
        }
    }

    /// `sequences` sequences of `periods` periods; `assigned` gives the arm of each row.
    fn switchback_design(
        sequences: usize,
        periods: usize,
        assigned: impl Fn(usize, usize) -> bool,
    ) -> JointDesignRecipe {
        let mut sequence_ids = Vec::new();
        let mut period_ids = Vec::new();
        let mut assignments = Vec::new();
        let mut values = Vec::new();
        for s in 0..sequences {
            let level = uniform(9000 + s as u64) * 2.0;
            for t in 0..periods {
                let treated = assigned(s, t);
                sequence_ids.push(format!("seq{s}"));
                period_ids.push(format!("p{t}"));
                assignments.push(treated);
                let noise = uniform((s * periods + t) as u64);
                values.push(level + noise + if treated { 1.0 } else { 0.0 });
            }
        }
        let n = assignments.len();
        JointDesignRecipe::DecisionExperiment {
            recipe: JointExperimentalRecipe {
                population: "stores".into(),
                treated_regime_id: "treated".into(),
                control_regime_id: "control".into(),
                horizon: 1,
                consistency: "one version".into(),
                exposure: no_interference(),
                design: ExperimentalDesign::Switchback {
                    sequence_ids,
                    period_ids,
                    treatment_probabilities: vec![0.5; n],
                    no_carryover: "no carryover".into(),
                    washout_rule: "zero washout".into(),
                    independent_sequences: "sequences assigned independently".into(),
                },
                assignments,
                outcomes: vec![outcome("sales", values)],
            },
        }
    }

    /// Sample covariance (divisor R - 1) of complete refits.
    fn refit_covariance(bootstrap: &JointUnitBootstrap) -> (usize, Vec<f64>) {
        let rows: Vec<&Vec<f64>> = bootstrap.realizations.iter().flatten().collect();
        let width = bootstrap.point.len();
        let r = rows.len() as f64;
        let mut mean = vec![0.0; width];
        for row in &rows {
            for (m, v) in mean.iter_mut().zip(*row) {
                *m += v / r;
            }
        }
        let mut cov = vec![0.0; width * width];
        for row in &rows {
            for i in 0..width {
                for j in 0..width {
                    cov[i * width + j] += (row[i] - mean[i]) * (row[j] - mean[j]) / (r - 1.0);
                }
            }
        }
        (width, cov)
    }

    /// Every entry within `tolerance` of the analytic one, on the scale of the analytic
    /// standard deviations.
    fn assert_close(width: usize, boot: &[f64], analytic: &[f64], tolerance: f64) {
        for i in 0..width {
            for j in 0..width {
                let scale = (analytic[i * width + i] * analytic[j * width + j]).sqrt();
                let gap = (boot[i * width + j] - analytic[i * width + j]).abs();
                assert!(
                    gap <= tolerance * scale,
                    "entry ({i},{j}): bootstrap {} analytic {} scale {scale}",
                    boot[i * width + j],
                    analytic[i * width + j]
                );
            }
        }
    }

    const REFITS: usize = 4000;

    #[test]
    fn cluster_refits_are_independent_across_arms_and_match_the_within_arm_covariance() {
        let ctx = ExecutionContext::for_tests(1);
        let design = cluster_design(40);
        let bootstrap = joint_unit_bootstrap(&design, REFITS, 11, "sites sampled", &ctx).unwrap();
        assert_eq!(bootstrap.unit, IndependentUnit::Cluster);
        assert_eq!(bootstrap.independent_units, 40);
        assert!(bootstrap.failures.iter().all(Option::is_none));
        let JointDesignRecipe::DecisionCluster { recipe } = &design else { unreachable!() };
        let law = joint_cluster(recipe, &ctx).unwrap();
        let (width, boot) = refit_covariance(&bootstrap);
        assert_eq!(width, 4);
        // The bootstrap of a mean over n units estimates (n - 1)/n of the unbiased Cov/n.
        let shrink = 39.0 / 40.0;
        let mut analytic = vec![0.0; 16];
        for a in 0..2 {
            for b in 0..2 {
                analytic[a * 4 + b] = law.treated_covariance[a * 2 + b] * shrink;
                analytic[(a + 2) * 4 + b + 2] = law.control_covariance[a * 2 + b] * shrink;
            }
        }
        // Monte Carlo error of a covariance entry over 4000 refits is about 0.02 of the scale.
        assert_close(width, &boot, &analytic, 0.08);
        for a in 0..2 {
            for b in 2..4 {
                let scale = (boot[a * 4 + a] * boot[b * 4 + b]).sqrt();
                assert!(boot[a * 4 + b].abs() < 0.08 * scale, "cross-arm entry ({a},{b})");
            }
        }
    }

    #[test]
    fn pair_refits_converge_to_the_pair_level_arm_covariance() {
        let ctx = ExecutionContext::for_tests(1);
        let design = pair_design(40);
        let bootstrap = joint_unit_bootstrap(&design, REFITS, 12, "pairs sampled", &ctx).unwrap();
        assert_eq!(bootstrap.unit, IndependentUnit::Pair);
        assert!(bootstrap.failures.iter().all(Option::is_none));
        let JointDesignRecipe::DecisionExperiment { recipe } = &design else { unreachable!() };
        let law = joint_experiment(recipe, &ctx).unwrap();
        assert_eq!(bootstrap.point, law.means);
        let (width, boot) = refit_covariance(&bootstrap);
        let analytic: Vec<f64> = law.covariance.iter().map(|v| v * 39.0 / 40.0).collect();
        // The pairs carry a strong cross-arm dependence the bootstrap must reproduce.
        assert!(analytic[2].abs() > 0.5 * (analytic[0] * analytic[2 * 4 + 2]).sqrt());
        assert_close(width, &boot, &analytic, 0.08);
    }

    #[test]
    fn sequence_refits_converge_to_the_polarised_switchback_covariance() {
        let ctx = ExecutionContext::for_tests(1);
        let design = switchback_design(40, 4, |s, t| uniform(20_000 + (s * 4 + t) as u64) < 0.5);
        let bootstrap =
            joint_unit_bootstrap(&design, REFITS, 13, "sequences sampled", &ctx).unwrap();
        assert_eq!(bootstrap.unit, IndependentUnit::Sequence);
        assert_eq!(bootstrap.independent_units, 40);
        assert!(bootstrap.failures.iter().all(Option::is_none));
        let JointDesignRecipe::DecisionExperiment { recipe } = &design else { unreachable!() };
        let law = joint_experiment(recipe, &ctx).unwrap();
        let (width, boot) = refit_covariance(&bootstrap);
        let analytic: Vec<f64> = law.covariance.iter().map(|v| v * 39.0 / 40.0).collect();
        assert_close(width, &boot, &analytic, 0.08);
    }

    #[test]
    fn a_seed_reproduces_every_refit_and_another_seed_draws_others() {
        let ctx = ExecutionContext::for_tests(1);
        let design = pair_design(30);
        let first = joint_unit_bootstrap(&design, 99, 5, "pairs sampled", &ctx).unwrap();
        let again = joint_unit_bootstrap(&design, 99, 5, "pairs sampled", &ctx).unwrap();
        let other = joint_unit_bootstrap(&design, 99, 6, "pairs sampled", &ctx).unwrap();
        let bits = |b: &JointUnitBootstrap| {
            b.realizations
                .iter()
                .flatten()
                .flat_map(|row| row.iter().map(|v| v.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&first), bits(&again));
        assert_ne!(bits(&first), bits(&other));
        // The execution context's own generator does not enter the draws.
        let reseeded = ExecutionContext::for_tests(77);
        let elsewhere = joint_unit_bootstrap(&design, 99, 5, "pairs sampled", &reseeded).unwrap();
        assert_eq!(bits(&first), bits(&elsewhere));
    }

    #[test]
    fn too_few_refits_units_or_no_premise_are_refused() {
        let ctx = ExecutionContext::for_tests(1);
        let refused = |design: &JointDesignRecipe, replicates: usize, sampling: &str| {
            joint_unit_bootstrap(design, replicates, 1, sampling, &ctx).unwrap_err()
        };
        assert!(refused(&pair_design(30), 98, "pairs sampled").contains("99 refits"));
        assert!(refused(&pair_design(30), 99, " ").contains("sampling premise"));
        assert!(refused(&cluster_design(29), 99, "sites sampled").contains("in each arm"));
        assert!(refused(&pair_design(29), 99, "pairs sampled").contains("30 pairs"));
        let short = switchback_design(29, 4, |_, t| t % 2 == 0);
        assert!(refused(&short, 99, "sequences sampled").contains("30 sequences"));
        assert_eq!(independent_units(&cluster_design(29)), Some((IndependentUnit::Cluster, 29)));
        assert_eq!(independent_units(&short), Some((IndependentUnit::Sequence, 29)));
    }

    #[test]
    fn a_refit_the_estimator_refuses_keeps_its_slot_and_reason() {
        let ctx = ExecutionContext::for_tests(1);
        // Fifteen always-treated and fifteen never-treated two-period sequences give exactly
        // thirty rows per arm, the switchback interval floor; a resample that draws fewer of
        // either kind leaves that arm below the floor and the estimator refuses the refit.
        let design = switchback_design(30, 2, |s, _| s < 15);
        let bootstrap = joint_unit_bootstrap(&design, 99, 3, "sequences sampled", &ctx).unwrap();
        assert_eq!(bootstrap.realizations.len(), 99);
        assert_eq!(bootstrap.failures.len(), 99);
        let failed = bootstrap.failures.iter().filter(|f| f.is_some()).count();
        assert!(failed > 0 && failed < 99, "{failed} failed refits");
        for (row, failure) in bootstrap.realizations.iter().zip(&bootstrap.failures) {
            assert_eq!(row.is_none(), failure.is_some());
        }
    }

    #[test]
    fn a_cancelled_bootstrap_is_refused_rather_than_truncated() {
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let error = joint_unit_bootstrap(&pair_design(30), 99, 1, "pairs sampled", &ctx);
        assert!(error.unwrap_err().contains("cancelled"));
    }
}
