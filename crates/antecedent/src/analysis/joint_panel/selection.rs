//! Observationally held-out whole-program selection using the native randomized
//! design owner. The bootstrap repeats the complete actual procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0
/// Continuous structural-dose selection with genuinely held-out original units.
pub mod instrumented;
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_design::program_search::{
    ProgramSearchError,
    selection::{
        CandidateEstimates, ObservationalSpan, SimultaneousProgramIntervals,
        procedure::{
            HeldoutProgramValue, ProgramSelectionProcedure, RankedProgram, SelectionDataBlock,
            WholeProcedureSelection, execute_heldout_selection,
        },
        simultaneous_program_intervals,
    },
};
use antecedent_estimate::joint_design::{
    JointDesignEstimate,
    randomized::{JointRandomizedDesign, JointRandomizedUnit, randomized_joint_means},
};
use serde::{Deserialize, Serialize};

/// Prespecified generation rule, repeated on every resampled training dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RandomizedCandidateGeneration {
    /// Search every declared complete program.
    All,
    /// Generate a subset using training HT utility estimates. The held-out
    /// evaluator never uses these training values as selected-value estimates.
    TrainingScreen {
        /// Maximum generated programs, at least one.
        top_k: usize,
    },
}
/// Utility and hard operational requirements of the retained selection procedure.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizedSelectionPolicy {
    /// Same target population's independent-unit sampling frame in both stages.
    /// Random assignment alone does not establish sampling/transport to this frame.
    pub independent_sampling_frame: String,
    /// Utility units shared by training, validation and every resampling slot.
    pub utility_units: String,
    /// Affine benefit coefficient on the actual experimental outcome mean.
    pub benefit_multiplier: f64,
    /// Utility cost of each complete program, in world order.
    pub utility_costs: Vec<f64>,
    /// Nonnegative resource requirement of each complete program.
    pub resource_costs: Vec<f64>,
    /// Inclusive resource budget, checked as a hard restriction.
    pub budget: f64,
    /// Fixed or data-dependent candidate generation.
    pub generation: RandomizedCandidateGeneration,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    design: JointRandomizedDesign,
    training: Vec<JointRandomizedUnit>,
    validation: Vec<JointRandomizedUnit>,
    policy: RandomizedSelectionPolicy,
}
/// Original observations, target and complete algorithm retained for execution/replay.
#[derive(Clone, Debug)]
pub struct PreparedRandomizedSelection {
    recipe: std::sync::Arc<Recipe>,
    identity: SemanticDigest,
}
/// Actual selected program with held-out observational uncertainty and bootstrap diagnostics.
#[derive(Debug)]
pub struct RandomizedSelectionResult {
    procedure: WholeProcedureSelection<usize>,
    identity: SemanticDigest,
    prepared: PreparedRandomizedSelection,
    seed: u64,
    repetitions: usize,
    tie_tolerance: f64,
}
/// Fewest complete programs a randomized selection compares.
const MIN_PROGRAMS: usize = 2;
/// Most complete programs a randomized selection compares: the held-out procedure refits and
/// ranks every program on every resample, so the count bounds the per-repetition work.
const MAX_PROGRAMS: usize = 256;

impl RandomizedSelectionResult {
    /// Generation, fitting, feasibility, search, selection, and evaluation receipt.
    #[must_use]
    pub const fn procedure(&self) -> &WholeProcedureSelection<usize> {
        &self.procedure
    }

    /// Complete original data/target/policy identity, rather than a simulation seed.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }
}

fn invalid(e: impl std::fmt::Display) -> ProgramSearchError {
    ProgramSearchError::Invalid(e.to_string())
}
impl PreparedRandomizedSelection {
    /// Bind actual distinct observational units, known randomization and common
    /// sampling/utility premises. Held-out units are never passed to selection.
    /// # Errors
    /// Malformed target/policy, overlapping original unit IDs or unsupported data.
    pub fn compile(
        design: JointRandomizedDesign,
        training: Vec<JointRandomizedUnit>,
        validation: Vec<JointRandomizedUnit>,
        policy: RandomizedSelectionPolicy,
    ) -> Result<Self, ProgramSearchError> {
        let k = design.worlds.len();
        if !(MIN_PROGRAMS..=MAX_PROGRAMS).contains(&k) {
            return Err(invalid(format!(
                "randomized selection admits {MIN_PROGRAMS}..{MAX_PROGRAMS} complete programs"
            )));
        }
        if policy.independent_sampling_frame.trim().is_empty()
            || policy.utility_units.trim().is_empty()
            || !policy.benefit_multiplier.is_finite()
            || !policy.budget.is_finite()
            || policy.budget < 0.0
            || policy.utility_costs.len() != k
            || policy.resource_costs.len() != k
            || policy.utility_costs.iter().any(|v| !v.is_finite())
            || policy.resource_costs.iter().any(|v| !v.is_finite() || *v < 0.0)
            || matches!(
                policy.generation,
                RandomizedCandidateGeneration::TrainingScreen { top_k: 0 }
            )
        {
            return Err(invalid("invalid shared population, utility, generation or hard budget"));
        }
        let ids = training.iter().map(|u| &u.id).collect::<std::collections::BTreeSet<_>>();
        if validation.iter().any(|u| ids.contains(&u.id)) {
            return Err(invalid(
                "training and held-out evaluation share original observational units",
            ));
        }
        // These admission checks do not use validation outcomes to generate,
        // fit, search or select a program.
        randomized_joint_means(&design, &training).map_err(invalid)?;
        randomized_joint_means(&design, &validation).map_err(invalid)?;
        let recipe = Recipe { design, training, validation, policy };
        let bytes = antecedent_io::convert::to_cbor(&recipe).map_err(invalid)?;
        let identity = SemanticDigest::from_bytes(CanonicalWriter::digest_bytes(
            "antecedent.joint_panel.randomized_selection.v1",
            &bytes,
        ));
        Ok(Self { recipe: std::sync::Arc::new(recipe), identity })
    }
    /// Execute complete selection on training units and fixed-action evaluation
    /// on independent held-out units. Original clusters resample as whole units.
    /// # Errors
    /// Failed stages, unsupported held-out program, cancellation or resources.
    pub fn execute(
        &self,
        repetitions: usize,
        tie_tolerance: f64,
        ctx: &ExecutionContext,
    ) -> Result<RandomizedSelectionResult, ProgramSearchError> {
        let blocks = |units: &[JointRandomizedUnit]| {
            units
                .iter()
                .map(|u| SelectionDataBlock {
                    span: ObservationalSpan { unit: u.id.clone(), first: 0, last: 0 },
                    observations: u.clone(),
                })
                .collect::<Vec<_>>()
        };
        let training = blocks(&self.recipe.training);
        let validation = blocks(&self.recipe.validation);
        let procedure = execute_heldout_selection(
            self,
            &training,
            &validation,
            0,
            repetitions,
            tie_tolerance,
            ctx,
        )?;
        Ok(RandomizedSelectionResult {
            procedure,
            identity: self.identity,
            prepared: self.clone(),
            seed: ctx.rng.master_seed(),
            repetitions,
            tie_tolerance,
        })
    }
    /// Verify replay against this actual current data, scientific target and procedure.
    /// # Errors
    /// Changed source/policy, forged output, malformed artifact or execution refusal.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<RandomizedSelectionResult, ProgramSearchError> {
        let result = replay_randomized_selection(bytes, ctx)?;
        if result.identity != self.identity {
            return Err(invalid("current selection data/target/procedure changed"));
        }
        Ok(result)
    }
    /// Joint max-statistic bootstrap intervals for the complete prespecified
    /// feasible program set. Every repetition refits the native randomized
    /// estimator on one common resample, preserving cross-program dependence.
    /// Data-dependent screening requires the held-out procedure instead.
    /// # Errors
    /// Changing candidate generation, failed replicate support, invalid level,
    /// cancellation, numerical or resource refusal.
    pub fn simultaneous_intervals(
        &self,
        repetitions: usize,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<SimultaneousProgramIntervals, ProgramSearchError> {
        if !matches!(self.recipe.policy.generation, RandomizedCandidateGeneration::All) {
            return Err(invalid(
                "fixed-set intervals cannot certify data-dependent candidate generation; execute held-out selection",
            ));
        }
        let arms = self
            .recipe
            .policy
            .resource_costs
            .iter()
            .enumerate()
            .filter_map(|(a, c)| (*c <= self.recipe.policy.budget).then_some(a))
            .collect::<Vec<_>>();
        if arms.is_empty() {
            return Err(invalid("no operationally feasible prespecified program"));
        }
        let quote = repetitions
            .checked_mul(arms.len())
            .and_then(|n| n.checked_mul(128))
            .ok_or_else(|| invalid("joint bootstrap storage overflow"))?;
        let _rows = ctx
            .acquire_admission(0, quote as u64)
            .map_err(|e| ProgramSearchError::Resource(e.to_string()))?;
        let estimates = |fit: &JointDesignEstimate| CandidateEstimates {
            ids: arms.iter().map(|a| fit.worlds[*a].clone()).collect(),
            values: arms.iter().map(|a| self.utility(fit, *a)).collect(),
            standard_errors: arms
                .iter()
                .map(|a| {
                    self.recipe.policy.benefit_multiplier.abs()
                        * fit.covariance[*a * fit.worlds.len() + *a].sqrt()
                })
                .collect(),
        };
        let data = self.recipe.training.iter().collect::<Vec<_>>();
        let observed = estimates(&self.fit_view(&data)?);
        let mut bootstrap = Vec::with_capacity(repetitions);
        for replicate in 0..repetitions {
            if ctx.cancellation.is_cancelled() {
                return Err(ProgramSearchError::Cancelled);
            }
            let mut rng = ctx.rng.stream_for(
                antecedent_core::StreamDomain::Design,
                0x5349_4d55_4c54_0000_u64.wrapping_add(replicate as u64),
            );
            let count = data.len() as u64;
            let threshold = count.wrapping_neg() % count;
            let sample = (0..data.len())
                .map(|_| {
                    let word = loop {
                        let word = rng.next_u64();
                        if word >= threshold {
                            break word;
                        }
                    };
                    data[usize::try_from(word % count).unwrap_or(0)]
                })
                .collect::<Vec<_>>();
            bootstrap.push(self.fit_view(&sample).ok().map(|fit| estimates(&fit)));
        }
        simultaneous_program_intervals(&observed, &bootstrap, level, ctx)
    }
    fn fit_view(
        &self,
        data: &[&JointRandomizedUnit],
    ) -> Result<JointDesignEstimate, ProgramSearchError> {
        // Bootstrap occurrences are separate empirical draws, but their original
        // observational identities remain in the retained partition/index receipt.
        let units = data
            .iter()
            .enumerate()
            .map(|(slot, unit)| {
                let mut copy = (*unit).clone();
                copy.id = format!("{}:bootstrap_occurrence:{slot}", unit.id);
                copy
            })
            .collect::<Vec<_>>();
        randomized_joint_means(&self.recipe.design, &units).map_err(invalid)
    }
    fn utility(&self, fit: &JointDesignEstimate, arm: usize) -> f64 {
        self.recipe.policy.benefit_multiplier * fit.means[arm]
            - self.recipe.policy.utility_costs[arm]
    }
}
impl ProgramSelectionProcedure<JointRandomizedUnit> for PreparedRandomizedSelection {
    type Candidates = Vec<usize>;
    type Fit = JointDesignEstimate;
    type Action = usize;
    fn generate(
        &self,
        data: &[&JointRandomizedUnit],
        _: usize,
        _: &ExecutionContext,
    ) -> Result<Vec<usize>, ProgramSearchError> {
        let mut arms = (0..self.recipe.design.worlds.len()).collect::<Vec<_>>();
        if let RandomizedCandidateGeneration::TrainingScreen { top_k } =
            self.recipe.policy.generation
        {
            let fit = self.fit_view(data)?;
            arms.sort_by(|a, b| {
                self.utility(&fit, *b)
                    .total_cmp(&self.utility(&fit, *a))
                    .then_with(|| self.recipe.design.worlds[*a].cmp(&self.recipe.design.worlds[*b]))
            });
            arms.truncate(top_k);
        }
        Ok(arms)
    }
    fn fit(
        &self,
        data: &[&JointRandomizedUnit],
        _: &Vec<usize>,
        _: usize,
        _: &ExecutionContext,
    ) -> Result<Self::Fit, ProgramSearchError> {
        self.fit_view(data)
    }
    fn feasible(
        &self,
        _: &Self::Fit,
        arms: Vec<usize>,
        _: &ExecutionContext,
    ) -> Result<Vec<usize>, ProgramSearchError> {
        Ok(arms
            .into_iter()
            .filter(|a| self.recipe.policy.resource_costs[*a] <= self.recipe.policy.budget)
            .collect())
    }
    fn search(
        &self,
        fit: &Self::Fit,
        arms: Vec<usize>,
        _: usize,
        _: &ExecutionContext,
    ) -> Result<Vec<RankedProgram<usize>>, ProgramSearchError> {
        Ok(arms
            .into_iter()
            .map(|arm| RankedProgram {
                id: self.recipe.design.worlds[arm].clone(),
                action: arm,
                training_value: self.utility(fit, arm),
            })
            .collect())
    }
    fn evaluate(
        &self,
        data: &[&JointRandomizedUnit],
        arm: &usize,
        _: usize,
        _: &ExecutionContext,
    ) -> Result<HeldoutProgramValue, ProgramSearchError> {
        let fit = self.fit_view(data)?;
        let mut coefficients = vec![0.0; fit.worlds.len()];
        coefficients[*arm] = self.recipe.policy.benefit_multiplier;
        let (value, variance) = fit.contrast(&coefficients).map_err(invalid)?;
        Ok(HeldoutProgramValue {
            target: format!(
                "{}:{}:{}",
                self.recipe.design.population,
                self.recipe.policy.independent_sampling_frame,
                self.recipe.policy.utility_units
            ),
            value: value - self.recipe.policy.utility_costs[*arm],
            standard_error: variance.sqrt(),
        })
    }
}
impl RandomizedSelectionResult {
    /// Failure-aware selected-value report. Sandwich uncertainty is asymptotic
    /// under the declared independent-unit model; no release coverage is licensed.
    #[must_use]
    pub fn report(&self) -> serde_json::Value {
        let observed = &self.procedure.observed;
        serde_json::json!({"identity":self.identity.to_hex(),"selected_program":observed.selected_id,
            "training_value":observed.training_value,"heldout":observed.heldout,
            "runner_up_gap":observed.runner_up_gap,"partition":self.procedure.partition,
            "repetitions":self.procedure.repetitions,"mean_optimism":self.procedure.mean_optimism,
            "changed_selection":self.procedure.changed_selection,"near_ties":self.procedure.near_ties,
            "uncertainty":"independent_unit_sandwich_on_reserved_observations",
            "bootstrap_role":"whole_procedure_diagnostics_no_ordinary_bootstrap_coverage_at_ties",
            "calibration":"unmeasured"})
    }
    /// Export original data and policy with a report that must be re-executed.
    /// # Errors
    /// Artifact encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ProgramSearchError> {
        antecedent_io::convert::to_cbor(&Document {
            version: 1,
            recipe: Recipe::clone(&self.prepared.recipe),
            seed: self.seed,
            repetitions: self.repetitions,
            tie_tolerance: self.tie_tolerance,
            report: self.report(),
        })
        .map_err(invalid)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    recipe: Recipe,
    seed: u64,
    repetitions: usize,
    tie_tolerance: f64,
    report: serde_json::Value,
}
/// Reconstruct actual randomized estimators and repeat every selection stage.
/// Historical replay alone grants no caller-current evidence authorization.
/// # Errors
/// Invalid/tampered artifact, unsupported design, resources or cancellation.
pub fn replay_randomized_selection(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<RandomizedSelectionResult, ProgramSearchError> {
    if bytes.len() > 32 * 1024 * 1024 {
        return Err(invalid("selection artifact size limit"));
    }
    let _input = ctx
        .acquire_admission(0, (bytes.len() as u64).saturating_mul(16))
        .map_err(|e| ProgramSearchError::Resource(e.to_string()))?;
    let d: Document = antecedent_io::convert::from_cbor(bytes).map_err(invalid)?;
    if d.version != 1 {
        return Err(invalid("unsupported selection artifact version"));
    }
    let mut execution = ctx.clone();
    execution.rng = antecedent_core::execution::RngFactory::from_seed(d.seed);
    let prepared = PreparedRandomizedSelection::compile(
        d.recipe.design,
        d.recipe.training,
        d.recipe.validation,
        d.recipe.policy,
    )?;
    let result = prepared.execute(d.repetitions, d.tie_tolerance, &execution)?;
    if result.report() != d.report {
        return Err(invalid("whole-procedure selection replay mismatch"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn fixture(
        generation: RandomizedCandidateGeneration,
    ) -> PreparedRandomizedSelection {
        let design = JointRandomizedDesign {
            worlds: vec!["a".into(), "b".into(), "c".into()],
            sampling_unit: "independently sampled randomized cluster".into(),
            population: "common sampling frame".into(),
            consistency: "exact complete program and common outcome measure".into(),
            between_unit_isolation: "no interference across sampled clusters".into(),
        };
        let units = |stage: &str| {
            (0..240)
                .map(|i| JointRandomizedUnit {
                    id: format!("{stage}:{i}"),
                    arm: i % 3,
                    probabilities: vec![1.0 / 3.0; 3],
                    outcome: [1.0, 1.1, 3.0][i % 3] + 0.1 * (i as f64).sin(),
                    population_weight: 1.0,
                })
                .collect()
        };
        PreparedRandomizedSelection::compile(
            design,
            units("train"),
            units("validation"),
            RandomizedSelectionPolicy {
                independent_sampling_frame: "IID clusters from the same predeclared frame".into(),
                utility_units: "outcome units".into(),
                benefit_multiplier: 1.0,
                utility_costs: vec![0.0; 3],
                resource_costs: vec![1.0, 1.0, 10.0],
                budget: 2.0,
                generation,
            },
        )
        .unwrap()
    }
    #[test]
    fn actual_randomized_selection_obeys_budget_uses_holdout_and_replays_every_stage() {
        let prepared = fixture(RandomizedCandidateGeneration::TrainingScreen { top_k: 3 });
        let ctx = ExecutionContext::for_tests(17);
        let result = prepared.execute(40, 0.05, &ctx).unwrap();
        assert_eq!(result.procedure.observed.selected_id, "b");
        assert!((result.procedure.observed.heldout.value - 1.1).abs() < 0.02);
        assert!(
            result
                .procedure
                .repetitions
                .iter()
                .all(|r| r.result.as_ref().is_some_and(|r| r.selected_id != "c"))
        );
        let bytes = result.export().unwrap();
        let replay = prepared.consume_current(&bytes, &ExecutionContext::for_tests(99)).unwrap();
        assert_eq!(replay.report(), result.report());
        let mut altered = prepared.clone();
        altered.identity = SemanticDigest::from_bytes([99; 32]);
        assert!(altered.consume_current(&bytes, &ctx).is_err());
        assert!(prepared.simultaneous_intervals(99, 0.95, &ctx).is_err());
    }
    #[test]
    fn actual_joint_bootstrap_preserves_the_fixed_feasible_family() {
        let prepared = fixture(RandomizedCandidateGeneration::All);
        let intervals =
            prepared.simultaneous_intervals(99, 0.95, &ExecutionContext::for_tests(1)).unwrap();
        assert_eq!(intervals.ids, ["a", "b"]);
        assert_eq!(intervals.roots.len(), 99);
        assert_eq!(intervals.selected_id, "b");
        for (bounds, truth) in intervals.intervals.iter().zip([1.0, 1.1]) {
            assert!(bounds[0] <= truth && bounds[1] >= truth);
        }
    }

    fn selected_value_simulation(
        seed: u64,
        n: usize,
        signal: f64,
        repetitions: usize,
    ) -> (bool, f64, f64) {
        let base = fixture(RandomizedCandidateGeneration::TrainingScreen { top_k: 2 });
        let mut rng = antecedent_core::execution::CausalRng::from_seed(seed);
        let mut units = |stage: &str| {
            (0..n)
                .map(|i| {
                    let arm = usize::try_from(rng.next_u64() % 3).unwrap();
                    let noise = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt()
                        * (std::f64::consts::TAU * rng.next_f64()).cos();
                    JointRandomizedUnit {
                        id: format!("{stage}:{i}"),
                        arm,
                        probabilities: vec![1.0 / 3.0; 3],
                        outcome: if arm == 1 { signal + noise } else { noise },
                        population_weight: 1.0,
                    }
                })
                .collect::<Vec<_>>()
        };
        let training = units("train");
        let validation = units("validation");
        let mut policy = base.recipe.policy.clone();
        policy.resource_costs = vec![1.0; 3];
        let prepared =
            PreparedRandomizedSelection::compile(
                base.recipe.design.clone(),
                training,
                validation,
                policy,
            )
                .unwrap();
        let result =
            prepared.execute(repetitions, 0.01, &ExecutionContext::for_tests(seed)).unwrap();
        let selected = &result.procedure.observed;
        let truth = if selected.action == 1 { signal } else { 0.0 };
        let error = selected.heldout.value - truth;
        (
            error.abs() <= 1.959963984540054 * selected.heldout.standard_error,
            selected.training_value - truth,
            error,
        )
    }

    #[test]
    fn actual_randomized_selected_value_coverage_exercises_winners_curse_across_datasets() {
        let outcomes =
            (0..400).map(|seed| selected_value_simulation(seed, 240, 0.0, 0)).collect::<Vec<_>>();
        let covered = outcomes.iter().filter(|r| r.0).count();
        assert!(covered >= 365, "selected coverage {covered}/400");
        assert!(outcomes.iter().map(|r| r.1).sum::<f64>() / 400.0 > 0.04);
        assert!((outcomes.iter().map(|r| r.2).sum::<f64>() / 400.0).abs() < 0.025);
    }

    #[test]
    #[ignore = "Q08/E14: execute only on the stable candidate; no implementation-stage calibration license"]
    fn whole_procedure_randomized_selected_value_calibration_l95() {
        let repetitions = std::env::var("ANTECEDENT_CALIBRATION_NSIM")
            .ok()
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(400)
            .max(400);
        // Source-fixed cells include exact ties, near ties, a separated winner,
        // two observational sample sizes, and actual whole-procedure resampling.
        for n in [240, 480] {
            for signal in [0.0, 0.02, 0.2] {
                let mut covered = 0usize;
                let mut optimism = 0.0;
                let mut holdout_bias = 0.0;
                for replicate in 0..repetitions {
                    let seed =
                        antecedent_core::derive_stream(0x5830_3453_454c_4543, replicate as u64);
                    let result = selected_value_simulation(seed, n, signal, 10);
                    covered += usize::from(result.0);
                    optimism += result.1;
                    holdout_bias += result.2;
                }
                let rate = covered as f64 / repetitions as f64;
                let mcse = (0.95 * 0.05 / (repetitions as f64)).sqrt();
                println!(
                    "selected-procedure-cell n={n} signal={signal} replicates={repetitions} covered={covered} coverage={rate} optimism={} heldout_bias={}",
                    optimism / repetitions as f64,
                    holdout_bias / repetitions as f64
                );
                assert!((rate - 0.95).abs() <= 3.0 * mcse + 0.005);
                assert!(
                    (holdout_bias / repetitions as f64).abs()
                        <= 4.0 * (3.0 / (n as f64 * repetitions as f64)).sqrt()
                );
            }
        }
    }
}
