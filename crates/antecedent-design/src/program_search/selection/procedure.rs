//! Execute every selection stage on training observations, then evaluate on a
//! separate observational sample. Bootstrap repetitions are diagnostics; at ties
//! they do not acquire an ordinary-bootstrap coverage interpretation.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    Arc, Deserialize, ExecutionContext, ObservationalSpan, ProgramHoldoutPartition,
    ProgramSearchError, Serialize, cancelled, invalid,
};

/// One original sampling block. Multiple blocks of a unit resample together.
#[derive(Clone, Debug)]
pub struct SelectionDataBlock<'a, D: Clone> {
    /// Actual observational unit and full preprocessing/lookback span.
    pub span: ObservationalSpan,
    /// Observations read by the method owner; borrowed when the caller already holds them.
    pub observations: std::borrow::Cow<'a, D>,
}
/// A program returned by the search stage, after operational feasibility checks.
#[derive(Clone, Debug)]
pub struct RankedProgram<A> {
    /// Stable complete program identity; continuous programs include their controls.
    pub id: String,
    /// Actual action consumed by held-out evaluation.
    pub action: A,
    /// Training-data utility estimate used for selection.
    pub training_value: f64,
}
/// Evaluation of one fixed, already selected action on reserved observations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeldoutProgramValue {
    /// Same scientific target and utility scale in every procedure repetition.
    pub target: String,
    /// Estimate evaluated using only validation observations.
    pub value: f64,
    /// Standard error under the evaluation owner's stated sampling model.
    pub standard_error: f64,
}
/// A method-owned selection pipeline. Each stage receives only its permitted
/// observational view. Implementations must retain preprocessing inside these
/// stages; pre-fitting with validation data violates this contract.
pub trait ProgramSelectionProcedure<D> {
    /// Generated candidates or a continuous domain.
    type Candidates;
    /// Fit used by feasibility and search.
    type Fit;
    /// Complete selected program, not just a value or a simulation seed.
    type Action;
    /// Generate candidates/domain from training observations.
    fn generate(
        &self,
        training: &[&D],
        replicate: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self::Candidates, ProgramSearchError>;
    /// Repeat all fitting, model selection and smoothing on this training sample.
    fn fit(
        &self,
        training: &[&D],
        candidates: &Self::Candidates,
        replicate: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, ProgramSearchError>;
    /// Admit operational/statistical feasibility using the fit for this repetition.
    fn feasible(
        &self,
        fit: &Self::Fit,
        candidates: Self::Candidates,
        ctx: &ExecutionContext,
    ) -> Result<Self::Candidates, ProgramSearchError>;
    /// Run the complete finite or continuous search on the admitted domain.
    fn search(
        &self,
        fit: &Self::Fit,
        candidates: Self::Candidates,
        replicate: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RankedProgram<Self::Action>>, ProgramSearchError>;
    /// Evaluate the fixed selected action on reserved observational data. This
    /// stage receives no training fit and may not regenerate or reselect actions.
    fn evaluate(
        &self,
        validation: &[&D],
        action: &Self::Action,
        replicate: usize,
        ctx: &ExecutionContext,
    ) -> Result<HeldoutProgramValue, ProgramSearchError>;
}
/// Exact stage at which a complete procedure failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStage {
    /// Data-dependent candidate/domain generation.
    Generation,
    /// Mechanism/nuisance fitting, model selection and smoothing.
    Fitting,
    /// Operational and statistical feasibility checks.
    Feasibility,
    /// Finite enumeration or continuous optimization.
    Search,
    /// Winner selection and deterministic tie handling.
    Selection,
    /// Fixed-action evaluation on reserved observations.
    Evaluation,
}
/// One actual procedure result, with stable deterministic tie handling.
#[derive(Debug)]
pub struct HeldoutSelection<A> {
    /// Selected complete action.
    pub action: A,
    /// Selected program identity.
    pub selected_id: String,
    /// Optimized training value, kept separate from held-out value.
    pub training_value: f64,
    /// Gap to runner-up; absent for a single candidate.
    pub runner_up_gap: Option<f64>,
    /// Selected-action estimate evaluated on independent observational data.
    pub heldout: HeldoutProgramValue,
}
/// One retained whole-procedure repetition, including failures and data indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionRepetition {
    /// Indices into the original training blocks, including repeated clusters.
    pub training_indices: Vec<usize>,
    /// Indices into original validation blocks, never into training data.
    pub validation_indices: Vec<usize>,
    /// Actual completed result, absent if any stage failed.
    pub result: Option<SelectionRepetitionValue>,
    /// Failed stage and concrete refusal, preserved instead of dropping the slot.
    pub failure: Option<(SelectionStage, String)>,
}
/// Numerical diagnostics for a completed repetition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionRepetitionValue {
    /// Complete selected program identity.
    pub selected_id: String,
    /// Training-data selected utility.
    pub training_value: f64,
    /// Independent observational evaluation of that selected program.
    pub heldout: HeldoutProgramValue,
    /// Training separation from the runner-up.
    pub runner_up_gap: Option<f64>,
}
/// Executed selection, independent evaluation and full-procedure diagnostics.
#[derive(Debug)]
pub struct WholeProcedureSelection<A> {
    /// Original-data selected action and held-out evaluation.
    pub observed: HeldoutSelection<A>,
    /// Exact observed-data split used by all stages.
    pub partition: ProgramHoldoutPartition,
    /// All resampling slots, including failed attempts.
    pub repetitions: Vec<SelectionRepetition>,
    /// Mean training-minus-heldout utility over successful repetitions. This is
    /// a selection-optimism diagnostic, not an automatic coverage correction.
    pub mean_optimism: Option<f64>,
    /// Successful repetitions choosing a different program identity.
    pub changed_selection: usize,
    /// Successful repetitions whose runner-up gap is within the declared tie tolerance.
    pub near_ties: usize,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl<A> WholeProcedureSelection<A> {
    /// Conservative retained reservation for partition, indices and diagnostics.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}
type StageFailure = (SelectionStage, ProgramSearchError);
fn run<D, P: ProgramSelectionProcedure<D>>(
    owner: &P,
    train: &[&D],
    validation: &[&D],
    replicate: usize,
    ctx: &ExecutionContext,
) -> Result<HeldoutSelection<P::Action>, StageFailure> {
    let failure = |stage, e: ProgramSearchError| (stage, e);
    let candidates = owner
        .generate(train, replicate, ctx)
        .map_err(|e| failure(SelectionStage::Generation, e))?;
    let fit = owner
        .fit(train, &candidates, replicate, ctx)
        .map_err(|e| failure(SelectionStage::Fitting, e))?;
    let candidates = owner
        .feasible(&fit, candidates, ctx)
        .map_err(|e| failure(SelectionStage::Feasibility, e))?;
    let mut ranked = owner
        .search(&fit, candidates, replicate, ctx)
        .map_err(|e| failure(SelectionStage::Search, e))?;
    let mut seen = std::collections::BTreeSet::new();
    if ranked.is_empty()
        || ranked.iter().any(|r| {
            r.id.trim().is_empty()
                || r.id.len() > 4096
                || !r.training_value.is_finite()
                || !seen.insert(&r.id)
        })
    {
        return Err((
            SelectionStage::Selection,
            invalid("empty, duplicate, oversized or nonfinite search result"),
        ));
    }
    drop(seen);
    ranked.sort_by(|a, b| {
        b.training_value.total_cmp(&a.training_value).then_with(|| a.id.cmp(&b.id))
    });
    let gap = ranked.get(1).map(|r| ranked[0].training_value - r.training_value);
    if gap.is_some_and(|value| !value.is_finite()) {
        return Err((SelectionStage::Selection, invalid("nonfinite runner-up utility gap")));
    }
    let selected = ranked.remove(0);
    let heldout = owner
        .evaluate(validation, &selected.action, replicate, ctx)
        .map_err(|e| failure(SelectionStage::Evaluation, e))?;
    if heldout.target.trim().is_empty()
        || heldout.target.len() > 4096
        || !heldout.value.is_finite()
        || !heldout.standard_error.is_finite()
        || heldout.standard_error < 0.0
    {
        return Err((SelectionStage::Evaluation, invalid("invalid held-out target/value/error")));
    }
    Ok(HeldoutSelection {
        action: selected.action,
        selected_id: selected.id,
        training_value: selected.training_value,
        runner_up_gap: gap,
        heldout,
    })
}
fn clusters<D: Clone>(data: &[SelectionDataBlock<'_, D>]) -> Vec<Vec<usize>> {
    let mut groups = std::collections::BTreeMap::<&str, Vec<usize>>::new();
    for (i, block) in data.iter().enumerate() {
        groups.entry(&block.span.unit).or_default().push(i);
    }
    groups.into_values().collect()
}
fn sample(groups: &[Vec<usize>], rng: &mut antecedent_core::execution::CausalRng) -> Vec<usize> {
    let count = groups.len() as u64;
    let threshold = count.wrapping_neg() % count;
    let mut indices = Vec::new();
    for _ in groups {
        let word = loop {
            let word = rng.next_u64();
            if word >= threshold {
                break word;
            }
        };
        let index = usize::try_from(word % count).unwrap_or(0);
        indices.extend_from_slice(&groups[index]);
    }
    indices
}
fn selection_quote(
    training: &[Vec<usize>],
    validation: &[Vec<usize>],
    repetitions: usize,
    partition_bytes: usize,
) -> Result<usize, ProgramSearchError> {
    // Sampling a long cluster repeatedly can exceed the original row count.
    let maximum = |clusters: &[Vec<usize>]| {
        clusters.iter().map(Vec::len).max().unwrap_or(0).checked_mul(clusters.len())
    };
    maximum(training)
        .and_then(|n| n.checked_add(maximum(validation)?))
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| n.checked_add(16384))
        .and_then(|n| n.checked_mul(repetitions.checked_add(1)?))
        .and_then(|n| n.checked_add(partition_bytes))
        .ok_or_else(|| invalid("selection procedure storage overflow"))
}
/// Execute selection and evaluate the selected action on genuinely reserved data.
/// Resampling repeats generation, fitting, feasibility, search, selection and
/// evaluation. Original unit clusters are resampled together within each stage.
///
/// Unit separation establishes disjoint observations. For a shared unit's time
/// blocks, the required gap must cover lookback and the evaluation method must
/// additionally justify its dependence model; a gap alone grants no independence.
/// Bootstrap repetitions are retained diagnostics even at ties; coverage follows
/// the held-out evaluation owner's fixed-action sampling model, not an assumed
/// ordinary bootstrap theorem for the winner.
///
/// # Errors
/// Invalid/overlapping partitions, original-procedure failure, changed targets,
/// cancellation, resources or invalid tie tolerance.
pub fn execute_heldout_selection<D: Clone, P: ProgramSelectionProcedure<D>>(
    owner: &P,
    training: &[SelectionDataBlock<'_, D>],
    validation: &[SelectionDataBlock<'_, D>],
    required_gap: u64,
    repetitions: usize,
    tie_tolerance: f64,
    ctx: &ExecutionContext,
) -> Result<WholeProcedureSelection<P::Action>, ProgramSearchError> {
    cancelled(ctx)?;
    if !tie_tolerance.is_finite() || tie_tolerance < 0.0 {
        return Err(invalid("invalid selection tie tolerance"));
    }
    let partition_bytes = training
        .iter()
        .chain(validation)
        .try_fold(0usize, |total, block| {
            total.checked_add(block.span.unit.len()).and_then(|n| n.checked_add(256))
        })
        .ok_or_else(|| invalid("selection partition storage overflow"))?;
    let _partition_scratch = ctx
        .acquire_admission(0, partition_bytes as u64)
        .map_err(|e| ProgramSearchError::Resource(e.to_string()))?;
    let partition = ProgramHoldoutPartition {
        training: training.iter().map(|b| b.span.clone()).collect(),
        validation: validation.iter().map(|b| b.span.clone()).collect(),
        required_gap,
    };
    partition.validate()?;
    let train_clusters = clusters(training);
    let validation_clusters = clusters(validation);
    let quote =
        selection_quote(&train_clusters, &validation_clusters, repetitions, partition_bytes)?;
    let retention = Arc::new(
        ctx.acquire_admission(0, quote as u64)
            .map_err(|e| ProgramSearchError::Resource(e.to_string()))?,
    );
    let train = training.iter().map(|b| &*b.observations).collect::<Vec<_>>();
    let valid = validation.iter().map(|b| &*b.observations).collect::<Vec<_>>();
    let observed = run(owner, &train, &valid, 0, ctx).map_err(|(stage, error)| match error {
        ProgramSearchError::Cancelled => ProgramSearchError::Cancelled,
        error @ ProgramSearchError::Resource(_) => error,
        error => ProgramSearchError::upstream(format!("{stage:?}"), error),
    })?;
    let mut rows = Vec::with_capacity(repetitions);
    let mut optimism = 0.0;
    let mut successful = 0usize;
    let mut changed_selection = 0;
    let mut near_ties = 0;
    for replicate in 0..repetitions {
        cancelled(ctx)?;
        let mut rng = ctx.rng.stream_for(
            antecedent_core::StreamDomain::Design,
            0x5345_4c45_4354_0000_u64.wrapping_add(replicate as u64),
        );
        let ti = sample(&train_clusters, &mut rng);
        let vi = sample(&validation_clusters, &mut rng);
        let train = ti.iter().map(|i| &*training[*i].observations).collect::<Vec<_>>();
        let valid = vi.iter().map(|i| &*validation[*i].observations).collect::<Vec<_>>();
        let result = run(owner, &train, &valid, replicate + 1, ctx);
        cancelled(ctx)?;
        let (result, failure) = match result {
            Ok(r) => {
                if r.heldout.target != observed.heldout.target {
                    return Err(invalid("whole-procedure evaluation changed scientific target"));
                }
                successful += 1;
                let difference = r.training_value - r.heldout.value;
                if !difference.is_finite() {
                    return Err(invalid("selection optimism overflow"));
                }
                let inverse_count = 1.0 / (successful as f64);
                optimism = optimism * (1.0 - inverse_count) + difference * inverse_count;
                changed_selection += usize::from(r.selected_id != observed.selected_id);
                near_ties += usize::from(r.runner_up_gap.is_some_and(|gap| gap <= tie_tolerance));
                (
                    Some(SelectionRepetitionValue {
                        selected_id: r.selected_id,
                        training_value: r.training_value,
                        heldout: r.heldout,
                        runner_up_gap: r.runner_up_gap,
                    }),
                    None,
                )
            }
            Err((_, ProgramSearchError::Cancelled)) => return Err(ProgramSearchError::Cancelled),
            Err((stage, error)) => {
                (None, Some((stage, error.to_string().chars().take(1024).collect())))
            }
        };
        rows.push(SelectionRepetition {
            training_indices: ti,
            validation_indices: vi,
            result,
            failure,
        });
    }
    Ok(WholeProcedureSelection {
        observed,
        partition,
        repetitions: rows,
        mean_optimism: (successful > 0).then_some(optimism),
        changed_selection,
        near_ties,
        retention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program_search::certified::*;
    use antecedent_core::{ValueExpr, VariableId};
    use std::cell::Cell;
    #[derive(Clone)]
    struct Observation {
        validation: bool,
        value: f64,
    }
    struct Pipeline {
        calls: [Cell<usize>; 5],
        fail: Option<usize>,
    }
    impl Pipeline {
        fn new(fail: Option<usize>) -> Self {
            Self { calls: std::array::from_fn(|_| Cell::new(0)), fail }
        }
        fn called(&self, stage: usize) {
            self.calls[stage].set(self.calls[stage].get() + 1);
        }
    }
    impl ProgramSelectionProcedure<Observation> for Pipeline {
        type Candidates = f64;
        type Fit = f64;
        type Action = f64;
        fn generate(
            &self,
            d: &[&Observation],
            _: usize,
            _: &ExecutionContext,
        ) -> Result<f64, ProgramSearchError> {
            self.called(0);
            assert!(d.iter().all(|o| !o.validation));
            // Domain generation itself depends on the observed training sample.
            Ok((d.iter().map(|o| o.value).sum::<f64>() / (d.len() as f64)).abs() + 1.0)
        }
        fn fit(
            &self,
            d: &[&Observation],
            _: &f64,
            replicate: usize,
            _: &ExecutionContext,
        ) -> Result<f64, ProgramSearchError> {
            self.called(1);
            assert!(d.iter().all(|o| !o.validation));
            if self.fail == Some(replicate) {
                return Err(invalid("deliberate fit failure"));
            }
            Ok(d.iter().map(|o| o.value).sum::<f64>() / (d.len() as f64))
        }
        fn feasible(
            &self,
            _: &f64,
            upper: f64,
            _: &ExecutionContext,
        ) -> Result<f64, ProgramSearchError> {
            self.called(2);
            Ok(upper.min(2.0))
        }
        fn search(
            &self,
            b: &f64,
            upper: f64,
            _: usize,
            ctx: &ExecutionContext,
        ) -> Result<Vec<RankedProgram<f64>>, ProgramSearchError> {
            self.called(3);
            let x = ValueExpr::var(VariableId::from_raw(0));
            let p = PreparedCertifiedSearch::compile(
                ValueExpr::constant(*b)
                    .mul(x.clone())
                    .sub(ValueExpr::constant(0.5).mul(x.clone().mul(x))),
                vec![CertifiedControl {
                    variable: VariableId::from_raw(0),
                    bounds: ObjectiveInterval { lower: 0.0, upper },
                }],
                vec![],
            )?;
            let c = p.execute(
                vec![0.0],
                CertificateOptions { absolute_gap: 1e-4, max_splits: 500 },
                ctx,
            )?;
            let action = c.controls[0];
            let mut programs = vec![RankedProgram {
                id: format!("control:{:016x}", action.to_bits()),
                action,
                training_value: b * action - 0.5 * action * action,
            }];
            if action != 0.0 {
                programs.push(RankedProgram {
                    id: "control:0000000000000000".into(),
                    action: 0.0,
                    training_value: 0.0,
                });
            }
            Ok(programs)
        }
        fn evaluate(
            &self,
            d: &[&Observation],
            x: &f64,
            _: usize,
            _: &ExecutionContext,
        ) -> Result<HeldoutProgramValue, ProgramSearchError> {
            self.called(4);
            assert!(d.iter().all(|o| o.validation));
            let mean = d.iter().map(|o| o.value).sum::<f64>() / (d.len() as f64);
            Ok(HeldoutProgramValue {
                target: "normal_linear_response_minus_known_quadratic_cost".into(),
                value: mean * x - 0.5 * x * x,
                standard_error: x.abs() / (d.len() as f64).sqrt(),
            })
        }
    }
    fn data(seed: u64, validation: bool, n: usize) -> Vec<SelectionDataBlock<'static, Observation>> {
        let mut rng = antecedent_core::execution::CausalRng::from_seed(seed);
        (0..n)
            .map(|i| {
                let normal = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt()
                    * (std::f64::consts::TAU * rng.next_f64()).cos();
                SelectionDataBlock {
                    span: ObservationalSpan {
                        unit: format!("{validation}:{i}"),
                        first: 0,
                        last: 0,
                    },
                    observations: std::borrow::Cow::Owned(Observation { validation, value: 0.3 + normal }),
                }
            })
            .collect()
    }
    #[test]
    fn complete_continuous_search_repeats_all_stages_and_retains_failed_slots() {
        let owner = Pipeline::new(Some(3));
        let result = execute_heldout_selection(
            &owner,
            &data(1, false, 8),
            &data(99, true, 16),
            0,
            30,
            1e-3,
            &ExecutionContext::for_tests(5),
        )
        .unwrap();
        assert_eq!(result.repetitions.len(), 30);
        assert_eq!(result.repetitions[2].failure.as_ref().unwrap().0, SelectionStage::Fitting);
        assert!(result.repetitions[2].result.is_none());
        assert_eq!(owner.calls[0].get(), 31);
        assert_eq!(owner.calls[1].get(), 31);
        assert_eq!(owner.calls[2].get(), 30);
        assert_eq!(owner.calls[3].get(), 30);
        assert_eq!(owner.calls[4].get(), 30);
        assert!(result.changed_selection > 0);
        assert!(result.mean_optimism.unwrap().is_finite());
        assert!(result.admitted_bytes() > 0);
    }
    #[test]
    fn heldout_selected_value_has_known_normal_coverage_despite_training_optimism() {
        let mut covered = 0;
        let mut optimism = 0.0;
        let mut holdout_error = 0.0;
        for seed in 0..400 {
            let owner = Pipeline::new(None);
            let result = execute_heldout_selection(
                &owner,
                &data(2 * seed + 1, false, 8),
                &data(2 * seed + 2, true, 16),
                0,
                0,
                0.0,
                &ExecutionContext::for_tests(seed),
            )
            .unwrap();
            let r = result.observed;
            // Independent analytic truth for this randomly selected action.
            let truth = 0.3 * r.action - 0.5 * r.action.powi(2);
            let error = r.heldout.value - truth;
            covered +=
                usize::from(error.abs() <= 1.959963984540054 * r.heldout.standard_error + 1e-14);
            optimism += r.training_value - truth;
            holdout_error += error;
        }
        assert!(covered >= 370, "{covered}/400 selected intervals covered");
        assert!(optimism / 400.0 > 0.04, "training winner should exhibit optimism");
        assert!((holdout_error / 400.0).abs() < 0.025);
        // This implementation scenario does not issue a release calibration license.
    }
}
