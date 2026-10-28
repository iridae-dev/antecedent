//! Frequentist circular-block mechanism-refit bootstrap of lag-aligned `TemporalDag`
//! attribution: known truth, reproducibility across thread budgets, the admission
//! refusal, and the declared-design refusal of a non-series resample.
//!
//! The interval is unlicensed and its calibration unmeasured; these tests pin what the
//! construction computes, not its coverage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ChangeAttributionQuery, DeclaredSamplingDesign, ExecutionContext, IntervalMethod,
    Lag, NonZeroThreadCount, Parallelism, PopulationSelector, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 400;

/// Deterministic standard normals (xorshift64* and Box–Muller).
struct Normals(u64);

impl Normals {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// `x_t = 0.5 x_{t-1} + u`, `y_t = 1 + 2 x_{t-1} + e` in the first half and
/// `y_t = 6 + 2 x_{t-1} + e` in the second: the outcome mechanism's intercept shifts
/// by 5, so the population mean change of `y` is 5 up to the sampling noise.
fn shifted_series(seed: u64) -> TimeSeriesData {
    let mut draw = Normals(seed | 1);
    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];
    for t in 1..N {
        x[t] = 0.5 * x[t - 1] + draw.next();
        let intercept = if t < N / 2 { 1.0 } else { 6.0 };
        y[t] = intercept + 2.0 * x[t - 1] + draw.next();
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn chain() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

fn change_query() -> CausalQuery {
    let half = N / 2;
    CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: half },
        PopulationSelector::TimeRange { start: half, end: N },
    ))
}

fn run(
    data: &TimeSeriesData,
    replicates: u32,
    design: Option<DeclaredSamplingDesign>,
    ctx: &ExecutionContext,
) -> Result<StudyResult, antecedent::CausalError> {
    let builder = Study::series(data.clone())
        .graph(chain())
        .query(change_query())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates);
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build()?.run(ctx)
}

#[test]
fn block_bootstrap_recovers_the_known_mean_shift_with_a_row_block_interval() {
    let data = shifted_series(41);
    let result = run(&data, 99, None, &ExecutionContext::for_tests(3)).unwrap();
    let total = result.change_attribution().unwrap().total_change;
    let estimate = result.estimate().as_effect().unwrap();
    let se = estimate.se_bootstrap().expect("a block-bootstrap standard error");
    // The two population means differ by 5; the published SE is the SD of the
    // replicate totals, of the order of a two-sample mean difference (about 0.2 here).
    assert!(se > 0.03 && se < 1.0, "se {se}");
    assert!((total - 5.0).abs() < 5.0 * se + 0.5, "total {total} se {se}");
    let binding = result.interval().as_ref().expect("an interval binding");
    assert_eq!(binding.method, IntervalMethod::CircularBlockSe);
    assert_eq!(binding.dependence, "circular_block:row_block");
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.frequentist_bootstrap"
            && d.message.contains("circular-block multinomial bootstrap")),
        "block disclosure"
    );
}

#[test]
fn block_bootstrap_is_the_same_for_any_thread_budget() {
    let data = shifted_series(43);
    let serial = ExecutionContext::for_tests(9);
    let mut parallel = ExecutionContext::for_tests(9);
    parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
    // The two budgets are independent runs of the same study, so they run side by side.
    let (a, b) = std::thread::scope(|scope| {
        let serial_run = scope.spawn(|| run(&data, 49, None, &serial).unwrap());
        let parallel_run = run(&data, 49, None, &parallel).unwrap();
        (
            serial_run.join().unwrap_or_else(|payload| std::panic::resume_unwind(payload)),
            parallel_run,
        )
    });
    let se = |result: &StudyResult| result.estimate().as_effect().unwrap().se_bootstrap().unwrap();
    assert_eq!(se(&a).to_bits(), se(&b).to_bits());
}

#[test]
fn a_tiny_pool_refuses_the_replicate_store_and_returns_its_lease() {
    let data = shifted_series(47);
    // The pool holds the planned design but not 20,000 replicate values.
    let tiny = ExecutionContext::for_tests(1)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 60_000));
    let before = tiny.admission.available();
    let error = run(&data, 20_000, None, &tiny).unwrap_err();
    assert!(error.to_string().contains("admission"), "{error}");
    assert_eq!(tiny.admission.available(), before);
}

#[test]
fn a_declared_independent_subject_design_refuses_the_block_resample() {
    let data = shifted_series(53);
    let error = run(
        &data,
        49,
        Some(DeclaredSamplingDesign::IndependentSubjectTrajectories),
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert!(error.to_string().contains("dependent series"), "{error}");
    // Without replicates the point value stands under any declaration.
    let point = run(
        &data,
        0,
        Some(DeclaredSamplingDesign::IndependentSubjectTrajectories),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert!(point.change_attribution().is_some());
}
