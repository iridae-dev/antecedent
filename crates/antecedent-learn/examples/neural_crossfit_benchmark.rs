//! Reproducible CPU baseline for the X7 accelerator decision.
//!
//! Run with `cargo run --release -p antecedent-learn --features ml-neural
//! --example neural_crossfit_benchmark`.

use std::hint::black_box;
use std::time::Instant;

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    DesignView, LearnerSpec, NeuralSpec, PredictionTask, TargetView, assign_folds,
    cross_fit_with_folds, resolve_for,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const ROWS: usize = 4_000;
    const COLS: usize = 8;
    const FOLDS: usize = 3;
    let ctx = ExecutionContext::for_tests(21);
    let mut design = vec![0.0; ROWS * COLS];
    let mut treatment = vec![0.0; ROWS];
    let mut outcome = vec![0.0; ROWS];
    for row in 0..ROWS {
        for col in 0..COLS {
            design[col * ROWS + row] = ((row * (col + 3) + col * 17) % 997) as f64 / 997.0 - 0.5;
        }
        treatment[row] = f64::from(u8::from(((row * 37 + 11) % 101) < 51));
        outcome[row] = 1.5 * treatment[row] + 0.7 * design[row] - 0.4 * design[ROWS + row]
            + 0.1 * ((row * 19 % 31) as f64 / 31.0);
    }
    let view = DesignView::from_column_major(&design, ROWS, COLS)?;
    let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Learner, 0xF01D);
    let start = Instant::now();
    let folds = assign_folds(ROWS, FOLDS, &mut rng, None)?;
    let fold_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let spec = NeuralSpec { hidden: 16, epochs: 20, learning_rate: 0.02 };
    let propensity = resolve_for(LearnerSpec::NeuralNet(spec), PredictionTask::BinaryProbability)?;
    let outcome_model = resolve_for(LearnerSpec::NeuralNet(spec), PredictionTask::Regression)?;

    let start = Instant::now();
    let p = cross_fit_with_folds(
        propensity.as_ref(),
        view,
        TargetView::new(&treatment),
        folds.clone(),
        &ctx,
        None,
    )?;
    let propensity_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let start = Instant::now();
    let y = cross_fit_with_folds(
        outcome_model.as_ref(),
        view,
        TargetView::new(&outcome),
        folds,
        &ctx,
        None,
    )?;
    let outcome_ms = start.elapsed().as_secs_f64() * 1_000.0;
    black_box((&p.predictions, &y.predictions));
    println!(
        "rows={ROWS} cols={COLS} folds={FOLDS} hidden=16 epochs=20 \
         fold_plan_ms={fold_ms:.3} propensity_fit_predict_ms={propensity_ms:.3} \
         outcome_fit_predict_ms={outcome_ms:.3} total_ms={:.3}",
        fold_ms + propensity_ms + outcome_ms,
    );
    println!(
        "backend=burn_ndarray_cpu gpu_transfer_ms=unmeasured \
         gpu_end_to_end_gain=unmeasured"
    );
    Ok(())
}
