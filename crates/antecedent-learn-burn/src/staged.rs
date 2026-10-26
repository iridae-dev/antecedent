//! Staged neural fit: the same arithmetic as [`super::train`], on any Burn autodiff backend,
//! with a wall clock around each stage the accelerator decision compares.
//!
//! Stages: prepare (validate, standardize, narrow to `f32`), upload (tensors placed on the
//! device and the device drained), fit (module build, forward, backward, and Adam step, with
//! the device drained at the end of each so queued kernels are billed to fit and not to the
//! next copy), synchronize (the per-epoch scalar loss copied onto the host), download (fitted
//! weights copied onto the host), predict (portable host forward on the held-out rows), and
//! output (held-out values written onto physical row slots as `f64`).
//!
//! [`run_workload`] drives a deterministic cross-fitted workload through those stages on one
//! backend so the same call times `NdArray<f32>`, `NdArray<f32>` with Accelerate, and
//! Burn's WGPU backend.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::hint::black_box;
use std::time::{Duration, Instant};

use burn::module::AutodiffModule;
use burn::nn::Linear;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};

use super::device::{EpochLoss, record_host_loss};
use super::{
    DenseLayer, LearnBurnError, Mlp, MlpSpec, TrainedMlp, colmajor_to_rowmajor_f32,
    column_standardization, epoch_loss, standardized, target_standardization, tensor2,
};

/// Names of the stages in report order.
pub const STAGE_NAMES: [&str; 7] =
    ["prepare", "upload", "fit", "synchronize", "download", "predict", "output"];

/// Wall clock of each stage. Every field is an [`Instant::elapsed`] of code that ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageTimings {
    /// Validate, standardize, and narrow the design and target.
    pub prepare: Duration,
    /// Place the training tensors on the device and drain it.
    pub upload: Duration,
    /// Module build, forward, backward, and optimizer step, drained after each.
    pub fit: Duration,
    /// Sum of the per-epoch scalar loss copies onto the host.
    pub synchronize: Duration,
    /// Copy the fitted weights onto the host.
    pub download: Duration,
    /// Portable host forward on the held-out rows.
    pub predict: Duration,
    /// Write held-out values onto physical row slots.
    pub output: Duration,
}

impl StageTimings {
    /// Sum of all stages.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.prepare
            + self.upload
            + self.fit
            + self.synchronize
            + self.download
            + self.predict
            + self.output
    }

    /// Stage clocks in nanoseconds, in [`STAGE_NAMES`] order.
    #[must_use]
    pub fn nanos(&self) -> [u128; 7] {
        [
            self.prepare.as_nanos(),
            self.upload.as_nanos(),
            self.fit.as_nanos(),
            self.synchronize.as_nanos(),
            self.download.as_nanos(),
            self.predict.as_nanos(),
            self.output.as_nanos(),
        ]
    }

    /// Stage-wise sum.
    #[must_use]
    pub fn plus(&self, other: &Self) -> Self {
        Self {
            prepare: self.prepare + other.prepare,
            upload: self.upload + other.upload,
            fit: self.fit + other.fit,
            synchronize: self.synchronize + other.synchronize,
            download: self.download + other.download,
            predict: self.predict + other.predict,
            output: self.output + other.output,
        }
    }
}

/// One staged fit: portable weights, host losses, and the clocks through download.
///
/// `predict` and `output` are zero here; [`run_workload`] times them.
#[derive(Clone, Debug)]
#[must_use = "the clocks are the measurement"]
pub struct StagedFit {
    /// Host weights. `Send + Sync`; inference needs no device.
    pub model: TrainedMlp,
    /// Clocks for prepare, upload, fit, synchronize, and download.
    pub timings: StageTimings,
    /// One host loss per epoch.
    pub losses: Vec<EpochLoss>,
}

/// Refuse with [`LearnBurnError::Cancelled`] when `stop` asks to cancel.
fn stopped(stop: &dyn Fn() -> bool) -> Result<(), LearnBurnError> {
    if stop() { Err(LearnBurnError::Cancelled) } else { Ok(()) }
}

/// Check a fit request the way [`super::train`] does.
pub(crate) fn validate_fit(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
) -> Result<(), LearnBurnError> {
    if nrows == 0 || ncols == 0 {
        return Err(LearnBurnError::EmptyDesign);
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) || y.len() != nrows {
        return Err(LearnBurnError::ShapeMismatch);
    }
    if spec.hidden == 0 || spec.epochs == 0 {
        return Err(LearnBurnError::Architecture);
    }
    if x_colmajor[..nrows * ncols].iter().chain(y).any(|v| !v.is_finite()) {
        return Err(LearnBurnError::NonFiniteInput);
    }
    #[allow(clippy::float_cmp, reason = "a binary target is exactly the labels 0 and 1")]
    let not_a_label = |v: f64| v != 0.0 && v != 1.0;
    if spec.binary && y.iter().any(|&v| not_a_label(v)) {
        return Err(LearnBurnError::BinaryLabels);
    }
    if !spec.learning_rate.is_finite() || spec.learning_rate <= 0.0 {
        return Err(LearnBurnError::LearningRate);
    }
    Ok(())
}

/// Fit on `device` of backend `B`, timing each stage.
///
/// The arithmetic, seeding, standardization, and finite-loss policy are those of
/// [`super::train`]. `stop` is polled before every epoch; a `true` returns
/// [`LearnBurnError::Cancelled`] with the device handle and tensors dropped.
///
/// # Errors
///
/// The input and divergence failures of [`super::train`], or cancellation.
#[must_use = "an error means no complete set of stage clocks"]
#[allow(clippy::too_many_lines, reason = "one linear pass over the timed stages")]
pub fn fit_staged<B: AutodiffBackend>(
    device: &B::Device,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
    stop: &dyn Fn() -> bool,
) -> Result<StagedFit, LearnBurnError> {
    let MlpSpec { binary, hidden, epochs, learning_rate, seed } = *spec;
    stopped(stop)?;

    let start = Instant::now();
    validate_fit(x_colmajor, nrows, ncols, y, spec)?;
    let (x_center, x_scale) = column_standardization(x_colmajor, nrows, ncols);
    let (y_center, y_scale) = if binary { (0.0, 1.0) } else { target_standardization(y) };
    let x_std = standardized(x_colmajor, nrows, ncols, &x_center, &x_scale);
    let x_rm = colmajor_to_rowmajor_f32(&x_std, nrows, ncols);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the network trains in f32, so narrowing the standardized target is the intended precision"
    )]
    let y_f: Vec<f32> = y.iter().map(|&v| ((v - y_center) / y_scale) as f32).collect();
    let prepare = start.elapsed();

    let start = Instant::now();
    let x_t = tensor2::<B, 2>(&x_rm, [nrows, ncols], device);
    let y_t = tensor2::<B, 2>(&y_f, [nrows, 1], device);
    B::sync(device);
    let upload = start.elapsed();

    let mut fit = Duration::ZERO;
    let mut synchronize = Duration::ZERO;
    let mut losses = Vec::with_capacity(epochs);

    let start = Instant::now();
    let mut model = Mlp::<B>::seeded(device, ncols, hidden, seed);
    let mut optim = AdamConfig::new().init::<B, Mlp<B>>();
    B::sync(device);
    fit += start.elapsed();

    for epoch in 0..epochs {
        stopped(stop)?;
        let start = Instant::now();
        let pred = black_box(model.forward(x_t.clone()));
        let loss = epoch_loss(pred, y_t.clone(), binary);
        B::sync(device);
        fit += start.elapsed();

        let start = Instant::now();
        let loss_value = black_box(
            loss.clone()
                .into_data()
                .as_slice::<f32>()
                .map_err(|e| LearnBurnError::Loss(format!("{e:?}")))?
                .first()
                .copied()
                .unwrap_or(f32::NAN),
        );
        synchronize += start.elapsed();
        losses.push(record_host_loss(epoch, loss_value)?);

        let start = Instant::now();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(learning_rate, model, grads);
        B::sync(device);
        fit += start.elapsed();
    }

    let start = Instant::now();
    let inner = model.valid();
    let trained = TrainedMlp {
        l1: extract_layer(&inner.l1)?,
        l2: extract_layer(&inner.l2)?,
        out: extract_layer(&inner.out)?,
        binary,
        ncols,
        x_center,
        x_scale,
        y_center,
        y_scale,
    };
    let finite = [&trained.l1, &trained.l2, &trained.out]
        .iter()
        .all(|layer| layer.weight.iter().chain(&layer.bias).all(|v| v.is_finite()));
    if !finite {
        return Err(LearnBurnError::NonFiniteWeights);
    }
    let download = start.elapsed();

    Ok(StagedFit {
        model: trained,
        timings: StageTimings {
            prepare,
            upload,
            fit,
            synchronize,
            download,
            predict: Duration::ZERO,
            output: Duration::ZERO,
        },
        losses,
    })
}

fn extract_layer<B: Backend>(layer: &Linear<B>) -> Result<DenseLayer, LearnBurnError> {
    let weight = layer.weight.val();
    let dims = weight.dims();
    let (din, dout) = (dims[0], dims[1]);
    let w = weight
        .into_data()
        .as_slice::<f32>()
        .map_err(|e| LearnBurnError::Weight(format!("{e:?}")))?
        .to_vec();
    let bias = layer.bias.as_ref().ok_or(LearnBurnError::MissingBias)?.val().into_data();
    let b = bias.as_slice::<f32>().map_err(|e| LearnBurnError::Bias(format!("{e:?}")))?.to_vec();
    if w.len() != din.saturating_mul(dout) || b.len() != dout {
        return Err(LearnBurnError::WeightShape);
    }
    Ok(DenseLayer { din, dout, weight: w, bias: b })
}

/// A deterministic cross-fitted regression workload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Workload {
    /// Name used in reports.
    pub name: &'static str,
    /// Physical rows.
    pub nrows: usize,
    /// Design columns, including the intercept column.
    pub ncols: usize,
    /// Cross-fit folds. Fold `f` predicts the rows with `row % folds == f`.
    pub folds: usize,
    /// Hidden width.
    pub hidden: usize,
    /// Adam epochs.
    pub epochs: usize,
    /// Adam step size.
    pub learning_rate: f64,
    /// Seed of fold 0. Fold `f` uses `seed + f`.
    pub seed: u64,
}

impl Workload {
    /// The small-fold shape: dominated by launch and copy overhead, so a CPU is expected to win.
    pub const SMALL: Self = Self {
        name: "small",
        nrows: 256,
        ncols: 4,
        folds: 2,
        hidden: 8,
        epochs: 8,
        learning_rate: 0.05,
        seed: 11,
    };
    /// The frozen CPU-control scale: 4000 rows, 8 columns, 3 folds, hidden 16, 20 epochs.
    pub const CONTROL: Self = Self {
        name: "control",
        nrows: 4_000,
        ncols: 8,
        folds: 3,
        hidden: 16,
        epochs: 20,
        learning_rate: 0.02,
        seed: 21,
    };
    /// A compute-bound shape where device throughput can pay for the copies.
    pub const LARGE: Self = Self {
        name: "large",
        nrows: 40_000,
        ncols: 32,
        folds: 2,
        hidden: 128,
        epochs: 15,
        learning_rate: 0.01,
        seed: 31,
    };

    /// Approximate floating-point operations of the whole workload (forward, backward, step).
    #[must_use]
    pub fn flops(&self) -> u64 {
        let train_rows = self.nrows - self.nrows / self.folds.max(1);
        fit_flops(train_rows, self.ncols, self.hidden, self.epochs)
            .saturating_mul(self.folds as u64)
    }

    /// The deterministic design (column-major, column 0 an intercept) and outcome.
    #[must_use]
    pub fn data(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.nrows;
        let mut x = vec![0.0; n * self.ncols];
        let mut y = vec![0.0; n];
        for row in 0..n {
            x[row] = 1.0;
            for col in 1..self.ncols {
                let g = (row.wrapping_mul(col + 3).wrapping_add(col * 17)) % 997;
                x[col * n + row] = (g as f64) / 997.0 - 0.5;
            }
            let a = if self.ncols > 1 { x[n + row] } else { 0.0 };
            let b = if self.ncols > 2 { x[2 * n + row] } else { 0.0 };
            y[row] = 1.5 + 2.0 * a - 1.2 * b + 0.8 * a * b + 0.1 * ((row * 19 % 31) as f64 / 31.0);
        }
        (x, y)
    }
}

/// Approximate floating-point operations of one fit: `6 * rows * epochs` times the
/// parameter count of the two-hidden-layer network.
#[must_use]
pub fn fit_flops(train_rows: usize, ncols: usize, hidden: usize, epochs: usize) -> u64 {
    let params = ncols
        .saturating_mul(hidden)
        .saturating_add(hidden.saturating_mul(hidden))
        .saturating_add(hidden);
    6u64.saturating_mul(train_rows as u64)
        .saturating_mul(epochs as u64)
        .saturating_mul(params as u64)
}

/// One fold of a [`WorkloadRun`].
#[derive(Clone, Debug)]
pub struct FoldRun {
    /// Fold index.
    pub fold: usize,
    /// Physical rows this fold predicted, ascending.
    pub predict_rows: Vec<usize>,
    /// Predictions for [`Self::predict_rows`], in that order.
    pub predictions: Vec<f64>,
    /// Stage clocks for this fold.
    pub timings: StageTimings,
    /// Host losses, one per epoch.
    pub losses: Vec<EpochLoss>,
}

/// A full cross-fit on one backend.
#[derive(Clone, Debug)]
#[must_use = "the run is the measurement"]
pub struct WorkloadRun {
    /// Per-fold results in fold order.
    pub folds: Vec<FoldRun>,
    /// Out-of-fold predictions by physical row.
    pub physical: Vec<f64>,
    /// Sum of fold clocks.
    pub timings: StageTimings,
    /// Wall clock of the whole run, including the gather between folds.
    pub wall: Duration,
}

/// Run `workload` on `device` of backend `B`, folds one after another.
///
/// # Errors
///
/// Any [`fit_staged`] failure, a fold with no rows, or cancellation.
pub fn run_workload<B: AutodiffBackend>(
    device: &B::Device,
    workload: &Workload,
    stop: &dyn Fn() -> bool,
) -> Result<WorkloadRun, LearnBurnError> {
    let Workload { nrows, ncols, folds, hidden, epochs, learning_rate, seed, .. } = *workload;
    if folds < 2 || nrows < folds * 2 {
        return Err(LearnBurnError::ShapeMismatch);
    }
    let wall_start = Instant::now();
    let (x, y) = workload.data();
    let mut physical = vec![0.0; nrows];
    let mut runs = Vec::with_capacity(folds);
    let mut total = StageTimings::default();
    for fold in 0..folds {
        let start = Instant::now();
        let train_rows: Vec<usize> = (0..nrows).filter(|r| r % folds != fold).collect();
        let predict_rows: Vec<usize> = (0..nrows).filter(|r| r % folds == fold).collect();
        let train_x = gather_colmajor(&x, nrows, ncols, &train_rows);
        let train_y: Vec<f64> = train_rows.iter().map(|&r| y[r]).collect();
        let predict_x = gather_colmajor(&x, nrows, ncols, &predict_rows);
        let gather = start.elapsed();

        let spec = MlpSpec {
            binary: false,
            hidden,
            epochs,
            learning_rate,
            seed: seed.wrapping_add(fold as u64),
        };
        let mut fitted =
            fit_staged::<B>(device, &train_x, train_rows.len(), ncols, &train_y, &spec, stop)?;
        fitted.timings.prepare += gather;

        let start = Instant::now();
        let mut predictions = vec![0.0; predict_rows.len()];
        fitted.model.predict(&predict_x, predict_rows.len(), ncols, &mut predictions)?;
        fitted.timings.predict = start.elapsed();

        let start = Instant::now();
        for (&row, &value) in predict_rows.iter().zip(&predictions) {
            physical[row] = value;
        }
        black_box(&physical);
        fitted.timings.output = start.elapsed();

        total = total.plus(&fitted.timings);
        runs.push(FoldRun {
            fold,
            predict_rows,
            predictions,
            timings: fitted.timings,
            losses: fitted.losses,
        });
    }
    Ok(WorkloadRun { folds: runs, physical, timings: total, wall: wall_start.elapsed() })
}

fn gather_colmajor(x: &[f64], nrows: usize, ncols: usize, rows: &[usize]) -> Vec<f64> {
    let m = rows.len();
    let mut out = vec![0.0; m * ncols];
    for col in 0..ncols {
        for (slot, &row) in rows.iter().enumerate() {
            out[col * m + slot] = x[col * nrows + row];
        }
    }
    out
}

/// Largest absolute gap between two prediction vectors, in units of the standard deviation
/// of `truth`. Infinite when the lengths differ or any value is non-finite.
#[must_use]
pub fn max_gap_in_target_sd(left: &[f64], right: &[f64], truth: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() || truth.is_empty() {
        return f64::INFINITY;
    }
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let sd = (truth.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let scale = if sd > 0.0 { sd } else { 1.0 };
    let mut worst = 0.0_f64;
    for (a, b) in left.iter().zip(right) {
        let gap = (a - b).abs() / scale;
        if !gap.is_finite() {
            return f64::INFINITY;
        }
        worst = worst.max(gap);
    }
    worst
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type Cpu = Autodiff<NdArray<f32>>;

    fn cpu_device() -> <NdArray<f32> as Backend>::Device {
        <NdArray<f32> as Backend>::Device::default()
    }

    #[test]
    fn staged_cpu_fit_replays_the_unstaged_fit_bit_for_bit() {
        let workload = Workload { nrows: 40, folds: 2, epochs: 6, hidden: 4, ..Workload::SMALL };
        let (x, y) = workload.data();
        let spec = MlpSpec { binary: false, hidden: 4, epochs: 6, learning_rate: 0.05, seed: 7 };
        let staged =
            fit_staged::<Cpu>(&cpu_device(), &x, 40, workload.ncols, &y, &spec, &|| false).unwrap();
        let plain = super::super::train(&x, 40, workload.ncols, &y, &spec).unwrap();
        let mut a = vec![0.0; 40];
        let mut b = vec![0.0; 40];
        staged.model.predict(&x, 40, workload.ncols, &mut a).unwrap();
        plain.predict(&x, 40, workload.ncols, &mut b).unwrap();
        for (left, right) in a.iter().zip(&b) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
        assert_eq!(staged.losses.len(), 6);
        assert!(staged.timings.fit > Duration::ZERO);
        assert!(staged.timings.prepare > Duration::ZERO);
    }

    #[test]
    fn workload_run_fills_every_physical_row_once_in_fold_order() {
        let workload = Workload { nrows: 60, folds: 3, epochs: 4, hidden: 4, ..Workload::SMALL };
        let run = run_workload::<Cpu>(&cpu_device(), &workload, &|| false).unwrap();
        assert_eq!(run.folds.len(), 3);
        let mut seen = vec![0u8; 60];
        for fold in &run.folds {
            assert!(fold.predict_rows.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(fold.predictions.len(), fold.predict_rows.len());
            assert_eq!(fold.losses.len(), 4);
            for (&row, &value) in fold.predict_rows.iter().zip(&fold.predictions) {
                seen[row] += 1;
                assert_eq!(run.physical[row].to_bits(), value.to_bits());
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
        assert!(run.timings.total() > Duration::ZERO);
        assert!(run.wall >= run.timings.fit);
        let again = run_workload::<Cpu>(&cpu_device(), &workload, &|| false).unwrap();
        for (left, right) in run.physical.iter().zip(&again.physical) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn cancellation_stops_between_epochs_and_leaves_the_backend_usable() {
        let workload = Workload { nrows: 40, folds: 2, epochs: 50, hidden: 4, ..Workload::SMALL };
        let (x, y) = workload.data();
        let spec = MlpSpec { binary: false, hidden: 4, epochs: 50, learning_rate: 0.05, seed: 3 };
        let polls = std::cell::Cell::new(0usize);
        let stop = || {
            polls.set(polls.get() + 1);
            polls.get() > 5
        };
        let error =
            fit_staged::<Cpu>(&cpu_device(), &x, 40, workload.ncols, &y, &spec, &stop).unwrap_err();
        assert!(matches!(error, LearnBurnError::Cancelled));
        assert!(polls.get() < 50, "cancellation must stop the epoch loop early");
        let ok = fit_staged::<Cpu>(&cpu_device(), &x, 40, workload.ncols, &y, &spec, &|| false);
        assert!(ok.is_ok());
    }

    #[test]
    fn staged_fit_refuses_the_inputs_train_refuses() {
        let spec = MlpSpec { binary: true, hidden: 4, epochs: 2, learning_rate: 0.05, seed: 1 };
        let x = vec![1.0; 8];
        let y = vec![0.0, 0.5, 1.0, 0.0];
        let error = fit_staged::<Cpu>(&cpu_device(), &x, 4, 2, &y, &spec, &|| false).unwrap_err();
        assert!(matches!(error, LearnBurnError::BinaryLabels));
    }

    #[test]
    fn gap_is_measured_in_target_sd_and_length_mismatch_is_infinite() {
        let truth = [0.0, 2.0, 4.0, 6.0];
        let sd = 5.0_f64.sqrt();
        let gap = max_gap_in_target_sd(&[0.0, 2.0], &[0.0, 2.0 + sd], &truth[..2]);
        assert!(gap.is_finite());
        assert!(max_gap_in_target_sd(&[1.0], &[1.0, 2.0], &truth).is_infinite());
        assert!(max_gap_in_target_sd(&[f64::NAN], &[1.0], &truth).is_infinite());
        let flops = fit_flops(10, 2, 3, 4);
        assert_eq!(flops, 6 * 10 * 4 * (2 * 3 + 9 + 3));
    }
}
