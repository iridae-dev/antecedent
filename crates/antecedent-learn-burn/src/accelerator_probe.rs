//! Timed stages for one small ndarray fold.
//!
//! [`probe_small_fold`] builds one held-out fold and times prepare, upload, fit, predict,
//! synchronize, download, and assemble on the default `NdArray<f32>` backend with
//! [`std::time::Instant`]. Each clock is the wall time of code that stage ran. Fit and
//! synchronize do not overlap: synchronize is the per-epoch host loss copy inside the
//! training loop. The bodies follow the ndarray steps of [`super::train`] (standardize,
//! tensor placement, Adam, host loss, weight copy, host forward, physical scatter). Those
//! steps are private and fused inside `train`, which returns no clocks.
//!
//! [`probe_cpu_workload`] and, with the `wgpu` feature, [`probe_wgpu_workload`] run any
//! [`Workload`](super::staged::Workload) through [`super::staged`], which times prepare,
//! upload, fit, synchronize, download, predict, and output on the named backend. Burn's
//! `metal` feature is Candle and is not enabled. On Apple, WGPU's graphics API is Metal.
//!
//! This is one small fold. It is not the frozen whole-workload control.
//!
//! Call [`probe_small_fold`] from one thread. This probe builds its layers with Burn's own
//! initializer, which draws from the process-global ndarray seed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::hint::black_box;
use std::time::{Duration, Instant};

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::relu;
use burn::tensor::{Tensor, TensorData, backend::Backend};

use super::{
    LearnBurnError, colmajor_to_rowmajor_f32, column_standardization, standardized,
    target_standardization,
};
use super::device::{self, LearnerBackend, LossSync};

type TrainBackend = Autodiff<NdArray<f32>>;

/// Rows in the probe design, including the held-out rows.
pub const FOLD_ROWS: usize = 32;

/// Columns in the probe design. Column 0 is an intercept.
pub const FOLD_COLS: usize = 4;

/// Leading rows used to fit. The rest are the held-out fold.
pub const TRAIN_ROWS: usize = 24;

/// Held-out rows. Assemble writes only these physical slots.
pub const PREDICT_ROWS: usize = FOLD_ROWS - TRAIN_ROWS;

/// Hidden width of the probe network.
pub const HIDDEN: usize = 8;

/// Adam epochs. At least one, so synchronize runs.
pub const EPOCHS: usize = 4;

/// Adam step size for this fold.
pub const LEARNING_RATE: f64 = 0.05;

/// `NdArray` seed applied at the start of fit, before the module is built.
pub const SEED: u64 = 11;

/// Comparison stages.
///
/// This is the report order. Execution order is prepare, upload, then fit with a
/// synchronize after each forward, then download, predict, and assemble.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStage {
    /// Host design, held-out split, and training-fold standardization.
    Prepare,
    /// Prepared train tensors placed on `NdArray`.
    Upload,
    /// Seed, module, forward, backward, and Adam step. Excludes host loss copies.
    Fit,
    /// Host forward on the held-out rows.
    Predict,
    /// Per-epoch copy of the scalar loss onto the host.
    Synchronize,
    /// Fitted weights copied onto the host.
    Download,
    /// Held-out predictions written onto physical row slots.
    Assemble,
}

/// Report order of [`ProbeStage`]. Not the execution order.
pub const STAGES: [ProbeStage; 7] = [
    ProbeStage::Prepare,
    ProbeStage::Upload,
    ProbeStage::Fit,
    ProbeStage::Predict,
    ProbeStage::Synchronize,
    ProbeStage::Download,
    ProbeStage::Assemble,
];

/// Exclusive wall clocks for one small ndarray fold.
///
/// Every field is [`Instant::elapsed`](std::time::Instant::elapsed) for a stage that ran.
/// `synchronize` is the sum of the per-epoch host loss copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageClocks {
    /// [`ProbeStage::Prepare`].
    pub prepare: Duration,
    /// [`ProbeStage::Upload`].
    pub upload: Duration,
    /// [`ProbeStage::Fit`].
    pub fit: Duration,
    /// [`ProbeStage::Predict`].
    pub predict: Duration,
    /// [`ProbeStage::Synchronize`].
    pub synchronize: Duration,
    /// [`ProbeStage::Download`].
    pub download: Duration,
    /// [`ProbeStage::Assemble`].
    pub assemble: Duration,
}

impl StageClocks {
    /// Clock for `stage`.
    #[must_use = "the clock is the stage that ran"]
    pub const fn elapsed(self, stage: ProbeStage) -> Duration {
        match stage {
            ProbeStage::Prepare => self.prepare,
            ProbeStage::Upload => self.upload,
            ProbeStage::Fit => self.fit,
            ProbeStage::Predict => self.predict,
            ProbeStage::Synchronize => self.synchronize,
            ProbeStage::Download => self.download,
            ProbeStage::Assemble => self.assemble,
        }
    }
}

/// One small fold timed on [`LearnerBackend::NdArrayCpu`].
#[derive(Clone, Debug)]
#[must_use = "stage clocks are the measurement"]
pub struct SmallFoldProbe {
    /// Always [`LearnerBackend::NdArrayCpu`].
    pub backend: LearnerBackend,
    /// Host loss policy applied inside synchronize.
    pub loss_sync: LossSync,
    /// Clocks for the seven stages that ran.
    pub clocks: StageClocks,
    /// Physical rows [`Self::assembled`] received. Length [`PREDICT_ROWS`].
    pub predict_rows: Vec<usize>,
    /// Length [`FOLD_ROWS`]. Slots outside [`Self::predict_rows`] stay `0.0`.
    pub assembled: Vec<f64>,
}

/// Time one small fold on the default `NdArray<f32>` backend.
///
/// # Errors
///
/// The ndarray fit can fail the same ways as [`super::train`]: a non-finite loss or
/// weights, or a Burn read of the loss or the weights.
#[must_use = "an error means no complete set of stage clocks"]
pub fn probe_small_fold() -> Result<SmallFoldProbe, LearnBurnError> {
    let (prepare, prepared) = measured_try(prepare_fold)?;
    let (upload, uploaded) = measured(|| upload_train(&prepared));
    let fitted = fit_train(&uploaded)?;
    let fit = fitted.fit;
    let synchronize = fitted.synchronize;
    let (download, host) = measured_try(|| download_weights(&fitted.model))?;
    let (predict, predictions) = measured_try(|| predict_held_out(&host, &prepared))?;
    let (assemble, assembled) =
        measured(|| assemble_physical(FOLD_ROWS, &prepared.predict_rows, &predictions));
    Ok(SmallFoldProbe {
        backend: LearnerBackend::NdArrayCpu,
        loss_sync: LossSync::HostEveryEpoch,
        clocks: StageClocks { prepare, upload, fit, predict, synchronize, download, assemble },
        predict_rows: prepared.predict_rows,
        assembled,
    })
}

/// Time `workload` on the default `NdArray<f32>` backend through the staged fit.
///
/// # Errors
///
/// Any [`super::staged::fit_staged`] failure.
#[must_use = "an error means no complete set of stage clocks"]
pub fn probe_cpu_workload(
    workload: &super::staged::Workload,
) -> Result<super::staged::WorkloadRun, LearnBurnError> {
    let device = <NdArray<f32> as Backend>::Device::default();
    super::staged::run_workload::<TrainBackend>(&device, workload, &|| false)
}

/// Time `workload` on Burn's WGPU backend (Metal on Apple) through the staged fit.
///
/// A device panic (no adapter, allocation failure) is returned as
/// [`LearnBurnError::Device`].
///
/// # Errors
///
/// Any [`super::staged::fit_staged`] failure, or a device panic.
#[cfg(feature = "wgpu")]
#[must_use = "an error means no complete set of stage clocks"]
pub fn probe_wgpu_workload(
    workload: &super::staged::Workload,
) -> Result<super::staged::WorkloadRun, LearnBurnError> {
    let device = burn::backend::wgpu::WgpuDevice::default();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::staged::run_workload::<device::WgpuTrain>(&device, workload, &|| false)
    }))
    .map_err(|payload| LearnBurnError::Device(device::panic_text(payload.as_ref())))?
}

fn measured<T>(work: impl FnOnce() -> T) -> (Duration, T) {
    let start = Instant::now();
    let value = black_box(work());
    (start.elapsed(), value)
}

fn measured_try<T>(
    work: impl FnOnce() -> Result<T, LearnBurnError>,
) -> Result<(Duration, T), LearnBurnError> {
    let start = Instant::now();
    let value = black_box(work()?);
    Ok((start.elapsed(), value))
}

struct PreparedFold {
    predict_rows: Vec<usize>,
    predict_x: Vec<f64>,
    x_center: Vec<f64>,
    x_scale: Vec<f64>,
    y_center: f64,
    y_scale: f64,
    x_rm: Vec<f32>,
    y_f: Vec<f32>,
}

struct Uploaded {
    x: Tensor<TrainBackend, 2>,
    y: Tensor<TrainBackend, 2>,
    device: <NdArray<f32> as Backend>::Device,
}

struct Fitted {
    model: FoldMlp<TrainBackend>,
    fit: Duration,
    synchronize: Duration,
}

struct HostLayer {
    din: usize,
    dout: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct HostNet {
    l1: HostLayer,
    l2: HostLayer,
    out: HostLayer,
}

fn prepare_fold() -> Result<PreparedFold, LearnBurnError> {
    let (x, y) = design();
    if x.iter().chain(&y).any(|value| !value.is_finite()) {
        return Err(LearnBurnError::NonFiniteInput);
    }
    let train_rows: Vec<usize> = (0..TRAIN_ROWS).collect();
    let predict_rows: Vec<usize> = (TRAIN_ROWS..FOLD_ROWS).collect();
    let train_x = gather_colmajor(&x, FOLD_ROWS, FOLD_COLS, &train_rows);
    let train_y = gather_target(&y, &train_rows);
    let predict_x = gather_colmajor(&x, FOLD_ROWS, FOLD_COLS, &predict_rows);
    let (x_center, x_scale) = column_standardization(&train_x, TRAIN_ROWS, FOLD_COLS);
    let (y_center, y_scale) = target_standardization(&train_y);
    let x_std = standardized(&train_x, TRAIN_ROWS, FOLD_COLS, &x_center, &x_scale);
    let x_rm = colmajor_to_rowmajor_f32(&x_std, TRAIN_ROWS, FOLD_COLS);
    let y_f = narrow_target(&train_y, y_center, y_scale);
    Ok(PreparedFold { predict_rows, predict_x, x_center, x_scale, y_center, y_scale, x_rm, y_f })
}

fn design() -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; FOLD_ROWS * FOLD_COLS];
    let mut y = vec![0.0; FOLD_ROWS];
    for row in 0..FOLD_ROWS {
        x[row] = 1.0;
        for col in 1..FOLD_COLS {
            let generated = (row * (col + 3) + col * 17) % 97;
            x[col * FOLD_ROWS + row] = (generated as f64) / 97.0 - 0.5;
        }
        y[row] = 1.5 + 0.25 * (row as f64) - 0.1 * x[FOLD_ROWS + row];
    }
    (x, y)
}

fn upload_train(prepared: &PreparedFold) -> Uploaded {
    let device = <NdArray<f32> as Backend>::Device::default();
    let x = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(prepared.x_rm.clone(), [TRAIN_ROWS, FOLD_COLS]),
        &device,
    );
    let y = Tensor::<TrainBackend, 2>::from_data(
        TensorData::new(prepared.y_f.clone(), [TRAIN_ROWS, 1]),
        &device,
    );
    Uploaded { x, y, device }
}

fn fit_train(uploaded: &Uploaded) -> Result<Fitted, LearnBurnError> {
    let mut fit = Duration::ZERO;
    let mut synchronize = Duration::ZERO;
    let start = Instant::now();
    <NdArray<f32> as Backend>::seed(SEED);
    let mut model = FoldMlp::<TrainBackend>::new(&uploaded.device, FOLD_COLS, HIDDEN);
    let mut optim = AdamConfig::new().init();
    fit += start.elapsed();
    for epoch in 0..EPOCHS {
        let start = Instant::now();
        let pred = black_box(model.forward(uploaded.x.clone()));
        let loss = black_box(pred.sub(uploaded.y.clone()).powf_scalar(2.0).mean());
        fit += start.elapsed();

        let start = Instant::now();
        let loss_value = black_box(copied_loss(loss.clone())?);
        let recorded = device::record_host_loss(epoch, loss_value);
        synchronize += start.elapsed();
        recorded?;

        let start = Instant::now();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(LEARNING_RATE, model, grads);
        fit += start.elapsed();
    }
    Ok(Fitted { model, fit, synchronize })
}

fn copied_loss<const D: usize, B: Backend>(loss: Tensor<B, D>) -> Result<f32, LearnBurnError> {
    Ok(loss
        .into_data()
        .as_slice::<f32>()
        .map_err(|error| LearnBurnError::Loss(format!("{error:?}")))?
        .first()
        .copied()
        .unwrap_or(f32::NAN))
}

fn download_weights(model: &FoldMlp<TrainBackend>) -> Result<HostNet, LearnBurnError> {
    let model = model.valid();
    let net =
        HostNet { l1: extract(&model.l1)?, l2: extract(&model.l2)?, out: extract(&model.out)? };
    if !weights_are_finite(&net) {
        return Err(LearnBurnError::NonFiniteWeights);
    }
    Ok(net)
}

fn predict_held_out(net: &HostNet, prepared: &PreparedFold) -> Result<Vec<f64>, LearnBurnError> {
    let nrows = prepared.predict_rows.len();
    let shaped = net.l1.din == FOLD_COLS
        && net.l2.din == net.l1.dout
        && net.out.din == net.l2.dout
        && net.out.dout == 1;
    if !shaped {
        return Err(LearnBurnError::PredictShape);
    }
    if prepared.predict_x.len() < nrows.saturating_mul(FOLD_COLS) {
        return Err(LearnBurnError::PredictBuffer);
    }
    let x_std =
        standardized(&prepared.predict_x, nrows, FOLD_COLS, &prepared.x_center, &prepared.x_scale);
    let x_rm = colmajor_to_rowmajor_f32(&x_std, nrows, FOLD_COLS);
    let mut h1 = vec![0.0f32; net.l1.dout];
    let mut h2 = vec![0.0f32; net.l2.dout];
    let mut ybuf = vec![0.0f32; net.out.dout];
    let mut out = vec![0.0; nrows];
    for (row, slot) in out.iter_mut().enumerate() {
        let start = row * FOLD_COLS;
        net.l1.forward_into(&x_rm[start..start + FOLD_COLS], &mut h1);
        relu_inplace(&mut h1);
        net.l2.forward_into(&h1, &mut h2);
        relu_inplace(&mut h2);
        net.out.forward_into(&h2, &mut ybuf);
        *slot = f64::from(ybuf[0]) * prepared.y_scale + prepared.y_center;
    }
    Ok(out)
}

fn assemble_physical(nrows: usize, predict_rows: &[usize], predictions: &[f64]) -> Vec<f64> {
    let mut physical = vec![0.0; nrows];
    for (&row, value) in predict_rows.iter().zip(predictions) {
        if let Some(slot) = physical.get_mut(row) {
            *slot = *value;
        }
    }
    physical
}

fn weights_are_finite(net: &HostNet) -> bool {
    [&net.l1, &net.l2, &net.out]
        .iter()
        .all(|layer| layer.weight.iter().chain(&layer.bias).all(|value| value.is_finite()))
}

fn extract<B: Backend>(layer: &Linear<B>) -> Result<HostLayer, LearnBurnError> {
    let weight = layer.weight.val();
    let dims = weight.dims();
    let din = dims[0];
    let dout = dims[1];
    let copied = weight
        .into_data()
        .as_slice::<f32>()
        .map_err(|error| LearnBurnError::Weight(format!("{error:?}")))?
        .to_vec();
    let bias = layer.bias.as_ref().ok_or(LearnBurnError::MissingBias)?.val().into_data();
    let bias = bias
        .as_slice::<f32>()
        .map_err(|error| LearnBurnError::Bias(format!("{error:?}")))?
        .to_vec();
    if copied.len() != din.saturating_mul(dout) || bias.len() != dout {
        return Err(LearnBurnError::WeightShape);
    }
    Ok(HostLayer { din, dout, weight: copied, bias })
}

impl HostLayer {
    #[allow(clippy::needless_range_loop, reason = "the subscript is the matrix coordinate")]
    fn forward_into(&self, x: &[f32], out: &mut [f32]) {
        out[..self.dout].copy_from_slice(&self.bias);
        for i in 0..self.din {
            let xi = x[i];
            let row = &self.weight[i * self.dout..(i + 1) * self.dout];
            for o in 0..self.dout {
                out[o] += xi * row[o];
            }
        }
    }
}

fn relu_inplace(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "f32 is the training dtype, so this narrowing is the intended precision"
)]
const fn narrow_f32(value: f64) -> f32 {
    value as f32
}

fn narrow_target(y: &[f64], y_center: f64, y_scale: f64) -> Vec<f32> {
    y.iter().map(|value| narrow_f32((value - y_center) / y_scale)).collect()
}

fn gather_colmajor(x: &[f64], nrows: usize, ncols: usize, rows: &[usize]) -> Vec<f64> {
    let gathered = rows.len();
    let mut out = vec![0.0; gathered.saturating_mul(ncols)];
    for col in 0..ncols {
        for (slot, &row) in rows.iter().enumerate() {
            out[col * gathered + slot] = x[col * nrows + row];
        }
    }
    out
}

fn gather_target(y: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&row| y[row]).collect()
}

#[derive(Module, Debug)]
struct FoldMlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> FoldMlp<B> {
    fn new(device: &B::Device, din: usize, hidden: usize) -> Self {
        Self {
            l1: LinearConfig::new(din, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            out: LinearConfig::new(hidden, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(x));
        let x = relu(self.l2.forward(x));
        self.out.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::{ProbeStage, STAGES, StageClocks};

    const fn zero_clock(stage: ProbeStage) -> std::time::Duration {
        StageClocks {
            prepare: std::time::Duration::ZERO,
            upload: std::time::Duration::ZERO,
            fit: std::time::Duration::ZERO,
            predict: std::time::Duration::ZERO,
            synchronize: std::time::Duration::ZERO,
            download: std::time::Duration::ZERO,
            assemble: std::time::Duration::ZERO,
        }
        .elapsed(stage)
    }

    #[test]
    fn stages_are_the_seven_comparison_points() {
        assert_eq!(
            STAGES,
            [
                ProbeStage::Prepare,
                ProbeStage::Upload,
                ProbeStage::Fit,
                ProbeStage::Predict,
                ProbeStage::Synchronize,
                ProbeStage::Download,
                ProbeStage::Assemble,
            ]
        );
        assert_eq!(zero_clock(ProbeStage::Prepare), std::time::Duration::ZERO);
        assert_eq!(zero_clock(ProbeStage::Assemble), std::time::Duration::ZERO);
    }

    #[test]
    fn small_fold_times_every_stage_on_the_ndarray_backend() {
        let probe = super::probe_small_fold().expect("ndarray fold");
        assert_eq!(probe.backend, crate::device::LearnerBackend::NdArrayCpu);
        assert_eq!(probe.predict_rows.len(), super::PREDICT_ROWS);
        assert_eq!(probe.assembled.len(), super::FOLD_ROWS);
        assert!(probe.assembled.iter().all(|value| value.is_finite()));
        for stage in STAGES {
            let _clock = probe.clocks.elapsed(stage);
        }
    }

    #[test]
    fn cpu_workload_probe_times_all_stages_and_fills_every_row() {
        let workload = crate::staged::Workload {
            nrows: 48,
            folds: 2,
            epochs: 3,
            hidden: 4,
            ..crate::staged::Workload::SMALL
        };
        let run = super::probe_cpu_workload(&workload).expect("ndarray workload");
        assert_eq!(run.physical.len(), 48);
        assert!(run.physical.iter().all(|value| value.is_finite()));
        assert!(run.timings.total() > std::time::Duration::ZERO);
        assert!(run.folds.iter().all(|fold| fold.losses.len() == 3));
    }
}
