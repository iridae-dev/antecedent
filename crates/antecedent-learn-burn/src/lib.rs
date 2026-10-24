//! Burn-backed MLP. Hidden behind `antecedent-learn`'s `neural_net` spec.
//!
//! The public surface is column-major `f64` in / `f64` out. The default build trains and
//! predicts on the CPU `NdArray<f32>` backend. The opt-in `wgpu` feature adds Burn's WGPU
//! device backend, selected by [`policy`], and `accelerate` adds Apple Accelerate BLAS under
//! the same ndarray backend. [`staged`] times each stage of either.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Tensor, TensorData, backend::Backend};

pub mod accelerator_probe;
pub mod admission;
pub mod device;
pub mod policy;
pub mod staged;

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

/// Burn's backend RNG is global: training holds this lock so concurrent folds cannot
/// reseed each other.
static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Trained two-hidden-layer MLP. Predicts one value per row.
///
/// Weights are copied out of Burn so the fitted artifact is `Send + Sync`.
#[derive(Clone, Debug)]
pub struct TrainedMlp {
    l1: DenseLayer,
    l2: DenseLayer,
    out: DenseLayer,
    binary: bool,
    ncols: usize,
    /// Per-column centre and scale applied to inputs (constant columns are untouched).
    x_center: Vec<f64>,
    x_scale: Vec<f64>,
    /// Centre and scale of a regression target (`0`, `1` for a binary target).
    y_center: f64,
    y_scale: f64,
}

#[derive(Clone, Debug)]
struct DenseLayer {
    din: usize,
    dout: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// Failure of [`train`] or [`TrainedMlp::predict`].
///
/// Display text matches the strings this crate used to return as `Result<_, String>`.
#[derive(Debug, Clone)]
pub enum LearnBurnError {
    /// No rows or no columns.
    EmptyDesign,
    /// Design length or target length does not match the declared shape.
    ShapeMismatch,
    /// Hidden width or epoch count is zero.
    Architecture,
    /// A design or target value is NaN or infinite.
    NonFiniteInput,
    /// A binary target is not exactly 0 or 1.
    BinaryLabels,
    /// Learning rate is non-finite or not positive.
    LearningRate,
    /// Another thread panicked while holding the training lock.
    LockPoisoned,
    /// Burn could not read the scalar loss.
    Loss(String),
    /// Training loss left the finite range.
    Diverged(f32),
    /// Fitted weights are NaN or infinite.
    NonFiniteWeights,
    /// Predict output or column count does not match the fit.
    PredictShape,
    /// Predict design buffer is shorter than `nrows * ncols`.
    PredictBuffer,
    /// Burn could not read a weight tensor.
    Weight(String),
    /// A linear layer has no bias.
    MissingBias,
    /// Burn could not read a bias tensor.
    Bias(String),
    /// Extracted weight or bias length does not match the layer.
    WeightShape,
    /// Cancellation was observed between epochs. No model was produced.
    Cancelled,
    /// The device runtime failed (allocation, adapter, or kernel). No model was produced.
    Device(String),
}

impl std::fmt::Display for LearnBurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDesign => f.write_str("neural_net needs a non-empty design"),
            Self::ShapeMismatch => f.write_str("neural_net design / target shape mismatch"),
            Self::Architecture => f.write_str("neural_net hidden and epochs must be ≥ 1"),
            Self::NonFiniteInput => f.write_str("neural_net design and target must be finite"),
            Self::BinaryLabels => f.write_str("neural_net binary target must be 0/1"),
            Self::LearningRate => {
                f.write_str("neural_net learning_rate must be finite and positive")
            }
            Self::LockPoisoned => f.write_str("neural_net training lock poisoned"),
            Self::Loss(detail) => write!(f, "neural_net loss: {detail}"),
            Self::Diverged(loss) => write!(f, "neural_net training diverged (loss {loss})"),
            Self::NonFiniteWeights => f.write_str("neural_net produced non-finite weights"),
            Self::PredictShape => f.write_str("neural_net predict shape mismatch"),
            Self::PredictBuffer => f.write_str("neural_net predict buffer shorter than design"),
            Self::Weight(detail) => write!(f, "neural_net weight: {detail}"),
            Self::MissingBias => f.write_str("neural_net layer missing bias"),
            Self::Bias(detail) => write!(f, "neural_net bias: {detail}"),
            Self::WeightShape => f.write_str("neural_net extracted weight shape mismatch"),
            Self::Cancelled => f.write_str("neural_net fit cancelled"),
            Self::Device(detail) => write!(f, "neural_net device: {detail}"),
        }
    }
}

impl std::error::Error for LearnBurnError {}

/// Task, architecture and optimizer settings of [`train`].
#[derive(Debug, Clone, Copy)]
pub struct MlpSpec {
    /// Binary-probability target (labels 0/1) rather than regression.
    pub binary: bool,
    /// Hidden units.
    pub hidden: usize,
    /// Training epochs.
    pub epochs: usize,
    /// Optimizer learning rate.
    pub learning_rate: f64,
    /// Initialization / stochastic-operation seed.
    pub seed: u64,
}

/// Fit a small MLP on a column-major design.
///
/// # Errors
///
/// Shape mismatch, empty design, or a Burn backend failure.
pub fn train(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
) -> Result<TrainedMlp, LearnBurnError> {
    let MlpSpec { binary, hidden, epochs, learning_rate, seed } = *spec;
    if nrows == 0 || ncols == 0 {
        return Err(LearnBurnError::EmptyDesign);
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) || y.len() != nrows {
        return Err(LearnBurnError::ShapeMismatch);
    }
    if hidden == 0 || epochs == 0 {
        return Err(LearnBurnError::Architecture);
    }
    if x_colmajor[..nrows * ncols].iter().chain(y).any(|v| !v.is_finite()) {
        return Err(LearnBurnError::NonFiniteInput);
    }
    #[allow(clippy::float_cmp, reason = "a binary target is exactly the labels 0 and 1")]
    let not_a_label = |v: f64| v != 0.0 && v != 1.0;
    if binary && y.iter().any(|&v| not_a_label(v)) {
        return Err(LearnBurnError::BinaryLabels);
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(LearnBurnError::LearningRate);
    }
    // Burn's backend RNG is global. Hold the lock through training so concurrent
    // folds cannot reseed each other's initialization or stochastic operations.
    let _guard = TRAIN_LOCK.lock().map_err(|_| LearnBurnError::LockPoisoned)?;
    InferBackend::seed(seed);
    let device = <InferBackend as Backend>::Device::default();
    // Standardize inputs and (for regression) the target before training. Raw f32 inputs
    // of scale 1e5 with a fixed Adam step saturate or overflow the network; the fitted
    // weights live on the standardized scale and `predict` applies the same transform.
    let (x_center, x_scale) = column_standardization(x_colmajor, nrows, ncols);
    let (y_center, y_scale) = if binary { (0.0, 1.0) } else { target_standardization(y) };
    let x_std = standardized(x_colmajor, nrows, ncols, &x_center, &x_scale);
    let x_rm = colmajor_to_rowmajor_f32(&x_std, nrows, ncols);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the network trains in f32, so narrowing the standardized target is the intended precision"
    )]
    let y_f: Vec<f32> = y.iter().map(|&v| ((v - y_center) / y_scale) as f32).collect();
    let mut model = Mlp::<TrainBackend>::new(&device, ncols, hidden);
    let mut optim = AdamConfig::new().init();
    let x_t = tensor2(&x_rm, [nrows, ncols], &device);
    let y_t = tensor2(&y_f, [nrows, 1], &device);
    let lr = learning_rate;
    for _ in 0..epochs {
        let pred = model.forward(x_t.clone());
        let loss = if binary {
            let p = sigmoid(pred.clone()).clamp(1e-6, 1.0 - 1e-6);
            let ones = p.ones_like();
            y_t.clone()
                .mul(p.clone().log())
                .add(ones.clone().sub(y_t.clone()).mul(ones.sub(p).log()))
                .mean()
                .mul_scalar(-1.0)
        } else {
            pred.sub(y_t.clone()).powf_scalar(2.0).mean()
        };
        // A diverged fit yields NaN/inf weights that `clamp` would carry into an `Ok` model;
        // stop at the first non-finite loss instead of returning it as a nuisance.
        let loss_value = loss
            .clone()
            .into_data()
            .as_slice::<f32>()
            .map_err(|e| LearnBurnError::Loss(format!("{e:?}")))?
            .first()
            .copied()
            .unwrap_or(f32::NAN);
        if !loss_value.is_finite() {
            return Err(LearnBurnError::Diverged(loss_value));
        }
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }
    let model = model.valid();
    let trained = TrainedMlp {
        l1: extract_linear(&model.l1)?,
        l2: extract_linear(&model.l2)?,
        out: extract_linear(&model.out)?,
        binary,
        ncols,
        x_center,
        x_scale,
        y_center,
        y_scale,
    };
    let all_finite = [&trained.l1, &trained.l2, &trained.out]
        .iter()
        .all(|l| l.weight.iter().chain(&l.bias).all(|v| v.is_finite()));
    if !all_finite {
        return Err(LearnBurnError::NonFiniteWeights);
    }
    Ok(trained)
}

impl TrainedMlp {
    /// Write one prediction per row into `out`.
    ///
    /// # Errors
    ///
    /// Shape mismatch.
    pub fn predict(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        out: &mut [f64],
    ) -> Result<(), LearnBurnError> {
        if ncols != self.ncols || out.len() != nrows {
            return Err(LearnBurnError::PredictShape);
        }
        if x_colmajor.len() < nrows.saturating_mul(ncols) {
            return Err(LearnBurnError::PredictBuffer);
        }
        if nrows == 0 {
            return Ok(());
        }
        let x_std = standardized(x_colmajor, nrows, ncols, &self.x_center, &self.x_scale);
        let x_rm = colmajor_to_rowmajor_f32(&x_std, nrows, ncols);
        let mut h1 = vec![0.0f32; self.l1.dout];
        let mut h2 = vec![0.0f32; self.l2.dout];
        let mut ybuf = vec![0.0f32; self.out.dout];
        for r in 0..nrows {
            let row = &x_rm[r * ncols..(r + 1) * ncols];
            self.l1.forward_into(row, &mut h1);
            relu_inplace(&mut h1);
            self.l2.forward_into(&h1, &mut h2);
            relu_inplace(&mut h2);
            self.out.forward_into(&h2, &mut ybuf);
            let mut yhat = ybuf[0];
            if self.binary {
                // Deliberately an f32 logistic: this optional provider has no antecedent
                // dependency, and its clip (1e-6) is the f32 resolution of the network.
                yhat = (1.0 / (1.0 + (-yhat).exp())).clamp(1e-6, 1.0 - 1e-6);
                out[r] = f64::from(yhat);
            } else {
                out[r] = f64::from(yhat) * self.y_scale + self.y_center;
            }
        }
        Ok(())
    }
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    out: Linear<B>,
}

impl<B: Backend> Mlp<B> {
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

impl DenseLayer {
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
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
    for v in values {
        *v = v.max(0.0);
    }
}

fn extract_linear(layer: &Linear<InferBackend>) -> Result<DenseLayer, LearnBurnError> {
    let weight = layer.weight.val();
    let dims = weight.dims();
    let din = dims[0];
    let dout = dims[1];
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

/// Centre and scale of each design column. A constant column (an intercept) keeps centre
/// `0` and scale `1` so its meaning survives standardization.
fn column_standardization(x: &[f64], nrows: usize, ncols: usize) -> (Vec<f64>, Vec<f64>) {
    let n = nrows as f64;
    let mut centers = vec![0.0; ncols];
    let mut scales = vec![1.0; ncols];
    for c in 0..ncols {
        let col = &x[c * nrows..(c + 1) * nrows];
        let mean = col.iter().sum::<f64>() / n;
        let sd = (col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
        if sd > 1e-12 * mean.abs().max(1.0) {
            centers[c] = mean;
            scales[c] = sd;
        }
    }
    (centers, scales)
}

fn target_standardization(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sd = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    if sd > 1e-12 * mean.abs().max(1.0) { (mean, sd) } else { (mean, 1.0) }
}

fn standardized(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    centers: &[f64],
    scales: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; nrows * ncols];
    for c in 0..ncols {
        for r in 0..nrows {
            out[c * nrows + r] = (x[c * nrows + r] - centers[c]) / scales[c];
        }
    }
    out
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the network consumes f32 inputs, so narrowing the standardized features is the intended precision"
)]
fn colmajor_to_rowmajor_f32(x: &[f64], nrows: usize, ncols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nrows.saturating_mul(ncols)];
    for r in 0..nrows {
        for c in 0..ncols {
            out[r * ncols + c] = x[c * nrows + r] as f32;
        }
    }
    out
}

fn tensor2<B: Backend, const D: usize>(
    values: &[f32],
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, D>::from_data(TensorData::new(values.to_vec(), shape), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize, offset: f64, slope: f64) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = offset + slope * (i as f64);
        }
        (x, y)
    }

    /// Mean absolute error relative to the target's SD.
    fn relative_mae(pred: &[f64], y: &[f64]) -> f64 {
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        pred.iter().zip(y).map(|(p, t)| (p - t).abs()).sum::<f64>() / n / sd
    }

    #[test]
    fn fits_a_line() {
        let n = 24usize;
        let (x, y) = line(n, 2.0, 0.5);
        let model = train(
            &x,
            n,
            2,
            &y,
            &MlpSpec { binary: false, hidden: 8, epochs: 80, learning_rate: 0.05, seed: 1 },
        )
        .unwrap();
        let mut out = vec![0.0; n];
        model.predict(&x, n, 2, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out[n - 1] > out[0]);
        let err = relative_mae(&out, &y);
        assert!(err < 0.3, "mean absolute error is {err} target SDs");
    }

    /// A target of scale 1e5 used to saturate the raw f32 network (predictions near 0 or
    /// NaN); standardization makes the fit scale-free.
    #[test]
    fn large_scale_targets_are_fit_on_their_own_scale() {
        let n = 24usize;
        let (x, y) = line(n, 1.0e5, 5.0e4);
        let model = train(
            &x,
            n,
            2,
            &y,
            &MlpSpec { binary: false, hidden: 8, epochs: 80, learning_rate: 0.05, seed: 1 },
        )
        .unwrap();
        let mut out = vec![0.0; n];
        model.predict(&x, n, 2, &mut out).unwrap();
        let err = relative_mae(&out, &y);
        assert!(err < 0.3, "mean absolute error is {err} target SDs");
    }

    #[test]
    fn diverged_or_invalid_training_is_refused() {
        let n = 24usize;
        let (x, y) = line(n, 2.0, 0.5);
        // An absurd step size overflows the loss to inf/NaN within a few epochs.
        assert!(
            train(
                &x,
                n,
                2,
                &y,
                &MlpSpec { binary: false, hidden: 8, epochs: 40, learning_rate: 1.0e30, seed: 1 }
            )
            .is_err()
        );
        // Binary training on a non-0/1 target.
        assert!(
            train(
                &x,
                n,
                2,
                &y,
                &MlpSpec { binary: true, hidden: 8, epochs: 5, learning_rate: 0.05, seed: 1 }
            )
            .is_err()
        );
        // Non-finite design.
        let mut bad = x.clone();
        bad[n + 3] = f64::NAN;
        assert!(
            train(
                &bad,
                n,
                2,
                &y,
                &MlpSpec { binary: false, hidden: 8, epochs: 5, learning_rate: 0.05, seed: 1 }
            )
            .is_err()
        );
    }
}
