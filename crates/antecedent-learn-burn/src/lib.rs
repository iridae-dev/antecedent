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
use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::{log_sigmoid, relu};
use burn::tensor::{Tensor, TensorData, backend::Backend};

pub mod accelerator_probe;
pub mod admission;
pub mod device;
pub mod policy;
pub mod staged;

type TrainBackend = Autodiff<NdArray<f32>>;

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
    train_cancellable(x_colmajor, nrows, ncols, y, spec, &|| false)
}

/// [`train`] that polls `stop` before every epoch and returns
/// [`LearnBurnError::Cancelled`] once it reports true.
///
/// # Errors
///
/// As [`train`], or cancellation.
pub fn train_cancellable(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
    stop: &dyn Fn() -> bool,
) -> Result<TrainedMlp, LearnBurnError> {
    // Every fit, on any backend, goes through the one staged implementation. The initial
    // weights are drawn on the host from `spec.seed` and no training step is stochastic, so
    // concurrent fits share no state and need no process-wide lock.
    let device = <TrainBackend as Backend>::Device::default();
    staged::fit_staged::<TrainBackend>(&device, x_colmajor, nrows, ncols, y, spec, stop)
        .map(|fit| fit.model)
}

/// Mean training loss of `pred` against `y` (both `rows x 1`) on the standardized scale.
///
/// A binary target uses the cross-entropy of the *logits*,
/// `-mean(y ln σ(z) + (1 - y) ln σ(-z))`, evaluated with Burn's stable log-sigmoid: a
/// saturated prediction keeps a nonzero gradient and a nonzero loss, which clamping a
/// probability away from 0 and 1 and taking its logarithm would flatten. A regression target
/// uses the mean squared error.
fn epoch_loss<B: Backend>(pred: Tensor<B, 2>, y: Tensor<B, 2>, binary: bool) -> Tensor<B, 1> {
    if binary {
        let log_p = log_sigmoid(pred.clone());
        let log_q = log_sigmoid(pred.neg());
        let complement = y.clone().neg().add_scalar(1.0);
        y.mul(log_p).add(complement.mul(log_q)).mean().neg()
    } else {
        pred.sub(y).powf_scalar(2.0).mean()
    }
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
    /// Initial weights drawn on the host from `seed`, so every backend starts from the same
    /// values. Backend RNGs differ between implementations; this does not depend on them.
    fn seeded(device: &B::Device, din: usize, hidden: usize, seed: u64) -> Self {
        let mut rng = HostRng(seed);
        Self {
            l1: seeded_linear(device, din, hidden, &mut rng),
            l2: seeded_linear(device, hidden, hidden, &mut rng),
            out: seeded_linear(device, hidden, 1, &mut rng),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = relu(self.l1.forward(x));
        let x = relu(self.l2.forward(x));
        self.out.forward(x)
    }
}

/// `SplitMix64`: a small deterministic generator whose stream is the same on every host.
struct HostRng(u64);

impl HostRng {
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        #[allow(clippy::cast_precision_loss, reason = "53 high bits fill the f64 mantissa exactly")]
        let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
        unit
    }
}

/// A linear layer with weight and bias uniform on `(-1/sqrt(din), 1/sqrt(din))`, the
/// distribution Burn's default initializer uses, drawn on the host.
fn seeded_linear<B: Backend>(
    device: &B::Device,
    din: usize,
    dout: usize,
    rng: &mut HostRng,
) -> Linear<B> {
    #[allow(clippy::cast_precision_loss, reason = "layer widths are small")]
    let bound = 1.0 / (din as f64).sqrt();
    #[allow(clippy::cast_possible_truncation, reason = "the network trains in f32")]
    let mut draw = |n: usize| -> Vec<f32> {
        (0..n).map(|_| ((rng.next_unit() * 2.0 - 1.0) * bound) as f32).collect()
    };
    let weight = draw(din * dout);
    let bias = draw(dout);
    Linear {
        weight: burn::module::Param::from_tensor(tensor2::<B, 2>(&weight, [din, dout], device)),
        bias: Some(burn::module::Param::from_tensor(tensor2::<B, 1>(&bias, [dout], device))),
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

    /// Fits share no state: the same request trained concurrently and sequentially, on the
    /// public route and on the staged route, produces bit-identical predictions.
    #[test]
    fn concurrent_fits_are_bit_identical_to_sequential_fits() {
        let (x, y) = line(20, 2.0, 0.5);
        let spec = MlpSpec { binary: false, hidden: 4, epochs: 2, learning_rate: 0.05, seed: 17 };
        let predict = |model: &TrainedMlp| {
            let mut out = vec![0.0; 20];
            model.predict(&x, 20, 2, &mut out).unwrap();
            out
        };
        let reference = predict(&train(&x, 20, 2, &y, &spec).unwrap());
        let device = <TrainBackend as Backend>::Device::default();
        let staged =
            staged::fit_staged::<TrainBackend>(&device, &x, 20, 2, &y, &spec, &|| false).unwrap();
        assert_eq!(bits(&reference), bits(&predict(&staged.model)));
        let concurrent: Vec<Vec<f64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| predict(&train(&x, 20, 2, &y, &spec).unwrap())))
                .collect();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect()
        });
        for out in &concurrent {
            assert_eq!(bits(&reference), bits(out));
        }
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// A saturated binary prediction keeps a finite loss and a nonzero gradient.
    #[test]
    fn binary_loss_stays_informative_when_saturated() {
        let device = <TrainBackend as Backend>::Device::default();
        let z = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(vec![40.0f32, -40.0], [2, 1]),
            &device,
        )
        .require_grad();
        let y = Tensor::<TrainBackend, 2>::from_data(TensorData::new(vec![0.0f32, 1.0], [2, 1]), &device);
        let loss = epoch_loss(z.clone(), y, true);
        let value = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        assert!(value.is_finite() && value > 30.0, "loss {value}");
        let grads = loss.backward();
        let g = z.grad(&grads).unwrap().into_data();
        assert!(g.as_slice::<f32>().unwrap().iter().all(|v| v.abs() > 0.1));
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
