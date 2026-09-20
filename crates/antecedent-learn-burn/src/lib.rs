//! Burn-backed MLP. Hidden behind `antecedent-learn`'s `neural_net` spec.
//!
//! The public surface is column-major `f64` in / `f64` out. GPU/WGPU is not
//! enabled here; `NdArray` is the portable default.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Tensor, TensorData, backend::Backend};

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

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
}

#[derive(Clone, Debug)]
struct DenseLayer {
    din: usize,
    dout: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
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
    binary: bool,
    hidden: usize,
    epochs: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<TrainedMlp, String> {
    if nrows == 0 || ncols == 0 {
        return Err("neural_net needs a non-empty design".into());
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) || y.len() != nrows {
        return Err("neural_net design / target shape mismatch".into());
    }
    if hidden == 0 || epochs == 0 {
        return Err("neural_net hidden and epochs must be ≥ 1".into());
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("neural_net learning_rate must be finite and positive".into());
    }
    // Burn's backend RNG is global. Hold the lock through training so concurrent
    // folds cannot reseed each other's initialization or stochastic operations.
    static TRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = TRAIN_LOCK.lock().map_err(|_| "neural_net training lock poisoned")?;
    InferBackend::seed(seed);
    let device = <InferBackend as Backend>::Device::default();
    let x_rm = colmajor_to_rowmajor_f32(x_colmajor, nrows, ncols);
    let y_f: Vec<f32> = y.iter().map(|&v| v as f32).collect();
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
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }
    let model = model.valid();
    Ok(TrainedMlp {
        l1: extract_linear(&model.l1)?,
        l2: extract_linear(&model.l2)?,
        out: extract_linear(&model.out)?,
        binary,
        ncols,
    })
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
    ) -> Result<(), String> {
        if ncols != self.ncols || out.len() != nrows {
            return Err("neural_net predict shape mismatch".into());
        }
        if x_colmajor.len() < nrows.saturating_mul(ncols) {
            return Err("neural_net predict buffer shorter than design".into());
        }
        if nrows == 0 {
            return Ok(());
        }
        let x_rm = colmajor_to_rowmajor_f32(x_colmajor, nrows, ncols);
        for r in 0..nrows {
            let row = &x_rm[r * ncols..(r + 1) * ncols];
            let h1 = self.l1.forward_relu(row);
            let h2 = self.l2.forward_relu(&h1);
            let mut yhat = self.out.forward(&h2)[0];
            if self.binary {
                yhat = (1.0 / (1.0 + (-yhat).exp())).clamp(1e-6, 1.0 - 1e-6);
            }
            out[r] = f64::from(yhat);
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
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.bias.clone();
        for i in 0..self.din {
            let xi = x[i];
            let row = &self.weight[i * self.dout..(i + 1) * self.dout];
            for o in 0..self.dout {
                out[o] += xi * row[o];
            }
        }
        out
    }

    fn forward_relu(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.forward(x);
        for v in &mut out {
            *v = v.max(0.0);
        }
        out
    }
}

fn extract_linear(layer: &Linear<InferBackend>) -> Result<DenseLayer, String> {
    let weight = layer.weight.val();
    let dims = weight.dims();
    let din = dims[0];
    let dout = dims[1];
    let w = weight
        .into_data()
        .as_slice::<f32>()
        .map_err(|e| format!("neural_net weight: {e:?}"))?
        .to_vec();
    let bias = layer
        .bias
        .as_ref()
        .ok_or_else(|| "neural_net layer missing bias".to_string())?
        .val()
        .into_data();
    let b = bias.as_slice::<f32>().map_err(|e| format!("neural_net bias: {e:?}"))?.to_vec();
    if w.len() != din.saturating_mul(dout) || b.len() != dout {
        return Err("neural_net extracted weight shape mismatch".into());
    }
    Ok(DenseLayer { din, dout, weight: w, bias: b })
}

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

    #[test]
    fn fits_a_line() {
        let n = 24usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 2.0 + 0.5 * (i as f64);
        }
        let model = train(&x, n, 2, &y, false, 8, 80, 0.05, 1).unwrap();
        let mut out = vec![0.0; n];
        model.predict(&x, n, 2, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out[n - 1] > out[0]);
    }
}
