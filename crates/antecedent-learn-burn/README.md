# antecedent-learn-burn

An optional neural-network learner for Antecedent's nuisance models, built on Burn. It trains
a multilayer perceptron with column-major `f64` input and output. The default build trains
and predicts on the `NdArray<f32>` CPU backend; the opt-in `wgpu` feature adds Burn's WGPU
device backend, and `accelerate` adds Apple Accelerate BLAS under the ndarray backend.

The learner is reached through `antecedent-learn`'s `ml-neural` feature (`neural_net`), which
is not part of `ml-full`. `antecedent-estimate` never names Burn.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which enables this learner with its `ml-neural` feature. Depend on this crate directly only to time the backends (see the `accelerator_stage_timing` example).

Main types: `MlpSpec`, `TrainedMlp`, `LearnBurnError`. API reference: <https://docs.rs/antecedent-learn-burn>.

## License

MIT OR Apache-2.0.
