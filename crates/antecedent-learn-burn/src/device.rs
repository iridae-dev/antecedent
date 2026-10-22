//! Opt-in Burn device candidate for transfer, fold orchestration, prediction order,
//! repeatability, and a same-seed CPU comparison.
//!
//! The default build stays on `NdArray<f32>`. Cargo feature `wgpu` compiles Burn 0.16.1's
//! WGPU backend, which selects Metal on Apple platforms. That feature is not default, so
//! the CPU wheel does not link the device library. Burn's `metal` feature (Candle) and
//! NVIDIA CUDA / cuDF are not this candidate. This module does not read a clock.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

#[cfg(feature = "wgpu")]
use burn::module::AutodiffModule;
#[cfg(feature = "wgpu")]
use burn::nn::Linear;
#[cfg(feature = "wgpu")]
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
#[cfg(feature = "wgpu")]
use burn::tensor::activation::sigmoid;
#[cfg(feature = "wgpu")]
use burn::tensor::backend::Backend;

use super::{LearnBurnError, MlpSpec, TrainedMlp};

#[cfg(feature = "wgpu")]
type WgpuBackend = burn::backend::Wgpu<f32, i32>;
#[cfg(feature = "wgpu")]
type WgpuTrain = burn::backend::Autodiff<WgpuBackend>;

const NOT_MEASURED_REASON: &str = "This platform has not been measured. No clock was read. \
     NVIDIA CUDA and cuDF are out of scope. The ndarray CPU path is unchanged.";

const NEXT_PLATFORM: &str =
    "Apple Silicon, Metal via Burn 0.16.1 WGPU with the opt-in wgpu feature, including transfer";

/// Where a fit runs. The default wheel is the CPU comparison backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearnerBackend {
    /// `NdArray<f32>` host backend. Numeric behavior is [`super::train`].
    NdArrayCpu,
    /// Burn 0.16.1 WGPU, compiled only with the `wgpu` feature.
    Wgpu,
}

/// Graphics API of the candidate. Presence of this value is not a device probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphicsApi {
    /// WGPU on Apple platforms. Metal is the GPU API.
    ///
    /// Burn 0.16.1's Cargo feature `metal` enables Candle, not this path, and is not enabled.
    MetalViaWgpu,
}

/// Speed evidence this crate stores.
///
/// There is no elapsed-time variant. This platform has not been measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measurement {
    /// No clock was read for the candidate or the CPU comparison.
    NotMeasured,
}

/// Static description of the WGPU candidate. Constructing it does not initialize a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateDescription {
    /// Candidate backend. The comparison backend remains [`LearnerBackend::NdArrayCpu`].
    pub backend: LearnerBackend,
    /// `true` only when this build enabled the `wgpu` feature.
    pub compiled: bool,
    /// Graphics API the candidate would use on this host.
    pub graphics_api: GraphicsApi,
    /// Always [`Measurement::NotMeasured`].
    pub measurement: Measurement,
    /// Why this crate records no timing.
    pub reason: &'static str,
    /// The next place a measurement, including transfer, could be taken.
    pub next_platform: &'static str,
}

/// Describes the candidate without probing a device or reading a clock.
#[must_use]
pub const fn candidate_description() -> CandidateDescription {
    CandidateDescription {
        backend: LearnerBackend::Wgpu,
        compiled: cfg!(feature = "wgpu"),
        graphics_api: GraphicsApi::MetalViaWgpu,
        measurement: Measurement::NotMeasured,
        reason: NOT_MEASURED_REASON,
        next_platform: NEXT_PLATFORM,
    }
}

/// When a scalar loss is copied to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossSync {
    /// After every epoch, before the optimizer step. Matches the CPU adapter's `into_data` read.
    HostEveryEpoch,
}

/// One host copy of a scalar training loss.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpochLoss {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Finite host scalar. Not a duration.
    pub loss: f32,
    /// Synchronization that produced [`Self::loss`].
    pub sync: LossSync,
}

/// Accept a host loss scalar, or refuse a non-finite one the way the CPU adapter does.
///
/// # Errors
///
/// [`LearnBurnError::Diverged`] when `loss` is NaN or infinite.
pub const fn record_host_loss(epoch: usize, loss: f32) -> Result<EpochLoss, LearnBurnError> {
    if !loss.is_finite() {
        return Err(LearnBurnError::Diverged(loss));
    }
    Ok(EpochLoss { epoch, loss, sync: LossSync::HostEveryEpoch })
}

/// How long a fit holds Burn's process-global backend RNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngLockScope {
    /// From the backend seed through the last stochastic step.
    ///
    /// The CPU adapter and the WGPU fit hold the process-global training lock for this
    /// whole scope. [`LeaseScheduler`] is the fold loop's matching protocol: it refuses
    /// a second lease while one is open. The scheduler does not replace that process lock.
    EntireFit,
}

/// A fold's seed while it holds the orchestration lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RngLease {
    /// Fold that owns the lease.
    pub fold: u32,
    /// Seed applied under the lease. Distinct folds keep distinct seeds.
    pub seed: u64,
}

/// Single-orchestrator protocol for [`RngLockScope::EntireFit`].
#[derive(Debug)]
pub struct LeaseScheduler {
    open: Option<RngLease>,
}

impl LeaseScheduler {
    /// No fold holds the lease.
    #[must_use]
    pub const fn new() -> Self {
        Self { open: None }
    }

    /// Take the lease for one fold.
    ///
    /// # Errors
    ///
    /// [`DeviceError::RngLockHeld`] when another fold's lease is still open.
    pub fn acquire(&mut self, fold: u32, seed: u64) -> Result<RngLease, DeviceError> {
        if let Some(held) = self.open {
            return Err(DeviceError::RngLockHeld { held_fold: held.fold });
        }
        let lease = RngLease { fold, seed };
        self.open = Some(lease);
        Ok(lease)
    }

    /// Release the lease this scheduler handed out.
    ///
    /// # Errors
    ///
    /// [`DeviceError::RngLockMismatch`] when `lease` is not the open one.
    pub fn release(&mut self, lease: RngLease) -> Result<(), DeviceError> {
        match self.open {
            Some(open) if open == lease => {
                self.open = None;
                Ok(())
            }
            Some(_) | None => Err(DeviceError::RngLockMismatch),
        }
    }
}

impl Default for LeaseScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// One host-device copy, or a cache reuse that performs none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferStep {
    /// Column-major design, standardized, copied toward the fit backend.
    UploadDesign,
    /// Standardized target copied toward the fit backend.
    UploadTarget,
    /// Scalar loss copied back to the host. One step per epoch.
    DownloadLoss,
    /// Weights copied out once. Prediction then uses [`TrainedMlp::predict`] on the host.
    DownloadWeights,
    /// The runtime cache already held this fold's portable weights. No copy ran.
    CacheReuse,
}

/// Transfer sequence for a fit that is not served from the cache.
///
/// Prediction is absent: after [`TransferStep::DownloadWeights`] the artifact is host-owned.
#[must_use]
pub fn transfer_plan(epochs: usize) -> Vec<TransferStep> {
    let mut steps = Vec::with_capacity(epochs.saturating_add(3));
    steps.push(TransferStep::UploadDesign);
    steps.push(TransferStep::UploadTarget);
    for _ in 0..epochs {
        steps.push(TransferStep::DownloadLoss);
    }
    steps.push(TransferStep::DownloadWeights);
    steps
}

/// Identity of a cached fit. A change of fold, seed, spec, or training data is a miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheKey {
    /// Fold id.
    pub fold: u32,
    /// Initialization seed.
    pub seed: u64,
    /// Hash of every [`MlpSpec`] field and the fold id.
    pub spec_tag: u64,
    /// 128-bit hash of the gathered training row list, design, and target, so the same
    /// fold id over different data is a different key.
    pub content: [u64; 2],
}

/// How long a cache entry stays valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLifetime {
    /// Resident weights live only inside the [`DeviceRuntime`] that inserted them.
    ///
    /// Dropping that runtime, or [`DeviceRuntime::invalidate`], drops the cache.
    /// The cache is not process-global.
    Runtime,
    /// This fit did not enter a runtime cache. The caller owns the portable weights.
    NotCached,
}

/// One cache insertion. `generation` advances on each miss and on invalidation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheEntry {
    /// Key that hit.
    pub key: CacheKey,
    /// Generation of the runtime after this entry was stored.
    pub generation: u64,
    /// Lifetime policy of the entry.
    pub lifetime: CacheLifetime,
}

/// Result of [`DeviceRuntime::cached_model`].
#[derive(Clone, Debug)]
pub struct CachedFit {
    /// Portable host weights.
    pub model: TrainedMlp,
    /// `true` when the closure did not run.
    pub hit: bool,
    /// Entry that served or stored the model.
    pub entry: CacheEntry,
}

/// Who owns the device handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceOwner {
    /// Comparison backend. No device library and no device handle.
    HostProcess,
    /// Created for one fit and dropped before portable weights are returned.
    ///
    /// Not stored on [`DeviceRuntime`] and not process-global.
    FitLocal,
}

/// Who owns the fitted weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelOwner {
    /// The runtime has no resident model.
    Vacant,
    /// [`TrainedMlp`] in host memory. `Send + Sync`. Prediction does not borrow a device.
    HostPortable,
}

/// Ownership and cache policy for a runtime or a finished fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeOwnership {
    /// Backend that produced, or would produce, the weights.
    pub backend: LearnerBackend,
    /// Where the device handle lives.
    pub device: DeviceOwner,
    /// Where the weights live.
    pub model: ModelOwner,
    /// How long a cache of those weights would remain valid.
    pub cache: CacheLifetime,
}

/// One cached host model. The device that produced it is not retained.
#[derive(Clone, Debug)]
struct Resident {
    entry: CacheEntry,
    model: TrainedMlp,
}

/// Host runtime for the CPU comparison path.
///
/// The WGPU device is not stored here. A device fit creates its handle locally and drops
/// it before returning portable weights. The cache stores those host weights only, one
/// entry per [`CacheKey`], until [`Self::invalidate`] or drop. A new key does not evict an old one.
#[derive(Clone, Debug)]
pub struct DeviceRuntime {
    backend: LearnerBackend,
    generation: u64,
    cache: Vec<Resident>,
}

impl DeviceRuntime {
    /// CPU comparison runtime. Does not link or initialize a device.
    #[must_use]
    pub const fn cpu() -> Self {
        Self { backend: LearnerBackend::NdArrayCpu, generation: 0, cache: Vec::new() }
    }

    /// Current ownership. An empty runtime has no resident model.
    #[must_use]
    pub fn ownership(&self) -> RuntimeOwnership {
        RuntimeOwnership {
            backend: self.backend,
            device: DeviceOwner::HostProcess,
            model: if self.cache.is_empty() {
                ModelOwner::Vacant
            } else {
                ModelOwner::HostPortable
            },
            cache: CacheLifetime::Runtime,
        }
    }

    /// Entry for `key`, if portable weights were stored under it.
    #[must_use]
    pub fn probe(&self, key: CacheKey) -> Option<CacheEntry> {
        self.cache.iter().find(|resident| resident.entry.key == key).map(|resident| resident.entry)
    }

    /// Drop every resident model. The next lookup of any key misses.
    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Return the cached model or run `fit` and store it.
    ///
    /// # Errors
    ///
    /// [`DeviceError::NotComparisonBackend`] when this runtime is not the ndarray CPU
    /// comparison. [`DeviceError::Backend`] when `fit` fails.
    pub fn cached_model(
        &mut self,
        key: CacheKey,
        fit: impl FnOnce() -> Result<TrainedMlp, LearnBurnError>,
    ) -> Result<CachedFit, DeviceError> {
        if self.backend != LearnerBackend::NdArrayCpu {
            return Err(DeviceError::NotComparisonBackend);
        }
        if let Some(found) = self.cache.iter().find(|resident| resident.entry.key == key) {
            return Ok(CachedFit { model: found.model.clone(), hit: true, entry: found.entry });
        }
        let model = fit()?;
        self.generation = self.generation.wrapping_add(1);
        let entry =
            CacheEntry { key, generation: self.generation, lifetime: CacheLifetime::Runtime };
        self.cache.push(Resident { entry, model: model.clone() });
        Ok(CachedFit { model, hit: false, entry })
    }
}

/// One cross-fit fold. Predict rows of a plan partition `0..nrows`. Train rows may overlap.
#[derive(Clone, Debug)]
pub struct FoldSpec {
    /// Stable id used in the cache key and the RNG lease.
    pub id: u32,
    /// Rows whose design and target are gathered for [`super::train`].
    pub train_rows: Vec<usize>,
    /// Rows written back in physical order. Disjoint across the plan.
    pub predict_rows: Vec<usize>,
    /// Architecture, step size, and seed. The seed is per fold.
    pub spec: MlpSpec,
}

/// Cache key for one fold over the design `x_colmajor` (`nrows` by `ncols`) and target `y`.
///
/// The key changes when the fold, seed, spec, training rows, or the values of the gathered
/// training design and target change. `x_colmajor` and `y` must be the arrays the fold will
/// be fitted on, with every training row inside `0..nrows`.
#[must_use]
pub fn cache_key(
    fold: &FoldSpec,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
) -> CacheKey {
    let train_x = gather_colmajor(x_colmajor, nrows, ncols, &fold.train_rows);
    let train_y = gather_target(y, &fold.train_rows);
    cache_key_for(fold, ncols, &train_x, &train_y)
}

fn cache_key_for(fold: &FoldSpec, ncols: usize, train_x: &[f64], train_y: &[f64]) -> CacheKey {
    CacheKey {
        fold: fold.id,
        seed: fold.spec.seed,
        spec_tag: spec_tag(&fold.spec, fold.id),
        content: content_tag(&fold.train_rows, ncols, train_x, train_y),
    }
}

/// Predictions for one fold, already in that fold's `predict_rows` order.
#[derive(Clone, Debug)]
pub struct FoldOutcome {
    /// [`FoldSpec::id`].
    pub id: u32,
    /// Seed the fit used.
    pub seed: u64,
    /// Physical rows these predictions belong to, in gather order.
    pub predict_rows: Vec<usize>,
    /// One value per [`Self::predict_rows`] entry.
    pub predictions: Vec<f64>,
    /// Portable host weights. Prediction does not need a device.
    pub model: TrainedMlp,
    /// Copies that ran, or [`TransferStep::CacheReuse`] when the cache hit.
    pub transfers: Vec<TransferStep>,
}

/// Physical-order predictions for a whole fold plan.
#[derive(Clone, Debug)]
pub struct Orchestration {
    /// Index `i` is the prediction for physical row `i`.
    pub physical: Vec<f64>,
    /// One outcome per input fold, in plan order.
    pub folds: Vec<FoldOutcome>,
    /// RNG protocol the plan enforced.
    pub rng_lock: RngLockScope,
    /// Loss synchronization the CPU adapter applies inside each fit.
    pub loss_sync: LossSync,
    /// Always [`Measurement::NotMeasured`].
    pub measurement: Measurement,
}

struct HostDesign<'a> {
    x: &'a [f64],
    y: &'a [f64],
    nrows: usize,
    ncols: usize,
}

/// Fit each fold on the ndarray CPU backend and scatter predictions to physical rows.
///
/// Repeat calls with the same seeds replay the CPU adapter. This function does not read a clock.
///
/// # Errors
///
/// A fold plan that is empty, overlapping, out of bounds, or short of a partition;
/// a shape mismatch; or a failure of [`super::train`] / [`TrainedMlp::predict`].
pub fn orchestrate_cpu(
    runtime: &mut DeviceRuntime,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    folds: &[FoldSpec],
) -> Result<Orchestration, DeviceError> {
    if runtime.backend != LearnerBackend::NdArrayCpu {
        return Err(DeviceError::NotComparisonBackend);
    }
    if nrows == 0
        || ncols == 0
        || y.len() != nrows
        || x_colmajor.len() < nrows.saturating_mul(ncols)
    {
        return Err(DeviceError::Shape);
    }
    validate_predict_partition(nrows, folds)?;
    let design = HostDesign { x: x_colmajor, y, nrows, ncols };
    let mut scheduler = LeaseScheduler::new();
    let mut physical = vec![0.0; nrows];
    let mut outcomes = Vec::with_capacity(folds.len());
    for fold in folds {
        let outcome = fit_one_fold(runtime, &mut scheduler, &design, fold)?;
        for (&row, value) in outcome.predict_rows.iter().zip(&outcome.predictions) {
            physical[row] = *value;
        }
        outcomes.push(outcome);
    }
    Ok(Orchestration {
        physical,
        folds: outcomes,
        rng_lock: RngLockScope::EntireFit,
        loss_sync: LossSync::HostEveryEpoch,
        measurement: Measurement::NotMeasured,
    })
}

fn fit_one_fold(
    runtime: &mut DeviceRuntime,
    scheduler: &mut LeaseScheduler,
    design: &HostDesign<'_>,
    fold: &FoldSpec,
) -> Result<FoldOutcome, DeviceError> {
    let lease = scheduler.acquire(fold.id, fold.spec.seed)?;
    let train_x = gather_colmajor(design.x, design.nrows, design.ncols, &fold.train_rows);
    let train_y = gather_target(design.y, &fold.train_rows);
    let key = cache_key_for(fold, design.ncols, &train_x, &train_y);
    let fitted = runtime.cached_model(key, || {
        super::train(&train_x, fold.train_rows.len(), design.ncols, &train_y, &fold.spec)
    });
    let release = scheduler.release(lease);
    let cached = fitted?;
    release?;
    let predict_x = gather_colmajor(design.x, design.nrows, design.ncols, &fold.predict_rows);
    let mut predictions = vec![0.0; fold.predict_rows.len()];
    cached.model.predict(&predict_x, fold.predict_rows.len(), design.ncols, &mut predictions)?;
    let transfers =
        if cached.hit { vec![TransferStep::CacheReuse] } else { transfer_plan(fold.spec.epochs) };
    Ok(FoldOutcome {
        id: fold.id,
        seed: fold.spec.seed,
        predict_rows: fold.predict_rows.clone(),
        predictions,
        model: cached.model,
        transfers,
    })
}

fn validate_predict_partition(nrows: usize, folds: &[FoldSpec]) -> Result<(), DeviceError> {
    if folds.is_empty() {
        return Err(DeviceError::EmptyFold);
    }
    let mut seen = vec![false; nrows];
    for fold in folds {
        if fold.train_rows.is_empty() || fold.predict_rows.is_empty() {
            return Err(DeviceError::EmptyFold);
        }
        for &row in fold.train_rows.iter().chain(&fold.predict_rows) {
            if row >= nrows {
                return Err(DeviceError::RowOutOfBounds { row, nrows });
            }
        }
        for &row in &fold.predict_rows {
            if seen[row] {
                return Err(DeviceError::FoldOverlap { row });
            }
            seen[row] = true;
        }
    }
    if let Some(row) = seen.iter().position(|&filled| !filled) {
        return Err(DeviceError::UnfilledRow { row });
    }
    Ok(())
}

fn gather_colmajor(x: &[f64], nrows: usize, ncols: usize, rows: &[usize]) -> Vec<f64> {
    let m = rows.len();
    let mut out = vec![0.0; m.saturating_mul(ncols)];
    for c in 0..ncols {
        for (i, &row) in rows.iter().enumerate() {
            out[c * m + i] = x[c * nrows + row];
        }
    }
    out
}

fn gather_target(y: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&row| y[row]).collect()
}

/// Hash of every [`MlpSpec`] field and the fold id. The cache lives in one process, so the
/// standard hasher's fixed-key output is stable for as long as the runtime is.
fn spec_tag(spec: &MlpSpec, fold: u32) -> u64 {
    let MlpSpec { binary, hidden, epochs, learning_rate, seed } = *spec;
    let mut hasher = DefaultHasher::new();
    "antecedent.learn_burn.spec_tag".hash(&mut hasher);
    fold.hash(&mut hasher);
    binary.hash(&mut hasher);
    hidden.hash(&mut hasher);
    epochs.hash(&mut hasher);
    learning_rate.to_bits().hash(&mut hasher);
    seed.hash(&mut hasher);
    hasher.finish()
}

/// Domain-separated blake3 digest (first 128 bits) of the training rows, width, design, and
/// target. The value is stable across processes and platforms.
fn content_tag(rows: &[usize], ncols: usize, x: &[f64], y: &[f64]) -> [u64; 2] {
    let mut hasher = blake3::Hasher::new_derive_key("antecedent.learn_burn.content_tag.v1");
    hasher.update(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        hasher.update(&(*row as u64).to_le_bytes());
    }
    hasher.update(&(ncols as u64).to_le_bytes());
    hasher.update(&(x.len() as u64).to_le_bytes());
    for value in x {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&(y.len() as u64).to_le_bytes());
    for value in y {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let word = |at: usize| {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(&bytes[at..at + 8]);
        u64::from_le_bytes(buf)
    };
    [word(0), word(8)]
}

/// Portable weights from a device fit, plus the policy that produced them.
///
/// Inference uses [`TrainedMlp::predict`]. The device handle is already dropped.
#[derive(Clone, Debug)]
pub struct PortableFit {
    /// Host weights. `Send + Sync`.
    pub model: TrainedMlp,
    /// Backend that trained. CPU comparison uses [`LearnerBackend::NdArrayCpu`].
    pub backend: LearnerBackend,
    /// Copies performed for this fit.
    pub transfers: Vec<TransferStep>,
    /// Loss synchronization applied during the fit.
    pub loss_sync: LossSync,
    /// RNG lock scope held during the fit.
    pub rng_lock: RngLockScope,
    /// Device and model ownership after the fit returns.
    pub ownership: RuntimeOwnership,
    /// Always [`Measurement::NotMeasured`].
    pub measurement: Measurement,
    /// Host loss copies that completed. Equal to the epoch count on success. Not a duration.
    pub epochs_synced: usize,
}

impl PortableFit {
    /// Host prediction from the downloaded weights.
    ///
    /// # Errors
    ///
    /// Shape mismatch. See [`TrainedMlp::predict`].
    pub fn predict(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        out: &mut [f64],
    ) -> Result<(), LearnBurnError> {
        self.model.predict(x_colmajor, nrows, ncols, out)
    }
}

/// Failure of fold orchestration or the runtime cache.
#[derive(Debug)]
pub enum DeviceError {
    /// A plan or a fold has no rows.
    EmptyFold,
    /// A row index is outside `0..nrows`.
    RowOutOfBounds {
        /// Offending row.
        row: usize,
        /// Physical row count.
        nrows: usize,
    },
    /// Two folds predict the same physical row.
    FoldOverlap {
        /// Row claimed twice.
        row: usize,
    },
    /// Some physical row is predicted by no fold.
    UnfilledRow {
        /// Missing row.
        row: usize,
    },
    /// Design or target length does not match the declared shape.
    Shape,
    /// A second lease was requested before the first was released.
    RngLockHeld {
        /// Fold that still holds the lease.
        held_fold: u32,
    },
    /// Release did not name the open lease.
    RngLockMismatch,
    /// The CPU orchestration entry point was used on a non-ndarray runtime.
    NotComparisonBackend,
    /// [`super::train`] or [`TrainedMlp::predict`] failed.
    Backend(LearnBurnError),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFold => f.write_str("neural fold plan is empty"),
            Self::RowOutOfBounds { row, nrows } => {
                write!(f, "neural fold row {row} is outside 0..{nrows}")
            }
            Self::FoldOverlap { row } => write!(f, "neural fold predict rows overlap at {row}"),
            Self::UnfilledRow { row } => {
                write!(f, "neural fold plan leaves row {row} unpredicted")
            }
            Self::Shape => f.write_str("neural fold design / target shape mismatch"),
            Self::RngLockHeld { held_fold } => {
                write!(f, "neural RNG lock is held by fold {held_fold}")
            }
            Self::RngLockMismatch => {
                f.write_str("neural RNG lock release does not match the holder")
            }
            Self::NotComparisonBackend => {
                f.write_str("CPU comparison accepts only the ndarray backend")
            }
            Self::Backend(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LearnBurnError> for DeviceError {
    fn from(error: LearnBurnError) -> Self {
        Self::Backend(error)
    }
}

/// Fit on Burn's WGPU backend and return host portable weights.
///
/// Available only with the `wgpu` feature. The default `WgpuDevice` is created here and
/// dropped before return. Each epoch copies the scalar loss to the host. This function
/// does not read a clock; [`PortableFit::measurement`] is [`Measurement::NotMeasured`].
///
/// # Errors
///
/// The same input and divergence failures as [`super::train`].
#[cfg(feature = "wgpu")]
pub fn fit_portable_wgpu(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
) -> Result<PortableFit, LearnBurnError> {
    let prepared = prepare_fit(x_colmajor, nrows, ncols, y, spec)?;
    let _guard = super::TRAIN_LOCK.lock().map_err(|_| LearnBurnError::LockPoisoned)?;
    <WgpuBackend as Backend>::seed(prepared.seed);
    let device = burn::backend::wgpu::WgpuDevice::default();
    let mut model = super::Mlp::<WgpuTrain>::new(&device, ncols, prepared.hidden);
    let mut optim = AdamConfig::new().init();
    let x_t = super::tensor2(&prepared.x_rm, [nrows, ncols], &device);
    let y_t = super::tensor2(&prepared.y_f, [nrows, 1], &device);
    for epoch in 0..prepared.epochs {
        let pred = model.forward(x_t.clone());
        let loss = if prepared.binary {
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
        let loss_value = loss
            .clone()
            .into_data()
            .as_slice::<f32>()
            .map_err(|error| LearnBurnError::Loss(format!("{error:?}")))?
            .first()
            .copied()
            .unwrap_or(f32::NAN);
        record_host_loss(epoch, loss_value)?;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(prepared.learning_rate, model, grads);
    }
    let model = model.valid();
    let trained = portable_from_inner(
        &model,
        prepared.binary,
        ncols,
        prepared.x_center,
        prepared.x_scale,
        prepared.y_center,
        prepared.y_scale,
    )?;
    Ok(PortableFit {
        model: trained,
        backend: LearnerBackend::Wgpu,
        transfers: transfer_plan(prepared.epochs),
        loss_sync: LossSync::HostEveryEpoch,
        rng_lock: RngLockScope::EntireFit,
        ownership: RuntimeOwnership {
            backend: LearnerBackend::Wgpu,
            device: DeviceOwner::FitLocal,
            model: ModelOwner::HostPortable,
            cache: CacheLifetime::NotCached,
        },
        measurement: Measurement::NotMeasured,
        epochs_synced: prepared.epochs,
    })
}

#[cfg(feature = "wgpu")]
struct PreparedFit {
    binary: bool,
    hidden: usize,
    epochs: usize,
    learning_rate: f64,
    seed: u64,
    x_center: Vec<f64>,
    x_scale: Vec<f64>,
    y_center: f64,
    y_scale: f64,
    x_rm: Vec<f32>,
    y_f: Vec<f32>,
}

#[cfg(feature = "wgpu")]
fn prepare_fit(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
) -> Result<PreparedFit, LearnBurnError> {
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
    let (x_center, x_scale) = super::column_standardization(x_colmajor, nrows, ncols);
    let (y_center, y_scale) = if binary { (0.0, 1.0) } else { super::target_standardization(y) };
    let x_std = super::standardized(x_colmajor, nrows, ncols, &x_center, &x_scale);
    let x_rm = super::colmajor_to_rowmajor_f32(&x_std, nrows, ncols);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the network trains in f32, so narrowing the standardized target is the intended precision"
    )]
    let y_f: Vec<f32> = y.iter().map(|&v| ((v - y_center) / y_scale) as f32).collect();
    Ok(PreparedFit {
        binary,
        hidden,
        epochs,
        learning_rate,
        seed,
        x_center,
        x_scale,
        y_center,
        y_scale,
        x_rm,
        y_f,
    })
}

#[cfg(feature = "wgpu")]
fn portable_from_inner<B: Backend>(
    model: &super::Mlp<B>,
    binary: bool,
    ncols: usize,
    x_center: Vec<f64>,
    x_scale: Vec<f64>,
    y_center: f64,
    y_scale: f64,
) -> Result<TrainedMlp, LearnBurnError> {
    let trained = TrainedMlp {
        l1: extract_portable(&model.l1)?,
        l2: extract_portable(&model.l2)?,
        out: extract_portable(&model.out)?,
        binary,
        ncols,
        x_center,
        x_scale,
        y_center,
        y_scale,
    };
    let all_finite = [&trained.l1, &trained.l2, &trained.out]
        .iter()
        .all(|layer| layer.weight.iter().chain(&layer.bias).all(|value| value.is_finite()));
    if !all_finite {
        return Err(LearnBurnError::NonFiniteWeights);
    }
    Ok(trained)
}

#[cfg(feature = "wgpu")]
fn extract_portable<B: Backend>(layer: &Linear<B>) -> Result<super::DenseLayer, LearnBurnError> {
    let weight = layer.weight.val();
    let dims = weight.dims();
    let din = dims[0];
    let dout = dims[1];
    let w = weight
        .into_data()
        .as_slice::<f32>()
        .map_err(|error| LearnBurnError::Weight(format!("{error:?}")))?
        .to_vec();
    let bias = layer.bias.as_ref().ok_or(LearnBurnError::MissingBias)?.val().into_data();
    let b = bias
        .as_slice::<f32>()
        .map_err(|error| LearnBurnError::Bias(format!("{error:?}")))?
        .to_vec();
    if w.len() != din.saturating_mul(dout) || b.len() != dout {
        return Err(LearnBurnError::WeightShape);
    }
    Ok(super::DenseLayer { din, dout, weight: w, bias: b })
}

#[cfg(test)]
fn zero_layer(din: usize, dout: usize) -> super::DenseLayer {
    super::DenseLayer { din, dout, weight: vec![0.0; din * dout], bias: vec![0.0; dout] }
}

#[cfg(test)]
fn zero_mlp(ncols: usize) -> TrainedMlp {
    let hidden = 1;
    TrainedMlp {
        l1: zero_layer(ncols, hidden),
        l2: zero_layer(hidden, hidden),
        out: zero_layer(hidden, 1),
        binary: false,
        ncols,
        x_center: vec![0.0; ncols],
        x_scale: vec![1.0; ncols],
        y_center: 0.0,
        y_scale: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::zero_mlp;
    use super::*;

    #[test]
    fn candidate_is_unmeasured_wgpu_metal_and_off_by_default() {
        let description = candidate_description();
        assert_eq!(description.backend, LearnerBackend::Wgpu);
        assert_eq!(description.graphics_api, GraphicsApi::MetalViaWgpu);
        assert_eq!(description.measurement, Measurement::NotMeasured);
        assert_eq!(description.compiled, cfg!(feature = "wgpu"));
        assert!(description.reason.contains("not been measured"));
        assert!(!description.reason.contains("ms"));
        assert!(description.next_platform.contains("Apple"));
        assert!(description.next_platform.contains("Metal"));
        assert!(!description.next_platform.contains("CUDA"));
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains("wgpu = [\"burn/wgpu\"]"));
        assert!(!manifest.lines().any(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with('#') && trimmed.starts_with("default") && trimmed.contains("wgpu")
        }));
        assert!(!manifest.lines().any(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with('#') && (trimmed.contains("metal") || trimmed.contains("cuda"))
        }));
    }

    #[test]
    fn rng_leases_do_not_overlap() {
        let mut scheduler = LeaseScheduler::new();
        let first = scheduler.acquire(0, 5).unwrap();
        assert!(matches!(scheduler.acquire(1, 9), Err(DeviceError::RngLockHeld { held_fold: 0 })));
        scheduler.release(first).unwrap();
        let second = scheduler.acquire(1, 9).unwrap();
        assert_eq!(second.seed, 9);
        assert_eq!(second.fold, 1);
        assert!(matches!(scheduler.release(first), Err(DeviceError::RngLockMismatch)));
        scheduler.release(second).unwrap();
        assert!(matches!(scheduler.release(second), Err(DeviceError::RngLockMismatch)));
    }

    #[test]
    fn host_loss_sync_rejects_non_finite_and_records_the_copy() {
        assert!(matches!(
            record_host_loss(0, f32::NAN),
            Err(LearnBurnError::Diverged(value)) if value.is_nan()
        ));
        assert!(matches!(
            record_host_loss(1, f32::INFINITY),
            Err(LearnBurnError::Diverged(value)) if value.is_infinite()
        ));
        let synced = record_host_loss(3, 0.25).unwrap();
        assert_eq!(synced.epoch, 3);
        assert_eq!(synced.sync, LossSync::HostEveryEpoch);
        #[allow(clippy::float_cmp, reason = "the host copy must preserve the finite scalar")]
        {
            assert_eq!(synced.loss, 0.25);
        }
    }

    #[test]
    fn transfer_plan_syncs_loss_every_epoch_and_downloads_weights() {
        let plan = transfer_plan(3);
        assert_eq!(
            plan,
            vec![
                TransferStep::UploadDesign,
                TransferStep::UploadTarget,
                TransferStep::DownloadLoss,
                TransferStep::DownloadLoss,
                TransferStep::DownloadLoss,
                TransferStep::DownloadWeights,
            ]
        );
    }

    #[test]
    fn cache_is_runtime_scoped_and_misses_on_a_new_key() {
        let mut runtime = DeviceRuntime::cpu();
        assert_eq!(runtime.ownership().model, ModelOwner::Vacant);
        assert_eq!(runtime.ownership().device, DeviceOwner::HostProcess);
        assert_eq!(runtime.ownership().cache, CacheLifetime::Runtime);
        let key = CacheKey { fold: 1, seed: 2, spec_tag: 3, content: [4, 5] };
        let mut calls = 0u32;
        let first = runtime
            .cached_model(key, || {
                calls += 1;
                Ok(zero_mlp(1))
            })
            .unwrap();
        assert!(!first.hit);
        assert_eq!(first.entry.generation, 1);
        assert_eq!(first.entry.lifetime, CacheLifetime::Runtime);
        let second = runtime
            .cached_model(key, || {
                calls += 1;
                Ok(zero_mlp(1))
            })
            .unwrap();
        assert!(second.hit);
        assert_eq!(calls, 1);
        assert_eq!(runtime.ownership().model, ModelOwner::HostPortable);
        assert!(runtime.probe(key).is_some());
        let other = CacheKey { fold: 2, seed: 2, spec_tag: 3, content: [4, 5] };
        let third = runtime
            .cached_model(other, || {
                calls += 1;
                Ok(zero_mlp(1))
            })
            .unwrap();
        assert!(!third.hit);
        assert_eq!(calls, 2);
        assert!(runtime.probe(key).is_some());
        assert!(runtime.probe(other).is_some());
        runtime.invalidate();
        assert_eq!(runtime.ownership().model, ModelOwner::Vacant);
        assert!(runtime.probe(other).is_none());
        assert_eq!(runtime.ownership().cache, CacheLifetime::Runtime);
    }

    #[test]
    fn portable_cpu_inference_does_not_need_the_device() {
        const fn assert_send<T: Send>() {}
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send::<DeviceRuntime>();
        assert_send_sync::<TrainedMlp>();
        assert_send_sync::<PortableFit>();
        let model = zero_mlp(2);
        let mut out = [0.0, 0.0];
        model.predict(&[0.0, 1.0, 2.0, 3.0], 2, 2, &mut out).unwrap();
        assert!(out.iter().all(|value| value.is_finite()));
        assert_eq!(candidate_description().measurement, Measurement::NotMeasured);
    }

    fn line(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 1.5 + 0.25 * (i as f64);
        }
        (x, y)
    }

    fn fold_plan(n: usize, epochs: usize) -> Vec<FoldSpec> {
        let mid = n / 2;
        let spec = |seed| MlpSpec { binary: false, hidden: 4, epochs, learning_rate: 0.05, seed };
        vec![
            FoldSpec {
                id: 0,
                train_rows: (0..mid).collect(),
                predict_rows: (mid..n).collect(),
                spec: spec(11),
            },
            FoldSpec {
                id: 1,
                train_rows: (mid..n).collect(),
                predict_rows: (0..mid).collect(),
                spec: spec(29),
            },
        ]
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "a repeated CPU seed is a bit-level replay")]
    fn cpu_folds_repeat_in_physical_order() {
        let n = 8usize;
        let (x, y) = line(n);
        let folds = fold_plan(n, 12);
        let mut runtime = DeviceRuntime::cpu();
        let first = orchestrate_cpu(&mut runtime, &x, n, 2, &y, &folds).unwrap();
        assert_eq!(first.physical.len(), n);
        assert!(first.physical.iter().all(|value| value.is_finite()));
        assert_eq!(first.measurement, Measurement::NotMeasured);
        assert_eq!(first.loss_sync, LossSync::HostEveryEpoch);
        assert_eq!(first.rng_lock, RngLockScope::EntireFit);
        let high = &first.folds[0];
        let low = &first.folds[1];
        assert_eq!(high.predict_rows[0], 4);
        assert_eq!(low.predict_rows[0], 0);
        assert_eq!(high.predictions[0], first.physical[4]);
        assert_eq!(low.predictions[0], first.physical[0]);
        assert_eq!(
            high.transfers.iter().filter(|step| matches!(step, TransferStep::DownloadLoss)).count(),
            12
        );
        assert_eq!(high.transfers.last(), Some(&TransferStep::DownloadWeights));
        let mut again = DeviceRuntime::cpu();
        let second = orchestrate_cpu(&mut again, &x, n, 2, &y, &folds).unwrap();
        assert_eq!(first.physical, second.physical);
        let third = orchestrate_cpu(&mut runtime, &x, n, 2, &y, &folds).unwrap();
        assert_eq!(first.physical, third.physical);
        assert_eq!(third.folds[0].transfers, vec![TransferStep::CacheReuse]);
        assert_eq!(third.folds[1].transfers, vec![TransferStep::CacheReuse]);
    }

    #[test]
    fn same_fold_id_over_different_data_is_a_miss() {
        let n = 8usize;
        let (x, y) = line(n);
        let folds = fold_plan(n, 12);
        let mut runtime = DeviceRuntime::cpu();
        let first = orchestrate_cpu(&mut runtime, &x, n, 2, &y, &folds).unwrap();
        assert!(first.folds.iter().all(|fold| fold.transfers != vec![TransferStep::CacheReuse]));
        let keys: Vec<CacheKey> = folds.iter().map(|f| cache_key(f, &x, n, 2, &y)).collect();
        assert!(keys.iter().all(|key| runtime.probe(*key).is_some()));

        // A different target under the same fold ids and specs must refit.
        let y_shifted: Vec<f64> = y.iter().map(|v| v + 3.0).collect();
        let shifted_keys: Vec<CacheKey> =
            folds.iter().map(|f| cache_key(f, &x, n, 2, &y_shifted)).collect();
        for (old, new) in keys.iter().zip(&shifted_keys) {
            assert_eq!((old.fold, old.seed, old.spec_tag), (new.fold, new.seed, new.spec_tag));
            assert_ne!(old.content, new.content);
        }
        assert!(shifted_keys.iter().all(|key| runtime.probe(*key).is_none()));
        let refit = orchestrate_cpu(&mut runtime, &x, n, 2, &y_shifted, &folds).unwrap();
        assert!(refit.folds.iter().all(|fold| fold.transfers != vec![TransferStep::CacheReuse]));
        assert_ne!(first.physical, refit.physical);

        // A different design under the same fold ids misses too.
        let mut x_moved = x.clone();
        x_moved[n + 1] += 0.5;
        let moved = cache_key(&folds[0], &x_moved, n, 2, &y);
        assert_ne!(moved, keys[0]);

        // Different training rows, same fold id, spec, and arrays, miss.
        let mut narrower = folds[0].clone();
        narrower.train_rows.pop();
        assert_ne!(cache_key(&narrower, &x, n, 2, &y), keys[0]);

        // Identical inputs hit and replay.
        let again = orchestrate_cpu(&mut runtime, &x, n, 2, &y, &folds).unwrap();
        assert!(again.folds.iter().all(|fold| fold.transfers == vec![TransferStep::CacheReuse]));
        assert_eq!(again.physical, first.physical);
    }

    #[test]
    fn spec_tag_depends_on_every_spec_field() {
        let base = MlpSpec { binary: false, hidden: 4, epochs: 12, learning_rate: 0.05, seed: 11 };
        let tag = spec_tag(&base, 0);
        assert_eq!(tag, spec_tag(&base, 0));
        let variants = [
            MlpSpec { binary: true, ..base },
            MlpSpec { hidden: 5, ..base },
            MlpSpec { epochs: 13, ..base },
            MlpSpec { learning_rate: 0.06, ..base },
            MlpSpec { seed: 12, ..base },
        ];
        for variant in variants {
            assert_ne!(spec_tag(&variant, 0), tag, "{variant:?}");
        }
        assert_ne!(spec_tag(&base, 1), tag);
        // Seed and fold id are not interchangeable.
        assert_ne!(
            spec_tag(&MlpSpec { seed: 1, ..base }, 0),
            spec_tag(&MlpSpec { seed: 0, ..base }, 1)
        );
    }

    #[test]
    fn rejected_fold_plans_do_not_fit() {
        let n = 4usize;
        let (x, y) = line(n);
        let mut runtime = DeviceRuntime::cpu();
        assert!(matches!(
            orchestrate_cpu(&mut runtime, &x, n, 2, &y, &[]),
            Err(DeviceError::EmptyFold)
        ));
        let mut overlap = fold_plan(n, 1);
        overlap[1].predict_rows = vec![2, 3];
        assert!(matches!(
            orchestrate_cpu(&mut runtime, &x, n, 2, &y, &overlap),
            Err(DeviceError::FoldOverlap { row: 2 })
        ));
        let mut bounds = fold_plan(n, 1);
        bounds[0].train_rows = vec![0, 9];
        assert!(matches!(
            orchestrate_cpu(&mut runtime, &x, n, 2, &y, &bounds),
            Err(DeviceError::RowOutOfBounds { row: 9, nrows: 4 })
        ));
        let only = vec![FoldSpec {
            id: 0,
            train_rows: vec![2, 3],
            predict_rows: vec![0, 1],
            spec: overlap[0].spec,
        }];
        assert!(matches!(
            orchestrate_cpu(&mut runtime, &x, n, 2, &y, &only),
            Err(DeviceError::UnfilledRow { row: 2 })
        ));
        let short_y = &y[..n - 1];
        assert!(matches!(
            orchestrate_cpu(&mut runtime, &x, n, 2, short_y, &fold_plan(n, 1)),
            Err(DeviceError::Shape)
        ));
        assert_eq!(runtime.ownership().model, ModelOwner::Vacant);
    }
}
