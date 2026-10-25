//! Public `cpu` / `gpu` / `auto` compute policy for the neural fit.
//!
//! [`plan`] decides the route from the request, the linked runtime, the adapter the runtime
//! reports, and the device-byte budget. The checks are, in order: dtype, operator, linked
//! library, device present, device memory, and (for `auto` only) the work crossover. A
//! `gpu` request that fails a check is refused with the reason. An `auto` request that
//! fails a check runs on the CPU and records the reason as [`Provenance::fallback`]. A
//! `cpu` request never probes a device.
//!
//! [`Provenance`] names the library, device, dtype, and replay scope of the route that
//! actually ran. A CPU fit is never labelled as device work: the label is built from the
//! route taken, not from the policy asked for.
//!
//! [`fit_with_policy`] runs the plan. Device memory is leased from the device pool for the
//! duration of the device fit and returned on every exit, including a device panic, which
//! is caught and (for `auto`) retried on the CPU.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use antecedent_core::device_reservation::DeviceReservationPool;

use super::admission::{self, Footprint};
use super::device::LearnerBackend;
use super::staged::{StageTimings, fit_flops};
use super::{LearnBurnError, MlpSpec, TrainedMlp};

/// Where the caller asks the fit to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ComputePolicy {
    /// The host CPU (`NdArray<f32>`). No device is probed.
    #[default]
    Cpu,
    /// The device backend. Refused, with the reason, when it cannot run.
    Gpu,
    /// The device when every check passes and the work is above the crossover, else the CPU.
    Auto,
}

impl ComputePolicy {
    /// Stable lowercase name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Auto => "auto",
        }
    }

    /// Parse `"cpu"`, `"gpu"`, or `"auto"`.
    ///
    /// # Errors
    ///
    /// The unrecognized text.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            other => Err(other.to_string()),
        }
    }
}

/// Arithmetic width requested for the fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// `f32`: the neural baseline.
    F32,
    /// `f64`: not available on the device backend.
    F64,
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        })
    }
}

/// Prediction task of the fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    /// Squared-error regression.
    Regression,
    /// Binary probability (cross-entropy).
    BinaryProbability,
}

/// What a fit asks of the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitRequest {
    /// Policy asked for.
    pub policy: ComputePolicy,
    /// Arithmetic width asked for.
    pub dtype: Dtype,
    /// Task.
    pub operator: OperatorKind,
    /// Sample weights supplied. The network has no weighted loss, so a device fit refuses.
    pub weighted: bool,
    /// Rows to train on.
    pub nrows: usize,
    /// Design columns.
    pub ncols: usize,
    /// Hidden width.
    pub hidden: usize,
    /// Epochs.
    pub epochs: usize,
}

impl FitRequest {
    /// Request for an unweighted `f32` neural fit described by `spec`.
    #[must_use]
    pub const fn neural(policy: ComputePolicy, spec: &MlpSpec, nrows: usize, ncols: usize) -> Self {
        Self {
            policy,
            dtype: Dtype::F32,
            operator: if spec.binary {
                OperatorKind::BinaryProbability
            } else {
                OperatorKind::Regression
            },
            weighted: false,
            nrows,
            ncols,
            hidden: spec.hidden,
            epochs: spec.epochs,
        }
    }

    /// Approximate floating-point work of the fit.
    #[must_use]
    pub fn work(&self) -> u64 {
        fit_flops(self.nrows, self.ncols, self.hidden, self.epochs)
    }
}

/// Facts about the adapter the device runtime reported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    /// Adapter name.
    pub name: String,
    /// Graphics API the runtime selected (`Metal`, `Vulkan`, ...).
    pub api: String,
    /// Adapter class (`IntegratedGpu`, `DiscreteGpu`, `Cpu`, ...).
    pub device_type: String,
    /// Largest single buffer the adapter allows.
    pub max_buffer_bytes: u64,
}

/// What this process can say about the device runtime.
pub trait DeviceProbe {
    /// Whether this build links the device library.
    fn compiled(&self) -> bool;

    /// The adapter the runtime would use.
    ///
    /// # Errors
    ///
    /// Why no adapter is usable.
    fn adapter(&self) -> Result<AdapterSummary, String>;
}

/// The real probe: the linked Burn WGPU runtime on this host.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemProbe;

impl DeviceProbe for SystemProbe {
    fn compiled(&self) -> bool {
        cfg!(feature = "wgpu")
    }

    fn adapter(&self) -> Result<AdapterSummary, String> {
        if let Some(reason) = device_disabled(std::env::var(DISABLE_DEVICE_ENV).ok().as_deref()) {
            return Err(reason);
        }
        system_adapter()
    }
}

/// Environment variable that hides the device from this process when set to a non-empty value
/// other than `0`. A device-capable build then behaves as if no adapter were present, so
/// `gpu` is refused with `no_device` and `auto` falls back to the CPU. Use it on shared hosts
/// that must keep jobs off the GPU, and to exercise the missing-device refusal.
pub const DISABLE_DEVICE_ENV: &str = "ANTECEDENT_DISABLE_DEVICE";

/// The refusal text when `value` of [`DISABLE_DEVICE_ENV`] disables the device.
fn device_disabled(value: Option<&str>) -> Option<String> {
    match value {
        Some(text) if !text.is_empty() && text != "0" => {
            Some(format!("the device is disabled by {DISABLE_DEVICE_ENV}"))
        }
        _ => None,
    }
}

#[cfg(feature = "wgpu")]
fn system_adapter() -> Result<AdapterSummary, String> {
    use std::sync::OnceLock;

    use burn::backend::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, init_setup};

    // Initializing a device twice is not allowed, so the one answer is cached. The first
    // call registers the default device that every later fit reuses.
    static ADAPTER: OnceLock<Result<AdapterSummary, String>> = OnceLock::new();
    ADAPTER
        .get_or_init(|| {
            std::panic::catch_unwind(|| {
                let device = WgpuDevice::default();
                let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
                let info = setup.adapter.get_info();
                AdapterSummary {
                    name: info.name.clone(),
                    api: format!("{:?}", info.backend),
                    device_type: format!("{:?}", info.device_type),
                    max_buffer_bytes: setup.adapter.limits().max_buffer_size,
                }
            })
            .map_err(|payload| {
                format!("no usable adapter: {}", super::device::panic_text(payload.as_ref()))
            })
        })
        .clone()
}

#[cfg(not(feature = "wgpu"))]
fn system_adapter() -> Result<AdapterSummary, String> {
    Err("the device library is not linked in this build".to_string())
}

/// The process-wide device-byte account for the adapter this build links.
///
/// Its budget is the adapter's largest allocation, the tightest bound the runtime reports
/// (Apple GPUs share system memory, so there is no separate VRAM figure). Every device fit in
/// the process leases from this one pool. `None` when no adapter is usable.
#[must_use]
pub fn system_device_pool() -> Option<&'static DeviceReservationPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<DeviceReservationPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        SystemProbe
            .adapter()
            .ok()
            .map(|adapter| DeviceReservationPool::fixed(adapter.max_buffer_bytes, 0))
    })
    .as_ref()
}

/// Why a device route was not taken.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FallbackReason {
    /// The build does not link the device library.
    RuntimeNotCompiled,
    /// The runtime reported no usable adapter.
    NoDevice(String),
    /// The requested arithmetic width is not available on the device.
    Dtype(Dtype),
    /// The operator or option is not implemented on the device.
    Operator(&'static str),
    /// The device account cannot hold the fit.
    DeviceMemory {
        /// Device bytes the fit needs.
        requested: u64,
        /// Device bytes available (the smaller of the pool remainder and the adapter limit).
        available: u64,
    },
    /// `auto` only: the work is below the crossover where the device pays for its copies.
    BelowCrossover {
        /// Approximate floating-point work of the fit.
        work: u64,
        /// Crossover in the same unit.
        threshold: u64,
    },
    /// The device run failed (allocation, kernel, or adapter loss) after it was chosen.
    DeviceFailure(String),
}

impl FallbackReason {
    /// Stable machine code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::RuntimeNotCompiled => "runtime_not_compiled",
            Self::NoDevice(_) => "no_device",
            Self::Dtype(_) => "unsupported_dtype",
            Self::Operator(_) => "unsupported_operator",
            Self::DeviceMemory { .. } => "device_memory",
            Self::BelowCrossover { .. } => "below_crossover",
            Self::DeviceFailure(_) => "device_failure",
        }
    }
}

impl fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeNotCompiled => f.write_str(
                "the device runtime is not linked in this build; install the accelerator \
                 distribution",
            ),
            Self::NoDevice(detail) => write!(f, "no usable device: {detail}"),
            Self::Dtype(dtype) => write!(f, "the device backend trains in f32, not {dtype}"),
            Self::Operator(what) => write!(f, "the device backend does not support {what}"),
            Self::DeviceMemory { requested, available } => {
                write!(f, "the fit needs {requested} device bytes and {available} are available")
            }
            Self::BelowCrossover { work, threshold } => write!(
                f,
                "work {work} is below the device crossover {threshold}; the cpu is faster"
            ),
            Self::DeviceFailure(detail) => write!(f, "the device run failed: {detail}"),
        }
    }
}

/// Route a fit takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `NdArray<f32>` on the host.
    Cpu,
    /// Burn WGPU on the selected adapter.
    Wgpu,
}

/// What a seeded replay of the route reproduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayScope {
    /// Same seed, same build, same host: identical bits.
    BitwiseOnHost,
    /// Same seed on the same adapter and driver: identical to within the device tolerance.
    /// Not bitwise across drivers, adapters, or against the CPU route.
    WithinToleranceOnDevice,
}

/// The route that ran, as the receipts record it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// Policy asked for.
    pub requested: ComputePolicy,
    /// Backend that trained.
    pub backend: LearnerBackend,
    /// Library that trained.
    pub library: &'static str,
    /// Device that trained: `"host-cpu"` or `"<api>:<adapter name>"`.
    pub device: String,
    /// Arithmetic width of the route.
    pub dtype: Dtype,
    /// Why a device route was not taken, when `auto` fell back.
    pub fallback: Option<FallbackReason>,
    /// What a seeded replay reproduces.
    pub replay: ReplayScope,
}

impl Provenance {
    fn cpu(requested: ComputePolicy, fallback: Option<FallbackReason>) -> Self {
        Self {
            requested,
            backend: LearnerBackend::NdArrayCpu,
            library: "burn-ndarray-0.16",
            device: "host-cpu".to_string(),
            dtype: Dtype::F32,
            fallback,
            replay: ReplayScope::BitwiseOnHost,
        }
    }

    fn wgpu(requested: ComputePolicy, adapter: &AdapterSummary) -> Self {
        Self {
            requested,
            backend: LearnerBackend::Wgpu,
            library: "burn-wgpu-0.16",
            device: format!("{}:{}", adapter.api, adapter.name),
            dtype: Dtype::F32,
            fallback: None,
            replay: ReplayScope::WithinToleranceOnDevice,
        }
    }

    /// One-line label for a learner receipt.
    ///
    /// The default CPU request keeps the plain `"burn"` label so existing receipts do not
    /// change. Every other combination names the library, device, and any fallback.
    #[must_use]
    pub fn implementation_label(&self) -> String {
        if self.requested == ComputePolicy::Cpu {
            return "burn".to_string();
        }
        let mut label = format!(
            "{} device={} dtype={} requested={}",
            self.library,
            self.device,
            self.dtype,
            self.requested.as_str()
        );
        if let Some(reason) = &self.fallback {
            label.push_str(" fallback=");
            label.push_str(reason.code());
        }
        label
    }
}

/// Smallest fit work, in [`fit_flops`] units, for which `auto` picks the device.
///
/// Measured on Apple M1 Max (Metal against the `NdArray<f32>` CPU backend): the per-fit work
/// of the smallest workload where the device end to end is at least twice as fast as the CPU
/// with the out-of-fold gap inside tolerance (40,000 rows, 32 columns, hidden 128, 15 epochs,
/// two folds). Below it, copies, kernel launches, and the fixed per-fit device cost mean the
/// CPU is faster or the gain is under the 2x rule; see `parity/accelerator_decision.md`.
pub const DEFAULT_MIN_DEVICE_WORK: u64 = 37_094_400_000;

/// Thresholds `auto` applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoPolicy {
    /// Smallest [`FitRequest::work`] for which `auto` picks the device.
    pub min_device_work: u64,
}

impl Default for AutoPolicy {
    fn default() -> Self {
        Self { min_device_work: DEFAULT_MIN_DEVICE_WORK }
    }
}

/// The decided route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Route to run.
    pub route: Route,
    /// Provenance the run will carry unless it later falls back.
    pub provenance: Provenance,
    /// Sized accounts of the fit.
    pub footprint: Footprint,
}

/// Failure of [`plan`] or [`fit_with_policy`].
#[derive(Debug)]
#[non_exhaustive]
pub enum PolicyError {
    /// A `gpu` request failed a check. Nothing ran and nothing is held.
    Refused(FallbackReason),
    /// The fit footprint overflowed.
    Footprint(admission::AdmissionRefusal),
    /// The fit itself failed or was cancelled.
    Fit(LearnBurnError),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(reason) => {
                write!(f, "gpu policy refused [{}]: {reason}", reason.code())
            }
            Self::Footprint(error) => write!(f, "{error}"),
            Self::Fit(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(_) => None,
            Self::Footprint(error) => Some(error),
            Self::Fit(error) => Some(error),
        }
    }
}

impl From<LearnBurnError> for PolicyError {
    fn from(error: LearnBurnError) -> Self {
        Self::Fit(error)
    }
}

/// First check a device route fails, or the adapter it would use.
fn device_check(
    request: &FitRequest,
    footprint: &Footprint,
    probe: &dyn DeviceProbe,
    device_pool: Option<&DeviceReservationPool>,
) -> Result<AdapterSummary, FallbackReason> {
    if request.dtype != Dtype::F32 {
        return Err(FallbackReason::Dtype(request.dtype));
    }
    if request.weighted {
        return Err(FallbackReason::Operator("sample weights"));
    }
    if !probe.compiled() {
        return Err(FallbackReason::RuntimeNotCompiled);
    }
    let adapter = probe.adapter().map_err(FallbackReason::NoDevice)?;
    if adapter.device_type == "Cpu" {
        return Err(FallbackReason::NoDevice(format!(
            "adapter {} is a software renderer",
            adapter.name
        )));
    }
    let pool_bytes = device_pool.map_or(u64::MAX, |pool| pool.available().device_bytes());
    let available = pool_bytes.min(adapter.max_buffer_bytes);
    if footprint.device_bytes > available {
        return Err(FallbackReason::DeviceMemory { requested: footprint.device_bytes, available });
    }
    Ok(adapter)
}

/// Decide the route for `request`.
///
/// # Errors
///
/// [`PolicyError::Refused`] when `request.policy` is [`ComputePolicy::Gpu`] and a check
/// fails; [`PolicyError::Footprint`] on size overflow.
pub fn plan(
    request: &FitRequest,
    probe: &dyn DeviceProbe,
    device_pool: Option<&DeviceReservationPool>,
    auto: AutoPolicy,
) -> Result<Plan, PolicyError> {
    let footprint = admission::footprint(request.nrows, request.ncols, request.hidden)
        .map_err(PolicyError::Footprint)?;
    let cpu = |fallback| Plan {
        route: Route::Cpu,
        provenance: Provenance::cpu(request.policy, fallback),
        footprint,
    };
    match request.policy {
        ComputePolicy::Cpu => Ok(cpu(None)),
        ComputePolicy::Gpu => match device_check(request, &footprint, probe, device_pool) {
            Ok(adapter) => Ok(Plan {
                route: Route::Wgpu,
                provenance: Provenance::wgpu(ComputePolicy::Gpu, &adapter),
                footprint,
            }),
            Err(reason) => Err(PolicyError::Refused(reason)),
        },
        ComputePolicy::Auto => match device_check(request, &footprint, probe, device_pool) {
            Err(reason) => Ok(cpu(Some(reason))),
            Ok(adapter) => {
                let work = request.work();
                if work < auto.min_device_work {
                    return Ok(cpu(Some(FallbackReason::BelowCrossover {
                        work,
                        threshold: auto.min_device_work,
                    })));
                }
                Ok(Plan {
                    route: Route::Wgpu,
                    provenance: Provenance::wgpu(ComputePolicy::Auto, &adapter),
                    footprint,
                })
            }
        },
    }
}

/// Inputs shared by every policy fit in a study.
#[derive(Clone, Copy)]
pub struct PolicyContext<'a> {
    /// Runtime probe.
    pub probe: &'a dyn DeviceProbe,
    /// Device-byte account shared by every concurrent fit. `None` means no account.
    pub device_pool: Option<&'a DeviceReservationPool>,
    /// `auto` thresholds.
    pub auto: AutoPolicy,
    /// Polled before every epoch. `true` cancels with no model.
    pub stop: &'a dyn Fn() -> bool,
}

/// A fit and the route that produced it.
#[derive(Clone, Debug)]
#[must_use = "the provenance names the route that ran"]
pub struct PolicyFit {
    /// Host weights. Prediction needs no device.
    pub model: TrainedMlp,
    /// Route that ran.
    pub provenance: Provenance,
    /// Stage clocks when a device fit ran. `None` on the CPU route, which reads no clock.
    pub timings: Option<StageTimings>,
}

/// Fit under `request.policy`.
///
/// # Errors
///
/// [`PolicyError::Refused`] for a refused `gpu` request or a `gpu` device failure,
/// [`PolicyError::Fit`] for input errors and cancellation. Cancellation never falls back.
pub fn fit_with_policy(
    request: &FitRequest,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
    context: &PolicyContext<'_>,
) -> Result<PolicyFit, PolicyError> {
    let plan = plan(request, context.probe, context.device_pool, context.auto)?;
    if (context.stop)() {
        return Err(PolicyError::Fit(LearnBurnError::Cancelled));
    }
    if plan.route == Route::Wgpu {
        let lease = match context.device_pool {
            Some(pool) => {
                if let Ok(lease) = pool.try_grant(plan.footprint.device_bytes, 0, false) {
                    Some(lease)
                } else {
                    let reason = FallbackReason::DeviceMemory {
                        requested: plan.footprint.device_bytes,
                        available: pool.available().device_bytes(),
                    };
                    return fall_back_or_refuse(
                        request, reason, x_colmajor, nrows, ncols, y, spec, context,
                    );
                }
            }
            None => None,
        };
        let outcome = run_device(x_colmajor, nrows, ncols, y, spec, context.stop);
        drop(lease);
        return match outcome {
            Ok((model, timings)) => {
                Ok(PolicyFit { model, provenance: plan.provenance, timings: Some(timings) })
            }
            Err(LearnBurnError::Device(detail)) => fall_back_or_refuse(
                request,
                FallbackReason::DeviceFailure(detail),
                x_colmajor,
                nrows,
                ncols,
                y,
                spec,
                context,
            ),
            Err(error) => Err(PolicyError::Fit(error)),
        };
    }
    let model = super::train_cancellable(x_colmajor, nrows, ncols, y, spec, context.stop)?;
    Ok(PolicyFit { model, provenance: plan.provenance, timings: None })
}

/// A device route failed after it was chosen: `auto` runs the CPU and records why, `gpu`
/// refuses.
#[allow(clippy::too_many_arguments, reason = "the fit inputs travel together")]
fn fall_back_or_refuse(
    request: &FitRequest,
    reason: FallbackReason,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
    context: &PolicyContext<'_>,
) -> Result<PolicyFit, PolicyError> {
    if request.policy != ComputePolicy::Auto {
        return Err(PolicyError::Refused(reason));
    }
    if (context.stop)() {
        return Err(PolicyError::Fit(LearnBurnError::Cancelled));
    }
    let model = super::train_cancellable(x_colmajor, nrows, ncols, y, spec, context.stop)?;
    Ok(PolicyFit {
        model,
        provenance: Provenance::cpu(ComputePolicy::Auto, Some(reason)),
        timings: None,
    })
}

#[cfg(feature = "wgpu")]
fn run_device(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    spec: &MlpSpec,
    stop: &dyn Fn() -> bool,
) -> Result<(TrainedMlp, StageTimings), LearnBurnError> {
    let fit =
        super::device::fit_portable_wgpu_cancellable(x_colmajor, nrows, ncols, y, spec, stop)?;
    let timings = match fit.measurement {
        super::device::Measurement::Staged(timings) => timings,
        super::device::Measurement::NotMeasured => StageTimings::default(),
    };
    Ok((fit.model, timings))
}

#[cfg(not(feature = "wgpu"))]
fn run_device(
    _x_colmajor: &[f64],
    _nrows: usize,
    _ncols: usize,
    _y: &[f64],
    _spec: &MlpSpec,
    _stop: &dyn Fn() -> bool,
) -> Result<(TrainedMlp, StageTimings), LearnBurnError> {
    Err(LearnBurnError::Device("the device library is not linked in this build".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        compiled: bool,
        adapter: Result<AdapterSummary, String>,
    }

    impl DeviceProbe for Fake {
        fn compiled(&self) -> bool {
            self.compiled
        }
        fn adapter(&self) -> Result<AdapterSummary, String> {
            self.adapter.clone()
        }
    }

    fn gpu_adapter() -> AdapterSummary {
        AdapterSummary {
            name: "Test GPU".to_string(),
            api: "Metal".to_string(),
            device_type: "IntegratedGpu".to_string(),
            max_buffer_bytes: 1 << 30,
        }
    }

    fn present() -> Fake {
        Fake { compiled: true, adapter: Ok(gpu_adapter()) }
    }

    fn spec() -> MlpSpec {
        MlpSpec { binary: false, hidden: 4, epochs: 6, learning_rate: 0.05, seed: 9 }
    }

    fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 1.0 + 0.5 * (i as f64);
        }
        (x, y)
    }

    fn request(policy: ComputePolicy, n: usize) -> FitRequest {
        FitRequest::neural(policy, &spec(), n, 2)
    }

    fn never() -> bool {
        false
    }

    fn context<'a>(
        probe: &'a dyn DeviceProbe,
        pool: Option<&'a DeviceReservationPool>,
        auto: AutoPolicy,
    ) -> PolicyContext<'a> {
        PolicyContext { probe, device_pool: pool, auto, stop: &never }
    }

    const EAGER: AutoPolicy = AutoPolicy { min_device_work: 0 };

    #[test]
    fn policy_names_round_trip_and_default_is_cpu() {
        for policy in [ComputePolicy::Cpu, ComputePolicy::Gpu, ComputePolicy::Auto] {
            assert_eq!(ComputePolicy::parse(policy.as_str()), Ok(policy));
        }
        assert_eq!(ComputePolicy::default(), ComputePolicy::Cpu);
        assert_eq!(ComputePolicy::parse("cuda"), Err("cuda".to_string()));
    }

    #[test]
    fn the_disable_switch_hides_the_device_only_for_set_values() {
        assert!(device_disabled(Some("1")).unwrap().contains(DISABLE_DEVICE_ENV));
        assert!(device_disabled(Some("yes")).is_some());
        for off in [None, Some(""), Some("0")] {
            assert_eq!(device_disabled(off), None);
        }
    }

    #[test]
    fn cpu_policy_never_probes_the_device() {
        struct Tripwire;
        impl DeviceProbe for Tripwire {
            fn compiled(&self) -> bool {
                panic!("cpu policy must not ask whether the device is compiled");
            }
            fn adapter(&self) -> Result<AdapterSummary, String> {
                panic!("cpu policy must not probe an adapter");
            }
        }
        let plan = plan(&request(ComputePolicy::Cpu, 16), &Tripwire, None, EAGER).unwrap();
        assert_eq!(plan.route, Route::Cpu);
        assert_eq!(plan.provenance.fallback, None);
        assert_eq!(plan.provenance.backend, LearnerBackend::NdArrayCpu);
        assert_eq!(plan.provenance.replay, ReplayScope::BitwiseOnHost);
        assert_eq!(plan.provenance.implementation_label(), "burn");
    }

    #[test]
    fn gpu_policy_refuses_each_failed_check_with_its_reason() {
        let absent = Fake { compiled: true, adapter: Err("no adapter".to_string()) };
        let not_linked = Fake { compiled: false, adapter: Ok(gpu_adapter()) };
        let software = Fake {
            compiled: true,
            adapter: Ok(AdapterSummary { device_type: "Cpu".to_string(), ..gpu_adapter() }),
        };
        let available = present();
        let cases: [(&dyn DeviceProbe, FitRequest, &str); 5] = [
            (
                &not_linked as &dyn DeviceProbe,
                request(ComputePolicy::Gpu, 16),
                "runtime_not_compiled",
            ),
            (&absent as &dyn DeviceProbe, request(ComputePolicy::Gpu, 16), "no_device"),
            (&software as &dyn DeviceProbe, request(ComputePolicy::Gpu, 16), "no_device"),
            (
                &available as &dyn DeviceProbe,
                FitRequest { dtype: Dtype::F64, ..request(ComputePolicy::Gpu, 16) },
                "unsupported_dtype",
            ),
            (
                &available as &dyn DeviceProbe,
                FitRequest { weighted: true, ..request(ComputePolicy::Gpu, 16) },
                "unsupported_operator",
            ),
        ];
        for (probe, req, code) in cases {
            match plan(&req, probe, None, EAGER) {
                Err(PolicyError::Refused(reason)) => assert_eq!(reason.code(), code),
                other => panic!("expected a {code} refusal, got {other:?}"),
            }
        }
        let message = plan(&request(ComputePolicy::Gpu, 16), &not_linked, None, EAGER)
            .unwrap_err()
            .to_string();
        assert!(message.contains("accelerator distribution"), "{message}");
    }

    #[test]
    fn gpu_policy_accepts_a_present_adapter_and_labels_the_device() {
        let plan = plan(&request(ComputePolicy::Gpu, 16), &present(), None, EAGER).unwrap();
        assert_eq!(plan.route, Route::Wgpu);
        assert_eq!(plan.provenance.backend, LearnerBackend::Wgpu);
        assert_eq!(plan.provenance.device, "Metal:Test GPU");
        assert_eq!(plan.provenance.library, "burn-wgpu-0.16");
        assert_eq!(plan.provenance.replay, ReplayScope::WithinToleranceOnDevice);
        let label = plan.provenance.implementation_label();
        assert!(label.contains("burn-wgpu-0.16") && label.contains("Metal:Test GPU"), "{label}");
        assert!(!label.contains("fallback"));
    }

    #[test]
    fn auto_policy_falls_back_with_the_reason_and_never_labels_cpu_work_as_gpu() {
        let absent = Fake { compiled: true, adapter: Err("no adapter".to_string()) };
        let planned = plan(&request(ComputePolicy::Auto, 16), &absent, None, EAGER).unwrap();
        assert_eq!(planned.route, Route::Cpu);
        assert_eq!(planned.provenance.backend, LearnerBackend::NdArrayCpu);
        assert_eq!(planned.provenance.device, "host-cpu");
        assert_eq!(planned.provenance.requested, ComputePolicy::Auto);
        assert_eq!(
            planned.provenance.fallback,
            Some(FallbackReason::NoDevice("no adapter".to_string()))
        );
        let label = planned.provenance.implementation_label();
        assert!(label.contains("host-cpu") && label.contains("fallback=no_device"), "{label}");
        assert!(!label.contains("wgpu"));

        let below =
            plan(&request(ComputePolicy::Auto, 16), &present(), None, AutoPolicy::default())
                .unwrap();
        assert_eq!(below.route, Route::Cpu);
        assert!(matches!(below.provenance.fallback, Some(FallbackReason::BelowCrossover { .. })));

        let at_threshold = AutoPolicy { min_device_work: request(ComputePolicy::Auto, 16).work() };
        let at = plan(&request(ComputePolicy::Auto, 16), &present(), None, at_threshold).unwrap();
        assert_eq!(at.route, Route::Wgpu, "work equal to the threshold takes the device");
        let past = AutoPolicy { min_device_work: at_threshold.min_device_work + 1 };
        let under = plan(&request(ComputePolicy::Auto, 16), &present(), None, past).unwrap();
        assert_eq!(under.route, Route::Cpu, "work one below the threshold stays on the cpu");

        let above = plan(&request(ComputePolicy::Auto, 16), &present(), None, EAGER).unwrap();
        assert_eq!(above.route, Route::Wgpu);
        assert_eq!(above.provenance.fallback, None);
    }

    #[test]
    fn device_memory_is_checked_against_the_pool_and_the_adapter() {
        let req = request(ComputePolicy::Gpu, 64);
        let fp = admission::footprint(64, 2, 4).unwrap();
        let small = DeviceReservationPool::fixed(fp.device_bytes - 1, 0);
        match plan(&req, &present(), Some(&small), EAGER) {
            Err(PolicyError::Refused(FallbackReason::DeviceMemory { requested, available })) => {
                assert_eq!(requested, fp.device_bytes);
                assert_eq!(available, fp.device_bytes - 1);
            }
            other => panic!("expected a device-memory refusal, got {other:?}"),
        }
        assert_eq!(small.available().device_bytes(), fp.device_bytes - 1);
        let tiny_adapter = Fake {
            compiled: true,
            adapter: Ok(AdapterSummary { max_buffer_bytes: 8, ..gpu_adapter() }),
        };
        assert!(matches!(
            plan(&req, &tiny_adapter, None, EAGER),
            Err(PolicyError::Refused(FallbackReason::DeviceMemory { .. }))
        ));
        let roomy = DeviceReservationPool::fixed(fp.device_bytes, 0);
        assert!(plan(&req, &present(), Some(&roomy), EAGER).is_ok());
        let auto =
            plan(&request(ComputePolicy::Auto, 64), &present(), Some(&small), EAGER).unwrap();
        assert_eq!(auto.route, Route::Cpu);
        assert_eq!(
            auto.provenance.fallback.as_ref().map(FallbackReason::code),
            Some("device_memory")
        );
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the fallback must replay the cpu fit bit for bit")]
    fn auto_fallback_fit_equals_the_plain_cpu_fit() {
        let n = 20;
        let (x, y) = data(n);
        let absent = Fake { compiled: false, adapter: Err("n/a".to_string()) };
        let fit = fit_with_policy(
            &request(ComputePolicy::Auto, n),
            &x,
            n,
            2,
            &y,
            &spec(),
            &context(&absent, None, EAGER),
        )
        .unwrap();
        assert_eq!(fit.provenance.backend, LearnerBackend::NdArrayCpu);
        assert!(fit.timings.is_none());
        let plain = crate::train(&x, n, 2, &y, &spec()).unwrap();
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        fit.model.predict(&x, n, 2, &mut a).unwrap();
        plain.predict(&x, n, 2, &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn gpu_request_in_a_cpu_only_build_refuses_without_fitting() {
        let n = 20;
        let (x, y) = data(n);
        let not_linked = Fake { compiled: false, adapter: Err("n/a".to_string()) };
        let error = fit_with_policy(
            &request(ComputePolicy::Gpu, n),
            &x,
            n,
            2,
            &y,
            &spec(),
            &context(&not_linked, None, EAGER),
        )
        .unwrap_err();
        assert!(matches!(error, PolicyError::Refused(FallbackReason::RuntimeNotCompiled)));
    }

    #[cfg(not(feature = "wgpu"))]
    #[test]
    fn a_device_failure_after_selection_falls_back_for_auto_and_refuses_for_gpu() {
        let n = 20;
        let (x, y) = data(n);
        let fp = admission::footprint(n, 2, 4).unwrap();
        let pool = DeviceReservationPool::fixed(fp.device_bytes, 0);
        // The probe claims a device, but this build links none, so the run fails after the
        // lease was granted. The lease must come back either way.
        let claims = present();
        let ctx = context(&claims, Some(&pool), EAGER);
        let fit =
            fit_with_policy(&request(ComputePolicy::Auto, n), &x, n, 2, &y, &spec(), &ctx).unwrap();
        assert_eq!(fit.provenance.backend, LearnerBackend::NdArrayCpu);
        assert_eq!(
            fit.provenance.fallback.as_ref().map(FallbackReason::code),
            Some("device_failure")
        );
        assert_eq!(pool.available().device_bytes(), fp.device_bytes);
        let error = fit_with_policy(&request(ComputePolicy::Gpu, n), &x, n, 2, &y, &spec(), &ctx)
            .unwrap_err();
        assert!(matches!(error, PolicyError::Refused(FallbackReason::DeviceFailure(_))));
        assert_eq!(pool.available().device_bytes(), fp.device_bytes);
    }

    #[test]
    fn a_lease_held_elsewhere_makes_auto_fall_back_and_gpu_refuse() {
        let n = 20;
        let (x, y) = data(n);
        let fp = admission::footprint(n, 2, 4).unwrap();
        let pool = DeviceReservationPool::fixed(fp.device_bytes, 0);
        let other = pool.try_grant(fp.device_bytes, 0, false).unwrap();
        let claims = present();
        let ctx = context(&claims, Some(&pool), EAGER);
        let auto =
            fit_with_policy(&request(ComputePolicy::Auto, n), &x, n, 2, &y, &spec(), &ctx).unwrap();
        assert_eq!(auto.provenance.backend, LearnerBackend::NdArrayCpu);
        assert_eq!(
            auto.provenance.fallback.as_ref().map(FallbackReason::code),
            Some("device_memory")
        );
        let gpu = fit_with_policy(&request(ComputePolicy::Gpu, n), &x, n, 2, &y, &spec(), &ctx);
        assert!(matches!(gpu, Err(PolicyError::Refused(FallbackReason::DeviceMemory { .. }))));
        drop(other);
        assert_eq!(pool.available().device_bytes(), fp.device_bytes);
    }

    #[test]
    fn cancellation_never_falls_back() {
        let n = 20;
        let (x, y) = data(n);
        let claims = present();
        let stop = || true;
        let ctx = PolicyContext { probe: &claims, device_pool: None, auto: EAGER, stop: &stop };
        for policy in [ComputePolicy::Cpu, ComputePolicy::Gpu, ComputePolicy::Auto] {
            let error =
                fit_with_policy(&request(policy, n), &x, n, 2, &y, &spec(), &ctx).unwrap_err();
            assert!(matches!(error, PolicyError::Fit(LearnBurnError::Cancelled)), "{policy:?}");
        }
    }

    #[cfg(feature = "wgpu")]
    mod device {
        use super::*;
        use crate::staged::{Workload, max_gap_in_target_sd};

        /// Largest tolerated gap between device and CPU out-of-fold predictions, in target
        /// standard deviations.
        const DEVICE_VS_CPU_TOLERANCE_SD: f64 = 0.05;

        #[test]
        fn device_fit_matches_the_cpu_fit_within_tolerance_and_reports_stage_clocks() {
            let Ok(_adapter) = SystemProbe.adapter() else {
                eprintln!("no adapter on this host; the device acceptance test is skipped");
                return;
            };
            let workload = Workload::CONTROL;
            let (x, y) = workload.data();
            let n = workload.nrows;
            let spec = MlpSpec {
                binary: false,
                hidden: workload.hidden,
                epochs: workload.epochs,
                learning_rate: workload.learning_rate,
                seed: workload.seed,
            };
            let ctx =
                PolicyContext { probe: &SystemProbe, device_pool: None, auto: EAGER, stop: &never };
            let cpu = fit_with_policy(
                &FitRequest::neural(ComputePolicy::Cpu, &spec, n, workload.ncols),
                &x,
                n,
                workload.ncols,
                &y,
                &spec,
                &ctx,
            )
            .unwrap();
            let gpu = fit_with_policy(
                &FitRequest::neural(ComputePolicy::Gpu, &spec, n, workload.ncols),
                &x,
                n,
                workload.ncols,
                &y,
                &spec,
                &ctx,
            )
            .unwrap();
            assert_eq!(gpu.provenance.backend, LearnerBackend::Wgpu);
            let clocks = gpu.timings.expect("device fit reports stage clocks");
            assert!(clocks.fit > std::time::Duration::ZERO);
            assert!(clocks.upload > std::time::Duration::ZERO);
            let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
            cpu.model.predict(&x, n, workload.ncols, &mut a).unwrap();
            gpu.model.predict(&x, n, workload.ncols, &mut b).unwrap();
            let gap = max_gap_in_target_sd(&a, &b, &y);
            assert!(gap < DEVICE_VS_CPU_TOLERANCE_SD, "device/cpu gap {gap} target SDs");
            // A seeded device replay is within tolerance of itself.
            let again = fit_with_policy(
                &FitRequest::neural(ComputePolicy::Gpu, &spec, n, workload.ncols),
                &x,
                n,
                workload.ncols,
                &y,
                &spec,
                &ctx,
            )
            .unwrap();
            let mut c = vec![0.0; n];
            again.model.predict(&x, n, workload.ncols, &mut c).unwrap();
            assert!(max_gap_in_target_sd(&b, &c, &y) < DEVICE_VS_CPU_TOLERANCE_SD);
        }

        #[test]
        fn device_memory_lease_is_returned_after_a_device_fit() {
            let Ok(_adapter) = SystemProbe.adapter() else {
                return;
            };
            let n = 64;
            let (x, y) = data(n);
            let fp = admission::footprint(n, 2, 4).unwrap();
            let pool = DeviceReservationPool::fixed(fp.device_bytes, 0);
            let ctx = PolicyContext {
                probe: &SystemProbe,
                device_pool: Some(&pool),
                auto: EAGER,
                stop: &never,
            };
            let fit = fit_with_policy(&request(ComputePolicy::Gpu, n), &x, n, 2, &y, &spec(), &ctx)
                .unwrap();
            assert_eq!(fit.provenance.backend, LearnerBackend::Wgpu);
            assert_eq!(pool.available().device_bytes(), fp.device_bytes);
        }
    }
}
