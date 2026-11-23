//! Resource and quality acceptance for the compute policy: shared CPU/RAM/device admission,
//! cancellation cleanup, fold concurrency, physical prediction order, numerical agreement,
//! and the workload where the CPU is the right choice.

use std::sync::atomic::{AtomicUsize, Ordering};

use antecedent_core::device_reservation::{
    CpuAdmissionError, DeviceReservationPool, place_physical_predictions,
};
use antecedent_core::execution::AdmissionPool;
use antecedent_core::{CancellationToken, ExecutionContext};
use antecedent_learn_burn::MlpSpec;
use antecedent_learn_burn::admission::{AdmissionRefusal, admit_fit, footprint};
use antecedent_learn_burn::device::LearnerBackend;
use antecedent_learn_burn::policy::{
    AdapterSummary, AutoPolicy, ComputePolicy, DeviceProbe, FallbackReason, FitRequest,
    PolicyContext, PolicyError, Route, fit_with_policy, plan,
};
use antecedent_learn_burn::staged::{Workload, max_gap_in_target_sd};

/// Largest gap between two f32 networks fit from the same seed on the same host, in target
/// standard deviations. Same-host CPU replays are bitwise, so this is exactly zero.
const SAME_HOST_REPLAY_SD: f64 = 0.0;

/// A device the test claims exists. The device library is not required: the policy layer is
/// what is under test.
struct Claimed;

impl DeviceProbe for Claimed {
    fn compiled(&self) -> bool {
        true
    }
    fn adapter(&self) -> Result<AdapterSummary, antecedent_learn_burn::policy::AdapterError> {
        Ok(AdapterSummary {
            name: "Claimed GPU".to_string(),
            api: "Metal".to_string(),
            device_type: "IntegratedGpu".to_string(),
            max_buffer_bytes: 1 << 32,
        })
    }
}

fn never() -> bool {
    false
}

fn spec(seed: u64) -> MlpSpec {
    MlpSpec { binary: false, hidden: 6, epochs: 12, learning_rate: 0.05, seed }
}

/// Column-major design (intercept, one ramp) and a linear target.
fn ramp(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![1.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        x[n + i] = i as f64 / n as f64;
        y[i] = 2.0 + 3.0 * x[n + i];
    }
    (x, y)
}

fn cpu_context<'a>(probe: &'a dyn DeviceProbe, stop: &'a dyn Fn() -> bool) -> PolicyContext<'a> {
    PolicyContext { probe, device_pool: None, auto: AutoPolicy::default(), stop }
}

fn predictions(model: &antecedent_learn_burn::TrainedMlp, x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    model.predict(x, n, 2, &mut out).unwrap();
    out
}

#[test]
fn concurrent_folds_never_exceed_the_shared_cpu_slots_and_replay_the_serial_fits() {
    const FOLDS: usize = 5;
    const SLOTS: u32 = 2;
    let n = 40;
    let (x, y) = ramp(n);
    let fp = footprint(n, 2, 6).unwrap();
    let pool = AdmissionPool::fixed(SLOTS, 100 * (fp.host_bytes + fp.scratch_bytes));
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let device = DeviceReservationPool::fixed(0, 0);

    let serial: Vec<Vec<f64>> = (0..FOLDS)
        .map(|fold| {
            let fit = fit_with_policy(
                &FitRequest::neural(ComputePolicy::Cpu, &spec(fold as u64), n, 2),
                &x,
                n,
                2,
                &y,
                &spec(fold as u64),
                &cpu_context(&Claimed, &never),
            )
            .unwrap();
            predictions(&fit.model, &x, n)
        })
        .collect();

    let running = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let refusals = AtomicUsize::new(0);
    let concurrent: Vec<(usize, Vec<f64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..FOLDS)
            .map(|fold| {
                let (ctx, device, x, y) = (&ctx, &device, &x, &y);
                let (running, peak, refusals) = (&running, &peak, &refusals);
                scope.spawn(move || {
                    let admission = loop {
                        match admit_fit(ctx, device, Route::Cpu, &fp) {
                            Ok(admission) => break admission,
                            Err(AdmissionRefusal::Cpu(CpuAdmissionError::FoldExceedsLease {
                                ..
                            })) => {
                                refusals.fetch_add(1, Ordering::SeqCst);
                                std::thread::yield_now();
                            }
                            Err(other) => panic!("unexpected refusal {other}"),
                        }
                    };
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    let fit = fit_with_policy(
                        &FitRequest::neural(ComputePolicy::Cpu, &spec(fold as u64), n, 2),
                        x,
                        n,
                        2,
                        y,
                        &spec(fold as u64),
                        &cpu_context(&Claimed, &never),
                    )
                    .unwrap();
                    running.fetch_sub(1, Ordering::SeqCst);
                    drop(admission);
                    (fold, predictions(&fit.model, x, n))
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });

    assert!(peak.load(Ordering::SeqCst) <= SLOTS as usize, "peak {}", peak.load(Ordering::SeqCst));
    for (fold, got) in concurrent {
        let gap = max_gap_in_target_sd(&got, &serial[fold], &y);
        assert!((gap - SAME_HOST_REPLAY_SD).abs() < f64::EPSILON, "fold {fold} gap {gap}");
        for (a, b) in got.iter().zip(&serial[fold]) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    assert_eq!(pool.available().threads(), SLOTS);
    assert_eq!(pool.available().bytes(), 100 * (fp.host_bytes + fp.scratch_bytes));
}

#[test]
fn cancellation_mid_fit_returns_cpu_ram_scratch_and_device_holds() {
    let n = 40;
    let (x, y) = ramp(n);
    let fp = footprint(n, 2, 6).unwrap();
    let pool = AdmissionPool::fixed(2, 10 * (fp.host_bytes + fp.scratch_bytes));
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let device = DeviceReservationPool::fixed(fp.device_bytes, 0);

    let mut admission = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap();
    assert_eq!(admission.held_device_bytes(), fp.device_bytes);
    assert_eq!(pool.available().threads(), 1);

    let token: CancellationToken = ctx.cancellation.clone();
    let polls = AtomicUsize::new(0);
    let stop = || {
        if polls.fetch_add(1, Ordering::SeqCst) == 4 {
            token.cancel();
        }
        token.is_cancelled()
    };
    let error = fit_with_policy(
        &FitRequest::neural(ComputePolicy::Cpu, &spec(1), n, 2),
        &x,
        n,
        2,
        &y,
        &spec(1),
        &cpu_context(&Claimed, &stop),
    )
    .unwrap_err();
    assert!(matches!(error, PolicyError::Fit(antecedent_learn_burn::LearnBurnError::Cancelled)));
    assert!(admission.observe_cancel(&ctx.cancellation));
    assert_eq!(admission.held_device_bytes(), 0);
    drop(admission);
    assert_eq!(pool.available().threads(), 2);
    assert_eq!(pool.available().bytes(), 10 * (fp.host_bytes + fp.scratch_bytes));
    assert_eq!(device.available().device_bytes(), fp.device_bytes);
    assert!(matches!(
        admit_fit(&ctx, &device, Route::Wgpu, &fp),
        Err(AdmissionRefusal::Cpu(CpuAdmissionError::Cancelled))
    ));
    assert_eq!(device.available().device_bytes(), fp.device_bytes);
}

#[test]
fn device_oom_refusal_leaves_every_account_unchanged_and_auto_runs_on_the_cpu() {
    let n = 40;
    let (x, y) = ramp(n);
    let fp = footprint(n, 2, 6).unwrap();
    let pool = AdmissionPool::fixed(2, 10 * (fp.host_bytes + fp.scratch_bytes));
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let device = DeviceReservationPool::fixed(fp.device_bytes - 1, 0);
    let before = pool.available();

    let refused = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap_err();
    assert!(matches!(refused, AdmissionRefusal::Device(_)));
    assert_eq!(pool.available(), before);
    assert_eq!(device.available().device_bytes(), fp.device_bytes - 1);

    let eager = AutoPolicy { min_device_work: 0 };
    let policy_ctx =
        PolicyContext { probe: &Claimed, device_pool: Some(&device), auto: eager, stop: &never };
    let auto = fit_with_policy(
        &FitRequest::neural(ComputePolicy::Auto, &spec(2), n, 2),
        &x,
        n,
        2,
        &y,
        &spec(2),
        &policy_ctx,
    )
    .unwrap();
    assert_eq!(auto.provenance.backend, LearnerBackend::NdArrayCpu);
    assert_eq!(auto.provenance.device, "host-cpu");
    assert!(matches!(auto.provenance.fallback, Some(FallbackReason::DeviceMemory { .. })));
    let gpu = fit_with_policy(
        &FitRequest::neural(ComputePolicy::Gpu, &spec(2), n, 2),
        &x,
        n,
        2,
        &y,
        &spec(2),
        &policy_ctx,
    );
    assert!(matches!(gpu, Err(PolicyError::Refused(FallbackReason::DeviceMemory { .. }))));
    assert_eq!(device.available().device_bytes(), fp.device_bytes - 1);
}

#[test]
fn out_of_fold_predictions_are_placed_by_physical_row_whatever_order_folds_finish() {
    let workload = Workload { nrows: 60, folds: 3, epochs: 6, hidden: 4, ..Workload::SMALL };
    let (x, y) = workload.data();
    let n = workload.nrows;
    let cols = workload.ncols;
    let mut per_fold: Vec<(Vec<u64>, Vec<f64>)> = Vec::new();
    for fold in 0..workload.folds {
        let train: Vec<usize> = (0..n).filter(|r| r % workload.folds != fold).collect();
        let held: Vec<usize> = (0..n).filter(|r| r % workload.folds == fold).collect();
        let gather = |rows: &[usize]| {
            let mut out = vec![0.0; rows.len() * cols];
            for c in 0..cols {
                for (slot, &r) in rows.iter().enumerate() {
                    out[c * rows.len() + slot] = x[c * n + r];
                }
            }
            out
        };
        let train_y: Vec<f64> = train.iter().map(|&r| y[r]).collect();
        let fold_spec = MlpSpec {
            binary: false,
            hidden: workload.hidden,
            epochs: workload.epochs,
            learning_rate: workload.learning_rate,
            seed: workload.seed + fold as u64,
        };
        let request = FitRequest::neural(ComputePolicy::Cpu, &fold_spec, train.len(), cols);
        let fit = fit_with_policy(
            &request,
            &gather(&train),
            train.len(),
            cols,
            &train_y,
            &fold_spec,
            &cpu_context(&Claimed, &never),
        )
        .unwrap();
        let mut out = vec![0.0; held.len()];
        fit.model.predict(&gather(&held), held.len(), cols, &mut out).unwrap();
        per_fold.push((held.iter().map(|&r| r as u64).collect(), out));
    }
    let forward: Vec<(&[u64], &[f64])> =
        per_fold.iter().map(|(r, v)| (r.as_slice(), v.as_slice())).collect();
    let reverse: Vec<(&[u64], &[f64])> = forward.iter().rev().copied().collect();
    let placed = place_physical_predictions(n, &forward).unwrap();
    assert_eq!(placed, place_physical_predictions(n, &reverse).unwrap());

    // The staged cross-fit reaches the same physical vector through a different code path.
    let run = antecedent_learn_burn::accelerator_probe::probe_cpu_workload(&workload).unwrap();
    for (a, b) in placed.iter().zip(&run.physical) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn the_cpu_fit_recovers_a_linear_target_within_a_declared_tolerance() {
    /// Largest in-sample gap to the truth, in target standard deviations.
    const LINEAR_FIT_TOLERANCE_SD: f64 = 0.15;
    let n = 64;
    let (x, y) = ramp(n);
    let spec = MlpSpec { binary: false, hidden: 8, epochs: 150, learning_rate: 0.05, seed: 3 };
    let fit = fit_with_policy(
        &FitRequest::neural(ComputePolicy::Cpu, &spec, n, 2),
        &x,
        n,
        2,
        &y,
        &spec,
        &cpu_context(&Claimed, &never),
    )
    .unwrap();
    let gap = max_gap_in_target_sd(&predictions(&fit.model, &x, n), &y, &y);
    assert!(gap < LINEAR_FIT_TOLERANCE_SD, "gap {gap} target SDs");
}

#[test]
fn a_small_workload_is_routed_to_the_cpu_even_when_a_device_is_present() {
    let w = Workload::SMALL;
    let spec = MlpSpec {
        binary: false,
        hidden: w.hidden,
        epochs: w.epochs,
        learning_rate: w.learning_rate,
        seed: w.seed,
    };
    let train_rows = w.nrows - w.nrows / w.folds;
    let request = FitRequest::neural(ComputePolicy::Auto, &spec, train_rows, w.ncols);
    let planned = plan(&request, &Claimed, None, AutoPolicy::default()).unwrap();
    assert_eq!(planned.route, Route::Cpu);
    match planned.provenance.fallback {
        Some(FallbackReason::BelowCrossover { work, threshold }) => assert!(work < threshold),
        other => panic!("expected a below-crossover fallback, got {other:?}"),
    }
    // The large workload clears the same threshold, so the small case is a real boundary.
    let big = Workload::LARGE;
    let big_spec = MlpSpec { hidden: big.hidden, epochs: big.epochs, ..spec };
    let big_request = FitRequest::neural(
        ComputePolicy::Auto,
        &big_spec,
        big.nrows - big.nrows / big.folds,
        big.ncols,
    );
    let big_plan = plan(&big_request, &Claimed, None, AutoPolicy::default()).unwrap();
    assert_eq!(big_plan.route, Route::Wgpu);
    assert_eq!(big_plan.provenance.fallback, None);
}

#[cfg(feature = "wgpu")]
mod device {
    use antecedent_learn_burn::accelerator_probe::{probe_cpu_workload, probe_wgpu_workload};
    use antecedent_learn_burn::policy::SystemProbe;

    use super::*;

    /// Largest device-versus-CPU gap in out-of-fold predictions, in target standard
    /// deviations.
    const DEVICE_VS_CPU_SD: f64 = 0.05;

    #[test]
    fn the_cpu_wins_the_small_workload_end_to_end_and_the_device_agrees_numerically() {
        if SystemProbe.adapter().is_err() {
            eprintln!("no adapter on this host; the device comparison is skipped");
            return;
        }
        let w = Workload::SMALL;
        let median = |f: &dyn Fn() -> std::time::Duration| {
            let mut runs: Vec<_> = (0..5).map(|_| f()).collect();
            runs.sort();
            runs[runs.len() / 2]
        };
        // Warm both runtimes once so the comparison excludes one-time shader compilation.
        let cpu_once = probe_cpu_workload(&w).unwrap();
        let device_once = probe_wgpu_workload(&w).unwrap();
        let cpu = median(&|| probe_cpu_workload(&w).unwrap().wall);
        let device = median(&|| probe_wgpu_workload(&w).unwrap().wall);
        assert!(cpu < device, "cpu {cpu:?} should beat the device {device:?} on a small fold");
        let (_, y) = w.data();
        let gap = max_gap_in_target_sd(&cpu_once.physical, &device_once.physical, &y);
        assert!(gap < DEVICE_VS_CPU_SD, "device/cpu gap {gap} target SDs");
    }
}
