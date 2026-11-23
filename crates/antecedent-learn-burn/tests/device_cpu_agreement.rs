//! The device and CPU backends start from the same host-drawn weights and see the same
//! full-batch data in the same order, so their out-of-fold predictions agree to float noise.
#![cfg(feature = "wgpu")]

use antecedent_learn_burn::accelerator_probe::{probe_cpu_workload, probe_wgpu_workload};
use antecedent_learn_burn::policy::{DeviceProbe, SystemProbe};
use antecedent_learn_burn::staged::{Workload, max_gap_in_target_sd};

/// Acceptance tolerance in target standard deviations.
const DEVICE_VS_CPU_SD: f64 = 0.05;

fn gap(workload: &Workload) -> f64 {
    let (_, y) = workload.data();
    let cpu = probe_cpu_workload(workload).unwrap();
    let gpu = probe_wgpu_workload(workload).unwrap();
    max_gap_in_target_sd(&cpu.physical, &gpu.physical, &y)
}

#[test]
fn device_and_cpu_out_of_fold_predictions_agree_on_small_and_control() {
    if SystemProbe.adapter().is_err() {
        eprintln!("no adapter on this host; skipped");
        return;
    }
    for workload in [Workload::SMALL, Workload::CONTROL] {
        let g = gap(&workload);
        eprintln!("{} gap {g:e} target SD", workload.name);
        assert!(g <= DEVICE_VS_CPU_SD, "{}: device/cpu gap {g} target SDs", workload.name);
    }
}
