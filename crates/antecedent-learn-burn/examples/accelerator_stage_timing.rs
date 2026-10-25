//! Stage timings for the accelerator candidates on this host.
//!
//! Runs deterministic cross-fitted workloads through `antecedent_learn_burn::staged` on the
//! `NdArray<f32>` CPU backend and, when built with `--features wgpu`, on Burn's WGPU backend
//! (Metal on Apple). Build with `--features accelerate` to put Apple Accelerate BLAS under
//! the ndarray backend; the report labels the CPU backend accordingly.
//!
//! Every number is a wall clock of work that ran: prepare, upload, fit, synchronize,
//! download, predict, and output. `cold` is the first run (device runtime and shader
//! compilation included); `warm` is the per-stage median of the remaining repeats.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p antecedent-learn-burn --features wgpu \
//!     --example accelerator_stage_timing -- [--repeats N] [--workloads small,control,large] \
//!     [--sweep] [--out PATH]
//! ```
//!
//! The output is TOML. `--out` also writes it to a file.

use std::fmt::Write as _;
use std::time::Duration;

use antecedent_learn_burn::accelerator_probe::probe_cpu_workload;
use antecedent_learn_burn::policy::{DeviceProbe, SystemProbe};
use antecedent_learn_burn::staged::{
    STAGE_NAMES, StageTimings, Workload, WorkloadRun, max_gap_in_target_sd,
};

/// The frozen gain requirement: the device end to end must be at least this many times
/// faster than the CPU control, transfer included.
const GAIN_THRESHOLD: f64 = 2.0;

/// Largest tolerated device-vs-CPU out-of-fold gap, in target standard deviations.
const DEVICE_VS_CPU_SD: f64 = 0.05;

struct Backend {
    label: String,
    run: fn(&Workload) -> Result<WorkloadRun, antecedent_learn_burn::LearnBurnError>,
}

struct Measured {
    backend: String,
    workload: Workload,
    cold: StageTimings,
    cold_wall: Duration,
    warm: StageTimings,
    warm_wall: Duration,
    physical: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut repeats = 5usize;
    let mut names = vec!["small".to_string(), "control".to_string(), "large".to_string()];
    let mut sweep = false;
    let mut out: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--repeats" => repeats = args.next().ok_or("--repeats needs a value")?.parse()?,
            "--workloads" => {
                names = args
                    .next()
                    .ok_or("--workloads needs a value")?
                    .split(',')
                    .map(str::to_string)
                    .collect();
            }
            "--sweep" => sweep = true,
            "--out" => out = Some(args.next().ok_or("--out needs a value")?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if repeats < 2 {
        return Err("--repeats must be at least 2 (one cold run and one warm run)".into());
    }
    let mut workloads: Vec<Workload> = Vec::new();
    for name in &names {
        workloads.push(match name.as_str() {
            "small" => Workload::SMALL,
            "control" => Workload::CONTROL,
            "large" => Workload::LARGE,
            other => return Err(format!("unknown workload {other}").into()),
        });
    }
    if sweep {
        workloads.extend(sweep_workloads());
    }

    let mut backends = vec![Backend {
        label: if cfg!(feature = "accelerate") {
            "ndarray_accelerate".into()
        } else {
            "ndarray".into()
        },
        run: probe_cpu_workload_fn,
    }];
    #[cfg(feature = "wgpu")]
    backends.push(Backend { label: "wgpu".into(), run: probe_wgpu_workload_fn });

    let mut report = String::new();
    writeln!(report, "# accelerator stage timings")?;
    writeln!(report, "[host]")?;
    writeln!(report, "os = \"{}\"", std::env::consts::OS)?;
    writeln!(report, "arch = \"{}\"", std::env::consts::ARCH)?;
    writeln!(
        report,
        "logical_cpus = {}",
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
    )?;
    writeln!(report, "repeats = {repeats}")?;
    writeln!(report, "gain_threshold = {GAIN_THRESHOLD}")?;
    match SystemProbe.adapter() {
        Ok(adapter) => {
            writeln!(report, "adapter = \"{}\"", adapter.name)?;
            writeln!(report, "adapter_api = \"{}\"", adapter.api)?;
            writeln!(report, "adapter_type = \"{}\"", adapter.device_type)?;
            writeln!(report, "adapter_max_buffer_bytes = {}", adapter.max_buffer_bytes)?;
        }
        Err(reason) => writeln!(report, "adapter_unavailable = {reason:?}")?,
    }

    let mut all: Vec<Measured> = Vec::new();
    for workload in &workloads {
        for backend in &backends {
            let measured = measure(backend, workload, repeats)?;
            write_measurement(&mut report, &measured)?;
            all.push(measured);
        }
    }
    write_comparisons(&mut report, &all, &backends)?;

    print!("{report}");
    if let Some(path) = out {
        std::fs::write(path, &report)?;
    }
    Ok(())
}

fn probe_cpu_workload_fn(
    workload: &Workload,
) -> Result<WorkloadRun, antecedent_learn_burn::LearnBurnError> {
    probe_cpu_workload(workload)
}

#[cfg(feature = "wgpu")]
fn probe_wgpu_workload_fn(
    workload: &Workload,
) -> Result<WorkloadRun, antecedent_learn_burn::LearnBurnError> {
    antecedent_learn_burn::accelerator_probe::probe_wgpu_workload(workload)
}

/// Hidden-width sweep at fixed rows: the smallest width where the device wins is the crossover.
fn sweep_workloads() -> Vec<Workload> {
    const WIDTHS: [(usize, &str); 8] = [
        (8, "sweep_h008"),
        (16, "sweep_h016"),
        (32, "sweep_h032"),
        (64, "sweep_h064"),
        (128, "sweep_h128"),
        (160, "sweep_h160"),
        (192, "sweep_h192"),
        (256, "sweep_h256"),
    ];
    WIDTHS
        .iter()
        .map(|&(hidden, name)| Workload {
            name,
            nrows: 20_000,
            ncols: 16,
            folds: 2,
            hidden,
            epochs: 10,
            learning_rate: 0.01,
            seed: 41,
        })
        .collect()
}

fn measure(
    backend: &Backend,
    workload: &Workload,
    repeats: usize,
) -> Result<Measured, Box<dyn std::error::Error>> {
    let cold_run = (backend.run)(workload)?;
    let mut warm: Vec<WorkloadRun> = Vec::new();
    for _ in 1..repeats {
        warm.push((backend.run)(workload)?);
    }
    let stage = |pick: fn(&StageTimings) -> Duration| {
        median(warm.iter().map(|run| pick(&run.timings)).collect())
    };
    let warm_timings = StageTimings {
        prepare: stage(|t| t.prepare),
        upload: stage(|t| t.upload),
        fit: stage(|t| t.fit),
        synchronize: stage(|t| t.synchronize),
        download: stage(|t| t.download),
        predict: stage(|t| t.predict),
        output: stage(|t| t.output),
    };
    let warm_wall = median(warm.iter().map(|run| run.wall).collect());
    let physical = warm.last().map(|run| run.physical.clone()).unwrap_or_default();
    Ok(Measured {
        backend: backend.label.clone(),
        workload: *workload,
        cold: cold_run.timings,
        cold_wall: cold_run.wall,
        warm: warm_timings,
        warm_wall,
        physical,
    })
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort();
    values.get(values.len() / 2).copied().unwrap_or(Duration::ZERO)
}

fn write_stage_table(
    report: &mut String,
    key: &str,
    timings: &StageTimings,
    wall: Duration,
) -> std::fmt::Result {
    writeln!(report, "[measurement.{key}]")?;
    for (name, nanos) in STAGE_NAMES.iter().zip(timings.nanos()) {
        writeln!(report, "{name}_ns = {nanos}")?;
    }
    writeln!(report, "stage_total_ns = {}", timings.total().as_nanos())?;
    writeln!(report, "wall_ns = {}", wall.as_nanos())
}

fn write_measurement(report: &mut String, m: &Measured) -> std::fmt::Result {
    writeln!(report, "\n[[measurement]]")?;
    writeln!(report, "backend = \"{}\"", m.backend)?;
    writeln!(report, "workload = \"{}\"", m.workload.name)?;
    writeln!(report, "rows = {}", m.workload.nrows)?;
    writeln!(report, "cols = {}", m.workload.ncols)?;
    writeln!(report, "folds = {}", m.workload.folds)?;
    writeln!(report, "hidden = {}", m.workload.hidden)?;
    writeln!(report, "epochs = {}", m.workload.epochs)?;
    writeln!(report, "flops = {}", m.workload.flops())?;
    write_stage_table(report, "cold", &m.cold, m.cold_wall)?;
    write_stage_table(report, "warm", &m.warm, m.warm_wall)
}

fn write_comparisons(
    report: &mut String,
    all: &[Measured],
    backends: &[Backend],
) -> std::fmt::Result {
    let Some(cpu_label) = backends.first().map(|b| b.label.as_str()) else {
        return Ok(());
    };
    let mut crossover: Option<(u64, &str)> = None;
    let mut promoted: Option<(u64, &str)> = None;
    for cpu in all.iter().filter(|m| m.backend == cpu_label) {
        for device in
            all.iter().filter(|m| m.backend != cpu_label && m.workload.name == cpu.workload.name)
        {
            let speedup = cpu.warm_wall.as_secs_f64() / device.warm_wall.as_secs_f64().max(1e-12);
            let (_, y) = cpu.workload.data();
            let gap = max_gap_in_target_sd(&cpu.physical, &device.physical, &y);
            writeln!(report, "\n[[comparison]]")?;
            writeln!(report, "workload = \"{}\"", cpu.workload.name)?;
            writeln!(report, "cpu_backend = \"{}\"", cpu.backend)?;
            writeln!(report, "device_backend = \"{}\"", device.backend)?;
            writeln!(report, "cpu_warm_wall_ns = {}", cpu.warm_wall.as_nanos())?;
            writeln!(report, "device_warm_wall_ns = {}", device.warm_wall.as_nanos())?;
            writeln!(report, "speedup = {speedup:.4}")?;
            writeln!(report, "meets_gain_threshold = {}", speedup >= GAIN_THRESHOLD)?;
            writeln!(report, "device_wins = {}", speedup > 1.0)?;
            writeln!(report, "max_oof_gap_target_sd = {gap:.6}")?;
            let transfer = device.warm.upload + device.warm.download + device.warm.synchronize;
            writeln!(report, "device_transfer_and_sync_ns = {}", transfer.as_nanos())?;
            if speedup > 1.0 && crossover.is_none_or(|(flops, _)| cpu.workload.flops() < flops) {
                crossover = Some((cpu.workload.flops(), cpu.workload.name));
            }
            if speedup >= GAIN_THRESHOLD
                && gap <= DEVICE_VS_CPU_SD
                && promoted.is_none_or(|(flops, _)| cpu.workload.flops() < flops)
            {
                promoted = Some((cpu.workload.flops(), cpu.workload.name));
            }
        }
    }
    writeln!(report, "\n[crossover]")?;
    match crossover {
        Some((flops, name)) => {
            writeln!(report, "smallest_device_win_flops = {flops}")?;
            writeln!(report, "workload = \"{name}\"")
        }
        None => writeln!(report, "smallest_device_win_flops = \"none\""),
    }?;
    // The auto policy compares per-fit work, so the promotion point is reported per fit.
    match promoted {
        Some((flops, name)) => {
            let folds = all.iter().find(|m| m.workload.name == name).map_or(1, |m| m.workload.folds);
            writeln!(report, "smallest_gain_threshold_flops = {flops}")?;
            writeln!(report, "smallest_gain_threshold_per_fit_flops = {}", flops / folds as u64)?;
            writeln!(report, "smallest_gain_threshold_workload = \"{name}\"")
        }
        None => writeln!(report, "smallest_gain_threshold_flops = \"none\""),
    }
}
