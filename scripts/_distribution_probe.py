"""Probe run INSIDE a clean virtual environment by ``distribution_acceptance.py``.

It imports the installed ``antecedent``, runs the neural learner under each device policy,
checks the outcome the distribution under test must produce, and prints one JSON report.
It imports nothing from the repository.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import antecedent as ac
import numpy as np
from antecedent.estimators import DML, DRLearner
from antecedent.learners import Logistic, NeuralNet
from antecedent.prediction import FittedEffectModel

PREDICT_AT = {"z": [-1.0, 0.0, 1.0]}


def data(n: int, seed: int = 19) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = rng.binomial(1, 0.5, size=n).astype(float)
    y = 2.0 * t + z + rng.normal(size=n) * 0.3
    return {"z": z, "t": t, "y": y}


def analyze(device: str, *, n: int = 160, hidden: int = 4, epochs: int = 3):
    return ac.analyze(
        data(n),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect(treatment="t", outcome="y"),
        estimator=DML(
            outcome=NeuralNet(hidden=hidden, epochs=epochs, learning_rate=0.01, device=device),
            treatment=Logistic(),
            folds=2,
        ),
        bootstrap=0,
        refute=False,
        seed=3,
    )


def analyze_portable(device: str):
    """The same neural analysis through the learner that keeps a portable fitted effect."""
    return ac.analyze(
        data(160),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect(treatment="t", outcome="y"),
        estimator=DRLearner(
            outcome=NeuralNet(hidden=4, epochs=3, learning_rate=0.01, device=device),
            treatment=Logistic(),
            folds=2,
        ),
        bootstrap=0,
        refute=False,
        seed=3,
    )


def labels(result) -> set[str]:
    return {impl for spec, impl, _ in result.estimate.learner_provenance if spec == "neural_net"}


def refusal(device: str) -> str | None:
    try:
        analyze(device)
    except Exception as error:  # the refusal is the observation
        return f"{type(error).__name__}: {error}"
    return None


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, ok: bool, detail: object = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    @property
    def ok(self) -> bool:
        return all(item["ok"] for item in self.checks)


def common_checks(report: Report, expect: str, version: str) -> None:
    root = Path(sys.prefix).resolve()
    where = Path(ac.__file__).resolve()
    report.check("imports_from_the_clean_environment", root in where.parents, where)
    report.check(
        "version_matches_the_wheel", ac.__version__ == version.split("+")[0], ac.__version__
    )
    cpu = analyze("cpu")
    report.check("cpu_default_label_is_plain", labels(cpu) == {"burn"}, labels(cpu))
    auto = analyze("auto")
    (label,) = labels(auto)
    report.check(
        "auto_small_fit_stays_on_the_cpu_and_says_why",
        "device=host-cpu" in label and "fallback=" in label and "wgpu" not in label,
        label,
    )
    report.check(
        "auto_fallback_replays_the_cpu_estimate",
        auto.estimate.ate == cpu.estimate.ate,
        (auto.estimate.ate, cpu.estimate.ate),
    )


def cpu_wheel_checks(report: Report) -> None:
    error = refusal("gpu")
    report.check(
        "gpu_is_refused_with_the_missing_runtime",
        error is not None and "runtime_not_compiled" in error,
        error,
    )


def accelerator_checks(report: Report) -> None:
    disabled = os.environ.get("ANTECEDENT_DISABLE_DEVICE", "") not in ("", "0")
    error = refusal("gpu")
    if disabled:
        report.check(
            "gpu_is_refused_when_the_device_is_hidden",
            error is not None and "no_device" in error,
            error,
        )
        (label,) = labels(analyze("auto"))
        report.check("auto_records_no_device", "fallback=no_device" in label, label)
        return
    if error is not None and "no_device" in error:
        report.check("adapter_present", False, f"accelerator wheel but no adapter: {error}")
        return
    report.check("gpu_fit_runs", error is None, error)
    cpu = analyze("cpu", n=2000, hidden=16, epochs=20)
    gpu = analyze("gpu", n=2000, hidden=16, epochs=20)
    (label,) = labels(gpu)
    report.check(
        "gpu_label_names_library_and_device",
        "burn-wgpu" in label and "requested=gpu" in label and "fallback" not in label,
        label,
    )
    se_cpu, se_gpu = cpu.estimate.se_analytic, gpu.estimate.se_analytic
    gap = abs(gpu.estimate.ate - cpu.estimate.ate)
    report.check(
        "downstream_effect_agrees_within_half_a_standard_error",
        gap < 0.5 * se_cpu,
        {"cpu": cpu.estimate.ate, "gpu": gpu.estimate.ate, "se_cpu": se_cpu},
    )
    report.check(
        "downstream_uncertainty_agrees_within_a_quarter",
        abs(se_gpu - se_cpu) < 0.25 * se_cpu,
        {"se_cpu": se_cpu, "se_gpu": se_gpu},
    )
    report.check(
        "both_estimates_cover_the_known_effect",
        abs(cpu.estimate.ate - 2.0) < 3 * se_cpu and abs(gpu.estimate.ate - 2.0) < 3 * se_gpu,
        {"truth": 2.0},
    )
    big = analyze("auto", n=200000, hidden=128, epochs=15)
    (big_label,) = labels(big)
    report.check(
        "auto_selects_the_device_for_a_large_fit",
        "burn-wgpu" in big_label and "requested=auto" in big_label and "fallback" not in big_label,
        big_label,
    )


def write_artifact(path: Path) -> None:
    model = analyze_portable("cpu").fitted_model
    exported = model.export()
    binary = isinstance(exported, bytes)
    body = base64.b64encode(exported).decode() if binary else exported
    expected = model.predict({k: np.array(v) for k, v in PREDICT_AT.items()})
    path.write_text(
        json.dumps({"binary": binary, "artifact": body, "expected": list(expected.values)}),
        encoding="utf-8",
    )


def consume_artifact(report: Report, path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = base64.b64decode(payload["artifact"]) if payload["binary"] else payload["artifact"]
    model = FittedEffectModel.load(raw)
    got = list(model.predict({k: np.array(v) for k, v in PREDICT_AT.items()}).values)
    report.check(
        "artifact_from_the_other_distribution_predicts_identically",
        got == payload["expected"],
        {"got": got, "expected": payload["expected"]},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("cpu", "accelerator"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--artifact-in", type=Path)
    args = parser.parse_args()
    report = Report()
    common_checks(report, args.expect, args.version)
    (cpu_wheel_checks if args.expect == "cpu" else accelerator_checks)(report)
    if args.artifact_out:
        write_artifact(args.artifact_out)
        report.check("artifact_written", args.artifact_out.exists(), args.artifact_out)
    if args.artifact_in:
        consume_artifact(report, args.artifact_in)
    print(json.dumps({"expect": args.expect, "ok": report.ok, "checks": report.checks}))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
