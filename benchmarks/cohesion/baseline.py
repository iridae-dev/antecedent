"""Release-extension lifecycle baseline; no before/after claim.

Python allocations and process high-water RSS are recorded separately. Native
allocator measurements live in the companion native-release.json report.
"""

import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "cohesion_example", ROOT / "examples/python/cohesive_ml_transport.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import antecedent as ac
from antecedent.prediction import FittedEffectModel


def measure(fn, repeats=15):
    times = []
    fn()
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - start) / 1e6)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    allocations = sum(item.count for item in snapshot.statistics("filename"))
    tracemalloc.stop()
    return dict(
        median_ms=statistics.median(times),
        p95_ms=sorted(times)[int(0.95 * (len(times) - 1))],
        python_peak_bytes=peak,
        python_retained_allocations=allocations,
        native_allocations=None,
        repeats=repeats,
    )


from antecedent import transport as tr


def grid_preparer():
    graph = ac.Admg.from_edges(["x", "y"], [("x", "y")])
    identified = tr.identify_classical(
        graph,
        tr.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
    )
    variables = [tr.VariableCoordinate(n, "binary") for n in ("x", "y")]
    catalog = tr.EvidenceCatalog(
        environments=[
            tr.Environment("source", variables),
            tr.Environment("target", variables),
        ],
        regimes=[
            tr.EvidenceRegime(
                "trial",
                "source",
                kind="experimental",
                interventions=["x"],
                measured=["y"],
            )
        ],
        bindings=[
            tr.RegimeBinding(
                "trial", "s", sampling="independent", dependence="independent_studies"
            )
        ],
    )
    data = tr.ExactTransportData(
        tuple(
            tr.ExactDiscreteLaw(
                "source",
                "trial",
                (("y", (0.0, 1.0)),),
                (1 - p, p),
                "s",
                interventions=(("x", float(i)),),
            )
            for i, p in enumerate((0.2, 0.8))
        )
    )
    request = tr.TransportResponseGridQuery(
        identified, catalog, ({"x": 0.0}, {"x": 1.0})
    )
    return lambda: tr.prepare(request, data)


prepare_grid = grid_preparer()
grid = prepare_grid()
grid_result = grid.estimate()
assert [grid_result.mean(i, "y") for i in range(2)] == [0.2, 0.8]

study, result, model = module.ml_example()
trial, trial_result = module.transport_example()
blob = model.export()
workloads = {
    "ml_prepare_fit_export_load": module.ml_example,
    "ml_cold_preparation": module.ml_prepare,
    "grid_cold_preparation": prepare_grid,
    "grid_repeated_evaluation": grid.estimate,
    "grid_inspection": grid.inspect,
    "grid_export": grid_result.export,
    "grid_load": lambda: ac.load(grid_result.export()),
    "ml_repeated_estimation_nuisance_reuse": study.estimate,
    "ml_inspection": study.inspect,
    "loaded_model_prediction": lambda: model.predict({"z": [-1.0, 0.0, 1.0] * 100}),
    "model_load": lambda: FittedEffectModel.load(blob),
    "trial_prepare_and_bootstrap": module.transport_example,
    "trial_repeated_joint_bootstrap": trial.estimate,
    "trial_inspection": trial.inspect,
    "trial_export": trial_result.export,
    "trial_load": lambda: ac.load(trial_result.export()),
}
report = dict(
    platform=platform.platform(),
    python=platform.python_version(),
    cpu_count=os.cpu_count(),
    ml_threads=1,
    transport_threads="ExecutionContext.production_default",
    extension_profile="release (build separately with maturin develop --release)",
    comparison="current working tree only; no speedup estimate",
    workloads={name: measure(fn) for name, fn in workloads.items()},
)
report["process_peak_rss_bytes"] = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
print(json.dumps(report, indent=2))
