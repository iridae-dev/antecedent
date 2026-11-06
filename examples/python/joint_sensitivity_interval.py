#!/usr/bin/env python3
"""Export a replayable sampling interval for a joint mechanism sensitivity range, then replay it.

Question: a transported effect rests on a source study whose outcome model and shared parent
distribution might deviate a little from the truth in the target. The joint mechanism sensitivity
analysis bounds the effect over a declared box of such deviations. That range is computed from
a sample, so how far could sampling noise move it, and can a second party check that number
without trusting the first?

Data: a small simulated binary transport problem with a counted source law (n = 80) built in the
script. The numbers are properties of that declared model.

The asserts check that the exported artifact replays independently to the same range and
interval, that the same seed gives the same bytes and a different seed gives different bootstrap
draws, that the current calibration status is derived by the replay rather than copied from the
producer, and that an edited artifact, a replay under tighter limits and an export from an
unestimated stage all refuse.

The result does not claim a confidence interval for the whole sensitivity set. It is a
conservative endpoint bootstrap that targets a fixed true deviation inside the declared box.
Replay shows the numbers follow from the stored premises, not that the premises are right.
"""

from __future__ import annotations

# %%
import json

import antecedent as ant
from antecedent import Admg
from antecedent.errors import CausalError, CausalUnsupportedError
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport

SEED = 7
REPLICATES = 199
LEVEL = 0.95
NAMES = ["w", "z", "x", "y"]
P_W = {0: 0.65, 1: 0.35}
P_Y1 = {(0, 0): 0.2, (0, 1): 0.5, (1, 0): 0.3, (1, 1): 0.8}
SAMPLE_SIZE = 80


def refused(call) -> CausalError:
    """Run ``call`` and return the typed error it raises; anything else is a failure."""
    try:
        call()
    except CausalError as error:
        return error
    raise AssertionError("the call should have been refused")


# %% [markdown]
# ## Build a counted transport problem
#
# The graph says the shared parent `w` affects the outcome `y`, and `z` is the controllable
# input that was set to zero in the source study. The source law is a table of eighty counted
# observations drawn to match the declared probabilities, so it carries sampling uncertainty.

# %%
graph = Admg.from_edges(
    NAMES,
    [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
    [("w", "y"), ("z", "y"), ("z", "x")],
)
query = transport.ZTransportQuery(
    transport.SelectionDiagram("source", "target", []),
    outcomes=["y"],
    treatments=["x"],
    controllable=["z"],
    experiment_assignment={"z": 0.0},
)
builder = transport.identify_z_transport(graph=graph, query=query)
assert builder.outcome == "identified"

catalog = transport.EvidenceCatalog(
    environments=(
        transport.Environment(
            "source", tuple(transport.VariableCoordinate(name, "binary") for name in NAMES)
        ),
    ),
    regimes=(
        transport.EvidenceRegime(
            "do_z_0",
            "source",
            kind="experimental",
            interventions=["z"],
            intervention_values={"z": 0.0},
            measured=NAMES,
        ),
    ),
    bindings=(
        transport.RegimeBinding(
            "do_z_0",
            "snapshot_z0",
            schema_names=NAMES,
            sampling="independent",
            dependence="independent_studies",
        ),
    ),
)

probabilities = []
for w in (0, 1):
    for x in (0, 1):
        for y in (0, 1):
            p_y = P_Y1[(w, x)] if y else 1.0 - P_Y1[(w, x)]
            probabilities.append(P_W[w] * (0.4 if x else 0.6) * p_y)
counts = [round(p * SAMPLE_SIZE) for p in probabilities]
counts[-1] += SAMPLE_SIZE - sum(counts)
law = transport.ExactDiscreteLaw(
    "source",
    "do_z_0",
    (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
    tuple(c / SAMPLE_SIZE for c in counts),
    "snapshot_z0",
    interventions=(("z", 0.0),),
    empirical_counts=tuple(counts),
)

# %% [markdown]
# ## Run the baseline and declare the deviation box
#
# The baseline estimate must run first: the artifact exports that executed baseline. The box
# lets the outcome model deviate by at most 0.2 and the shared parent distribution by at most 0.3.
# The range over that box is exact for the sample; the interval below adds sampling noise to it.

# %%
prepared = builder.prepare_exact(catalog, (law,), {"x": 1.0})
plan = json.loads(require_success(prepared.estimate()))
assert plan["status"] == "available"

box = transport.JointDeviation(
    {"outcome_kernel": 0.2, "shared_parent_marginal": 0.3},
    decision_threshold=0.5,
)
joint = transport.joint_mechanism_sensitivity(prepared, box)
plug_in = joint["assumption_range"]
assert joint["inference_claim"] == "assumption_range"
print(f"baseline effect {joint['baseline']:.3f}")
print(f"assumption range over the box [{plug_in['minimum']:.3f}, {plug_in['maximum']:.3f}]")

# %% [markdown]
# ## Export the interval artifact
#
# The export runs the endpoint bootstrap once, with a declared seed, replicate count and level,
# and seals the box, the seed and every replicate in portable bytes. The same seed gives the same
# bytes. Another seed draws different replicates.

# %%
artifact = transport.export_joint_sensitivity_interval(
    prepared, box, seed=SEED, replicates=REPLICATES, level=LEVEL
)
again = transport.export_joint_sensitivity_interval(
    prepared, box, seed=SEED, replicates=REPLICATES, level=LEVEL
)
other = transport.export_joint_sensitivity_interval(
    prepared, box, seed=SEED + 1, replicates=REPLICATES, level=LEVEL
)
assert bytes(artifact) == bytes(again)
print(f"artifact: {len(bytes(artifact))} bytes")

# %% [markdown]
# ## Replay it independently
#
# Replay needs only the bytes. It replays the baseline and the range, reruns the seeded bootstrap
# and compares every number to the stored ones. If anything differs it refuses rather than
# returning a result. The replayed interval must contain the plug-in range, because it is a
# conservative bound on each endpoint.

# %%
del prepared, builder  # the replay below uses nothing but the bytes
read = artifact.replay()
assert read.acceptance.replayed
assert read["seed"] == SEED
assert read["numerical"]["level"] == LEVEL

requested, successful, failed = read["numerical"]["counts"]
assert requested == REPLICATES and successful + failed == requested
lower, upper = read["numerical"]["endpoints"]
print(f"declared box {read['deviation']}")
print(f"seed {read['seed']}, level {read['numerical']['level']}")
print(f"replicates requested {requested}, successful {successful}, failed {failed}")
print(f"endpoint interval [{lower:.3f}, {upper:.3f}]")
print(f"coverage target: {read['numerical']['coverage_target']}")

assert lower < plug_in["minimum"] < plug_in["maximum"] < upper
twice = artifact.replay()
assert twice["numerical"] == read["numerical"]
assert again.replay()["numerical"] == read["numerical"]
assert other.replay()["numerical"]["baselines"] != read["numerical"]["baselines"]
# The generic loader reads the same bytes through the same consumer.
assert require_success(ant.load(artifact)).acceptance.replayed

# %% [markdown]
# ## Read the calibration status
#
# The producer stores the status it saw as history. The replay derives the status again from the
# replayed construction and the coverage records that exist now. An edited historical status
# cannot promote the interval.

# %%
status = read["calibration"]["status"]
print(f"producer recorded: {read['recorded_calibration']['status']}")
print(f"current status: {status}")
assert status in {"calibrated", "outside_measured_scope", "measured_below_nominal", "not_measured"}

# %% [markdown]
# ## See what refuses
#
# Editing the bytes breaks the content digest. A replay under limits tighter than the stored
# request refuses before any work, and so does exporting from a stage whose baseline has not
# run. None of these returns a number.

# %%
edited = bytearray(bytes(artifact))
edited[-3] ^= 0x01
error = refused(lambda: transport.consume_joint_sensitivity_interval(bytes(edited)))
print("edited bytes:", type(error).__name__, getattr(error, "reason_code", None))

error = refused(lambda: transport.consume_joint_sensitivity_interval(bytes(artifact)[:-7]))
print("truncated bytes:", type(error).__name__)

error = refused(lambda: artifact.replay(max_replicates=REPLICATES - 1))
assert isinstance(error, CausalUnsupportedError) and error.reason_code == "route_not_supported"
print("tighter limit:", error.reason_code)

unestimated = transport.identify_z_transport(graph=graph, query=query).prepare_exact(
    catalog, (law,), {"x": 1.0}
)
error = refused(
    lambda: transport.export_joint_sensitivity_interval(
        unestimated, box, seed=SEED, replicates=REPLICATES
    )
)
print("unestimated stage:", type(error).__name__)
