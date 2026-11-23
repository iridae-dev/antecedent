# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pooled multi-environment lagged frame with space and time dummies, by direct construction.

Regenerate with:  uv run --script jpcmci_plus_pooled_frame_reference.py > jpcmci_plus_pooled_frame_reference.json

Three environments of different lengths hold variables v0, v1 and an unrequested v2.  The frame
requests v0 and v1 at depth d, so each variable has d + 1 lag slots and each environment
contributes its rows d..n_e - 1 (no window crosses an environment boundary).  The column of
variable slot s and lag l has, at the row of time t = d + j of environment e, the value
series_e[var][t - l].  A space dummy has M - 1 columns (one per environment but the last, the
reference), the k-th equal to 1 on the rows of environment k and 0 elsewhere; a time dummy is the
raw time index t, or one-hot over the sorted distinct raw times except the largest, which is the
reference.  Every dummy is repeated on all d + 1 lag slots after the observed block, and the
synthetic variable ids start past both the requested variables and the whole schema (3 here).
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random


def build(envs, variables, depth, space, time):
    n_lags = depth + 1
    rows = []
    env_rows = []
    for e, series in enumerate(envs):
        n = len(series[0])
        for j in range(n - depth):
            rows.append((e, depth + j))
        env_rows.append(n - depth)
    cols = []
    for v in variables:
        for lag in range(n_lags):
            cols.append([envs[e][v][t - lag] for e, t in rows])
    ids = {"space": [], "time": []}
    cursor = max(max(variables) + 1, len(envs[0]))
    dummy_cols = []
    if space and len(envs) > 1:
        for k in range(len(envs) - 1):
            ids["space"].append(cursor)
            cursor += 1
            dummy_cols.append([1.0 if e == k else 0.0 for e, _ in rows])
    if time == "integer":
        ids["time"].append(cursor)
        cursor += 1
        dummy_cols.append([float(t) for _, t in rows])
    elif time == "one_hot":
        levels = sorted({t for _, t in rows})
        if len(levels) > 1:
            for level in levels[:-1]:
                ids["time"].append(cursor)
                cursor += 1
                dummy_cols.append([1.0 if t == level else 0.0 for _, t in rows])
    for col in dummy_cols:
        for _ in range(n_lags):
            cols.append(col)
    return {"columns": cols, "env_rows": env_rows, "space_ids": ids["space"], "time_ids": ids["time"]}


def main():
    rng = random.Random(404)
    lengths = [12, 15, 9]
    envs = [[[round(rng.gauss(0, 1), 6) for _ in range(n)] for _ in range(3)] for n in lengths]
    out = {"lengths": lengths, "envs": envs, "variables": [0, 1], "cases": []}
    for depth in (2, 4):
        for name, space, time in [("none", False, None), ("space", True, None), ("space_time_integer", True, "integer"),
                                  ("space_time_one_hot", True, "one_hot"), ("time_one_hot_only", False, "one_hot")]:
            out["cases"].append({"name": name, "depth": depth, "space": space, "time": time,
                                 **build(envs, [0, 1], depth, space, time)})
    print(json.dumps(out))


main()
