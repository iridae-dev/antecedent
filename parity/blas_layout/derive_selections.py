#!/usr/bin/env python3
"""Derive `[[selection]]` blocks of layout.toml from its measured `[[case]]` records.

Rule, applied to every (op, layout, rows, cols, nonzeros_per_column) group:

* the winner is the configuration with the lowest median;
* a simpler configuration (reference before faer, then fewer workers) replaces the winner
  when its median is within `HYSTERESIS` of the winner's, so a parallel scope is chosen only
  when it earns more than noise.

Usage:
    derive_selections.py layout.toml [--cases more_cases.toml ...] [--qr-crossover crossover.toml]
        [--write] [--write-rust crates/antecedent-stats/src/layout_measured.rs]

`--cases` merges `[[case]]` records emitted by `cargo bench -p antecedent-stats --bench
blas_layout` (stdout, or the file named by `ANTECEDENT_LAYOUT_RECORD`). `--qr-crossover`
merges the serial and parallel means of the column-pivoted QR record as `qr` cases. With
`--write` the file is rewritten: every case, then the derived selections. Without it the
selections print to stdout. `--write-rust` also regenerates the rule table the crate compiles
from the derived selections.
"""

import sys
import tomllib
from collections import defaultdict

HYSTERESIS = 0.03
PRODUCT_RANK = {"reference": 0, "faer": 1}


def key_of(case):
    return (
        case.get("op", "gram"),
        case["layout"],
        case["rows"],
        case["cols"],
        case.get("nonzeros_per_column", 0),
    )


def config_of(case):
    return (case["product"], case["workers"])


def pick(cases):
    best = min(cases, key=lambda c: c["median_ns"])
    limit = best["median_ns"] * (1.0 + HYSTERESIS)
    simple = [c for c in cases if c["median_ns"] <= limit]
    chosen = min(simple, key=lambda c: (PRODUCT_RANK[c["product"]], c["workers"]))
    return best, chosen


def qr_cases(path):
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    workers = data.get("parallel_workers", data.get("workers", 10))
    out = []
    for design in data["design"]:
        base = {
            "op": "qr",
            "layout": "dense",
            "rows": design["rows"],
            "cols": design["cols"],
            "product": "faer",
        }
        for count, field in ((1, "serial_ns"), (workers, "parallel_ns")):
            value = design[field]
            out.append(
                dict(base, workers=count, median_ns=value, low_ns=value, high_ns=value)
            )
    return out


def render_case(case):
    lines = ["[[case]]"]
    for field in (
        "op",
        "layout",
        "rows",
        "cols",
        "nonzeros_per_column",
        "product",
        "workers",
        "median_ns",
        "low_ns",
        "high_ns",
        "scratch_bytes",
    ):
        if field in case:
            value = case[field]
            lines.append(f'{field} = "{value}"' if isinstance(value, str) else f"{field} = {value}")
    return "\n".join(lines)


def render_selection(group, best, chosen, cases):
    op, layout, rows, cols, nnz = group
    lines = [
        "[[selection]]",
        f'op = "{op}"',
        f'layout = "{layout}"',
        f"rows = {rows}",
        f"cols = {cols}",
    ]
    if nnz:
        lines.append(f"nonzeros_per_column = {nnz}")
    lines.append(f'product = "{chosen["product"]}"')
    lines.append(f'workers = {chosen["workers"]}')
    runner = sorted(
        (c for c in cases if config_of(c) != config_of(chosen)),
        key=lambda c: c["median_ns"],
    )
    if runner:
        ratio = runner[0]["median_ns"] / chosen["median_ns"]
        contrast = (
            f'next best {runner[0]["product"]} at {runner[0]["workers"]} workers is '
            f"{ratio:.2f}x the chosen median"
        )
    else:
        contrast = "only configuration measured"
    lines.append(
        f'reason = "lowest median within {HYSTERESIS:.0%} after preferring the simpler '
        f'configuration; {contrast}"'
    )
    return "\n".join(lines)


RUST_OPS = {
    "gram": "Gram",
    "qr": "ColumnPivotedQr",
    "svd": "ThinSvd",
    "eigen": "SymmetricEigen",
}


def rust_row(group, chosen):
    op, layout, rows, cols, nnz = group
    product = "Reference" if chosen["product"] == "reference" else "Faer"
    workers = chosen["workers"]
    if layout == "sparse":
        return f"    sparse({rows}, {cols}, {nnz}, Product::{product}, {workers}),"
    helper = "dense_rowmajor" if layout == "dense_rowmajor" else "dense"
    return f"    {helper}(LayoutOp::{RUST_OPS[op]}, {rows}, {cols}, Product::{product}, {workers}),"


def main(argv):
    args = argv[1:]
    write = "--write" in args
    args = [a for a in args if a != "--write"]
    rust_path = None
    if "--write-rust" in args:
        at = args.index("--write-rust")
        rust_path = args[at + 1]
        del args[at : at + 2]
    path = args.pop(0)
    extra, qr = [], []
    while args:
        flag = args.pop(0)
        if flag == "--cases":
            extra.append(args.pop(0))
        elif flag == "--qr-crossover":
            qr.append(args.pop(0))
        else:
            raise SystemExit(f"unknown argument {flag}")
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    cases = list(data.get("case", []))
    for case in cases:
        case.setdefault("op", "gram")
    for extra_path in extra:
        with open(extra_path, "rb") as handle:
            more = tomllib.load(handle).get("case", [])
        for case in more:
            case.setdefault("op", "gram")
        cases.extend(more)
    for qr_path in qr:
        cases.extend(qr_cases(qr_path))
    seen = {}
    for case in cases:
        seen[key_of(case) + config_of(case)] = case
    cases = list(seen.values())
    groups = defaultdict(list)
    for case in cases:
        groups[key_of(case)].append(case)
    ordered = sorted(groups, key=lambda g: (g[0], g[1], g[2] * g[3] * g[3], g[2], g[3]))
    selections = []
    rust_rows = []
    for group in ordered:
        best, chosen = pick(groups[group])
        selections.append(render_selection(group, best, chosen, groups[group]))
        rust_rows.append(rust_row(group, chosen))
    if rust_path:
        with open(rust_path, "w", encoding="utf-8") as handle:
            handle.write(
                "pub(crate) const MEASURED: &[MeasuredPoint] = &[\n"
                + "\n".join(rust_rows)
                + "\n];\n"
            )
    if not write:
        print("\n\n".join(selections))
        return
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    head = text.split("\n[[case]]", 1)[0].rstrip() + "\n"
    ordered_cases = sorted(
        cases,
        key=lambda c: (
            c["op"],
            c["layout"],
            c["rows"] * c["cols"] * c["cols"],
            c["rows"],
            c["cols"],
            PRODUCT_RANK[c["product"]],
            c["workers"],
        ),
    )
    body = "\n\n".join(render_case(c) for c in ordered_cases)
    footer = (
        "# Selections are derived from the cases above by derive_selections.py:\n"
        "# the lowest median wins, and a simpler configuration within noise replaces it.\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(head + "\n" + body + "\n\n" + footer + "\n" + "\n\n".join(selections) + "\n")


if __name__ == "__main__":
    main(sys.argv)
