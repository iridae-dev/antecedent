#!/usr/bin/env python3
"""Generate the licensed rows of the counterfactual and nested-effect validation cells on a DAG.

The rows are a deterministic function of the table below. They live between two marker lines of
parity/support_licensed.toml; `--check` fails when the file differs from the generated block and
`--write` replaces (or appends) the block.

    python3 scripts/generate_cross_world_dag_support_rows.py --check
    python3 scripts/generate_cross_world_dag_support_rows.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# BEGIN generated rows: scripts/generate_cross_world_dag_support_rows.py"
END = "# END generated rows: scripts/generate_cross_world_dag_support_rows.py"

FIXTURE = "conformance/estimate/cross_world_dag_validation"
TEST = "crates/antecedent/tests/cross_world_dag_validation_truth.rs"

FAMILIES = [
    {
        "query": "Counterfactual",
        "estimator": "gcm.fit",
        "structures": ["explicit", "accepted"],
        "validations": {"explicit": ["cheap", "full"], "accepted": ["cheap", "full"]},
        "evidence": "counterfactual_dag_cheap_and_full_validation_refutes_the_structural_slope",
        "checked": "counterfactual_dag_validation_executes_from_the_retained_plan_without_the_builder",
        "calibration_reason": {"Frequentist": "no_interval_reported", "Bayesian": "estimator_grid_not_measured"},
        "limitations": (
            "Mean unit effect of the fitted structural model on a supplied or accepted DAG, validated "
            "by the functional refuter state: the placebo treatment (the permuted treatment removes the "
            "mechanism the abducted noise is pushed through), the cross-world sensitivity range, "
            "subset stability and, under full, the placebo outcome and the bootstrap stability of the "
            "whole procedure. Known truth: the closed form of the generating outcome equation "
            "(conformance/estimate/cross_world_dag_validation) gives a mean unit effect of 3 for the "
            "contrast 0 to 1.5; a planted null slope is flagged by the cross-world sensitivity refuter "
            "and its clean twin passes; on Gaussian disturbances the Bayesian posterior-predictive check "
            "passes the clean case. Subset stability of a homogeneous linear effect has a constant "
            "target across subsets and is reported as a refuter that did not run, not asserted. The "
            "verdicts are diagnostics, not a calibration claim. The licence of the point and of the unit-effect "
            "intervals is that of the corresponding validation none cell; the interval's calibration "
            "is not measured."
        ),
    },
    {
        "query": "NestedCounterfactualEffect",
        "estimator": "mediation.linear",
        "structures": ["explicit", "accepted"],
        "validations": {"explicit": ["cheap", "full"], "accepted": ["none", "cheap", "full"]},
        "evidence": "nested_dag_validation_refutes_the_structural_direct_effect",
        "checked": "nested_dag_validation_executes_from_the_retained_plan_without_the_builder",
        "calibration_reason": {"Frequentist": "no_interval_reported", "Bayesian": "no_interval_reported"},
        "limitations": (
            "Natural direct effect Y(active, M(control)) - Y(control, M(control)) on a supplied or "
            "accepted DAG under the shared-exogenous linear-Gaussian route, point only. Validation runs "
            "the functional refuter state: the placebo treatment (which removes the direct and indirect "
            "paths the cross-world functional reads), the mediator placebo (which returns the total "
            "effect), the cross-world sensitivity range, subset stability and, under full, the placebo "
            "outcome and the bootstrap stability of the whole procedure. Known truth: the closed forms "
            "of the generating equations (conformance/estimate/cross_world_dag_validation) give a direct "
            "effect of 0.3 and a total effect of 1.26; the planted outcome error coupled to the "
            "mediator error at 0.3 moves the read of the direct effect to 0.06 and is flagged by the "
            "cross-world sensitivity refuter while its uncoupled twin passes. The verdicts are "
            "diagnostics, not a calibration claim; on Gaussian disturbances the Bayesian "
            "posterior-predictive check passes the clean case. No interval is "
            "published. The accepted-structure cell executes exactly as the supplied one; acceptance "
            "does not establish the cross-world assumptions."
        ),
    },
]


def quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def rows() -> list[str]:
    out: list[str] = []
    for family in FAMILIES:
        for structure in family["structures"]:
            for inference in ("Frequentist", "Bayesian"):
                for validation in family["validations"][structure]:
                    out.append(
                        "\n".join(
                            [
                                "[[cell]]",
                                f'query = "{family["query"]}"',
                                'graph_class = "Dag"',
                                f'structure = "{structure}"',
                                f'inference = "{inference}"',
                                f'validation = "{validation}"',
                                f'estimators = ["{family["estimator"]}"]',
                                "staged = true",
                                'evidence_kind = "internal_known_truth"',
                                f'known_truth_fixture = "{FIXTURE}"',
                                f'evidence_test = "{TEST}"',
                                f'evidence_assertion = "{family["evidence"]}"',
                                "checked_execution = [",
                                f'  {{ estimator = "{family["estimator"]}", test = "{TEST}", assertion = "{family["checked"]}" }},',
                                "]",
                                f"limitations = {quote(family['limitations'])}",
                                f'calibration_reason = "{family["calibration_reason"][inference]}"',
                            ]
                        )
                    )
    return out


def block() -> str:
    return BEGIN + "\n\n" + "\n\n".join(rows()) + "\n\n" + END + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in {"--check", "--write"}:
        print(__doc__)
        return 2
    text = LICENSED.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = head + block().rstrip("\n") + tail
    else:
        new = text.rstrip("\n") + "\n\n" + block()
    if argv[0] == "--check":
        if new != text:
            print("parity/support_licensed.toml differs from the generated cross-world DAG rows")
            return 1
        print("ok")
        return 0
    LICENSED.write_text(new, encoding="utf-8")
    print(f"wrote {len(rows())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
