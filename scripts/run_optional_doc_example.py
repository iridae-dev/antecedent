#!/usr/bin/env python3
"""Require the external half of an optional documentation handoff to execute."""

import argparse
import importlib
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dependency", choices=["dowhy", "econml"])
    args = parser.parse_args()
    importlib.import_module(args.dependency)  # A missing dependency is a failure here.
    if args.dependency == "dowhy":
        ns = runpy.run_path(str(ROOT / "examples/python/dowhy_handoff.py"))
        data = ns["confounded_scm"]()
        ant = ns["antecedent_side"](data)
        result = ns["dowhy_side"](data, ant["dot"])
        assert result is not None, "DoWhy half was skipped"
        assert (
            result["adjustment_set"]
            == result["round_trip_adjustment_set"]
            == ns["EXPECTED_ADJUSTMENT"]
        )
        assert abs(result["ate"] - ns["TRUE_ATE"]) < 0.5
    else:
        ns = runpy.run_path(
            str(ROOT / "examples/python/econml_cate_handoff.py"), run_name="__main__"
        )
        assert "econml_cate" in ns["LAST_HANDOFF"], "EconML fit was skipped"
        assert abs(ns["LAST_HANDOFF"]["econml_cate"] - ns["TRUE_ATE"]) < 0.5
    print(f"Required {args.dependency} handoff passed")


if __name__ == "__main__":
    main()
