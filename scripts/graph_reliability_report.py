#!/usr/bin/env python3
"""Check directed-edge reliability measurements in calibration logs independently.

Each dataset contains ten equal-width bins of [edge count, sum of predicted
probabilities, observed edge count]. Edges within a dataset share a graph, so the
denominator of the reliability statistic uses the squared residual per dataset.
Exit 0 for compatible reports, 1 for scientific failures, 2 for invalid reports.
This command reads raw measurements and never modifies gate receipts.
"""
import argparse
import json
import math
from pathlib import Path

MARKER = "graph-reliability-report "
DESIGNS = {"per_edge", "soft_tiers", "per_edge_and_soft_tiers", "per_edge_pooled",
           "per_edge_and_soft_tiers_pooled", "per_edge_observed", "per_edge_and_soft_tiers_observed"}


def required_report(group: str) -> tuple[str, str] | None:
    """Score and simulation design for each fixed discovery reliability gate."""
    prefix = "antecedent-discovery__"
    if not group.startswith(prefix):
        return None
    test = group.removeprefix(prefix)
    for design, prior in (("per_edge", "per_edge_prior"), ("soft_tiers", "soft_tier"),
                          ("per_edge_and_soft_tiers", "per_edge_and_soft_tier")):
        if test == f"{prior}_marginals_are_reliable_under_draws_from_the_prior":
            return "GaussianBic", design
        if test == f"signed_uniform_{design}_edge_probabilities_match_generating_frequencies":
            return "GaussianSignedUniform", design
    if test == "observed_score_six_variable_per_edge_reliability":
        return "GaussianSignedUniform", "per_edge_observed"
    if test == "observed_score_six_variable_soft_tier_reliability":
        return "GaussianSignedUniform", "per_edge_and_soft_tiers_observed"
    if test == "pooled_marginals_are_reliable_under_regression_imputation_of_a_hidden_column":
        return "GaussianBic", "per_edge_pooled"
    if test == "pooled_marginals_are_reliable_under_imputation_with_per_edge_and_soft_tier_priors":
        return "GaussianBic", "per_edge_and_soft_tiers_pooled"
    return None


def parse_report(text: str) -> dict:
    reports = [json.loads(line[len(MARKER):]) for line in text.splitlines()
               if line.startswith(MARKER)]
    if len(reports) != 1:
        raise ValueError("expected exactly one complete directed-edge reliability report")
    return reports[0]


def check_report(report: dict) -> list[int]:
    """Recompute the fixed six-variable study's bin statistics and failed bins."""
    if (report["design"] not in DESIGNS or report["score"] not in {"GaussianBic", "GaussianSignedUniform"}
            or (report["design"].endswith("_observed") and report["score"] != "GaussianSignedUniform")
            or (report["score"] == "GaussianSignedUniform" and report["design"].endswith("_pooled"))
            or report["rows_per_dataset"] != 400 or report["variables"] != 6
            or report["minimum_bin_cells"] != 200 or report["maximum_abs_z"] != 3.5):
        raise ValueError("unknown directed-edge reliability design")
    datasets, bins = report["dataset_bins"], report["bins"]
    if len(datasets) != 400 or len(bins) != 10:
        raise ValueError("the study requires 400 complete datasets and ten bins")
    computed = [[0.0] * 4 for _ in bins]
    for dataset in datasets:
        if len(dataset) != 10:
            raise ValueError("missing dataset bins")
        edges = 0
        for index, (count, prediction, observed) in enumerate(dataset):
            if not all(type(x) in (float, int) and math.isfinite(x)
                       for x in (count, prediction, observed)):
                raise ValueError("non-numerical bin statistic")
            if not (0 <= observed <= count <= 30 and int(observed) == observed and int(count) == count):
                raise ValueError("invalid edge counts")
            if not count * index / 10 - 1e-12 <= prediction <= count * (index + 1) / 10 + 1e-12:
                raise ValueError("prediction total outside its bin")
            edges += count
            totals = computed[index]
            for k, value in enumerate((count, prediction, observed, (observed - prediction) ** 2)):
                totals[k] += value
        if edges != 30:
            raise ValueError("dataset did not report all 30 directed pairs")
    failed = []
    for i, ((count, prediction, observed, squares), recorded) in enumerate(zip(computed, bins)):
        for field, value in zip(("cells", "predicted_sum", "observed_sum", "cluster_residual_squares"),
                                (count, prediction, observed, squares)):
            if not math.isclose(recorded[field], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"bin {i}: inconsistent {field}")
        if count >= 200 and abs(observed - prediction) > 3.5 * math.sqrt(squares):
            failed.append(i)
    if report["reliability_passed"] is not (not failed):
        raise ValueError("reported scientific verdict disagrees with the bin statistics")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    status = 0
    for path in parser.parse_args().logs:
        try:
            report = parse_report(path.read_text())
            failed = check_report(report)
        except (KeyError, OSError, TypeError, ValueError, OverflowError) as error:
            parser.exit(2, f"{path}: invalid reliability report: {error}\n")
        if failed:
            print(f"{report['design']}: failed reliability in bins {failed}")
            status = 1
        else:
            print(f"{report['design']}: passed the registered reliability check")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
