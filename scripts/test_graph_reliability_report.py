"""The graph reliability report distinguishes failed science from incomplete execution."""
import unittest

from graph_reliability_report import check_report


def two_populated_bins(observed):
    # 400 datasets, 20 edges with mean p=.05, 10 with mean p=.95. The
    # high-probability bin alternates 9 and 10 observed edges across datasets.
    raw = [[[20, 1, observed], *[[0, 0, 0] for _ in range(8)], [10, 9.5, 9 + i % 2]]
           for i in range(400)]
    bins = [dict(cells=0, predicted_sum=0, observed_sum=0, cluster_residual_squares=0)
            for _ in range(10)]
    bins[0].update(cells=8000, predicted_sum=400, observed_sum=400 * observed,
                   cluster_residual_squares=400 * (observed - 1) ** 2)
    bins[9].update(cells=4000, predicted_sum=3800, observed_sum=3800,
                   cluster_residual_squares=100)
    return dict(design="per_edge", score="GaussianBic", variables=6, rows_per_dataset=400,
                minimum_bin_cells=200, maximum_abs_z=3.5, dataset_bins=raw, bins=bins,
                reliability_passed=observed == 1)


class ReliabilityReportTests(unittest.TestCase):
    def test_reconstructs_pass_and_fail(self):
        self.assertEqual(check_report(two_populated_bins(1)), [])
        self.assertEqual(check_report(two_populated_bins(0)), [0])

    def test_rejects_false_pass_and_inconsistent_cluster_denominator(self):
        report = two_populated_bins(0)
        report["reliability_passed"] = True
        with self.assertRaisesRegex(ValueError, "scientific verdict"):
            check_report(report)
        report["bins"][0]["cluster_residual_squares"] = 40_000
        with self.assertRaisesRegex(ValueError, "cluster_residual_squares"):
            check_report(report)

    def test_rejects_missing_dataset_and_missing_edges(self):
        report = two_populated_bins(1)
        report["dataset_bins"][1][0][0] = 19
        with self.assertRaisesRegex(ValueError, "all 30"):
            check_report(report)
        report["dataset_bins"].pop()
        with self.assertRaisesRegex(ValueError, "400 complete datasets"):
            check_report(report)
