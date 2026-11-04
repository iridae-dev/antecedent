import unittest

import calibration_status as status


class CandidateGroups(unittest.TestCase):
    def test_graph_tokens_do_not_confuse_classes(self):
        key = ("PointDerivative", "Dag", "explicit", "Frequentist")
        groups = [
            ("point_derivative_dag_frequentist", ""),
            ("point_derivative_cpdag_frequentist", ""),
            ("point_derivative_temporal_dag_frequentist", ""),
        ]
        self.assertEqual(status.candidate_groups(key, groups), ["point_derivative_dag_frequentist"])


if __name__ == "__main__":
    unittest.main()
