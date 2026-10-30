import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import calibration_readiness as gate

class ReadinessParsing(unittest.TestCase):
    def test_graph_tokens_do_not_confuse_classes(self):
        key = ("PointDerivative", "Dag", "explicit", "Frequentist")
        self.assertTrue(gate._candidate(key, "point_derivative_dag_frequentist"))
        self.assertFalse(gate._candidate(key, "point_derivative_cpdag_frequentist"))
        self.assertFalse(gate._candidate(key, "point_derivative_temporal_dag_frequentist"))
    def test_ignored_attribute_is_local_and_braces_are_lexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "crates/antecedent/tests"
            directory.mkdir(parents=True)
            source = directory / "point_derivative_dag_frequentist.rs"
            name = "point_derivative_dag_frequentist"
            body = 'let name = "point_derivative_dag_frequentist"; let x = "}"; helper(name);'
            helper = 'fn helper(test: &str) { let key = RecordKey { test, }; bind(key); let mut c = CoverageTally::for_record(key); c.record(1); }'
            prefix = '#[test] #[ignore = "calibration: run via scripts/gate_calibration.sh"] fn earlier() {}\n'
            key = ("PointDerivative", "Dag", "explicit", "Frequentist")
            groups = [(name, name)]
            with patch.object(gate, "ROOT", root):
                source.write_text(prefix + '#[test] fn ' + name + '() {' + body + '}\n' + helper)
                self.assertFalse(gate.has_record_emitter(key, groups))
                source.write_text(prefix + '#[test] #[ignore = "calibration: run via scripts/gate_calibration.sh"] fn ' + name + '() {' + body + '}\n' + helper)
                self.assertTrue(gate.has_record_emitter(key, groups))
                source.write_text('#[test] #[ignore = "calibration: run via scripts/gate_calibration.sh"] fn ' + name + '();')
                self.assertFalse(gate.has_record_emitter(key, groups))
