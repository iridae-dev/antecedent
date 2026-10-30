"""The wheel import check requires a real path inside the intended environment."""
import contextlib
import io
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

GATE = Path(__file__).resolve().with_name("gate_release_candidate_2.py")


class WheelImportPath(unittest.TestCase):
    def test_real_containment_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "venv"
            for module_file, accepted in [
                (str(root / "lib" / "antecedent" / "__init__.py"), True),
                (str(Path(str(root) + "-src") / "antecedent.py"), False),
                (None, False),
            ]:
                module = types.SimpleNamespace(__file__=module_file, __version__="test")
                with self.subTest(path=module_file), patch.dict(sys.modules, {"antecedent": module}), \
                     patch.object(sys, "argv", [str(GATE), str(root)]), contextlib.redirect_stdout(io.StringIO()):
                    if accepted:
                        runpy.run_path(str(GATE), run_name="__main__")
                    else:
                        with self.assertRaises(SystemExit):
                            runpy.run_path(str(GATE), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
