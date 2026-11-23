"""The detail-literal gate finds exactly the literals a constant can replace."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_reason_detail_literals as gate  # noqa: E402


class RustLexing(unittest.TestCase):
    def test_strings_comments_and_raw_strings(self):
        src = 'let a = "x.y"; // "x.z"\nlet b = r#"{"k":"x.y"}"#; let c = \'"\'; let d = "p.q";\n'
        strings, mask = gate.rust_lex(src)
        self.assertEqual([t for _, _, t in strings], ["x.y", "p.q"])
        self.assertNotIn("x.z", mask)

    def test_format_string_position_is_not_replaceable(self):
        src = 'format!("x.y"); assert_eq!(a, "x.y"); assert!(c, "x.y");'
        strings, mask = gate.rust_lex(src)
        got = [gate._enclosing_macro(mask, a) for a, _, _ in strings]
        self.assertEqual(got, [("format", 0), ("assert_eq", 1), ("assert", 1)])
        self.assertTrue(gate.FORMAT_INDEX["format"] == 0 and gate.FORMAT_INDEX["assert_eq"] == 2)

    def test_cfg_test_module_range(self):
        src = 'fn f() { "a.b" }\n#[cfg(test)]\nmod tests {\n  fn g() { "a.b" }\n}\n'
        _, mask = gate.rust_lex(src)
        (lo, hi), = gate.cfg_test_ranges(mask)
        self.assertGreater(lo, src.index("fn f"))
        self.assertGreater(src.rindex('"a.b"'), lo)
        self.assertLessEqual(src.rindex('"a.b"'), hi)


if __name__ == "__main__":
    unittest.main()
