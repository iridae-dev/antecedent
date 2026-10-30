"""TOML strings emitted by the collector round-trip escaped characters."""
import tomllib
import unittest

from collect_coverage_records import toml_value


class CoverageTomlEscaping(unittest.TestCase):
    def test_text_round_trips_without_fabricating_data(self):
        value = 'path\\segment "quoted"\nline\tend\x7f'
        with self.assertRaises(ValueError):
            toml_value("text", "\ud800")
        encoded = toml_value("text", value)
        self.assertEqual(tomllib.loads("text = " + encoded)["text"], value)


if __name__ == "__main__":
    unittest.main()
