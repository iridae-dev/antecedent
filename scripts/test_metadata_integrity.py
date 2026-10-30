import unittest
from metadata_integrity import author_orcid_problems, oracle_packages, oracle_rows

class MetadataIntegrity(unittest.TestCase):
    def test_authors_are_scoped_nonempty_and_quoted_or_unquoted(self):
        creators = [{"orcid": "0009-0000-2421-4351"}]
        for value in ('"https://orcid.org/0009-0000-2421-4351"', 'https://orcid.org/0009-0000-2421-4351'):
            self.assertEqual(author_orcid_problems("authors:\n  - orcid: " + value + "\nversion: 1\n", creators), [])
        self.assertTrue(author_orcid_problems("authors:\n  - name: A\n", []))
        self.assertTrue(author_orcid_problems("authors:\n  - orcid: invalid\n", creators))
    def test_ledger_and_actual_oracle_fields(self):
        self.assertTrue(oracle_rows({"methods": []})[1])
        self.assertTrue(oracle_rows({"method": []})[1])
        self.assertFalse(oracle_rows({"method": [{"id": "x"}]})[1])
        self.assertNotIn("scipy", oracle_packages({"notes":"scipy"}))
        self.assertIn("scipy", oracle_packages({"oracle":{"project":"SciPy 1.2"}}))

    def test_nested_declared_projects_and_named_crosscheck(self):
        fixture = {"oracle": {"projects": {"fci": {"project": "causal-learn"},
                                             "lingam": {"project": "lingam"}},
                              "cross_check": "scipy==1.18.1 scipy.stats.entropy"}}
        self.assertTrue({"causal-learn", "lingam", "scipy"} <= oracle_packages(fixture))
        self.assertNotIn("scipy", oracle_packages({"historical_oracle": {"project": "scipy"}}))
