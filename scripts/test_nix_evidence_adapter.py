"""Engineering tests for preparatory Nix evidence build diagnostics."""
import unittest
from unittest.mock import patch
import json
import subprocess

from nix_evidence_adapter import client_policy_diagnostics, main


class ClientPolicyTests(unittest.TestCase):
    def config(self):
        return {
            "sandbox": {"value": True},
            "sandbox-fallback": {"value": False},
            "substitute": {"value": False},
            "builders": {"value": ""},
            "sandbox-paths": {"value": ["/usr/lib", "/private/tmp"]},
        }

    def test_client_configuration_is_never_daemon_attestation(self):
        result = client_policy_diagnostics(self.config())
        self.assertTrue(result["client_policy_ok"])
        self.assertFalse(result["boundary_verified"])
        self.assertEqual(result["sandbox_paths_requiring_review"], ["/usr/lib", "/private/tmp"])

    def test_unsafe_and_missing_settings_fail_closed(self):
        config = self.config()
        config["sandbox"] = {"value": False}
        config["sandbox-fallback"] = {"value": True}
        config["builders"] = {"value": "@/etc/nix/machines"}
        del config["substitute"]
        result = client_policy_diagnostics(config)
        self.assertFalse(result["client_policy_ok"])
        self.assertEqual(len(result["problems"]), 4)
        self.assertFalse(result["boundary_verified"])

    def test_sandbox_paths_must_be_known(self):
        config = self.config()
        del config["sandbox-paths"]
        self.assertIn("sandbox paths are unknown", client_policy_diagnostics(config)["problems"])


class DiagnosticCommandTests(unittest.TestCase):
    def test_cli_reports_that_a_safe_client_is_not_an_attested_boundary(self):
        result = subprocess.CompletedProcess([], 0, json.dumps(ClientPolicyTests().config()), "")
        with patch("nix_evidence_adapter.subprocess.run", return_value=result) as run, \
             patch("builtins.print") as print_message:
            self.assertEqual(main(), 1)
        self.assertEqual(run.call_args.args[0][-3:], ["config", "show", "--json"])
        self.assertFalse(json.loads(print_message.call_args.args[0])["boundary_verified"])

    def test_cli_fails_closed_if_nix_is_missing_or_returns_invalid_json(self):
        for result in (FileNotFoundError(), subprocess.CompletedProcess([], 0, "not json", "")):
            with self.subTest(result=result), patch("nix_evidence_adapter.subprocess.run") as run, \
                 patch("builtins.print"):
                if isinstance(result, Exception):
                    run.side_effect = result
                else:
                    run.return_value = result
                self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
