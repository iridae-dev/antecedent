"""The committed registries, checked against two edits that used to owe hundreds of records.

* An edit to the body of `fn invert` in `crates/antecedent-estimate/src/synthetic_rollout.rs`
  (the grid search of the conformal synthetic-control set) owed every calibration record
  bound through the facade's dispatch: associations, attributions, effects. It owes the
  measurements of the conformal sets, and no record outside the test targets whose code
  can build a synthetic-control rollout: the synthetic-control measurements (one helper
  builds every interval kind under a flag), the joint-response measurements (their recipe
  chooses its inference from a deserialisable value), and the synthetic-control rows of
  the range and graphless wider-range measurements. Seven records the rebinding left
  broad still owe.
* A comment in a Nix evidence recipe, or in the policy code, owed every record. It owes
  none.

Nothing is written: the edits are laid over the checkout for one currency check. A record
that is owed for another reason is not counted against either edit, and a calibration record
kept current by a probe attestation is checked through that attestation (its own row in the
records), not through the measurement binding the attestation replaced.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import calibration_status
import evidence_manifests as manifests
import narrow_evidence_binding as narrowing
from rust_items import File

ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = "crates/antecedent-estimate/src/synthetic_rollout.rs"
RECIPES = ("nix/evidence-target.nix", "nix/evidence-deps.nix", "nix/evidence-common.nix", "nix/evidence-driver.nix",
           "nix/evidence-toolchain.nix", "nix/calibration-release.nix")
POLICY = ("scripts/nix_evidence_policy.py", "scripts/nix_evidence_deps.py", "scripts/compile_time_environment.py")
# The measurements of the conformal synthetic-control sets: the routine `invert` serves.
SYNTHETIC = "crates/antecedent/tests/synthetic_control_calibration.rs"
CONFORMAL = {f"{SYNTHETIC}::conformal_synthetic_control_{name}_set"
             for name in ("first_period", "last_period", "constant_effect")}
# The test targets whose code can build a synthetic-control rollout.
ROLLOUT_TARGETS = (
    SYNTHETIC,
    "crates/antecedent/tests/joint_response_calibration.rs",
    "crates/antecedent/tests/synthetic_control_range_calibration.rs",
    "crates/antecedent/tests/graphless_wider_range_calibration.rs",
)
# Seven measurements of this target keep the broad selection: their manifests name items
# the source of their build does not hold under the same names, so they were not narrowed.
LEFT_BROAD = "crates/antecedent/tests/randomized_interval_calibration.rs"


class Overlaid(manifests.Checker):
    """A checker that reads some files from elsewhere: the checkout as it would be after an edit."""

    def __init__(self, root: Path, files: dict[str, Path]):
        super().__init__(root)
        self.files = files

    def path(self, rel: str) -> Path:
        return self.files.get(rel) or super().path(rel)


@unittest.skipUnless((ROOT / "parity/coverage_records.toml").is_file() and (ROOT / ROLLOUT).is_file(),
                     "needs the committed registries")
class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        edited: dict[str, Path] = {}

        def lay(rel: str, text: str) -> None:
            path = Path(cls.tmp.name) / rel.replace("/", "__")
            path.write_text(text, encoding="utf-8")
            edited[rel] = path

        source = (ROOT / ROLLOUT).read_text(encoding="utf-8")
        invert = File(source).by_locator["fn:invert#0"]
        lay(ROLLOUT, source[:invert.head_end + 1] + "\n    let _edited = 0_u8;" + source[invert.head_end + 1:])
        for rel in RECIPES:
            lay(rel, "# An edit to a comment.\n" + (ROOT / rel).read_text(encoding="utf-8") + "\n# Another.\n")
        for rel in POLICY:
            lay(rel, (ROOT / rel).read_text(encoding="utf-8") + "\n# An edit to a comment.\n")
        checker = Overlaid(ROOT, edited)
        coverage = calibration_status.load_records(ROOT / "parity" / "coverage_records.toml")
        # A probe confirms one measurement. Compare the active binding for that
        # measurement and report the route's original test, including when several
        # historical attestations exist for the same record.
        active = calibration_status.tree_fingerprints(coverage, root=ROOT)
        tests = {row["id"]: row["test"] for row in coverage}
        cls.records = []
        for kind, bound, ref, test in narrowing.records(ROOT, "all"):
            rid = bound["key"]
            if kind == "scientific":
                cls.records.append((kind, bound, ref, test))
            elif active.get(rid) == "manifest:" + ref or (kind == "calibration" and not active.get(rid)):
                cls.records.append(("calibration", bound, ref, tests[rid]))
        cls.reasons: dict[tuple[str, str], list[str]] = {}
        cls.tests: dict[tuple[str, str], str] = {}
        for kind, binding, ref, test in cls.records:
            label = (kind, binding["key"])
            cls.reasons[label] = checker.check(binding, ref)["reasons"]
            cls.tests[label] = test
        # Owed whatever these edits: a reason that names none of the edited files.
        cls.owed_before = {label for label, reasons in cls.reasons.items()
                           if any(not reason.endswith(tuple(":" + rel for rel in edited)) for reason in reasons)}

    def owed_by(self, path: str) -> set[tuple[str, str]]:
        return {label for label, reasons in self.reasons.items() if label not in self.owed_before
                and any(reason.startswith("input_changed:") and reason.endswith(":" + path) for reason in reasons)}

    def test_most_records_were_current_before_the_edits(self):
        self.assertLess(len(self.owed_before), len(self.reasons) // 50)

    def test_the_edit_reaches_the_measurements_of_the_routine_it_serves(self):
        owed = {label for label in self.owed_by(ROLLOUT) if label[0] == "calibration"}
        self.assertTrue({self.tests[label] for label in owed} >= CONFORMAL)

    def test_the_edit_owes_no_record_of_another_route(self):
        owed = {label for label in self.owed_by(ROLLOUT) if label[0] == "calibration"}
        beyond = sorted(self.tests[label] for label in owed if not self.tests[label].startswith(ROLLOUT_TARGETS))
        self.assertLessEqual(len(beyond), 7)
        self.assertEqual([test for test in beyond if not test.startswith(LEFT_BROAD)], [])
        for family in ("cov.association.", "cov.attribute.", "cov.anomaly_attribution.", "cov.average_effect.",
                       "cov.conditional_effect.", "cov.mediation_effect.", "cov.response_curve."):
            measured = [label for label in self.reasons if label[0] == "calibration" and label[1].startswith(family)]
            self.assertTrue(measured, family)
            self.assertFalse({label for label in set(measured) & owed
                              if not self.tests[label].startswith(LEFT_BROAD)}, family)
        self.assertLessEqual(len(owed), 45)

    def test_a_comment_in_a_recipe_or_in_policy_code_owes_nothing(self):
        for rel in (*RECIPES, *POLICY):
            with self.subTest(rel=rel):
                self.assertEqual(self.owed_by(rel), set())

    def test_no_current_record_binds_a_recipe_by_its_bytes(self):
        checker = manifests.Checker(ROOT)
        bound = set()
        for kind, binding, ref, _ in self.records:
            if (kind, binding["key"]) in self.owed_before:
                continue
            for entry in checker.load(ref)["inputs"]:
                if entry["kind"] == "evidence_file" and entry["path"] in (*RECIPES, *POLICY, "flake.lock"):
                    bound.add(entry["path"])
        self.assertEqual(bound, set())


if __name__ == "__main__":
    unittest.main()
