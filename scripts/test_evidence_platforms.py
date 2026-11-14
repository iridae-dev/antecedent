"""Platform-keyed evidence policy: selection, receipts, registries and digest replay."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import calibration_probe as probe
import evidence_manifests as em
import evidence_platforms as platforms
import nix_evidence_policy as darwin_policy

REPO = Path(__file__).resolve().parent.parent
MAC, LINUX, LINUX_ARM = "aarch64-apple-darwin", "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"
NAR = "sha256-" + "A" * 43 + "="


def receipt(target: str, *, policy: str | None = None, host: str | None = None, recipes=None) -> tuple[dict, list]:
    """A well-formed scientific receipt earned on `target`, and the inputs that bind it."""
    platform = platforms.PLATFORMS[target]
    recipe = {rel: "1" * 64 for rel in (recipes or platform.recipes)}
    source = {"commit": "c" * 40, "store_path": "/nix/store/" + "a" * 32 + "-source", "nar_hash": NAR}
    artifact = {"schema": 2, "crate": "c", "target_kind": "test", "target_name": "t",
                "binary": "targets/c/test/t/bin/test", "binary_sha256": "b" * 64,
                "build_targets": [{"crate": "c", "kind": "test", "name": "t"}],
                "source": source["store_path"], "source_nar_hash": NAR, "output_nar_hash": NAR,
                "recipe_sha256": recipe}
    built = {"schema": "nix-build-v1", "execution": "exact-binary", "artifact": artifact, "source": source,
             "tools": {"rustc": "1", "rustc_commit": "x", "cargo": "c", "host": host or target},
             "policy": {"id": policy or platform.policy, "target": target, "source_projection_allowed": True,
                        "audit_selection_sha256": "d" * 64},
             "compiler_graph_sha256": "e" * 64}
    return built, [{"kind": "evidence_file", "path": rel, "digest": digest} for rel, digest in recipe.items()]


BINDING = {"kind": "scientific", "key": "c::t::x"}


class SelectionTests(unittest.TestCase):
    def test_each_reviewed_platform_has_its_own_policy_system_and_registry(self):
        self.assertEqual(sorted(platforms.PLATFORMS), sorted([MAC, LINUX, LINUX_ARM]))
        mac = platforms.PLATFORMS[MAC]
        # The Apple-silicon policy is exactly the one existing records name.
        self.assertEqual((mac.policy, mac.nix_system, mac.registry),
                         ("nix-darwin-private-tmp-audit-v1", "aarch64-darwin", "nix/compile-time-reviewed.json"))
        self.assertEqual(mac.recipes, ("nix/evidence-target.nix", "nix/evidence-deps.nix", "nix/evidence-common.nix",
                                       "nix/evidence-driver.nix", "nix/evidence-toolchain.nix", "flake.lock"))
        self.assertEqual(mac.sandbox_paths, darwin_policy.ALLOWED_SANDBOX_PATHS)
        self.assertEqual({p.nix_system for p in platforms.PLATFORMS.values()},
                         {"aarch64-darwin", "x86_64-linux", "aarch64-linux"})
        registries = [p.registry for p in platforms.PLATFORMS.values()]
        self.assertEqual(len(set(registries)), 3)
        for target in (LINUX, LINUX_ARM):
            linux = platforms.PLATFORMS[target]
            self.assertNotEqual(linux.policy, mac.policy)
            self.assertFalse(set(linux.recipes) & set(mac.recipes) - {"flake.lock"})
            self.assertNotIn("/private/tmp", linux.sandbox_paths)

    def test_the_host_selects_its_policy_and_an_unreviewed_host_refuses(self):
        for target in platforms.PLATFORMS:
            with patch.object(platforms, "host_target", return_value=target):
                self.assertIs(platforms.host_platform(), platforms.PLATFORMS[target])
        with patch.object(platforms, "host_target", return_value="x86_64-apple-darwin"), \
             self.assertRaisesRegex(platforms.UnreviewedPlatform, "cannot be earned on x86_64-apple-darwin"):
            platforms.host_platform()

    def test_host_target_is_a_rust_triple(self):
        with patch.object(platforms.sys, "platform", "darwin"), \
             patch.object(platforms.host_info, "machine", return_value="arm64"):
            self.assertEqual(platforms.host_target(), MAC)
        with patch.object(platforms.sys, "platform", "linux"), \
             patch.object(platforms.host_info, "machine", return_value="x86_64"), \
             patch.object(platforms.host_info, "libc_ver", return_value=("glibc", "2.39")):
            self.assertEqual(platforms.host_target(), LINUX)
        with patch.object(platforms.sys, "platform", "linux"), \
             patch.object(platforms.host_info, "machine", return_value="aarch64"), \
             patch.object(platforms.host_info, "libc_ver", return_value=("", "")):
            self.assertNotIn(platforms.host_target(), platforms.PLATFORMS)  # musl is not reviewed

    def test_apple_silicon_is_ready_and_linux_refuses_until_bring_up(self):
        self.assertEqual(platforms.bring_up_gaps(REPO, platforms.PLATFORMS[MAC]), [])
        with patch.object(platforms, "host_target", return_value=MAC):
            self.assertIs(platforms.earning_platform(REPO), platforms.PLATFORMS[MAC])
        for target in (LINUX, LINUX_ARM):
            gaps = "\n".join(platforms.bring_up_gaps(REPO, platforms.PLATFORMS[target]))
            self.assertIn("no verified SHA-256 for llvm-tools, rust, rustc-dev", gaps)
            self.assertIn(f"nix/compile-time-reviewed.{target}.json", gaps)
            with patch.object(platforms, "host_target", return_value=target), \
                 self.assertRaisesRegex(platforms.UnreviewedPlatform, "bring-up is incomplete"):
                platforms.earning_platform(REPO)

    def test_a_complete_bring_up_earns(self):
        linux = platforms.PLATFORMS[LINUX]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel in (*linux.recipes, *linux.release_recipes, linux.registry):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("{}", encoding="utf-8")
            (root / "tools/evidence-analyzer").mkdir(parents=True)
            (root / "tools/evidence-analyzer/pin.toml").write_text('release = "1.2.3"\n', encoding="utf-8")
            pins = {"release": "1.2.3", "targets": {LINUX: {"rust": "a" * 64, "llvm-tools": None}}}
            (root / linux.toolchain_pins).write_text(json.dumps(pins), encoding="utf-8")
            self.assertEqual(platforms.bring_up_gaps(root, linux),
                             [f"{linux.toolchain_pins}: no verified SHA-256 for llvm-tools"])
            pins["targets"][LINUX]["llvm-tools"] = "b" * 64
            (root / linux.toolchain_pins).write_text(json.dumps(pins), encoding="utf-8")
            with patch.object(platforms, "host_target", return_value=LINUX):
                self.assertIs(platforms.earning_platform(root), linux)
            pins["release"] = "1.2.4"
            (root / linux.toolchain_pins).write_text(json.dumps(pins), encoding="utf-8")
            self.assertIn("release is not the evidence compiler pin's", platforms.bring_up_gaps(root, linux)[0])


class ReceiptTests(unittest.TestCase):
    def test_a_receipt_of_each_platform_validates_with_its_own_recipes(self):
        for target in platforms.PLATFORMS:
            with self.subTest(target=target):
                built, inputs = receipt(target)
                em.validate_nix_receipt(built, BINDING, inputs)
                self.assertEqual(platforms.manifest_platform({"nix_build": built}), target)

    def test_a_receipt_claiming_another_platforms_policy_is_rejected(self):
        mac, linux = platforms.PLATFORMS[MAC], platforms.PLATFORMS[LINUX]
        cases = {
            "mac receipt, linux policy id": receipt(MAC, policy=linux.policy),
            "linux receipt, mac policy id": receipt(LINUX, policy=mac.policy),
            "mac receipt, linux compiler": receipt(MAC, host=LINUX),
            "linux receipt, mac compiler": receipt(LINUX, host=MAC),
            "linux receipt, other linux compiler": receipt(LINUX, host=LINUX_ARM),
            "mac receipt, linux recipes": receipt(MAC, recipes=linux.recipes),
            "linux receipt, mac recipes": receipt(LINUX, recipes=mac.recipes),
        }
        for label, (built, inputs) in cases.items():
            with self.subTest(label), self.assertRaises(em.InvalidManifest):
                em.validate_nix_receipt(built, BINDING, inputs)
        built, inputs = receipt(MAC)
        built["policy"]["target"] = LINUX  # the Mac policy id under a Linux target
        with self.assertRaises(em.InvalidManifest):
            em.validate_nix_receipt(built, BINDING, inputs)
        for missing in (("policy", "target"), ("tools", "host")):
            built, inputs = receipt(MAC)
            del built[missing[0]][missing[1]]
            with self.subTest(missing=missing), self.assertRaises(em.InvalidManifest):
                em.validate_nix_receipt(built, BINDING, inputs)

    def test_stored_receipts_name_their_platform_and_still_validate(self):
        store = REPO / "parity/evidence_manifests"
        stored = sorted(store.glob("*.json"))[:40]
        self.assertTrue(stored)
        checker = em.Checker(REPO)
        for path in stored:
            manifest = checker.load(path.stem)
            self.assertEqual(platforms.manifest_platform(manifest), MAC)
            self.assertEqual(platforms.stored_platform(REPO, path.stem), MAC)
            binding = manifest["record"]
            em.validate_nix_receipt(manifest["nix_build"], binding, manifest["inputs"])
        self.assertIsNone(platforms.stored_platform(REPO, "0" * 64))
        self.assertIsNone(platforms.manifest_platform({}))


class RegistryTests(unittest.TestCase):
    """A Linux registry is checked exactly as the Apple-silicon one is."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "nix").mkdir()
        (self.root / "Cargo.lock").write_bytes((REPO / "Cargo.lock").read_bytes())
        (self.root / darwin_policy.REGISTRY).write_bytes((REPO / darwin_policy.REGISTRY).read_bytes())
        self.reviewed = json.loads((REPO / darwin_policy.REGISTRY).read_bytes())
        self.linux = platforms.PLATFORMS[LINUX]
        rows = [r for r in self.reviewed["units"] if r["source_projection"] == "reviewed-safe"]
        script = next(r for r in rows if r["mode"] == "run-custom-build")
        build = next(r for r in rows if r["mode"] == "build" and r["name"] == script["name"]
                     and r["version"] == script["version"] and r["target_kind"] == ["custom-build"])
        unit = lambda row, deps: {**{k: row[k] for k in darwin_policy.UNIT_KEYS}, "dependencies": deps}  # noqa: E731
        self.graph = {"units": [unit(script, [{"index": 1}]), unit(build, [])], "roots": [0]}
        self.lock = (self.root / "Cargo.lock").read_bytes()

    def write(self, registry: dict) -> None:
        (self.root / self.linux.registry).write_text(json.dumps(registry), encoding="utf-8")

    def test_a_draft_is_unresolved_until_reviewed_and_keeps_reviewed_rows(self):
        draft = platforms.draft_registry(self.linux, self.lock, [self.graph, self.graph])
        self.assertEqual((draft["policy"], draft["target"]), (self.linux.policy, LINUX))
        self.assertEqual([row["source_projection"] for row in draft["units"]], ["unresolved"] * 2)
        self.write(draft)
        policy = platforms.validate_units(self.root, self.graph, platform=self.linux)
        self.assertEqual((policy["id"], policy["target"], policy["covered_units"]), (self.linux.policy, LINUX, 2))
        # Unreviewed rows cover the graph but never license precise evidence.
        self.assertFalse(policy["source_projection_allowed"])
        for row in draft["units"]:
            row["source_projection"] = "reviewed-safe"
        draft["units"][0]["review"] = "read on Linux"
        again = platforms.draft_registry(self.linux, self.lock, [self.graph], draft)
        self.assertEqual(again["units"], draft["units"])
        self.write(again)
        policy = platforms.validate_units(self.root, self.graph, platform=self.linux)
        self.assertTrue(policy["source_projection_allowed"])
        platforms.recheck_selection(self.root, policy)
        again["units"][0]["review"] = "changed"
        self.write(again)
        with self.assertRaisesRegex(ValueError, "audit changed"):
            platforms.recheck_selection(self.root, policy)

    def test_a_missing_stale_or_foreign_registry_refuses(self):
        with self.assertRaisesRegex(ValueError, "invalid or incomplete"):
            platforms.validate_units(self.root, self.graph, platform=self.linux)
        # The Apple-silicon registry is not a Linux review.
        self.write(self.reviewed)
        with self.assertRaisesRegex(ValueError, "stale or unrecognized"):
            platforms.validate_units(self.root, self.graph, platform=self.linux)
        draft = platforms.draft_registry(self.linux, self.lock, [self.graph])
        self.write({**draft, "units": draft["units"][:1]})
        with self.assertRaisesRegex(ValueError, "unreviewed Nix compile-time unit"):
            platforms.validate_units(self.root, self.graph, platform=self.linux)
        self.write({**draft, "cargo_lock_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "stale or unrecognized"):
            platforms.validate_units(self.root, self.graph, platform=self.linux)

    def test_the_shared_check_agrees_with_the_apple_silicon_policy(self):
        # Rename the reviewed registry to a Linux one: the same rows must select alike.
        self.write({**self.reviewed, "policy": self.linux.policy, "target": LINUX})
        linux = platforms.validate_units(self.root, self.graph, platform=self.linux)
        mac = platforms.validate_units(self.root, self.graph, platform=platforms.DARWIN)
        self.assertEqual(mac, darwin_policy.validate_units(self.root, self.graph))
        differing = {"id", "target", "audit_selection_sha256"}
        self.assertEqual({k: v for k, v in linux.items() if k not in differing},
                         {k: v for k, v in mac.items() if k not in differing})
        self.assertNotEqual(linux["audit_selection_sha256"], mac["audit_selection_sha256"])
        # Each receipt is rechecked against the registry of the platform it names.
        platforms.recheck_selection(self.root, mac)
        platforms.recheck_selection(self.root, linux)
        with self.assertRaises(ValueError):
            platforms.recheck_selection(self.root, {**mac, "target": LINUX})
        with self.assertRaises(ValueError):
            platforms.recheck_selection(self.root, {**linux, "id": darwin_policy.POLICY})

    def test_a_build_is_validated_on_the_host_that_made_it(self):
        built = {"compiler_graph": self.graph, "tools": {"host": MAC}}
        with patch.object(platforms, "host_target", return_value=LINUX), \
             self.assertRaisesRegex(ValueError, "compiler host is not this host"):
            platforms.validate_build(self.root, built)
        with patch.object(platforms, "host_target", return_value=MAC), \
             patch.object(darwin_policy, "validate_build", return_value={"id": "mac"}) as mac:
            self.assertEqual(platforms.validate_build(self.root, built), {"id": "mac"})
        mac.assert_called_once_with(self.root, built)
        self.write(platforms.draft_registry(self.linux, self.lock, [self.graph]))
        with patch.object(platforms, "host_target", return_value=LINUX), \
             patch.object(platforms, "_verify_sandbox") as sandbox:
            policy = platforms.validate_build(self.root, {**built, "tools": {"host": LINUX}})
        sandbox.assert_called_once_with(self.linux)
        self.assertEqual(policy["target"], LINUX)


class LinuxSandboxTests(unittest.TestCase):
    def check(self, *, lines=("sandbox = true", "sandbox-fallback = false"), sandbox=True, fallback=False,
              paths=("/bin/sh=/nix/store/abc-busybox/bin/busybox",)):
        linux = platforms.PLATFORMS[LINUX]
        config = {"sandbox": {"value": sandbox}, "sandbox-fallback": {"value": fallback},
                  "sandbox-paths": {"value": list(paths)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "nix.conf"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stat = path.stat()
            owned = type("Stat", (), {"st_uid": 0, "st_mode": stat.st_mode & ~0o002})()
            done = type("Done", (), {"stdout": json.dumps(config)})()
            with patch.object(platforms, "SYSTEM_CONFIG", path), \
                 patch.object(Path, "stat", return_value=owned), \
                 patch.object(Path, "is_file", return_value=True), \
                 patch.object(platforms.subprocess, "run", return_value=done), \
                 patch.dict(platforms.os.environ, clear=False) as env:
                for name in platforms.NIX_OVERRIDES:
                    env.pop(name, None)
                platforms._verify_sandbox(linux)

    def test_the_default_linux_sandbox_passes(self):
        self.check()
        self.check(paths=())

    def test_a_weakened_sandbox_refuses(self):
        cases = {
            "explicitly set sandbox": {"lines": ("sandbox-fallback = false",)},
            "explicitly set sandbox-fallback": {"lines": ("sandbox = true",)},
            "included": {"lines": ("sandbox = true", "sandbox-fallback = false", "include /etc/extra.conf")},
            "disagree": {"sandbox": "relaxed"},
            "disagree ": {"fallback": True},
            "unreviewed sandbox path": {"paths": ("/tmp",)},
            "unreviewed sandbox path ": {"paths": ("/bin/sh=/usr/bin/dash",)},
            "unreviewed sandbox path  ": {"paths": ("/private/tmp",)},
        }
        for message, arguments in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message.strip()):
                self.check(**arguments)


class ReplayTests(unittest.TestCase):
    def test_replay_is_same_platform_only(self):
        self.assertIsNone(platforms.replay_refusal(MAC, MAC))
        self.assertIsNone(platforms.replay_refusal(None, LINUX))
        self.assertEqual(platforms.replay_refusal(MAC, LINUX),
                         f"earned on {MAC}; this host ({LINUX}) cannot replay its digests")

    def test_the_probe_sets_aside_records_earned_elsewhere(self):
        earned = {"mac": MAC, "linux": LINUX, "old": None}
        records = [{"id": rid} for rid in earned]
        local, foreign = probe.replayable(records, LINUX, lambda rec: earned[rec["id"]])
        self.assertEqual([rec["id"] for rec in local], ["linux", "old"])
        self.assertEqual(foreign, {"mac": f"earned on {MAC}; this host ({LINUX}) cannot replay its digests"})

    def test_a_cross_platform_probe_runs_nothing_and_reports_no_mismatch(self):
        record = {"id": "cov.example", "dependency_manifest": "f" * 64}
        out = io.StringIO()
        with patch.object(probe, "select_records", return_value=[record]), \
             patch.object(platforms, "stored_platform", return_value=MAC), \
             patch.object(platforms, "host_target", return_value=LINUX), \
             patch.object(probe, "run_tasks") as run, \
             patch.object(probe, "plan") as plan, \
             contextlib.redirect_stdout(out):
            self.assertEqual(probe.main(["quick"]), 0)
        run.assert_not_called()
        plan.assert_not_called()
        text = out.getvalue()
        self.assertIn(f"cov.example: not probed: earned on {MAC}; this host ({LINUX}) cannot replay its digests", text)
        self.assertNotIn("mismatch", text)

    def test_a_probe_speaks_only_for_a_measurement_of_its_own_platform(self):
        import calibration_status as status
        earned = {"m": MAC, "p-mac": MAC, "p-linux": LINUX}
        with patch.object(platforms, "stored_platform", side_effect=lambda root, ref: earned.get(ref)):
            record = {"dependency_manifest": "m"}
            self.assertTrue(status._same_platform(REPO, record, {"dependency_manifest": "p-mac"}))
            self.assertFalse(status._same_platform(REPO, record, {"dependency_manifest": "p-linux"}))
            self.assertTrue(status._same_platform(REPO, {}, {"dependency_manifest": "p-linux"}))


if __name__ == "__main__":
    unittest.main()
