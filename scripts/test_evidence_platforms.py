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

    def test_the_tree_holds_every_platforms_policy(self):
        # Nothing a record needs is left to be produced on a particular host.
        for target, platform in platforms.PLATFORMS.items():
            with self.subTest(target=target):
                self.assertEqual(platforms.bring_up_gaps(REPO, platform), [])
                with patch.object(platforms, "host_target", return_value=target):
                    self.assertIs(platforms.earning_platform(REPO), platform)

    def test_a_tree_without_a_platforms_policy_refuses(self):
        linux = platforms.PLATFORMS[LINUX]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(platforms, "host_target", return_value=LINUX), \
             self.assertRaisesRegex(platforms.UnreviewedPlatform, "lacks part of its policy") as refused:
            platforms.earning_platform(Path(directory))
        self.assertIn(f"{linux.build_recipe} is missing", str(refused.exception))
        self.assertIn(f"{linux.registry} (the reviewed compile-time registry) is missing", str(refused.exception))

    def test_the_tree_and_its_pins_make_a_platform_ready(self):
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


def unit_graph(rows: list[dict]) -> dict:
    """A compiler graph whose compile-time units are exactly `rows` (one must run at compile time)."""
    units = [{**{k: row[k] for k in darwin_policy.UNIT_KEYS}, "dependencies": []} for row in rows]
    runs = [i for i, unit in enumerate(units) if compile_time(unit)]
    units[runs[0]]["dependencies"] = [{"index": i} for i in range(len(units)) if i not in runs]
    return {"units": units, "roots": [runs[0]]}


def compile_time(unit: dict) -> bool:
    return unit["mode"] == "run-custom-build" or "proc-macro" in unit["target_kind"]


class CommittedLinuxPolicyTests(unittest.TestCase):
    """What a Linux host needs is in the tree: pins, recipes and reviewed registries."""

    def test_every_toolchain_archive_is_pinned_to_the_compiler_pin(self):
        import tomllib
        pins = json.loads((REPO / platforms.LINUX_PINS).read_bytes())
        pin = tomllib.loads((REPO / "tools/evidence-analyzer/pin.toml").read_text(encoding="utf-8"))
        self.assertEqual(pins["release"], pin["release"])
        self.assertEqual(sorted(pins["targets"]), sorted([LINUX, LINUX_ARM]))
        seen = set()
        for target, archives in pins["targets"].items():
            self.assertEqual(sorted(archives), ["llvm-tools", "rust", "rustc-dev"])
            for name, value in archives.items():
                with self.subTest(target=target, archive=name):
                    self.assertRegex(value, r"\A[0-9a-f]{64}\Z")
                    seen.add(value)
        self.assertEqual(len(seen), 6)  # six different archives

    def test_the_linux_recipes_name_no_other_platform(self):
        for platform in (platforms.PLATFORMS[LINUX], platforms.PLATFORMS[LINUX_ARM]):
            for rel in {*platform.recipes, *platform.release_recipes} - {"flake.lock", platforms.LINUX_PINS}:
                text = (REPO / rel).read_text(encoding="utf-8")
                with self.subTest(recipe=rel):
                    self.assertTrue(rel.startswith("nix/linux/"))
                    self.assertNotIn("darwin", text.lower())
                    self.assertNotIn("oso_prefix", text)  # a Mach-O linker flag
        # The Apple-silicon recipes are bound by content in stored records: still theirs alone.
        for rel in set(platforms.DARWIN.recipes) - {"flake.lock"}:
            self.assertNotIn("linux", (REPO / rel).read_text(encoding="utf-8").lower())

    def test_each_linux_registry_is_reviewed_and_bound_to_the_lock(self):
        lock = (REPO / "Cargo.lock").read_bytes()
        for target in (LINUX, LINUX_ARM):
            platform = platforms.PLATFORMS[target]
            registry, _ = platforms._registry(REPO, platform)
            with self.subTest(target=target):
                self.assertEqual(registry["cargo_lock_sha256"], darwin_policy._sha(lock))
                self.assertEqual(registry["scope"], platform.scope)
                self.assertTrue(registry["audit_documents"])
                for document in registry["audit_documents"]:
                    self.assertTrue((REPO / document).is_file(), document)
                self.assertIn("nix/linux/compile-time-audit.md", registry["audit_documents"])
                states = {row["source_projection"] for row in registry["units"]}
                self.assertEqual(states, {"reviewed-safe", "complete-source"})
                for row in registry["units"]:
                    self.assertEqual(row["repo_source_review"]["projection"], row["source_projection"])
                    self.assertNotIn("arwin", row["repo_source_review"]["reason"])
                # Every row licenses a build that selects it, and the receipt rechecks.
                self.assertEqual(platforms.uncovered_units(REPO, platform, [unit_graph(registry["units"])]), [])
                safe = [row for row in registry["units"] if row["source_projection"] == "reviewed-safe"]
                policy = platforms.validate_units(REPO, unit_graph(safe), platform=platform)
                self.assertTrue(policy["source_projection_allowed"])
                self.assertEqual((policy["id"], policy["target"]), (platform.policy, target))
                platforms.recheck_selection(REPO, policy)

    def test_a_unit_reviewed_on_two_platforms_agrees(self):
        # One package, checksum and feature set is one source: its review state is one finding.
        reviewed = {darwin_policy._signature(row): row
                    for row in json.loads((REPO / darwin_policy.REGISTRY).read_bytes())["units"]}
        for target in (LINUX, LINUX_ARM):
            shared = 0
            for row in json.loads((REPO / platforms.PLATFORMS[target].registry).read_bytes())["units"]:
                other = reviewed.get(darwin_policy._signature(row))
                if other is not None and other["checksum"] == row["checksum"]:
                    shared += 1
                    self.assertEqual(row["source_projection"], other["source_projection"], row["name"])
            self.assertGreater(shared, 100)


class PrepareTests(unittest.TestCase):
    """`prepare` checks the host, realises the inputs once and names what resolves a refusal."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.linux = platforms.PLATFORMS[LINUX]
        for rel in (*self.linux.recipes, *self.linux.release_recipes, platforms.INPUTS_RECIPE,
                    "tools/evidence-analyzer/pin.toml"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_bytes((REPO / rel).read_bytes())
        (self.root / "Cargo.lock").write_bytes((REPO / "Cargo.lock").read_bytes())
        self.registry = json.loads((REPO / self.linux.registry).read_bytes())
        self.write(self.registry)
        self.graph = unit_graph(self.registry["units"][:3])
        self.ran: list[list[str]] = []
        host = patch.object(platforms, "host_target", return_value=LINUX)
        host.start()
        self.addCleanup(host.stop)

    def write(self, registry: dict) -> None:
        (self.root / self.linux.registry).write_text(json.dumps(registry), encoding="utf-8")

    def run_nix(self, code: int = 0):
        def run(command, **kwargs):
            self.ran.append(command)
            return type("Done", (), {"returncode": code})()
        return run

    def prepare(self, *, code: int = 0, sandbox=None):
        with patch.object(platforms, "_verify_sandbox", side_effect=sandbox) as checked, \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                return platforms.prepare_host(self.root, run=self.run_nix(code), graphs=[self.graph])
            finally:
                self.sandbox_checked = checked.call_count

    def test_a_ready_host_realises_the_inputs_once(self):
        self.assertFalse(platforms.prepared(self.root, self.linux))
        self.assertIs(self.prepare(), self.linux)
        self.assertEqual(self.sandbox_checked, 1)
        self.assertEqual(len(self.ran), 1)
        command = self.ran[0]
        self.assertEqual(command[-2:], ["--file", str(self.root / platforms.INPUTS_RECIPE)])
        # Realising inputs may substitute; it is never an evidence build's own flags.
        self.assertNotIn("--offline", command)
        self.assertTrue(platforms.prepared(self.root, self.linux))
        # Earning prepares on first use only.
        with patch.object(platforms, "prepare_host") as again:
            self.assertIs(platforms.earning_platform(self.root, prepare=True), self.linux)
        again.assert_not_called()

    def test_earning_prepares_a_linux_host_on_first_use_and_never_apple_silicon(self):
        with patch.object(platforms, "prepare_host") as first:
            platforms.earning_platform(self.root, prepare=True)
            first.assert_called_once_with(self.root)
            platforms.earning_platform(self.root)  # a check alone prepares nothing
            first.assert_called_once()
        self.assertTrue(self.linux.prepares and platforms.PLATFORMS[LINUX_ARM].prepares)
        self.assertFalse(platforms.DARWIN.prepares)
        with patch.object(platforms, "host_target", return_value=MAC), \
             patch.object(platforms, "prepare_host") as never:
            self.assertIs(platforms.earning_platform(REPO, prepare=True), platforms.DARWIN)
        never.assert_not_called()

    def test_a_changed_lock_recipe_or_registry_prepares_again(self):
        self.prepare()
        for rel in ("Cargo.lock", self.linux.build_recipe, self.linux.registry, platforms.INPUTS_RECIPE):
            original = (self.root / rel).read_bytes()
            (self.root / rel).write_bytes(original + b"\n")
            with self.subTest(changed=rel):
                self.assertFalse(platforms.prepared(self.root, self.linux))
            (self.root / rel).write_bytes(original)
        self.assertTrue(platforms.prepared(self.root, self.linux))

    def test_a_weak_sandbox_refuses_with_the_command_that_fixes_it(self):
        refusal = ValueError("Nix daemon configuration must explicitly set sandbox = true")
        with self.assertRaises(platforms.UnreviewedPlatform) as refused:
            self.prepare(sandbox=refusal)
        text = str(refused.exception)
        self.assertIn("must explicitly set sandbox = true", text)
        self.assertIn(platforms.SANDBOX_REMEDY, text)
        self.assertIn(platforms.PREPARE, text)
        self.assertEqual(self.ran, [])  # nothing is fetched or built before the host complies
        # A setting no appended line can fix is named without that command.
        with self.assertRaises(platforms.UnreviewedPlatform) as refused:
            self.prepare(sandbox=ValueError("Nix exposes an unreviewed sandbox path"))
        self.assertIn("unreviewed sandbox path", str(refused.exception))
        self.assertNotIn(platforms.SANDBOX_REMEDY, str(refused.exception))
        self.assertIn(platforms.PREPARE, str(refused.exception))
        self.assertFalse(platforms.prepared(self.root, self.linux))

    def test_an_unreviewed_or_unresolved_unit_refuses_before_any_build(self):
        missing = {**self.registry, "units": self.registry["units"][1:]}
        self.write(missing)
        with self.assertRaises(platforms.UnreviewedPlatform) as refused:
            self.prepare()
        unit = self.registry["units"][0]
        self.assertIn(f"{unit['name']} {unit['version']} {unit['mode']} (not in the registry)", str(refused.exception))
        self.assertIn("draft-registry", str(refused.exception))
        unresolved = json.loads(json.dumps(self.registry))
        unresolved["units"][1]["source_projection"] = "unresolved"
        self.write(unresolved)
        with self.assertRaisesRegex(platforms.UnreviewedPlatform, r"\(unresolved\)"):
            self.prepare()
        self.write({**self.registry, "cargo_lock_sha256": "0" * 64})
        with self.assertRaisesRegex(platforms.UnreviewedPlatform, "draft-registry --all"):
            self.prepare()
        self.assertEqual(self.ran, [])
        self.assertFalse(platforms.prepared(self.root, self.linux))

    def test_inputs_that_do_not_build_refuse(self):
        with self.assertRaisesRegex(platforms.UnreviewedPlatform, "could not be realised") as refused:
            self.prepare(code=1)
        self.assertIn(platforms.PREPARE, str(refused.exception))
        self.assertFalse(platforms.prepared(self.root, self.linux))

    def test_a_tree_without_the_policy_refuses_first(self):
        (self.root / self.linux.registry).unlink()
        with self.assertRaisesRegex(platforms.UnreviewedPlatform, "lacks part of its policy"):
            self.prepare()
        self.assertEqual((self.sandbox_checked, self.ran), (0, []))


class RegistryMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.linux = platforms.PLATFORMS[LINUX]
        self.lock = (REPO / "Cargo.lock").read_bytes()
        self.registry = json.loads((REPO / self.linux.registry).read_bytes())

    def test_a_redraft_keeps_reviewed_rows_and_adds_new_units_unresolved(self):
        rows = [row for row in self.registry["units"] if compile_time(row)]
        known = {**self.registry, "units": rows[:5] + [{**rows[5], "source_projection": "unresolved"}]}
        draft = platforms.draft_registry(self.linux, self.lock, [unit_graph(rows[3:8])], known)
        by_name = {json.dumps(darwin_policy._signature(row)): row for row in draft["units"]}
        signature = lambda row: json.dumps(darwin_policy._signature(row))  # noqa: E731
        # Reviewed rows stay as they were, including those this graph does not select.
        for row in rows[:5]:
            self.assertEqual(by_name[signature(row)], row)
        # A unit never reviewed is unresolved, however often it is redrafted.
        for row in rows[5:8]:
            self.assertEqual(by_name[signature(row)]["source_projection"], "unresolved")
        self.assertEqual(len(draft["units"]), 8)
        self.assertEqual((draft["scope"], draft["audit_documents"]),
                         (self.registry["scope"], self.registry["audit_documents"]))
        # An unresolved row whose unit is gone is dropped; nothing selected it.
        gone = platforms.draft_registry(self.linux, self.lock, [unit_graph(rows[:5])], known)
        self.assertEqual(gone["units"], sorted(rows[:5], key=lambda r: json.dumps(darwin_policy._signature(r))))

    def test_the_committed_registries_are_fixed_points(self):
        for target in (LINUX, LINUX_ARM):
            platform = platforms.PLATFORMS[target]
            registry = json.loads((REPO / platform.registry).read_bytes())
            again = platforms.draft_registry(platform, self.lock, [unit_graph(registry["units"])], registry)
            self.assertEqual(again, registry)

    def test_cargo_resolves_as_the_named_host(self):
        import os
        import shutil
        import subprocess
        if shutil.which("rustc") is None:
            self.skipTest("no rustc on PATH")
        before = os.environ.get("RUSTC")
        with patch.object(platforms, "host_target", return_value=MAC), platforms._resolving_as(LINUX):
            wrapper = os.environ["RUSTC"]
            version = subprocess.run([wrapper, "-vV"], capture_output=True, text=True, check=True).stdout
            cfg = subprocess.run([wrapper, "--print", "cfg"], capture_output=True, text=True, check=True).stdout
        self.assertIn(f"host: {LINUX}\n", version)
        self.assertIn('target_os="linux"', cfg)
        self.assertIn('target_arch="x86_64"', cfg)
        self.assertEqual(os.environ.get("RUSTC"), before)
        # A host resolves its own graph with its own compiler, unwrapped.
        with patch.object(platforms, "host_target", return_value=LINUX), platforms._resolving_as(LINUX):
            self.assertEqual(os.environ.get("RUSTC"), before)


class CrossPlatformCurrencyTests(unittest.TestCase):
    """A record earned on one platform is checked, not re-earned, on every other."""

    @classmethod
    def setUpClass(cls):
        cls.checker = em.Checker(REPO)
        # Stored records, earned on Apple silicon, that the tree holds current.
        cls.stored = []
        for binding, ref in em.inventory(REPO):
            if len(cls.stored) == 3:
                break
            if ref and cls.checker.check(binding, ref)["current"]:
                cls.stored.append((binding, ref))

    def as_earned_on(self, manifest: dict, target: str) -> dict:
        """`manifest` as the same pass earned on `target`: its policy, compiler host and recipes."""
        platform = platforms.PLATFORMS[target]
        registry = json.loads((REPO / platform.registry).read_bytes())
        rows = {darwin_policy._sha(darwin_policy._canonical(darwin_policy._signature(row))): row
                for row in registry["units"]}
        manifest = json.loads(json.dumps(manifest))
        build = manifest["nix_build"]
        # A unit's identity is the same on every platform; one this platform does not
        # build is not in its registry and would not be selected there.
        selected = {i: rows[i] for i in build["policy"]["selected_unit_ids"] if i in rows}
        build["policy"].update(id=platform.policy, target=target, selected_unit_ids=sorted(selected),
                               audit_selection_sha256=darwin_policy._selection_digest(registry, selected))
        build["tools"]["host"] = target
        drop = set(platforms.DARWIN.recipes) | set(platforms.DARWIN.release_recipes)
        bound = {rel: self.checker.input_digest("evidence_file", rel)
                 for rel in {*platform.recipes, *platform.release_recipes, *platform.policy_sources}}
        build["artifact"]["recipe_sha256"] = {rel: bound[rel] for rel in platform.recipes}
        if "release" in build:
            build["release"]["recipe_sha256"] = {rel: bound[rel] for rel in platform.release_recipes}
        manifest["inputs"] = [entry for entry in manifest["inputs"]
                              if not (entry["kind"] == "evidence_file" and entry["path"] in drop | set(bound))]
        manifest["inputs"] += [{"kind": "evidence_file", "path": rel, "digest": value, "role": "external"}
                               for rel, value in sorted(bound.items())]
        return manifest

    def check(self, binding: dict, ref: str, host: str) -> dict:
        with patch.object(platforms, "host_target", return_value=host):
            return self.checker.check(binding, ref)

    def test_a_record_is_current_on_every_host(self):
        self.assertEqual(len(self.stored), 3)
        for binding, ref in self.stored:
            self.assertEqual(platforms.stored_platform(REPO, ref), MAC)
            for host in (MAC, LINUX, LINUX_ARM, "riscv64gc-unknown-linux-gnu"):
                with self.subTest(record=binding["key"], host=host):
                    self.assertEqual(self.check(binding, ref, host),
                                     {"record": f"{binding['kind']}:{binding['key']}", "current": True, "reasons": []})

    def test_currency_never_asks_which_host_it_runs_on(self):
        binding, ref = self.stored[0]
        with patch.object(platforms, "host_target", side_effect=AssertionError("currency read the host")), \
             patch.object(platforms, "host_platform", side_effect=AssertionError("currency read the host")):
            self.assertTrue(self.checker.check(binding, ref)["current"])

    def test_a_linux_earned_record_is_current_on_apple_silicon_and_back(self):
        binding, ref = self.stored[0]
        earned = self.checker.load(ref)
        for target in (LINUX_ARM, LINUX):
            manifest = self.as_earned_on(earned, target)
            self.assertEqual(platforms.manifest_platform(manifest), target)
            fake = "f" * 63 + ("a" if target == LINUX else "b")
            with patch.dict(self.checker.manifests, {fake: manifest}):
                for host in (MAC, LINUX, LINUX_ARM):
                    with self.subTest(earned=target, host=host):
                        self.assertEqual(self.check(binding, fake, host)["reasons"], [])
                # It is held to the registry of the platform it names, not the host's.
                moved = json.loads(json.dumps(manifest))
                moved["nix_build"]["policy"]["target"] = MAC
                with patch.dict(self.checker.manifests, {fake: moved}):
                    self.assertFalse(self.check(binding, fake, target)["current"])
                # A changed review of a unit it selected owes it, on every host.
                registry = json.loads((REPO / platforms.PLATFORMS[target].registry).read_bytes())
                chosen = set(manifest["nix_build"]["policy"]["selected_unit_ids"])
                for row in registry["units"]:
                    identity = darwin_policy._sha(darwin_policy._canonical(darwin_policy._signature(row)))
                    if identity in chosen:
                        row["repo_source_review"]["reason"] += " (changed)"
                        break
                real = platforms._registry

                def changed(root, platform, registry=registry, target=target, real=real):
                    found, lock = real(root, platform)
                    return (registry if platform.target == target else found), lock
                with patch.object(platforms, "_registry", side_effect=changed):
                    for host in (MAC, target):
                        owed = self.check(binding, fake, host)
                        self.assertFalse(owed["current"])
                        self.assertIn("audit changed", owed["reasons"][0])

    def test_replay_follows_the_platform_that_earned_not_the_gates(self):
        import calibration_status as status
        binding, ref = self.stored[0]
        linux = self.as_earned_on(self.checker.load(ref), LINUX)
        self.assertEqual(platforms.manifest_platform(linux), LINUX)
        earned = {"mac-record": MAC, "linux-record": LINUX}
        records = [{"id": rid, "dependency_manifest": rid} for rid in earned]
        with patch.object(platforms, "stored_platform", side_effect=lambda root, ref: earned.get(ref)):
            for host, replayed in ((MAC, "mac-record"), (LINUX, "linux-record")):
                local, foreign = probe.replayable(records, host)
                self.assertEqual([rec["id"] for rec in local], [replayed])
                self.assertEqual(len(foreign), 1)
            self.assertEqual(probe.replayable(records, LINUX_ARM)[0], [])  # another Linux is another platform
            # A probe attests a measurement of its own platform only, whichever host asks.
            for host in (MAC, LINUX):
                with patch.object(platforms, "host_target", return_value=host):
                    self.assertTrue(status._same_platform(REPO, records[1], {"dependency_manifest": "linux-record"}))
                    self.assertFalse(status._same_platform(REPO, records[1], {"dependency_manifest": "mac-record"}))


if __name__ == "__main__":
    unittest.main()
