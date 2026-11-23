"""Evidence primitives shared by the method inventories.

A method child and a method spec rest on the same few facts: a named test exists and is run, a named artifact consumer
reads the artifact, and the uncertainty a route publishes is the one its implementation
or its coverage records show. This module answers those questions from the sources, so
no inventory has to infer them from prose.

    fixture     path[::test_name] of a registered test, script or fixture record
    consumer    path[::symbol] of a non-test source that reads the artifact
    interval    the uncertainty kind an implementation or a measured record shows
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from functools import cache
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import load_toml
from test_evidence import rust_test_declaration_problems, static_python_test

UNLISTED = {"", "unlisted", "none"}
REGISTRY = "parity/test_registry.toml"
ROOTS = "crates|python|parity|scripts|docs|examples|conformance|provenance|tests"
PATH_TOKEN = re.compile(rf"\b(?:{ROOTS})(?:/[A-Za-z0-9_.\-]+)+/?(?:::[A-Za-z0-9_]+)*")
TOKEN = re.compile(r"^(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+(?:::[A-Za-z0-9_]+)*$")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The interval vocabulary an execution records (antecedent_core::IntervalMethod::as_str) and
# the uncertainty kind of the counterpart inventory each one is. A coverage record also names
# the standard-error kind; a Monte Carlo standard error makes a band a resampling procedure.
INTERVAL_KIND = {
    "analytic_se": "analytic",
    "anderson_rubin": "analytic",
    "bootstrap_se": "design_aware_resampling",
    "circular_block_se": "design_aware_resampling",
    "posterior_quantile": "posterior",
    "posterior_absolute_deviation": "posterior",
    "unit_posterior_quantile": "posterior",
    "identified_set": "enclosure",
    "simultaneous_band": "analytic",
    "none": "none",
}
# Interval constructions that coverage records name beyond the closed execution vocabulary.
RECORD_INTERVAL_KIND = INTERVAL_KIND | {
    "simultaneous_wald_normal": "analytic",
    "bonferroni_confidence_set": "analytic",
    "wald_normal": "analytic",
    "wald_z": "analytic",
    "confidence_sequence": "analytic",
    "wilson_effective_sample": "analytic",
    "student_t_iid": "analytic",
    "independent_gaussian_study_t": "analytic",
    "percentile_bootstrap": "design_aware_resampling",
    "normal_approximation": "analytic",
    "student_t_hc3_sandwich": "analytic",
    "posterior_eti": "posterior",
    "hksj_t": "analytic",
    "repeated_confidence_interval": "analytic",
    "population_linear_aipw_score": "analytic",
    "student_analytic": "analytic",
    "normal_delta_method_clipped": "analytic",
    "delta_method_wald": "analytic",
    "student_t_bartlett_hac": "analytic",
    "student_t_cluster_robust": "analytic",
    "bootstrap_studentized": "design_aware_resampling",
    "block_percentile_bootstrap": "design_aware_resampling",
    "balanced_population_linear_score": "analytic",
    "balanced_population_score": "analytic",
    "population_linear_atc_hc3": "analytic",
    "student_t_prediction_interval": "analytic",
    "percentile_bootstrap_or_logical_support": "design_aware_resampling",
    "bootstrap_bca": "design_aware_resampling",
    "student_t_sandwich_effective_n": "analytic",
    "bootstrap_basic": "design_aware_resampling",
    "bootstrap_percentile": "design_aware_resampling",
}
VARIANT_METHOD = {
    "AnalyticSe": "analytic_se",
    "BootstrapSe": "bootstrap_se",
    "PosteriorQuantile": "posterior_quantile",
    "PosteriorAbsoluteDeviation": "posterior_absolute_deviation",
    "UnitPosteriorQuantile": "unit_posterior_quantile",
    "IdentifiedSet": "identified_set",
    "CircularBlockSe": "circular_block_se",
    "SimultaneousBand": "simultaneous_band",
    "AndersonRubin": "anderson_rubin",
}
INTERVAL_REF = re.compile(r"\bIntervalMethod::(" + "|".join(VARIANT_METHOD) + r")\b")


class Repo:
    """Read-only view of the files the reconciliation matches against."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._text: dict[str, str | None] = {}
        self._grep: dict[str, list[str]] = {}
        self._corpus: dict[str, str] | None = None
        self._cargo: dict[str, dict] = {}
        self._registry: set[str] | None = None
        self._registered: dict[str, bool] = {}

    def load(self, rel: str) -> dict:
        return load_toml(self.root / rel)

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def text(self, rel: str) -> str | None:
        if rel not in self._text:
            path = self.root / rel
            self._text[rel] = (
                path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
            )
        return self._text[rel]

    def sources_reading(self, symbol: str) -> list[str]:
        """Rust and python sources under crates/ and python/antecedent that mention the symbol."""
        if symbol not in self._grep:
            found: list[str] = []
            for base in ("crates", "python/antecedent"):
                if not (self.root / base).is_dir():
                    continue
                proc = subprocess.run(
                    [
                        "grep",
                        "-rlw",
                        "--include=*.rs",
                        "--include=*.py",
                        "--exclude-dir=tests",
                        "--exclude-dir=target",
                        "--",
                        symbol,
                        base,
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                found.extend(sorted(proc.stdout.split()))
            self._grep[symbol] = found
        return self._grep[symbol]

    def cargo(self, crate_dir: str) -> dict:
        if crate_dir not in self._cargo:
            path = self.root / crate_dir / "Cargo.toml"
            self._cargo[crate_dir] = (
                load_toml(path) if path.is_file() else {}
            )
        return self._cargo[crate_dir]

    def registry_paths(self) -> set[str]:
        if self._registry is None:
            try:
                data = self.load(REGISTRY)
            except (OSError, tomllib.TOMLDecodeError):
                data = {}
            self._registry = {t.get("path", "") for t in data.get("test", [])}
        return self._registry

    def corpus(self) -> dict[str, str]:
        """Sources that run things: registered Rust tests, python tests, scripts and gates."""
        if self._corpus is None:
            files: dict[str, str] = {}
            patterns = (
                "crates/*/tests/*.rs",
                "crates/*/tests/**/*.rs",
                "python/tests/*.py",
                "python/tests/**/*.py",
                "scripts/*.py",
                "scripts/*.sh",
                "examples/**/*.py",
                "examples/**/*.rs",
            )
            for pattern in patterns:
                for path in sorted(self.root.glob(pattern)):
                    rel = path.relative_to(self.root).as_posix()
                    if rel in files or not path.is_file():
                        continue
                    if rel.endswith(".rs") and not self.rust_test_registered(rel):
                        continue
                    files[rel] = path.read_text(encoding="utf-8", errors="replace")
            self._corpus = files
        return self._corpus

    def rust_test_registered(self, rel: str) -> bool:
        match = re.match(r"(crates/[^/]+)/(tests/.+\.rs)$", rel)
        return bool(match) and match.group(2) in self.crate_tests(match.group(1))

    def crate_tests(self, crate_dir: str) -> set[str]:
        """Crate-relative test sources that cargo compiles into some test target."""
        if crate_dir in self._registered:
            return self._registered[crate_dir]
        manifest = self.cargo(crate_dir)
        explicit = {e.get("path", f"tests/{e.get('name')}.rs") for e in manifest.get("test", [])}
        tests_dir = self.root / crate_dir / "tests"
        found = set(explicit)
        if manifest.get("package", {}).get("autotests", True):
            found |= {f"tests/{p.name}" for p in tests_dir.glob("*.rs")}
        suite_map = tests_dir / "suite_map.txt"
        if suite_map.is_file():
            for raw in suite_map.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                suite, rest = line.split(":", 1)
                if f"tests/{suite.strip()}.rs" in explicit:
                    found |= {
                        f"tests/{part.strip()}.rs" for part in rest.split(",") if part.strip()
                    }
        # a module that a compiled test pulls in with #[path = "..."] runs with it
        pending = list(found)
        while pending:
            host = pending.pop()
            body = self.text(f"{crate_dir}/{host}") or ""
            for included in re.findall(r'#\[path\s*=\s*"([^"]+\.rs)"\]', body):
                target = f"tests/{included.rsplit('/', 1)[-1]}"
                if target not in found and (self.root / crate_dir / target).is_file():
                    found.add(target)
                    pending.append(target)
        self._registered[crate_dir] = found
        return found


def split_fixture(token: str) -> tuple[str, str | None]:
    path, _, name = token.partition("::")
    return path, (name.rsplit("::", 1)[-1] if name else None)


def paths_in(value: object) -> list[str]:
    """Path-like tokens named by a string or a list of strings; prose is ignored."""
    parts = [value] if isinstance(value, str) else [v for v in (value or []) if isinstance(v, str)]
    found: list[str] = []
    for part in parts:
        if part.strip() in UNLISTED:
            continue
        found.extend(
            m.group(0).rstrip("/") if "::" not in m.group(0) else m.group(0)
            for m in PATH_TOKEN.finditer(part)
        )
    return list(dict.fromkeys(found))


def record_evidence(repo: Repo, path: str) -> list[str]:
    """Test citations held by a fixture record: every `evidence` and `test` entry in it."""
    try:
        data = tomllib.loads(repo.text(path) or "")
    except tomllib.TOMLDecodeError:
        return []
    found: list[str] = []

    def walk(node: object, key: str = "") -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, name)
        elif isinstance(node, list):
            for value in node:
                walk(value, key)
        elif isinstance(node, str) and key in ("evidence", "test", "tests", "public_workflow"):
            found.extend(paths_in(node))

    walk(data)
    return list(dict.fromkeys(found))


def fixture_problem(repo: Repo, token: str) -> str | None:
    """None when the fixture exists and a registered test runs it, else the exact reason."""
    path, name = split_fixture(token)
    if not repo.exists(path):
        return f"fixture {path} does not exist"
    if name is not None:
        body = repo.text(path)
        if body is None or not re.search(rf"\b(?:fn|def|class)\s+{re.escape(name)}\b", body):
            return f"fixture {path} defines no `{name}`"
    if path.endswith(".rs") and "/tests/" in path:
        return (
            None
            if repo.rust_test_registered(path)
            else f"fixture {path} is not a registered test target"
        )
    if path.endswith(".rs") and "/src/" in path:
        body = repo.text(path) or ""
        if path in repo.registry_paths() or "#[cfg(test)]" in body or "#[test]" in body:
            return None
        return f"fixture {path} holds no unit tests and is not in the test registry"
    if re.match(r"python/tests/.+\.py$", path):
        return None
    if path.endswith(".toml"):
        listed = record_evidence(repo, path)
        if listed:
            for token in listed:
                if "::" not in token and "/src/" in token:  # a cited source file, not a test
                    if not repo.exists(token):
                        return f"fixture record {path} cites {token}, which does not exist"
                    continue
                problem = fixture_problem(repo, token)
                if problem:
                    return f"fixture record {path} cites evidence that fails: {problem}"
            return None
    if path.startswith("examples/") and path.endswith(".py"):
        catalog = repo.text("examples/catalog.toml") or ""
        if path in catalog and repo.rust_test_registered("crates/antecedent/tests/examples_run.rs"):
            return None
    base = path.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    for rel, body in repo.corpus().items():
        if rel == path:
            continue
        if path in body or base in body:
            return None
        # a golden file a test finds by scenario name, in a directory the test names
        if "/golden/" in path and f'"{stem}"' in body:
            return None
    if path.startswith("scripts/") and path.endswith(".py"):
        gates = (repo.text("scripts/gate_selftests.sh") or "") + (
            repo.text("scripts/gate_release.sh") or ""
        )
        if base in gates:
            return None
    return f"fixture {path} is read by no registered test, gate or script"


def owner_problem(repo: Repo, owner: str, symbol: str | None = None) -> str | None:
    path = owner.split(" ")[0].rstrip("/")
    if not repo.exists(path):
        return f"owner {path} does not exist"
    if symbol:
        body = repo.text(path)
        if body is None:  # a directory owner defines the symbol somewhere beneath it
            proc = subprocess.run(
                ["grep", "-rlw", "--", symbol, path],
                cwd=repo.root,
                capture_output=True,
                text=True,
                check=False,
            )
            if not proc.stdout.strip():
                return f"owner {path} defines no `{symbol}`"
        elif not re.search(rf"\b{re.escape(symbol)}\b", body):
            return f"owner {path} defines no `{symbol}`"
    return None


SYMBOL = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+|[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"
)


def consumer_problem(
    repo: Repo, texts: list[str], symbol: str | None, owners: list[str]
) -> str | None:
    """A consumer is a path that exists or a symbol that source code outside tests reads."""
    owner_set = {o.split(" ")[0].rstrip("/") for o in owners}
    paths = paths_in(texts)
    for path in paths:
        if not repo.exists(split_fixture(path)[0]):
            return f"consumer {split_fixture(path)[0]} does not exist"
    readers = [p for p in paths if split_fixture(p)[0] not in owner_set]
    if readers:
        return None
    symbols = [symbol] if symbol else []
    for text in texts:
        if text.strip() in UNLISTED:
            continue
        symbols.extend(m.group(0).rsplit("::", 1)[-1] for m in SYMBOL.finditer(text))
    if not paths and not symbols and not any(t.strip() not in UNLISTED for t in texts):
        return "no consumer is named"
    for name in symbols:
        if (name == symbol or len(name) > 5) and any(
            src not in owner_set for src in repo.sources_reading(name)
        ):
            return None
    if paths:
        return "the only consumer named is the owner itself"
    return (
        f"no source reads the consumer named ({texts[0][:60]!r}...)"
        if texts
        else "no consumer is named"
    )



class Evidence:
    """Read-only evidence over one tree (default the repository)."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.repo = Repo(root)

    # ---- tokens -------------------------------------------------------------------------
    @staticmethod
    def tokens(value: object) -> list[str]:
        parts = [str(v) for v in value] if isinstance(value, list) else str(value).split(";")
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def split(token: str) -> tuple[str, str | None]:
        path, _, name = token.partition("::")
        return path, (name.rsplit("::", 1)[-1] if name else None)

    def text(self, path: str) -> str:
        return self.repo.text(path) or ""

    # ---- fixtures -----------------------------------------------------------------------
    def is_test_name(self, path: str, name: str) -> bool:
        if path.endswith(".py"):
            return not static_python_test(self.root / path, name)
        if path.endswith(".rs"):
            return not rust_test_declaration_problems(self.root / path, name)
        return True

    def fixture_findings(self, token: str) -> list[str]:
        if not TOKEN.match(token):
            return [f"fixture {token!r} is not a path[::test] token"]
        path, name = self.split(token)
        problem = fixture_problem(self.repo, token)
        if problem:
            return [problem]
        if name and path.endswith((".rs", ".py")) and not self.is_test_name(path, name):
            return [f"fixture {path}::{name} is not a test function"]
        return []

    # ---- consumers ----------------------------------------------------------------------
    @staticmethod
    def is_test_path(path: str) -> bool:
        return "/tests/" in path or "/test_" in path or path.startswith(("scripts/test_",))

    def consumer_findings(self, token: str) -> list[str]:
        if not TOKEN.match(token):
            return [f"consumer {token!r} is not a path[::symbol] token"]
        path, symbol = self.split(token)
        if not self.repo.exists(path):
            return [f"consumer {path} does not exist"]
        if symbol:
            body = self.text(path)
            if not re.search(rf"\b{re.escape(symbol)}\b", body):
                return [f"consumer {path} does not mention `{symbol}`"]
        return []

    # ---- relevance ----------------------------------------------------------------------
    def anchors(self, child: dict) -> set[str]:
        """Names a test of this method has to mention somewhere in its file."""
        found: set[str] = set()
        target = str(child.get("target", ""))
        if IDENT.fullmatch(target) and target not in UNLISTED:
            found.add(target)
        for module in str(child.get("module", "")).split(";"):
            module = module.strip().split(" ")[0]
            if module.endswith(".rs"):
                stem = Path(module).stem
                found.add(Path(module).parent.name if stem == "mod" else stem)
        for part in re.split(r"[^A-Za-z0-9_.]+", str(child.get("inference", ""))):
            if len(part) > 4 and ("." in part or "_" in part):
                found.add(part)
        for module in str(child.get("module", "")).split(";"):
            found |= self.public_symbols(module.strip().split(" ")[0])
        route = str(child.get("route", ""))
        for part in route.split("."):
            if len(part) > 5 and part not in (
                "supported",
                "graphless",
                "capability",
                "execution_only",
                "closed",
                "unlisted",
            ):
                found.add(part)
        return {a for a in found if len(a) > 3}

    def public_symbols(self, module: str) -> set[str]:
        """Public type and function names a module defines: a test that uses one runs the module."""
        if not module.endswith(".rs"):
            return set()
        names = re.findall(
            r"^\s*pub(?:\([a-z]+\))?\s+(?:const\s+)?(?:struct|enum|fn|trait)\s+([A-Za-z_][A-Za-z0-9_]*)",
            self.text(module),
            re.M,
        )
        return {n for n in names if len(n) > 7}

    def mentions(self, path: str, anchors: set[str]) -> bool:
        body = self.text(path)
        squashed = re.sub(r"[^a-z0-9]", "", body.lower())
        return any(a in body or re.sub(r"[^a-z0-9]", "", a.lower()) in squashed for a in anchors)

    # ---- uncertainty --------------------------------------------------------------------
    @cache  # noqa: B019
    def coverage_records(self) -> list[dict]:
        return tomllib.loads(self.text("parity/coverage_records.toml")).get("record", [])

    def record_kind(self, record: dict) -> str | None:
        method = record.get("interval_method", "")
        kind = RECORD_INTERVAL_KIND.get(method)
        if kind is None:
            return None
        if method == "simultaneous_band" and record.get("se_kind") == "monte_carlo":
            return "design_aware_resampling"
        return kind

    def implementation_intervals(self, sources: list[str]) -> set[str]:
        """Interval methods the implementing sources construct (a closed-vocabulary scan)."""
        found: set[str] = set()
        for source in sources:
            path = source.split(" ")[0].split("::")[0]
            files = [path] if (self.root / path).is_file() else self._files_under(path)
            for file in files:
                for match in INTERVAL_REF.finditer(self.text(file)):
                    found.add(VARIANT_METHOD[match.group(1)])
        return found

    def _files_under(self, rel: str) -> list[str]:
        base = self.root / rel
        if not base.is_dir():
            return []
        return [p.relative_to(self.root).as_posix() for p in sorted(base.rglob("*.rs"))]
