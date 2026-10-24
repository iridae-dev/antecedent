"""Evidence primitives shared by the method inventories.

A method child, a route-manifest method, a counterpart row and a readiness verdict all
rest on the same few facts: a named test exists and is run, a named artifact consumer
reads the artifact, and the uncertainty a route publishes is the one its implementation
or its coverage records show. This module answers those questions from the sources, so
no inventory has to infer them from prose.

    fixture     path[::test_name] of a registered test, script or fixture record
    consumer    path[::symbol] of a non-test source that reads the artifact
    interval    the uncertainty kind an implementation or a measured record shows
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import reconcile_release_scope as scope  # noqa: E402

UNLISTED = {"", "unlisted", "none"}
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
    "UnitPosteriorQuantile": "unit_posterior_quantile",
    "IdentifiedSet": "identified_set",
    "CircularBlockSe": "circular_block_se",
    "SimultaneousBand": "simultaneous_band",
    "AndersonRubin": "anderson_rubin",
}
INTERVAL_REF = re.compile(r"\bIntervalMethod::(" + "|".join(VARIANT_METHOD) + r")\b")


class Evidence:
    """Read-only evidence over one tree (default the repository)."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.repo = scope.Repo(root)

    # ---- tokens -------------------------------------------------------------------------
    @staticmethod
    def tokens(value: object) -> list[str]:
        if isinstance(value, list):
            parts = [str(v) for v in value]
        else:
            parts = str(value).split(";")
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def split(token: str) -> tuple[str, str | None]:
        path, _, name = token.partition("::")
        return path, (name.rsplit("::", 1)[-1] if name else None)

    def text(self, path: str) -> str:
        return self.repo.text(path) or ""

    # ---- fixtures -----------------------------------------------------------------------
    def is_test_name(self, path: str, name: str) -> bool:
        body = self.text(path)
        if path.endswith(".py"):
            return bool(re.search(rf"^\s*def {re.escape(name)}\b", body, re.M)) and (
                name.startswith("test") or "/tests/" in path or path.startswith("scripts/test")
            )
        if path.endswith(".rs"):
            return bool(
                re.search(
                    rf"#\[(?:tokio::)?test[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub )?(?:async )?fn {re.escape(name)}\b",
                    body,
                )
            )
        return True

    def fixture_findings(self, token: str) -> list[str]:
        if not TOKEN.match(token):
            return [f"fixture {token!r} is not a path[::test] token"]
        path, name = self.split(token)
        problem = scope.fixture_problem(self.repo, token)
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
            if len(part) > 5 and part not in ("licensed", "graphless", "execution_only", "closed", "unlisted"):
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
    @lru_cache(maxsize=None)  # noqa: B019
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
