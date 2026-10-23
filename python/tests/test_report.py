"""The readable report of a prepared handle, end to end from Python.

``prepared.report(format)`` renders Markdown or one self-contained HTML page from the
records the handle holds: the executed claim and contract, or the prepared contract alone,
with the plan view. The renderer is the Rust one; Python only selects the format. Golden
files pin the output for a licensed static effect, a route that executes without a licence,
a Bayesian effect and a temporal effect; regenerate them deliberately with
``ANTECEDENT_REGENERATE_REPORTS=1 python -m pytest python/tests/test_report.py`` and review
the diff. A cell the support matrix refuses raises at prepare, so no handle exists to report
on; that refusal is pinned here and its report is the Rust one.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.estimation import PreparedAnalysis
from antecedent.policy import ConditionalDoseResponse

GOLDEN = Path(__file__).parent / "golden" / "report"
REGENERATE = "ANTECEDENT_REGENERATE_REPORTS"

_SECTIONS = [
    "Summary",
    "Claim",
    "Identification",
    "Assumptions",
    "Diagnostics and evidence",
    "Uncertainty",
    "Support and licence",
    "Refusals and limits",
    "Program plan",
    "Identities",
]

_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


def _table(n: int = 240, names: tuple[str, str, str] = ("t", "y", "z")) -> dict[str, np.ndarray]:
    t_name, y_name, z_name = names
    z = np.array([math.sin(i / 17.0) for i in range(n)])
    t = (z + np.array([math.cos(i / 11.0) for i in range(n)]) > 0).astype(float)
    y = 1.0 + 2.0 * t + 0.8 * z
    return {t_name: t, y_name: y, z_name: z}


def _licensed():
    prepared = ant.prepare(
        _table(),
        graph=_EDGES,
        query=ant.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
        seed=3,
    )
    prepared.estimate(_table(), seed=3)
    return prepared


def _bayesian():
    prepared = ant.prepare(
        _table(),
        graph=_EDGES,
        query=ant.AverageEffect("t", "y"),
        inference=ant.Bayesian(backend="conjugate", n_draws=64, prior_scale=30.0),
        refute="none",
        bootstrap=0,
        seed=3,
    )
    prepared.estimate(_table(), seed=3)
    return prepared


def _dose_data():
    doses = [-1.0, -0.5, 0.0, 0.5, 1.0]
    groups = [group for group in ("control", "treated") for _ in doses]
    dose = doses * 2
    outcome = [
        (1.0 if group == "control" else 10.0) + 2.0 * value
        for group, value in zip(groups, dose, strict=True)
    ]
    return {
        "y": np.asarray(outcome),
        "dose": np.asarray(dose),
        "group": groups,
        "density": np.full(len(dose), 0.5),
    }


def _unlicensed():
    prepared = ant.prepare(
        _dose_data(),
        query=ConditionalDoseResponse(
            outcome="y",
            dose="dose",
            baseline_group="group",
            dose_density="density",
            target_doses=(0.0, 0.5),
            bandwidth=0.6,
            density_provenance="known",
        ),
    )
    prepared.estimate()
    return prepared


def _temporal():
    n = 400
    rng = np.random.default_rng(7)
    pressure = np.sin(0.04 * np.arange(n))
    defect = np.zeros(n)
    for t in range(1, n):
        defect[t] = 0.9 * pressure[t - 1] + 0.05 * rng.uniform(-1.0, 1.0)
    data = {"pressure": pressure, "defect": defect}
    prepared = PreparedAnalysis.prepare(
        data,
        graph=[("pressure", 1, "defect", 0)],
        query=ant.PulseEffect(
            treatment="pressure",
            outcome="defect",
            treatment_lag=1,
            horizon_steps=1,
            active_level=1.0,
        ),
        refute=False,
        seed=7,
    )
    prepared.estimate(data, seed=7)
    return prepared


SCENARIOS: dict[str, Callable[[], PreparedAnalysis]] = {
    "licensed_static_effect": _licensed,
    "unlicensed_executing_route": _unlicensed,
    "bayesian_effect": _bayesian,
    "temporal_effect": _temporal,
}


def _check_golden(name: str, extension: str, actual: str) -> None:
    path = GOLDEN / f"{name}.{extension}"
    if os.environ.get(REGENERATE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8", newline="\n")
        return
    if not path.is_file():
        pytest.fail(f"golden report {path} is missing; generate it with {REGENERATE}=1")
    assert actual == path.read_text(encoding="utf-8"), (
        f"report {name}.{extension} differs from its golden file; regenerate deliberately "
        f"with {REGENERATE}=1 and review the diff"
    )


def _fact(markdown: str, key: str) -> str:
    prefix = f"- **{key}:** "
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise AssertionError(f"no fact {key!r} in\n{markdown}")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_report_matches_its_golden_markdown_and_html(name: str) -> None:
    prepared = SCENARIOS[name]()
    _check_golden(name, "md", prepared.report("markdown"))
    _check_golden(name, "html", prepared.report("html"))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_report_is_deterministic_from_the_study_to_the_bytes(name: str) -> None:
    first = SCENARIOS[name]()
    second = SCENARIOS[name]()
    for form in ("markdown", "html"):
        rendered = first.report(form)
        assert rendered == first.report(form), "rendering twice differs"
        assert rendered == second.report(form), f"rebuilding the study changed the {form} report"
    assert first.report() == first.report("markdown"), "markdown is the default"


def test_report_carries_every_section_and_the_claim_of_the_execution() -> None:
    prepared = _licensed()
    markdown = prepared.report("markdown")
    assert re.findall(r"^## (.+)$", markdown, re.MULTILINE) == _SECTIONS
    result = prepared.last_result
    assert result is not None
    assert float(_fact(markdown, "Value")) == pytest.approx(result.ate, abs=1e-9)
    assert _fact(markdown, "Support status") == "licensed"
    assert _fact(markdown, "Claim kind") == "point"
    assert "adjustment" in markdown.lower() or "backdoor" in markdown.lower()


def test_report_before_an_estimate_reads_the_contract_and_executes_nothing() -> None:
    prepared = ant.prepare(
        _table(),
        graph=_EDGES,
        query=ant.AverageEffect("t", "y"),
        refute="none",
        bootstrap=0,
        seed=3,
    )
    markdown = prepared.report("markdown")
    assert prepared.last_result is None, "a report executes nothing"
    assert "a contract before execution" in markdown
    assert "No claim is attached" in markdown
    assert "## Claim" not in markdown
    assert re.findall(r"^## (.+)$", markdown, re.MULTILINE) == [
        section for section in _SECTIONS if section != "Claim"
    ]
    assert prepared.report("markdown") == markdown, "reading it twice changes nothing"
    prepared.estimate(_table(), seed=3)
    assert "## Claim" in prepared.report("markdown")


def test_bayesian_report_states_its_posterior_plan_and_calibration() -> None:
    markdown = _bayesian().report("markdown")
    assert _fact(markdown, "Inference").lower() == "bayesian"
    assert "posterior" in markdown
    assert _fact(markdown, "Calibration status")
    assert _fact(markdown, "Governing record")


def test_the_unlicensed_route_states_that_no_licence_applies() -> None:
    markdown = _unlicensed().report("markdown")
    assert _fact(markdown, "Support status") != "licensed"
    assert "## Refusals and limits" in markdown


def test_a_refused_cell_raises_at_prepare_so_no_handle_reports_on_it() -> None:
    data = _table()
    with pytest.raises(Exception):  # noqa: B017 - the refusal's type is pinned by its own suites
        ant.prepare(
            data,
            graph=_EDGES,
            query=ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
            refute="none",
            bootstrap=0,
        )


def test_an_unknown_format_is_a_typed_refusal() -> None:
    prepared = _licensed()
    with pytest.raises(CausalValueError, match="unknown report format"):
        prepared.report("pdf")  # type: ignore[arg-type]


def test_html_is_inert_and_escapes_every_value() -> None:
    hostile = ("t<b>", 'y&"q"\'', "z|`<script>alert(1)</script>*_[l](u)")
    prepared = ant.prepare(
        _table(names=hostile),
        graph=[(hostile[2], hostile[0]), (hostile[2], hostile[1]), (hostile[0], hostile[1])],
        query=ant.AverageEffect(hostile[0], hostile[1]),
        refute="none",
        bootstrap=0,
        seed=3,
    )
    prepared.estimate(_table(names=hostile), seed=3)

    html = prepared.report("html")
    assert "<script" not in html
    assert "<b>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "y&amp;&quot;q&quot;&#39;" in html
    assert "t&lt;b&gt;" in html
    assert html.startswith("<!doctype html>")
    assert "default-src 'none'" in html
    assert "http://" not in html and "https://" not in html

    markdown = prepared.report("markdown")
    assert "<script>" not in markdown and "<b>" not in markdown
    assert "\\<script\\>alert(1)\\</script\\>" in markdown
    assert "z\\|\\`\\<script" in markdown
