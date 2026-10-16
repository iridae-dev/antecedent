"""The transported-route surface keeps checked constructors and typed results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_transport_route_types_and_misspelled_names_are_rejected(tmp_path):
    fixture = tmp_path / "positive.py"
    mypy = [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "mypy-cache")]
    fixture.write_text("""from typing import assert_type
from antecedent import cross_world_execution as cwx
from antecedent import discretized_composition as dc
from antecedent import inverse_sources as src
from antecedent import mechanism_composition as mc
from antecedent import transported_response as tr
from antecedent.transport import revalidation as rv
from antecedent.transport import sensitivity as ts

def checked(
    response: tr.TransportedResponse,
    composition: mc.PreparedMechanismComposition,
    account: cwx.WholePathAccount,
    scene: rv.TransportScene,
) -> None:
    source = src.transported_forward_source(response, accepts_transported=True)
    assert_type(source, src.ForwardSource)
    assert_type(source.origin, src.ForwardOrigin)
    assert_type(source.origin.transported, bool)
    assert_type(response.identity, str)
    assert_type(response.means, tuple[float, ...] | None)
    assert_type(response.receipt, tr.TransportReceipt)

    discretized = dc.discretize(composition, 4)
    assert_type(discretized, dc.PreparedDiscretizedComposition)
    contrast = discretized.mean_contrast(dc.DiscreteContrast.ate("Y", "X", 3, 0))
    assert_type(contrast, dc.DiscretizedContrast)
    assert_type(contrast.error_bound, float)

    assert_type(account.unit_mean_sampling_variance, float | None)
    assert_type(account.units[0].predictive_variance, float)

    report = rv.revalidate(None, scene)
    assert_type(report, rv.TransportRevalidation)
    assert_type(report.claims, tuple[rv.ClaimRevalidation, ...])
    assert_type(report.claim("effect").calibration_withdrawn, bool)
    ts.transport_sensitivity
""")
    result = subprocess.run(
        [*mypy, str(fixture)], cwd=Path(__file__).parents[1], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fixture.write_text("""from antecedent import inverse_sources as src
from antecedent import transported_response as tr
from antecedent.transport import revalidation as rv

def misuse(response: tr.TransportedResponse) -> None:
    src.transported_forward_source(response, accept_transported=True)
    rv.ObservedEffect("claim:effect", "stale", "text")
    tr.TransportDeclaration(route="declared")
""")
    result = subprocess.run(
        [*mypy, str(fixture)], cwd=Path(__file__).parents[1], text=True, capture_output=True
    )
    assert result.returncode != 0
    assert 'Unexpected keyword argument "accept_transported"' in result.stdout, result.stdout
    assert "incompatible" in result.stdout.lower(), result.stdout
