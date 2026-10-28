#!/usr/bin/env python3
"""Run conformance reference generators in ``--check`` mode.

Exact float documents are compared with ``numpy.isclose(..., rtol=0)`` inside
the generators (not ``dict !=``). ``generate_frontdoor.py`` compares the
committed CSV columns the same way. None of these rerun a Monte Carlo study.

    python3 scripts/check_conformance_references.py
"""

from __future__ import annotations

import subprocess
import sys

from _repo import REPO_ROOT

GENERATORS = (
    "conformance/response/joint_panel/reference.py",
    "conformance/estimate/rd_sharp/reference.py",
    "conformance/estimate/cpdag_ate_envelope/reference.py",
    "conformance/estimate/pag_ate_envelope_identified/reference.py",
    "conformance/estimate/path_specific_edge_gformula/reference.py",
    "conformance/gcm/sparse_feedback_equilibrium/reference.py",
    "conformance/gcm/external_mechanism_feedback/reference.py",
    "conformance/gcm/ou_trajectory_likelihood/reference.py",
    "conformance/gcm/nonlinear_sde_ensemble/reference.py",
    "conformance/estimate/temporal_class_envelope/identified_pag_reference.py",
    "conformance/validate/mnar_sensitivity/reference.py",
    "conformance/design/adaptive_policy_search/reference.py",
    "conformance/bayesian/posterior_moments/generate.py",
    "conformance/estimate/noisy_estimators/generate_frontdoor.py",
    "conformance/composition/release_boundary/reference.py",
    "conformance/composition/rollout/reference.py",
    "conformance/composition/source_projection/reference.py",
    "conformance/estimate/frontdoor_functional/reference.py",
    "conformance/recalculation/bayesian_retention/reference.py",
    "conformance/recalculation/checked_static_design/reference.py",
    "conformance/recalculation/design_ar/reference.py",
    "conformance/recalculation/temporal_history/reference.py",
    "conformance/response/temporal_confounded_pulse/reference.py",
    "conformance/transport/nested_fisher/reference.py",
    "conformance/transport/quadratic_dose/reference.py",
    "conformance/attribution/anomaly_root_cause/reference.py",
    "conformance/attribution/arrow_strength/reference.py",
    "conformance/attribution/distribution_change_grid/reference.py",
    "conformance/attribution/mechanism_unit_change/reference.py",
    "conformance/attribution/path_allocation/reference.py",
    "conformance/bayesian/graph_effect_envelope/reference.py",
    "conformance/bayesian/prior_bank_ess/reference.py",
    "conformance/bayesian/prior_bank_power_mixture/reference.py",
    "conformance/bayesian/prior_conjugate_moment_match/reference.py",
    "conformance/estimate/bayesian_structural_transport/reference.py",
    "conformance/validate/riesz_sensitivity/reference.py",
    "conformance/attribution/distribution_change_y_shift/reference.py",
    "conformance/attribution/structure_change_grid/reference.py",
    "conformance/bayesian/conjugate_gaussian/reference.py",
    "conformance/context/prediction_smoke/reference.py",
    "conformance/estimate/aipw/reference.py",
    "conformance/estimate/distance_matching/reference.py",
    "conformance/estimate/efficient_backdoor/reference.py",
    "conformance/estimate/frontdoor/reference.py",
    "conformance/estimate/glm_adjustment/reference.py",
    "conformance/estimate/iv_2sls/reference.py",
    "conformance/estimate/propensity_ipw/reference.py",
    "conformance/gcm/do_sampling_grid/reference.py",
    "conformance/gcm/do_sampling_weighting/reference.py",
    "conformance/gcm/gcm_fit_intervene/reference.py",
    "conformance/population_time/effect_constancy_consumers/reference.py",
    "conformance/response/intervention_response/reference.py",
    "conformance/response/observation_primitives/reference.py",
    "conformance/response/staged_derivatives/reference.py",
    "conformance/response/temporal_dose_horizon/reference.py",
    "conformance/response/temporal_observation/reference.py",
    "conformance/response/trial_transport/reference.py",
    "conformance/response/two_point_curve_average_effect/reference.py",
)


def python_executable() -> str:
    venv = REPO_ROOT / "python" / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def has_inline_environment(path: Path) -> bool:
    """A PEP 723 header means the generator declares (and locks) its own environment."""
    with path.open(encoding="utf-8") as handle:
        return any(line.strip() == "# /// script" for _, line in zip(range(40), handle))


def main() -> int:
    executable = python_executable()
    failed = False
    for rel in GENERATORS:
        path = REPO_ROOT / rel
        if has_inline_environment(path):
            command = ["uv", "run", "--script", str(path), "--check"]
        else:
            command = [executable, str(path), "--check"]
        print(" ".join(command))
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            failed = True
    if failed:
        print("conformance reference check FAILED", file=sys.stderr)
        return 1
    print("conformance reference check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
