#!/usr/bin/env bash
# Named-test existence: every filter the feature gates used to run through
# scripts/counted_cargo.sh still selects at least one non-ignored test.
#
# Those gates no longer re-run Rust suites (`cargo nextest run --workspace` in the
# Rust job runs them), so a renamed test or module would silently drop out of what
# the filter names. This reads `cargo nextest list`, which reuses the Rust job's
# build, and applies libtest's substring filter to each suite's test names.
# Run it after the Rust job's test steps (same profile and features):
#   bash scripts/gate_named_tests.sh
#   bash scripts/gate_named_tests.sh --self-test   # canned listings, no compile
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - "$@" <<'PY'
import json
import subprocess
import sys

# (listing, package, target, filter). `target` is "lib", a `--test` target name,
# or "*" for every test binary of the package (a filter without --lib/--test).
# `listing` names the cargo feature set the Rust job builds the test under.
NAMED = [
    # gate_bayesian.sh
    *(("workspace", "antecedent-discovery", "lib", f) for f in (
        "graph_posterior::", "exact_enumeration::", "structure_mcmc::", "order_mcmc::",
        "ci_screened_posterior::", "dbn_posterior::")),
    ("workspace", "antecedent-estimate", "lib", "bayesian"),
    ("workspace", "antecedent-estimate", "lib", "envelope"),
    ("workspace", "antecedent-validate", "lib", "bayesian_checks"),
    ("workspace", "antecedent-io", "lib", "posterior"),
    ("workspace", "antecedent-io", "lib", "prior_bank"),
    ("workspace", "antecedent-data", "lib", "resample"),
    ("workspace", "antecedent", "lib", "bayesian_transport_providers_match_independent_dirichlet_moments"),
    # gate_causal_artifacts.sh
    ("workspace", "antecedent-io", "*", "causal_artifact"),
    # gate_context.sh
    ("workspace", "antecedent-identify", "lib", "temporal_mediation::"),
    ("workspace", "antecedent-validate", "lib", "functional::"),
    # gate_estimate_ci.sh
    ("workspace", "antecedent-stats", "lib", "ci::calibration"),
    # gate_gcm.sh
    ("gaussian-process", "antecedent-model", "lib", "gaussian_process_matches_exact_logdet_oracle"),
    # gate_pag.sh
    ("workspace", "antecedent", "lib", "refuses_dag_only"),
    # gate_estimate_reuse.sh
    ("workspace", "antecedent-estimate", "lib", "matching_index_reused_across_compatible_point_fits"),
    ("workspace", "antecedent-estimate", "lib", "bootstrap_reuses_propensity_workspace_buffers"),
    ("workspace", "antecedent-stats", "lib", "matching::tests"),
    # gate_response_calibration.sh (also run by scripts/gate_calibration.sh)
    ("workspace", "antecedent-identify", "lib", "matches_frozen_bpbounds_table1_oracle"),
    ("workspace", "antecedent-identify", "causaleffect_transport_subset",
     "matches_frozen_causaleffect_supported_sid_subset"),
    ("workspace", "antecedent-stats", "lib", "observation_primitives_match_frozen_paper_equation_fixture"),
    ("workspace", "antecedent-estimate", "lib", "matches_frozen_trial_transport_equation_fixture"),
    ("workspace", "antecedent-estimate", "lib", "matches_frozen_exact_design_calibration_fixture"),
    ("workspace", "antecedent", "response_facade",
     "two_point_curve_contrast_conforms_to_average_effect_under_shared_linear_contract"),
    ("workspace", "antecedent-stats", "lib", "cox_ipcw"),
]

LISTINGS = {
    "workspace": ["--workspace"],
    "gaussian-process": ["-p", "antecedent-model", "--features", "gaussian-process"],
}


def problems(listings: dict[str, dict]) -> list[str]:
    found = []
    for listing, package, target, needle in NAMED:
        suites = listings[listing]["rust-suites"]
        if target == "*":
            ids = [i for i, s in suites.items() if s["package-name"] == package]
        else:
            ids = [package if target == "lib" else f"{package}::{target}"]
        ids = [i for i in ids if i in suites and (target == "*" or suites[i]["kind"] == ("lib" if target == "lib" else "test"))]
        where = f"{package} {'--lib' if target == 'lib' else '' if target == '*' else '--test ' + target}".rstrip()
        if not ids:
            found.append(f"{where}: no such test binary")
            continue
        names = [
            (name, case["ignored"])
            for i in ids
            for name, case in suites[i]["testcases"].items()
            if needle in name
        ]
        if not names:
            found.append(f"{where} {needle}: the filter selects no test")
        elif all(ignored for _, ignored in names):
            found.append(f"{where} {needle}: every selected test is #[ignore]d")
    return found


def nextest(args: list[str]) -> dict:
    command = ["cargo", "nextest", "list", *args, "--message-format", "json"]
    print("==", " ".join(command), flush=True)
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        sys.exit(f"`{' '.join(command)}` failed")
    return json.loads(proc.stdout)


if sys.argv[1:] == ["--self-test"]:
    def suite(package: str, kind: str, cases: dict[str, bool]) -> dict:
        return {"package-name": package, "kind": kind,
                "testcases": {n: {"ignored": ig} for n, ig in cases.items()}}

    def good() -> dict[str, dict]:
        suites: dict[str, dict] = {}
        for listing, package, target, needle in NAMED:
            sid = package if target in ("lib", "*") else f"{package}::{target}"
            kind = "lib" if target in ("lib", "*") else "test"
            suites.setdefault(sid, suite(package, kind, {}))["testcases"][f"m::{needle}_t"] = {"ignored": False}
        return {"workspace": {"rust-suites": suites},
                "gaussian-process": {"rust-suites": suites}}

    def case(name: str, listings: dict, want_fail: bool) -> None:
        got = problems(listings)
        if bool(got) != want_fail:
            sys.exit(f"named-test self-test {name}: expected {'failure' if want_fail else 'pass'}, got {got}")
        print(f"ok  {name}")

    case("positive control", good(), False)
    renamed = good()
    tc = renamed["workspace"]["rust-suites"]["antecedent-estimate"]["testcases"]
    del tc["m::envelope_t"]
    tc["m::envelop_t"] = {"ignored": False}
    case("renamed test fails", renamed, True)
    ignored = good()
    ignored["workspace"]["rust-suites"]["antecedent-stats"]["testcases"]["m::cox_ipcw_t"]["ignored"] = True
    case("ignored-only selection fails", ignored, True)
    missing = good()
    del missing["workspace"]["rust-suites"]["antecedent::response_facade"]
    case("missing --test target fails", missing, True)
    wrong_target = good()
    suites = wrong_target["workspace"]["rust-suites"]
    moved = suites["antecedent-identify::causaleffect_transport_subset"]
    suites["antecedent-identify"]["testcases"].update(moved["testcases"])
    del suites["antecedent-identify::causaleffect_transport_subset"]
    case("test present only in another target fails", wrong_target, True)
    print("named-test self-test: ok")
    sys.exit(0)

found = problems({key: nextest(args) for key, args in LISTINGS.items()})
if found:
    print("a filter the feature gates relied on no longer selects a test:", file=sys.stderr)
    for problem in found:
        print(f"  {problem}", file=sys.stderr)
    sys.exit(1)
print(f"named tests: ok ({len(NAMED)} filters)")
PY
