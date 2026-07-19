#!/usr/bin/env python3
"""One-shot migration: normalize reference schema, rename paths, rewrite inventories.

Run from repo root after Phase-0 oracle generation. Not retained as a project tool.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOWHY_COMMIT = "178ecc9c690a02f2801c1f70da2695f5744186cc"
TIG_COMMIT = "5a8768754e6103755b006e9357e21c1a58534927"
TIG_EXTENDED = "ff3ff13e1481073b8c5833a6fde1c304627a208e"

ID_MAP = {
    # DoWhy → estimate / gcm
    "dowhy.model_graph.construction": "estimate.model.construction",
    "dowhy.model_graph.treatment_outcome": "estimate.model.treatment_outcome",
    "dowhy.model_graph.parsing": "estimate.model.parsing",
    "dowhy.model_graph.assumptions": "estimate.model.assumptions",
    "dowhy.model_graph.workflow": "estimate.model.workflow",
    "dowhy.identify.adjustment_sets": "estimate.identify.adjustment_sets",
    "dowhy.identify.automatic": "estimate.identify.automatic",
    "dowhy.identify.backdoor": "estimate.identify.backdoor",
    "dowhy.identify.efficient_backdoor": "estimate.identify.efficient_backdoor",
    "dowhy.identify.general_id": "estimate.identify.general_id",
    "dowhy.identify.estimand": "estimate.identify.estimand",
    "dowhy.estimate.linear_regression": "estimate.linear_regression",
    "dowhy.estimate.glm": "estimate.glm",
    "dowhy.estimate.propensity": "estimate.propensity",
    "dowhy.estimate.matching": "estimate.matching",
    "dowhy.estimate.doubly_robust": "estimate.doubly_robust",
    "dowhy.estimate.iv": "estimate.iv",
    "dowhy.estimate.rd": "estimate.rd",
    "dowhy.estimate.two_stage": "estimate.two_stage",
    "dowhy.estimate.conditional": "estimate.conditional",
    "dowhy.refute.placebo": "estimate.refute.placebo",
    "dowhy.refute.random_common_cause": "estimate.refute.random_common_cause",
    "dowhy.refute.bootstrap": "estimate.refute.bootstrap",
    "dowhy.refute.unobserved_common_cause": "estimate.refute.unobserved_common_cause",
    "dowhy.refute.overlap": "estimate.refute.overlap",
    "dowhy.refute.data_subset": "estimate.refute.data_subset",
    "dowhy.refute.dummy_outcome": "estimate.refute.dummy_outcome",
    "dowhy.refute.evalue": "estimate.refute.evalue",
    "dowhy.refute.graph": "estimate.refute.graph",
    "dowhy.refute.sensitivity": "estimate.refute.sensitivity",
    "dowhy.do_sampling": "gcm.do_sampling",
    "dowhy.gcm": "gcm.surface",
    "dowhy.secondary": "estimate.secondary_surfaces",
    # Tigramite → discovery
    "tigramite.data.dataframe": "discovery.data.dataframe",
    "tigramite.data.masks": "discovery.data.masks",
    "tigramite.data.multiple_datasets": "discovery.data.multiple_datasets",
    "tigramite.data.offsets": "discovery.data.offsets",
    "tigramite.data.vector_variables": "discovery.data.vector_variables",
    "tigramite.data.transforms": "discovery.data.transforms",
    "tigramite.data.bootstrap": "discovery.data.bootstrap",
    "tigramite.ci.partial_corr": "discovery.ci.partial_corr",
    "tigramite.ci.multivariate_partial_corr": "discovery.ci.multivariate_partial_corr",
    "tigramite.ci.weighted_partial_corr": "discovery.ci.weighted_partial_corr",
    "tigramite.ci.robust_partial_corr": "discovery.ci.robust_partial_corr",
    "tigramite.ci.regression": "discovery.ci.regression",
    "tigramite.ci.cmi_knn": "discovery.ci.knn_dependence",
    "tigramite.ci.mixed_cmi_knn": "discovery.ci.mixed_knn_dependence",
    "tigramite.ci.symbolic_cmi": "discovery.ci.symbolic_cmi",
    "tigramite.ci.gpdc": "discovery.ci.gpdc",
    "tigramite.ci.gsquared": "discovery.ci.gsquared",
    "tigramite.ci.oracle": "discovery.ci.oracle",
    "tigramite.discovery.pcmci": "discovery.pcmci",
    "tigramite.discovery.pcmci_plus": "discovery.pcmci_plus",
    "tigramite.discovery.lpcmci": "discovery.lpcmci",
    "tigramite.discovery.jpcmci_plus": "discovery.jpcmci_plus",
    "tigramite.discovery.rpcmci": "discovery.rpcmci",
    "tigramite.discovery.fdr": "discovery.fdr",
    "tigramite.graphs.ts_graph": "discovery.graphs.ts_graph",
    "tigramite.graphs.separation": "discovery.graphs.separation",
    "tigramite.graphs.endpoints": "discovery.graphs.endpoints",
    "tigramite.effects": "discovery.effects",
    "tigramite.simulation": "discovery.simulation",
}


def to_reference(obj: dict, project: str) -> dict:
    """Lift branded block into unified reference schema."""
    block = obj.pop(project, None)
    if block is None:
        return obj
    ref = {
        "project": project,
        "available": bool(block.get("available", False)),
        "version": block.get("pinned_version")
        or block.get(f"{project}_version")
        or block.get("version"),
        "commit": block.get("pinned_commit"),
        "command": block.get("command")
        or (obj.get("generation") or {}).get("env")
        or (obj.get("generation") or {}).get("script"),
        "note": block.get("note") or block.get("product_contract"),
        "outputs": {
            k: v
            for k, v in block.items()
            if k
            not in {
                "available",
                "note",
                "product_contract",
                "pinned_version",
                "pinned_commit",
                "command",
                "dowhy_version",
                "tigramite_version",
                "version",
            }
        },
    }
    if project == "dowhy" and not ref.get("commit"):
        ref["commit"] = DOWHY_COMMIT
    if project == "tigramite" and not ref.get("commit"):
        ref["commit"] = TIG_COMMIT
    # Drop empty outputs noise
    if not ref["outputs"]:
        ref.pop("outputs")
    obj["reference"] = ref
    gen = obj.get("generation") or {}
    if "baseline_pin" in gen:
        if "dowhy" in str(gen["baseline_pin"]):
            gen["baseline_pin"] = "parity/baselines/dowhy.toml"
        if "tigramite" in str(gen["baseline_pin"]):
            gen["baseline_pin"] = "parity/baselines/tigramite.toml"
        obj["generation"] = gen
    return obj


def normalize_json_files() -> None:
    for path in (ROOT / "conformance").rglob("expected.json"):
        data = json.loads(path.read_text())
        if "dowhy" in data:
            data = to_reference(data, "dowhy")
        if "tigramite" in data:
            data = to_reference(data, "tigramite")
        # Fix hedge status heuristic
        if path.parent.name == "general_id_hedge":
            est = ((data.get("reference") or {}).get("outputs") or {}).get("estimand", "")
            if "No such variable" in est:
                data["expected_status_family"] = "unidentified"
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"normalized {path.relative_to(ROOT)}")


def move_trees() -> None:
    moves = [
        ("conformance/dowhy/linear_gaussian_ate", "conformance/estimate/linear_gaussian_ate"),
        ("conformance/dowhy/noisy_estimators", "conformance/estimate/noisy_estimators"),
    ]
    tig = ROOT / "conformance" / "tigramite"
    if tig.exists():
        for child in sorted(tig.iterdir()):
            if child.is_dir():
                moves.append(
                    (
                        f"conformance/tigramite/{child.name}",
                        f"conformance/discovery/{child.name}",
                    )
                )
    for src_s, dst_s in moves:
        src, dst = ROOT / src_s, ROOT / dst_s
        if not src.exists():
            print(f"skip missing {src_s}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"dst exists, merging content {dst_s}")
            for item in src.iterdir():
                target = dst / item.name
                if target.exists():
                    if item.is_file():
                        shutil.copy2(item, target)
                else:
                    shutil.move(str(item), str(target))
            shutil.rmtree(src)
        else:
            shutil.move(str(src), str(dst))
        print(f"moved {src_s} -> {dst_s}")
    # Remove empty branded dirs
    for d in [ROOT / "conformance" / "dowhy", ROOT / "conformance" / "tigramite"]:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()


def write_baselines() -> None:
    base = ROOT / "parity" / "baselines"
    base.mkdir(parents=True, exist_ok=True)
    (base / "dowhy.toml").write_text(
        f"""# Pinned external baseline (oracle pin only). Capability inventory lives in domain TOMLs.
project = "dowhy"
version = "0.14"
commit = "{DOWHY_COMMIT}"
notes = "Pinned per ADR 0009. Used only to interpret conformance/**/expected.json reference blocks."
"""
    )
    (base / "tigramite.toml").write_text(
        f"""# Pinned external baseline (oracle pin only). Capability inventory lives in domain TOMLs.
project = "tigramite"
version = "5.2.1.25"
commit = "{TIG_COMMIT}"
extended_snapshot_commit = "{TIG_EXTENDED}"
notes = "Pinned per ADR 0009. Extended snapshot covers post-release features. Used only for reference blocks."
"""
    )


def rewrite_capability_block(block: str) -> str:
    def repl_id(m: re.Match[str]) -> str:
        old = m.group(1)
        new = ID_MAP.get(old, old)
        return f'id = "{new}"'

    return re.sub(r'id = "([^"]+)"', repl_id, block)


def split_inventories() -> None:
    """Create parity/estimate.toml and parity/discovery.toml from old branded files."""
    dowhy = (ROOT / "parity" / "dowhy.toml").read_text()
    tig = (ROOT / "parity" / "tigramite.toml").read_text()

    def caps_only(text: str, header: str) -> str:
        # Drop [baseline] section; keep capabilities + excluded/optional
        parts = re.split(r"\n(?=\[\[capabilities\]\]|\[[a-z_]+\])", text)
        body = []
        for p in parts:
            if p.startswith("[[capabilities]]") or p.startswith("[excluded]") or p.startswith("[optional"):
                body.append(rewrite_capability_block(p).rstrip() + "\n")
        # Also rewrite path notes
        joined = "\n".join(body)
        joined = joined.replace("conformance/dowhy/", "conformance/estimate/")
        joined = joined.replace("conformance/tigramite/", "conformance/discovery/")
        joined = joined.replace("parity/dowhy.toml", "parity/baselines/dowhy.toml")
        joined = joined.replace("parity/tigramite.toml", "parity/baselines/tigramite.toml")
        # Scrub project names from notes where possible
        joined = re.sub(r"\bDoWhy\b", "pinned baseline", joined)
        joined = re.sub(r"\bTigramite\b", "pinned baseline", joined)
        joined = re.sub(r"\bdowhy\b", "baseline", joined)
        joined = re.sub(r"\btigramite\b", "baseline", joined)
        return header + joined

    estimate_header = """# Estimate / identify / refute capability inventory.
# Status values: pending | in_progress | done
# Baseline pin: parity/baselines/dowhy.toml (oracle reference only)

"""
    discovery_header = """# Discovery / CI / temporal-graph capability inventory.
# Status values: pending | in_progress | done
# Baseline pin: parity/baselines/tigramite.toml (oracle reference only)

"""
    (ROOT / "parity" / "estimate.toml").write_text(caps_only(dowhy, estimate_header))
    (ROOT / "parity" / "discovery.toml").write_text(caps_only(tig, discovery_header))

    # Remove branded inventories
    (ROOT / "parity" / "dowhy.toml").unlink(missing_ok=True)
    (ROOT / "parity" / "tigramite.toml").unlink(missing_ok=True)


def rewrite_fixtures() -> None:
    fix = ROOT / "parity" / "fixtures"
    for path in fix.glob("*.toml"):
        text = path.read_text()
        text = text.replace("conformance/dowhy/", "conformance/estimate/")
        text = text.replace("conformance/tigramite/", "conformance/discovery/")
        text = text.replace("parity/dowhy.toml", "parity/baselines/dowhy.toml")
        text = text.replace("parity/tigramite.toml", "parity/baselines/tigramite.toml")
        for old, new in ID_MAP.items():
            text = text.replace(old, new)
        # Rename files
        new_name = path.name
        new_name = new_name.replace("dowhy_", "estimate_")
        new_name = new_name.replace("tigramite_", "discovery_")
        out = fix / new_name
        out.write_text(text)
        if out != path:
            path.unlink()
        print(f"fixture {out.name}")


def rewrite_text_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text()
    orig = text
    text = text.replace("parity/dowhy.toml", "parity/estimate.toml")
    text = text.replace("parity/tigramite.toml", "parity/discovery.toml")
    text = text.replace("conformance/dowhy/", "conformance/estimate/")
    text = text.replace("conformance/tigramite/", "conformance/discovery/")
    for old, new in sorted(ID_MAP.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(old, new)
    # Test binary names in gates
    text = text.replace("dowhy_linear_gaussian_ate", "estimate_linear_gaussian_ate")
    text = text.replace("dowhy_noisy_estimators", "estimate_noisy_estimators")
    text = text.replace("tigramite_pcmci_lag1", "discovery_pcmci_lag1")
    text = text.replace("tigramite_pcmci_plus_lag0", "discovery_pcmci_plus_lag0")
    text = text.replace("tigramite_pcmci_multivar", "discovery_pcmci_multivar")
    text = text.replace("tigramite_masked_mci_lag1", "discovery_masked_mci_lag1")
    text = text.replace("tigramite_vector_vars_pcmci", "discovery_vector_vars_pcmci")
    text = text.replace("tigramite_jpcmci_plus_two_env_edges", "discovery_jpcmci_plus_two_env_edges")
    text = text.replace("tigramite_ci_stats", "discovery_ci_stats")
    if text != orig:
        path.write_text(text)
        print(f"rewrote {path.relative_to(ROOT)}")


def rename_tests() -> None:
    renames = [
        (
            "crates/causal/tests/dowhy_linear_gaussian_ate.rs",
            "crates/causal/tests/estimate_linear_gaussian_ate.rs",
        ),
        (
            "crates/causal/tests/dowhy_noisy_estimators.rs",
            "crates/causal/tests/estimate_noisy_estimators.rs",
        ),
        (
            "crates/causal-discovery/tests/tigramite_pcmci_lag1.rs",
            "crates/causal-discovery/tests/discovery_pcmci_lag1.rs",
        ),
        (
            "crates/causal-discovery/tests/tigramite_pcmci_plus_lag0.rs",
            "crates/causal-discovery/tests/discovery_pcmci_plus_lag0.rs",
        ),
        (
            "crates/causal-discovery/tests/tigramite_pcmci_multivar.rs",
            "crates/causal-discovery/tests/discovery_pcmci_multivar.rs",
        ),
        (
            "crates/causal-discovery/tests/tigramite_masked_mci_lag1.rs",
            "crates/causal-discovery/tests/discovery_masked_mci_lag1.rs",
        ),
        (
            "crates/causal-discovery/tests/tigramite_vector_vars_pcmci.rs",
            "crates/causal-discovery/tests/discovery_vector_vars_pcmci.rs",
        ),
        (
            "crates/causal-discovery/tests/tigramite_jpcmci_plus_two_env_edges.rs",
            "crates/causal-discovery/tests/discovery_jpcmci_plus_two_env_edges.rs",
        ),
        (
            "crates/causal-stats/tests/tigramite_ci_stats.rs",
            "crates/causal-stats/tests/discovery_ci_stats.rs",
        ),
    ]
    for src_s, dst_s in renames:
        src, dst = ROOT / src_s, ROOT / dst_s
        if not src.exists():
            continue
        text = src.read_text()
        text = text.replace("conformance/dowhy/", "conformance/estimate/")
        text = text.replace("conformance/tigramite/", "conformance/discovery/")
        # Update JSON key access from branded to reference
        text = text.replace('["dowhy"]', '["reference"]')
        text = text.replace('["tigramite"]', '["reference"]')
        text = text.replace(".dowhy", ".reference")  # unlikely
        # Soft scrub module docs
        text = re.sub(r"`?DoWhy`?", "estimate-parity", text)
        text = re.sub(r"`?Tigramite`?", "discovery-parity", text)
        text = re.sub(r"\bdowhy\b", "estimate", text, flags=re.I)
        text = re.sub(r"\btigramite\b", "discovery", text, flags=re.I)
        dst.write_text(text)
        if dst != src:
            src.unlink()
        print(f"test {dst.relative_to(ROOT)}")


def main() -> None:
    normalize_json_files()
    move_trees()
    write_baselines()
    split_inventories()
    rewrite_fixtures()
    rename_tests()
    for rel in [
        "scripts/gate_estimate_ci.sh",
        "scripts/gate_release.sh",
        "scripts/gate_context.sh",
        "scripts/gate_gcm.sh",
        "scripts/gate_pag.sh",
        "parity/README.md",
        "parity/release.toml",
        "parity/context.toml",
        "parity/attribution.toml",
        "parity/gcm.toml",
        "parity/pag.toml",
        "crates/causal/tests/context_effects.rs",
        "crates/causal/tests/estimate_conformance.rs",
    ]:
        rewrite_text_file(ROOT / rel)
    # Mark RD unavailable reference if missing
    rd = ROOT / "conformance" / "estimate" / "rd_sharp" / "expected.json"
    if rd.exists():
        data = json.loads(rd.read_text())
        if "reference" not in data:
            data["reference"] = {
                "project": "dowhy",
                "available": False,
                "version": "0.14",
                "commit": DOWHY_COMMIT,
                "command": "uv run --python 3.12 --with dowhy==0.14 … generate_dowhy_estimate_oracles.py",
                "note": "iv.regression_discontinuity returned None on pin; analytic true_effect retained",
            }
            rd.write_text(json.dumps(data, indent=2) + "\n")
    print("migration done")


if __name__ == "__main__":
    main()
