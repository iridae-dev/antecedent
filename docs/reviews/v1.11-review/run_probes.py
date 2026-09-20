"""Run public-API counterexamples without adding tests to production crates.

This diagnostic prints actual and expected behavior; successful execution means
the probes ran, not that the candidate is correct. See the adjacent review.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CRATES = ("core", "prob", "stats", "graph", "identify", "state", "estimate")


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_facets as calibration

    surface = calibration.load_surface()
    records = calibration.load_records()
    print("Calibration dependency probe", flush=True)
    for rel in (
        "crates/antecedent-estimate/src/util.rs",
        "crates/antecedent-stats/src/faer_backend.rs",
        "crates/antecedent-prob/src/conjugate.rs",
        "crates/antecedent-kernels/src/rng.rs",
        "crates/antecedent/src/analysis/execute/dispatch.rs",
    ):
        facet = surface.facet_of(rel)
        affected = sum(facet in calibration.record_facets(r, surface) for r in records)
        print(f"{rel}: facet={facet}, invalidates {affected}/{len(records)} records", flush=True)

    with tempfile.TemporaryDirectory(prefix="antecedent-review-") as directory:
        work = Path(directory)
        (work / "src").mkdir()
        shutil.copyfile(HERE / "probes.rs", work / "src/main.rs")
        shutil.copyfile(ROOT / "Cargo.lock", work / "Cargo.lock")
        lines = [
            '[package]', 'name = "antecedent-review-probes"',
            'version = "0.0.0"', 'edition = "2024"', '[dependencies]',
        ]
        for crate in CRATES:
            # TOML literal strings preserve platform path spelling.
            path = ROOT / "crates" / f"antecedent-{crate}"
            lines.append(f"antecedent-{crate} = {{ path = '{path}' }}")
        for crate in ("antecedent-stats", "antecedent-kernels", "faer"):
            lines += [f"[profile.dev.package.{crate}]", "opt-level = 2"]
        (work / "Cargo.toml").write_text("\n".join(lines) + "\n")
        subprocess.run([
            "cargo", "run", "--offline", "--manifest-path", str(work / "Cargo.toml"),
            "--target-dir", str(ROOT / "target"),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
