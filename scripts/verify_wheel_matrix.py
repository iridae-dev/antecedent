"""Require one release wheel for every CI Python and platform combination."""

import re
import sys
from pathlib import Path


def problems(directory: Path, version: str) -> list[str]:
    expected = {(f"cp3{minor}", platform) for minor in (11, 12, 13, 14)
                for platform in ("linux-x86_64", "linux-aarch64", "macos-arm64", "windows-x86_64")}
    found: set[tuple[str, str]] = set()
    errors: list[str] = []
    wheels = sorted(directory.glob("*.whl"))
    for wheel in wheels:
        parts = wheel.name.removesuffix(".whl").split("-")
        if len(parts) != 5 or not wheel.name.endswith(".whl") or parts[0] != "antecedent":
            errors.append(f"unexpected wheel filename: {wheel.name}")
            continue
        _, built_version, python, abi, tag = parts
        if re.sub(r"[-._]", "", built_version) != re.sub(r"[-._]", "", version):
            errors.append(f"wrong version in {wheel.name}")
        if python not in {f"cp3{minor}" for minor in (11, 12, 13, 14)} or abi != python:
            errors.append(f"unexpected Python/ABI tag in {wheel.name}")
            continue
        if tag.startswith("manylinux") and tag.endswith("x86_64"):
            platform = "linux-x86_64"
        elif tag.startswith("manylinux") and tag.endswith("aarch64"):
            platform = "linux-aarch64"
        elif tag.startswith("macosx") and tag.endswith("arm64"):
            platform = "macos-arm64"
        elif tag == "win_amd64":
            platform = "windows-x86_64"
        else:
            errors.append(f"unexpected platform tag in {wheel.name}")
            continue
        key = (python, platform)
        if key in found:
            errors.append(f"duplicate wheel for {key}: {wheel.name}")
        found.add(key)
    if len(wheels) != 16:
        errors.append(f"expected 16 wheels; found {len(wheels)}")
    if found != expected:
        errors.append(f"missing wheel combinations: {sorted(expected - found)}")
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_wheel_matrix.py DIR VERSION")
    issues = problems(Path(sys.argv[1]), sys.argv[2])
    if issues:
        raise SystemExit("\n".join(f"FAIL: {issue}" for issue in issues))
    print("verified: all 16 Python/platform wheel combinations")
