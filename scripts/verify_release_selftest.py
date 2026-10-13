"""Exercise release wheel acceptance and its fail-closed cases without publishing."""
import os
import subprocess
import tempfile
from pathlib import Path

from _repo import REPO_ROOT as root

verifier = root / "scripts/verify_release.sh"
# Synthetic versions: the check compares wheel names to the version it is given.
VERSION = "9.8.7"
OTHER_VERSION = "9.8.8"
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

with tempfile.TemporaryDirectory() as temp:
    work = Path(temp)
    wheels = work / "wheels"
    wheels.mkdir()
    bin_dir = work / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\nprintf '%s\\n' \"$MOCK_RUN_SHA\"\n", encoding="utf-8", newline="\n")
    gh.chmod(0o755)
    tags = ("manylinux_2_17_x86_64", "manylinux_2_17_aarch64", "macosx_11_0_arm64", "win_amd64")
    for minor in (11, 12, 13, 14):
        for tag in tags:
            (wheels / f"antecedent-{VERSION}-cp3{minor}-cp3{minor}-{tag}.whl").touch()

    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "GITHUB_RUN_ID": "tag-run", "MOCK_RUN_SHA": sha}

    def check(version: str = VERSION, run_sha: str = sha) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(verifier), "wheels", str(wheels), version, "main-run"],
                              cwd=root, env={**env, "MOCK_RUN_SHA": run_sha},
                              text=True, capture_output=True)

    good = check()
    assert good.returncode == 0, good.stdout + good.stderr
    assert "16 wheel(s)" in good.stdout

    wrong_sha = check(run_sha="0" * 40)
    assert wrong_sha.returncode and "did not build" in wrong_sha.stderr

    wrong_version = check(version=OTHER_VERSION)
    assert wrong_version.returncode and "does not carry version" in wrong_version.stderr

    missing = next(wheels.glob("*win_amd64.whl"))
    missing.unlink()
    incomplete = check()
    assert incomplete.returncode and "expected 16 wheels" in incomplete.stderr

print("release wheel verification self-test: ok")
