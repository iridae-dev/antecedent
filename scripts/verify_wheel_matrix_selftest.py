"""Exercise fail-closed checks for reused release wheels without building them."""

import tempfile
from pathlib import Path

from verify_wheel_matrix import problems


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    for minor in (11, 12, 13, 14):
        for platform in ("manylinux_2_17_x86_64", "manylinux_2_17_aarch64",
                         "macosx_11_0_arm64", "win_amd64"):
            (root / f"antecedent-2.2.0-cp3{minor}-cp3{minor}-{platform}.whl").touch()
    assert not problems(root, "2.2.0")

    missing = root / "antecedent-2.2.0-cp311-cp311-win_amd64.whl"
    missing.unlink()
    assert any("missing wheel" in issue for issue in problems(root, "2.2.0"))
    missing.touch()

    stale = root / "antecedent-2.2.0-cp312-cp312-win_amd64.whl"
    stale.rename(root / "antecedent-2.1.0-cp312-cp312-win_amd64.whl")
    assert any("wrong version" in issue for issue in problems(root, "2.2.0"))

print("wheel matrix self-test: ok")
