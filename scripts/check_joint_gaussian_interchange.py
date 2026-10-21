#!/usr/bin/env python3
"""Consume an actual Rust Gaussian posterior fixture after its producer exits.

The Rust acceptance test emits independent expected identity and IEEE-754 bits.
This uses the existing Python artifact consumer, not a second estimator or a
claim that loading historical numerical evidence grants current authority.

Set ANTECEDENT_GAUSSIAN_INTERCHANGE_DIR=/private/tmp/antecedent-x-gaussian-interchange
when running joint_gaussian_posterior_keeps_whole_contrast_draws_aligned_and_changed_data_refits.
After that producer process exits, run this script with that directory as its
argument using the worktree's built Python extension. No fixture is skipped or
synthesized when producer files are missing.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def float_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def check(directory: Path) -> dict[str, object]:
    # Select the extension built in this worktree, never an unrelated installation.
    sys.path.insert(0, str(ROOT / "python"))
    import numpy as np
    from antecedent import _native
    from antecedent.errors import CausalSerializationError

    identity = json.loads((directory / "identity.json").read_text())
    expected = json.loads((directory / "expected.json").read_text())
    payload = (directory / "law.bin").read_bytes()
    artifact = _native.JointDistributionArtifact.load(payload, json.dumps(identity))
    shape = tuple(expected["shape"])
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise AssertionError("fixture must contain a genuine multi-world posterior law")
    assert artifact.shape == shape
    assert artifact.semantic == "causal_functional_posterior"
    metadata = json.loads(artifact.metadata_json)
    assert metadata["identity"] == identity
    assert metadata["calibration"] == "unmeasured"
    assert artifact.source_evidence is None, "historical loading cannot mint current authority"
    rows = artifact.draws_copy()
    assert rows.shape == shape and rows.dtype == np.float64
    assert rows.view(np.uint64).ravel().tolist() == expected["draw_bits"]
    for left in range(shape[1]):
        assert float_bits(artifact.mean(left)) == expected["mean_bits"][left]
        for right in range(shape[1]):
            assert (
                float_bits(artifact.covariance(left, right))
                == expected["covariance_bits"][left][right]
            )
    # The consumer must independently require the exact retained source identity.
    changed = dict(identity, snapshot_id=identity["snapshot_id"] + ":changed")
    try:
        _native.JointDistributionArtifact.load(payload, json.dumps(changed))
    except CausalSerializationError:
        pass
    else:
        raise AssertionError("changed consumer authority identity was accepted")
    return {
        "status": "passed",
        "shape": list(shape),
        "draw_bits": "exact",
        "means_and_covariance": "exact",
        "current_authority": False,
        "calibration": "unmeasured",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="files emitted by the actual Rust acceptance")
    args = parser.parse_args()
    print(json.dumps(check(args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
