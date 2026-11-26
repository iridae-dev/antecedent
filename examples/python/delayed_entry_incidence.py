"""Compare cause-1 incidence at day three in a randomized delayed-entry cohort.

Pass an output filename to save the result with its source data. Retain the printed
identity separately and use `--read FILE IDENTITY` to replay in a fresh process.
The Rust example accepts the same file and identity.
"""

import sys
from pathlib import Path

import antecedent as ant
import pandas as pd
from antecedent.artifacts import artifact_identity
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.survival import CompetingRisksOutcome


def cohort(thin: bool) -> pd.DataFrame:
    rows = []
    for arm in (0, 1):
        for i in range(160):
            slot = i % 8
            cause = 1 if slot < 2 + 2 * arm else 2 if slot == 7 else 0
            # All events occur on day two. Early censoring in the second cohort
            # leaves only twenty treated subjects under observation at day three.
            duration = 2.0 if cause else 2.5 if thin and arm and slot != 4 else 3.0
            rows.append((duration, cause, arm, i % 2))
    return pd.DataFrame(rows, columns=["duration", "cause", "arm", "entry"])


def read(raw: bytes, identity: str) -> None:
    replay = require_success(ant.load(raw, retained_identity=identity)).replay()
    assert replay.acceptance.replayed, replay.acceptance
    curve = replay.decoded_artifact.payload["survival"]
    assert curve["target_cause"] == 1 and curve["tau"] == 3.0
    assert abs(curve["treated"][-1] - curve["control"][-1] - 0.25) < 1e-12
    assert curve["graphless_support_status"] == "supported"
    low, high = curve["difference_at_tau_interval"]
    assert low < 0.25 < high
    print("Replayed day-three interval", (low, high))


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--read":
        read(Path(sys.argv[2]).read_bytes(), sys.argv[3])
        return
    if len(sys.argv) > 2:
        raise SystemExit("usage: delayed_entry_incidence.py [FILE | --read FILE IDENTITY]")
    for thin in (False, True):
        data = cohort(thin)
        query = CompetingRisksOutcome(
            "duration",
            "cause",
            "arm",
            target_cause=1,
            tau=3.0,
            randomized=True,
            delayed_entry="entry",
            observation_assumption=IndependentGiven(()),
        )
        result = require_success(
            ant.analyze(
                data,
                query=query,
                inference=ant.Frequentist(bootstrap=299),
                execution=ant.Execution(seed=297),
            )
        )
        curve = result.survival
        assert curve is not None and curve.incidence_difference_interval is not None
        # Four versus two cause-1 events per block of eight randomized subjects.
        assert abs(curve.incidence_difference - 0.25) < 1e-12
        assert curve.minimum_horizon_risk_set == (20 if thin else 60)
        assert curve.support_status == ("unsupported_pointwise_interval" if thin else "supported")
        print("At risk, support:", curve.minimum_horizon_risk_set, curve.support_status)
        if not thin:
            raw = bytes(result.export(source_data=data))
            identity = artifact_identity(raw)
            read(raw, identity)
            if len(sys.argv) == 2:
                Path(sys.argv[1]).write_bytes(raw)
                print("Retain this identity:", identity)


if __name__ == "__main__":
    main()
