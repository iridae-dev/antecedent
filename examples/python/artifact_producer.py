"""Write scientific bytes and a separately retainable expected identity.

With no arguments, runs in a temporary directory. Use --output to retain files.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import antecedent as ant
import numpy as np


def produce(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(24)
    t = rng.binomial(1, 0.5, 160).astype(float)
    result = ant.analyze(
        {"t": t, "y": 2 * t + rng.normal(scale=0.2, size=len(t))},
        graph=[("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    envelope = result.export()
    assert envelope.expectation is not None
    (directory / "analysis.bin").write_bytes(envelope.data)
    (directory / "expected.json").write_text(
        json.dumps(envelope.expectation.to_dict()["identity"]), encoding="utf-8"
    )
    assert ant.load(envelope.data, expected=envelope.expectation).acceptance.identity_bound
    print("Producer effect:", result.answer.value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        produce(args.output)
    else:
        with TemporaryDirectory() as directory:
            produce(directory)


if __name__ == "__main__":
    main()
