"""Consume scientific bytes against a separately retained identity.

Use --input for a producer's directory. With no arguments, create a temporary
producer fixture and demonstrate valid consumption and two invalid requests.
"""

import argparse
import json
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory

import antecedent as ant


def consume(directory):
    directory = Path(directory)
    data = (directory / "analysis.bin").read_bytes()
    expected = json.loads((directory / "expected.json").read_text(encoding="utf-8"))
    loaded = ant.load(data, expected=expected)
    assert loaded.acceptance.identity_bound
    print("Consumer effect:", loaded.answer.value)
    print("Acceptance:", loaded.acceptance)
    # Obtain a different valid expected identity, rather than guessing a field.
    import numpy as np

    t = np.array([0.0, 1.0] * 80)
    different = ant.analyze(
        {"t": t, "y": 3 * t},
        graph=[("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    ).artifact()
    wrong_expected = different.expectation
    for label, content, identity in [
        ("wrong identity", data, wrong_expected),
        ("altered bytes", b"not-an-artifact", expected),
    ]:
        try:
            ant.load(content, expected=identity)
        except ant.CausalError as error:
            print("Refused", label, type(error).__name__)
        else:
            raise AssertionError(f"Accepted {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.input:
        consume(args.input)
    else:
        producer = runpy.run_path(str(Path(__file__).with_name("artifact_producer.py")))
        with TemporaryDirectory() as directory:
            producer["produce"](directory)
            consume(directory)


if __name__ == "__main__":
    main()
