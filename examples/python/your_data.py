"""How do you take your own CSV table to an average-effect estimate?

The script writes a temporary 240-row synthetic CSV (randomized binary treatment, effect 2,
covariate z affecting the outcome); replace its path with your own reviewed table. It stops
on missing values in the required columns rather than imputing or dropping rows, and asserts
the estimate lies within 0.3 of 2. On your own table the effect holds only under the graph you
declare; the script does not test it. Requires pandas in addition to Antecedent.

Run: ``python examples/python/your_data.py``.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import antecedent as ant
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(24)
    z = rng.normal(size=240)
    t = rng.binomial(1, 0.5, len(z)).astype(float)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "observations.csv"
        pd.DataFrame(
            {
                "treatment": t,
                "outcome": 2 * t + z + rng.normal(scale=0.2, size=len(z)),
                "z": z,
            }
        ).to_csv(path, index=False)
        frame = pd.read_csv(path)
        required = ["treatment", "outcome", "z"]
        if frame[required].isna().any().any():
            raise ValueError("Declare a missing-observation policy before analysis")
        data = {name: frame[name].to_numpy(dtype=float) for name in required}
        study = ant.prepare(
            data,
            graph=[("treatment", "outcome"), ("z", "outcome")],
            query=ant.AverageEffect("treatment", "outcome"),
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
        )
        print(study.inspect())
        result = ant.estimate(study)
        print(result.inspect())
        print(
            "Effect:",
            result.answer.value,
            "calibration:",
            result.value.calibration.status,
        )
        assert result.answer.kind == "point"
        assert abs(result.answer.value - 2) < 0.3


if __name__ == "__main__":
    main()
