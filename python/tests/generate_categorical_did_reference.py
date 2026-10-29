"""Exact rational oracle for the multi-level difference-in-differences contrast.

Run to regenerate ``fixtures/categorical_did_reference.json``. Every figure is a ``Fraction``
formed from the definitions alone (no library kernel):

- 2x2 repeated cross-section: the four cell means of a level and of the reference give
  ``(post_L - pre_L) - (post_R - pre_R)``.
- Balanced two-period panel: the change score of each subject is ``post - pre``; the contrast is
  the difference of the two groups' mean change; the cluster-robust variance (subject clusters) is
  ``G / (G - 1) * sum_g sum_i (d_i - mean_g)^2 / n_g^2`` for ``G`` subjects in the two groups.
- Augmented panel: the level's mean untreated-change residual ``d - m`` minus the odds-weighted
  (``p / (1 - p)``, normalised to sum to one) mean of the reference's residuals.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def mean(values):
    return sum(values, Fraction(0)) / len(values)


def frac(value):
    return Fraction(value)


def did_2x2(rows, level, reference):
    cell = {}
    for y, group, post in rows:
        cell.setdefault((group, post), []).append(frac(y))
    def change(g):
        return mean(cell[(g, 1)]) - mean(cell[(g, 0)])
    return change(level) - change(reference)


def panel(subjects, level, reference):
    """subjects: list of (pre, post, group). Returns (estimate, variance)."""
    delta = {level: [], reference: []}
    for pre, post, group in subjects:
        if group in delta:
            delta[group].append(frac(post) - frac(pre))
    estimate = mean(delta[level]) - mean(delta[reference])
    big_g = len(delta[level]) + len(delta[reference])
    meat = Fraction(0)
    for values in delta.values():
        m = mean(values)
        meat += sum(((d - m) ** 2 for d in values), Fraction(0)) / (len(values) ** 2)
    variance = Fraction(big_g, big_g - 1) * meat
    return estimate, variance


def augmented(subjects, level, reference, propensity, prediction):
    treated, control = [], []
    for index, (pre, post, group) in enumerate(subjects):
        if group == level:
            treated.append(frac(post) - frac(pre) - frac(prediction[index]))
        elif group == reference:
            p = frac(propensity[index])
            control.append((p / (1 - p), frac(post) - frac(pre) - frac(prediction[index])))
    total = sum((w for w, _ in control), Fraction(0))
    return mean(treated) - sum((w * r for w, r in control), Fraction(0)) / total


def text(value):
    return f"{value.numerator}/{value.denominator}"


def cases():
    out = []
    # Unbalanced groups of 5, 3 and 2 subjects (the smallest is a tiny sample).
    subjects = (
        [(10 + i, 12 + 2 * i + (i % 2), 0) for i in range(5)]
        + [(20 + i, 25 + 3 * i, 1) for i in range(3)]
        + [(5, 5 + 7, 2), (6, 6 + 9, 2)]
    )
    out.append(("unbalanced_three_levels", subjects, 0))
    # Ties: every change within a level is identical, so a level has zero variance.
    subjects = [(1, 4, 0)] * 3 + [(2, 7, 1)] * 4 + [(0, 1, 2)] * 2 + [(3, 3, 2)]
    out.append(("tied_changes", subjects, 1))
    # A large outcome offset and a non-smallest reference level.
    offset = 10**8
    subjects = (
        [(offset + 3 * i, offset + 3 * i + 2 + i, "a") for i in range(4)]
        + [(offset + i, offset + i + 9 - i, "b") for i in range(4)]
        + [(offset + 2 * i, offset + 2 * i + 5, "c") for i in range(3)]
        + [(offset + i * i, offset + i * i + 1, "d") for i in range(3)]
    )
    out.append(("offset_four_string_levels", subjects, "c"))
    return out


def main():
    result = []
    for name, subjects, reference in cases():
        levels = sorted({g for _, _, g in subjects}, key=lambda g: (str(type(g)), g))
        propensity = [0.25 + 0.5 * ((i * 7) % 11) / 11 for i in range(len(subjects))]
        prediction = [((i * 5) % 9) / 3 for i in range(len(subjects))]
        rows2x2 = []
        for pre, post, group in subjects:
            rows2x2.append((pre, group, 0))
            rows2x2.append((post, group, 1))
        contrasts = {}
        for level in levels:
            if level == reference:
                continue
            estimate, variance = panel(subjects, level, reference)
            contrasts[str(level)] = {
                "level": level,
                "did_2x2": text(did_2x2(rows2x2, level, reference)),
                "panel": text(estimate),
                "panel_variance": text(variance),
                "augmented": text(augmented(subjects, level, reference, propensity, prediction)),
                "level_subjects": sum(1 for _, _, g in subjects if g == level),
                "reference_subjects": sum(1 for _, _, g in subjects if g == reference),
            }
        result.append(
            {
                "name": name,
                "reference": reference,
                "subjects": [list(row) for row in subjects],
                "propensity": propensity,
                "prediction": prediction,
                "contrasts": contrasts,
            }
        )
    path = Path(__file__).parent / "fixtures" / "categorical_did_reference.json"
    path.write_text(json.dumps({"cases": result}, indent=1) + "\n")


if __name__ == "__main__":
    main()
