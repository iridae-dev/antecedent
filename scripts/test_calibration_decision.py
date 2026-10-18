"""Independent probability/count and provenance tests for the coverage FWER protocol."""

from __future__ import annotations

import math
import unittest
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_decision as decision


def oracle_lower(n: int, k: int, a: int, b: int) -> Decimal:
    if k == n:
        return Decimal(1)
    if k * b >= n * a:
        return Decimal(1) - oracle_lower(n, n - k - 1, b - a, b)
    with localcontext() as ctx:
        ctx.prec = 80
        # Independent exact integer binomial coefficient and rational probability.
        mass = (
            Decimal(math.comb(n, k))
            * Decimal(a) ** k
            * Decimal(b - a) ** (n - k)
            / Decimal(b) ** n
        )
        term = total = Decimal(1)
        for j in range(k, 0, -1):
            term *= Decimal(j * (b - a)) / Decimal((n - j + 1) * a)
            total += term
            if term < total * Decimal("1e-72"):
                break
        return +(mass * total)


class DecisionTest(unittest.TestCase):
    def config(self):
        return decision.config_for_manifest(
            {"groups": [{"index": i, "label": f"case{i}"} for i in range(1, 755)]}
        )

    def test_all_small_binomial_tails_match_exact_rational_enumeration(self):
        for n in range(1, 26):
            for a, b in ((1, 2), (19, 20), (1, 10)):
                masses = [
                    Fraction(math.comb(n, k) * a**k * (b - a) ** (n - k), b**n)
                    for k in range(n + 1)
                ]
                for k in range(n + 1):
                    lower = float(sum(masses[: k + 1]))
                    upper = float(sum(masses[k:]))
                    self.assertAlmostEqual(
                        decision.binomial_p_value(n, k, a / b, True), lower, delta=2e-13
                    )
                    self.assertAlmostEqual(
                        decision.binomial_p_value(n, k, a / b, False),
                        min(1, 2 * min(lower, upper)),
                        delta=2e-13,
                    )

    def test_large_counts_match_independent_eighty_digit_integer_oracle(self):
        for n, values in (
            (400, (300, 340, 360, 380, 399, 400)),
            (1953, (1750, 1829, 1855, 1900)),
            (2000, (1750, 1829, 1880, 1900, 1950)),
            (20000, (18000, 18800, 18820, 18926, 19000, 19150, 19200, 19900)),
        ):
            for k in values:
                lower = float(oracle_lower(n, k, 19, 20))
                upper = float(oracle_lower(n, n - k, 1, 20))
                for lower_only, expected in (
                    (True, lower),
                    (False, min(1, 2 * min(lower, upper))),
                ):
                    actual = decision.binomial_p_value(n, k, 0.95, lower_only)
                    self.assertTrue(
                        math.isclose(actual, expected, rel_tol=5e-9, abs_tol=2e-13),
                        (n, k, lower_only, actual, expected),
                    )

    def test_discreteness_and_spending_bound_complete_family_false_rejections(self):
        config = self.config()
        n = 400
        for lower_only in (False, True):
            alpha = decision.ALPHA * 0.5 / config["family_cells"]
            rejection_mass = sum(
                Fraction(math.comb(n, k) * 19**k, 20**n)
                for k in range(n + 1)
                if decision.binomial_p_value(n, k, 0.95, lower_only) <= alpha
            )
            self.assertLessEqual(float(rejection_mass), alpha)
            self.assertLessEqual(
                float(rejection_mass) * config["family_cells"], decision.ALPHA * 0.5
            )
        self.assertEqual(sum(look["weight"] for look in decision.LOOKS), 1)

    def test_frozen_757_group_boundaries_match_independent_decimal_oracle(self):
        # Reviewed 100-digit Decimal integer-mass oracle; fixed historical family
        # size keeps this a regression fixture when the live source roster grows.
        config = decision.config_for_manifest(
            {"groups": [{"index": i} for i in range(757)]}
        )
        cases = (
            (0.95, False, 400, 352, 399),
            (0.95, False, 2000, 1841, 1950),
            (0.95, False, 20000, 18823, 19168),
            (0.95, True, 400, 353, 400),
            (0.95, True, 2000, 1842, 2000),
            (0.95, True, 20000, 18827, 20000),
            (0.90, False, 400, 324, 388),
            (0.90, False, 2000, 1721, 1871),
            (0.90, False, 20000, 17758, 18233),
            (0.90, True, 400, 325, 400),
            (0.90, True, 2000, 1723, 2000),
            (0.90, True, 20000, 17764, 20000),
        )
        for nominal, lower_only, n, low, high in cases:
            with self.subTest(nominal=nominal, lower_only=lower_only, n=n):
                for k in (low, high):
                    self.assertTrue(
                        decision.evaluate(config, n, n, k, nominal, lower_only)[
                            "compatible"
                        ]
                    )
                self.assertFalse(
                    decision.evaluate(config, n, n, low - 1, nominal, lower_only)[
                        "compatible"
                    ]
                )
                if high < n:
                    self.assertFalse(
                        decision.evaluate(config, n, n, high + 1, nominal, lower_only)[
                            "compatible"
                        ]
                    )

    def test_exact_counts_use_planned_look_not_random_reported_denominator(self):
        value = decision.evaluate(self.config(), 2000, 1953, 1829, 0.95, False)
        self.assertTrue(value["compatible"])
        self.assertEqual(value["look_weight"], 0.25)
        with self.assertRaises(ValueError):
            decision.evaluate(self.config(), 1953, 1953, 1829, 0.95, False)
        with self.assertRaises(ValueError):
            decision.evaluate(self.config(), 400, 2000, 1900, 0.95, False)

    def test_near_threshold_noise_and_clear_miscalibration_remain_distinct(self):
        config = self.config()
        self.assertTrue(
            decision.evaluate(config, 20000, 20000, 18926, 0.95, False)["compatible"]
        )
        self.assertFalse(
            decision.evaluate(config, 20000, 20000, 17200, 0.95, False)["compatible"]
        )
        self.assertFalse(
            decision.evaluate(config, 20000, 20000, 19900, 0.95, False)["compatible"]
        )
        self.assertTrue(
            decision.evaluate(config, 20000, 20000, 19900, 0.95, True)["compatible"]
        )

    def test_unknown_family_fields_counts_looks_and_risks_fail_closed(self):
        config = self.config()
        for field, value in (
            ("validation_domain", "another_domain"),
            ("family_groups", 1),
            ("family_alpha", 0.1),
            ("cell_cap_per_group", 64),
            ("family_grid_points", 1),
            ("family_digest", "not-a-hash"),
            ("family_groups", True),
        ):
            with self.assertRaises(ValueError):
                decision.validate_config({**config, field: value})
        altered = deepcopy(config)
        altered["looks"][0]["weight"] = 0.75
        with self.assertRaises(ValueError):
            decision.validate_config(altered)
        with self.assertRaises(ValueError):
            decision.validate_config({**config, "unknown": True})
        for scored, covered in ((0, 0), (20, 21), (20, -1), (20.0, 19), (20, True)):
            with self.assertRaises(ValueError):
                decision.binomial_p_value(scored, covered, 0.95, False)

    def test_metadata_cannot_relabel_counts_look_probability_family_or_scientific_tail(
        self,
    ):
        config = self.config()
        value = decision.evaluate(config, 2000, 2000, 1890, 0.95, False)
        decision.validate_decision(
            value,
            0.95,
            observed=0.945,
            replicates=2000,
            expected_config=config,
            role="gated",
        )
        for field, wrong in (
            ("covered", 1891),
            ("look_weight", 0.5),
            ("p_value", 1),
            ("compatible", False),
            ("scored", 1999),
        ):
            with self.assertRaises(ValueError):
                decision.validate_decision(
                    {**value, field: wrong},
                    0.95,
                    observed=0.945,
                    replicates=2000,
                    expected_config=config,
                    role="gated",
                )
        with self.assertRaises(ValueError):
            decision.validate_decision(
                {**value, "family_digest": "b" * 64}, 0.95, expected_config=config
            )
        with self.assertRaises(ValueError):
            decision.validate_decision(value, 0.95, role="conservative_point")
        incompatible = decision.evaluate(config, 20000, 20000, 17000, 0.95, False)
        with self.assertRaises(ValueError):
            decision.validate_decision(incompatible, 0.95, role="gated")

    def test_probability_endpoints_and_boolean_metadata_are_not_ambiguous(self):
        self.assertEqual(decision.binomial_p_value(20, 20, 1, False), 1)
        self.assertEqual(decision.binomial_p_value(20, 19, 1, False), 0)
        self.assertEqual(decision.binomial_p_value(20, 0, 0, False), 1)
        with self.assertRaises(ValueError):
            decision.binomial_p_value(20, 19, 0.95, 1)
        with self.assertRaises(ValueError):
            decision.canonical_json({"p": float("nan")})


if __name__ == "__main__":
    unittest.main()
