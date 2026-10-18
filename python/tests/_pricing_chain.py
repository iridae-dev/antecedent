"""Simulated populations of the policy-workflow acceptance cases.

``pricing_chain`` is price -> demand -> revenue with a lognormal price: ``log(price)`` is
``Normal(price_location, price_scale)``, ``demand = 50 + demand_slope * price + Normal(0, 1)`` and
``revenue = 1.5 * demand + Normal(0, revenue_noise)``. ``linear_chain`` is ``a -> b -> c`` with unit
coefficients ``b = 1.5 a + e_b`` and ``c = 2 b + 0.5 e_c``. Both are deterministic in ``seed``, so a
fresh process regenerates the same rows.
"""

from __future__ import annotations

import antecedent
import numpy as np

PRICING_EDGES = [("price", "demand"), ("demand", "revenue")]
LINEAR_EDGES = [("a", "b"), ("b", "c")]


def pricing_chain(
    n: int,
    seed: int,
    *,
    price_location: float = 3.0,
    price_scale: float = 0.4,
    demand_slope: float = -0.8,
    revenue_noise: float = 0.5,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    price = np.exp(rng.normal(price_location, price_scale, size=n))
    demand = 50.0 + demand_slope * price + rng.normal(size=n)
    revenue = 1.5 * demand + rng.normal(scale=revenue_noise, size=n)
    return {"price": price, "demand": demand, "revenue": revenue}


def linear_chain(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 1.5 * a + rng.normal(size=n)
    c = 2.0 * b + 0.5 * rng.normal(size=n)
    return {"a": a, "b": b, "c": c}


def fit(data: dict[str, np.ndarray], edges: list[tuple[str, str]], **options):
    names = list(data)
    return antecedent.model.fit_gcm(names, [data[name] for name in names], edges, **options)


def fit_pricing_chain(n: int, seed: int, **options):
    return fit(pricing_chain(n, seed), PRICING_EDGES, **options)


def fit_linear_chain(n: int, seed: int, **options):
    return fit(linear_chain(n, seed), LINEAR_EDGES, **options)
