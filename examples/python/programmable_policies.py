# SPDX-License-Identifier: MIT OR Apache-2.0
"""Price, replenish and replay a 12-week policy on the same synthetic store-weeks.

Fitted one-period price/demand mechanisms and declared stock-flow accounting form
one model. Closed-form policies execute in Antecedent; a separate NumPy recurrence
checks every counterfactual value against the generating equations.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import antecedent as ant
import numpy as np
from antecedent.errors import CausalError, CausalSerializationError
from antecedent.intervention_callbacks import (
    GENERATED,
    CallbackProgram,
    InterventionCallback,
    callback_records,
    recover_callback,
    required_callbacks,
    sample_do as sample_callbacks,
)
from antecedent.intervention_programs import (
    assign,
    clip,
    consume_policy_program,
    counterfactual_temporal_policy,
    decode_policy_program,
    encode_policy_program,
    generated,
    lag,
    maximum,
    minimum,
    policy_program_identity,
    policy_support,
    prepare_policy,
    sample_policy,
    sample_temporal_policy,
    transform,
    var,
    where,
)

NAMES = ["inventory", "price", "demand", "sales", "replenishment"]
EDGES = [
    ("inventory", "price"),
    ("price", "demand"),
    ("inventory", "sales"),
    ("demand", "sales"),
]
STORES, WEEKS = 24, 12
EXECUTION = ant.Execution(seed=20261030)
# Generator-only parameters; the fit is given observations and the graph, not these numbers.
TRUTH = {
    "price_intercept": 15.0,
    "inventory_slope": -0.02,
    "demand_intercept": 31.0,
    "price_slope": -1.1,
}


def residual_noise(rng, design, scale):
    """Orthogonal training errors make fitted-coefficient truth independently checkable."""
    noise = rng.normal(size=len(design))
    noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
    return noise * scale / np.std(noise)


def fit_model(seed):
    rng = np.random.default_rng(seed)
    inventory = rng.uniform(40, 120, 512)
    price = (
        TRUTH["price_intercept"]
        + TRUTH["inventory_slope"] * inventory
        + residual_noise(rng, np.column_stack([np.ones(512), inventory]), 0.8)
    )
    demand = (
        TRUTH["demand_intercept"]
        + TRUTH["price_slope"] * price
        + residual_noise(rng, np.column_stack([np.ones(512), price]), 1.5)
    )
    rows = {
        "inventory": inventory,
        "price": price,
        "demand": demand,
        "sales": np.minimum(inventory, np.maximum(demand, 0)),
        "replenishment": rng.uniform(8, 20, 512),
    }
    model = ant.model.fit_gcm(
        NAMES,
        [rows[n] for n in NAMES],
        EDGES,
        constraints={
            name: {"required": "linear_gaussian"}
            for name in ("price", "demand", "sales")
        },
    )
    return model


def programs():
    # Known bookkeeping, shared by every world. Orders placed last week arrive now.
    common = [
        assign(
            "inventory",
            maximum(0, lag("inventory", 1) - lag("sales", 1)) + lag("replenishment", 1),
        ),
        assign("sales", minimum(var("inventory"), maximum(0, var("demand")))),
    ]
    conditional = transform(
        "price",
        where(var("inventory") > 70, maximum(10, 0.85 * generated()), generated()),
    )
    adaptive = transform(
        "price",
        clip(
            where(
                var("inventory") > 70,
                0.85 * generated(),
                generated() + 0.12 * (lag("sales", 1) - 16),
            ),
            10,
            18,
        ),
    )
    return conditional, {
        "baseline": [*common, assign("replenishment", 12)],
        "fixed": [*common, assign("replenishment", 12), assign("price", 10)],
        "conditional": [*common, assign("replenishment", 12), conditional],
        "adaptive": [
            *common,
            assign(
                "replenishment",
                clip(lag("sales", 1) + 0.4 * (70 - var("inventory")), 0, 32),
            ),
            adaptive,
        ],
    }


def history():
    return [
        {
            "inventory": np.linspace(44, 104, STORES),
            "price": np.full(STORES, 14.0),
            "demand": np.full(STORES, 16.0),
            "sales": np.full(STORES, 16.0),
            "replenishment": np.full(STORES, 12.0),
        }
    ]


def oracle(strategy, price_noise, demand_noise):
    """Independent stock-flow recurrence, with no policy evaluator or fitted-model calls."""
    previous = history()[0]
    steps = []
    for t in range(WEEKS):
        stock = (
            np.maximum(0, previous["inventory"] - previous["sales"])
            + previous["replenishment"]
        )
        generated_price = (
            TRUTH["price_intercept"] + TRUTH["inventory_slope"] * stock + price_noise[t]
        )
        price = generated_price.copy()
        orders = np.full(STORES, 12.0)
        if strategy == "fixed":
            price = np.full(STORES, 10.0)
        elif strategy == "conditional":
            price = np.where(
                stock > 70, np.maximum(10, 0.85 * generated_price), generated_price
            )
        elif strategy == "adaptive":
            price = np.clip(
                np.where(
                    stock > 70,
                    0.85 * generated_price,
                    generated_price + 0.12 * (previous["sales"] - 16),
                ),
                10,
                18,
            )
            orders = np.clip(previous["sales"] + 0.4 * (70 - stock), 0, 32)
        demand = (
            TRUTH["demand_intercept"] + TRUTH["price_slope"] * price + demand_noise[t]
        )
        sales = np.minimum(stock, np.maximum(0, demand))
        previous = {
            "inventory": stock,
            "price": price,
            "demand": demand,
            "sales": sales,
            "replenishment": orders,
        }
        steps.append(previous)
    return steps


def columns(steps):
    return [
        {name: np.asarray(step.draws[i]) for i, name in enumerate(NAMES)}
        for step in steps
    ]


def contribution(steps):
    # Unit acquisition cost 6 is charged on units sold; 0.10 per unit-week held,
    # and 0.05 handling cost per unit ordered. This is contribution, not cash flow.
    return sum(
        (s["price"] - 6) * s["sales"]
        - 0.1 * (s["inventory"] - s["sales"])
        - 0.05 * s["replenishment"]
        for s in steps
    )


def expect_refusal(code, operation):
    try:
        operation()
    except CausalError as refusal:
        assert refusal.reason_code == code, str(refusal)
        return code
    raise AssertionError(f"Expected {code}")


def callback_example(model, conditional):
    source = (
        "def discount(g, inventory):\n"
        "    return np.where(inventory > 70.0, np.maximum(10.0, 0.85 * g), g)\n"
    )
    batches = []

    def discount(g, inventory):
        batches.append(g.shape)
        return np.where(inventory > 70, np.maximum(10, 0.85 * g), g)

    callback = InterventionCallback(
        discount,
        inputs=["g", "inventory"],
        name="discount",
    )
    applied = CallbackProgram(
        "price", callback, reads=[GENERATED, "inventory"], mode="transform"
    )
    expected = sample_policy(model, [conditional], 128, execution=EXECUTION).draws
    actual = sample_callbacks(model, [applied], 128, execution=EXECUTION).draws
    np.testing.assert_array_equal(actual, expected)
    assert batches == [(128,)]
    refusals = [
        expect_refusal(
            "policy_callback_not_portable", lambda: callback_records([applied])
        )
    ]
    invalid = InterventionCallback(
        lambda g, stock: g[:-1], inputs=["g", "inventory"], name="wrong_shape"
    )
    refusals.append(
        expect_refusal(
            "policy_callback_output_invalid",
            lambda: sample_callbacks(
                model,
                [
                    CallbackProgram(
                        "price",
                        invalid,
                        reads=[GENERATED, "inventory"],
                        mode="transform",
                    )
                ],
                128,
                execution=EXECUTION,
            ),
        )
    )
    refusals.append(
        expect_refusal(
            "policy_dependency_unavailable",
            lambda: sample_callbacks(
                model,
                [
                    CallbackProgram(
                        "price",
                        callback,
                        reads=[GENERATED, "unknown_stock"],
                        mode="transform",
                    )
                ],
                128,
                execution=EXECUTION,
            ),
        )
    )
    recoverable = InterventionCallback.from_source(
        source, entry_point="discount", inputs=["g", "inventory"], name="discount"
    )
    artifact = encode_policy_program(
        NAMES,
        [
            CallbackProgram(
                "price", recoverable, reads=[GENERATED, "inventory"], mode="transform"
            )
        ],
    )
    (record,) = required_callbacks(artifact)
    recovered = recover_callback(source, record)
    restored = decode_policy_program(NAMES, artifact, callbacks=[recovered])
    np.testing.assert_array_equal(
        sample_callbacks(model, restored, 128, execution=EXECUTION).draws, expected
    )
    refusals.append(
        expect_refusal(
            "policy_callback_unrecoverable",
            lambda: recover_callback(source.replace("0.85", "0.90"), record),
        )
    )
    return {
        "refusals": refusals,
        "runtime_batches": batches,
        "source_sha256": record.source_sha256,
        "dependency_pins": record.dependencies,
    }


def main():
    started = time.perf_counter()
    model = fit_model(23)
    conditional, policies = programs()
    # Conventional hard intervention and a compiled constant assignment agree.
    hard = model.sample_do({"price": 10.0}, 128, execution=EXECUTION)
    assigned = sample_policy(model, [assign("price", 10)], 128, execution=EXECUTION)
    np.testing.assert_array_equal(hard.draws, assigned.draws)
    raw = model.sample_do({}, 128, execution=EXECUTION).draws[NAMES.index("price")]
    floor = sample_policy(
        model,
        [transform("price", maximum(10, 0.85 * generated()))],
        128,
        execution=EXECUTION,
    ).draws[NAMES.index("price")]
    np.testing.assert_array_equal(floor, np.maximum(10, 0.85 * raw))
    support = policy_support(model, [conditional])
    assert support.simulation.licensed and support.counterfactual.licensed
    assert not support.identification.licensed
    assert support.identification.reason_code == "policy_route_not_licensed"
    circular = assign("price", var("sales"))
    cycle = expect_refusal(
        "policy_dependency_circular",
        lambda: sample_policy(model, [circular], 8, execution=EXECUTION),
    )
    # Reading inventory corrects the dependency: inventory precedes price.
    sample_policy(model, [conditional], 8, execution=EXECUTION)
    missing = expect_refusal(
        "policy_history_unavailable",
        lambda: sample_temporal_policy(
            model,
            policies["adaptive"],
            n_rows=STORES,
            n_steps=WEEKS,
            window=1,
            execution=EXECUTION,
        ),
    )

    rng = np.random.default_rng(91)
    price_noise = rng.normal(0, 0.8, (WEEKS, STORES))
    demand_noise = rng.normal(0, 1.5, (WEEKS, STORES))
    factual = oracle("baseline", price_noise, demand_noise)
    results = {}
    for strategy, program in policies.items():
        replay = columns(
            counterfactual_temporal_policy(
                model,
                program,
                factual,
                window=1,
                history=history(),
                execution=EXECUTION,
            )
        )
        expected = oracle(strategy, price_noise, demand_noise)
        for actual, truth in zip(replay, expected, strict=True):
            for name in NAMES:
                np.testing.assert_allclose(actual[name], truth[name], rtol=0, atol=1e-8)
        # Exactly the same units, disturbances and prehistory in every counterfactual world.
        values = contribution(replay)
        delta = values - contribution(factual)
        results[strategy] = {
            "mean_contribution": float(np.mean(values)),
            "mean_paired_change": float(np.mean(delta)),
            "unit_changes_first_three": delta[:3].tolist(),
            "mean_units_sold": float(np.mean(sum(s["sales"] for s in replay))),
            "mean_lost_demand": float(
                np.mean(sum(np.maximum(s["demand"] - s["sales"], 0) for s in replay))
            ),
        }
        assert np.isclose(
            np.mean(delta), np.mean(values) - np.mean(contribution(factual)), atol=1e-10
        )
    # Forward forecasts are new model draws, not counterfactuals of the factual stores.
    forecast = sample_temporal_policy(
        model,
        policies["adaptive"],
        n_rows=STORES,
        n_steps=WEEKS,
        window=1,
        history=history(),
        execution=EXECUTION,
    )
    repeated = sample_temporal_policy(
        model,
        policies["adaptive"],
        n_rows=STORES,
        n_steps=WEEKS,
        window=1,
        history=history(),
        execution=EXECUTION,
    )
    for first, second in zip(forecast, repeated, strict=True):
        np.testing.assert_array_equal(first.draws, second.draws)
    # Recover retained forward disturbances and independently check their whole recurrence.
    forward = columns(forecast)
    baseline_draw = columns(
        sample_temporal_policy(
            model,
            policies["baseline"],
            n_rows=STORES,
            n_steps=WEEKS,
            window=1,
            history=history(),
            execution=EXECUTION,
        )
    )
    u_price = np.array(
        [
            s["price"]
            - (TRUTH["price_intercept"] + TRUTH["inventory_slope"] * s["inventory"])
            for s in baseline_draw
        ]
    )
    u_demand = np.array(
        [
            s["demand"]
            - (TRUTH["demand_intercept"] + TRUTH["price_slope"] * s["price"])
            for s in baseline_draw
        ]
    )
    for actual, expected in zip(
        forward, oracle("adaptive", u_price, u_demand), strict=True
    ):
        for name in NAMES:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-8)

    artifact = encode_policy_program(NAMES, policies["adaptive"])
    identity = policy_program_identity(NAMES, policies["adaptive"])
    verified = consume_policy_program(
        artifact, expected_identity=identity, supplied=NAMES
    )
    assert verified.identity == identity
    decoded = decode_policy_program(NAMES, artifact)
    assert encode_policy_program(NAMES, decoded) == artifact
    replay = columns(
        counterfactual_temporal_policy(
            model,
            decoded,
            factual,
            window=1,
            history=history(),
            execution=EXECUTION,
        )
    )
    np.testing.assert_allclose(
        contribution(replay),
        contribution(oracle("adaptive", price_noise, demand_noise)),
        rtol=0,
        atol=1e-8,
    )
    changed = [*policies["adaptive"][:-1], transform("price", maximum(11, generated()))]
    other = policy_program_identity(NAMES, changed)
    assert other != identity
    try:
        consume_policy_program(
            encode_policy_program(NAMES, changed),
            expected_identity=identity,
            supplied=NAMES,
        )
    except CausalSerializationError as refusal:
        assert "expected identity" in str(refusal)
    else:
        raise AssertionError("A changed program cannot inherit the old identity")
    prepared = prepare_policy(model, [conditional])
    np.testing.assert_array_equal(
        prepared.sample(128, execution=EXECUTION).draws,
        sample_policy(model, [conditional], 128, execution=EXECUTION).draws,
    )
    rebound = prepared.rebind(fit_model(24))
    rebound.sample(128, execution=EXECUTION)
    assert rebound.identity == prepared.identity
    assert rebound.program_compilations == 1 and rebound.model_bindings == 2
    callback = callback_example(model, conditional)
    print(
        json.dumps(
            {
                "worlds": results,
                "forward_adaptive_mean_contribution": float(
                    np.mean(contribution(forward))
                ),
                "support": asdict(support),
                "refusals": [cycle, missing],
                "program_identity": identity,
                "changed_program_identity": other,
                "reuse": {
                    "program_compilations": rebound.program_compilations,
                    "model_bindings": rebound.model_bindings,
                },
                "callback": callback,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
