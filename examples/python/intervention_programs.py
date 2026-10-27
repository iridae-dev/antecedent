"""A price floor as an intervention program, run on every route and saved as bytes.

``price := max(10, 0.85 * price)`` replaces the price the node's own mechanism generated; the
node keeps its incoming edges and its noise, so this is not ``do(price = c)``. The same program
text is drawn forward, replayed on each factual row's own abduced noise, refused when it is
malformed or reads backwards through the graph, encoded as versioned bytes under a content
identity, and, as a plain Python function, shown to be a runtime binding that cannot leave the
process unless it is declared from self-contained source."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.intervention_callbacks import (
    GENERATED,
    CallbackProgram,
    InterventionCallback,
    callback_records,
    recover_callback,
    required_callbacks,
)
from antecedent.intervention_callbacks import sample_do as sample_callbacks
from antecedent.intervention_programs import (
    assign,
    counterfactual_policy,
    decode_policy_program,
    encode_policy_program,
    generated,
    maximum,
    policy_program_identity,
    policy_support,
    policy_value,
    sample_policy,
    transform,
    var,
)

NAMES = ["price", "demand"]
N_DRAWS = 512

rng = np.random.default_rng(23)
price = 11.0 + rng.normal(scale=2.0, size=600)
demand = 50.0 - 0.8 * price + rng.normal(scale=1.0, size=600)
model = antecedent.model.fit_gcm(NAMES, [price, demand], [("price", "demand")])

floor = transform("price", maximum(10, 0.85 * generated()))

# 1. Forward sampling: on the same noise the program is the hand-applied floor.
observed = model.sample_do({}, N_DRAWS, execution=antecedent.Execution(seed=7)).draws[0]
floored = sample_policy(model, [floor], N_DRAWS, execution=antecedent.Execution(seed=7)).draws[0]
np.testing.assert_array_equal(floored, np.maximum(10.0, 0.85 * observed))
assert (floored >= 10.0).all() and (0.85 * observed < 10.0).any()

# 2. Counterfactual execution: each factual row's own noise goes through the same floor.
worlds = counterfactual_policy(
    model, [floor], ["price", "demand"], execution=antecedent.Execution(seed=1)
)
np.testing.assert_allclose(worlds["price"], np.maximum(10.0, 0.85 * price), atol=1e-9)
raised = worlds["price"] > price
assert raised.any() and (worlds["demand"][raised] < demand[raised]).all()

# Each route is licensed separately before anything runs. A transform is simulable and replayable
# but is a function of each unit's noise, not a level, so it has no observational identification.
support = policy_support(model, [floor])
assert support.simulation.licensed and support.counterfactual.licensed
assert not support.identification.licensed
assert support.identification.reason_code == "policy_route_not_licensed"

# Its model-based value is the paired difference against the unintervened model on the same noise;
# only the simulation error is quantified, and the model's own uncertainty says it is not.
value = policy_value(model, [floor], "demand", n=4000, execution=antecedent.Execution(seed=3))
# The floor lowers most prices, and demand falls with price.
assert value.mean_difference > 5 * value.difference_simulation_stderr
assert value.model_uncertainty == "unquantified"

# 3. Validation happens before any draw: a program that reads its own descendant is refused
#    with a registered reason code, and a malformed expression never reaches the library.
circular = assign("price", var("demand"))
assert policy_support(model, [circular]).simulation.reason_code == "policy_dependency_circular"
try:
    sample_policy(model, [circular], N_DRAWS, execution=antecedent.Execution(seed=1))
except CausalError as refusal:
    assert refusal.reason_code == "policy_dependency_circular", refusal
else:
    raise AssertionError("a circular read must be refused")
try:
    generated() + True
except CausalValueError:
    pass
else:
    raise AssertionError("a boolean is not a number")

# 4. A closed-form program has a portable encoding and a content identity that moves with it.
data = encode_policy_program(NAMES, [floor])
assert decode_policy_program(NAMES, data) == (floor,)
identity = policy_program_identity(NAMES, [floor])
assert identity == policy_program_identity(NAMES, decode_policy_program(NAMES, data))
other = transform("price", maximum(10, 0.9 * generated()))
assert policy_program_identity(NAMES, [other]) != identity

# 5. The same floor as a batch callback runs once for the whole batch and agrees with the
#    expression, but an in-process function has no durable encoding.
runtime = InterventionCallback(lambda g: np.maximum(10.0, 0.85 * g), inputs=["g"], name="floor")
by_callback = CallbackProgram(
    "price",
    runtime,
    reads=[GENERATED],
    mode="transform",
)
via_callback = sample_callbacks(
    model, [by_callback], N_DRAWS, execution=antecedent.Execution(seed=7)
).draws[0]
np.testing.assert_array_equal(via_callback, floored)
try:
    callback_records([by_callback])
except CausalUnsupportedError as refusal:
    assert refusal.reason_code == "policy_callback_not_portable", refusal
else:
    raise AssertionError("a runtime callback must not be exportable")

# 6. Declared from self-contained source with pinned dependencies, the callback leaves the process
#    only as a record (name, inputs, source digest, entry point, pins), never as code; a consumer
#    replays it from the source it holds, and edited source is refused.
SOURCE = "def floor(g):\n    return np.maximum(10.0, 0.85 * g)\n"
recoverable = InterventionCallback.from_source(
    SOURCE, entry_point="floor", inputs=["g"], name="floor"
)
(record,) = callback_records([recoverable])
replayed = recover_callback(SOURCE, record)
program = CallbackProgram(
    "price",
    replayed,
    reads=[GENERATED],
    mode="transform",
)
np.testing.assert_array_equal(
    sample_callbacks(model, [program], N_DRAWS, execution=antecedent.Execution(seed=7)).draws[0],
    floored,
)
artifact = encode_policy_program(
    NAMES, [CallbackProgram("price", recoverable, reads=[GENERATED], mode="transform")]
)
assert required_callbacks(artifact) == (record,)
(restored,) = decode_policy_program(NAMES, artifact, callbacks=[replayed])
np.testing.assert_array_equal(
    sample_callbacks(model, [restored], N_DRAWS, execution=antecedent.Execution(seed=7)).draws[0],
    floored,
)
try:
    encode_policy_program(NAMES, [by_callback])
except CausalUnsupportedError as refusal:
    assert refusal.reason_code == "policy_callback_not_portable", refusal
else:
    raise AssertionError("a runtime callback must not be encodable")
try:
    recover_callback(SOURCE.replace("0.85", "0.9"), record)
except CausalUnsupportedError as refusal:
    assert refusal.reason_code == "policy_callback_unrecoverable", refusal
else:
    raise AssertionError("edited source must not replay")
print("price floor: forward, counterfactual, saved and recovered programs agree")
