# Use a causal analysis as a component in your application

A retention service wants to know how many more weekly sessions a reminder message produces, and it wants to answer that for its own callers without a person reading each analysis. This tutorial builds the part of that service that talks to Antecedent. It shows how your code submits a request, reads the plan before spending compute, handles an answer and a refusal, decides what an answer permits, adapts to new data and hands a result to another service.

Antecedent supplies the causal computation and its contracts. Your application supplies planning, permissions and the policy for acting on an answer. The tutorial keeps that line visible: every decision in the code below belongs to the host, and every field the host reads comes from Antecedent.

**The setting.** The users are simulated, so the effect can be checked. The reminder is assigned at random, prior activity is a precision covariate, and the true effect of the reminder on weekly sessions is 2 in this model. That is a fixture truth of the simulated model, not a claim about any real service. Run the whole example with:

```bash
python examples/python/application_component.py
```

## What the host declares and what Antecedent computes

The caller declares three things: the data, a graph stating which variables drive which, and one question. Here the question is the average effect of `reminder` on `sessions`, and the graph says prior activity drives sessions.

```python
@dataclass(frozen=True)
class Request:
    """What a caller submits: prepared data, a declared graph and one question."""

    data: dict
    graph: object
    query: object


def observed_request(data):
    """The caller declares that prior activity is the only other driver of sessions."""
    graph = [("reminder", "sessions"), ("prior_sessions", "sessions")]
    return Request(data, graph, ant.AverageEffect("reminder", "sessions"))
```

The graph is a declaration, not something Antecedent learned from the data. Antecedent works out what the declared structure identifies (here, that the effect is estimable by adjusting for prior activity), runs the estimator it selected and reports what it did. If the graph is wrong, the answer is conditional on a wrong graph, and no field in the result can tell you that. That judgement stays with whoever declares the structure.

A request carries no executable content: data, a graph and a question. Nothing in it names code to run.

## Prepare and inspect before running

Preparing a request compiles it and identifies the question. Nothing is fitted.

```python
def prepare_request(request):
    """Compile the request. Nothing is fitted; the study is what was compiled."""
    return ant.prepare(
        request.data,
        graph=request.graph,
        query=request.query,
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=19),
    )
```

The inference declaration requests a point estimate with no bootstrap interval, so the host has no interval to trust or discard in this tutorial. `Execution` carries the seed and, for a real service, the resource limits, cache policy and a cancellation token.

The plan view reads what compile retained and runs nothing. The host takes the few fields it needs to decide whether to proceed:

```python
def read_plan(study):
    """What the host reads before it spends any compute."""
    if study.refusal is not None:
        return {"stage": "refused", "code": study.refusal.code}
    view = study.plan_view()
    return {
        "stage": view.stage,
        "route": view.route,
        "support_status": view.support.status,
        "identification": view.identification.status,
        "program": view.identity,
    }
```

`support_status` is the support-matrix status of the selected route. A `licensed` route is one the support matrix licenses for this combination of question, graph class, structure, inference and validation. Antecedent can also run a route that is not licensed and record that status. Successful execution does not upgrade the scientific claim. `program` is the identity of the compiled program, and every result of the program carries it. This is what lets a host log, compare and cache by what was asked, not by when it ran.

## Run it and read the claim

```python
def execute(study):
    """Run a prepared study. A scientific refusal and an answer are both returned."""
    outcome = ant.estimate(study)
    if not outcome.accepted:
        try:
            outcome.refusal.raise_error()
        except ant.CausalError as error:
            return {
                "status": "unsupported",
                "refusal": host.refusal_record(error),
            }, outcome
    report = outcome.inspect()
    claim = host.claim_record(report.contract["claim"])
    response = {
        "status": claim["status"],
        "claim": claim,
        "value": float(outcome.answer.value),
        "support_status": report.support.payload["matrix_status"],
        "calibration": report.calibration.status,
    }
    return response, outcome
```

`ant.estimate` returns an `Outcome`. It either holds a result or holds a typed scientific refusal, and the host asks which with `outcome.accepted`. An operational failure (a cancellation, a resource limit, a corrupt artifact) is different: it raises an exception and never becomes an answer.

For a successful outcome the host reads four things.

| Field | What it tells the host |
| --- | --- |
| `claim["status"]` | `supported`, `partial` or `unsupported`, decided by one rule shared with Rust. `supported` means the claim is identified under the declared structure and evaluated, so the host may use it as stated. It does not certify the declared structure. |
| `claim["kind"]` | The kind of answer: a `point`, or `bounds` or `incomplete` when the structure identifies less. A `partial` claim must travel with its limit. |
| `support_status` | The licence status of the route that ran. |
| `calibration` | Whether an interval carries a calibration record. Read the status field; do not infer it from whether an interval is present. |

In this run the status is `supported`, the kind is a `point`, and the value is near 2, as the simulated model says it should be. The example asserts that, with a tolerance, because one sample of 160 users estimates the effect with sampling error. It also asserts that the calibration field is a status string. This request asks for no interval, so there is nothing to calibrate, and the tutorial makes no statement about interval coverage.

## Apply your own acceptance policy

Antecedent reports fields. It does not decide what to do about them. That is the host's policy, and it can be as small as this:

```python
@dataclass(frozen=True)
class AcceptancePolicy:
    """The host's own rule for acting on an answer. Antecedent does not supply one."""

    accepted_support: frozenset = frozenset({"licensed"})


def decide(response, policy):
    """The host's policy: Antecedent reports fields, the host chooses what they permit."""
    if response["status"] == "unsupported":
        return "reject"
    if response["status"] == "partial":
        return "relay_with_limits"
    if response["support_status"] not in policy.accepted_support:
        return "send_to_review"
    return "act"
```

This service acts only on a supported claim from a licensed route, relays a partial claim with its limit and rejects the rest. A stricter service would change `accepted_support`, and the same answer would go to review instead. The example asserts both decisions. The policy is your code because it encodes what your service is willing to do on a claim, which depends on stakes Antecedent cannot see.

## Handle a refusal

A refusal is scientific information. It says what was asked, what the declared structure cannot support, and what would change that. Here the caller sends the same data with a different graph: an unmeasured common cause of the reminder and the sessions.

```python
def confounded_request(data):
    """The caller declares an unmeasured common cause of reminder and sessions."""
    graph = ant.Admg.from_edges(
        ["prior_sessions", "reminder", "sessions"],
        [
            ("prior_sessions", "reminder"),
            ("prior_sessions", "sessions"),
            ("reminder", "sessions"),
        ],
        [("reminder", "sessions")],
    )
    return Request(data, graph, ant.AverageEffect("reminder", "sessions"))
```

Preparing the request retains the refusal on the study, so the host can decline before running anything (`study.refusal`, and the plan reads `refused`). Executing the study returns an outcome that is not accepted, with `outcome.answer.kind == "refused"`. The host classifies it with `host.refusal_record`, which gives a record with a registered `reason_code` (here `effect_not_identified`), a stable `family`, the message and the remedy, and the example asserts the code is in `host.refusal_codes()`. The record is never a number, so the policy rejects it.

Three situations look alike to a caller and stay distinct here:

* **The data cannot answer the question under the declared structure.** This refusal. Changing the data or the graph, or adding evidence such as measuring the common cause, changes what is identifiable. `study.inspect()` lists `next_actions` and licensed `neighbors` when Antecedent has them.
* **An implementation is unavailable.** The refusal names a route or capability that does not exist. Retrying will not help.
* **Something operational failed.** A resource limit or a cancellation raises an exception. Retrying with more resources may help, and nothing was answered.

A host that retries refusals, or substitutes another route silently when one refuses, answers a different question than the caller asked.

## Branch onto changed data

A week later the service has new observations. Rebinding returns a new branch of the same study on the new data. The question, the graph and the estimator are retained, and the earlier result is untouched.

```python
changed = {**data, "sessions": data["sessions"] + data["reminder"]}
branch = ant.rebind(study, data=changed)
branch_response, _ = execute(branch)
```

The changed data add one session for every user who got the reminder, so the effect rises by exactly 1. The example asserts that difference, that `branch` is a different object from `study`, that estimating the original study again returns the original value, and that the plan identities differ because each program binds the digest of the observations it reads. A host can keep both branches, compare them, and refuse to mix them up.

A changed *design* is a different question and needs a new `ant.prepare`: `ant.rebind` refuses to carry a randomized-assignment declaration across changed assignments.

## Export, then consume as an independent consumer

The service now hands its result to another service. Export produces bytes and an expectation, which the producer retains separately:

```python
artifact = outcome.export()
retained = json.dumps(artifact.expectation.to_dict()["identity"])
loaded = consume(bytes(artifact), retained)
```

The consumer holds only those bytes and the identity it retained. It loads against the expected identity and checks the acceptance fields:

```python
def consume(encoded, retained_identity):
    """An independent consumer: it has only the bytes and the identity it retained."""
    expected = json.loads(retained_identity)
    loaded = ant.load(encoded, expected=expected)
    if loaded.acceptance.identity_bound and loaded.acceptance.answer_available:
        return loaded
    raise AssertionError(f"not accepted as a claim: {loaded.acceptance}")
```

The loaded value equals the producer's. Loading restores a recorded claim. It does not recreate the study: `loaded.study` is `None`, so the consumer cannot refresh, explore or re-execute anything, and the artifact itself stays what it recorded. A consumer that wants a new answer prepares a new study on its own data.

The consumer also refuses what it must. Altered bytes raise an error that classifies into a stable family, and a different request's identity does not accept these bytes. In the example both are asserted. In production the identity belongs in a record the byte channel cannot rewrite, as [save, hand off and accept results](../guides/build/artifacts-and-handoff.md) explains, and that guide also separates the producer and consumer into two processes.

## How an agent uses the same interface

An agent that calls Antecedent as a tool uses exactly these steps. A tool layer exposes the functions in this tutorial (prepare a request, read the plan, execute, read the decision) and the model chooses which to call and with what. Each tool returns a plain record: a plan, a response with a `status` and a claim, or a refusal with a `reason_code` and a remedy. A model can branch on those fields. It cannot turn a refusal into a number, because Antecedent returned no number.

Antecedent contains no agent logic: no planner, no prompt, no memory, no tool loop. The planning, the permissions and the acceptance policy remain in your code. [Embed Antecedent in an application or agent](../guides/build/embedding.md) lists the versioned JSON Schema for these records, the stable status, refusal and error codes, and the rules for resources, cancellation and callbacks.

## What this established

You have a host that submits a prepared request, reads the plan, distinguishes an answer from a typed refusal and an operational failure, applies a policy of its own to the status, support and calibration fields, keeps the earlier result when new data arrive and hands a result to a consumer that can verify it without the study behind it. In the simulated model, the recovered effect matches the known effect of 2 within sampling error, and the structure that cannot identify it is refused with a registered code.

The next questions are about the host's environment. To keep results current as data change, see [retained refresh and state publication](../guides/retained-refresh.md) and [staleness and drift diagnostics](../guides/staleness.md). To connect this analysis to a decision or another component's output, see [connect scientific components and bring in outside results](../guides/build/scientific-interfaces.md).

## Complete runnable source

<!-- example: examples/python/application_component.py -->
