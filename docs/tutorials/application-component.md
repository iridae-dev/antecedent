# Use a causal analysis as a component in your application

Does a reminder bring users back more often? A retention service estimates how many extra sessions a reminder produces each week. Your application needs to decide what to send the caller when it gets an estimate, when the data cannot answer the question, and when the job fails.

You’ll build the part of the service that makes those decisions. Antecedent computes the estimate and reports the assumptions behind it. Your code decides whether to use it, send it for review, or ask the caller for different evidence.

In the simulated data, users receive reminders at random. A reminder adds two weekly sessions on average. Prior activity helps predict weekly sessions, so the analysis uses it to improve precision.

Run `python examples/python/application_component.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/application_component.ipynb). See the [complete example](#complete-runnable-source).

For a first look at the objects the service will handle, see [your first analysis](../start/first-analysis.md) and [reading a result](../start/reading-a-result.md).

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

Antecedent uses the graph you supply to determine whether it can estimate the effect. Here it can compare users who did and did not receive reminders, accounting for prior activity. The person who specified the graph is responsible for whether it describes the service.

The request contains data, a graph and a question. Antecedent selects the code that runs.

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

Inspect the plan to see which route Antecedent selected. The host reads these fields before committing resources to the fit.

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

For the first request the example prints the plan as one JSON line and asserts that `stage` is `prepared` and `support_status` is `supported`.

`support_status` tells you whether the release supports this combination of question, graph, inference and validation. For an unsupported route, the caller receives `not_supported`. Use `program` to match a result to the compiled request that produced it, including when you log or cache results.

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
| `claim["status"]` | `supported`, `partial` or `unsupported`, decided by one rule shared with Rust. `supported` means the claim is identified under the declared structure and evaluated, so the host may use it as stated. It does not check the declared structure. |
| `claim["kind"]` | The kind of answer: a `point`, or `bounds` or `incomplete` when the structure identifies less. A `partial` claim must travel with its limit. |
| `support_status` | The support status of the route that ran. |
| `calibration` | Whether an interval carries a calibration record. Read the status field; do not infer it from whether an interval is present. |

With 160 simulated users, the estimate should be close to two extra sessions per week. Sampling variation explains why the script checks it with a tolerance. This request returns a `supported` point estimate. It requests no interval.

## Apply your own acceptance policy

Decide which results your service can use automatically and which need review. This example implements that decision in one function.

```python
@dataclass(frozen=True)
class AcceptancePolicy:
    """The host's own rule for acting on an answer. Antecedent does not supply one."""

    accepted_support: frozenset = frozenset({"supported"})


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

This service acts only on a supported claim from a supported route, relays a partial claim with its limit and rejects the rest. A stricter service would change `accepted_support`, and the same answer would go to review instead. The example asserts both decisions. The policy is your code because it encodes what your service is willing to do on a claim, which depends on stakes Antecedent cannot see.

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

Check `study.refusal` before executing. If you do execute, `outcome.answer.kind` is `"refused"`. Use `host.refusal_record` to get the registered `reason_code` (`effect_not_identified` here), its `family`, a message for the caller and a proposed remedy.

Three situations look alike to a caller and stay distinct here:

* **The data cannot answer the question under the declared structure.** This refusal. Changing the data or the graph, or adding evidence such as measuring the common cause, changes what is identifiable. `study.inspect()` lists `next_actions` and supported `neighbors` when Antecedent has them.
* **An implementation is unavailable.** The refusal names a route or capability that does not exist. Retrying will not help.
* **Something operational failed.** A resource limit or a cancellation raises an exception. Retrying with more resources may help, and nothing was answered.

Retry a cancelled job if resources permit. For an identification refusal, return the reason and remedy to the caller. Switching estimators silently could change the question.

## Branch onto changed data

A week later the service has new observations. Rebinding returns a new branch of the same study on the new data. The question, the graph and the estimator are retained, and the earlier result is untouched.

```python
changed = {**data, "sessions": data["sessions"] + data["reminder"]}
branch = ant.rebind(study, data=changed)
branch_response, _ = execute(branch)
```

In the new table, each user who received a reminder has one more session. The estimated effect therefore rises by exactly one. Each branch has a distinct program identity because the identity includes a digest of the data. You can still estimate the original study and recover its earlier value.

A changed *design* is a different question and needs a new `ant.prepare`: `ant.rebind` refuses to carry a randomized-assignment declaration across changed assignments.

## Hand the result to another service

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
    raise AssertionError(f"not intact for the expected identity: {loaded.acceptance}")
```

The loaded value equals the producer's. Loading checks that the artifact is intact and restores a recorded claim; it recomputes nothing. It does not recreate the study: `loaded.study` is `None`, so the consumer cannot refresh, explore or re-execute anything, and the artifact itself stays what it recorded. A consumer that wants a new answer prepares a new study on its own data. To cite the methods behind the result in a handoff note or report, the producer reads `outcome.method_provenance` before export; see [cite the methods behind a study](../guides/citing-a-study.md).

The consumer rejects altered bytes and an artifact that belongs to a different request. Store the expected identity somewhere the sender of the bytes cannot overwrite. [Save, hand off and accept results](../guides/build/artifacts-and-handoff.md) shows the two services running in separate processes.

## How an agent uses the same interface

An agent that calls Antecedent as a tool uses exactly these steps. A tool layer exposes the functions in this tutorial (prepare a request, read the plan, execute, read the decision) and the model chooses which to call and with what. Each tool returns a plain record: a plan, a response with a `status` and a claim, or a refusal with a `reason_code` and a remedy. A model can branch on those fields. It cannot turn a refusal into a number, because Antecedent returned no number.

Antecedent contains no agent logic: no planner, no prompt, no memory, no tool loop. The planning, the permissions and the acceptance policy remain in your code. [Embed Antecedent in an application or agent](../guides/build/embedding.md) lists the versioned JSON Schema for these records, the stable status, refusal and error codes, and the rules for resources, cancellation and callbacks.

## What this established

The host can now return two useful responses to the same caller. With random assignment it reports about two extra sessions per week. With an unmeasured common cause it returns `effect_not_identified` and explains what evidence would help.

## What this did not establish

None of this shows that a declared graph is true: a `supported` status is conditional on the structure the caller declared. The request asked for no interval, so nothing here bears on interval coverage. The acceptance policy is an example of where a host's rule goes, not a recommended rule.

## Where to go next

The next questions are about the host's environment. To keep results current as data change, see [retained refresh and state publication](../guides/retained-refresh.md) and [staleness and drift diagnostics](../guides/staleness.md). To connect this analysis to a decision or another component's output, see [connect scientific components and bring in outside results](../guides/build/scientific-interfaces.md).

## Complete runnable source

<!-- example: examples/python/application_component.py -->

[Open the Python script](../../examples/python/application_component.py)
