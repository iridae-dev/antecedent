"""Strict evidence for stateless measured factories with retained native source replay."""

import ast
import tomllib

CONTRACTS = {
    "antecedent.transport.joint_bayesian": (
        "transport.joint_bayesian",
        "joint_bayesian_transport",
    ),
    "antecedent.learned.joint_transport": (
        "transport.learned_joint",
        "learned_joint_transport",
    ),
    "antecedent.transport.binary_nested_markov": (
        "identify.binary_nested_markov",
        "binary_nested_markov",
    ),
    "antecedent.transport.binary_nested_markov_fisher_interval": (
        "identify.binary_nested_markov_fisher",
        "binary_nested_markov_fisher_interval",
    ),
    "antecedent.transport.temporal_dependent_interval": (
        "temporal.dependent_interval",
        "temporal_dependent_interval",
    ),
    "antecedent.transport.sampled_observation_recovery": (
        "recovery.sampled_observation",
        "sampled_observation_recovery",
    ),
}


def replay_body_problems(body, assertion, producer):
    try:
        tree = ast.parse(body)
        test = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == assertion)
        factory = next(
            n
            for n in test.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "measured" for t in n.targets)
        )
        assert isinstance(factory.value, ast.Call)
        assert ast.unparse(factory.value.func) == f"tr.{producer}"
        assert any(
            k.arg is None and isinstance(k.value, ast.Name) and k.value.id == "source_inputs"
            for k in factory.value.keywords
        )
        deleted = {
            t.id: n.lineno
            for n in test.body
            if isinstance(n, ast.Delete)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        replay = next(
            n.value
            for n in test.body
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "independent_replay"
        )
        assert [ast.unparse(arg) for arg in replay.args] == [
            "payload",
            "identity",
            "original",
            "tmp_path",
        ]
        helper = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "independent_replay"
        )
        assert not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
            for statement in helper.body
            for n in ast.walk(statement)
        )
        local_load = next(
            n.value
            for n in helper.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "loaded" for t in n.targets)
        )
        assert ast.unparse(local_load) == "MeasuredInference.load(payload, expected=identity)"
        capture = next(
            n
            for n in test.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
        )
        assert ast.unparse(capture.targets[0]) == "(payload, identity, original)"
        assert isinstance(capture.value, ast.Tuple) and len(capture.value.elts) == 3
        exported, identity, original = map(ast.unparse, capture.value.elts)
        assert exported in {"measured.export()", "bytes(measured.export())"}
        assert identity == "measured.expected_identity"
        assert original == "measured.inspect()"
        assert factory.lineno < capture.lineno < min(deleted.values())
        tree = ast.Module(body=[test, helper], type_ignores=[])
        assert deleted["source_inputs"] < replay.lineno and deleted["measured"] < replay.lineno
        assert factory.lineno < deleted["source_inputs"] and factory.lineno < deleted["measured"]
        loads = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "MeasuredInference.load"
        ]
        assert loads and all(any(k.arg == "expected" for k in n.keywords) for n in loads)
        observed = next(
            n.value
            for n in helper.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "observed" for t in n.targets)
        )
        assert isinstance(observed, ast.Call) and ast.unparse(observed.func) == "json.loads"
        child = observed.args[0]
        assert isinstance(child, ast.Call) and ast.unparse(child.func) == "subprocess.check_output"
        assert (
            ast.unparse(child.args[0])
            == "[sys.executable, '-c', script, str(path), json.dumps(identity._wire())]"
        )
        script = next(
            n.value
            for n in helper.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "script" for t in n.targets)
        )
        assert isinstance(script, ast.Constant) and isinstance(script.value, str)
        assert "expected=I._from_wire(json.loads(sys.argv[2]))" in script.value
        assert "pathlib.Path(sys.argv[1]).read_bytes()" in script.value
        assertions = [ast.unparse(n.test) for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        assert "loaded.inspect() == original" in assertions and "observed == original" in assertions
        assert any(
            "data_digest" in ast.unparse(n) and "CausalError" in ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.With)
        )
        assert any(
            "source_artifact" in ast.unparse(n) and "CausalError" in ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, ast.With)
        )
    except (SyntaxError, AssertionError, StopIteration, KeyError, IndexError):
        return [
            "requires exact factory execution, discarded producer inputs/handle, expected-identity local and fresh-process replay, equality and substitution refusals"
        ]
    return []


def source_replay_problems(root, route, body):
    name = route["route"]
    contract = CONTRACTS.get(name)
    if contract is None or route.get("stage") != "uncertainty":
        return [
            "source-bound replay is only defined for the six exact measured uncertainty factories"
        ]
    record_id, producer = contract
    records = tomllib.loads((root / "parity/promotion.toml").read_text(encoding="utf-8"))["record"]
    record = next((r for r in records if r["id"] == record_id), {})
    if (
        route.get("promotion_record") != record_id
        or record.get("status") != "promoted"
        or not record.get("coverage_records")
    ):
        return [
            "source-bound replay requires its promoted measured owner and actual scalar coverage records"
        ]
    if not any(
        r.get("name") == name
        and r.get("stage") == "uncertainty"
        and r.get("status") == "licensed"
        and r.get("claim") == "calibrated"
        for r in record.get("routes", [])
    ):
        return ["source-bound replay must match the owning calibrated public uncertainty route"]
    proof = tomllib.loads((root / f"provenance/{record_id}.toml").read_text(encoding="utf-8"))
    if route["evidence_test"] not in proof.get("test_sources", []):
        return ["source-bound replay evidence must be registered in its original method provenance"]
    return replay_body_problems(body, route["evidence_assertion"], producer)


def shared_evidence_problems(routes):
    cited_by = {}
    for route in routes:
        if route.get("status") == "licensed":
            key = (route.get("evidence_test", ""), route.get("evidence_assertion", ""))
            cited_by.setdefault(key, []).append(route.get("route", "?"))
    return [
        f"{path}::{assertion} is the sole evidence for {len(names)} routes "
        f"({', '.join(names)}); cite a distinct assertion per route"
        for (path, assertion), names in sorted(cited_by.items())
        if len(names) > 1
    ]
