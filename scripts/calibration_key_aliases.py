"""Validate the single reviewed identity migration without rewriting measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

RECORD_ID = (
    "cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95."
    "binary_missingness_whole_row_recovery_bca_l95"
)
MEASUREMENT_SHA = "5c76724fbac09c655ec77576a648afb0ed1a0779"
COLLECTION_SHA = "e4bc216c0f01bf913f373df721863e0730e0ed14"
RECORD_SHA256 = "9302ebf37b4ce12679561198caf2c2cf0319b826e0cf3b37d08de53d5ccdffaa"
FUNCTIONAL_SHA256 = "0067df569dc7b83db8c7f9b8a465e35ab298b4c750a25af619870026a0a0f33a"


def canonical_digest(value: object) -> str:
    """Hash complete registry data, including its original measurement provenance."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validated_aliases(root: Path) -> list[dict[str, str]]:
    """Return only proven mappings; pending declarations confer no authority."""
    data = tomllib.loads((root / "parity/calibration_key_aliases.toml").read_text(encoding="utf-8"))
    if set(data) != {"version", "alias"} or data["version"] != 1:
        raise ValueError("unsupported calibration identity mapping schema")
    aliases = data["alias"]
    if not isinstance(aliases, list) or len(aliases) != 1:
        raise ValueError("only the original BCa recovery identity mapping is permitted")
    row = aliases[0]
    expected = {
        "record_id": RECORD_ID,
        "measurement_sha": MEASUREMENT_SHA,
        "collection_sha": COLLECTION_SHA,
        "original_record_sha256": RECORD_SHA256,
        "original_functional_sha256": FUNCTIONAL_SHA256,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("mapping must bind the exact original recovery measurement")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", COLLECTION_SHA, "HEAD"],
        cwd=root, check=True, capture_output=True,
    )
    original_text = subprocess.run(
        ["git", "show", f"{COLLECTION_SHA}:parity/coverage_records.toml"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    original = next(
        record for record in tomllib.loads(original_text)["record"] if record["id"] == RECORD_ID
    )
    current_records = tomllib.loads(
        (root / "parity/coverage_records.toml").read_text(encoding="utf-8")
    )["record"]
    current = [record for record in current_records if record["id"] == RECORD_ID]
    if len(current) != 1 or current[0] != original or canonical_digest(original) != RECORD_SHA256:
        raise ValueError("original recovery coverage record must remain unchanged")
    if hashlib.sha256(original["functional"].encode("utf-8")).hexdigest() != FUNCTIONAL_SHA256:
        raise ValueError("original recovery functional identity changed")
    if row.get("status") == "pending_identity_replay":
        if set(row) != set(expected) | {"status"}:
            raise ValueError("pending mapping cannot carry unchecked runtime claims")
        return []
    receipt_fields = {"native_receipt", "native_receipt_sha256", "replay_receipt", "replay_receipt_sha256"}
    if row.get("status") != "verified_identity_replay" or set(row) != set(expected) | {"status"} | receipt_fields:
        raise ValueError("identity mapping requires reviewed native construction and replay evidence")
    native = read_receipt(root, row["native_receipt"], row["native_receipt_sha256"])
    replay = read_receipt(root, row["replay_receipt"], row["replay_receipt_sha256"])
    return [validate_evidence(root, original, native, replay)]


MAX_RECEIPT_BYTES = 4 * 1024 * 1024
PROOF_MARKER = "RECOVERY_IDENTITY_PROOF="


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_receipt(root: Path, relative: str, digest: str) -> dict:
    if not isinstance(relative, str):
        raise ValueError("identity evidence must name a registry JSON receipt")
    path = root / relative
    if (not relative.startswith("parity/")
            or path.suffix != ".json" or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError("identity evidence must be a bounded registry JSON receipt")
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("identity evidence exceeds receipt bounds")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("identity evidence receipt digest changed")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("identity evidence receipt must be an object")
    return result


def parse_native_log(log: str) -> dict:
    proofs = [json.loads(line[len(PROOF_MARKER):]) for line in log.splitlines()
              if line.startswith(PROOF_MARKER)]
    if (len(proofs) != 1 or not isinstance(proofs[0], dict)
            or "test result: ok. 1 passed; 0 failed;" not in log):
        raise ValueError("native construction evidence requires one actual passing proof test")
    return proofs[0]


def validate_evidence(root: Path, original: dict, native: dict, replay: dict) -> dict[str, str]:
    """Check actual producer logs and every emitted value before one exact alias."""
    import calibration_facets
    import collect_coverage_records

    source = native.get("source_sha")
    if (native.get("version") != 1 or replay.get("version") != 1
            or not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source)
            or replay.get("source_sha") != source or replay.get("record_id") != RECORD_ID
            or native.get("branch") != "2.3.0" or replay.get("branch") != "2.3.0"):
        raise ValueError("identity evidence must bind one real main-branch source commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"],
                   cwd=root, check=True, capture_output=True)
    log = native.get("log")
    if not isinstance(log, str) or sha256(log) != native.get("log_sha256"):
        raise ValueError("native construction log digest changed")
    proof = parse_native_log(log)
    if not calibration_facets._bit_equal(proof, native.get("proof")):
        raise ValueError("native construction proof differs from actual test output")
    config = {
        "interval_method": "bootstrap_bca", "replicates": 2000, "seed": 0x46ab0000 + 100000,
        "max_failed_fraction_bits": "0000000000000000",
        "normalization_tolerance_bits": "3fb999999999999a", "small_cell_count_bits": "4014000000000000",
        "treated_level_bits": "3ff0000000000000", "control_level_bits": "0000000000000000",
    }
    negatives = proof.get("negative_mutations")
    if (proof.get("version") != 1 or proof.get("record_id") != RECORD_ID
            or proof.get("original_functional") != original["functional"]
            or proof.get("config") != config or proof.get("rows") != 1000
            or proof.get("identical_repeated_native_receipt") is not True
            or any(not isinstance(proof.get(field), str)
                   or not re.fullmatch(r"[0-9a-f]{32}", proof[field])
                   for field in ("input_digest", "receipt_digest"))
            or not isinstance(negatives, dict) or set(negatives) != {"graph", "effect", "sampling"}
            or any(value not in {"refused", "different_scientific_identity"} for value in negatives.values())):
        raise ValueError("native construction does not prove the original BCa graph/effect/row protocol")
    identity = proof.get("scientific_derivation_identity")
    if not isinstance(identity, str) or not identity.startswith("recovery_scientific.v1:"):
        raise ValueError("canonical identity must come from checked native recovery")
    canonical = "recovered_effect:" + identity + ":treated=3ff0000000000000:control=0000000000000000"
    if proof.get("canonical_functional") != canonical or canonical == original["functional"]:
        raise ValueError("canonical functional differs from the original native scalar construction")
    logs = replay.get("logs")
    if not isinstance(logs, dict) or not 1 <= len(logs) <= 6:
        raise ValueError("replay evidence must retain the actual bounded recovery logs")
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for name, entry in logs.items():
            if (not isinstance(name, str) or Path(name).name != name or not isinstance(entry, dict)
                    or set(entry) != {"text", "sha256"} or not isinstance(entry["text"], str)
                    or sha256(entry["text"]) != entry["sha256"]):
                raise ValueError("replay evidence log digest changed")
            path = Path(directory) / name
            path.write_text(entry["text"], encoding="utf-8")
            problems, _ = collect_coverage_records.log_problems(path)
            if problems:
                raise ValueError("recovery replay did not pass: " + "; ".join(problems))
            paths.append(path)
        try:
            actual = collect_coverage_records.merged_records(paths, sha=source)
        except SystemExit as exc:
            raise ValueError("recovery replay logs do not bind their actual main source") from exc
    if set(actual) != {RECORD_ID}:
        raise ValueError("identity evidence must replay only its governing recovery record")
    raw = actual.get(RECORD_ID)
    if not isinstance(raw, dict) or not calibration_facets._bit_equal(raw, replay.get("raw_main_replay_record")):
        raise ValueError("raw recovery replay record differs from actual emitted logs")
    if raw.get("functional") != canonical:
        raise ValueError("replay functional differs from original checked native construction")
    differences = calibration_facets.compare_replay(original, raw | {"functional": original["functional"]})
    if differences:
        raise ValueError("recovery replay changes measured values or scope: " + "; ".join(differences))
    return {"record_id": RECORD_ID, "measurement_sha": MEASUREMENT_SHA,
            "original_functional": original["functional"], "canonical_functional": canonical,
            "source_sha": source}


def clean_main_source(root: Path) -> str:
    """Captures must execute on a clean primary 2.3.0 checkout, never a worktree."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                              text=True).stdout.strip()
    if not (root / ".git").is_dir() or git("branch", "--show-current") != "2.3.0":
        raise ValueError("capture requires the primary 2.3.0 branch checkout")
    if git("status", "--porcelain"):
        raise ValueError("capture requires clean committed main source")
    return git("rev-parse", "HEAD")


def capture_native(root: Path) -> dict:
    source = clean_main_source(root)
    result = subprocess.run([
        "cargo", "test", "-p", "antecedent-estimate", "--test", "recovery_whole_method_calibration",
        "recovery_bca_original_measurement_identity_receipt", "--", "--exact", "--nocapture",
    ], cwd=root, check=True, capture_output=True, text=True)
    if clean_main_source(root) != source:
        raise ValueError("source changed during native construction capture")
    return {"version": 1, "source_sha": source, "branch": "2.3.0", "log": result.stdout,
            "log_sha256": sha256(result.stdout), "proof": parse_native_log(result.stdout)}


def capture_replay(root: Path, path: Path) -> dict:
    import calibration_facets

    source = clean_main_source(root)
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("raw replay capture exceeds receipt bounds")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payloads = calibration_facets.load_replay_capture(path.parent, source, raw["records"])
    if RECORD_ID not in payloads:
        raise ValueError("actual replay selection lacks the original recovery record")
    logs = {}
    for name in raw["logs"]:
        text = (path.parent / name).read_text(encoding="utf-8")
        if RECORD_ID in text:
            logs[name] = {"text": text, "sha256": sha256(text)}
    if clean_main_source(root) != source:
        raise ValueError("source changed during raw replay capture")
    return {"version": 1, "source_sha": source, "branch": "2.3.0", "record_id": RECORD_ID,
            "logs": logs, "raw_main_replay_record": payloads[RECORD_ID]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture-native", "capture-replay"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "capture-native":
        receipt = capture_native(root)
    else:
        if args.input is None:
            parser.error("capture-replay requires --input raw-replay-capture.json")
        receipt = capture_replay(root, args.input)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise ValueError("captured receipt exceeds evidence bounds")
    args.output.write_text(encoded, encoding="utf-8")
    print(f"Captured actual main-source evidence at {receipt['source_sha']}; alias remains pending.")


if __name__ == "__main__":
    main()
