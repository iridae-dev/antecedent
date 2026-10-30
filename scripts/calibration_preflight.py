"""Read-only preflight for direct formal calibration-gate execution."""
import json
import os
import sys
import calibration_decision as decision
import calibration_family as family


def validate_inherited(value, expected):
    if not value:
        raise ValueError("missing frozen calibration decision context; use scripts/measure_calibration.sh")
    try:
        supplied = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid calibration decision context JSON") from error
    decision.validate_config(supplied)
    if supplied != expected:
        raise ValueError("calibration decision context differs from the complete frozen family")


def main():
    try:
        value = os.environ.get(decision.ENV)
        if not value:
            raise ValueError("missing frozen calibration decision context; use scripts/measure_calibration.sh")
        validate_inherited(value, family.current_config())
    except (ValueError, OSError, KeyError) as error:
        print(f"calibration preflight: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
