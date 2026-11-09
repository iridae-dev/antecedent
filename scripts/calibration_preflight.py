"""The frozen decision context a formal calibration measurement must carry
(checked by `calibration_runner.execute`)."""
import json

import calibration_decision as decision


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
