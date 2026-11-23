"""Design payloads refuse an absent identifier instead of spelling it as text."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from antecedent.errors import CausalValueError
from antecedent.estimation import _panel_did_payload
from antecedent.quasi import PanelDifferenceInDifferences


def _data(subjects):
    return {
        "y": np.arange(4.0),
        "id": subjects,
        "group": np.array([1.0, 1.0, 0.0, 0.0]),
        "after": np.array([0.0, 1.0, 0.0, 1.0]),
    }


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_missing_subject_id_refused(missing):
    subjects = np.array(["a", missing, "b", "b"], dtype=object)
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    with pytest.raises(CausalValueError, match="missing identifier"):
        _panel_did_payload(_data(subjects), query)


def test_present_subject_ids_are_kept():
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    _, _, design = _panel_did_payload(_data(np.array(["a", "a", "b", "b"], dtype=object)), query)
    assert design["subjects"] == ("a", "a", "b", "b")
