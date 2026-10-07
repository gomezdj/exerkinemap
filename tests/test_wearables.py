from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exerkinemap.pooling.frame.dhdr_tools import load_dhdr_cgm_csv
from Exerkinetics.wearables.flirt_features import causal_window_features
from Exerkinetics.wearables.gate import ExtrinsicGate, apply_illness_mask, modulate_scores
from Exerkinetics.wearables.posture import attach_deep_postures
from Exerkinetics.wearables.schema import WearableSchemaError, normalize_wearable_frame


def test_cgm_loader_normalizes_the_common_wearable_schema(tmp_path) -> None:
    input_path = tmp_path / "cgm.csv"
    pd.DataFrame(
        {
            "person": ["p2", "p1"],
            "time": ["2026-01-01T00:05:00Z", "2026-01-01T00:00:00Z"],
            "glucose": [120, 100],
        }
    ).to_csv(input_path, index=False)

    frame = load_dhdr_cgm_csv(
        input_path,
        subject_column="person",
        timestamp_column="time",
        glucose_column="glucose",
    )

    assert list(frame.columns) == [
        "subject_id",
        "timestamp",
        "signal",
        "value",
        "source",
        "unit",
    ]
    assert frame.iloc[0]["subject_id"] == "p1"
    assert set(frame["signal"]) == {"glucose_mg_dl"}


def test_schema_rejects_invalid_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "person": ["p1"],
            "time": ["not-a-time"],
            "value": [100],
        }
    )

    with pytest.raises(WearableSchemaError, match="invalid timestamps"):
        normalize_wearable_frame(
            raw,
            column_map={"subject_id": "person", "timestamp": "time", "value": "value"},
            source="test",
            default_signal="glucose_mg_dl",
        )


def test_causal_windows_do_not_use_future_observations() -> None:
    frame = normalize_wearable_frame(
        pd.DataFrame(
            {
                "person": ["p1", "p1", "p1"],
                "time": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:10:00Z",
                    "2026-01-01T00:40:00Z",
                ],
                "value": [100, 110, 200],
            }
        ),
        column_map={"subject_id": "person", "timestamp": "time", "value": "value"},
        source="test",
        default_signal="glucose_mg_dl",
        default_unit="mg/dL",
    )

    features = causal_window_features(frame, window="30min")

    second_window = features.loc[features["window_end"] == pd.Timestamp("2026-01-01T00:10:00Z")].iloc[0]
    assert second_window["mean"] == 105
    assert second_window["max"] == 110


def test_gate_is_bounded_and_illness_mask_is_neutral() -> None:
    gate = ExtrinsicGate(weights=np.array([2.0, -1.0]))
    weights = gate.transform(np.array([[0.0, 0.0], [10.0, -10.0]]))

    assert np.all(weights >= 0.5)
    assert np.all(weights <= 1.5)
    np.testing.assert_allclose(apply_illness_mask(weights, [False, True]), [1.0, 1.0])
    np.testing.assert_allclose(
        modulate_scores(np.array([2.0, 3.0]), weights),
        np.array([2.0, 3.0]) * weights,
    )


def test_posture_adapter_requires_one_label_per_feature_row() -> None:
    features = pd.DataFrame({"mean": [1.0, 2.0]})

    labeled = attach_deep_postures(features, lambda _: ["standing", "walking"])
    assert list(labeled["posture"]) == ["standing", "walking"]
    with pytest.raises(ValueError, match="returned 1 labels"):
        attach_deep_postures(features, lambda _: ["standing"])