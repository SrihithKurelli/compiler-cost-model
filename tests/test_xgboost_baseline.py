import numpy as np

from scripts.train_xgboost_baseline import (
    FEATURE_NAMES,
    TRANSFORM_TYPES,
    extract_features,
)


def _program():
    return {
        "program_annotation": {
            "iterators": ["i", "j"],
            "computations": ["C0", "C1"],
            "buffers": {"buffers_info": ["A", "B"]},
            "memory_size": 4096,
        }
    }


def test_feature_vector_has_stable_finite_schema():
    program = _program()
    schedule = {
        "transformations_list": [
            {
                "type": "tiling",
                "loop_levels": [0, 1],
                "parameters": [32, 4],
                "computations": ["C0", "C1"],
            },
            {
                "type": "parallelization",
                "loop_levels": [0],
                "parameters": [1],
                "computations": ["C0"],
            },
        ]
    }
    features = extract_features(program, schedule)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert len(TRANSFORM_TYPES) == 8


def test_features_are_invariant_to_transformation_order():
    program = _program()
    first = {
        "type": "tiling",
        "loop_levels": [0, 1],
        "parameters": [32, 4],
        "computations": ["C0"],
    }
    second = {
        "type": "parallelization",
        "loop_levels": [0],
        "parameters": [1],
        "computations": ["C0", "C1"],
    }
    forward = {"transformations_list": [first, second]}
    reverse = {"transformations_list": [second, first]}
    np.testing.assert_allclose(
        extract_features(program, forward),
        extract_features(program, reverse),
    )


def test_empty_identity_schedule_has_zero_transformation_features():
    features = extract_features(_program(), {"transformations_list": []})
    assert features.shape == (len(FEATURE_NAMES),)
    assert features[4] == 0
    assert features[5] == 0
