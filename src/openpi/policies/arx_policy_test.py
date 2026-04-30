import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import arx_policy


# === Delta EE Mode (14D state / 14D action) ===

def test_arx_delta_ee_inputs_maps_fixed_schema():
    transform = arx_policy.ArxDeltaEEInputs(model_type=_model.ModelType.PI05)
    data = {
        "observation/state": np.arange(14, dtype=np.float32),
        "observation/image": np.zeros((3, 8, 6), dtype=np.float32),
        "observation/wrist_image": np.ones((3, 8, 6), dtype=np.float32),
        "observation/right_wrist_image": np.full((3, 8, 6), 0.5, dtype=np.float32),
        "actions": np.ones((4, 14), dtype=np.float32),
        "prompt": "pick up the block",
    }

    result = transform(data)

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (4, 14)
    assert result["prompt"] == "pick up the block"
    assert result["image"]["base_0_rgb"].shape == (8, 6, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (8, 6, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (8, 6, 3)
    assert set(result["image_mask"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert result["image"]["base_0_rgb"].dtype == np.uint8


def test_arx_delta_ee_inputs_extracts_policy_state_from_full_lerobot_state():
    transform = arx_policy.ArxDeltaEEInputs(model_type=_model.ModelType.PI05)
    full_state = np.arange(28, dtype=np.float32)
    data = {
        "observation/state": full_state,
        "observation/image": np.zeros((8, 6, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((8, 6, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.full((8, 6, 3), 2, dtype=np.uint8),
        "actions": np.ones((4, 14), dtype=np.float32),
        "prompt": "pick up the block",
    }

    result = transform(data)

    np.testing.assert_array_equal(
        result["state"],
        np.asarray([*range(12, 24), 24, 26], dtype=np.float32),
    )


def test_arx_delta_ee_outputs_slice_to_14d():
    transform = arx_policy.ArxDeltaEEOutputs(action_dim=14)
    outputs = transform({"actions": np.ones((6, 32), dtype=np.float32)})

    assert outputs["actions"].shape == (6, 14)
    np.testing.assert_array_equal(outputs["actions"], np.ones((6, 14), dtype=np.float32))


# === Full Joint + Chassis Mode (59D state / 32D action) ===

def test_arx_full_inputs_maps_59d_state_and_images():
    transform = arx_policy.ArxFullInputs()
    data = {
        "state": np.arange(59, dtype=np.float32),
        "images": {
            "head": np.zeros((3, 240, 424), dtype=np.float32),
            "left_wrist": np.ones((3, 240, 424), dtype=np.float32),
            "right_wrist": np.full((3, 240, 424), 0.5, dtype=np.float32),
        },
        "actions": np.ones((4, 32), dtype=np.float32),
        "prompt": "pick up the bottle",
    }

    result = transform(data)

    assert result["state"].shape == (59,)
    assert result["actions"].shape == (4, 32)
    assert result["prompt"] == "pick up the bottle"
    # head → base_0_rgb, left/right wrist mapped correctly
    assert result["image"]["base_0_rgb"].shape == (240, 424, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (240, 424, 3)
    assert result["image"]["right_wrist_0_rgb"].shape == (240, 424, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    # all three cameras present → all masks True
    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.True_


def test_arx_full_inputs_masks_missing_images():
    transform = arx_policy.ArxFullInputs()
    data = {
        "state": np.zeros(59, dtype=np.float32),
        "images": {
            # head absent — mask should be False; wrists present
            "left_wrist": np.zeros((240, 424, 3), dtype=np.uint8),
            "right_wrist": np.zeros((240, 424, 3), dtype=np.uint8),
        },
        "prompt": "test",
    }

    result = transform(data)

    assert result["image_mask"]["base_0_rgb"] == np.False_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.True_
    # placeholder zeros for missing head
    assert result["image"]["base_0_rgb"].dtype == np.uint8


def test_arx_full_outputs_slice_to_32d():
    transform = arx_policy.ArxFullOutputs(action_dim=32)
    outputs = transform({"actions": np.ones((6, 64), dtype=np.float32)})

    assert outputs["actions"].shape == (6, 32)
    np.testing.assert_array_equal(outputs["actions"], np.ones((6, 32), dtype=np.float32))
