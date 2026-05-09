"""
ARX LIFT2 双臂移动机器人 Pi0.5 Policy Transform

将 ARX 机器人的观测/动作格式转换为 openpi 模型的标准格式。
融合了基于全身关节控制（Full Joint + Chassis）和基于末端增量控制（Delta EE）的两种模式。
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# === 1. Full Joint & Chassis Mode (32D Action, 59D State) ===
def make_arx_lift2_full_example() -> dict:
    """Creates a random input example for the ARX policy (for warmup)."""
    return {
        "state": np.ones((59,), dtype=np.float32),
        "images": {
            "head": np.random.randint(256, size=(3, 240, 424), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(3, 240, 424), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(3, 240, 424), dtype=np.uint8),
        },
        "prompt": "pick up the object",
    }

def _parse_image(img) -> np.ndarray:
    """Parse image to HWC uint8 format."""
    img = np.asarray(img)
    # Convert float images to uint8
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # Convert CHW → HWC if needed
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img

@dataclasses.dataclass(frozen=True)
class ArxLift2FullInputs(transforms.DataTransformFn):
    """Convert ARX full observations to openpi model input format."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)

        in_images = data.get("images", {})
        parsed_images = {name: _parse_image(in_images[name]) for name in in_images}

        ref_shape = (240, 424, 3)
        for name in ("head", "left_wrist", "right_wrist"):
            if name in parsed_images:
                ref_shape = parsed_images[name].shape
                break

        if "head" in parsed_images:
            images = {"base_0_rgb": parsed_images["head"]}
            image_masks = {"base_0_rgb": np.True_}
        else:
            images = {"base_0_rgb": np.zeros(ref_shape, dtype=np.uint8)}
            image_masks = {"base_0_rgb": np.False_}

        wrist_mapping = {
            "left_wrist_0_rgb": "left_wrist",
            "right_wrist_0_rgb": "right_wrist",
        }
        for dest, source in wrist_mapping.items():
            if source in parsed_images:
                images[dest] = parsed_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros(ref_shape, dtype=np.uint8)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs

@dataclasses.dataclass(frozen=True)
class ArxLift2FullOutputs(transforms.DataTransformFn):
    """Convert model output actions to ARX 32D full action format."""

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :self.action_dim], dtype=np.float32)
        return {"actions": actions}


# === 1.5. R5 Full Joint Mode (40D Dataset -> 28D Action / 56D State) ===
# This handles datasets with 40D actions and 68D states by extracting the core
# 28D/56D components. Padding to 32D is handled automatically by the model transforms.

# Action (40D -> 28D): 
# [0-13] joints, [14-25] tcp, [26-37] delta_tcp (skip), [38-39] grippers
ARX_R5_ACTION_INDICES = (*range(26), 38, 39)

# State (68D -> 56D):
# [0-41] joints(pvc), [42-53] tcp, [54-65] delta_tcp (skip), [66-67] grippers
ARX_R5_STATE_INDICES = (*range(54), 66, 67)

@dataclasses.dataclass(frozen=True)
class ArxR5FullInputs(transforms.DataTransformFn):
    """Convert ARX R5 observations to 56D model input format.
    
    Handles both training (68D state from dataset) and inference (56D state from robot adapter).
    """

    def __call__(self, data: dict) -> dict:
        # 1. Process State (68D training -> 56D, or 56D inference -> 56D)
        raw_state = np.asarray(data["state"], dtype=np.float32)
        
        # If state is 68D (training), extract to 56D; if already 56D (inference), use as-is
        if raw_state.shape[-1] == 68:
            state = np.take(raw_state, ARX_R5_STATE_INDICES, axis=-1)
        elif raw_state.shape[-1] == 56:
            state = raw_state
        else:
            raise ValueError(
                f"ARX R5 state must be 68D (training) or 56D (inference), got {raw_state.shape[-1]}D"
            )

        # 2. Process Images
        in_images = data.get("images", {})
        parsed_images = {name: _parse_image(in_images[name]) for name in in_images}

        ref_shape = (240, 424, 3)
        for name in ("head", "left_wrist", "right_wrist"):
            if name in parsed_images:
                ref_shape = parsed_images[name].shape
                break

        images = {}
        image_masks = {}
        
        # Head camera -> base_0_rgb
        if "head" in parsed_images:
            images["base_0_rgb"] = parsed_images["head"]
            image_masks["base_0_rgb"] = np.True_
        else:
            images["base_0_rgb"] = np.zeros(ref_shape, dtype=np.uint8)
            image_masks["base_0_rgb"] = np.False_

        # Wrist cameras
        wrist_mapping = {"left_wrist_0_rgb": "left_wrist", "right_wrist_0_rgb": "right_wrist"}
        for dest, source in wrist_mapping.items():
            if source in parsed_images:
                images[dest] = parsed_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros(ref_shape, dtype=np.uint8)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # 3. Process Actions (40D -> 28D)
        if "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            # Extract 28 core dimensions (Padding to 32D is handled by PadStatesAndActions)
            inputs["actions"] = np.take(raw_actions, ARX_R5_ACTION_INDICES, axis=-1)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs

@dataclasses.dataclass(frozen=True)
class ArxR5FullOutputs(transforms.DataTransformFn):
    """Convert 32D model output back to ARX 40D action format."""

    def __call__(self, data: dict) -> dict:
        # model_actions is [batch, 32] (automatically padded by the model)
        model_actions = np.asarray(data["actions"], dtype=np.float32)
        
        # Initialize 40D action with zeros
        actions = np.zeros((*model_actions.shape[:-1], 40), dtype=np.float32)
        
        # Map the first 28D of the 32D output back to 40D positions
        # 0-25: joints and tcp (from model 0-25)
        actions[..., :26] = model_actions[..., :26]
        # 26-37: delta_tcp (remains zero)
        # 38-39: grippers (from model 26-27)
        actions[..., 38] = model_actions[..., 26]
        actions[..., 39] = model_actions[..., 27]
        
        return {"actions": actions}


# === 2. Delta End-Effector Mode (14D Action, 14D State) ===
ARX_STATE_DIM = 14
ARX_ACTION_DIM = 14
# The ARX LeRobot recorder stores the full robot state as:
# left joints 6D, right joints 6D, left EE pose 6D, right EE pose 6D,
# left gripper state/cmd, right gripper state/cmd. The policy consumes only
# EE poses and gripper states.
ARX_FULL_STATE_TO_POLICY_INDICES = (*range(12, 24), 24, 26)

def make_arx_delta_ee_example() -> dict:
    """Creates a random input example for the ARX R5 dual-arm policy."""
    return {
        "observation/state": np.random.rand(ARX_STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }

def _parse_ee_state(state) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == ARX_STATE_DIM:
        return state
    if state.shape[-1] <= max(ARX_FULL_STATE_TO_POLICY_INDICES):
        raise ValueError(
            f"ARX state must be either {ARX_STATE_DIM}D policy state or full LeRobot state with at least "
            f"{max(ARX_FULL_STATE_TO_POLICY_INDICES) + 1} dims, got shape {state.shape}"
        )
    return np.take(state, ARX_FULL_STATE_TO_POLICY_INDICES, axis=-1)

@dataclasses.dataclass(frozen=True)
class ArxDeltaEEInputs(transforms.DataTransformFn):
    """Convert ARX EE observations into the common model input format."""

    model_type: _model.ModelType
    state_key: str = "observation/state"
    base_image_key: str = "observation/image"
    left_wrist_image_key: str = "observation/wrist_image"
    right_wrist_image_key: str = "observation/right_wrist_image"
    prompt_key: str = "prompt"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data[self.base_image_key])
        left_wrist_image = _parse_image(data[self.left_wrist_image_key])
        right_wrist_image = _parse_image(data[self.right_wrist_image_key])

        inputs = {
            "state": _parse_ee_state(data[self.state_key]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if self.prompt_key in data:
            prompt = data[self.prompt_key]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs

@dataclasses.dataclass(frozen=True)
class ArxDeltaEEOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the 14D ARX action format."""

    action_dim: int = ARX_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}
