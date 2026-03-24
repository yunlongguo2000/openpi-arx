"""
ARX LIFT2 双臂移动机器人 Pi0.5 Policy Transform

将 ARX 机器人的观测/动作格式转换为 openpi 模型的标准格式。

ARX 数据维度定义 (来自 LeRobot 数据集 info.json):
  Action (32D):
    [0:7]   left_joint_pos (7)   - 左臂6关节+夹爪关节
    [7:14]  right_joint_pos (7)  - 右臂6关节+夹爪关节
    [14:20] left_tcp_pose (6)    - 左TCP位姿 (x,y,z,roll,pitch,yaw)
    [20:26] right_tcp_pose (6)   - 右TCP位姿
    [26]    left_gripper (1)     - 左夹爪位置
    [27]    right_gripper (1)    - 右夹爪位置
    [28:32] chassis (4)          - 底盘 (vx, vy, wz, height)

  State (59D):
    [0:7]   left_joint_pos (7)
    [7:14]  left_joint_vel (7)
    [14:21] left_joint_cur (7)
    [21:28] right_joint_pos (7)
    [28:35] right_joint_vel (7)
    [35:42] right_joint_cur (7)
    [42:48] left_tcp_pose (6)
    [48:54] right_tcp_pose (6)
    [54]    left_gripper (1)
    [55]    right_gripper (1)
    [56:59] chassis (3)          - (height, head_yaw, head_pitch)

  Images:
    head_image:        (240, 424, 3) video/av1
    left_wrist_image:  (240, 424, 3) video/av1
    right_wrist_image: (240, 424, 3) video/av1
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_arx_example() -> dict:
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


def _parse_image(img: np.ndarray) -> np.ndarray:
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
class ArxInputs(transforms.DataTransformFn):
    """Convert ARX observations to openpi model input format.

    Expected inputs (from inference environment or dataset after repack):
      - state: np.ndarray[59]
      - images: dict with 'head', 'left_wrist' and/or 'right_wrist'
      - actions: np.ndarray[action_horizon, 32] (training only)
      - prompt: str (language instruction)
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        # Parse state
        state = np.asarray(data["state"], dtype=np.float32)

        # Parse images
        in_images = data.get("images", {})
        parsed_images = {name: _parse_image(in_images[name]) for name in in_images}

        # Determine reference shape from any available camera
        ref_shape = (240, 424, 3)
        for name in ("head", "left_wrist", "right_wrist"):
            if name in parsed_images:
                ref_shape = parsed_images[name].shape
                break

        # Map head camera to base_0_rgb (exterior view)
        if "head" in parsed_images:
            images = {"base_0_rgb": parsed_images["head"]}
            image_masks = {"base_0_rgb": np.True_}
        else:
            images = {"base_0_rgb": np.zeros(ref_shape, dtype=np.uint8)}
            image_masks = {"base_0_rgb": np.False_}

        # Map wrist cameras to model slots
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

        # Actions are only available during training
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Convert model output actions to ARX 32D action format.

    Model outputs [action_horizon, padded_action_dim], we take the first 32 dims.
    """

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :self.action_dim])
        return {"actions": actions}
