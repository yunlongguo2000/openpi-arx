from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from openpi_client import image_tools

from openpi.arx.arx_ros2_rpc_client import ArxROS2RPCClient

log = logging.getLogger(__name__)

CAMERA_KEYS = ("head_image", "left_wrist_image", "right_wrist_image")

@dataclasses.dataclass(frozen=True)
class DualArmCommandResult:
    command: Any
    ack: dict[str, Any]

def state_59d_from_full_state(full_state: dict) -> np.ndarray:
    """Convert an ARX RPC state dict into the 59D OpenPI ARX state vector (Full Joint + Chassis)."""
    la = full_state["left_arm"]
    ra = full_state["right_arm"]
    ch = full_state["chassis"]

    return np.concatenate([
        la["joint_positions"],          # [0:7]   left pos
        la["joint_velocities"],         # [7:14]  left vel
        la["joint_currents"],           # [14:21] left cur
        ra["joint_positions"],          # [21:28] right pos
        ra["joint_velocities"],         # [28:35] right vel
        ra["joint_currents"],           # [35:42] right cur
        la["end_pose"],                 # [42:48] left tcp
        ra["end_pose"],                 # [48:54] right tcp
        [la["gripper"]],                # [54]    left gripper
        [ra["gripper"]],                # [55]    right gripper
        [ch["height"],                  # [56]    chassis height
         ch["head_yaw"],               # [57]    chassis head yaw
         ch["head_pitch"]],            # [58]    chassis head pitch
    ]).astype(np.float32)

def make_arx_lift_observation(
    state: np.ndarray,
    images: Mapping[str, np.ndarray],
    prompt: str,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Pack ARX LIFT2 59D state and 3 RGB streams into the observation schema."""
    obs = {
        "state": np.asarray(state, dtype=np.float32),
        "images": {
            "head": images["head_image"],
            "left_wrist": images["left_wrist_image"],
            "right_wrist": images["right_wrist_image"],
        },
        "prompt": prompt,
    }
    return obs

class ArxLiftRobotAdapter:
    """Specialized ARX LIFT2 inference adapter: dual-arm + chassis, 32D actions, 59D state."""

    def __init__(
        self,
        *,
        rpc_client: ArxROS2RPCClient | None = None,
        camera_rig: Any | None = None,
        dry_run: bool = False,
    ):
        self._client = rpc_client
        self._camera_rig = camera_rig
        self._dry_run = dry_run

    def connect(self) -> None:
        if self._camera_rig is not None:
            self._camera_rig.connect()
        if not self._dry_run and self._client is not None:
            if not self._client.system_connect():
                raise RuntimeError("Failed to connect to the ARX RPC server")

    def disconnect(self) -> None:
        if self._camera_rig is not None:
            self._camera_rig.disconnect()
        if self._client is not None:
            self._client.disconnect() if not self._dry_run else self._client.close()

    def get_full_state(self) -> dict[str, Any]:
        if self._dry_run:
            return {
                "left_arm": {"joint_positions": np.zeros(7), "joint_velocities": np.zeros(7), "joint_currents": np.zeros(7), "end_pose": np.zeros(6), "gripper": 0.0},
                "right_arm": {"joint_positions": np.zeros(7), "joint_velocities": np.zeros(7), "joint_currents": np.zeros(7), "end_pose": np.zeros(6), "gripper": 0.0},
                "chassis": {"height": 0.0, "head_yaw": 0.0, "head_pitch": 0.0}
            }
        return self._client.get_full_state()

    def read_policy_observation(self, *, image_height: int, image_width: int, prompt: str) -> dict[str, Any]:
        full_state = self.get_full_state()
        images = self._camera_rig.read()
        state = state_59d_from_full_state(full_state)
        return make_arx_lift_observation(state, images, prompt, height=image_height, width=image_width)

    def apply_action_chunk(self, actions: np.ndarray, *, action_horizon: int) -> list[DualArmCommandResult]:
        """Apply 32D action targets to the LIFT2 robot (including chassis/lift)."""
        results = []
        for i in range(action_horizon):
            action = actions[i]
            left_joints = action[0:7]
            right_joints = action[7:14]
            left_gripper = float(action[26])
            right_gripper = float(action[27])
            chassis_vx = float(action[28])
            chassis_vy = float(action[29])
            chassis_wz = float(action[30])
            chassis_height = float(action[31])

            if not self._dry_run and self._client is not None:
                self._client.set_full_command(left_joints, right_joints, chassis_vx, chassis_vy, chassis_wz, chassis_height)
                self._client.set_left_gripper(left_gripper)
                self._client.set_right_gripper(right_gripper)
            
            results.append(DualArmCommandResult(command=action, ack={"accepted": True}))
        return results
