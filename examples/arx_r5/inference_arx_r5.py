from __future__ import annotations

import argparse
import logging
import time
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

# ARX Bridge Path
_ARX_BRIDGE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "ros2_bridge",
))
if _ARX_BRIDGE_PATH not in sys.path:
    sys.path.insert(0, _ARX_BRIDGE_PATH)

from openpi.arx.arx_r5.arx_r5_robot_adapter import ArxR5RobotAdapter as ArxRobotAdapter, DummyCameraRig, RealSenseCameraRig
from openpi.arx.arx_ros2_rpc_client import ArxROS2RPCClient
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi_client import websocket_client_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class DummyArxPolicy:
    """Fixed-action policy for dry-runs."""

    def __init__(self, action_horizon: int, action_dim: int, control_mode: str):
        self.action_dim = action_dim
        self.control_mode = control_mode
        action = np.zeros((action_horizon, action_dim), dtype=np.float32)
        if self.control_mode == "delta_ee":
            action[:, 0] = 0.005
            action[:, 6] = -0.005
            action[:, 12] = 0.2
            action[:, 13] = 0.2
        self._action_chunk = action

    def infer(self, _obs: dict[str, object]) -> dict[str, np.ndarray]:
        return {"actions": self._action_chunk.copy()}


class RemotePolicyWrapper:
    """Wrapper for WebSocket policy server to match LocalPolicy interface."""

    def __init__(self, host: str, port: int):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self.client.infer(obs)


class ArxUnifiedInference:
    """Unified inference for ARX robots (R5 / LIFT2, full_joint / delta_ee).

    control_mode:
      - "full_joint": 32D model output, used by pi05_arx / pi05_arx_r5_bottle_handoff / pi05_arx_lora
      - "delta_ee":   14D model output, used by pi05_arx_delta_ee
    """

    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file)

        # Inference Type
        self.inference_type = self.cfg.get("inference_type", "remote")

        # Model & Train Config
        model_cfg = self.cfg["model"]
        self.model_name = model_cfg["name"]

        # Robot Config -> determines control mode
        robot_cfg = self.cfg["robot"]
        self.robot_ip = robot_cfg["ip"]
        self.robot_port = int(robot_cfg.get("port", 4242))
        self.run_mode = robot_cfg.get("mode", "execute")
        self.robot_type: Literal["arx_r5", "arx_lift"] = robot_cfg.get("robot_type", "arx_lift")
        if self.robot_type not in ("arx_r5", "arx_lift"):
            raise ValueError(f"Unsupported robot.robot_type: {self.robot_type}")

        self.control_mode = robot_cfg.get("control_mode", "full_joint")
        if self.control_mode not in ("full_joint", "delta_ee"):
            raise ValueError(f"Unsupported robot.control_mode: {self.control_mode}")

        self.action_dim = 14 if self.control_mode == "delta_ee" else 32

        # Validate model <-> control_mode consistency
        model_is_delta_ee = (self.model_name == "pi05_arx_delta_ee")
        if model_is_delta_ee and self.control_mode != "delta_ee":
            raise ValueError(
                f"Model {self.model_name} requires control_mode=delta_ee, got control_mode={self.control_mode}."
            )
        if not model_is_delta_ee and self.control_mode != "full_joint":
            raise ValueError(
                f"Model {self.model_name} requires control_mode=full_joint, got control_mode={self.control_mode}."
            )

        # Validate robot_type <-> control_mode
        if self.control_mode == "full_joint" and self.robot_type not in ("arx_lift", "arx_r5"):
            raise ValueError(
                f"full_joint control not supported for robot_type={self.robot_type}."
            )
        if self.control_mode == "delta_ee" and self.robot_type not in ("arx_lift", "arx_r5"):
            raise ValueError(
                f"delta_ee control not supported for robot_type={self.robot_type}."
            )
        if self.control_mode == "delta_ee" and self.robot_type == "arx_lift":
            log.warning("Using delta_ee control on arx_lift robot_type; chassis commands are not used in this mode.")

        # Control Params
        control_cfg = self.cfg["control"]
        self.action_fps = float(control_cfg.get("action_fps", 15))
        self.action_horizon = int(control_cfg.get("action_horizon", 16))
        self.max_steps = int(control_cfg.get("max_steps", 0))

        # Image Params
        image_cfg = self.cfg["image"]
        self.image_height = int(image_cfg.get("height", 240))
        self.image_width = int(image_cfg.get("width", 424))

        # Task
        self.task_description = self.cfg.get("task_description", "")

        # Initialize Hardware
        if self.run_mode == "mock":
            camera_rig = DummyCameraRig(width=self.image_width, height=self.image_height)
            rpc_client = ArxROS2RPCClient(ip=self.robot_ip, port=self.robot_port, autoconnect=False)
        else:
            camera_rig = DummyCameraRig(width=self.image_width, height=self.image_height)
            rpc_client = ArxROS2RPCClient(ip=self.robot_ip, port=self.robot_port)

        self.robot = ArxRobotAdapter(
            rpc_client=rpc_client,
            camera_rig=camera_rig,
            dry_run=(self.run_mode == "mock"),
        )
        log.info(
            "Resolved robot profile: type=%s, control_mode=%s, model=%s",
            self.robot_type,
            self.control_mode,
            self.model_name,
        )

    def _load_policy(self):
        if self.inference_type == "remote":
            server_cfg = self.cfg["policy_server"]
            log.info(f"Connecting to Remote Policy Server at {server_cfg['host']}:{server_cfg['port']}")
            return RemotePolicyWrapper(server_cfg['host'], server_cfg['port'])

        # Local model loading
        log.info(f"Loading Local Model: {self.model_name}")
        train_config = _config.get_config(self.model_name)
        checkpoint_dir = os.path.expanduser(self.cfg["model"]["checkpoint_dir"])
        norm_stats_dir = self.cfg["model"].get("norm_stats_dir")
        norm_stats = _normalize.load(Path(norm_stats_dir).expanduser()) if norm_stats_dir else None

        if not Path(checkpoint_dir).exists() and self.run_mode == "mock":
            log.warning(f"Checkpoint {checkpoint_dir} not found. Using dummy policy.")
            return DummyArxPolicy(self.action_horizon, self.action_dim, self.control_mode)

        return _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=self.task_description,
            norm_stats=norm_stats,
        )

    def run(self) -> None:
        policy = self._load_policy()
        self.robot.connect()
        log.info("Robot Connected")

        try:
            step = 0
            while True:
                started = time.perf_counter()

                obs = self.robot.read_policy_observation(
                    image_height=self.image_height,
                    image_width=self.image_width,
                    prompt=self.task_description,
                )

                result = policy.infer(obs)
                actions = np.asarray(result["actions"], dtype=np.float32)

                self.robot.apply_action_chunk(
                    obs["state"],
                    actions,
                    action_horizon=self.action_horizon,
                )

                log.info(f"Step {step} completed")

                step += 1
                if self.max_steps > 0 and step >= self.max_steps:
                    break

                elapsed = time.perf_counter() - started
                sleep_time = (1.0 / self.action_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self.robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Unified ARX Inference")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "cfg_arx_r5_pi.yaml")
    args = parser.parse_args()

    ArxUnifiedInference(args.config).run()


if __name__ == "__main__":
    main()
