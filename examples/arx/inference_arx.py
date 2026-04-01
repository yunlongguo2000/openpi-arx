"""
ARX Pi0.5 推理主循环

在机器人端运行，负责:
  1. 从机器人获取观测 (59D state + 3张图像)
  2. 通过 WebSocket 发给 Policy Server
  3. 接收 action chunk [action_horizon, 32]
  4. 逐步执行动作到机器人

用法:
  cd /home/yunlong/ARX_new/openpi_arx
  python inference/inference_arx.py --config inference/cfg_arx_pi.yaml

前置条件:
  - Policy Server 已启动: python serve_policy.py --checkpoint_dir <path>
  - 机器人端 arx_ros2_rpc_server.py 已运行

通信方式: ZMQ + msgpack (ArxROS2RPCClient)

State 59D 拼接顺序 (与 info.json 一致):
  [0:7]   left_joint_pos      ← left_arm.joint_positions
  [7:14]  left_joint_vel      ← left_arm.joint_velocities
  [14:21] left_joint_cur      ← left_arm.joint_currents
  [21:28] right_joint_pos     ← right_arm.joint_positions
  [28:35] right_joint_vel     ← right_arm.joint_velocities
  [35:42] right_joint_cur     ← right_arm.joint_currents
  [42:48] left_tcp_pose       ← left_arm.end_pose
  [48:54] right_tcp_pose      ← right_arm.end_pose
  [54]    left_gripper         ← left_arm.gripper
  [55]    right_gripper        ← right_arm.gripper
  [56:59] chassis              ← chassis.height, chassis.head_yaw, chassis.head_pitch

Action 32D 解析顺序 (与 info.json 一致):
  [0:7]   left_joint_pos      → set_full_command(left_positions=...)
  [7:14]  right_joint_pos     → set_full_command(right_positions=...)
  [14:20] left_tcp_pose       → (暂不使用, 关节位置控制优先)
  [20:26] right_tcp_pose      → (暂不使用)
  [26]    left_gripper         → set_left_gripper()
  [27]    right_gripper        → set_right_gripper()
  [28:32] chassis (vx,vy,wz,h)→ set_full_command(vx, vy, wz, height)
"""

import argparse
import logging
import sys
import os
import time

import numpy as np
import yaml

# ArxROS2RPCClient 位于 ARX_new 工程根目录下的 ros2_bridge/
_ARX_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if _ARX_ROOT not in sys.path:
    sys.path.insert(0, _ARX_ROOT)

from openpi_client import websocket_client_policy
from ros2_bridge.arx_ros2_rpc_client import ArxROS2RPCClient

log = logging.getLogger(__name__)


def state_from_full_state(full_state: dict) -> np.ndarray:
    """将 RPC get_full_state() 返回值拼接为 59D state 向量。

    拼接顺序严格对应数据集 info.json 中 observation.state 的 names 定义。
    """
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


class ArxInference:
    """ARX Pi0.5 推理控制器"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.action_horizon = self.cfg["control"]["action_horizon"]
        self.control_freq = self.cfg["control"]["control_freq"]
        self.control_interval = 1.0 / self.control_freq
        self.task_instruction = self.cfg.get("task_instruction", "")

        # 连接 Policy Server (WebSocket)
        server_cfg = self.cfg["policy_server"]
        log.info(f"Connecting to Policy Server at {server_cfg['host']}:{server_cfg['port']}")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=server_cfg["host"],
            port=server_cfg["port"],
        )
        log.info("Policy Server connected")

        # 连接机器人 RPC (ZMQ + msgpack)
        robot_cfg = self.cfg["robot"]
        log.info(f"Connecting to robot RPC at {robot_cfg['ip']}:{robot_cfg['port']}")
        self.robot = ArxROS2RPCClient(ip=robot_cfg["ip"], port=robot_cfg["port"])

        # 连接硬件
        if not self.robot.system_connect(timeout=10.0):
            raise RuntimeError("Failed to connect to ARX robot system")
        log.info("Robot system connected")

        # TODO: 初始化相机 (当前用占位图像, 后续接入 RealSense)
        self.cameras_enabled = False

    def get_observation(self) -> dict:
        """从机器人获取当前观测，组装为 openpi 推理格式。

        Returns:
            dict with keys matching ArxInputs 期望的格式:
              - "state": np.ndarray[59]
              - "images": {"head": HWC uint8, "left_wrist": HWC uint8, "right_wrist": HWC uint8}
              - "prompt": str
        """
        # 获取完整机器人状态
        full_state = self.robot.get_full_state()
        if full_state is None:
            raise RuntimeError("Failed to get robot state")

        # 拼接 59D state 向量
        state_59d = state_from_full_state(full_state)

        # 获取相机图像
        if self.cameras_enabled:
            # TODO: 从 RealSense 获取图像
            # head_img = self.head_camera.read()
            # left_img = self.left_camera.read()
            # right_img = self.right_camera.read()
            pass
        else:
            # 占位黑图 — Policy Server 的 ArxInputs 会设置 image_mask=False
            head_img = np.zeros((240, 424, 3), dtype=np.uint8)
            left_img = np.zeros((240, 424, 3), dtype=np.uint8)
            right_img = np.zeros((240, 424, 3), dtype=np.uint8)

        obs = {
            "state": state_59d,
            "images": {
                "head": head_img,
                "left_wrist": left_img,
                "right_wrist": right_img,
            },
            "prompt": self.task_instruction,
        }
        return obs

    def execute_actions(self, actions: np.ndarray):
        """逐步执行 action chunk 到机器人。

        Args:
            actions: np.ndarray[action_horizon, 32]
        """
        for i, action in enumerate(actions[:self.action_horizon]):
            step_start = time.time()

            # 解析 32D action
            left_joints = action[0:7]
            right_joints = action[7:14]
            # left_tcp = action[14:20]    # 关节位置控制优先, TCP 暂不使用
            # right_tcp = action[20:26]
            left_gripper = float(action[26])
            right_gripper = float(action[27])
            chassis_vx = float(action[28])
            chassis_vy = float(action[29])
            chassis_wz = float(action[30])
            chassis_height = float(action[31])

            # 发送全部命令 (单次原子调用, server 端有安全限幅)
            self.robot.set_full_command(
                left_joints, right_joints,
                chassis_vx, chassis_vy, chassis_wz, chassis_height,
            )
            self.robot.set_left_gripper(left_gripper)
            self.robot.set_right_gripper(right_gripper)

            # 频率控制
            elapsed = time.time() - step_start
            sleep_time = self.control_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        """推理主循环"""
        log.info(f"Starting inference loop")
        log.info(f"  Task: '{self.task_instruction}'")
        log.info(f"  Action horizon: {self.action_horizon}")
        log.info(f"  Control freq: {self.control_freq} Hz")

        step_count = 0
        try:
            while True:
                loop_start = time.time()

                # 1. 获取观测
                obs = self.get_observation()

                # 2. 发送到 Policy Server 推理
                result = self.policy.infer(obs)
                actions = result["actions"]  # [action_horizon, 32]

                infer_time = time.time() - loop_start
                log.info(f"Step {step_count}: infer={infer_time*1000:.1f}ms, "
                         f"actions shape={actions.shape}")

                # 3. 逐步执行动作
                self.execute_actions(actions)

                # 4. 心跳保活
                self.robot.heartbeat()

                step_count += 1

        except KeyboardInterrupt:
            log.info("User interrupted")
        except Exception as e:
            log.error(f"Error during inference: {e}", exc_info=True)
        finally:
            log.info("Triggering emergency stop...")
            self.robot.emergency_stop()
            self.robot.disconnect()
            self.robot.close()
            log.info(f"Inference stopped after {step_count} steps")


def main():
    parser = argparse.ArgumentParser(description="ARX Pi0.5 Inference")
    parser.add_argument("--config", type=str, default="inference/cfg_arx_pi.yaml",
                        help="Path to deployment config YAML")
    args = parser.parse_args()

    inferencer = ArxInference(args.config)
    inferencer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
