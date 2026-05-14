#!/usr/bin/env python3
"""
LeRobot框架下Pi0.5部署脚本
完全兼容LeRobot标准接口，训练完成后直接使用
"""
import argparse
import logging
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# LeRobot导入
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.cameras.realsense import RealSenseCamera
from lerobot.configs.policies import PreTrainedConfig

# 导入我们的适配器
from ros2_bridge.pi05_action_adapter import Pi05ActionAdapter

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Pi0.5 LeRobot Deployment")
    parser.add_argument("--model-path", type=str, required=True, help="Trained Pi0.5 model directory")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.100", help="ARX robot RPC server IP")
    parser.add_argument("--robot-port", type=int, default=4242, help="ARX robot RPC server port")
    parser.add_argument("--instruction", type=str, required=True, help="Task instruction for Pi0.5")
    parser.add_argument("--control-freq", type=int, default=10, help="Control frequency (Hz)")
    return parser.parse_args()


def main():
    args = parse_args()
    control_interval = 1.0 / args.control_freq

    # -------------------------------------------------------------------------
    # 1. 加载训练好的Pi0.5模型（LeRobot标准接口）
    # -------------------------------------------------------------------------
    log.info(f"Loading Pi0.5 model from {args.model_path}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.model_path)
    policy = make_policy(policy_cfg, pretrained_path=args.model_path)
    policy.eval()
    log.info("Model loaded successfully")

    # -------------------------------------------------------------------------
    # 2. 初始化硬件：摄像头 + RPC适配器
    # -------------------------------------------------------------------------
    log.info("Initializing cameras")
    cameras = {
        "front": RealSenseCamera(serial_number="your_front_camera_serial", width=640, height=480, fps=30),
        "left_wrist": RealSenseCamera(serial_number="your_left_camera_serial", width=640, height=480, fps=30),
        "right_wrist": RealSenseCamera(serial_number="your_right_camera_serial", width=640, height=480, fps=30),
    }

    # 从训练配置中加载归一化参数
    normalization = policy.config.dataset_config.normalization
    action_mean = np.array(normalization["action_mean"])
    action_std = np.array(normalization["action_std"])

    log.info(f"Connecting to robot at {args.robot_ip}:{args.robot_port}")
    with Pi05ActionAdapter(
        rpc_ip=args.robot_ip,
        rpc_port=args.robot_port,
        action_mean=action_mean,
        action_std=action_std
    ) as adapter:
        log.info("Robot connected successfully")
        log.info(f"Task instruction: {args.instruction}")

        # -------------------------------------------------------------------------
        # 3. 推理控制主循环
        # -------------------------------------------------------------------------
        import time
        try:
            while True:
                loop_start = time.time()

                # 采集相机图像
                images = {}
                for cam_name, cam in cameras.items():
                    img = cam.read()
                    if img is None:
                        log.warning(f"Failed to read from {cam_name} camera")
                        continue
                    images[f"observation.image.{cam_name}"] = img

                # 采集机器人状态
                state = adapter.get_state_observation()
                if state is None:
                    log.error("Failed to get robot state, stopping")
                    break

                # 组装观测输入（LeRobot标准格式）
                obs = {
                    **images,
                    "observation.state": state,
                    "task": args.instruction,
                }

                # 模型推理生成动作
                action = policy.select_action(obs)

                # 发送动作到机器人
                if not adapter.send_action(action):
                    log.error("Failed to send action, stopping")
                    break

                # 控制频率校准
                loop_time = time.time() - loop_start
                if loop_time < control_interval:
                    time.sleep(control_interval - loop_time)

        except KeyboardInterrupt:
            log.info("User interrupted, stopping")
        except Exception as e:
            log.error(f"Error during execution: {e}", exc_info=True)
        finally:
            # 停止所有相机
            for cam in cameras.values():
                cam.stop()
            log.info("Cameras stopped")


if __name__ == "__main__":
    main()
