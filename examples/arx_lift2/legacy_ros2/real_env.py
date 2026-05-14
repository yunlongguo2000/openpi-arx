# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))
sys.path.append(str(ROOT.parents[2]))  # src/learning/ for shared detr & robomimic

import argparse
import yaml

# from interbotix_xs_modules.arm import InterbotixManipulatorXS
# from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np

from examples.aloha_arx_lift_ros2_real import constants
from examples.aloha_arx_lift_ros2_real import robot_utils

from functools import partial
import signal
import rclpy


# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]


obs_dict = collections.OrderedDict()


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(
        self,
        init_node,
        *,
        reset_position: Optional[List[float]] = None,
        setup_robots: bool = True,
    ):
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = (
            reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        )
        
        # Initialize rclpy if needed
        if init_node and not rclpy.ok():
            rclpy.init()
        
        self.args = parse_args()
        self.data = load_yaml(self.args.data)
        self.ros_operator = robot_utils.RosOperator(self.args, self.data, in_collect=False)

        if self.args.use_base:
            signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=self.ros_operator))
        # self.puppet_bot_left = InterbotixManipulatorXS(
        #     robot_model="vx300s",
        #     group_name="arm",
        #     gripper_name="gripper",
        #     robot_name="puppet_left",
        #     init_node=init_node,
        # )
        # self.puppet_bot_right = InterbotixManipulatorXS(
        #     robot_model="vx300s",
        #     group_name="arm",
        #     gripper_name="gripper",
        #     robot_name="puppet_right",
        #     init_node=False,
        # )
        if setup_robots:
            self.setup_robots()

        # self.recorder_left = robot_utils.Recorder("left", init_node=False)
        # self.recorder_right = robot_utils.Recorder("right", init_node=False)
        # self.image_recorder = robot_utils.ImageRecorder(init_node=False)
        # self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        # robot_utils.setup_puppet_bot(self.puppet_bot_left)
        # robot_utils.setup_puppet_bot(self.puppet_bot_right)
        self.ros_operator.init_robot_base_pose()
    # def get_qpos(self):
    #     left_qpos_raw = self.recorder_left.qpos
    #     right_qpos_raw = self.recorder_right.qpos
    #     left_arm_qpos = left_qpos_raw[:6]
    #     right_arm_qpos = right_qpos_raw[:6]
    #     left_gripper_qpos = [
    #         constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
    #     ]  # this is position not joint
    #     right_gripper_qpos = [
    #         constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
    #     ]  # this is position not joint
    #     return np.concatenate(
    #         [left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos]
    #     )

    # def get_qvel(self):
    #     left_qvel_raw = self.recorder_left.qvel
    #     right_qvel_raw = self.recorder_right.qvel
    #     left_arm_qvel = left_qvel_raw[:6]
    #     right_arm_qvel = right_qvel_raw[:6]
    #     left_gripper_qvel = [
    #         constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])
    #     ]
    #     right_gripper_qvel = [
    #         constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])
    #     ]
    #     return np.concatenate(
    #         [left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel]
    #     )

    # def get_effort(self):
    #     left_effort_raw = self.recorder_left.effort
    #     right_effort_raw = self.recorder_right.effort
    #     left_robot_effort = left_effort_raw[:7]
    #     right_robot_effort = right_effort_raw[:7]
    #     return np.concatenate([left_robot_effort, right_robot_effort])

    # def get_images(self):
    #     return self.image_recorder.get_images()

    # def set_gripper_pose(
    #     self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized
    # ):
    #     left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    #         left_gripper_desired_pos_normalized
    #     )
    #     self.gripper_command.cmd = left_gripper_desired_joint
    #     self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

    #     right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    #         right_gripper_desired_pos_normalized
    #     )
    #     self.gripper_command.cmd = right_gripper_desired_joint
    #     self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    # def _reset_joints(self):
    #     robot_utils.move_arms(
    #         [self.puppet_bot_left, self.puppet_bot_right],
    #         [self._reset_position, self._reset_position],
    #         move_time=1,
    #     )

    # def _reset_gripper(self):
    #     """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
    #     robot_utils.move_grippers(
    #         [self.puppet_bot_left, self.puppet_bot_right],
    #         [constants.PUPPET_GRIPPER_JOINT_OPEN] * 2,
    #         move_time=0.5,
    #     )
    #     robot_utils.move_grippers(
    #         [self.puppet_bot_left, self.puppet_bot_right],
    #         [constants.PUPPET_GRIPPER_JOINT_CLOSE] * 2,
    #         move_time=1,
    #     )
        
    # def build_image_dict(self,img_front: np.ndarray,
    #                     img_left:  np.ndarray,
    #                     img_right: np.ndarray) -> dict[str, np.ndarray | None]:
    #     """将三路 RGB 帧封装成 ImageRecorder.get_images 同格式 dict。"""
    #     return {
    #         "cam_high":           img_front,
    #         "cam_high_depth":     None,          # 若无深度帧可置 None
    #         "cam_left_wrist":     img_left,
    #         "cam_left_wrist_depth":  None,
    #         "cam_right_wrist":    img_right,
    #         "cam_right_wrist_depth": None,
    #         }

    # def get_observation(self):
    #     obs = collections.OrderedDict()
    #     obs["qpos"] = self.get_qpos()
    #     obs["qvel"] = self.get_qvel()
    #     obs["effort"] = self.get_effort()
    #     obs["images"] = self.get_images()
    #     return obs

    def get_observation(self):
        global obs_dict

        rate = robot_utils.Rate(self.args.frame_rate)
        while True and rclpy.ok():
            obs_dict = self.ros_operator.get_observations()
            if not obs_dict:
                print("syn fail")
                rate.sleep()

                continue

            return obs_dict

    
    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            # self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            # self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            # self._reset_joints()
            # self._reset_gripper()

            init_robot(self.ros_operator, self.args.use_base)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action):
        # state_len = int(len(action) / 2)
        # left_action = action[:state_len]
        # right_action = action[state_len:]
        # self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        # self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
        # self.set_gripper_pose(left_action[-1], right_action[-1])
        robot_action(self.ros_operator, self.args, action)
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )



def make_real_env(
    init_node,
    *,
    reset_position: Optional[List[float]] = None,
    setup_robots: bool = True,
) -> RealEnv:
    return RealEnv(init_node, reset_position=reset_position, setup_robots=setup_robots)



def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')

    # 数据集和检查点设置
    parser.add_argument('--ckpt_dir', type=str, default=Path.joinpath(ROOT, 'weights'),
                        help='ckpt dir')
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt',
                        help='ckpt name')
    parser.add_argument('--pretrain_ckpt', type=str, default='',
                        help='pretrain ckpt')
    parser.add_argument('--ckpt_stats_name', type=str, default='dataset_stats.pkl',
                        help='ckpt stats name')

    # 配置文件
    parser.add_argument('--data', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 推理设置
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay rate')
    parser.add_argument('--loss_function', type=str, choices=['l1', 'l2', 'l1+l2'],
                        default='l1', help='loss function')
    parser.add_argument('--pos_lookahead_step', type=int, default=0, help='position lookahead step')

    # 模型结构设置
    parser.add_argument('--policy_class', type=str, choices=['CNNMLP', 'ACT', 'Diffusion'], default='ACT',
                        help='policy class selection')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone model architecture')
    parser.add_argument('--chunk_size', type=int, default=30, help='chunk size for input data')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden layer dimension size')

    # 摄像头和位置嵌入设置
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
                        help='camera names to use')
    parser.add_argument('--position_embedding', type=str, choices=('sine', 'learned'), default='sine',
                        help='type of positional embedding to use')
    parser.add_argument('--masks', action='store_true', help='train segmentation head if provided')
    parser.add_argument('--dilation', action='store_true',
                        help='replace stride with dilation in the last convolutional block (DC5)')

    # 机器人设置
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')

    # ACT模型专用设置
    parser.add_argument('--enc_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=7, help='number of decoder layers')
    parser.add_argument('--nheads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate in transformer layers')
    parser.add_argument('--pre_norm', action='store_true', help='use pre-normalization in transformer')
    parser.add_argument('--states_dim', type=int, default=14, help='state dimension size')
    parser.add_argument('--kl_weight', type=int, default=10, help='KL divergence weight')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='feedforward network dimension')
    parser.add_argument('--temporal_agg', type=bool, default=True, help='use temporal aggregation')

    # Diffusion模型专用设置
    parser.add_argument('--observation_horizon', type=int, default=1, help='observation horizon length')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon length')
    parser.add_argument('--num_inference_timesteps', type=int, default=10,
                        help='number of inference timesteps')
    parser.add_argument('--ema_power', type=int, default=0.75, help='EMA power for diffusion process')

    # 图像设置
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')

    # 状态和动作设置
    parser.add_argument('--use_qvel', action='store_true', help='include qvel in state information')
    parser.add_argument('--use_effort', action='store_true', help='include effort data in state')
    parser.add_argument('--use_eef_states', action='store_true', help='use eef data in state')

    parser.add_argument('--gripper_gate', type=float, default=-1, help='gripper gate threshold')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None
    
    
    
def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')

    # 底盘给零
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)
    
    
    
def init_robot(ros_operator, use_base):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    # 发布初始位置（关节空间姿态）
    ros_operator.follow_arm_publish_continuous(init0, init0)
    # ros_operator.robot_base_shutdown()
    input("Enter any key to continue :")

    ros_operator.follow_arm_publish_continuous(init1, init1)
    # if use_base:
    #     ros_operator.start_base_control_thread()


def robot_action(ros_operator, args, action):
    gripper_gate = args.gripper_gate
    max_gripper = 5

    gripper_idx = [6, 13]

    left_action = action[:gripper_idx[0] + 1]  # 取8维度
    if gripper_gate != -1:
        left_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

    right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
    if gripper_gate != -1:
        right_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

    ros_operator.follow_arm_publish(left_action, right_action)

    if args.use_base:
        action_base = action[gripper_idx[1] + 1:gripper_idx[1] + 1 + 10]
        ros_operator.set_robot_base_target(action_base)

def apply_gripper_gate(action_value, gate):
    return 0 if action_value < gate else action_value
