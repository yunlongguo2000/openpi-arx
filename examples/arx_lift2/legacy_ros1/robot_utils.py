

import collections


# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import datetime
import json
import time

# from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
# from interbotix_xs_msgs.msg import JointGroupCommand
# from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import yaml
import rospy



import os
import threading

import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation as R  # eef:ZXY

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, Imu
from tf.msg import tfMessage
from tf.transformations import euler_from_quaternion

from examples.aloha_arx_lift_real.msg._JointControl import JointControl
from examples.aloha_arx_lift_real.msg._JointInformation import JointInformation
from examples.aloha_arx_lift_real.msg._PosCmd import PosCmd

from examples.aloha_arx_lift_real.utils.controller import PIDController
from sensor_msgs.msg import JointState

from examples.aloha_arx_lift_real import constants


# class ImageRecorder:
#     def __init__(self, init_node=True, is_debug=False):
#         self.is_debug = is_debug
#         self.bridge = CvBridge()
#         self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

#         if init_node:
#             rospy.init_node("image_recorder", anonymous=True)
#         for cam_name in self.camera_names:
#             setattr(self, f"{cam_name}_rgb_image", None)
#             setattr(self, f"{cam_name}_depth_image", None)
#             setattr(self, f"{cam_name}_timestamp", 0.0)
#             if cam_name == "cam_high":
#                 callback_func = self.image_cb_cam_high
#             elif cam_name == "cam_low":
#                 callback_func = self.image_cb_cam_low
#             elif cam_name == "cam_left_wrist":
#                 callback_func = self.image_cb_cam_left_wrist
#             elif cam_name == "cam_right_wrist":
#                 callback_func = self.image_cb_cam_right_wrist
#             else:
#                 raise NotImplementedError
#             rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
#             if self.is_debug:
#                 setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

#         self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}
#         time.sleep(0.5)

#     def image_cb(self, cam_name, data):
#         setattr(
#             self,
#             f"{cam_name}_rgb_image",
#             self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="bgr8"),
#         )
#         # setattr(
#         #     self,
#         #     f"{cam_name}_depth_image",
#         #     self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
#         # )
#         setattr(
#             self,
#             f"{cam_name}_timestamp",
#             data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
#         )
#         # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
#         # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
#         # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
#         if self.is_debug:
#             getattr(self, f"{cam_name}_timestamps").append(
#                 data.images[0].header.stamp.secs
#                 + data.images[0].header.stamp.nsecs * 1e-9
#             )

#     def image_cb_cam_high(self, data):
#         cam_name = "cam_high"
#         return self.image_cb(cam_name, data)

#     def image_cb_cam_low(self, data):
#         cam_name = "cam_low"
#         return self.image_cb(cam_name, data)

#     def image_cb_cam_left_wrist(self, data):
#         cam_name = "cam_left_wrist"
#         return self.image_cb(cam_name, data)

#     def image_cb_cam_right_wrist(self, data):
#         cam_name = "cam_right_wrist"
#         return self.image_cb(cam_name, data)

#     def get_images(self):
#         image_dict = {}
#         for cam_name in self.camera_names:
#             while (
#                 getattr(self, f"{cam_name}_timestamp")
#                 <= self.cam_last_timestamps[cam_name]
#             ):
#                 time.sleep(0.00001)
#             rgb_image = getattr(self, f"{cam_name}_rgb_image")
#             depth_image = getattr(self, f"{cam_name}_depth_image")
#             self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
#             image_dict[cam_name] = rgb_image
#             image_dict[f"{cam_name}_depth"] = depth_image
#         return image_dict

#     def print_diagnostics(self):
#         def dt_helper(l):
#             l = np.array(l)
#             diff = l[1:] - l[:-1]
#             return np.mean(diff)

#         for cam_name in self.camera_names:
#             image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
#             print(f"{cam_name} {image_freq=:.2f}")
#         print()


# class Recorder:
#     def __init__(self, side, init_node=True, is_debug=False):
#         self.secs = None
#         self.nsecs = None
#         self.qpos = None
#         self.effort = None
#         self.arm_command = None
#         self.gripper_command = None
#         self.is_debug = is_debug

#         if init_node:
#             rospy.init_node("recorder", anonymous=True)
#         rospy.Subscriber(
#             f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb
#         )
#         rospy.Subscriber(
#             f"/puppet_{side}/commands/joint_group",
#             JointGroupCommand,
#             self.puppet_arm_commands_cb,
#         )
#         rospy.Subscriber(
#             f"/puppet_{side}/commands/joint_single",
#             JointSingleCommand,
#             self.puppet_gripper_commands_cb,
#         )
#         if self.is_debug:
#             self.joint_timestamps = deque(maxlen=50)
#             self.arm_command_timestamps = deque(maxlen=50)
#             self.gripper_command_timestamps = deque(maxlen=50)
#         time.sleep(0.1)

#     def puppet_state_cb(self, data):
#         self.qpos = data.position
#         self.qvel = data.velocity
#         self.effort = data.effort
#         self.data = data
#         if self.is_debug:
#             self.joint_timestamps.append(time.time())

#     def puppet_arm_commands_cb(self, data):
#         self.arm_command = data.cmd
#         if self.is_debug:
#             self.arm_command_timestamps.append(time.time())

#     def puppet_gripper_commands_cb(self, data):
#         self.gripper_command = data.cmd
#         if self.is_debug:
#             self.gripper_command_timestamps.append(time.time())

#     def print_diagnostics(self):
#         def dt_helper(l):
#             l = np.array(l)
#             diff = l[1:] - l[:-1]
#             return np.mean(diff)

#         joint_freq = 1 / dt_helper(self.joint_timestamps)
#         arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
#         gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

#         print(
#             f"{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n"
#         )


# def get_arm_joint_positions(bot):
#     return bot.arm.core.joint_states.position[:6]


# def get_arm_gripper_positions(bot):
#     return bot.gripper.core.joint_states.position[6]


# def move_arms(bot_list, target_pose_list, move_time=1):
#     num_steps = int(move_time / constants.DT)
#     curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
#     traj_list = [
#         np.linspace(curr_pose, target_pose, num_steps)
#         for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
#     ]
#     for t in range(num_steps):
#         for bot_id, bot in enumerate(bot_list):
#             bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
#         time.sleep(constants.DT)


# def move_grippers(bot_list, target_pose_list, move_time):
#     print(f"Moving grippers to {target_pose_list=}")
#     gripper_command = JointSingleCommand(name="gripper")
#     num_steps = int(move_time / constants.DT)
#     curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
#     traj_list = [
#         np.linspace(curr_pose, target_pose, num_steps)
#         for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
#     ]

#     with open(
#         f"/data/gripper_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
#         "a",
#     ) as f:
#         for t in range(num_steps):
#             d = {}
#             for bot_id, bot in enumerate(bot_list):
#                 gripper_command.cmd = traj_list[bot_id][t]
#                 bot.gripper.core.pub_single.publish(gripper_command)
#                 d[bot_id] = {
#                     "obs": get_arm_gripper_positions(bot),
#                     "act": traj_list[bot_id][t],
#                 }
#             f.write(json.dumps(d) + "\n")
#             time.sleep(constants.DT)


# def setup_puppet_bot(bot):
#     bot.dxl.robot_reboot_motors("single", "gripper", True)
#     bot.dxl.robot_set_operating_modes("group", "arm", "position")
#     bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
#     torque_on(bot)


# def setup_master_bot(bot):
#     bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
#     bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
#     torque_off(bot)


# def set_standard_pid_gains(bot):
#     bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 800)
#     bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


# def set_low_pid_gains(bot):
#     bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 100)
#     bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


# def torque_off(bot):
#     bot.dxl.robot_torque_enable("group", "arm", False)
#     bot.dxl.robot_torque_enable("single", "gripper", False)


# def torque_on(bot):
#     bot.dxl.robot_torque_enable("group", "arm", True)
#     bot.dxl.robot_torque_enable("single", "gripper", True)


# for DAgger
# def sync_puppet_to_master(
#     master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right
# ):
#     print("\nSyncing!")

#     # activate master arms
#     torque_on(master_bot_left)
#     torque_on(master_bot_right)

#     # get puppet arm positions
#     puppet_left_qpos = get_arm_joint_positions(puppet_bot_left)
#     puppet_right_qpos = get_arm_joint_positions(puppet_bot_right)

#     # get puppet gripper positions
#     puppet_left_gripper = get_arm_gripper_positions(puppet_bot_left)
#     puppet_right_gripper = get_arm_gripper_positions(puppet_bot_right)

#     # move master arms to puppet positions
#     move_arms(
#         [master_bot_left, master_bot_right],
#         [puppet_left_qpos, puppet_right_qpos],
#         move_time=1,
#     )

#     # move master grippers to puppet positions
#     move_grippers(
#         [master_bot_left, master_bot_right],
#         [puppet_left_gripper, puppet_right_gripper],
#         move_time=1,
#     )
class RosOperator:
    def __init__(self, args, config, in_collect=False):
        rospy.init_node('robot_operator', anonymous=True)

        self.args = args
        self.config = config

        self.in_collect = in_collect

        self.base_enable = False
        self.robot_base_pose_init = [0, 0, 0]  # rlative, the head_pitch and height and head yaw is the adsolutly
        self.robot_base_target = np.zeros((6,))
        self.base_control_thread = None

        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()

        self.bridge = CvBridge()

        self.img_head_deque = deque()
        self.img_left_deque = deque()
        self.img_right_deque = deque()

        self.img_head_depth_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()

        self.master_arm_right_deque = deque()
        self.master_arm_left_deque = deque()
        self.follow_arm_left_deque = deque()
        self.follow_arm_right_deque = deque()

        self.base_pose_deque = deque()
        self.robot_base_origin = deque()

        # eef
        self.follow_arm_right_eef_deque = deque()
        self.follow_arm_left_eef_deque = deque()
        self.robot_base_deque = deque()

        self.master_arm_right_eef_deque = deque()
        self.master_arm_left_eef_deque = deque()

        self.master_VR_left_deque = deque()
        self.master_VR_right_deque = deque()
        self.master_VR_left_eef_deque = deque()
        self.master_VR_right_eef_deque = deque()

        self.follow_arm_publish_lock = threading.Lock()
        self.follow_arm_publish_lock.acquire()

        image_type = 'compress_image' if self.args.is_compress and self.in_collect else 'original_image'
        callback_type = CompressedImage if self.args.is_compress and self.in_collect else Image

        if in_collect:
            joint_topic_type = JointInformation
        else:
            joint_topic_type = JointControl

        # 摄像头订阅
        img_topics = {
            'img_head': 'img_head_topic',
            'img_left': 'img_left_topic',
            'img_right': 'img_right_topic',
        }
        for key, topic in img_topics.items():
            rospy.Subscriber(self.config['camera_config'][image_type][topic],
                             callback_type, getattr(self, f"{key}_callback"),
                             queue_size=2, tcp_nodelay=True)

        if self.args.use_depth_image:
            depth_img_topics = {
                'img_head_depth': 'img_head_depth_topic',
                'img_left_depth': 'img_left_depth_topic',
                'img_right_depth': 'img_right_depth_topic',
            }
            for key, topic in depth_img_topics.items():
                rospy.Subscriber(self.config['camera_config'][image_type][topic],
                                 callback_type, getattr(self, f"{key}_callback"),
                                 queue_size=2, tcp_nodelay=True)

        # 机械臂订阅
        arm_topics = {
            'follow_arm_left': ('follow_arm_left_topic', joint_topic_type),
            'follow_arm_right': ('follow_arm_right_topic', joint_topic_type),
            'follow_arm_left_eef': ('follow_arm_left_eef_topic', PosCmd),
            'follow_arm_right_eef': ('follow_arm_right_eef_topic', PosCmd),
            'master_VR_left_eef': ('master_VR_left_eef_topic', PosCmd),
            'master_VR_right_eef': ('master_VR_right_eef_topic', PosCmd)
        }
        for key, (topic_key, msg_type) in arm_topics.items():
            rospy.Subscriber(self.config['arm_config'][topic_key],
                             msg_type, getattr(self, f"{key}_callback"),
                             queue_size=2, tcp_nodelay=True)

        # 底盘订阅
        if self.args.use_base:
            rospy.Subscriber(self.config['robot_base_config']['robot_base_topic'],
                             PosCmd, self.robot_base_callback, queue_size=2, tcp_nodelay=True)
            rospy.Subscriber('/tf', tfMessage, self.base_pose_callback, queue_size=2, tcp_nodelay=True)

        # 采集模式相关订阅
        if self.in_collect:
            collect_topics = {
                'master_arm_left_eef': 'master_arm_left_eef_topic',
                'master_arm_right_eef': 'master_arm_right_eef_topic'
            }
            for key, topic in collect_topics.items():
                rospy.Subscriber(self.config['arm_config'][topic],
                                 PosCmd, getattr(self, f"{key}_callback"), queue_size=2, tcp_nodelay=True)
        # 推理模式相关发布
        else:
            self.follow_arm_left_publisher = rospy.Publisher(
                self.config['arm_config']['follow_arm_left_cmd_topic'], joint_topic_type, queue_size=10)
            self.follow_arm_right_publisher = rospy.Publisher(
                self.config['arm_config']['follow_arm_right_cmd_topic'], joint_topic_type, queue_size=10)
            self.base_robot_publisher = rospy.Publisher(
                self.config['robot_base_config']['robot_base_topic'], PosCmd, queue_size=10)

    # 推理
    def follow_arm_publish(self, left, right):
        joint_state_msg = JointControl()

        joint_state_msg.joint_pos = left
        self.follow_arm_left_publisher.publish(joint_state_msg)  # /joint_control
        if len(right) != 0:
            joint_state_msg.joint_pos = right
            self.follow_arm_right_publisher.publish(joint_state_msg)  # /joint_control2

    def init_robot_base_pose(self):
        if len(self.robot_base_origin) == 0:
            print(r'there is no base_pose_deque')

            return None
        base_pose = self.robot_base_origin.pop()
        tf_info = base_pose.transforms[0].transform
        base_quaternion = [tf_info.rotation.x, tf_info.rotation.y,
                           tf_info.rotation.z, tf_info.rotation.w]
        r = R.from_quat(base_quaternion)
        _, _, base_pose_yaw = r.as_euler('xyz', degrees=False)
        base_pose = [tf_info.translation.x, -tf_info.translation.y, base_pose_yaw]
        self.robot_base_pose_init = base_pose

        self.robot_base_target = np.zeros((6,))

        return True

    def set_robot_base_target(self, target_base):
        self.robot_base_target[0] = target_base[0]  # x
        self.robot_base_target[1] = target_base[1]  # y
        self.robot_base_target[2] = target_base[2]  # Wz
        self.robot_base_target[3] = target_base[3]  # height
        self.robot_base_target[4] = target_base[4]  # head_pitch
        self.robot_base_target[5] = target_base[5]  # head_yaw

    def start_base_control_thread(self):
        if self.args.base:
            self.init_robot_base_pose()
            self.base_enable = True
            self.base_control_thread = threading.Thread(target=self.robot_base_control_thread,
                                                        args=())  # 执行指令单独的线程,，可以边说话边执行，多线程操作
            self.base_control_thread.start()

            return

    def visualize_pid_base(self, states, target, plot_path=None):
        STATE_NAMES = ["DX", "DY", "Yaw"]
        label1, label2 = 'states', 'target'
        states = np.array(states)
        target = np.array(target)

        num_ts, num_dim = states.shape
        fig, axs = plt.subplots(num_dim, 1, figsize=(8, 2 * num_dim))

        all_names = [f"{name}_left" for name in STATE_NAMES] + [f"{name}_right" for name in STATE_NAMES]

        for dim_idx, ax in enumerate(axs):
            ax.plot(states[:, dim_idx], label=label1, color='orangered')
            ax.plot(target[:, dim_idx], label=label2)
            ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
            ax.legend()

        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path)
            print(f'Saved pid control plot to: {plot_path}')
        else:
            plt.show()

        plt.close()

    def robot_base_shutdown(self):
        rate = rospy.Rate(self.args.frame_rate)

        shutdown_control = PosCmd()
        shutdown_control.height = self.robot_base_target[3]

        for mode in [1, 2]:
            shutdown_control.mode1 = mode
            self.base_robot_publisher.publish(shutdown_control)

            rate.sleep()

        self.base_enable = False

        return

    def robot_base_control_thread(self):  # inference init robot arm in qpos
        rate = rospy.Rate(self.args.frame_rate)
        control = PosCmd()
        max_velocity = 1.0

        pid_controllers = {
            'x': PIDController(kp=10.0, ki=0.0, kd=0.0, max_i=1.0, max_output=max_velocity),
            'y': PIDController(kp=10.0, ki=0.0, kd=0.0, max_i=1.0, max_output=max_velocity),
            'z': PIDController(kp=1.0, ki=0.0, kd=0.0, max_i=1.0, max_output=max_velocity)
        }

        recorded_base_poses = []
        recorded_target_poses = []
        recorded_control_outputs = []
        timeout = 0

        while (not rospy.is_shutdown()) and self.base_enable:
            if len(self.base_pose_deque) == 0:
                print('\033[33mThere is no base_pose_deque\033[0m')

                timeout += 1
                if timeout > 100:
                    self.base_enable = False
                    break
                rate.sleep()

                continue

            base_pose = self.base_pose_deque.pop()
            current_x, current_y, current_Wz = base_pose
            target_x, target_y, target_Wz, target_height, target_pitch, target_yaw = self.robot_base_target

            # 更新控制命令
            control.chx = pid_controllers['x'].update(current_x, target_x, dt=0.017)
            control.chy = pid_controllers['y'].update(current_y, target_y, dt=0.017)
            control.chz = pid_controllers['z'].update(current_Wz, target_Wz, dt=0.017)
            control.height = target_height
            control.head_pit = target_pitch
            control.head_yaw = target_yaw
            control.mode1 = 1

            # 记录数据
            target_pose = [target_x, target_y, current_Wz]
            output_control = [control.chx, control.chy, control.chz]

            recorded_base_poses.append(base_pose)
            recorded_target_poses.append(target_pose)
            recorded_control_outputs.append(output_control)

            self.base_robot_publisher.publish(control)
            rate.sleep()

        if not self.base_enable:
            self.robot_base_shutdown()

            plot_path = (
                os.path.join(self.args.ckpt_dir, f"{self.args.ckpt_name}_PID.png")
                if self.args.episode_path == "./datasets"
                else os.path.join(f"{self.args.episode_path}_PID.png")
            )
            self.visualize_pid_base(recorded_base_poses, recorded_target_poses, plot_path=plot_path)

        return

    def follow_arm_publish_continuous(self, left_target, right_target):
        arm_steps_length = [0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.2]
        left_arm = None
        right_arm = None

        rate = rospy.Rate(self.args.frame_rate)
        while not rospy.is_shutdown():
            if len(self.follow_arm_left_deque) != 0:
                left_arm = list(self.follow_arm_left_deque[-1].joint_pos)

            if len(self.follow_arm_right_deque) != 0:
                right_arm = list(self.follow_arm_right_deque[-1].joint_pos)

            if left_arm is not None and right_arm is not None:
                break

        # 计算方向标志位
        left_symbol = [1 if left_target[i] - left_arm[i] > 0 else -1 for i in range(len(left_target))]
        if right_arm:
            right_symbol = [1 if right_target[i] - right_arm[i] > 0 else -1 for i in
                            range(len(right_target))] if right_arm else None

        step = 0
        while not rospy.is_shutdown():
            left_done = 0
            right_done = 0

            if self.follow_arm_publish_lock.acquire(False):
                return

            left_done = self._update_arm_position(left_target, left_arm, left_symbol, arm_steps_length)

            if right_arm:
                right_done = self._update_arm_position(right_target, right_arm, right_symbol, arm_steps_length)

            if right_arm:
                if left_done > len(left_target) - 1 and right_done > len(right_target) - 1:
                    print('left_done and right_done')

                    break
            elif left_done > len(left_target) - 1:
                break

            # JointControl topic
            joint_state_msg = JointControl()

            joint_state_msg.joint_pos = left_arm
            self.follow_arm_left_publisher.publish(joint_state_msg)
            rate.sleep()

            if right_arm:
                joint_state_msg.joint_pos = right_arm
                self.follow_arm_right_publisher.publish(joint_state_msg)

            step += 1
            print("follow_arm_publish_continuous:", step)
            rate.sleep()

    def get_observation(self, ts=-1):  # get the robot observation
        img_data = {
            'cam_high': None,
            'cam_left_wrist': None,
            'cam_right_wrist': None,
        }
        img_depth_data = {
            'cam_high': None,
            'cam_left_wrist': None,
            'cam_right_wrist': None,
        }
        arm_data = {
            'follow_arm_left': None,
            'follow_arm_right': None,
            'follow_arm_left_eef': None,
            'follow_arm_right_eef': None,
        }

        # 获取图像信息
        for cam_name in self.args.camera_names:
            if cam_name in img_data:
                deque_map = {
                    'cam_high': self.img_head_deque,
                    'cam_left_wrist': self.img_left_deque,
                    'cam_right_wrist': self.img_right_deque,
                }
                if len(deque_map[cam_name]) == 0:
                    print(f'there is no {cam_name}_deque')

                    return None

                # 是否压缩处理图像
                if self.args.is_compress and self.in_collect:
                    img_data[cam_name] = self.bridge.compressed_imgmsg_to_cv2(deque_map[cam_name].pop(),
                                                                              'passthrough')
                else:
                    img_data[cam_name] = self.bridge.imgmsg_to_cv2(deque_map[cam_name].pop(),
                                                                   'passthrough')

            if self.args.use_depth_image:
                if cam_name in img_depth_data:
                    deque_map = {
                        'head_depth': self.img_head_depth_deque,
                        'left_wrist_depth': self.img_left_depth_deque,
                        'right_wrist_depth': self.img_right_depth_deque,
                    }

                    key = cam_name + '_depth'

                    if len(deque_map[key]) == 0:
                        print(f'there is no {key}_deque')

                        return None

                    if self.args.is_compress:
                        img_depth_data[key] = self.bridge.compressed_imgmsg_to_cv2(deque_map[key].pop(),
                                                                                   'passthrough')
                    else:
                        img_depth_data[key] = self.bridge.imgmsg_to_cv2(deque_map[key].pop(),
                                                                        'passthrough')

        # 获取机械臂状态
        for arm_name in ['follow_arm_left', 'follow_arm_right', 'follow_arm_left_eef', 'follow_arm_right_eef']:
            deque_map = {
                'follow_arm_left': self.follow_arm_left_deque,
                'follow_arm_right': self.follow_arm_right_deque,
                'follow_arm_left_eef': self.follow_arm_left_eef_deque,
                'follow_arm_right_eef': self.follow_arm_right_eef_deque,
            }
            if len(deque_map[arm_name]) == 0:
                print(f'there is no {arm_name}_deque')

                return None

            arm_data[arm_name] = deque_map[arm_name].pop()

        obs_dict = collections.OrderedDict()  # 有序的字典

        # 保存图像
        obs_dict['images'] = {cam: img for cam, img in img_data.items() if cam in self.args.camera_names}

        if self.args.use_depth_image:
            obs_dict['images_depth'] = {cam: img_depth_data[cam] for cam in img_depth_data if
                                        cam in self.args.camera_names}

        # 保存机械臂状态
        follow_arm_left_eef_array = [arm_data['follow_arm_left_eef'].x, arm_data['follow_arm_left_eef'].y,
                                     arm_data['follow_arm_left_eef'].z,
                                     arm_data['follow_arm_left_eef'].roll, arm_data['follow_arm_left_eef'].pitch,
                                     arm_data['follow_arm_left_eef'].yaw, arm_data['follow_arm_left_eef'].gripper]

        follow_arm_right_eef_array = [arm_data['follow_arm_right_eef'].x, arm_data['follow_arm_right_eef'].y,
                                      arm_data['follow_arm_right_eef'].z,
                                      arm_data['follow_arm_right_eef'].roll, arm_data['follow_arm_right_eef'].pitch,
                                      arm_data['follow_arm_right_eef'].yaw, arm_data['follow_arm_right_eef'].gripper]
        
        
        
        
        
        qpos = np.concatenate((np.array(arm_data['follow_arm_left'].joint_pos),
                                           np.array(arm_data['follow_arm_right'].joint_pos)), axis=0)       
        
        qpos[6]  = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos[6])
        qpos[13] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos[13])
        # qvel[6]  = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qvel[6])
        # qvel[13] = constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qvel[13])

        obs_dict['qpos'] = qpos
        # obs_dict['qpos'] = np.concatenate((np.array(arm_data['follow_arm_left'].joint_pos),
        #                                    np.array(arm_data['follow_arm_right'].joint_pos)), axis=0)
        obs_dict['qvel'] = np.concatenate((np.array(arm_data['follow_arm_left'].joint_vel),
                                           np.array(arm_data['follow_arm_right'].joint_vel)), axis=0)
        obs_dict['effort'] = np.concatenate((np.array(arm_data['follow_arm_left'].joint_cur),
                                             np.array(arm_data['follow_arm_right'].joint_cur)), axis=0)
        obs_dict['eef'] = np.concatenate((follow_arm_left_eef_array, follow_arm_right_eef_array), axis=0)

        # 保存底盘状态
        # if self.args.use_base and ts != 0:
        #     if len(self.robot_base_deque) == 0:
        #         print(r'there is no robot_base_deque, maby there is no VR message')

        #         return None

        #     if len(self.base_pose_deque) == 0:
        #         print(r'there is no base_pose_deque')

        #         return None

        #     robot_base = self.robot_base_deque.pop()
        #     base_pose = self.base_pose_deque.pop()
        #     obs_dict['robot_base'] = [base_pose[0], base_pose[1], base_pose[2], robot_base.height,
        #                               robot_base.head_pit, robot_base.head_yaw]
        # else:
        obs_dict['robot_base'] = np.zeros((6,))

        return obs_dict

    def get_action(self):
        joints_dim = 7

        action_dict = collections.OrderedDict()

        def extract_eef_data(eef):
            return [eef.x, eef.y, eef.z, eef.roll, eef.pitch, eef.yaw, eef.gripper]

        deque_map = {
            'master_VR_left_eef_deque': self.master_VR_left_eef_deque,
            'master_VR_right_eef_deque': self.master_VR_right_eef_deque,
        }

        for name, deque in deque_map.items():
            if len(deque) == 0:
                print(f'there is no {name}')

                return None

        # 获取主臂状态
        master_arm_left_eef = self.master_VR_left_eef_deque.pop()
        master_arm_right_eef = self.master_VR_right_eef_deque.pop()

        # 主臂保存状态
        master_arm_left_eef_array = extract_eef_data(master_arm_left_eef)
        master_arm_right_eef_array = extract_eef_data(master_arm_right_eef)

        # 构建动作字典
        action_dict['action'] = np.zeros((joints_dim * 2,))
        action_dict['action_qvel'] = np.zeros((joints_dim * 2,))
        action_dict['action_eef'] = np.concatenate((master_arm_left_eef_array, master_arm_right_eef_array), axis=0)
        action_dict['action_base'] = np.zeros((13,))  # waiting for the obersevation

        return action_dict

    def img_head_callback(self, msg):
        if len(self.img_head_deque) >= 2000:
            self.img_head_deque.popleft()
        self.img_head_deque.append(msg)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_head_depth_callback(self, msg):
        if len(self.img_head_depth_deque) >= 2000:
            self.img_head_depth_deque.popleft()
        self.img_head_depth_deque.append(msg)

    # master qpos and eef
    def master_arm_left_eef_callback(self, msg):
        if len(self.master_arm_left_eef_deque) >= 2:
            self.master_arm_left_eef_deque.popleft()
        self.master_arm_left_eef_deque.append(msg)

    def master_arm_right_eef_callback(self, msg):
        if len(self.master_arm_right_eef_deque) >= 2:
            self.master_arm_right_eef_deque.popleft()
        self.master_arm_right_eef_deque.append(msg)

    # VR Master
    def master_VR_left_eef_callback(self, msg):
        if len(self.master_VR_left_eef_deque) >= 2:
            self.master_VR_left_eef_deque.popleft()
        self.master_VR_left_eef_deque.append(msg)

    def master_VR_right_eef_callback(self, msg):
        if len(self.master_VR_right_eef_deque) >= 2:
            self.master_VR_right_eef_deque.popleft()
        self.master_VR_right_eef_deque.append(msg)

    # follow qpos and eef
    def follow_arm_left_callback(self, msg):
        if len(self.follow_arm_left_deque) >= 2:
            self.follow_arm_left_deque.popleft()
        self.follow_arm_left_deque.append(msg)

    def follow_arm_left_eef_callback(self, msg):
        if len(self.follow_arm_left_eef_deque) >= 2:
            self.follow_arm_left_eef_deque.popleft()
        self.follow_arm_left_eef_deque.append(msg)

    def follow_arm_right_callback(self, msg):
        if len(self.follow_arm_right_deque) >= 2:
            self.follow_arm_right_deque.popleft()
        self.follow_arm_right_deque.append(msg)

    def follow_arm_right_eef_callback(self, msg):
        if len(self.follow_arm_right_eef_deque) >= 2:
            self.follow_arm_right_eef_deque.popleft()
        self.follow_arm_right_eef_deque.append(msg)

    # robot robot_base
    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def base_pose_callback(self, msg):
        if len(self.base_pose_deque) >= 2:
            self.base_pose_deque.popleft()

        if len(self.robot_base_origin) >= 2:
            self.robot_base_origin.popleft()
        self.robot_base_origin.append(msg)

        tf_info = msg.transforms[0].transform
        base_quaternion = [tf_info.rotation.x, tf_info.rotation.y,
                           tf_info.rotation.z, tf_info.rotation.w]
        r = R.from_quat(base_quaternion)
        _, _, base_pose_yaw = r.as_euler('xyz', degrees=False)
        base_pose = [tf_info.translation.x, -tf_info.translation.y, base_pose_yaw]

        base_pose[0] = base_pose[0] - self.robot_base_pose_init[0]  # 如果这个值是负的
        base_pose[1] = base_pose[1] - self.robot_base_pose_init[1]
        base_pose[2] = base_pose[2] - self.robot_base_pose_init[2]

        self.base_pose_deque.append(base_pose)

    def _update_arm_position(self, target, arm, symbol, steps_length):
        diff = [abs(target[i] - arm[i]) for i in range(len(target))]
        done = 0
        for i in range(len(target)):
            if diff[i] < steps_length[i]:
                arm[i] = target[i]
                done += 1
            else:
                arm[i] += symbol[i] * steps_length[i]

        return done

