"""Offline platform integration; no GPU model or robot required."""
import numpy as np
from arx_client import ArxROS2RPCClient
from openpi.arx.arx_ros2_rpc_client import ArxROS2RPCClient as CompatibilityClient
from openpi.arx.arx_r5.arx_r5_robot_adapter import state_56d_from_full_state
from openpi.arx.arx_lift2.arx_lift2_robot_adapter import state_59d_from_full_state


def test_model_uses_the_installed_platform_client():
    assert CompatibilityClient is ArxROS2RPCClient


def test_model_state_adapters_preserve_field_order():
    def arm(offset):
        return {'joint_positions': np.arange(7)+offset,
                'joint_velocities': np.arange(7)+offset+10,
                'joint_currents': np.arange(7)+offset+20,
                'end_pose': np.arange(6)+offset+30, 'gripper': offset+40}
    state = {'left_arm': arm(0), 'right_arm': arm(100),
             'chassis': {'height': 1, 'head_yaw': 2, 'head_pitch': 3}}
    r5 = state_56d_from_full_state(state)
    lift = state_59d_from_full_state(state)
    assert r5.shape == (56,) and lift.shape == (59,)
    np.testing.assert_array_equal(r5[:7], state['left_arm']['joint_positions'])
    np.testing.assert_array_equal(r5[21:28], state['right_arm']['joint_positions'])
    np.testing.assert_array_equal(lift[:56], r5)
    np.testing.assert_array_equal(lift[56:], [1, 2, 3])
