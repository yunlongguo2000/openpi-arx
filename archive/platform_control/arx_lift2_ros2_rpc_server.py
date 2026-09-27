from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time

import zerorpc

# ARX LIFT 2 specialized RPC server (Dual Arms + Chassis + Lift)
# This server includes full chassis and lift support.

log = logging.getLogger(__name__)

class ArxLift2ROS2RPCServer:
    """RPC server for ARX LIFT 2 (Dual Arms + Mobile Base + Lift)."""

    def __init__(self, ip="0.0.0.0", port=4242):
        self._ip = ip
        self._port = port
        # In a real setup, this would connect to the arx_lift_controller ROS2 nodes
        self._system_connected = False

    def system_connect(self, timeout=10.0) -> bool:
        log.info("ARX LIFT 2 System connecting...")
        # Placeholder for actual ROS2 initialization
        self._system_connected = True
        return True

    def get_full_state(self):
        """Returns the current state of the dual arms, grippers, and chassis."""
        # Mocking LIFT 2 state for now.
        return {
            "left_arm": {
                "joint_positions": [0.0]*7, 
                "joint_velocities": [0.0]*7, 
                "joint_currents": [0.0]*7, 
                "end_pose": [0.0]*6, 
                "gripper": 0.0
            },
            "right_arm": {
                "joint_positions": [0.0]*7, 
                "joint_velocities": [0.0]*7, 
                "joint_currents": [0.0]*7, 
                "end_pose": [0.0]*6, 
                "gripper": 0.0
            },
            "chassis": {
                "height": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "wz": 0.0
            }
        }

    def set_full_command(self, left_joints, right_joints, vx=0.0, vy=0.0, wz=0.0, h=0.0):
        """Executes arm, chassis, and lift commands."""
        log.debug(f"LIFT 2 Command - Arms: {left_joints}/{right_joints}, Chassis: {vx},{vy},{wz}, Lift: {h}")
        # Logic to publish to ROS2
        return {"accepted": True}

    def set_left_gripper(self, pos):
        return {"accepted": True}

    def set_right_gripper(self, pos):
        return {"accepted": True}

    def heartbeat(self):
        pass

    def emergency_stop(self):
        log.warning("EMERGENCY STOP TRIGGERED")
        return {"accepted": True}

def main():
    parser = argparse.ArgumentParser(description="ARX LIFT 2 ROS2 RPC Server")
    parser.add_argument("--port", type=int, default=4242, help="Port to listen on")
    args = parser.parse_args()

    server = zerorpc.Server(ArxLift2ROS2RPCServer(port=args.port))
    server.bind(f"tcp://0.0.0.0:{args.port}")
    log.info(f"ARX LIFT 2 ROS2 RPC Server listening on port {args.port}")
    
    def stop_server(signum, frame):
        log.info("Stopping server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    server.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
