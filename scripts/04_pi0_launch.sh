#!/bin/bash
set -x

# Get ARX_ROOT dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

# CAN (legacy - CAN daemons should be managed by systemd or hardware scripts)
# See scripts/hardware/start_can_daemon.sh for CAN interface management

# Body (LIFT controller)
gnome-terminal --title="body" -x $shell_type -i -c "cd ${ARX_ROOT}/ros2_ws; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; $shell_exec"
sleep 1

# Arms (R5 dual arm - no VR for Pi0 inference)
gnome-terminal --title="arms" -x $shell_type -i -c "cd ${ARX_ROOT}/ros2_ws; source install/setup.bash; ros2 launch arx_r5_controller open_double_arm.launch.py; $shell_exec"
sleep 1

# RealSense cameras
gnome-terminal --title="realsense" -x $shell_type -i -c "cd ${ARX_ROOT}/ros2_ws; source install/setup.bash; ros2 launch realsense2_camera rs_launch.py; $shell_exec"
sleep 3

