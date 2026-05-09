#!/bin/bash
# ARX Pi0.5 推理专用硬件启动脚本
#
# 启动顺序: LIFT body → R5 dual arm (无VR) → RealSense cameras
# 注意: Pi0.5 推理模式下不启动 VR 遥操 (使用 open_double_arm.launch.py)
#
# 用法:
#   ./scripts/launch_hardware.sh
#   ./scripts/launch_hardware.sh --no-cameras   # 不启动相机 (用 RPC 取图时)
#
# 前置条件:
#   - ROS2 Jazzy workspace 已编译 (ARX_ROOT/install/ 存在)
#   - CAN 接口已配置 (scripts/hardware/setup_can.sh)

set -e

# 自动检测 ARX workspace 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# openpi-arx/scripts/ → 向上两级到 ARX_new (如果 openpi-arx 是子目录)
# 也支持 openpi-arx 作为独立仓库 (此时需设置 ARX_ROOT 环境变量)
if [ -n "$ARX_ROOT" ]; then
    : # 使用环境变量
elif [ -d "${SCRIPT_DIR}/../../install" ]; then
    ARX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
elif [ -d "${SCRIPT_DIR}/../../../install" ]; then
    ARX_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
    echo "[ERROR] Cannot find ARX ROS2 workspace. Set ARX_ROOT environment variable."
    exit 1
fi

LAUNCH_CAMERAS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cameras)
            LAUNCH_CAMERAS=false
            shift
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Usage: $0 [--no-cameras]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  ARX Pi0.5 Hardware Launch (Inference Mode)"
echo "============================================================"
echo "  ARX_ROOT: ${ARX_ROOT}"
echo "  Cameras:  ${LAUNCH_CAMERAS}"
echo ""

# 检查 workspace
if [ ! -f "${ARX_ROOT}/install/setup.bash" ]; then
    echo "[ERROR] ROS2 workspace not built: ${ARX_ROOT}/install/setup.bash not found"
    echo "  Run: cd ${ARX_ROOT} && colcon build --symlink-install"
    exit 1
fi

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

# 1. LIFT body controller
echo "[1/3] Launching LIFT body controller..."
gnome-terminal \
    --title="[Pi0.5] LIFT Body" \
    -- $shell_type -i -c "\
        cd ${ARX_ROOT} && \
        source install/setup.bash && \
        echo '[BODY] Starting LIFT controller...' && \
        ros2 launch arx_lift_controller lift.launch.py; \
        $shell_exec"
sleep 2

# 2. R5 dual arm (无 VR — 推理模式)
echo "[2/3] Launching R5 dual arm (no VR)..."
gnome-terminal \
    --title="[Pi0.5] R5 Arms" \
    -- $shell_type -i -c "\
        cd ${ARX_ROOT} && \
        source install/setup.bash && \
        echo '[ARMS] Starting R5 dual arm (inference mode, no VR)...' && \
        ros2 launch arx_r5_controller open_double_arm.launch.py; \
        $shell_exec"
sleep 2

# 3. RealSense cameras (可选)
if [ "$LAUNCH_CAMERAS" = true ]; then
    echo "[3/3] Launching RealSense cameras..."
    gnome-terminal \
        --title="[Pi0.5] RealSense" \
        -- $shell_type -i -c "\
            cd ${ARX_ROOT} && \
            source install/setup.bash && \
            echo '[CAMERA] Starting RealSense cameras...' && \
            ros2 launch realsense2_camera rs_launch.py; \
            $shell_exec"
    sleep 3
else
    echo "[3/3] Skipping cameras (--no-cameras)"
fi

echo ""
echo "[OK] Hardware components launched."
echo ""
echo "Next steps:"
echo "  1. GPU machine:  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_arx --policy.dir=<checkpoint>"
echo "  2. Robot machine: python examples/arx_r5/inference_arx_r5.py --config examples/arx_r5/config/cfg_arx_r5_pi.yaml"
echo ""
