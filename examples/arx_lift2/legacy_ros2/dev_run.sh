#!/usr/bin/env bash
# 开发调试快捷脚本
# 使用方法: ./examples/aloha_arx_lift_ros2_real/dev_run.sh [参数]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================"
echo "🤖 启动 ALOHA ROS2 Real 调试模式"
echo "========================================"

# 检查 ROS2 环境
if [ ! -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "❌ 错误: 未找到 ROS 2 Jazzy，请先安装" >&2
    exit 1
fi

# 加载环境
echo "📦 加载环境..."

# 1. Conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "uv_envs" ]; then
    echo "  - 激活 Conda 环境: uv_envs"
    source /home/arx/miniconda3/bin/activate uv_envs
fi

# 2. 项目虚拟环境
if [ -z "$VIRTUAL_ENV" ] || [ ! "$VIRTUAL_ENV" -ef "$PROJECT_ROOT/.venv" ]; then
    echo "  - 激活项目虚拟环境"
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 3. ROS 2 环境
echo "  - 加载 ROS 2 Jazzy"
source /opt/ros/jazzy/setup.bash

# 4. 解决库冲突
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# 5. 添加自定义消息路径
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MSG_PATH="$SCRIPT_DIR/msg/$PYTHON_VERSION"
if [ -d "$MSG_PATH" ]; then
    export PYTHONPATH="$MSG_PATH:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$MSG_PATH/lib:${LD_LIBRARY_PATH}"
fi

# 6. 预加载系统 libstdc++
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD+:$LD_PRELOAD}"
fi

echo ""
echo "✅ 环境配置完成"
echo "🚀 启动程序..."
echo ""

# 运行主程序
exec python -m examples.aloha_arx_lift_ros2_real.main "$@"
