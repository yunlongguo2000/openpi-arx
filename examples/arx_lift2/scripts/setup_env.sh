#!/usr/bin/env bash
# 用于开发调试的环境设置脚本
# 使用方法: source examples/aloha_arx_lift_ros2_real/setup_env.sh

# 1) 激活 Conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "uv_envs" ]; then
    echo "激活 Conda 环境: uv_envs"
    source /home/arx/miniconda3/bin/activate uv_envs
fi

# 2) 激活项目虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "激活项目虚拟环境"
    source ./.venv/bin/activate
fi

# 3) 加载 ROS 2 环境
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    echo "加载 ROS 2 Jazzy 环境"
    source /opt/ros/jazzy/setup.bash
else
    echo "[警告] /opt/ros/jazzy/setup.bash 未找到" >&2
fi

# 4) 解决库冲突：让系统 libstdc++ 优先于 Conda 的
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# 5) 添加 ROS 2 自定义消息路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MSG_PATH="$SCRIPT_DIR/msg/$PYTHON_VERSION"
if [ -d "$MSG_PATH" ]; then
    export PYTHONPATH="$MSG_PATH:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$MSG_PATH/lib:${LD_LIBRARY_PATH}"
fi

# 6) 预加载系统 libstdc++.so.6
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD+:$LD_PRELOAD}"
fi

echo "============================================"
echo "环境已配置完成！"
echo "现在可以直接运行:"
echo "  python -m examples.aloha_arx_lift_ros2_real.main"
echo "或者使用 VSCode 调试器"
echo "============================================"
