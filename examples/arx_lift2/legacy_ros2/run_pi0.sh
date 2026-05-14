#!/usr/bin/env bash
# set -euo pipefail
# Conda 环境里直接运行 collect.py 时导入 rclpy 报错，核心原因是 Conda 自带的 libstdc++.so.6 太旧，优先级压过了系统/ROS2 的 libstdc++，导致需要的符号 GLIBCXX_3.4.30 找不到

# 优先系统/ROS2 库，预加载系统 libstdc++

# 进入脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1) 加载 ROS 2 环境（提供 rclpy 的 Python 与相关共享库路径）
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
else
  echo "[ERROR] /opt/ros/jazzy/setup.bash 未找到，请确认已安装 ROS 2 Jazzy。" >&2
  exit 1
fi

# 2) 让系统 libstdc++ 与 ROS 库路径在最前，避免被 Conda 的旧版覆盖
#    注：Conda 环境常把 $CONDA_PREFIX/lib 放在 LD_LIBRARY_PATH 前面，会触发 GLIBCXX 版本冲突
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# 3) 保险起见，预加载系统 libstdc++.so.6（若存在）
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD+:$LD_PRELOAD}"
fi

# 4) 选择 Python 解释器：优先当前 shell 的 python，其次 python3
PYTHON_BIN="$(command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "[ERROR] 未找到 python 或 python3 可执行文件。" >&2
  exit 1
fi

# 5) 运行采集脚本，转发所有参数
exec "$PYTHON_BIN" "$SCRIPT_DIR/collect.py" "$@"
