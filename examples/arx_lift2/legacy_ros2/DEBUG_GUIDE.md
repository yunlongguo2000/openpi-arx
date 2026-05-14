# 解决 ROS2 与 Conda 环境冲突指南

## 问题原因
Conda 环境中的 `libstdc++.so.6` 版本较旧,不满足 ROS2 Jazzy 要求的 `GLIBCXX_3.4.30` 版本。

## 解决方案

### 方法 1: 使用 wrapper 脚本运行 (生产环境推荐)

```bash
# 直接运行
./examples/aloha_arx_lift_ros2_real/run_main.sh

# 或者使用开发脚本
./examples/aloha_arx_lift_ros2_real/dev_run.sh
```

### 方法 2: 在终端中手动设置环境 (调试推荐)

```bash
# 1. 激活环境
source examples/aloha_arx_lift_ros2_real/setup_env.sh

# 2. 现在可以直接运行 Python 命令
python -m examples.aloha_arx_lift_ros2_real.main

# 或者运行其他 Python 脚本
python examples/aloha_arx_lift_ros2_real/main.py
```

### 方法 3: 使用 VSCode 调试器 (开发调试推荐)

1. 打开 VSCode 调试面板 (F5 或点击左侧调试图标)
2. 选择配置: **"🤖 ALOHA ROS2 Real (调试模式)"**
3. 点击运行或按 F5

这个配置已经设置好了所有必要的环境变量,可以直接使用断点调试。

### 方法 4: 一键运行脚本

```bash
# 快速启动开发模式
./examples/aloha_arx_lift_ros2_real/dev_run.sh
```

## 环境变量说明

关键的环境变量设置:

- `LD_LIBRARY_PATH`: 优先使用系统和 ROS2 的库路径
- `LD_PRELOAD`: 预加载系统的 libstdc++.so.6
- `PYTHONPATH`: 添加 ROS2 和自定义消息的 Python 路径

## 验证环境

```bash
# 验证 libstdc++ 版本
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

# 应该能看到 GLIBCXX_3.4.30 或更高版本
```

## 故障排查

### 如果仍然出现 GLIBCXX 错误:

1. 检查 Conda 是否在干扰:
```bash
echo $LD_LIBRARY_PATH
# 确保 /usr/lib/x86_64-linux-gnu 在 conda lib 路径之前
```

2. 手动设置预加载:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

3. 临时禁用 Conda 的库路径:
```bash
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 如果 ROS2 消息找不到:

确保消息路径正确:
```bash
ls examples/aloha_arx_lift_ros2_real/msg/3.12/
# 应该能看到编译好的消息文件
```

## 推荐工作流程

**日常开发:**
```bash
# 新终端中
source examples/aloha_arx_lift_ros2_real/setup_env.sh
python -m examples.aloha_arx_lift_ros2_real.main
```

**需要调试:**
- 使用 VSCode 调试配置 "🤖 ALOHA ROS2 Real (调试模式)"

**快速测试:**
```bash
./examples/aloha_arx_lift_ros2_real/dev_run.sh
```
