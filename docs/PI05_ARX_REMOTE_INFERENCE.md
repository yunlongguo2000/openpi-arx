# π₀.₅ 模型在 ARX LIFT2 上的远程推理部署方案

本文档描述如何使用 **分离式部署**：在一台 NVIDIA RTX 4090 机器上运行模型推理，机器人机载电脑运行数据采集和运动控制，两者通过网络通信。

## 架构概述

```
┌─────────────────────────┐      WebSocket       ┌─────────────────────────┐
│  GPU服务器 (RTX 4090)   │ ←──────────────────→ │  机器人机载电脑          │
│  • 运行 pi0.5 模型推理  │                       │  • 采集传感器数据        │
│  • 监听 8000 端口      │                       │  • ROS2 控制机器人      │
│  • 输出动作序列        │                       │  • 执行运动控制          │
└─────────────────────────┘                       └─────────────────────────┘
```

**优势**：
- 机器人不需要强大GPU，降低机载电脑成本
- 可以利用工作站/开发机的强大GPU资源
- 环境分离，便于调试和维护

## 系统要求

| 机器 | 硬件要求 | 软件要求 |
|------|---------|---------|
| GPU服务器 | NVIDIA GPU ≥ 8GB VRAM（RTX 4090 满足） | Ubuntu 22.04, CUDA驱动 |
| 机器人机载 | x86_64 或 ARM64 处理器，网络连通 | Ubuntu 22.04/24.04, ROS2 Jazzy |

---

## 第一步：克隆代码仓库

**两台机器都需要执行**：

```bash
git clone --recurse-submodules https://github.com/yunlongguo2000/openpi-arx.git
cd openpi-arx
```

如果你已经克隆过，更新子模块：

```bash
git submodule update --init --recursive
```

---

## 第二步：环境安装

### 🖥️ **在 GPU服务器 (4090) 上**

```bash
# 1. 创建conda环境
conda create -n openpi python=3.11 -y
conda activate openpi

# 2. 安装uv
pip install uv

# 3. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4. 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev

# 5.（可选）如果使用 PyTorch 模型，应用 transformers 补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

### 🤖 **在 机器人本机 上**

```bash
# 1. 创建conda环境
conda create -n openpi python=3.11 -y
conda activate openpi

# 2. 安装uv
pip install uv

# 3. 安装依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4. 安装 openpi-client（官方远程推理客户端）
cd packages/openpi-client
pip install -e .
cd ../..

# 5. 安装机器人通信额外依赖（当前使用纯 ZMQ + msgpack，已替换掉 ZeroRPC）
uv pip install pyzmq msgpack-python

# 6. 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

---

## 第三步：在 GPU服务器 (4090) 上 - 启动策略服务端

### 选项 A：使用你自己微调好的模型

```bash
conda activate openpi
cd openpi-arx

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=checkpoints/pi05_arx/your_task_name/30000 \
    --host 0.0.0.0 \
    --port 8000
```

- `--policy.config`: 配置名称，`pi05_arx` 或 `pi05_arx_lora`
- `--policy.dir`: 你的checkpoint目录路径
- `--host`: 监听所有网络接口 (`0.0.0.0`)
- `--port`: 监听端口，默认 `8000`

### 选项 B：使用官方预训练模型测试

```bash
conda activate openpi
cd openpi-arx

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
    --host 0.0.0.0 \
    --port 8000
```

模型会自动下载到 `~/.cache/openpi/`。

### 验证服务启动

**预期输出**：
```
* Running on http://0.0.0.0:8000
```

**检查网络连通性（从机器人本机测试）**：

```bash
# 在机器人本机上执行
nc -zv <YOUR_GPU_SERVER_IP> 8000
# 或者
telnet <YOUR_GPU_SERVER_IP> 8000
```

如果连接成功，说明网络配置正确。如果失败，请检查：
- GPU服务器防火墙是否开放8000端口
- 两台机器是否在同一局域网
- IP地址是否正确

---

## 第四步：在 机器人本机 上 - 配置和启动

### 1. 启动 ROS2 ARX 控制服务

打开第一个终端，启动ARX ROS2 RPC服务端：

```bash
# source 你的ROS2环境
conda deactivate  # 如果conda环境有冲突
source /opt/ros/jazzy/setup.bash
source /home/arx/ARX_new/lerobot_data_collection/install/setup.bash

# 启动ARX ROS2 RPC服务端（监听 4242 端口）
cd /home/arx/ARX_new/lerobot_data_collection/arx_vr_data_collection
python ros2_bridge/arx_ros2_rpc_server.py
```

当前使用 **纯 ZMQ + msgpack** RPC 通信（已替换 ZeroRPC 以降低延迟），服务默认监听 `4242` 端口。

### 2. 配置推理参数

打开第二个终端，复制并编辑配置文件：

```bash
conda activate openpi
cd openpi-arx

cp examples/arx/config/cfg_arx_pi.yaml my_deployment.yaml
nano my_deployment.yaml
```

编辑以下内容：

```yaml
policy_server:
  host: "192.168.1.100"  # 这里替换成你的4090机器的实际IP地址
  port: 8000              # 与服务端端口一致

robot:
  ip: "127.0.0.1"        # ROS2 RPC服务地址，本机一般是127.0.0.1
  port: 4242              # ROS2 RPC端口

task_instruction: "pick up the red cup"  # 这里替换成你的任务描述

control:
  action_horizon: 16       # 动作 horizon，与训练配置一致
  control_freq: 15         # 控制频率 (Hz)，与数据集采集频率一致
```

### 3. 复制 norm_stats.json（如果需要）

如果你在GPU服务器上训练计算了 `norm_stats.json`，需要将它复制到机器人本机：

```bash
# 在GPU服务器上，找到norm_stats.json位置
# 通常在你的数据集目录下

# 复制到机器人本机
scp /path/to/norm_stats.json robot@<ROBOT_IP>:~/openpi-arx/path/to/norm_stats.json
```

### 4. 检查目录路径

`inference_arx.py` 默认假设你的目录结构如下：

```
/home/arx/ARX_new/
├── openpi-arx/                 (本仓库)
│   └── examples/arx/inference_arx.py
└── lerobot_data_collection/
    └── arx_vr_data_collection/
        └── ros2_bridge/arx_ros2_rpc_client.py
```

如果你的 `lerobot_data_collection` 位置不同，需要修改 `inference_arx.py` 中的 `_ARX_BRIDGE_PATH`。

### 5. 启动推理客户端

```bash
conda activate openpi
cd openpi-arx

python examples/arx/inference_arx.py \
    --config my_deployment.yaml
```

---

## 第五步：运行流程

启动成功后，推理循环会执行以下步骤：

1. **读取状态**：通过RPC从ROS2获取59维机器人状态
   - 左右臂关节位置/速度/电流
   - 左右臂TCP位姿
   - 夹爪位置
   - 底盘高度/云台角度
   - 状态可以保持未归一化，归一化由GPU服务器处理

2. **获取图像**：从腕部RealSense相机获取图像（如果已连接）
   - 在客户端对图像进行resize填充并转换为uint8格式，减少网络带宽和延迟
   - 预训练pi0模型的典型尺寸为 `224x224`

3. **发送请求**：通过WebSocket将观测发送给GPU服务器

4. **获取动作**：接收模型输出的 `32D × 16步` 动作序列
   - 不需要每次控制步都请求模型，通常每N步请求一次，剩余步骤开环执行动作序列中的后续动作

5. **执行控制**：以15Hz频率逐步发送动作命令给ROS2控制器执行

---

## 完整的一键启动脚本示例

你可以创建启动脚本来简化操作。

### GPU服务器 `start_server.sh`:

```bash
#!/bin/bash
conda activate openpi
cd ~/openpi-arx
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=checkpoints/pi05_arx/your_task/30000 \
    --host 0.0.0.0 --port 8000
```

### 机器人本机 `start_inference.sh`:

```bash
#!/bin/bash
conda activate openpi
cd ~/openpi-arx
python examples/arx/inference_arx.py --config my_deployment.yaml
```

添加执行权限：
```bash
chmod +x start_server.sh start_inference.sh
```

---

## 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| **连接超时** | 防火墙/网络/IP错误 | 1. 检查GPU服务器IP 2. 检查防火墙是否开放8000端口 3. 测试ping和telnet连接 |
| **Missing `norm_stats.json`** | 归一化统计文件缺失 | 将GPU服务器上 `compute_norm_stats.py` 生成的 `norm_stats.json` 复制到机器人对应路径 |
| **Action dimension mismatch** | 动作维度不匹配 | 检查 `ArxOutputs.action_dim = 32` 是否正确，确认数据集动作维度一致 |
| **GPU Out of Memory** | GPU显存不足 | 1. 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 再启动 2. 使用更小的batch |
| **Images all black** | 相机未连接 | 当前代码用零填充占位，这是预期行为；连接RealSense后替换 `get_observation()` 中的图像获取代码 |
| **Robot RPC connection failed** | ROS2 RPC服务未启动 | 1. 确认 `arx_ros2_rpc_server.py` 正在运行 2. 检查配置中的IP/端口是否一致 |
| **`ArxROS2RPCClient` import error** | 桥接路径不正确 | 检查 `inference_arx.py` 中的 `_ARX_BRIDGE_PATH` 指向正确的 `arx_vr_data_collection` 路径 |
| **`uv sync` 依赖冲突** | 依赖缓存问题 | 删除 `.venv` 目录重新运行：`rm -rf .venv && GIT_LFS_SKIP_SMUDGE=1 uv sync` |
| **`No module named openpi_client'`** | openpi-client 未安装 | 在机器人本机执行：`cd packages/openpi-client && pip install -e .` |
| **`No module named ros2_bridge.arx_ros2_rpc_client'`** | 找不到 arx_vr_data_collection | 检查 `inference_arx.py` 中的 `_ARX_BRIDGE_PATH` 是否指向正确的 `lerobot_data_collection/arx_vr_data_collection` 路径 |
| **ZMQ connection refused** | arx_ros2_rpc_server 未启动 | 确认第一个终端已经启动 `python arx_ros2_rpc_server.py`，且IP/端口配置一致 |

---

## 快速命令总览

| 步骤 | GPU服务器 (4090) | 机器人本机 |
|------|-----------------|-----------|
| 克隆 | `git clone --recurse-submodules https://github.com/yunlongguo2000/openpi-arx.git` | 同上 |
| 环境 | `conda create -n openpi python=3.11 && conda activate openpi && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` | 同上 + `cd packages/openpi-client && pip install -e . && cd ../.. && uv pip install pyzmq msgpack-python` |
| 启动 | `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_arx --policy.dir=path/to/checkpoint --host 0.0.0.0 --port 8000` | 终端1: `python /path/to/arx_vr_data_collection/ros2_bridge/arx_ros2_rpc_server.py`<br>终端2: `python examples/arx/inference_arx.py --config my_deployment.yaml` |

---

## 数据流向

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  GPU服务器 (RTX 4090)                                 │
  │                                                                      │
  │  ┌─────────────┐                                                    │
  │  │  pi0.5 Model │  ←───  接收观测  ←───  WebSocket  ←───┐           │
  │  └─────────────┘  →───  输出动作  →───  WebSocket  →───┘           │
  │                                                                      │
  └─────────────────────────────────────────────────────────────────────┘
                                     ↕
                               局域网通信
                                     ↕
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  机器人机载电脑                                       │
  │                                                                      │
  │  1. arx_ros2_rpc_server (ROS2)                                      │
  │      ↓ 获取59维机器人状态                                            │
  │  2. inference_arx.py (客户端)                                       │
  │      ↓ 拼接观测 (state + images + prompt)                            │
  │      ↓ 发送给GPU服务器                                               │
  │      ↓ 接收 32D × 16-step 动作 chunk                                │
  │      ↓ 以15Hz频率发送给ROS2控制器执行                                │
  │                                                                      │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 参考文献

- [README.md](../README.md) - 项目主文档
- [PI05_FRANKA.md](../PI05_FRANKA.md) - Franka 部署参考
- [PI05_ARX.md](../PI05_ARX.md) - ARX LIFT2 完整文档
- [docs/remote_inference.md](../docs/remote_inference.md) - OpenPI 官方远程推理文档

---

**最后更新**: 2026-03-20
