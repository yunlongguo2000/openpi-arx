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

| 机器 | 主机名 | 硬件要求 | 软件要求 |
|------|--------|---------|---------|
| GPU服务器 | deepcybo | NVIDIA GPU ≥ 8GB VRAM（RTX 4090） | Ubuntu 22.04, CUDA驱动 |
| 机器人机载 | arx | x86_64 或 ARM64 处理器，网络连通 | Ubuntu 22.04/24.04, ROS2 Jazzy |

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

### 🖥️ **在 GPU服务器 (deepcybo, 4090) 上**

GPU 服务器需要完整安装所有依赖（JAX、PyTorch、模型推理栈等）。

#### ⚠️ **重要：lerobot 版本锁定**

本项目对 lerobot 版本有严格要求，**必须使用 0.3.4**，不能使用最新版。原因如下：

| 组件 | 要求 | 冲突说明 |
|------|------|--------|
| **lerobot** | 0.3.4（必须） | 0.5.1+ 要求 huggingface-hub>=1.0.0 |
| **transformers** | 4.53.2（锁定） | 要求 huggingface-hub<1.0 |
| **Python** | 3.11（必须） | lerobot 0.3.4 支持 >=3.10；0.5.1+ 要求 >=3.12 |

**数据格式兼容性**：本机数据采集使用 lerobot 0.3.4，GPU 服务器训练也必须使用相同版本，否则数据集格式不兼容。

#### 安装步骤

```bash
# 1. 克隆 lerobot（必须在创建conda环境之前）
cd /home/deepcybo
git clone https://github.com/huggingface/lerobot.git

# 2. 切换到 0.3.4 版本（重要！）
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
cd ..

# 3. 创建conda环境（Python 3.11）
conda create -n openpi_arx python=3.11 -y
conda activate openpi_arx

# 4. 安装uv
pip install uv

# 5. 确保 pyproject.toml 中 lerobot 路径正确
#    openpi-arx/pyproject.toml 应包含：
#    lerobot = {path = "/home/deepcybo/lerobot"}
#    如果路径不同，请先修改 pyproject.toml 中的 [tool.uv.sources] 部分

# 6. 安装依赖（下载量较大，约 5-8GB，含 JAX CUDA12 + PyTorch 2.7.1）
cd /home/deepcybo/openpi-arx
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 7. 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev

# 8. 应用 transformers 补丁（必须）
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

### 🤖 **在 机器人本机 (arx) 上**

机器人端**不需要**安装 JAX/PyTorch 等大型 GPU 依赖，只需轻量安装推理客户端和通信组件。

```bash
# 1. 创建conda环境
conda create -n openpi_arx python=3.11 -y
conda activate openpi_arx

# 2. 安装 openpi-client（官方远程推理客户端，轻量依赖）
cd packages/openpi-client
pip install -e .
cd ../..

# 3. 安装机器人通信和感知依赖
pip install pyzmq msgpack-python pyrealsense2 opencv-python

# 4. 锁定 numpy 版本（opencv-python 可能拉升 numpy>=2.0，但 openpi-client 要求 <2.0）
pip install "numpy>=1.22.4,<2.0.0"

# 5. 安装系统依赖（如果尚未安装）
sudo apt update
sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

> **注意**：机器人端不需要运行 `uv sync`，这避免了 `pyproject.toml` 中 lerobot 本地路径在不同机器上不一致的问题。

---

## 第三步：在 GPU服务器 (4090) 上 - 生成 norm_stats 和启动策略服务端

### 3.1 准备 pi05_base 基础权重

第一次训练需要下载 pi0.5 的基础权重（约 12GB）。**不建议** 直接从 Google Cloud Storage 下载（速度极慢，约 250KB/s，需要 13 小时）。

**推荐方案**：从其他途径获取后放置到本地：

```bash
# 创建权重目录
mkdir -p /home/deepcybo/models/pi05_models/pi05_base/

# 确保权重结构如下：
/home/deepcybo/models/pi05_models/pi05_base/
├── params/                # 模型参数（约 12GB）
│   ├── _CHECKPOINT_METADATA
│   ├── array_metadatas
│   ├── state.safetensors
│   └── ...
└── assets/                # 模型资产文件
```

### 3.2 计算归一化统计（norm_stats）

训练和推理前必须计算归一化统计：

```bash
cd /home/deepcybo/openpi-arx

# repo_id 指向实际使用的数据集，59D 表示采用全部状态维度
OBS_INDICES=$(seq -s, 1 59) .venv/bin/python scripts/compute_norm_stats.py \
    --config-name pi05_arx
```

输出存储在 `assets/pi05_arx/deepcybo/{dataset_name}/norm_stats_64.json`。

需要将其复制到机器人本机或训练配置中指定。

### 3.3 启动推理服务

#### 选项 A：使用自己微调好的模型

```bash
conda activate openpi_arx
cd /home/deepcybo/openpi-arx

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=path/to/your/checkpoint/30000 \
    --port 8000
```

- `--policy.config`: 配置名称，`pi05_arx` 或 `pi05_arx_lora`
- `--policy.dir`: 你的checkpoint目录路径
- `--port`: 监听端口，默认 `8000`

#### 选项 B：使用单臂模型测试（动作维度不匹配，仅用于验证通信）

```bash
conda activate openpi_arx
cd /home/deepcybo/openpi-arx

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
    --port 8000
```

⚠️ **注意**：`pi05_droid` 是单臂模型（动作 10D），而 ARX LIFT2 是双臂（动作 32D），不能直接使用。此方案仅用于验证客户端与服务器的网络通信。

### 3.4 验证服务启动

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

**后台运行**：使用 `nohup` 确保 SSH 断开后服务继续运行：

```bash
cd /home/deepcybo/openpi-arx

nohup uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=path/to/checkpoint \
    --port 8000 > /tmp/serve_policy.log 2>&1 &

# 查看日志
tail -f /tmp/serve_policy.log
```

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
conda activate openpi_arx
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
conda activate openpi_arx
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
conda activate openpi_arx
cd ~/openpi-arx
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=checkpoints/pi05_arx/your_task/30000 \
    --host 0.0.0.0 --port 8000
```

### 机器人本机 `start_inference.sh`:

```bash
#!/bin/bash
conda activate openpi_arx
cd ~/openpi-arx
python examples/arx/inference_arx.py --config my_deployment.yaml
```

添加执行权限：
```bash
chmod +x start_server.sh start_inference.sh
```

---

## 环境变量参考

| 环境变量 | 示例值 | 说明 | 何时需要 |
|---------|------|------|--------|
| `OBS_INDICES` | `1,2,3,...,59` | 控制 observation.state 的列选择和顺序 | 运行 `train.py` 或 `compute_norm_stats.py` 时必须设置 |
| `WANDB_MODE` | `disabled` | 禁用 Weights & Biases 日志记录 | 没有 W&B 账户时必须设置 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.9` | JAX 可使用的 GPU 显存比例（0.0-1.0） | GPU 显存不足时尝试从 0.75 调至 0.9 |
| `GIT_LFS_SKIP_SMUDGE` | `1` | 安装依赖时跳过 Git LFS 大文件下载 | 首次 `uv sync` 时设置 |

**典型训练命令**：
```bash
OBS_INDICES=$(seq -s, 1 59) WANDB_MODE=disabled \
    .venv/bin/python scripts/train.py pi05_arx \
    --exp-name my_experiment \
    --num-train-steps 10000
```

---

## 已知问题及修复

### Bug 1: reorder_state 操作时机错误

**现象**：运行 `compute_norm_stats.py` 或 `train.py` 时报错 `'TransformedDataset' object has no attribute 'hf_dataset'`

**原因**：`reorder_state` 操作在 `TransformedDataset` 包装之后执行，而包装后的对象没有 `hf_dataset` 属性。

**状态**：✅ 已修复（src/openpi/training/data_loader.py）

修复方案：将 `reorder_state` 移到 `TransformedDataset` 包装之前执行。

### Bug 2: dataset_meta.tasks 数据格式不兼容

**现象**：运行训练时报错 `task_index=0 not found in task mapping` 或 `'NoneType' object is not subscriptable`

**原因**：lerobot 0.3.4 中 `dataset_meta.tasks` 返回 DataFrame 格式，而代码期望 dict 格式。

**状态**：✅ 已修复（src/openpi/training/data_loader.py）

修复方案：添加类型检查，自动转换 DataFrame 为 dict：
```python
if hasattr(tasks, 'iterrows'):  # DataFrame
    tasks = {int(row['task_index']): str(name) for name, row in tasks.iterrows()}
```

---

## 故障排查

### GPU 服务器相关

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| **`uv sync` 依赖冲突** | lerobot 版本错误或 lerobot 路径不对 | 1. 确认 lerobot 已切换到 0.3.4：`git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6` 2. 删除 `.venv` 重新运行：`rm -rf .venv && GIT_LFS_SKIP_SMUDGE=1 uv sync` |
| **transformers 补丁未应用** | pip install 后忘记复制补丁 | 运行：`cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/` |
| **训练时 'hf_dataset' 错误** | reorder_state Bug（已修复） | 确保使用最新的 data_loader.py（见"已知问题"章节） |
| **训练时 task mapping 错误** | dataset_meta.tasks 格式错误（已修复） | 确保使用最新的 data_loader.py（见"已知问题"章节） |
| **Missing `norm_stats.json`** | 未运行 `compute_norm_stats.py` | 设置 OBS_INDICES 后运行：`OBS_INDICES=$(seq -s, 1 59) .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_arx` |
| **权重下载超时或极慢** | 从 Google Cloud Storage 下载 | 不建议直接下载；改用其他途径获取后放置到 `/home/deepcybo/models/pi05_models/pi05_base/` |
| **GPU Out of Memory** | 显存不足 | 1. 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 2. 减小 batch_size 3. 考虑使用 LoRA 微调而非全量微调 |

### 推理服务相关

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| **连接超时** | 防火墙/网络/IP错误 | 1. 检查GPU服务器IP 2. 检查防火墙是否开放8000端口 3. 从机器人端测试：`nc -zv <GPU_IP> 8000` |
| **服务启动后立即退出** | 模型文件默认从 GCS 下载失败 | 事先准备好权重，或在 serve 命令中指定本地路径 |
| **Action dimension mismatch** | 模型动作维度与 ARX 不匹配 | 检查 `ArxOutputs.action_dim = 32`，确认数据集动作维度一致；不要用 pi05_droid（10D 单臂） |

### 机器人端相关

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| **numpy 版本冲突** | opencv-python 拉升 numpy≥2.0 | 执行 `pip install "numpy>=1.22.4,<2.0.0"` 降级 |
| **`No module named openpi_client`** | openpi-client 未安装 | 在机器人本机执行：`cd packages/openpi-client && pip install -e .` |
| **`No module named ros2_bridge.arx_ros2_rpc_client`** | 桥接路径不正确 | 检查 `inference_arx.py` 中的 `_ARX_BRIDGE_PATH` 是否指向正确的 `lerobot_data_collection/arx_vr_data_collection` 路径 |
| **Robot RPC connection failed** | ROS2 RPC服务未启动 | 1. 确认 `arx_ros2_rpc_server.py` 正在运行 2. 检查配置中的IP/端口是否一致 |
| **ZMQ connection refused** | arx_ros2_rpc_server 未启动 | 确认第一个终端已经启动 `python arx_ros2_rpc_server.py`，且IP/端口配置一致 |
| **Images all black** | 相机未连接 | 当前代码用零填充占位，这是预期行为；连接RealSense后替换 `get_observation()` 中的图像获取代码 |

---

## 快速命令总览

### GPU 服务器部署顺序

```bash
# 1. 克隆代码及 lerobot 0.3.4
cd /home/deepcybo
git clone https://github.com/yunlongguo2000/openpi-arx.git
git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout da5d2f3e && cd ..

# 2. 创建环境
conda create -n openpi_arx python=3.11 -y
conda activate openpi_arx
pip install uv

# 3. 安装依赖
cd openpi-arx
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 4. 系统依赖
sudo apt update && sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev

# 5. 应用补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# 6. 下载或获取 pi05_base 权重（约 12GB）
mkdir -p /home/deepcybo/models/pi05_models/pi05_base/params/
# 将权重放置到上述目录

# 7. 计算 norm_stats（需要有数据集）
OBS_INDICES=$(seq -s, 1 59) .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_arx

# 8. 启动服务
nohup .venv/bin/python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=path/to/checkpoint \
    --port 8000 > /tmp/serve.log 2>&1 &
```

### 机器人端部署

```bash
# 1. 克隆代码
git clone --recurse-submodules https://github.com/yunlongguo2000/openpi-arx.git

# 2. 创建环境
conda create -n openpi_arx python=3.11 -y
conda activate openpi_arx

# 3. 安装 openpi-client
cd packages/openpi-client && pip install -e . && cd ../..

# 4. 安装依赖
pip install pyzmq msgpack-python pyrealsense2 opencv-python "numpy>=1.22.4,<2.0.0"
sudo apt update && sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev

# 5. 启动 ROS2 RPC 服务
source /opt/ros/jazzy/setup.bash
cd /home/arx/ARX_new/lerobot_data_collection/arx_vr_data_collection
python ros2_bridge/arx_ros2_rpc_server.py

# 6. （新终端）启动推理
conda activate openpi_arx
cd openpi-arx
python examples/arx/inference_arx.py --config my_deployment.yaml
```

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

## 完整工作流总结

```
┌─────────────────────────────────────────────────────────┐
│  第一步：数据采集（本机）                                  │
│  • 用 VR 遥操作采集 ARX LIFT2 演示数据                    │
│  • 存储为 LeRobot 0.3.4 格式数据集                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  第二步：环境部署                                          │
│  • GPU 服务器：lerobot 0.3.4 + JAX + PyTorch               │
│  • 机器人本机：openpi-client 轻量安装                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  第三步：训练前准备（GPU 服务器）                          │
│  • 获取/下载 pi05_base 权重（12GB）                       │
│  • 计算 norm_stats（OBS_INDICES 全 59D）                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  第四步：模型训练（GPU 服务器）                            │
│  • 基于 pi05_base 微调 pi05_arx 模型                      │
│  • 生成 checkpoint                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  第五步：启动推理服务                                      │
│  • GPU 服务器：serve 微调后的模型                         │
│  • 机器人本机：运行 inference_arx.py                      │
│  • 两端通过 WebSocket 通信                                │
└─────────────────────────────────────────────────────────┘
```

---

## 版本历史

| 日期 | 版本 | 主要变更 |
|------|------|--------|
| 2026-03-23 | v2.0 | 根据实际部署经验更新：lerobot 0.3.4 版本锁定、权重本地获取、Bug 修复说明、环境变量参考 |
| 2026-03-23 | v1.0 | 初版文档 |

---

**最后更新**: 2026-03-23

**维护者**: ARX Robot Team

**相关文档**:
- GPU 服务器配置详记：[docs/GPU_SERVER_SETUP_LOG.md](./GPU_SERVER_SETUP_LOG.md)
- ARX LIFT2 完整部署指南：[docs/PI05_ARX.md](./PI05_ARX.md)
