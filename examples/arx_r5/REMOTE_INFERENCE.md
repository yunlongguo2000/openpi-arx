# π₀.₅ 模型在 ARX R5 上的远程推理部署方案 (2026年5月版)

本文档描述如何使用 **分离式部署**：在 NVIDIA RTX 4090 机器上运行模型推理（Policy Server），机器人端运行推理客户端，两者通过网络通信。

> **更新说明**：本指南已根据 2026-05-14 的成功实验数据更新，采用了 **异步推理 (Async Inference)** 和 **平滑缓冲 (ActionBuffer)** 技术，解决了动作卡顿问题。

---

## 1. 架构概述

```
┌─────────────────────────┐      WebSocket       ┌─────────────────────────┐
│  GPU服务器 (RTX 4090)   │ ←──────────────────→ │  机器人端 (机载/中控)    │
│  • 运行 Policy Server   │       (Port 8000)      │  • 采集 RealSense 图像   │
│  • 加载 Pi0.5 权重      │                       │  • 获取机器人 56D 状态   │
│  • 计算 40D 动作序列    │                       │  • 15Hz 平滑执行动作     │
└─────────────────────────┘                       └─────────────────────────┘
```

**关键改进**：
- **异步推理**：推理客户端在后台请求模型，前台以恒定 15Hz 频率从缓冲区提取动作，彻底消除“突进-停顿”现象。
- **只读模式**：支持 `mode: "read_only"`，可在不实际移动机器人的情况下验证全链路逻辑。

---

## 2. 环境安装 (基于 Conda)

根据 `DEPLOYMENT_LOG.md` 验证，必须严格锁定以下版本以确保 JAX 和 PyTorch 兼容：

### 🖥️ GPU 服务器与客户端通用环境

```bash
# 1. 创建环境
conda create -n openpi_arx python=3.11 -y
conda activate openpi_arx

# 2. 安装核心 ML 框架 (严格锁定版本)
pip install "numpy>=1.26.0,<2.0.0" "ml-dtypes==0.4.1" "jax[cuda12]==0.5.3" "flax==0.10.2"
pip install "torch==2.7.1" "transformers==4.53.2"

# 3. 安装其他依赖
pip install opencv-python pillow sentencepiece wandb polars pyrealsense2 zerorpc gym-aloha
pip install -e /home/yunlong/lerobot

# 4. 安装 OpenPi 及其客户端
cd /home/yunlong/ARX_new/openpi-arx
pip install -e packages/openpi-client
pip install -e . --no-deps

# 5. 应用 Transformers 补丁 (关键：用于支持特定的模型加载逻辑)
# 获取 conda 环境路径
CONDA_PREFIX=$(conda info --base)/envs/openpi_arx
cp -r src/openpi/models_pytorch/transformers_replace/* $CONDA_PREFIX/lib/python3.11/site-packages/transformers/
```

---

## 3. 模型准备 (GPU 服务器)

推理前需要确保 Checkpoint 结构正确，且包含必要的资产文件。

### 3.1 权重目录结构
确保你的模型目录如下所示（以 `bottle_handoff_v2/13000` 为例）：
```text
checkpoints/bottle_handoff_v2/13000/
├── params/                # JAX 模型参数目录
│   ├── _CHECKPOINT_METADATA
│   ├── array_metadatas
│   └── ...
└── assets/                # 归一化统计资产
    └── arx_r5_bottle_handoff/
        └── norm_stats.json
```

### 3.2 手动准备 assets (如果缺失)
如果 `params` 目录下没有 `assets`，需要手动创建并放入 `norm_stats.json`，否则 Policy Server 启动时会报错找不到归一化参数。

---

## 4. 部署步骤 (三部曲)

为了确保任务成功（如瓶子交接），必须遵循 **安全复位 -> 姿态初始化 -> 推理** 的流程。

### 第一步：启动 Policy Server (GPU 服务器)

```bash
conda activate openpi_arx
cd /home/yunlong/ARX_new
python openpi-arx/scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_arx_r5_bottle_handoff \
    --policy.dir pi05_deploy/checkpoints/bottle_handoff_v2/13000
```
- **显存占用**: 约 18 GB。
- **稳态延迟**: 约 80-100ms。

### 第二步：机器人初始化 (机器人端)

```bash
# 1. 安全复位到零位
python pi05_deploy/reset_to_zero.py

# 2. 移动到操作姿态 (训练均值姿态)
python pi05_deploy/init_to_operating_pose.py
```

### 第三步：启动推理客户端

确保 `openpi-arx/examples/arx_r5/config/cfg_arx_r5_pi.yaml` 中的 IP 地址配置正确。

```bash
python openpi-arx/examples/arx_r5/inference_arx_r5.py \
    --config openpi-arx/examples/arx_r5/config/cfg_arx_r5_pi.yaml
```

---

## 5. 关键配置说明

在 `cfg_arx_r5_pi.yaml` 中，以下参数至关重要：

- `mode: "execute"`: 正常运行。
- `mode: "read_only"`: **推荐测试用**。只推理，不下发指令。
- `head_serial: "218622271302"`: 确保序列号与物理连接的 D405 一致。
- `task_description`: 推荐使用详尽的任务描述。

---

## 6. 常见问题排查 (FAQ)

### Q1: 机器人动作卡顿？
**A**: 检查是否启动了 `ActionBuffer` 平滑逻辑。最新版 `inference_arx_r5.py` 已默认包含此功能。

### Q2: JAX 报错 CudaDevice 找不到？
**A**: 执行 `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 并检查 `jax[cuda12]==0.5.3`。

---

**最后更新**: 2026-05-14  
**维护者**: Yunlong Guo  
**相关日志**: [pi05_deploy/DEPLOYMENT_LOG.md](../../pi05_deploy/DEPLOYMENT_LOG.md)
