# GPU 服务器 openpi-arx 远程推理环境配置指南

> **文档性质**: 操作记录 + 部署指南
> **记录日期**: 2026-03-23
> **目标服务器**: 192.168.110.26 (用户: deepcybo, GPU: RTX 4090 24GB)
> **本机**: ARX LIFT2 双臂机器人机载电脑 (用户: arx)

---

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. SSH 免密登录配置](#2-ssh-免密登录配置)
- [3. 克隆代码仓库](#3-克隆代码仓库)
- [4. lerobot 版本选择 (关键)](#4-lerobot-版本选择-关键)
- [5. 创建 Conda 环境](#5-创建-conda-环境)
- [6. 安装 Python 依赖](#6-安装-python-依赖)
- [7. 安装系统依赖](#7-安装系统依赖)
- [8. 应用 transformers 补丁](#8-应用-transformers-补丁)
- [9. 验证 GPU 环境](#9-验证-gpu-环境)
- [10. 数据集准备与传输](#10-数据集准备与传输)
- [11. 修改训练配置](#11-修改训练配置)
- [12. 修复代码 Bug](#12-修复代码-bug)
- [13. 计算归一化统计](#13-计算归一化统计)
- [14. 下载 pi05_base 预训练权重](#14-下载-pi05_base-预训练权重)
- [15. 训练尝试与显存分析](#15-训练尝试与显存分析)
- [16. 关键环境变量参考](#16-关键环境变量参考)
- [17. 当前状态与后续计划](#17-当前状态与后续计划)

---

## 1. 背景与目标

### 硬件架构

| 角色 | 设备 | 说明 |
|------|------|------|
| 本机 (机载电脑) | ARX LIFT2 双臂机器人 | 用户 `arx`，负责数据采集和机器人控制 |
| GPU 服务器 | 192.168.110.26 | 用户 `deepcybo`，RTX 4090 24GB，负责模型微调和推理 |

### 目标

在 GPU 服务器上配置 openpi-arx 环境，用于：
- pi0.5 模型微调 (LoRA / 全量)
- 远程推理服务 (供机载电脑调用)

---

## 2. SSH 免密登录配置

在本机 (arx) 上执行，将公钥复制到 GPU 服务器：

```bash
ssh-copy-id deepcybo@192.168.110.26
```

验证：

```bash
ssh deepcybo@192.168.110.26  # 应无需输入密码
```

---

## 3. 克隆代码仓库

SSH 登录到 GPU 服务器后，克隆以下两个仓库：

```bash
# openpi-arx 主仓库 (含子模块)
cd /home/deepcybo
git clone --recurse-submodules https://github.com/yunlongguo2000/openpi-arx.git

# lerobot 数据处理库
git clone https://github.com/huggingface/lerobot.git
```

> **注意**: lerobot 克隆时可能因 GitHub 网络问题失败，需要多次重试。如多次失败可考虑使用代理或从本机 scp 拷贝。

---

## 4. lerobot 版本选择 (关键)

### 问题

lerobot 最新版 (0.5.1) 与 openpi-arx 存在 **双重兼容性冲突**：

| 冲突项 | lerobot 0.5.1 要求 | openpi-arx 要求 |
|--------|-------------------|-----------------|
| Python 版本 | `>=3.12` | `3.11` (conda 环境) |
| huggingface-hub | `>=1.0` | `<1.0` (transformers==4.53.2 依赖) |

### 决策依据

- 本机所有 lerobot 副本 (如 `/home/arx/ARX_new/lerobot/`) 均为 **0.3.4** 版本
- 数据采集和训练推理的 lerobot 版本 **必须一致**，否则数据格式不兼容

### 操作

将 GPU 服务器上的 lerobot 切换到 0.3.4 版本：

```bash
cd /home/deepcybo/lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6  # lerobot 0.3.4
```

---

## 5. 创建 Conda 环境

GPU 服务器上 conda 安装在 `/home/deepcybo/anaconda3/`，但可能未初始化。

```bash
# 初始化 conda (如果 conda 命令不可用)
source /home/deepcybo/anaconda3/etc/profile.d/conda.sh

# 创建 Python 3.11 环境
conda create -n openpi_arx python=3.11 -y
conda activate openpi_arx
```

> **提示**: 如需每次登录自动激活，可将 `source` 行添加到 `~/.bashrc`。

---

## 6. 安装 Python 依赖

### 前置条件

确认 `openpi-arx/pyproject.toml` 中 lerobot 路径已指向 GPU 服务器上的 lerobot 目录：

```toml
# pyproject.toml 中应包含类似配置:
# lerobot 路径指向 /home/deepcybo/lerobot
```

如果路径不匹配，需要先修改 `pyproject.toml`。

### 安装

```bash
pip install uv
cd /home/deepcybo/openpi-arx

# 跳过 Git LFS 大文件下载，加速安装
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

安装完成后共约 **272 个包**，关键组件版本：

| 组件 | 版本 |
|------|------|
| JAX | 0.5.3 |
| PyTorch | 2.7.1+cu126 |
| transformers | 4.53.2 |

---

## 7. 安装系统依赖

```bash
sudo apt update
sudo apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

> **说明**: `libavcodec-dev`、`libavformat-dev`、`libavutil-dev` 通常已预装，主要需要的是 `ffmpeg`。

---

## 8. 应用 transformers 补丁

openpi-arx 对 transformers 库有自定义修改，需要将补丁文件覆盖到虚拟环境中：

```bash
cd /home/deepcybo/openpi-arx
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

> **注意**: 每次重新安装 transformers 后，都需要重新执行此步骤。

---

## 9. 验证 GPU 环境

在 GPU 服务器上进入 Python 环境验证：

```bash
cd /home/deepcybo/openpi-arx
.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import jax
print(f'JAX: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
"
```

预期输出：

```
PyTorch: 2.7.1+cu126
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
JAX: 0.5.3
JAX devices: [CudaDevice(id=0)]
```

---

## 10. 数据集准备与传输

### 数据集存储位置

lerobot 数据集默认存储在 `~/.cache/huggingface/lerobot/` 目录下。

### 本机数据集情况

本机 `/home/arx/.cache/huggingface/lerobot/deepcybo/` 下有 34 个数据集，其中：
- **20 个有效** (包含完整的 `data/` + `videos/` + `parquet` 文件)
- **14 个损坏** (缺失关键文件)

### 选择并传输数据集

选择最大的有效数据集先跑通流程：

- **数据集**: `arx_lift_task_20260319_v31`
- **规模**: 2 episodes, 1798 frames, 约 21MB

从本机传输到 GPU 服务器：

```bash
# 在本机 (arx) 执行
scp -r /home/arx/.cache/huggingface/lerobot/deepcybo/arx_lift_task_20260319_v31 \
    deepcybo@192.168.110.26:/home/deepcybo/.cache/huggingface/lerobot/deepcybo/
```

> **提示**: 传输大量数据集时建议用 `rsync` 代替 `scp`，支持断点续传：
> ```bash
> rsync -avzP /home/arx/.cache/huggingface/lerobot/deepcybo/ \
>     deepcybo@192.168.110.26:/home/deepcybo/.cache/huggingface/lerobot/deepcybo/
> ```

---

## 11. 修改训练配置

修改文件: `src/openpi/training/config.py`

### 11.1 更新数据集 repo_id

将 `pi05_arx` 和 `pi05_arx_lora` 配置中的 `repo_id` 更新为实际使用的数据集：

```python
# 修改前
repo_id = "deepcybo/arx_lift_task_20260312_v03"

# 修改后
repo_id = "deepcybo/arx_lift_task_20260319_v31"
```

### 11.2 更新预训练权重路径

将 `weight_loader` 从 GCS 路径改为本地路径：

```python
# 修改前
weight_loader = "gs://openpi-assets/checkpoints/pi05_base/params"

# 修改后
weight_loader = "/home/deepcybo/models/pi05_models/pi05_base/params"
```

---

## 12. 修复代码 Bug

修改文件: `src/openpi/training/data_loader.py`

### Bug 1: reorder_state 执行时机错误

**现象**: `'TransformedDataset' object has no attribute 'hf_dataset'`

**原因**: `reorder_state` 操作在 `TransformedDataset` 包装之后执行，而 `TransformedDataset` 没有 `hf_dataset` 属性。

**修复**: 将 `reorder_state` 移到 `prompt_from_task` 包装之前执行，并将 `OBS_INDICES` 改为可选 (不设置时不做 reorder)。

### Bug 2: dataset_meta.tasks 返回类型不兼容

**现象**: `task_index=0 not found in task mapping`

**原因**: 在 lerobot 0.3.4 中，`dataset_meta.tasks` 返回的是 `DataFrame` 而非 `dict`，导致 task 查找失败。

**修复**: 添加 DataFrame 到 dict 的转换逻辑：

```python
# 添加类型检查和转换
if isinstance(tasks, pd.DataFrame):
    tasks = dict(zip(tasks['task_index'], tasks['task']))
```

---

## 13. 计算归一化统计

归一化统计 (norm_stats) 是训练和推理的前置依赖。

```bash
cd /home/deepcybo/openpi-arx

OBS_INDICES=$(seq -s, 1 59) .venv/bin/python scripts/compute_norm_stats.py \
    --config-name pi05_arx
```

输出目录:

```
/home/deepcybo/openpi-arx/assets/pi05_arx/deepcybo/arx_lift_task_20260319_v31/
```

> **说明**: `OBS_INDICES=$(seq -s, 1 59)` 表示选择 observation.state 的全部 59 个维度 (索引 1~59)。

---

## 14. 下载 pi05_base 预训练权重

### 方法一: 从 GCS 直接下载 (不推荐)

```bash
# 速度极慢，约 250KB/s，预计 13 小时
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base /home/deepcybo/models/pi05_models/
```

### 方法二: 从其他途径获取 (推荐)

从已有的下载源或其他机器拷贝，放置到以下路径：

```
/home/deepcybo/models/pi05_models/pi05_base/
├── params/        # 模型参数
└── assets/        # 模型资产文件
```

总大小约 **12GB**。

### 验证权重完整性

```bash
ls -la /home/deepcybo/models/pi05_models/pi05_base/params/
ls -la /home/deepcybo/models/pi05_models/pi05_base/assets/
```

确保 `params/` 和 `assets/` 目录都存在且非空。

---

## 15. 训练尝试与显存分析

### 全量微调 (pi05_arx) - RTX 4090 上 OOM

在 RTX 4090 (24GB) 上尝试全量微调 pi0.5 模型：

```bash
cd /home/deepcybo/openpi-arx
WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    .venv/bin/python scripts/train.py --config-name pi05_arx
```

| batch_size | 结果 |
|-----------|------|
| 16 (默认) | OOM |
| 4 | OOM |
| 2 | OOM |

### 显存需求分析

pi0.5 模型约 **3B 参数**，全量微调的显存需求：

| 组件 | 显存估算 |
|------|---------|
| 模型参数 (FP32) | ~6 GB |
| Adam 优化器状态 (2x) | ~12 GB |
| 梯度 | ~6 GB |
| 激活值 (batch_size=2) | ~2-4 GB |
| **合计** | **~26-28 GB** |

RTX 4090 的 24GB 显存不足以进行全量微调。

### 解决方案

| 方案 | 说明 | 推荐度 |
|------|------|--------|
| **LoRA 微调** (`pi05_arx_lora`) | 仅微调少量参数，大幅降低显存需求 | 推荐 |
| 更换 GPU | 使用 A100 40GB/80GB 或 H100 | 视硬件条件 |
| 梯度检查点 | 用计算换显存 | 可与 LoRA 结合 |

---

## 16. 关键环境变量参考

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `OBS_INDICES` | `1,2,3,...,59` | 控制 `observation.state` 的列选择和排序，59D 全选 |
| `WANDB_MODE` | `disabled` | 禁用 Weights & Biases 日志记录 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.95` | JAX 可使用的 GPU 显存比例 (默认 0.75) |
| `GIT_LFS_SKIP_SMUDGE` | `1` | 安装依赖时跳过 Git LFS 大文件下载 |

---

## 17. 当前状态与后续计划

### 已完成

- [x] SSH 免密登录配置
- [x] 代码仓库克隆 (openpi-arx + lerobot 0.3.4)
- [x] Conda 环境创建 (openpi_arx, Python 3.11)
- [x] Python 依赖安装 (272 packages)
- [x] 系统依赖安装 (ffmpeg)
- [x] transformers 补丁应用
- [x] GPU 环境验证 (PyTorch + JAX + CUDA)
- [x] 数据集传输 (arx_lift_task_20260319_v31)
- [x] 训练配置修改 (config.py)
- [x] 代码 Bug 修复 (data_loader.py)
- [x] 归一化统计计算 (norm_stats)
- [x] pi05_base 预训练权重就位 (~12GB)

### 待完成

- [ ] 尝试 LoRA 微调 (`pi05_arx_lora` 配置)
- [ ] 启动远程推理服务
- [ ] 配置本机到 GPU 服务器的推理调用链路
- [ ] 传输更多数据集进行多任务训练

### 全量微调状态

RTX 4090 (24GB) 显存不足，全量微调失败。下一步应优先尝试 **LoRA 微调** 方案。

---

## 附录: GPU 服务器目录结构

```
/home/deepcybo/
├── anaconda3/                          # Conda 安装目录
├── openpi-arx/                         # 主项目
│   ├── .venv/                          # Python 虚拟环境
│   ├── src/openpi/                     # 源代码
│   ├── scripts/                        # 训练/推理脚本
│   └── assets/                         # 归一化统计等资产文件
├── lerobot/                            # lerobot 0.3.4
├── models/
│   └── pi05_models/
│       └── pi05_base/                  # 预训练权重 (~12GB)
│           ├── params/
│           └── assets/
└── .cache/
    └── huggingface/
        └── lerobot/
            └── deepcybo/
                └── arx_lift_task_20260319_v31/  # 训练数据集
```

---

## 附录: 快速部署检查清单 (适用于新 GPU 服务器)

在新的 GPU 服务器上部署时，按以下顺序执行：

1. **SSH 免密登录** -- 从本机 `ssh-copy-id`
2. **克隆仓库** -- openpi-arx (含子模块) + lerobot
3. **lerobot 切换到 0.3.4** -- `git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6`
4. **创建 conda 环境** -- Python 3.11
5. **安装依赖** -- `uv sync` + `uv pip install -e .`
6. **安装系统依赖** -- ffmpeg 等
7. **应用 transformers 补丁** -- 复制覆盖
8. **验证 GPU** -- PyTorch CUDA + JAX CUDA
9. **传输数据集** -- scp/rsync 到 `~/.cache/huggingface/lerobot/`
10. **修改配置** -- `config.py` 中的 repo_id 和 weight_loader 路径
11. **修复 Bug** -- `data_loader.py` (如已合入主分支则跳过)
12. **计算 norm_stats** -- `scripts/compute_norm_stats.py`
13. **准备权重** -- pi05_base (~12GB)
14. **开始训练** -- 根据显存选择全量微调或 LoRA
