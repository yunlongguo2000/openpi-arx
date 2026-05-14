# $\pi_{0.5}$ 用于 ARX LIFT2 双臂移动机器人

本文档介绍如何使用本仓库在 **ARX LIFT2** 双臂移动机器人平台上微调和部署 $\pi_{0.5}$。

---

## 机器人概览: 两种不同的协议

openpi-arx 仓库支持 ARX 系列的两种主要硬件配置。虽然两者都可以使用相同的 $\pi_{0.5}$ 基础模型进行微调，但它们的物理差异需要不同的观测和动作协议。

### 1. ARX LIFT 2（移动底盘 + 双臂）
这是默认的、功能完整的配置，包含底盘和升降机构控制。

| 项目 | 规格 |
|------|------|
| 平台 | ARX LIFT2 移动底盘 + 双 ARX R5/X5 手臂 |
| State 维度 | **59D**（关节 PVC、TCP、夹爪、底盘） |
| Action 维度 | **32D**（关节、TCP、夹爪、底盘） |
| TrainConfig | `pi05_arx` / `LeRobotArxLift2FullDataConfig` |

### 2. ARX R5（纯双臂）
此配置专门针对在无移动底盘的纯双臂装置上采集的数据集进行了优化。它明确移除了底盘相关维度，为手臂提供干净、无噪声的学习环境。

| 项目 | 规格 |
|------|------|
| 平台 | ARX R5 纯双臂装置（桌面式） |
| State 维度 | **56D**（42D 关节 PVC + 12D TCP 位姿 + 2D 夹爪） |
| Action 维度 | **28D**（14D 关节目标 + 12D TCP 目标 + 2D 夹爪指令） |
| TrainConfig | `pi05_arx_r5_bottle_handoff` / `LeRobotArxR5FullDataConfig` |

**$\pi_{0.5}$ 如何处理 R5（28D/56D）维度:**
- **State (56D):** 在 $\pi_{0.5}$ 中，连续 state 输入被 tokenize 为离散语言 token（例如 `State: 142 201 56...;`）。LLM 天然接受可变长度文本，因此 56D state 直接原生传入，无需零填充。
- **Action (28D):** 预训练的 flow-matching expert 具有固定的 32D 输出投影（继承自 LIFT 2 预训练）。在训练期间，openpi 数据加载器使用内部的 `PadStatesAndActions` 变换自动将 28D action 目标填充至 32D。在推理期间，我们将 32D 预测切片回 28D 以供机器人使用。

---

## 环境设置

### 1. 克隆仓库

```bash
git clone --recurse-submodules <this-repo-url>
# 或者如果已经克隆:
git submodule update --init --recursive
```

### 2. 创建环境并安装依赖

```bash
conda create -n openpi_arx python=3.11
conda activate openpi_arx
pip install uv
```

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

安装系统包:

```bash
apt update
apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

### 3. 额外依赖

推理脚本依赖 `ArxROS2RPCClient`（连接机器人 ROS2 节点的 ZMQ + msgpack 桥接）。
确保 `lerobot_data_collection/arx_vr_data_collection/` 与本仓库位于同一目录下（或调整 `examples/arx/inference_arx.py` 中的 `_ARX_BRIDGE_PATH`）。

---

## 关键源文件

| 文件 | 作用 |
|------|------|
| [`src/openpi/policies/arx_policy.py`](src/openpi/policies/arx_policy.py) | `ArxInputs` / `ArxOutputs` — 观测/动作变换 |
| [`src/openpi/training/config.py`](src/openpi/training/config.py) | `LeRobotArxLift2FullDataConfig`、`pi05_arx`、`pi05_arx_lora` TrainConfig |
| [`examples/arx/inference_arx.py`](examples/arx/inference_arx.py) | 推理主循环（机器人端） |
| [`examples/arx/config/cfg_arx_pi.yaml`](examples/arx/config/cfg_arx_pi.yaml) | 部署配置（服务器地址、机器人 IP、控制参数） |

---

## 数据配置（`LeRobotArxLift2FullDataConfig` 和 `LeRobotArxR5FullDataConfig`）

定义在 [`config.py`](src/openpi/training/config.py) 中，这些类负责将数据集适配为所需的模型格式。

### ARX LIFT 2（`LeRobotArxLift2FullDataConfig`）
- **Repack 变换**: 将 LeRobot 数据集键映射到推理环境键。
- **Data 变换**: 使用 `ArxLift2FullInputs`（填充至 59D state）和 `ArxLift2FullOutputs`（返回 32D action）。
- **Delta action 变换**（`use_delta_joint_actions=True`）: 对关节（14D）和 TCP（12D）应用 delta 转换；夹爪和底盘保持绝对值。

### ARX R5（`LeRobotArxR5FullDataConfig`）
- **Repack 变换**: 将 LeRobot 数据集键映射到推理环境键。
- **Data 变换**: 使用 `ArxR5FullInputs`（提取纯 56D state）和 `ArxR5FullOutputs`（将 32D 预测映射回 28D action）。
- **Delta action 变换**: 默认不使用（actions 通常为绝对关节/TCP 目标）。

---

## 训练

### 方案 A — 全量微调（`pi05_arx`）

使用 $\pi_{0.5}$ 基础模型进行全权重更新。

**1. 计算归一化统计量:**

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx
```

**2. 启动训练:**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_arx \
    --exp-name=arx_task_v01 --overwrite
```

### 方案 B — LoRA 微调（`pi05_arx_lora`）

更低的 GPU 内存需求（约 22.5 GB）。在 `paligemma_2b` + `gemma_300m` 上使用 LoRA。

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_arx_lora \
    --exp-name=arx_task_lora_v01 --overwrite
```

### 方案 C — ARX R5 全关节微调（`pi05_arx_r5_bottle_handoff`）

适用于无移动底盘的纯双臂数据集。使用 $\pi_{0.5}$ 基础模型在 28D actions 上进行全权重更新。

> **环境说明**: 位于 `/vepfs-mlp2/c20250510/250404002/venvs/openpi_venv` 的 venv
> 使用 lerobot 0.1.0（v2.1 格式）并指向基础 `openpi` 包。ARX R5 数据集
> 是 lerobot **v3.0 格式**。请始终按如下所示设置 `PYTHONPATH` 以覆盖两者。

**1. 计算归一化统计量:**

```bash
cd /root/projects/openpi-arx
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src \
  /vepfs-mlp2/c20250510/250404002/venvs/openpi_venv/bin/python \
  scripts/compute_norm_stats.py --config-name pi05_arx_r5_bottle_handoff
```

Norm stats 保存在数据集旁边:
```
/vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/norm_stats.json
```

**2. 启动训练:**

```bash
cd /root/projects/openpi-arx
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  /vepfs-mlp2/c20250510/250404002/venvs/openpi_venv/bin/python \
  scripts/train.py pi05_arx_r5_bottle_handoff \
  --exp_name bottle_handoff_v1 \
  --checkpoint_base_dir /vepfs-mlp2/c20250510/250404002/checkpoints
```

> **状态（2026-05-11）**: norm stats 已计算 ✅，所有代码 bug 已修复 ✅，训练就绪可启动。

### 训练配置参考

所有 `TrainConfig` 参数定义在 [`config.py`](src/openpi/training/config.py) 中。以下是完整的参考。

#### 模型架构

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pi05` | `bool` | `False` | 启用 pi05 模式（离散化状态 token、adaRMSNorm 注入 timestep） |
| `paligemma_variant` | `Variant` | `"gemma_2b"` | VLM backbone 型号（见下方冻结/微调说明） |
| `action_expert_variant` | `Variant` | `"gemma_300m"` | Action Expert 型号 |
| `action_dim` | `int` | `32` | 输出动作维度 |
| `action_horizon` | `int` | `50` | 预测动作序列长度 |
| `max_token_len` | `int` | `200`（pi05: 256） | 离散化状态的最大 token 长度 |
| `dtype` | `str` | `"bfloat16"` | 模型计算精度 |

#### 冻结 / 全量 / LoRA 微调

这是最容易被忽略的配置。冻结由 `paligemma_variant` 和 `action_expert_variant` **隐式控制**，没有显式的 `freeze=True/False` 开关。

`Variant` 可选值（定义在 `gemma.py:55`）:
- `"gemma_2b"` — 全量训练 VLM
- `"gemma_2b_lora"` — 冻结 VLM backbone，只训 LoRA 适配器
- `"gemma_300m"` — 全量训练 Action Expert
- `"gemma_300m_lora"` — 冻结 AE backbone，只训 LoRA 适配器

| paligemma_variant | action_expert_variant | VLM 状态 | Action Expert 状态 |
|---|---|---|---|
| `"gemma_2b"` | `"gemma_300m"` | 全量 | 全量 |
| `"gemma_2b_lora"` | `"gemma_300m_lora"` | 冻结 + LoRA | 冻结 + LoRA |
| `"gemma_2b_lora"` | `"gemma_300m"` | 冻结 + LoRA | 全量 |
| `"gemma_2b"` | `"gemma_300m_lora"` | 全量 | 冻结 + LoRA |

`freeze_filter` 参数（默认 `nnx.Nothing`）由 `Pi0Config.get_freeze_filter()` 自动生成: 若 variant 含 `_lora`，冻结对应模块的 base weights，但排除 LoRA 参数（`.*lora.*`）使其仍可训练。`trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))`。

**项目中各配置的微调方式:**

| 配置名 | 微调方式 | VLM | Action Expert |
|--------|---------|-----|---------------|
| `pi05_arx_r5_bottle_handoff` | **全量** | 全量 | 全量 |
| `pi05_arx` | 全量 | 全量 | 全量 |
| `pi05_arx_delta_ee` | 全量 | 全量 | 全量 |
| `pi05_arx_lora` | LoRA | 冻结 | 冻结 |
| `pi05_full_droid_finetune` | 全量 | 全量 | 全量 |
| `pi05_droid_finetune` | LoRA | 冻结 | 冻结 |
| `pi05_droid_finetune_franka` | LoRA | 冻结 | 冻结 |
| `pi0_libero_low_mem_finetune` | LoRA | 冻结 | 冻结 |
| `pi0_fast_libero_low_mem_finetune` | LoRA | 冻结 | 全量 |

#### 优化器 & 学习率

| 参数 | 子参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `lr_schedule` | `CosineDecaySchedule` | — | 余弦衰减调度 |
| | `warmup_steps` | `1000` | 线性预热步数 |
| | `peak_lr` | `2.5e-5` | 峰值学习率 |
| | `decay_lr` | `2.5e-6` | 最终衰减学习率 |
| `optimizer` | `AdamW` | — | AdamW 优化器 |
| | `b1` | `0.9` | 一阶矩衰减 |
| | `b2` | `0.95` | 二阶矩衰减 |
| | `eps` | `1e-8` | 数值稳定性 |
| | `weight_decay` | `1e-10` | 权重衰减 |
| | `clip_gradient_norm` | `1.0` | 全局梯度裁剪 |

#### 训练控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_steps` | `30000` | 总训练步数 |
| `batch_size` | `32` | 全局 batch size |
| `fsdp_devices` | `1` | FSDP 分片设备数（`>1` 启用多卡 FSDP） |
| `seed` | `42` | 随机种子 |
| `log_interval` | `100` | 每 N 步记录一次指标到 WandB |
| `save_interval` | `1000` | 每 N 步保存 checkpoint |
| `keep_period` | `5000` | 步数为该值整数倍的 checkpoint 不自动删除 |
| `ema_decay` | `0.99` | 指数滑动平均衰减率 |
| `overwrite` | `False` | 覆盖已有 checkpoint 目录 |
| `resume` | `False` | 从最新 checkpoint 恢复训练 |
| `wandb_enabled` | `True` | 启用 WandB 日志 |
| `project_name` | `"openpi"` | WandB 项目名 |
| `num_workers` | `2` | DataLoader 工作进程数 |

#### 数据配置 (DataConfig)

| 参数 | 说明 |
|------|------|
| `repo_id` | 数据集路径（HuggingFace repo ID 或本地绝对路径） |
| `prompt_from_task` | `True` 时从数据集 parquet 的 `task` 列读取 prompt |
| `default_prompt` | 若 `prompt_from_task=False`，使用此固定 prompt |

**R5 数据变换链路:**

```
Dataset (68D state / 40D action) 
  → RepackTransform (key mapping)
  → ArxR5FullJointInputs:
      - 68D state → 56D (via ARX_R5_STATE_INDICES)  [训练]
      - 56D state → 56D (identity)                    [推理]
      - 40D action → 28D (via ARX_R5_ACTION_INDICES)
  → Normalize (norm_stats)
  → Tokenize (discrete state tokens)
  → PadStatesAndActions → 32D action

Inference output:
  32D → ArxR5FullJointOutputs → 40D → Robot adapter
```

---

## 推理

推理采用 **Policy Server + Robot Client** 分离架构:

```
[GPU 服务器]  serve_policy.py  ←→  WebSocket  ←→  [机器人]  inference_arx.py
                                                      ↕ ZMQ/msgpack
                                                  arx_ros2_rpc_server.py
```

### 第 1 步 — 启动 Policy Server（GPU 机器）

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=checkpoints/pi05_arx/arx_task_v01/30000
```

服务器默认监听 **8000** 端口。

### 第 2 步 — 编辑部署配置（机器人机器）

编辑 [`examples/arx/config/cfg_arx_pi.yaml`](examples/arx/config/cfg_arx_pi.yaml):

```yaml
policy_server:
  host: "<GPU_SERVER_IP>"
  port: 8000

robot:
  ip: "<ROBOT_IP>"         # 例如 192.168.1.100
  port: 4242

task_instruction: "pick up the object"

control:
  action_horizon: 16
  control_freq: 15         # Hz，必须匹配数据集 FPS
```

### 第 3 步 — 运行推理（机器人机器）

```bash
cd /path/to/openpi-arx
python examples/arx/inference_arx.py \
    --config examples/arx/config/cfg_arx_pi.yaml
```

推理循环将:
1. 通过 RPC 从机器人读取 59D state
2. 捕获手腕摄像头图像（在摄像头连接之前使用占位零值）
3. 通过 WebSocket 将观测发送到 Policy Server
4. 接收 32D × 16 步 action chunk
5. 以 15 Hz 频率执行动作，并附带安全心跳

> **注意**: 摄像头集成目前使用零值图像占位。连接 RealSense 摄像头并替换 `ArxInference.get_observation()` 中的占位符以启用视觉输入。

---

## 故障排除

| 问题 | 解决方法 |
|-------|-----------|
| 机器人机器上缺少 `norm_stats` | 运行 `compute_norm_stats.py` 后从 GPU 机器复制 `norm_stats.json` |
| 机器人 RPC 连接失败 | 确保 `arx_ros2_rpc_server.py` 在机器人上运行，且 IP/端口与配置文件匹配 |
| Action 维度不匹配（LIFT2） | `ArxOutputs.action_dim` 必须为 32；验证数据集 `info.json` 匹配 |
| Action 维度不匹配（R5） | 对于 R5，模型输出 32D（填充后）；适配器变换为 40D（夹爪索引 38,39）。参见 [ADAPTATION_FIXES_SUMMARY.md](ADAPTATION_FIXES_SUMMARY.md) |
| 训练 OOM | 使用 `pi05_arx_lora` 配置，或设置 `--fsdp-devices <n>` 进行多 GPU |
| 推理时图像全黑 | RealSense 集成完成前为预期行为；`image_mask=False` 告知模型忽略图像 |
| `ArxROS2RPCClient` 导入错误 | 检查 `inference_arx.py` 中的 `_ARX_BRIDGE_PATH` 是否指向 `arx_vr_data_collection/` |
| R5 推理崩溃并出现 IndexError | 确保代码是最新的（commit 5492c76+）。旧代码存在维度不匹配 bug。参见 [ADAPTATION_FIXES_SUMMARY.md](ADAPTATION_FIXES_SUMMARY.md) |
| `TypeError: Cannot overwrite attribute __setattr__` | `ArxR5FullInputs` 上重复的 `@dataclasses.dataclass` — 已在新代码中修复 |
| 训练时出现 token 截断警告 | `max_token_len` 太小；对 56D state 的 R5 配置设置 `max_token_len=256` |
| `ModuleNotFoundError: lerobot.datasets` | venv 的 lerobot 使用旧路径；添加 `PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src` |
| `HFValidationError: Repo id must be in the form...` | lerobot 0.1.0 不支持绝对路径；按上述方式设置 PYTHONPATH 以使用 hilserl lerobot |
