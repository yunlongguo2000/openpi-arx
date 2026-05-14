# $\pi_{0.5}$ for ARX LIFT2 Dual-Arm Mobile Robot

This document describes how to finetune and deploy $\pi_{0.5}$ on the **ARX LIFT2** dual-arm mobile robot platform using this repository.

---

## Robot Overview: Two Distinct Protocols

The openpi-arx repository supports two primary hardware configurations for the ARX family. While both can be finetuned using the same $\pi_{0.5}$ base model, their physical differences require distinct observation and action protocols.

### 1. ARX LIFT 2 (Mobile Base + Dual Arms)
This is the default, feature-complete configuration that includes chassis and lift control.

| Item | Spec |
|------|------|
| Platform | ARX LIFT2 mobile base + dual ARX R5/X5 arms |
| State dim | **59D** (Joints PVC, TCP, Grippers, Chassis) |
| Action dim | **32D** (Joints, TCP, Grippers, Chassis) |
| TrainConfig | `pi05_arx` / `LeRobotArxLift2FullDataConfig` |

### 2. ARX R5 (Pure Dual-Arm)
This configuration is specifically optimized for datasets collected on pure dual-arm setups without a mobile base. It explicitly removes chassis-related dimensions to provide a clean, noise-free learning environment for the arms.

| Item | Spec |
|------|------|
| Platform | ARX R5 pure dual-arm setup (desktop) |
| State dim | **56D** (42D Joint PVC + 12D TCP Pose + 2D Grippers) |
| Action dim | **28D** (14D Joint Targets + 12D TCP Targets + 2D Gripper Commands) |
| TrainConfig | `pi05_arx_r5_bottle_handoff` / `LeRobotArxR5FullDataConfig` |

**How $\pi_{0.5}$ handles the R5 (28D/56D) dimensions:**
- **State (56D):** In $\pi_{0.5}$, continuous state inputs are tokenized into discrete language tokens (e.g., `State: 142 201 56...;`). The LLM naturally accepts variable-length text, so the 56D state is passed natively without any zero-padding.
- **Action (28D):** The pre-trained flow-matching expert has a fixed 32D output projection (inherited from the LIFT 2 pre-training). During training, the openpi data loader automatically pads the 28D action targets to 32D using the internal `PadStatesAndActions` transform. During inference, we slice the 32D prediction back to 28D for the robot.

---

## Environment Setup

### 1. Clone the repository

```bash
git clone --recurse-submodules <this-repo-url>
# Or if already cloned:
git submodule update --init --recursive
```

### 2. Create environment and install dependencies

```bash
conda create -n openpi_arx python=3.11
conda activate openpi_arx
pip install uv
```

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Install system packages:

```bash
apt update
apt install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev
```

### 3. Additional dependencies

The inference script depends on `ArxROS2RPCClient` (ZMQ + msgpack bridge to the robot ROS2 node).
Make sure `lerobot_data_collection/arx_vr_data_collection/` is present alongside this repo (or adjust `_ARX_BRIDGE_PATH` in `examples/arx/inference_arx.py`).

---

## Key Source Files

| File | Role |
|------|------|
| [`src/openpi/policies/arx_policy.py`](src/openpi/policies/arx_policy.py) | `ArxInputs` / `ArxOutputs` — observation/action transform |
| [`src/openpi/training/config.py`](src/openpi/training/config.py) | `LeRobotArxLift2FullDataConfig`, `pi05_arx`, `pi05_arx_lora` TrainConfigs |
| [`examples/arx/inference_arx.py`](examples/arx/inference_arx.py) | Inference main loop (robot-side) |
| [`examples/arx/config/cfg_arx_pi.yaml`](examples/arx/config/cfg_arx_pi.yaml) | Deployment config (server address, robot IP, control params) |

---

## Data Configs (`LeRobotArxLift2FullDataConfig` & `LeRobotArxR5FullDataConfig`)

Defined in [`config.py`](src/openpi/training/config.py), these classes handle adapting datasets to the required model formats.

### For ARX LIFT 2 (`LeRobotArxLift2FullDataConfig`)
- **Repack transform**: maps LeRobot dataset keys to inference environment keys.
- **Data transform**: uses `ArxLift2FullInputs` (pads to 59D state) and `ArxLift2FullOutputs` (returns 32D action).
- **Delta action transform** (`use_delta_joint_actions=True`): applies delta conversion to joints (14D) and TCP (12D); grippers and chassis remain absolute.

### For ARX R5 (`LeRobotArxR5FullDataConfig`)
- **Repack transform**: maps LeRobot dataset keys to inference environment keys.
- **Data transform**: uses `ArxR5FullInputs` (extracts pure 56D state) and `ArxR5FullOutputs` (maps 32D prediction back to 28D action).
- **Delta action transform**: Not used by default (actions are typically absolute joints/TCP targets).

---

## Training

### Option A — Full finetune (`pi05_arx`)

Uses $\pi_{0.5}$ base with full weight updates.

**1. Compute normalization statistics:**

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx
```

**2. Start training:**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_arx \
    --exp-name=arx_task_v01 --overwrite
```

### Option B — LoRA finetune (`pi05_arx_lora`)

Lower GPU memory requirement (~22.5 GB). Uses LoRA on `paligemma_2b` + `gemma_300m`.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_arx_lora \
    --exp-name=arx_task_lora_v01 --overwrite
```

### Option C — ARX R5 Full Joint finetune (`pi05_arx_r5_bottle_handoff`)

For pure dual-arm datasets without mobile base. Uses $\pi_{0.5}$ base with full weight updates on 28D actions.

> **Environment note**: The venv at `/vepfs-mlp2/c20250510/250404002/venvs/openpi_venv`
> uses lerobot 0.1.0 (v2.1 format) and points to the base `openpi` package. ARX R5 datasets
> are in lerobot **v3.0 format**. Always set `PYTHONPATH` as shown below to override both.

**1. Compute normalization statistics:**

```bash
cd /root/projects/openpi-arx
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src \
  /vepfs-mlp2/c20250510/250404002/venvs/openpi_venv/bin/python \
  scripts/compute_norm_stats.py --config-name pi05_arx_r5_bottle_handoff
```

Norm stats are saved next to the dataset:
```
/vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/norm_stats.json
```

**2. Start training:**

```bash
cd /root/projects/openpi-arx
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
  /vepfs-mlp2/c20250510/250404002/venvs/openpi_venv/bin/python \
  scripts/train.py pi05_arx_r5_bottle_handoff \
  --exp_name bottle_handoff_v1 \
  --checkpoint_base_dir /vepfs-mlp2/c20250510/250404002/checkpoints
```

> **Status (2026-05-11)**: norm stats computed ✅, all code bugs fixed ✅, training ready to launch.

### Training Configuration Reference

All `TrainConfig` parameters are defined in [`config.py`](src/openpi/training/config.py). Below is the comprehensive reference.

#### Model Architecture

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

`Variant` 可选值（定义在 `gemma.py:55`）：
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

`freeze_filter` 参数（默认 `nnx.Nothing`）由 `Pi0Config.get_freeze_filter()` 自动生成：若 variant 含 `_lora`，freeze 对应模块的 base weights，但排除 LoRA 参数（`.*lora.*`）使其仍可训练。`trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))`。

**项目中各配置的微调方式：**

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

**R5 数据变换链路：**

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

## Inference

Inference uses a **Policy Server + Robot Client** split:

```
[GPU Server]  serve_policy.py  ←→  WebSocket  ←→  [Robot]  inference_arx.py
                                                      ↕ ZMQ/msgpack
                                                  arx_ros2_rpc_server.py
```

### Step 1 — Launch Policy Server (GPU machine)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_arx \
    --policy.dir=checkpoints/pi05_arx/arx_task_v01/30000
```

The server listens on port **8000** by default.

### Step 2 — Edit deployment config (robot machine)

Edit [`examples/arx/config/cfg_arx_pi.yaml`](examples/arx/config/cfg_arx_pi.yaml):

```yaml
policy_server:
  host: "<GPU_SERVER_IP>"
  port: 8000

robot:
  ip: "<ROBOT_IP>"         # e.g. 192.168.1.100
  port: 4242

task_instruction: "pick up the object"

control:
  action_horizon: 16
  control_freq: 15         # Hz, must match dataset FPS
```

### Step 3 — Run inference (robot machine)

```bash
cd /path/to/openpi-arx
python examples/arx/inference_arx.py \
    --config examples/arx/config/cfg_arx_pi.yaml
```

The inference loop will:
1. Read 59D state from the robot via RPC
2. Capture wrist camera images (placeholder zeros until cameras are connected)
3. Send observation to Policy Server via WebSocket
4. Receive 32D × 16-step action chunk
5. Execute actions at 15 Hz with safety heartbeat

> **Note**: Camera integration is currently stubbed with zero images. Connect RealSense cameras and replace the placeholder in `ArxInference.get_observation()` to enable visual input.

---

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| `norm_stats` missing on robot machine | Copy `norm_stats.json` from GPU machine after running `compute_norm_stats.py` |
| Robot RPC connection failed | Ensure `arx_ros2_rpc_server.py` is running on robot and IP/port match config file |
| Action dimension mismatch (LIFT2) | `ArxOutputs.action_dim` must be 32; verify dataset `info.json` matches |
| Action dimension mismatch (R5) | For R5, model outputs 32D (padded); adapter transforms to 40D (indices 38,39 for grippers). See [ADAPTATION.md](ADAPTATION_FIXES_SUMMARY.md) |
| Training OOM | Use `pi05_arx_lora` config, or set `--fsdp-devices <n>` for multi-GPU |
| Images all black during inference | Expected until RealSense integration is complete; `image_mask=False` tells model to ignore them |
| `ArxROS2RPCClient` import error | Check `_ARX_BRIDGE_PATH` in `inference_arx.py` points to `arx_vr_data_collection/` |
| R5 inference crashes with IndexError | Ensure code is up-to-date (commit 5492c76+). Old code has dimension mismatch bug. See [ADAPTATION.md](ADAPTATION_FIXES_SUMMARY.md) |
| `TypeError: Cannot overwrite attribute __setattr__` | Duplicate `@dataclasses.dataclass` on `ArxR5FullInputs` — fixed in latest code |
| Token truncation warning during training | `max_token_len` too small; set `max_token_len=256` for 56D-state R5 configs |
| `ModuleNotFoundError: lerobot.datasets` | venv lerobot uses old path; add `PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src` |
| `HFValidationError: Repo id must be in the form...` | lerobot 0.1.0 doesn't support absolute paths; set PYTHONPATH as above to use hilserl lerobot |
