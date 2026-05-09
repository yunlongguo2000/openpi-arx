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

**1. Compute normalization statistics:**

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx_r5_bottle_handoff
```

**2. Start training:**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_arx_r5_bottle_handoff \
    --exp-name=arx_r5_bottle_handoff_v01 --overwrite
```

> **Note on R5 Adaptation (May 2026)**: The ARX R5 adaptation has been verified and fixed. See [ADAPTATION_FIXES_SUMMARY.md](ADAPTATION_FIXES_SUMMARY.md) for details on critical fixes to dimension handling, gripper indices, and method signatures.

### Configuration Notes

Edit `TrainConfig` in [`config.py`](src/openpi/training/config.py) to set:
- `repo_id` — your HuggingFace/local LeRobot dataset path
- `weight_loader` — base checkpoint path (default: `gs://openpi-assets/checkpoints/pi05_base/params`)
- `num_train_steps` — default 30,000
- `batch_size` — default 32

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
| Action dimension mismatch (R5) | For R5, model outputs 32D (padded); adapter transforms to 40D (indices 38,39 for grippers). See [ADAPTATION_FIXES_SUMMARY.md](ADAPTATION_FIXES_SUMMARY.md) |
| Training OOM | Use `pi05_arx_lora` config, or set `--fsdp-devices <n>` for multi-GPU |
| Images all black during inference | Expected until RealSense integration is complete; `image_mask=False` tells model to ignore them |
| `ArxROS2RPCClient` import error | Check `_ARX_BRIDGE_PATH` in `inference_arx.py` points to `arx_vr_data_collection/` |
| R5 inference crashes with IndexError | Ensure code is up-to-date (commit 5492c76+). Old code has dimension mismatch bug. See [ADAPTATION_FIXES_SUMMARY.md](ADAPTATION_FIXES_SUMMARY.md) |
