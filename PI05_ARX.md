# $\pi_{0.5}$ for ARX LIFT2 Dual-Arm Mobile Robot

This document describes how to finetune and deploy $\pi_{0.5}$ on the **ARX LIFT2** dual-arm mobile robot platform using this repository.

---

## Robot Overview

| Item | Spec |
|------|------|
| Platform | ARX LIFT2 mobile base + dual ARX R5/X5 arms |
| State dim | **59D** |
| Action dim | **32D** |
| Cameras | Left wrist (424×240) + Right wrist (424×240) |
| Control freq | 15 Hz |
| Action horizon | 16 steps |

### State Vector (59D)

| Index | Field | Source |
|-------|-------|--------|
| `[0:7]` | `left_joint_pos` | left arm joint positions |
| `[7:14]` | `left_joint_vel` | left arm joint velocities |
| `[14:21]` | `left_joint_cur` | left arm joint currents |
| `[21:28]` | `right_joint_pos` | right arm joint positions |
| `[28:35]` | `right_joint_vel` | right arm joint velocities |
| `[35:42]` | `right_joint_cur` | right arm joint currents |
| `[42:48]` | `left_tcp_pose` | left end-effector pose (x,y,z,roll,pitch,yaw) |
| `[48:54]` | `right_tcp_pose` | right end-effector pose |
| `[54]` | `left_gripper` | left gripper position |
| `[55]` | `right_gripper` | right gripper position |
| `[56:59]` | `chassis` | height, head_yaw, head_pitch |

### Action Vector (32D)

| Index | Field | Notes |
|-------|-------|-------|
| `[0:7]` | `left_joint_pos` | delta joint commands |
| `[7:14]` | `right_joint_pos` | delta joint commands |
| `[14:20]` | `left_tcp_pose` | delta TCP pose (not used in joint-control mode) |
| `[20:26]` | `right_tcp_pose` | delta TCP pose (not used in joint-control mode) |
| `[26]` | `left_gripper` | absolute gripper position |
| `[27]` | `right_gripper` | absolute gripper position |
| `[28:32]` | `chassis (vx, vy, wz, h)` | absolute chassis velocities + height |

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
Make sure `lerobot_data_collection/lerobot_ur_dual_vrteleop/` is present alongside this repo (or adjust `_ARX_BRIDGE_PATH` in `examples/arx/inference_arx.py`).

---

## Key Source Files

| File | Role |
|------|------|
| [`src/openpi/policies/arx_policy.py`](src/openpi/policies/arx_policy.py) | `ArxInputs` / `ArxOutputs` — observation/action transform |
| [`src/openpi/training/config.py`](src/openpi/training/config.py) | `LeRobotArxDataConfig`, `pi05_arx`, `pi05_arx_lora` TrainConfigs |
| [`examples/arx/inference_arx.py`](examples/arx/inference_arx.py) | Inference main loop (robot-side) |
| [`examples/arx/config/cfg_arx_pi.yaml`](examples/arx/config/cfg_arx_pi.yaml) | Deployment config (server address, robot IP, control params) |

---

## Data Config (`LeRobotArxDataConfig`)

Defined in [`config.py`](src/openpi/training/config.py). It handles:

- **Repack transform**: maps LeRobot dataset keys to inference environment keys:
  ```
  observation.state              → state
  observation.images.left_wrist_image  → images.left_wrist
  observation.images.right_wrist_image → images.right_wrist
  action                         → actions
  task                           → prompt
  ```
- **Delta action transform** (`use_delta_joint_actions=True`): applies delta conversion to joints (14D) and TCP (12D); grippers and chassis remain absolute.

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
| Robot RPC connection failed | Ensure `arx_ros2_rpc_server.py` is running on robot and IP/port match `cfg_arx_pi.yaml` |
| Action dimension mismatch | `ArxOutputs.action_dim` must be 32; verify dataset `info.json` matches |
| Training OOM | Use `pi05_arx_lora` config, or set `--fsdp-devices <n>` for multi-GPU |
| Images all black during inference | Expected until RealSense integration is complete; `image_mask=False` tells model to ignore them |
| `ArxROS2RPCClient` import error | Check `_ARX_BRIDGE_PATH` in `inference_arx.py` points to `lerobot_ur_dual_vrteleop/` |
