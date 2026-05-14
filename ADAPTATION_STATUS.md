# ARX R5 Fine-Tuning Adaptation Status

## ✅ Completed Fixes (May 2026)

### Code Quality
- [x] Fixed ArxR5FullInputs to handle 56D inference states
- [x] Fixed ArxR5RobotAdapter.apply_action_chunk signature and indices
- [x] Fixed inference_arx_r5.py method calls
- [x] Updated cfg_arx_r5_pi.yaml model configuration
- [x] All Python files compile without syntax errors
- [x] Code changes preserve existing functionality for 14D EE and 59D LIFT configs
- [x] **Fixed duplicate `@dataclasses.dataclass(frozen=True)` decorator on `ArxR5FullInputs`** (caused `TypeError: Cannot overwrite attribute __setattr__` at import time)
- [x] **Fixed `max_token_len` too small for 56D discrete state** (increased 200 → 256; 56D state tokenizes to ~213 tokens, exceeding default of 200)
- [x] **Renamed control mode variables** for clarity (`is_14d` → `control_mode`, `full_joint` / `delta_ee`)
- [x] **Fixed dataset metadata**: `robot_type` → `arx_r5`, task description corrected

### Training Pipeline
- [x] **Norm stats computed** for `arx_r5_bottle_handoff` dataset (v2, after task fix)
  - Output: `/vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/norm_stats.json`
  - Full data pipeline verified end-to-end: state (56D), actions (16×32D padded), images ✓
- [x] **Training completed**: `pi05_arx_r5_bottle_handoff`, exp `bottle_handoff_v2`
  - 20,000 steps, 4-GPU FSDP
  - Final loss: **0.0021** (initial: ~0.6, ~99.7% reduction)
  - Checkpoints: 5000, 10000, 13000, 20000
  - WandB: [openpi project](https://wandb.ai/yunlong-guo2000-beijing-institute-of-technology/openpi)
  - Config: full fine-tuning (gemma_2b + gemma_300m, no LoRA), peak_lr=2.5e-5

### Documentation
- [x] Created ADAPTATION_FIXES_SUMMARY.md detailing all changes
- [x] Documented dimension mapping (40D↔28D actions, 68D↔56D states)
- [x] Data flow diagrams showing training vs inference paths
- [x] Comparison with Nero (14D) reference implementation
- [x] **Updated PI05_ARX.md with comprehensive Training Configuration Reference**
  - Model architecture, freeze/LoRA/full fine-tuning variants, optimizer & LR schedule, training control, data config

### Critical Data Flow
- [x] Training path: Dataset (40D actions/68D states) → Extract (28D/56D) → Model
- [x] Inference path: Robot (56D state) → Direct pass → Model → Transform (40D) → Execute

## ⏳ Remaining Tasks (Deployment)

### 1. Deploy to 4090 Inference Machine
- [ ] Set up Python 3.11 venv with required dependencies (jax[cuda12], torch, flax, openpi-client, etc.)
- [ ] Download checkpoint `params/` (12GB) from TOS: `tos://c20250510/yunlong/checkpoints/pi05_arx_r5_bottle_handoff/bottle_handoff_v2/13000/params/`
- [ ] Download `norm_stats.json` from TOS
- [ ] Configure `tosutil` on 4090 machine with credentials

### 2. Dry-Run Inference Test
```bash
cd /path/to/openpi-arx
python examples/arx_r5/inference_arx_r5.py \
  --config examples/arx_r5/config/cfg_arx_r5_pi.yaml \
  --robot-mode mock \
  --max-steps 100
```

### 3. Live Robot Testing
- [ ] Deploy to R5 robot with real camera feeds
- [ ] Verify motor command execution
- [ ] Validate gripper control at indices 38-39
- [ ] Check episode completion

## 🔍 Verification Checklist

### Code-Level Verification ✅
- [x] ArxR5FullInputs handles 68D: ✓ (training extraction)
- [x] ArxR5FullInputs handles 56D: ✓ (inference direct pass)
- [x] ArxR5RobotAdapter.apply_action_chunk accepts state: ✓
- [x] Gripper indices are 38, 39: ✓ (not 26, 27)
- [x] inference_arx_r5.py uses correct method signatures: ✓
- [x] cfg_arx_r5_pi.yaml references pi05_arx_r5_bottle_handoff: ✓
- [x] No duplicate dataclass decorator on ArxR5FullInputs: ✓
- [x] max_token_len=256 accommodates 56D discrete state: ✓

### Data-Level Verification ✅
- [x] norm_stats computed for dataset: ✓ (`norm_stats.json` written)
- [x] Full transform pipeline verified: state (56D), actions (16×32D), images (3×HWC): ✓
- [x] Dataset metadata corrected (robot_type, task description): ✓
- [x] Model training completed (20000 steps, loss 0.0021): ✓
- [ ] Inference produces 40D actions (testing: required)
- [ ] Actions correctly extract to 28D/grippers (testing: required)

## 📊 Key Metrics

| Metric | Value | Note |
|--------|-------|------|
| Dataset size | 10055 frames | 15 episodes |
| Action format | 40D | Includes unused TCP dims |
| Core actions | 28D | 14L joints + 14R joints + 2 grippers |
| Training state | 56D | joint_pvc + tcp + grippers |
| Inference state | 56D | Same as training (from robot RPC) |
| Model config | pi05_arx_r5_bottle_handoff | Full fine-tuning (not LoRA) |
| Control mode | full_joint | 28D joint + 2D grippers |
| Training steps | 20000 | bottle_handoff_v2 |
| Final loss | 0.0021 | Initial loss 0.6 |
| Best checkpoint | 20000 | Also 13000 available |
| FSDP | 4 GPU | GPU 0-3 |

## 🎯 Comparison Matrix

| Feature | Nero (14D) | ARX R5 (28D) | Note |
|---------|-----------|------------|------|
| State format (training) | 14D | 68D | R5 has velocity/current data |
| State format (inference) | 14D | 56D | RPC-limited, no velocities |
| Dimension divergence | None | Yes (68→56) | Handled by shape detection |
| Transform complexity | Simple | Complex | Training ≠ inference state size |
| Gripper indices | N/A | 38, 39 | 40D format |
| Reference match | Reference | ✓ | Adapts Nero pattern successfully |

## 📝 Environment Notes (vepfs server)

The pre-installed venv at `/vepfs-mlp2/c20250510/250404002/venvs/openpi_venv` points to
`/root/projects/openpi` (base openpi) and includes lerobot 0.1.0 (v2.1 dataset format), which
is **incompatible** with the v3.0 datasets used here. Always prepend:

```
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src
```

This ensures Python resolves:
- `lerobot.*` → `/root/projects/hilserl/lerobot` (v3.0 format, local absolute path support)
- `openpi.*` → `/root/projects/openpi-arx/src` (ARX-specific policies and configs)

## 📝 Technical Notes

### Why 56D ≠ 68D Handling Was Necessary
- **Training dataset**: 68D (42D joint_pvc + 12D tcp + 12D delta_tcp + 2D grippers)
- **Robot RPC**: 56D (42D joint_pvc + 12D tcp + 2D grippers) - no velocities/currents
- **Solution**: Detect input shape and extract only if 68D, else use directly

### Action Format Clarification
```
40D from model transform:
[0-13]   = left arm joints
[14-25]  = right arm joints (note: indices 14-20 extracted for 7-joint arms)
[26-31]  = left TCP position (unused)
[32-37]  = right TCP position (unused)
[38]     = left gripper
[39]     = right gripper
```

### Why Gripper Indices Matter
- Old code: read from indices 26-27 (correct for 32D model output, wrong for 40D transform)
- New code: read from indices 38-39 (correct for 40D transform output)
- Impact: Gripper commands were going to wrong dimensions in old code

### Training Config Overview
- **Model**: pi05 (gemma_2b VLM + gemma_300m AE), no LoRA, full fine-tuning
- **Optimizer**: AdamW (b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_grad_norm=1.0)
- **LR Schedule**: CosineDecay (warmup=1000 steps, peak=2.5e-5, decay=2.5e-6)
- **Batch**: 32 global, 4-GPU FSDP
- **EMA**: decay 0.99
- **Prompt**: "Hand the bottle from the left arm to the right arm and place it in the basket on the right"

---
**Last Updated**: 2026-05-13
**Adaptation Scope**: pi05_arx_r5_bottle_handoff (28D full joint + 2D gripper)
**Status**: Training complete ✅; deployment to 4090 in progress ⏳
