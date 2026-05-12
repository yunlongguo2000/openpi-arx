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

### Training Pipeline
- [x] **Norm stats computed** for `arx_r5_bottle_handoff` dataset
  - Output: `/vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/norm_stats.json`
  - Full data pipeline verified end-to-end: state (56D), actions (16×32D padded), images ✓
- [x] **Training-ready**: all blockers resolved, training command verified

### Documentation
- [x] Created ADAPTATION_FIXES_SUMMARY.md detailing all changes
- [x] Documented dimension mapping (40D↔28D actions, 68D↔56D states)
- [x] Data flow diagrams showing training vs inference paths
- [x] Comparison with Nero (14D) reference implementation

### Critical Data Flow
- [x] Training path: Dataset (40D actions/68D states) → Extract (28D/56D) → Model
- [x] Inference path: Robot (56D state) → Direct pass → Model → Transform (40D) → Execute

## ⏳ Remaining Tasks (Before Deployment)

### 1. ~~Compute norm_stats~~ ✅ Done
```
Output: /vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/norm_stats.json
```

### 2. Train Model ⏳
```bash
cd /root/projects/openpi-arx
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src \
  /vepfs-mlp2/c20250510/250404002/venvs/openpi_venv/bin/python \
  scripts/train.py pi05_arx_r5_bottle_handoff \
  --exp_name bottle_handoff_v1 \
  --checkpoint_base_dir /vepfs-mlp2/c20250510/250404002/checkpoints
```
**Status**: All blockers resolved; ready to launch

### 3. Dry-Run Inference Test
```bash
uv run python3 examples/arx_r5/inference_arx_r5.py \
  --config examples/arx_r5/config/cfg_arx_r5_pi.yaml \
  --robot-mode mock \
  --max-steps 100
```
**Status**: Blocked pending trained checkpoint

### 4. Live Robot Testing
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
- [ ] Model training completes (infrastructure: required)
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
| Model config | pi05_arx_r5_bottle_handoff | Fine-tuned baseline |
| Control space | Full joint | 28D (not 14D EE) |

## 🎯 Comparison Matrix

| Feature | Nero (14D) | ARX R5 (28D) | Note |
|---------|-----------|------------|------|
| State format (training) | 14D | 68D | R5 has velocity/current data |
| State format (inference) | 14D | 56D | RPC-limited, no velocities |
| Dimension divergence | None | Yes (68→56) | Handled by shape detection |
| Transform complexity | Simple | Complex | Training ≠ inference state size |
| Gripper indices | N/A | 38, 39 | 40D format |
| Reference match | Reference | ✓ | Adapts Nero pattern successfully |

## 💡 Next Actions

1. **Immediate**: Launch training with the command in Section 2 above
2. **Short-term**: Deploy and test on actual R5 robot
3. **Medium-term**: Validate task performance (bottle handoff success rate)

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

---
**Last Updated**: 2026-05-11
**Adaptation Scope**: pi05_arx_r5_bottle_handoff (28D full joint + 2D gripper)
**Status**: Code + data pipeline complete; training ready to launch
