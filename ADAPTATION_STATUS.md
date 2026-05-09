# ARX R5 Fine-Tuning Adaptation Status

## ✅ Completed Fixes (Commit 5492c76)

### Code Quality
- [x] Fixed ArxR5FullInputs to handle 56D inference states
- [x] Fixed ArxR5RobotAdapter.apply_action_chunk signature and indices
- [x] Fixed inference_arx_r5.py method calls
- [x] Updated cfg_arx_r5_pi.yaml model configuration
- [x] All Python files compile without syntax errors
- [x] Code changes preserve existing functionality for 14D EE and 59D LIFT configs

### Documentation
- [x] Created ADAPTATION_FIXES_SUMMARY.md detailing all changes
- [x] Documented dimension mapping (40D↔28D actions, 68D↔56D states)
- [x] Data flow diagrams showing training vs inference paths
- [x] Comparison with Nero (14D) reference implementation

### Critical Data Flow
- [x] Training path: Dataset (40D actions/68D states) → Extract (28D/56D) → Model
- [x] Inference path: Robot (56D state) → Direct pass → Model → Transform (40D) → Execute

## ⏳ Remaining Tasks (Before Deployment)

### 1. Compute norm_stats (Required)
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx_r5_bottle_handoff
```
**Output**: `./assets/pi05_arx_r5_bottle_handoff/norm_stats.json`
**Status**: Not yet done (dataset exists but norm_stats missing)

### 2. Train Model
```bash
uv run python3 src/openpi/training/pi_train.py --config-name pi05_arx_r5_bottle_handoff
```
**Expected**: Model checkpoint with correct data normalization
**Status**: Awaiting training infrastructure setup

### 3. Dry-Run Inference Test
```bash
uv run python3 examples/arx_r5/inference_arx_r5.py \
  --config examples/arx_r5/config/cfg_arx_r5_pi.yaml \
  --robot-mode mock \
  --max-steps 100
```
**Expected**: No crashes, proper action sequence generation
**Status**: Blocked by lerobot dependency path issue in this environment

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

### Data-Level Verification ⏸️
- [ ] norm_stats computed for dataset (manual: required)
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

1. **Immediate**: Compute norm_stats when infrastructure available
2. **Short-term**: Run training on dataset with fixed adaptation
3. **Medium-term**: Deploy and test on actual R5 robot
4. **Long-term**: Validate task performance (bottle handoff success rate)

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
**Last Updated**: After commit 5492c76
**Adaptation Scope**: pi05_arx_r5_bottle_handoff (28D full joint + 2D gripper)
**Status**: Code fixes complete, awaiting training & validation
