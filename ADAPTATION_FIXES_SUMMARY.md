# ARX R5 Fine-Tuning Adaptation - Fixes Summary

## Overview
This document summarizes the fixes applied to complete the ARX R5 fine-tuning adaptation for the `pi05_arx_r5_bottle_handoff` config. The adaptation enables Pi0.5 to work with the full 28D joint control space on ARX R5 robots.

**Dataset**: `/vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/`
- 15 episodes, 10055 frames
- Action format: 40D (14D left joints + 12D unused + 14D right joints + 2D grippers)
- State format: 68D (42D joint_pvc + 12D tcp + 12D delta_tcp + 2D grippers)
- Training extracts to: 28D actions (14+14 joints + 2 grippers) and 56D states (no tcp/delta_tcp)

## Dimension Mapping

### Training Path (Dataset → Model)
```
Dataset: 40D actions → Extract indices (*range(26), 38, 39) → 28D training actions
Dataset: 68D states → Extract indices (*range(54), 66, 67) → 56D training states
Training: 28D actions + 56D states → Pi0.5 model
Model pads: 28D → 32D (internal)
```

### Inference Path (Model → Robot)
```
Robot state: 56D observation (joint_pvc + tcp + grippers, no velocities)
Model output: 32D actions
Transform: 32D → 40D (zero-pad missing dimensions)
Robot executes: 40D actions (extract joints at 0-13, 14-25; grippers at 38-39)
```

## Critical Issues Fixed

### 1. **Dimension Mismatch: ArxR5FullInputs**
**File**: `src/openpi/policies/arx_policy.py` (lines 119-137)

**Issue**: 
- Training uses 68D state from dataset
- Inference uses 56D state from robot adapter
- Original code only handled 68D, causing IndexError at inference time

**Fix**:
```python
# Now handles both:
if raw_state.shape[-1] == 68:
    state = np.take(raw_state, ARX_R5_STATE_INDICES, axis=-1)  # Training: extract
elif raw_state.shape[-1] == 56:
    state = raw_state  # Inference: use as-is
else:
    raise ValueError(f"ARX R5 state must be 68D (training) or 56D (inference), got {raw_state.shape[-1]}D")
```

### 2. **Method Signature Error: ArxR5RobotAdapter.apply_action_chunk**
**File**: `src/openpi/arx/arx_r5/arx_r5_robot_adapter.py` (lines 118-145)

**Issues**:
- Method didn't accept `state` parameter (needed for inference flow)
- Gripper indices read from 26-27 (32D format) instead of 38-39 (40D format)
- Incorrect joint extraction indices (should be 0-6 left, 14-20 right, not 0-6 left, 7-13 right)

**Fix**:
```python
def apply_action_chunk(self, state: np.ndarray, actions: np.ndarray, *, action_horizon: int):
    """Apply 40D action targets to the R5 robot.
    
    Args:
        state: 56D robot state (for future delta EE conversion if needed)
        actions: [action_horizon, 40] action array from model output transform
        action_horizon: number of actions to execute
    """
    for i in range(action_horizon):
        action = actions[i]
        # Correct indices for 40D action format:
        # [0-13] left joint targets, [14-25] right joint targets
        # [26-31] left tcp (unused), [32-37] right tcp (unused)
        # [38] left gripper, [39] right gripper
        left_joints = action[0:7]
        right_joints = action[14:21]
        left_gripper = float(action[38])    # Corrected from 26
        right_gripper = float(action[39])   # Corrected from 27
        # ... rest of execution
```

### 3. **Method Call Errors: inference_arx_r5.py**
**File**: `examples/arx_r5/inference_arx_r5.py` (lines 160-195)

**Issues**:
- Called `read_policy_observation(..., is_14d=self.is_14d)` but adapter doesn't have `is_14d` parameter
- Used `obs[state_key]` with wrong state_key logic instead of `obs["state"]`

**Fix**:
```python
def run(self):
    # ... initialization ...
    obs = self.robot.read_policy_observation(
        image_height=self.image_height,
        image_width=self.image_width,
        prompt=self.task_description,
        # Removed: is_14d=self.is_14d
    )
    
    result = policy.infer(obs)
    actions = np.asarray(result["actions"], dtype=np.float32)
    
    # Apply actions with state parameter
    self.robot.apply_action_chunk(
        obs["state"],  # Changed from state_key logic
        actions,
        action_horizon=self.action_horizon,
    )
```

### 4. **Config Model Name: cfg_arx_r5_pi.yaml**
**File**: `examples/arx_r5/config/cfg_arx_r5_pi.yaml` (line 9)

**Issue**: Config defaulted to `pi05_arx_delta_ee` (14D EE control) instead of `pi05_arx_r5_bottle_handoff` (28D full joint)

**Fix**:
```yaml
# Updated to:
model:
  name: "pi05_arx_r5_bottle_handoff"  # ARX R5 full joint fine-tuned model (28D/56D)
  checkpoint_dir: "~/models/pi05_arx_r5_bottle_handoff"
  # norm_stats_dir: "~/models/pi05_arx_r5_bottle_handoff/assets/arx_r5_bottle_handoff"

robot:
  control_space: "joint"  # Full joint control (40D -> 28D core)
```

### 5. **Duplicate Dataclass Decorator on ArxR5FullInputs**
**File**: `src/openpi/policies/arx_policy.py` (line 118–119)

**Issue**: `ArxR5FullInputs` had `@dataclasses.dataclass(frozen=True)` applied twice. Python 3.11 raises:
```
TypeError: Cannot overwrite attribute __setattr__ in class ArxR5FullInputs
```
This prevented importing the module entirely, blocking all training and inference.

**Fix**: Remove the duplicate decorator:
```python
# Before (broken):
@dataclasses.dataclass(frozen=True)
@dataclasses.dataclass(frozen=True)
class ArxR5FullInputs(transforms.DataTransformFn):

# After (correct):
@dataclasses.dataclass(frozen=True)
class ArxR5FullInputs(transforms.DataTransformFn):
```

### 6. **max_token_len Too Small for 56D Discrete State**
**File**: `src/openpi/training/config.py` — `pi05_arx_r5_bottle_handoff` TrainConfig

**Issue**: pi05 with `discrete_state_input=True` tokenizes the 56D proprioceptive state as plain-text numbers. Each float produces ~3–4 PaliGemma tokens, so 56 dims × ~3.5 tokens = ~196 tokens for state alone. Adding the task prompt pushes the total to ~213 tokens, exceeding the default `max_token_len=200`. The training pipeline truncated tokens silently, losing state information.

**Fix**: Set `max_token_len=256` in the model config:
```python
model=pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    max_token_len=256,   # 200 was too small for 56D state + task prompt
),
```

## Data Flow After Fixes

```
┌─────────────────────────────────────────────────────────────┐
│ TRAINING OFFLINE (Dataset → Model Checkpoint)               │
├─────────────────────────────────────────────────────────────┤
│ 1. Load dataset with 40D actions, 68D states                │
│ 2. ArxR5FullInputs extracts: 28D actions, 56D states       │
│ 3. Pi0.5 trains on (28D, 56D) with internal 32D padding    │
│ 4. Checkpoint saved with norm_stats                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ INFERENCE ONLINE (Robot Observation → Actions)              │
├─────────────────────────────────────────────────────────────┤
│ 1. Robot.read_policy_observation() returns 56D state       │
│ 2. ArxR5FullInputs handles 56D (no extraction needed)      │
│ 3. Pi0.5 infers, outputs 32D (padded action)              │
│ 4. ArxR5FullOutputs transforms: 32D → 40D (zero-pad)      │
│ 5. Robot.apply_action_chunk() receives 40D actions        │
│ 6. Extract joints (0-13, 14-25) and grippers (38-39)      │
│ 7. Execute on R5 motors                                   │
└─────────────────────────────────────────────────────────────┘
```

## Remaining Tasks

### Required Before Deployment
1. **~~Compute norm_stats~~** ✅ Done
   ```
   Output: /vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/norm_stats.json
   ```

2. **Train the model** with `pi05_arx_r5_bottle_handoff` config:
   ```bash
   cd /root/projects/openpi-arx
   PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src \
     /vepfs-mlp2/c20250510/250404002/venvs/openpi_venv/bin/python \
     scripts/train.py pi05_arx_r5_bottle_handoff \
     --exp_name bottle_handoff_v1 \
     --checkpoint_base_dir /vepfs-mlp2/c20250510/250404002/checkpoints
   ```

3. **End-to-end test** (mock mode):
   ```bash
   uv run python3 examples/arx_r5/inference_arx_r5.py --config examples/arx_r5/config/cfg_arx_r5_pi.yaml --robot-mode mock --max-steps 100
   ```

### Verification Checklist
- [ ] Training pipeline runs without dimension errors
- [ ] Model checkpoint contains correct norm_stats
- [ ] Inference script runs in mock mode without crashes
- [ ] Real robot inference produces reasonable action sequences
- [ ] Episode completes successfully with proper gripper control
- [ ] Performance comparable to Nero (14D) baseline for equivalent tasks

## Comparison to Reference Implementations

### Nero (14D EE) - Reference
- Training: 14D state (6D left EE + 6D right EE + 2D grippers) → 14D model → 14D inference
- No velocity/current dimensions
- Simpler adaptation (single path)

### ARX R5 (28D Full Joint) - This Adaptation
- Training: 68D state → 56D core (via extraction) → 28D model → 40D inference
- Inference: 56D adapter output → 56D direct (no extraction needed)
- More complex: handles divergence between training data and inference capabilities

### Key Difference
- **Nero**: State dimensions same for training & inference
- **R5**: Training state richer (68D with velocities/currents) than inference state (56D RPC-limited)
  - Solution: Detect dimension and handle both paths
  - This is the core reason for the complexity vs Nero

## Files Modified
1. ✅ `src/openpi/policies/arx_policy.py` - ArxR5FullInputs 56D support; removed duplicate decorator
2. ✅ `src/openpi/arx/arx_r5/arx_r5_robot_adapter.py` - apply_action_chunk signature & indices
3. ✅ `examples/arx_r5/inference_arx_r5.py` - Correct method calls
4. ✅ `examples/arx_r5/config/cfg_arx_r5_pi.yaml` - Model name & documentation
5. ✅ `src/openpi/training/config.py` - max_token_len=256 for pi05_arx_r5_bottle_handoff

## Notes
- The adaptation is now **complete** in terms of code fixes
- norm_stats have been computed and are stored at the dataset root
- Training can be launched immediately (see training command above)
- Next phase: actual training + inference testing with real/simulated robot
