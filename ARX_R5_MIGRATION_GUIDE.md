# ARX R5 Pi0.5 Adaptation - Migration & Update Guide

## For Users Upgrading from Old Code

If you have been working with the ARX R5 adaptation and experience any of the following issues, you **must update** to the latest code (commit 5492c76 or later).

### Critical Issues Fixed (May 2026)

| Issue | Symptom | Status |
|-------|---------|--------|
| **Dimension Mismatch** | `IndexError: index 66 is out of bounds` during inference | ✅ Fixed |
| **Gripper Index Error** | Gripper commands go to wrong motors | ✅ Fixed |
| **Method Signature** | `TypeError: unexpected keyword argument 'is_14d'` | ✅ Fixed |
| **Model Config** | Loads wrong model (`pi05_arx_delta_ee` instead of R5) | ✅ Fixed |

### What Changed

#### 1. Code Changes
- **src/openpi/policies/arx_policy.py** - ArxR5FullInputs now detects 68D vs 56D
- **src/openpi/arx/arx_r5/arx_r5_robot_adapter.py** - Gripper indices corrected to 38,39
- **examples/arx_r5/inference_arx_r5.py** - Method calls fixed
- **examples/arx_r5/config/cfg_arx_r5_pi.yaml** - Model name updated

#### 2. Update Steps

**Step 1: Pull latest code**
```bash
cd /path/to/openpi-arx
git pull origin main
git log --oneline | head -5  # Verify you have commit 5492c76+
```

**Step 2: Verify installation**
```bash
uv sync  # Update dependencies if needed
```

**Step 3: Check configuration**
- If using R5, ensure `examples/arx_r5/config/cfg_arx_r5_pi.yaml` has:
  ```yaml
  model:
    name: "pi05_arx_r5_bottle_handoff"  # Must be this exact name
  ```

**Step 4: Recompute norm_stats (if training)**
```bash
# Old norm_stats may be incompatible
uv run scripts/compute_norm_stats.py --config-name pi05_arx_r5_bottle_handoff
```

#### 3. No Breaking Changes for Other Configs
- ✅ LIFT2 training/inference: **unaffected**
- ✅ 14D EE mode: **unaffected**
- ✅ Existing LoRA configs: **unaffected**

---

## Technical Details for Developers

### Problem: Dimension Divergence

The core issue: ARX R5 has different state dimensions in training vs inference.

**Training (Dataset):**
```
68D state = 42D joint_pvc + 12D tcp + 12D delta_tcp + 2D grippers
```

**Inference (Robot RPC):**
```
56D state = 42D joint_pvc + 12D tcp + 2D grippers
(Robot doesn't send velocity/current data)
```

**Old Code (BROKEN):**
```python
def __call__(self, data: dict) -> dict:
    raw_state = np.asarray(data["state"], dtype=np.float32)
    # Always assumes 68D, crashes on 56D inference
    state = np.take(raw_state, ARX_R5_STATE_INDICES, axis=-1)
```

**New Code (FIXED):**
```python
def __call__(self, data: dict) -> dict:
    raw_state = np.asarray(data["state"], dtype=np.float32)
    # Shape detection handles both paths
    if raw_state.shape[-1] == 68:
        state = np.take(raw_state, ARX_R5_STATE_INDICES, axis=-1)
    elif raw_state.shape[-1] == 56:
        state = raw_state  # Use directly
    else:
        raise ValueError(...)
```

### Problem: Action Dimension Confusion

The 40D action format has historically been confusing:

**Training:**
```
Dataset 40D = 14D left joints + 12D unused + 14D right joints + 2D grippers
```

**Model output (32D, padded):**
```
Model 32D (padded by Pi0.5 from 28D)
```

**Inference transform (40D):**
```
40D = [0-13] left, [14-25] right, [26-31] left TCP, [32-37] right TCP, [38-39] grippers
```

**Old Code (BROKEN):**
```python
left_gripper = float(action[26])  # WRONG! This is left TCP, not gripper
right_gripper = float(action[27]) # WRONG! This is left TCP, not gripper
```

**New Code (FIXED):**
```python
left_gripper = float(action[38])  # CORRECT for 40D format
right_gripper = float(action[39]) # CORRECT for 40D format
```

---

## Verification Checklist

After updating, verify:

- [ ] Code is at commit 5492c76 or later
  ```bash
  git log --oneline | head -1 | grep 5492c76
  ```

- [ ] No syntax errors
  ```bash
  python3 -m py_compile src/openpi/policies/arx_policy.py
  python3 -m py_compile src/openpi/arx/arx_r5/arx_r5_robot_adapter.py
  python3 -m py_compile examples/arx_r5/inference_arx_r5.py
  ```

- [ ] Config file is correct
  ```bash
  grep "pi05_arx_r5_bottle_handoff" examples/arx_r5/config/cfg_arx_r5_pi.yaml
  ```

- [ ] norm_stats recomputed (if doing fresh training)
  ```bash
  ls -la assets/pi05_arx_r5_bottle_handoff/norm_stats.json
  ```

---

## Documentation References

- **ADAPTATION_FIXES_SUMMARY.md** - Detailed technical changes
- **ADAPTATION_STATUS.md** - Verification checklist and metrics
- **PI05_ARX.md** - Updated training and inference instructions
- **Code Comments** - Inline documentation of fixes

---

## Support

If you encounter issues after updating:

1. Check [PI05_ARX.md Troubleshooting](PI05_ARX.md#troubleshooting)
2. Review [ADAPTATION_FIXES_SUMMARY.md](ADAPTATION_FIXES_SUMMARY.md)
3. Verify your code is at commit 5492c76+
4. Check that configuration matches examples in [cfg_arx_r5_pi.yaml](examples/arx_r5/config/cfg_arx_r5_pi.yaml)

---

**Last Updated:** May 9, 2026  
**Status:** All ARX R5 critical issues resolved and verified
