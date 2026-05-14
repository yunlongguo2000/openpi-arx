# ARX R5 Pi0.5 微调适配 — 完整文档

本文档涵盖 ARX R5 全关节（28D）微调适配的技术详情、训练指标、环境说明及升级迁移指南。

---

## 目录
1. [概述与数据集](#概述与数据集)
2. [维度映射与数据流](#维度映射与数据流)
3. [已修复的关键问题](#已修复的关键问题)
4. [关键指标与训练结果](#关键指标与训练结果)
5. [与 Nero 参考实现的对比](#与-nero-参考实现的对比)
6. [环境说明](#环境说明)
7. [迁移与升级指南](#迁移与升级指南)

---

## 概述与数据集

本文档总结了为完成 `pi05_arx_r5_bottle_handoff` 配置的 ARX R5 微调适配所做的全部修复。该适配使 Pi0.5 能够在 ARX R5 机器人上使用完整的 28D 关节控制空间。

**数据集**: `/vepfs-mlp2/c20250510/250404002/arx_r5_datasets/arx_r5_bottle_handoff/`
- 15 个 episode，10055 帧
- Action 格式: 40D（14D 左臂关节 + 12D 未使用 + 14D 右臂关节 + 2D 夹爪）
- State 格式: 68D（42D joint_pvc + 12D tcp + 12D delta_tcp + 2D 夹爪）
- 训练提取为: 28D actions（14+14 关节 + 2 夹爪）和 56D states（不含 tcp/delta_tcp）

### 两种硬件配置

| 配置 | State 维度 | Action 维度 | TrainConfig |
|------|-----------|------------|-------------|
| ARX LIFT2（移动底盘+双臂） | 59D | 32D | `pi05_arx` |
| ARX R5（纯双臂） | 56D | 28D | `pi05_arx_r5_bottle_handoff` |

**$\pi_{0.5}$ 如何处理 R5（28D/56D）维度:**
- **State (56D):** 连续 state 输入被 tokenize 为离散语言 token，LLM 天然接受可变长度文本，56D state 直接原生传入，无需零填充。
- **Action (28D):** 预训练的 flow-matching expert 具有固定的 32D 输出投影（继承自 LIFT 2 预训练）。训练时 openpi 数据加载器自动将 28D action 填充至 32D；推理时切片回 28D。

---

## 维度映射与数据流

### 训练路径（数据集 → 模型）
```
数据集: 40D actions → 提取索引 (*range(26), 38, 39) → 28D 训练 actions
数据集: 68D states → 提取索引 (*range(54), 66, 67) → 56D 训练 states
训练: 28D actions + 56D states → Pi0.5 模型
模型填充: 28D → 32D（内部）
```

### 推理路径（模型 → 机器人）
```
机器人 state: 56D 观测（joint_pvc + tcp + 夹爪，无速度）
模型输出: 32D actions
变换: 32D → 40D（零填充缺失维度）
机器人执行: 40D actions（提取关节 0-13, 14-25；夹爪 38-39）
```

### 完整数据流图

```
┌─────────────────────────────────────────────────────────────┐
│ 离线训练（数据集 → 模型检查点）                               │
├─────────────────────────────────────────────────────────────┤
│ 1. 加载数据集（40D actions, 68D states）                    │
│ 2. ArxR5FullInputs 提取: 28D actions, 56D states          │
│ 3. Pi0.5 在 (28D, 56D) 上训练，内部 32D 填充               │
│ 4. 保存检查点及 norm_stats                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 在线推理（机器人观测 → Actions）                              │
├─────────────────────────────────────────────────────────────┤
│ 1. Robot.read_policy_observation() 返回 56D state          │
│ 2. ArxR5FullInputs 处理 56D（无需提取）                     │
│ 3. Pi0.5 推理，输出 32D（填充后的 action）                  │
│ 4. ArxR5FullOutputs 变换: 32D → 40D（零填充）              │
│ 5. Robot.apply_action_chunk() 接收 40D actions             │
│ 6. 提取关节 (0-13, 14-25) 和夹爪 (38-39)                   │
│ 7. 在 R5 电机上执行                                         │
└─────────────────────────────────────────────────────────────┘
```

### R5 数据变换链路

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

### Action 格式说明

```
模型变换输出的 40D:
[0-13]   = 左臂关节
[14-25]  = 右臂关节（注意: 7 关节臂提取索引 14-20）
[26-31]  = 左臂 TCP 位置（未使用）
[32-37]  = 右臂 TCP 位置（未使用）
[38]     = 左夹爪
[39]     = 右夹爪
```

---

## 已修复的关键问题

### 1. 维度不匹配: ArxR5FullInputs
**文件**: `src/openpi/policies/arx_policy.py`（第 119-137 行）

**问题**:
- 训练使用数据集中的 68D state
- 推理使用机器人适配器输出的 56D state
- 原始代码仅处理 68D，导致推理时出现 IndexError

**修复**:
```python
# 现在同时处理两种情况:
if raw_state.shape[-1] == 68:
    state = np.take(raw_state, ARX_R5_STATE_INDICES, axis=-1)  # 训练: 提取
elif raw_state.shape[-1] == 56:
    state = raw_state  # 推理: 直接使用
else:
    raise ValueError(f"ARX R5 state 必须为 68D（训练）或 56D（推理），当前为 {raw_state.shape[-1]}D")
```

### 2. 方法签名错误: ArxR5RobotAdapter.apply_action_chunk
**文件**: `src/openpi/arx/arx_r5/arx_r5_robot_adapter.py`（第 118-145 行）

**问题**:
- 方法未接受 `state` 参数（推理流程需要）
- 夹爪索引从 26-27 读取（32D 格式），而非 38-39（40D 格式）
- 关节提取索引错误（应为 0-6 左臂, 14-20 右臂，而非 0-6 左臂, 7-13 右臂）

**修复**:
```python
def apply_action_chunk(self, state: np.ndarray, actions: np.ndarray, *, action_horizon: int):
    """将 40D action 目标应用到 R5 机器人。

    Args:
        state: 56D 机器人 state（用于将来可能的 delta EE 转换）
        actions: [action_horizon, 40] action 数组，来自模型输出变换
        action_horizon: 要执行的 action 数量
    """
    for i in range(action_horizon):
        action = actions[i]
        left_joints = action[0:7]
        right_joints = action[14:21]
        left_gripper = float(action[38])    # 从 26 修正
        right_gripper = float(action[39])   # 从 27 修正
        # ... 其余执行逻辑
```

### 3. 方法调用错误: inference_arx_r5.py
**文件**: `examples/arx_r5/inference_arx_r5.py`（第 160-195 行）

**问题**:
- 调用 `read_policy_observation(..., is_14d=self.is_14d)` 但适配器没有 `is_14d` 参数
- 使用 `obs[state_key]` 搭配错误的 state_key 逻辑，而非 `obs["state"]`

**修复**:
```python
def run(self):
    # ... 初始化 ...
    obs = self.robot.read_policy_observation(
        image_height=self.image_height,
        image_width=self.image_width,
        prompt=self.task_description,
        # 已移除: is_14d=self.is_14d
    )

    result = policy.infer(obs)
    actions = np.asarray(result["actions"], dtype=np.float32)

    # 使用 state 参数应用 actions
    self.robot.apply_action_chunk(
        obs["state"],  # 从 state_key 逻辑改为直接使用
        actions,
        action_horizon=self.action_horizon,
    )
```

### 4. 配置模型名称: cfg_arx_r5_pi.yaml
**文件**: `examples/arx_r5/config/cfg_arx_r5_pi.yaml`（第 9 行）

**问题**: 配置默认为 `pi05_arx_delta_ee`（14D EE 控制），而非 `pi05_arx_r5_bottle_handoff`（28D 全关节）

**修复**:
```yaml
# 更新为:
model:
  name: "pi05_arx_r5_bottle_handoff"  # ARX R5 全关节微调模型（28D/56D）
  checkpoint_dir: "~/models/pi05_arx_r5_bottle_handoff"

robot:
  control_space: "joint"  # 全关节控制（40D → 28D 核心）
```

### 5. ArxR5FullInputs 上的重复 dataclass 装饰器
**文件**: `src/openpi/policies/arx_policy.py`（第 118–119 行）

**问题**: `ArxR5FullInputs` 被两次应用 `@dataclasses.dataclass(frozen=True)`。Python 3.11 会抛出:
```
TypeError: Cannot overwrite attribute __setattr__ in class ArxR5FullInputs
```
这导致无法导入模块，完全阻塞了所有训练和推理。

**修复**: 移除重复的装饰器:
```python
# 之前（损坏）:
@dataclasses.dataclass(frozen=True)
@dataclasses.dataclass(frozen=True)
class ArxR5FullInputs(transforms.DataTransformFn):

# 之后（正确）:
@dataclasses.dataclass(frozen=True)
class ArxR5FullInputs(transforms.DataTransformFn):
```

### 6. max_token_len 对 56D 离散 State 过小
**文件**: `src/openpi/training/config.py` — `pi05_arx_r5_bottle_handoff` TrainConfig

**问题**: pi05 配合 `discrete_state_input=True` 将 56D 本体感知 state 以纯文本数字形式 tokenize。每个浮点数产生约 3–4 个 PaliGemma token，因此 56 维 × ~3.5 tokens = 仅 state 就约 196 tokens。加上任务 prompt，总计约 213 tokens，超过了默认的 `max_token_len=200`。训练管线会静默截断 token，导致 state 信息丢失。

**修复**: 在模型配置中设置 `max_token_len=256`:
```python
model=pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    max_token_len=256,   # 200 对于 56D state + 任务 prompt 来说太小
),
```

### 其他修复
- **重命名控制模式变量**以提高清晰度（`is_14d` → `control_mode`，`full_joint` / `delta_ee`）
- **修复数据集元数据**: `robot_type` → `arx_r5`，修正任务描述
- 所有代码更改保留 14D EE 和 59D LIFT 配置的现有功能

### 修改的文件汇总
1. ✅ `src/openpi/policies/arx_policy.py` - ArxR5FullInputs 56D 支持；移除重复装饰器
2. ✅ `src/openpi/arx/arx_r5/arx_r5_robot_adapter.py` - apply_action_chunk 签名和索引
3. ✅ `examples/arx_r5/inference_arx_r5.py` - 修正方法调用
4. ✅ `examples/arx_r5/config/cfg_arx_r5_pi.yaml` - 模型名称和文档
5. ✅ `src/openpi/training/config.py` - pi05_arx_r5_bottle_handoff 的 max_token_len=256

---

## 关键指标与训练结果

| 指标 | 数值 | 备注 |
|--------|-------|------|
| 数据集大小 | 10055 帧 | 15 个 episode |
| Action 格式 | 40D | 包含未使用的 TCP 维度 |
| 核心 actions | 28D | 14L 关节 + 14R 关节 + 2 夹爪 |
| 训练 state | 56D | joint_pvc + tcp + 夹爪 |
| 推理 state | 56D | 与训练相同（来自机器人 RPC） |
| 模型配置 | pi05_arx_r5_bottle_handoff | 全量微调（非 LoRA） |
| 控制模式 | full_joint | 28D 关节 + 2D 夹爪 |
| 训练步数 | 20000 | 实验名 bottle_handoff_v2 |
| 最终 loss | 0.0021 | 初始 loss 0.6（约 99.7% 下降） |
| 最佳检查点 | 20000 | 13000 也可用 |
| FSDP | 4 GPU | GPU 0-3 |
| WandB | [openpi 项目](https://wandb.ai/yunlong-guo2000-beijing-institute-of-technology/openpi) | — |

### 训练配置概览
- **模型**: pi05（gemma_2b VLM + gemma_300m AE），无 LoRA，全量微调
- **优化器**: AdamW（b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_grad_norm=1.0）
- **学习率调度**: CosineDecay（warmup=1000 步, peak=2.5e-5, decay=2.5e-6）
- **批次**: 32 全局, 4-GPU FSDP
- **EMA**: 衰减 0.99
- **Prompt**: "Hand the bottle from the left arm to the right arm and place it in the basket on the right"

### 验证检查清单

**代码级验证** ✅
- [x] ArxR5FullInputs 处理 68D（训练提取）
- [x] ArxR5FullInputs 处理 56D（推理直接传入）
- [x] ArxR5RobotAdapter.apply_action_chunk 接受 state
- [x] 夹爪索引为 38, 39（而非 26, 27）
- [x] inference_arx_r5.py 使用正确的方法签名
- [x] cfg_arx_r5_pi.yaml 引用 pi05_arx_r5_bottle_handoff
- [x] ArxR5FullInputs 无重复 dataclass 装饰器
- [x] max_token_len=256 可容纳 56D 离散 state

**数据级验证** ✅
- [x] 数据集 norm_stats 已计算（`norm_stats.json` 已写入）
- [x] 完整变换管线已验证: state (56D), actions (16×32D), images (3×HWC)
- [x] 数据集元数据已更正（robot_type, 任务描述）
- [x] 模型训练已完成（20000 步，loss 0.0021）
- [ ] 推理产生 40D actions（待测试）
- [ ] Actions 正确提取为 28D/夹爪（待测试）

### 剩余部署任务

1. **部署到 4090 推理机器**
   - [ ] 设置 Python 3.11 venv，安装所需依赖（jax[cuda12], torch, flax, openpi-client 等）
   - [ ] 从 TOS 下载检查点 `params/`（12GB）
   - [ ] 从 TOS 下载 `norm_stats.json`
   - [ ] 在 4090 机器上配置 `tosutil` 及凭证

2. **空跑推理测试**
   ```bash
   cd /path/to/openpi-arx
   python examples/arx_r5/inference_arx_r5.py \
     --config examples/arx_r5/config/cfg_arx_r5_pi.yaml \
     --robot-mode mock \
     --max-steps 100
   ```

3. **真实机器人测试**
   - [ ] 部署到 R5 机器人，接入真实摄像头
   - [ ] 验证电机指令执行
   - [ ] 验证索引 38-39 的夹爪控制
   - [ ] 检查 episode 完成情况

---

## 与 Nero 参考实现的对比

| 特性 | Nero (14D) | ARX R5 (28D) | 备注 |
|---------|-----------|------------|------|
| State 格式（训练） | 14D | 68D | R5 有速度/电流数据 |
| State 格式（推理） | 14D | 56D | RPC 限制，无速度 |
| 维度差异 | 无 | 有 (68→56) | 通过形状检测处理 |
| 变换复杂度 | 简单 | 复杂 | 训练 ≠ 推理 state 大小 |
| 夹爪索引 | 不适用 | 38, 39 | 40D 格式 |
| 参考匹配 | 参考 | ✓ | 成功借鉴 Nero 模式 |

### 关键区别
- **Nero**: 训练和推理的 state 维度相同，单一路径
- **R5**: 训练 state 更丰富（68D，含速度/电流），推理 state 受限（56D，RPC 限制）
  - 解决方案: 检测维度并处理两种路径
  - 这是相比 Nero 更复杂的根本原因

---

## 环境说明

### vepfs 服务器

位于 `/vepfs-mlp2/c20250510/250404002/venvs/openpi_venv` 的预装 venv 指向
`/root/projects/openpi`（基础 openpi），包含 lerobot 0.1.0（v2.1 数据集格式），
与本文使用的 v3.0 数据集**不兼容**。请始终在命令前添加:

```
PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src
```

这确保 Python 按以下顺序解析:
- `lerobot.*` → `/root/projects/hilserl/lerobot`（v3.0 格式，支持本地绝对路径）
- `openpi.*` → `/root/projects/openpi-arx/src`（ARX 专用策略和配置）

### 为什么需要 56D ≠ 68D 处理
- **训练数据集**: 68D（42D joint_pvc + 12D tcp + 12D delta_tcp + 2D 夹爪）
- **机器人 RPC**: 56D（42D joint_pvc + 12D tcp + 2D 夹爪）— 无速度/电流数据
- **解决方案**: 检测输入形状，如果是 68D 则提取，否则直接使用

### 为什么夹爪索引很重要
- 旧代码: 从索引 26-27 读取（适用于 32D 模型输出，但不适用于 40D 变换）
- 新代码: 从索引 38-39 读取（适用于 40D 变换输出）
- 影响: 旧代码中夹爪指令被发送到了错误的维度

---

## 迁移与升级指南

### 面向从旧代码升级的用户

如果你之前一直在使用 ARX R5 适配，并遇到以下任何问题，你**必须**更新到最新代码（commit 5492c76 或更高版本）。

| 问题 | 症状 | 状态 |
|-------|---------|--------|
| **维度不匹配** | 推理时出现 `IndexError: index 66 is out of bounds` | ✅ 已修复 |
| **夹爪索引错误** | 夹爪指令发送到错误的电机 | ✅ 已修复 |
| **方法签名** | `TypeError: unexpected keyword argument 'is_14d'` | ✅ 已修复 |
| **模型配置** | 加载错误的模型（`pi05_arx_delta_ee` 而非 R5） | ✅ 已修复 |

### 升级步骤

**第 1 步: 拉取最新代码**
```bash
cd /path/to/openpi-arx
git pull origin main
git log --oneline | head -5  # 验证你已有 commit 5492c76+
```

**第 2 步: 验证安装**
```bash
uv sync  # 如有需要，更新依赖
```

**第 3 步: 检查配置**

如果使用 R5，确保 `examples/arx_r5/config/cfg_arx_r5_pi.yaml` 包含:
```yaml
model:
  name: "pi05_arx_r5_bottle_handoff"  # 必须是这个确切的名称
```

**第 4 步: 重新计算 norm_stats（如果进行训练）**
```bash
# 旧的 norm_stats 可能不兼容
uv run scripts/compute_norm_stats.py --config-name pi05_arx_r5_bottle_handoff
```

### 升级后验证

- [ ] 代码位于 commit 5492c76 或更高版本
  ```bash
  git log --oneline | head -1 | grep 5492c76
  ```
- [ ] 无语法错误
  ```bash
  python3 -m py_compile src/openpi/policies/arx_policy.py
  python3 -m py_compile src/openpi/arx/arx_r5/arx_r5_robot_adapter.py
  python3 -m py_compile examples/arx_r5/inference_arx_r5.py
  ```
- [ ] 配置文件正确
  ```bash
  grep "pi05_arx_r5_bottle_handoff" examples/arx_r5/config/cfg_arx_r5_pi.yaml
  ```
- [ ] norm_stats 已重新计算（如果进行全新训练）
  ```bash
  ls -la assets/pi05_arx_r5_bottle_handoff/norm_stats.json
  ```

### 其他配置无破坏性更改
- ✅ LIFT2 训练/推理: **不受影响**
- ✅ 14D EE 模式: **不受影响**
- ✅ 现有 LoRA 配置: **不受影响**

### 故障排除

| 问题 | 解决方法 |
|-------|-----------|
| R5 推理崩溃并出现 IndexError | 确保代码是最新的（commit 5492c76+）。旧代码存在维度不匹配 bug |
| `TypeError: Cannot overwrite attribute __setattr__` | `ArxR5FullInputs` 上重复的 `@dataclasses.dataclass` — 已在新代码中修复 |
| 训练时出现 token 截断警告 | `max_token_len` 太小；对 56D state 的 R5 配置设置 `max_token_len=256` |
| `ModuleNotFoundError: lerobot.datasets` | venv 的 lerobot 使用旧路径；添加 `PYTHONPATH=/root/projects/hilserl/lerobot/src:/root/projects/openpi-arx/src` |
| `HFValidationError: Repo id must be in the form...` | lerobot 0.1.0 不支持绝对路径；按上述方式设置 PYTHONPATH 以使用 hilserl lerobot |

---

## 相关文档

- **README.md** — 训练和推理完整说明（含训练配置参考）
- **代码注释** — 修复的内联文档

---

**最后更新**: 2026-05-14
**适配范围**: pi05_arx_r5_bottle_handoff（28D 全关节 + 2D 夹爪）
**状态**: 训练已完成 ✅；部署到 4090 进行中 ⏳
