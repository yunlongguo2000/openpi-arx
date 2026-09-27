# Pi0.5 推理部署日志

**日期**: 2026-05-14  
**分支**: `yunlong/feature/pi05-deployment`  
**模型**: `pi05_arx_r5_bottle_handoff`  
**机器人**: ARX R5 双臂 (IP: 10.10.10.2:4242)  
**开发机**: RTX 4090 (IP: 10.10.10.100, WiFi: 192.168.110.202)

---

## 1. 环境重建

### 1.1 初始问题

`openpi_arx` conda 环境中 JAX/Flax/Numpy 版本与 `openpi-arx/pyproject.toml` 严重不匹配：

| 包名 | 已安装 | 要求 | 状态 |
|------|--------|------|------|
| jax[cuda12] | 0.10.0 | 0.5.3 | ❌ 跨5个大版本 |
| flax | 0.12.7 | 0.10.2 | ❌ |
| numpy | 2.4.4 | <2.0.0 | ❌ |
| ml-dtypes | 0.5.4 | 0.4.1 | ❌ |
| transformers | 5.8.1 | 4.53.2 | ❌ |
| torch | 2.12.0 | 2.7.1 | ⚠️ |

### 1.2 重建步骤

```bash
# 删除旧环境
conda remove -n openpi_arx --all -y

# 创建新环境
conda create -n openpi_arx python=3.11 -y

# 分批安装依赖
# Batch 1: 核心 ML 框架
pip install "numpy>=1.26.0,<2.0.0" "ml-dtypes==0.4.1" "jax[cuda12]==0.5.3" "flax==0.10.2"

# Batch 2: PyTorch + Transformers
pip install "torch==2.7.1" "transformers==4.53.2" ...

# Batch 3: 其他依赖 + lerobot + openpi-client + openpi (editable)
pip install opencv-python pillow sentencepiece wandb polars pyrealsense2 zerorpc gym-aloha ...
pip install -e /home/yunlong/lerobot
pip install -e /home/yunlong/ARX_new/openpi-arx/packages/openpi-client
pip install -e /home/yunlong/ARX_new/openpi-arx --no-deps
```

### 1.3 最终兼容状态

| 包名 | 最终版本 | 要求 | 状态 |
|------|---------|------|------|
| jax[cuda12] | 0.5.3 | 0.5.3 | ✅ |
| flax | 0.10.2 | 0.10.2 | ✅ |
| numpy | 1.26.4 | <2.0.0 | ✅ |
| ml-dtypes | 0.4.1 | 0.4.1 | ✅ |
| torch | 2.7.1 | 2.7.1 | ✅ |
| transformers | 4.53.2 | 4.53.2 | ✅ |
| chex | 0.1.89 | jax>=0.4.27 | ✅ |
| orbax-checkpoint | 0.11.13 | 0.11.13 | ✅ |
| JAX GPU | CudaDevice(id=0) | — | ✅ |

**已知无害冲突**:
- `opencv-python-headless` 需 numpy>=2（已用 4.10.0.84 兼容 1.x）
- `rerun-sdk` 需 numpy>=2（已用 0.21.0 兼容 1.x）
- `datasets` 需 fsspec<=2025.9.0（已用 2025.9.0）
- `lerobot` 需 rerun-sdk>=0.24.0（已用 0.21.0，仅影响可视化）

---

## 2. Policy Server 启动

```bash
conda activate openpi_arx
cd /home/yunlong/ARX_new
python openpi-arx/scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_arx_r5_bottle_handoff \
    --policy.dir /home/yunlong/ARX_new/pi05_deploy/checkpoints/bottle_handoff_v2/13000
```

- WebSocket Server: `ws://0.0.0.0:8000`
- GPU 内存占用: ~18 GB
- 推理延迟: 首帧 ~10s (JIT编译), 稳态 ~82ms

---

## 3. 代码修改

### 3.1 文件清单

| 文件 | 修改内容 |
|------|---------|
| `openpi-arx/examples/arx_r5/config/cfg_arx_r5_pi.yaml` | 配置 IP、camera serials、mode、max_steps、prompt |
| `openpi-arx/examples/arx_r5/inference_arx_r5.py` | 新增 read_only 模式、receding horizon 平滑缓冲、后台推理 |
| `openpi-arx/src/openpi/arx/realsense_camera_rig.py` | **新建** - RealSense 三路相机读取（重试机制） |
| `openpi-arx/src/openpi/arx/arx_r5/arx_r5_robot_adapter.py` | 新增 `apply_single_action()` 方法 |
| `openpi-arx/src/openpi/arx/arx_ros2_rpc_client.py` | 修复 `disconnect()` 警告 |
| `ros2_bridge/arx_ros2_rpc_client.py` | 修复 `disconnect()` 警告 |
| `pi05_deploy/test_inference_readonly.py` | **新建** - 只读推理测试脚本 |
| `pi05_deploy/init_to_operating_pose.py` | **新建** - 平滑移动到操作姿态 |
| `pi05_deploy/reset_to_zero.py` | **新建** - 平滑复位到零位 |

### 3.2 关键修复: logging.basicConfig 位置

原脚本 `logging.basicConfig` 在 imports 之后调用，但 `datasets` 等库 import 时已触发 log 调用，导致 `basicConfig` 成为 no-op。修复：将 `basicConfig` 移到 imports 之前。

### 3.3 read_only 模式

新增 `mode: "read_only"` 配置选项：
- 连接真实机器人 RPC 获取状态
- 连接 Policy Server 运行推理
- **不向机器人发送动作指令**

实现方式：
```python
if self.run_mode == "read_only":
    rpc_client = ArxROS2RPCClient(...)  # 正常连接
    adapter_dry_run = False             # 读取真实状态
# ...
# 在 run() 中跳过 apply_single_action
if self.read_only:
    log.info("... (NOT sent)")
else:
    self.robot.apply_single_action(action)
```

### 3.4 平滑运动缓冲 (Receding Horizon)

**问题**: 旧代码每轮推理后一口气发完 16 步动作，然后等待 ~114ms（相机+推理），形成"突进→停顿→突进"的卡顿。

**方案**:
1. 推理结果存入 `ActionBuffer`（16 步 × 40D）
2. 每个控制周期 (66.7ms = 15Hz) 从 buffer 弹出 1 步执行
3. Buffer 耗尽时触发新的推理
4. 后台推理线程与动作执行并行

**关键参数**:
- `REPLAN_EARLY = 0`（不提前截断，完整执行 16 步轨迹）
- `pending_actions` 预取：旧 buffer 耗尽后才替换新轨迹

**效果**:
- 步间间隔: 66.7ms (15Hz) 均匀
- 推理频率: ~1 Hz（16 步触发一次）
- 不再有"突进→停顿"现象

### 3.5 相机集成

```python
# calibration_params.yaml 配置
head:        serial=218622271302 (D405 1280×720 → resize 424×240)
left_wrist:  serial=218622274803 (D405)
right_wrist: serial=218622278394 (D405)
```

相机硬件初始化慢（realsense-viewer 也需要多次点击），`RealSenseCameraRig` 实现了重试+pipeline 重启机制（5 次重试，20 帧预热）。

---

## 4. 实验记录

### 4.1 实验矩阵

| # | 起始位置 | 步数 | 模式 | 关键结果 |
|---|---------|------|------|---------|
| 1 | 操作姿态 | 3 | burst | 左 j1: 1.77→1.86, 夹爪闭合 |
| 2 | 零位 | 3 | burst | arm 几乎不动, 夹爪闭合 |
| 3 | 零位 | 30 | burst | arm 微动, 夹爪闭合 |
| 4 | 零位 | 60 | smooth | arm 微动 (0.01-0.08 rad) |
| 5 | 操作姿态 | 60 | smooth | arm 微动 (0.01-0.03 rad) |
| 6 | 零位 | 300 | smooth | **左 j0: 0→-0.375, 左 j1: 0→0.453** |
| 7 | **零位** | **3000** | smooth | 左 j0:-0.402, j1:0.258（平台期） |
| 8 | **操作姿态** | **3000** | smooth | **左 j3:-1.05, 右 j1:-0.97, 右 j3:-1.28 (任务行为)** |

### 4.2 操作姿态定义

从 `norm_stats.json` 训练数据统计均值推算出操作姿态（非官方定义）：

```python
OPERATING_POSE = {
    "left":  [-1.04, 1.79, 0.76, 1.05, -0.42, 0.07, 0.0],
    "right": [0.34, 0.99, 0.45, 0.53, -0.12, 0.05, 0.0],
}
```

### 4.3 关键发现

1. **从零位启动**: 模型会慢慢朝训练分布（操作姿态）靠拢，但 300 步左右到达平台期不再前进
2. **从操作姿态启动**: 模型需要足够多步数（~3000）才能完整展开任务轨迹
3. **行为克隆局限**: 模型在训练数据分布内的状态表现好，分布外则不敢做大动作
4. **Task Description 重要**: 正确 prompt `"Hand the bottle from the left arm to the right arm and place it in the basket on the right."` 比 `"pick up the object"` 更有效

### 4.4 最终 3000 步结果（实验 #8）

```
左臂: j0:-1.04→-1.20, j3:1.03→-0.02 (-60°) | gr: 0→0.009 (松开)
右臂: j1:0.98→0.01 (-56°), j3:0.52→-0.76 (-73°), j5:0.04→-0.59 | gr: 0→0.979 (抓紧)
```

符合 "左手交出瓶子，右手接住" 的任务语义。

---

## 5. 运行命令

```bash
# 启动 Policy Server
conda activate openpi_arx
python openpi-arx/scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config pi05_arx_r5_bottle_handoff \
    --policy.dir pi05_deploy/checkpoints/bottle_handoff_v2/13000

# 复位到零位
python pi05_deploy/reset_to_zero.py

# 移动到操作姿态
python pi05_deploy/init_to_operating_pose.py

# 推理（read_only / execute / mock, 见 cfg_arx_r5_pi.yaml）
python openpi-arx/examples/arx_r5/inference_arx_r5.py \
    --config openpi-arx/examples/arx_r5/config/cfg_arx_r5_pi.yaml
```

### 配置文件关键字段

```yaml
# openpi-arx/examples/arx_r5/config/cfg_arx_r5_pi.yaml
inference_type: "remote"        # 连接已有 Policy Server
policy_server:
  host: "127.0.0.1"
  port: 8000
robot:
  robot_type: "arx_r5"
  ip: "10.10.10.2"
  port: 4242
  mode: "execute"               # execute / mock / read_only
  control_mode: "full_joint"
cameras:
  enabled: true
  head_serial: "218622271302"
  left_wrist_serial: "218622274803"
  right_wrist_serial: "218622278394"
control:
  action_horizon: 16
  action_fps: 15
  max_steps: 3000
task_description: "Hand the bottle from the left arm to the right arm..."
```

---

## 6. 已知问题 & TODO

- [ ] 零位启动到操作姿态的过渡需要 300+ 步，中间阶段模型犹豫
- [ ] 推理频率 ~1Hz (模型侧)，控制频率 15Hz (执行侧) — 当前方案 OK，但推理延迟更高时可能断流
- [ ] 相机启动偶尔需要 pipeline 重启（硬件问题，非代码问题）
- [ ] 没有 head camera 外参实时更新（当前用 dummy 占位，未启用 calibrate_board_to_world）
- [ ] LeRobot 数据集实际 episode 起始帧可作为更精确的操作姿态
- [ ] 长时间运行 GPU 内存可能泄漏（未测试 > 1 小时连续运行）
