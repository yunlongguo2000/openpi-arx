# 🚀 Pi0.5 并行开发路线图 (yunlong)

**当前状态**: wuhao0120 在调试 VR 遥操作延迟问题
**你的任务**: 并行开展 Pi0.5 部署前期开发工作
**目标**: 等 VR 调试完成时，所有代码和环境都已准备就绪

---

## 📊 工作优先级矩阵

```
优先级    工作量    收益    截止日期    状态
─────────────────────────────────────────
🔴 P0    1-2天    最高   立即开始    → 下周一前完成
🟡 P1    3-5天    高     下周三前完成  → 备选并行任务
🟢 P2    灵活     中等   长期优化    → VR调试后再做
```

---

## 🎯 推荐并行方案（3周内完成）

根据你已完成的 **数据集分析工作** 和 **现有代码基础**，建议采用以下并行方案：

```
第1周 (本周)：
├─ ✅ 数据集分析完成 (已做)
├─ 🔴 P0.1: Rosbridge 安全加固 (2-3天)
└─ 🔴 P0.2: Pi0.5 推理对接代码 (2-3天)

第2周：
├─ 🟡 P1.1: 数据处理工具链 (3-4天)
└─ 🟡 P1.2: Pi0.5 训练配置模板 (1-2天)

第3周：
├─ ☁️  P0.3: 开发服务器训练环境搭建 (2-3天)
└─ 🟢 可选: 模型优化工具链开发
```

---

## 📋 具体任务分解

### 🔴 **第一优先级 (P0) - 立即开展**

#### P0.1: Rosbridge 安全加固开发 ⏱️ 2-3 天

**现状**:
- ✅ 已有 `arx_ros2_rpc_server.py` 基础框架
- ⚠️ 缺少关键安全验证和心跳机制

**你需要做**:

```python
# 1. 关节角度/速度限幅 (已部分实现在 server，需要完善)
# 位置: arx_ros2_rpc_server.py

class ArxROS2RPCServer:
    def _clamp_joint_positions(self, positions, is_left):
        """关节位置限幅 + 速度限制"""
        # ✅ 已有: 位置范围限幅
        # ✅ 已有: 速度限幅
        # TODO: 完善错误日志和告警机制

    def set_chassis_velocity(self, vx, vy, wz):
        """底盘速度限幅"""
        # ✅ 已有框架
        # TODO: 增加硬限制，防止突加速

# 2. 心跳检测机制 (基础框架存在，需强化)
class ArxROS2RPCServer:
    def _heartbeat_check_loop(self):
        """心跳超时自动急停"""
        # ✅ 已有超时检测
        # ✅ 已有急停触发
        # TODO: 增加故障恢复、日志完善

# 3. 客户端重试逻辑 (需要完全实现)
# 位置: arx_ros2_rpc_client.py

class ArxROS2RPCClient:
    def _call_with_retry(self, method, *args, max_retries=2):
        """RPC调用重试 + 失败急停"""
        for attempt in range(max_retries + 1):
            try:
                return self._call(method, *args)
            except Exception as e:
                if attempt < max_retries:
                    log.warning(f"重试 {attempt+1}/{max_retries}")
                    time.sleep(0.1)
                else:
                    log.error(f"失败后急停")
                    self.emergency_stop()
                    raise
```

**实现步骤**:
1. 阅读现有 `arx_ros2_rpc_server.py` 中的 `_clamp_joint_positions()` 实现
2. 完善和测试限幅逻辑
3. 增强现有心跳机制的日志和恢复能力
4. 在客户端实现重试装饰器
5. 编写单元测试验证所有边界情况

**预期产物**:
- `arx_ros2_rpc_server.py` (加强版)
- `arx_ros2_rpc_client.py` (加强版)
- `test_rpc_safety.py` (安全测试用例)

---

#### P0.2: Pi0.5 推理对接代码开发 ⏱️ 2-3 天

**现状**:
- ✅ 有 `ArxROS2RPCClient` 基础类
- ✅ 有完整数据集格式定义
- ❌ 缺少 Pi0.5 推理 → RPC 命令的转换层

**你需要做**:

```python
# 位置: lerobot_data_collection/arx_vr_data_collection/
#       pi05_deployment/pi05_inference_node.py (新建)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.normalize_utils import Normalizer
from ros2_bridge.arx_ros2_rpc_client import ArxROS2RPCClient
import numpy as np

class Pi05InferenceNode:
    """Pi0.5 推理和 RPC 对接节点"""

    def __init__(self, model_path, dataset_repo_id):
        """初始化推理节点"""
        # 1. 加载预训练模型
        self.model = self._load_model(model_path)

        # 2. 加载统计数据用于反归一化
        self.dataset = LeRobotDataset(dataset_repo_id)
        self.normalizer = self._build_normalizer(self.dataset)

        # 3. 连接 RPC 客户端
        self.rpc_client = ArxROS2RPCClient(ip="localhost", port=4242)

        # 4. 维护推理状态缓冲
        self.obs_buffer = []  # 历史20帧观测
        self.context_size = 20
        self.horizon = 16

    def _load_model(self, model_path):
        """加载 Pi0.5 预训练模型"""
        # 从 HuggingFace 下载或本地加载
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path)
        return model.eval()

    def _build_normalizer(self, dataset):
        """从数据集统计信息构建归一化器"""
        import json
        stats = json.load(...)  # 从 meta/stats.json 读取
        normalizer = Normalizer(stats)
        return normalizer

    def step(self, observation):
        """推理一步，返回动作"""
        # 1. 观测归一化
        obs_normalized = self.normalizer.normalize(observation)

        # 2. 维护观测缓冲(滑动窗口)
        self.obs_buffer.append(obs_normalized)
        if len(self.obs_buffer) > self.context_size:
            self.obs_buffer.pop(0)

        # 3. 如果缓冲未满，返回空操作
        if len(self.obs_buffer) < self.context_size:
            return np.zeros(32)

        # 4. Pi0.5 推理获得未来 16 步动作
        context = np.array(self.obs_buffer)  # (20, obs_dim)
        with torch.no_grad():
            actions_pred = self.model.predict(context)  # (16, 32)

        # 5. 取第一步作为当前命令
        action_normalized = actions_pred[0]

        # 6. 反归一化
        action = self.normalizer.denormalize(action_normalized)

        return action

    def act_on_robot(self):
        """主控制循环"""
        # 连接机器人
        connected = self.rpc_client.system_connect(timeout=10.0)
        if not connected:
            raise RuntimeError("无法连接到机器人")

        try:
            while True:
                # 1. 获取当前观测
                obs = self.rpc_client.get_full_state()
                if obs is None:
                    continue

                # 2. Pi0.5 推理
                action = self.step(obs)

                # 3. 发送命令到机器人
                # 解析 action 为左臂、右臂、底盘命令
                left_joints = action[:7]
                right_joints = action[7:14]
                left_tcp = action[14:20]
                right_tcp = action[20:26]
                left_gripper = action[26]
                right_gripper = action[27]
                chassis_vx = action[28]
                chassis_vy = action[29]
                chassis_wz = action[30]
                chassis_height = action[31]

                # 发送到机器人
                self.rpc_client.set_full_command(
                    left_joints, right_joints,
                    chassis_vx, chassis_vy, chassis_wz, chassis_height
                )

                time.sleep(1/15)  # 15 FPS

        except KeyboardInterrupt:
            print("推理中止")
        finally:
            self.rpc_client.disconnect()

# 命令行入口
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lerobot/pi05_base")
    parser.add_argument("--dataset", default="deepcybo/arx_lift_task_20260312_v03")
    args = parser.parse_args()

    node = Pi05InferenceNode(args.model, args.dataset)
    node.act_on_robot()
```

**实现步骤**:
1. 创建 `pi05_deployment/` 目录
2. 实现 `Pi05InferenceNode` 类框架
3. 集成 LeRobot 推理接口
4. 对接 RPC 客户端
5. 编写集成测试（不需要真实运行）

**预期产物**:
- `pi05_deployment/pi05_inference_node.py`
- `pi05_deployment/normalizer.py` (数据归一化工具)
- `tests/test_pi05_inference.py` (单元测试)

---

### 🟡 **第二优先级 (P1) - 3-5 天**

#### P1.1: 数据处理工具链 ⏱️ 3-4 天

**模块list**:

```python
# scripts/data_pipeline/

1. validate_dataset.py (已有基础)
   ├─ 完善数据字段检查
   ├─ 时间同步性验证
   └─ 输出验证报告

2. clean_dataset.py (新建)
   ├─ 过滤无效轨迹 (关节突变>阈值)
   ├─ 去除异常片段 (动作速度>限制)
   └─ 输出清洗后数据集

3. upload_dataset.py (新建)
   ├─ 自动上传到训练服务器
   ├─ 支持断点续传
   └─ 上传完成自动触发预处理

4. preprocess_dataset.py (新建)
   ├─ 计算 action/state 归一化统计量
   ├─ 生成训练用 stats.json
   └─ 分割 train/val/test
```

---

#### P1.2: Pi0.5 训练配置模板 ⏱️ 1-2 天

**新建文件**: `configs/train_pi05_arx.yaml`

```yaml
# Pi0.5 微调配置 - ARX R5 + LIFT
name: "pi05_arx_finetune"

# 数据配置
dataset_repo_id: "deepcybo/arx_lift_task_20260312_v03"
batch_size: 8
num_workers: 4

# 模型配置
policy:
  _target_: "lerobot.policies.pi05.pi05_policy.Pi05Policy"
  model_name: "lerobot/pi05_base"  # 预训练模型

# 观测空间 (59维)
observation_features:
  - joint_pos_left: (7,)      # 左臂关节位置
  - joint_vel_left: (7,)      # 左臂关节速度
  - joint_cur_left: (7,)      # 左臂关节电流
  - joint_pos_right: (7,)     # 右臂关节位置
  - joint_vel_right: (7,)     # 右臂关节速度
  - joint_cur_right: (7,)     # 右臂关节电流
  - tcp_pose_left: (6,)       # 左TCP位姿
  - tcp_pose_right: (6,)      # 右TCP位姿
  - gripper_left: (1,)        # 左夹爪
  - gripper_right: (1,)       # 右夹爪
  - chassis_state: (3,)       # 底盘状态
  - images:
      left_wrist: (3, 240, 424)   # 左手腕相机
      right_wrist: (3, 240, 424)  # 右手腕相机

# 动作空间 (32维)
action_features:
  - joint_pos_left: (7,)
  - joint_pos_right: (7,)
  - tcp_pose_left: (6,)
  - tcp_pose_right: (6,)
  - gripper_left: (1,)
  - gripper_right: (1,)
  - chassis_cmd: (4,)  # vx, vy, wz, height

# 训练配置
training:
  lr: 1e-4
  num_epochs: 100
  steps_per_epoch: 500
  gradient_clip: 1.0

  # 数据增强
  augmentation:
    image_augment: true
    action_noise: 0.01

# 评估配置
eval:
  eval_episodes: 10
  eval_every_steps: 2000
```

---

### ☁️ **开发服务器任务 (P0.3) - 2-3 天**

#### P0.3: 训练环境搭建 ⏱️ 2-3 天

**在开发服务器上执行** (完全不涉及机器人):

```bash
# 1. 拉取 LeRobot 代码
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout <pi05_branch>

# 2. 安装依赖
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers lerobot[diffusion]

# 3. 下载预训练模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('lerobot/pi05_base')"

# 4. 验证训练 pipeline
# 使用公开数据集 (lerobot/libero_object_dataset_singlearm_02)
python scripts/train.py \
    --config-path config_dir \
    --config-name train_libero_base.yaml
```

**目标**:
- ✅ 环境搭建完毕
- ✅ 能成功运行示例训练
- ✅ 验证模型推理工作正常

---

## 🎬 立即行动计划 (本周)

### 第1天 (今天) ⏱️ 2-3小时
- [ ] 阅读现有 `arx_ros2_rpc_server.py` 代码
- [ ] 理解 `_clamp_joint_positions()` 实现
- [ ] 规划 P0.1 改进点

### 第2-3天 ⏱️ 8-10小时
- [ ] **P0.1**: 完善 Rosbridge 安全加固
  - [ ] 增强关节限幅逻辑
  - [ ] 完善心跳机制日志
  - [ ] 实现客户端重试装饰器
  - [ ] 编写测试用例

### 第4-5天 ⏱️ 8-10小时
- [ ] **P0.2**: 实现 Pi0.5 推理对接
  - [ ] 创建 `Pi05InferenceNode` 框架
  - [ ] 集成 LeRobot 推理
  - [ ] 对接 RPC 客户端
  - [ ] 编写集成测试

### 并行任务 (第1-5天)
- [ ] **开发服务器**: 搭建 Pi0.5 训练环境
  - [ ] 安装依赖
  - [ ] 下载模型
  - [ ] 验证示例训练

---

## 📌 优先级提示

**DO NOW (立即做)**:
1. ✅ P0.1 - Rosbridge 安全加固 (最有价值，最快有成果)
2. ✅ P0.2 - Pi0.5 推理对接 (核心功能，必须完成)
3. ✅ P0.3 - 训练环境搭建 (在开发机上做，不阻塞你)

**DO LATER (下周)**:
1. 🟡 P1.1 - 数据处理工具链
2. 🟡 P1.2 - 训练配置模板

**OPTIONAL (可选，长期)**:
1. 🟢 P2 - Rosbridge 性能优化

---

## 📊 工作量估算

| 任务 | 工作量 | 难度 | 依赖 |
|------|--------|------|------|
| P0.1 | 1-2天 | ⭐⭐ | 无 |
| P0.2 | 2-3天 | ⭐⭐⭐ | 无 |
| P0.3 | 2-3天 | ⭐ | 无 |
| P1.1 | 3-4天 | ⭐⭐ | P0.1✓ |
| P1.2 | 1-2天 | ⭐ | 无 |
| **总计** | **8-14天** | | |

**预期**: 10天内所有 P0 和 P1 任务完成

---

## ✨ 成果预期

完成以上任务后，你将拥有：

```
✅ 硬件安全保障
   ├─ 关节/速度限幅完善
   ├─ 心跳超时自动急停
   └─ RPC 调用失败重试机制

✅ Pi0.5 完整推理链路
   ├─ 推理节点 (Pi05InferenceNode)
   ├─ 归一化工具 (Normalizer)
   ├─ RPC 对接完成
   └─ 端到端测试通过

✅ 数据处理自动化
   ├─ 数据验证脚本
   ├─ 数据清洗脚本
   ├─ 自动上传脚本
   └─ 预处理流程完整

✅ 训练环境就绪
   ├─ LeRobot 环境搭建
   ├─ 模型下载完成
   ├─ 示例训练验证
   └─ 模型推理测试

🎯 最终: 只需替换数据集路径即可立刻开始训练！
```

---

## 🔗 相关文件

- **参考文档**: `/home/yunlong/ARX_new/docs/Pi0.5部署并行任务安排.md`
- **数据集分析**: `/home/yunlong/ARX_new/data/DATASET_ANALYSIS.md`
- **已有代码**:
  - `lerobot_data_collection/arx_vr_data_collection/ros2_bridge/arx_ros2_rpc_server.py`
  - `lerobot_data_collection/arx_vr_data_collection/ros2_bridge/arx_ros2_rpc_client.py`
- **ROS2 Bridge**: `/home/yunlong/ARX_new/lerobot_data_collection/arx_vr_data_collection/ros2_bridge/arx_lift_ros2_bridge.py`

---

## 🚨 注意事项

**DO**:
- ✅ 所有代码都是纯开发，不需要运行
- ✅ 可以在本地 IDE 中完成所有 P0、P1 任务
- ✅ P0.3 在开发服务器上进行
- ✅ 充分利用并行性，加快进度

**DON'T**:
- ❌ 不要修改机器人电脑上的任何代码
- ❌ 不要占用机器人网络
- ❌ 不要干扰 wuhao0120 的 VR 调试工作
- ❌ 不要在机器人上运行新代码

---

**预期工作完成时间**: 2-3 周
**最终目标**: Pi0.5 全链路部署就绪，等数据即可训练！
