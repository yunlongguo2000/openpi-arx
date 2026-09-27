# 🚀 Pi0.5 部署修订计划 (本地4090显卡版本)

**重大变化**: 拥有本地4090显卡，可以进行**完整的本地训练和验证**！

---

## 📊 资源清单更新

### 硬件配置
- **处理器**: NVIDIA RTX 4090 (24GB VRAM)
- **内存**: 足够进行完整训练
- **存储**: 足够存放模型和数据集

### 可做的工作
- ✅ 完整的 LeRobot 环境搭建
- ✅ 模型下载和本地推理测试
- ✅ Pi0.5 预训练模型微调（可运行）
- ✅ 本地推理验证（可实际运行）
- ✅ 模型导出和优化（ONNX/TensorRT）

---

## 🎯 修订后的任务优先级

### 🔴 **P0 - 第一优先级** (1周内完成)

#### P0.1: Rosbridge 安全加固 ⏱️ 2-3天 (纯代码)
**状态**: 本地编码，不需要运行
- 完善关节/速度限幅
- 增强心跳机制
- RPC 重试装饰器

**文件**:
- `lerobot_data_collection/arx_vr_data_collection/ros2_bridge/arx_ros2_rpc_server.py`
- `lerobot_data_collection/arx_vr_data_collection/ros2_bridge/arx_ros2_rpc_client.py`

---

#### P0.2: Pi0.5 推理对接 ⏱️ 2-3天 (纯代码 + 可测试)
**状态**: 本地编码，后期可运行测试

**代码框架**:
```python
class Pi05InferenceNode:
    def __init__(self, model_path, dataset_repo_id):
        # 加载模型
        # 初始化 RPC 客户端
        pass

    def step(self, observation):
        # 推理一步
        pass

    def act_on_robot(self):
        # 主控制循环
        pass
```

**文件**:
- `lerobot_data_collection/arx_vr_data_collection/pi05_deployment/pi05_inference_node.py`
- `lerobot_data_collection/arx_vr_data_collection/pi05_deployment/normalizer.py`

---

#### P0.3: 本地训练环境搭建 ⏱️ 2-3天 (本地4090执行)
**状态**: 立刻可在本地运行！

```bash
# 在这台电脑上执行
cd ~/workspace
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 安装依赖 (使用 CUDA 支持)
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate

# 验证 PyTorch + CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# 下载预训练模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('lerobot/pi05_base')"

# 使用公开数据集验证训练 pipeline
python scripts/train.py \
    --config-name train_libero_base.yaml \
    --offline_steps 5000  # 完整验证
```

**检查清单**:
```
✅ PyTorch + CUDA 正常
✅ 模型下载到本地
✅ 推理速度测试 (4090应该 30+ FPS)
✅ 示例训练运行正常
✅ 内存占用监控
```

---

### 🟡 **P1 - 第二优先级** (第2-3周)

#### P1.1: 本地完整训练流程 ⏱️ 3-4天 (可实际执行)
**现在可实际做**：

```bash
# 1. 准备 arx 数据集
export DATASET_PATH="/home/yunlong/ARX_new/data/lerobot/arx_lift_task_20260312_v03"

# 2. 创建训练配置
cat > configs/train_pi05_arx_local.yaml << 'EOF'
name: "pi05_arx_finetune_local"
dataset_repo_id: "deepcybo/arx_lift_task_20260312_v03"
batch_size: 4  # 适合 4090 VRAM
num_workers: 4

policy:
  _target_: "lerobot.policies.pi05.pi05_policy.Pi05Policy"
  model_name: "lerobot/pi05_base"

training:
  lr: 1e-4
  num_epochs: 50  # 快速验证
  steps_per_epoch: 100

eval:
  eval_episodes: 5
  eval_every_steps: 500
EOF

# 3. 启动本地训练
python scripts/train.py \
    --config-path configs \
    --config-name train_pi05_arx_local.yaml \
    --output-dir ./outputs/pi05_arx_v01
```

**收益**:
- ✅ 验证方案的正确性
- ✅ 获得 baseline 性能
- ✅ 理解训练过程中的问题
- ✅ 预热 4090 显卡

---

#### P1.2: 推理节点本地验证 ⏱️ 2-3天 (可实际执行)
**代码框架和测试**:

```python
# test_pi05_local_inference.py
import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pi05_deployment.pi05_inference_node import Pi05InferenceNode

def test_inference_speed():
    """测试本地推理速度"""
    node = Pi05InferenceNode(
        model_path="lerobot/pi05_base",
        dataset_repo_id="deepcybo/arx_lift_task_20260312_v03"
    )

    # 模拟观测
    obs = {
        'state': np.random.randn(59).astype(np.float32),
        'images': {
            'left_wrist_image': np.random.randint(0, 255, (240, 424, 3), dtype=np.uint8),
            'right_wrist_image': np.random.randint(0, 255, (240, 424, 3), dtype=np.uint8),
        }
    }

    # 热身
    for _ in range(5):
        node.step(obs)

    # 计时推理
    import time
    t0 = time.time()
    for _ in range(100):
        action = node.step(obs)
    t1 = time.time()

    fps = 100 / (t1 - t0)
    print(f"推理速度: {fps:.1f} FPS")
    print(f"单步延迟: {(t1-t0)*10:.1f} ms")
    print(f"Action shape: {action.shape}")

    assert fps > 20, f"推理速度太慢: {fps} FPS"
    print("✅ 推理速度测试通过")

if __name__ == "__main__":
    test_inference_speed()
```

**运行命令**:
```bash
python test_pi05_local_inference.py
# 预期输出: ✅ 推理速度测试通过 (4090应该 30+ FPS)
```

---

#### P1.3: 模型导出优化 ⏱️ 2-3天 (可实际执行)
**新增任务：在本地完成**：

```python
# export_pi05_model.py
import torch
from transformers import AutoModel

def export_onnx():
    """导出为 ONNX 格式 (推理加速)"""
    model = AutoModel.from_pretrained("lerobot/pi05_base")

    # 导出 ONNX
    dummy_input = torch.randn(1, 20, 59)  # (batch, context, obs_dim)
    torch.onnx.export(
        model, dummy_input,
        "pi05_model.onnx",
        input_names=['observations'],
        output_names=['actions'],
        dynamic_axes={'observations': {0: 'batch_size'}},
        opset_version=14
    )
    print("✅ ONNX 导出完成")

def quantize_model():
    """INT8 量化 (减小模型体积)"""
    import torch.quantization as quantization

    model = AutoModel.from_pretrained("lerobot/pi05_base")
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    quantization.prepare_qat(model, inplace=True)
    # ... 量化过程 ...
    print("✅ 量化完成")

if __name__ == "__main__":
    export_onnx()
    quantize_model()
```

---

### 🟢 **P2 - 长期优化** (可选)

#### P2.1: 本地 Rosbridge 性能优化 ⏱️ 2-3天 (可实际测试)
```python
# 在本地测试关节轨迹插值和速度平滑
# 验证端到端延迟 (~120ms 目标)
# 测试不同的控制频率设置
```

---

## 📈 修订后的工作流

### 第1周: P0 建设 (7天)
```
Day 1-2: P0.1 安全加固 (纯代码)
Day 3-4: P0.2 推理对接 (纯代码)
Day 5-7: P0.3 本地训练环境 (本地4090执行验证)
         ├─ 环境搭建 (1.5天)
         ├─ 模型下载 (0.5天)
         ├─ 推理测试 (1天)
         └─ 示例训练验证 (2天)
```

### 第2周: P1 加强 (6-7天)
```
Day 8-12: P1.1 本地完整训练流程
          ├─ 训练配置完成 (1天)
          ├─ 首次训练 (2-3天, 4090跑 50 epoch)
          └─ 性能评估 (1-2天)

Day 13-14: P1.2 推理节点验证
           ├─ 推理速度测试 (1天)
           └─ 故障处理 (1天)
```

### 第3周: P2 优化 (可选)
```
Day 15-19: P1.3 模型导出优化
           ├─ ONNX 导出 (1天)
           ├─ INT8 量化 (2天)
           └─ 部署包制作 (1-2天)

Day 19-21: P2.1 Rosbridge 性能优化
```

---

## 💻 本地 4090 训练设置

### 环境变量配置
```bash
# ~/.bashrc 或 ~/.zshrc 添加
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=~/.cache/torch
export HF_HOME=~/.cache/huggingface

# 验证
nvidia-smi  # 应显示 RTX 4090
```

### 内存优化 (4090 VRAM 充足)
```python
# training_config.yaml
batch_size: 4      # 可用更大的 batch size
gradient_accumulation_steps: 2
use_fp16: true     # 混合精度训练，加速+省显存
num_workers: 4     # 多进程数据加载
pin_memory: true   # 锁定显存提高速度
```

### 训练监控
```bash
# 实时监控显卡占用
watch -n 1 nvidia-smi

# 或使用 tensorboard 监控训练
tensorboard --logdir ./outputs/pi05_arx_v01 --port 6006
# 访问: http://localhost:6006
```

---

## 🎯 立刻可执行的任务列表

### 今天 (第1天)
- [ ] P0.1 开始 - 审查 RPC 代码 (1小时)
- [ ] P0.1 开始 - 设计安全加固方案 (1小时)
- [ ] P0.3 开始 - 验证 CUDA 环境 (30分钟)
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  nvidia-smi
  ```

### 本周 (Day 2-7)
- [ ] P0.1 完成 - Rosbridge 安全加固
- [ ] P0.2 完成 - Pi0.5 推理对接
- [ ] P0.3 完成 - 本地训练环境搭建 + 验证

### 第2周
- [ ] P1.1 开始 - 本地完整训练
- [ ] P1.2 开始 - 推理速度测试

### 第3周
- [ ] P1.3 - 模型导出优化
- [ ] P2.1 - Rosbridge 性能优化 (可选)

---

## 📊 性能预期 (4090)

| 任务 | 预期性能 | 备注 |
|------|---------|------|
| 推理速度 | 30-50 FPS | Pi0.5 小模型，非常快 |
| 训练速度 | ~1000 samples/sec | 单个 4090 能力 |
| 完整训练 (50 epoch) | ~2-3 小时 | 快速验证 |
| 模型导出 | 实时完成 | ONNX 转换非常快 |
| 量化时间 | ~30 分钟 | 全量化过程 |

---

## 🔑 关键改变

### 之前计划 (远程服务器)
- ❌ 环境搭建在"开发服务器"
- ❌ 模型下载在服务器
- ❌ 推理代码只能测试逻辑，不能实运行
- ❌ 等 VR 数据后再开始训练

### 现在计划 (本地4090)
- ✅ 环境搭建在本地 (立刻验证)
- ✅ 模型本地下载 (快速迭代)
- ✅ 推理代码可实际运行测试 (获得真实性能)
- ✅ 收到 VR 数据后立刻开始训练 (零等待)

---

## 🚀 立刻开始

### Step 1: 验证 CUDA (5分钟)
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name()}')"
# 预期输出: CUDA Available: True, Device: NVIDIA RTX 4090
```

### Step 2: 查看修订计划 (15分钟)
```bash
cat /home/yunlong/ARX_new/REVISED_PI05_PLAN.md
```

### Step 3: 开始 P0.1 (现在！)
```bash
cd /home/yunlong/ARX_new
git checkout yunlong/feature/pi05-deployment
cat lerobot_data_collection/arx_vr_data_collection/ros2_bridge/arx_ros2_rpc_server.py
# 开始编码 P0.1
```

---

## 💪 现在你的能力

```
你不仅能:
  ✅ 写完整的代码
  ✅ 本地测试代码逻辑
  ✅ 实际运行推理 (测试性能)
  ✅ 进行完整的模型训练 (50 epoch 只需 2-3 小时)
  ✅ 导出优化模型
  ✅ 进行性能基准测试

你现在是:
  🚀 完整的 Pi0.5 部署工程师
```

---

## ⏱️ 最终时间表

```
Week 1: P0 全部完成 (可验证+可测试)
Week 2: P1 主要任务完成 (可实运行训练)
Week 3: P2 完善流程 (生产就绪)

总耗时: 3周 (相比之前节省 1-2 周)
最终结果: 完整的本地 Pi0.5 部署系统 + 训练流程
```

---

<div style="background: #c8e6c9; padding: 20px; border-radius: 8px; border-left: 4px solid #4caf50;">

## 🎉 关键优势

**你拥有本地 4090 意味着:**

1. **立刻验证** - 所有代码都能本地运行，不用等远程
2. **快速迭代** - 推理代码修改后秒级验证
3. **完整训练** - 收到 VR 数据后立刻开始，无关键路径延迟
4. **性能基准** - 建立本地性能标准，部署时有参考
5. **优化调试** - 显存占用、推理速度都能本地测试优化

**Time to Production: 从 4-5 周 → 2-3 周！**

</div>

