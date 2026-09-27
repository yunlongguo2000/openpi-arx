# ARX 平台接入

OpenPI 可以独立克隆到任意目录。ARX 控制器与真实 RPC 服务统一维护在 ARX_new；本工程仅维护模型训练、数据映射与推理适配。

在现有模型环境中安装平台发布版本的轻量客户端，然后运行 examples/arx_r5 或 examples/arx_lift2。arx-client 提供原有 ArxROS2RPCClient API，兼容 import 保留在 openpi.arx 中。

本次整理保留原有 ZeroRPC 控制传输，不更改动作布局或模型参数。模型历史文档中仍可能描述旧 ZMQ 协议，应以平台 protocol-v1.json 为准。旧控制副本和 mock 服务在 archive/platform_control。

完整模型训练/真机闭环没有在 Windows 整理机上验证；客户端与适配边界由离线测试验证。
新 uv.lock 已按固定 Git 依赖解析成功，旧本地路径锁仅作为历史材料保存。
固定 arx-client 来源为 ARX_new 的 e44224bb5350ee6c22fbd41c6dee58590dbe7b68 提交。
