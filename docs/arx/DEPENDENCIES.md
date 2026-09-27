# 独立模型环境

原 uv.lock 中的 ../lerobot 本地依赖已归档 archive/platform_control/uv.platform-local.lock。
pyproject.toml 使用固定上游 LeRobot 提交和轻量 arx-client；新 uv.lock 已成功解析 272 个包，
不依赖 ../lerobot 或机器绝对路径。在目标 Linux 模型机运行 uv sync --locked。
平台为私有仓库时需要 GitHub 读取权限。Windows 检查 Git 依赖时可能需要 core.longpaths=true。
本次验证依赖解析、客户端 / adapter 解耦；完整 GPU 安装、训练和真机闭环仍需目标机器回归。
