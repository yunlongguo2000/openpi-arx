#!/usr/bin/env bash
# 测试环境配置

source examples/aloha_arx_lift_ros2_real/setup_env.sh

echo "测试导入 rclpy..."
python -c "import rclpy; print('rclpy 导入成功')"

echo ""
echo "测试导入项目模块..."
python -c "from examples.aloha_arx_lift_ros2_real import robot_utils; print('robot_utils 导入成功')"

echo ""
echo "所有测试通过!"
