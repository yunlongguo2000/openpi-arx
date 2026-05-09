#!/bin/bash
# ARX Pi0.5 端到端部署脚本
#
# 在 GPU 机器上运行, 完成:
#   1. 环境检查 (Python, openpi, checkpoint)
#   2. 可选: 启动硬件 (如果在机器人电脑上)
#   3. 启动 Policy Server (serve_policy.py)
#
# 用法:
#   # 仅启动 policy server (GPU 开发机)
#   ./scripts/deploy.sh --checkpoint checkpoints/pi05_arx/arx_task_v01/30000
#
#   # 启动硬件 + policy server (一体机部署)
#   ./scripts/deploy.sh --checkpoint <path> --hardware
#
#   # 使用 LoRA 配置
#   ./scripts/deploy.sh --config pi05_arx_r5_bottle_handoff_lora --checkpoint <path>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 默认参数
CONFIG_NAME="pi05_arx" # Set to pi05_arx_r5_bottle_handoff for R5
CHECKPOINT_DIR=""
LAUNCH_HARDWARE=false
SERVER_PORT=8000

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --hardware)
            LAUNCH_HARDWARE=true
            shift
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --checkpoint <path> [--config pi05_arx_r5_bottle_handoff] [--hardware] [--port 8000]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH  Trained model checkpoint directory (required)"
            echo "  --config NAME      Training config name (default: pi05_arx)"
            echo "  --hardware         Also launch robot hardware (LIFT, R5, cameras)"
            echo "  --port PORT        Policy server port (default: 8000)"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "============================================================"
echo "  ARX Pi0.5 Deployment"
echo "============================================================"
echo "  Config:     ${CONFIG_NAME}"
echo "  Checkpoint: ${CHECKPOINT_DIR:-'(not set)'}"
echo "  Hardware:   ${LAUNCH_HARDWARE}"
echo "  Port:       ${SERVER_PORT}"
echo ""

# ---------------------------------------------------------------
# 1. 环境检查
# ---------------------------------------------------------------
echo "[1/3] Checking environment..."

# Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found"
    exit 1
fi
echo "  Python: $(python3 --version)"

# openpi
cd "${OPENPI_ROOT}"
if ! python3 -c "import openpi" 2>/dev/null; then
    echo "[WARNING] openpi not importable. Installing..."
    if command -v uv &>/dev/null; then
        GIT_LFS_SKIP_SMUDGE=1 uv sync
        GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    else
        pip install -e .
    fi
fi
echo "  openpi: OK"

# Checkpoint
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] --checkpoint is required"
    echo "  Example: --checkpoint checkpoints/pi05_arx/arx_task_v01/30000"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ] && [[ ! "$CHECKPOINT_DIR" == gs://* ]]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT_DIR}"
    exit 1
fi
echo "  Checkpoint: OK"

# norm_stats
NORM_STATS_DIR="${OPENPI_ROOT}/assets"
if [ -d "$NORM_STATS_DIR" ]; then
    NORM_COUNT=$(find "$NORM_STATS_DIR" -name "norm_stats.json" 2>/dev/null | wc -l)
    echo "  Norm stats: found ${NORM_COUNT} file(s)"
else
    echo "  Norm stats: assets/ not found (will be loaded from checkpoint)"
fi

echo ""

# ---------------------------------------------------------------
# 2. 可选: 启动硬件
# ---------------------------------------------------------------
if [ "$LAUNCH_HARDWARE" = true ]; then
    echo "[2/3] Launching hardware..."
    HARDWARE_SCRIPT="${SCRIPT_DIR}/launch_hardware.sh"
    if [ -x "$HARDWARE_SCRIPT" ]; then
        bash "$HARDWARE_SCRIPT"
    else
        echo "[ERROR] Hardware script not found: ${HARDWARE_SCRIPT}"
        exit 1
    fi
    echo ""
else
    echo "[2/3] Skipping hardware launch (use --hardware to enable)"
    echo ""
fi

# ---------------------------------------------------------------
# 3. 启动 Policy Server
# ---------------------------------------------------------------
echo "[3/3] Starting Policy Server..."
echo "  Config: ${CONFIG_NAME}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Listening on: 0.0.0.0:${SERVER_PORT}"
echo ""

# 显示本机 IP 方便机器人端配置
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$LOCAL_IP" ]; then
    echo "  -> Set policy_server.host to '${LOCAL_IP}' in cfg_arx_pi.yaml on the robot"
    echo ""
fi

cd "${OPENPI_ROOT}"
exec uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config="${CONFIG_NAME}" \
    --policy.dir="${CHECKPOINT_DIR}" \
    --port="${SERVER_PORT}"
