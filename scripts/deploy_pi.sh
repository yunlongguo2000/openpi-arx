#!/bin/bash
# ARX Pi VLA Deployment Script
#
# Deploys the Physical Intelligence Pi model for autonomous operation
#
# Prerequisites:
#   1. Pi model weights downloaded to src/learning/vla/weights/
#   2. Configuration file created at src/learning/vla/config/pi_arx_lift.yaml
#   3. Python environment with OpenPI installed
#
# Usage:
#   ./deploy_pi.sh [--model pi0] [--config custom.yaml]

set -e

# Auto-detect ARX workspace root (portable across systems)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PI_ROOT="${ARX_ROOT}/src/learning/vla"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default settings
MODEL_NAME="pi0"
CONFIG_FILE="${PI_ROOT}/config/pi_arx_lift.yaml"
LAUNCH_HARDWARE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --no-hardware)
            LAUNCH_HARDWARE=false
            shift
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Usage: $0 [--model pi0] [--config config.yaml] [--no-hardware]"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "  ${CYAN}ARX Pi VLA Deployment${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "${BLUE}[INFO]${NC} ARX Root: ${ARX_ROOT}"
echo -e "${BLUE}[INFO]${NC} Pi Root: ${PI_ROOT}"
echo -e "${BLUE}[INFO]${NC} Model: ${MODEL_NAME}"
echo -e "${BLUE}[INFO]${NC} Config: ${CONFIG_FILE}"
echo ""

# Check if Pi directory exists
if [ ! -d "${PI_ROOT}" ]; then
    echo -e "${RED}[ERROR]${NC} Pi directory not found at: ${PI_ROOT}"
    exit 1
fi

# Check if model weights exist
WEIGHTS_DIR="${PI_ROOT}/weights/${MODEL_NAME}"
if [ ! -d "${WEIGHTS_DIR}" ]; then
    echo -e "${RED}[ERROR]${NC} Model weights not found at: ${WEIGHTS_DIR}"
    echo ""
    echo "Please download Pi model weights first:"
    echo "  1. Visit: https://physicalintelligence.company/download"
    echo "  2. Download ${MODEL_NAME} weights"
    echo "  3. Extract to: ${WEIGHTS_DIR}"
    echo ""
    exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${YELLOW}[WARNING]${NC} Config file not found: ${CONFIG_FILE}"
    echo -e "${YELLOW}[WARNING]${NC} A default config will be created"
    echo ""

    # Create default config
    mkdir -p "$(dirname ${CONFIG_FILE})"
    cat > ${CONFIG_FILE} << 'EOF'
# ARX LIFT2 Pi VLA Configuration
# Generated automatically

robot:
  type: arx_lift2
  control_frequency: 10  # Hz

model:
  name: pi0
  weights_path: auto  # Auto-detect from weights directory

inference:
  batch_size: 1
  chunk_size: 100
  temporal_ensemble: true

hardware:
  use_ros2: true
  topics:
    camera: /camera/color/image_raw
    depth: /camera/depth/image_rect_raw
    joint_states: /joint_states
    action: /arx_action

logging:
  level: INFO
  save_episodes: true
  episode_dir: ${ARX_ROOT}/data/episodes
EOF

    echo -e "${GREEN}[SUCCESS]${NC} Default config created at: ${CONFIG_FILE}"
    echo ""
fi

# Check ROS2 workspace
if [ ! -d "${ARX_ROOT}/ros2_ws/install" ]; then
    echo -e "${RED}[ERROR]${NC} ROS2 workspace not built!"
    echo "Please run: cd ${ARX_ROOT}/ros2_ws && colcon build --symlink-install"
    exit 1
fi

# Source ROS2 environment
echo -e "${BLUE}[INFO]${NC} Sourcing ROS2 environment..."
source ~/.bashrc
source "${ARX_ROOT}/ros2_ws/install/setup.bash"

# Launch hardware if requested
if [ "$LAUNCH_HARDWARE" = true ]; then
    echo ""
    echo -e "${BLUE}[INFO]${NC} Launching ARX hardware components..."
    echo ""

    # Launch body controller in background
    echo -e "${BLUE}[1/3]${NC} Launching LIFT body controller..."
    gnome-terminal \
        --title="ARX Body (Pi VLA)" \
        --geometry=100x30+0+0 \
        -- bash -c "\
            source ${ARX_ROOT}/ros2_ws/install/setup.bash && \
            echo '${GREEN}[BODY]${NC} Starting for Pi VLA...' && \
            ros2 launch arx_lift_controller lift.launch.py; \
            exec bash" &

    sleep 3

    # Launch R5 arms
    echo -e "${BLUE}[2/3]${NC} Launching R5 arms..."
    gnome-terminal \
        --title="ARX R5 Arms (Pi VLA)" \
        --geometry=100x30+800+0 \
        -- bash -c "\
            source ${ARX_ROOT}/ros2_ws/install/setup.bash && \
            echo '${GREEN}[R5 ARMS]${NC} Starting for Pi VLA...' && \
            ros2 launch arx_r5_controller double_arm.launch.py; \
            exec bash" &

    sleep 2

    # Launch camera
    echo -e "${BLUE}[3/3]${NC} Launching RealSense camera..."
    gnome-terminal \
        --title="RealSense Camera (Pi VLA)" \
        --geometry=80x20+0+500 \
        -- bash -c "\
            source ${ARX_ROOT}/ros2_ws/install/setup.bash && \
            echo '${GREEN}[CAMERA]${NC} Starting RealSense...' && \
            ros2 launch realsense2_camera rs_launch.py; \
            exec bash" &

    sleep 2

    echo -e "${GREEN}[SUCCESS]${NC} Hardware components launched"
    echo ""
fi

# Set up Python environment for Pi
echo -e "${BLUE}[INFO]${NC} Setting up Pi Python environment..."
cd ${PI_ROOT}

# Check if OpenPI is installed
if ! python3 -c "import openpi" 2>/dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} OpenPI not installed in current Python environment"
    echo ""
    echo "Installing OpenPI..."
    pip install -e . || {
        echo -e "${RED}[ERROR]${NC} Failed to install OpenPI"
        exit 1
    }
    echo -e "${GREEN}[SUCCESS]${NC} OpenPI installed"
fi

# Export environment variables
export PI_WEIGHTS="${WEIGHTS_DIR}"
export PI_CONFIG="${CONFIG_FILE}"
export ROS_DOMAIN_ID=0

echo ""
echo -e "${GREEN}[SUCCESS]${NC} Environment configured"
echo ""
echo "Environment variables:"
echo "  PI_WEIGHTS=${PI_WEIGHTS}"
echo "  PI_CONFIG=${PI_CONFIG}"
echo ""

# Check if example integration exists
EXAMPLE_DIR="${PI_ROOT}/examples/aloha_arx_lift_ros2_real"
if [ -d "${EXAMPLE_DIR}" ]; then
    echo -e "${BLUE}[INFO]${NC} Found ARX LIFT2 ROS2 integration example"
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "  ${CYAN}Starting Pi VLA Inference${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo ""

    # Run Pi inference
    cd ${EXAMPLE_DIR}

    echo -e "${BLUE}[INFO]${NC} Running Pi policy on ARX LIFT2..."
    echo ""

    python3 run_policy.py \
        --weights ${PI_WEIGHTS} \
        --config ${PI_CONFIG} \
        --robot arx_lift2 \
        --ros2
else
    echo -e "${YELLOW}[WARNING]${NC} ARX LIFT2 integration example not found"
    echo ""
    echo "You can run Pi inference manually:"
    echo "  cd ${PI_ROOT}"
    echo "  python -m openpi.inference.run_policy \\"
    echo "      --weights ${PI_WEIGHTS} \\"
    echo "      --config ${PI_CONFIG} \\"
    echo "      --robot arx_lift2"
    echo ""
fi

echo ""
echo -e "${GREEN}[SUCCESS]${NC} Pi VLA deployment complete!"
echo ""
