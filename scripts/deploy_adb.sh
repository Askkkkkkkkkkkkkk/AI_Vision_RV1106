#!/usr/bin/env bash
set -euo pipefail

RKMPI_ROOT="${RKMPI_ROOT:-$HOME/projAI_CAM/luckfox_pico_rkmpi_example}"
APP_NAME="luckfox_pico_rtsp_yolov5"
DEPLOY_DIR="${DEPLOY_DIR:-$RKMPI_ROOT/install/uclibc/${APP_NAME}_demo}"
TARGET_DIR="${TARGET_DIR:-/root/ai_vision_v3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/configs/ai_vision.conf}"

if [[ ! -d "$DEPLOY_DIR" ]]; then
    echo "找不到编译部署目录：$DEPLOY_DIR"
    echo "请先完成编译，并确认 install/uclibc 下的实际目录。"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "找不到配置文件：$CONFIG_FILE"
    exit 1
fi

adb wait-for-device
adb shell "mkdir -p '$TARGET_DIR'"

echo "部署到板端：$TARGET_DIR"
adb push "$DEPLOY_DIR/." "$TARGET_DIR/"
adb shell "mkdir -p '$TARGET_DIR/config'"
adb push "$CONFIG_FILE" "$TARGET_DIR/config/ai_vision.conf"
adb shell "chmod +x '$TARGET_DIR/$APP_NAME'"

echo "部署完成。运行：TARGET_DIR=$TARGET_DIR ./scripts/run_board.sh"
