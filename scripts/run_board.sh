#!/usr/bin/env bash
set -euo pipefail

APP_NAME="luckfox_pico_rtsp_yolov5"
TARGET_DIR="${TARGET_DIR:-/root/ai_vision_v3}"

adb wait-for-device

adb shell "test -x '$TARGET_DIR/$APP_NAME'" || {
    echo "板端找不到：$TARGET_DIR/$APP_NAME"
    echo "请先运行 deploy_adb.sh"
    exit 1
}

adb shell "if [ -x /oem/usr/bin/RkLunch-stop.sh ]; then /oem/usr/bin/RkLunch-stop.sh; fi"

echo "启动 V3..."
adb shell "cd '$TARGET_DIR' && \
export LD_LIBRARY_PATH=./lib:/oem/usr/lib:\$LD_LIBRARY_PATH && \
exec ./$APP_NAME"
