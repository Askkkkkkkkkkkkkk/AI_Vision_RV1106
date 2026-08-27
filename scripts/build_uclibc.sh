#!/usr/bin/env bash
set -euo pipefail

RKMPI_ROOT="${RKMPI_ROOT:-$HOME/projAI_CAM/luckfox_pico_rkmpi_example}"
LUCKFOX_SDK_PATH="${LUCKFOX_SDK_PATH:-$HOME/luckfox-pico}"
APP_NAME="luckfox_pico_rtsp_yolov5"

if [[ ! -f "$RKMPI_ROOT/build.sh" ]]; then
    echo "找不到 RKMPI 工程：$RKMPI_ROOT"
    echo "请设置：export RKMPI_ROOT=完整RKMPI工程路径"
    exit 1
fi

if [[ ! -d "$RKMPI_ROOT/example/$APP_NAME" ]]; then
    echo "找不到 V3 源码：$RKMPI_ROOT/example/$APP_NAME"
    exit 1
fi

export LUCKFOX_SDK_PATH

echo "RKMPI 工程：$RKMPI_ROOT"
echo "SDK 路径：$LUCKFOX_SDK_PATH"
echo "请在菜单中选择：uclibc → $APP_NAME"

cd "$RKMPI_ROOT"
./build.sh
