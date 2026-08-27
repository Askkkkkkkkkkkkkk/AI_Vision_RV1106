第 1 项：根目录 `README.md`。当前分支是 V3 源码快照，因此 README 明确说明它需要放入完整的 Luckfox RKMPI 工程中构建。

# AI Vision RV1106

基于 Luckfox Pico RV1106、SC3336 摄像头和 RKNN 的嵌入式端侧 AI 视觉识别系统。

本项目实现摄像头实时采集、YOLOv5 NPU 推理、检测框叠加与 RTSP 视频输出。当前 `v3-clean` 分支为性能优化后的 V3 版本，采用采集、预处理、推理、输出解耦的实时流水线。

## 项目能力

- SC3336 摄像头经 MIPI CSI 接入 RV1106；
- RKMPI 视频采集、ISP、H.264 编码与 RTSP 推流；
- RKNN Runtime 调用 RV1106 NPU 执行 YOLOv5 推理；
- 视频叠加检测框、类别、置信度与性能信息；
- V3 多线程流水线：Capture → Preprocess → Inference → Output；
- 有界最新帧队列，推理跟不上时主动丢弃旧帧，避免延迟累积；
- 当前 COCO YOLOv5 模型的 V3 版本实测视频输出约 15 FPS。

## 硬件与软件环境

| 项目 | 环境 |
|---|---|
| 开发板 | Luckfox Pico Pro/Max/Ultra，RV1106 |
| 摄像头 | SC3336 |
| 板端系统 | Luckfox Buildroot（uClibc） |
| 开发主机 | Ubuntu 虚拟机 |
| 模型格式 | RKNN |
| 推理运行时 | RKNN Runtime |
| 视频输出 | H.264 RTSP |
| 播放工具 | VLC |

## V3 架构

```text
SC3336
  ↓
VI 摄像头采集线程
  ↓
最新帧队列（丢弃旧帧）
  ↓
预处理线程
  ↓
最新帧队列（丢弃旧帧）
  ↓
RKNN NPU 推理线程
  ↓
DetectionBuffer（只保留最新检测结果）
  ↓
视频叠加、H.264 编码、RTSP 输出
```

V3 的核心目标是让 AI 推理与视频输出互不阻塞。即使 NPU 推理速度低于摄像头帧率，RTSP 视频也不会因为等待旧推理结果而持续积压。

## 当前仓库内容

```text
example/luckfox_pico_rtsp_yolov5/
├── include/                 # V3 头文件
├── model/
│   ├── anchors_yolov5.txt
│   ├── coco_80_labels_list.txt
│   └── yolov5.rknn
└── src/
    ├── ai_thread.cc
    ├── detection_buffer.cc
    ├── luckfox_mpi.cc
    ├── main.cc
    ├── postprocess.cc
    ├── video_thread.cc
    └── yolov5.cc
```

说明：本仓库当前保存的是 V3 应用源码与模型文件。构建仍依赖完整的 Luckfox RKMPI 示例工程和 Luckfox SDK。

## 获取代码

```bash
git clone https://github.com/Askkkkkkkkkkkkkk/AI_Vision_RV1106.git
cd AI_Vision_RV1106
git checkout v3-clean
```

将仓库中的 `example/luckfox_pico_rtsp_yolov5` 放入完整 Luckfox RKMPI 工程的 `example/` 目录中：

```text
~/projAI_CAM/luckfox_pico_rkmpi_example/example/luckfox_pico_rtsp_yolov5
```

## 编译

进入完整 Luckfox RKMPI 示例工程根目录：

```bash
cd ~/projAI_CAM/luckfox_pico_rkmpi_example
./build.sh
```

在交互菜单中选择：

```text
1) uclibc
luckfox_pico_rtsp_yolov5
```

生成的部署文件通常位于 `install/` 下。实际生成目录以本机 `build.sh` 输出为准。

## 板端运行

通过 USB ADB 或网络登录 RV1106 后，先停止默认视频服务，避免其占用摄像头：

```sh
/oem/usr/bin/RkLunch-stop.sh
```

当前已确认的 V3 可执行文件位于：

```text
/root/luckfox_pico_rtsp_yolov5
```

启动命令：

```sh
cd /root
export LD_LIBRARY_PATH=/oem/usr/lib:$LD_LIBRARY_PATH
./luckfox_pico_rtsp_yolov5
```

停止程序：

```text
Ctrl + C
```

## 查看 RTSP 视频

在 VLC 中打开网络串流：

```text
rtsp://<RV1106板端IP>/live/0
```

例如板端 IP 为 `192.168.20.2`：

```text
rtsp://192.168.20.2/live/0
```

IP 会随 USB 网卡、以太网或虚拟机网络设置变化。板端可通过以下命令查看：

```sh
ip -4 addr
```

## 模型文件

当前默认模型文件：

```text
model/yolov5.rknn
model/coco_80_labels_list.txt
model/anchors_yolov5.txt
```

后续替换为三类食材模型时，不能只替换 `.rknn` 文件，还需要同步修改：

1. 标签文件：`pizza`、`cookie`、`egg_tart`；
2. 类别数：COCO 80 类改为食材模型的 3 类；
3. 锚框：替换为模型导出时生成的 `RK_anchors.txt`；
4. 后处理：按三分类模型输出解析。

V3 的视频线程、DetectionBuffer 和 RTSP 流水线不需要因为模型替换而重写。

## 已完成的验证

- SC3336 摄像头正常出图；
- H.264 RTSP 视频流可由 VLC 播放；
- YOLOv5 RKNN 模型已在 RV1106 NPU 上完成推理；
- 摄像头实时检测、框绘制与 RTSP 输出已打通；
- V1 单线程基线约 6 FPS；
- V2 四线程流水线约 9 FPS；
- V3 去除重复视频取帧与颜色转换、解耦 AI 和视频输出后，视频输出达到约 15 FPS。

## 已知限制

- 当前仓库尚未提供统一部署脚本、开机自启动脚本和运行配置文件；
- 当前默认模型为 COCO 80 类模型，不是最终食材模型；
- 未完成食材模型的 PT → ONNX → RKNN → 板端实时识别验证；
- 未完成检测事件图片、JSON/CSV 事件记录和事件冷却机制；
- 未完成 8 小时稳定性、断网恢复、模型缺失和磁盘空间不足等可靠性测试；
- RTSP 地址依赖板端当前网络配置，换网络后需要重新确认 IP。

## 后续交付计划

1. 补齐构建、部署和运行脚本；
2. 增加可配置的模型路径、阈值和 RTSP 参数；
3. 增加 JPEG 与 CSV/JSON 事件输出；
4. 增加 Buildroot 开机自启动服务；
5. 接入三类食材自训练模型；
6. 补齐模型转换、性能测试、故障排查和部署文档；
7. 完成稳定性测试与最终验收记录。

## 参考

- Luckfox Pico RKMPI 示例工程：<https://github.com/LuckfoxTECH/luckfox_pico_rkmpi_example>
- 本项目 V3 Clean 分支：<https://github.com/Askkkkkkkkkkkkkk/AI_Vision_RV1106/tree/v3-clean>

下一项将是第 2 项文档包，我会先输出 `docs/hardware.md`。
