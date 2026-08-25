#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>

#include <deque>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define DISP_WIDTH  720
#define DISP_HEIGHT 480

// Keep only the newest packet at each stage.
// This is important for a real-time camera: when AI is slower than capture,
// we prefer dropping an old frame instead of accumulating seconds of latency.
#define PIPELINE_QUEUE_SIZE 1
#define PERF_REPORT_FRAMES  30

// display size
static const int width  = DISP_WIDTH;
static const int height = DISP_HEIGHT;

// model size
static const int model_width  = 640;
static const int model_height = 640;

static volatile sig_atomic_t g_running = 1;

static uint64_t now_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static double us_to_ms(uint64_t us)
{
    return (double)us / 1000.0;
}

static int clamp_int(int value, int low, int high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

// -----------------------------------------------------------------------------
// LatestQueue
// -----------------------------------------------------------------------------
// A bounded queue with "drop oldest" policy.
// If the consumer is too slow, push() discards the oldest queued packet and
// keeps the newest one. This prevents latency from growing without bound.
// -----------------------------------------------------------------------------
template <typename T>
class LatestQueue {
public:
    explicit LatestQueue(size_t capacity)
        : capacity_(capacity), stopped_(false), dropped_(0)
    {
        pthread_mutex_init(&mutex_, NULL);
        pthread_cond_init(&cond_, NULL);
    }

    ~LatestQueue()
    {
        pthread_cond_destroy(&cond_);
        pthread_mutex_destroy(&mutex_);
    }

    bool push(const T &item)
    {
        pthread_mutex_lock(&mutex_);

        if (stopped_) {
            pthread_mutex_unlock(&mutex_);
            return false;
        }

        while (queue_.size() >= capacity_) {
            queue_.pop_front();
            ++dropped_;
        }

        queue_.push_back(item);
        pthread_cond_signal(&cond_);
        pthread_mutex_unlock(&mutex_);
        return true;
    }

    bool pop(T *item)
    {
        pthread_mutex_lock(&mutex_);

        while (queue_.empty() && !stopped_) {
            pthread_cond_wait(&cond_, &mutex_);
        }

        if (queue_.empty() && stopped_) {
            pthread_mutex_unlock(&mutex_);
            return false;
        }

        *item = queue_.front();
        queue_.pop_front();
        pthread_mutex_unlock(&mutex_);
        return true;
    }

    void stop()
    {
        pthread_mutex_lock(&mutex_);
        stopped_ = true;
        queue_.clear();
        pthread_cond_broadcast(&cond_);
        pthread_mutex_unlock(&mutex_);
    }

    uint64_t dropped()
    {
        pthread_mutex_lock(&mutex_);
        uint64_t value = dropped_;
        pthread_mutex_unlock(&mutex_);
        return value;
    }

private:
    size_t capacity_;
    bool stopped_;
    uint64_t dropped_;
    std::deque<T> queue_;
    pthread_mutex_t mutex_;
    pthread_cond_t cond_;
};

// -----------------------------------------------------------------------------
// Per-frame packets passed between pipeline stages
// -----------------------------------------------------------------------------
struct CapturePacket {
    uint64_t frame_id;
    uint64_t capture_ts_us;
    double capture_copy_ms;
    cv::Mat yuv420sp;
};

struct PreprocessPacket {
    uint64_t frame_id;
    uint64_t capture_ts_us;
    double capture_copy_ms;
    double preprocess_ms;

    // Original camera image used later for drawing / streaming.
    cv::Mat frame_bgr;

    // 640x640 input copied into RKNN input memory by inference thread.
    cv::Mat model_input;

    // Letterbox parameters MUST belong to the same frame.
    // They cannot be global in a multi-threaded pipeline.
    float scale;
    int left_padding;
    int top_padding;
};

struct InferencePacket {
    uint64_t frame_id;
    uint64_t capture_ts_us;
    double capture_copy_ms;
    double preprocess_ms;
    double inference_ms;

    cv::Mat frame_bgr;
    float scale;
    int left_padding;
    int top_padding;

    object_detect_result_list od_results;
};

// -----------------------------------------------------------------------------
// Thread contexts
// -----------------------------------------------------------------------------
struct CaptureThreadContext {
    LatestQueue<CapturePacket> *output_queue;
};

struct PreprocessThreadContext {
    LatestQueue<CapturePacket> *input_queue;
    LatestQueue<PreprocessPacket> *output_queue;
};

struct InferenceThreadContext {
    LatestQueue<PreprocessPacket> *input_queue;
    LatestQueue<InferencePacket> *output_queue;
    rknn_app_context_t *rknn_app_ctx;
};

struct OutputThreadContext {
    LatestQueue<InferencePacket> *input_queue;

    unsigned char *rgb_dma_data;
    VIDEO_FRAME_INFO_S *h264_frame;
    VENC_STREAM_S *venc_stream;

    rtsp_demo_handle rtsplive;
    rtsp_session_handle rtsp_session;

    LatestQueue<CapturePacket> *capture_queue;
    LatestQueue<PreprocessPacket> *preprocess_queue;
    LatestQueue<InferencePacket> *inference_queue;
};

// -----------------------------------------------------------------------------
// Capture thread
// VI frame -> private YUV copy -> release VI immediately -> capture queue
// -----------------------------------------------------------------------------
static void *capture_thread(void *arg)
{
    CaptureThreadContext *ctx = (CaptureThreadContext *)arg;
    uint64_t frame_id = 0;

    printf("[V2] Capture thread started\n");

    while (g_running) {
        VIDEO_FRAME_INFO_S vi_frame;
        memset(&vi_frame, 0, sizeof(vi_frame));

        // Use a finite timeout so Ctrl+C can stop the pipeline cleanly.
        RK_S32 ret = RK_MPI_VI_GetChnFrame(0, 0, &vi_frame, 1000);
        if (ret != RK_SUCCESS) {
            if (g_running) {
                RK_LOGE("RK_MPI_VI_GetChnFrame fail %x", ret);
            }
            continue;
        }

        void *vi_data = RK_MPI_MB_Handle2VirAddr(vi_frame.stVFrame.pMbBlk);
        uint64_t t0 = now_us();

        CapturePacket packet;
        packet.frame_id = ++frame_id;
        packet.capture_ts_us = t0;
        packet.yuv420sp.create(height + height / 2, width, CV_8UC1);

        // The VI buffer belongs to RKMPI. Copy it before releasing the frame,
        // otherwise another thread would be reading memory that RKMPI may reuse.
        memcpy(packet.yuv420sp.data, vi_data, width * height * 3 / 2);

        uint64_t t1 = now_us();
        packet.capture_copy_ms = us_to_ms(t1 - t0);

        ret = RK_MPI_VI_ReleaseChnFrame(0, 0, &vi_frame);
        if (ret != RK_SUCCESS) {
            RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", ret);
        }

        if (!ctx->output_queue->push(packet)) {
            break;
        }
    }

    printf("[V2] Capture thread stopped\n");
    return NULL;
}

// -----------------------------------------------------------------------------
// Preprocess thread
// YUV420SP -> BGR -> resize + letterbox -> preprocess queue
// -----------------------------------------------------------------------------
static void *preprocess_thread(void *arg)
{
    PreprocessThreadContext *ctx = (PreprocessThreadContext *)arg;
    CapturePacket input;

    printf("[V2] Preprocess thread started\n");

    while (g_running && ctx->input_queue->pop(&input)) {
        uint64_t t0 = now_us();

        PreprocessPacket output;
        output.frame_id = input.frame_id;
        output.capture_ts_us = input.capture_ts_us;
        output.capture_copy_ms = input.capture_copy_ms;

        // Convert camera NV12/YUV420SP to BGR.
        cv::cvtColor(input.yuv420sp, output.frame_bgr, cv::COLOR_YUV420sp2BGR);

        // Letterbox to 640x640 while keeping aspect ratio.
        float scale_x = (float)model_width / (float)width;
        float scale_y = (float)model_height / (float)height;
        output.scale = scale_x < scale_y ? scale_x : scale_y;

        int input_width = (int)((float)width * output.scale);
        int input_height = (int)((float)height * output.scale);

        output.left_padding = (model_width - input_width) / 2;
        output.top_padding = (model_height - input_height) / 2;

        cv::Mat resized;
        cv::resize(output.frame_bgr,
                   resized,
                   cv::Size(input_width, input_height),
                   0,
                   0,
                   cv::INTER_LINEAR);

        output.model_input = cv::Mat(model_height,
                                     model_width,
                                     CV_8UC3,
                                     cv::Scalar(0, 0, 0));

        cv::Rect roi(output.left_padding,
                     output.top_padding,
                     input_width,
                     input_height);
        resized.copyTo(output.model_input(roi));

        uint64_t t1 = now_us();
        output.preprocess_ms = us_to_ms(t1 - t0);

        if (!ctx->output_queue->push(output)) {
            break;
        }
    }

    printf("[V2] Preprocess thread stopped\n");
    return NULL;
}

// -----------------------------------------------------------------------------
// Inference thread
// Copy prepared 640x640 image -> RKNN input memory -> NPU run -> postprocess
// -----------------------------------------------------------------------------
static void *inference_thread(void *arg)
{
    InferenceThreadContext *ctx = (InferenceThreadContext *)arg;
    PreprocessPacket input;

    printf("[V2] Inference thread started\n");

    while (g_running && ctx->input_queue->pop(&input)) {
        uint64_t t0 = now_us();

        InferencePacket output;
        memset(&output.od_results, 0, sizeof(output.od_results));

        output.frame_id = input.frame_id;
        output.capture_ts_us = input.capture_ts_us;
        output.capture_copy_ms = input.capture_copy_ms;
        output.preprocess_ms = input.preprocess_ms;
        output.frame_bgr = input.frame_bgr;
        output.scale = input.scale;
        output.left_padding = input.left_padding;
        output.top_padding = input.top_padding;

        memcpy(ctx->rknn_app_ctx->input_mems[0]->virt_addr,
               input.model_input.data,
               model_width * model_height * 3);

        int ret = inference_yolov5_model(ctx->rknn_app_ctx, &output.od_results);
        if (ret != 0) {
            printf("[V2] inference_yolov5_model failed, ret=%d, frame=%llu\n",
                   ret,
                   (unsigned long long)output.frame_id);
            continue;
        }

        uint64_t t1 = now_us();
        output.inference_ms = us_to_ms(t1 - t0);

        if (!ctx->output_queue->push(output)) {
            break;
        }
    }

    printf("[V2] Inference thread stopped\n");
    return NULL;
}

// Map one YOLO 640x640 coordinate back to the original 720x480 image.
static int map_x(const InferencePacket &packet, int x)
{
    int mapped = (int)((float)(x - packet.left_padding) / packet.scale);
    return clamp_int(mapped, 0, width - 1);
}

static int map_y(const InferencePacket &packet, int y)
{
    int mapped = (int)((float)(y - packet.top_padding) / packet.scale);
    return clamp_int(mapped, 0, height - 1);
}

// -----------------------------------------------------------------------------
// Output thread
// Draw detection -> copy to RGB DMA buffer -> VENC H264 -> RTSP
// -----------------------------------------------------------------------------
static void *output_thread(void *arg)
{
    OutputThreadContext *ctx = (OutputThreadContext *)arg;
    InferencePacket input;

    RK_U32 h264_time_ref = 0;

    uint64_t report_start_us = now_us();
    uint64_t report_frames = 0;
    double sum_capture_copy_ms = 0.0;
    double sum_preprocess_ms = 0.0;
    double sum_inference_ms = 0.0;
    double sum_output_ms = 0.0;
    double sum_e2e_ms = 0.0;

    printf("[V2] Output thread started\n");

    while (g_running && ctx->input_queue->pop(&input)) {
        uint64_t output_t0 = now_us();

        // After pop(), this frame is owned by the output stage.
        // Drawing here does not interfere with preprocess/inference of newer frames.
        cv::Mat frame = input.frame_bgr;

        char text[64];
        for (int i = 0; i < input.od_results.count; ++i) {
            object_detect_result *det_result = &(input.od_results.results[i]);

            int sx = map_x(input, (int)det_result->box.left);
            int sy = map_y(input, (int)det_result->box.top);
            int ex = map_x(input, (int)det_result->box.right);
            int ey = map_y(input, (int)det_result->box.bottom);

            // Protect against malformed / inverted boxes after clipping.
            if (ex <= sx || ey <= sy) {
                continue;
            }

            printf("%s @ (%d %d %d %d) %.3f [frame=%llu]\n",
                   coco_cls_to_name(det_result->cls_id),
                   sx,
                   sy,
                   ex,
                   ey,
                   det_result->prop,
                   (unsigned long long)input.frame_id);

            cv::rectangle(frame,
                          cv::Point(sx, sy),
                          cv::Point(ex, ey),
                          cv::Scalar(0, 255, 0),
                          3);

            snprintf(text,
                     sizeof(text),
                     "%s %.1f%%",
                     coco_cls_to_name(det_result->cls_id),
                     det_result->prop * 100.0f);

            int text_y = sy - 8;
            if (text_y < 20) text_y = sy + 24;

            cv::putText(frame,
                        text,
                        cv::Point(sx, text_y),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        cv::Scalar(0, 255, 0),
                        2);
        }

        // RGB/BGR memory path is kept identical to your V1 encoder path.
        // V1 also used RK_FMT_RGB888 while feeding OpenCV BGR bytes, so we preserve
        // that behavior here instead of changing two variables at the same time.
        memcpy(ctx->rgb_dma_data, frame.data, width * height * 3);

        ctx->h264_frame->stVFrame.u32TimeRef = h264_time_ref++;
        ctx->h264_frame->stVFrame.u64PTS = TEST_COMM_GetNowUs();

        RK_S32 ret = RK_MPI_VENC_SendFrame(0, ctx->h264_frame, 1000);
        if (ret != RK_SUCCESS) {
            RK_LOGE("RK_MPI_VENC_SendFrame fail %x", ret);
            continue;
        }

        ret = RK_MPI_VENC_GetStream(0, ctx->venc_stream, 1000);
        if (ret == RK_SUCCESS) {
            if (ctx->rtsplive && ctx->rtsp_session) {
                void *encoded_data =
                    RK_MPI_MB_Handle2VirAddr(ctx->venc_stream->pstPack->pMbBlk);

                rtsp_tx_video(ctx->rtsp_session,
                              (uint8_t *)encoded_data,
                              ctx->venc_stream->pstPack->u32Len,
                              ctx->venc_stream->pstPack->u64PTS);
                rtsp_do_event(ctx->rtsplive);
            }

            RK_S32 release_ret = RK_MPI_VENC_ReleaseStream(0, ctx->venc_stream);
            if (release_ret != RK_SUCCESS) {
                RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", release_ret);
            }
        } else if (g_running) {
            RK_LOGE("RK_MPI_VENC_GetStream fail %x", ret);
        }

        uint64_t output_t1 = now_us();
        double output_ms = us_to_ms(output_t1 - output_t0);
        double e2e_ms = us_to_ms(output_t1 - input.capture_ts_us);

        ++report_frames;
        sum_capture_copy_ms += input.capture_copy_ms;
        sum_preprocess_ms += input.preprocess_ms;
        sum_inference_ms += input.inference_ms;
        sum_output_ms += output_ms;
        sum_e2e_ms += e2e_ms;

        if (report_frames >= PERF_REPORT_FRAMES) {
            uint64_t now = now_us();
            double seconds = (double)(now - report_start_us) / 1000000.0;
            double fps = seconds > 0.0 ? (double)report_frames / seconds : 0.0;

            printf("\n[V2 PERF] FPS=%.2f | capture-copy=%.2f ms | preprocess=%.2f ms | "
                   "inference=%.2f ms | output=%.2f ms | e2e=%.2f ms\n",
                   fps,
                   sum_capture_copy_ms / report_frames,
                   sum_preprocess_ms / report_frames,
                   sum_inference_ms / report_frames,
                   sum_output_ms / report_frames,
                   sum_e2e_ms / report_frames);

            printf("[V2 DROP] capture->pre=%llu | pre->infer=%llu | infer->out=%llu\n\n",
                   (unsigned long long)ctx->capture_queue->dropped(),
                   (unsigned long long)ctx->preprocess_queue->dropped(),
                   (unsigned long long)ctx->inference_queue->dropped());

            report_start_us = now;
            report_frames = 0;
            sum_capture_copy_ms = 0.0;
            sum_preprocess_ms = 0.0;
            sum_inference_ms = 0.0;
            sum_output_ms = 0.0;
            sum_e2e_ms = 0.0;
        }
    }

    printf("[V2] Output thread stopped\n");
    return NULL;
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    system("RkLunch-stop.sh");

    RK_S32 s32Ret = 0;
    int ret = 0;

    // -------------------------------------------------------------------------
    // RKNN model init
    // -------------------------------------------------------------------------
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    const char *model_path = "./model/yolov5.rknn";
    ret = init_yolov5_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_yolov5_model failed! ret=%d\n", ret);
        return -1;
    }

    printf("init rknn model success!\n");
    init_post_process();

    // -------------------------------------------------------------------------
    // Create one DMA RGB frame for VENC input
    // Only output_thread touches this buffer.
    // -------------------------------------------------------------------------
    MB_POOL_CONFIG_S pool_cfg;
    memset(&pool_cfg, 0, sizeof(MB_POOL_CONFIG_S));
    pool_cfg.u64MBSize = width * height * 3;
    pool_cfg.u32MBCnt = 1;
    pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA;

    MB_POOL src_pool = RK_MPI_MB_CreatePool(&pool_cfg);
    printf("Create Pool success !\n");

    MB_BLK src_blk = RK_MPI_MB_GetMB(src_pool, width * height * 3, RK_TRUE);
    unsigned char *rgb_dma_data =
        (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_blk);

    VIDEO_FRAME_INFO_S h264_frame;
    memset(&h264_frame, 0, sizeof(h264_frame));
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = width;
    h264_frame.stVFrame.u32VirHeight = height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_RGB888;
    h264_frame.stVFrame.u32FrameFlag = 160;
    h264_frame.stVFrame.pMbBlk = src_blk;

    VENC_STREAM_S venc_stream;
    memset(&venc_stream, 0, sizeof(venc_stream));
    venc_stream.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    if (venc_stream.pstPack == NULL) {
        printf("malloc venc_stream.pstPack failed\n");
        RK_MPI_MB_ReleaseMB(src_blk);
        RK_MPI_MB_DestroyPool(src_pool);
        release_yolov5_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }
    memset(venc_stream.pstPack, 0, sizeof(VENC_PACK_S));

    // -------------------------------------------------------------------------
    // ISP / RKMPI / RTSP / VI / VENC init
    // -------------------------------------------------------------------------
    RK_BOOL multi_sensor = RK_FALSE;
    const char *iq_dir = "/etc/iqfiles";
    rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;

    SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
    SAMPLE_COMM_ISP_Run(0);

    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        RK_LOGE("rk mpi sys init fail!");
        g_running = 0;
    }

    rtsp_demo_handle g_rtsplive = NULL;
    rtsp_session_handle g_rtsp_session = NULL;

    if (g_running) {
        g_rtsplive = create_rtsp_demo(554);
        g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
        rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
        rtsp_sync_video_ts(g_rtsp_session,
                           rtsp_get_reltime(),
                           rtsp_get_ntptime());

        vi_dev_init();
        vi_chn_init(0, width, height);

        RK_CODEC_ID_E codec_type = RK_VIDEO_ID_AVC;
        venc_init(0, width, height, codec_type);
        printf("venc init success\n");
    }

    // -------------------------------------------------------------------------
    // V2 pipeline queues
    // -------------------------------------------------------------------------
    LatestQueue<CapturePacket> capture_queue(PIPELINE_QUEUE_SIZE);
    LatestQueue<PreprocessPacket> preprocess_queue(PIPELINE_QUEUE_SIZE);
    LatestQueue<InferencePacket> inference_queue(PIPELINE_QUEUE_SIZE);

    CaptureThreadContext capture_ctx;
    capture_ctx.output_queue = &capture_queue;

    PreprocessThreadContext preprocess_ctx;
    preprocess_ctx.input_queue = &capture_queue;
    preprocess_ctx.output_queue = &preprocess_queue;

    InferenceThreadContext inference_ctx;
    inference_ctx.input_queue = &preprocess_queue;
    inference_ctx.output_queue = &inference_queue;
    inference_ctx.rknn_app_ctx = &rknn_app_ctx;

    OutputThreadContext output_ctx;
    output_ctx.input_queue = &inference_queue;
    output_ctx.rgb_dma_data = rgb_dma_data;
    output_ctx.h264_frame = &h264_frame;
    output_ctx.venc_stream = &venc_stream;
    output_ctx.rtsplive = g_rtsplive;
    output_ctx.rtsp_session = g_rtsp_session;
    output_ctx.capture_queue = &capture_queue;
    output_ctx.preprocess_queue = &preprocess_queue;
    output_ctx.inference_queue = &inference_queue;

    pthread_t capture_tid;
    pthread_t preprocess_tid;
    pthread_t inference_tid;
    pthread_t output_tid;

    bool capture_started = false;
    bool preprocess_started = false;
    bool inference_started = false;
    bool output_started = false;

    if (g_running && pthread_create(&capture_tid, NULL, capture_thread, &capture_ctx) == 0)
        capture_started = true;
    else
        g_running = 0;

    if (g_running && pthread_create(&preprocess_tid, NULL, preprocess_thread, &preprocess_ctx) == 0)
        preprocess_started = true;
    else
        g_running = 0;

    if (g_running && pthread_create(&inference_tid, NULL, inference_thread, &inference_ctx) == 0)
        inference_started = true;
    else
        g_running = 0;

    if (g_running && pthread_create(&output_tid, NULL, output_thread, &output_ctx) == 0)
        output_started = true;
    else
        g_running = 0;

    printf("\n===============================================\n");
    printf(" V2 multi-thread pipeline started\n");
    printf(" Capture -> Preprocess -> Inference -> Output\n");
    printf(" Queue size: %d (drop-oldest / keep-latest)\n", PIPELINE_QUEUE_SIZE);
    printf(" Press Ctrl+C to stop\n");
    printf("===============================================\n\n");

    // Main thread only supervises shutdown.
    while (g_running) {
        sleep(1);
    }

    // Wake all consumers blocked on an empty queue.
    capture_queue.stop();
    preprocess_queue.stop();
    inference_queue.stop();

    if (capture_started) pthread_join(capture_tid, NULL);
    if (preprocess_started) pthread_join(preprocess_tid, NULL);
    if (inference_started) pthread_join(inference_tid, NULL);
    if (output_started) pthread_join(output_tid, NULL);

    printf("[V2] All pipeline threads joined\n");

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    RK_MPI_MB_ReleaseMB(src_blk);
    RK_MPI_MB_DestroyPool(src_pool);

    RK_MPI_VI_DisableChn(0, 0);
    RK_MPI_VI_DisableDev(0);

    SAMPLE_COMM_ISP_Stop(0);

    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);

    free(venc_stream.pstPack);

    if (g_rtsplive)
        rtsp_del_demo(g_rtsplive);

    s32Ret = RK_MPI_SYS_Exit();
    (void)s32Ret;

    release_yolov5_model(&rknn_app_ctx);
    deinit_post_process();

    printf("[V2] Exit cleanly\n");
    return 0;
}
