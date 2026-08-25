// main.cc 1/2

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


#include "pipeline_common.h"

#include "ai_thread.h"
#include "video_thread.h"
#include "detection_buffer.h"


#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "yolov5.h"


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



#define DISP_WIDTH  720
#define DISP_HEIGHT 480


static const int width = DISP_WIDTH;
static const int height = DISP_HEIGHT;



static volatile sig_atomic_t g_running = 1;



static void signal_handler(int sig)
{

    (void)sig;

    g_running = 0;

}






// ============================================================
// Capture Packet
// ============================================================

struct CapturePacket
{

    uint64_t frame_id;


    uint64_t capture_ts_us;


    double capture_copy_ms;


    cv::Mat yuv420sp;

};






struct CaptureThreadContext
{

    LatestQueue<CapturePacket>* output_queue;

};






static void* capture_thread(void* arg)
{

    CaptureThreadContext* ctx =
        (CaptureThreadContext*)arg;



    uint64_t frame_id = 0;



    printf("[V3] Capture thread started\n");



    while(g_running)
    {


        VIDEO_FRAME_INFO_S vi_frame;


        memset(
            &vi_frame,
            0,
            sizeof(vi_frame)
        );



        RK_S32 ret =
            RK_MPI_VI_GetChnFrame(
                0,
                0,
                &vi_frame,
                1000
            );



        if(ret != RK_SUCCESS)
        {
            continue;
        }



        void* vi_data =
            RK_MPI_MB_Handle2VirAddr(
                vi_frame.stVFrame.pMbBlk
            );



        CapturePacket packet;


        packet.frame_id =
            ++frame_id;



        packet.capture_ts_us =
            TEST_COMM_GetNowUs();



        packet.yuv420sp.create(
            height + height / 2,
            width,
            CV_8UC1
        );



        memcpy(
            packet.yuv420sp.data,
            vi_data,
            width * height * 3 / 2
        );



        RK_MPI_VI_ReleaseChnFrame(
            0,
            0,
            &vi_frame
        );



        if(!ctx->output_queue->push(packet))
        {
            break;
        }


    }



    printf("[V3] Capture thread stopped\n");


    return NULL;

}







// ============================================================
// Preprocess Thread
//
// Capture
//     |
// YUV420SP
//     |
// BGR
//     |
// +-----------+
// |           |
// AI        Video
//
// ============================================================



struct PreprocessThreadContext
{

    LatestQueue<CapturePacket>* input_queue;


    LatestQueue<PreprocessPacket>* output_queue;


    LatestQueue<VideoPacket>* video_queue;

};






static void* preprocess_thread(void* arg)
{

    PreprocessThreadContext* ctx =
        (PreprocessThreadContext*)arg;



    CapturePacket input;



    printf("[V3] Preprocess thread started\n");



    while(
        g_running &&
        ctx->input_queue->pop(&input)
    )
    {


        PreprocessPacket output;



        output.frame_id =
            input.frame_id;



        output.capture_ts_us =
            input.capture_ts_us;



        cv::cvtColor(
            input.yuv420sp,
            output.frame_bgr,
            cv::COLOR_YUV420sp2BGR
        );



        // send frame to video thread

        VideoPacket video;


        video.frame_id =
            output.frame_id;


        video.capture_ts_us =
            output.capture_ts_us;


        video.frame_bgr =
            output.frame_bgr;



        ctx->video_queue->push(video);





        float scale_x =
            640.0f / width;



        float scale_y =
            640.0f / height;



        output.scale =
            scale_x < scale_y ?
            scale_x :
            scale_y;



        int input_width =
            width * output.scale;



        int input_height =
            height * output.scale;



        output.left_padding =
            (640 - input_width) / 2;



        output.top_padding =
            (640 - input_height) / 2;



        cv::Mat resized;



        cv::resize(
            output.frame_bgr,
            resized,
            cv::Size(
                input_width,
                input_height
            )
        );



        output.model_input =
            cv::Mat(
                640,
                640,
                CV_8UC3,
                cv::Scalar(
                    0,
                    0,
                    0
                )
            );



        cv::Rect roi(
            output.left_padding,
            output.top_padding,
            input_width,
            input_height
        );



        resized.copyTo(
            output.model_input(roi)
        );



        if(!ctx->output_queue->push(output))
        {
            break;
        }


    }



    printf("[V3] Preprocess thread stopped\n");


    return NULL;

}

// main.cc 2/2


int main(int argc, char *argv[])
{

    (void)argc;
    (void)argv;



    signal(
        SIGINT,
        signal_handler
    );


    signal(
        SIGTERM,
        signal_handler
    );



    system(
        "RkLunch-stop.sh"
    );



    int ret = 0;



    // ========================================================
    // RKNN init
    // ========================================================


    rknn_app_context_t rknn_app_ctx;


    memset(
        &rknn_app_ctx,
        0,
        sizeof(rknn_app_ctx)
    );



    const char* model_path =
        "./model/yolov5.rknn";



    ret =
        init_yolov5_model(
            model_path,
            &rknn_app_ctx
        );



    if(ret != 0)
    {

        printf(
            "init yolov5 model failed\n"
        );

        return -1;

    }



    init_post_process();





    // ========================================================
    // RGB DMA buffer
    // ========================================================


    MB_POOL_CONFIG_S pool_cfg;


    memset(
        &pool_cfg,
        0,
        sizeof(pool_cfg)
    );



    pool_cfg.u64MBSize =
        width * height * 3;


    pool_cfg.u32MBCnt =
        1;


    pool_cfg.enAllocType =
        MB_ALLOC_TYPE_DMA;



    MB_POOL src_pool =
        RK_MPI_MB_CreatePool(
            &pool_cfg
        );



    MB_BLK src_blk =
        RK_MPI_MB_GetMB(
            src_pool,
            width * height * 3,
            RK_TRUE
        );



    unsigned char* rgb_dma_data =
        (unsigned char*)
        RK_MPI_MB_Handle2VirAddr(
            src_blk
        );





    VIDEO_FRAME_INFO_S h264_frame;


    memset(
        &h264_frame,
        0,
        sizeof(h264_frame)
    );


    h264_frame.stVFrame.u32Width =
        width;


    h264_frame.stVFrame.u32Height =
        height;


    h264_frame.stVFrame.u32VirWidth =
        width;


    h264_frame.stVFrame.u32VirHeight =
        height;


    h264_frame.stVFrame.enPixelFormat =
        RK_FMT_RGB888;


    h264_frame.stVFrame.pMbBlk =
        src_blk;





    VENC_STREAM_S venc_stream;


    memset(
        &venc_stream,
        0,
        sizeof(venc_stream)
    );



    venc_stream.pstPack =
        (VENC_PACK_S*)
        malloc(
            sizeof(VENC_PACK_S)
        );






    // ========================================================
    // ISP / VI / VENC / RTSP
    // ========================================================


    RK_BOOL multi_sensor =
        RK_FALSE;


    const char* iq_dir =
        "/etc/iqfiles";



    rk_aiq_working_mode_t hdr_mode =
        RK_AIQ_WORKING_MODE_NORMAL;



    SAMPLE_COMM_ISP_Init(
        0,
        hdr_mode,
        multi_sensor,
        iq_dir
    );


    SAMPLE_COMM_ISP_Run(0);




    if(
        RK_MPI_SYS_Init()
        != RK_SUCCESS
    )
    {

        printf(
            "RK_MPI_SYS_Init failed\n"
        );

        return -1;

    }




    rtsp_demo_handle rtsplive =
        create_rtsp_demo(
            554
        );



    rtsp_session_handle rtsp_session =
        rtsp_new_session(
            rtsplive,
            "/live/0"
        );



    rtsp_set_video(
        rtsp_session,
        RTSP_CODEC_ID_VIDEO_H264,
        NULL,
        0
    );



    rtsp_sync_video_ts(
        rtsp_session,
        rtsp_get_reltime(),
        rtsp_get_ntptime()
    );



    vi_dev_init();


    vi_chn_init(
        0,
        width,
        height
    );


    venc_init(
        0,
        width,
        height,
        RK_VIDEO_ID_AVC
    );





    // ========================================================
    // Queues
    // ========================================================


    LatestQueue<CapturePacket> capture_queue(
        PIPELINE_QUEUE_SIZE
    );


    LatestQueue<PreprocessPacket> preprocess_queue(
        PIPELINE_QUEUE_SIZE
    );


    LatestQueue<VideoPacket> video_queue(
        PIPELINE_QUEUE_SIZE
    );



    DetectionBuffer detection_buffer;





    // ========================================================
    // Context
    // ========================================================


    CaptureThreadContext capture_ctx;


    capture_ctx.output_queue =
        &capture_queue;





    PreprocessThreadContext preprocess_ctx;


    preprocess_ctx.input_queue =
        &capture_queue;


    preprocess_ctx.output_queue =
        &preprocess_queue;


    preprocess_ctx.video_queue =
        &video_queue;





    AIThreadContext ai_ctx;


    ai_ctx.input_queue =
        &preprocess_queue;


    ai_ctx.detection_buffer =
        &detection_buffer;


    ai_ctx.rknn_ctx =
        &rknn_app_ctx;


    ai_ctx.running =
        &g_running;





    VideoThreadContext video_ctx;


    video_ctx.input_queue =
        &video_queue;


    video_ctx.detection_buffer =
        &detection_buffer;


    video_ctx.rgb_dma_data =
        rgb_dma_data;


    video_ctx.h264_frame =
        &h264_frame;


    video_ctx.venc_stream =
        &venc_stream;


    video_ctx.rtsplive =
        rtsplive;


    video_ctx.rtsp_session =
        rtsp_session;


    video_ctx.running =
        &g_running;







    // ========================================================
    // Create threads
    // ========================================================


    pthread_t capture_tid;

    pthread_t preprocess_tid;

    pthread_t ai_tid;

    pthread_t video_tid;



    pthread_create(
        &capture_tid,
        NULL,
        capture_thread,
        &capture_ctx
    );


    pthread_create(
        &preprocess_tid,
        NULL,
        preprocess_thread,
        &preprocess_ctx
    );


    pthread_create(
        &ai_tid,
        NULL,
        ai_thread,
        &ai_ctx
    );


    pthread_create(
        &video_tid,
        NULL,
        video_thread,
        &video_ctx
    );





    printf(
        "\n=============================\n"
    );


    printf(
        " V3 pipeline started\n"
    );


    printf(
        " Capture\n"
        "    |\n"
        " Preprocess\n"
        "   |      |\n"
        "   AI   Video\n"
        "          |\n"
        "        RTSP\n"
    );


    printf(
        "=============================\n"
    );






    while(g_running)
    {

        sleep(1);

    }






    // ========================================================
    // Stop
    // ========================================================


    capture_queue.stop();


    preprocess_queue.stop();


    video_queue.stop();





    pthread_join(
        capture_tid,
        NULL
    );


    pthread_join(
        preprocess_tid,
        NULL
    );


    pthread_join(
        ai_tid,
        NULL
    );


    pthread_join(
        video_tid,
        NULL
    );







    // ========================================================
    // Cleanup
    // ========================================================


    RK_MPI_MB_ReleaseMB(
        src_blk
    );


    RK_MPI_MB_DestroyPool(
        src_pool
    );



    RK_MPI_VI_DisableChn(
        0,
        0
    );


    RK_MPI_VI_DisableDev(
        0
    );



    SAMPLE_COMM_ISP_Stop(
        0
    );



    RK_MPI_VENC_StopRecvFrame(
        0
    );


    RK_MPI_VENC_DestroyChn(
        0
    );



    if(rtsplive)
    {

        rtsp_del_demo(
            rtsplive
        );

    }



    RK_MPI_SYS_Exit();



    release_yolov5_model(
        &rknn_app_ctx
    );


    deinit_post_process();



    printf(
        "[V3] Exit cleanly\n"
    );



    return 0;

}