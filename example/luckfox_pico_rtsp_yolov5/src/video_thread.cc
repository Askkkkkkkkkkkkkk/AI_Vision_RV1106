#include "video_thread.h"

#include <stdio.h>
#include <string.h>


void* video_thread(void* arg)
{

    VideoThreadContext* ctx =
        (VideoThreadContext*)arg;


    printf("[V3] Video thread started\n");


    RK_U32 h264_time_ref = 0;


    while(*(ctx->running))
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



        cv::Mat yuv420sp(
            480 + 480 / 2,
            720,
            CV_8UC1,
            vi_data
        );



        cv::Mat frame;


        cv::cvtColor(
            yuv420sp,
            frame,
            cv::COLOR_YUV420sp2BGR
        );



        object_detect_result_list od_results;


        memset(
            &od_results,
            0,
            sizeof(od_results)
        );


        od_results =
            ctx->detection_buffer->get();



        for(int i = 0;
            i < od_results.count;
            i++)
        {

            object_detect_result* det =
                &(od_results.results[i]);



            int x1 = det->box.left;
            int y1 = det->box.top;
            int x2 = det->box.right;
            int y2 = det->box.bottom;



            cv::rectangle(
                frame,
                cv::Point(x1,y1),
                cv::Point(x2,y2),
                cv::Scalar(0,255,0),
                3
            );



            char text[64];


            snprintf(
                text,
                sizeof(text),
                "%s %.1f%%",
                coco_cls_to_name(det->cls_id),
                det->prop * 100
            );



            cv::putText(
                frame,
                text,
                cv::Point(x1,y1-5),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(0,255,0),
                2
            );

        }



        memcpy(
            ctx->rgb_dma_data,
            frame.data,
            720 * 480 * 3
        );



        ctx->h264_frame->stVFrame.u32TimeRef =
            h264_time_ref++;



        ctx->h264_frame->stVFrame.u64PTS =
            TEST_COMM_GetNowUs();



        ret =
            RK_MPI_VENC_SendFrame(
                0,
                ctx->h264_frame,
                1000
            );



        if(ret != RK_SUCCESS)
        {

            RK_LOGE(
                "RK_MPI_VENC_SendFrame failed %x",
                ret
            );

            RK_MPI_VI_ReleaseChnFrame(
                0,
                0,
                &vi_frame
            );

            continue;
        }



        ret =
            RK_MPI_VENC_GetStream(
                0,
                ctx->venc_stream,
                1000
            );



        if(ret == RK_SUCCESS)
        {

            void* encoded_data =
                RK_MPI_MB_Handle2VirAddr(
                    ctx->venc_stream->pstPack->pMbBlk
                );



            if(ctx->rtsplive &&
               ctx->rtsp_session)
            {

                rtsp_tx_video(
                    ctx->rtsp_session,
                    (uint8_t*)encoded_data,
                    ctx->venc_stream->pstPack->u32Len,
                    ctx->venc_stream->pstPack->u64PTS
                );


                rtsp_do_event(
                    ctx->rtsplive
                );

            }



            RK_MPI_VENC_ReleaseStream(
                0,
                ctx->venc_stream
            );

        }



        RK_MPI_VI_ReleaseChnFrame(
            0,
            0,
            &vi_frame
        );


    }


    printf("[V3] Video thread stopped\n");


    return NULL;

}