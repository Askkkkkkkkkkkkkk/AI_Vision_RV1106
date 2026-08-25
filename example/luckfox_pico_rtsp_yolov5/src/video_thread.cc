#include "video_thread.h"
#include "perf_monitor.h"

#include <stdio.h>
#include <string.h>



void* video_thread(void* arg)
{

    VideoThreadContext* ctx =
        (VideoThreadContext*)arg;



    printf("[V3] Video thread started\n");



    RK_U32 h264_time_ref = 0;



    uint64_t frame_count = 0;


    uint64_t fps_start =
        get_time_us();



    double video_time_sum = 0;



    VideoPacket input;



    while(*(ctx->running))
    {


        uint64_t frame_start =
            get_time_us();



        /*
            Get BGR frame from preprocess

            No:
                RK_MPI_VI_GetChnFrame()

            No:
                cvtColor()

        */


        if(!ctx->input_queue->pop(&input))
        {
            break;
        }



        cv::Mat frame =
            input.frame_bgr;



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



            cv::rectangle(
                frame,
                cv::Point(
                    det->box.left,
                    det->box.top
                ),
                cv::Point(
                    det->box.right,
                    det->box.bottom
                ),
                cv::Scalar(
                    0,
                    255,
                    0
                ),
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
                cv::Point(
                    det->box.left,
                    det->box.top - 5
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
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





        RK_S32 ret =
            RK_MPI_VENC_SendFrame(
                0,
                ctx->h264_frame,
                1000
            );



        if(ret == RK_SUCCESS)
        {


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

        }






        uint64_t frame_end =
            get_time_us();




        double video_ms =
            (frame_end-frame_start)/1000.0;



        frame_count++;


        video_time_sum += video_ms;




        if(frame_count % 30 == 0)
        {


            uint64_t now =
                get_time_us();



            double fps =
                frame_count /
                ((now-fps_start)/1000000.0);



            printf(
                "\n[V3 VIDEO PERF]\n"
                "FPS: %.2f\n"
                "Average video time: %.2f ms\n"
                "Frames: %llu\n\n",
                fps,
                video_time_sum/frame_count,
                (unsigned long long)frame_count
            );


        }


    }



    printf("[V3] Video thread stopped\n");


    return NULL;

}