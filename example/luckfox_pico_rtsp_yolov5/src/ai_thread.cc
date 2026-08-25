#include "ai_thread.h"
#include "perf_monitor.h"

#include <stdio.h>
#include <string.h>


void* ai_thread(void* arg)
{

    AIThreadContext* ctx =
        (AIThreadContext*)arg;


    printf("[V3] AI thread started\n");


    PreprocessPacket input;


    uint64_t ai_frame_count = 0;

    double ai_time_sum = 0;



    while(*(ctx->running))
    {


        if(!ctx->input_queue->pop(&input))
        {
            break;
        }



        uint64_t t0 =
            get_time_us();



        object_detect_result_list od_results;


        memset(
            &od_results,
            0,
            sizeof(od_results)
        );



        memcpy(
            ctx->rknn_ctx->input_mems[0]->virt_addr,

            input.model_input.data,

            640 * 640 * 3
        );




        int ret =
            inference_yolov5_model(
                ctx->rknn_ctx,
                &od_results
            );



        if(ret != 0)
        {

            printf(
                "[V3] inference failed ret=%d\n",
                ret
            );

            continue;

        }



        uint64_t t1 =
            get_time_us();



        double ai_ms =
            (t1 - t0) / 1000.0;



        ai_frame_count++;

        ai_time_sum += ai_ms;



        ctx->detection_buffer->update(
            od_results
        );



        if(ai_frame_count % 30 == 0)
        {

            printf(
                "\n[V3 AI PERF]\n"
                "Frames: %llu\n"
                "Average inference: %.2f ms\n\n",
                (unsigned long long)ai_frame_count,
                ai_time_sum / ai_frame_count
            );

        }



        printf(
            "[V3 AI] frame=%llu detect=%d time=%.2f ms\n",
            (unsigned long long)input.frame_id,
            od_results.count,
            ai_ms
        );


    }



    printf("[V3] AI thread stopped\n");


    return NULL;

}