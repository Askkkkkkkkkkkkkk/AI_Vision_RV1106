#ifndef AI_THREAD_H
#define AI_THREAD_H
#include <signal.h>

#include "pipeline_common.h"
#include "detection_buffer.h"
#include "yolov5.h"


struct AIThreadContext
{

    LatestQueue<PreprocessPacket>* input_queue;


    DetectionBuffer* detection_buffer;


    rknn_app_context_t* rknn_ctx;


    volatile sig_atomic_t* running;

};


void* ai_thread(void* arg);


#endif