#ifndef VIDEO_THREAD_H
#define VIDEO_THREAD_H


#include <signal.h>


#include "pipeline_common.h"
#include "detection_buffer.h"
#include "luckfox_mpi.h"
#include "rtsp_demo.h"


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



struct VideoThreadContext
{

    // Receive processed BGR frame
    // from preprocess thread

    LatestQueue<VideoPacket>* input_queue;



    // AI detection result buffer

    DetectionBuffer* detection_buffer;



    // VENC input buffer

    unsigned char* rgb_dma_data;



    VIDEO_FRAME_INFO_S* h264_frame;



    VENC_STREAM_S* venc_stream;



    rtsp_demo_handle rtsplive;



    rtsp_session_handle rtsp_session;



    volatile sig_atomic_t* running;

};



void* video_thread(void* arg);



#endif