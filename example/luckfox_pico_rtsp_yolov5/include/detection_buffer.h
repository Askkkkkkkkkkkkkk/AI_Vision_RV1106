#ifndef DETECTION_BUFFER_H
#define DETECTION_BUFFER_H


#include <mutex>
#include <string.h>

#include "yolov5.h"


// ============================================================
// DetectionBuffer
//
// AI thread:
//      RKNN inference
//          |
//          v
//      object_detect_result_list
//          |
//          v
//      update()
//
//
// Video thread:
//      get()
//          |
//          v
//      draw boxes
//          |
//          v
//      RTSP output
//
// Only keep newest detection result.
// ============================================================


class DetectionBuffer
{

public:


    DetectionBuffer()
    {
        memset(
            &result_,
            0,
            sizeof(result_)
        );
    }



    // AI thread writes newest result
    void update(
        const object_detect_result_list& result
    );



    // Video thread reads newest result
    object_detect_result_list get();



private:


    // protect read/write
    std::mutex mutex_;



    // newest YOLO result
    object_detect_result_list result_;


};


#endif