#include "detection_buffer.h"



void DetectionBuffer::update(
    const object_detect_result_list& result
)
{

    std::lock_guard<std::mutex> lock(
        mutex_
    );


    memcpy(
        &result_,
        &result,
        sizeof(object_detect_result_list)
    );

}



object_detect_result_list DetectionBuffer::get()
{

    std::lock_guard<std::mutex> lock(
        mutex_
    );


    object_detect_result_list result;


    memcpy(
        &result,
        &result_,
        sizeof(object_detect_result_list)
    );


    return result;

}