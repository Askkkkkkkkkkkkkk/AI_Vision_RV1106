#include "detection_buffer.h"


void DetectionBuffer::update(
    const std::vector<DetectBox>& result
)
{
    std::lock_guard<std::mutex> lock(mutex_);

    boxes_ = result;
}



std::vector<DetectBox> DetectionBuffer::get()
{
    std::lock_guard<std::mutex> lock(mutex_);

    return boxes_;
}