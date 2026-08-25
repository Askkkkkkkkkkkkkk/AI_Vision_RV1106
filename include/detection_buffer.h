#ifndef DETECTION_BUFFER_H
#define DETECTION_BUFFER_H


#include <vector>
#include <mutex>


// 单个检测框
struct DetectBox
{
    int class_id;     // 类别ID

    float score;      // 置信度

    int x1;
    int y1;

    int x2;
    int y2;
};



class DetectionBuffer
{

public:


    // AI线程更新检测结果
    void update(
        const std::vector<DetectBox>& result
    );


    // 视频线程获取最新结果
    std::vector<DetectBox> get();



private:


    // 防止AI写入和视频读取冲突
    std::mutex mutex_;


    // 保存最新检测结果
    std::vector<DetectBox> boxes_;


};


#endif
