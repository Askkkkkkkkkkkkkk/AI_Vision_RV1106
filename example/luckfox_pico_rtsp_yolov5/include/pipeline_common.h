#ifndef PIPELINE_COMMON_H
#define PIPELINE_COMMON_H


#include <deque>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "opencv2/core/core.hpp"


#define PIPELINE_QUEUE_SIZE 1



template <typename T>
class LatestQueue
{

public:

    explicit LatestQueue(size_t capacity)
        :
        capacity_(capacity),
        stopped_(false),
        dropped_(0)
    {

        pthread_mutex_init(
            &mutex_,
            NULL
        );

        pthread_cond_init(
            &cond_,
            NULL
        );
    }



    ~LatestQueue()
    {

        pthread_cond_destroy(
            &cond_
        );

        pthread_mutex_destroy(
            &mutex_
        );
    }



    bool push(const T& item)
    {

        pthread_mutex_lock(
            &mutex_
        );


        if(stopped_)
        {

            pthread_mutex_unlock(
                &mutex_
            );

            return false;
        }



        while(queue_.size() >= capacity_)
        {

            queue_.pop_front();

            dropped_++;

        }



        queue_.push_back(item);



        pthread_cond_signal(
            &cond_
        );


        pthread_mutex_unlock(
            &mutex_
        );


        return true;

    }




    bool pop(T* item)
    {

        pthread_mutex_lock(
            &mutex_
        );


        while(queue_.empty() && !stopped_)
        {

            pthread_cond_wait(
                &cond_,
                &mutex_
            );

        }



        if(queue_.empty() && stopped_)
        {

            pthread_mutex_unlock(
                &mutex_
            );

            return false;

        }



        *item =
            queue_.front();



        queue_.pop_front();



        pthread_mutex_unlock(
            &mutex_
        );


        return true;

    }




    void stop()
    {

        pthread_mutex_lock(
            &mutex_
        );


        stopped_ = true;


        queue_.clear();



        pthread_cond_broadcast(
            &cond_
        );


        pthread_mutex_unlock(
            &mutex_
        );

    }





    uint64_t dropped()
    {

        pthread_mutex_lock(
            &mutex_
        );


        uint64_t value =
            dropped_;



        pthread_mutex_unlock(
            &mutex_
        );


        return value;

    }



private:

    size_t capacity_;

    bool stopped_;

    uint64_t dropped_;


    std::deque<T> queue_;


    pthread_mutex_t mutex_;

    pthread_cond_t cond_;

};







// ============================================================
// PreprocessPacket
//
// Capture
//   |
// YUV420SP
//   |
// BGR
//   |
// resize + letterbox
//   |
// AI Thread
//
// ============================================================

struct PreprocessPacket
{

    uint64_t frame_id;


    uint64_t capture_ts_us;



    double capture_copy_ms;


    double preprocess_ms;



    // Original BGR frame
    // Used for detection coordinate mapping

    cv::Mat frame_bgr;



    // 640x640 input

    cv::Mat model_input;



    // Letterbox parameters

    float scale;


    int left_padding;


    int top_padding;

};







// ============================================================
// VideoPacket
//
// Preprocess Thread
//        |
//        v
// Video Thread
//        |
//        v
// Draw + VENC + RTSP
//
// Avoid duplicate:
// YUV -> BGR conversion
//
// ============================================================

struct VideoPacket
{

    uint64_t frame_id;


    uint64_t capture_ts_us;



    // Already converted BGR image

    cv::Mat frame_bgr;

};




#endif