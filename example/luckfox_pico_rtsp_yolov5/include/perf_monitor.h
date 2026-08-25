#ifndef PERF_MONITOR_H
#define PERF_MONITOR_H


#include <stdint.h>
#include <time.h>
#include <stdio.h>


static inline uint64_t get_time_us()
{
    struct timespec ts;

    clock_gettime(
        CLOCK_MONOTONIC,
        &ts
    );

    return 
        ts.tv_sec * 1000000ULL +
        ts.tv_nsec / 1000;
}



struct PerfMonitor
{

    uint64_t start_time;

    uint64_t frame_count;


    double ai_time_sum;

    double video_time_sum;

    double latency_sum;


    PerfMonitor()
    {
        start_time = get_time_us();

        frame_count = 0;

        ai_time_sum = 0;

        video_time_sum = 0;

        latency_sum = 0;
    }



    void add(
        double ai_ms,
        double video_ms,
        double latency_ms
    )
    {

        frame_count++;

        ai_time_sum += ai_ms;

        video_time_sum += video_ms;

        latency_sum += latency_ms;


        if(frame_count % 30 == 0)
        {

            uint64_t now = get_time_us();


            double sec =
                (now-start_time)/1000000.0;


            printf("\n========== V3 PERFORMANCE ==========\n");


            printf(
                "FPS: %.2f\n",
                frame_count/sec
            );


            printf(
                "AI avg: %.2f ms\n",
                ai_time_sum/frame_count
            );


            printf(
                "Video avg: %.2f ms\n",
                video_time_sum/frame_count
            );


            printf(
                "Latency avg: %.2f ms\n",
                latency_sum/frame_count
            );


            printf(
                "====================================\n\n"
            );

        }

    }

};


#endif