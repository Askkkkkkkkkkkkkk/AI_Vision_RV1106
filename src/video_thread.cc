#include "video_thread.h"

#include <stdio.h>
#include <unistd.h>



void* video_thread(void* arg)
{

    DetectionBuffer* buffer =
        static_cast<DetectionBuffer*>(arg);



    while(1)
    {


        /*
            后续替换：

            RK_MPI_VI_GetChnFrame()

                    |

            RK_MPI_VENC_SendFrame()

                    |

            RTSP


        */


        std::vector<DetectBox> boxes =
            buffer->get();



        printf(
            "[Video Thread] output frame, boxes=%ld\n",
            boxes.size()
        );


        sleep(1);

    }


    return nullptr;
}