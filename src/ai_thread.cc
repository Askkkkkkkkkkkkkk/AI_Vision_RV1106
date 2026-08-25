#include "ai_thread.h"

#include <stdio.h>
#include <unistd.h>



void* ai_thread(void* arg)
{

    DetectionBuffer* buffer =
        static_cast<DetectionBuffer*>(arg);


    while(1)
    {

        /*
            后续替换为你的:

            1. 获取camera frame

            2. preprocess

            3. rknn_run()

            4. postprocess()

            5. 得到vector<DetectBox>
        */


        std::vector<DetectBox> result;


        // 更新最新检测结果
        buffer->update(result);



        printf(
            "[AI Thread] inference finished\n"
        );


        sleep(1);

    }


    return nullptr;
}