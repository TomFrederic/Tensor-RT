#include "trt_engine.h"

namespace nmtSample
{
    TRTEngine::TRTEngine(nvinfer1::ICudaEngine * engine)
        : mEngine(engine)
    {
    }

    TRTEngine::~TRTEngine()
    {
        if (mEngine)
        {
            mEngine->destroy();
            mEngine = nullptr;
        }
    }

    nvinfer1::ICudaEngine* TRTEngine::getImpl() const
    {
        return mEngine;
    }
}
