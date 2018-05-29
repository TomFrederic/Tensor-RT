#ifndef SAMPLE_NMT_TRT_ENGINE_
#define SAMPLE_NMT_TRT_ENGINE_

#include <memory>

#include "NvInfer.h"

namespace nmtSample
{
    /** \class TRTEngine
    *
    * \brief convenient wrapper for TensorRT ICudaEngine
    *
    */
    class TRTEngine
    {
    public:
        typedef std::shared_ptr<TRTEngine> ptr;

        TRTEngine() = delete;

        TRTEngine(nvinfer1::ICudaEngine * engine);

        virtual ~TRTEngine();

        nvinfer1::ICudaEngine* getImpl() const;

    private:
        nvinfer1::ICudaEngine * mEngine;
    };
}

#endif // SAMPLE_NMT_TRT_ENGINE_
