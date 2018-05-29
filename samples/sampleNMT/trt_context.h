#ifndef SAMPLE_NMT_TRT_CONTEXT_
#define SAMPLE_NMT_TRT_CONTEXT_

#include <memory>

#include "NvInfer.h"

namespace nmtSample
{
    /** \class TRTContext
    *
    * \brief convenient wrapper for TensorRT IExecutionContext
    *
    */
    class TRTContext
    {
    public:
        typedef std::shared_ptr<TRTContext> ptr;

        TRTContext() = delete;

        TRTContext(nvinfer1::IExecutionContext * context);

        virtual ~TRTContext();

        nvinfer1::IExecutionContext* getImpl() const;

    private:
        nvinfer1::IExecutionContext * mContext;
    };
}

#endif // SAMPLE_NMT_TRT_CONTEXT_
