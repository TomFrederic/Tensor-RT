#include "trt_context.h"

namespace nmtSample
{
    TRTContext::TRTContext(nvinfer1::IExecutionContext * context)
        : mContext(context)
    {
    }

    TRTContext::~TRTContext()
    {
        if (mContext)
        {
            mContext->destroy();
            mContext = nullptr;
        }
    }

    nvinfer1::IExecutionContext* TRTContext::getImpl() const
    {
        return mContext;
    }
}
