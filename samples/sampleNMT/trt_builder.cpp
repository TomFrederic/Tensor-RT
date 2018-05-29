#include "trt_builder.h"

#include <iostream>

namespace nmtSample
{
    TRTBuilder::TRTBuilder(int maxBatchSize, int maxWorkspaceSize, bool verbose)
        : mBuilder(nullptr)
        , mLogger(verbose)
    {
     	mBuilder = nvinfer1::createInferBuilder(mLogger);
        mBuilder->setMaxBatchSize(maxBatchSize);
        mBuilder->setMaxWorkspaceSize(maxWorkspaceSize);
    }

    TRTBuilder::~TRTBuilder()
    {
        if (mBuilder)
        {
            mBuilder->destroy();
            mBuilder = nullptr;
        }
    }

    nvinfer1::IBuilder* TRTBuilder::getImpl() const
    {
        return mBuilder;
    }

    TRTBuilder::Logger::Logger(bool verbose)
        : mVerbose(verbose)
    {
    }

    void TRTBuilder::Logger::log(Logger::Severity severity, const char * msg)
    {
        // suppress info-level messages
        if (mVerbose || (severity != Logger::Severity::kINFO))
            std::cout << msg << std::endl;
    }
}
