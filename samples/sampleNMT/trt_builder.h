#ifndef SAMPLE_NMT_TRT_BUILDER_
#define SAMPLE_NMT_TRT_BUILDER_

#include <memory>

#include "NvInfer.h"

namespace nmtSample
{
    /** \class TRTBuilder
    *
    * \brief convenient wrapper for TensorRT IBuilder
    *
    */
    class TRTBuilder
    {
    private:
        class Logger : public nvinfer1::ILogger
        {
        public:
            Logger(bool verbose);

            void log(Severity severity, const char * msg) override;

        private:
            bool mVerbose;
        };

    public:
        typedef std::shared_ptr<TRTBuilder> ptr;

        TRTBuilder() = delete;

        TRTBuilder(int maxBatchSize, int maxWorkspaceSize, bool verbose);

        virtual ~TRTBuilder();

        nvinfer1::IBuilder* getImpl() const;

    private:
        nvinfer1::IBuilder * mBuilder;
        Logger mLogger;
    };
}

#endif // SAMPLE_NMT_TRT_BUILDER_
